//! On-disk format for the exact-vector store.
//!
//! The vector file is laid out as:
//!
//! ```text
//! +------------------------------------------------------------------+
//! | FileHeader (HEADER_SIZE bytes, zero-padded)                       |
//! +------------------------------------------------------------------+
//! | Block 0  (block_size bytes)                                       |
//! | Block 1  (block_size bytes)                                       |
//! | ...                                                               |
//! | Block N-1 (block_size bytes)                                      |
//! +------------------------------------------------------------------+
//! ```
//!
//! Each block holds a whole number of fixed-width vectors packed back to
//! back; the tail of a block is zero padding. A vector never straddles a
//! block boundary, so fetching any vector touches exactly one block — one
//! seek plus one sequential read on a spinning disk.
//!
//! All multi-byte integers are little-endian. The two supported targets
//! (x86_64, aarch64) are little-endian, so the on-disk and in-memory
//! representations coincide and `mmap` views need no byte swapping.

use slate_core::{Dtype, Error, Result};

/// Magic bytes identifying a Slate-ANN vector store: `"SLATEVEC"`.
pub const MAGIC: [u8; 8] = *b"SLATEVEC";

/// On-disk format version. Bumped on any incompatible layout change.
pub const FORMAT_VERSION: u32 = 1;

/// Size reserved for the file header, in bytes.
///
/// The logical header is smaller than this; the remainder is zero padding
/// so the first data block starts at a fixed, cache-line- and
/// page-friendly offset. 4 KiB also matches the smallest common page size,
/// keeping block starts page-aligned when `block_size` is a multiple of it.
pub const HEADER_SIZE: usize = 4096;

/// Numeric tag for a [`Dtype`] as stored on disk.
///
/// Kept stable and independent of the in-memory enum's discriminants so the
/// on-disk format does not change if the enum is reordered.
fn dtype_tag(dtype: Dtype) -> u8 {
    match dtype {
        Dtype::F32 => 1,
        Dtype::F16 => 2,
        Dtype::I8 => 3,
    }
}

/// Inverse of [`dtype_tag`].
fn dtype_from_tag(tag: u8) -> Result<Dtype> {
    match tag {
        1 => Ok(Dtype::F32),
        2 => Ok(Dtype::F16),
        3 => Ok(Dtype::I8),
        other => Err(Error::corrupt(format!("unknown dtype tag {other}"))),
    }
}

/// Parsed, validated header of a vector-store file.
///
/// This is the self-describing preamble: opening a store reads it back and
/// checks the magic, version, and geometry before trusting any block data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileHeader {
    /// Element type of every stored vector.
    pub dtype: Dtype,
    /// Dimensionality (element count) of every stored vector.
    pub dimensions: u32,
    /// Number of vectors stored.
    pub count: u64,
    /// Size of each block in bytes (power of two, `>= HEADER_SIZE`-friendly).
    pub block_size: u32,
    /// Number of vectors packed into each fully-occupied block.
    pub vectors_per_block: u32,
}

impl FileHeader {
    /// Bytes occupied by a single vector of this header's dtype/dims.
    #[must_use]
    pub fn vector_bytes(&self) -> usize {
        self.dtype.vector_bytes(self.dimensions as usize)
    }

    /// Serialize the header into a fixed `HEADER_SIZE` buffer (zero-padded).
    #[must_use]
    pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        let mut off = 0;

        let mut put = |bytes: &[u8], off: &mut usize| {
            buf[*off..*off + bytes.len()].copy_from_slice(bytes);
            *off += bytes.len();
        };

        put(&MAGIC, &mut off);
        put(&FORMAT_VERSION.to_le_bytes(), &mut off);
        put(&[dtype_tag(self.dtype)], &mut off);
        // one reserved/alignment byte after the dtype tag
        put(&[0u8], &mut off);
        // two reserved bytes to keep the following u32s 4-byte aligned
        put(&[0u8, 0u8], &mut off);
        put(&self.dimensions.to_le_bytes(), &mut off);
        put(&self.block_size.to_le_bytes(), &mut off);
        put(&self.vectors_per_block.to_le_bytes(), &mut off);
        put(&self.count.to_le_bytes(), &mut off);

        buf
    }

    /// Parse and validate a header from the start of a file.
    ///
    /// # Errors
    /// Returns [`Error::Corrupt`] if the buffer is too small, the magic or
    /// version does not match, the dtype tag is unknown, or the geometry is
    /// internally inconsistent.
    pub fn from_bytes(buf: &[u8]) -> Result<Self> {
        if buf.len() < HEADER_SIZE {
            return Err(Error::corrupt(format!(
                "header buffer too small: {} < {HEADER_SIZE}",
                buf.len()
            )));
        }

        if buf[0..8] != MAGIC {
            return Err(Error::corrupt("bad magic: not a Slate-ANN vector store"));
        }

        let version = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        if version != FORMAT_VERSION {
            return Err(Error::corrupt(format!(
                "unsupported format version {version} (expected {FORMAT_VERSION})"
            )));
        }

        let dtype = dtype_from_tag(buf[12])?;
        // buf[13..16] reserved
        let dimensions = u32::from_le_bytes(buf[16..20].try_into().unwrap());
        let block_size = u32::from_le_bytes(buf[20..24].try_into().unwrap());
        let vectors_per_block = u32::from_le_bytes(buf[24..28].try_into().unwrap());
        let count = u64::from_le_bytes(buf[28..36].try_into().unwrap());

        let header = FileHeader {
            dtype,
            dimensions,
            count,
            block_size,
            vectors_per_block,
        };
        header.validate()?;
        Ok(header)
    }

    /// Check internal consistency of the geometry fields.
    ///
    /// # Errors
    /// Returns [`Error::Corrupt`] if dimensions are zero, the block size
    /// cannot hold a single vector, or `vectors_per_block` disagrees with the
    /// block size and vector width.
    pub fn validate(&self) -> Result<()> {
        if self.dimensions == 0 {
            return Err(Error::corrupt("header dimensions must be non-zero"));
        }
        let vbytes = self.vector_bytes();
        if vbytes == 0 {
            return Err(Error::corrupt("computed vector size is zero"));
        }
        if (self.block_size as usize) < vbytes {
            return Err(Error::corrupt(format!(
                "block_size {} too small for a {vbytes}-byte vector",
                self.block_size
            )));
        }
        let expected_vpb = (self.block_size as usize / vbytes) as u32;
        if self.vectors_per_block != expected_vpb {
            return Err(Error::corrupt(format!(
                "vectors_per_block {} disagrees with block_size/vector_bytes {expected_vpb}",
                self.vectors_per_block
            )));
        }
        if self.vectors_per_block == 0 {
            return Err(Error::corrupt("vectors_per_block computed as zero"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> FileHeader {
        // 64 KiB blocks, 128-dim f32 vectors => 512 bytes each => 128/block.
        let dtype = Dtype::F32;
        let dimensions = 128u32;
        let block_size = 64 * 1024u32;
        let vbytes = dtype.vector_bytes(dimensions as usize);
        let vectors_per_block = (block_size as usize / vbytes) as u32;
        FileHeader {
            dtype,
            dimensions,
            count: 1000,
            block_size,
            vectors_per_block,
        }
    }

    #[test]
    fn round_trips_through_bytes() {
        let h = sample();
        let bytes = h.to_bytes();
        let parsed = FileHeader::from_bytes(&bytes).unwrap();
        assert_eq!(h, parsed);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = sample().to_bytes();
        bytes[0] = b'X';
        let err = FileHeader::from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, Error::Corrupt(_)));
    }

    #[test]
    fn rejects_bad_version() {
        let mut bytes = sample().to_bytes();
        // Corrupt the version field (bytes 8..12).
        bytes[8] = 0xFF;
        let err = FileHeader::from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, Error::Corrupt(_)));
    }

    #[test]
    fn rejects_unknown_dtype_tag() {
        let mut bytes = sample().to_bytes();
        bytes[12] = 0x7F;
        let err = FileHeader::from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, Error::Corrupt(_)));
    }

    #[test]
    fn rejects_inconsistent_vectors_per_block() {
        let mut h = sample();
        h.vectors_per_block += 1;
        let err = h.validate().unwrap_err();
        assert!(matches!(err, Error::Corrupt(_)));
    }

    #[test]
    fn rejects_block_too_small_for_vector() {
        let dtype = Dtype::F32;
        let dimensions = 128u32;
        // 256 bytes can't hold a 512-byte vector.
        let h = FileHeader {
            dtype,
            dimensions,
            count: 1,
            block_size: 256,
            vectors_per_block: 0,
        };
        assert!(h.validate().is_err());
    }

    #[test]
    fn all_dtype_tags_round_trip() {
        for dt in [Dtype::F32, Dtype::F16, Dtype::I8] {
            assert_eq!(dtype_from_tag(dtype_tag(dt)).unwrap(), dt);
        }
    }
}
