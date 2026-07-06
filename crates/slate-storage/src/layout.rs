//! Block-packing geometry and the on-disk writer.
//!
//! The layout maps a dense vector index `0..count` onto `(block, slot)`
//! coordinates and byte offsets within the file. Vectors are packed
//! back-to-back inside a block; the tail of each block is zero padding so no
//! vector crosses a block boundary.
//!
//! Phase 2 stores `f32` vectors only. The `Dtype` is still recorded in the
//! header so f16/i8 layouts (Phase 7) can reuse this geometry unchanged —
//! only the per-element width differs.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use slate_core::{Dtype, Error, Result};

use crate::format::{FileHeader, HEADER_SIZE};

/// Geometry of a block-packed vector file.
///
/// Pure arithmetic over `(dtype, dimensions, block_size)`; carries no data
/// and is cheap to copy. Used by both the writer and the reader so they
/// agree on every byte offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockLayout {
    dtype: Dtype,
    dimensions: usize,
    block_size: usize,
    vector_bytes: usize,
    vectors_per_block: usize,
}

impl BlockLayout {
    /// Construct a layout, validating that a block can hold at least one
    /// vector and that `block_size` is a power of two.
    ///
    /// # Errors
    /// Returns [`Error::InvalidConfig`] if `dimensions` is zero, `block_size`
    /// is not a power of two, or a single vector does not fit in a block.
    pub fn new(dtype: Dtype, dimensions: usize, block_size: usize) -> Result<Self> {
        if dimensions == 0 {
            return Err(Error::invalid_config("dimensions must be non-zero"));
        }
        if !block_size.is_power_of_two() {
            return Err(Error::invalid_config(format!(
                "block_size {block_size} must be a power of two"
            )));
        }
        let vector_bytes = dtype.stored_vector_bytes(dimensions);
        if vector_bytes == 0 {
            return Err(Error::invalid_config("vector_bytes computed as zero"));
        }
        if block_size < vector_bytes {
            return Err(Error::invalid_config(format!(
                "block_size {block_size} too small for {vector_bytes}-byte vector"
            )));
        }
        let vectors_per_block = block_size / vector_bytes;
        Ok(Self {
            dtype,
            dimensions,
            block_size,
            vector_bytes,
            vectors_per_block,
        })
    }

    /// Element type of stored vectors.
    #[must_use]
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    /// Vector dimensionality.
    #[must_use]
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Block size in bytes.
    #[must_use]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Bytes occupied by one vector.
    #[must_use]
    pub fn vector_bytes(&self) -> usize {
        self.vector_bytes
    }

    /// Number of vectors in a fully-occupied block.
    #[must_use]
    pub fn vectors_per_block(&self) -> usize {
        self.vectors_per_block
    }

    /// Number of blocks required to store `count` vectors.
    #[must_use]
    pub fn blocks_for(&self, count: usize) -> usize {
        count.div_ceil(self.vectors_per_block)
    }

    /// Total file size in bytes for `count` vectors (header + whole blocks).
    #[must_use]
    pub fn file_size_for(&self, count: usize) -> usize {
        HEADER_SIZE + self.blocks_for(count) * self.block_size
    }

    /// Block index holding the vector at dense index `i`.
    #[must_use]
    pub fn block_of(&self, i: usize) -> usize {
        i / self.vectors_per_block
    }

    /// Slot (position within its block) of the vector at dense index `i`.
    #[must_use]
    pub fn slot_of(&self, i: usize) -> usize {
        i % self.vectors_per_block
    }

    /// Absolute byte offset of block `block` within the file.
    #[must_use]
    pub fn block_offset(&self, block: usize) -> usize {
        HEADER_SIZE + block * self.block_size
    }

    /// Absolute byte offset of the vector at dense index `i` within the file.
    #[must_use]
    pub fn vector_offset(&self, i: usize) -> usize {
        self.block_offset(self.block_of(i)) + self.slot_of(i) * self.vector_bytes
    }

    /// Byte span `(offset, len)` covering the contiguous run of dense indices
    /// `first..=last` inclusive (`first <= last`).
    ///
    /// Because [`Self::vector_offset`] is monotonic in the dense index, the run
    /// occupies one contiguous range from the start of `first` to the end of
    /// `last`. The span may include block-boundary padding between the two if
    /// they straddle adjacent blocks; it is still a single sequential range the
    /// disk head streams without re-seeking. The elevator scheduler reads one
    /// such span per coalesced run.
    #[must_use]
    pub fn run_span(&self, first: usize, last: usize) -> (usize, usize) {
        debug_assert!(first <= last, "run_span requires first <= last");
        let start = self.vector_offset(first);
        let end = self.vector_offset(last) + self.vector_bytes;
        (start, end - start)
    }

    /// Build the matching [`FileHeader`] for `count` vectors.
    #[must_use]
    pub fn header(&self, count: u64) -> FileHeader {
        FileHeader {
            dtype: self.dtype,
            dimensions: self.dimensions as u32,
            count,
            block_size: self.block_size as u32,
            vectors_per_block: self.vectors_per_block as u32,
        }
    }
}

/// Streaming writer that packs `f32` vectors into the block layout.
///
/// Vectors are appended in order; their dense index is the order of
/// insertion (0, 1, 2, ...). The caller is responsible for mapping its own
/// [`slate_core::VectorId`]s to that dense order and persisting the map (the
/// reader takes the dense index directly).
pub struct StoreWriter<W: Write> {
    layout: BlockLayout,
    writer: W,
    count: usize,
    /// Bytes already written into the current (partial) block.
    block_cursor: usize,
}

impl StoreWriter<BufWriter<File>> {
    /// Create a new store file at `path`, reserving space for the header.
    ///
    /// The header is rewritten with the final `count` on [`Self::finish`].
    ///
    /// # Errors
    /// Returns [`Error::Io`] on filesystem errors.
    pub fn create(path: impl AsRef<Path>, layout: BlockLayout) -> Result<Self> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        // Reserve the header region with a placeholder; final header is
        // written by `finish` via seek-back through the inner File.
        let placeholder = [0u8; HEADER_SIZE];
        writer.write_all(&placeholder)?;
        Ok(Self {
            layout,
            writer,
            count: 0,
            block_cursor: 0,
        })
    }
}

impl<W: Write> StoreWriter<W> {
    /// Append one vector, encoding it into the layout's dtype. Its dense index
    /// is returned. The caller always supplies `f32`; `f16`/`i8` layouts narrow
    /// it on the way to disk.
    ///
    /// # Errors
    /// Returns [`Error::DimensionMismatch`] if the vector length differs from
    /// the layout's dimensionality, or [`Error::Io`] on write failure.
    pub fn push(&mut self, vector: &[f32]) -> Result<usize> {
        if vector.len() != self.layout.dimensions {
            return Err(Error::DimensionMismatch {
                expected: self.layout.dimensions,
                got: vector.len(),
            });
        }

        // If this vector would not fit in the remaining space of the current
        // block, pad the block out and start a new one.
        let remaining = self.layout.block_size - self.block_cursor;
        if remaining < self.layout.vector_bytes {
            self.pad(remaining)?;
            self.block_cursor = 0;
        }

        // Encode into the on-disk dtype. The BufWriter coalesces the small
        // element-wise writes used by the narrow paths.
        match self.layout.dtype {
            Dtype::F32 => {
                // bytemuck gives a zero-copy little-endian view on LE targets.
                let bytes: &[u8] = bytemuck::cast_slice(vector);
                self.writer.write_all(bytes)?;
            }
            Dtype::F16 => {
                for x in vector {
                    self.writer
                        .write_all(&half::f16::from_f32(*x).to_le_bytes())?;
                }
            }
            Dtype::I8 => {
                let max_abs = vector.iter().fold(0.0f32, |m, x| m.max(x.abs()));
                let scale = if max_abs == 0.0 { 0.0 } else { max_abs / 127.0 };
                self.writer.write_all(&scale.to_le_bytes())?;
                if scale == 0.0 {
                    for _ in vector {
                        self.writer.write_all(&[0u8])?;
                    }
                } else {
                    let inv = 1.0 / scale;
                    for x in vector {
                        let q = (x * inv).round().clamp(-127.0, 127.0) as i8;
                        self.writer.write_all(&[q as u8])?;
                    }
                }
            }
        }
        self.block_cursor += self.layout.vector_bytes;
        let index = self.count;
        self.count += 1;
        Ok(index)
    }

    /// Number of vectors written so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Whether no vectors have been written.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Write `n` zero bytes (block tail padding).
    fn pad(&mut self, n: usize) -> Result<()> {
        const ZEROS: [u8; 1024] = [0u8; 1024];
        let mut left = n;
        while left > 0 {
            let chunk = left.min(ZEROS.len());
            self.writer.write_all(&ZEROS[..chunk])?;
            left -= chunk;
        }
        Ok(())
    }

    /// Pad the final partial block and flush, returning the header that
    /// describes the file. The caller must write this header at offset 0.
    ///
    /// For the [`File`]-backed constructor, prefer [`Self::finish`], which
    /// also seeks back and writes the header in place.
    ///
    /// # Errors
    /// Returns [`Error::Io`] on write failure.
    pub fn finish_blocks(&mut self) -> Result<FileHeader> {
        if self.block_cursor > 0 {
            let remaining = self.layout.block_size - self.block_cursor;
            self.pad(remaining)?;
            self.block_cursor = 0;
        }
        self.writer.flush()?;
        Ok(self.layout.header(self.count as u64))
    }
}

impl StoreWriter<BufWriter<File>> {
    /// Finalize the file: pad the last block, then seek back and write the
    /// real header (with the final vector count) at offset 0.
    ///
    /// # Errors
    /// Returns [`Error::Io`] on write or seek failure.
    pub fn finish(mut self) -> Result<FileHeader> {
        use std::io::Seek;
        let header = self.finish_blocks()?;
        let mut file = self
            .writer
            .into_inner()
            .map_err(|e| Error::Io(e.into_error()))?;
        file.seek(std::io::SeekFrom::Start(0))?;
        file.write_all(&header.to_bytes())?;
        file.flush()?;
        Ok(header)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_basic() {
        // 1 KiB blocks, 4-dim f32 => 16 bytes/vector => 64 vectors/block.
        let l = BlockLayout::new(Dtype::F32, 4, 1024).unwrap();
        assert_eq!(l.vector_bytes(), 16);
        assert_eq!(l.vectors_per_block(), 64);
        assert_eq!(l.blocks_for(0), 0);
        assert_eq!(l.blocks_for(1), 1);
        assert_eq!(l.blocks_for(64), 1);
        assert_eq!(l.blocks_for(65), 2);
        assert_eq!(l.block_of(63), 0);
        assert_eq!(l.block_of(64), 1);
        assert_eq!(l.slot_of(65), 1);
        assert_eq!(l.block_offset(0), HEADER_SIZE);
        assert_eq!(l.block_offset(1), HEADER_SIZE + 1024);
        // vector 64 is slot 0 of block 1
        assert_eq!(l.vector_offset(64), HEADER_SIZE + 1024);
        // vector 65 is slot 1 of block 1
        assert_eq!(l.vector_offset(65), HEADER_SIZE + 1024 + 16);
    }

    #[test]
    fn rejects_non_power_of_two_block() {
        assert!(BlockLayout::new(Dtype::F32, 4, 1000).is_err());
    }

    #[test]
    fn rejects_block_smaller_than_vector() {
        // 4-dim f32 = 16 bytes; block of 8 can't hold it.
        assert!(BlockLayout::new(Dtype::F32, 4, 8).is_err());
    }

    #[test]
    fn file_size_accounts_for_header_and_padding() {
        let l = BlockLayout::new(Dtype::F32, 4, 1024).unwrap();
        // 65 vectors => 2 blocks => header + 2*1024.
        assert_eq!(l.file_size_for(65), HEADER_SIZE + 2 * 1024);
    }

    #[test]
    fn push_rejects_wrong_dimensions() {
        let l = BlockLayout::new(Dtype::F32, 4, 1024).unwrap();
        let mut w = StoreWriter {
            layout: l,
            writer: Vec::<u8>::new(),
            count: 0,
            block_cursor: 0,
        };
        let err = w.push(&[1.0, 2.0, 3.0]).unwrap_err();
        assert!(matches!(
            err,
            Error::DimensionMismatch {
                expected: 4,
                got: 3
            }
        ));
    }

    #[test]
    fn push_into_vec_packs_and_pads_blocks() {
        // 2-dim f32 = 8 bytes; 32-byte block => 4 vectors/block.
        let l = BlockLayout::new(Dtype::F32, 2, 32).unwrap();
        let mut w = StoreWriter {
            layout: l,
            writer: Vec::<u8>::new(),
            count: 0,
            block_cursor: 0,
        };
        // Write 5 vectors => spills into a 2nd block.
        for i in 0..5 {
            let v = [i as f32, (i + 100) as f32];
            assert_eq!(w.push(&v).unwrap(), i);
        }
        let header = w.finish_blocks().unwrap();
        assert_eq!(header.count, 5);
        // Buffer must be exactly 2 full blocks (no header here; Vec writer).
        assert_eq!(w.writer.len(), 2 * 32);
        // Vector 4 sits at slot 0 of block 1: bytes [32..40].
        let v4: &[f32] = bytemuck::cast_slice(&w.writer[32..40]);
        assert_eq!(v4, &[4.0, 104.0]);
        // Slot 1 of block 0 (vector 1): bytes [8..16].
        let v1: &[f32] = bytemuck::cast_slice(&w.writer[8..16]);
        assert_eq!(v1, &[1.0, 101.0]);
    }
}
