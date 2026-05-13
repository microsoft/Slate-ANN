//! On-disk vector element types.

use serde::{Deserialize, Serialize};

/// Element type used to store exact vectors on disk.
///
/// Slate-ANN streams exact vectors from disk for re-ranking, so the on-disk
/// element type is the primary knob for **read volume per query** — the
/// dominant cost on low-bandwidth stores like 7200rpm HDDs. Narrower types cut
/// I/O at some precision cost:
///
/// | Dtype | Bytes/dim | Relative I/O | Precision                       |
/// |-------|-----------|--------------|---------------------------------|
/// | `F32` | 4         | 1.0x         | full                            |
/// | `F16` | 2         | 0.5x         | ~3 decimal digits               |
/// | `I8`  | 1         | 0.25x        | scalar-quantized (per-vector s) |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum Dtype {
    /// IEEE-754 single precision (4 bytes per dimension).
    #[default]
    F32,
    /// IEEE-754 half precision (2 bytes per dimension).
    F16,
    /// 8-bit signed integer, scalar-quantized with a stored scale factor.
    I8,
}

impl Dtype {
    /// Number of bytes one element of this dtype occupies on disk.
    #[inline]
    pub const fn size_bytes(self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::F16 => 2,
            Dtype::I8 => 1,
        }
    }

    /// Byte length of a `dimensions`-long vector stored in this dtype.
    ///
    /// Note: `I8` vectors additionally carry a small per-vector scale factor in
    /// the storage layer; this returns only the element payload size. Use
    /// [`stored_vector_bytes`](Self::stored_vector_bytes) for the full on-disk
    /// footprint including that metadata.
    #[inline]
    pub const fn vector_bytes(self, dimensions: usize) -> usize {
        self.size_bytes() * dimensions
    }

    /// Per-vector metadata bytes the storage layer prepends to each vector slot.
    ///
    /// `I8` stores a single `f32` scale factor inline at the start of every
    /// vector; `F32` and `F16` carry no per-vector metadata.
    #[inline]
    pub const fn metadata_bytes(self) -> usize {
        match self {
            Dtype::I8 => core::mem::size_of::<f32>(),
            Dtype::F32 | Dtype::F16 => 0,
        }
    }

    /// Full on-disk footprint of one `dimensions`-long vector slot: the element
    /// payload plus any per-vector metadata (the `I8` scale factor).
    ///
    /// This is the stride the block geometry uses, and the number of bytes a
    /// single-vector read actually moves off the store.
    #[inline]
    pub const fn stored_vector_bytes(self, dimensions: usize) -> usize {
        self.vector_bytes(dimensions) + self.metadata_bytes()
    }

    /// Whether this dtype is lossy relative to the caller's `f32` input.
    #[inline]
    pub const fn is_lossy(self) -> bool {
        matches!(self, Dtype::F16 | Dtype::I8)
    }

    /// Lower-case identifier used in the on-disk metadata file.
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Dtype::F32 => "f32",
            Dtype::F16 => "f16",
            Dtype::I8 => "i8",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn element_sizes() {
        assert_eq!(Dtype::F32.size_bytes(), 4);
        assert_eq!(Dtype::F16.size_bytes(), 2);
        assert_eq!(Dtype::I8.size_bytes(), 1);
    }

    #[test]
    fn vector_byte_lengths() {
        assert_eq!(Dtype::F32.vector_bytes(768), 768 * 4);
        assert_eq!(Dtype::F16.vector_bytes(768), 768 * 2);
        assert_eq!(Dtype::I8.vector_bytes(768), 768);
    }

    #[test]
    fn metadata_and_stored_footprint() {
        // Only I8 carries per-vector metadata (a single f32 scale).
        assert_eq!(Dtype::F32.metadata_bytes(), 0);
        assert_eq!(Dtype::F16.metadata_bytes(), 0);
        assert_eq!(Dtype::I8.metadata_bytes(), 4);

        // Stored footprint = payload + metadata.
        assert_eq!(Dtype::F32.stored_vector_bytes(768), 768 * 4);
        assert_eq!(Dtype::F16.stored_vector_bytes(768), 768 * 2);
        assert_eq!(Dtype::I8.stored_vector_bytes(768), 768 + 4);

        // Narrowing is strictly monotone in footprint even at small dims where
        // the I8 scale is most visible.
        let d = 16;
        assert!(
            Dtype::I8.stored_vector_bytes(d) < Dtype::F16.stored_vector_bytes(d)
                && Dtype::F16.stored_vector_bytes(d) < Dtype::F32.stored_vector_bytes(d)
        );
    }

    #[test]
    fn lossiness() {
        assert!(!Dtype::F32.is_lossy());
        assert!(Dtype::F16.is_lossy());
        assert!(Dtype::I8.is_lossy());
    }
}
