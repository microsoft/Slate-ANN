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
    /// the storage layer; this returns only the element payload size.
    #[inline]
    pub const fn vector_bytes(self, dimensions: usize) -> usize {
        self.size_bytes() * dimensions
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
    fn lossiness() {
        assert!(!Dtype::F32.is_lossy());
        assert!(Dtype::F16.is_lossy());
        assert!(Dtype::I8.is_lossy());
    }
}
