//! Custom memory-mapping layer over [`memmap2`].
//!
//! This wraps a read-only `Mmap` with:
//!
//! * typed, bounds-checked slice views over the mapped bytes, and
//! * `madvise` access-pattern hints ([`Advice`]) so callers can tell the
//!   kernel whether the workload is sequential (build/scan) or random
//!   (graph traversal) — the single most important knob separating HDD and
//!   SSD behaviour, since random page faults on a spinning disk each cost a
//!   full seek.
//!
//! The map is read-only: the writer produces files with explicit buffered
//! writes (see [`crate::layout`]); the reader maps them for zero-copy access.

use std::fs::File;
use std::path::Path;

use memmap2::{Mmap, MmapOptions};
use slate_core::{Error, Result};

/// Access-pattern hint forwarded to the kernel via `madvise(2)`.
///
/// These map onto `memmap2::Advice` / POSIX `posix_madvise` values. They are
/// advisory: the kernel may ignore them, and on platforms without `madvise`
/// they are no-ops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Advice {
    /// No expectation; let the kernel use its default read-ahead.
    Normal,
    /// Pages will be accessed in order — enables aggressive read-ahead.
    /// Use for the build pipeline and full-store scans.
    Sequential,
    /// Pages will be accessed in random order — suppresses read-ahead so the
    /// disk is not made to pre-fetch blocks a graph walk will never touch.
    Random,
    /// These pages will be needed soon — ask the kernel to pre-fault them.
    WillNeed,
}

impl Advice {
    #[cfg(unix)]
    fn to_memmap(self) -> memmap2::Advice {
        match self {
            Advice::Normal => memmap2::Advice::Normal,
            Advice::Sequential => memmap2::Advice::Sequential,
            Advice::Random => memmap2::Advice::Random,
            Advice::WillNeed => memmap2::Advice::WillNeed,
        }
    }
}

/// A read-only memory map with typed views and advice control.
#[derive(Debug)]
pub struct MmapView {
    mmap: Mmap,
}

impl MmapView {
    /// Map an entire file read-only.
    ///
    /// # Errors
    /// Returns [`Error::Io`] if the file cannot be opened or mapped.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        Self::from_file(&file)
    }

    /// Map an already-open file read-only.
    ///
    /// # Errors
    /// Returns [`Error::Io`] if the mapping fails.
    pub fn from_file(file: &File) -> Result<Self> {
        // SAFETY: the file is opened read-only and the `Mmap` borrows it for
        // the duration of the call. The resulting mapping is treated as
        // immutable for its whole lifetime; we never expose a `&mut` into it.
        let mmap = unsafe { MmapOptions::new().map(file)? };
        Ok(Self { mmap })
    }

    /// Total length of the mapping in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    /// Whether the mapping is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }

    /// The whole mapping as a byte slice.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap
    }

    /// A bounds-checked byte sub-slice `[offset, offset + len)`.
    ///
    /// # Errors
    /// Returns [`Error::Corrupt`] if the requested range is out of bounds.
    pub fn slice(&self, offset: usize, len: usize) -> Result<&[u8]> {
        let end = offset
            .checked_add(len)
            .ok_or_else(|| Error::corrupt("mmap slice range overflow"))?;
        self.mmap.get(offset..end).ok_or_else(|| {
            Error::corrupt(format!(
                "mmap slice {offset}..{end} out of bounds (len {})",
                self.mmap.len()
            ))
        })
    }

    /// Apply an access-pattern [`Advice`] to the entire mapping.
    ///
    /// On non-Unix platforms this is a no-op that still validates the call.
    ///
    /// # Errors
    /// Returns [`Error::Io`] if the underlying `madvise` call fails.
    pub fn advise(&self, advice: Advice) -> Result<()> {
        #[cfg(unix)]
        {
            self.mmap.advise(advice.to_memmap())?;
        }
        #[cfg(not(unix))]
        {
            let _ = advice;
        }
        Ok(())
    }

    /// Apply an access-pattern [`Advice`] to a byte sub-range.
    ///
    /// Lets a caller hint, for example, `WillNeed` on a specific block about
    /// to be read while leaving the rest of the map `Random`.
    ///
    /// # Errors
    /// Returns [`Error::Corrupt`] if the range is out of bounds, or
    /// [`Error::Io`] if the `madvise` call fails.
    pub fn advise_range(&self, offset: usize, len: usize, advice: Advice) -> Result<()> {
        // Validate the range against the mapping bounds regardless of platform.
        let _ = self.slice(offset, len)?;
        #[cfg(unix)]
        {
            self.mmap.advise_range(advice.to_memmap(), offset, len)?;
        }
        #[cfg(not(unix))]
        {
            let _ = advice;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_file_with(bytes: &[u8]) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(bytes).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn maps_and_reads_bytes() {
        let data: Vec<u8> = (0..=255u8).collect();
        let f = temp_file_with(&data);
        let view = MmapView::open(f.path()).unwrap();
        assert_eq!(view.len(), 256);
        assert!(!view.is_empty());
        assert_eq!(view.as_bytes(), &data[..]);
        assert_eq!(view.slice(10, 4).unwrap(), &data[10..14]);
    }

    #[test]
    fn slice_out_of_bounds_errors() {
        let f = temp_file_with(&[1, 2, 3, 4]);
        let view = MmapView::open(f.path()).unwrap();
        assert!(view.slice(2, 10).is_err());
        assert!(view.slice(usize::MAX, 1).is_err());
    }

    #[test]
    fn advice_calls_succeed() {
        let data = vec![7u8; 8192];
        let f = temp_file_with(&data);
        let view = MmapView::open(f.path()).unwrap();
        // All advice variants should be accepted on this platform.
        view.advise(Advice::Normal).unwrap();
        view.advise(Advice::Sequential).unwrap();
        view.advise(Advice::Random).unwrap();
        view.advise(Advice::WillNeed).unwrap();
        // Sub-range advice within bounds.
        view.advise_range(0, 4096, Advice::WillNeed).unwrap();
        // Out-of-range advice is rejected.
        assert!(view.advise_range(4096, 1 << 20, Advice::WillNeed).is_err());
    }
}
