//! The I/O backend seam: how raw block bytes are pulled off disk.
//!
//! Two strategies are provided now, behind one trait so the search layer is
//! oblivious to which is in use and the Phase 7 HDD elevator scheduler can be
//! slotted in later:
//!
//! * [`MmapBackend`] — zero-copy reads straight out of a memory map. Each
//!   touched page is faulted in on demand (one 4 KiB read per fault), which
//!   is ideal on SSDs and warm page cache but pathological for cold random
//!   reads on a spinning disk.
//! * [`PreadBackend`] — explicit positioned reads (`read_at`) into a
//!   caller-provided buffer. This issues one large sequential read per block,
//!   which is what a 7200 rpm drive wants: a single seek amortised over the
//!   whole 64 KiB block instead of many page-fault seeks.
//!
//! Both are addressed in whole blocks; the [`crate::reader`] turns a vector
//! index into a `(block, offset-within-block, len)` request.

use std::fs::File;
use std::path::Path;

use slate_core::{Error, Result};

use crate::mmap::{Advice, MmapView};

/// Abstraction over reading a contiguous byte range from the store file.
///
/// Implementors must treat the file as immutable for their lifetime.
pub trait IoBackend: Send + Sync {
    /// Total readable length of the underlying file, in bytes.
    fn len(&self) -> usize;

    /// Whether the underlying file is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read exactly `buf.len()` bytes starting at `offset` into `buf`.
    ///
    /// # Errors
    /// Returns [`Error::Corrupt`] if the range is out of bounds, or
    /// [`Error::Io`] on a read failure.
    fn read_exact_at(&self, offset: usize, buf: &mut [u8]) -> Result<()>;

    /// Hint the expected access pattern for a byte range. Default: no-op.
    ///
    /// # Errors
    /// Implementations backed by `mmap` may return [`Error::Io`].
    fn advise_range(&self, _offset: usize, _len: usize, _advice: Advice) -> Result<()> {
        Ok(())
    }
}

/// Zero-copy backend reading from a memory map.
#[derive(Debug)]
pub struct MmapBackend {
    view: MmapView,
}

impl MmapBackend {
    /// Map `path` read-only.
    ///
    /// # Errors
    /// Returns [`Error::Io`] if opening or mapping fails.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            view: MmapView::open(path)?,
        })
    }

    /// Borrow the underlying view (for callers wanting true zero-copy slices).
    #[must_use]
    pub fn view(&self) -> &MmapView {
        &self.view
    }
}

impl IoBackend for MmapBackend {
    fn len(&self) -> usize {
        self.view.len()
    }

    fn read_exact_at(&self, offset: usize, buf: &mut [u8]) -> Result<()> {
        let src = self.view.slice(offset, buf.len())?;
        buf.copy_from_slice(src);
        Ok(())
    }

    fn advise_range(&self, offset: usize, len: usize, advice: Advice) -> Result<()> {
        self.view.advise_range(offset, len, advice)
    }
}

/// Explicit positioned-read backend (`pread`-style).
#[derive(Debug)]
pub struct PreadBackend {
    file: File,
    len: usize,
}

impl PreadBackend {
    /// Open `path` read-only for positioned reads.
    ///
    /// # Errors
    /// Returns [`Error::Io`] if the file cannot be opened or sized.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let len = file.metadata()?.len() as usize;
        Ok(Self { file, len })
    }
}

impl IoBackend for PreadBackend {
    fn len(&self) -> usize {
        self.len
    }

    fn read_exact_at(&self, offset: usize, buf: &mut [u8]) -> Result<()> {
        let end = offset
            .checked_add(buf.len())
            .ok_or_else(|| Error::corrupt("pread range overflow"))?;
        if end > self.len {
            return Err(Error::corrupt(format!(
                "pread {offset}..{end} out of bounds (len {})",
                self.len
            )));
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            self.file.read_exact_at(buf, offset as u64)?;
        }
        #[cfg(not(unix))]
        {
            use std::io::{Read, Seek, SeekFrom};
            // Fallback for non-Unix: seek+read. Needs &mut, so clone a handle.
            let mut f = self.file.try_clone()?;
            f.seek(SeekFrom::Start(offset as u64))?;
            f.read_exact(buf)?;
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

    fn check_backend(b: &dyn IoBackend, data: &[u8]) {
        assert_eq!(b.len(), data.len());
        assert!(!b.is_empty());
        let mut buf = [0u8; 16];
        b.read_exact_at(0, &mut buf).unwrap();
        assert_eq!(&buf, &data[0..16]);
        b.read_exact_at(100, &mut buf).unwrap();
        assert_eq!(&buf, &data[100..116]);
        // Out-of-bounds read is rejected.
        let mut tail = [0u8; 32];
        assert!(b.read_exact_at(data.len() - 4, &mut tail).is_err());
        // advise_range within bounds must succeed.
        b.advise_range(0, 64, Advice::WillNeed).unwrap();
    }

    #[test]
    fn mmap_and_pread_backends_agree() {
        let data: Vec<u8> = (0..=255u8).cycle().take(4096).collect();
        let f = temp_file_with(&data);

        let mmap = MmapBackend::open(f.path()).unwrap();
        check_backend(&mmap, &data);

        let pread = PreadBackend::open(f.path()).unwrap();
        check_backend(&pread, &data);

        // Both yield identical bytes for the same request.
        let mut a = [0u8; 64];
        let mut b = [0u8; 64];
        mmap.read_exact_at(1000, &mut a).unwrap();
        pread.read_exact_at(1000, &mut b).unwrap();
        assert_eq!(a, b);
    }
}
