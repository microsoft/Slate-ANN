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

    /// Scatter one contiguous file region starting at `offset` into the
    /// destination buffers `bufs`, filled in order (`preadv`-style).
    ///
    /// The source bytes `offset .. offset + Σ bufs[i].len()` are read as a
    /// single contiguous run and distributed across the destination slices.
    /// Inter-slot gaps must be modelled by the caller as throwaway sink
    /// buffers in `bufs`; this method does not skip bytes.
    ///
    /// The default implementation issues one [`Self::read_exact_at`] per
    /// destination buffer; backends that can do a single vectored syscall
    /// override this.
    ///
    /// # Errors
    /// Returns [`Error::Corrupt`] if the combined range is out of bounds, or
    /// [`Error::Io`] on a read failure.
    fn read_vectored_at(&self, offset: usize, bufs: &mut [std::io::IoSliceMut<'_>]) -> Result<()> {
        let mut cur = offset;
        for buf in bufs.iter_mut() {
            self.read_exact_at(cur, buf)?;
            cur += buf.len();
        }
        Ok(())
    }

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

    #[cfg(unix)]
    fn read_vectored_at(&self, offset: usize, bufs: &mut [std::io::IoSliceMut<'_>]) -> Result<()> {
        use std::os::unix::io::AsRawFd;

        let total: usize = bufs.iter().map(|b| b.len()).sum();
        let end = offset
            .checked_add(total)
            .ok_or_else(|| Error::corrupt("preadv range overflow"))?;
        if end > self.len {
            return Err(Error::corrupt(format!(
                "preadv {offset}..{end} out of bounds (len {})",
                self.len
            )));
        }

        let fd = self.file.as_raw_fd();
        // One `preadv` per syscall; loop only to absorb short reads (rare on a
        // regular file) by advancing the iovec list and the file offset. The
        // std positioned-vectored API is unstable on this toolchain, so we call
        // `libc::preadv` directly through the raw fd.
        let mut cur = offset;
        let mut bufs: &mut [std::io::IoSliceMut<'_>] = bufs;
        while !bufs.is_empty() {
            // Rebuild the iovec view of the (possibly shrunken) destinations.
            let iovs: Vec<libc::iovec> = bufs
                .iter_mut()
                .map(|b| libc::iovec {
                    iov_base: b.as_mut_ptr().cast::<libc::c_void>(),
                    iov_len: b.len(),
                })
                .collect();
            // SAFETY: `fd` is valid for the lifetime of `self.file`; each iovec
            // points to a live, writable buffer of exactly `iov_len` bytes (it
            // borrows `bufs` mutably); `iovs.len()` is the slice length which
            // fits `c_int` for any realistic run. `preadv` does not retain the
            // pointers past the call.
            let n = unsafe {
                libc::preadv(
                    fd,
                    iovs.as_ptr(),
                    iovs.len() as libc::c_int,
                    cur as libc::off_t,
                )
            };
            if n < 0 {
                return Err(Error::Io(std::io::Error::last_os_error()));
            }
            if n == 0 {
                return Err(Error::corrupt("unexpected EOF in vectored read"));
            }
            let n = n as usize;
            cur += n;
            std::io::IoSliceMut::advance_slices(&mut bufs, n);
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

    fn check_vectored(b: &dyn IoBackend, data: &[u8]) {
        // Scatter a 4000-byte contiguous region at offset 32 into three uneven
        // destination buffers; the concatenation must equal one flat read.
        let mut d0 = vec![0u8; 1000];
        let mut d1 = vec![0u8; 1234];
        let mut d2 = vec![0u8; 1766];
        {
            let mut bufs = [
                std::io::IoSliceMut::new(&mut d0),
                std::io::IoSliceMut::new(&mut d1),
                std::io::IoSliceMut::new(&mut d2),
            ];
            b.read_vectored_at(32, &mut bufs).unwrap();
        }
        let mut flat = vec![0u8; 4000];
        b.read_exact_at(32, &mut flat).unwrap();
        assert_eq!(&d0[..], &flat[0..1000]);
        assert_eq!(&d1[..], &flat[1000..2234]);
        assert_eq!(&d2[..], &flat[2234..4000]);
        assert_eq!(&flat[..], &data[32..4032]);

        // Out-of-bounds combined range is rejected.
        let mut tail = vec![0u8; 64];
        let mut bufs = [std::io::IoSliceMut::new(&mut tail)];
        assert!(b.read_vectored_at(data.len() - 4, &mut bufs).is_err());
    }

    #[test]
    fn vectored_reads_match_flat_reads_on_both_backends() {
        let data: Vec<u8> = (0..=255u8).cycle().take(8192).collect();
        let f = temp_file_with(&data);

        let mmap = MmapBackend::open(f.path()).unwrap();
        check_vectored(&mmap, &data);

        let pread = PreadBackend::open(f.path()).unwrap();
        check_vectored(&pread, &data);
    }
}
