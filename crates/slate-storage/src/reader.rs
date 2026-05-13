//! The read side: open a store file and fetch exact vectors by dense index.
//!
//! [`VectorStore`] is generic over the [`IoBackend`], so the same reader
//! serves both the zero-copy [`MmapBackend`] and the explicit-read
//! [`PreadBackend`]; the Phase 7 elevator scheduler will be a third backend.
//!
//! Phase 2 decodes `f32` only. The header records the dtype so later phases
//! can dispatch f16/i8 decode here without changing the file format.

use std::path::Path;

use slate_core::{Dtype, Error, Result, VectorId};

use crate::format::{FileHeader, HEADER_SIZE};
use crate::io::{IoBackend, MmapBackend, PreadBackend};
use crate::layout::BlockLayout;
use crate::mmap::Advice;

/// A read-only, block-packed exact-vector store.
///
/// Dense indices `0..count` address vectors in insertion order. Mapping
/// application-level [`VectorId`]s onto that dense order is the index layer's
/// responsibility; convenience methods accepting a [`VectorId`] treat its
/// numeric value as the dense index.
#[derive(Debug)]
pub struct VectorStore<B: IoBackend> {
    backend: B,
    header: FileHeader,
    layout: BlockLayout,
}

impl VectorStore<MmapBackend> {
    /// Open a store at `path` using the memory-map backend (SSD/warm-cache
    /// friendly, zero-copy).
    ///
    /// # Errors
    /// Returns [`Error::Io`] / [`Error::Corrupt`] on open or validation
    /// failure.
    pub fn open_mmap(path: impl AsRef<Path>) -> Result<Self> {
        Self::with_backend(MmapBackend::open(path)?)
    }
}

impl VectorStore<PreadBackend> {
    /// Open a store at `path` using the positioned-read backend (HDD friendly,
    /// one large read per block).
    ///
    /// # Errors
    /// Returns [`Error::Io`] / [`Error::Corrupt`] on open or validation
    /// failure.
    pub fn open_pread(path: impl AsRef<Path>) -> Result<Self> {
        Self::with_backend(PreadBackend::open(path)?)
    }
}

impl<B: IoBackend> VectorStore<B> {
    /// Build a store over an arbitrary backend, parsing and validating the
    /// header and checking the file is large enough for the claimed geometry.
    ///
    /// # Errors
    /// Returns [`Error::Corrupt`] if the header is invalid or the file is
    /// shorter than the header says it should be.
    pub fn with_backend(backend: B) -> Result<Self> {
        let mut header_buf = [0u8; HEADER_SIZE];
        backend.read_exact_at(0, &mut header_buf)?;
        let header = FileHeader::from_bytes(&header_buf)?;

        let layout = BlockLayout::new(
            header.dtype,
            header.dimensions as usize,
            header.block_size as usize,
        )?;

        // Ensure the backing file is large enough for the claimed vectors.
        let need = layout.file_size_for(header.count as usize);
        if backend.len() < need {
            return Err(Error::corrupt(format!(
                "file too small: have {} bytes, need {need} for {} vectors",
                backend.len(),
                header.count
            )));
        }

        Ok(Self {
            backend,
            header,
            layout,
        })
    }

    /// Number of stored vectors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.header.count as usize
    }

    /// Whether the store holds no vectors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.header.count == 0
    }

    /// Vector dimensionality.
    #[must_use]
    pub fn dimensions(&self) -> usize {
        self.header.dimensions as usize
    }

    /// Element type of stored vectors.
    #[must_use]
    pub fn dtype(&self) -> Dtype {
        self.header.dtype
    }

    /// The parsed file header.
    #[must_use]
    pub fn header(&self) -> &FileHeader {
        &self.header
    }

    /// The block layout/geometry.
    #[must_use]
    pub fn layout(&self) -> &BlockLayout {
        &self.layout
    }

    /// Borrow the underlying I/O backend.
    #[must_use]
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Fetch the vector at dense `index` into a freshly allocated `f32` `Vec`.
    ///
    /// Narrow stores are decoded to `f32`.
    ///
    /// # Errors
    /// Returns [`Error::NotFound`] if `index >= len`, or an I/O error from the
    /// backend.
    pub fn get(&self, index: usize) -> Result<Vec<f32>> {
        let mut out = vec![0.0f32; self.dimensions()];
        self.get_into(index, &mut out)?;
        Ok(out)
    }

    /// Fetch the vector at dense `index` into a caller-provided `f32` buffer,
    /// avoiding allocation on the f32 hot path.
    ///
    /// Narrow stores (`f16`/`i8`) are decoded to `f32` here, so callers always
    /// receive `f32` regardless of the on-disk dtype.
    ///
    /// # Errors
    /// Returns [`Error::NotFound`] if `index >= len`,
    /// [`Error::DimensionMismatch`] if `out.len() != dimensions`, or an I/O
    /// error from the backend.
    pub fn get_into(&self, index: usize, out: &mut [f32]) -> Result<()> {
        if index >= self.len() {
            return Err(Error::NotFound(VectorId::new(index as u64)));
        }
        if out.len() != self.dimensions() {
            return Err(Error::DimensionMismatch {
                expected: self.dimensions(),
                got: out.len(),
            });
        }
        let offset = self.layout.vector_offset(index);
        match self.header.dtype {
            Dtype::F32 => {
                // Zero-copy: read straight into the output buffer's bytes.
                let dst: &mut [u8] = bytemuck::cast_slice_mut(out);
                self.backend.read_exact_at(offset, dst)?;
            }
            Dtype::F16 => {
                // Read the narrow slot, then widen to f32.
                let mut buf = vec![0u8; self.layout.vector_bytes()];
                self.backend.read_exact_at(offset, &mut buf)?;
                crate::codec::decode_f16(&buf, out)?;
            }
            Dtype::I8 => {
                let mut buf = vec![0u8; self.layout.vector_bytes()];
                self.backend.read_exact_at(offset, &mut buf)?;
                crate::codec::decode_i8(&buf, out)?;
            }
        }
        Ok(())
    }

    /// Convenience: fetch by [`VectorId`], treating its value as dense index.
    ///
    /// # Errors
    /// As [`Self::get`].
    pub fn get_id(&self, id: VectorId) -> Result<Vec<f32>> {
        self.get(id.as_index())
    }

    /// Hint the access pattern for the block holding `index` (e.g. `WillNeed`
    /// just before a fetch, or `Random` for graph traversal).
    ///
    /// # Errors
    /// Returns an I/O error from the backend; a no-op for backends without
    /// `madvise` support.
    pub fn advise_vector(&self, index: usize, advice: Advice) -> Result<()> {
        if index >= self.len() {
            return Err(Error::NotFound(VectorId::new(index as u64)));
        }
        let block = self.layout.block_of(index);
        let offset = self.layout.block_offset(block);
        self.backend
            .advise_range(offset, self.layout.block_size(), advice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::StoreWriter;
    use slate_core::StorageParams;

    /// Write `count` deterministic vectors of `dims` dims into a temp file and
    /// return the path (kept alive by the returned `TempDir`).
    fn write_store(dims: usize, block_size: usize, count: usize) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vectors.bin");
        let layout = BlockLayout::new(Dtype::F32, dims, block_size).unwrap();
        let mut w = StoreWriter::create(&path, layout).unwrap();
        for i in 0..count {
            let v: Vec<f32> = (0..dims).map(|d| (i * 1000 + d) as f32).collect();
            w.push(&v).unwrap();
        }
        let header = w.finish().unwrap();
        assert_eq!(header.count, count as u64);
        (dir, path)
    }

    fn expected(i: usize, dims: usize) -> Vec<f32> {
        (0..dims).map(|d| (i * 1000 + d) as f32).collect()
    }

    #[test]
    fn mmap_round_trip_reads_every_vector() {
        let dims = 16;
        let (_dir, path) = write_store(dims, 1024, 200);
        let store = VectorStore::open_mmap(&path).unwrap();
        assert_eq!(store.len(), 200);
        assert_eq!(store.dimensions(), dims);
        for i in 0..200 {
            assert_eq!(store.get(i).unwrap(), expected(i, dims), "vector {i}");
        }
    }

    #[test]
    fn pread_round_trip_matches_mmap() {
        let dims = 10;
        let (_dir, path) = write_store(dims, 512, 137);
        let mmap = VectorStore::open_mmap(&path).unwrap();
        let pread = VectorStore::open_pread(&path).unwrap();
        assert_eq!(mmap.len(), pread.len());
        for i in 0..137 {
            assert_eq!(mmap.get(i).unwrap(), pread.get(i).unwrap(), "vector {i}");
        }
    }

    #[test]
    fn out_of_range_get_is_not_found() {
        let (_dir, path) = write_store(8, 256, 5);
        let store = VectorStore::open_mmap(&path).unwrap();
        let err = store.get(5).unwrap_err();
        assert!(matches!(err, Error::NotFound(_)));
    }

    #[test]
    fn get_into_rejects_wrong_buffer_len() {
        let (_dir, path) = write_store(8, 256, 3);
        let store = VectorStore::open_mmap(&path).unwrap();
        let mut buf = [0.0f32; 7];
        assert!(matches!(
            store.get_into(0, &mut buf),
            Err(Error::DimensionMismatch { expected: 8, got: 7 })
        ));
    }

    #[test]
    fn default_storage_block_size_works_end_to_end() {
        // Use the production default block size from slate-core config.
        let block_size = StorageParams::default().block_size;
        let dims = 64;
        let (_dir, path) = write_store(dims, block_size, 500);
        let store = VectorStore::open_mmap(&path).unwrap();
        store.advise_vector(0, Advice::WillNeed).unwrap();
        for i in (0..500).step_by(37) {
            assert_eq!(store.get(i).unwrap(), expected(i, dims));
        }
    }

    /// Write `count` deterministic vectors with a chosen dtype.
    fn write_store_dtype(
        dtype: Dtype,
        dims: usize,
        block_size: usize,
        count: usize,
    ) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("narrow.bin");
        let layout = BlockLayout::new(dtype, dims, block_size).unwrap();
        let mut w = StoreWriter::create(&path, layout).unwrap();
        for i in 0..count {
            // Bounded payload in [-1, 1] so i8/f16 error stays small & checkable.
            let v: Vec<f32> = (0..dims)
                .map(|d| (((i * 7 + d * 13) % 200) as f32 / 100.0) - 1.0)
                .collect();
            w.push(&v).unwrap();
        }
        let header = w.finish().unwrap();
        assert_eq!(header.count, count as u64);
        assert_eq!(header.dtype, dtype);
        (dir, path)
    }

    fn bounded(i: usize, dims: usize) -> Vec<f32> {
        (0..dims)
            .map(|d| (((i * 7 + d * 13) % 200) as f32 / 100.0) - 1.0)
            .collect()
    }

    #[test]
    fn f16_store_round_trips_within_tolerance() {
        let dims = 16;
        let count = 200;
        let (_dir, path) = write_store_dtype(Dtype::F16, dims, 1024, count);
        let store = VectorStore::open_mmap(&path).unwrap();
        assert_eq!(store.dtype(), Dtype::F16);
        // f16 footprint is half of f32 (no per-vector metadata).
        assert_eq!(store.layout().vector_bytes(), dims * 2);
        for i in 0..count {
            let got = store.get(i).unwrap();
            let want = bounded(i, dims);
            for (a, b) in want.iter().zip(got.iter()) {
                assert!((a - b).abs() <= 1e-2, "f16 vec {i}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn i8_store_round_trips_within_tolerance_and_matches_on_both_backends() {
        let dims = 12;
        let count = 137;
        let (_dir, path) = write_store_dtype(Dtype::I8, dims, 512, count);
        let mmap = VectorStore::open_mmap(&path).unwrap();
        let pread = VectorStore::open_pread(&path).unwrap();
        assert_eq!(mmap.dtype(), Dtype::I8);
        // i8 footprint is dims code bytes + a 4-byte scale.
        assert_eq!(mmap.layout().vector_bytes(), dims + 4);
        for i in 0..count {
            let m = mmap.get(i).unwrap();
            let p = pread.get(i).unwrap();
            assert_eq!(m, p, "backends disagree on i8 vec {i}");
            let want = bounded(i, dims);
            // Symmetric per-vector quant: error <= half a step = max_abs/254.
            let max_abs = want.iter().fold(0.0f32, |mx, x| mx.max(x.abs()));
            let tol = max_abs / 254.0 + 1e-6;
            for (a, b) in want.iter().zip(m.iter()) {
                assert!((a - b).abs() <= tol, "i8 vec {i}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn narrow_dtypes_shrink_the_file() {
        let dims = 64;
        let count = 300;
        let block = 4096;
        let f32_len = {
            let (_d, p) = write_store(dims, block, count);
            std::fs::metadata(&p).unwrap().len()
        };
        let f16_len = {
            let (_d, p) = write_store_dtype(Dtype::F16, dims, block, count);
            std::fs::metadata(&p).unwrap().len()
        };
        let i8_len = {
            let (_d, p) = write_store_dtype(Dtype::I8, dims, block, count);
            std::fs::metadata(&p).unwrap().len()
        };
        // f16 strictly smaller than f32; i8 strictly smaller than f16 (the
        // +4B scale is dwarfed by dims=64).
        assert!(f16_len < f32_len, "f16 {f16_len} !< f32 {f32_len}");
        assert!(i8_len < f16_len, "i8 {i8_len} !< f16 {f16_len}");
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use crate::layout::StoreWriter;
    use proptest::prelude::*;

    proptest! {
        // Random geometry + payload must survive a write -> read round trip
        // identically on both backends, for any block size (in power-of-two
        // steps) that can hold the vector, exercising block boundaries and
        // tail padding.
        #![proptest_config(ProptestConfig::with_cases(200))]
        #[test]
        fn write_read_round_trip(
            dims in 1usize..=48,
            count in 0usize..=300,
            log2_block in 6u32..=12,
        ) {
            let block_size = 1usize << log2_block;
            let vbytes = dims * 4;
            // Skip geometries where a vector cannot fit in a block.
            prop_assume!(block_size >= vbytes);

            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("p.bin");
            let layout = BlockLayout::new(Dtype::F32, dims, block_size).unwrap();
            let mut w = StoreWriter::create(&path, layout).unwrap();
            // Deterministic but geometry-dependent payload.
            let payload = |i: usize| -> Vec<f32> {
                (0..dims).map(|d| (i as f32) * 0.5 - (d as f32)).collect()
            };
            for i in 0..count {
                w.push(&payload(i)).unwrap();
            }
            let header = w.finish().unwrap();
            prop_assert_eq!(header.count, count as u64);

            let mmap = VectorStore::open_mmap(&path).unwrap();
            let pread = VectorStore::open_pread(&path).unwrap();
            prop_assert_eq!(mmap.len(), count);
            prop_assert_eq!(pread.len(), count);
            for i in 0..count {
                let want = payload(i);
                prop_assert_eq!(mmap.get(i).unwrap(), want.clone());
                prop_assert_eq!(pread.get(i).unwrap(), want);
            }
            // Reading one past the end is always NotFound.
            prop_assert!(mmap.get(count).is_err());
        }
    }
}
