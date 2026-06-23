//! Self-describing on-disk index *bundle*.
//!
//! A bundle is a directory that packages everything needed to serve queries
//! without re-specifying any build parameters:
//!
//! ```text
//! my-index/
//!   manifest.json   bundle magic + version + file names + resolved BuildConfig
//!   vectors.svec    the slate-storage vector store (MAGIC "SLATEVEC")
//!   index.sidx      the slate-index graph/list frame (MAGIC "SLATEIDX")
//! ```
//!
//! The point of the storage-aware design is *build expensively once, serve
//! cheaply off disk forever*. That promise only holds if a loader can trust the
//! files themselves — the on-disk dtype, block size, metric and backend — rather
//! than a remembered command line. The manifest stamps the resolved
//! [`BuildConfig`] next to the data so the seek-minimising layout is fully
//! reproducible from disk alone.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use slate_core::{BuildConfig, Error, IndexBackend, Neighbor, QueryCounters, Result, SearchConfig};
use slate_graph::{HnswIndex, IvfIndex};
use slate_storage::{BlockLayout, MmapBackend, StoreWriter, VectorStore};

use crate::format;

/// Magic string written at the head of every bundle manifest.
pub const BUNDLE_MAGIC: &str = "SLATEANN-BUNDLE";
/// On-disk bundle layout version. Bumped on incompatible manifest changes.
pub const BUNDLE_FORMAT_VERSION: u32 = 1;
/// Manifest file name inside the bundle directory.
pub const MANIFEST_FILE: &str = "manifest.json";
/// Vector-store file name inside the bundle directory.
pub const STORE_FILE: &str = "vectors.svec";
/// Index file name inside the bundle directory.
pub const INDEX_FILE: &str = "index.sidx";

/// The JSON manifest that makes a bundle self-describing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BundleManifest {
    /// Always [`BUNDLE_MAGIC`]; rejected on load otherwise.
    pub magic: String,
    /// Always [`BUNDLE_FORMAT_VERSION`]; rejected on load otherwise.
    pub version: u32,
    /// Relative name of the vector-store file (normally [`STORE_FILE`]).
    pub store_file: String,
    /// Relative name of the index file (normally [`INDEX_FILE`]).
    pub index_file: String,
    /// The fully resolved build configuration the bundle was produced with.
    pub config: BuildConfig,
}

impl BundleManifest {
    fn new(config: BuildConfig) -> Self {
        Self {
            magic: BUNDLE_MAGIC.to_string(),
            version: BUNDLE_FORMAT_VERSION,
            store_file: STORE_FILE.to_string(),
            index_file: INDEX_FILE.to_string(),
            config,
        }
    }

    fn validate(&self) -> Result<()> {
        if self.magic != BUNDLE_MAGIC {
            return Err(Error::corrupt(format!(
                "not a slate bundle manifest: magic {:?}",
                self.magic
            )));
        }
        if self.version != BUNDLE_FORMAT_VERSION {
            return Err(Error::corrupt(format!(
                "unsupported bundle version {} (expected {BUNDLE_FORMAT_VERSION})",
                self.version
            )));
        }
        Ok(())
    }
}

/// One of the concrete index backends a bundle can hold.
///
/// The variant is fixed by [`BuildConfig::backend`]; the on-disk index frame
/// carries its own backend tag, which is cross-checked on load.
#[derive(Debug)]
pub enum BundleIndex {
    /// A graph (HNSW) index.
    Hnsw(HnswIndex),
    /// An inverted-file (IVF) index.
    Ivf(IvfIndex),
}

impl BundleIndex {
    /// The backend discriminant this index corresponds to.
    #[must_use]
    pub fn backend(&self) -> IndexBackend {
        match self {
            BundleIndex::Hnsw(_) => IndexBackend::Hnsw,
            BundleIndex::Ivf(_) => IndexBackend::Ivf,
        }
    }

    /// Run an exact streaming search, returning neighbours and the unified
    /// [`QueryCounters`] (both backends report the same counter shape).
    pub fn search<B: slate_storage::IoBackend>(
        &self,
        store: &VectorStore<B>,
        query: &[f32],
        config: &SearchConfig,
    ) -> Result<(Vec<Neighbor>, QueryCounters)> {
        match self {
            BundleIndex::Hnsw(index) => {
                let (neighbors, stats) = index.search(store, query, config)?;
                Ok((neighbors, stats.counters))
            }
            BundleIndex::Ivf(index) => {
                let (neighbors, stats) = index.search(store, query, config)?;
                Ok((neighbors, stats.counters))
            }
        }
    }

    /// Run a PQ-gated hybrid search. Errors with [`Error::unsupported`] if the
    /// index was built without a PQ tier.
    pub fn search_hybrid<B: slate_storage::IoBackend>(
        &self,
        store: &VectorStore<B>,
        query: &[f32],
        config: &SearchConfig,
    ) -> Result<(Vec<Neighbor>, QueryCounters)> {
        match self {
            BundleIndex::Hnsw(index) => {
                let (neighbors, stats) = index.search_hybrid(store, query, config)?;
                Ok((neighbors, stats.counters))
            }
            BundleIndex::Ivf(index) => {
                let (neighbors, stats) = index.search_hybrid(store, query, config)?;
                Ok((neighbors, stats.counters))
            }
        }
    }
}

/// A bundle opened from disk: the resolved config, the memory-mapped vector
/// store, and the deserialized index, ready to serve queries.
#[derive(Debug)]
pub struct Bundle {
    config: BuildConfig,
    store: VectorStore<MmapBackend>,
    index: BundleIndex,
}

impl Bundle {
    /// The resolved build configuration recorded in the manifest.
    #[must_use]
    pub fn config(&self) -> &BuildConfig {
        &self.config
    }

    /// The memory-mapped vector store.
    #[must_use]
    pub fn store(&self) -> &VectorStore<MmapBackend> {
        &self.store
    }

    /// The deserialized index.
    #[must_use]
    pub fn index(&self) -> &BundleIndex {
        &self.index
    }

    /// Exact search over the bundle's store + index using the manifest metric.
    pub fn search(
        &self,
        query: &[f32],
        config: &SearchConfig,
    ) -> Result<(Vec<Neighbor>, QueryCounters)> {
        self.index.search(&self.store, query, config)
    }

    /// PQ-gated hybrid search (errors if the index has no PQ tier).
    pub fn search_hybrid(
        &self,
        query: &[f32],
        config: &SearchConfig,
    ) -> Result<(Vec<Neighbor>, QueryCounters)> {
        self.index.search_hybrid(&self.store, query, config)
    }
}

fn manifest_path(dir: &Path) -> PathBuf {
    dir.join(MANIFEST_FILE)
}

/// Build a complete bundle directory from in-memory vectors.
///
/// Writes the vector store, builds the configured backend over it, then saves
/// the index and manifest. `config.dimensions` must match the vector width and
/// `config.validate()` must pass.
pub fn build_bundle(
    dir: impl AsRef<Path>,
    config: &BuildConfig,
    vectors: &[Vec<f32>],
    seed: u64,
) -> Result<()> {
    config.validate()?;
    let dir = dir.as_ref();
    std::fs::create_dir_all(dir)?;

    let dims = config.dimensions;
    for (i, v) in vectors.iter().enumerate() {
        if v.len() != dims {
            return Err(Error::DimensionMismatch {
                expected: dims,
                got: v.len(),
            });
        }
        // Touch i so a stray malformed row names itself in debug builds.
        debug_assert!(i < vectors.len());
    }

    // 1. Write the vector store directly inside the bundle directory.
    let store_path = dir.join(STORE_FILE);
    let layout = BlockLayout::new(config.storage.dtype, dims, config.storage.block_size)?;
    let mut writer = StoreWriter::create(&store_path, layout)?;
    for v in vectors {
        writer.push(v)?;
    }
    writer.finish()?;

    // 2. Build the configured backend over the freshly written store.
    let store = VectorStore::open_mmap(&store_path)?;
    let index = match config.backend {
        IndexBackend::Hnsw => {
            BundleIndex::Hnsw(HnswIndex::build(&store, config.metric, &config.hnsw, seed)?)
        }
        IndexBackend::Ivf => {
            BundleIndex::Ivf(IvfIndex::build(&store, config.metric, &config.ivf, seed)?)
        }
    };

    // 3. Save the index frame and the manifest.
    save_index(dir, &index)?;
    write_manifest(dir, &BundleManifest::new(config.clone()))?;
    Ok(())
}

fn save_index(dir: &Path, index: &BundleIndex) -> Result<()> {
    let index_path = dir.join(INDEX_FILE);
    match index {
        BundleIndex::Hnsw(idx) => format::save_hnsw(&index_path, idx),
        BundleIndex::Ivf(idx) => format::save_ivf(&index_path, idx),
    }
}

fn write_manifest(dir: &Path, manifest: &BundleManifest) -> Result<()> {
    let json = serde_json::to_vec_pretty(manifest)
        .map_err(|e| Error::corrupt(format!("failed to serialize bundle manifest: {e}")))?;
    std::fs::write(manifest_path(dir), json)?;
    Ok(())
}

fn read_manifest(dir: &Path) -> Result<BundleManifest> {
    let bytes = std::fs::read(manifest_path(dir))?;
    let manifest: BundleManifest = serde_json::from_slice(&bytes)
        .map_err(|e| Error::corrupt(format!("malformed bundle manifest: {e}")))?;
    manifest.validate()?;
    Ok(manifest)
}

/// Open a bundle directory: validate the manifest, memory-map the store, and
/// load the index (whose backend tag must match the manifest's `config.backend`).
pub fn open_bundle(dir: impl AsRef<Path>) -> Result<Bundle> {
    let dir = dir.as_ref();
    let manifest = read_manifest(dir)?;

    let store = VectorStore::open_mmap(dir.join(&manifest.store_file))?;

    let index_path = dir.join(&manifest.index_file);
    let index = match manifest.config.backend {
        IndexBackend::Hnsw => BundleIndex::Hnsw(format::load_hnsw(&index_path)?),
        IndexBackend::Ivf => BundleIndex::Ivf(format::load_ivf(&index_path)?),
    };

    Ok(Bundle {
        config: manifest.config,
        store,
        index,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use slate_core::{Dtype, IndexBackend, Metric, StorageParams};
    use std::collections::HashSet;

    /// Deterministic dependency-free pseudo-random vectors in [0, 1).
    fn gen_vectors(seed: u64, n: usize, dims: usize) -> Vec<Vec<f32>> {
        let mut state = seed | 1;
        let mut next = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..n)
            .map(|_| (0..dims).map(|_| next() as f32).collect())
            .collect()
    }

    fn config_for(dims: usize, backend: IndexBackend) -> BuildConfig {
        let mut config = BuildConfig::new(dims, Metric::L2, backend);
        // Keep the divisibility constraint satisfied for any test dims by using
        // a single subquantizer (PQ isn't exercised by the plain build path).
        config.pq.num_subquantizers = 1;
        config.storage = StorageParams {
            dtype: Dtype::F32,
            block_size: 4096,
            ..StorageParams::default()
        };
        config
    }

    fn cfg(k: usize) -> SearchConfig {
        SearchConfig {
            k,
            ..SearchConfig::default()
        }
    }

    fn assert_round_trips(backend: IndexBackend) {
        let dims = 12;
        let vectors = gen_vectors(2024, 300, dims);
        let dir = tempfile::tempdir().unwrap();
        let config = config_for(dims, backend);
        build_bundle(dir.path(), &config, &vectors, 777).unwrap();

        let bundle = open_bundle(dir.path()).unwrap();
        assert_eq!(bundle.config(), &config);
        assert_eq!(bundle.index().backend(), backend);

        // A loaded bundle answers identically to a freshly built reference.
        let store = bundle.store();
        let reference = match backend {
            IndexBackend::Hnsw => {
                BundleIndex::Hnsw(HnswIndex::build(store, Metric::L2, &config.hnsw, 777).unwrap())
            }
            IndexBackend::Ivf => {
                BundleIndex::Ivf(IvfIndex::build(store, Metric::L2, &config.ivf, 777).unwrap())
            }
        };

        let queries = gen_vectors(99, 25, dims);
        for q in &queries {
            let (got, _) = bundle.search(q, &cfg(10)).unwrap();
            let (want, _) = reference.search(store, q, &cfg(10)).unwrap();
            let got_ids: Vec<u64> = got.iter().map(|n| n.id.get()).collect();
            let want_ids: Vec<u64> = want.iter().map(|n| n.id.get()).collect();
            assert_eq!(got_ids, want_ids, "{backend:?} bundle search differs");
        }
    }

    #[test]
    fn hnsw_bundle_round_trips() {
        assert_round_trips(IndexBackend::Hnsw);
    }

    #[test]
    fn ivf_bundle_round_trips() {
        assert_round_trips(IndexBackend::Ivf);
    }

    #[test]
    fn bundle_files_are_named_as_documented() {
        let dims = 8;
        let vectors = gen_vectors(7, 50, dims);
        let dir = tempfile::tempdir().unwrap();
        build_bundle(dir.path(), &config_for(dims, IndexBackend::Hnsw), &vectors, 1).unwrap();
        let present: HashSet<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert!(present.contains(MANIFEST_FILE));
        assert!(present.contains(STORE_FILE));
        assert!(present.contains(INDEX_FILE));
    }

    #[test]
    fn manifest_bad_magic_is_rejected() {
        let dims = 8;
        let vectors = gen_vectors(7, 40, dims);
        let dir = tempfile::tempdir().unwrap();
        build_bundle(dir.path(), &config_for(dims, IndexBackend::Hnsw), &vectors, 1).unwrap();

        // Corrupt the magic field in the manifest.
        let mut manifest = read_manifest(dir.path()).unwrap();
        manifest.magic = "NOT-A-BUNDLE".to_string();
        let json = serde_json::to_vec_pretty(&manifest).unwrap();
        std::fs::write(manifest_path(dir.path()), json).unwrap();

        let err = open_bundle(dir.path()).unwrap_err();
        assert!(matches!(err, Error::Corrupt(_)), "got {err:?}");
    }

    #[test]
    fn manifest_bad_version_is_rejected() {
        let dims = 8;
        let vectors = gen_vectors(7, 40, dims);
        let dir = tempfile::tempdir().unwrap();
        build_bundle(dir.path(), &config_for(dims, IndexBackend::Hnsw), &vectors, 1).unwrap();

        let mut manifest = read_manifest(dir.path()).unwrap();
        manifest.version = BUNDLE_FORMAT_VERSION + 1;
        let json = serde_json::to_vec_pretty(&manifest).unwrap();
        std::fs::write(manifest_path(dir.path()), json).unwrap();

        // read_manifest itself validates, so opening must fail.
        let err = open_bundle(dir.path()).unwrap_err();
        assert!(matches!(err, Error::Corrupt(_)), "got {err:?}");
    }

    #[test]
    fn dimension_mismatch_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let config = config_for(8, IndexBackend::Hnsw);
        // A row with the wrong width.
        let vectors = vec![vec![0.0f32; 8], vec![0.0f32; 7]];
        let err = build_bundle(dir.path(), &config, &vectors, 1).unwrap_err();
        assert!(
            matches!(err, Error::DimensionMismatch { expected: 8, got: 7 }),
            "got {err:?}"
        );
    }

    #[test]
    fn empty_bundle_round_trips() {
        let dims = 8;
        let dir = tempfile::tempdir().unwrap();
        build_bundle(dir.path(), &config_for(dims, IndexBackend::Hnsw), &[], 1).unwrap();
        let bundle = open_bundle(dir.path()).unwrap();
        let (res, counters) = bundle.search(&vec![0.1f32; dims], &cfg(10)).unwrap();
        assert!(res.is_empty());
        assert_eq!(counters.exact_distances, 0);
    }
}
