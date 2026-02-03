//! Build, search, and storage configuration.
//!
//! These structs are the tunable surface of the engine. They derive
//! `Serialize`/`Deserialize` because the resolved [`BuildConfig`] is persisted
//! verbatim into the on-disk metadata file, making an index self-describing.

use serde::{Deserialize, Serialize};

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::metric::Metric;

/// Which approximate-nearest-neighbor backend to build.
///
/// Both backends sit behind the same index trait and share the two-level
/// hybrid search; this only selects the graph/partition structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum IndexBackend {
    /// Hierarchical Navigable Small World graph with high-degree-preserving
    /// pruning. Best recall/latency tradeoff; the default.
    #[default]
    Hnsw,
    /// Inverted file (k-means) partitioning. Naturally sequential per-list
    /// reads make it friendly to spinning disks.
    Ivf,
}

/// Storage device profile, used to pick I/O strategy and `madvise` hints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum IoProfile {
    /// Spinning disk: favor large sequential `pread`s and the elevator
    /// scheduler; avoid page-fault-driven 4KB random reads.
    Hdd,
    /// Solid-state: random reads are cheap; mmap page faults are fine.
    Ssd,
    /// Detect at open time, defaulting to the more conservative `Hdd` strategy
    /// when unsure.
    #[default]
    Auto,
}

/// On-disk vector storage parameters.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StorageParams {
    /// Element type for exact vectors on disk (controls read volume/precision).
    pub dtype: Dtype,
    /// Disk block size in bytes. Large by default (64 KiB) so each seek on a
    /// spinning disk amortizes over many co-located vectors.
    pub block_size: usize,
    /// Device profile controlling I/O strategy.
    pub io_profile: IoProfile,
}

impl Default for StorageParams {
    fn default() -> Self {
        Self {
            dtype: Dtype::F32,
            block_size: 64 * 1024,
            io_profile: IoProfile::Auto,
        }
    }
}

impl StorageParams {
    /// Validate the storage parameters.
    pub fn validate(&self) -> Result<()> {
        if self.block_size < 512 {
            return Err(Error::invalid_config(format!(
                "block_size must be >= 512 bytes, got {}",
                self.block_size
            )));
        }
        if !self.block_size.is_power_of_two() {
            return Err(Error::invalid_config(format!(
                "block_size must be a power of two, got {}",
                self.block_size
            )));
        }
        Ok(())
    }
}

/// Product-quantization parameters for the RAM-resident approximate tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PqParams {
    /// Number of subquantizers (subspaces). The vector dimensionality must be
    /// divisible by this value.
    pub num_subquantizers: usize,
    /// Bits per subspace code. `8` gives the standard 256-centroid codebook.
    pub bits_per_code: u8,
}

impl Default for PqParams {
    fn default() -> Self {
        Self {
            num_subquantizers: 16,
            bits_per_code: 8,
        }
    }
}

impl PqParams {
    /// Number of centroids per subspace (`2^bits_per_code`).
    #[inline]
    pub const fn centroids_per_subspace(&self) -> usize {
        1usize << self.bits_per_code
    }

    /// Validate the PQ parameters in isolation (divisibility against the vector
    /// dimensionality is checked by [`BuildConfig::validate`]).
    pub fn validate(&self) -> Result<()> {
        if self.num_subquantizers == 0 {
            return Err(Error::invalid_config("num_subquantizers must be >= 1"));
        }
        if self.bits_per_code == 0 || self.bits_per_code > 8 {
            return Err(Error::invalid_config(format!(
                "bits_per_code must be in 1..=8, got {}",
                self.bits_per_code
            )));
        }
        Ok(())
    }
}

/// HNSW construction parameters, including LEANN-style high-degree-preserving
/// pruning.
///
/// Most nodes are capped at out-degree `m`; the top `hub_fraction` of nodes by
/// degree ("navigation hubs") may grow to `m_max`. Every node is additionally
/// allowed to form bidirectional links with newly inserted nodes up to `m_max`,
/// which preserves navigability toward hubs even for low-degree nodes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HnswParams {
    /// Base out-degree cap for ordinary nodes (LEANN `m`).
    pub m: usize,
    /// Maximum out-degree for hub nodes and new-node links (LEANN `M`).
    pub m_max: usize,
    /// Candidate-list size during construction (`ef_construction`).
    pub ef_construction: usize,
    /// Fraction of nodes preserved as high-degree hubs (LEANN `beta`, ~0.02).
    pub hub_fraction: f32,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            m: 8,
            m_max: 32,
            ef_construction: 200,
            hub_fraction: 0.02,
        }
    }
}

impl HnswParams {
    /// Validate the HNSW parameters.
    pub fn validate(&self) -> Result<()> {
        if self.m == 0 {
            return Err(Error::invalid_config("hnsw.m must be >= 1"));
        }
        if self.m_max < self.m {
            return Err(Error::invalid_config(format!(
                "hnsw.m_max ({}) must be >= hnsw.m ({})",
                self.m_max, self.m
            )));
        }
        if self.ef_construction == 0 {
            return Err(Error::invalid_config("hnsw.ef_construction must be >= 1"));
        }
        if !(0.0..=1.0).contains(&self.hub_fraction) {
            return Err(Error::invalid_config(format!(
                "hnsw.hub_fraction must be in [0, 1], got {}",
                self.hub_fraction
            )));
        }
        Ok(())
    }
}

/// IVF (inverted file / k-means) construction parameters.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct IvfParams {
    /// Number of coarse centroids / inverted lists (`k`).
    pub num_lists: usize,
    /// Lists probed per query (`nprobe`).
    pub num_probes: usize,
    /// Number of nearest lists each vector is assigned to (LEANN soft
    /// assignment uses 2 for cross-list connectivity).
    pub soft_assign: usize,
    /// Maximum k-means iterations during training.
    pub max_kmeans_iters: usize,
}

impl Default for IvfParams {
    fn default() -> Self {
        Self {
            num_lists: 256,
            num_probes: 16,
            soft_assign: 2,
            max_kmeans_iters: 25,
        }
    }
}

impl IvfParams {
    /// Validate the IVF parameters.
    pub fn validate(&self) -> Result<()> {
        if self.num_lists == 0 {
            return Err(Error::invalid_config("ivf.num_lists must be >= 1"));
        }
        if self.num_probes == 0 || self.num_probes > self.num_lists {
            return Err(Error::invalid_config(format!(
                "ivf.num_probes must be in 1..={}, got {}",
                self.num_lists, self.num_probes
            )));
        }
        if self.soft_assign == 0 || self.soft_assign > self.num_lists {
            return Err(Error::invalid_config(format!(
                "ivf.soft_assign must be in 1..={}, got {}",
                self.num_lists, self.soft_assign
            )));
        }
        if self.max_kmeans_iters == 0 {
            return Err(Error::invalid_config("ivf.max_kmeans_iters must be >= 1"));
        }
        Ok(())
    }
}

/// Top-level configuration for building an index.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BuildConfig {
    /// Vector dimensionality. Must be set (> 0) and divisible by
    /// `pq.num_subquantizers`.
    pub dimensions: usize,
    /// Distance metric.
    pub metric: Metric,
    /// Which ANN backend to build.
    pub backend: IndexBackend,
    /// Disk storage parameters.
    pub storage: StorageParams,
    /// Product-quantization parameters (RAM approximate tier).
    pub pq: PqParams,
    /// HNSW parameters (used when `backend == Hnsw`).
    pub hnsw: HnswParams,
    /// IVF parameters (used when `backend == Ivf`).
    pub ivf: IvfParams,
    /// Number of shards for the storage-efficient sharded build pipeline.
    /// `1` disables sharding.
    pub num_shards: usize,
}

impl Default for BuildConfig {
    fn default() -> Self {
        Self {
            dimensions: 0,
            metric: Metric::L2,
            backend: IndexBackend::Hnsw,
            storage: StorageParams::default(),
            pq: PqParams::default(),
            hnsw: HnswParams::default(),
            ivf: IvfParams::default(),
            num_shards: 1,
        }
    }
}

impl BuildConfig {
    /// Create a build configuration with the given dimensionality, metric, and
    /// backend, leaving all other parameters at their defaults.
    pub fn new(dimensions: usize, metric: Metric, backend: IndexBackend) -> Self {
        Self {
            dimensions,
            metric,
            backend,
            ..Self::default()
        }
    }

    /// Validate the entire configuration, including cross-field constraints.
    pub fn validate(&self) -> Result<()> {
        if self.dimensions == 0 {
            return Err(Error::invalid_config("dimensions must be set (> 0)"));
        }
        self.storage.validate()?;
        self.pq.validate()?;
        self.hnsw.validate()?;
        self.ivf.validate()?;
        if !self.dimensions.is_multiple_of(self.pq.num_subquantizers) {
            return Err(Error::invalid_config(format!(
                "dimensions ({}) must be divisible by pq.num_subquantizers ({})",
                self.dimensions, self.pq.num_subquantizers
            )));
        }
        if self.num_shards == 0 {
            return Err(Error::invalid_config("num_shards must be >= 1"));
        }
        Ok(())
    }
}

/// Per-query search configuration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SearchConfig {
    /// Number of nearest neighbors to return.
    pub k: usize,
    /// Search-time candidate-list size (`ef`); the primary recall/latency knob.
    pub ef_search: usize,
    /// Re-ranking ratio `alpha`: the top fraction of the approximate queue whose
    /// exact vectors are fetched from disk and re-ranked each step.
    pub rerank_ratio: f32,
    /// Number of exact-vector fetches accumulated before issuing a batched,
    /// seek-ordered disk read (amortizes HDD seeks across exploration steps).
    pub fetch_batch_size: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            k: 10,
            ef_search: 64,
            rerank_ratio: 0.2,
            fetch_batch_size: 64,
        }
    }
}

impl SearchConfig {
    /// Validate the search configuration.
    pub fn validate(&self) -> Result<()> {
        if self.k == 0 {
            return Err(Error::invalid_config("k must be >= 1"));
        }
        if self.ef_search < self.k {
            return Err(Error::invalid_config(format!(
                "ef_search ({}) must be >= k ({})",
                self.ef_search, self.k
            )));
        }
        if !(self.rerank_ratio > 0.0 && self.rerank_ratio <= 1.0) {
            return Err(Error::invalid_config(format!(
                "rerank_ratio must be in (0, 1], got {}",
                self.rerank_ratio
            )));
        }
        if self.fetch_batch_size == 0 {
            return Err(Error::invalid_config("fetch_batch_size must be >= 1"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    // Several negative tests deliberately mutate one field of a defaulted
    // struct and re-validate between mutations; struct-update syntax does not
    // fit that pattern cleanly.
    #![allow(clippy::field_reassign_with_default)]

    use super::*;

    #[test]
    fn default_build_config_requires_dimensions() {
        let cfg = BuildConfig::default();
        assert!(cfg.validate().is_err(), "dimensions=0 must be rejected");
    }

    #[test]
    fn valid_build_config_passes() {
        let cfg = BuildConfig::new(768, Metric::Cosine, IndexBackend::Hnsw);
        assert!(cfg.validate().is_ok(), "{:?}", cfg.validate());
    }

    #[test]
    fn pq_divisibility_enforced() {
        let mut cfg = BuildConfig::new(770, Metric::L2, IndexBackend::Hnsw);
        cfg.pq.num_subquantizers = 16; // 770 % 16 != 0
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn pq_centroid_count() {
        assert_eq!(PqParams::default().centroids_per_subspace(), 256);
    }

    #[test]
    fn storage_block_size_must_be_pow2() {
        let mut sp = StorageParams::default();
        sp.block_size = 1000;
        assert!(sp.validate().is_err());
        sp.block_size = 4096;
        assert!(sp.validate().is_ok());
    }

    #[test]
    fn hnsw_degree_ordering() {
        let mut p = HnswParams::default();
        p.m = 40;
        p.m_max = 32;
        assert!(p.validate().is_err());
    }

    #[test]
    fn ivf_probe_bounds() {
        let mut p = IvfParams::default();
        p.num_probes = p.num_lists + 1;
        assert!(p.validate().is_err());
    }

    #[test]
    fn search_config_bounds() {
        let mut s = SearchConfig::default();
        assert!(s.validate().is_ok());
        s.ef_search = 1;
        s.k = 10;
        assert!(s.validate().is_err());
        s = SearchConfig::default();
        s.rerank_ratio = 0.0;
        assert!(s.validate().is_err());
    }

    #[test]
    fn config_roundtrips_through_json() {
        // Exercises serde wiring that the on-disk metadata format relies on.
        let cfg = BuildConfig::new(128, Metric::InnerProduct, IndexBackend::Ivf);
        let json = serde_json::to_string(&cfg).expect("serialize");
        let back: BuildConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(cfg, back);
    }
}
