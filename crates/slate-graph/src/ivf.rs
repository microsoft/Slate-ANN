//! IVF (inverted file) backend with a k-means coarse quantizer and
//! soft-assigned posting lists.
//!
//! # What this is (Phase 8)
//!
//! IVF partitions the corpus into `num_lists` Voronoi cells around k-means
//! centroids. Each cell owns a **posting list** of the vector ids assigned to
//! it. A query:
//!
//! 1. measures its distance to every centroid (cheap, in RAM),
//! 2. **probes** the `num_probes` nearest cells,
//! 3. unions their posting lists into a candidate id set, and
//! 4. ranks those candidates by **exact** distance streamed from the store.
//!
//! To keep cells connected the way LEANN's graph keeps hubs reachable, every
//! vector is **soft-assigned** to its `soft_assign` nearest centroids, so a
//! query that lands on a slightly-wrong cell still finds neighbours that live
//! in an adjacent probed cell.
//!
//! # Storage-aware fetch (reuses Phase 7)
//!
//! The candidate ids of the probed lists are exactly the kind of batch the
//! Phase-7 scheduler was built for. We hand them to
//! [`slate_storage::FetchSchedule::plan`] and stream them with
//! [`slate_storage::VectorStore::fetch_scheduled`], which sorts them into
//! ascending physical offset, coalesces same/adjacent-block reads into runs,
//! and issues one positioned read per run. The cost model is charged a single
//! [`QueryCounters::add_read`] for the whole probe — so IVF inherits the
//! seek-coalescing payoff without re-deriving any I/O path. This is the
//! "naturally sequential per-list reads … friendly to spinning disks" property
//! that [`slate_core::IndexBackend::Ivf`] advertises.
//!
//! # Quantizer metric vs index metric
//!
//! The coarse quantizer is trained with squared-L2 (via
//! [`slate_pq::kmeans::train`]), and both vector assignment and query probing
//! use squared-L2 so they agree with the partition geometry the centroids
//! describe. The index's own [`Metric`] governs only the **exact** ranking of
//! the fetched candidates — L2 partitioning is used purely as a routing proxy,
//! exactly as the PQ tier uses L2-in-subspace to gate fetches.

use slate_core::{
    Error, IvfParams, Metric, Neighbor, QueryCounters, Result, SearchConfig, TopK, VectorId,
};
use slate_storage::{FetchSchedule, IoBackend, VectorStore};

use serde::{Deserialize, Serialize};

/// Per-search statistics, exposing the physical work the storage-aware cost
/// model prices. Mirrors [`crate::HnswStats`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IvfStats {
    /// Counters accumulated during the search (centroid comparisons as
    /// approximate distances, exact candidate distances, bytes/seeks/runs read
    /// from the store).
    pub counters: QueryCounters,
}

/// An in-RAM inverted-file index over a [`VectorStore`].
///
/// The centroids and posting lists are resident; the vectors themselves stay on
/// disk and are streamed (in coalesced seek order) only for the candidates of a
/// probed query. The index serializes to disk via [`serde`] (see
/// `slate_index::format`); the vectors stay in their own store file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IvfIndex {
    /// Coarse centroids, row-major: centroid `c` occupies
    /// `[c*dims .. (c+1)*dims]`. There are `num_lists` of them.
    centroids: Vec<f32>,
    /// One posting list per centroid, each a **sorted** list of vector ids.
    postings: Vec<Vec<u32>>,
    /// Number of centroids / posting lists.
    num_lists: usize,
    /// Vector dimensionality.
    dims: usize,
    /// Total number of vectors indexed.
    len: usize,
    /// Metric used for the exact ranking of fetched candidates.
    metric: Metric,
    /// Number of nearest cells to probe per query (resolved from config).
    num_probes: usize,
}

/// k-means iterations are taken from [`IvfParams::max_kmeans_iters`]; the seed
/// is mixed so it is decorrelated from any graph seed sharing the same value.
const IVF_SEED_MIX: u64 = 0x2545_F491_4F6C_DD1D;

impl IvfIndex {
    /// Build an IVF index over every vector in `store`.
    ///
    /// Reads all vectors into RAM once, trains the coarse quantizer with
    /// `params.num_lists` centroids, and soft-assigns each vector to its
    /// `params.soft_assign` nearest centroids, producing sorted posting lists.
    ///
    /// `seed` makes the clustering (and therefore the whole index)
    /// deterministic.
    ///
    /// # Errors
    ///
    /// Returns [`Error::unsupported`] if the store is not f32-decodable into the
    /// expected dimensionality (it always is via `get_into`, which decodes
    /// narrow dtypes), and propagates distance-kernel / I/O errors.
    pub fn build<B: IoBackend>(
        store: &VectorStore<B>,
        metric: Metric,
        params: &IvfParams,
        seed: u64,
    ) -> Result<Self> {
        let dims = store.dimensions();
        let len = store.len();

        // Empty store: a valid, empty index.
        if len == 0 {
            return Ok(Self {
                centroids: Vec::new(),
                postings: Vec::new(),
                num_lists: 0,
                dims,
                len: 0,
                metric,
                num_probes: params.num_probes,
            });
        }

        // Read the whole corpus into RAM once (row-major), decoding whatever
        // on-disk dtype the store uses into f32.
        let mut data = vec![0.0f32; len * dims];
        let mut scratch = vec![0.0f32; dims];
        for i in 0..len {
            store.get_into(i, &mut scratch)?;
            data[i * dims..(i + 1) * dims].copy_from_slice(&scratch);
        }

        // Train the coarse quantizer. k-means returns at most `num_lists`
        // centroids (fewer if the corpus is tiny), so trust its reported `k`.
        let km = slate_pq::kmeans::train(
            &data,
            dims,
            params.num_lists,
            params.max_kmeans_iters,
            seed ^ IVF_SEED_MIX,
        )?;
        let num_lists = km.k;
        let centroids = km.centroids;

        // Soft-assign every vector to its `soft_assign` nearest centroids,
        // capped by how many centroids actually exist.
        let soft = params.soft_assign.min(num_lists).max(1);
        let mut postings: Vec<Vec<u32>> = vec![Vec::new(); num_lists];
        for i in 0..len {
            let v = &data[i * dims..(i + 1) * dims];
            for c in nearest_centroids(&centroids, num_lists, dims, v, soft)? {
                postings[c].push(i as u32);
            }
        }

        // Posting lists are kept ascending so candidate unions and the fetch
        // schedule see sorted ids. Soft-assign appends in centroid order, which
        // is already ascending per list, but sort defensively.
        for list in &mut postings {
            list.sort_unstable();
        }

        Ok(Self {
            centroids,
            postings,
            num_lists,
            dims,
            len,
            metric,
            num_probes: params.num_probes,
        })
    }

    /// Number of vectors indexed.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the index holds no vectors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of coarse centroids / posting lists.
    #[must_use]
    pub fn num_lists(&self) -> usize {
        self.num_lists
    }

    /// Borrow posting list `list` (sorted vector ids). Used by tests and, later,
    /// by the on-disk serializer.
    #[must_use]
    pub fn posting_list(&self, list: usize) -> &[u32] {
        &self.postings[list]
    }

    /// Search for the `config.k` nearest neighbours of `query`.
    ///
    /// Probes the `num_probes` nearest cells, unions their posting lists, fetches
    /// the candidate vectors in coalesced seek order, and ranks them exactly.
    /// Returns the best `k` ascending plus the per-search [`IvfStats`].
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`] if `query.len() != dims`; propagates
    /// distance-kernel / I/O errors.
    pub fn search<B: IoBackend>(
        &self,
        store: &VectorStore<B>,
        query: &[f32],
        config: &SearchConfig,
    ) -> Result<(Vec<Neighbor>, IvfStats)> {
        if query.len() != self.dims {
            return Err(Error::DimensionMismatch {
                expected: self.dims,
                got: query.len(),
            });
        }

        let mut counters = QueryCounters::new();

        // Empty index: nothing to return.
        if self.len == 0 || self.num_lists == 0 {
            return Ok((Vec::new(), IvfStats { counters }));
        }

        // ---- PROBE: distance from the query to every centroid (RAM work,
        // modelled as approximate distances), keep the `num_probes` nearest.
        let probes = self.num_probes.min(self.num_lists).max(1);
        let mut centroid_scores: Vec<(f32, usize)> = Vec::with_capacity(self.num_lists);
        for c in 0..self.num_lists {
            let cc = &self.centroids[c * self.dims..(c + 1) * self.dims];
            let d = slate_simd::l2_sq(query, cc)?;
            centroid_scores.push((d, c));
        }
        counters.add_approx(self.num_lists as u64);
        // Partial-select the `probes` smallest by distance (ties by list id for
        // determinism).
        centroid_scores
            .sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        centroid_scores.truncate(probes);

        // ---- CANDIDATES: union the probed posting lists, deduped. Sorting the
        // merged ids puts them in ascending (== physical-offset) order, which is
        // exactly what the scheduler wants.
        let mut candidates: Vec<usize> = Vec::new();
        for &(_, c) in &centroid_scores {
            for &id in &self.postings[c] {
                candidates.push(id as usize);
            }
        }
        candidates.sort_unstable();
        candidates.dedup();

        if candidates.is_empty() {
            return Ok((Vec::new(), IvfStats { counters }));
        }

        // ---- FETCH + EXACT RANK: stream the candidates in coalesced seek order
        // and rank them by the index's exact metric. One positioned read per run.
        let plan = FetchSchedule::plan(store.layout(), &candidates);
        let metric = self.metric;
        let mut topk = TopK::new(config.k);
        let mut scratch = vec![0.0f32; self.dims];
        store.fetch_scheduled(&plan, &mut scratch, |idx, vec| {
            let exact = slate_simd::distance(metric, query, vec)?;
            counters.add_exact(1);
            topk.offer(Neighbor::new(VectorId::new(idx as u64), exact));
            Ok(())
        })?;
        // One coalesced storage charge for the whole probe.
        counters.add_read(plan.span_bytes(), plan.seeks(), plan.runs());

        Ok((topk.into_sorted_vec(), IvfStats { counters }))
    }
}

/// Indices of the `count` centroids nearest to `v` by squared-L2, smallest
/// first (ties by centroid id). `count` is assumed `>= 1` and `<= num_lists`.
fn nearest_centroids(
    centroids: &[f32],
    num_lists: usize,
    dims: usize,
    v: &[f32],
    count: usize,
) -> Result<Vec<usize>> {
    let mut scored: Vec<(f32, usize)> = Vec::with_capacity(num_lists);
    for c in 0..num_lists {
        let cc = &centroids[c * dims..(c + 1) * dims];
        let d = slate_simd::l2_sq(v, cc)?;
        scored.push((d, c));
    }
    scored.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    scored.truncate(count);
    Ok(scored.into_iter().map(|(_, c)| c).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SplitMix64;
    use slate_core::{Dtype, StorageParams};
    use slate_storage::{Advice, BlockLayout, MmapBackend, StoreWriter};
    use std::collections::HashSet;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tempfile::NamedTempFile;

    /// Write `vectors` (row-major, `dims` each) to a temp store and open it
    /// mmap-backed, using the default (large) block size.
    fn build_store(
        vectors: &[Vec<f32>],
        dims: usize,
    ) -> (NamedTempFile, VectorStore<MmapBackend>) {
        build_store_blocked(vectors, dims, StorageParams::default().block_size)
    }

    /// Like [`build_store`] but with an explicit (small) block size, so vectors
    /// span many blocks and the fetch path exercises real run coalescing.
    fn build_store_blocked(
        vectors: &[Vec<f32>],
        dims: usize,
        block_size: usize,
    ) -> (NamedTempFile, VectorStore<MmapBackend>) {
        let tmp = NamedTempFile::new().unwrap();
        let layout = BlockLayout::new(Dtype::F32, dims, block_size).unwrap();
        let mut writer = StoreWriter::create(tmp.path(), layout).unwrap();
        for v in vectors {
            writer.push(v).unwrap();
        }
        writer.finish().unwrap();
        let store = VectorStore::open_mmap(tmp.path()).unwrap();
        (tmp, store)
    }

    /// Independent brute-force KNN over the in-memory vectors, the recall
    /// reference. Returns ids of the `k` nearest, ascending.
    fn naive_knn(vectors: &[Vec<f32>], query: &[f32], metric: Metric, k: usize) -> Vec<u64> {
        let mut scored: Vec<Neighbor> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let s = slate_simd::distance(metric, query, v).unwrap();
                Neighbor::new(VectorId::new(i as u64), s)
            })
            .collect();
        scored.sort_unstable_by(slate_core::cmp_ascending);
        scored.truncate(k);
        scored.into_iter().map(|n| n.id.get()).collect()
    }

    /// `n` clustered vectors in `dims` dimensions: a handful of Gaussian-ish
    /// blobs so the IVF partition is meaningful (uniform noise would make every
    /// cell equivalent and recall hard to interpret).
    fn clustered_vectors(seed: u64, n: usize, dims: usize) -> Vec<Vec<f32>> {
        let mut rng = SplitMix64::new(seed);
        let num_centers = 12;
        let centers: Vec<Vec<f32>> = (0..num_centers)
            .map(|_| (0..dims).map(|_| rng.next_f64() as f32 * 10.0).collect())
            .collect();
        (0..n)
            .map(|_| {
                let c = &centers[(rng.next_u64() as usize) % num_centers];
                (0..dims)
                    .map(|j| c[j] + (rng.next_f64() as f32 - 0.5))
                    .collect()
            })
            .collect()
    }

    /// IVF params with a modest list count appropriate for the small test
    /// corpora (so each list holds enough vectors for recall to be meaningful).
    fn ivf_params(num_lists: usize, num_probes: usize, soft_assign: usize) -> IvfParams {
        IvfParams {
            num_lists,
            num_probes,
            soft_assign,
            max_kmeans_iters: 25,
        }
    }

    /// Mean recall@`k` over `queries` random queries, scored against the
    /// in-RAM `naive_knn` oracle under the L2 metric (the only metric these
    /// IVF tests exercise). `dims` is derived from the dataset.
    fn mean_recall(
        index: &IvfIndex,
        store: &VectorStore<MmapBackend>,
        vectors: &[Vec<f32>],
        k: usize,
        queries: usize,
        query_seed: u64,
    ) -> f64 {
        let dims = vectors[0].len();
        let mut rng = SplitMix64::new(query_seed);
        let cfg = SearchConfig {
            k,
            ..SearchConfig::default()
        };
        let mut total = 0.0f64;
        for _ in 0..queries {
            let query: Vec<f32> = (0..dims).map(|_| rng.next_f64() as f32 * 10.0).collect();
            let (res, _stats) = index.search(store, &query, &cfg).unwrap();
            let truth: HashSet<u64> = naive_knn(vectors, &query, Metric::L2, k)
                .into_iter()
                .collect();
            let hits = res.iter().filter(|n| truth.contains(&n.id.get())).count();
            total += hits as f64 / k as f64;
        }
        total / queries as f64
    }

    #[test]
    fn empty_store_yields_empty_index_and_search() {
        let (_tmp, store) = build_store(&[], 8);
        let index = IvfIndex::build(&store, Metric::L2, &ivf_params(16, 4, 2), 1).unwrap();
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
        let cfg = SearchConfig::default();
        let (res, stats) = index.search(&store, &[0.0; 8], &cfg).unwrap();
        assert!(res.is_empty());
        assert_eq!(stats.counters.exact_distances, 0);
        assert_eq!(stats.counters.seeks, 0);
    }

    #[test]
    fn single_vector_is_its_own_nearest() {
        let vectors = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let (_tmp, store) = build_store(&vectors, 4);
        let index = IvfIndex::build(&store, Metric::L2, &ivf_params(16, 4, 2), 7).unwrap();
        assert_eq!(index.len(), 1);
        let cfg = SearchConfig {
            k: 1,
            ..SearchConfig::default()
        };
        let (res, _stats) = index.search(&store, &[1.0, 2.0, 3.0, 4.0], &cfg).unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].id.get(), 0);
    }

    #[test]
    fn dimension_mismatch_is_reported() {
        let vectors = vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]];
        let (_tmp, store) = build_store(&vectors, 4);
        let index = IvfIndex::build(&store, Metric::L2, &ivf_params(16, 4, 2), 1).unwrap();
        let cfg = SearchConfig::default();
        let err = index.search(&store, &[1.0, 2.0, 3.0], &cfg).unwrap_err();
        match err {
            Error::DimensionMismatch { expected, got } => {
                assert_eq!(expected, 4);
                assert_eq!(got, 3);
            }
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn recall_is_high_on_clustered_data() {
        let dims = 16;
        let n = 400;
        let vectors = clustered_vectors(2024, n, dims);
        let (_tmp, store) = build_store(&vectors, dims);
        // 16 lists over 12 true clusters, probe 8, soft-assign 2.
        let params = ivf_params(16, 8, 2);
        let index = IvfIndex::build(&store, Metric::L2, &params, 555).unwrap();
        let recall = mean_recall(&index, &store, &vectors, 10, 30, 7);
        assert!(
            recall >= 0.85,
            "IVF mean recall@10 {recall:.3} below floor 0.85"
        );
    }

    #[test]
    fn recall_rises_with_more_probes() {
        let dims = 16;
        let n = 400;
        let vectors = clustered_vectors(2024, n, dims);
        let (_tmp, store) = build_store(&vectors, dims);

        // Same index, two probe budgets. More probes can only union in more
        // (or equal) candidates, so recall is monotone non-decreasing.
        let few = IvfIndex::build(&store, Metric::L2, &ivf_params(24, 1, 1), 555).unwrap();
        let many = IvfIndex::build(&store, Metric::L2, &ivf_params(24, 12, 1), 555).unwrap();
        let recall_few = mean_recall(&few, &store, &vectors, 10, 30, 7);
        let recall_many = mean_recall(&many, &store, &vectors, 10, 30, 7);
        assert!(
            recall_many >= recall_few,
            "more probes lowered recall: {recall_few:.3} -> {recall_many:.3}"
        );
        // And the high-probe configuration should be genuinely good.
        assert!(
            recall_many >= 0.80,
            "high-probe recall@10 {recall_many:.3} below floor 0.80"
        );
    }

    #[test]
    fn build_and_search_are_deterministic() {
        let dims = 16;
        let n = 300;
        let vectors = clustered_vectors(99, n, dims);
        let (_tmp, store) = build_store(&vectors, dims);
        let params = ivf_params(16, 6, 2);

        let a = IvfIndex::build(&store, Metric::L2, &params, 4242).unwrap();
        let b = IvfIndex::build(&store, Metric::L2, &params, 4242).unwrap();

        // Identical centroids and posting lists.
        assert_eq!(a.num_lists(), b.num_lists());
        for l in 0..a.num_lists() {
            assert_eq!(a.posting_list(l), b.posting_list(l));
        }

        // Identical search results + counters.
        let cfg = SearchConfig {
            k: 10,
            ..SearchConfig::default()
        };
        let query: Vec<f32> = (0..dims).map(|j| (j as f32) * 0.3).collect();
        let (res_a, stats_a) = a.search(&store, &query, &cfg).unwrap();
        let (res_b, stats_b) = b.search(&store, &query, &cfg).unwrap();
        let ids_a: Vec<u64> = res_a.iter().map(|n| n.id.get()).collect();
        let ids_b: Vec<u64> = res_b.iter().map(|n| n.id.get()).collect();
        assert_eq!(ids_a, ids_b);
        assert_eq!(stats_a.counters, stats_b.counters);
    }

    /// An [`IoBackend`] that tallies every `read_exact_at`, so a test can see the
    /// real syscall count the coalesced fetch path issues.
    struct CountingBackend<B: IoBackend> {
        inner: B,
        reads: AtomicUsize,
    }

    impl<B: IoBackend> CountingBackend<B> {
        fn new(inner: B) -> Self {
            Self {
                inner,
                reads: AtomicUsize::new(0),
            }
        }

        fn reads(&self) -> usize {
            self.reads.load(Ordering::Relaxed)
        }
    }

    impl<B: IoBackend> IoBackend for CountingBackend<B> {
        fn len(&self) -> usize {
            self.inner.len()
        }

        fn read_exact_at(&self, offset: usize, buf: &mut [u8]) -> Result<()> {
            self.reads.fetch_add(1, Ordering::Relaxed);
            self.inner.read_exact_at(offset, buf)
        }

        fn advise_range(&self, offset: usize, len: usize, advice: Advice) -> Result<()> {
            self.inner.advise_range(offset, len, advice)
        }
    }

    #[test]
    fn probe_fetch_coalesces_reads_into_runs() {
        // On a small-block store, a probe's candidates share/neighbour blocks,
        // so the coalesced fetch must issue strictly fewer reads than candidates
        // — and exactly `seeks`/`runs` reads — while recall stays high.
        let dims = 16;
        let n = 400;
        let vectors = clustered_vectors(2025, n, dims);
        // block_size 256, dims-16 f32 => 64 B/vec => 4 vectors/block.
        let (tmp, _open) = build_store_blocked(&vectors, dims, 256);
        let index = {
            let store = VectorStore::open_mmap(tmp.path()).unwrap();
            IvfIndex::build(&store, Metric::L2, &ivf_params(16, 8, 2), 555).unwrap()
        };
        // Re-open the SAME file behind a counting backend.
        let store =
            VectorStore::with_backend(CountingBackend::new(MmapBackend::open(tmp.path()).unwrap()))
                .unwrap();

        let cfg = SearchConfig {
            k: 10,
            ..SearchConfig::default()
        };
        let mut rng = SplitMix64::new(7);
        let mut total_reads = 0usize;
        let mut total_runs = 0u64;
        let mut total_candidates = 0u64;
        let mut total_recall = 0.0f64;
        let queries = 30;
        for _ in 0..queries {
            let query: Vec<f32> = (0..dims).map(|_| rng.next_f64() as f32 * 10.0).collect();
            let before = store.backend().reads();
            let (res, stats) = index.search(&store, &query, &cfg).unwrap();
            let delta = store.backend().reads() - before;

            // Real syscall count equals the planned run/seek count for this probe.
            assert_eq!(delta as u64, stats.counters.seeks);
            assert_eq!(stats.counters.seeks, stats.counters.sequential_runs);
            // Each candidate produced exactly one exact distance.
            assert!(stats.counters.seeks <= stats.counters.exact_distances);

            total_reads += delta;
            total_runs += stats.counters.sequential_runs;
            total_candidates += stats.counters.exact_distances;

            let truth: HashSet<u64> =
                naive_knn(&vectors, &query, Metric::L2, 10).into_iter().collect();
            let hits = res.iter().filter(|n| truth.contains(&n.id.get())).count();
            total_recall += hits as f64 / 10.0;
        }

        // Coalescing actually fired: fewer reads than candidates fetched.
        assert!(
            (total_reads as u64) < total_candidates,
            "no coalescing: reads {total_reads} >= candidates {total_candidates}"
        );
        assert_eq!(total_reads as u64, total_runs);
        let recall = total_recall / queries as f64;
        assert!(recall >= 0.80, "recall@10 {recall:.3} below floor 0.80");
    }
}
