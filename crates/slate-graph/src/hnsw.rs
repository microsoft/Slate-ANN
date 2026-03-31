//! In-RAM HNSW (Hierarchical Navigable Small World) proximity graph with
//! exact-distance ranking.
//!
//! # What this is (Phase 4)
//!
//! A from-scratch HNSW build + search. The graph topology (per-node adjacency
//! lists, one set per layer) lives in RAM; vector data is **not** held by the
//! index after construction — at query time exact vectors are streamed from a
//! [`VectorStore`] through its [`IoBackend`], which is the seam the
//! storage-aware cost model measures. Every search accumulates a
//! [`QueryCounters`] (nodes visited, exact distances computed, bytes read).
//!
//! # Algorithm
//!
//! Standard HNSW (Malkov & Yashunin, 2016):
//!
//! - Each node is assigned a maximum **level** drawn from a geometric
//!   distribution `floor(-ln(U) * mL)` with `mL = 1 / ln(m)`. Most nodes live
//!   only on layer 0; a logarithmically thinning subset reaches higher layers,
//!   forming an express skip-list-like structure over the base graph.
//! - **Insertion** greedily descends the upper layers (beam width 1) from the
//!   current entry point to find a good entry into the node's top layer, then
//!   runs a width-`ef_construction` best-first search on each layer at or below
//!   it, selecting up to `m` (`m_max` on layer 0) neighbors and adding
//!   bidirectional links, pruning back any neighbor that exceeds its cap.
//! - **Search** descends the upper layers to layer 0, then runs a width-
//!   `ef_search` best-first search there and returns the best `k`.
//!
//! Neighbor selection here is the simple "closest first" heuristic; LEANN's
//! high-degree-preserving pruning is a later phase. Distances follow the
//! engine-wide ascending convention via [`slate_simd::distance`].

use slate_core::{
    Error, Metric, Neighbor, QueryCounters, Result, SearchConfig, TopK, VectorId,
};
use slate_storage::{IoBackend, VectorStore};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::rng::SplitMix64;

/// Parameters controlling graph shape, mirroring [`slate_core::HnswParams`] but
/// resolved into the fields the builder uses directly.
#[derive(Debug, Clone, Copy)]
struct GraphParams {
    /// Neighbor cap on layers above 0.
    m: usize,
    /// Neighbor cap on layer 0 (denser base layer).
    m_max: usize,
    /// Best-first beam width during construction.
    ef_construction: usize,
    /// Level-generation normalization factor `mL = 1 / ln(m)`.
    level_mult: f64,
}

impl GraphParams {
    fn from_config(p: &slate_core::HnswParams) -> Self {
        // m >= 2 guarantees a finite, positive level multiplier. `validate`
        // already enforces m >= 1; guard the degenerate m == 1 to avoid
        // ln(1) == 0 producing an infinite multiplier.
        let m = p.m.max(2);
        let level_mult = 1.0 / (m as f64).ln();
        Self {
            m: p.m,
            m_max: p.m_max,
            ef_construction: p.ef_construction,
            level_mult,
        }
    }

    /// Neighbor cap for a given layer (layer 0 is the dense base layer).
    #[inline]
    fn cap_for_layer(&self, layer: usize) -> usize {
        if layer == 0 {
            self.m_max
        } else {
            self.m
        }
    }
}

/// Per-search statistics, exposing the physical work the storage-aware cost
/// model prices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HnswStats {
    /// Counters accumulated during the search (nodes visited, exact distances,
    /// bytes read from the store).
    pub counters: QueryCounters,
}

/// A candidate on the search frontier, ordered for a **min**-heap by score.
///
/// `BinaryHeap` is a max-heap, so we invert the comparison: the smallest score
/// (closest, most promising) compares greatest and is popped first.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Candidate {
    node: u32,
    score: f32,
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        // Inverted: smaller score => "greater" so it surfaces first in a
        // max-heap. Tie-break by id for determinism (smaller id first =>
        // "greater").
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| other.node.cmp(&self.node))
    }
}

/// An in-RAM HNSW index over a fixed set of vectors.
///
/// The index stores only graph topology and a little per-node metadata; vectors
/// are read from the backing [`VectorStore`] on demand at search time. Build it
/// with [`HnswIndex::build`], query it with [`HnswIndex::search`].
#[derive(Debug)]
pub struct HnswIndex {
    /// `adjacency[layer][node]` = neighbor ids of `node` on `layer`.
    /// Layer 0 covers all nodes; higher layers cover progressively fewer.
    adjacency: Vec<Vec<Vec<u32>>>,
    /// Maximum level each node participates in (0-based; 0 = base layer only).
    node_levels: Vec<u8>,
    /// Entry point: the node with the highest level. `None` iff the graph is
    /// empty.
    entry_point: Option<u32>,
    /// Highest layer index that exists in the graph.
    max_layer: usize,
    /// Number of nodes (== number of vectors in the store).
    len: usize,
    /// Vector dimensionality.
    dims: usize,
    /// Distance metric.
    metric: Metric,
    /// Resolved graph parameters.
    params: GraphParams,
}

impl HnswIndex {
    /// Build an HNSW graph over every vector in `store`.
    ///
    /// `params` controls graph shape; `seed` makes level assignment (and hence
    /// the whole graph) deterministic. Vectors are read once from the store and
    /// held in RAM only for the duration of the build.
    ///
    /// # Errors
    ///
    /// Returns an error if the store's dtype is unsupported (non-`F32`) or a
    /// vector read fails.
    pub fn build<B: IoBackend>(
        store: &VectorStore<B>,
        metric: Metric,
        params: &slate_core::HnswParams,
        seed: u64,
    ) -> Result<Self> {
        let dims = store.dimensions();
        let len = store.len();
        let gp = GraphParams::from_config(params);

        // Pull every vector into RAM for the build. The build is distance-bound
        // and touches vectors repeatedly; streaming each from disk per
        // comparison would be pathological. Search, by contrast, streams.
        let mut data = vec![0.0f32; len.saturating_mul(dims)];
        for i in 0..len {
            let start = i * dims;
            store.get_into(i, &mut data[start..start + dims])?;
        }

        let mut index = Self {
            adjacency: vec![vec![Vec::new(); len]],
            node_levels: vec![0u8; len],
            entry_point: None,
            max_layer: 0,
            len,
            dims,
            metric,
            params: gp,
        };

        let mut rng = SplitMix64::new(seed);
        for node in 0..len {
            let level = index.assign_level(&mut rng);
            index.node_levels[node] = level as u8;
            index.insert_node(node as u32, level, &data)?;
        }

        Ok(index)
    }

    /// Number of indexed vectors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the index holds no vectors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The highest layer index in the graph (0 if flat or empty).
    #[must_use]
    pub fn max_layer(&self) -> usize {
        self.max_layer
    }

    /// Draw a level from the geometric distribution `floor(-ln(U) * mL)`.
    fn assign_level(&self, rng: &mut SplitMix64) -> usize {
        // U in (0, 1]; guard against log(0). next_f64 is [0,1), so use 1 - U.
        let u = 1.0 - rng.next_f64();
        (-(u.ln()) * self.params.level_mult).floor() as usize
    }

    /// Exact distance between an in-RAM query slice and node `b`'s vector,
    /// also held in RAM (build path).
    #[inline]
    fn dist_ram(&self, query: &[f32], b: u32, data: &[f32]) -> Result<f32> {
        let start = b as usize * self.dims;
        slate_simd::distance(self.metric, query, &data[start..start + self.dims])
    }

    /// Insert one node into the graph, wiring its neighbors on every layer up to
    /// `level`. Operates entirely on in-RAM `data`.
    fn insert_node(&mut self, node: u32, level: usize, data: &[f32]) -> Result<()> {
        let node_start = node as usize * self.dims;
        let query = &data[node_start..node_start + self.dims];

        // Grow the per-layer adjacency structure if this node introduces new
        // upper layers.
        if level > self.max_layer {
            for _ in self.max_layer..level {
                self.adjacency.push(vec![Vec::new(); self.len]);
            }
            self.max_layer = level;
        }

        // First node becomes the entry point and has no neighbors to wire.
        let Some(entry) = self.entry_point else {
            self.entry_point = Some(node);
            return Ok(());
        };

        // Phase 1: greedily descend from the entry point through the layers
        // ABOVE this node's top level, beam width 1, to find a good entry.
        let mut current = entry;
        let mut current_dist = self.dist_ram(query, current, data)?;
        let mut layer = self.max_layer;
        while layer > level {
            self.greedy_descend(query, &mut current, &mut current_dist, layer, data)?;
            layer -= 1;
        }

        // Phase 2: from this node's top layer down to 0, run a width-
        // ef_construction search and connect.
        let mut entry_points = vec![current];
        for layer in (0..=level).rev() {
            let candidates =
                self.search_layer(query, &entry_points, self.params.ef_construction, layer, data)?;

            let cap = self.params.cap_for_layer(layer);
            let selected = select_neighbors(&candidates, cap);

            // Wire bidirectional links and prune over-full neighbors.
            for &nbr in &selected {
                self.connect(node, nbr, layer);
                self.prune_node(nbr, layer, data)?;
            }

            // Next lower layer starts its search from this layer's found
            // candidates.
            entry_points = candidates.iter().map(|c| c.node).collect();
            if entry_points.is_empty() {
                entry_points = vec![current];
            }
        }

        Ok(())
    }

    /// Move `current` to its closest neighbor on `layer` repeatedly until no
    /// neighbor improves the distance (beam width 1 greedy hill-climb).
    fn greedy_descend(
        &self,
        query: &[f32],
        current: &mut u32,
        current_dist: &mut f32,
        layer: usize,
        data: &[f32],
    ) -> Result<()> {
        let mut improved = true;
        while improved {
            improved = false;
            let neighbors = &self.adjacency[layer][*current as usize];
            for &nbr in neighbors {
                let d = self.dist_ram(query, nbr, data)?;
                if d < *current_dist {
                    *current_dist = d;
                    *current = nbr;
                    improved = true;
                }
            }
        }
        Ok(())
    }

    /// Best-first search confined to a single `layer`, starting from
    /// `entry_points`, returning up to `ef` closest nodes (unsorted set, as
    /// [`Candidate`]s). In-RAM distance variant used during construction.
    fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[u32],
        ef: usize,
        layer: usize,
        data: &[f32],
    ) -> Result<Vec<Candidate>> {
        let mut visited = vec![false; self.len];
        // Frontier: min-heap by score (closest pops first).
        let mut frontier: BinaryHeap<Candidate> = BinaryHeap::new();
        // Results: we keep the ef best; track the worst via a max-heap-like
        // TopK over node ids.
        let mut results: BinaryHeap<ResultEntry> = BinaryHeap::new();

        for &ep in entry_points {
            if visited[ep as usize] {
                continue;
            }
            visited[ep as usize] = true;
            let d = self.dist_ram(query, ep, data)?;
            frontier.push(Candidate { node: ep, score: d });
            results.push(ResultEntry { node: ep, score: d });
            if results.len() > ef {
                results.pop();
            }
        }

        while let Some(cand) = frontier.pop() {
            // Stop when the closest frontier node is worse than the worst kept
            // result and we already have ef of them.
            if results.len() >= ef {
                if let Some(worst) = results.peek() {
                    if cand.score > worst.score {
                        break;
                    }
                }
            }
            let neighbors = &self.adjacency[layer][cand.node as usize];
            for &nbr in neighbors {
                if visited[nbr as usize] {
                    continue;
                }
                visited[nbr as usize] = true;
                let d = self.dist_ram(query, nbr, data)?;
                let worst_score = results.peek().map(|r| r.score);
                if results.len() < ef || worst_score.is_none_or(|w| d < w) {
                    frontier.push(Candidate { node: nbr, score: d });
                    results.push(ResultEntry { node: nbr, score: d });
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        Ok(results
            .into_iter()
            .map(|r| Candidate {
                node: r.node,
                score: r.score,
            })
            .collect())
    }

    /// Add a directed link both ways between `a` and `b` on `layer`, avoiding
    /// duplicates and self-loops.
    fn connect(&mut self, a: u32, b: u32, layer: usize) {
        if a == b {
            return;
        }
        let la = &mut self.adjacency[layer][a as usize];
        if !la.contains(&b) {
            la.push(b);
        }
        let lb = &mut self.adjacency[layer][b as usize];
        if !lb.contains(&a) {
            lb.push(a);
        }
    }

    /// If `node` exceeds its neighbor cap on `layer`, keep only the closest
    /// `cap` neighbors (simple distance-based pruning).
    fn prune_node(&mut self, node: u32, layer: usize, data: &[f32]) -> Result<()> {
        let cap = self.params.cap_for_layer(layer);
        let neighbors = self.adjacency[layer][node as usize].clone();
        if neighbors.len() <= cap {
            return Ok(());
        }
        let node_start = node as usize * self.dims;
        let query = &data[node_start..node_start + self.dims];

        let mut scored: Vec<Candidate> = Vec::with_capacity(neighbors.len());
        for &nbr in &neighbors {
            let d = self.dist_ram(query, nbr, data)?;
            scored.push(Candidate { node: nbr, score: d });
        }
        let kept = select_neighbors(&scored, cap);
        self.adjacency[layer][node as usize] = kept;
        Ok(())
    }

    /// Search the graph for the `k` nearest neighbors of `query`, streaming
    /// exact vectors from `store`.
    ///
    /// Returns the neighbors (ascending, best first) together with the
    /// [`HnswStats`] accumulated during traversal. `config.k` sets the result
    /// size; `config.ef_search` sets the layer-0 beam width.
    ///
    /// # Errors
    ///
    /// Returns an error on dimension mismatch, unsupported dtype, or a failed
    /// vector read.
    pub fn search<B: IoBackend>(
        &self,
        store: &VectorStore<B>,
        query: &[f32],
        config: &SearchConfig,
    ) -> Result<(Vec<Neighbor>, HnswStats)> {
        if query.len() != self.dims {
            return Err(Error::DimensionMismatch {
                expected: self.dims,
                got: query.len(),
            });
        }
        let mut counters = QueryCounters::new();

        let Some(entry) = self.entry_point else {
            // Empty graph.
            return Ok((Vec::new(), HnswStats { counters }));
        };

        let vector_bytes = (self.dims * std::mem::size_of::<f32>()) as u64;
        let mut scratch = vec![0.0f32; self.dims];

        // Descend the upper layers greedily (beam width 1) to find a good entry
        // into layer 0.
        let mut current = entry;
        let mut current_dist =
            self.dist_disk(store, query, current, &mut scratch, &mut counters, vector_bytes)?;
        let mut layer = self.max_layer;
        while layer > 0 {
            let mut improved = true;
            while improved {
                improved = false;
                let neighbors = &self.adjacency[layer][current as usize];
                for &nbr in neighbors {
                    let d = self.dist_disk(
                        store,
                        query,
                        nbr,
                        &mut scratch,
                        &mut counters,
                        vector_bytes,
                    )?;
                    if d < current_dist {
                        current_dist = d;
                        current = nbr;
                        improved = true;
                    }
                }
            }
            layer -= 1;
        }

        // Width-ef_search best-first search on layer 0.
        let ef = config.ef_search.max(config.k).max(1);
        let results =
            self.search_layer_disk(store, query, current, ef, &mut scratch, &mut counters, vector_bytes)?;

        // Collect the best k.
        let mut topk = TopK::new(config.k);
        for c in results {
            topk.offer(Neighbor::new(VectorId::new(u64::from(c.node)), c.score));
        }
        let neighbors = topk.into_sorted_vec();

        Ok((neighbors, HnswStats { counters }))
    }

    /// Exact distance to node `b`, streaming its vector from the store and
    /// bumping the storage + distance counters.
    #[inline]
    fn dist_disk<B: IoBackend>(
        &self,
        store: &VectorStore<B>,
        query: &[f32],
        b: u32,
        scratch: &mut [f32],
        counters: &mut QueryCounters,
        vector_bytes: u64,
    ) -> Result<f32> {
        store.get_into(b as usize, scratch)?;
        counters.visit_node();
        counters.add_exact(1);
        // One vector fetch == one read of `vector_bytes`, modeled as a single
        // seek + transfer. The HDD-aware I/O scheduler (Phase 7) will coalesce
        // these; here each is independent.
        counters.add_read(vector_bytes, 1, 1);
        slate_simd::distance(self.metric, query, scratch)
    }

    /// Layer-0 best-first search reading exact vectors from the store.
    #[allow(clippy::too_many_arguments)]
    fn search_layer_disk<B: IoBackend>(
        &self,
        store: &VectorStore<B>,
        query: &[f32],
        entry: u32,
        ef: usize,
        scratch: &mut [f32],
        counters: &mut QueryCounters,
        vector_bytes: u64,
    ) -> Result<Vec<Candidate>> {
        let mut visited = vec![false; self.len];
        let mut frontier: BinaryHeap<Candidate> = BinaryHeap::new();
        let mut results: BinaryHeap<ResultEntry> = BinaryHeap::new();

        visited[entry as usize] = true;
        let d0 = self.dist_disk(store, query, entry, scratch, counters, vector_bytes)?;
        frontier.push(Candidate {
            node: entry,
            score: d0,
        });
        results.push(ResultEntry {
            node: entry,
            score: d0,
        });

        while let Some(cand) = frontier.pop() {
            if results.len() >= ef {
                if let Some(worst) = results.peek() {
                    if cand.score > worst.score {
                        break;
                    }
                }
            }
            let neighbors = &self.adjacency[0][cand.node as usize];
            for &nbr in neighbors {
                if visited[nbr as usize] {
                    continue;
                }
                visited[nbr as usize] = true;
                let d = self.dist_disk(store, query, nbr, scratch, counters, vector_bytes)?;
                let worst_score = results.peek().map(|r| r.score);
                if results.len() < ef || worst_score.is_none_or(|w| d < w) {
                    frontier.push(Candidate { node: nbr, score: d });
                    results.push(ResultEntry { node: nbr, score: d });
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        Ok(results
            .into_iter()
            .map(|r| Candidate {
                node: r.node,
                score: r.score,
            })
            .collect())
    }
}

/// An entry in the bounded result set of a layer search, ordered for a max-heap
/// so the **worst** (largest score) is at the top for eviction.
#[derive(Debug, Clone, Copy, PartialEq)]
struct ResultEntry {
    node: u32,
    score: f32,
}

impl Eq for ResultEntry {}

impl PartialOrd for ResultEntry {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ResultEntry {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        // Larger score => greater => top of max-heap (evicted first).
        self.score
            .total_cmp(&other.score)
            .then_with(|| self.node.cmp(&other.node))
    }
}

/// Select up to `cap` closest neighbors from a candidate set (simple heuristic:
/// sort by ascending score, take the front).
fn select_neighbors(candidates: &[Candidate], cap: usize) -> Vec<u32> {
    let mut sorted: Vec<Candidate> = candidates.to_vec();
    sorted.sort_unstable_by(|a, b| {
        a.score
            .total_cmp(&b.score)
            .then_with(|| a.node.cmp(&b.node))
    });
    sorted.truncate(cap);
    sorted.into_iter().map(|c| c.node).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use slate_core::{Dtype, HnswParams, StorageParams};
    use slate_storage::{BlockLayout, StoreWriter, VectorStore};
    use tempfile::NamedTempFile;

    /// Write `vectors` (row-major, `dims` each) to a temp store and open it
    /// mmap-backed. Returns the temp file (kept alive) and the store.
    fn build_store(
        vectors: &[Vec<f32>],
        dims: usize,
    ) -> (NamedTempFile, VectorStore<slate_storage::MmapBackend>) {
        let tmp = NamedTempFile::new().unwrap();
        let block_size = StorageParams::default().block_size;
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

    fn default_params() -> HnswParams {
        HnswParams::default()
    }

    #[test]
    fn empty_graph_returns_empty() {
        let (_tmp, store) = build_store(&[], 4);
        let index = HnswIndex::build(&store, Metric::L2, &default_params(), 1).unwrap();
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
        let cfg = SearchConfig::default();
        let (res, stats) = index.search(&store, &[0.0, 0.0, 0.0, 0.0], &cfg).unwrap();
        assert!(res.is_empty());
        assert_eq!(stats.counters.nodes_visited, 0);
    }

    #[test]
    fn single_vector_is_its_own_nearest() {
        let vectors = vec![vec![1.0, 2.0, 3.0]];
        let (_tmp, store) = build_store(&vectors, 3);
        let index = HnswIndex::build(&store, Metric::L2, &default_params(), 7).unwrap();
        assert_eq!(index.len(), 1);
        let cfg = SearchConfig {
            k: 1,
            ..SearchConfig::default()
        };
        let (res, _stats) = index.search(&store, &[1.0, 2.0, 3.0], &cfg).unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].id, VectorId::new(0));
        assert!(res[0].score.abs() < 1e-6);
    }

    #[test]
    fn finds_exact_nearest_on_small_grid() {
        // 2D grid of well-separated points; nearest is unambiguous.
        let mut vectors = Vec::new();
        for x in 0..5 {
            for y in 0..5 {
                vectors.push(vec![x as f32, y as f32]);
            }
        }
        let (_tmp, store) = build_store(&vectors, 2);
        let index = HnswIndex::build(&store, Metric::L2, &default_params(), 123).unwrap();

        let query = vec![2.1, 2.9];
        let cfg = SearchConfig {
            k: 1,
            ef_search: 32,
            ..SearchConfig::default()
        };
        let (res, _stats) = index.search(&store, &query, &cfg).unwrap();
        let expect = naive_knn(&vectors, &query, Metric::L2, 1);
        assert_eq!(res[0].id.get(), expect[0]);
    }

    #[test]
    fn dimension_mismatch_is_reported() {
        let vectors = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let (_tmp, store) = build_store(&vectors, 2);
        let index = HnswIndex::build(&store, Metric::L2, &default_params(), 1).unwrap();
        let cfg = SearchConfig::default();
        let err = index.search(&store, &[1.0, 2.0, 3.0], &cfg).unwrap_err();
        assert!(matches!(err, Error::DimensionMismatch { .. }));
    }

    #[test]
    fn counters_are_populated() {
        let vectors: Vec<Vec<f32>> = (0..50).map(|i| vec![i as f32, (i % 7) as f32]).collect();
        let (_tmp, store) = build_store(&vectors, 2);
        let index = HnswIndex::build(&store, Metric::L2, &default_params(), 99).unwrap();
        let cfg = SearchConfig {
            k: 5,
            ef_search: 16,
            ..SearchConfig::default()
        };
        let (_res, stats) = index.search(&store, &[10.0, 3.0], &cfg).unwrap();
        // Search must visit at least k nodes and read a vector per visit.
        assert!(stats.counters.nodes_visited >= 5);
        assert_eq!(stats.counters.exact_distances, stats.counters.nodes_visited);
        let expected_bytes = stats.counters.nodes_visited * 2 * 4;
        assert_eq!(stats.counters.bytes_read, expected_bytes);
    }

    #[test]
    fn recall_is_high_on_clustered_data() {
        // Deterministic pseudo-random clustered vectors.
        let mut rng = SplitMix64::new(2024);
        let dims = 16;
        let n = 400;
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dims).map(|_| rng.next_f64() as f32).collect())
            .collect();
        let (_tmp, store) = build_store(&vectors, dims);
        let params = HnswParams {
            m: 16,
            m_max: 32,
            ef_construction: 200,
            ..HnswParams::default()
        };
        let index = HnswIndex::build(&store, Metric::L2, &params, 555).unwrap();

        let k = 10;
        let cfg = SearchConfig {
            k,
            ef_search: 64,
            ..SearchConfig::default()
        };

        let queries = 30;
        let mut total_recall = 0.0f64;
        for _ in 0..queries {
            let query: Vec<f32> = (0..dims).map(|_| rng.next_f64() as f32).collect();
            let (res, _stats) = index.search(&store, &query, &cfg).unwrap();
            let truth = naive_knn(&vectors, &query, Metric::L2, k);
            let truth_set: std::collections::HashSet<u64> = truth.into_iter().collect();
            let hits = res.iter().filter(|n| truth_set.contains(&n.id.get())).count();
            total_recall += hits as f64 / k as f64;
        }
        let mean_recall = total_recall / f64::from(queries);
        assert!(
            mean_recall >= 0.90,
            "mean recall@{k} was {mean_recall}, expected >= 0.90"
        );
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;
    use slate_core::{Dtype, HnswParams, StorageParams};
    use slate_storage::{BlockLayout, StoreWriter, VectorStore};
    use std::collections::HashSet;
    use tempfile::NamedTempFile;

    prop_compose! {
        fn dataset()(
            dims in 2usize..=12,
            count in 20usize..=120,
        )(
            query in prop::collection::vec(-10.0f32..10.0, dims),
            vectors in prop::collection::vec(
                prop::collection::vec(-10.0f32..10.0, dims),
                count,
            ),
            k in 1usize..=10,
            metric_idx in 0usize..3,
        ) -> (Vec<f32>, Vec<Vec<f32>>, usize, Metric) {
            let metric = [Metric::L2, Metric::InnerProduct, Metric::Cosine][metric_idx];
            (query, vectors, k, metric)
        }
    }

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

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(60))]

        /// HNSW with a generous beam recovers most of the true nearest
        /// neighbors. We assert a recall floor (not exact match) because HNSW is
        /// approximate; the floor is high because ef_search is set well above k.
        #[test]
        fn recall_meets_floor_vs_oracle((query, vectors, k, metric) in dataset()) {
            let dims = query.len();
            let tmp = NamedTempFile::new().unwrap();
            let block_size = StorageParams::default().block_size;
            let layout = BlockLayout::new(Dtype::F32, dims, block_size).unwrap();
            let mut writer = StoreWriter::create(tmp.path(), layout).unwrap();
            for v in &vectors {
                writer.push(v).unwrap();
            }
            writer.finish().unwrap();
            let store = VectorStore::open_mmap(tmp.path()).unwrap();

            let params = HnswParams {
                m: 16,
                m_max: 32,
                ef_construction: 128,
                ..HnswParams::default()
            };
            let index = HnswIndex::build(&store, metric, &params, 0xC0FFEE).unwrap();

            let cfg = SearchConfig {
                k,
                ef_search: 64,
                ..SearchConfig::default()
            };
            let (res, _stats) = index.search(&store, &query, &cfg).unwrap();

            let truth = naive_knn(&vectors, &query, metric, k);
            let truth_set: HashSet<u64> = truth.iter().copied().collect();
            let hits = res.iter().filter(|n| truth_set.contains(&n.id.get())).count();
            let recall = hits as f64 / truth.len() as f64;

            // Result count never exceeds k or the dataset size.
            prop_assert!(res.len() <= k);
            prop_assert!(res.len() <= vectors.len());
            // Approximate recall floor.
            prop_assert!(
                recall >= 0.6,
                "recall {recall} below floor; k={k} metric={metric:?} n={}",
                vectors.len()
            );
        }
    }
}
