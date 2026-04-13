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
//!
//! # Two-level hybrid search (Phase 5)
//!
//! Built with [`HnswIndex::build_with_pq`], the index additionally trains a
//! product-quantization [`slate_pq::PqCodebook`] and keeps every vector's
//! compact PQ code resident in RAM. [`HnswIndex::search_hybrid`] then runs
//! LEANN's two-level traversal: cheap **approximate** (ADC) distances, computed
//! from the in-RAM codes with zero disk I/O, decide which nodes are worth an
//! **exact** vector fetch from the [`VectorStore`]; the exact distances — the
//! only ones trusted for ranking — drive the result set and which nodes are
//! expanded. Approximate scores never enter the result ranking, only the
//! fetch-gating decision, so the configured [`Metric`] still governs the final
//! answer even though the PQ codebook is trained with subspace L2. The payoff
//! is storage-access reduction: most discovered nodes are scored approximately
//! and never incur a disk seek. Exact fetches are gathered into batches
//! (`fetch_batch_size`) so the Phase 7 elevator scheduler can later coalesce
//! their seeks; until then each is still priced as an independent read.

use slate_core::{
    Error, Metric, Neighbor, QueryCounters, Result, SearchConfig, TopK, VectorId,
};
use slate_storage::{IoBackend, VectorStore};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::rng::SplitMix64;

/// k-means iterations used to train the PQ codebook during a PQ-enabled build.
const PQ_TRAIN_ITERS: usize = 25;

/// Mixing constant so the PQ training seed is decorrelated from the graph seed
/// while staying a deterministic function of it.
const PQ_SEED_MIX: u64 = 0x9E37_79B9_7F4A_7C15;

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
    /// PQ codebook for the approximate tier. `Some` only when the index was
    /// built with [`HnswIndex::build_with_pq`]; required by
    /// [`HnswIndex::search_hybrid`].
    codebook: Option<slate_pq::PqCodebook>,
    /// Flat PQ codes for every node (`codes[node * code_len ..]`), empty unless
    /// a `codebook` is present. Resident in RAM so approximate distances cost
    /// no disk I/O.
    codes: Vec<u8>,
}

impl HnswIndex {
    /// Build an HNSW graph over every vector in `store`.
    ///
    /// `params` controls graph shape; `seed` makes level assignment (and hence
    /// the whole graph) deterministic. Vectors are read once from the store and
    /// held in RAM only for the duration of the build.
    ///
    /// The resulting index has no PQ tier, so only [`HnswIndex::search`] (exact
    /// streaming search) is available. Use [`HnswIndex::build_with_pq`] to also
    /// enable [`HnswIndex::search_hybrid`].
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
        Self::build_inner(store, metric, params, None, seed)
    }

    /// Build an HNSW graph **and** a product-quantization tier over every vector
    /// in `store`, enabling [`HnswIndex::search_hybrid`].
    ///
    /// The graph is identical to [`HnswIndex::build`] for the same `params` and
    /// `seed`; additionally the PQ codebook is trained on the same in-RAM copy
    /// of the vectors and every vector is encoded into a compact code kept
    /// resident for the index's lifetime. `pq` must satisfy
    /// `store.dimensions() % pq.num_subquantizers == 0`.
    ///
    /// # Errors
    ///
    /// Returns an error if the store's dtype is unsupported, a vector read
    /// fails, or the PQ parameters are incompatible with the dimensionality.
    pub fn build_with_pq<B: IoBackend>(
        store: &VectorStore<B>,
        metric: Metric,
        params: &slate_core::HnswParams,
        pq: &slate_core::PqParams,
        seed: u64,
    ) -> Result<Self> {
        Self::build_inner(store, metric, params, Some(pq), seed)
    }

    /// Shared build path. When `pq` is `Some` and the store is non-empty, trains
    /// a PQ codebook on the resident vectors and encodes them all.
    fn build_inner<B: IoBackend>(
        store: &VectorStore<B>,
        metric: Metric,
        params: &slate_core::HnswParams,
        pq: Option<&slate_core::PqParams>,
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

        // Train the PQ tier (if requested) while the vectors are resident, then
        // encode them all. The PQ seed is derived from the graph seed so both
        // are deterministic yet independent. An empty store has nothing to
        // quantize, so it stays PQ-less (an empty hybrid search returns empty).
        let (codebook, codes) = match pq {
            Some(pq) if len > 0 => {
                let cb = slate_pq::PqCodebook::train(
                    &data,
                    dims,
                    pq,
                    PQ_TRAIN_ITERS,
                    seed ^ PQ_SEED_MIX,
                )?;
                let codes = cb.encode_batch(&data)?;
                (Some(cb), codes)
            }
            _ => (None, Vec::new()),
        };

        let mut index = Self {
            adjacency: vec![vec![Vec::new(); len]],
            node_levels: vec![0u8; len],
            entry_point: None,
            max_layer: 0,
            len,
            dims,
            metric,
            params: gp,
            codebook,
            codes,
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

    /// Whether this index carries a PQ tier (built via
    /// [`HnswIndex::build_with_pq`]). [`HnswIndex::search_hybrid`] requires it.
    #[must_use]
    pub fn has_pq(&self) -> bool {
        self.codebook.is_some()
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

    /// Approximate (ADC) distance to node `b` from its in-RAM PQ code. Records
    /// one approximate-distance evaluation and performs **no** disk I/O.
    #[inline]
    fn approx_dist(
        &self,
        adc: &slate_pq::AdcTable,
        b: u32,
        counters: &mut QueryCounters,
    ) -> Result<f32> {
        let d = adc.distance_at(&self.codes, b as usize)?;
        counters.add_approx(1);
        Ok(d)
    }

    /// Two-level hybrid search (LEANN): approximate PQ distances gate which
    /// nodes get an exact disk fetch; exact distances rank results and steer
    /// expansion. Returns the best `config.k` neighbors (ascending) with the
    /// [`HnswStats`] accumulated during traversal.
    ///
    /// The upper layers are descended purely on approximate distances (no disk
    /// I/O at all); only the dense layer 0 fetches exact vectors, and only for
    /// the most promising approximate candidates. Approximate scores never enter
    /// the result ranking — the configured [`Metric`] governs the final answer.
    ///
    /// # Errors
    ///
    /// Returns an error on dimension mismatch, a failed vector read, or if the
    /// index was built without a PQ tier (use [`HnswIndex::build_with_pq`]).
    pub fn search_hybrid<B: IoBackend>(
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
            // Empty graph (also the case for an empty PQ-requested build).
            return Ok((Vec::new(), HnswStats { counters }));
        };

        let Some(codebook) = self.codebook.as_ref() else {
            return Err(Error::unsupported(
                "search_hybrid requires an index built with build_with_pq (no PQ tier present)",
            ));
        };

        // Asymmetric distance table: the (un-quantized) query's subvectors vs
        // each subspace's centroids. Built once, reused for every node.
        let adc = slate_pq::AdcTable::build(codebook, query)?;

        let vector_bytes = (self.dims * std::mem::size_of::<f32>()) as u64;
        let mut scratch = vec![0.0f32; self.dims];

        // Upper layers: greedy beam-1 descent on APPROXIMATE distance only. The
        // PQ codes are in RAM, so the entire funnel down to layer 0 costs zero
        // disk seeks; layer 0's exact ranking then corrects the entry choice.
        let mut current = entry;
        let mut current_approx = self.approx_dist(&adc, current, &mut counters)?;
        let mut layer = self.max_layer;
        while layer > 0 {
            let mut improved = true;
            while improved {
                improved = false;
                let neighbors = &self.adjacency[layer][current as usize];
                for &nbr in neighbors {
                    let d = self.approx_dist(&adc, nbr, &mut counters)?;
                    if d < current_approx {
                        current_approx = d;
                        current = nbr;
                        improved = true;
                    }
                }
            }
            layer -= 1;
        }

        // Layer 0: PQ-gated interleaved beam.
        let ef = config.ef_search.max(config.k).max(1);
        let results = self.search_layer_hybrid(
            store,
            query,
            &adc,
            current,
            ef,
            config,
            &mut scratch,
            &mut counters,
            vector_bytes,
        )?;

        let mut topk = TopK::new(config.k);
        for c in results {
            topk.offer(Neighbor::new(VectorId::new(u64::from(c.node)), c.score));
        }
        Ok((topk.into_sorted_vec(), HnswStats { counters }))
    }

    /// Layer-0 PQ-gated hybrid search (LEANN two-level).
    ///
    /// Maintains three structures: an **approximate** frontier (min-heap by ADC
    /// score) of discovered-but-unfetched nodes, an **exact** frontier
    /// (min-heap by true distance) of fetched-but-unexpanded nodes, and the
    /// bounded `ef` result set (max-heap by true distance). Each round promotes
    /// the top `rerank_ratio` fraction of the approximate frontier (capped at
    /// `fetch_batch_size`) to exact via batched disk reads, then expands the
    /// closest exact node, scoring its neighbors approximately. The storage
    /// saving is structural: nodes that never rank highly enough approximately
    /// are never fetched.
    #[allow(clippy::too_many_arguments)]
    fn search_layer_hybrid<B: IoBackend>(
        &self,
        store: &VectorStore<B>,
        query: &[f32],
        adc: &slate_pq::AdcTable,
        entry: u32,
        ef: usize,
        config: &SearchConfig,
        scratch: &mut [f32],
        counters: &mut QueryCounters,
        vector_bytes: u64,
    ) -> Result<Vec<Candidate>> {
        // `discovered`: approximate score computed (queued or beyond).
        // `exact_known`: exact distance already fetched.
        let mut discovered = vec![false; self.len];
        let mut exact_known = vec![false; self.len];

        // Discovered, not yet fetched (min-heap by approximate score).
        let mut approx_pq: BinaryHeap<Candidate> = BinaryHeap::new();
        // Fetched, not yet expanded (min-heap by exact score).
        let mut exact_frontier: BinaryHeap<Candidate> = BinaryHeap::new();
        // Best `ef` by exact score (max-heap so the worst is evicted first).
        let mut results: BinaryHeap<ResultEntry> = BinaryHeap::new();

        discovered[entry as usize] = true;
        let entry_approx = self.approx_dist(adc, entry, counters)?;
        approx_pq.push(Candidate {
            node: entry,
            score: entry_approx,
        });

        let rerank_ratio = config.rerank_ratio;
        let fetch_batch_size = config.fetch_batch_size.max(1);

        loop {
            // ---- FETCH PHASE: promote the most promising approximate
            // candidates to exact, batched to amortize seeks. Take the top
            // `rerank_ratio` fraction of the approximate frontier, capped by
            // `fetch_batch_size`; always at least one so the search progresses.
            let mut fetched_this_round = 0usize;
            if !approx_pq.is_empty() {
                let want = (approx_pq.len() as f32 * rerank_ratio).ceil() as usize;
                let budget = want.clamp(1, fetch_batch_size);

                // Gather the batch of node ids first; a future I/O scheduler
                // (Phase 7) will issue these as one seek-ordered scatter read.
                // Until then each is priced as an independent read by dist_disk.
                let mut batch: Vec<u32> = Vec::with_capacity(budget);
                while batch.len() < budget {
                    let Some(cand) = approx_pq.pop() else {
                        break;
                    };
                    if exact_known[cand.node as usize] {
                        continue;
                    }
                    batch.push(cand.node);
                }

                for &node in &batch {
                    let exact =
                        self.dist_disk(store, query, node, scratch, counters, vector_bytes)?;
                    exact_known[node as usize] = true;
                    exact_frontier.push(Candidate { node, score: exact });
                    results.push(ResultEntry { node, score: exact });
                    if results.len() > ef {
                        results.pop();
                    }
                    fetched_this_round += 1;
                }
            }

            // ---- EXPAND PHASE: expand the closest exact node; its graph
            // neighbors get cheap approximate scores and join the approximate
            // frontier for a future fetch decision.
            let mut expanded_this_round = false;
            if let Some(best) = exact_frontier.pop() {
                // Best-first stop (exact vs exact — valid; approximate scores
                // are never compared here): once the closest unexpanded exact
                // node is no better than the worst kept result and the result
                // set is full, no further expansion can improve it.
                let can_improve = if results.len() >= ef {
                    results.peek().is_none_or(|w| best.score <= w.score)
                } else {
                    true
                };
                if can_improve {
                    expanded_this_round = true;
                    let neighbors = &self.adjacency[0][best.node as usize];
                    for &nbr in neighbors {
                        if discovered[nbr as usize] {
                            continue;
                        }
                        discovered[nbr as usize] = true;
                        let a = self.approx_dist(adc, nbr, counters)?;
                        approx_pq.push(Candidate { node: nbr, score: a });
                    }
                } else {
                    // The exact frontier can no longer improve the results; the
                    // remaining approximate candidates rank worse, so stop.
                    break;
                }
            }

            // ---- TERMINATION: no fetch and no expansion means both frontiers
            // are exhausted of useful work.
            if fetched_this_round == 0 && !expanded_this_round {
                break;
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
    use slate_core::{Dtype, HnswParams, PqParams, StorageParams};
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

    /// PQ config used by the hybrid tests: 4 subspaces of 4 dims each, 64
    /// centroids per subspace — a healthy `n >> k` regime for the test sizes.
    fn hybrid_pq() -> PqParams {
        PqParams {
            num_subquantizers: 4,
            bits_per_code: 6,
        }
    }

    #[test]
    fn build_without_pq_has_none() {
        let vectors = vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]];
        let (_tmp, store) = build_store(&vectors, 4);
        let index = HnswIndex::build(&store, Metric::L2, &default_params(), 1).unwrap();
        assert!(!index.has_pq());
    }

    #[test]
    fn search_hybrid_without_pq_is_unsupported() {
        let vectors = vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]];
        let (_tmp, store) = build_store(&vectors, 4);
        let index = HnswIndex::build(&store, Metric::L2, &default_params(), 1).unwrap();
        let cfg = SearchConfig::default();
        let err = index
            .search_hybrid(&store, &[1.0, 2.0, 3.0, 4.0], &cfg)
            .unwrap_err();
        // Robust against the exact error variant name.
        assert!(
            err.to_string().contains("build_with_pq"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn hybrid_empty_graph_returns_empty() {
        let (_tmp, store) = build_store(&[], 4);
        let index = HnswIndex::build_with_pq(&store, Metric::L2, &default_params(), &hybrid_pq(), 1)
            .unwrap();
        assert!(!index.has_pq(), "empty store has nothing to quantize");
        let cfg = SearchConfig::default();
        let (res, stats) = index
            .search_hybrid(&store, &[0.0, 0.0, 0.0, 0.0], &cfg)
            .unwrap();
        assert!(res.is_empty());
        assert_eq!(stats.counters.nodes_visited, 0);
    }

    #[test]
    fn hybrid_single_vector_is_its_own_nearest() {
        let vectors = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let (_tmp, store) = build_store(&vectors, 4);
        let index = HnswIndex::build_with_pq(&store, Metric::L2, &default_params(), &hybrid_pq(), 7)
            .unwrap();
        assert!(index.has_pq());
        let cfg = SearchConfig {
            k: 1,
            ..SearchConfig::default()
        };
        let (res, _stats) = index
            .search_hybrid(&store, &[1.0, 2.0, 3.0, 4.0], &cfg)
            .unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].id, VectorId::new(0));
        assert!(res[0].score.abs() < 1e-6);
    }

    #[test]
    fn hybrid_recall_matches_oracle_and_saves_fetches() {
        // Random vectors; the hybrid path must approach exact recall because the
        // exact re-ranking governs the final ordering, while fetching far fewer
        // than every vector from "disk".
        let mut rng = SplitMix64::new(2025);
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
        let index =
            HnswIndex::build_with_pq(&store, Metric::L2, &params, &hybrid_pq(), 555).unwrap();

        let k = 10;
        let cfg = SearchConfig {
            k,
            ef_search: 64,
            ..SearchConfig::default()
        };

        let queries = 30;
        let mut total_recall = 0.0f64;
        let mut total_approx = 0u64;
        let mut total_exact = 0u64;
        for q in 0..queries {
            let query: Vec<f32> = (0..dims).map(|_| rng.next_f64() as f32).collect();
            let (res, stats) = index.search_hybrid(&store, &query, &cfg).unwrap();
            let truth = naive_knn(&vectors, &query, Metric::L2, k);
            let hits = res
                .iter()
                .filter(|nb| truth.contains(&nb.id.get()))
                .count();
            total_recall += hits as f64 / k as f64;

            // Approximate distances must be recorded (the new cost term), and we
            // must approximate-score at least as many nodes as we fetch exactly.
            assert!(stats.counters.approx_distances > 0, "query {q}");
            assert!(stats.counters.approx_distances >= stats.counters.exact_distances);
            // Exact distances == disk fetches == nodes visited in this tier.
            assert_eq!(stats.counters.exact_distances, stats.counters.nodes_visited);
            // The whole point: not every vector is fetched from disk.
            assert!(
                stats.counters.exact_distances < n as u64,
                "fetched {} of {n} vectors on query {q}",
                stats.counters.exact_distances
            );
            total_approx += stats.counters.approx_distances;
            total_exact += stats.counters.exact_distances;
        }
        let mean_recall = total_recall / queries as f64;
        assert!(
            mean_recall >= 0.80,
            "hybrid recall@{k} too low: {mean_recall:.3}"
        );
        // Sanity: across all queries we genuinely skipped many exact fetches.
        assert!(
            total_exact < total_approx,
            "no fetch saving: exact={total_exact} approx={total_approx}"
        );
    }

    #[test]
    fn hybrid_search_is_deterministic() {
        let mut rng = SplitMix64::new(77);
        let dims = 16;
        let n = 200;
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dims).map(|_| rng.next_f64() as f32).collect())
            .collect();
        let (_tmp, store) = build_store(&vectors, dims);
        let index =
            HnswIndex::build_with_pq(&store, Metric::L2, &default_params(), &hybrid_pq(), 9).unwrap();
        let cfg = SearchConfig {
            k: 10,
            ef_search: 48,
            ..SearchConfig::default()
        };
        let query: Vec<f32> = (0..dims).map(|_| rng.next_f64() as f32).collect();
        let (a, sa) = index.search_hybrid(&store, &query, &cfg).unwrap();
        let (b, sb) = index.search_hybrid(&store, &query, &cfg).unwrap();
        let ids_a: Vec<u64> = a.iter().map(|n| n.id.get()).collect();
        let ids_b: Vec<u64> = b.iter().map(|n| n.id.get()).collect();
        assert_eq!(ids_a, ids_b);
        assert_eq!(sa.counters, sb.counters);
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
