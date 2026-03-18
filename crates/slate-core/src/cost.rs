//! The storage-aware query-cost model.
//!
//! Slate-ANN is built on the thesis that approximate-nearest-neighbour search
//! should be **storage-aware**: the physical medium and the on-disk layout are
//! first-class terms in the query-cost objective, not an afterthought hidden
//! behind a uniform-latency assumption. See `docs/storage-aware-search.md` for
//! the full argument.
//!
//! The objective decomposes the cost of answering a query into three additive
//! terms:
//!
//! ```text
//! QueryCost = TraversalCost + StorageAccessCost + DistanceComputationCost
//! ```
//!
//! This module provides the **vocabulary and an estimator** for that objective:
//!
//! * [`StorageProfile`] — the parameters of a storage medium (seek latency,
//!   sequential bandwidth, block granularity), with presets for an HDD, an
//!   NVMe SSD, and RAM.
//! * [`DistanceCost`] — the per-operation cost of approximate and exact distance
//!   computations on the active CPU.
//! * [`QueryCounters`] — the **physical counters** a search accumulates as it
//!   runs (nodes visited, seeks issued, bytes read, distance ops). These are
//!   what the engine actually measures.
//! * [`QueryCost`] — the three-term decomposition, obtained by pricing a set of
//!   [`QueryCounters`] against a [`StorageProfile`] and a [`DistanceCost`].
//!
//! It is deliberately **not** a query planner: nothing here decides traversal
//! order or fetch policy. It names the terms later phases optimize and lets a
//! measured query be priced after the fact, so the paradigm is evaluated with
//! numbers rather than asserted.

use serde::{Deserialize, Serialize};

/// Parameters of a storage medium, expressed in the units the cost model needs.
///
/// The two parameters that matter most on non-uniform storage are the **seek
/// latency** (the fixed price of positioning at a read) and the **sequential
/// bandwidth** (the marginal price per byte once positioned). On a 7200rpm HDD
/// the former dominates the latter by orders of magnitude, which is precisely
/// why minimizing *seek count* — not bytes, and not graph hops — is the lever.
///
/// Classical in-RAM ANN implicitly assumes [`StorageProfile::memory`], where the
/// seek latency is ~0 and bandwidth is effectively unbounded; under that profile
/// the storage-access term vanishes and only graph hops remain. Slate-ANN takes
/// the profile as an *input* instead.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StorageProfile {
    /// Latency to position at a fresh random read, in seconds.
    ///
    /// For an HDD this is head-seek plus average rotational latency; for an SSD
    /// it is a small controller/queue constant; for RAM it is ~0.
    pub seek_latency_s: f64,
    /// Sustained sequential transfer rate, in bytes per second.
    pub sequential_bandwidth_bps: f64,
    /// Natural transfer granularity of the medium, in bytes.
    ///
    /// Reads smaller than this still pay for at least this many bytes of
    /// transfer (HDD sector / SSD page / OS page). Used when pricing a read
    /// whose payload is smaller than one block.
    pub block_bytes: u64,
}

impl StorageProfile {
    /// A representative 7200rpm consumer hard disk drive.
    ///
    /// ~9 ms seek (≈4.2 ms average rotational latency at 7200rpm plus head
    /// seek), ~160 MB/s sequential, 4 KiB minimum transfer. This is the headline
    /// target medium: the regime where the uniform-latency assumption is most
    /// catastrophically wrong.
    #[inline]
    pub const fn hdd_7200rpm() -> Self {
        Self {
            seek_latency_s: 0.009,
            sequential_bandwidth_bps: 160.0 * 1_000_000.0,
            block_bytes: 4096,
        }
    }

    /// A representative consumer NVMe solid-state drive.
    ///
    /// ~100 µs effective random-access latency, ~3.5 GB/s sequential, 4 KiB
    /// page. Random reads are cheap here, which is why SSD-tuned designs (e.g.
    /// DiskANN) can afford one small random read per hop.
    #[inline]
    pub const fn ssd_nvme() -> Self {
        Self {
            seek_latency_s: 0.000_1,
            sequential_bandwidth_bps: 3_500.0 * 1_000_000.0,
            block_bytes: 4096,
        }
    }

    /// Resident memory: the implicit medium of classical in-RAM ANN.
    ///
    /// Negligible seek, very high bandwidth, cache-line granularity. Under this
    /// profile [`QueryCost::storage_access_s`] is ~0 and query cost is dominated
    /// by traversal and distance computation — the classical special case.
    #[inline]
    pub const fn memory() -> Self {
        Self {
            seek_latency_s: 0.0,
            sequential_bandwidth_bps: 20_000.0 * 1_000_000.0,
            block_bytes: 64,
        }
    }

    /// Seconds to transfer `bytes` at the medium's sequential bandwidth.
    ///
    /// This is the honest transfer time for an already-summed byte count: zero
    /// bytes costs zero. Per-read block-granularity flooring lives in
    /// [`read_s`](Self::read_s), since "a read smaller than a block still pays
    /// for a block" is a property of an individual read, not of an aggregate.
    #[inline]
    pub fn transfer_s(self, bytes: u64) -> f64 {
        bytes as f64 / self.sequential_bandwidth_bps
    }

    /// Modelled latency of a single read of `bytes`: one seek plus the transfer
    /// of at least one block.
    #[inline]
    pub fn read_s(self, bytes: u64) -> f64 {
        self.seek_latency_s + self.transfer_s(bytes.max(self.block_bytes))
    }
}

/// Per-operation cost of distance computation on the active CPU.
///
/// Approximate distances are PQ/ADC table lookups over RAM-resident codes and
/// are cheap; exact distances are full `d`-dimensional SIMD reductions over a
/// fetched vector. Keeping the two priced separately is what makes the
/// approximate-gate trade — spend cheap approximate ops to avoid expensive exact
/// fetches — legible in the cost model.
///
/// Costs are in seconds per operation and are expected to be obtained by offline
/// micro-benchmarking (the SIMD bench harness) for a given dimensionality,
/// dtype, and instruction set.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DistanceCost {
    /// Seconds per approximate (PQ/ADC lookup) distance.
    pub approx_s: f64,
    /// Seconds per exact (full SIMD) distance.
    pub exact_s: f64,
}

impl DistanceCost {
    /// Construct an explicit per-op cost pair.
    #[inline]
    pub const fn new(approx_s: f64, exact_s: f64) -> Self {
        Self { approx_s, exact_s }
    }
}

/// Physical counters accumulated by a single query as it executes.
///
/// These are the quantities the engine actually measures and the quantities the
/// storage-aware claims are stated in terms of. A hops-only view records only
/// `nodes_visited`; the storage-access terms (`seeks`, `bytes_read`,
/// `sequential_runs`) are exactly what that view omits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct QueryCounters {
    /// Nodes popped from the candidate queue (graph hops). Drives traversal cost.
    pub nodes_visited: u64,
    /// Approximate (PQ/ADC) distances computed.
    pub approx_distances: u64,
    /// Exact (full SIMD) distances computed on fetched vectors.
    pub exact_distances: u64,
    /// Physical positioning operations (seeks) issued to storage.
    pub seeks: u64,
    /// Total bytes read from storage.
    pub bytes_read: u64,
    /// Number of coalesced sequential runs the reads collapsed into.
    ///
    /// With perfect demand paging this equals `seeks`; with elevator scheduling
    /// and graph-aware layout it can be far smaller, which is the effect the
    /// scheduler is meant to produce.
    pub sequential_runs: u64,
}

impl QueryCounters {
    /// A zeroed counter set.
    #[inline]
    pub const fn new() -> Self {
        Self {
            nodes_visited: 0,
            approx_distances: 0,
            exact_distances: 0,
            seeks: 0,
            bytes_read: 0,
            sequential_runs: 0,
        }
    }

    /// Record a node pop (one graph hop).
    #[inline]
    pub fn visit_node(&mut self) {
        self.nodes_visited += 1;
    }

    /// Record `n` approximate distance computations.
    #[inline]
    pub fn add_approx(&mut self, n: u64) {
        self.approx_distances += n;
    }

    /// Record `n` exact distance computations.
    #[inline]
    pub fn add_exact(&mut self, n: u64) {
        self.exact_distances += n;
    }

    /// Record a physical read of `bytes` that required `seeks` positioning
    /// operations and collapsed into `runs` sequential runs.
    #[inline]
    pub fn add_read(&mut self, bytes: u64, seeks: u64, runs: u64) {
        self.bytes_read += bytes;
        self.seeks += seeks;
        self.sequential_runs += runs;
    }

    /// Merge another counter set into this one (e.g. across parallel workers).
    #[inline]
    pub fn merge(&mut self, other: &QueryCounters) {
        self.nodes_visited += other.nodes_visited;
        self.approx_distances += other.approx_distances;
        self.exact_distances += other.exact_distances;
        self.seeks += other.seeks;
        self.bytes_read += other.bytes_read;
        self.sequential_runs += other.sequential_runs;
    }
}

/// The three-term query-cost decomposition, in seconds.
///
/// Produced by [`QueryCost::estimate`], which prices a set of [`QueryCounters`]
/// against a [`StorageProfile`] and a [`DistanceCost`]. The point of keeping the
/// terms separate (rather than only their sum) is that the storage-aware claims
/// are about how a change shifts cost *between* terms — e.g. a narrower dtype
/// moves cost out of `storage_access_s`, a larger approximate gate moves cost
/// out of `storage_access_s` and into `distance_s`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QueryCost {
    /// `c_hop · |V|` — best-first bookkeeping over visited nodes (RAM-bound).
    pub traversal_s: f64,
    /// `Σ (t_seek + bytes/B_seq)` — physical reads. The term prior work assumes
    /// uniform; the dominant term on HDD-class media.
    pub storage_access_s: f64,
    /// `c_dist · (|A| + |E|)` — approximate plus exact distance computation.
    pub distance_s: f64,
}

impl QueryCost {
    /// Per-visited-node traversal overhead, in seconds.
    ///
    /// Best-first bookkeeping (queue ops, adjacency expansion, visited-set
    /// maintenance) is small and RAM-bound; this is a representative constant,
    /// not a tuned value. Traversal is intentionally the cheap term in the
    /// storage-aware regime.
    pub const HOP_OVERHEAD_S: f64 = 50e-9;

    /// Price `counters` against a storage `profile` and distance `cost`.
    ///
    /// Storage access is modelled as one seek per `seeks` counted plus transfer
    /// of `bytes_read` at the medium's sequential bandwidth — so a query that
    /// coalesces its reads into fewer seeks (via layout + scheduling) is priced
    /// strictly lower at equal bytes, which is the behaviour the paradigm
    /// predicts and the scheduler is built to produce.
    #[inline]
    pub fn estimate(
        counters: &QueryCounters,
        profile: StorageProfile,
        cost: DistanceCost,
    ) -> Self {
        let traversal_s = counters.nodes_visited as f64 * Self::HOP_OVERHEAD_S;
        let storage_access_s = counters.seeks as f64 * profile.seek_latency_s
            + profile.transfer_s(counters.bytes_read);
        let distance_s = counters.approx_distances as f64 * cost.approx_s
            + counters.exact_distances as f64 * cost.exact_s;
        Self {
            traversal_s,
            storage_access_s,
            distance_s,
        }
    }

    /// Total modelled query latency: the sum of the three terms.
    #[inline]
    pub fn total_s(self) -> f64 {
        self.traversal_s + self.storage_access_s + self.distance_s
    }

    /// Fraction of total cost attributable to storage access, in `[0, 1]`.
    ///
    /// A diagnostic for *which regime a query is in*: near 1 means the query is
    /// storage-bound (the paradigm's target regime), near 0 means it is
    /// compute- or traversal-bound (the classical in-RAM regime).
    #[inline]
    pub fn storage_fraction(self) -> f64 {
        let total = self.total_s();
        if total == 0.0 {
            0.0
        } else {
            self.storage_access_s / total
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hdd_seek_dominates_single_vector_transfer() {
        // One 768-dim f32 vector is ~3 KiB; on HDD the seek must dwarf the
        // transfer, which is the entire motivation for minimizing seeks.
        let hdd = StorageProfile::hdd_7200rpm();
        let transfer = hdd.transfer_s(3 * 1024);
        assert!(
            hdd.seek_latency_s > transfer * 100.0,
            "seek {} should dwarf transfer {}",
            hdd.seek_latency_s,
            transfer
        );
    }

    #[test]
    fn transfer_is_honest_and_read_floors_to_a_block() {
        let hdd = StorageProfile::hdd_7200rpm();
        // Aggregate transfer is honest: zero bytes is free, bytes scale linearly.
        assert_eq!(hdd.transfer_s(0), 0.0);
        assert!(hdd.transfer_s(2 * 1024) > hdd.transfer_s(1024));
        // A single sub-block read is charged for a full block by read_s.
        assert_eq!(hdd.read_s(1), hdd.read_s(hdd.block_bytes));
        // A read larger than a block is charged for its true size.
        let big = hdd.block_bytes * 10;
        assert!(hdd.read_s(big) > hdd.read_s(hdd.block_bytes));
    }

    #[test]
    fn read_is_seek_plus_floored_transfer() {
        let ssd = StorageProfile::ssd_nvme();
        let bytes = 8192; // larger than the 4 KiB block, so the floor is a no-op
        assert_eq!(
            ssd.read_s(bytes),
            ssd.seek_latency_s + ssd.transfer_s(bytes)
        );
    }

    #[test]
    fn memory_profile_has_negligible_seek() {
        let mem = StorageProfile::memory();
        assert_eq!(mem.seek_latency_s, 0.0);
        // Under the memory profile a read is pure transfer, no seek. A 64-byte
        // read equals one block, so the floor is a no-op here.
        assert_eq!(mem.read_s(64), mem.transfer_s(64));
    }

    #[test]
    fn counters_accumulate_and_merge() {
        let mut a = QueryCounters::new();
        a.visit_node();
        a.add_approx(10);
        a.add_exact(2);
        a.add_read(6144, 2, 1);
        assert_eq!(a.nodes_visited, 1);
        assert_eq!(a.approx_distances, 10);
        assert_eq!(a.exact_distances, 2);
        assert_eq!(a.bytes_read, 6144);
        assert_eq!(a.seeks, 2);
        assert_eq!(a.sequential_runs, 1);

        let mut b = QueryCounters::new();
        b.add_read(1024, 1, 1);
        b.merge(&a);
        assert_eq!(b.bytes_read, 6144 + 1024);
        assert_eq!(b.seeks, 3);
        assert_eq!(b.nodes_visited, 1);
    }

    #[test]
    fn estimate_decomposes_three_terms() {
        let mut c = QueryCounters::new();
        c.nodes_visited = 100;
        c.approx_distances = 1000;
        c.exact_distances = 50;
        c.seeks = 50;
        c.bytes_read = 50 * 3072;
        let cost = DistanceCost::new(5e-9, 200e-9);
        let qc = QueryCost::estimate(&c, StorageProfile::hdd_7200rpm(), cost);

        assert_eq!(qc.traversal_s, 100.0 * QueryCost::HOP_OVERHEAD_S);
        assert_eq!(qc.distance_s, 1000.0 * 5e-9 + 50.0 * 200e-9);
        // 50 random seeks on HDD ≈ 50 * 9 ms = 0.45 s, the dominant term.
        assert!(qc.storage_access_s > 0.4);
        assert!((qc.total_s()
            - (qc.traversal_s + qc.storage_access_s + qc.distance_s))
            .abs()
            < 1e-12);
    }

    #[test]
    fn coalescing_seeks_lowers_cost_at_equal_bytes() {
        // Same bytes, same exact-distance work; the only difference is how many
        // seeks the reads collapsed into. The storage-aware model must price the
        // coalesced plan strictly cheaper — this is claim (3) in miniature.
        let bytes = 64 * 3072;
        let cost = DistanceCost::new(5e-9, 200e-9);
        let hdd = StorageProfile::hdd_7200rpm();

        let mut scattered = QueryCounters::new();
        scattered.add_read(bytes, 64, 64);
        let mut coalesced = QueryCounters::new();
        coalesced.add_read(bytes, 1, 1);

        let scattered_cost = QueryCost::estimate(&scattered, hdd, cost);
        let coalesced_cost = QueryCost::estimate(&coalesced, hdd, cost);
        assert!(scattered_cost.storage_access_s > coalesced_cost.storage_access_s);
        // And the scattered query is firmly storage-bound.
        assert!(scattered_cost.storage_fraction() > 0.9);
    }

    #[test]
    fn memory_profile_collapses_the_seek_term() {
        // The same query, priced under RAM vs HDD. The seek-driven storage cost
        // (100 random seeks) is what the uniform-latency assumption omits; under
        // the memory profile it collapses by orders of magnitude, leaving the
        // query compute/traversal bound — the classical degenerate case.
        let mut c = QueryCounters::new();
        c.nodes_visited = 100;
        c.exact_distances = 100;
        c.seeks = 100;
        c.bytes_read = 100 * 3072;
        let cost = DistanceCost::new(5e-9, 200e-9);

        let hdd = QueryCost::estimate(&c, StorageProfile::hdd_7200rpm(), cost);
        let mem = QueryCost::estimate(&c, StorageProfile::memory(), cost);

        // HDD is dominated by seeks; RAM is not storage-bound at all.
        assert!(hdd.storage_fraction() > 0.9);
        assert!(mem.storage_fraction() < 0.5);
        // The storage term shrinks by orders of magnitude across the two media.
        assert!(mem.storage_access_s < hdd.storage_access_s / 100.0);
    }

    #[test]
    fn zero_cost_has_zero_storage_fraction() {
        let qc = QueryCost::estimate(
            &QueryCounters::new(),
            StorageProfile::hdd_7200rpm(),
            DistanceCost::new(5e-9, 200e-9),
        );
        assert_eq!(qc.total_s(), 0.0);
        assert_eq!(qc.storage_fraction(), 0.0);
    }
}
