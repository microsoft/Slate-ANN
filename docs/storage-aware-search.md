# Storage-Aware Approximate Nearest Neighbor Search

*A positioning and concept note for Slate-ANN.*

**Status:** working draft (the conceptual contribution; intended to be citable
independently of the implementation).

---

## 1. Thesis

Mainstream approximate-nearest-neighbor (ANN) search optimizes **graph
traversal** under an unstated assumption: that accessing any vector costs the
same as accessing any other. Indexes are tuned to minimize the number of graph
hops (equivalently, distance computations), because under **uniform storage
latency** hop count is a faithful proxy for query latency.

That assumption holds when the entire index — graph *and* vectors — lives in
RAM. It breaks the moment vectors live on a medium whose access cost is
**non-uniform**: a 7200 rpm hard disk pays ~8–12 ms for a random seek but
streams ~100–200 MB/s sequentially, a 4–5 order-of-magnitude gap between the
*worst* and *best* access pattern for the *same* number of bytes. On such media
two query plans that perform the **same number of hops** can differ by more than
an order of magnitude in wall-clock latency, depending solely on *where the
touched vectors physically sit* and *in what order they are read*.

**Slate-ANN's claim is that ANN search should be storage-aware:** the physical
storage medium and the on-disk layout are first-class terms in the query-cost
objective, and the index should **jointly optimize graph traversal and physical
layout** against that objective — not minimize hops in isolation.

We state the objective explicitly:

```
QueryCost  =  TraversalCost  +  StorageAccessCost  +  DistanceComputationCost
```

Existing systems optimize (mostly) the first term and treat the third as a
fixed per-hop constant, while the second is assumed away (uniform, or zero
because everything is resident). Slate-ANN makes all three explicit and
co-optimizes them.

> This note argues the paradigm. The companion engineering — a from-scratch Rust
> implementation with custom mmap, a disk-resident layout, an elevator I/O
> scheduler, and multi-tier SIMD distance kernels — is the *evidence* that the
> paradigm is realizable and that optimizing the full objective beats optimizing
> hops alone on non-uniform storage.

---

## 2. The cost model

Let a query `q` execute a best-first graph search that **visits** a set of nodes
`V` (popped from the candidate queue) and, for some subset, **fetches** the
exact vector from storage and **computes** an exact distance. We decompose the
cost of answering `q` into three additive terms.

### 2.1 Traversal cost

```
TraversalCost = c_hop · |V|
```

The bookkeeping of best-first search itself: popping the priority queue,
expanding adjacency lists, maintaining the visited set. `|V|` is the number of
nodes visited; `c_hop` is the (small, RAM-bound) per-node overhead. This is the
term classical ANN minimizes, and the term graph quality (degree distribution,
hub preservation, ef) controls. **In Slate-ANN the graph topology and PQ codes
are RAM-resident, so this term is genuinely cheap and genuinely the only term
that behaves like the classical assumption.**

### 2.2 Storage access cost — *the term prior work assumes uniform*

```
StorageAccessCost = Σ_b ( t_seek(b)  +  bytes(b) / B_seq )
```

summed over the **physical read operations** `b` the query issues. For each read:

- `t_seek(b)` — the latency to *position* at the read (HDD head seek + rotational
  latency; for SSD a much smaller, queue-depth-dependent constant; for RAM ~0).
- `bytes(b) / B_seq` — the transfer time at the medium's sequential bandwidth
  `B_seq`.

The structure of this term is the whole point:

- It is **not** a function of "how many vectors were touched" but of "how many
  *physical positioning operations* were paid and how many bytes moved." Two
  plans touching the same vectors differ here if one **coalesces** neighbor
  reads into few sequential runs and the other scatters them into many random
  seeks.
- On a 7200 rpm HDD, `t_seek ≈ 8–12 ms` dominates everything else by orders of
  magnitude; `bytes/B_seq` for a single 768-dim f32 vector (~3 KB) is ~15–30 µs.
  **Therefore minimizing the *number of seeks* — not the number of bytes, and
  not the number of hops — is the dominant lever.**
- Under the **uniform-latency assumption** of classical ANN, `t_seek ≡ 0` and
  `B_seq → ∞`, so this entire term vanishes and only `|V|` (hops) remains. That
  is precisely the special case prior work optimizes; it is correct *only* when
  the index is RAM-resident.

Levers that move this term (and the Slate-ANN mechanism that pulls each):

| Lever | Effect on `StorageAccessCost` | Slate-ANN mechanism |
|---|---|---|
| **Layout locality** | co-locate graph neighbors physically → neighbor fetches fall in the same block/run → fewer seeks | graph-aware on-disk vector ordering (build-time layout pass) |
| **Access reordering** | sort/coalesce pending reads by offset → convert random seeks into sequential runs | elevator (SCAN) I/O scheduler, batched fetches |
| **Bytes per vector** | narrower dtype → fewer bytes → smaller transfer *and* more vectors per block/seek | f32 / f16 / int8 on-disk dtypes |
| **Block granularity** | larger blocks amortize one seek over more useful vectors (HDD) vs. waste bandwidth (SSD) | medium-aware block size (64 KiB+ on HDD) |
| **Resident proxies** | answer "should I fetch at all?" without touching disk | PQ codes in RAM (approximate distance gate) |

### 2.3 Distance computation cost

```
DistanceComputationCost = c_dist(d, dtype, isa) · ( |A| + |E| )
```

- `|A|` — approximate distances computed (PQ/ADC table lookups over RAM-resident
  codes), cheap per op.
- `|E|` — exact distances computed on fetched vectors, `d`-dimensional SIMD dot/
  L2.
- `c_dist` depends on dimensionality `d`, the on-disk `dtype`, and the CPU `isa`
  (AVX-512 / AVX2 / NEON / scalar).

Classical in-RAM ANN folds this into a per-hop constant. We keep it explicit
because on **low-power CPUs** it is *not* negligible relative to a (now-removed
or reduced) storage term, and because **it trades against the storage term**:
spending more approximate distances `|A|` (cheap, resident) to shrink the set of
exact fetches `|E|` (expensive, on-disk) is a direct `StorageAccessCost ↓` for
`DistanceComputationCost ↑` exchange. This is the LEANN two-level hybrid-distance
idea, *reinterpreted as a deliberate move along the cost-model trade surface*
rather than merely a recall trick.

### 2.4 The objective

A storage-aware index chooses its **layout** `L` and its **search policy** `π`
(traversal order, fetch/rerank decisions, batching) to minimize expected query
cost subject to a recall floor and a storage budget:

```
minimize_{L, π}   E_q [ TraversalCost + StorageAccessCost + DistanceComputationCost ]
subject to        Recall@k ≥ τ
                  IndexSize(L) ≤ S
```

Classical ANN is the degenerate instance where `StorageAccessCost ≡ 0` (resident
data) and `DistanceComputationCost` is a constant per hop, collapsing the
objective to `minimize |V|`. **Slate-ANN refuses that collapse.**

---

## 3. Prior work as special cases

The paradigm is clarifying because well-known systems fall out of it by holding
one or more terms constant or assuming them away.

- **HNSW / in-memory graph ANN.** `StorageAccessCost ≡ 0` (everything resident);
  `DistanceComputationCost` ≈ constant·`|V|`. Objective collapses to *minimize
  hops*. Excellent under the uniform-latency assumption; silently mispriced the
  moment data spills to disk.

- **DiskANN (SSD).** Acknowledges storage by **co-locating each node's vector and
  adjacency in one 4 KB sector** and bounding reads to ~`|V|` sector fetches —
  this is a *layout* optimization of `StorageAccessCost`, and historically the
  first major crack in the uniform-latency assumption. But it is **tuned to SSD's
  cost curve** (cheap random 4 KB reads, negligible seek) and uses
  **pure-approximate traversal + final re-rank**. On HDD the 4 KB-per-node random
  pattern is pathological (one expensive seek per hop, tiny transfer), and the
  sector padding inflates index size. DiskANN optimizes `StorageAccessCost` *for
  one medium's parameters*; it is storage-*specialized*, not storage-*aware* in
  the parametric sense we mean.

- **LEANN.** Drives `IndexSize` to near-zero by **not storing vectors at all** and
  recomputing them with a GPU encoder; `StorageAccessCost` for vectors becomes
  `RecomputeCost` on an accelerator. Brilliant when a fast GPU is present and
  storage is the binding constraint. On a **low-power CPU with no accelerator**,
  recompute is the new bottleneck — the term simply moved, it did not vanish.
  Slate-ANN keeps LEANN's *resident approximate proxy* (PQ codes) and *hub-
  preserving pruning* but **re-stores exact vectors on disk** and attacks the
  resulting `StorageAccessCost` directly, because on our target the disk, not the
  encoder, is the cheap-to-improve resource.

The contribution is not "beat system X." It is: **these systems each implicitly
fix a different subset of the cost model; naming the full model exposes that
their tunings are medium-specific points, and lets an index instead take the
storage medium's parameters as input and co-optimize layout + policy for them.**

---

## 4. Falsifiable claims

The paradigm earns its keep only if it makes predictions a hops-only view does
not. Slate-ANN's evaluation is built to test these:

1. **Hop count is a poor latency predictor on non-uniform storage.** Two search
   policies matched on `|V|` (and on Recall@k) can differ substantially in
   wall-clock latency on HDD; the difference is explained by **seek count and
   sequential-run length**, i.e. by `StorageAccessCost`, not by hops.

2. **Layout locality reduces query latency at fixed recall and fixed hops.**
   Graph-aware physical ordering of vectors lowers seek count versus an
   insertion-order layout, with no change to the graph or the traversal, hence
   no change to recall — isolating `StorageAccessCost` as the moved term.

3. **Access reordering (elevator scheduling + batched fetch) beats demand
   paging on HDD.** Coalescing pending exact-vector fetches by disk offset
   converts random seeks into sequential runs and reduces total query time, the
   I/O-domain analog of trading staleness for throughput.

4. **The dtype/PQ knobs trade `DistanceComputationCost` against
   `StorageAccessCost` along a measurable Pareto front.** Narrower on-disk dtype
   and a larger approximate-gate (more `|A|`, fewer `|E|`) reduce bytes-read and
   seeks at a quantifiable recall/precision cost — a front a hops-only model
   cannot even express.

Each claim references **physical counters** (seeks, bytes read, run lengths,
distance-op counts) that the engine exposes — see §5 — so the paradigm is
measured, not asserted. A **seek-counting I/O backend** (planned alongside the
HDD scheduler) lets these be evaluated deterministically without special
hardware, then confirmed on a real 7200 rpm drive.

---

## 5. How the implementation instantiates the model

The 0–12 build roadmap is re-narrated here as **instantiating named terms of the
objective** rather than as a feature checklist. Each phase pulls specific levers:

| Phase | Deliverable | Primary cost term(s) attacked |
|---|---|---|
| 1 | Multi-tier SIMD distance kernels (AVX-512/AVX2/NEON/scalar) | `DistanceComputationCost` (`c_dist` over `isa`) |
| 2 | Disk-resident block layout, custom mmap, `pread`/mmap backends, `madvise` | `StorageAccessCost` (block granularity, medium hint); the *seam* for the scheduler |
| 3 | Brute-force exact KNN | none directly — establishes the **recall oracle** `τ` is measured against |
| 4–6 | HNSW + high-degree-preserving pruning | `TraversalCost` (`|V|`), and `StorageAccessCost` indirectly (hubs ⇒ fewer fetched nodes) |
| 5 | PQ codes + ADC two-level hybrid search | the `|A|` ↔ `|E|` trade: `DistanceComputationCost` up to push `StorageAccessCost` down |
| 7 | **Elevator I/O scheduler**, graph-aware layout ordering, f16/int8 dtypes, seek-counting backend | `StorageAccessCost` — the paradigm's core term (seeks, runs, bytes) |
| 8 | IVF backend behind the same trait | alternative `TraversalCost`/`StorageAccessCost` profile (sequential list scans) |
| 9 | Storage-efficient sharded build | build-time `IndexSize(L)` budget `S` |
| 10 | Soft-delete + buffered insert | keeps the objective valid under mutation |
| 11 | Intra-/inter-query parallelism + shared scheduler | hides latency of all terms; arbitrates `StorageAccessCost` |
| 12 | Benchmarks + recall harness | **measures all three terms** and tests §4's claims |

The throughline: a storage profile (medium latency/bandwidth/block parameters)
is an **input** to the engine, and phases 5–8 are the machinery that takes that
input and minimizes the stated objective for it — that is what makes the system
storage-*aware* rather than storage-*specialized*.

---

## 6. Relationship to the engineering contribution

The paradigm and the artifact reinforce each other but stand alone:

- **The paradigm** (this note) is citable on its own: it names the
  uniform-latency assumption baked into mainstream ANN, gives the three-term
  cost model, and shows existing systems are special cases. A reviewer can adopt
  "storage-aware ANN search" and the `Query Cost = Traversal + Storage Access +
  Distance Computation` objective without using a line of our code.

- **The artifact** (Slate-ANN) is the existence proof: a low-power-CPU, HDD-class
  vector search engine, written from scratch in Rust, that takes a storage
  profile as input and co-optimizes physical layout and search policy to
  minimize the full objective — and the benchmarks that show doing so beats
  minimizing hops on non-uniform storage.
