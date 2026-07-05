# Slate-ANN

Slate-ANN is a high-performance, disk-backed approximate nearest-neighbor (ANN) search engine for vector retrieval when the full vector corpus cannot stay resident in memory.

The engine is built around storage-aware search: query latency is modeled as a combination of graph traversal, storage access, and distance computation. That makes physical reads, seek count, layout locality, vector encoding, and reranking policy first-class concerns instead of treating vector fetches as uniform-latency memory loads.

## Features

- Rust workspace with layered crates for core types, SIMD distance kernels, storage, product quantization, graph indexes, bundle persistence, and CLI tooling.
- Exact brute-force baseline used as a recall oracle for approximate backends.
- HNSW backend with exact disk-streamed distance evaluation.
- IVF backend with soft-assigned posting lists.
- Resident product-quantization tier for candidate generation and hybrid reranking.
- Graph-aware on-disk layout ordering and HDD-oriented elevator scheduling.
- Narrow on-disk vector dtypes (`f32`, `f16`, `i8`) with reusable decode scratch buffers.
- Self-describing index bundles with save/load support, soft deletes, and buffered inserts.
- Cost counters and benchmark harness for comparing modeled storage and distance costs.

## Workspace Layout

| Crate | Purpose |
| --- | --- |
| `slate-core` | Shared IDs, metrics, config, error types, search results, and query-cost model. |
| `slate-simd` | Runtime-dispatched distance kernels for scalar, AVX2, AVX-512, and NEON targets. |
| `slate-storage` | Disk vector format, mmap/pread backends, layout, scheduling, and typed decoding. |
| `slate-pq` | Product-quantization training, encoding, and ADC lookup tables. |
| `slate-graph` | HNSW and IVF approximate-search backends. |
| `slate-index` | Brute-force oracle, bundle format, persistence, and update log. |
| `slate-cli` | `slate` command-line interface for build, query, bench, insert, and delete workflows. |

## Build

Slate-ANN is a Cargo workspace. The repository declares its supported Rust toolchain in `Cargo.toml`.

```powershell
cargo build --workspace
cargo test --workspace
```

For release builds:

```powershell
cargo build --workspace --release
```

## CLI Quick Start

The CLI reads plain-text vectors: one vector per line, whitespace-separated `f32` values.

```text
0.1 0.2 0.3 0.4
0.2 0.1 0.4 0.3
```

Build a bundle:

```powershell
cargo run -p slate-cli -- build vectors.txt bundle --backend hnsw --metric l2
```

Query the bundle:

```powershell
cargo run -p slate-cli -- query bundle query.txt --k 10
```

Benchmark modeled storage cost:

```powershell
cargo run -p slate-cli -- bench bundle queries.txt --k 10 --profile hdd --recall
```

Apply buffered updates:

```powershell
cargo run -p slate-cli -- insert bundle new-vector.txt
cargo run -p slate-cli -- delete bundle --id 42
```

## Design Notes

Traditional ANN implementations optimize graph hops under an implicit uniform-latency assumption. That assumption is reasonable when both the graph and vectors are memory-resident, but it breaks down when exact vectors live on storage where random seeks and sequential reads have very different costs.

Slate-ANN separates the cost model into traversal, storage access, and distance computation. This allows the engine to evaluate design choices such as graph-aware layout, batched/elevator reads, narrower vector encodings, resident PQ proxies, and exact reranking as explicit tradeoffs.

See `docs/storage-aware-search.md` for the longer design rationale.

## License

Licensed under either of:

- MIT license (`LICENSE-MIT`)
- Apache License, Version 2.0 (`LICENSE-APACHE`)

at your option.
