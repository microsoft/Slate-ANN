//! `slate` — the command-line front end for the Slate-ANN disk-backed vector
//! search engine.
//!
//! Three subcommands are wired in Phase 10:
//!
//! - `slate build <vectors> <out>` reads a plain-text vector file (one vector
//!   per line, whitespace-separated `f32`), writes a self-describing index
//!   *bundle* directory (vector store + index + resolved [`BuildConfig`]
//!   manifest), so the index can later be opened without re-stating any
//!   parameters.
//! - `slate query <bundle> <query> --k N` opens such a bundle and prints the
//!   `k` nearest neighbours of the first query vector as `id score` lines.
//! - `slate bench <bundle> <queries> --profile P` runs a query workload, times
//!   it, accumulates the [`QueryCounters`] the engine records, and prices them
//!   through [`QueryCost::estimate`] against a [`StorageProfile`] so the
//!   storage fraction of modeled latency is a printed number.
//! - `slate delete <bundle> --id N` and `slate insert <bundle> <vector>` edit a
//!   bundle through its `updates.json` log: deletes tombstone a stored or
//!   buffered id, inserts append a vector that is brute-scanned and merged into
//!   results, all without rewriting the seek-optimised store or index.
//!
//! The plain-text format is deliberately dependency-free; richer binary inputs
//! (`fvecs`/`npy`) and incremental updates are deferred to later Phase-10 turns.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use slate_core::{
    BuildConfig, DistanceCost, Error, IndexBackend, Metric, QueryCost, QueryCounters, Result,
    SearchConfig, StorageProfile, VectorId,
};

#[derive(Parser)]
#[command(
    name = "slate",
    about = "Disk-backed vector search engine (storage-aware ANN)",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Build a self-describing index bundle from a plain-text vector file.
    Build {
        /// Plain-text vectors: one vector per line, whitespace-separated floats.
        vectors: PathBuf,
        /// Output bundle directory (created if missing).
        out: PathBuf,
        /// Index backend: `hnsw` or `ivf`.
        #[arg(long, default_value = "hnsw")]
        backend: String,
        /// Distance metric: `l2`, `cosine`, or `inner` (aka `ip`).
        #[arg(long, default_value = "l2")]
        metric: String,
        /// Deterministic build seed.
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Query an index bundle with the first vector of a plain-text query file.
    Query {
        /// Bundle directory produced by `slate build`.
        bundle: PathBuf,
        /// Plain-text query file; only the first vector is used.
        query: PathBuf,
        /// Number of neighbours to return.
        #[arg(long, default_value_t = 10)]
        k: usize,
    },
    /// Benchmark a bundle over a query workload and price the storage cost.
    Bench {
        /// Bundle directory produced by `slate build`.
        bundle: PathBuf,
        /// Plain-text query file; every vector is run as one query.
        queries: PathBuf,
        /// Number of neighbours to return per query.
        #[arg(long, default_value_t = 10)]
        k: usize,
        /// Storage profile to price the cost model against: `hdd`, `ssd`, or `memory`.
        #[arg(long, default_value = "hdd")]
        profile: String,
        /// Also measure recall@k against the exact brute-force oracle.
        #[arg(long)]
        recall: bool,
    },
    /// Soft-delete a vector id from a bundle (records a tombstone in the log).
    Delete {
        /// Bundle directory produced by `slate build`.
        bundle: PathBuf,
        /// Vector id to tombstone (a stored id or a previously-inserted id).
        #[arg(long)]
        id: u64,
    },
    /// Insert a vector into a bundle's update log (buffered, brute-scanned).
    Insert {
        /// Bundle directory produced by `slate build`.
        bundle: PathBuf,
        /// Plain-text vector file; only the first vector is used.
        vector: PathBuf,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Commands::Build {
            vectors,
            out,
            backend,
            metric,
            seed,
        } => run_build(&vectors, &out, &backend, &metric, seed),
        Commands::Query { bundle, query, k } => run_query(&bundle, &query, k),
        Commands::Bench {
            bundle,
            queries,
            k,
            profile,
            recall,
        } => run_bench(&bundle, &queries, k, &profile, recall),
        Commands::Delete { bundle, id } => run_delete(&bundle, id),
        Commands::Insert { bundle, vector } => run_insert(&bundle, &vector),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run_build(vectors: &Path, out: &Path, backend: &str, metric: &str, seed: u64) -> Result<()> {
    let (dims, rows) = read_vectors_text(vectors)?;
    let backend = parse_backend(backend)?;
    let metric = parse_metric(metric)?;

    let mut config = BuildConfig::new(dims, metric, backend);
    // The CLI builds a plain (no-PQ) index, so the product-quantiser subspace
    // count must not constrain the allowed dimensionality. Setting it to 1
    // makes `dims % num_subquantizers == 0` hold for any `dims`.
    config.pq.num_subquantizers = 1;

    slate_index::build_bundle(out, &config, &rows, seed)?;

    println!(
        "built {backend:?} bundle: {n} vectors x {dims} dims ({metric:?}) -> {path}",
        n = rows.len(),
        path = out.display(),
    );
    Ok(())
}

fn run_query(bundle: &Path, query: &Path, k: usize) -> Result<()> {
    let index = slate_index::open_bundle(bundle)?;
    let expected = index.config().dimensions;

    let (dims, rows) = read_vectors_text(query)?;
    let first = rows
        .first()
        .ok_or_else(|| Error::invalid_config(format!("query file {} is empty", query.display())))?;
    if dims != expected {
        return Err(Error::DimensionMismatch {
            expected,
            got: dims,
        });
    }

    let config = SearchConfig {
        k,
        ..SearchConfig::default()
    };
    let (neighbours, counters) = index.search(first, &config)?;

    for n in &neighbours {
        println!("{} {}", n.id.get(), n.score);
    }
    eprintln!(
        "visited {} nodes, {} exact / {} approx distances, {} seeks, {} bytes",
        counters.nodes_visited,
        counters.exact_distances,
        counters.approx_distances,
        counters.seeks,
        counters.bytes_read,
    );
    Ok(())
}

/// Run every vector in `queries` against the bundle, time the workload, and
/// price the accumulated counters through the cost model for `profile`.
fn run_bench(bundle: &Path, queries: &Path, k: usize, profile: &str, recall: bool) -> Result<()> {
    let index = slate_index::open_bundle(bundle)?;
    let expected = index.config().dimensions;
    let metric = index.config().metric;
    let profile_name = profile;
    let profile = parse_profile(profile)?;

    let (dims, rows) = read_vectors_text(queries)?;
    if dims != expected {
        return Err(Error::DimensionMismatch {
            expected,
            got: dims,
        });
    }
    if rows.is_empty() {
        return Err(Error::invalid_config(format!(
            "query file {} contains no vectors",
            queries.display()
        )));
    }

    let config = SearchConfig {
        k,
        ..SearchConfig::default()
    };

    let mut total = QueryCounters::new();
    let mut elapsed = std::time::Duration::ZERO;
    let mut total_recall = 0.0f64;

    for query in &rows {
        let start = std::time::Instant::now();
        let (neighbours, counters) = index.search(query, &config)?;
        elapsed += start.elapsed();
        total.merge(&counters);

        if recall {
            let truth = slate_index::brute_force_search(index.store(), query, metric, &config)?;
            let truth_ids: HashSet<u64> = truth.iter().map(|n| n.id.get()).collect();
            let hits = neighbours
                .iter()
                .filter(|n| truth_ids.contains(&n.id.get()))
                .count();
            let denom = config.k.max(1) as f64;
            total_recall += hits as f64 / denom;
        }
    }

    // Representative per-distance compute costs; calibrated values come from
    // offline SIMD micro-benchmarking per dim/dtype/ISA.
    let cost = DistanceCost::new(5e-9, 200e-9);
    let modeled = QueryCost::estimate(&total, profile, cost);

    let n = rows.len() as f64;
    let measured_ms = elapsed.as_secs_f64() / n * 1_000.0;

    println!(
        "bench: {q} queries, k={k}, profile={profile_name}",
        q = rows.len()
    );
    println!("  measured mean latency: {measured_ms:.4} ms/query");
    println!("  mean counters per query:");
    println!(
        "    nodes_visited:    {:.1}",
        total.nodes_visited as f64 / n
    );
    println!(
        "    approx_distances: {:.1}",
        total.approx_distances as f64 / n
    );
    println!(
        "    exact_distances:  {:.1}",
        total.exact_distances as f64 / n
    );
    println!("    seeks:            {:.1}", total.seeks as f64 / n);
    println!(
        "    sequential_runs:  {:.1}",
        total.sequential_runs as f64 / n
    );
    println!("    bytes_read:       {:.0}", total.bytes_read as f64 / n);
    println!("  modeled QueryCost per query (profile={profile_name}):");
    println!(
        "    traversal:        {:.4} ms",
        modeled.traversal_s / n * 1_000.0
    );
    println!(
        "    storage_access:   {:.4} ms",
        modeled.storage_access_s / n * 1_000.0
    );
    println!(
        "    distance:         {:.4} ms",
        modeled.distance_s / n * 1_000.0
    );
    println!(
        "    total:            {:.4} ms",
        modeled.total_s() / n * 1_000.0
    );
    println!("    storage_fraction: {:.3}", modeled.storage_fraction());
    if recall {
        println!("  mean recall@{k}: {:.4}", total_recall / n);
    }
    Ok(())
}

/// Tombstone a vector id in the bundle's update log and persist it.
fn run_delete(bundle: &Path, id: u64) -> Result<()> {
    let mut bundle_handle = slate_index::open_bundle(bundle)?;
    bundle_handle.delete(VectorId::new(id));
    bundle_handle.flush()?;
    println!("deleted id {id} from {path}", path = bundle.display());
    Ok(())
}

/// Append the first vector of `vector` to the bundle's insert buffer and persist.
fn run_insert(bundle: &Path, vector: &Path) -> Result<()> {
    let mut bundle_handle = slate_index::open_bundle(bundle)?;
    let (_dims, rows) = read_vectors_text(vector)?;
    let first = rows
        .first()
        .ok_or_else(|| Error::invalid_config(format!("{} contains no vector", vector.display())))?;
    let new_id = bundle_handle.insert(first.clone())?;
    bundle_handle.flush()?;
    println!("inserted id {id}", id = new_id.get());
    Ok(())
}

/// Parse a whitespace-separated plain-text vector file. The dimensionality is
/// inferred from the first non-blank line; blank lines are skipped and any row
/// whose width differs is reported as a [`Error::DimensionMismatch`].
fn read_vectors_text(path: &Path) -> Result<(usize, Vec<Vec<f32>>)> {
    let text = std::fs::read_to_string(path)?;
    let mut rows: Vec<Vec<f32>> = Vec::new();
    let mut dims: Option<usize> = None;

    for (lineno, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut row = Vec::new();
        for field in trimmed.split_whitespace() {
            let value: f32 = field.parse().map_err(|_| {
                Error::invalid_config(format!(
                    "{}:{}: not a float: {field:?}",
                    path.display(),
                    lineno + 1
                ))
            })?;
            row.push(value);
        }
        match dims {
            None => dims = Some(row.len()),
            Some(expected) if row.len() != expected => {
                return Err(Error::DimensionMismatch {
                    expected,
                    got: row.len(),
                });
            }
            Some(_) => {}
        }
        rows.push(row);
    }

    let dims = dims
        .ok_or_else(|| Error::invalid_config(format!("{} contains no vectors", path.display())))?;
    Ok((dims, rows))
}

fn parse_backend(s: &str) -> Result<IndexBackend> {
    match s.to_ascii_lowercase().as_str() {
        "hnsw" => Ok(IndexBackend::Hnsw),
        "ivf" => Ok(IndexBackend::Ivf),
        other => Err(Error::invalid_config(format!(
            "unknown backend {other:?} (expected `hnsw` or `ivf`)"
        ))),
    }
}

fn parse_metric(s: &str) -> Result<Metric> {
    match s.to_ascii_lowercase().as_str() {
        "l2" | "euclidean" => Ok(Metric::L2),
        "cosine" | "cos" => Ok(Metric::Cosine),
        "inner" | "ip" | "dot" => Ok(Metric::InnerProduct),
        other => Err(Error::invalid_config(format!(
            "unknown metric {other:?} (expected `l2`, `cosine`, or `inner`)"
        ))),
    }
}

fn parse_profile(s: &str) -> Result<StorageProfile> {
    match s.to_ascii_lowercase().as_str() {
        "hdd" | "hdd_7200rpm" => Ok(StorageProfile::hdd_7200rpm()),
        "ssd" | "nvme" => Ok(StorageProfile::ssd_nvme()),
        "memory" | "mem" | "ram" => Ok(StorageProfile::memory()),
        other => Err(Error::invalid_config(format!(
            "unknown storage profile {other:?} (expected `hdd`, `ssd`, or `memory`)"
        ))),
    }
}
