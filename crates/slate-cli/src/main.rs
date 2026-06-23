//! `slate` — the command-line front end for the Slate-ANN disk-backed vector
//! search engine.
//!
//! Two subcommands are wired in this first Phase-10 cut:
//!
//! - `slate build <vectors> <out>` reads a plain-text vector file (one vector
//!   per line, whitespace-separated `f32`), writes a self-describing index
//!   *bundle* directory (vector store + index + resolved [`BuildConfig`]
//!   manifest), so the index can later be opened without re-stating any
//!   parameters.
//! - `slate query <bundle> <query> --k N` opens such a bundle and prints the
//!   `k` nearest neighbours of the first query vector as `id score` lines.
//!
//! The plain-text format is deliberately dependency-free; richer binary inputs
//! (`fvecs`/`npy`), a `bench` subcommand, and incremental updates are deferred
//! to later Phase-10 turns.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use slate_core::{BuildConfig, Error, IndexBackend, Metric, Result, SearchConfig};

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
    let first = rows.first().ok_or_else(|| {
        Error::invalid_config(format!("query file {} is empty", query.display()))
    })?;
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

    let dims = dims.ok_or_else(|| {
        Error::invalid_config(format!("{} contains no vectors", path.display()))
    })?;
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
