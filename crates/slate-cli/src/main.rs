//! # slate (CLI)
//!
//! Command-line build/query/bench tool for the Slate-ANN vector search engine.
//! Fleshed out alongside the engine; for now it reports the build identity so
//! the workspace has a runnable binary from Phase 0.

fn main() {
    println!(
        "slate-ann {} — disk-backed vector search engine (scaffold)",
        env!("CARGO_PKG_VERSION")
    );
}
