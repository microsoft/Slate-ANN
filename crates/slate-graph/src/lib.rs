//! # slate-graph
//!
//! Pluggable approximate-nearest-neighbor index backends for Slate-ANN behind
//! a common `VectorIndex` trait:
//!
//! - **HNSW** with LEANN-style high-degree-preserving pruning.
//! - **IVF** (k-means) with soft assignment for connectivity.
//!
//! Both share the two-level hybrid search: approximate PQ distances steer
//! which exact vectors to batch-fetch from disk, exact distances steer
//! traversal.
//!
//! Populated in Phases 4, 5, 6, and 8.

#![doc(html_root_url = "https://docs.rs/slate-graph")]
