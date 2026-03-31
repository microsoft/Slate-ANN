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
//! ## Status
//!
//! Phase 4 populates the **HNSW** backend with a plain in-RAM proximity graph
//! that ranks with **exact** distances streamed from a
//! [`slate_storage::VectorStore`]. It accumulates a
//! [`slate_core::QueryCounters`] per search so the storage-aware cost model can
//! price the traversal. LEANN high-degree-preserving pruning (Phase 6), the PQ
//! approximate tier and two-level hybrid search (Phase 5), the IVF backend
//! (Phase 8), and on-disk graph persistence all land in later phases.

#![doc(html_root_url = "https://docs.rs/slate-graph")]
#![deny(unsafe_op_in_unsafe_fn)]

pub mod hnsw;
pub mod rng;

pub use hnsw::{HnswIndex, HnswStats};
