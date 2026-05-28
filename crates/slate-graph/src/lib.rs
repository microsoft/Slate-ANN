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
//! The **HNSW** backend (Phases 4–7) is a proximity graph with LEANN
//! high-degree-preserving pruning that ranks with **exact** distances streamed
//! from a [`slate_storage::VectorStore`], optionally gated by a PQ approximate
//! tier (two-level hybrid search), and fetches candidates in coalesced seek
//! order via the Phase-7 [`slate_storage::FetchSchedule`]. The **IVF** backend
//! (Phase 8) adds a k-means coarse quantizer with soft-assigned posting lists,
//! reusing the same storage-aware fetch path to stream a probe's candidates in
//! seek order. Both accumulate a [`slate_core::QueryCounters`] per search so the
//! storage-aware cost model can price the work. On-disk index persistence lands
//! in Phase 9.

#![doc(html_root_url = "https://docs.rs/slate-graph")]
#![deny(unsafe_op_in_unsafe_fn)]

pub mod hnsw;
pub mod ivf;
pub mod rng;

pub use hnsw::{HnswIndex, HnswStats};
pub use ivf::{IvfIndex, IvfStats};
