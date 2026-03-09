//! # slate-index
//!
//! Top-level Slate-ANN engine: the storage-efficient sharded build pipeline,
//! the multi-file on-disk index format, soft-delete / buffered-insert updates,
//! and the public query API.
//!
//! Phase 3 adds [`brute_force_search`] — exact KNN that streams every vector
//! from a [`slate_storage::VectorStore`]. It is both the first end-to-end query
//! path and the recall oracle for the approximate backends added later.
//!
//! Populated in Phases 3, 9, and 10.

#![doc(html_root_url = "https://docs.rs/slate-index")]
#![forbid(unsafe_code)]

pub mod brute;
pub mod neighbor;
pub mod topk;

pub use brute::brute_force_search;
pub use neighbor::Neighbor;
pub use topk::TopK;
