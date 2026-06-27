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
//! The shared result vocabulary ([`Neighbor`], [`TopK`], `cmp_ascending`) now
//! lives in `slate-core` so the graph backends can use it without a dependency
//! cycle; it is re-exported here for convenience and backward compatibility.
//!
//! Populated in Phases 3, 9, and 10.

#![doc(html_root_url = "https://docs.rs/slate-index")]
#![forbid(unsafe_code)]

pub mod brute;
pub mod bundle;
pub mod format;
pub mod update;

pub use brute::brute_force_search;
pub use bundle::{build_bundle, open_bundle, Bundle, BundleIndex, BundleManifest};
pub use slate_core::{cmp_ascending, Neighbor, TopK};
pub use update::UpdateLog;
