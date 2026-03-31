//! # slate-core
//!
//! Foundational, dependency-light types shared across every Slate-ANN crate:
//! identifiers, distance metrics, on-disk dtypes, the central error type, and
//! the build/search/storage configuration structs.
//!
//! Keeping these here lets the SIMD, storage, PQ, graph, and index layers
//! depend on a common vocabulary without depending on each other.

#![forbid(unsafe_code)]
#![doc(html_root_url = "https://docs.rs/slate-core")]

pub mod config;
pub mod cost;
mod dtype;
mod error;
mod id;
mod metric;
pub mod search;

pub use config::{
    BuildConfig, HnswParams, IndexBackend, IoProfile, IvfParams, PqParams, SearchConfig,
    StorageParams,
};
pub use cost::{DistanceCost, QueryCost, QueryCounters, StorageProfile};
pub use dtype::Dtype;
pub use error::{Error, Result};
pub use id::VectorId;
pub use metric::Metric;
pub use search::{cmp_ascending, Neighbor, TopK};
