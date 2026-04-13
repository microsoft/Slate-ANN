//! # slate-pq
//!
//! Product quantization for Slate-ANN's approximate, RAM-resident distance
//! tier. Trains a per-subspace codebook, encodes every vector into a compact
//! PQ code, and computes asymmetric distances (ADC) via precomputed lookup
//! tables.
//!
//! This is the substrate for LEANN's two-level hybrid search: cheap
//! approximate distances (table lookups over RAM-resident codes) decide which
//! nodes to traverse and which exact vectors to fetch from disk, while the
//! exact streamed distances govern the final ranking.
//!
//! ## Pieces
//!
//! * [`PqCodebook`] — trained per-subspace centroids; encodes a vector to its
//!   `M`-byte code.
//! * [`AdcTable`] — a per-query lookup table giving the approximate distance to
//!   any code as a sum of `M` table entries.
//! * [`kmeans`] — the seeded Lloyd's k-means used to train each subspace.
//!
//! Populated in Phase 5.

#![doc(html_root_url = "https://docs.rs/slate-pq")]
#![deny(unsafe_op_in_unsafe_fn)]

pub mod adc;
pub mod codebook;
pub mod kmeans;
pub mod rng;

pub use adc::AdcTable;
pub use codebook::PqCodebook;
