//! # slate-pq
//!
//! Product quantization for Slate-ANN's approximate, RAM-resident distance
//! tier. Trains a per-subspace codebook, encodes every vector into a compact
//! PQ code, and computes asymmetric distances (ADC) via precomputed lookup
//! tables.
//!
//! This is the substrate for LEANN's two-level hybrid search: cheap
//! approximate distances decide which nodes to traverse and which exact
//! vectors to fetch from disk.
//!
//! Populated in Phase 5.

#![doc(html_root_url = "https://docs.rs/slate-pq")]
