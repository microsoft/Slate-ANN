//! # slate-storage
//!
//! Disk-resident vector storage for Slate-ANN: a custom memory-mapping layer
//! over `memmap2`, the seek-minimizing block layout, dtype codecs
//! (f32/f16/i8), and the HDD elevator I/O scheduler.
//!
//! Only the graph topology, PQ codes, and the ID→offset map live in RAM;
//! exact vectors are streamed from this layer on demand.
//!
//! Populated in Phase 2 and Phase 7.

#![doc(html_root_url = "https://docs.rs/slate-storage")]
