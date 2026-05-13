//! # slate-storage
//!
//! Disk-resident vector storage for Slate-ANN: a custom memory-mapping layer
//! over `memmap2`, the seek-minimizing block layout, dtype codecs
//! (f32/f16/i8), and the HDD elevator I/O scheduler.
//!
//! Only the graph topology, PQ codes, and the ID→offset map live in RAM;
//! exact vectors are streamed from this layer on demand.
//!
//! ## Layout
//!
//! A store file is a self-describing [`format::FileHeader`] followed by
//! fixed-size blocks of back-to-back vectors ([`layout::BlockLayout`]). A
//! vector never crosses a block boundary, so any fetch touches exactly one
//! block — one seek plus one sequential read on a spinning disk.
//!
//! ## Reading
//!
//! [`reader::VectorStore`] opens a file over an [`io::IoBackend`]:
//! [`io::MmapBackend`] for zero-copy SSD/warm-cache access, or
//! [`io::PreadBackend`] for explicit large positioned reads suited to HDDs.
//! Access-pattern [`mmap::Advice`] hints expose the `madvise` knob.
//!
//! The HDD elevator I/O scheduler ([`schedule::FetchSchedule`]) and the narrow
//! dtype codecs ([`codec`], f16/i8) land in Phase 7; narrow stores are decoded
//! to `f32` on the read path so the rest of the engine stays dtype-blind.

#![doc(html_root_url = "https://docs.rs/slate-storage")]
#![deny(unsafe_op_in_unsafe_fn)]

pub mod codec;
pub mod format;
pub mod io;
pub mod layout;
pub mod mmap;
pub mod reader;
pub mod schedule;

pub use format::{FileHeader, FORMAT_VERSION, HEADER_SIZE, MAGIC};
pub use io::{IoBackend, MmapBackend, PreadBackend};
pub use layout::{BlockLayout, StoreWriter};
pub use mmap::{Advice, MmapView};
pub use reader::VectorStore;
pub use schedule::FetchSchedule;
