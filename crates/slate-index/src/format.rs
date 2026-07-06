//! On-disk index format: persist and reload a built ANN index.
//!
//! A built [`HnswIndex`] or [`IvfIndex`] is pure RAM. This module gives each a
//! durable on-disk form so the expensive build (the `O(N·ef·M)` graph
//! construction) can happen once on a capable machine and the resulting index
//! can be shipped to and reopened on a low-power serving device with zero
//! rebuild. The vectors themselves already persist in their own
//! [`slate_storage::VectorStore`] file; this format persists the *structure*
//! (graph adjacency / posting lists and the little metadata search needs).
//!
//! # Frame
//!
//! Every index file begins with a fixed header so a loader validates it before
//! trusting it, mirroring [`slate_storage`]'s `FileHeader`:
//!
//! ```text
//! offset  size  field
//! 0       8     MAGIC = b"SLATEIDX"
//! 8       4     FORMAT_VERSION (u32, little-endian)
//! 12      1     backend tag (1 = HNSW, 2 = IVF)
//! 13      3     reserved (zero)
//! 16      8     payload length (u64, little-endian)
//! 24      …     payload (serde_json of the index struct)
//! ```
//!
//! The payload is JSON for simplicity and because `serde_json` is already in the
//! dependency tree (it is what the PQ codebook persists with). Swapping the
//! payload encoder later does not change the frame.
//!
//! # Validation
//!
//! [`load_hnsw`] / [`load_ivf`] reject a file whose magic, version, or backend
//! tag does not match with [`Error::Corrupt`]. A backend mismatch (loading an
//! IVF file as HNSW or vice-versa) is reported distinctly so the caller can tell
//! "wrong file" from "garbage file".

use std::io::{Read, Write};
use std::path::Path;

use slate_core::{Error, Result};
use slate_graph::{HnswIndex, IvfIndex};

/// Magic bytes at the start of every Slate-ANN index file. Distinct from the
/// vector store's `SLATEVEC` so the two file kinds can never be confused.
pub const MAGIC: [u8; 8] = *b"SLATEIDX";

/// On-disk index format version. Bump on any incompatible frame/payload change.
pub const FORMAT_VERSION: u32 = 1;

/// Size of the fixed header preceding the JSON payload.
pub const HEADER_LEN: usize = 24;

/// Which backend a serialized index holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BackendTag {
    Hnsw,
    Ivf,
}

impl BackendTag {
    /// The on-disk byte for this backend.
    const fn to_byte(self) -> u8 {
        match self {
            BackendTag::Hnsw => 1,
            BackendTag::Ivf => 2,
        }
    }

    /// Parse an on-disk backend byte.
    fn from_byte(b: u8) -> Result<Self> {
        match b {
            1 => Ok(BackendTag::Hnsw),
            2 => Ok(BackendTag::Ivf),
            other => Err(Error::corrupt(format!(
                "unknown index backend tag {other} (expected 1 = HNSW or 2 = IVF)"
            ))),
        }
    }

    /// Human-readable name for error messages.
    const fn name(self) -> &'static str {
        match self {
            BackendTag::Hnsw => "HNSW",
            BackendTag::Ivf => "IVF",
        }
    }
}

/// Serialize `payload_json` with the given backend tag into `writer`.
fn write_framed<W: Write>(mut writer: W, tag: BackendTag, payload: &[u8]) -> Result<()> {
    let mut header = [0u8; HEADER_LEN];
    header[0..8].copy_from_slice(&MAGIC);
    header[8..12].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
    header[12] = tag.to_byte();
    // header[13..16] reserved, already zero.
    header[16..24].copy_from_slice(&(payload.len() as u64).to_le_bytes());
    writer.write_all(&header)?;
    writer.write_all(payload)?;
    Ok(())
}

/// Validate the frame header in `bytes`, returning the backend tag and the
/// payload slice. Rejects bad magic/version/length with [`Error::Corrupt`].
fn parse_frame(bytes: &[u8]) -> Result<(BackendTag, &[u8])> {
    if bytes.len() < HEADER_LEN {
        return Err(Error::corrupt(format!(
            "index file too short: {} bytes < {HEADER_LEN}-byte header",
            bytes.len()
        )));
    }
    if bytes[0..8] != MAGIC {
        return Err(Error::corrupt(
            "bad index magic (not a Slate-ANN index file)",
        ));
    }
    let version = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    if version != FORMAT_VERSION {
        return Err(Error::corrupt(format!(
            "unsupported index format version {version} (expected {FORMAT_VERSION})"
        )));
    }
    let tag = BackendTag::from_byte(bytes[12])?;
    let payload_len = u64::from_le_bytes([
        bytes[16], bytes[17], bytes[18], bytes[19], bytes[20], bytes[21], bytes[22], bytes[23],
    ]) as usize;
    let payload = &bytes[HEADER_LEN..];
    if payload.len() != payload_len {
        return Err(Error::corrupt(format!(
            "index payload length mismatch: header says {payload_len}, file has {}",
            payload.len()
        )));
    }
    Ok((tag, payload))
}

/// Decode a framed index of the expected backend from `bytes`.
fn decode<T: serde::de::DeserializeOwned>(bytes: &[u8], expected: BackendTag) -> Result<T> {
    let (tag, payload) = parse_frame(bytes)?;
    if tag != expected {
        return Err(Error::corrupt(format!(
            "index backend mismatch: file holds {}, expected {}",
            tag.name(),
            expected.name()
        )));
    }
    serde_json::from_slice(payload)
        .map_err(|e| Error::corrupt(format!("malformed {} index payload: {e}", expected.name())))
}

/// Serialize an [`HnswIndex`] frame into `writer`.
pub fn write_hnsw<W: Write>(writer: W, index: &HnswIndex) -> Result<()> {
    let payload = serde_json::to_vec(index)
        .map_err(|e| Error::corrupt(format!("failed to serialize HNSW index: {e}")))?;
    write_framed(writer, BackendTag::Hnsw, &payload)
}

/// Deserialize an [`HnswIndex`] from a framed byte slice, validating the header.
pub fn read_hnsw(bytes: &[u8]) -> Result<HnswIndex> {
    decode(bytes, BackendTag::Hnsw)
}

/// Serialize an [`IvfIndex`] frame into `writer`.
pub fn write_ivf<W: Write>(writer: W, index: &IvfIndex) -> Result<()> {
    let payload = serde_json::to_vec(index)
        .map_err(|e| Error::corrupt(format!("failed to serialize IVF index: {e}")))?;
    write_framed(writer, BackendTag::Ivf, &payload)
}

/// Deserialize an [`IvfIndex`] from a framed byte slice, validating the header.
pub fn read_ivf(bytes: &[u8]) -> Result<IvfIndex> {
    decode(bytes, BackendTag::Ivf)
}

/// Save an [`HnswIndex`] to `path`, overwriting any existing file.
pub fn save_hnsw(path: impl AsRef<Path>, index: &HnswIndex) -> Result<()> {
    let file = std::fs::File::create(path)?;
    write_hnsw(std::io::BufWriter::new(file), index)
}

/// Load an [`HnswIndex`] previously written by [`save_hnsw`].
pub fn load_hnsw(path: impl AsRef<Path>) -> Result<HnswIndex> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut bytes)?;
    read_hnsw(&bytes)
}

/// Save an [`IvfIndex`] to `path`, overwriting any existing file.
pub fn save_ivf(path: impl AsRef<Path>, index: &IvfIndex) -> Result<()> {
    let file = std::fs::File::create(path)?;
    write_ivf(std::io::BufWriter::new(file), index)
}

/// Load an [`IvfIndex`] previously written by [`save_ivf`].
pub fn load_ivf(path: impl AsRef<Path>) -> Result<IvfIndex> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut bytes)?;
    read_ivf(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use slate_core::{Dtype, HnswParams, IvfParams, Metric, SearchConfig, StorageParams};
    use slate_graph::{HnswStats, IvfStats};
    use slate_storage::{BlockLayout, MmapBackend, StoreWriter, VectorStore};
    use tempfile::NamedTempFile;

    /// Deterministic pseudo-random vectors (small dependency-free LCG-ish hash
    /// so the test needs no rng crate).
    fn gen_vectors(seed: u64, n: usize, dims: usize) -> Vec<Vec<f32>> {
        let mut state = seed | 1;
        let mut next = || {
            // SplitMix64-style scramble, mapped to [0, 1).
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..n)
            .map(|_| (0..dims).map(|_| next() as f32).collect())
            .collect()
    }

    fn build_store(vectors: &[Vec<f32>], dims: usize) -> (NamedTempFile, VectorStore<MmapBackend>) {
        let tmp = NamedTempFile::new().unwrap();
        let block_size = StorageParams::default().block_size;
        let layout = BlockLayout::new(Dtype::F32, dims, block_size).unwrap();
        let mut writer = StoreWriter::create(tmp.path(), layout).unwrap();
        for v in vectors {
            writer.push(v).unwrap();
        }
        writer.finish().unwrap();
        let store = VectorStore::open_mmap(tmp.path()).unwrap();
        (tmp, store)
    }

    fn hnsw_params() -> HnswParams {
        HnswParams {
            m: 8,
            m_max: 16,
            ef_construction: 100,
            ..HnswParams::default()
        }
    }

    fn cfg(k: usize) -> SearchConfig {
        SearchConfig {
            k,
            ..SearchConfig::default()
        }
    }

    /// A loaded HNSW index must return byte-identical results (ids + counters)
    /// to the in-RAM original across many queries.
    #[test]
    fn hnsw_round_trip_preserves_search() {
        let dims = 12;
        let vectors = gen_vectors(2024, 300, dims);
        let (_tmp, store) = build_store(&vectors, dims);
        let original = HnswIndex::build(&store, Metric::L2, &hnsw_params(), 777).unwrap();

        // Round-trip through an in-memory buffer (no temp file needed).
        let mut buf = Vec::new();
        write_hnsw(&mut buf, &original).unwrap();
        let loaded = read_hnsw(&buf).unwrap();

        let queries = gen_vectors(99, 40, dims);
        for q in &queries {
            let (orig_res, orig_stats): (_, HnswStats) =
                original.search(&store, q, &cfg(10)).unwrap();
            let (load_res, load_stats): (_, HnswStats) =
                loaded.search(&store, q, &cfg(10)).unwrap();
            let orig_ids: Vec<_> = orig_res.iter().map(|n| n.id).collect();
            let load_ids: Vec<_> = load_res.iter().map(|n| n.id).collect();
            assert_eq!(orig_ids, load_ids, "loaded HNSW search ids differ");
            assert_eq!(orig_stats, load_stats, "loaded HNSW counters differ");
        }
    }

    /// A loaded IVF index must return byte-identical results to the in-RAM
    /// original across many queries.
    #[test]
    fn ivf_round_trip_preserves_search() {
        let dims = 12;
        let vectors = gen_vectors(4242, 400, dims);
        let (_tmp, store) = build_store(&vectors, dims);
        let params = IvfParams {
            num_lists: 16,
            num_probes: 8,
            soft_assign: 2,
            ..IvfParams::default()
        };
        let original = IvfIndex::build(&store, Metric::L2, &params, 555).unwrap();

        let mut buf = Vec::new();
        write_ivf(&mut buf, &original).unwrap();
        let loaded = read_ivf(&buf).unwrap();

        let queries = gen_vectors(7, 40, dims);
        for q in &queries {
            let (orig_res, orig_stats): (_, IvfStats) =
                original.search(&store, q, &cfg(10)).unwrap();
            let (load_res, load_stats): (_, IvfStats) = loaded.search(&store, q, &cfg(10)).unwrap();
            let orig_ids: Vec<_> = orig_res.iter().map(|n| n.id).collect();
            let load_ids: Vec<_> = load_res.iter().map(|n| n.id).collect();
            assert_eq!(orig_ids, load_ids, "loaded IVF search ids differ");
            assert_eq!(orig_stats, load_stats, "loaded IVF counters differ");
        }
    }

    /// The path-based save/load entry points round-trip through a real file.
    #[test]
    fn save_and_load_via_file_round_trip() {
        let dims = 8;
        let vectors = gen_vectors(11, 120, dims);
        let (_tmp, store) = build_store(&vectors, dims);
        let original = HnswIndex::build(&store, Metric::L2, &hnsw_params(), 1).unwrap();

        let idx_file = NamedTempFile::new().unwrap();
        save_hnsw(idx_file.path(), &original).unwrap();
        let loaded = load_hnsw(idx_file.path()).unwrap();

        let q = &gen_vectors(123, 1, dims)[0];
        let orig_ids: Vec<_> = original
            .search(&store, q, &cfg(5))
            .unwrap()
            .0
            .iter()
            .map(|n| n.id)
            .collect();
        let load_ids: Vec<_> = loaded
            .search(&store, q, &cfg(5))
            .unwrap()
            .0
            .iter()
            .map(|n| n.id)
            .collect();
        assert_eq!(orig_ids, load_ids);
    }

    /// An empty index (no vectors) round-trips and searches to empty.
    #[test]
    fn empty_index_round_trips() {
        let dims = 4;
        let (_tmp, store) = build_store(&[], dims);
        let original = HnswIndex::build(&store, Metric::L2, &hnsw_params(), 0).unwrap();
        assert!(original.is_empty());

        let mut buf = Vec::new();
        write_hnsw(&mut buf, &original).unwrap();
        let loaded = read_hnsw(&buf).unwrap();
        assert!(loaded.is_empty());

        let (res, _stats) = loaded.search(&store, &[0.0; 4], &cfg(5)).unwrap();
        assert!(res.is_empty());
    }

    /// A valid HNSW frame is rejected when read as IVF, and vice-versa.
    #[test]
    fn backend_mismatch_is_rejected() {
        let dims = 6;
        let vectors = gen_vectors(5, 60, dims);
        let (_tmp, store) = build_store(&vectors, dims);
        let hnsw = HnswIndex::build(&store, Metric::L2, &hnsw_params(), 3).unwrap();

        let mut buf = Vec::new();
        write_hnsw(&mut buf, &hnsw).unwrap();
        let err = read_ivf(&buf).unwrap_err();
        assert!(
            matches!(err, Error::Corrupt(_)),
            "expected corrupt, got {err:?}"
        );
        assert!(err.to_string().contains("backend mismatch"));
    }

    /// Bad magic, wrong version, and a truncated payload each fail to load.
    #[test]
    fn corrupt_frames_are_rejected() {
        let dims = 6;
        let vectors = gen_vectors(8, 60, dims);
        let (_tmp, store) = build_store(&vectors, dims);
        let hnsw = HnswIndex::build(&store, Metric::L2, &hnsw_params(), 4).unwrap();
        let mut good = Vec::new();
        write_hnsw(&mut good, &hnsw).unwrap();

        // Too short to even hold a header.
        assert!(matches!(read_hnsw(&good[..10]), Err(Error::Corrupt(_))));

        // Bad magic.
        let mut bad_magic = good.clone();
        bad_magic[0] = b'X';
        assert!(matches!(read_hnsw(&bad_magic), Err(Error::Corrupt(_))));

        // Wrong version.
        let mut bad_version = good.clone();
        bad_version[8] = 0xFF;
        let err = read_hnsw(&bad_version).unwrap_err();
        assert!(err.to_string().contains("version"));

        // Truncated payload (header intact, body cut).
        let truncated = &good[..good.len() - 5];
        assert!(matches!(read_hnsw(truncated), Err(Error::Corrupt(_))));

        // Unknown backend tag.
        let mut bad_tag = good.clone();
        bad_tag[12] = 9;
        assert!(matches!(read_hnsw(&bad_tag), Err(Error::Corrupt(_))));
    }

    /// The frame header is exactly as documented: magic, version, tag, length.
    #[test]
    fn frame_header_layout_is_stable() {
        let dims = 4;
        let vectors = gen_vectors(1, 20, dims);
        let (_tmp, store) = build_store(&vectors, dims);
        let hnsw = HnswIndex::build(&store, Metric::L2, &hnsw_params(), 2).unwrap();
        let mut buf = Vec::new();
        write_hnsw(&mut buf, &hnsw).unwrap();

        assert_eq!(&buf[0..8], &MAGIC);
        assert_eq!(
            u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]),
            FORMAT_VERSION
        );
        assert_eq!(buf[12], 1); // HNSW tag
        let payload_len = u64::from_le_bytes([
            buf[16], buf[17], buf[18], buf[19], buf[20], buf[21], buf[22], buf[23],
        ]) as usize;
        assert_eq!(payload_len, buf.len() - HEADER_LEN);
    }
}
