//! Brute-force exact k-nearest-neighbor search.
//!
//! Streams every vector from a [`VectorStore`] and ranks it with the
//! [`slate_simd`] distance kernels. This is the simplest correct search in the
//! engine — no graph, no approximation — and doubles as the **recall oracle**
//! the approximate backends (HNSW, IVF) are measured against.
//!
//! Complexity is O(N · d) distance work and O(k) memory: vectors are decoded
//! one at a time into a reusable scratch buffer, never all held at once.

use slate_core::{Error, Metric, Result, SearchConfig, VectorId};
use slate_storage::{IoBackend, VectorStore};

use crate::neighbor::Neighbor;
use crate::topk::TopK;

/// Brute-force exact KNN over an entire [`VectorStore`].
///
/// Scans all `store.len()` vectors, computes `metric` distance to `query`, and
/// returns the best `config.k` as [`Neighbor`]s sorted ascending (closest
/// first). Fewer than `k` are returned when the store holds fewer vectors.
///
/// `query.len()` must equal `store.dimensions()`. The store must hold `f32`
/// vectors (the only dtype materialized so far); other dtypes yield
/// [`Error::Unsupported`].
///
/// `Cosine` uses the raw (per-call normalizing) path via
/// [`slate_simd::distance`]; when cosine builds later pre-normalize vectors,
/// callers can switch to the cheaper inner-product path.
///
/// # Errors
/// - [`Error::DimensionMismatch`] if `query.len() != store.dimensions()`.
/// - [`Error::Unsupported`] if the store dtype is not `f32`.
/// - Propagates [`Error::Io`] / [`Error::Corrupt`] from vector reads.
pub fn brute_force_search<B: IoBackend>(
    store: &VectorStore<B>,
    query: &[f32],
    metric: Metric,
    config: &SearchConfig,
) -> Result<Vec<Neighbor>> {
    let dims = store.dimensions();
    if query.len() != dims {
        return Err(Error::DimensionMismatch {
            expected: dims,
            got: query.len(),
        });
    }
    if store.dtype() != slate_core::Dtype::F32 {
        return Err(Error::unsupported(format!(
            "brute-force search requires f32 store, found {}",
            store.dtype().as_str()
        )));
    }

    let mut topk = TopK::new(config.k);
    let mut scratch = vec![0.0f32; dims];

    for index in 0..store.len() {
        store.get_into(index, &mut scratch)?;
        let score = slate_simd::distance(metric, query, &scratch)?;
        topk.offer(Neighbor::new(VectorId::new(index as u64), score));
    }

    Ok(topk.into_sorted_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use slate_core::{Dtype, StorageParams};
    use slate_storage::{BlockLayout, StoreWriter};
    use tempfile::NamedTempFile;

    /// Write `vectors` to a temp store and open it via the mmap backend.
    fn build_store(vectors: &[Vec<f32>], dims: usize) -> (NamedTempFile, VectorStore<slate_storage::MmapBackend>) {
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

    fn cfg(k: usize) -> SearchConfig {
        SearchConfig {
            k,
            ..SearchConfig::default()
        }
    }

    #[test]
    fn l2_finds_nearest() {
        let vectors = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![5.0, 5.0],
            vec![0.5, 0.5],
        ];
        let (_tmp, store) = build_store(&vectors, 2);
        let got = brute_force_search(&store, &[0.0, 0.0], Metric::L2, &cfg(2)).unwrap();
        assert_eq!(got.len(), 2);
        // Closest is index 0 (itself, dist 0), then index 3 (0.5).
        assert_eq!(got[0].id, VectorId::new(0));
        assert!((got[0].score - 0.0).abs() < 1e-6);
        assert_eq!(got[1].id, VectorId::new(3));
        assert!((got[1].score - 0.5).abs() < 1e-6);
    }

    #[test]
    fn k_exceeds_count_returns_all() {
        let vectors = vec![vec![1.0, 1.0], vec![2.0, 2.0]];
        let (_tmp, store) = build_store(&vectors, 2);
        let got = brute_force_search(&store, &[0.0, 0.0], Metric::L2, &cfg(10)).unwrap();
        assert_eq!(got.len(), 2);
    }

    #[test]
    fn inner_product_ranks_by_negated_dot() {
        let vectors = vec![
            vec![1.0, 0.0],
            vec![10.0, 0.0],
            vec![0.0, 1.0],
        ];
        let (_tmp, store) = build_store(&vectors, 2);
        let got = brute_force_search(&store, &[1.0, 0.0], Metric::InnerProduct, &cfg(1)).unwrap();
        // Largest dot (10) => smallest negated score => best.
        assert_eq!(got[0].id, VectorId::new(1));
        assert!((got[0].score + 10.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_ignores_magnitude() {
        let vectors = vec![
            vec![1.0, 0.0],   // same direction as query
            vec![100.0, 0.0], // same direction, bigger magnitude
            vec![0.0, 1.0],   // orthogonal
        ];
        let (_tmp, store) = build_store(&vectors, 2);
        let got = brute_force_search(&store, &[2.0, 0.0], Metric::Cosine, &cfg(2)).unwrap();
        // Both index 0 and 1 are colinear => cosine distance ~0; orthogonal last.
        assert!((got[0].score).abs() < 1e-6);
        assert!((got[1].score).abs() < 1e-6);
        let top_ids = [got[0].id, got[1].id];
        assert!(top_ids.contains(&VectorId::new(0)));
        assert!(top_ids.contains(&VectorId::new(1)));
    }

    #[test]
    fn empty_store_returns_empty() {
        let (_tmp, store) = build_store(&[], 3);
        let got = brute_force_search(&store, &[0.0, 0.0, 0.0], Metric::L2, &cfg(5)).unwrap();
        assert!(got.is_empty());
    }

    #[test]
    fn dimension_mismatch_is_reported() {
        let vectors = vec![vec![1.0, 2.0, 3.0]];
        let (_tmp, store) = build_store(&vectors, 3);
        let err = brute_force_search(&store, &[1.0, 2.0], Metric::L2, &cfg(1)).unwrap_err();
        assert!(matches!(
            err,
            Error::DimensionMismatch { expected: 3, got: 2 }
        ));
    }
}

#[cfg(test)]
mod proptests {
    //! Validate brute force against an independent naive in-memory reference.
    //!
    //! The disk-streaming `brute_force_search` must agree exactly with a
    //! straightforward "compute every distance, sort" implementation — pinning
    //! the recall oracle to a second, obviously-correct algorithm.

    use super::*;
    use crate::neighbor::cmp_ascending;
    use proptest::prelude::*;
    use slate_core::{Dtype, StorageParams};
    use slate_storage::{BlockLayout, StoreWriter, VectorStore};
    use tempfile::NamedTempFile;

    /// Independent reference: compute all distances in memory and sort with the
    /// same total order brute force uses.
    fn naive_reference(
        vectors: &[Vec<f32>],
        query: &[f32],
        metric: Metric,
        k: usize,
    ) -> Vec<Neighbor> {
        let mut all: Vec<Neighbor> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let score = slate_simd::distance(metric, query, v).unwrap();
                Neighbor::new(VectorId::new(i as u64), score)
            })
            .collect();
        all.sort_unstable_by(cmp_ascending);
        all.truncate(k);
        all
    }

    fn write_and_open(
        vectors: &[Vec<f32>],
        dims: usize,
    ) -> (NamedTempFile, VectorStore<slate_storage::MmapBackend>) {
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

    prop_compose! {
        /// A random dataset: `count` vectors of `dims` dimensions plus a query.
        fn dataset()(
            dims in 1usize..=32,
            count in 0usize..=200,
        )(
            query in prop::collection::vec(-10.0f32..10.0, dims),
            vectors in prop::collection::vec(
                prop::collection::vec(-10.0f32..10.0, dims),
                count,
            ),
            dims in Just(dims),
        ) -> (usize, Vec<f32>, Vec<Vec<f32>>) {
            (dims, query, vectors)
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(150))]

        #[test]
        fn matches_naive_reference(
            (dims, query, vectors) in dataset(),
            k in 1usize..=64,
            metric_idx in 0usize..3,
        ) {
            let metric = [Metric::L2, Metric::InnerProduct, Metric::Cosine][metric_idx];
            let (_tmp, store) = write_and_open(&vectors, dims);

            let got = brute_force_search(&store, &query, metric, &SearchConfig { k, ..SearchConfig::default() }).unwrap();
            let want = naive_reference(&vectors, &query, metric, k);

            prop_assert_eq!(got.len(), want.len());
            for (g, w) in got.iter().zip(want.iter()) {
                prop_assert_eq!(g.id, w.id);
                // Same kernels on both sides => bit-identical scores expected,
                // but allow a tiny epsilon for safety.
                prop_assert!((g.score - w.score).abs() <= 1e-6 * g.score.abs().max(1.0));
            }
        }
    }
}
