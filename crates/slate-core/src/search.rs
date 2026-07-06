//! Search result types and the bounded top-k collector shared by every
//! Slate-ANN backend.
//!
//! These live in `slate-core` — the shared-vocabulary crate — so that both the
//! graph backends (`slate-graph`) and the top-level engine (`slate-index`) can
//! produce and rank results with one canonical [`Neighbor`] type and one
//! [`TopK`] collector, without any inter-crate dependency cycle.

use crate::VectorId;
use std::collections::BinaryHeap;

/// A single search result: a vector identity paired with its distance score.
///
/// Scores follow the engine-wide **ascending** ranking convention (smaller =
/// closer), matching [`crate::Metric`]. For `L2` the score is squared
/// Euclidean distance; for `InnerProduct` it is the negated dot product; for
/// `Cosine` it is `1 − cos`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neighbor {
    /// Identity of the matched vector.
    pub id: VectorId,
    /// Distance score (ascending: smaller is closer).
    pub score: f32,
}

impl Neighbor {
    /// Construct a neighbor from an id and score.
    #[inline]
    #[must_use]
    pub const fn new(id: VectorId, score: f32) -> Self {
        Self { id, score }
    }
}

/// Total ordering over `(score, id)` for ranking.
///
/// `f32` has no total `Ord` (NaN), so we order by [`f32::total_cmp`] on the
/// score and break ties by ascending `id`. This makes results **deterministic**
/// regardless of visitation order. Smaller scores order first (ascending = best
/// first); NaN scores — which would signal an upstream bug — sort to the end via
/// `total_cmp` and so are never preferred over real neighbors.
#[inline]
#[must_use]
pub fn cmp_ascending(a: &Neighbor, b: &Neighbor) -> core::cmp::Ordering {
    a.score.total_cmp(&b.score).then_with(|| a.id.cmp(&b.id))
}

/// Wrapper giving [`Neighbor`] a total `Ord` via [`cmp_ascending`].
///
/// A larger score (worse neighbor) compares **greater**, so a
/// [`BinaryHeap`] of these — a max-heap — keeps the current worst neighbor at
/// its top, ready for eviction.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Ranked(Neighbor);

impl Eq for Ranked {}

impl PartialOrd for Ranked {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Ranked {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        cmp_ascending(&self.0, &other.0)
    }
}

/// Collects the `k` best (smallest-score) neighbors from a stream of candidates
/// in O(N log k) time and O(k) memory.
///
/// Never materializes all candidate distances — suitable for a brute-force scan
/// over an arbitrarily large store, or for bounding a graph beam.
#[derive(Debug)]
pub struct TopK {
    k: usize,
    /// Max-heap: the top is the *worst* kept neighbor (largest score).
    heap: BinaryHeap<Ranked>,
}

impl TopK {
    /// Create a collector retaining up to `k` neighbors.
    ///
    /// `k == 0` yields a collector that keeps nothing.
    #[must_use]
    pub fn new(k: usize) -> Self {
        Self {
            k,
            heap: BinaryHeap::with_capacity(k),
        }
    }

    /// Offer a candidate. Kept only if it ranks among the best `k` seen so far.
    pub fn offer(&mut self, candidate: Neighbor) {
        if self.k == 0 {
            return;
        }
        if self.heap.len() < self.k {
            self.heap.push(Ranked(candidate));
            return;
        }
        // Heap is full: replace the current worst iff the candidate is strictly
        // better (smaller). `peek` is the max = worst kept neighbor.
        if let Some(worst) = self.heap.peek() {
            if cmp_ascending(&candidate, &worst.0) == core::cmp::Ordering::Less {
                self.heap.pop();
                self.heap.push(Ranked(candidate));
            }
        }
    }

    /// The current worst (largest) retained score, or `None` if empty.
    ///
    /// Useful as a beam-search cutoff: once `k` neighbors are held, any
    /// candidate not better than this can be skipped.
    #[must_use]
    pub fn worst_score(&self) -> Option<f32> {
        self.heap.peek().map(|r| r.0.score)
    }

    /// Whether the collector is full (holds exactly `k` neighbors).
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.heap.len() >= self.k
    }

    /// Number of neighbors currently retained.
    #[must_use]
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Whether no neighbors are retained.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Consume the collector, returning neighbors sorted ascending (best first).
    #[must_use]
    pub fn into_sorted_vec(self) -> Vec<Neighbor> {
        let mut out: Vec<Neighbor> = self.heap.into_iter().map(|r| r.0).collect();
        out.sort_unstable_by(cmp_ascending);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cmp::Ordering;

    fn n(id: u64, score: f32) -> Neighbor {
        Neighbor::new(VectorId::new(id), score)
    }

    #[test]
    fn orders_by_score_then_id() {
        let a = Neighbor::new(VectorId::new(5), 1.0);
        let b = Neighbor::new(VectorId::new(2), 2.0);
        assert_eq!(cmp_ascending(&a, &b), Ordering::Less);
    }

    #[test]
    fn breaks_ties_by_id() {
        let a = Neighbor::new(VectorId::new(2), 1.0);
        let b = Neighbor::new(VectorId::new(5), 1.0);
        assert_eq!(cmp_ascending(&a, &b), Ordering::Less);
        assert_eq!(cmp_ascending(&b, &a), Ordering::Greater);
    }

    #[test]
    fn nan_sorts_last() {
        let real = Neighbor::new(VectorId::new(1), 1.0);
        let nan = Neighbor::new(VectorId::new(0), f32::NAN);
        // total_cmp places NaN above any finite value, so the real neighbor is
        // "less" (ranked better).
        assert_eq!(cmp_ascending(&real, &nan), Ordering::Less);
    }

    #[test]
    fn keeps_k_smallest() {
        let mut t = TopK::new(3);
        for (i, s) in [5.0, 1.0, 4.0, 2.0, 3.0].iter().enumerate() {
            t.offer(n(i as u64, *s));
        }
        let got = t.into_sorted_vec();
        let scores: Vec<f32> = got.iter().map(|x| x.score).collect();
        assert_eq!(scores, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn k_larger_than_input_keeps_all_sorted() {
        let mut t = TopK::new(10);
        t.offer(n(0, 2.0));
        t.offer(n(1, 1.0));
        let got = t.into_sorted_vec();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].id, VectorId::new(1));
        assert_eq!(got[1].id, VectorId::new(0));
    }

    #[test]
    fn k_zero_keeps_nothing() {
        let mut t = TopK::new(0);
        t.offer(n(0, 1.0));
        assert!(t.is_empty());
        assert!(t.into_sorted_vec().is_empty());
    }

    #[test]
    fn deterministic_tie_break_by_id() {
        // All equal scores; only the k smallest ids should survive.
        let mut t = TopK::new(2);
        for id in [9, 3, 7, 1, 5] {
            t.offer(n(id, 1.0));
        }
        let got = t.into_sorted_vec();
        let ids: Vec<u64> = got.iter().map(|x| x.id.get()).collect();
        assert_eq!(ids, vec![1, 3]);
    }

    #[test]
    fn worst_score_and_fullness_track_the_beam() {
        let mut t = TopK::new(2);
        assert!(!t.is_full());
        assert_eq!(t.worst_score(), None);
        t.offer(n(0, 3.0));
        assert!(!t.is_full());
        assert_eq!(t.worst_score(), Some(3.0));
        t.offer(n(1, 1.0));
        assert!(t.is_full());
        assert_eq!(t.worst_score(), Some(3.0));
        // A better candidate evicts the worst and lowers the cutoff.
        t.offer(n(2, 2.0));
        assert!(t.is_full());
        assert_eq!(t.worst_score(), Some(2.0));
    }
}
