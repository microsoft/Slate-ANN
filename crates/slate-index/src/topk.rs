//! Bounded top-k collector.

use crate::neighbor::{cmp_ascending, Neighbor};
use std::collections::BinaryHeap;

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
/// over an arbitrarily large store.
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
    use slate_core::VectorId;

    fn n(id: u64, score: f32) -> Neighbor {
        Neighbor::new(VectorId::new(id), score)
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
}
