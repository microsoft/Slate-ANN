//! Search result types shared by every Slate-ANN backend.

use slate_core::VectorId;

/// A single search result: a vector identity paired with its distance score.
///
/// Scores follow the engine-wide **ascending** ranking convention (smaller =
/// closer), matching [`slate_core::Metric`]. For `L2` the score is squared
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
    a.score
        .total_cmp(&b.score)
        .then_with(|| a.id.cmp(&b.id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cmp::Ordering;

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
}
