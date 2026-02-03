//! Distance metrics and the engine's ranking convention.

use serde::{Deserialize, Serialize};

/// Distance metric used to compare query and database vectors.
///
/// ## Ranking convention
///
/// Throughout Slate-ANN, search ranks candidates by an **ascending score**:
/// smaller means closer. Each metric defines how a raw similarity maps onto
/// that convention, so a single "smaller-is-better" priority queue works for
/// every metric:
///
/// | Metric         | Score computed by kernels        | Smaller = closer? |
/// |----------------|----------------------------------|-------------------|
/// | `L2`           | squared Euclidean distance       | yes (natural)     |
/// | `InnerProduct` | negated inner product (`-<a,b>`) | yes (negated)     |
/// | `Cosine`       | `1 - cosine_similarity`          | yes               |
///
/// `L2` uses the **squared** distance to avoid a per-comparison `sqrt`; the
/// ordering is identical to true Euclidean distance and the square root can be
/// applied once to final results if an actual distance is needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum Metric {
    /// Squared Euclidean (L2) distance.
    #[default]
    L2,
    /// Inner (dot) product similarity, scored as its negation.
    InnerProduct,
    /// Cosine distance (`1 - cosine_similarity`).
    Cosine,
}

impl Metric {
    /// Whether this metric requires inputs to be L2-normalized for correct
    /// results.
    ///
    /// `Cosine` is implemented as inner product over unit-normalized vectors,
    /// so the engine normalizes both database and query vectors when this is
    /// `true`.
    #[inline]
    pub const fn requires_normalized_input(self) -> bool {
        matches!(self, Metric::Cosine)
    }

    /// Lower-case identifier used in the on-disk metadata file.
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Metric::L2 => "l2",
            Metric::InnerProduct => "inner_product",
            Metric::Cosine => "cosine",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalization_requirement() {
        assert!(Metric::Cosine.requires_normalized_input());
        assert!(!Metric::L2.requires_normalized_input());
        assert!(!Metric::InnerProduct.requires_normalized_input());
    }

    #[test]
    fn stable_string_tags() {
        assert_eq!(Metric::L2.as_str(), "l2");
        assert_eq!(Metric::InnerProduct.as_str(), "inner_product");
        assert_eq!(Metric::Cosine.as_str(), "cosine");
    }
}
