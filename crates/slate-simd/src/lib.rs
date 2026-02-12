//! # slate-simd
//!
//! SIMD distance kernels for Slate-ANN with runtime CPU-feature dispatch.
//!
//! Provides L2², inner-product, and cosine distance over `f32` (with `f16` /
//! `i8` to follow in later phases). Four implementation tiers are selected at
//! runtime — AVX-512, AVX2+FMA, ARM NEON, and a portable scalar fallback (also
//! the correctness oracle for the vectorized paths).
//!
//! ## Ranking convention
//! Mirrors [`slate_core::Metric`]: all distances rank by **ascending** score
//! (smaller = closer).
//! - [`l2_sq`] — squared Euclidean (no `sqrt`; preserves ordering).
//! - [`inner_product`] — **negated** dot product (`−⟨a,b⟩`).
//! - [`cosine`] — `1 − cos(a,b)` over raw inputs;
//!   [`cosine_normalized`] is the cheaper `1 − ⟨a,b⟩` for pre-normalized inputs.
//!
//! ## Safety model
//! Vectorized kernels use `#[target_feature]` intrinsics (`unsafe`). They are
//! only ever invoked behind the runtime dispatcher in [`dispatch`], which
//! confirms CPU support before selecting a tier. The **public API below is
//! entirely safe** and validates that input slices have equal length.
//!
//! Populated in Phase 1 (f32 kernels + dispatch).

#![doc(html_root_url = "https://docs.rs/slate-simd")]
// The vectorized tiers require raw-intrinsic `unsafe`. We forbid *implicit*
// unsafe (every intrinsic call must sit in an explicit `unsafe` block with a
// safety comment) rather than forbidding unsafe entirely.
#![deny(unsafe_op_in_unsafe_fn)]
#![cfg_attr(feature = "portable_simd", feature(portable_simd))]

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;
mod dispatch;
#[cfg(target_arch = "aarch64")]
mod neon;
pub mod scalar;

pub use dispatch::{active_tier, detect_tier, Tier};
use slate_core::{Error, Result};

/// Validate that two operand slices have identical, non-zero length.
#[inline]
fn check(a: &[f32], b: &[f32]) -> Result<()> {
    if a.len() != b.len() {
        return Err(Error::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    Ok(())
}

/// Squared Euclidean (L2²) distance, dispatched to the best available tier.
///
/// # Errors
/// Returns [`Error::DimensionMismatch`] if `a.len() != b.len()`.
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> Result<f32> {
    check(a, b)?;
    Ok(dispatch::l2_sq_kernel()(a, b))
}

/// Inner-product distance `−⟨a,b⟩`, dispatched to the best available tier.
///
/// # Errors
/// Returns [`Error::DimensionMismatch`] if `a.len() != b.len()`.
#[inline]
pub fn inner_product(a: &[f32], b: &[f32]) -> Result<f32> {
    check(a, b)?;
    Ok(-dispatch::dot_kernel()(a, b))
}

/// Raw inner product `⟨a,b⟩` (not negated), dispatched to the best tier.
///
/// Exposed for callers that need the similarity directly (e.g. PQ table
/// construction). Most search code wants [`inner_product`].
///
/// # Errors
/// Returns [`Error::DimensionMismatch`] if `a.len() != b.len()`.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> Result<f32> {
    check(a, b)?;
    Ok(dispatch::dot_kernel()(a, b))
}

/// Cosine distance `1 − cos(a,b)` over raw (un-normalized) inputs.
///
/// Zero-norm operands yield distance `1.0`.
///
/// # Errors
/// Returns [`Error::DimensionMismatch`] if `a.len() != b.len()`.
#[inline]
pub fn cosine(a: &[f32], b: &[f32]) -> Result<f32> {
    check(a, b)?;
    let (d, na, nb) = dispatch::cosine_parts(a, b);
    let denom = (na * nb).sqrt();
    if denom == 0.0 {
        Ok(1.0)
    } else {
        Ok(1.0 - d / denom)
    }
}

/// Cosine distance for pre-normalized inputs: `1 − ⟨a,b⟩`.
///
/// # Errors
/// Returns [`Error::DimensionMismatch`] if `a.len() != b.len()`.
#[inline]
pub fn cosine_normalized(a: &[f32], b: &[f32]) -> Result<f32> {
    check(a, b)?;
    Ok(1.0 - dispatch::dot_kernel()(a, b))
}

/// Dispatch a distance computation by [`slate_core::Metric`].
///
/// Convenience for callers that hold a runtime `Metric`. `Cosine` here uses the
/// raw-input path ([`cosine`]); when the storage layer pre-normalizes vectors,
/// prefer calling [`cosine_normalized`] (or [`inner_product`]) directly to skip
/// the redundant norm computation.
///
/// # Errors
/// Returns [`Error::DimensionMismatch`] if `a.len() != b.len()`.
#[inline]
pub fn distance(metric: slate_core::Metric, a: &[f32], b: &[f32]) -> Result<f32> {
    use slate_core::Metric;
    match metric {
        Metric::L2 => l2_sq(a, b),
        Metric::InnerProduct => inner_product(a, b),
        Metric::Cosine => cosine(a, b),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimension_mismatch_is_reported() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [1.0f32, 2.0];
        assert!(matches!(
            l2_sq(&a, &b),
            Err(Error::DimensionMismatch { expected: 3, got: 2 })
        ));
    }

    #[test]
    fn l2_sq_known_value() {
        let a = [0.0f32, 0.0, 0.0];
        let b = [1.0f32, 2.0, 2.0];
        // 1 + 4 + 4 = 9
        assert!((l2_sq(&a, &b).unwrap() - 9.0).abs() < 1e-6);
    }

    #[test]
    fn inner_product_is_negated() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [1.0f32, 1.0, 1.0];
        // dot = 6 -> distance = -6
        assert!((inner_product(&a, &b).unwrap() + 6.0).abs() < 1e-6);
        assert!((dot(&a, &b).unwrap() - 6.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_identical_is_zero() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        assert!(cosine(&a, &a).unwrap().abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_is_one() {
        let a = [1.0f32, 0.0];
        let b = [0.0f32, 1.0];
        assert!((cosine(&a, &b).unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_zero_norm_is_one() {
        let a = [0.0f32, 0.0, 0.0];
        let b = [1.0f32, 2.0, 3.0];
        assert!((cosine(&a, &b).unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_normalized_matches_cosine_on_unit_vectors() {
        // Build two unit vectors.
        let a = [0.6f32, 0.8];
        let b = [1.0f32, 0.0];
        let raw = cosine(&a, &b).unwrap();
        let norm = cosine_normalized(&a, &b).unwrap();
        assert!((raw - norm).abs() < 1e-6);
    }

    #[test]
    fn active_tier_is_reported() {
        // Just ensure detection runs and returns a stable value.
        let t = active_tier();
        assert_eq!(t, active_tier());
        println!("active tier: {}", t.as_str());
    }
}

/// Property tests: the dispatched (vectorized) public API must agree with the
/// scalar oracle within a relative epsilon, for arbitrary vectors and lengths.
///
/// On this machine the dispatcher selects AVX2, so these directly validate the
/// AVX2 tier against scalar. On AVX-512 / NEON hardware the same tests validate
/// those tiers. Lengths deliberately include non-multiples of every lane width
/// (8 for AVX2, 16 for AVX-512, 4 for NEON) to exercise the remainder/mask tail.
#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    /// Absolute-error comparison whose tolerance scales with an explicit error
    /// bound `scale`. For sums/dot-products the rounding error is bounded by
    /// `ε · Σ|terms|`, NOT by the magnitude of the (possibly cancelled) result —
    /// so callers pass the sum of absolute term magnitudes as `scale`. This is
    /// the numerically correct way to compare a sequential (scalar) reduction
    /// against a tree (SIMD) reduction.
    fn approx_eq_scaled(got: f32, want: f32, scale: f32) -> bool {
        let tol = 1e-4 * scale.max(1.0);
        (got - want).abs() <= tol
    }

    /// Sum of absolute element-wise products `Σ|aᵢ·bᵢ|` — the summation error
    /// scale for dot products and (negated) inner products.
    fn dot_scale(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x * y).abs()).sum()
    }

    /// Sum of absolute squared differences `Σ(aᵢ−bᵢ)²` — the error scale for L2².
    fn l2_scale(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
    }

    // Vectors in a bounded range keep accumulation well-conditioned; lengths
    // 0..=257 span several lane-width multiples plus odd tails.
    prop_compose! {
        fn vec_pair()(len in 0usize..=257)
                     (a in prop::collection::vec(-10.0f32..10.0, len),
                      b in prop::collection::vec(-10.0f32..10.0, len))
                     -> (Vec<f32>, Vec<f32>) {
            (a, b)
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2000))]

        #[test]
        fn l2_sq_matches_oracle((a, b) in vec_pair()) {
            let got = l2_sq(&a, &b).unwrap();
            let want = scalar::l2_sq(&a, &b);
            let scale = l2_scale(&a, &b);
            prop_assert!(approx_eq_scaled(got, want, scale),
                "got={got} want={want} scale={scale} len={}", a.len());
        }

        #[test]
        fn dot_matches_oracle((a, b) in vec_pair()) {
            let got = dot(&a, &b).unwrap();
            let want = scalar::dot(&a, &b);
            let scale = dot_scale(&a, &b);
            prop_assert!(approx_eq_scaled(got, want, scale),
                "got={got} want={want} scale={scale} len={}", a.len());
        }

        #[test]
        fn inner_product_is_negated_dot((a, b) in vec_pair()) {
            let got = inner_product(&a, &b).unwrap();
            let want = -scalar::dot(&a, &b);
            let scale = dot_scale(&a, &b);
            prop_assert!(approx_eq_scaled(got, want, scale));
        }

        #[test]
        fn cosine_matches_oracle((a, b) in vec_pair()) {
            let got = cosine(&a, &b).unwrap();
            let want = scalar::cosine_distance(&a, &b);
            // Cosine is bounded in [0, 2]; its denominator normalizes magnitudes,
            // so a small fixed absolute tolerance is appropriate here.
            prop_assert!((got - want).abs() <= 1e-4,
                "got={got} want={want} len={}", a.len());
        }
    }
}
