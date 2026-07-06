//! # slate-simd
//!
//! SIMD distance kernels for Slate-ANN with runtime CPU-feature dispatch.
//!
//! Provides L2², inner-product, and cosine distance over `f32`, plus
//! asymmetric `f32`-query vs narrow-stored kernels ([`distance_f16`],
//! [`distance_i8`]) that widen the on-disk representation inside the SIMD
//! reduction so narrow stores skip a decode-to-`f32` pass. Four implementation
//! tiers are selected at runtime — AVX-512, AVX2+FMA, ARM NEON, and a portable
//! scalar fallback (also the correctness oracle for the vectorized paths).
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
//! Populated in Phase 1 (f32 kernels + dispatch); narrow-store kernels added in
//! the Phase-9.5 deferred clean-ups.

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

/// Distance between an `f32` query and an `f16`-stored vector, by [`Metric`].
///
/// `stored` holds `query.len()` little-endian `f16` elements (`2·len` bytes),
/// exactly as written by the storage codec. The result is numerically identical
/// to decoding `stored` to `f32` and calling [`distance`] — the native kernels
/// just fuse the widen into the SIMD reduction. `Cosine` uses the raw-input
/// path; zero-norm operands yield `1.0`.
///
/// [`Metric`]: slate_core::Metric
///
/// # Errors
/// Returns [`Error::DimensionMismatch`] if `stored.len() != 2 * query.len()`.
#[inline]
pub fn distance_f16(metric: slate_core::Metric, query: &[f32], stored: &[u8]) -> Result<f32> {
    use slate_core::Metric;
    if stored.len() != 2 * query.len() {
        return Err(Error::DimensionMismatch {
            expected: 2 * query.len(),
            got: stored.len(),
        });
    }
    match metric {
        Metric::L2 => Ok(dispatch::l2_sq_f16(query, stored)),
        Metric::InnerProduct => Ok(-dispatch::dot_f16(query, stored)),
        Metric::Cosine => {
            let (d, nq, ns) = dispatch::cosine_parts_f16(query, stored);
            let denom = (nq * ns).sqrt();
            if denom == 0.0 {
                Ok(1.0)
            } else {
                Ok(1.0 - d / denom)
            }
        }
    }
}

/// Distance between an `f32` query and an `i8`-stored vector, by [`Metric`].
///
/// `codes` holds `query.len()` signed codes whose dequantized value is
/// `code * scale` (the symmetric per-vector scale written by the storage codec).
/// The result is numerically identical to decoding to `f32` and calling
/// [`distance`]. `Cosine` uses the raw-input path; zero-norm operands yield
/// `1.0`.
///
/// [`Metric`]: slate_core::Metric
///
/// # Errors
/// Returns [`Error::DimensionMismatch`] if `codes.len() != query.len()`.
#[inline]
pub fn distance_i8(
    metric: slate_core::Metric,
    query: &[f32],
    scale: f32,
    codes: &[i8],
) -> Result<f32> {
    use slate_core::Metric;
    if codes.len() != query.len() {
        return Err(Error::DimensionMismatch {
            expected: query.len(),
            got: codes.len(),
        });
    }
    match metric {
        Metric::L2 => Ok(dispatch::l2_sq_i8(query, scale, codes)),
        Metric::InnerProduct => Ok(-dispatch::dot_i8(query, scale, codes)),
        Metric::Cosine => {
            let (d, nq, ns) = dispatch::cosine_parts_i8(query, scale, codes);
            let denom = (nq * ns).sqrt();
            if denom == 0.0 {
                Ok(1.0)
            } else {
                Ok(1.0 - d / denom)
            }
        }
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
            Err(Error::DimensionMismatch {
                expected: 3,
                got: 2
            })
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

    #[test]
    fn distance_f16_equals_decode_then_distance() {
        use half::f16;
        use slate_core::Metric;
        let query = [0.5f32, -1.25, 3.0, 0.0, -2.5, 7.5, -0.125, 4.0, 1.0];
        let raw = [0.4f32, -1.0, 3.25, 0.5, -2.0, 7.0, -0.25, 4.5, 0.75];
        let stored: Vec<u8> = raw
            .iter()
            .flat_map(|&x| f16::from_f32(x).to_le_bytes())
            .collect();
        // half -> single is exact, so the native kernel must equal decode-then-f32.
        let decoded: Vec<f32> = raw.iter().map(|&x| f16::from_f32(x).to_f32()).collect();
        for metric in [Metric::L2, Metric::InnerProduct, Metric::Cosine] {
            let native = distance_f16(metric, &query, &stored).unwrap();
            let reference = distance(metric, &query, &decoded).unwrap();
            assert!(
                (native - reference).abs() <= 1e-6,
                "metric={metric:?} native={native} reference={reference}"
            );
        }
    }

    #[test]
    fn distance_i8_equals_decode_then_distance() {
        use slate_core::Metric;
        let query = [0.5f32, -1.25, 3.0, 0.0, -2.5, 7.5, -0.125, 4.0, 1.0];
        let scale = 0.05f32;
        let codes = [10i8, -20, 60, 0, -50, 127, -3, 80, 15];
        let decoded: Vec<f32> = codes.iter().map(|&c| f32::from(c) * scale).collect();
        for metric in [Metric::L2, Metric::InnerProduct, Metric::Cosine] {
            let native = distance_i8(metric, &query, scale, &codes).unwrap();
            let reference = distance(metric, &query, &decoded).unwrap();
            assert!(
                (native - reference).abs() <= 1e-6,
                "metric={metric:?} native={native} reference={reference}"
            );
        }
    }

    #[test]
    fn narrow_distance_rejects_wrong_length() {
        use slate_core::Metric;
        let query = [1.0f32, 2.0, 3.0];
        // f16 stored must be 2*len bytes.
        assert!(matches!(
            distance_f16(Metric::L2, &query, &[0u8; 4]),
            Err(Error::DimensionMismatch {
                expected: 6,
                got: 4
            })
        ));
        assert!(matches!(
            distance_i8(Metric::L2, &query, 1.0, &[0i8; 2]),
            Err(Error::DimensionMismatch {
                expected: 3,
                got: 2
            })
        ));
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

    use half::f16;
    use slate_core::Metric;

    /// Encode an `f32` vector to the on-disk f16 byte layout (2 bytes/elem, LE),
    /// matching the storage codec without depending on `slate-storage`.
    fn encode_f16(v: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(2 * v.len());
        for &x in v {
            out.extend_from_slice(&f16::from_f32(x).to_le_bytes());
        }
        out
    }

    /// Symmetric per-vector i8 quantization, mirroring the storage codec:
    /// `scale = max|x| / 127`, `code = round(x / scale)` clamped to `[-127, 127]`;
    /// an all-zero vector yields `scale = 0` and all-zero codes.
    fn encode_i8(v: &[f32]) -> (f32, Vec<i8>) {
        let max_abs = v.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        let scale = if max_abs == 0.0 { 0.0 } else { max_abs / 127.0 };
        let codes = v
            .iter()
            .map(|&x| {
                if scale == 0.0 {
                    0i8
                } else {
                    (x / scale).round().clamp(-127.0, 127.0) as i8
                }
            })
            .collect();
        (scale, codes)
    }

    /// Scalar reference for f16 distance: decode-then-compose, identical shape to
    /// the public `distance_f16` but always through the scalar kernels.
    fn scalar_distance_f16(metric: Metric, query: &[f32], stored: &[u8]) -> f32 {
        match metric {
            Metric::L2 => scalar::l2_sq_f16(query, stored),
            Metric::InnerProduct => -scalar::dot_f16(query, stored),
            Metric::Cosine => {
                let (d, nq, ns) = scalar::cosine_parts_f16(query, stored);
                let denom = (nq * ns).sqrt();
                if denom == 0.0 {
                    1.0
                } else {
                    1.0 - d / denom
                }
            }
        }
    }

    fn scalar_distance_i8(metric: Metric, query: &[f32], scale: f32, codes: &[i8]) -> f32 {
        match metric {
            Metric::L2 => scalar::l2_sq_i8(query, scale, codes),
            Metric::InnerProduct => -scalar::dot_i8(query, scale, codes),
            Metric::Cosine => {
                let (d, nq, ns) = scalar::cosine_parts_i8(query, scale, codes);
                let denom = (nq * ns).sqrt();
                if denom == 0.0 {
                    1.0
                } else {
                    1.0 - d / denom
                }
            }
        }
    }

    /// `Σ|query·stored|` error scale for the f16/i8 dot reductions.
    fn dot_scale_decoded(query: &[f32], stored: &[f32]) -> f32 {
        query.iter().zip(stored).map(|(x, y)| (x * y).abs()).sum()
    }

    /// `Σ(query−stored)²` error scale for the f16/i8 L2 reductions.
    fn l2_scale_decoded(query: &[f32], stored: &[f32]) -> f32 {
        query
            .iter()
            .zip(stored)
            .map(|(x, y)| (x - y) * (x - y))
            .sum()
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2000))]

        #[test]
        fn l2_sq_f16_matches_oracle((q, v) in vec_pair()) {
            let stored = encode_f16(&v);
            let decoded: Vec<f32> = v.iter().map(|&x| f16::from_f32(x).to_f32()).collect();
            let got = distance_f16(Metric::L2, &q, &stored).unwrap();
            let want = scalar_distance_f16(Metric::L2, &q, &stored);
            let scale = l2_scale_decoded(&q, &decoded);
            prop_assert!(approx_eq_scaled(got, want, scale),
                "got={got} want={want} scale={scale} len={}", q.len());
        }

        #[test]
        fn dot_f16_matches_oracle((q, v) in vec_pair()) {
            let stored = encode_f16(&v);
            let decoded: Vec<f32> = v.iter().map(|&x| f16::from_f32(x).to_f32()).collect();
            let got = distance_f16(Metric::InnerProduct, &q, &stored).unwrap();
            let want = scalar_distance_f16(Metric::InnerProduct, &q, &stored);
            let scale = dot_scale_decoded(&q, &decoded);
            prop_assert!(approx_eq_scaled(got, want, scale),
                "got={got} want={want} scale={scale} len={}", q.len());
        }

        #[test]
        fn cosine_f16_matches_oracle((q, v) in vec_pair()) {
            let stored = encode_f16(&v);
            let got = distance_f16(Metric::Cosine, &q, &stored).unwrap();
            let want = scalar_distance_f16(Metric::Cosine, &q, &stored);
            prop_assert!((got - want).abs() <= 1e-4,
                "got={got} want={want} len={}", q.len());
        }

        #[test]
        fn l2_sq_i8_matches_oracle((q, v) in vec_pair()) {
            let (scale_q, codes) = encode_i8(&v);
            let decoded: Vec<f32> = codes.iter().map(|&c| f32::from(c) * scale_q).collect();
            let got = distance_i8(Metric::L2, &q, scale_q, &codes).unwrap();
            let want = scalar_distance_i8(Metric::L2, &q, scale_q, &codes);
            let scale = l2_scale_decoded(&q, &decoded);
            prop_assert!(approx_eq_scaled(got, want, scale),
                "got={got} want={want} scale={scale} len={}", q.len());
        }

        #[test]
        fn dot_i8_matches_oracle((q, v) in vec_pair()) {
            let (scale_q, codes) = encode_i8(&v);
            let decoded: Vec<f32> = codes.iter().map(|&c| f32::from(c) * scale_q).collect();
            let got = distance_i8(Metric::InnerProduct, &q, scale_q, &codes).unwrap();
            let want = scalar_distance_i8(Metric::InnerProduct, &q, scale_q, &codes);
            let scale = dot_scale_decoded(&q, &decoded);
            prop_assert!(approx_eq_scaled(got, want, scale),
                "got={got} want={want} scale={scale} len={}", q.len());
        }

        #[test]
        fn cosine_i8_matches_oracle((q, v) in vec_pair()) {
            let (scale_q, codes) = encode_i8(&v);
            let got = distance_i8(Metric::Cosine, &q, scale_q, &codes).unwrap();
            let want = scalar_distance_i8(Metric::Cosine, &q, scale_q, &codes);
            prop_assert!((got - want).abs() <= 1e-4,
                "got={got} want={want} len={}", q.len());
        }
    }
}
