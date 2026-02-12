//! Scalar reference kernels.
//!
//! These are the portable fallback **and** the correctness oracle: every
//! vectorized tier (AVX2 / AVX-512 / NEON) is property-tested to agree with
//! these functions within a relative epsilon. They must therefore be obviously
//! correct and free of any platform-specific behavior.
//!
//! All functions assume `a.len() == b.len()`; callers in [`crate`] validate
//! lengths before dispatch.

/// Squared Euclidean (L2²) distance: `Σ (a_i − b_i)²`.
///
/// No square root is taken — search ranks by ascending score and `sqrt` is
/// monotonic, so omitting it preserves ordering and saves work.
#[inline]
#[must_use]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc
}

/// Raw inner product `⟨a, b⟩ = Σ a_i · b_i`.
///
/// This is the *unnegated* dot product. The metric-facing score negates it (see
/// [`inner_product_distance`]) so that "more similar" maps to "smaller".
#[inline]
#[must_use]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

/// Inner-product *distance* = `−⟨a, b⟩` (ascending-rank convention).
#[inline]
#[must_use]
pub fn inner_product_distance(a: &[f32], b: &[f32]) -> f32 {
    -dot(a, b)
}

/// Cosine *distance* = `1 − cos(a, b)` over raw (un-normalized) inputs.
///
/// Computes `1 − ⟨a,b⟩ / (‖a‖·‖b‖)` in a single pass. If either vector has zero
/// norm the cosine similarity is defined here as `0` (distance `1`), which keeps
/// the result finite and within `[0, 2]`.
#[inline]
#[must_use]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot_acc = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot_acc += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = (na * nb).sqrt();
    if denom == 0.0 {
        1.0
    } else {
        1.0 - dot_acc / denom
    }
}

/// Cosine *distance* for inputs already L2-normalized: `1 − ⟨a,b⟩`.
///
/// When the storage layer normalizes vectors on ingest (cosine metric), the
/// per-query cost collapses to a single dot product.
#[inline]
#[must_use]
pub fn cosine_distance_normalized(a: &[f32], b: &[f32]) -> f32 {
    1.0 - dot(a, b)
}
