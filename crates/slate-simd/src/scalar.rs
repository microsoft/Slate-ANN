//! Scalar reference kernels.
//!
//! These are the portable fallback **and** the correctness oracle: every
//! vectorized tier (AVX2 / AVX-512 / NEON) is property-tested to agree with
//! these functions within a relative epsilon. They must therefore be obviously
//! correct and free of any platform-specific behavior.
//!
//! All functions assume `a.len() == b.len()`; callers in [`crate`] validate
//! lengths before dispatch.
//!
//! # Narrow-stored kernels
//!
//! The `*_f16` / `*_i8` variants below are the oracle for the native narrow
//! kernels (AVX2 / AVX-512 / NEON). They take an `f32` query against a vector
//! stored in a narrow on-disk representation and decode each stored element to
//! `f32` *exactly* as the storage codec does, then run the same `f32`
//! arithmetic. The native SIMD kernels fuse that widening into the reduction
//! and must reproduce these results.

use half::f16;

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

/// Decode one IEEE binary16 element stored as two little-endian bytes to `f32`.
///
/// `half → single` is exact (every `f16` is representable in `f32`), so this is
/// bit-identical to the storage codec's decode path.
#[inline]
#[must_use]
fn decode_f16_chunk(chunk: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
    f16::from_bits(bits).to_f32()
}

/// L2² distance between an `f32` query and an `f16`-stored vector.
///
/// `stored` holds `query.len()` little-endian `f16` elements (`2·len` bytes).
#[inline]
#[must_use]
pub fn l2_sq_f16(query: &[f32], stored: &[u8]) -> f32 {
    debug_assert_eq!(stored.len(), 2 * query.len());
    let mut acc = 0.0f32;
    for (&q, chunk) in query.iter().zip(stored.chunks_exact(2)) {
        let d = q - decode_f16_chunk(chunk);
        acc += d * d;
    }
    acc
}

/// Raw inner product `⟨query, f16-stored⟩`.
#[inline]
#[must_use]
pub fn dot_f16(query: &[f32], stored: &[u8]) -> f32 {
    debug_assert_eq!(stored.len(), 2 * query.len());
    let mut acc = 0.0f32;
    for (&q, chunk) in query.iter().zip(stored.chunks_exact(2)) {
        acc += q * decode_f16_chunk(chunk);
    }
    acc
}

/// Cosine accumulators `(⟨q,s⟩, ‖q‖², ‖s‖²)` for an `f32` query vs `f16` store.
#[inline]
#[must_use]
pub fn cosine_parts_f16(query: &[f32], stored: &[u8]) -> (f32, f32, f32) {
    debug_assert_eq!(stored.len(), 2 * query.len());
    let mut dot_acc = 0.0f32;
    let mut nq = 0.0f32;
    let mut ns = 0.0f32;
    for (&q, chunk) in query.iter().zip(stored.chunks_exact(2)) {
        let s = decode_f16_chunk(chunk);
        dot_acc += q * s;
        nq += q * q;
        ns += s * s;
    }
    (dot_acc, nq, ns)
}

/// L2² distance between an `f32` query and an `i8`-stored vector.
///
/// The stored vector dequantizes as `code · scale` (symmetric per-vector scale),
/// matching the storage codec exactly.
#[inline]
#[must_use]
pub fn l2_sq_i8(query: &[f32], scale: f32, codes: &[i8]) -> f32 {
    debug_assert_eq!(codes.len(), query.len());
    let mut acc = 0.0f32;
    for (&q, &c) in query.iter().zip(codes) {
        let d = q - f32::from(c) * scale;
        acc += d * d;
    }
    acc
}

/// Raw inner product `⟨query, i8-stored⟩` with per-vector `scale`.
#[inline]
#[must_use]
pub fn dot_i8(query: &[f32], scale: f32, codes: &[i8]) -> f32 {
    debug_assert_eq!(codes.len(), query.len());
    let mut acc = 0.0f32;
    for (&q, &c) in query.iter().zip(codes) {
        acc += q * (f32::from(c) * scale);
    }
    acc
}

/// Cosine accumulators `(⟨q,s⟩, ‖q‖², ‖s‖²)` for an `f32` query vs `i8` store.
#[inline]
#[must_use]
pub fn cosine_parts_i8(query: &[f32], scale: f32, codes: &[i8]) -> (f32, f32, f32) {
    debug_assert_eq!(codes.len(), query.len());
    let mut dot_acc = 0.0f32;
    let mut nq = 0.0f32;
    let mut ns = 0.0f32;
    for (&q, &c) in query.iter().zip(codes) {
        let s = f32::from(c) * scale;
        dot_acc += q * s;
        nq += q * q;
        ns += s * s;
    }
    (dot_acc, nq, ns)
}
