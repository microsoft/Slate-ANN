//! AVX-512F kernels (x86-64).
//!
//! 16 `f32` lanes per iteration. These are **compiled but runtime-gated**: the
//! Zen 3 dev box has no AVX-512, so this tier is never selected locally. It is
//! validated for correctness on AVX-512 hardware / CI; the property tests still
//! compile it. Carefully written against the Intel intrinsics reference.
//!
//! # Safety
//! Every public function is `#[target_feature(enable = "avx512f")]` and
//! `unsafe`: the caller MUST ensure AVX-512F is present. [`crate::dispatch`]
//! guarantees this. Intrinsic calls are wrapped in an explicit `unsafe` block
//! (the crate denies `unsafe_op_in_unsafe_fn`).

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// L2² distance over `f32` using AVX-512F.
///
/// Uses a mask-tail: the final partial chunk is handled with a `loadu` masked by
/// the active-lane bitmask, avoiding a separate scalar loop.
///
/// # Safety
/// Requires AVX-512F. `a.len() == b.len()` assumed.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    // SAFETY: AVX-512F guaranteed by target_feature precondition. Full 16-lane
    // chunks loaded only while `i + 16 <= n`. The masked tail load activates
    // exactly the `rem` valid trailing lanes (mask = (1<<rem)-1); inactive lanes
    // read as zero and never touch out-of-bounds memory.
    unsafe {
        let mut acc = _mm512_setzero_ps();
        let mut i = 0usize;
        while i + 16 <= n {
            let va = _mm512_loadu_ps(pa.add(i));
            let vb = _mm512_loadu_ps(pb.add(i));
            let d = _mm512_sub_ps(va, vb);
            acc = _mm512_fmadd_ps(d, d, acc);
            i += 16;
        }
        let rem = n - i;
        if rem > 0 {
            let mask: __mmask16 = (1u16 << rem) - 1;
            let va = _mm512_maskz_loadu_ps(mask, pa.add(i));
            let vb = _mm512_maskz_loadu_ps(mask, pb.add(i));
            let d = _mm512_sub_ps(va, vb);
            acc = _mm512_fmadd_ps(d, d, acc);
        }
        _mm512_reduce_add_ps(acc)
    }
}

/// Raw inner product `⟨a,b⟩` over `f32` using AVX-512F.
///
/// # Safety
/// Requires AVX-512F. `a.len() == b.len()` assumed.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    // SAFETY: AVX-512F guaranteed; full chunks under bound; masked tail touches
    // only `rem` valid lanes. See `l2_sq` for the detailed argument.
    unsafe {
        let mut acc = _mm512_setzero_ps();
        let mut i = 0usize;
        while i + 16 <= n {
            let va = _mm512_loadu_ps(pa.add(i));
            let vb = _mm512_loadu_ps(pb.add(i));
            acc = _mm512_fmadd_ps(va, vb, acc);
            i += 16;
        }
        let rem = n - i;
        if rem > 0 {
            let mask: __mmask16 = (1u16 << rem) - 1;
            let va = _mm512_maskz_loadu_ps(mask, pa.add(i));
            let vb = _mm512_maskz_loadu_ps(mask, pb.add(i));
            acc = _mm512_fmadd_ps(va, vb, acc);
        }
        _mm512_reduce_add_ps(acc)
    }
}

/// Single-pass cosine accumulators `(⟨a,b⟩, ‖a‖², ‖b‖²)` using AVX-512F.
///
/// # Safety
/// Requires AVX-512F. `a.len() == b.len()` assumed.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn cosine_parts(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    // SAFETY: AVX-512F guaranteed; full chunks under bound; masked tail touches
    // only `rem` valid lanes. See `l2_sq` for the detailed argument.
    unsafe {
        let mut dot_acc = _mm512_setzero_ps();
        let mut na_acc = _mm512_setzero_ps();
        let mut nb_acc = _mm512_setzero_ps();
        let mut i = 0usize;
        while i + 16 <= n {
            let va = _mm512_loadu_ps(pa.add(i));
            let vb = _mm512_loadu_ps(pb.add(i));
            dot_acc = _mm512_fmadd_ps(va, vb, dot_acc);
            na_acc = _mm512_fmadd_ps(va, va, na_acc);
            nb_acc = _mm512_fmadd_ps(vb, vb, nb_acc);
            i += 16;
        }
        let rem = n - i;
        if rem > 0 {
            let mask: __mmask16 = (1u16 << rem) - 1;
            let va = _mm512_maskz_loadu_ps(mask, pa.add(i));
            let vb = _mm512_maskz_loadu_ps(mask, pb.add(i));
            dot_acc = _mm512_fmadd_ps(va, vb, dot_acc);
            na_acc = _mm512_fmadd_ps(va, va, na_acc);
            nb_acc = _mm512_fmadd_ps(vb, vb, nb_acc);
        }
        (
            _mm512_reduce_add_ps(dot_acc),
            _mm512_reduce_add_ps(na_acc),
            _mm512_reduce_add_ps(nb_acc),
        )
    }
}

// --- Narrow-stored kernels (f32 query vs f16/i8 store) -----------------------
//
// The query is always `f32`; only the stored vector is narrow. Each iteration
// widens 16 stored elements to a `__m512` of `f32` and runs the identical
// sub/fmadd reduction as the `f32` kernels above, then a scalar tail handles the
// final partial chunk. AVX-512F covers `vcvtph2ps` (`_mm512_cvtph_ps`), so the
// `f16` kernels need no separate `f16c` gate.

/// Decode one little-endian `f16` element at logical index `i` of `stored`.
///
/// # Safety
/// `2*i + 1 < stored.len()`. No target feature required (scalar conversion).
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn f16_at(p: *const u8, i: usize) -> f32 {
    // SAFETY: caller guarantees the two bytes at `2*i` are readable.
    let bits = unsafe { u16::from(*p.add(2 * i)) | (u16::from(*p.add(2 * i + 1)) << 8) };
    half::f16::from_bits(bits).to_f32()
}

/// Widen 16 little-endian `f16` bytes at `p` to a `__m512` of 16 `f32`.
///
/// # Safety
/// Requires AVX-512F. `p` must point to at least 32 readable bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn widen_f16(p: *const u8) -> __m512 {
    // SAFETY: caller guarantees 32 readable bytes at `p` and AVX-512F support.
    // `loadu` is unaligned-safe; `cvtph_ps` widens 16 packed halves to 16 f32.
    unsafe {
        let packed = _mm256_loadu_si256(p.cast::<__m256i>());
        _mm512_cvtph_ps(packed)
    }
}

/// Widen 16 `i8` codes at `p` to a `__m512` of 16 `f32` scaled by `scale`.
///
/// # Safety
/// Requires AVX-512F. `p` must point to at least 16 readable bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn widen_i8(p: *const i8, scale: __m512) -> __m512 {
    // SAFETY: caller guarantees 16 readable bytes at `p` and AVX-512F support.
    unsafe {
        let packed = _mm_loadu_si128(p.cast::<__m128i>());
        let widened = _mm512_cvtepi8_epi32(packed);
        let floats = _mm512_cvtepi32_ps(widened);
        _mm512_mul_ps(floats, scale)
    }
}

/// L2² distance between an `f32` query and an `f16`-stored vector (AVX-512F).
///
/// `stored` holds `query.len()` little-endian `f16` elements (`2·len` bytes).
///
/// # Safety
/// Requires AVX-512F. `stored.len() == 2 * query.len()` assumed.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn l2_sq_f16(query: &[f32], stored: &[u8]) -> f32 {
    debug_assert_eq!(stored.len(), 2 * query.len());
    let n = query.len();
    let pq = query.as_ptr();
    let ps = stored.as_ptr();
    // SAFETY: AVX-512F guaranteed. Full 16-lane chunks load 16 f32 from `pq` and
    // 32 bytes (16 f16) from `ps` while `i + 16 <= n`. The scalar tail reads f16
    // at byte offset `2*i` (< stored.len()) and `query[i]`.
    unsafe {
        let mut acc = _mm512_setzero_ps();
        let mut i = 0usize;
        while i + 16 <= n {
            let vq = _mm512_loadu_ps(pq.add(i));
            let vs = widen_f16(ps.add(2 * i));
            let d = _mm512_sub_ps(vq, vs);
            acc = _mm512_fmadd_ps(d, d, acc);
            i += 16;
        }
        let mut tail = _mm512_reduce_add_ps(acc);
        while i < n {
            let s = f16_at(ps, i);
            let d = *pq.add(i) - s;
            tail += d * d;
            i += 1;
        }
        tail
    }
}

/// Raw inner product `⟨query, f16-stored⟩` (AVX-512F).
///
/// # Safety
/// Requires AVX-512F. `stored.len() == 2 * query.len()` assumed.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn dot_f16(query: &[f32], stored: &[u8]) -> f32 {
    debug_assert_eq!(stored.len(), 2 * query.len());
    let n = query.len();
    let pq = query.as_ptr();
    let ps = stored.as_ptr();
    // SAFETY: as in `l2_sq_f16`.
    unsafe {
        let mut acc = _mm512_setzero_ps();
        let mut i = 0usize;
        while i + 16 <= n {
            let vq = _mm512_loadu_ps(pq.add(i));
            let vs = widen_f16(ps.add(2 * i));
            acc = _mm512_fmadd_ps(vq, vs, acc);
            i += 16;
        }
        let mut tail = _mm512_reduce_add_ps(acc);
        while i < n {
            tail += *pq.add(i) * f16_at(ps, i);
            i += 1;
        }
        tail
    }
}

/// Single-pass cosine accumulators `(⟨q,s⟩, ‖q‖², ‖s‖²)` for an `f16` store
/// (AVX-512F).
///
/// # Safety
/// Requires AVX-512F. `stored.len() == 2 * query.len()` assumed.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn cosine_parts_f16(query: &[f32], stored: &[u8]) -> (f32, f32, f32) {
    debug_assert_eq!(stored.len(), 2 * query.len());
    let n = query.len();
    let pq = query.as_ptr();
    let ps = stored.as_ptr();
    // SAFETY: as in `l2_sq_f16`.
    unsafe {
        let mut dot_acc = _mm512_setzero_ps();
        let mut nq_acc = _mm512_setzero_ps();
        let mut ns_acc = _mm512_setzero_ps();
        let mut i = 0usize;
        while i + 16 <= n {
            let vq = _mm512_loadu_ps(pq.add(i));
            let vs = widen_f16(ps.add(2 * i));
            dot_acc = _mm512_fmadd_ps(vq, vs, dot_acc);
            nq_acc = _mm512_fmadd_ps(vq, vq, nq_acc);
            ns_acc = _mm512_fmadd_ps(vs, vs, ns_acc);
            i += 16;
        }
        let mut dot = _mm512_reduce_add_ps(dot_acc);
        let mut nq = _mm512_reduce_add_ps(nq_acc);
        let mut ns = _mm512_reduce_add_ps(ns_acc);
        while i < n {
            let q = *pq.add(i);
            let s = f16_at(ps, i);
            dot += q * s;
            nq += q * q;
            ns += s * s;
            i += 1;
        }
        (dot, nq, ns)
    }
}

/// L2² distance between an `f32` query and an `i8`-stored vector (AVX-512F).
///
/// `codes` holds `query.len()` signed codes; the dequantized value is
/// `code * scale`.
///
/// # Safety
/// Requires AVX-512F. `codes.len() == query.len()` assumed.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn l2_sq_i8(query: &[f32], scale: f32, codes: &[i8]) -> f32 {
    debug_assert_eq!(codes.len(), query.len());
    let n = query.len();
    let pq = query.as_ptr();
    let pc = codes.as_ptr();
    // SAFETY: AVX-512F guaranteed. Full chunks load 16 f32 from `pq` and 16 i8
    // from `pc` while `i + 16 <= n`. The scalar tail reads `query[i]`/`codes[i]`.
    unsafe {
        let vscale = _mm512_set1_ps(scale);
        let mut acc = _mm512_setzero_ps();
        let mut i = 0usize;
        while i + 16 <= n {
            let vq = _mm512_loadu_ps(pq.add(i));
            let vs = widen_i8(pc.add(i), vscale);
            let d = _mm512_sub_ps(vq, vs);
            acc = _mm512_fmadd_ps(d, d, acc);
            i += 16;
        }
        let mut tail = _mm512_reduce_add_ps(acc);
        while i < n {
            let s = f32::from(*pc.add(i)) * scale;
            let d = *pq.add(i) - s;
            tail += d * d;
            i += 1;
        }
        tail
    }
}

/// Raw inner product `⟨query, i8-stored⟩` (AVX-512F).
///
/// # Safety
/// Requires AVX-512F. `codes.len() == query.len()` assumed.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn dot_i8(query: &[f32], scale: f32, codes: &[i8]) -> f32 {
    debug_assert_eq!(codes.len(), query.len());
    let n = query.len();
    let pq = query.as_ptr();
    let pc = codes.as_ptr();
    // SAFETY: as in `l2_sq_i8`.
    unsafe {
        let vscale = _mm512_set1_ps(scale);
        let mut acc = _mm512_setzero_ps();
        let mut i = 0usize;
        while i + 16 <= n {
            let vq = _mm512_loadu_ps(pq.add(i));
            let vs = widen_i8(pc.add(i), vscale);
            acc = _mm512_fmadd_ps(vq, vs, acc);
            i += 16;
        }
        let mut tail = _mm512_reduce_add_ps(acc);
        while i < n {
            tail += *pq.add(i) * (f32::from(*pc.add(i)) * scale);
            i += 1;
        }
        tail
    }
}

/// Single-pass cosine accumulators `(⟨q,s⟩, ‖q‖², ‖s‖²)` for an `i8` store
/// (AVX-512F).
///
/// # Safety
/// Requires AVX-512F. `codes.len() == query.len()` assumed.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn cosine_parts_i8(query: &[f32], scale: f32, codes: &[i8]) -> (f32, f32, f32) {
    debug_assert_eq!(codes.len(), query.len());
    let n = query.len();
    let pq = query.as_ptr();
    let pc = codes.as_ptr();
    // SAFETY: as in `l2_sq_i8`.
    unsafe {
        let vscale = _mm512_set1_ps(scale);
        let mut dot_acc = _mm512_setzero_ps();
        let mut nq_acc = _mm512_setzero_ps();
        let mut ns_acc = _mm512_setzero_ps();
        let mut i = 0usize;
        while i + 16 <= n {
            let vq = _mm512_loadu_ps(pq.add(i));
            let vs = widen_i8(pc.add(i), vscale);
            dot_acc = _mm512_fmadd_ps(vq, vs, dot_acc);
            nq_acc = _mm512_fmadd_ps(vq, vq, nq_acc);
            ns_acc = _mm512_fmadd_ps(vs, vs, ns_acc);
            i += 16;
        }
        let mut dot = _mm512_reduce_add_ps(dot_acc);
        let mut nq = _mm512_reduce_add_ps(nq_acc);
        let mut ns = _mm512_reduce_add_ps(ns_acc);
        while i < n {
            let q = *pq.add(i);
            let s = f32::from(*pc.add(i)) * scale;
            dot += q * s;
            nq += q * q;
            ns += s * s;
            i += 1;
        }
        (dot, nq, ns)
    }
}
