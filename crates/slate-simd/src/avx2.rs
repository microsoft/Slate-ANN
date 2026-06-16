//! AVX2 + FMA kernels (x86-64).
//!
//! These execute on the Zen 3 dev box (AVX2 present, AVX-512 absent). Each
//! function processes 8 `f32` lanes per iteration and handles a scalar
//! remainder tail for lengths that are not a multiple of 8.
//!
//! # Safety
//! Every public function is `#[target_feature(enable = "avx2,fma")]` and
//! therefore `unsafe`: the caller MUST ensure the CPU supports AVX2 + FMA. The
//! runtime dispatcher in [`crate::dispatch`] guarantees this before taking a
//! pointer to any of these functions. Intrinsic calls are wrapped in an explicit
//! `unsafe` block (the crate denies `unsafe_op_in_unsafe_fn`).

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Horizontally sum the 8 lanes of a `__m256` into a single `f32`.
///
/// # Safety
/// Requires AVX. Caller guarantees the target feature is present.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum256(v: __m256) -> f32 {
    // NOTE: every intrinsic used here is a register-only shuffle/add that Rust
    // classifies as safe; no `unsafe` block is needed. The fn stays `unsafe`
    // because it is `#[target_feature]`-gated (callers must have AVX).
    // Reduce 256 -> 128 by adding the high and low 128-bit halves.
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let sum128 = _mm_add_ps(lo, hi);
    // Reduce 128 -> scalar via two horizontal adds.
    let shuf = _mm_movehdup_ps(sum128); // [1,1,3,3]
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    let sums2 = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(sums2)
}

/// L2² distance over `f32` using AVX2 + FMA.
///
/// # Safety
/// Requires AVX2 + FMA. `a.len() == b.len()` assumed.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    // SAFETY: AVX2+FMA guaranteed by target_feature precondition. Only full
    // 8-lane chunks are loaded while `i + 8 <= n`; `loadu` is unaligned-safe and
    // reads exactly 8 valid elements. The scalar tail dereferences indices
    // `i < n`, all in bounds. `hsum256` requires AVX (subset of AVX2).
    unsafe {
        let mut acc = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 8 <= n {
            let va = _mm256_loadu_ps(pa.add(i));
            let vb = _mm256_loadu_ps(pb.add(i));
            let d = _mm256_sub_ps(va, vb);
            acc = _mm256_fmadd_ps(d, d, acc); // acc += d*d
            i += 8;
        }
        let mut tail = hsum256(acc);
        while i < n {
            let d = *pa.add(i) - *pb.add(i);
            tail += d * d;
            i += 1;
        }
        tail
    }
}

/// Raw inner product `⟨a,b⟩` over `f32` using AVX2 + FMA.
///
/// # Safety
/// Requires AVX2 + FMA. `a.len() == b.len()` assumed.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    // SAFETY: as in `l2_sq` — target feature guaranteed; only full chunks loaded
    // under the bound; scalar tail indices are in bounds.
    unsafe {
        let mut acc = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 8 <= n {
            let va = _mm256_loadu_ps(pa.add(i));
            let vb = _mm256_loadu_ps(pb.add(i));
            acc = _mm256_fmadd_ps(va, vb, acc);
            i += 8;
        }
        let mut acc_s = hsum256(acc);
        while i < n {
            acc_s += *pa.add(i) * *pb.add(i);
            i += 1;
        }
        acc_s
    }
}

/// Single-pass cosine accumulators `(⟨a,b⟩, ‖a‖², ‖b‖²)` using AVX2 + FMA.
///
/// # Safety
/// Requires AVX2 + FMA. `a.len() == b.len()` assumed.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn cosine_parts(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    // SAFETY: target feature guaranteed; only full 8-lane chunks loaded while
    // `i + 8 <= n`; scalar tail indices `i < n` are in bounds.
    unsafe {
        let mut dot_acc = _mm256_setzero_ps();
        let mut na_acc = _mm256_setzero_ps();
        let mut nb_acc = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 8 <= n {
            let va = _mm256_loadu_ps(pa.add(i));
            let vb = _mm256_loadu_ps(pb.add(i));
            dot_acc = _mm256_fmadd_ps(va, vb, dot_acc);
            na_acc = _mm256_fmadd_ps(va, va, na_acc);
            nb_acc = _mm256_fmadd_ps(vb, vb, nb_acc);
            i += 8;
        }
        let mut d = hsum256(dot_acc);
        let mut na = hsum256(na_acc);
        let mut nb = hsum256(nb_acc);
        while i < n {
            let x = *pa.add(i);
            let y = *pb.add(i);
            d += x * y;
            na += x * x;
            nb += y * y;
            i += 1;
        }
        (d, na, nb)
    }
}

// --- Narrow-stored kernels (f32 query vs f16/i8 store) -----------------------
//
// The query is always `f32`; only the stored vector is narrow. Each iteration
// widens 8 stored elements to a `__m256` of `f32` and then runs the identical
// sub/fmadd reduction as the `f32` kernels above. The `f16` widen uses the F16C
// `vcvtph2ps` instruction, so these functions additionally require `f16c`; the
// dispatcher gates that separately (`is_x86_feature_detected!("f16c")`).

/// Widen 8 little-endian `f16` bytes at `p` to a `__m256` of 8 `f32`.
///
/// # Safety
/// Requires AVX + F16C. `p` must point to at least 16 readable bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
#[inline]
unsafe fn widen_f16(p: *const u8) -> __m256 {
    // SAFETY: caller guarantees 16 readable bytes at `p` and AVX2+F16C support.
    // `loadu` is unaligned-safe; `cvtph_ps` widens 8 packed halves to 8 f32.
    unsafe {
        let packed = _mm_loadu_si128(p.cast::<__m128i>());
        _mm256_cvtph_ps(packed)
    }
}

/// L2² distance between an `f32` query and an `f16`-stored vector (AVX2 + F16C).
///
/// `stored` holds `query.len()` little-endian `f16` elements (`2·len` bytes).
///
/// # Safety
/// Requires AVX2 + FMA + F16C. `stored.len() == 2 * query.len()` assumed.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn l2_sq_f16(query: &[f32], stored: &[u8]) -> f32 {
    debug_assert_eq!(stored.len(), 2 * query.len());
    let n = query.len();
    let pq = query.as_ptr();
    let ps = stored.as_ptr();
    // SAFETY: AVX2+FMA+F16C guaranteed. Full 8-lane chunks load 8 f32 from `pq`
    // and 16 bytes (8 f16) from `ps` while `i + 8 <= n`. The scalar tail reads
    // f16 elements at byte offset `2*i` (< stored.len()) and `query[i]`.
    unsafe {
        let mut acc = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 8 <= n {
            let vq = _mm256_loadu_ps(pq.add(i));
            let vs = widen_f16(ps.add(2 * i));
            let d = _mm256_sub_ps(vq, vs);
            acc = _mm256_fmadd_ps(d, d, acc);
            i += 8;
        }
        let mut tail = hsum256(acc);
        while i < n {
            let s = f16_at(ps, i);
            let d = *pq.add(i) - s;
            tail += d * d;
            i += 1;
        }
        tail
    }
}

/// Raw inner product `⟨query, f16-stored⟩` (AVX2 + F16C).
///
/// # Safety
/// Requires AVX2 + FMA + F16C. `stored.len() == 2 * query.len()` assumed.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn dot_f16(query: &[f32], stored: &[u8]) -> f32 {
    debug_assert_eq!(stored.len(), 2 * query.len());
    let n = query.len();
    let pq = query.as_ptr();
    let ps = stored.as_ptr();
    // SAFETY: as in `l2_sq_f16`.
    unsafe {
        let mut acc = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 8 <= n {
            let vq = _mm256_loadu_ps(pq.add(i));
            let vs = widen_f16(ps.add(2 * i));
            acc = _mm256_fmadd_ps(vq, vs, acc);
            i += 8;
        }
        let mut acc_s = hsum256(acc);
        while i < n {
            acc_s += *pq.add(i) * f16_at(ps, i);
            i += 1;
        }
        acc_s
    }
}

/// Single-pass cosine accumulators for `f32` query vs `f16` store (AVX2 + F16C).
///
/// # Safety
/// Requires AVX2 + FMA + F16C. `stored.len() == 2 * query.len()` assumed.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn cosine_parts_f16(query: &[f32], stored: &[u8]) -> (f32, f32, f32) {
    debug_assert_eq!(stored.len(), 2 * query.len());
    let n = query.len();
    let pq = query.as_ptr();
    let ps = stored.as_ptr();
    // SAFETY: as in `l2_sq_f16`.
    unsafe {
        let mut dot_acc = _mm256_setzero_ps();
        let mut nq_acc = _mm256_setzero_ps();
        let mut ns_acc = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 8 <= n {
            let vq = _mm256_loadu_ps(pq.add(i));
            let vs = widen_f16(ps.add(2 * i));
            dot_acc = _mm256_fmadd_ps(vq, vs, dot_acc);
            nq_acc = _mm256_fmadd_ps(vq, vq, nq_acc);
            ns_acc = _mm256_fmadd_ps(vs, vs, ns_acc);
            i += 8;
        }
        let mut d = hsum256(dot_acc);
        let mut nq = hsum256(nq_acc);
        let mut ns = hsum256(ns_acc);
        while i < n {
            let q = *pq.add(i);
            let s = f16_at(ps, i);
            d += q * s;
            nq += q * q;
            ns += s * s;
            i += 1;
        }
        (d, nq, ns)
    }
}

/// Decode one little-endian `f16` element (scalar tail helper).
///
/// # Safety
/// `p` must point to at least `2*(i+1)` readable bytes. Requires F16C only for
/// symmetry with the SIMD path; the conversion itself is a safe library call.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "f16c")]
#[inline]
unsafe fn f16_at(p: *const u8, i: usize) -> f32 {
    // SAFETY: caller guarantees `p.add(2*i)` and `p.add(2*i+1)` are readable.
    let bits = unsafe { u16::from(*p.add(2 * i)) | (u16::from(*p.add(2 * i + 1)) << 8) };
    half::f16::from_bits(bits).to_f32()
}

/// Widen 8 `i8` codes at `p` to a `__m256` of 8 `f32`, scaled by `scale`.
///
/// # Safety
/// Requires AVX2. `p` must point to at least 8 readable bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn widen_i8(p: *const i8, scale: __m256) -> __m256 {
    // SAFETY: caller guarantees 8 readable bytes at `p` and AVX2 support.
    // `loadl_epi64` reads the low 8 bytes; `cvtepi8_epi32` sign-extends the
    // low 8 `i8` lanes to 8 `i32`; `cvtepi32_ps` converts to f32.
    unsafe {
        let packed = _mm_loadl_epi64(p.cast::<__m128i>());
        let widened = _mm256_cvtepi8_epi32(packed);
        let floats = _mm256_cvtepi32_ps(widened);
        _mm256_mul_ps(floats, scale)
    }
}

/// L2² distance between an `f32` query and an `i8`-stored vector (AVX2).
///
/// The stored vector dequantizes as `code · scale` (symmetric per-vector scale).
///
/// # Safety
/// Requires AVX2 + FMA. `codes.len() == query.len()` assumed.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn l2_sq_i8(query: &[f32], scale: f32, codes: &[i8]) -> f32 {
    debug_assert_eq!(codes.len(), query.len());
    let n = query.len();
    let pq = query.as_ptr();
    let pc = codes.as_ptr();
    // SAFETY: AVX2+FMA guaranteed. Full chunks load 8 f32 from `pq` and 8 i8
    // from `pc` while `i + 8 <= n`; scalar tail indices `i < n` are in bounds.
    unsafe {
        let vscale = _mm256_set1_ps(scale);
        let mut acc = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 8 <= n {
            let vq = _mm256_loadu_ps(pq.add(i));
            let vs = widen_i8(pc.add(i), vscale);
            let d = _mm256_sub_ps(vq, vs);
            acc = _mm256_fmadd_ps(d, d, acc);
            i += 8;
        }
        let mut tail = hsum256(acc);
        while i < n {
            let s = f32::from(*pc.add(i)) * scale;
            let d = *pq.add(i) - s;
            tail += d * d;
            i += 1;
        }
        tail
    }
}

/// Raw inner product `⟨query, i8-stored⟩` with per-vector `scale` (AVX2).
///
/// # Safety
/// Requires AVX2 + FMA. `codes.len() == query.len()` assumed.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_i8(query: &[f32], scale: f32, codes: &[i8]) -> f32 {
    debug_assert_eq!(codes.len(), query.len());
    let n = query.len();
    let pq = query.as_ptr();
    let pc = codes.as_ptr();
    // SAFETY: as in `l2_sq_i8`.
    unsafe {
        let vscale = _mm256_set1_ps(scale);
        let mut acc = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 8 <= n {
            let vq = _mm256_loadu_ps(pq.add(i));
            let vs = widen_i8(pc.add(i), vscale);
            acc = _mm256_fmadd_ps(vq, vs, acc);
            i += 8;
        }
        let mut acc_s = hsum256(acc);
        while i < n {
            acc_s += *pq.add(i) * (f32::from(*pc.add(i)) * scale);
            i += 1;
        }
        acc_s
    }
}

/// Single-pass cosine accumulators for `f32` query vs `i8` store (AVX2).
///
/// # Safety
/// Requires AVX2 + FMA. `codes.len() == query.len()` assumed.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn cosine_parts_i8(query: &[f32], scale: f32, codes: &[i8]) -> (f32, f32, f32) {
    debug_assert_eq!(codes.len(), query.len());
    let n = query.len();
    let pq = query.as_ptr();
    let pc = codes.as_ptr();
    // SAFETY: as in `l2_sq_i8`.
    unsafe {
        let vscale = _mm256_set1_ps(scale);
        let mut dot_acc = _mm256_setzero_ps();
        let mut nq_acc = _mm256_setzero_ps();
        let mut ns_acc = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 8 <= n {
            let vq = _mm256_loadu_ps(pq.add(i));
            let vs = widen_i8(pc.add(i), vscale);
            dot_acc = _mm256_fmadd_ps(vq, vs, dot_acc);
            nq_acc = _mm256_fmadd_ps(vq, vq, nq_acc);
            ns_acc = _mm256_fmadd_ps(vs, vs, ns_acc);
            i += 8;
        }
        let mut d = hsum256(dot_acc);
        let mut nq = hsum256(nq_acc);
        let mut ns = hsum256(ns_acc);
        while i < n {
            let q = *pq.add(i);
            let s = f32::from(*pc.add(i)) * scale;
            d += q * s;
            nq += q * q;
            ns += s * s;
            i += 1;
        }
        (d, nq, ns)
    }
}
