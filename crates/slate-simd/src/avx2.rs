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
