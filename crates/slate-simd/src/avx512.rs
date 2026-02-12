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
