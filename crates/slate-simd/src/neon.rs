//! ARM NEON kernels (aarch64).
//!
//! 4 `f32` lanes per iteration. Targets low-power ARM devices (Raspberry Pi and
//! similar SBCs), which are archetypal deployment targets for this engine. NEON
//! is mandatory on aarch64, so this tier is always available there — but we
//! still gate selection through [`crate::dispatch`] for symmetry with x86.
//!
//! # Safety
//! Functions are `#[target_feature(enable = "neon")]`. On aarch64 NEON is
//! baseline, but we keep the gate explicit. Intrinsic calls are wrapped in an
//! explicit `unsafe` block (the crate denies `unsafe_op_in_unsafe_fn`).

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

/// L2² distance over `f32` using NEON.
///
/// # Safety
/// Requires NEON (baseline on aarch64). `a.len() == b.len()` assumed.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    // SAFETY: NEON guaranteed by target_feature precondition. Only full 4-lane
    // chunks loaded while `i + 4 <= n`; `vld1q` is unaligned-safe. Scalar tail
    // indices `i < n` are in bounds.
    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        let mut i = 0usize;
        while i + 4 <= n {
            let va = vld1q_f32(pa.add(i));
            let vb = vld1q_f32(pb.add(i));
            let d = vsubq_f32(va, vb);
            acc = vfmaq_f32(acc, d, d); // acc += d*d
            i += 4;
        }
        let mut tail = vaddvq_f32(acc); // horizontal add of 4 lanes
        while i < n {
            let d = *pa.add(i) - *pb.add(i);
            tail += d * d;
            i += 1;
        }
        tail
    }
}

/// Raw inner product `⟨a,b⟩` over `f32` using NEON.
///
/// # Safety
/// Requires NEON (baseline on aarch64). `a.len() == b.len()` assumed.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    // SAFETY: NEON guaranteed; only full 4-lane chunks under the bound; scalar
    // tail indices in bounds.
    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        let mut i = 0usize;
        while i + 4 <= n {
            let va = vld1q_f32(pa.add(i));
            let vb = vld1q_f32(pb.add(i));
            acc = vfmaq_f32(acc, va, vb);
            i += 4;
        }
        let mut acc_s = vaddvq_f32(acc);
        while i < n {
            acc_s += *pa.add(i) * *pb.add(i);
            i += 1;
        }
        acc_s
    }
}

/// Single-pass cosine accumulators `(⟨a,b⟩, ‖a‖², ‖b‖²)` using NEON.
///
/// # Safety
/// Requires NEON (baseline on aarch64). `a.len() == b.len()` assumed.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn cosine_parts(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    // SAFETY: NEON guaranteed; only full 4-lane chunks under the bound; scalar
    // tail indices in bounds.
    unsafe {
        let mut dot_acc = vdupq_n_f32(0.0);
        let mut na_acc = vdupq_n_f32(0.0);
        let mut nb_acc = vdupq_n_f32(0.0);
        let mut i = 0usize;
        while i + 4 <= n {
            let va = vld1q_f32(pa.add(i));
            let vb = vld1q_f32(pb.add(i));
            dot_acc = vfmaq_f32(dot_acc, va, vb);
            na_acc = vfmaq_f32(na_acc, va, va);
            nb_acc = vfmaq_f32(nb_acc, vb, vb);
            i += 4;
        }
        let mut d = vaddvq_f32(dot_acc);
        let mut na = vaddvq_f32(na_acc);
        let mut nb = vaddvq_f32(nb_acc);
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
