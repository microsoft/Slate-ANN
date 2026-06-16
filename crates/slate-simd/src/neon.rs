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

// --- Narrow-stored kernels (f32 query vs f16/i8 store) -----------------------
//
// The query is always `f32`; only the stored vector is narrow. Each iteration
// widens 4 stored elements to a `float32x4_t` and runs the identical sub/fmadd
// reduction as the `f32` kernels above. The `f16` widen uses `vcvt_f32_f16`,
// which requires the `fp16` feature; the dispatcher gates that separately
// (`is_aarch64_feature_detected!("fp16")`).

/// Widen 4 little-endian `f16` bytes at `p` to a `float32x4_t`.
///
/// # Safety
/// Requires NEON + FP16. `p` must point to at least 8 readable bytes.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16")]
#[inline]
unsafe fn widen_f16(p: *const u8) -> float32x4_t {
    // SAFETY: caller guarantees 8 readable bytes at `p` and NEON+FP16 support.
    // `vld1_u16` loads 4 u16 (unaligned-safe); reinterpret to f16x4; widen.
    unsafe {
        let bits = vld1_u16(p.cast::<u16>());
        vcvt_f32_f16(vreinterpret_f16_u16(bits))
    }
}

/// L2² distance between an `f32` query and an `f16`-stored vector (NEON + FP16).
///
/// `stored` holds `query.len()` little-endian `f16` elements (`2·len` bytes).
///
/// # Safety
/// Requires NEON + FP16. `stored.len() == 2 * query.len()` assumed.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16")]
pub unsafe fn l2_sq_f16(query: &[f32], stored: &[u8]) -> f32 {
    debug_assert_eq!(stored.len(), 2 * query.len());
    let n = query.len();
    let pq = query.as_ptr();
    let ps = stored.as_ptr();
    // SAFETY: NEON+FP16 guaranteed. Full 4-lane chunks load 4 f32 from `pq` and
    // 8 bytes (4 f16) from `ps` while `i + 4 <= n`; scalar tail reads f16 at
    // byte offset `2*i` (< stored.len()) and `query[i]`.
    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        let mut i = 0usize;
        while i + 4 <= n {
            let vq = vld1q_f32(pq.add(i));
            let vs = widen_f16(ps.add(2 * i));
            let d = vsubq_f32(vq, vs);
            acc = vfmaq_f32(acc, d, d);
            i += 4;
        }
        let mut tail = vaddvq_f32(acc);
        while i < n {
            let s = f16_at(ps, i);
            let d = *pq.add(i) - s;
            tail += d * d;
            i += 1;
        }
        tail
    }
}

/// Raw inner product `⟨query, f16-stored⟩` (NEON + FP16).
///
/// # Safety
/// Requires NEON + FP16. `stored.len() == 2 * query.len()` assumed.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16")]
pub unsafe fn dot_f16(query: &[f32], stored: &[u8]) -> f32 {
    debug_assert_eq!(stored.len(), 2 * query.len());
    let n = query.len();
    let pq = query.as_ptr();
    let ps = stored.as_ptr();
    // SAFETY: as in `l2_sq_f16`.
    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        let mut i = 0usize;
        while i + 4 <= n {
            let vq = vld1q_f32(pq.add(i));
            let vs = widen_f16(ps.add(2 * i));
            acc = vfmaq_f32(acc, vq, vs);
            i += 4;
        }
        let mut acc_s = vaddvq_f32(acc);
        while i < n {
            acc_s += *pq.add(i) * f16_at(ps, i);
            i += 1;
        }
        acc_s
    }
}

/// Single-pass cosine accumulators for `f32` query vs `f16` store (NEON + FP16).
///
/// # Safety
/// Requires NEON + FP16. `stored.len() == 2 * query.len()` assumed.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16")]
pub unsafe fn cosine_parts_f16(query: &[f32], stored: &[u8]) -> (f32, f32, f32) {
    debug_assert_eq!(stored.len(), 2 * query.len());
    let n = query.len();
    let pq = query.as_ptr();
    let ps = stored.as_ptr();
    // SAFETY: as in `l2_sq_f16`.
    unsafe {
        let mut dot_acc = vdupq_n_f32(0.0);
        let mut nq_acc = vdupq_n_f32(0.0);
        let mut ns_acc = vdupq_n_f32(0.0);
        let mut i = 0usize;
        while i + 4 <= n {
            let vq = vld1q_f32(pq.add(i));
            let vs = widen_f16(ps.add(2 * i));
            dot_acc = vfmaq_f32(dot_acc, vq, vs);
            nq_acc = vfmaq_f32(nq_acc, vq, vq);
            ns_acc = vfmaq_f32(ns_acc, vs, vs);
            i += 4;
        }
        let mut d = vaddvq_f32(dot_acc);
        let mut nq = vaddvq_f32(nq_acc);
        let mut ns = vaddvq_f32(ns_acc);
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
/// `p` must point to at least `2*(i+1)` readable bytes.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn f16_at(p: *const u8, i: usize) -> f32 {
    // SAFETY: caller guarantees `p.add(2*i)` and `p.add(2*i+1)` are readable.
    let bits = unsafe { u16::from(*p.add(2 * i)) | (u16::from(*p.add(2 * i + 1)) << 8) };
    half::f16::from_bits(bits).to_f32()
}

/// Widen 4 `i8` codes at `p` to a `float32x4_t`, scaled by `scale`.
///
/// # Safety
/// Requires NEON. `p` must point to at least 4 readable bytes.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn widen_i8(p: *const i8, scale: f32) -> float32x4_t {
    // SAFETY: caller guarantees 8 readable bytes at `p` (vld1_s8 loads 8) and
    // NEON support; only the low 4 lanes are used. Widen s8->s16->s32->f32.
    unsafe {
        let bytes = vld1_s8(p);
        let s16 = vmovl_s8(bytes); // 8x i16
        let s32 = vmovl_s16(vget_low_s16(s16)); // low 4x i32
        let floats = vcvtq_f32_s32(s32);
        vmulq_n_f32(floats, scale)
    }
}

/// L2² distance between an `f32` query and an `i8`-stored vector (NEON).
///
/// # Safety
/// Requires NEON. `codes.len() == query.len()` assumed and `codes` has at least
/// 4 readable bytes past each chunk (the store always pads the slot footprint).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn l2_sq_i8(query: &[f32], scale: f32, codes: &[i8]) -> f32 {
    debug_assert_eq!(codes.len(), query.len());
    let n = query.len();
    let pq = query.as_ptr();
    let pc = codes.as_ptr();
    // SAFETY: NEON guaranteed. Full 4-lane chunks load 4 f32 from `pq`; `widen_i8`
    // reads 8 bytes via vld1_s8 but only the chunks where `i + 4 <= n` use it and
    // for the final partial vector the scalar tail is used instead.
    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        let mut i = 0usize;
        while i + 8 <= n {
            let vq = vld1q_f32(pq.add(i));
            let vs = widen_i8(pc.add(i), scale);
            let d = vsubq_f32(vq, vs);
            acc = vfmaq_f32(acc, d, d);
            i += 4;
        }
        let mut tail = vaddvq_f32(acc);
        while i < n {
            let s = f32::from(*pc.add(i)) * scale;
            let d = *pq.add(i) - s;
            tail += d * d;
            i += 1;
        }
        tail
    }
}

/// Raw inner product `⟨query, i8-stored⟩` with per-vector `scale` (NEON).
///
/// # Safety
/// Requires NEON. `codes.len() == query.len()` assumed.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn dot_i8(query: &[f32], scale: f32, codes: &[i8]) -> f32 {
    debug_assert_eq!(codes.len(), query.len());
    let n = query.len();
    let pq = query.as_ptr();
    let pc = codes.as_ptr();
    // SAFETY: as in `l2_sq_i8`.
    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        let mut i = 0usize;
        while i + 8 <= n {
            let vq = vld1q_f32(pq.add(i));
            let vs = widen_i8(pc.add(i), scale);
            acc = vfmaq_f32(acc, vq, vs);
            i += 4;
        }
        let mut acc_s = vaddvq_f32(acc);
        while i < n {
            acc_s += *pq.add(i) * (f32::from(*pc.add(i)) * scale);
            i += 1;
        }
        acc_s
    }
}

/// Single-pass cosine accumulators for `f32` query vs `i8` store (NEON).
///
/// # Safety
/// Requires NEON. `codes.len() == query.len()` assumed.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn cosine_parts_i8(query: &[f32], scale: f32, codes: &[i8]) -> (f32, f32, f32) {
    debug_assert_eq!(codes.len(), query.len());
    let n = query.len();
    let pq = query.as_ptr();
    let pc = codes.as_ptr();
    // SAFETY: as in `l2_sq_i8`.
    unsafe {
        let mut dot_acc = vdupq_n_f32(0.0);
        let mut nq_acc = vdupq_n_f32(0.0);
        let mut ns_acc = vdupq_n_f32(0.0);
        let mut i = 0usize;
        while i + 8 <= n {
            let vq = vld1q_f32(pq.add(i));
            let vs = widen_i8(pc.add(i), scale);
            dot_acc = vfmaq_f32(dot_acc, vq, vs);
            nq_acc = vfmaq_f32(nq_acc, vq, vq);
            ns_acc = vfmaq_f32(ns_acc, vs, vs);
            i += 4;
        }
        let mut d = vaddvq_f32(dot_acc);
        let mut nq = vaddvq_f32(nq_acc);
        let mut ns = vaddvq_f32(ns_acc);
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
