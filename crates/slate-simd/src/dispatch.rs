//! Runtime CPU-feature dispatch.
//!
//! The best available kernel tier is detected **once** (first use) and the
//! resolved `fn` pointers are cached in `OnceLock`s. After warm-up, each call is
//! a plain indirect call with no feature-detection overhead.
//!
//! Selection order (best first):
//! - x86-64: AVX-512F → AVX2+FMA → scalar
//! - aarch64: NEON → scalar
//! - other:   scalar
//!
//! Public kernels in [`crate`] route through here. Tests can also force a tier
//! to compare a specific implementation against the scalar oracle.

use std::sync::OnceLock;

use crate::scalar;
#[cfg(target_arch = "x86_64")]
use crate::{avx2, avx512};
#[cfg(target_arch = "aarch64")]
use crate::neon;

/// Which implementation tier the dispatcher selected for this CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Portable scalar fallback (always available).
    Scalar,
    /// AVX2 + FMA (x86-64).
    Avx2,
    /// AVX-512F (x86-64).
    Avx512,
    /// ARM NEON (aarch64).
    Neon,
}

impl Tier {
    /// Human-readable tier name (for logs / CLI `bench` output).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Tier::Scalar => "scalar",
            Tier::Avx2 => "avx2",
            Tier::Avx512 => "avx512",
            Tier::Neon => "neon",
        }
    }
}

/// Detect the best tier supported by the current CPU.
///
/// Uses compile-time `cfg` to pick the candidate set per architecture and
/// runtime `is_*_feature_detected!` macros to confirm availability. The result
/// is cached by [`active_tier`].
#[must_use]
pub fn detect_tier() -> Tier {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return Tier::Avx512;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return Tier::Avx2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return Tier::Neon;
        }
    }
    Tier::Scalar
}

/// The cached active tier for this process.
#[must_use]
pub fn active_tier() -> Tier {
    static TIER: OnceLock<Tier> = OnceLock::new();
    *TIER.get_or_init(detect_tier)
}

/// Signature shared by the L2² / dot kernels.
pub type BinaryKernel = fn(&[f32], &[f32]) -> f32;

// --- Thin safe wrappers that assert the target feature before the unsafe call.
// Each wrapper is only ever installed as the active pointer when `detect_tier`
// confirmed the corresponding feature, so the precondition holds.

#[cfg(target_arch = "x86_64")]
fn l2_sq_avx2(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: only selected when AVX2+FMA detected (see `resolve_l2_sq`).
    unsafe { avx2::l2_sq(a, b) }
}
#[cfg(target_arch = "x86_64")]
fn l2_sq_avx512(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: only selected when AVX-512F detected.
    unsafe { avx512::l2_sq(a, b) }
}
#[cfg(target_arch = "x86_64")]
fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: only selected when AVX2+FMA detected.
    unsafe { avx2::dot(a, b) }
}
#[cfg(target_arch = "x86_64")]
fn dot_avx512(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: only selected when AVX-512F detected.
    unsafe { avx512::dot(a, b) }
}

#[cfg(target_arch = "aarch64")]
fn l2_sq_neon(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: only selected when NEON detected.
    unsafe { neon::l2_sq(a, b) }
}
#[cfg(target_arch = "aarch64")]
fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: only selected when NEON detected.
    unsafe { neon::dot(a, b) }
}

fn resolve_l2_sq() -> BinaryKernel {
    match active_tier() {
        #[cfg(target_arch = "x86_64")]
        Tier::Avx512 => l2_sq_avx512,
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 => l2_sq_avx2,
        #[cfg(target_arch = "aarch64")]
        Tier::Neon => l2_sq_neon,
        _ => scalar::l2_sq,
    }
}

fn resolve_dot() -> BinaryKernel {
    match active_tier() {
        #[cfg(target_arch = "x86_64")]
        Tier::Avx512 => dot_avx512,
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 => dot_avx2,
        #[cfg(target_arch = "aarch64")]
        Tier::Neon => dot_neon,
        _ => scalar::dot,
    }
}

/// Cached best `l2_sq` kernel.
#[must_use]
pub fn l2_sq_kernel() -> BinaryKernel {
    static K: OnceLock<BinaryKernel> = OnceLock::new();
    *K.get_or_init(resolve_l2_sq)
}

/// Cached best `dot` kernel.
#[must_use]
pub fn dot_kernel() -> BinaryKernel {
    static K: OnceLock<BinaryKernel> = OnceLock::new();
    *K.get_or_init(resolve_dot)
}

/// Compute the raw cosine accumulators `(dot, ‖a‖², ‖b‖²)` with the best tier.
///
/// Cosine returns three values, so it does not share [`BinaryKernel`]. We branch
/// on the cached tier each call; the branch is trivially predictable.
#[must_use]
pub fn cosine_parts(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    match active_tier() {
        #[cfg(target_arch = "x86_64")]
        Tier::Avx512 => unsafe { avx512::cosine_parts(a, b) }, // SAFETY: tier-gated
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 => unsafe { avx2::cosine_parts(a, b) }, // SAFETY: tier-gated
        #[cfg(target_arch = "aarch64")]
        Tier::Neon => unsafe { neon::cosine_parts(a, b) }, // SAFETY: tier-gated
        _ => {
            // Scalar single-pass parts.
            let mut d = 0.0f32;
            let mut na = 0.0f32;
            let mut nb = 0.0f32;
            for i in 0..a.len() {
                d += a[i] * b[i];
                na += a[i] * a[i];
                nb += b[i] * b[i];
            }
            (d, na, nb)
        }
    }
}
