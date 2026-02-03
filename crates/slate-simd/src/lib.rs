//! # slate-simd
//!
//! SIMD distance kernels for Slate-ANN with runtime CPU feature dispatch.
//!
//! Provides L2², inner-product, and cosine distance over `f32`, `f16`, and
//! `i8` vectors with four implementation tiers selected at runtime:
//! AVX-512, AVX2, ARM NEON, and a portable scalar fallback (also the
//! correctness oracle for the vectorized paths).
//!
//! Populated in Phase 1.

#![doc(html_root_url = "https://docs.rs/slate-simd")]
