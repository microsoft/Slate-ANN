//! Micro-benchmarks for the dispatched distance kernels.
//!
//! Benches the public (runtime-dispatched) API across the three metrics at
//! representative embedding dimensions. On this dev box the active tier is
//! AVX2; the same harness will exercise AVX-512 or NEON on hardware that has
//! them, since dispatch is resolved at runtime.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use slate_simd::{active_tier, cosine, inner_product, l2_sq};

/// Deterministic pseudo-random vector (no rand dependency); values span a
/// bounded range so accumulation stays well-conditioned.
fn make_vec(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..len)
        .map(|_| {
            // xorshift64* step
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let bits = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
            // Map to [-10, 10).
            ((bits >> 40) as f32 / (1u64 << 24) as f32) * 20.0 - 10.0
        })
        .collect()
}

fn bench_distance(c: &mut Criterion) {
    // Common embedding widths: 128 (small), 384 (MiniLM), 768 (Contriever/BERT).
    const DIMS: [usize; 3] = [128, 384, 768];

    let mut group = c.benchmark_group(format!("distance/{}", active_tier().as_str()));
    for &dims in &DIMS {
        let a = make_vec(dims, 1);
        let b = make_vec(dims, 2);
        group.throughput(Throughput::Elements(dims as u64));

        group.bench_with_input(BenchmarkId::new("l2_sq", dims), &dims, |bn, _| {
            bn.iter(|| l2_sq(black_box(&a), black_box(&b)).unwrap());
        });
        group.bench_with_input(BenchmarkId::new("inner_product", dims), &dims, |bn, _| {
            bn.iter(|| inner_product(black_box(&a), black_box(&b)).unwrap());
        });
        group.bench_with_input(BenchmarkId::new("cosine", dims), &dims, |bn, _| {
            bn.iter(|| cosine(black_box(&a), black_box(&b)).unwrap());
        });
    }
    group.finish();
}

criterion_group!(benches, bench_distance);
criterion_main!(benches);
