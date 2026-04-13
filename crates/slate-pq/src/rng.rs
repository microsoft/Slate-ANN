//! Dependency-free seeded PRNG (SplitMix64), used for deterministic k-means
//! centroid initialization.
//!
//! This is a deliberate duplicate of the generator in `slate-graph::rng`: the
//! two crates must not depend on each other, and the generator is a dozen lines
//! of well-known constants. Reproducibility (byte-identical codebooks per seed)
//! matters more than sharing code here, and it keeps `slate-pq` free of a `rand`
//! dependency.

/// A fast, deterministic [SplitMix64] pseudo-random generator.
///
/// Not cryptographically secure. Suitable for reproducible centroid seeding and
/// tie-breaking, where determinism across runs and architectures is the goal.
///
/// [SplitMix64]: https://prng.di.unimi.it/splitmix64.c
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Golden-ratio odd increment (the SplitMix64 gamma).
    const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

    /// Create a generator from any seed (including 0).
    #[inline]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next 64-bit value.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(Self::GAMMA);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Next `f64` uniformly distributed in `[0, 1)` (53-bit mantissa).
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        // Top 53 bits scaled by 2^-53.
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform integer in `[0, n)` for `n > 0` (modulo-biased, acceptable for
    /// centroid selection).
    #[inline]
    pub fn next_below(&mut self, n: usize) -> usize {
        debug_assert!(n > 0, "next_below requires n > 0");
        (self.next_u64() % n as u64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_deterministic_for_a_seed() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = SplitMix64::new(1);
        let mut b = SplitMix64::new(2);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn f64_is_in_unit_interval() {
        let mut r = SplitMix64::new(7);
        for _ in 0..1000 {
            let x = r.next_f64();
            assert!((0.0..1.0).contains(&x), "x={x}");
        }
    }

    #[test]
    fn next_below_is_in_range() {
        let mut r = SplitMix64::new(123);
        for _ in 0..1000 {
            let x = r.next_below(10);
            assert!(x < 10, "x={x}");
        }
    }
}
