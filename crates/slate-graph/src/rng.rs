//! A tiny, dependency-free, deterministic pseudo-random generator.
//!
//! HNSW assigns each inserted node a level drawn from a geometric distribution,
//! which requires a source of uniform `f64` values. We deliberately avoid
//! pulling in the `rand` crate: the requirement is only for *reproducible*
//! randomness so that a given seed produces a byte-identical graph (essential
//! for deterministic tests, cross-architecture validation, and debuggable
//! builds), not for cryptographic quality.
//!
//! [`SplitMix64`] is the generator Java's `SplittableRandom` and the reference
//! `xoshiro` seeding routine use. It is a single 64-bit state advanced by a
//! fixed odd increment and finalized with a strong avalanche mix, giving
//! well-distributed output that passes BigCrush — far more than enough for
//! level assignment.

/// A `SplitMix64` pseudo-random generator: 64 bits of state, fast and
/// deterministic.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// The fixed odd increment (the 64-bit golden-ratio constant `2^64 / φ`).
    const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

    /// Create a generator from a seed. Any seed (including 0) is valid.
    #[inline]
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Advance the state and return the next 64-bit value.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(Self::GAMMA);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Return a uniform `f64` in the half-open interval `[0, 1)`.
    ///
    /// Uses the top 53 bits (the `f64` mantissa width) so every representable
    /// fraction is reachable and the distribution is uniform.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        // 53 high bits -> [0, 2^53), scaled by 2^-53 into [0, 1).
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_deterministic_for_a_seed() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = SplitMix64::new(1);
        let mut b = SplitMix64::new(2);
        // Overwhelmingly likely to differ within a few draws.
        let mut any_diff = false;
        for _ in 0..8 {
            if a.next_u64() != b.next_u64() {
                any_diff = true;
                break;
            }
        }
        assert!(any_diff);
    }

    #[test]
    fn f64_is_in_unit_interval() {
        let mut r = SplitMix64::new(12345);
        for _ in 0..100_000 {
            let x = r.next_f64();
            assert!((0.0..1.0).contains(&x), "out of range: {x}");
        }
    }

    #[test]
    fn f64_mean_is_roughly_half() {
        let mut r = SplitMix64::new(999);
        let n = 200_000;
        let sum: f64 = (0..n).map(|_| r.next_f64()).sum();
        let mean = sum / f64::from(n);
        // Loose bound; just guards against gross distribution bugs.
        assert!((mean - 0.5).abs() < 0.01, "mean was {mean}");
    }
}
