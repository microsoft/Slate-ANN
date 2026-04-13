//! Seeded Lloyd's k-means over a set of equal-length vectors.
//!
//! Used per PQ subspace to learn that subspace's centroid codebook. The
//! implementation is intentionally small and deterministic:
//!
//! * **Initialization** is k-means++ (D² weighting) driven by the seeded
//!   [`SplitMix64`] so a given (data, k, seed) triple always yields the same
//!   centroids, on any architecture.
//! * **Assignment** uses squared-L2 distance (the natural PQ quantization
//!   metric; the index's own metric governs the *exact* tier, not the codebook).
//! * **Empty clusters** are re-seeded to the point farthest from its assigned
//!   centroid, which keeps all `k` codes useful.
//!
//! Distances here are computed with [`slate_simd::l2_sq`] so the codebook
//! training benefits from the same SIMD kernels as search.

use crate::rng::SplitMix64;
use slate_core::Result;

/// Result of training k-means on one subspace: the centroids, flattened
/// row-major as `k * sub_dim` floats.
#[derive(Debug, Clone)]
pub struct KMeansResult {
    /// Number of centroids actually produced (`<= requested k`, clamped to the
    /// number of distinct-ish points available).
    pub k: usize,
    /// Sub-vector dimensionality.
    pub sub_dim: usize,
    /// Centroids, row-major: centroid `c` occupies `[c*sub_dim .. (c+1)*sub_dim]`.
    pub centroids: Vec<f32>,
}

impl KMeansResult {
    /// Borrow centroid `c` as a slice.
    #[inline]
    pub fn centroid(&self, c: usize) -> &[f32] {
        &self.centroids[c * self.sub_dim..(c + 1) * self.sub_dim]
    }
}

/// Train k-means on `points` (each `sub_dim` long, row-major in `points`).
///
/// Produces up to `k` centroids. If there are fewer than `k` points, every
/// point becomes its own centroid (a degenerate but correct codebook). Runs at
/// most `max_iters` Lloyd iterations, stopping early when no point changes
/// assignment.
///
/// `seed` makes initialization deterministic.
///
/// # Errors
///
/// Propagates distance-kernel errors (only on dimension mismatch, which cannot
/// occur here since all slices are `sub_dim`).
pub fn train(
    points: &[f32],
    sub_dim: usize,
    k: usize,
    max_iters: usize,
    seed: u64,
) -> Result<KMeansResult> {
    debug_assert!(sub_dim > 0);
    let n = if sub_dim == 0 { 0 } else { points.len() / sub_dim };

    // Fewer points than requested centroids: each point is its own centroid.
    if n <= k {
        return Ok(KMeansResult {
            k: n,
            sub_dim,
            centroids: points[..n * sub_dim].to_vec(),
        });
    }

    let mut rng = SplitMix64::new(seed);
    let mut centroids = kmeans_plus_plus_init(points, sub_dim, k, n, &mut rng)?;

    let mut assignment = vec![0usize; n];
    for _ in 0..max_iters.max(1) {
        let changed = assign_all(points, sub_dim, &centroids, k, &mut assignment)?;
        recompute_centroids(points, sub_dim, k, &assignment, &mut centroids, &mut rng)?;
        if !changed {
            break;
        }
    }

    Ok(KMeansResult {
        k,
        sub_dim,
        centroids,
    })
}

/// k-means++ initialization: first centroid uniformly at random, each subsequent
/// centroid chosen with probability proportional to squared distance from the
/// nearest already-chosen centroid.
fn kmeans_plus_plus_init(
    points: &[f32],
    sub_dim: usize,
    k: usize,
    n: usize,
    rng: &mut SplitMix64,
) -> Result<Vec<f32>> {
    let mut centroids = Vec::with_capacity(k * sub_dim);

    // First centroid: uniform random point.
    let first = rng.next_below(n);
    centroids.extend_from_slice(point(points, sub_dim, first));

    // Running nearest-centroid squared distance per point.
    let mut d2 = vec![f32::INFINITY; n];

    for c in 1..k {
        // Update d2 against the most recently added centroid.
        let last = &centroids[(c - 1) * sub_dim..c * sub_dim];
        let mut sum = 0.0f64;
        for (i, d2i) in d2.iter_mut().enumerate() {
            let dist = slate_simd::l2_sq(point(points, sub_dim, i), last)?;
            if dist < *d2i {
                *d2i = dist;
            }
            sum += *d2i as f64;
        }

        // Choose the next centroid weighted by d2. If all points coincide
        // (sum == 0), fall back to a uniform pick.
        let next = if sum <= 0.0 {
            rng.next_below(n)
        } else {
            let target = rng.next_f64() * sum;
            let mut acc = 0.0f64;
            let mut chosen = n - 1;
            for (i, &d2i) in d2.iter().enumerate() {
                acc += d2i as f64;
                if acc >= target {
                    chosen = i;
                    break;
                }
            }
            chosen
        };
        centroids.extend_from_slice(point(points, sub_dim, next));
    }

    Ok(centroids)
}

/// Assign every point to its nearest centroid. Returns whether any assignment
/// changed from the incoming `assignment`.
fn assign_all(
    points: &[f32],
    sub_dim: usize,
    centroids: &[f32],
    k: usize,
    assignment: &mut [usize],
) -> Result<bool> {
    let mut changed = false;
    for (i, assign_i) in assignment.iter_mut().enumerate() {
        let p = point(points, sub_dim, i);
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for c in 0..k {
            let cc = &centroids[c * sub_dim..(c + 1) * sub_dim];
            let d = slate_simd::l2_sq(p, cc)?;
            if d < best_d {
                best_d = d;
                best = c;
            }
        }
        if *assign_i != best {
            *assign_i = best;
            changed = true;
        }
    }
    Ok(changed)
}

/// Recompute each centroid as the mean of its assigned points. Empty clusters
/// are re-seeded to the point currently farthest from its own centroid, so all
/// `k` codes stay in use.
fn recompute_centroids(
    points: &[f32],
    sub_dim: usize,
    k: usize,
    assignment: &[usize],
    centroids: &mut [f32],
    rng: &mut SplitMix64,
) -> Result<()> {
    let mut sums = vec![0.0f64; k * sub_dim];
    let mut counts = vec![0usize; k];

    for (i, &c) in assignment.iter().enumerate() {
        counts[c] += 1;
        let p = point(points, sub_dim, i);
        let base = c * sub_dim;
        for (j, &val) in p.iter().enumerate() {
            sums[base + j] += val as f64;
        }
    }

    for c in 0..k {
        if counts[c] == 0 {
            // Re-seed the empty cluster to a far-away point to avoid a dead code.
            let reseed = farthest_point(points, sub_dim, centroids, assignment, rng)?;
            let src = point(points, sub_dim, reseed).to_vec();
            centroids[c * sub_dim..(c + 1) * sub_dim].copy_from_slice(&src);
        } else {
            let inv = 1.0 / counts[c] as f64;
            let base = c * sub_dim;
            for j in 0..sub_dim {
                centroids[base + j] = (sums[base + j] * inv) as f32;
            }
        }
    }
    Ok(())
}

/// Find the point farthest from its assigned centroid (used to re-seed empty
/// clusters). Falls back to a random point if distances are degenerate.
fn farthest_point(
    points: &[f32],
    sub_dim: usize,
    centroids: &[f32],
    assignment: &[usize],
    rng: &mut SplitMix64,
) -> Result<usize> {
    let mut best = rng.next_below(assignment.len());
    let mut best_d = -1.0f32;
    for (i, &c) in assignment.iter().enumerate() {
        let cc = &centroids[c * sub_dim..(c + 1) * sub_dim];
        let d = slate_simd::l2_sq(point(points, sub_dim, i), cc)?;
        if d > best_d {
            best_d = d;
            best = i;
        }
    }
    Ok(best)
}

/// Borrow point `i` (length `sub_dim`).
#[inline]
fn point(points: &[f32], sub_dim: usize, i: usize) -> &[f32] {
    &points[i * sub_dim..(i + 1) * sub_dim]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fewer_points_than_k_makes_each_its_own_centroid() {
        // 3 points, ask for 8 centroids.
        let pts = vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0];
        let res = train(&pts, 2, 8, 10, 1).unwrap();
        assert_eq!(res.k, 3);
        assert_eq!(res.centroids, pts);
    }

    #[test]
    fn recovers_two_well_separated_clusters() {
        // Two tight blobs far apart; k=2 should place one centroid in each.
        let mut pts = Vec::new();
        for _ in 0..50 {
            pts.push(0.0);
            pts.push(0.0);
        }
        for _ in 0..50 {
            pts.push(100.0);
            pts.push(100.0);
        }
        let res = train(&pts, 2, 2, 25, 7).unwrap();
        assert_eq!(res.k, 2);
        // One centroid near origin, one near (100,100), in some order.
        let c0 = res.centroid(0);
        let c1 = res.centroid(1);
        let near_origin = |c: &[f32]| c[0].abs() < 1.0 && c[1].abs() < 1.0;
        let near_100 = |c: &[f32]| (c[0] - 100.0).abs() < 1.0 && (c[1] - 100.0).abs() < 1.0;
        assert!(
            (near_origin(c0) && near_100(c1)) || (near_origin(c1) && near_100(c0)),
            "centroids did not separate: {c0:?} {c1:?}"
        );
    }

    #[test]
    fn is_deterministic_for_a_seed() {
        let mut pts = Vec::new();
        let mut r = SplitMix64::new(0);
        for _ in 0..200 {
            pts.push(r.next_f64() as f32);
            pts.push(r.next_f64() as f32);
            pts.push(r.next_f64() as f32);
        }
        let a = train(&pts, 3, 16, 20, 99).unwrap();
        let b = train(&pts, 3, 16, 20, 99).unwrap();
        assert_eq!(a.centroids, b.centroids);
    }
}
