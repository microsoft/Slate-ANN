//! The product-quantization codebook: per-subspace centroids plus encode.
//!
//! A [`PqCodebook`] holds `M = num_subquantizers` independent sub-codebooks,
//! each with up to `2^bits_per_code` centroids of length `sub_dim = dims / M`.
//! Training runs k-means separately on each subspace; encoding maps a full
//! vector to its `M`-byte code (one nearest-centroid id per subspace).
//!
//! The codes are the **RAM-resident approximate tier**: ~`M` bytes per vector
//! versus `dims * 4` bytes for the exact on-disk vector, the asymmetry that lets
//! the engine keep an approximate distance for *every* vector in memory while
//! the exact vectors stay on disk.

use serde::{Deserialize, Serialize};
use slate_core::{Error, PqParams, Result};

use crate::kmeans;

/// A trained product-quantization codebook.
///
/// Serializable so it can be persisted alongside the index (the build pipeline
/// in a later phase writes it into the index metadata). Codes produced by
/// [`encode`](Self::encode) are `code_len()` bytes long.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PqCodebook {
    /// Full vector dimensionality.
    dims: usize,
    /// Number of subspaces (`M`).
    num_subspaces: usize,
    /// Sub-vector dimensionality (`dims / M`).
    sub_dim: usize,
    /// Centroid count per subspace requested (`2^bits`). The actual count for a
    /// given subspace may be smaller if the training set had fewer points; see
    /// [`subspace_k`](Self::subspace_k).
    centroids_per_subspace: usize,
    /// Per-subspace centroid counts actually produced (`len == num_subspaces`).
    ks: Vec<usize>,
    /// Per-subspace centroids, row-major within each subspace, subspaces
    /// concatenated. Subspace `s` occupies `centroids[offsets[s] ..
    /// offsets[s] + ks[s] * sub_dim]`.
    centroids: Vec<f32>,
    /// Start offset (in floats) of each subspace's centroid block.
    offsets: Vec<usize>,
}

impl PqCodebook {
    /// Train a codebook over `vectors` (row-major, `dims` each).
    ///
    /// `params` supplies `num_subquantizers` and `bits_per_code`. `max_iters`
    /// bounds k-means per subspace; `seed` makes training deterministic (each
    /// subspace is seeded from `seed` mixed with its index).
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidConfig`] if `dims` is not divisible by
    /// `num_subquantizers` or the inputs are empty/ragged, and propagates
    /// k-means errors.
    pub fn train(
        vectors: &[f32],
        dims: usize,
        params: &PqParams,
        max_iters: usize,
        seed: u64,
    ) -> Result<Self> {
        params.validate()?;
        if dims == 0 {
            return Err(Error::invalid_config("dims must be > 0"));
        }
        let m = params.num_subquantizers;
        if !dims.is_multiple_of(m) {
            return Err(Error::invalid_config(format!(
                "dims ({dims}) must be divisible by num_subquantizers ({m})"
            )));
        }
        if vectors.is_empty() {
            return Err(Error::invalid_config(
                "cannot train a PQ codebook on zero vectors",
            ));
        }
        if !vectors.len().is_multiple_of(dims) {
            return Err(Error::invalid_config(format!(
                "vectors length ({}) is not a multiple of dims ({dims})",
                vectors.len()
            )));
        }

        let n = vectors.len() / dims;
        let sub_dim = dims / m;
        let centroids_per_subspace = params.centroids_per_subspace();

        let mut centroids: Vec<f32> = Vec::new();
        let mut offsets: Vec<usize> = Vec::with_capacity(m);
        let mut ks: Vec<usize> = Vec::with_capacity(m);

        // Scratch holding one subspace's column-extracted sub-vectors.
        let mut sub_points = vec![0.0f32; n * sub_dim];

        for s in 0..m {
            // Gather subspace `s`: the sub_dim-wide slice of every vector.
            let col_start = s * sub_dim;
            for (i, chunk) in sub_points.chunks_exact_mut(sub_dim).enumerate() {
                let src = &vectors[i * dims + col_start..i * dims + col_start + sub_dim];
                chunk.copy_from_slice(src);
            }

            // Mix the subspace index into the seed so subspaces differ.
            let sub_seed = seed ^ ((s as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let res = kmeans::train(
                &sub_points,
                sub_dim,
                centroids_per_subspace,
                max_iters,
                sub_seed,
            )?;

            offsets.push(centroids.len());
            ks.push(res.k);
            centroids.extend_from_slice(&res.centroids);
        }

        Ok(Self {
            dims,
            num_subspaces: m,
            sub_dim,
            centroids_per_subspace,
            ks,
            centroids,
            offsets,
        })
    }

    /// Full vector dimensionality.
    #[inline]
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Number of subspaces (`M`).
    #[inline]
    pub fn num_subspaces(&self) -> usize {
        self.num_subspaces
    }

    /// Sub-vector dimensionality (`dims / M`).
    #[inline]
    pub fn sub_dim(&self) -> usize {
        self.sub_dim
    }

    /// Length in bytes of an encoded code (one byte per subspace).
    ///
    /// `bits_per_code` is fixed at 8 in this phase, so a code is exactly
    /// `num_subspaces` bytes.
    #[inline]
    pub fn code_len(&self) -> usize {
        self.num_subspaces
    }

    /// Centroids requested per subspace (`2^bits`).
    #[inline]
    pub fn centroids_per_subspace(&self) -> usize {
        self.centroids_per_subspace
    }

    /// Number of centroids actually produced for subspace `s`.
    #[inline]
    pub fn subspace_k(&self, s: usize) -> usize {
        self.ks[s]
    }

    /// Borrow centroid `c` of subspace `s` (length `sub_dim`).
    #[inline]
    pub fn centroid(&self, s: usize, c: usize) -> &[f32] {
        let base = self.offsets[s] + c * self.sub_dim;
        &self.centroids[base..base + self.sub_dim]
    }

    /// Encode a full vector into its `code_len()`-byte PQ code, writing into
    /// `out`.
    ///
    /// Each output byte is the id of the nearest centroid in the corresponding
    /// subspace. `bits_per_code == 8` guarantees ids fit in a `u8`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] if `vector.len() != dims` or
    /// `out.len() != code_len()`, and propagates distance-kernel errors.
    pub fn encode_into(&self, vector: &[f32], out: &mut [u8]) -> Result<()> {
        if vector.len() != self.dims {
            return Err(Error::DimensionMismatch {
                expected: self.dims,
                got: vector.len(),
            });
        }
        if out.len() != self.code_len() {
            return Err(Error::DimensionMismatch {
                expected: self.code_len(),
                got: out.len(),
            });
        }
        for (s, out_s) in out.iter_mut().enumerate() {
            let col = s * self.sub_dim;
            let sub = &vector[col..col + self.sub_dim];
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for c in 0..self.ks[s] {
                let d = slate_simd::l2_sq(sub, self.centroid(s, c))?;
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            *out_s = best as u8;
        }
        Ok(())
    }

    /// Encode a full vector into a freshly allocated code.
    ///
    /// # Errors
    ///
    /// See [`encode_into`](Self::encode_into).
    pub fn encode(&self, vector: &[f32]) -> Result<Vec<u8>> {
        let mut out = vec![0u8; self.code_len()];
        self.encode_into(vector, &mut out)?;
        Ok(out)
    }

    /// Encode every vector in a row-major batch into a flat code buffer.
    ///
    /// Returns a `Vec<u8>` of length `count * code_len()`, code `i` occupying
    /// `[i*code_len .. (i+1)*code_len]`.
    ///
    /// # Errors
    ///
    /// See [`encode_into`](Self::encode_into); also errors if `vectors.len()` is
    /// not a multiple of `dims`.
    pub fn encode_batch(&self, vectors: &[f32]) -> Result<Vec<u8>> {
        if !vectors.len().is_multiple_of(self.dims) {
            return Err(Error::invalid_config(format!(
                "vectors length ({}) is not a multiple of dims ({})",
                vectors.len(),
                self.dims
            )));
        }
        let count = vectors.len() / self.dims;
        let code_len = self.code_len();
        let mut codes = vec![0u8; count * code_len];
        for i in 0..count {
            let v = &vectors[i * self.dims..(i + 1) * self.dims];
            let out = &mut codes[i * code_len..(i + 1) * code_len];
            self.encode_into(v, out)?;
        }
        Ok(codes)
    }

    /// Reconstruct an approximate vector from its code (concatenated centroids),
    /// writing into `out` (length `dims`).
    ///
    /// Mainly useful for tests and diagnostics; the search path never needs the
    /// reconstruction because ADC works directly on codes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] on a wrong-length `code` or `out`.
    pub fn reconstruct_into(&self, code: &[u8], out: &mut [f32]) -> Result<()> {
        if code.len() != self.code_len() {
            return Err(Error::DimensionMismatch {
                expected: self.code_len(),
                got: code.len(),
            });
        }
        if out.len() != self.dims {
            return Err(Error::DimensionMismatch {
                expected: self.dims,
                got: out.len(),
            });
        }
        for (s, &code_s) in code.iter().enumerate() {
            let c = code_s as usize;
            let col = s * self.sub_dim;
            out[col..col + self.sub_dim].copy_from_slice(self.centroid(s, c));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SplitMix64;

    fn random_vectors(n: usize, dims: usize, seed: u64) -> Vec<f32> {
        let mut r = SplitMix64::new(seed);
        (0..n * dims).map(|_| r.next_f64() as f32).collect()
    }

    #[test]
    fn train_rejects_indivisible_dims() {
        let v = vec![0.0f32; 30];
        let params = PqParams {
            num_subquantizers: 4,
            bits_per_code: 8,
        };
        // dims=5 not divisible by 4.
        let err = PqCodebook::train(&v, 5, &params, 10, 1).unwrap_err();
        assert!(matches!(err, Error::InvalidConfig(_)));
    }

    #[test]
    fn train_rejects_empty() {
        let params = PqParams::default();
        let err = PqCodebook::train(&[], 16, &params, 10, 1).unwrap_err();
        assert!(matches!(err, Error::InvalidConfig(_)));
    }

    #[test]
    fn code_len_is_num_subspaces() {
        let v = random_vectors(100, 8, 1);
        let params = PqParams {
            num_subquantizers: 4,
            bits_per_code: 8,
        };
        let cb = PqCodebook::train(&v, 8, &params, 10, 1).unwrap();
        assert_eq!(cb.code_len(), 4);
        assert_eq!(cb.encode(&v[..8]).unwrap().len(), 4);
    }

    #[test]
    fn encoding_is_deterministic() {
        let v = random_vectors(200, 16, 5);
        let params = PqParams::default();
        let cb = PqCodebook::train(&v, 16, &params, 15, 42).unwrap();
        let a = cb.encode(&v[..16]).unwrap();
        let b = cb.encode(&v[..16]).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn reconstruction_is_close_for_clustered_data() {
        // When data is highly clustered, PQ reconstruction error is small.
        // Build vectors from a small set of "true" centers plus tiny noise.
        let dims = 8;
        let mut r = SplitMix64::new(11);
        let centers: Vec<Vec<f32>> = (0..4)
            .map(|_| (0..dims).map(|_| (r.next_f64() * 10.0) as f32).collect())
            .collect();
        let mut vectors = Vec::new();
        for _ in 0..400 {
            let c = &centers[r.next_below(centers.len())];
            for &x in c {
                vectors.push(x + (r.next_f64() as f32 - 0.5) * 0.01);
            }
        }
        let params = PqParams {
            num_subquantizers: 4,
            bits_per_code: 8,
        };
        let cb = PqCodebook::train(&vectors, dims, &params, 25, 1).unwrap();

        let v = &vectors[..dims];
        let code = cb.encode(v).unwrap();
        let mut recon = vec![0.0f32; dims];
        cb.reconstruct_into(&code, &mut recon).unwrap();
        let err = slate_simd::l2_sq(v, &recon).unwrap();
        assert!(err < 1.0, "reconstruction error too high: {err}");
    }

    #[test]
    fn codebook_roundtrips_through_json() {
        let v = random_vectors(100, 8, 3);
        let params = PqParams {
            num_subquantizers: 4,
            bits_per_code: 8,
        };
        let cb = PqCodebook::train(&v, 8, &params, 10, 1).unwrap();
        let json = serde_json::to_string(&cb).unwrap();
        let back: PqCodebook = serde_json::from_str(&json).unwrap();
        assert_eq!(cb, back);
    }
}
