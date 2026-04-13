//! Asymmetric distance computation (ADC) tables.
//!
//! Given a query and a trained [`PqCodebook`], an [`AdcTable`] precomputes — once
//! per query — the squared-L2 distance from each query sub-vector to every
//! centroid in that subspace. The approximate distance to any encoded vector is
//! then a sum of `M` table lookups, with no per-vector arithmetic over the full
//! dimensionality.
//!
//! "Asymmetric" because the query stays in full precision (it is not quantized);
//! only the database vectors are quantized. This is more accurate than the
//! symmetric variant (quantizing the query too) at no extra query-time cost.
//!
//! The table is the engine's cheap, RAM-resident distance oracle: it lets a
//! search rank candidates and decide which exact vectors to fetch from disk
//! without touching the disk at all.

use slate_core::{Error, Result};

use crate::codebook::PqCodebook;

/// Per-query ADC lookup table over a [`PqCodebook`].
///
/// Holds `num_subspaces * centroids_per_subspace` precomputed squared-L2
/// distances. [`distance`](Self::distance) prices an encoded vector by summing
/// one entry per subspace.
#[derive(Debug, Clone)]
pub struct AdcTable<'cb> {
    codebook: &'cb PqCodebook,
    /// `table[s * stride + c]` = squared-L2 distance from query subspace `s` to
    /// centroid `c` of that subspace. `stride == centroids_per_subspace`.
    table: Vec<f32>,
    stride: usize,
}

impl<'cb> AdcTable<'cb> {
    /// Build the ADC table for `query` against `codebook`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] if `query.len() != codebook.dims()`,
    /// and propagates distance-kernel errors.
    pub fn build(codebook: &'cb PqCodebook, query: &[f32]) -> Result<Self> {
        if query.len() != codebook.dims() {
            return Err(Error::DimensionMismatch {
                expected: codebook.dims(),
                got: query.len(),
            });
        }
        let m = codebook.num_subspaces();
        let sub_dim = codebook.sub_dim();
        let stride = codebook.centroids_per_subspace();
        let mut table = vec![f32::INFINITY; m * stride];

        for s in 0..m {
            let col = s * sub_dim;
            let q_sub = &query[col..col + sub_dim];
            let k = codebook.subspace_k(s);
            for c in 0..k {
                table[s * stride + c] = slate_simd::l2_sq(q_sub, codebook.centroid(s, c))?;
            }
        }

        Ok(Self {
            codebook,
            table,
            stride,
        })
    }

    /// The codebook this table was built against.
    #[inline]
    pub fn codebook(&self) -> &PqCodebook {
        self.codebook
    }

    /// Approximate squared-L2 distance from the query to the vector encoded by
    /// `code`, as the sum of one table entry per subspace.
    ///
    /// This is the hot lookup: `M` indexed loads and adds, all in RAM.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] if `code.len() != codebook.code_len()`.
    #[inline]
    pub fn distance(&self, code: &[u8]) -> Result<f32> {
        if code.len() != self.codebook.code_len() {
            return Err(Error::DimensionMismatch {
                expected: self.codebook.code_len(),
                got: code.len(),
            });
        }
        let mut sum = 0.0f32;
        for (s, &c) in code.iter().enumerate() {
            sum += self.table[s * self.stride + c as usize];
        }
        Ok(sum)
    }

    /// Approximate distance for code `i` in a flat code buffer (length
    /// `count * code_len`).
    ///
    /// Avoids slicing bookkeeping in hot loops over a whole code table.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Corrupt`] if `codes` is not a whole number of codes or
    /// `i` is out of range.
    #[inline]
    pub fn distance_at(&self, codes: &[u8], i: usize) -> Result<f32> {
        let code_len = self.codebook.code_len();
        let start = i * code_len;
        let end = start + code_len;
        if end > codes.len() {
            return Err(Error::corrupt(format!(
                "code index {i} out of range for buffer of {} codes",
                codes.len() / code_len
            )));
        }
        let mut sum = 0.0f32;
        for (s, &c) in codes[start..end].iter().enumerate() {
            sum += self.table[s * self.stride + c as usize];
        }
        Ok(sum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SplitMix64;
    use slate_core::PqParams;

    fn random_vectors(n: usize, dims: usize, seed: u64) -> Vec<f32> {
        let mut r = SplitMix64::new(seed);
        (0..n * dims).map(|_| r.next_f64() as f32).collect()
    }

    #[test]
    fn adc_matches_reconstructed_distance() {
        // ADC distance to a code must equal the exact L2 between the query and
        // the code's reconstruction (both are the sum of per-subspace
        // query-to-centroid distances).
        let dims = 16;
        let vectors = random_vectors(300, dims, 1);
        let params = PqParams::default();
        let cb = PqCodebook::train(&vectors, dims, &params, 20, 7).unwrap();

        let query = &vectors[..dims];
        let table = AdcTable::build(&cb, query).unwrap();

        // Take some database vector, encode it, compare ADC vs reconstruction.
        let dbv = &vectors[dims..2 * dims];
        let code = cb.encode(dbv).unwrap();
        let mut recon = vec![0.0f32; dims];
        cb.reconstruct_into(&code, &mut recon).unwrap();

        let adc = table.distance(&code).unwrap();
        let exact_recon = slate_simd::l2_sq(query, &recon).unwrap();
        assert!(
            (adc - exact_recon).abs() <= 1e-3 * exact_recon.max(1.0),
            "adc {adc} != reconstructed {exact_recon}"
        );
    }

    #[test]
    fn adc_ranks_like_exact_on_clustered_data() {
        // On clustered data, ADC ordering should correlate strongly with exact
        // ordering. We check the nearest-by-ADC is among the nearest-by-exact.
        let dims = 16;
        let mut r = SplitMix64::new(3);
        let centers: Vec<Vec<f32>> = (0..8)
            .map(|_| (0..dims).map(|_| (r.next_f64() * 10.0) as f32).collect())
            .collect();
        let mut vectors = Vec::new();
        for _ in 0..500 {
            let c = &centers[r.next_below(centers.len())];
            for &x in c {
                vectors.push(x + (r.next_f64() as f32 - 0.5) * 0.5);
            }
        }
        let params = PqParams::default();
        let cb = PqCodebook::train(&vectors, dims, &params, 25, 9).unwrap();
        let codes = cb.encode_batch(&vectors).unwrap();

        let query: Vec<f32> = centers[0].clone();
        let table = AdcTable::build(&cb, &query).unwrap();

        let n = vectors.len() / dims;
        // Exact nearest.
        let mut exact_best = 0usize;
        let mut exact_best_d = f32::INFINITY;
        for i in 0..n {
            let d = slate_simd::l2_sq(&query, &vectors[i * dims..(i + 1) * dims]).unwrap();
            if d < exact_best_d {
                exact_best_d = d;
                exact_best = i;
            }
        }
        // ADC nearest.
        let mut adc_best = 0usize;
        let mut adc_best_d = f32::INFINITY;
        for i in 0..n {
            let d = table.distance_at(&codes, i).unwrap();
            if d < adc_best_d {
                adc_best_d = d;
                adc_best = i;
            }
        }
        // The two should land in the same cluster (very close exact distance).
        let exact_of_adc_best =
            slate_simd::l2_sq(&query, &vectors[adc_best * dims..(adc_best + 1) * dims]).unwrap();
        assert!(
            exact_of_adc_best <= exact_best_d + 5.0,
            "adc best {adc_best} (exact {exact_of_adc_best}) far from exact best {exact_best} ({exact_best_d})"
        );
    }

    #[test]
    fn wrong_query_dims_is_reported() {
        let vectors = random_vectors(100, 8, 1);
        let cb = PqCodebook::train(
            &vectors,
            8,
            &PqParams {
                num_subquantizers: 4,
                bits_per_code: 8,
            },
            10,
            1,
        )
        .unwrap();
        let err = AdcTable::build(&cb, &[0.0; 7]).unwrap_err();
        assert!(matches!(err, Error::DimensionMismatch { .. }));
    }
}
