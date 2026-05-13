//! Narrow on-disk dtype codecs: `f16` and symmetric per-vector `i8`.
//!
//! These are pure byte-buffer transforms — no I/O, no `unsafe`. The storage
//! layer encodes vectors with [`encode_f16`] / [`encode_i8`] when writing a
//! narrow store, and decodes them back to `f32` with [`decode_f16`] /
//! [`decode_i8`] on the read path so the rest of the engine stays dtype-blind.
//!
//! # Layouts
//!
//! - **`f16`**: `dims` little-endian [`half::f16`] elements, 2 bytes each. No
//!   per-vector metadata.
//! - **`i8`**: a 4-byte little-endian `f32` *scale* followed by `dims` signed
//!   bytes. The scale is `max|x_i| / 127`; each code is `round(x_i / scale)`
//!   clamped to `[-127, 127]` (the `-128` slot is unused so the range is
//!   symmetric). Decode is `code · scale`. An all-zero vector encodes to
//!   `scale = 0` and all-zero codes and decodes back to zero exactly.

use half::f16;
use slate_core::{Dtype, Error, Result};

/// Largest magnitude a symmetric `i8` code may take (`-128` is left unused).
const I8_MAX: f32 = 127.0;

/// Encode an `f32` vector into `out` as little-endian `f16`.
///
/// `out` must be exactly `dims * 2` bytes. Errors with
/// [`Error::DimensionMismatch`] otherwise.
pub fn encode_f16(vector: &[f32], out: &mut [u8]) -> Result<()> {
    let expected = Dtype::F16.stored_vector_bytes(vector.len());
    if out.len() != expected {
        return Err(Error::DimensionMismatch {
            expected,
            got: out.len(),
        });
    }
    for (x, slot) in vector.iter().zip(out.chunks_exact_mut(2)) {
        slot.copy_from_slice(&f16::from_f32(*x).to_le_bytes());
    }
    Ok(())
}

/// Decode a little-endian `f16` byte slot into an `f32` vector.
///
/// `bytes` must be `dims * 2` long and `out` exactly `dims` long.
pub fn decode_f16(bytes: &[u8], out: &mut [f32]) -> Result<()> {
    let expected = Dtype::F16.stored_vector_bytes(out.len());
    if bytes.len() != expected {
        return Err(Error::DimensionMismatch {
            expected,
            got: bytes.len(),
        });
    }
    for (slot, x) in bytes.chunks_exact(2).zip(out.iter_mut()) {
        // chunks_exact yields exactly 2-byte slices.
        let raw = u16::from_le_bytes([slot[0], slot[1]]);
        *x = f16::from_bits(raw).to_f32();
    }
    Ok(())
}

/// Encode an `f32` vector into `out` as a symmetric per-vector `i8` slot.
///
/// `out` must be exactly `dims + 4` bytes (a 4-byte `f32` scale followed by
/// `dims` signed codes). Errors with [`Error::DimensionMismatch`] otherwise.
pub fn encode_i8(vector: &[f32], out: &mut [u8]) -> Result<()> {
    let expected = Dtype::I8.stored_vector_bytes(vector.len());
    if out.len() != expected {
        return Err(Error::DimensionMismatch {
            expected,
            got: out.len(),
        });
    }

    let max_abs = vector.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    let scale = if max_abs == 0.0 { 0.0 } else { max_abs / I8_MAX };

    let (scale_bytes, code_bytes) = out.split_at_mut(core::mem::size_of::<f32>());
    scale_bytes.copy_from_slice(&scale.to_le_bytes());

    if scale == 0.0 {
        // All-zero vector: every code is zero, decodes back to zero exactly.
        for b in code_bytes.iter_mut() {
            *b = 0;
        }
        return Ok(());
    }

    let inv = 1.0 / scale;
    for (x, b) in vector.iter().zip(code_bytes.iter_mut()) {
        let q = (x * inv).round().clamp(-I8_MAX, I8_MAX) as i8;
        *b = q as u8;
    }
    Ok(())
}

/// Decode a symmetric per-vector `i8` slot into an `f32` vector.
///
/// `bytes` must be `dims + 4` long (scale + codes) and `out` exactly `dims`.
pub fn decode_i8(bytes: &[u8], out: &mut [f32]) -> Result<()> {
    let expected = Dtype::I8.stored_vector_bytes(out.len());
    if bytes.len() != expected {
        return Err(Error::DimensionMismatch {
            expected,
            got: bytes.len(),
        });
    }
    let (scale_bytes, code_bytes) = bytes.split_at(core::mem::size_of::<f32>());
    // split_at guarantees scale_bytes is exactly 4 bytes.
    let scale = f32::from_le_bytes([
        scale_bytes[0],
        scale_bytes[1],
        scale_bytes[2],
        scale_bytes[3],
    ]);
    for (b, x) in code_bytes.iter().zip(out.iter_mut()) {
        *x = f32::from(*b as i8) * scale;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrip_is_near_lossless() {
        let v = [0.0f32, 1.0, -1.0, 0.5, -0.25, 3.5, -2.75, 100.0];
        let mut enc = vec![0u8; Dtype::F16.stored_vector_bytes(v.len())];
        encode_f16(&v, &mut enc).unwrap();
        let mut dec = vec![0.0f32; v.len()];
        decode_f16(&enc, &mut dec).unwrap();
        for (a, b) in v.iter().zip(dec.iter()) {
            // f16 has ~3 decimal digits; relative error well under 1e-2.
            let tol = 1e-2 * a.abs().max(1.0);
            assert!((a - b).abs() <= tol, "f16 {a} -> {b}");
        }
    }

    #[test]
    fn f16_known_bit_pattern() {
        // 1.0 in IEEE binary16 is 0x3C00.
        let mut enc = vec![0u8; 2];
        encode_f16(&[1.0], &mut enc).unwrap();
        assert_eq!(u16::from_le_bytes([enc[0], enc[1]]), 0x3C00);
    }

    #[test]
    fn i8_roundtrip_within_half_a_step() {
        let v = [0.0f32, 1.0, -1.0, 0.3, -0.7, 0.92, -0.42, 0.5];
        let mut enc = vec![0u8; Dtype::I8.stored_vector_bytes(v.len())];
        encode_i8(&v, &mut enc).unwrap();
        let mut dec = vec![0.0f32; v.len()];
        decode_i8(&enc, &mut dec).unwrap();
        // Quantization error is at most half a step = max_abs / 254.
        let max_abs = v.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        let step = max_abs / 254.0;
        for (a, b) in v.iter().zip(dec.iter()) {
            assert!((a - b).abs() <= step + 1e-6, "i8 {a} -> {b}");
        }
    }

    #[test]
    fn i8_all_zero_vector_is_exact() {
        let v = [0.0f32; 6];
        let mut enc = vec![0u8; Dtype::I8.stored_vector_bytes(v.len())];
        encode_i8(&v, &mut enc).unwrap();
        // Scale is zero.
        assert_eq!(f32::from_le_bytes([enc[0], enc[1], enc[2], enc[3]]), 0.0);
        let mut dec = vec![9.0f32; v.len()];
        decode_i8(&enc, &mut dec).unwrap();
        assert!(dec.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn i8_extremes_hit_symmetric_range() {
        // The element equal to max_abs must quantize to +127; its negation to -127.
        let v = [2.0f32, -2.0, 0.0, 1.0];
        let mut enc = vec![0u8; Dtype::I8.stored_vector_bytes(v.len())];
        encode_i8(&v, &mut enc).unwrap();
        let codes: Vec<i8> = enc[4..].iter().map(|&b| b as i8).collect();
        assert_eq!(codes[0], 127);
        assert_eq!(codes[1], -127);
        assert_eq!(codes[2], 0);
    }

    #[test]
    fn wrong_buffer_lengths_error() {
        let v = [1.0f32, 2.0];
        let mut small = vec![0u8; 1];
        assert!(encode_f16(&v, &mut small).is_err());
        assert!(encode_i8(&v, &mut small).is_err());
        let mut out = vec![0.0f32; 2];
        assert!(decode_f16(&[0u8; 3], &mut out).is_err());
        assert!(decode_i8(&[0u8; 3], &mut out).is_err());
    }
}
