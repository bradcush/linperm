//! Fiat-Shamir transcript wrapper around `merlin`.
//!
//! Merlin works on byte strings; this module adapts it to absorb arkworks
//! field elements (via `CanonicalSerialize`) and squeeze unbiased prime-field
//! challenges. All transcript labels must be static byte strings, use them
//! consistently on both prover and verifier sides.

use alloc::vec::Vec;

use ark_ff::PrimeField;
use ark_serialize::CanonicalSerialize;
use ark_std::vec;

const CHALLENGE_SECURITY_PAD_BYTES: usize = 16;

/// Domain-separated Fiat-Shamir transcript.
pub struct Transcript {
    inner: merlin::Transcript,
}

impl Transcript {
    pub fn new(label: &'static [u8]) -> Self {
        Self {
            inner: merlin::Transcript::new(label),
        }
    }

    /// Absorb arbitrary bytes under `label`.
    pub fn append_message(&mut self, label: &'static [u8], message: &[u8]) {
        self.inner.append_message(label, message);
    }

    /// Absorb a serializable arkworks value
    /// (typically a field or group element).
    pub fn append<T: CanonicalSerialize>(
        &mut self,
        label: &'static [u8],
        value: &T,
    ) {
        let mut buf = Vec::with_capacity(value.compressed_size());
        value
            .serialize_compressed(&mut buf)
            .expect("serializing into a Vec cannot fail");
        self.inner.append_message(label, &buf);
    }

    /// Absorb a slice of serializable values under one label.
    pub fn append_slice<T: CanonicalSerialize>(
        &mut self,
        label: &'static [u8],
        values: &[T],
    ) {
        let total: usize = values.iter().map(|v| v.compressed_size()).sum();
        let mut buf = Vec::with_capacity(total);
        for v in values {
            v.serialize_compressed(&mut buf)
                .expect("serializing into a Vec cannot fail");
        }
        self.inner.append_message(label, &buf);
    }

    /// Squeeze a single unbiased prime-field challenge.
    ///
    /// Pulls $\lceil \log_2 p \rceil / 8 + 16$ bytes from the
    /// transcript and reduces modulo $p$, giving statistical distance
    /// $< 2^{-128}$ from uniform, aligned w/ the rest of protocol.
    pub fn challenge<F: PrimeField>(&mut self, label: &'static [u8]) -> F {
        let modulus_bytes = (F::MODULUS_BIT_SIZE as usize).div_ceil(8);
        let mut buf = vec![0u8; modulus_bytes + CHALLENGE_SECURITY_PAD_BYTES];
        self.inner.challenge_bytes(label, &mut buf);
        F::from_be_bytes_mod_order(&buf)
    }

    /// Squeeze a length-`n` vector of independent prime-field challenges.
    pub fn challenge_vec<F: PrimeField>(
        &mut self,
        label: &'static [u8],
        n: usize,
    ) -> Vec<F> {
        (0..n).map(|_| self.challenge(label)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_ff::UniformRand;
    use ark_std::rand::RngCore;
    use ark_std::test_rng;

    #[test]
    fn deterministic_with_same_inputs() {
        let mut a = Transcript::new(b"test");
        let mut b = Transcript::new(b"test");
        a.append_message(b"x", b"hello");
        b.append_message(b"x", b"hello");
        let ca: Fr = a.challenge(b"c");
        let cb: Fr = b.challenge(b"c");
        assert_eq!(ca, cb);
    }

    #[test]
    fn diverges_on_different_inputs() {
        let mut a = Transcript::new(b"test");
        let mut b = Transcript::new(b"test");
        a.append_message(b"x", b"hello");
        b.append_message(b"x", b"world");
        let ca: Fr = a.challenge(b"c");
        let cb: Fr = b.challenge(b"c");
        assert_ne!(ca, cb);
    }

    #[test]
    fn absorbs_arbitrary_bytes() {
        let mut rng = test_rng();
        let mut value = [0u8; 32];
        rng.fill_bytes(&mut value);
        let mut baseline = Transcript::new(b"test");
        let mut absorbed = Transcript::new(b"test");
        absorbed.append_message(b"v", &value);
        let c_base: Fr = baseline.challenge(b"c");
        let c_abs: Fr = absorbed.challenge(b"c");
        assert_ne!(c_base, c_abs);
    }

    #[test]
    fn absorbs_field_element() {
        let mut rng = test_rng();
        let value = Fr::rand(&mut rng);
        let mut baseline = Transcript::new(b"test");
        let mut absorbed = Transcript::new(b"test");
        absorbed.append(b"v", &value);
        let c_base: Fr = baseline.challenge(b"c");
        let c_abs: Fr = absorbed.challenge(b"c");
        assert_ne!(c_base, c_abs);
    }

    #[test]
    fn absorbs_field_elements() {
        let mut rng = test_rng();
        let values = [Fr::rand(&mut rng), Fr::rand(&mut rng)];
        let mut baseline = Transcript::new(b"test");
        let mut absorbed = Transcript::new(b"test");
        absorbed.append_slice(b"v", &values);
        let c_base: Fr = baseline.challenge(b"c");
        let c_abs: Fr = absorbed.challenge(b"c");
        assert_ne!(c_base, c_abs);
    }

    #[test]
    fn challenge_vec_returns_distinct_elements() {
        let mut t = Transcript::new(b"test");
        let v: Vec<Fr> = t.challenge_vec(b"c", 4);
        for i in 0..v.len() {
            for j in (i + 1)..v.len() {
                assert_ne!(v[i], v[j]);
            }
        }
    }
}
