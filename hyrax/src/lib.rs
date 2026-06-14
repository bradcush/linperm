//! Hyrax multilinear polynomial commitment backend.
//!
//! Implements [`permcore::PolynomialCommitment`] following Wahby et al.
//! (ePrint 2017/1132, §6). A $\nu$-variate polynomial's evaluation vector is
//! reshaped into a $2^{\nu_r} \times 2^{\nu_c}$ matrix $M$; the commitment is
//! one Pedersen commitment per row against a fixed column-generator set. An
//! evaluation at $z$ is proved by the combined row $w = L^\top M$: the
//! verifier checks $\sum_i L_i C_i = \sum_j w_j g_j$ (Pedersen binding forces
//! $w = L^\top M$) and $\langle w, R \rangle = \mathrm{value}$.
//!
//! This is the **binding-only, naive** variant: no blinding (so not hiding /
//! zero-knowledge) and no inner-product compression (so the proof is the
//! $O(\sqrt{N})$ vector $w$ rather than $O(\log N)$). The transcript is
//! therefore unused, there are no squeezed challenges yet; Commitment and
//! verifier costs are already $O(\sqrt{N})$; only the prover's commit/open
//! time is $O(N)$ pending the sparse backend.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod matrix;

use alloc::vec::Vec;
use core::marker::PhantomData;

use ark_ec::CurveGroup;
use ark_std::rand::RngCore;

use permcore::pcs::MleRef;
use permcore::{PolynomialCommitment, Transcript};

use matrix::{col_tensor, dot, lt_times_m, row_tensor, split_vars};

/// Public parameters: the column generators (with unknown pairwise discrete
/// logs) and the variable bound they were sized for. `HyraxKey` equals Hyrax's
/// verifier key equals its prover key, both hold the full generator set, which
/// is why the verifier is $O(\sqrt{N})$ rather than constant-time.
#[derive(Clone, Debug)]
pub struct HyraxKey<G: CurveGroup> {
    pub generators: Vec<G::Affine>,
    pub max_num_vars: usize,
}

/// Errors from the Hyrax backend.
#[derive(Debug, PartialEq, Eq)]
pub enum HyraxError {
    /// A polynomial has more variables
    /// than `setup` provisioned for.
    TooManyVars {
        num_vars: usize,
        max_num_vars: usize,
    },
    /// `point` length did not match the poly's var count.
    PointLenMismatch { expected: usize, got: usize },
}

/// Hyrax commitment scheme over the group `G`.
/// Acts as though we store G but we don't actually.
pub struct Hyrax<G: CurveGroup>(PhantomData<G>);

impl<G: CurveGroup> PolynomialCommitment<G::ScalarField> for Hyrax<G> {
    type ProverKey = HyraxKey<G>;
    type VerifierKey = HyraxKey<G>;
    type Commitment = Vec<G::Affine>;
    type Proof = Vec<G::ScalarField>;
    type Error = HyraxError;

    /// Sample the column generators, one for each.
    ///
    /// Generators come from `rng`, so binding rests on the sampler
    /// being honest (nobody learns the pairwise discrete logs),
    /// trusted-setup. A transparent, recomputable variant would
    /// derive them by hashing a public seed to the curve.
    fn setup<R: RngCore>(
        max_num_vars: usize,
        rng: &mut R,
    ) -> Result<(Self::ProverKey, Self::VerifierKey), Self::Error> {
        let (_, n_col) = split_vars(max_num_vars);
        let count = 1usize << n_col;
        let generators: Vec<G::Affine> =
            (0..count).map(|_| G::rand(rng).into_affine()).collect();
        let key = HyraxKey {
            generators,
            max_num_vars,
        };
        Ok((key.clone(), key))
    }

    /// Commit to the rows of the `M` matrix using dense
    /// evaluations, single MSM for each row is used.
    fn commit(
        pk: &Self::ProverKey,
        poly: MleRef<'_, G::ScalarField>,
    ) -> Result<Self::Commitment, Self::Error> {
        let num_vars = poly.num_vars();
        if num_vars > pk.max_num_vars {
            return Err(HyraxError::TooManyVars {
                num_vars,
                max_num_vars: pk.max_num_vars,
            });
        }
        let dense = poly.to_dense();
        let (n_row, n_col) = split_vars(num_vars);
        let cols = 1usize << n_col;
        let bases = &pk.generators[..cols];
        // One Pedersen MSM per row. The sparse backend will pass only a row's
        // nonzero (generator, scalar) pairs here instead of the full slice.
        // That will allow us to get an optimized prover when sparse.
        let commitment = (0..(1usize << n_row))
            .map(|row| {
                let base = row * cols;
                let scalars = &dense.evaluations[base..base + cols];
                G::msm_unchecked(bases, scalars).into_affine()
            })
            .collect();

        Ok(commitment)
    }

    /// Opens a commitment at the evaluation point.
    /// Uses full `w` for proof of correct opening.
    fn open(
        pk: &Self::ProverKey,
        poly: MleRef<'_, G::ScalarField>,
        point: &[G::ScalarField],
        _transcript: &mut Transcript,
    ) -> Result<(G::ScalarField, Self::Proof), Self::Error> {
        let num_vars = poly.num_vars();
        if num_vars > pk.max_num_vars {
            return Err(HyraxError::TooManyVars {
                num_vars,
                max_num_vars: pk.max_num_vars,
            });
        }
        if point.len() != num_vars {
            return Err(HyraxError::PointLenMismatch {
                expected: num_vars,
                got: point.len(),
            });
        }
        let dense = poly.to_dense();
        let (_, n_col) = split_vars(num_vars);
        // We're not returning l yet
        let l = row_tensor(point, n_col);
        let r = col_tensor(point, n_col);
        // `dense.evaluations` is M in a vector form
        let w = lt_times_m(&dense.evaluations, &l, n_col);
        let value = dot(&w, &r);
        Ok((value, w))
    }

    /// Verify consistency between a commitment, point, value, and proof.
    /// Legitimate evaluations of the poly w/ extremely high probability.
    fn verify(
        vk: &Self::VerifierKey,
        commitment: &Self::Commitment,
        point: &[G::ScalarField],
        value: G::ScalarField,
        proof: &Self::Proof,
        _transcript: &mut Transcript,
    ) -> Result<bool, Self::Error> {
        let num_vars = point.len();
        let (n_row, n_col) = split_vars(num_vars);
        let cols = 1usize << n_col;
        // Structural shape check, malformed proof is a rejection.
        if commitment.len() != (1usize << n_row) || proof.len() != cols {
            return Ok(false);
        }
        let l = row_tensor(point, n_col);
        let r = col_tensor(point, n_col);
        // Binding means the homomorphic combination of row commitments must
        // equal the Pedersen commitment of the claimed combined row w.
        let from_commitment = G::msm_unchecked(commitment, &l);
        let from_proof = G::msm_unchecked(&vk.generators[..cols], proof);
        if from_commitment != from_proof {
            return Ok(false);
        }
        // Evaluation <w, R> must be the claimed value.
        Ok(dot(proof, &r) == value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::{Fr, G1Projective};
    use ark_ff::UniformRand;
    use ark_poly::{DenseMultilinearExtension, Polynomial};
    use ark_std::test_rng;

    type H = Hyrax<G1Projective>;

    /// Random polynomial over $2^{num_vars}$
    fn random_poly(
        num_vars: usize,
        rng: &mut impl RngCore,
    ) -> DenseMultilinearExtension<Fr> {
        let evals: Vec<Fr> =
            (0..(1usize << num_vars)).map(|_| Fr::rand(rng)).collect();
        DenseMultilinearExtension::from_evaluations_vec(num_vars, evals)
    }

    /// Helper to exercise even and odd variable counts,
    /// including the degenerate 0, containing the assertion.
    fn round_trip_at(num_vars: usize) {
        let mut rng = test_rng();
        let (pk, vk) = H::setup(num_vars, &mut rng).unwrap();
        let poly = random_poly(num_vars, &mut rng);
        let comm = H::commit(&pk, (&poly).into()).unwrap();
        let point: Vec<Fr> =
            (0..num_vars).map(|_| Fr::rand(&mut rng)).collect();
        let mut p_t = Transcript::new(b"hyrax");
        let (value, proof) =
            H::open(&pk, (&poly).into(), &point, &mut p_t).unwrap();
        // The opened value must be the true MLE evaluation: this is the
        // test that pins the reshape + eq-tensor ordering convention.
        // Helpful to log `num_vars`, quickly identify what broke.
        assert_eq!(value, poly.evaluate(&point), "num_vars = {num_vars}");
        let mut v_t = Transcript::new(b"hyrax");
        assert!(
            H::verify(&vk, &comm, &point, value, &proof, &mut v_t).unwrap(),
            "num_vars = {num_vars}"
        );
    }

    #[test]
    // Encapsulates many tests
    fn round_trip_various_num_vars() {
        for num_vars in 0..=6 {
            round_trip_at(num_vars);
        }
    }

    #[test]
    fn rejects_wrong_value() {
        let mut rng = test_rng();
        let num_vars = 4;
        let (pk, vk) = H::setup(num_vars, &mut rng).unwrap();
        let poly = random_poly(num_vars, &mut rng);
        let comm = H::commit(&pk, (&poly).into()).unwrap();
        let point: Vec<Fr> =
            (0..num_vars).map(|_| Fr::rand(&mut rng)).collect();
        let mut p_t = Transcript::new(b"hyrax");
        let (value, proof) =
            H::open(&pk, (&poly).into(), &point, &mut p_t).unwrap();
        let mut v_t = Transcript::new(b"hyrax");
        assert!(!H::verify(
            &vk,
            &comm,
            &point,
            value + Fr::from(1u64),
            &proof,
            &mut v_t
        )
        .unwrap());
    }

    #[test]
    fn rejects_tampered_proof() {
        let mut rng = test_rng();
        let num_vars = 4;
        let (pk, vk) = H::setup(num_vars, &mut rng).unwrap();
        let poly = random_poly(num_vars, &mut rng);
        let comm = H::commit(&pk, (&poly).into()).unwrap();
        let point: Vec<Fr> =
            (0..num_vars).map(|_| Fr::rand(&mut rng)).collect();
        let mut p_t = Transcript::new(b"hyrax");
        let (value, mut proof) =
            H::open(&pk, (&poly).into(), &point, &mut p_t).unwrap();
        // Tampering w (proof) breaks the binding check in the verifier,
        // the prover can't also match the commitment w/ the proof.
        proof[0] += Fr::from(1u64);
        let mut v_t = Transcript::new(b"hyrax");
        assert!(
            !H::verify(&vk, &comm, &point, value, &proof, &mut v_t).unwrap()
        );
    }

    #[test]
    fn rejects_wrong_point() {
        let mut rng = test_rng();
        let num_vars = 4;
        let (pk, vk) = H::setup(num_vars, &mut rng).unwrap();
        let poly = random_poly(num_vars, &mut rng);
        let comm = H::commit(&pk, (&poly).into()).unwrap();
        let point: Vec<Fr> =
            (0..num_vars).map(|_| Fr::rand(&mut rng)).collect();
        let mut p_t = Transcript::new(b"hyrax");
        let (value, proof) =
            H::open(&pk, (&poly).into(), &point, &mut p_t).unwrap();
        let mut other: Vec<Fr> = point.clone();
        // Fails `from_commitment != from_proof` check
        other[0] += Fr::from(1u64);
        let mut v_t = Transcript::new(b"hyrax");
        assert!(
            !H::verify(&vk, &comm, &other, value, &proof, &mut v_t).unwrap()
        );
    }

    #[test]
    fn distinct_polys_distinct_commitments() {
        let mut rng = test_rng();
        let num_vars = 4;
        let (pk, _) = H::setup(num_vars, &mut rng).unwrap();
        let a = random_poly(num_vars, &mut rng);
        let b = random_poly(num_vars, &mut rng);
        let c_a = H::commit(&pk, (&a).into()).unwrap();
        let c_b = H::commit(&pk, (&b).into()).unwrap();
        assert_ne!(c_a, c_b);
    }

    #[test]
    fn rejects_too_many_vars() {
        let mut rng = test_rng();
        let (pk, _) = H::setup(2, &mut rng).unwrap();
        let poly = random_poly(4, &mut rng);
        assert_eq!(
            H::commit(&pk, (&poly).into()),
            // Setup required variable count aware
            Err(HyraxError::TooManyVars {
                num_vars: 4,
                max_num_vars: 2,
            })
        );
    }
}
