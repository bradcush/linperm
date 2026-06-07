//! Polynomial-commitment trait for multilinear polynomials.
//!
//! Models Definitions 6 and 7 of §4.4 of the paper. The trait is intentionally
//! minimal so swapping in Hyrax / Dory / KZH / multilinear-KZG / FRI-based
//! schemes later is purely a matter of binding the type parameter at the call
//! site — the PIOP layer never sees the concrete commitment type.
//!
//! [`MockPcs`] is a non-cryptographic placeholder that stores the committed
//! polynomial in plaintext. It is meant for unit-testing the PIOPs and must
//! never be used outside tests.

use core::convert::Infallible;
use core::marker::PhantomData;

use ark_ff::PrimeField;
use ark_poly::{
    DenseMultilinearExtension, MultilinearExtension, Polynomial,
    SparseMultilinearExtension,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::RngCore;

use crate::transcript::Transcript;

/// Borrowed multilinear polynomial in either representation. Both variants
/// denote the same mathematical object, a multilinear polynomial given by its
/// evaluations over $B_\mu$. We only support both for benchmarking.
#[derive(Clone, Copy)]
pub enum MleRef<'a, F: PrimeField> {
    Dense(&'a DenseMultilinearExtension<F>),
    Sparse(&'a SparseMultilinearExtension<F>),
}

impl<F: PrimeField> MleRef<'_, F> {
    /// Number of variables $\mu$.
    pub fn num_vars(&self) -> usize {
        match self {
            Self::Dense(poly) => poly.num_vars,
            Self::Sparse(poly) => poly.num_vars,
        }
    }

    /// Evaluate at `point`, `num_vars()` coordinates.
    pub fn evaluate(&self, point: &[F]) -> F {
        match self {
            Self::Dense(poly) => poly.evaluate(&point.to_vec()),
            Self::Sparse(poly) => poly.evaluate(&point.to_vec()),
        }
    }

    /// Dense evaluation table, $O(2^\mu)$ time and memory either way.
    pub fn to_dense(&self) -> DenseMultilinearExtension<F> {
        match self {
            Self::Dense(poly) => (*poly).clone(),
            Self::Sparse(poly) => {
                DenseMultilinearExtension::from_evaluations_vec(
                    poly.num_vars,
                    poly.to_evaluations(),
                )
            }
        }
    }
}

impl<'a, F: PrimeField> From<&'a DenseMultilinearExtension<F>>
    for MleRef<'a, F>
{
    fn from(poly: &'a DenseMultilinearExtension<F>) -> Self {
        Self::Dense(poly)
    }
}

impl<'a, F: PrimeField> From<&'a SparseMultilinearExtension<F>>
    for MleRef<'a, F>
{
    fn from(poly: &'a SparseMultilinearExtension<F>) -> Self {
        Self::Sparse(poly)
    }
}

/// A multilinear polynomial commitment scheme.
///
/// All operations are written transcript-aware so the `Eval` sub-protocol
/// can be made non-interactive via Fiat-Shamir at compile time.
pub trait PolynomialCommitment<F: PrimeField> {
    type ProverKey;
    type VerifierKey;
    type Commitment: Clone + CanonicalSerialize + CanonicalDeserialize;
    type Proof: CanonicalSerialize + CanonicalDeserialize;
    type Error: core::fmt::Debug;

    /// Generate prover and verifier keys for polynomials with
    /// up to `max_num_vars` variables. SRS is sized in advance,
    /// how many variables a committed polynomial can have. Note
    /// that `max_num_vars = 3` means $B_\mu = 8$ constraints.
    fn setup<R: RngCore>(
        max_num_vars: usize,
        // Toxic waste management internal
        rng: &mut R,
    ) -> Result<(Self::ProverKey, Self::VerifierKey), Self::Error>;

    /// Commit to a multilinear polynomial. Commitments must bind the
    /// polynomial, not its representation, so a sparse polynomial
    /// and its densification will commit identically.
    fn commit(
        pk: &Self::ProverKey,
        poly: MleRef<'_, F>,
    ) -> Result<Self::Commitment, Self::Error>;

    /// Prove an evaluation `poly(point) = value` and return both.
    fn open(
        pk: &Self::ProverKey,
        poly: MleRef<'_, F>,
        point: &[F],
        transcript: &mut Transcript,
        // Returning the point w/ its proof
    ) -> Result<(F, Self::Proof), Self::Error>;

    /// Verify an evaluation proof against a commitment.
    fn verify(
        vk: &Self::VerifierKey,
        commitment: &Self::Commitment,
        point: &[F],
        value: F,
        proof: &Self::Proof,
        transcript: &mut Transcript,
    ) -> Result<bool, Self::Error>;
}

/// A trivial, non-cryptographic PCS that stores the committed polynomial
/// verbatim. Useful for exercising PIOPs end-to-end in tests.
pub struct MockPcs<F: PrimeField>(PhantomData<F>);

impl<F: PrimeField> PolynomialCommitment<F> for MockPcs<F> {
    type ProverKey = ();
    type VerifierKey = ();
    // Commitment here the polynomial itself
    type Commitment = DenseMultilinearExtension<F>;
    type Proof = ();
    // Like Unreachable
    type Error = Infallible;

    fn setup<R: RngCore>(
        _max_num_vars: usize,
        _rng: &mut R,
    ) -> Result<(Self::ProverKey, Self::VerifierKey), Self::Error> {
        Ok(((), ()))
    }

    fn commit(
        _pk: &Self::ProverKey,
        poly: MleRef<'_, F>,
    ) -> Result<Self::Commitment, Self::Error> {
        // Densifying makes the commitment
        // representation-independent.
        Ok(poly.to_dense())
    }

    fn open(
        _pk: &Self::ProverKey,
        poly: MleRef<'_, F>,
        point: &[F],
        _transcript: &mut Transcript,
    ) -> Result<(F, Self::Proof), Self::Error> {
        Ok((poly.evaluate(point), ()))
    }

    fn verify(
        _vk: &Self::VerifierKey,
        commitment: &Self::Commitment,
        point: &[F],
        value: F,
        _proof: &Self::Proof,
        _transcript: &mut Transcript,
    ) -> Result<bool, Self::Error> {
        // Since it's the polynomial evaluation is the check
        Ok(commitment.evaluate(&point.to_vec()) == value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    #[test]
    // Contract-level test for the trait
    fn mock_pcs_round_trip() {
        let mut rng = test_rng();
        let num_vars = 3;
        let evals: Vec<Fr> =
            (0..(1 << num_vars)).map(|_| Fr::rand(&mut rng)).collect();
        let poly =
            DenseMultilinearExtension::from_evaluations_vec(num_vars, evals);

        let (pk, vk) = MockPcs::<Fr>::setup(num_vars, &mut rng).unwrap();
        let comm = MockPcs::commit(&pk, (&poly).into()).unwrap();

        // Random point $\alpha \in F^3$
        let point: Vec<Fr> =
            (0..num_vars).map(|_| Fr::rand(&mut rng)).collect();

        // Real openings would absorb `comm` and `point` here before opening;
        // `MockPcs` doesn't bother since it doesn't squeeze any challenges.
        // Domain separation is absorption the first transcript.
        let mut prover_t = Transcript::new(b"mock");
        let (value, proof) =
            MockPcs::open(&pk, (&poly).into(), &point, &mut prover_t).unwrap();

        // Real backends do pairing checks / inner-product
        // checks against comm without ever seeing poly
        let mut verifier_t = Transcript::new(b"mock");
        assert!(MockPcs::verify(
            &vk,
            &comm,
            &point,
            value,
            &proof,
            &mut verifier_t
        )
        .unwrap());

        // Brand new transcript for the second check. Must be fresh.
        // Doesn't matter for MockPcs since it doesn't squeeze.
        let mut verifier_t = Transcript::new(b"mock");
        assert!(!MockPcs::verify(
            &vk,
            &comm,
            &point,
            // Wrong value
            value + Fr::from(1u64),
            &proof,
            &mut verifier_t
        )
        .unwrap());
    }

    #[test]
    // Commitments bind the polynomial, not the
    // representation it was handed over in.
    fn sparse_and_dense_commit_identically() {
        let mut rng = test_rng();
        let num_vars = 4;
        let entries = [(3usize, Fr::rand(&mut rng)), (9, Fr::rand(&mut rng))];
        let sparse =
            SparseMultilinearExtension::from_evaluations(num_vars, &entries);
        let dense = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            sparse.to_evaluations(),
        );
        let (pk, _vk) = MockPcs::<Fr>::setup(num_vars, &mut rng).unwrap();
        let c_sparse = MockPcs::commit(&pk, (&sparse).into()).unwrap();
        let c_dense = MockPcs::commit(&pk, (&dense).into()).unwrap();
        assert_eq!(c_sparse, c_dense);
        let point: Vec<Fr> =
            (0..num_vars).map(|_| Fr::rand(&mut rng)).collect();
        assert_eq!(
            MleRef::from(&sparse).evaluate(&point),
            MleRef::from(&dense).evaluate(&point),
        );
    }
}
