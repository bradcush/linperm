use ark_bn254::Fr;
use ark_ff::UniformRand;
use ark_poly::DenseMultilinearExtension;
use ark_std::test_rng;

use permcore::{MockPcs, Permutation, PolynomialCommitment, Transcript};

#[test]
// Confirm `permcore`'s public API composes from a downstream crate.
// Catches reachability regressions (e.g. a `pub` narrowed to `pub(crate)`).
fn public_api_round_trips() {
    let mut rng = test_rng();
    let num_vars = 4;
    let perm = Permutation::identity(num_vars);
    let evals: Vec<Fr> = perm.bit_evaluations(0);
    let poly = DenseMultilinearExtension::from_evaluations_vec(num_vars, evals);
    let (pk, vk) = MockPcs::<Fr>::setup(num_vars, &mut rng).unwrap();
    let commitment = MockPcs::commit(&pk, (&poly).into()).unwrap();
    let point: Vec<Fr> = (0..num_vars).map(|_| Fr::rand(&mut rng)).collect();
    let mut prover_t = Transcript::new(b"integration");
    let (value, proof) =
        MockPcs::open(&pk, (&poly).into(), &point, &mut prover_t).unwrap();
    let mut verifier_t = Transcript::new(b"integration");
    assert!(MockPcs::verify(
        &vk,
        &commitment,
        &point,
        value,
        &proof,
        &mut verifier_t
    )
    .unwrap());
}
