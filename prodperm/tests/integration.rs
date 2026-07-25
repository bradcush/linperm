use ark_bn254::{Fr, G1Projective};
use ark_ff::UniformRand;
use ark_poly::DenseMultilinearExtension;
use ark_std::test_rng;

use hyrax::Hyrax;
use prodperm::permcore::{
    MockPcs, Permutation, PolynomialCommitment, Transcript,
};
use prodperm::{index, prove, verify};

/// $\sigma$ on $B_4$ plus a consistent $(f, g)$ with $g(\sigma(x)) = f(x)$,
/// the same relation BiPerm's integration test builds, so both protocols
/// are exercised against an identical statement.
fn instance(
    rng: &mut impl ark_std::rand::RngCore,
) -> (
    Permutation,
    DenseMultilinearExtension<Fr>,
    DenseMultilinearExtension<Fr>,
) {
    let perm = Permutation::new(vec![
        // All indices 0-15 represented, w/o duplicates
        5, 3, 7, 1, 0, 6, 2, 4, 9, 11, 8, 10, 13, 15, 12, 14,
    ])
    .unwrap();
    let num_vars = perm.num_vars();
    let f_evals: Vec<Fr> =
        (0..(1 << num_vars)).map(|_| Fr::rand(rng)).collect();
    let mut g_evals = vec![Fr::from(0u64); perm.size()];
    for x in 0..perm.size() {
        g_evals[perm.apply(x)] = f_evals[x];
    }
    let f = DenseMultilinearExtension::from_evaluations_vec(num_vars, f_evals);
    let g = DenseMultilinearExtension::from_evaluations_vec(num_vars, g_evals);
    (perm, f, g)
}

#[test]
// ProdPerm index, prove, and verify over the
// mock PCS implementation, just polynomials
fn prodperm_round_trip_mock() {
    let mut rng = test_rng();
    let (perm, f, g) = instance(&mut rng);
    // SRS covers the largest commitment, the
    // $(\mu + 1)$-variate product tree.
    let (pk, vk) = MockPcs::<Fr>::setup(perm.num_vars() + 1, &mut rng).unwrap();
    // Preprocess $\sigma$ once; deployer trusted in this test
    let (p_idx, v_idx) = index::<Fr, MockPcs<Fr>>(&pk, &perm).unwrap();
    let mut prover_t = Transcript::new(b"integration");
    let proof = prove(&pk, &p_idx, &f, &g, &mut prover_t).unwrap();
    let mut verifier_t = Transcript::new(b"integration");
    verify(&vk, &v_idx, &proof, &mut verifier_t).unwrap();
}

#[test]
// The same protocol over a real Hyrax backend, exercising the generic
// `PolynomialCommitment` boundary end-to-end with actual curve ops.
fn prodperm_round_trip_hyrax() {
    let mut rng = test_rng();
    let (perm, f, g) = instance(&mut rng);
    let (pk, vk) =
        Hyrax::<G1Projective>::setup(perm.num_vars() + 1, &mut rng).unwrap();
    let (p_idx, v_idx) = index::<Fr, Hyrax<G1Projective>>(&pk, &perm).unwrap();
    let mut prover_t = Transcript::new(b"integration");
    let proof = prove(&pk, &p_idx, &f, &g, &mut prover_t).unwrap();
    let mut verifier_t = Transcript::new(b"integration");
    verify(&vk, &v_idx, &proof, &mut verifier_t).unwrap();
}
