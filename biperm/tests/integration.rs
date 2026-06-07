use ark_bn254::Fr;
use ark_ff::UniformRand;
use ark_poly::DenseMultilinearExtension;
use ark_std::test_rng;

use biperm::permcore::{
    MockPcs, Permutation, PolynomialCommitment, Transcript,
};
use biperm::{index, prove, verify};

#[test]
// BiPerm index, prove, and verify public API. Checks the
// protocol works end-to-end via the public surface.
fn biperm_round_trip() {
    let mut rng = test_rng();
    // $\sigma$ (permutation) on $B_4$ (16 elements).
    // $\mu = 4$ splits into two halves of 2 bits each.
    let perm = Permutation::new(vec![
        // All indices 0-15 represented, w/o duplicates
        5, 3, 7, 1, 0, 6, 2, 4, 9, 11, 8, 10, 13, 15, 12, 14,
    ])
    .unwrap();
    let num_vars = perm.num_vars();
    // Build $(f, g)$ with $g(\sigma(x)) = f(x)$,
    // (eg. $g = f \cdot \sigma^{-1}$), the
    // relation Lemma 4 lets BiPerm prove.
    let f_evals: Vec<Fr> =
        (0..(1 << num_vars)).map(|_| Fr::rand(&mut rng)).collect();
    let mut g_evals = vec![Fr::from(0u64); perm.size()];
    for x in 0..perm.size() {
        g_evals[perm.apply(x)] = f_evals[x];
    }
    // Create the multilinear extensions of $f$ and $g$
    let f = DenseMultilinearExtension::from_evaluations_vec(num_vars, f_evals);
    let g = DenseMultilinearExtension::from_evaluations_vec(num_vars, g_evals);
    // SRS covers largest commitment, $3\mu/2$-variate sparse indicators.
    let (pk, vk) = MockPcs::<Fr>::setup(num_vars * 3 / 2, &mut rng).unwrap();
    // Preprocess $\sigma$ once; deployer trusted in this test
    let (p_idx, v_idx) = index::<Fr, MockPcs<Fr>>(&pk, &perm).unwrap();
    let mut prover_t = Transcript::new(b"integration");
    let proof = prove(&pk, &p_idx, &f, &g, &mut prover_t).unwrap();
    let mut verifier_t = Transcript::new(b"integration");
    verify(&vk, &v_idx, &proof, &mut verifier_t).unwrap();
}
