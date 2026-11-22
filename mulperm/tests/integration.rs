use ark_bn254::{Fr, G1Projective};
use ark_ff::UniformRand;
use ark_poly::DenseMultilinearExtension;
use ark_std::test_rng;

use hyrax::Hyrax;
use mulperm::permcore::{
    MockPcs, Permutation, PolynomialCommitment, Transcript,
};
use mulperm::{index, prove, verify};

/// $\sigma$ on $B_4$ (16 elements; $\mu = 4$, so the default $\ell = 2$)
/// plus a consistent $(f, g)$ with $g(\sigma(x)) = f(x)$, the relation
/// Lemma 4 lets MulPerm prove ($g = f \cdot \sigma^{-1}$).
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

/// The largest committed polynomial is
/// $\tilde{\sigma}$, over $\log \mu + \mu$ variables.
fn max_num_vars(perm: &Permutation) -> usize {
    let num_vars = perm.num_vars();
    num_vars.trailing_zeros() as usize + num_vars
}

#[test]
// MulPerm index, prove, and verify over the
// mock PCS implementation, just polynomials
fn mulperm_round_trip_mock() {
    let mut rng = test_rng();
    let (perm, f, g) = instance(&mut rng);
    let (pk, vk) = MockPcs::<Fr>::setup(max_num_vars(&perm), &mut rng).unwrap();
    let (p_idx, v_idx) = index::<Fr, MockPcs<Fr>>(&pk, &perm).unwrap();
    let mut prover_t = Transcript::new(b"integration");
    let proof = prove(&pk, &p_idx, &f, &g, &mut prover_t).unwrap();
    let mut verifier_t = Transcript::new(b"integration");
    verify(&vk, &v_idx, &proof, &mut verifier_t).unwrap();
}

#[test]
// The same protocol over a real (binding-only) Hyrax backend, exercising the
// generic `PolynomialCommitment` boundary end-to-end with actual curve ops.
fn mulperm_round_trip_hyrax() {
    let mut rng = test_rng();
    let (perm, f, g) = instance(&mut rng);
    let (pk, vk) =
        Hyrax::<G1Projective>::setup(max_num_vars(&perm), &mut rng).unwrap();
    let (p_idx, v_idx) = index::<Fr, Hyrax<G1Projective>>(&pk, &perm).unwrap();
    let mut prover_t = Transcript::new(b"integration");
    let proof = prove(&pk, &p_idx, &f, &g, &mut prover_t).unwrap();
    let mut verifier_t = Transcript::new(b"integration");
    verify(&vk, &v_idx, &proof, &mut verifier_t).unwrap();
}
