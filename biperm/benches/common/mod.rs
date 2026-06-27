//! Shared instance for BiPerm benchmarks.

use ark_bn254::Fr;
use ark_ff::UniformRand;
use ark_poly::DenseMultilinearExtension;
use ark_std::rand::RngCore;

use biperm::permcore::Permutation;

/// Create an instance for benchmarking,
/// the permutation and dense MLEs.
pub fn instance(
    mu: usize,
    rng: &mut impl RngCore,
) -> (
    Permutation,
    DenseMultilinearExtension<Fr>,
    DenseMultilinearExtension<Fr>,
) {
    let n = 1usize << mu;
    let perm = Permutation::new((0..n).map(|x| (x + 1) % n).collect()).unwrap();
    let f_evals: Vec<Fr> = (0..n).map(|_| Fr::rand(rng)).collect();
    let mut g_evals = vec![Fr::from(0u64); n];
    for x in 0..n {
        g_evals[perm.apply(x)] = f_evals[x];
    }
    (
        perm,
        DenseMultilinearExtension::from_evaluations_vec(mu, f_evals),
        DenseMultilinearExtension::from_evaluations_vec(mu, g_evals),
    )
}
