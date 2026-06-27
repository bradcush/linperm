//! Shared instance for BiPerm benchmarks.

use ark_bn254::Fr;
use ark_ff::UniformRand;
use ark_poly::DenseMultilinearExtension;
use ark_std::rand::RngCore;
use criterion::Criterion;
use pprof::criterion::{Output, PProfProfiler};

use biperm::permcore::Permutation;

/// Criterion configured to emit a flamegraph of the timed closure under
/// `--profile-time`. Sampling at 100 Hz; the profiler is dormant on normal
/// `cargo bench` runs, so timing/report output is unaffected.
pub fn criterion() -> Criterion {
    Criterion::default()
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)))
}

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
