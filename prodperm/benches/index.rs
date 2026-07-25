// MockPcs::ProverKey is `()`, so the generic `prover_key::<MockPcs<_>>`
// binding is a unit value, expected for the generic-over-PCS bench shape.
#![allow(clippy::let_unit_value)]

mod common;
mod config;

use ark_bn254::{Fr, G1Projective};
use ark_std::rand::RngCore;
use ark_std::test_rng;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use hyrax::Hyrax;
use prodperm::index;
use prodperm::permcore::{MockPcs, Permutation, PolynomialCommitment};

use common::instance;

/// Generic SRS generation (untimed), so the bench times only the
/// $\sigma$ commitment inside `index`, not key setup. The SRS is
/// sized for the largest polynomial ProdPerm commits, the
/// $(\mu + 1)$-variate product tree built during `prove`.
fn prover_key<P: PolynomialCommitment<Fr>>(
    perm: &Permutation,
    rng: &mut impl RngCore,
) -> P::ProverKey {
    P::setup(perm.num_vars() + 1, rng).unwrap().0
}

fn bench(c: &mut Criterion) {
    // `index` preprocesses $\sigma$ once, building and committing the single
    // $\mu$-variate $\tilde{s}_\sigma$. Same $\mu$ values as the BiPerm
    // benches (which need even $\mu$) so the two are read side by side.
    const MUS: [usize; 4] = [8, 10, 12, 14];

    let mut idx = c.benchmark_group("prodperm_index");
    idx.sample_size(10);
    for mu in MUS {
        let mut rng = test_rng();
        let (perm, _, _) = instance(mu, &mut rng);
        let mpk = prover_key::<MockPcs<Fr>>(&perm, &mut rng);
        let hpk = prover_key::<Hyrax<G1Projective>>(&perm, &mut rng);
        idx.bench_with_input(BenchmarkId::new("mock", mu), &mu, |b, _| {
            b.iter(|| index::<Fr, MockPcs<Fr>>(&mpk, &perm).unwrap())
        });
        idx.bench_with_input(BenchmarkId::new("hyrax", mu), &mu, |b, _| {
            b.iter(|| index::<Fr, Hyrax<G1Projective>>(&hpk, &perm).unwrap())
        });
    }
    idx.finish();
}

criterion_group! {
    name = benches;
    config = config::profiled();
    targets = bench
}
criterion_main!(benches);
