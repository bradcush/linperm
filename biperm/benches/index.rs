// MockPcs::ProverKey is `()`, so the generic `prover_key::<MockPcs<_>>`
// binding is a unit value, expected for the generic-over-PCS bench shape.
#![allow(clippy::let_unit_value)]

mod common;
mod config;

use ark_bn254::{Fr, G1Projective};
use ark_std::rand::RngCore;
use ark_std::test_rng;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use biperm::permcore::{MockPcs, Permutation, PolynomialCommitment};
use biperm::{index, index_with, IndicatorRepr};
use hyrax::Hyrax;

use common::instance;

/// Generic SRS generation (untimed), so the bench times only
/// the indicator commitment inside `index`, not key setup.
fn prover_key<P: PolynomialCommitment<Fr>>(
    perm: &Permutation,
    rng: &mut impl RngCore,
) -> P::ProverKey {
    P::setup(perm.num_vars() * 3 / 2, rng).unwrap().0
}

fn bench(c: &mut Criterion) {
    // `index` preprocesses $\sigma$ once, builds + commits the
    // two $n^{1.5}$ indicators. The dense-Hyrax commit dominates,
    // so keep μ modest. Even μ only (BiPerm requires it).
    const MUS: [usize; 4] = [8, 10, 12, 14];

    let mut idx = c.benchmark_group("biperm_index");
    // The dense Hyrax MSM is slow;
    // 10 keeps wall-clock sane
    idx.sample_size(10);
    for mu in MUS {
        let mut rng = test_rng();
        let (perm, _, _) = instance(mu, &mut rng);
        let mpk = prover_key::<MockPcs<Fr>>(&perm, &mut rng);
        // One SRS; "shyrax" commits the indicators sparse, "dhyrax" forces
        // the dense path via `index_with`, same Hyrax, biperm picks the rep.
        let hpk = prover_key::<Hyrax<G1Projective>>(&perm, &mut rng);
        idx.bench_with_input(BenchmarkId::new("mock", mu), &mu, |b, _| {
            b.iter(|| index::<Fr, MockPcs<Fr>>(&mpk, &perm).unwrap())
        });
        idx.bench_with_input(BenchmarkId::new("shyrax", mu), &mu, |b, _| {
            b.iter(|| index::<Fr, Hyrax<G1Projective>>(&hpk, &perm).unwrap())
        });
        idx.bench_with_input(BenchmarkId::new("dhyrax", mu), &mu, |b, _| {
            b.iter(|| {
                index_with::<Fr, Hyrax<G1Projective>>(
                    &hpk,
                    &perm,
                    IndicatorRepr::Dense,
                )
                .unwrap()
            })
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
