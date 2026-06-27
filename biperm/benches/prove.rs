// MockPcs::ProverKey is `()`, so the `pk` from `prover_index::<MockPcs<_>>`
// is a unit value, expected for the generic-over-PCS bench shape.
#![allow(clippy::let_unit_value)]

mod common;
mod config;

use ark_bn254::{Fr, G1Projective};
use ark_std::rand::RngCore;
use ark_std::test_rng;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use biperm::permcore::{
    MockPcs, Permutation, PolynomialCommitment, Transcript,
};
use biperm::{index, prove, BiPermProverIndex};
use hyrax::Hyrax;

use common::instance;

/// SRS + a prebuilt prover index (untimed), so the bench times
/// only `prove` over a fixed, already-committed $\sigma$.
fn prover_index<P: PolynomialCommitment<Fr>>(
    perm: &Permutation,
    rng: &mut impl RngCore,
) -> (P::ProverKey, BiPermProverIndex<Fr, P>) {
    let pk = P::setup(perm.num_vars() * 3 / 2, rng).unwrap().0;
    let (p_idx, _) = index::<Fr, P>(&pk, perm).unwrap();
    (pk, p_idx)
}

fn bench(c: &mut Criterion) {
    // Commit f/g, sumcheck, open f/g + indicators. Keep μ modest, dense
    // commit/open is in the hot loop. Even $\mu$ only (BiPerm requires).
    const MUS: [usize; 3] = [8, 10, 12];

    let mut prv = c.benchmark_group("biperm_prove");
    for mu in MUS {
        let mut rng = test_rng();
        let (perm, f, g) = instance(mu, &mut rng);
        let (mpk, mp_idx) = prover_index::<MockPcs<Fr>>(&perm, &mut rng);
        let (hpk, hp_idx) =
            prover_index::<Hyrax<G1Projective>>(&perm, &mut rng);
        prv.bench_with_input(BenchmarkId::new("mock", mu), &mu, |b, _| {
            b.iter(|| {
                let mut t = Transcript::new(b"bench");
                prove(&mpk, &mp_idx, &f, &g, &mut t).unwrap()
            })
        });
        prv.bench_with_input(BenchmarkId::new("hyrax", mu), &mu, |b, _| {
            b.iter(|| {
                let mut t = Transcript::new(b"bench");
                prove(&hpk, &hp_idx, &f, &g, &mut t).unwrap()
            })
        });
    }
    prv.finish();
}

criterion_group! {
    name = benches;
    config = config::profiled();
    targets = bench
}
criterion_main!(benches);
