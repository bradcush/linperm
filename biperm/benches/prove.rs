// MockPcs::ProverKey is `()`, so the generic `prover_key::<MockPcs<_>>`
// binding is a unit value, expected for the generic-over-PCS bench shape.
#![allow(clippy::let_unit_value)]

mod common;

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

/// Generic SRS generation (untimed), for the index
/// bench, which times the indicator commitment itself.
fn prover_key<P: PolynomialCommitment<Fr>>(
    perm: &Permutation,
    rng: &mut impl RngCore,
) -> P::ProverKey {
    P::setup(perm.num_vars() * 3 / 2, rng).unwrap().0
}

/// SRS + a prebuilt prover index (untimed), for the prove bench,
/// which times only `prove` over a fixed, already-committed $\sigma$.
fn prover_index<P: PolynomialCommitment<Fr>>(
    perm: &Permutation,
    rng: &mut impl RngCore,
) -> (P::ProverKey, BiPermProverIndex<Fr, P>) {
    let pk = P::setup(perm.num_vars() * 3 / 2, rng).unwrap().0;
    let (p_idx, _) = index::<Fr, P>(&pk, perm).unwrap();
    (pk, p_idx)
}

fn bench(c: &mut Criterion) {
    // Proving puts the dense commit/open in the hot loop, so keep μ modest.
    // $\mu=14$ dense-Hyrax index is ~seconds *per iteration* (exactly the cost
    // the sparse backend will erase). Even μ only (BiPerm requires it).
    const MUS: [usize; 3] = [8, 10, 12];

    // index: builds + commits the two $n^{1.5}$ indicators
    let mut idx = c.benchmark_group("biperm_index");
    // The dense Hyrax MSM is slow;
    // 10 keeps wall-clock sane
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

    // prove: commit f/g, sumcheck, open f/g + indicators
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
    config = common::criterion();
    targets = bench
}
criterion_main!(benches);
