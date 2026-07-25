mod common;
mod config;

use ark_bn254::{Fr, G1Projective};
use ark_poly::DenseMultilinearExtension;
use ark_serialize::CanonicalSerialize;
use ark_std::rand::RngCore;
use ark_std::test_rng;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use hyrax::Hyrax;
use prodperm::permcore::{
    MockPcs, Permutation, PolynomialCommitment, Transcript,
};
use prodperm::{
    index, prove, verify, Opening, ProdPermProof, ProdPermVerifierIndex,
};

use common::instance;

/// Generic over the PCS: setup + index + prove (untimed).
fn prep<P: PolynomialCommitment<Fr>>(
    perm: &Permutation,
    f: &DenseMultilinearExtension<Fr>,
    g: &DenseMultilinearExtension<Fr>,
    rng: &mut impl RngCore,
) -> (
    P::VerifierKey,
    ProdPermVerifierIndex<Fr, P>,
    ProdPermProof<Fr, P>,
) {
    let (pk, vk) = P::setup(perm.num_vars() + 1, rng).unwrap();
    let (p_idx, v_idx) = index::<Fr, P>(&pk, perm).unwrap();
    let mut t = Transcript::new(b"bench");
    let proof = prove(&pk, &p_idx, f, g, &mut t).unwrap();
    (vk, v_idx, proof)
}

/// An opening costs its claimed value and proof.
fn opening<P: PolynomialCommitment<Fr>>(open: &Opening<Fr, P>) -> usize {
    open.value.compressed_size() + open.proof.compressed_size()
}

/// Bytes the verifier holds/receives: index commitment +
/// the proof. So we can check/verify sizes are consistent.
fn footprint<P: PolynomialCommitment<Fr>>(
    vidx: &ProdPermVerifierIndex<Fr, P>,
    proof: &ProdPermProof<Fr, P>,
) -> usize {
    // Concat, compressed, for verifier
    vidx.sigma_commit.compressed_size()
        + proof.f_commit.compressed_size()
        + proof.g_commit.compressed_size()
        + proof.v_commit.compressed_size()
        + proof.sumcheck.round_polys.compressed_size()
        + opening(&proof.f)
        + opening(&proof.g)
        + opening(&proof.sigma)
        + opening(&proof.v_bot)
        + opening(&proof.v_top)
        + opening(&proof.v_left)
        + opening(&proof.v_right)
        + opening(&proof.v_root)
}

fn bench(c: &mut Criterion) {
    // Same $\mu$ values as the BiPerm benches.
    const MUS: [usize; 4] = [8, 10, 12, 14];

    let mut group = c.benchmark_group("prodperm_verify");
    let mut sizes = Vec::new();

    for mu in MUS {
        let mut rng = test_rng();
        let (perm, f, g) = instance(mu, &mut rng);
        // Outside the timed region, just benching verify
        let (mvk, mvidx, mproof) = prep::<MockPcs<Fr>>(&perm, &f, &g, &mut rng);
        let (hvk, hvidx, hproof) =
            prep::<Hyrax<G1Projective>>(&perm, &f, &g, &mut rng);
        sizes.push((
            mu,
            footprint::<MockPcs<Fr>>(&mvidx, &mproof),
            footprint::<Hyrax<G1Projective>>(&hvidx, &hproof),
        ));
        group.bench_with_input(BenchmarkId::new("mock", mu), &mu, |b, _| {
            b.iter(|| {
                let mut t = Transcript::new(b"bench");
                verify(&mvk, &mvidx, &mproof, &mut t).unwrap();
            })
        });
        group.bench_with_input(BenchmarkId::new("hyrax", mu), &mu, |b, _| {
            b.iter(|| {
                let mut t = Transcript::new(b"bench");
                verify(&hvk, &hvidx, &hproof, &mut t).unwrap();
            })
        });
    }
    group.finish();

    // Size table (deterministic, printed once, not part of the
    // timing). Directly comparable to BiPerm table at equal $\mu$.
    println!("\nverifier footprint (bytes)");
    println!("{:>4}  {:>12}  {:>12}", "mu", "mock", "hyrax");
    for (mu, m, h) in sizes {
        println!("{mu:>4}  {m:>12}  {h:>12}");
    }
}

criterion_group! {
    name = benches;
    config = config::profiled();
    targets = bench
}
criterion_main!(benches);
