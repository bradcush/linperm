mod common;

use ark_bn254::{Fr, G1Projective};
use ark_poly::DenseMultilinearExtension;
use ark_serialize::CanonicalSerialize;
use ark_std::rand::RngCore;
use ark_std::test_rng;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use biperm::permcore::{
    MockPcs, Permutation, PolynomialCommitment, Transcript,
};
use biperm::{index, prove, verify, BiPermProof, BiPermVerifierIndex};
use hyrax::Hyrax;

use common::instance;

/// Generic over the PCS: setup + index + prove (untimed).
fn prep<P: PolynomialCommitment<Fr>>(
    perm: &Permutation,
    f: &DenseMultilinearExtension<Fr>,
    g: &DenseMultilinearExtension<Fr>,
    rng: &mut impl RngCore,
) -> (
    P::VerifierKey,
    BiPermVerifierIndex<Fr, P>,
    BiPermProof<Fr, P>,
) {
    let (pk, vk) = P::setup(perm.num_vars() * 3 / 2, rng).unwrap();
    let (p_idx, v_idx) = index::<Fr, P>(&pk, perm).unwrap();
    let mut t = Transcript::new(b"bench");
    let proof = prove(&pk, &p_idx, f, g, &mut t).unwrap();
    (vk, v_idx, proof)
}

/// Bytes the verifier holds/receives: index commitments +
/// the proof. So we can check/vefify sizes are consistent.
fn footprint<P: PolynomialCommitment<Fr>>(
    vidx: &BiPermVerifierIndex<Fr, P>,
    proof: &BiPermProof<Fr, P>,
) -> usize {
    // Concat, compressed, for verifier
    vidx.ind_l_commit.compressed_size()
        + vidx.ind_r_commit.compressed_size()
        + proof.f_commit.compressed_size()
        + proof.g_commit.compressed_size()
        + proof.sumcheck.round_polys.compressed_size()
        + proof.g_at_alpha.compressed_size()
        + proof.g_opening.compressed_size()
        + proof.f_at_r.compressed_size()
        + proof.f_opening.compressed_size()
        + proof.ind_l_at_r.compressed_size()
        + proof.ind_l_opening.compressed_size()
        + proof.ind_r_at_r.compressed_size()
        + proof.ind_r_opening.compressed_size()
}

fn bench(c: &mut Criterion) {
    // BiPerm needs even $\mu$. 8,10 below
    // the verify crossover; 12,14 above.
    const MUS: [usize; 4] = [8, 10, 12, 14];

    let mut group = c.benchmark_group("biperm_verify");
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

    // Size table (deterministic, printed once, not part of the timing).
    // Here for reference but really doesn't below in benches. We could
    // choose to iterate bytes if we want to make more of a timed bench.
    println!("\nverifier footprint (bytes)");
    println!("{:>4}  {:>12}  {:>12}", "mu", "mock", "hyrax");
    for (mu, m, h) in sizes {
        println!("{mu:>4}  {m:>12}  {h:>12}");
    }
}

criterion_group! {
    name = benches;
    config = common::criterion();
    targets = bench
}
criterion_main!(benches);
