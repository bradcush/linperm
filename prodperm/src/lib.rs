//! ProdPerm: Grand product permutation protocol.
//!
//! Proves the same statement BiPerm does, $g(\sigma(x)) = f(x)$ for all
//! $x \in B_\mu$. ProdPerm compresses each $(index, value)$ pair with
//! challenges $\beta, \gamma$ and checks a grand product:
//!
//! $$\prod_{x \in B_\mu} \frac{f(x) + \beta \tilde{s}\_\sigma(x) + \gamma}{g(x) + \beta \tilde{s}\_{id}(x) + \gamma} = 1$$
//!
//! The numerator's factors encode the multiset $\\{(\sigma(x), f(x))\\}$
//! and the denominator's encode $\\{(y, g(y))\\}$. Equal products at random
//! $\beta, \gamma$ means those multisets agree except with probability
//! $O(n/|F|)$, and agreement forces $g(\sigma(x)) = f(x)$. The product
//! itself is discharged by [`permcore::prodcheck`], which commits one
//! $(\mu+1)$-variate product tree and reduces to a single zerocheck.
//!
//! # Shape
//!
//! Indexed like BiPerm: [`index`] preprocesses a fixed $\sigma$ once by
//! committing $\tilde{s}\_\sigma$ (one $\mu$-variate polynomial, against
//! BiPerm's two $3\mu/2$-variate ones), and the verifier holds only
//! commitments and openings.
//!
//! $\tilde{s}\_{id}$ needs no commitment. The identity map $x \mapsto x$ is
//! $\sum_j 2^j x_j$, already multilinear, so the verifier evaluates it
//! directly in $O(\mu)$, saving an opening over a textbook rendering.
//!
//! # Cost
//!
//! The prover commits three $\mu$-or-larger polynomials ($f$, $g$, and the
//! double-width tree $v$) against BiPerm's two, and the proof carries eight
//! openings against BiPerm's four. Nothing here is batched yet; every
//! opening is an independent PCS call, on both sides of the comparison.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub use permcore;

use alloc::vec::Vec;

use ark_ff::PrimeField;
use ark_poly::DenseMultilinearExtension;
use permcore::prodcheck::{
    self, ProdcheckError, ProdcheckEvals, ProdcheckPoints,
};
use permcore::sumcheck::{SumcheckError, SumcheckProof};
use permcore::{MleRef, Permutation, PolynomialCommitment, Transcript};
use tracing::info_span;

/// PCS opening, claimed evaluation, proof backing it.
pub struct Opening<F: PrimeField, P: PolynomialCommitment<F>> {
    pub value: F,
    pub proof: P::Proof,
}

/// ProdPerm proof: commitments to $f$, $g$, and the product
/// tree $v$, the product-check sumcheck messages, and eight
/// openings, $f$, $g$, $\tilde{s}\_\sigma$ at the sumcheck point
/// $\rho$ plus the five the product check needs from $v$.
pub struct ProdPermProof<F: PrimeField, P: PolynomialCommitment<F>> {
    pub f_commit: P::Commitment,
    pub g_commit: P::Commitment,
    pub v_commit: P::Commitment,
    pub sumcheck: SumcheckProof<F>,
    pub f: Opening<F, P>,
    pub g: Opening<F, P>,
    pub sigma: Opening<F, P>,
    pub v_bot: Opening<F, P>,
    pub v_top: Opening<F, P>,
    pub v_left: Opening<F, P>,
    pub v_right: Opening<F, P>,
    pub v_root: Opening<F, P>,
}

/// Built once by [`index`] and reused across proofs.
pub struct ProdPermProverIndex<F: PrimeField, P: PolynomialCommitment<F>> {
    /// $\tilde{s}\_\sigma$, the image of $\sigma$ as field values.
    pub sigma: DenseMultilinearExtension<F>,
    pub sigma_commit: P::Commitment,
    /// $\tilde{s}\_{id}$'s evaluation table. Cached here
    /// rather than rebuilt per proof because it depends
    /// on neither the instance nor a challenge.
    pub identity: Vec<F>,
}

/// Verifier half of the index: $\mu$ and the $\sigma$ commitment. Indexing
/// is deterministic and public, so anyone can recompute this from $\sigma$.
pub struct ProdPermVerifierIndex<F: PrimeField, P: PolynomialCommitment<F>> {
    pub num_vars: usize,
    pub sigma_commit: P::Commitment,
}

#[derive(Debug, PartialEq, Eq)]
pub enum ProdPermError<E> {
    Sumcheck(SumcheckError),
    /// The product tree could not be built, in practice a denominator
    /// $g(x) + \beta x + \gamma$ that vanished on the cube.
    Prodcheck(ProdcheckError),
    /// PCS operation
    /// produced an error
    Pcs(E),
    /// PCS opening verification failed
    PcsVerifyFailed,
    /// The tree's root did not open to $1$: the
    /// products differ, so the multisets differ.
    RootNotOne,
    /// The product-check constraints did not
    /// match the sumcheck's final claim.
    FinalCheckFailed,
}

impl<E> From<SumcheckError> for ProdPermError<E> {
    fn from(e: SumcheckError) -> Self {
        Self::Sumcheck(e)
    }
}

impl<E> From<ProdcheckError> for ProdPermError<E> {
    fn from(e: ProdcheckError) -> Self {
        Self::Prodcheck(e)
    }
}

/// Output of [`index`], the
/// prover and verifier halves.
pub type ProdPermIndex<F, P> =
    (ProdPermProverIndex<F, P>, ProdPermVerifierIndex<F, P>);

/// Preprocess a fixed $\sigma$: build and commit
/// $\tilde{s}\_\sigma$, and cache $\tilde{s}\_{id}$. Runs
/// once per permutation, independent of any $f, g$ instance.
pub fn index<F: PrimeField, P: PolynomialCommitment<F>>(
    pk: &P::ProverKey,
    perm: &Permutation,
) -> Result<ProdPermIndex<F, P>, ProdPermError<P::Error>> {
    let num_vars = perm.num_vars();
    let sigma = DenseMultilinearExtension::from_evaluations_vec(
        num_vars,
        perm.image_evaluations::<F>(),
    );
    let sigma_commit =
        P::commit(pk, (&sigma).into()).map_err(ProdPermError::Pcs)?;
    let identity = Permutation::identity(num_vars).image_evaluations::<F>();
    Ok((
        ProdPermProverIndex {
            sigma,
            sigma_commit: sigma_commit.clone(),
            identity,
        },
        ProdPermVerifierIndex {
            num_vars,
            sigma_commit,
        },
    ))
}

/// Prove $g(\sigma(x)) = f(x)$ for all $x \in B_\mu$.
///
/// Commits $f, g$, absorbs them with the index commitment before squeezing
/// $\beta, \gamma$, builds the two compressed multiset polynomials and their
/// product tree, and commits the tree before the product check draws any
/// challenge of its own. Opens everything at the resulting sumcheck point.
///
/// # Panics
///
/// Panics if $f$ or $g$ disagrees with the index on $\mu$.
pub fn prove<F: PrimeField, P: PolynomialCommitment<F>>(
    pk: &P::ProverKey,
    index: &ProdPermProverIndex<F, P>,
    f: &DenseMultilinearExtension<F>,
    g: &DenseMultilinearExtension<F>,
    transcript: &mut Transcript,
) -> Result<ProdPermProof<F, P>, ProdPermError<P::Error>> {
    let num_vars = index.sigma.num_vars;
    assert_eq!(num_vars, f.num_vars, "f num_vars must match μ");
    assert_eq!(num_vars, g.num_vars, "g num_vars must match μ");
    // Span names match BiPerm's so the phase breakdowns line up.
    // Nothing forces this, see if we can find something better.
    let commit_span = info_span!("commit").entered();
    let f_commit = P::commit(pk, f.into()).map_err(ProdPermError::Pcs)?;
    let g_commit = P::commit(pk, g.into()).map_err(ProdPermError::Pcs)?;
    drop(commit_span);
    transcript.append(b"sigma_commit", &index.sigma_commit);
    transcript.append(b"f_commit", &f_commit);
    transcript.append(b"g_commit", &g_commit);
    let beta: F = transcript.challenge(b"beta");
    let gamma: F = transcript.challenge(b"gamma");
    let aux_span = info_span!("aux").entered();
    let p = compress(f, &index.sigma.evaluations, beta, gamma);
    let q = compress(g, &index.identity, beta, gamma);
    let v = prodcheck::build(&p, &q)?;
    drop(aux_span);
    // $v$ must be bound before the product check squeezes anything,
    // otherwise the prover gets to pick its tree after seeing $\alpha$.
    let commit_span = info_span!("commit").entered();
    let v_commit = P::commit(pk, (&v).into()).map_err(ProdPermError::Pcs)?;
    drop(commit_span);
    transcript.append(b"v_commit", &v_commit);
    let sumcheck_span = info_span!("sumcheck").entered();
    let output = prodcheck::prove(&v, &p, &q, transcript);
    drop(sumcheck_span);
    // Everything needs $\rho$
    let rho = &output.challenges;
    let points = prodcheck::points(rho);
    let v_ref = MleRef::from(&v);
    let opens_span = info_span!("opens").entered();
    let f_open = open::<F, P>(pk, f.into(), rho, transcript)?;
    let g_open = open::<F, P>(pk, g.into(), rho, transcript)?;
    let sigma_open = open::<F, P>(pk, (&index.sigma).into(), rho, transcript)?;
    let v_bot = open::<F, P>(pk, v_ref, &points.bot, transcript)?;
    let v_top = open::<F, P>(pk, v_ref, &points.top, transcript)?;
    let v_left = open::<F, P>(pk, v_ref, &points.left, transcript)?;
    let v_right = open::<F, P>(pk, v_ref, &points.right, transcript)?;
    let v_root = open::<F, P>(pk, v_ref, &points.root, transcript)?;
    drop(opens_span);
    Ok(ProdPermProof {
        f_commit,
        g_commit,
        v_commit,
        sumcheck: output.proof,
        f: f_open,
        g: g_open,
        sigma: sigma_open,
        v_bot,
        v_top,
        v_left,
        v_right,
        v_root,
    })
}

/// Verify a ProdPerm proof. The verifier holds only the PCS verifier
/// key and the [`ProdPermVerifierIndex`]; $f$, $g$, and $\sigma$ are
/// accessible only through commitments and the proof's openings.
pub fn verify<F: PrimeField, P: PolynomialCommitment<F>>(
    vk: &P::VerifierKey,
    index: &ProdPermVerifierIndex<F, P>,
    proof: &ProdPermProof<F, P>,
    transcript: &mut Transcript,
) -> Result<(), ProdPermError<P::Error>> {
    let num_vars = index.num_vars;
    transcript.append(b"sigma_commit", &index.sigma_commit);
    transcript.append(b"f_commit", &proof.f_commit);
    transcript.append(b"g_commit", &proof.g_commit);
    let beta: F = transcript.challenge(b"beta");
    let gamma: F = transcript.challenge(b"gamma");
    transcript.append(b"v_commit", &proof.v_commit);
    let out = prodcheck::verify::<F>(num_vars, &proof.sumcheck, transcript)?;
    let rho = &out.challenges;
    // Same order as `prove` opened them, the transcript is shared
    check(vk, &proof.f_commit, rho, &proof.f, transcript)?;
    check(vk, &proof.g_commit, rho, &proof.g, transcript)?;
    check(vk, &index.sigma_commit, rho, &proof.sigma, transcript)?;
    let ProdcheckPoints {
        bot,
        top,
        left,
        right,
        root,
    } = &out.points;
    check(vk, &proof.v_commit, bot, &proof.v_bot, transcript)?;
    check(vk, &proof.v_commit, top, &proof.v_top, transcript)?;
    check(vk, &proof.v_commit, left, &proof.v_left, transcript)?;
    check(vk, &proof.v_commit, right, &proof.v_right, transcript)?;
    check(vk, &proof.v_commit, root, &proof.v_root, transcript)?;
    // The grand product itself. Everything above only says the
    // tree is well formed; this is what says the multisets match.
    if proof.v_root.value != F::one() {
        return Err(ProdPermError::RootNotOne);
    }
    // $p$ and $q$ are virtual, reconstructed from the openings we have.
    // $\tilde{s}_{id}$ is public, so it costs an evaluation, not an opening.
    let p_at_rho = proof.f.value + beta * proof.sigma.value + gamma;
    let q_at_rho = proof.g.value + beta * identity_at(rho) + gamma;
    let evals = ProdcheckEvals {
        bot: proof.v_bot.value,
        top: proof.v_top.value,
        left: proof.v_left.value,
        right: proof.v_right.value,
    };
    // Batched (linear-combination), w/ alpha,
    // single zerocheck over two constraints.
    if !out.final_check(&evals, p_at_rho, q_at_rho) {
        return Err(ProdPermError::FinalCheckFailed);
    }
    Ok(())
}

/// $values(x) + \beta \cdot ids\[x\] + \gamma$ over the cube: one
/// field element per point, compressing the pair $(id, value)$ so
/// a multiset of pairs becomes a multiset of scalars.
fn compress<F: PrimeField>(
    values: &DenseMultilinearExtension<F>,
    ids: &[F],
    beta: F,
    gamma: F,
) -> DenseMultilinearExtension<F> {
    let evals: Vec<F> = values
        .evaluations
        .iter()
        .zip(ids)
        .map(|(value, id)| *value + beta * id + gamma)
        .collect();
    DenseMultilinearExtension::from_evaluations_vec(values.num_vars, evals)
}

/// $\tilde{s}\_{id}(\rho) = \sum_j 2^j \rho_j$.
///
/// The identity map is already multilinear (degree one in each
/// variable), so this closed form *is* its multilinear extension.
/// Little-endian, matching the index order everywhere else.
fn identity_at<F: PrimeField>(rho: &[F]) -> F {
    let mut acc = F::zero();
    let mut weight = F::one();
    for r in rho {
        acc += weight * r;
        weight.double_in_place();
    }
    acc
}

/// Open `poly` at `point`, pairing the value with its proof.
fn open<F: PrimeField, P: PolynomialCommitment<F>>(
    pk: &P::ProverKey,
    poly: MleRef<'_, F>,
    point: &[F],
    transcript: &mut Transcript,
) -> Result<Opening<F, P>, ProdPermError<P::Error>> {
    let (value, proof) =
        P::open(pk, poly, point, transcript).map_err(ProdPermError::Pcs)?;
    Ok(Opening { value, proof })
}

/// Verify one opening against its commitment, mapping
/// a false result to [`ProdPermError::PcsVerifyFailed`].
fn check<F: PrimeField, P: PolynomialCommitment<F>>(
    vk: &P::VerifierKey,
    commitment: &P::Commitment,
    point: &[F],
    opening: &Opening<F, P>,
    transcript: &mut Transcript,
) -> Result<(), ProdPermError<P::Error>> {
    let ok = P::verify(
        vk,
        commitment,
        point,
        opening.value,
        &opening.proof,
        transcript,
    )
    .map_err(ProdPermError::Pcs)?;
    if ok {
        Ok(())
    } else {
        Err(ProdPermError::PcsVerifyFailed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_ff::UniformRand;
    use ark_poly::Polynomial;
    use ark_std::test_rng;
    use permcore::MockPcs;

    /// $\sigma$ on $B_\mu$ with a consistent
    /// $(f, g)$, $g(\sigma(x)) = f(x)$.
    fn instance(
        num_vars: usize,
        rng: &mut impl ark_std::rand::RngCore,
    ) -> (
        Permutation,
        DenseMultilinearExtension<Fr>,
        DenseMultilinearExtension<Fr>,
    ) {
        let n = 1usize << num_vars;
        let perm =
            Permutation::new((0..n).map(|x| (x + 1) % n).collect()).unwrap();
        let f_evals: Vec<Fr> = (0..n).map(|_| Fr::rand(rng)).collect();
        let mut g_evals = alloc::vec![Fr::from(0u64); n];
        for x in 0..n {
            g_evals[perm.apply(x)] = f_evals[x];
        }
        (
            perm,
            DenseMultilinearExtension::from_evaluations_vec(num_vars, f_evals),
            DenseMultilinearExtension::from_evaluations_vec(num_vars, g_evals),
        )
    }

    #[test]
    // The closed form really is the MLE of $x \mapsto x$, which is
    // what lets the verifier skip committing $\tilde{s}_{id}$.
    fn identity_at_matches_the_extension() {
        let mut rng = test_rng();
        let num_vars = 4;
        let table = Permutation::identity(num_vars).image_evaluations::<Fr>();
        let mle =
            DenseMultilinearExtension::from_evaluations_vec(num_vars, table);
        let rho: Vec<Fr> = (0..num_vars).map(|_| Fr::rand(&mut rng)).collect();
        assert_eq!(identity_at(&rho), mle.evaluate(&rho));
    }

    #[test]
    fn round_trip_accepts_consistent_instance() {
        let mut rng = test_rng();
        let num_vars = 5;
        let (perm, f, g) = instance(num_vars, &mut rng);
        let (pk, vk) = MockPcs::<Fr>::setup(num_vars + 1, &mut rng).unwrap();
        let (p_idx, v_idx) = index::<Fr, MockPcs<Fr>>(&pk, &perm).unwrap();
        let mut p_t = Transcript::new(b"prodperm");
        let proof = prove(&pk, &p_idx, &f, &g, &mut p_t).unwrap();
        let mut v_t = Transcript::new(b"prodperm");
        verify(&vk, &v_idx, &proof, &mut v_t).unwrap();
    }

    #[test]
    // $g$ inconsistent with $\sigma$: the multisets differ, so the
    // grand product isn't one and the root opening reveals that.
    // Honest prover given an incorrect $g$, so a valid tree.
    fn rejects_mismatched_g() {
        let mut rng = test_rng();
        let num_vars = 4;
        let (perm, f, mut g) = instance(num_vars, &mut rng);
        g.evaluations[2] += Fr::from(1u64);
        let (pk, vk) = MockPcs::<Fr>::setup(num_vars + 1, &mut rng).unwrap();
        let (p_idx, v_idx) = index::<Fr, MockPcs<Fr>>(&pk, &perm).unwrap();
        let mut p_t = Transcript::new(b"prodperm");
        let proof = prove(&pk, &p_idx, &f, &g, &mut p_t).unwrap();
        let mut v_t = Transcript::new(b"prodperm");
        assert_eq!(
            verify(&vk, &v_idx, &proof, &mut v_t),
            Err(ProdPermError::RootNotOne)
        );
    }

    #[test]
    // A different $\sigma$ than the one indexed. The index
    // commitment is absorbed before $\beta, \gamma$, so this cannot
    // even reach the product check with a consistent transcript.
    fn rejects_mismatched_index() {
        let mut rng = test_rng();
        let num_vars = 4;
        let (perm, f, g) = instance(num_vars, &mut rng);
        let n = perm.size();
        let other =
            Permutation::new((0..n).map(|x| (x + 2) % n).collect()).unwrap();
        let (pk, vk) = MockPcs::<Fr>::setup(num_vars + 1, &mut rng).unwrap();
        let (p_idx, _) = index::<Fr, MockPcs<Fr>>(&pk, &perm).unwrap();
        let (_, v_idx) = index::<Fr, MockPcs<Fr>>(&pk, &other).unwrap();
        let mut p_t = Transcript::new(b"prodperm");
        let proof = prove(&pk, &p_idx, &f, &g, &mut p_t).unwrap();
        let mut v_t = Transcript::new(b"prodperm");
        let err = verify(&vk, &v_idx, &proof, &mut v_t).unwrap_err();
        assert!(matches!(
            err,
            ProdPermError::Sumcheck(SumcheckError::RoundCheckFailed { .. })
        ));
    }

    /// A named single-field mutation applied to a proof before verifying.
    type Tamper = (&'static str, fn(&mut ProdPermProof<Fr, MockPcs<Fr>>));

    #[test]
    // Tampering with an opened value after the fact: the PCS catches it before
    // any of the protocol's own checks run. `final_check` is one equation, so
    // a prover holding one unbound value can solve for whatever satisfies it.
    fn rejects_tampered_openings() {
        let mut rng = test_rng();
        let num_vars = 4;
        let (perm, f, g) = instance(num_vars, &mut rng);
        let (pk, vk) = MockPcs::<Fr>::setup(num_vars + 1, &mut rng).unwrap();
        let (p_idx, v_idx) = index::<Fr, MockPcs<Fr>>(&pk, &perm).unwrap();
        let tampers: [Tamper; 8] = [
            ("f", |p| p.f.value += Fr::from(1u64)),
            ("g", |p| p.g.value += Fr::from(1u64)),
            ("sigma", |p| p.sigma.value += Fr::from(1u64)),
            ("v_bot", |p| p.v_bot.value += Fr::from(1u64)),
            ("v_top", |p| p.v_top.value += Fr::from(1u64)),
            ("v_left", |p| p.v_left.value += Fr::from(1u64)),
            ("v_right", |p| p.v_right.value += Fr::from(1u64)),
            ("v_root", |p| p.v_root.value += Fr::from(1u64)),
        ];
        for (name, tamper) in tampers {
            let mut p_t = Transcript::new(b"prodperm");
            let mut proof = prove(&pk, &p_idx, &f, &g, &mut p_t).unwrap();
            tamper(&mut proof);
            let mut v_t = Transcript::new(b"prodperm");
            assert_eq!(
                verify(&vk, &v_idx, &proof, &mut v_t),
                Err(ProdPermError::PcsVerifyFailed),
                "tampering {name} was not caught by the PCS",
            );
        }
    }
}
