//! BiPerm linear-time permutation argument.
//!
//! Proves $f(\sigma(x)) = g(x)$ for all $x \in B_\mu$, where $\sigma$ is a
//! permutation on the boolean hypercube and $f, g$ are multilinear. The
//! reduction is via Lemma 4 of the paper:
//!
//! $$\sum_{x \in B_\mu} f(x) \cdot \tilde{\mathbb{1}}\_\sigma(x, \alpha) = g(\alpha)$$
//!
//! BiPerm factorizes the indicator as $\tilde{\mathbb{1}}\_\sigma(X, Y) = \tilde{\mathbb{1}}\_{\sigma_L}(X, Y_L) \cdot \tilde{\mathbb{1}}\_{\sigma_R}(X, Y_R)$
//! on the boolean cube, giving a degree-3 sumcheck on the product $f \cdot \tilde{\mathbb{1}}\_{\sigma_L} \cdot \tilde{\mathbb{1}}\_{\sigma_R}$.
//!
//! The protocol is *indexed* meaning [`index`] preprocesses a fixed $\sigma$
//! once, committing to the two sparse $3\mu/2$-variate indicator polynomials
//! ($n$ nonzero entries out of $n^{1.5}$, hence a friendly PCS requirement).
//!
//! The verifier never holds $f$, $g$, or $\sigma$ directly. They hold a PCS
//! verifier key, the [`BiPermVerifierIndex`] with the indicator commitments,
//! and the commitments and openings carried in the rest of the proof.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub use permcore;

use alloc::vec::Vec;

use ark_ff::PrimeField;
use ark_poly::{DenseMultilinearExtension, SparseMultilinearExtension};
use permcore::eq::eq_eval_table;
use permcore::sumcheck::{self, SumcheckError, SumcheckProof};
use permcore::{CoreError, Permutation, PolynomialCommitment, Transcript};
use tracing::info_span;

/// Perm proof: PCS commitments to $f, g$, the sumcheck transcript, and PCS
/// openings of $g$ at $\alpha$, $f$ at the sumcheck challenge $r$, and the
/// two indicator polynomials at $(r \Vert \alpha_L)$ / $(r \Vert \alpha_R)$.
pub struct BiPermProof<F: PrimeField, P: PolynomialCommitment<F>> {
    pub f_commit: P::Commitment,
    pub g_commit: P::Commitment,
    pub sumcheck: SumcheckProof<F>,
    pub g_at_alpha: F,
    pub g_opening: P::Proof,
    pub f_at_r: F,
    pub f_opening: P::Proof,
    pub ind_l_at_r: F,
    pub ind_l_opening: P::Proof,
    pub ind_r_at_r: F,
    pub ind_r_opening: P::Proof,
}

/// Prover half of the index for a fixed $\sigma$: the
/// sparse indicator polynomials plus their commitments.
/// Built once by [`index`] and reused across proofs.
pub struct BiPermProverIndex<F: PrimeField, P: PolynomialCommitment<F>> {
    pub perm: Permutation,
    pub ind_l: SparseMultilinearExtension<F>,
    pub ind_r: SparseMultilinearExtension<F>,
    pub ind_l_commit: P::Commitment,
    pub ind_r_commit: P::Commitment,
}

/// Verifier half of the index: just $\mu$ and the indicator commitments.
/// Indexing is deterministic and public, so anyone can recompute this
/// from $\sigma$, the prover never gets to choose it arbitrarily.
pub struct BiPermVerifierIndex<F: PrimeField, P: PolynomialCommitment<F>> {
    pub num_vars: usize,
    pub ind_l_commit: P::Commitment,
    pub ind_r_commit: P::Commitment,
}

#[derive(Debug, PartialEq, Eq)]
pub enum BiPermError<E> {
    Sumcheck(SumcheckError),
    Core(CoreError),
    /// PCS operation produced an error
    Pcs(E),
    /// PCS opening verification failed
    PcsVerifyFailed,
    /// $f(r) \cdot ind_L(r) \cdot ind_R(r)$ did
    /// not equal the sumcheck's final claim.
    FinalCheckFailed,
}

impl<E> From<SumcheckError> for BiPermError<E> {
    fn from(e: SumcheckError) -> Self {
        Self::Sumcheck(e)
    }
}

impl<E> From<CoreError> for BiPermError<E> {
    fn from(e: CoreError) -> Self {
        Self::Core(e)
    }
}

/// Build $[\mathrm{eq}(\sigma_{\mathrm{half}}(x), \alpha_{\mathrm{half}}) : x \in B_\mu]$
/// with `eq_eval_table($\alpha_half$)` as a lookup over $\sigma$'s image bits.
fn indicator_table<F: PrimeField>(
    perm: &Permutation,
    alpha_half: &[F],
    // Maybe "left", "right"
    use_left_half: bool,
) -> Result<Vec<F>, CoreError> {
    let eq_t = eq_eval_table::<F>(alpha_half);
    (0..perm.size())
        .map(|x| {
            let (lo, hi) = perm.halves(x)?;
            Ok(eq_t[if use_left_half { lo } else { hi }])
        })
        .collect()
}

/// Output of [`index`], computing
/// the prover and verifier halves.
pub type BiPermIndex<F, P> =
    (BiPermProverIndex<F, P>, BiPermVerifierIndex<F, P>);

/// Preprocess a fixed $\sigma$: build and commit the two sparse indicator
/// polynomials $\tilde{\mathbb{1}}\_{\sigma_L}, \tilde{\mathbb{1}}\_{\sigma_R}$.
/// Runs once per permutation, independent of any $f, g$ instance.
pub fn index<F: PrimeField, P: PolynomialCommitment<F>>(
    pk: &P::ProverKey,
    perm: &Permutation,
) -> Result<BiPermIndex<F, P>, BiPermError<P::Error>> {
    let ind_l = perm.half_indicator::<F>(true)?;
    let ind_r = perm.half_indicator::<F>(false)?;
    let ind_l_commit =
        P::commit(pk, (&ind_l).into()).map_err(BiPermError::Pcs)?;
    let ind_r_commit =
        P::commit(pk, (&ind_r).into()).map_err(BiPermError::Pcs)?;
    Ok((
        BiPermProverIndex {
            perm: perm.clone(),
            ind_l,
            ind_r,
            ind_l_commit: ind_l_commit.clone(),
            ind_r_commit: ind_r_commit.clone(),
        },
        BiPermVerifierIndex {
            num_vars: perm.num_vars(),
            ind_l_commit,
            ind_r_commit,
        },
    ))
}

/// Prove $f(\sigma(x)) = g(x)$ for all $x \in B_\mu$.
///
/// Commits to $f, g$ via the PCS, absorbs the index and instance
/// commitments into the transcript, then squeezes $\alpha$ so $\alpha$
/// depends on both. After sumcheck rounds, opens $g$ at $\alpha$, $f$ at
/// the sumcheck challenge $r$, and the committed indicators at
/// $(r \Vert \alpha_{half})$, which equal $h_{L/R}(r)$ since partial
/// evaluation commutes for MLEs. Put another way, we have
/// extend-then-substitute = substitute-then-extend.
pub fn prove<F: PrimeField, P: PolynomialCommitment<F>>(
    pk: &P::ProverKey,
    index: &BiPermProverIndex<F, P>,
    f: &DenseMultilinearExtension<F>,
    g: &DenseMultilinearExtension<F>,
    transcript: &mut Transcript,
) -> Result<BiPermProof<F, P>, BiPermError<P::Error>> {
    let num_vars = index.perm.num_vars();
    assert_eq!(num_vars, f.num_vars, "f num_vars must match μ");
    assert_eq!(num_vars, g.num_vars, "g num_vars must match μ");
    // Span names group the bench phase breakdown
    let commit_span = info_span!("commit").entered();
    let f_commit = P::commit(pk, f.into()).map_err(BiPermError::Pcs)?;
    let g_commit = P::commit(pk, g.into()).map_err(BiPermError::Pcs)?;
    drop(commit_span);
    transcript.append(b"ind_l_commit", &index.ind_l_commit);
    transcript.append(b"ind_r_commit", &index.ind_r_commit);
    transcript.append(b"f_commit", &f_commit);
    transcript.append(b"g_commit", &g_commit);
    let alpha: Vec<F> = transcript.challenge_vec(b"alpha", num_vars);
    let opens_span = info_span!("opens").entered();
    let (g_at_alpha, g_opening) =
        P::open(pk, g.into(), &alpha, transcript).map_err(BiPermError::Pcs)?;
    drop(opens_span);
    let (alpha_l, alpha_r) = alpha.split_at(num_vars / 2);
    let aux_span = info_span!("aux").entered();
    let h_l_evals = indicator_table(&index.perm, alpha_l, true)?;
    let h_r_evals = indicator_table(&index.perm, alpha_r, false)?;
    let h_l =
        DenseMultilinearExtension::from_evaluations_vec(num_vars, h_l_evals);
    let h_r =
        DenseMultilinearExtension::from_evaluations_vec(num_vars, h_r_evals);
    drop(aux_span);
    let sumcheck_span = info_span!("sumcheck").entered();
    let output = sumcheck::prove(&[f.clone(), h_l, h_r], transcript);
    drop(sumcheck_span);
    let r = &output.challenges;
    let opens_span = info_span!("opens").entered();
    let (f_at_r, f_opening) =
        P::open(pk, f.into(), r, transcript).map_err(BiPermError::Pcs)?;
    drop(opens_span);
    let point_l: Vec<F> = r.iter().chain(alpha_l).copied().collect();
    let point_r: Vec<F> = r.iter().chain(alpha_r).copied().collect();
    let opens_span = info_span!("opens").entered();
    let (ind_l_at_r, ind_l_opening) =
        P::open(pk, (&index.ind_l).into(), &point_l, transcript)
            .map_err(BiPermError::Pcs)?;
    let (ind_r_at_r, ind_r_opening) =
        P::open(pk, (&index.ind_r).into(), &point_r, transcript)
            .map_err(BiPermError::Pcs)?;
    drop(opens_span);
    Ok(BiPermProof {
        f_commit,
        g_commit,
        sumcheck: output.proof,
        g_at_alpha,
        g_opening,
        f_at_r,
        f_opening,
        ind_l_at_r,
        ind_l_opening,
        ind_r_at_r,
        ind_r_opening,
    })
}

/// Verify a BiPerm proof. The verifier holds only the PCS verifier key and
/// the [`BiPermVerifierIndex`]; $f$, $g$, and the $\sigma$ indicators are
/// accessible only through commitments and openings in the proof.
pub fn verify<F: PrimeField, P: PolynomialCommitment<F>>(
    vk: &P::VerifierKey,
    index: &BiPermVerifierIndex<F, P>,
    proof: &BiPermProof<F, P>,
    transcript: &mut Transcript,
) -> Result<(), BiPermError<P::Error>> {
    let num_vars = index.num_vars;
    transcript.append(b"ind_l_commit", &index.ind_l_commit);
    transcript.append(b"ind_r_commit", &index.ind_r_commit);
    transcript.append(b"f_commit", &proof.f_commit);
    transcript.append(b"g_commit", &proof.g_commit);
    let alpha: Vec<F> = transcript.challenge_vec(b"alpha", num_vars);
    let alpha_ok = P::verify(
        vk,
        &proof.g_commit,
        &alpha,
        proof.g_at_alpha,
        &proof.g_opening,
        transcript,
    )
    .map_err(BiPermError::Pcs)?;
    if !alpha_ok {
        return Err(BiPermError::PcsVerifyFailed);
    }
    let out = sumcheck::verify(
        proof.g_at_alpha,
        num_vars,
        3,
        &proof.sumcheck,
        transcript,
    )?;
    let r = &out.challenges;
    let r_ok = P::verify(
        vk,
        &proof.f_commit,
        r,
        proof.f_at_r,
        &proof.f_opening,
        transcript,
    )
    .map_err(BiPermError::Pcs)?;
    if !r_ok {
        return Err(BiPermError::PcsVerifyFailed);
    }
    // Verify the indicator openings
    let (alpha_l, alpha_r) = alpha.split_at(num_vars / 2);
    let point_l: Vec<F> = r.iter().chain(alpha_l).copied().collect();
    let point_r: Vec<F> = r.iter().chain(alpha_r).copied().collect();
    let ind_l_ok = P::verify(
        vk,
        &index.ind_l_commit,
        &point_l,
        proof.ind_l_at_r,
        &proof.ind_l_opening,
        transcript,
    )
    .map_err(BiPermError::Pcs)?;
    if !ind_l_ok {
        return Err(BiPermError::PcsVerifyFailed);
    }
    let ind_r_ok = P::verify(
        vk,
        &index.ind_r_commit,
        &point_r,
        proof.ind_r_at_r,
        &proof.ind_r_opening,
        transcript,
    )
    .map_err(BiPermError::Pcs)?;
    if !ind_r_ok {
        return Err(BiPermError::PcsVerifyFailed);
    }

    if proof.f_at_r * proof.ind_l_at_r * proof.ind_r_at_r != out.final_claim {
        return Err(BiPermError::FinalCheckFailed);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_ff::UniformRand;
    use ark_std::test_rng;
    use permcore::MockPcs;

    /// Create the MLE of polynomials $f, g$ s.t.
    /// $f(\sigma(x)) = g(x)$ for all $x \in B_\mu$.
    fn consistent_pair(
        perm: &Permutation,
        rng: &mut impl ark_std::rand::RngCore,
    ) -> (DenseMultilinearExtension<Fr>, DenseMultilinearExtension<Fr>) {
        let num_vars = perm.num_vars();
        let f_evals: Vec<Fr> =
            (0..(1 << num_vars)).map(|_| Fr::rand(rng)).collect();
        // Lemma 4: $\sum_x f(x) \cdot eq(\sigma(x),\alphaα) = g(\alpha)$
        // holds iff $g(\sigma(x)) = f(x)$, eg. $g = f \cdot \sigma^{-1}$.
        // Encode that directly: $g[\sigma(x)] = f[x]$.
        let mut g_evals = alloc::vec![Fr::from(0u64); perm.size()];
        for x in 0..perm.size() {
            g_evals[perm.apply(x)] = f_evals[x];
        }
        (
            DenseMultilinearExtension::from_evaluations_vec(num_vars, f_evals),
            DenseMultilinearExtension::from_evaluations_vec(num_vars, g_evals),
        )
    }

    /// 16-element perm so $\mu = 4$ (even, halves cleanly)
    fn sample_perm() -> Permutation {
        Permutation::new(alloc::vec![
            5, 3, 7, 1, 0, 6, 2, 4, 9, 11, 8, 10, 13, 15, 12, 14
        ])
        .unwrap()
    }

    /// MockPcs setup sized for the largest committed
    /// polynomial, the $3\mu/2$-variate indicators.
    fn setup_keys(
        perm: &Permutation,
        rng: &mut impl ark_std::rand::RngCore,
    ) -> ((), ()) {
        MockPcs::<Fr>::setup(perm.num_vars() * 3 / 2, rng).unwrap()
    }

    #[test]
    fn honest_round_trip() {
        let mut rng = test_rng();
        let perm = sample_perm();
        let (f, g) = consistent_pair(&perm, &mut rng);
        let (pk, vk) = setup_keys(&perm, &mut rng);
        let (p_idx, v_idx) = index::<Fr, MockPcs<Fr>>(&pk, &perm).unwrap();
        let mut p_t = Transcript::new(b"biperm");
        let proof = prove(&pk, &p_idx, &f, &g, &mut p_t).unwrap();
        let mut v_t = Transcript::new(b"biperm");
        verify(&vk, &v_idx, &proof, &mut v_t).unwrap();
    }

    #[test]
    fn rejects_inconsistent_pair() {
        let mut rng = test_rng();
        let perm = sample_perm();
        let (f, _g_good) = consistent_pair(&perm, &mut rng);
        // Random $g$ unrelated to $f$ and $\sigma$.
        let bad_evals: Vec<Fr> = (0..(1 << perm.num_vars()))
            .map(|_| Fr::rand(&mut rng))
            .collect();
        let g_bad = DenseMultilinearExtension::from_evaluations_vec(
            perm.num_vars(),
            bad_evals,
        );
        let (pk, vk) = setup_keys(&perm, &mut rng);
        let (p_idx, v_idx) = index::<Fr, MockPcs<Fr>>(&pk, &perm).unwrap();
        let mut p_t = Transcript::new(b"biperm");
        let proof = prove(&pk, &p_idx, &f, &g_bad, &mut p_t).unwrap();
        // Verifier absorbs `g_bad`, gets a different $\alpha$ than the
        // prover, and the round-0 boundary check fails immediately.
        let mut v_t = Transcript::new(b"biperm");
        let err = verify(&vk, &v_idx, &proof, &mut v_t).unwrap_err();
        assert!(matches!(
            err,
            BiPermError::Sumcheck(SumcheckError::RoundCheckFailed { round: 0 }),
        ));
    }

    #[test]
    // Tamper (minimal) a single round polynomial
    fn rejects_tampered_sumcheck() {
        let mut rng = test_rng();
        let perm = sample_perm();
        let (f, g) = consistent_pair(&perm, &mut rng);
        let (pk, vk) = setup_keys(&perm, &mut rng);
        let (p_idx, v_idx) = index::<Fr, MockPcs<Fr>>(&pk, &perm).unwrap();
        let mut p_t = Transcript::new(b"biperm");
        let mut proof = prove(&pk, &p_idx, &f, &g, &mut p_t).unwrap();
        proof.sumcheck.round_polys[1][0] += Fr::from(1u64);
        let mut v_t = Transcript::new(b"biperm");
        let err = verify(&vk, &v_idx, &proof, &mut v_t).unwrap_err();
        assert!(matches!(
            err,
            BiPermError::Sumcheck(SumcheckError::RoundCheckFailed { round: 1 }),
        ));
    }

    #[test]
    fn rejects_mismatched_index() {
        // The indicator commitments are absorbed into the transcript, so a
        // verifier holding a different $\sigma$'s index derives a different
        // $\alpha$ and the $g$-opening check fails immediately.
        let mut rng = test_rng();
        let perm = sample_perm();
        let (f, g) = consistent_pair(&perm, &mut rng);
        // Canonical identity permutation, `0..perm.num_vars()`
        let other_perm = Permutation::identity(perm.num_vars());
        let (pk, vk) = setup_keys(&perm, &mut rng);
        let (p_idx, _) = index::<Fr, MockPcs<Fr>>(&pk, &perm).unwrap();
        let (_, v_idx_other) =
            index::<Fr, MockPcs<Fr>>(&pk, &other_perm).unwrap();
        let mut p_t = Transcript::new(b"biperm");
        let proof = prove(&pk, &p_idx, &f, &g, &mut p_t).unwrap();
        let mut v_t = Transcript::new(b"biperm");
        let err = verify(&vk, &v_idx_other, &proof, &mut v_t).unwrap_err();
        assert!(matches!(err, BiPermError::PcsVerifyFailed));
    }

    #[test]
    // A forged indicator value must fail against the
    // preprocessed commitment, not be taken on faith.
    fn rejects_tampered_indicator_opening() {
        let mut rng = test_rng();
        let perm = sample_perm();
        let (f, g) = consistent_pair(&perm, &mut rng);
        let (pk, vk) = setup_keys(&perm, &mut rng);
        let (p_idx, v_idx) = index::<Fr, MockPcs<Fr>>(&pk, &perm).unwrap();
        let mut p_t = Transcript::new(b"biperm");
        let mut proof = prove(&pk, &p_idx, &f, &g, &mut p_t).unwrap();
        proof.ind_l_at_r += Fr::from(1u64);
        let mut v_t = Transcript::new(b"biperm");
        let err = verify(&vk, &v_idx, &proof, &mut v_t).unwrap_err();
        assert!(matches!(err, BiPermError::PcsVerifyFailed));
    }
}
