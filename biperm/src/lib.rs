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
//! This implementation operates on dense multilinear polynomials and assumes
//! the verifier holds $f$ and $g$ directly. A SNARK requires committing to $f,
//! g$ via PCS upfront and replacing the direct evaluations with PCS openings.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub use permcore;

use alloc::vec::Vec;

use ark_ff::PrimeField;
use ark_poly::{DenseMultilinearExtension, Polynomial};
use permcore::eq::eq_eval_table;
use permcore::sumcheck::{self, SumcheckError, SumcheckProof};
use permcore::{CoreError, Permutation, Transcript};

/// Currently just the underlying sumcheck transcript; the `f` and `g`
/// evaluations the verifier needs are computed directly for now.
/// When a PCS layer is added we'll include commitments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BiPermProof<F> {
    pub sumcheck: SumcheckProof<F>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum BiPermError {
    Sumcheck(SumcheckError),
    Core(CoreError),
    /// $f(r) \cdot ind_L(r) \cdot ind_R(r)$ did
    /// not equal the sumcheck's final claim.
    FinalCheckFailed,
}

impl From<SumcheckError> for BiPermError {
    fn from(e: SumcheckError) -> Self {
        Self::Sumcheck(e)
    }
}

impl From<CoreError> for BiPermError {
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

/// Prove $f(\sigma(x)) = g(x)$ for all $x \in B_\mu$.
///
/// Absorbs $f$ and $g$ into the transcript before squeezing $\alpha$ so that
/// $\alpha$ depends on the specific instance, not just the domain separator.
pub fn prove<F: PrimeField>(
    perm: &Permutation,
    f: &DenseMultilinearExtension<F>,
    g: &DenseMultilinearExtension<F>,
    transcript: &mut Transcript,
) -> Result<BiPermProof<F>, BiPermError> {
    let num_vars = perm.num_vars();
    assert_eq!(num_vars, f.num_vars, "f num_vars must match μ");
    assert_eq!(num_vars, g.num_vars, "g num_vars must match μ");
    transcript.append_slice(b"f", &f.evaluations);
    transcript.append_slice(b"g", &g.evaluations);
    let alpha: Vec<F> = transcript.challenge_vec(b"alpha", num_vars);
    let (alpha_l, alpha_r) = alpha.split_at(num_vars / 2);
    let h_l_evals = indicator_table(perm, alpha_l, true)?;
    let h_r_evals = indicator_table(perm, alpha_r, false)?;
    let h_l =
        DenseMultilinearExtension::from_evaluations_vec(num_vars, h_l_evals);
    let h_r =
        DenseMultilinearExtension::from_evaluations_vec(num_vars, h_r_evals);
    let output = sumcheck::prove(&[f.clone(), h_l, h_r], transcript);
    Ok(BiPermProof {
        sumcheck: output.proof,
    })
}

/// Verify a BiPerm proof against $f$ and $g$.
///
/// $\alpha$ is squeezed from the transcript, mirroring the prover's move.
/// The verifier then computes the initial claim $g(\alpha)$, runs the
/// sumcheck verifier, and directly evaluates $f(r)$, $\mathrm{ind}_L(r)$,
/// $\mathrm{ind}_R(r)$ to check the final product.
pub fn verify<F: PrimeField>(
    perm: &Permutation,
    f: &DenseMultilinearExtension<F>,
    g: &DenseMultilinearExtension<F>,
    proof: &BiPermProof<F>,
    transcript: &mut Transcript,
) -> Result<(), BiPermError> {
    let num_vars = perm.num_vars();
    assert_eq!(num_vars, f.num_vars, "f num_vars must match μ");
    assert_eq!(num_vars, g.num_vars, "g num_vars must match μ");
    transcript.append_slice(b"f", &f.evaluations);
    transcript.append_slice(b"g", &g.evaluations);
    let alpha: Vec<F> = transcript.challenge_vec(b"alpha", num_vars);
    let initial_claim = g.evaluate(&alpha);
    let out = sumcheck::verify(
        initial_claim,
        num_vars,
        3,
        &proof.sumcheck,
        transcript,
    )?;
    let (alpha_l, alpha_r) = alpha.split_at(num_vars / 2);
    // Direct evaluation for until we have a commitment scheme. We should
    // still use the interface of one so that this code looks the same.
    let h_l_evals = indicator_table(perm, alpha_l, true)?;
    let h_r_evals = indicator_table(perm, alpha_r, false)?;
    let h_l =
        DenseMultilinearExtension::from_evaluations_vec(num_vars, h_l_evals);
    let h_r =
        DenseMultilinearExtension::from_evaluations_vec(num_vars, h_r_evals);
    let f_at_r = f.evaluate(&out.challenges);
    let ind_l_at_r = h_l.evaluate(&out.challenges);
    let ind_r_at_r = h_r.evaluate(&out.challenges);
    if f_at_r * ind_l_at_r * ind_r_at_r != out.final_claim {
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

    #[test]
    fn honest_round_trip() {
        let mut rng = test_rng();
        let perm = sample_perm();
        let (f, g) = consistent_pair(&perm, &mut rng);
        let mut p_t = Transcript::new(b"biperm");
        let proof = prove(&perm, &f, &g, &mut p_t).unwrap();
        let mut v_t = Transcript::new(b"biperm");
        verify(&perm, &f, &g, &proof, &mut v_t).unwrap();
    }

    #[test]
    fn rejects_inconsistent_pair() {
        let mut rng = test_rng();
        let perm = sample_perm();
        let (f, g_good) = consistent_pair(&perm, &mut rng);
        // Random $g$ unrelated to $f$ and $\sigma$.
        let bad_evals: Vec<Fr> = (0..(1 << perm.num_vars()))
            .map(|_| Fr::rand(&mut rng))
            .collect();
        let g_bad = DenseMultilinearExtension::from_evaluations_vec(
            perm.num_vars(),
            bad_evals,
        );
        let mut p_t = Transcript::new(b"biperm");
        let proof = prove(&perm, &f, &g_good, &mut p_t).unwrap();
        // Verifier absorbs `g_bad`, gets a different $\alpha$ than the
        // prover, and the round-0 boundary check fails immediately.
        let mut v_t = Transcript::new(b"biperm");
        let err = verify(&perm, &f, &g_bad, &proof, &mut v_t).unwrap_err();
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
        let mut p_t = Transcript::new(b"biperm");
        let mut proof = prove(&perm, &f, &g, &mut p_t).unwrap();
        proof.sumcheck.round_polys[1][0] += Fr::from(1u64);
        let mut v_t = Transcript::new(b"biperm");
        let err = verify(&perm, &f, &g, &proof, &mut v_t).unwrap_err();
        assert!(matches!(
            err,
            BiPermError::Sumcheck(SumcheckError::RoundCheckFailed { round: 1 }),
        ));
    }

    #[test]
    fn rejects_wrong_perm() {
        // $\sigma$ isn't (yet) absorbed into the transcript, so
        // prover and verifier agree on $\alpha$ but produce different
        // indicators, the final product check fails.
        let mut rng = test_rng();
        let perm = sample_perm();
        let (f, g) = consistent_pair(&perm, &mut rng);
        // Canonical identity permutation, `0..perm.num_vars()`
        let other_perm = Permutation::identity(perm.num_vars());
        let mut p_t = Transcript::new(b"biperm");
        let proof = prove(&perm, &f, &g, &mut p_t).unwrap();
        let mut v_t = Transcript::new(b"biperm");
        let err = verify(&other_perm, &f, &g, &proof, &mut v_t).unwrap_err();
        assert!(matches!(err, BiPermError::FinalCheckFailed));
    }
}
