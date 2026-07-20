//! Zerocheck PIOP: prove a constraint vanishes on the boolean hypercube.
//!
//! Given a constraint $C(x) = \sum_i c_i \prod_k g_{i,k}(x)$ over
//! multilinear factors, proves $C(x) = 0$ for **all** $x \in B_\mu$. The
//! verifier draws a random $r \in F^\mu$ and the claim reduces to the
//! sumcheck $\sum_{x \in B_\mu} eq(r, x) \cdot C(x) = 0$.
//!
//! A plain sumcheck with claim $0$ would only show the values of $C$
//! *sum* to zero, which cancellation across the cube can fake. The $eq$
//! factor closes the gap: as a function of $r$ the weighted sum is the
//! multilinear extension of $C$'s cube values, so if $C$ is nonzero
//! anywhere on the cube it is nonzero at a random $r$ except with
//! probability $\le \mu / |F|$ (Schwartz-Zippel).

use alloc::vec::Vec;

use ark_ff::PrimeField;
use ark_poly::DenseMultilinearExtension;

use crate::eq::{eq, eq_eval_table};
use crate::sumcheck::{
    self, SumcheckError, SumcheckProof, SumcheckProverOutput, Term,
};
use crate::transcript::Transcript;

/// Output of a successful zerocheck verify call.
///
/// `challenges` is the sumcheck point $\rho$; `eq_eval` is $eq(r, \rho)$,
/// computed by the verifier from its own $r$. The caller must check
/// $eq\\_eval \cdot C(\rho) == final\\_claim$, evaluating the constraint's
/// factors at $\rho$ by external means (eg. PCS openings).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZerocheckOutput<F> {
    /// The sumcheck challenge point $\rho$.
    pub challenges: Vec<F>,
    /// $eq(r, \rho)$ for the zerocheck point $r$.
    pub eq_eval: F,
    /// The raw sumcheck final claim, $eq(r, \rho) \cdot C(\rho)$.
    pub final_claim: F,
}

/// Prove that the constraint given by `terms` over `factors` vanishes
/// everywhere on $B_\mu$. Draws the zerocheck point $r$ from `transcript`,
/// multiplies $eq(r, \cdot)$ into every term, and runs the sumcheck with
/// initial claim zero. Returns the sumcheck prover output; its challenge
/// point is where the caller must later open the factors.
///
/// # Panics
///
/// Panics if `factors` is empty; otherwise inherits the
/// [`sumcheck::prove_terms`] contract on `factors` and `terms`.
pub fn prove<F: PrimeField>(
    factors: &[DenseMultilinearExtension<F>],
    terms: &[Term<F>],
    transcript: &mut Transcript,
) -> SumcheckProverOutput<F> {
    assert!(
        !factors.is_empty(),
        "zerocheck::prove: factors must not be empty",
    );
    let num_vars = factors[0].num_vars;
    let r = transcript.challenge_vec(b"zerocheck_r", num_vars);
    let eq_mle = DenseMultilinearExtension::from_evaluations_vec(
        num_vars,
        eq_eval_table(&r),
    );
    // The eq factor is shared by every term, appended once to the
    // slice. This makes it simple to include it in every term.
    let mut all = factors.to_vec();
    let eq_idx = all.len();
    all.push(eq_mle);
    let eq_terms: Vec<Term<F>> = terms
        .iter()
        .map(|t| {
            let mut factors = t.factors.clone();
            factors.push(eq_idx);
            Term {
                coeff: t.coeff,
                factors,
            }
        })
        .collect();
    sumcheck::prove_terms(&all, &eq_terms, transcript)
}

/// Verify a zerocheck proof.
///
/// `degree` is the constraint's maximum term factor count; the
/// $eq$ factor is accounted for internally. Draws the same $r$ as
/// the prover, verifies the sumcheck against initial claim zero,
/// and returns the output for the caller's final-claim check.
pub fn verify<F: PrimeField>(
    num_vars: usize,
    degree: usize,
    proof: &SumcheckProof<F>,
    transcript: &mut Transcript,
) -> Result<ZerocheckOutput<F>, SumcheckError> {
    let r: Vec<F> = transcript.challenge_vec(b"zerocheck_r", num_vars);
    let out =
        sumcheck::verify(F::zero(), num_vars, degree + 1, proof, transcript)?;
    let eq_eval = eq(&r, &out.challenges);
    Ok(ZerocheckOutput {
        challenges: out.challenges,
        eq_eval,
        final_claim: out.final_claim,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_ff::UniformRand;
    use ark_poly::Polynomial;
    use ark_std::{test_rng, vec};

    /// Generate a random multi-linear extension
    /// over the $B_\mu$ of `num_vars` bits
    fn random_mle(
        num_vars: usize,
        rng: &mut impl ark_std::rand::RngCore,
    ) -> DenseMultilinearExtension<Fr> {
        let evals: Vec<Fr> =
            (0..(1 << num_vars)).map(|_| Fr::rand(rng)).collect();
        DenseMultilinearExtension::from_evaluations_vec(num_vars, evals)
    }

    #[test]
    // The layer-consistency shape: $a \cdot b − c \equiv 0$
    // on the cube, where $c$ is the MLE of the pointwise products.
    // Off the cube $a \cdot b \neq c$ (product of MLEs is not the
    // MLE of the products), so the final claim is checked against
    // factor evaluations at $\rho$, not against zero.
    fn product_relation_round_trip() {
        let mut rng = test_rng();
        let num_vars = 4;
        let a = random_mle(num_vars, &mut rng);
        let b = random_mle(num_vars, &mut rng);
        let c_evals: Vec<Fr> = (0..(1 << num_vars))
            .map(|x| a.evaluations[x] * b.evaluations[x])
            .collect();
        let c =
            DenseMultilinearExtension::from_evaluations_vec(num_vars, c_evals);
        let factors = vec![a, b, c];
        let terms = vec![
            Term {
                coeff: Fr::from(1u64),
                factors: vec![0, 1],
            },
            Term {
                coeff: -Fr::from(1u64),
                factors: vec![2],
            },
        ];
        let mut p_t = Transcript::new(b"zerocheck");
        let proof = prove(&factors, &terms, &mut p_t).proof;
        let mut v_t = Transcript::new(b"zerocheck");
        let out = verify(num_vars, 2, &proof, &mut v_t).unwrap();
        let a_r = factors[0].evaluate(&out.challenges);
        let b_r = factors[1].evaluate(&out.challenges);
        let c_r = factors[2].evaluate(&out.challenges);
        assert_eq!(out.eq_eval * (a_r * b_r - c_r), out.final_claim);
    }

    #[test]
    // An honest prover of a false statement: C nonzero on the cube
    // makes the weighted total nonzero, failing round 0 against claim 0.
    // Honest prover just passes what it computes directly from prove.
    fn rejects_nonvanishing_constraint() {
        let mut rng = test_rng();
        let num_vars = 3;
        let factors = vec![random_mle(num_vars, &mut rng)];
        let terms = vec![Term {
            coeff: Fr::from(1u64),
            factors: vec![0],
        }];
        let mut p_t = Transcript::new(b"zerocheck");
        let proof = prove(&factors, &terms, &mut p_t).proof;
        let mut v_t = Transcript::new(b"zerocheck");
        let err = verify(num_vars, 1, &proof, &mut v_t).unwrap_err();
        assert!(matches!(err, SumcheckError::RoundCheckFailed { round: 0 }));
    }

    #[test]
    // The property zerocheck adds over plain sumcheck: values that
    // sum to zero across the cube but are nonzero pointwise. A plain
    // sumcheck against claim 0 accepts; the eq weighting rejects.
    fn rejects_zero_sum_but_nonvanishing() {
        let num_vars = 3;
        let mut evals = vec![Fr::from(0u64); 1 << num_vars];
        evals[0] = Fr::from(1u64);
        evals[1] = -Fr::from(1u64);
        let g =
            DenseMultilinearExtension::from_evaluations_vec(num_vars, evals);
        let factors = vec![g];
        let terms = vec![Term {
            coeff: Fr::from(1u64),
            factors: vec![0],
        }];
        // Plain sumcheck accepts the zero-sum claim
        let mut p_t = Transcript::new(b"sumcheck");
        let plain = sumcheck::prove(&factors, &mut p_t).proof;
        let mut v_t = Transcript::new(b"sumcheck");
        assert!(sumcheck::verify(
            Fr::from(0u64),
            num_vars,
            1,
            &plain,
            &mut v_t
        )
        .is_ok());
        // Zerocheck rejects the same constraint
        let mut p_t = Transcript::new(b"zerocheck");
        let proof = prove(&factors, &terms, &mut p_t).proof;
        let mut v_t = Transcript::new(b"zerocheck");
        let err = verify(num_vars, 1, &proof, &mut v_t).unwrap_err();
        assert!(matches!(err, SumcheckError::RoundCheckFailed { round: 0 }));
    }
}
