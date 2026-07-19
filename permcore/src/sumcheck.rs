//! Sumcheck protocol for sums of products of multilinear polynomials.
//!
//! Reduces the claim $T = \sum_{x \in B_\mu} \sum_i c_i \prod_k g_{i,k}(x)$
//! to a single evaluation claim $\sum_i c_i \prod_k g_{i,k}(r) = u$ at a
//! random $r \in F^\mu$. The verifier then needs to evaluate the final claim
//! by some external means, typically PCS openings of the underlying
//! polynomials at $r$, or direct computation when the verifier can.
//!
//! Each factor $g_{i,k}$ must be multilinear; the protocol degree is the
//! maximum number of factors in any term. Each round produces a univariate
//! polynomial of degree $d$, encoded as $d+1$ evaluations at
//! $\{0, 1, \ldots, d\}$. [`prove`] covers the single-product case
//! $T = \sum_x \prod_k g_k(x)$; [`prove_terms`] the general form.

use alloc::vec::Vec;

use ark_ff::PrimeField;
use ark_poly::DenseMultilinearExtension;
use ark_std::vec;

use crate::transcript::Transcript;

/// Errors produced by the sumcheck verifier.
#[derive(Debug, PartialEq, Eq)]
pub enum SumcheckError {
    /// Proof has the wrong number of rounds for `num_vars`.
    WrongNumRounds { expected: usize, got: usize },
    /// A round polynomial has the wrong number of
    /// evaluations for the declared `degree`.
    MalformedRoundPoly {
        round: usize,
        expected: usize,
        got: usize,
    },
    /// $s_i(0) + s_i(1)$ did not match
    /// the previous round's claim.
    RoundCheckFailed { round: usize },
}

/// Transcript of a sumcheck proof.
///
/// `round_polys[i]` is the round-$i$ univariate polynomial sent
/// by the prover, encoded as evaluations at `0, 1, ..., degree`.
/// Obviously with $d+1$ points, can use Lagrange build it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SumcheckProof<F> {
    pub round_polys: Vec<Vec<F>>,
}

/// Output of sumcheck proving, the
/// proof plus the challenge point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SumcheckProverOutput<F> {
    pub proof: SumcheckProof<F>,
    pub challenges: Vec<F>,
}

/// Output of a successful sumcheck verify call.
///
/// `challenges` is the random point $r = (r_1, \ldots, r_\mu)$.
/// `final_claim` is the value the proved expression must
/// equal at $r$. ($\prod_k g_k(r)$ for [`prove`],
/// $\sum_i c_i \prod_k g_{i,k}(r)$ for [`prove_terms`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SumcheckOutput<F> {
    // Each challenge, or single random point
    pub challenges: Vec<F>,
    pub final_claim: F,
}

/// One product term $c \cdot \prod_k g_{i_k}$ of a sum of products,
/// referencing its factors by index into a shared factor slice so a
/// polynomial appearing in several terms is stored (and folded) once.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Term<F> {
    /// Coefficient $c$ scaling the product.
    pub coeff: F,
    /// Indices into the shared factor slice; may repeat
    /// (e.g. squaring a factor) and may appear in other terms.
    pub factors: Vec<usize>,
}

/// Prove $T = \sum_{x \in B_\mu} \prod_k g_k(x)$ where each $g_k$ is
/// multilinear. Returns the proof transcript; the initial claim $T$ is
/// implicit and must be supplied to the verifier out-of-band.
///
/// Single-term case of [`prove_terms`]; produces an identical transcript
/// for the product of all `factors` with coefficient one.
///
/// # Panics
///
/// Panics if `factors` is empty or
/// factors have inconsistent `num_vars`.
pub fn prove<F: PrimeField>(
    factors: &[DenseMultilinearExtension<F>],
    transcript: &mut Transcript,
) -> SumcheckProverOutput<F> {
    let term = Term {
        coeff: F::one(),
        factors: (0..factors.len()).collect(),
    };
    prove_terms(factors, &[term], transcript)
}

/// Prove $T = \sum_{x \in B_\mu} \sum_i c_i \prod_k g_{i,k}(x)$ where each
/// factor $g_{i,k}$ is multilinear and each [`Term`] indexes into `factors`.
/// The protocol degree is the maximum factor count over `terms`; the verifier
/// must be given the same value. Returns the proof transcript; the initial
/// claim $T$ is implicit and must be supplied to the verifier out-of-band.
///
/// # Panics
///
/// Panics if `factors` or `terms` is empty, factors have
/// inconsistent `num_vars`, a term references a factor out
/// of range, or every term is empty (degree zero).
pub fn prove_terms<F: PrimeField>(
    factors: &[DenseMultilinearExtension<F>],
    terms: &[Term<F>],
    transcript: &mut Transcript,
) -> SumcheckProverOutput<F> {
    assert!(
        !factors.is_empty(),
        "sumcheck::prove_terms: factors must not be empty",
    );
    assert!(
        !terms.is_empty(),
        "sumcheck::prove_terms: terms must not be empty",
    );
    // Factors `num_vars` must all be the same.
    // In the case of BiPerm, we have `num_vars = \mu`.
    // So we're saying all polynomials over boolean hypercube.
    let num_vars = factors[0].num_vars;
    for f in factors.iter().skip(1) {
        assert_eq!(
            f.num_vars, num_vars,
            "sumcheck::prove_terms: inconsistent num_vars",
        );
    }
    // Check index range
    for t in terms {
        for &i in &t.factors {
            assert!(
                i < factors.len(),
                "sumcheck::prove_terms: factor index out of range",
            );
        }
    }
    // Degree is just the max # of MLEs in a term
    let degree = terms.iter().map(|t| t.factors.len()).max().unwrap();
    assert!(
        degree >= 1,
        "sumcheck::prove_terms: degree must be at least 1"
    );
    // Working copies of each factor's eval table; updated in place per
    // round. Shared across terms, so each factor folds once per round.
    let mut tables: Vec<Vec<F>> =
        factors.iter().map(|f| f.evaluations.clone()).collect();
    let mut round_polys = Vec::with_capacity(num_vars);
    let mut challenges = Vec::with_capacity(num_vars);

    for _ in 0..num_vars {
        // Folding is halving
        let half = tables[0].len() / 2;
        // s(c) for c = 0, 1, ..., degree.
        let mut evals = vec![F::zero(); degree + 1];
        for (c_idx, eval) in evals.iter_mut().enumerate() {
            let c = F::from(c_idx as u64);
            let one_minus_c = F::one() - c;
            let mut sum = F::zero();
            for y in 0..half {
                // Reusing tables is an optimization, we get all the
                // values over the entire cube to start from initially
                for term in terms {
                    let mut prod = term.coeff;
                    for &i in &term.factors {
                        let even = tables[i][2 * y];
                        let odd = tables[i][2 * y + 1];
                        prod *= one_minus_c * even + c * odd;
                    }
                    sum += prod;
                }
            }
            // Because single eval is a sum over $B_\mu$, but
            // $|evals| = degree + 1$ for the round univariate poly
            *eval = sum;
        }
        // For each round we're producing evals for what we're sum-checking
        // in that round. Update the transcript and the round polynomials.
        transcript.append_slice(b"round", &evals);
        round_polys.push(evals);
        // Bind LSB to r in each factor's table.
        // Same thing here with the challenge, update.
        let r: F = transcript.challenge(b"r");
        challenges.push(r);
        let one_minus_r = F::one() - r;
        for table in tables.iter_mut() {
            for y in 0..half {
                let even = table[2 * y];
                let odd = table[2 * y + 1];
                table[y] = one_minus_r * even + r * odd;
            }
            // We only need half to continue
            table.truncate(half);
        }
    }
    // Each round produces a small univariate polynomial and a
    // challenge, using Fiat-Shamir of evals up to this point.
    SumcheckProverOutput {
        proof: SumcheckProof { round_polys },
        challenges,
    }
}

/// Verify a sumcheck proof against `initial_claim`.
///
/// Checks per-round consistency $s_i(0) + s_i(1) == prev_claim$ and the
/// proof's structural shape; `degree` is the number of factors for [`prove`]
/// or the maximum term factor count for [`prove_terms`]. Returns the
/// challenge point and the final claim the caller must check
/// $\sum_i c_i \prod_k g_{i,k}(r) == final_claim$ against. The caller
/// evaluates claimed poly the challenge point to check the `final_claim`.
pub fn verify<F: PrimeField>(
    initial_claim: F,
    num_vars: usize,
    degree: usize,
    proof: &SumcheckProof<F>,
    transcript: &mut Transcript,
) -> Result<SumcheckOutput<F>, SumcheckError> {
    if proof.round_polys.len() != num_vars {
        return Err(SumcheckError::WrongNumRounds {
            expected: num_vars,
            got: proof.round_polys.len(),
        });
    }
    let mut claim = initial_claim;
    let mut challenges = Vec::with_capacity(num_vars);
    for (round, round_poly) in proof.round_polys.iter().enumerate() {
        if round_poly.len() != degree + 1 {
            return Err(SumcheckError::MalformedRoundPoly {
                round,
                expected: degree + 1,
                got: round_poly.len(),
            });
        }
        if round_poly[0] + round_poly[1] != claim {
            return Err(SumcheckError::RoundCheckFailed { round });
        }
        transcript.append_slice(b"round", round_poly);
        let r: F = transcript.challenge(b"r");
        challenges.push(r);
        claim = lagrange_eval(round_poly, r);
    }
    Ok(SumcheckOutput {
        challenges,
        final_claim: claim,
    })
}

/// Evaluate a univariate polynomial at `r`, given its evaluations at
/// `0, 1, ..., evals.len() - 1`. Standard Lagrange interpolation.
/// We interpolate and evaluate the point, all at once.
fn lagrange_eval<F: PrimeField>(evals: &[F], r: F) -> F {
    let n = evals.len();
    let mut result = F::zero();
    for (i, &eval_i) in evals.iter().enumerate() {
        let xi = F::from(i as u64);
        let mut num = F::one();
        let mut den = F::one();
        for j in 0..n {
            if i == j {
                continue;
            }
            let xj = F::from(j as u64);
            // Point in evaluation
            num *= r - xj;
            den *= xi - xj;
        }
        result +=
            eval_i * num * den.inverse().expect("nonzero by construction");
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_ff::UniformRand;
    use ark_poly::Polynomial;
    use ark_std::test_rng;

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

    /// Just sum over $B_\mu$, what we're checking
    fn initial_claim(factors: &[DenseMultilinearExtension<Fr>]) -> Fr {
        let n = 1 << factors[0].num_vars;
        (0..n)
            .map(|x| factors.iter().map(|f| f.evaluations[x]).product::<Fr>())
            .sum()
    }

    #[test]
    fn single_factor_round_trip() {
        let mut rng = test_rng();
        let num_vars = 4;
        let factor = random_mle(num_vars, &mut rng);
        let claim = initial_claim(core::slice::from_ref(&factor));
        let mut p_t = Transcript::new(b"sumcheck");
        let proof = prove(core::slice::from_ref(&factor), &mut p_t).proof;
        let mut v_t = Transcript::new(b"sumcheck");
        let out = verify(claim, num_vars, 1, &proof, &mut v_t).unwrap();
        assert_eq!(factor.evaluate(&out.challenges), out.final_claim);
    }

    #[test]
    // Closer to BiPerm and MulPerm
    fn product_of_three_factors_round_trip() {
        let mut rng = test_rng();
        let num_vars = 3;
        let factors: Vec<_> =
            (0..3).map(|_| random_mle(num_vars, &mut rng)).collect();
        let claim = initial_claim(&factors);
        let mut p_t = Transcript::new(b"sumcheck");
        let proof = prove(&factors, &mut p_t).proof;
        let mut v_t = Transcript::new(b"sumcheck");
        let out = verify(claim, num_vars, 3, &proof, &mut v_t).unwrap();
        let final_product: Fr = factors
            .iter()
            .map(|f| f.evaluate(&out.challenges))
            .product();
        assert_eq!(final_product, out.final_claim);
    }

    /// Sum over $B_\mu$ of the term form,
    /// what `prove_terms` is checking
    fn terms_claim(
        factors: &[DenseMultilinearExtension<Fr>],
        terms: &[Term<Fr>],
    ) -> Fr {
        let n = 1 << factors[0].num_vars;
        (0..n)
            .map(|x| {
                terms
                    .iter()
                    .map(|t| {
                        t.factors
                            .iter()
                            .map(|&i| factors[i].evaluations[x])
                            .product::<Fr>()
                            * t.coeff
                    })
                    .sum::<Fr>()
            })
            .sum()
    }

    #[test]
    // Mixed-degree terms sharing factors: c0*g0*g1*g2 + c1*g1
    fn multi_term_round_trip() {
        let mut rng = test_rng();
        let num_vars = 4;
        let factors: Vec<_> =
            (0..3).map(|_| random_mle(num_vars, &mut rng)).collect();
        let terms = vec![
            Term {
                coeff: Fr::rand(&mut rng),
                factors: vec![0, 1, 2],
            },
            Term {
                coeff: Fr::rand(&mut rng),
                factors: vec![1],
            },
        ];
        let claim = terms_claim(&factors, &terms);
        let mut p_t = Transcript::new(b"sumcheck");
        let proof = prove_terms(&factors, &terms, &mut p_t).proof;
        let mut v_t = Transcript::new(b"sumcheck");
        // Degree 3 (max), # of factors in the first term
        let out = verify(claim, num_vars, 3, &proof, &mut v_t).unwrap();
        // Evaluation at the random point
        let expected: Fr = terms
            .iter()
            .map(|t| {
                t.factors
                    .iter()
                    .map(|&i| factors[i].evaluate(&out.challenges))
                    .product::<Fr>()
                    * t.coeff
            })
            .sum();
        assert_eq!(expected, out.final_claim);
    }

    #[test]
    // A repeated index within a term squares the factor
    fn repeated_factor_round_trip() {
        let mut rng = test_rng();
        let num_vars = 3;
        let factor = random_mle(num_vars, &mut rng);
        let terms = vec![Term {
            coeff: Fr::from(1u64),
            factors: vec![0, 0],
        }];
        let factors = vec![factor];
        let claim = terms_claim(&factors, &terms);
        let mut p_t = Transcript::new(b"sumcheck");
        let proof = prove_terms(&factors, &terms, &mut p_t).proof;
        let mut v_t = Transcript::new(b"sumcheck");
        let out = verify(claim, num_vars, 2, &proof, &mut v_t).unwrap();
        let eval = factors[0].evaluate(&out.challenges);
        assert_eq!(eval * eval, out.final_claim);
    }

    #[test]
    fn rejects_wrong_initial_claim() {
        let mut rng = test_rng();
        let num_vars = 3;
        let factor = random_mle(num_vars, &mut rng);
        let claim = initial_claim(core::slice::from_ref(&factor));
        let mut p_t = Transcript::new(b"sumcheck");
        let proof = prove(&[factor], &mut p_t).proof;
        let mut v_t = Transcript::new(b"sumcheck");
        let err = verify(claim + Fr::from(1u64), num_vars, 1, &proof, &mut v_t)
            .unwrap_err();
        assert!(matches!(err, SumcheckError::RoundCheckFailed { round: 0 }));
    }

    #[test]
    fn rejects_tampered_round_poly() {
        let mut rng = test_rng();
        let num_vars = 3;
        let factor = random_mle(num_vars, &mut rng);
        let claim = initial_claim(core::slice::from_ref(&factor));
        let mut p_t = Transcript::new(b"sumcheck");
        let mut proof = prove(&[factor], &mut p_t).proof;
        // Tamper round 1's s(0); round 0's boundary check still passes
        // since we didn't touch it, so verify fails at round 1.
        proof.round_polys[1][0] += Fr::from(1u64);
        let mut v_t = Transcript::new(b"sumcheck");
        let err = verify(claim, num_vars, 1, &proof, &mut v_t).unwrap_err();
        assert!(matches!(err, SumcheckError::RoundCheckFailed { round: 1 }));
    }

    #[test]
    fn rejects_wrong_num_rounds() {
        let mut rng = test_rng();
        let num_vars = 3;
        let factor = random_mle(num_vars, &mut rng);
        let claim = initial_claim(core::slice::from_ref(&factor));
        let mut p_t = Transcript::new(b"sumcheck");
        let proof = prove(&[factor], &mut p_t).proof;
        let mut v_t = Transcript::new(b"sumcheck");
        let err = verify(claim, num_vars + 1, 1, &proof, &mut v_t).unwrap_err();
        assert!(matches!(err, SumcheckError::WrongNumRounds { .. }));
    }

    #[test]
    fn rejects_malformed_round_poly() {
        let mut rng = test_rng();
        let num_vars = 3;
        let factor = random_mle(num_vars, &mut rng);
        let claim = initial_claim(core::slice::from_ref(&factor));
        let mut p_t = Transcript::new(b"sumcheck");
        let mut proof = prove(&[factor], &mut p_t).proof;
        // Drop evaluation, degree evaluations
        proof.round_polys[0].pop();
        let mut v_t = Transcript::new(b"sumcheck");
        let err = verify(claim, num_vars, 1, &proof, &mut v_t).unwrap_err();
        assert!(matches!(
            err,
            SumcheckError::MalformedRoundPoly {
                round: 0,
                expected: 2,
                got: 1,
            }
        ));
    }
}
