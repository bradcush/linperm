//! Product-check a grand product of ratios equals one.
//!
//! Given multilinear $p, q$ over $B_\mu$, proves
//! $\prod_{x \in B_\mu} \frac{p(x)}{q(x)} = 1$ by committing
//! a single $(\mu + 1)$-variate polynomial $v$ holding the whole
//! product tree, then reducing to one [`zerocheck`] over $B_\mu$.
//!
//! # The layout of $v$
//!
//! With $N = 2^\mu$, the evaluation vector $V$ of $v$ has $2N$
//! slots and holds the tree in layer order, leaves first:
//!
//! * $V\[y\] = p(y)/q(y)$ for $y \in \[0, N)$ - the leaves,
//! * $V\[i\] = V\[2(i - N)\] \cdot V\[2(i - N) + 1\]$ for
//!   $i \in \[N, 2N-1)$ - every internal node, one rule across all layers,
//! * $V\[2N - 1\] = 0$ - the one spare slot ($2N$ slots, $2N - 1$ nodes).
//!
//! The grand product lands at $V\[2N - 2\]$, the root.
//!
//! # The constraints
//!
//! Reading indices as little-endian bits, $i \ge N$ is exactly "last
//! variable is $1$" and $2y + b$ is exactly "first variable is $b$". So the
//! recurrence above becomes a statement about four $\mu$-variate partial
//! evaluations of $v$, and the check is two constraints over $B_\mu$:
//!
//! * $C_1(y) = v(y, 1) - v(0, y) \cdot v(1, y)$, the tree,
//! * $C_2(y) = v(y, 0) \cdot q(y) - p(y)$, the leaves are $p/q$.
//!
//! $C_1$ holds at the spare slot for free: there $v(y, 1)$ *is* the spare
//! $0$ and $v(1, y)$ is the spare as well, so both sides vanish. The two are
//! batched with a transcript challenge $\alpha$ into the single zerocheck
//! $C_1 + \alpha C_2 \equiv 0$. Nothing here constrains the root, the
//! caller checks $v$ opens to $1$ at [`ProdcheckPoints::root`],
//! which is what makes the product a *unit* product.
//!
//! # What the caller owes
//!
//! $v$ must be committed and absorbed into the transcript *before* calling
//! [`prove`], otherwise the prover picks its tree after seeing $\alpha$ and
//! the zerocheck point. The caller then opens $v$ at the five points from
//! [`points`] and finishes with [`ProdcheckOutput::final_check`], supplying
//! $p(\rho), q(\rho)$ by whatever means it has (eg. virtual polynomials).

use alloc::vec;
use alloc::vec::Vec;

use ark_ff::{batch_inversion, PrimeField};
use ark_poly::DenseMultilinearExtension;

use crate::sumcheck::{
    SumcheckError, SumcheckProof, SumcheckProverOutput, Term,
};
use crate::transcript::Transcript;
use crate::zerocheck;

/// Errors from building the product tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProdcheckError {
    /// $p/q$ does not exist.
    ZeroDenominator { index: usize },
    /// $p$ and $q$ disagree on size $\mu$.
    NumVarsMismatch { expected: usize, got: usize },
}

/// The five points at which the committed $v$ must be opened.
///
/// The first four are the partial evaluations the zerocheck's final claim is
/// checked against; `root` is the fixed boolean point holding the grand
/// product. Both prover and verifier derive these from the same
/// $\rho$ via [`points`], so neither side chooses them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProdcheckPoints<F> {
    /// $(\rho, 0)$ - the leaf layer, $p/q$.
    pub bot: Vec<F>,
    /// $(\rho, 1)$ - the internal nodes.
    pub top: Vec<F>,
    /// $(0, \rho)$ - even slots, the left child of each node.
    pub left: Vec<F>,
    /// $(1, \rho)$ - odd slots, the right child of each node.
    pub right: Vec<F>,
    /// $(0, 1, \ldots, 1)$ - slot $2N - 2$, the root. Must open to $1$.
    pub root: Vec<F>,
}

/// The four partial evaluations of $v$
/// at $\rho$, as opened by the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProdcheckEvals<F> {
    /// $v(\rho, 0)$.
    pub bot: F,
    /// $v(\rho, 1)$.
    pub top: F,
    /// $v(0, \rho)$.
    pub left: F,
    /// $v(1, \rho)$.
    pub right: F,
}

/// Output of a successful [`verify`] call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProdcheckOutput<F> {
    /// The zerocheck challenge point $\rho$.
    pub challenges: Vec<F>,
    /// $eq(r, \rho)$ from the zerocheck.
    pub eq_eval: F,
    /// The raw sumcheck final claim.
    pub final_claim: F,
    /// The constraint-batching challenge $\alpha$.
    pub alpha: F,
    /// Where $v$ must be opened.
    pub points: ProdcheckPoints<F>,
}

impl<F: PrimeField> ProdcheckOutput<F> {
    /// Check the zerocheck's final claim against the opened
    /// values of $v$ and the caller's $p(\rho), q(\rho)$. Whether
    /// $eq(r, \rho) \cdot (C_1(\rho) + \alpha C_2(\rho))$ matches the claim.
    ///
    /// This does **not** check the root opening; the caller must separately
    /// require that $v$ opens to $1$ at [`ProdcheckPoints::root`].
    pub fn final_check(
        &self,
        evals: &ProdcheckEvals<F>,
        p_at_rho: F,
        q_at_rho: F,
    ) -> bool {
        let tree = evals.top - evals.left * evals.right;
        let leaves = evals.bot * q_at_rho - p_at_rho;
        // Because the sumcheck is run on $eq(r, \rho) \cdot C(\rho)$.
        self.eq_eval * (tree + self.alpha * leaves) == self.final_claim
    }
}

/// Build the five opening points from the zerocheck challenge $\rho$.
///
/// `rho` has $\mu$ coordinates; every returned point has $\mu + 1$, matching
/// $v$. Variable order is little-endian, so pushing to the back fixes the
/// last variable and inserting at the front fixes the first.
pub fn points<F: PrimeField>(rho: &[F]) -> ProdcheckPoints<F> {
    let last = |bit: F| {
        let mut point = rho.to_vec();
        point.push(bit);
        point
    };
    let first = |bit: F| {
        let mut point = Vec::with_capacity(rho.len() + 1);
        point.push(bit);
        point.extend_from_slice(rho);
        point
    };
    // Slot $2N - 2 = 2(N - 1)$: first bit clear, the rest set.
    let mut root = vec![F::one(); rho.len() + 1];
    root[0] = F::zero();
    ProdcheckPoints {
        bot: last(F::zero()),
        top: last(F::one()),
        left: first(F::zero()),
        right: first(F::one()),
        root,
    }
}

/// Build the product-tree polynomial $v$ for $p / q$. One batch
/// inversion plus one pass over the tree, so $O(2^\mu)$ field
/// operations and $2^{\mu+1}$ field elements of memory.
///
/// The caller commits the result, absorbs the commitment, and passes it
/// back to [`prove`]. Errors if $q$ vanishes anywhere on the cube.
///
/// # Panics
///
/// Panics if $\mu = 0$; the reduction needs
/// at least one variable to sumcheck over.
pub fn build<F: PrimeField>(
    p: &DenseMultilinearExtension<F>,
    q: &DenseMultilinearExtension<F>,
) -> Result<DenseMultilinearExtension<F>, ProdcheckError> {
    let num_vars = p.num_vars;
    if q.num_vars != num_vars {
        return Err(ProdcheckError::NumVarsMismatch {
            expected: num_vars,
            got: q.num_vars,
        });
    }
    // Panic when $\mu = 0$, consider returning an error
    assert!(num_vars > 0, "prodcheck::build: mu must be at least 1");
    let n = 1usize << num_vars;
    // `batch_inversion` leaves zeros untouched rather than failing, so the
    // zeros have to be ruled out up front or they'd silently become 0/0.
    if let Some(index) = q.evaluations.iter().position(|e| e.is_zero()) {
        return Err(ProdcheckError::ZeroDenominator { index });
    }
    let mut evals = q.evaluations.clone();
    batch_inversion(&mut evals);
    for (leaf, p_at) in evals.iter_mut().zip(&p.evaluations) {
        *leaf *= p_at;
    }
    evals.resize(2 * n, F::zero());
    // Both children of slot `i` sit at indices `< i`, so a single
    // forward pass fills every layer. The last slot stays zero.
    for i in n..(2 * n - 1) {
        let child = 2 * (i - n);
        evals[i] = evals[child] * evals[child + 1];
    }
    Ok(DenseMultilinearExtension::from_evaluations_vec(
        num_vars + 1,
        evals,
    ))
}

/// $v$ with its last variable fixed to `bit`:
/// the low or high half of the evaluation vector.
fn fix_last<F: PrimeField>(
    v: &DenseMultilinearExtension<F>,
    bit: bool,
) -> DenseMultilinearExtension<F> {
    let num_vars = v.num_vars - 1;
    let half = 1usize << num_vars;
    let base = if bit { half } else { 0 };
    DenseMultilinearExtension::from_evaluations_slice(
        num_vars,
        &v.evaluations[base..base + half],
    )
}

/// $v$ with its first variable fixed to `bit`: every other slot,
/// since the first variable is the low bit of the index.
fn fix_first<F: PrimeField>(
    v: &DenseMultilinearExtension<F>,
    bit: bool,
) -> DenseMultilinearExtension<F> {
    let num_vars = v.num_vars - 1;
    let offset = usize::from(bit);
    let evals: Vec<F> = (0..(1usize << num_vars))
        .map(|y| v.evaluations[2 * y + offset])
        .collect();
    DenseMultilinearExtension::from_evaluations_vec(num_vars, evals)
}

/// Prove $\prod_{x} p(x)/q(x) = 1$ given the tree `v`.
///
/// Squeezes the batching challenge $\alpha$, runs the zerocheck for
/// $C_1 + \alpha C_2$ over $B_\mu$. Returns the sumcheck prover output; its
/// challenge point $\rho$ is where the caller opens $v$, $p$, and $q$.
///
/// # Panics
///
/// Panics if `v` does not have exactly one more variable
/// than `p` and `q`, or if `p` and `q` disagree on $\mu$.
pub fn prove<F: PrimeField>(
    v: &DenseMultilinearExtension<F>,
    p: &DenseMultilinearExtension<F>,
    q: &DenseMultilinearExtension<F>,
    transcript: &mut Transcript,
) -> SumcheckProverOutput<F> {
    let num_vars = p.num_vars;
    assert_eq!(
        q.num_vars, num_vars,
        "prodcheck::prove: q num_vars must be mu"
    );
    assert_eq!(
        v.num_vars,
        num_vars + 1,
        "prodcheck::prove: v num_vars must be mu + 1",
    );
    let alpha: F = transcript.challenge(b"prodcheck_alpha");
    let factors = [
        fix_last(v, true),
        fix_first(v, false),
        fix_first(v, true),
        fix_last(v, false),
        q.clone(),
        p.clone(),
    ];
    zerocheck::prove(&factors, &terms(alpha), transcript)
}

/// Verify a product-check proof, `num_vars` being $\mu$ (one fewer than
/// $v$). Squeezes the same $\alpha$, verifies the zerocheck, and returns
/// what the caller needs to finish: the points to open $v$ at and the
/// final-claim check in [`ProdcheckOutput::final_check`].
pub fn verify<F: PrimeField>(
    num_vars: usize,
    proof: &SumcheckProof<F>,
    transcript: &mut Transcript,
) -> Result<ProdcheckOutput<F>, SumcheckError> {
    let alpha: F = transcript.challenge(b"prodcheck_alpha");
    // Both constraints are degree 2 in the factors.
    let out = zerocheck::verify(num_vars, 2, proof, transcript)?;
    Ok(ProdcheckOutput {
        points: points(&out.challenges),
        challenges: out.challenges,
        eq_eval: out.eq_eval,
        final_claim: out.final_claim,
        alpha,
    })
}

/// $C_1 + \alpha C_2$ over the factor slice built in [`prove`]:
/// `0` top, `1` left, `2` right, `3` bot, `4` q, `5` p.
fn terms<F: PrimeField>(alpha: F) -> [Term<F>; 4] {
    [
        Term {
            coeff: F::one(),
            factors: vec![0],
        },
        Term {
            coeff: -F::one(),
            factors: vec![1, 2],
        },
        Term {
            coeff: alpha,
            factors: vec![3, 4],
        },
        Term {
            coeff: -alpha,
            factors: vec![5],
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_ff::UniformRand;
    use ark_poly::Polynomial;
    use ark_std::test_rng;

    /// $(p, q)$ over $B_\mu$ whose ratios multiply to one: `q` holds the
    /// same values as `p`, reordered, so the products match term for term.
    fn unit_instance(
        num_vars: usize,
        rng: &mut impl ark_std::rand::RngCore,
    ) -> (DenseMultilinearExtension<Fr>, DenseMultilinearExtension<Fr>) {
        let n = 1usize << num_vars;
        let p_evals: Vec<Fr> = (0..n).map(|_| Fr::rand(rng)).collect();
        let q_evals: Vec<Fr> = (0..n).map(|x| p_evals[(x + 1) % n]).collect();
        (
            DenseMultilinearExtension::from_evaluations_vec(num_vars, p_evals),
            DenseMultilinearExtension::from_evaluations_vec(num_vars, q_evals),
        )
    }

    /// Open `v` at the four points
    /// the final check needs.
    fn evals_at(
        v: &DenseMultilinearExtension<Fr>,
        points: &ProdcheckPoints<Fr>,
    ) -> ProdcheckEvals<Fr> {
        ProdcheckEvals {
            bot: v.evaluate(&points.bot),
            top: v.evaluate(&points.top),
            left: v.evaluate(&points.left),
            right: v.evaluate(&points.right),
        }
    }

    #[test]
    // The tree really does accumulate the grand product
    // at its root, and the spare slot stays zero.
    fn root_holds_the_grand_product() {
        let mut rng = test_rng();
        let num_vars = 4;
        let n = 1usize << num_vars;
        let (p, q) = unit_instance(num_vars, &mut rng);
        let v = build(&p, &q).unwrap();
        let expected = (0..n)
            .map(|x| p.evaluations[x] / q.evaluations[x])
            .product();
        assert_eq!(v.evaluations[2 * n - 2], expected);
        assert_eq!(v.evaluations[2 * n - 1], Fr::from(0u64));
        // A permuted denominator makes it a unit product
        assert_eq!(expected, Fr::from(1u64));
    }

    #[test]
    // The convention the whole reduction rests on: the points handed to the
    // PCS pick out exactly the partial evaluations the sumcheck folded.
    // Little-endian, so `bot`/`top` are contiguous halves and
    // `left`/`right` are the even/odd strides.
    fn points_match_the_partial_evaluations() {
        let mut rng = test_rng();
        let num_vars = 4;
        let (p, q) = unit_instance(num_vars, &mut rng);
        let v = build(&p, &q).unwrap();
        let rho: Vec<Fr> = (0..num_vars).map(|_| Fr::rand(&mut rng)).collect();
        let points = points(&rho);
        assert_eq!(v.evaluate(&points.bot), fix_last(&v, false).evaluate(&rho));
        assert_eq!(v.evaluate(&points.top), fix_last(&v, true).evaluate(&rho));
        assert_eq!(
            v.evaluate(&points.left),
            fix_first(&v, false).evaluate(&rho)
        );
        assert_eq!(
            v.evaluate(&points.right),
            fix_first(&v, true).evaluate(&rho)
        );
        // The root point addresses slot $2N - 2$
        assert_eq!(
            v.evaluate(&points.root),
            v.evaluations[(1 << (num_vars + 1)) - 2],
        );
    }

    #[test]
    fn round_trip_accepts_unit_product() {
        let mut rng = test_rng();
        let num_vars = 5;
        let (p, q) = unit_instance(num_vars, &mut rng);
        let v = build(&p, &q).unwrap();
        let mut p_t = Transcript::new(b"prodcheck");
        let proof = prove(&v, &p, &q, &mut p_t).proof;
        let mut v_t = Transcript::new(b"prodcheck");
        let out = verify(num_vars, &proof, &mut v_t).unwrap();
        let opened = evals_at(&v, &out.points);
        assert!(out.final_check(
            &opened,
            p.evaluate(&out.challenges),
            q.evaluate(&out.challenges),
        ));
        assert_eq!(v.evaluate(&out.points.root), Fr::from(1u64));
    }

    #[test]
    // A well-formed tree over a non-unit product: the constraints still
    // hold, so only the root check can reject. This is why the caller
    // owes that check, `final_check` alone is not the whole protocol.
    fn non_unit_product_fails_only_at_the_root() {
        let mut rng = test_rng();
        let num_vars = 4;
        let n = 1usize << num_vars;
        let p_evals: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        let q_evals: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        let p =
            DenseMultilinearExtension::from_evaluations_vec(num_vars, p_evals);
        let q =
            DenseMultilinearExtension::from_evaluations_vec(num_vars, q_evals);
        let v = build(&p, &q).unwrap();
        let mut p_t = Transcript::new(b"prodcheck");
        let proof = prove(&v, &p, &q, &mut p_t).proof;
        let mut v_t = Transcript::new(b"prodcheck");
        let out = verify(num_vars, &proof, &mut v_t).unwrap();
        let opened = evals_at(&v, &out.points);
        assert!(out.final_check(
            &opened,
            p.evaluate(&out.challenges),
            q.evaluate(&out.challenges),
        ));
        assert_ne!(v.evaluate(&out.points.root), Fr::from(1u64));
    }

    #[test]
    // Forging the root to one to fake a unit product breaks the recurrence
    // at that node. An honest prover of the forged tree then sumchecks
    // a constraint that doesn't vanish, so this dies at round
    // 0 rather than surviving to the final check.
    fn rejects_forged_root() {
        let mut rng = test_rng();
        let num_vars = 4;
        let n = 1usize << num_vars;
        let p_evals: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        let q_evals: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        let p =
            DenseMultilinearExtension::from_evaluations_vec(num_vars, p_evals);
        let q =
            DenseMultilinearExtension::from_evaluations_vec(num_vars, q_evals);
        let mut v = build(&p, &q).unwrap();
        v.evaluations[2 * n - 2] = Fr::from(1u64);
        let mut p_t = Transcript::new(b"prodcheck");
        let proof = prove(&v, &p, &q, &mut p_t).proof;
        let mut v_t = Transcript::new(b"prodcheck");
        let err = verify::<Fr>(num_vars, &proof, &mut v_t).unwrap_err();
        assert!(matches!(err, SumcheckError::RoundCheckFailed { round: 0 }));
    }

    #[test]
    // Leaves that aren't $p/q$: the second constraint is what catches this,
    // so it also pins that $\alpha$ actually mixes both constraints in.
    // The total goes non-zero due to Schwartz-Zippel. Round 0 is
    // where a false statement from an honest prover fails.
    fn rejects_tampered_leaf() {
        let mut rng = test_rng();
        let num_vars = 4;
        let (p, q) = unit_instance(num_vars, &mut rng);
        let mut v = build(&p, &q).unwrap();
        v.evaluations[3] += Fr::from(1u64);
        let mut p_t = Transcript::new(b"prodcheck");
        let proof = prove(&v, &p, &q, &mut p_t).proof;
        let mut v_t = Transcript::new(b"prodcheck");
        let err = verify::<Fr>(num_vars, &proof, &mut v_t).unwrap_err();
        assert!(matches!(err, SumcheckError::RoundCheckFailed { round: 0 }));
    }

    #[test]
    // The final check constrains all four openings, not just some.
    // A cheating prover reaches here by passing the round checks
    // and then claiming an opening the PCS wouldn't back. No
    // incentive for attacking but still something we expect.
    fn final_check_rejects_wrong_opening() {
        let mut rng = test_rng();
        let num_vars = 4;
        let (p, q) = unit_instance(num_vars, &mut rng);
        let v = build(&p, &q).unwrap();
        let mut p_t = Transcript::new(b"prodcheck");
        let proof = prove(&v, &p, &q, &mut p_t).proof;
        let mut v_t = Transcript::new(b"prodcheck");
        let out = verify(num_vars, &proof, &mut v_t).unwrap();
        let honest = evals_at(&v, &out.points);
        let p_at = p.evaluate(&out.challenges);
        let q_at = q.evaluate(&out.challenges);
        assert!(out.final_check(&honest, p_at, q_at));
        for tamper in [
            ProdcheckEvals {
                top: honest.top + Fr::from(1u64),
                ..honest
            },
            ProdcheckEvals {
                left: honest.left + Fr::from(1u64),
                ..honest
            },
            ProdcheckEvals {
                right: honest.right + Fr::from(1u64),
                ..honest
            },
            ProdcheckEvals {
                bot: honest.bot + Fr::from(1u64),
                ..honest
            },
        ] {
            assert!(!out.final_check(&tamper, p_at, q_at));
        }
    }

    #[test]
    fn build_rejects_zero_denominator() {
        let mut rng = test_rng();
        let num_vars = 3;
        let (p, mut q) = unit_instance(num_vars, &mut rng);
        q.evaluations[5] = Fr::from(0u64);
        assert_eq!(
            build(&p, &q),
            Err(ProdcheckError::ZeroDenominator { index: 5 })
        );
    }

    #[test]
    fn build_rejects_num_vars_mismatch() {
        let mut rng = test_rng();
        let (p, _) = unit_instance(3, &mut rng);
        let (_, q) = unit_instance(4, &mut rng);
        assert_eq!(
            build(&p, &q),
            Err(ProdcheckError::NumVarsMismatch {
                expected: 3,
                got: 4
            })
        );
    }
}
