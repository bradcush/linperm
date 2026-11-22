//! Proves $g(\sigma(x)) = f(x)$ for all $x \in B_\mu$ via the same Lemma 4
//! reduction as BiPerm, but splits the indicator into $\ell$ groups of $w$
//! bits each instead of two halves. The partial products are never
//! committed, so the only index oracle is $\tilde{\sigma}$, which is
//! dense and works with any PCS. For now $\mu$ must be a power of two, so
//! the $\mu$ bit slots split evenly into $\ell \cdot w = \mu$ with no padding.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub use permcore;

use alloc::vec::Vec;

use ark_ff::PrimeField;
use ark_poly::{DenseMultilinearExtension, Polynomial};
use permcore::eq::eq_eval_table;
use permcore::sumcheck::{self, SumcheckProof, SumcheckProverOutput};
use permcore::{CoreError, Permutation, PolynomialCommitment, Transcript};

/// How the $\mu$ bit slots of $\sigma(x)$ split into groups. Slot
/// $b = j \cdot w + i$ is bit $i \in \[w\]$ of group $j \in [\ell]$.
/// Remember we have $j \in [\ell]$, so $j$ the group #.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Groups {
    /// $\ell$, number of groups
    pub ell: usize,
    /// $w = \mu / \ell$,
    /// number of bits per group
    pub width: usize,
}

impl Groups {
    /// Validate $\ell$ for $\mu$ = `num_vars`. Both must be powers of two
    /// with $\ell \le \mu$, so $w = \mu / \ell$ is a power of two too. If
    /// $\ell$ wasn't a power of 2, we'd need padding when representing.
    pub fn new(num_vars: usize, ell: usize) -> Result<Self, ParamsError> {
        if !num_vars.is_power_of_two() {
            return Err(ParamsError::NumVarsNotPowerOfTwo { num_vars });
        }
        if !ell.is_power_of_two() || ell > num_vars {
            return Err(ParamsError::InvalidEll { ell, num_vars });
        }
        Ok(Self {
            ell,
            width: num_vars / ell,
        })
    }

    /// The default $\ell = 2^{\lfloor \log(\mu) / 2 \rfloor}$, the
    /// paper's $\ell = \sqrt{\mu}$ rounded down to a power of two.
    /// This is asymptotically optimal for the prover's cost.
    pub fn default_for(num_vars: usize) -> Result<Self, ParamsError> {
        let log_mu = num_vars.trailing_zeros();
        Self::new(num_vars, 1 << (log_mu / 2))
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum ParamsError {
    /// Only power-of-two $\mu$ is supported
    NumVarsNotPowerOfTwo { num_vars: usize },
    /// $\ell$ must be a power of two with $\ell \le \mu$
    InvalidEll { ell: usize, num_vars: usize },
}

#[derive(Debug, PartialEq, Eq)]
pub enum MulPermError<E> {
    Params(ParamsError),
    Core(CoreError),
    /// PCS operation produced an error
    Pcs(E),
}

impl<E> From<ParamsError> for MulPermError<E> {
    fn from(e: ParamsError) -> Self {
        Self::Params(e)
    }
}

impl<E> From<CoreError> for MulPermError<E> {
    fn from(e: CoreError) -> Self {
        Self::Core(e)
    }
}

/// Built once by [`index`] and reused across proofs
/// because it only depends on $\sigma$ and $\ell$.
pub struct MulPermProverIndex<F: PrimeField, P: PolynomialCommitment<F>> {
    pub perm: Permutation,
    pub groups: Groups,
    pub sigma: DenseMultilinearExtension<F>,
    pub sigma_commit: P::Commitment,
}

/// Verifier half of the index: $\mu$, the grouping, and the
/// commitment to $\tilde{\sigma}$. Indexing is deterministic and
/// public, so anyone can recompute this from $\sigma$ and $\ell$.
pub struct MulPermVerifierIndex<F: PrimeField, P: PolynomialCommitment<F>> {
    pub num_vars: usize,
    pub groups: Groups,
    pub sigma_commit: P::Commitment,
}

/// Output of [`index`], computing
/// the prover and verifier halves.
pub type MulPermIndex<F, P> =
    (MulPermProverIndex<F, P>, MulPermVerifierIndex<F, P>);

/// The paper's $\tilde{\sigma}[\mu]$ over $\log \mu + \mu$ variables, with
/// $\tilde{\sigma}(\langle b \rangle, x) = \sigma_b(x)$, the $b$-th bit of
/// $\sigma(x)$. $b$ occupies the low $\log \mu$ variables and $x$ the
/// high $\mu$, so entry $b + \mu \cdot x$ holds $\sigma_b(x)$.
///
/// Since $b = j \cdot w + i$, the low $\log w$ variables of $b$ are $i$
/// and the next $\log \ell$ are $j$. The table is the same for every
/// $\ell$; only how a query point is read changes.
pub fn sigma_mle<F: PrimeField>(
    perm: &Permutation,
) -> DenseMultilinearExtension<F> {
    let num_vars = perm.num_vars();
    assert!(num_vars.is_power_of_two(), "μ must be 2^k");
    let log_mu = num_vars.trailing_zeros() as usize;
    // Bit b of $\sigma(x)$ as 0 or 1, for each x and
    // slot b. Flat map for a single array of evaluations.
    let evals: Vec<F> =
        (0..perm.size())
            .flat_map(|x| {
                (0..num_vars).map(move |b| {
                    if perm.bit(b, x) {
                        F::one()
                    } else {
                        F::zero()
                    }
                })
            })
            .collect();
    DenseMultilinearExtension::from_evaluations_vec(log_mu + num_vars, evals)
}

/// $\tilde{\alpha}$ over $\log \mu$ variables, the paper's reading of the
/// challenge $\alpha \in \mathbb{F}^\mu$ as a polynomial with
/// $\tilde{\alpha}(\langle b \rangle) = \alpha_b$. Laid out like the $b$ part
/// of [`sigma_mle`], so $\tilde{\alpha}(\langle i \rangle, \langle j \rangle)
/// = \alpha_{j \cdot w + i}$. `alpha.len()` = $\mu$ must be a power of two.
/// Stated differently, $\alpha$ as a poly, so we can use in the relation.
pub fn alpha_mle<F: PrimeField>(alpha: &[F]) -> DenseMultilinearExtension<F> {
    assert!(alpha.len().is_power_of_two(), "α length must be 2^k");
    let log_mu = alpha.len().trailing_zeros() as usize;
    DenseMultilinearExtension::from_evaluations_slice(log_mu, alpha)
}

/// Preprocess a fixed $\sigma$ with an explicit $\ell$:
/// build and commit $\tilde{\sigma}$. Runs once per
/// permutation, independent of any $f, g$ instance.
pub fn index_with<F: PrimeField, P: PolynomialCommitment<F>>(
    pk: &P::ProverKey,
    perm: &Permutation,
    ell: usize,
) -> Result<MulPermIndex<F, P>, MulPermError<P::Error>> {
    let groups = Groups::new(perm.num_vars(), ell)?;
    let sigma = sigma_mle::<F>(perm);
    let sigma_commit =
        P::commit(pk, (&sigma).into()).map_err(MulPermError::Pcs)?;
    Ok((
        MulPermProverIndex {
            perm: perm.clone(),
            groups,
            sigma,
            sigma_commit: sigma_commit.clone(),
        },
        MulPermVerifierIndex {
            num_vars: perm.num_vars(),
            groups,
            sigma_commit,
        },
    ))
}

/// Preprocess a fixed $\sigma$ with the default
/// $\ell$ from [`Groups::default_for`]; see [`index_with`].
pub fn index<F: PrimeField, P: PolynomialCommitment<F>>(
    pk: &P::ProverKey,
    perm: &Permutation,
) -> Result<MulPermIndex<F, P>, MulPermError<P::Error>> {
    let groups = Groups::default_for(perm.num_vars())?;
    index_with(pk, perm, groups.ell)
}

/// Algorithm 4: the partial products $p(x, \langle j \rangle) =
/// \mathrm{eq}(\alpha_{I_j}, \sigma_{I_j}(x))$ of Eq. 6 for every
/// $x \in B_\mu$ and $j \in [\ell]$, where $I_j$ is group $j$'s $w$ bit
/// slots. Their product over $j$ is $\mathrm{eq}(\alpha, \sigma(x))$.
/// Group $j$'s bits take only $2^w$ values, so `eq_eval_table` of
/// $\alpha_{I_j}$ holds every possible $p(\cdot, \langle j \rangle)$, and
/// each entry is one lookup by `perm.group`. Cost is $O(\ell \cdot 2^w)$ field
/// operations (Lemma 5) plus $n \ell$ lookups, instead of the naive $n \mu$
/// multiplications required to build a table of size $n \dot \ell$.
///
/// Takes $\sigma$ as `perm`, its grouping as `groups`, and the challenge
/// $\alpha \in \mathbb{F}^\mu$ as `alpha`. Returns $\ell$ multilinear
/// extensions over $\mu$ variables, entry $j$ being $\tilde{p}(\cdot, \langle j \rangle)$.
/// Laid end to end they are the table of $\tilde{p}$ over $\mu + \log \ell$
/// variables, $x$ low and $j$ high.
///
/// # Panics
///
/// Panics if `alpha.len()` or $\ell \cdot w$ isn't $\mu$.
pub fn partial_products<F: PrimeField>(
    perm: &Permutation,
    groups: Groups,
    alpha: &[F],
) -> Result<Vec<DenseMultilinearExtension<F>>, CoreError> {
    let Groups { ell, width } = groups;
    let num_vars = perm.num_vars();
    assert_eq!(alpha.len(), num_vars, "α length must be μ");
    assert_eq!(ell * width, num_vars, "groups must cover μ");
    (0..ell)
        .map(|j| {
            // Covers $j \cdot w$ through $j \cdot w + w − 1$, $I_j$.
            // Picking out values from $\alpha $ for the decomposition.
            // The group #, the spread in that group, the group position.
            // $eq_t[s] = eq(\alpha_{I_j}, s)$ for each of the $2^w$ patterns.
            let eq_t = eq_eval_table(&alpha[j * width..(j + 1) * width]);
            let evals = (0..perm.size())
                // We need every $x$ evaluation for each $j$.
                // We get the entry from the evaluation based on
                // the $\ell$, $j$, and $x$ we're interested in.
                .map(|x| Ok(eq_t[perm.group(ell, j, x)?]))
                .collect::<Result<Vec<F>, CoreError>>()?;
            Ok(DenseMultilinearExtension::from_evaluations_vec(
                num_vars, evals,
            ))
        })
        .collect()
}

/// Algorithm 5's prover. Sumcheck 1 proves Eq. 8,
/// $\sum_{x \in B_\mu} f(x) \prod_j \tilde{p}(x, \langle j \rangle) =
/// g(\alpha)$, over the factors $f$ and the $\ell$ partial products, so
/// its degree is $\ell + 1$. The sum is Lemma 4's, with
/// $\mathrm{eq}(\alpha, \sigma(x))$ split into its groups.
///
/// Takes $f$ as `f`, the output of [`partial_products`]
/// as `parts`, and the Fiat–Shamir `transcript`. Returns the
/// sumcheck output, whose challenges are $\beta \in \mathbb{F}^\mu$, and
/// $v_j = \tilde{p}(\beta, \langle j \rangle)$ for each $j$. The verifier
/// checks the final claim as $C_1 = f(\beta) \prod_j v_j$, then
/// sumcheck 2 proves the $v_j$.
pub fn sumcheck1_prove<F: PrimeField>(
    f: &DenseMultilinearExtension<F>,
    parts: Vec<DenseMultilinearExtension<F>>,
    transcript: &mut Transcript,
) -> (SumcheckProverOutput<F>, Vec<F>) {
    let mut factors = Vec::with_capacity(parts.len() + 1);
    factors.push(f.clone());
    factors.extend(parts);
    let output = sumcheck::prove(&factors, transcript);
    let p_at_beta = factors[1..]
        .iter()
        .map(|p| p.evaluate(&output.challenges))
        .collect();
    (output, p_at_beta)
}

/// Algorithm 7's prover, by direct computation in every
/// round, with no bucketing yet. Sumcheck 2 proves Eq. 9,
/// $\sum_{x, j} \mathrm{eq}(\beta', (x, j)) \prod_i E(\tilde{\alpha}_i,
/// \tilde{\sigma}_i) = \tilde{p}(\beta')$, with $\tilde{\alpha}_i =
/// \tilde{\alpha}(\langle i \rangle, j)$, $\tilde{\sigma}_i =
/// \tilde{\sigma}(\langle i \rangle, j, x)$ and $E(a, s) = a s + (1 - a)(1 -
/// s)$, Definition 2's $\mathrm{eq}$ for one variable. Variables are $x$
/// low and $j$ high, so the $\mu$ rounds over $x$ come first. Each $E$ has
/// degree 2 in $j$, so the degree is $2w + 1$ in every round.
///
/// Each round evaluates the summand at $c = 0, \ldots, 2w + 1$
/// directly, $O(w)$ per point and table pair, so $O(n \mu w)$ over
/// all rounds. $\tilde{\alpha}_i$ doesn't depend on $x$, so it's
/// kept as $\ell$ values and only folds in the $j$ rounds.
///
/// Takes the index's $\tilde{\sigma}$ as `sigma`, its grouping as `groups`,
/// $\alpha$ as `alpha`, $\beta' = (\beta, t)$ as `beta_t` and the
/// Fiat–Shamir `transcript`. Returns the sumcheck output, whose challenges
/// are $\gamma = (x^*, j^*)$. Transcript labels match [`sumcheck::prove`],
/// so [`sumcheck::verify`] checks the proof at degree $2w + 1$.
///
/// # Panics
///
/// Panics if `alpha.len()` isn't $\ell \cdot w$, `sigma` isn't over
/// $\log \mu + \mu$ variables, or `beta_t` isn't over $\mu + \log \ell$.
pub fn sumcheck2_prove<F: PrimeField>(
    sigma: &DenseMultilinearExtension<F>,
    groups: Groups,
    alpha: &[F],
    // $\beta$ w/ $t$ appended, $\mu + \log \ell$ field elements,
    // sumcheck 1's challenge point and a random point in $J$ variables
    beta_t: &[F],
    transcript: &mut Transcript,
) -> SumcheckProverOutput<F> {
    let Groups { ell, width } = groups;
    let num_vars = alpha.len();
    let n = 1 << num_vars;
    let total = num_vars + ell.trailing_zeros() as usize;
    assert_eq!(ell * width, num_vars, "groups must cover μ");
    // Because one entry per (bit slot $b$, input $x$)
    assert_eq!(sigma.evaluations.len(), num_vars * n, "σ̃ size");
    assert_eq!(beta_t.len(), total, "β′ must have μ + log ℓ entries");
    let degree = 2 * width + 1;
    let mut eq_t = eq_eval_table(beta_t);
    // $\tilde{\sigma}_i$ at entry $x + n \cdot j$ is $\tilde{\sigma}$'s
    // entry $(j \cdot w + i) + \mu \cdot x$, moving $x$ to the low end.
    // Just a reshuffle of $\tilde{\sigma}$, to use in the prover
    let mut s: Vec<Vec<F>> = (0..width)
        .map(|i| {
            (0..ell)
                .flat_map(|j| {
                    (0..n).map(move |x| {
                        sigma.evaluations[j * width + i + num_vars * x]
                    })
                })
                .collect()
        })
        .collect();
    let mut a: Vec<Vec<F>> = (0..width)
        .map(|i| (0..ell).map(|j| alpha[j * width + i]).collect())
        .collect();
    // Size of the $x$ part of each
    // table, 1 once $x$ is bound
    let mut x_len = n;
    let mut round_polys = Vec::with_capacity(total);
    let mut challenges = Vec::with_capacity(total);
    // Each factor's value at the current $c$, and its step to $c + 1$
    let mut a_c = alloc::vec![F::zero(); width];
    let mut a_step = alloc::vec![F::zero(); width];
    let mut s_c = alloc::vec![F::zero(); width];
    let mut s_step = alloc::vec![F::zero(); width];
    // For each round we update the transcript, add the
    // round poly (as evals), and push the challenges.
    for _ in 0..total {
        let half = eq_t.len() / 2;
        let mut evals = alloc::vec![F::zero(); degree + 1];
        for y in 0..half {
            for i in 0..width {
                s_c[i] = s[i][2 * y];
                s_step[i] = s[i][2 * y + 1] - s_c[i];
                // In an $x$ round both entries of the pair share
                // $j = y / (x_{len} / 2)$, so $\tilde{\alpha}_i$ is constant.
                (a_c[i], a_step[i]) = if x_len > 1 {
                    (a[i][y / (x_len / 2)], F::zero())
                } else {
                    (a[i][2 * y], a[i][2 * y + 1] - a[i][2 * y])
                };
            }
            let mut e_c = eq_t[2 * y];
            let e_step = eq_t[2 * y + 1] - e_c;
            for eval in evals.iter_mut() {
                let mut prod = e_c;
                for i in 0..width {
                    // $E(a, s) = 1 - a - s + 2as$, one multiplication
                    let t = a_c[i] * s_c[i];
                    prod *= F::one() - a_c[i] - s_c[i] + t + t;
                    a_c[i] += a_step[i];
                    s_c[i] += s_step[i];
                }
                *eval += prod;
                e_c += e_step;
            }
        }
        transcript.append_slice(b"round", &evals);
        round_polys.push(evals);
        let r: F = transcript.challenge(b"r");
        challenges.push(r);
        // Folding fixes the variable just bound to the challenge $r$
        // in every table to optimize the work we do in the next round.
        // For completeness and soundness we don't need to do it.
        fold(&mut eq_t, r);
        s.iter_mut().for_each(|t| fold(t, r));
        if x_len > 1 {
            x_len /= 2;
        } else {
            a.iter_mut().for_each(|t| fold(t, r));
        }
    }
    SumcheckProverOutput {
        proof: SumcheckProof { round_polys },
        challenges,
    }
}

/// Bind the low variable of an evaluation table to $r$, halving it.
/// Takes the table as `table`, folded in place, and the challenge `r`.
fn fold<F: PrimeField>(table: &mut Vec<F>, r: F) {
    let half = table.len() / 2;
    for y in 0..half {
        // The line computes the value at $r$: start at the 0-value and move
        // $r$ times the gap toward the 1-value. Rebuild on top of the existing
        // table where previous values won't be needed in future computation.
        table[y] = table[2 * y] + r * (table[2 * y + 1] - table[2 * y]);
    }
    table.truncate(half);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_ff::UniformRand;
    use ark_std::test_rng;
    use permcore::eq::eq;
    use permcore::sumcheck::SumcheckError;
    use permcore::MockPcs;

    /// The low `k` bits of `v` as a
    /// boolean point, little-endian.
    fn bits(v: usize, k: usize) -> Vec<Fr> {
        // We're just moving each over to pick off the bits. This is just a
        // helper to get exact bits for little-endian point representation.
        (0..k).map(|t| Fr::from(((v >> t) & 1) as u64)).collect()
    }

    /// 16-element perm, so $\mu = 4$ (a power of two)
    fn sample_perm() -> Permutation {
        // All unique (0..16)
        Permutation::new(alloc::vec![
            5, 3, 7, 1, 0, 6, 2, 4, 9, 11, 8, 10, 13, 15, 12, 14
        ])
        .unwrap()
    }

    /// $\mu = 8$, each $x$ maps to $x + 1$ and $255$
    /// wraps to $0$. Gives groups of up to 8 bits.
    fn shift_perm() -> Permutation {
        Permutation::new((0..256).map(|x| (x + 1) % 256).collect()).unwrap()
    }

    /// Naive Eq. 6, one bit-wise eq per slot of group $j$, so $w$
    /// multiplications per entry and $O(n \mu)$ for the whole table.
    /// We do the naive partial product for an easy correctness check.
    fn partial_product_naive(
        perm: &Permutation,
        width: usize,
        alpha: &[Fr],
        j: usize,
        x: usize,
    ) -> Fr {
        // For a single group $j$ which
        // is what we'll check against
        (0..width)
            .map(|i| {
                let b = j * width + i;
                let bit = Fr::from(perm.bit(b, x) as u64);
                eq(&[alpha[b]], &[bit])
            })
            .product()
    }

    /// Random $f$ and $g$ with $g(\sigma(x)) = f(x)$
    /// for all $x \in B_\mu$ like we have elsewhere.
    /// Given some permutation gives consistent pair.
    fn consistent_pair(
        perm: &Permutation,
        rng: &mut impl ark_std::rand::RngCore,
    ) -> (DenseMultilinearExtension<Fr>, DenseMultilinearExtension<Fr>) {
        let num_vars = perm.num_vars();
        let f_evals: Vec<Fr> =
            (0..perm.size()).map(|_| Fr::rand(rng)).collect();
        let mut g_evals = alloc::vec![Fr::from(0u64); perm.size()];
        for x in 0..perm.size() {
            g_evals[perm.apply(x)] = f_evals[x];
        }
        (
            DenseMultilinearExtension::from_evaluations_vec(num_vars, f_evals),
            DenseMultilinearExtension::from_evaluations_vec(num_vars, g_evals),
        )
    }

    /// Prover side of sumcheck 1 at a random $\alpha$. Returns
    /// the claim $g(\alpha)$, which the verifier gets from
    /// $g$'s opening, the sumcheck output, and the $v_j$.
    /// Working at the PIOP level, just evaluate $g$.
    fn run_sumcheck1(
        perm: &Permutation,
        ell: usize,
        f: &DenseMultilinearExtension<Fr>,
        g: &DenseMultilinearExtension<Fr>,
        rng: &mut impl ark_std::rand::RngCore,
    ) -> (Fr, SumcheckProverOutput<Fr>, Vec<Fr>) {
        let num_vars = perm.num_vars();
        let alpha: Vec<Fr> = (0..num_vars).map(|_| Fr::rand(rng)).collect();
        let groups = Groups::new(num_vars, ell).unwrap();
        let parts = partial_products(perm, groups, &alpha).unwrap();
        let mut p_t = Transcript::new(b"mulperm");
        let (out, v) = sumcheck1_prove(f, parts, &mut p_t);
        (g.evaluate(&alpha), out, v)
    }

    /// The final check $C_2 = \mathrm{eq}(\beta', \gamma) \prod_i
    /// E(\tilde{\alpha}(\langle i \rangle, j^*), \tilde{\sigma}(\langle i
    /// \rangle, j^*, x^*))$ at $\gamma = (x^*, j^*)$, with $\tilde{\sigma}$
    /// evaluated directly in place of its $w$ openings.
    fn c2(
        sigma: &DenseMultilinearExtension<Fr>,
        groups: Groups,
        alpha: &[Fr],
        beta_t: &[Fr],
        // $\gamma$ is the random
        // point Sumcheck 2 ends at
        gamma: &[Fr],
    ) -> Fr {
        let log_w = groups.width.trailing_zeros() as usize;
        let (x_star, j_star) = gamma.split_at(alpha.len());
        let alpha_t = alpha_mle(alpha);
        let prod: Fr = (0..groups.width)
            .map(|i| {
                let i_j: Vec<Fr> =
                    bits(i, log_w).into_iter().chain(j_star.to_vec()).collect();
                let i_j_x: Vec<Fr> =
                    i_j.iter().chain(x_star).copied().collect();
                // $E(a, s)$ is the one-variable eq mentioned
                eq(&[alpha_t.evaluate(&i_j)], &[sigma.evaluate(&i_j_x)])
            })
            .product();
        eq(beta_t, gamma) * prod
    }

    #[test]
    fn default_groups() {
        let cases = [(1, 1, 1), (2, 1, 2), (4, 2, 2), (8, 2, 4), (16, 4, 4)];
        // Both $\ell$ and $\mu$ are powers of two.
        for (num_vars, ell, width) in cases {
            assert_eq!(
                Groups::default_for(num_vars).unwrap(),
                Groups { ell, width }
            );
        }
    }

    #[test]
    fn groups_reject_bad_params() {
        assert_eq!(
            Groups::new(6, 2),
            Err(ParamsError::NumVarsNotPowerOfTwo { num_vars: 6 })
        );
        for ell in [0, 3, 16] {
            assert_eq!(
                Groups::new(8, ell),
                Err(ParamsError::InvalidEll { ell, num_vars: 8 })
            );
        }
    }

    #[test]
    fn sigma_matches_bits() {
        let perm = sample_perm();
        // Note $\mu$ = `num_vars`
        let num_vars = perm.num_vars();
        let log_mu = num_vars.trailing_zeros() as usize;
        let sigma = sigma_mle::<Fr>(&perm);
        assert_eq!(sigma.num_vars, log_mu + num_vars);
        for x in 0..perm.size() {
            for b in 0..num_vars {
                let want = Fr::from(perm.bit(b, x) as u64);
                // Point is $(\langle b \rangle, x)$, $b$ in the low variables.
                // We want to reduce to the # of bits we need specifically.
                let point: Vec<Fr> = bits(b, log_mu)
                    .into_iter()
                    .chain(bits(x, num_vars))
                    .collect();
                assert_eq!(sigma.evaluate(&point), want);
            }
        }
    }

    #[test]
    fn alpha_splits_into_groups() {
        let mut rng = test_rng();
        let num_vars = 8;
        let alpha: Vec<Fr> =
            (0..num_vars).map(|_| Fr::rand(&mut rng)).collect();
        let alpha_t = alpha_mle(&alpha);
        for ell in [1, 2, 8] {
            let Groups { ell, width } = Groups::new(num_vars, ell).unwrap();
            let log_w = width.trailing_zeros() as usize;
            let log_ell = ell.trailing_zeros() as usize;
            for j in 0..ell {
                for i in 0..width {
                    // Point is $(\langle i \rangle, \langle j \rangle)$
                    let point: Vec<Fr> = bits(i, log_w)
                        .into_iter()
                        .chain(bits(j, log_ell))
                        .collect();
                    // Just want to make sure we can access elements of
                    // $\alpha$ given the group # $j$ and $i$-th entry.
                    // Needed for relation/arithmetization stated before.
                    assert_eq!(alpha_t.evaluate(&point), alpha[j * width + i]);
                }
            }
        }
    }

    #[test]
    fn index_commits_sigma_for_every_ell() {
        let perm = sample_perm();
        let sigma = sigma_mle::<Fr>(&perm);
        for ell in [1, 2, 4] {
            let (p_idx, v_idx) =
                index_with::<Fr, MockPcs<Fr>>(&(), &perm, ell).unwrap();
            assert_eq!(p_idx.sigma, sigma);
            // $\tilde{\sigma}$ doesn't depend on $\ell$.
            // `sigma_commit = sigma` because mock uses poly.
            assert_eq!(v_idx.sigma_commit, sigma);
            assert_eq!(v_idx.groups, Groups::new(4, ell).unwrap());
        }
    }

    #[test]
    fn partial_products_match_naive() {
        let mut rng = test_rng();
        // Using both for $\mu = 4$ and $\mu = 8$ even though
        // there isn't much we gain from testing across both.
        for perm in [sample_perm(), shift_perm()] {
            let num_vars = perm.num_vars();
            let alpha: Vec<Fr> =
                (0..num_vars).map(|_| Fr::rand(&mut rng)).collect();
            // Different $\ell$s cover the different extremes at each
            // end. Minimum # of groups up to maximum # of groups.
            for ell in [1, 2, num_vars] {
                let groups = Groups::new(num_vars, ell).unwrap();
                let parts = partial_products(&perm, groups, &alpha).unwrap();
                assert_eq!(parts.len(), ell);
                for (j, p) in parts.iter().enumerate() {
                    assert_eq!(p.num_vars, num_vars);
                    for x in 0..perm.size() {
                        let want = partial_product_naive(
                            &perm,
                            groups.width,
                            &alpha,
                            j,
                            x,
                        );
                        assert_eq!(p.evaluations[x], want);
                    }
                }
            }
        }
    }

    #[test]
    fn partial_products_multiply_to_indicator() {
        let mut rng = test_rng();
        for perm in [sample_perm(), shift_perm()] {
            let num_vars = perm.num_vars();
            let alpha: Vec<Fr> =
                (0..num_vars).map(|_| Fr::rand(&mut rng)).collect();
            for ell in [1, 2, num_vars] {
                let groups = Groups::new(num_vars, ell).unwrap();
                let parts = partial_products(&perm, groups, &alpha).unwrap();
                for x in 0..perm.size() {
                    let prod: Fr =
                        parts.iter().map(|p| p.evaluations[x]).product();
                    // Lemma 4's $\tilde{1}_\sigma(x, \alpha)$
                    // Basically just an arithmetization check
                    let want = eq(&alpha, &bits(perm.apply(x), num_vars));
                    assert_eq!(prod, want);
                }
            }
        }
    }

    #[test]
    fn index_rejects_non_power_of_two_mu() {
        let perm = Permutation::identity(3);
        assert!(matches!(
            index::<Fr, MockPcs<Fr>>(&(), &perm),
            Err(MulPermError::Params(ParamsError::NumVarsNotPowerOfTwo {
                num_vars: 3
            }))
        ));
    }

    #[test]
    fn sumcheck1_round_trip() {
        let mut rng = test_rng();
        for perm in [sample_perm(), shift_perm()] {
            let num_vars = perm.num_vars();
            let (f, g) = consistent_pair(&perm, &mut rng);
            for ell in [1, 2, num_vars] {
                let (claim, out, v) =
                    run_sumcheck1(&perm, ell, &f, &g, &mut rng);
                assert_eq!(v.len(), ell);
                let mut v_t = Transcript::new(b"mulperm");
                let res = sumcheck::verify(
                    claim,
                    num_vars,
                    // Because $f$ and $v$s
                    ell + 1,
                    &out.proof,
                    &mut v_t,
                )
                .unwrap();
                // Both challenges output on each side because we
                // need them to prove/verify sumcheck2 later.
                assert_eq!(res.challenges, out.challenges);
                // $C_1 = f(\beta) \prod_j v_j$
                let c1 = f.evaluate(&res.challenges) * v.iter().product::<Fr>();
                assert_eq!(res.final_claim, c1);
            }
        }
    }

    #[test]
    fn sumcheck1_rejects_inconsistent_pair() {
        let mut rng = test_rng();
        let perm = shift_perm();
        let ell = 2;
        let (f, _g_good) = consistent_pair(&perm, &mut rng);
        // Random $g$ unrelated to $f$ and $\sigma$.
        let g_bad = DenseMultilinearExtension::from_evaluations_vec(
            perm.num_vars(),
            (0..perm.size()).map(|_| Fr::rand(&mut rng)).collect(),
        );
        let (claim, out, _) = run_sumcheck1(&perm, ell, &f, &g_bad, &mut rng);
        // The honest round 0 sums to the true $\sum_x f(x) \cdot
        // eq(\alpha, \sigma(x))$, which isn't `g_bad`'s $g(\alpha)$.
        let mut v_t = Transcript::new(b"mulperm");
        assert_eq!(
            sumcheck::verify(
                claim,
                perm.num_vars(),
                ell + 1,
                &out.proof,
                &mut v_t
            ),
            Err(SumcheckError::RoundCheckFailed { round: 0 })
        );
    }

    #[test]
    fn sumcheck2_round_trip() {
        let mut rng = test_rng();
        for (perm, ells) in [
            // Turning both into slices
            // gives the same type `&[usize]`.
            (sample_perm(), &[1, 2, 4][..]),
            (shift_perm(), &[1, 2, 4, 8][..]),
        ] {
            let num_vars = perm.num_vars();
            let sigma = sigma_mle::<Fr>(&perm);
            let alpha: Vec<Fr> =
                (0..num_vars).map(|_| Fr::rand(&mut rng)).collect();
            for &ell in ells {
                let groups = Groups::new(num_vars, ell).unwrap();
                let total_vars = num_vars + ell.trailing_zeros() as usize;
                let beta_t: Vec<Fr> =
                    (0..total_vars).map(|_| Fr::rand(&mut rng)).collect();
                // The claim $S_p = \tilde{p}(\beta')$ comes from Alg. 4's
                // tables laid end to end, independent of $\tilde{\sigma}$.
                let parts = partial_products(&perm, groups, &alpha).unwrap();
                let p = DenseMultilinearExtension::from_evaluations_vec(
                    total_vars,
                    parts.iter().flat_map(|p| p.evaluations.clone()).collect(),
                );
                // $\mu + \log \ell$ entries
                let claim = p.evaluate(&beta_t);
                let mut p_t = Transcript::new(b"mulperm");
                let out =
                    sumcheck2_prove(&sigma, groups, &alpha, &beta_t, &mut p_t);
                let mut v_t = Transcript::new(b"mulperm");
                let res = sumcheck::verify(
                    claim,
                    total_vars,
                    2 * groups.width + 1,
                    &out.proof,
                    &mut v_t,
                )
                .unwrap();
                assert_eq!(res.challenges, out.challenges);
                let want = c2(&sigma, groups, &alpha, &beta_t, &res.challenges);
                assert_eq!(res.final_claim, want);
            }
        }
    }
}
