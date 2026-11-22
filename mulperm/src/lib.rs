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
use permcore::eq::{eq, eq_eval_table};
use permcore::sumcheck::{
    self, SumcheckError, SumcheckProof, SumcheckProverOutput,
};
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
    /// A PCS opening didn't verify
    /// against its commitment
    PcsVerifyFailed,
    /// The proof has the wrong number of
    /// $v_j$ or $\tilde{\sigma}$ openings
    MalformedProof,
    /// Sumcheck 1's rounds
    Sumcheck1(SumcheckError),
    /// $C_1 \ne f(\beta) \prod_j v_j$
    C1Failed,
    /// Sumcheck 2's rounds
    Sumcheck2(SumcheckError),
    /// $C_2 \ne \mathrm{eq}(\beta', \gamma) \prod_i
    /// E(\tilde{\alpha}(\langle i \rangle, j^*), V_i)$
    C2Failed,
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

/// Perm proof: PCS commitments to $f, g$, both sumchecks' messages, the
/// $v_j = \tilde{p}(\beta, \langle j \rangle)$, and PCS openings of $g$ at
/// $\alpha$, $f$ at $\beta$, and $\tilde{\sigma}$ at $(\langle i \rangle,
/// j^*, x^*)$ for each $i \in [w]$, where $\gamma = (x^*, j^*)$.
pub struct MulPermProof<F: PrimeField, P: PolynomialCommitment<F>> {
    pub f_commit: P::Commitment,
    pub g_commit: P::Commitment,
    pub g_at_alpha: F,
    pub g_opening: P::Proof,
    pub sumcheck1: SumcheckProof<F>,
    pub f_at_beta: F,
    pub f_opening: P::Proof,
    pub p_at_beta: Vec<F>,
    pub sumcheck2: SumcheckProof<F>,
    pub sigma_at_gamma: Vec<F>,
    pub sigma_openings: Vec<P::Proof>,
}

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

/// The $w$ points $(\langle i \rangle, j^*, x^*)$ where
/// $\tilde{\sigma}$ is opened after sumcheck 2, one per bit
/// $i$ of a group. Takes the grouping as `groups` and
/// sumcheck 2's challenges $\gamma = (x^*, j^*)$ as
/// `gamma`. Returns the points in order of $i$.
fn sigma_points<F: PrimeField>(groups: Groups, gamma: &[F]) -> Vec<Vec<F>> {
    let Groups { ell, width } = groups;
    let log_w = width.trailing_zeros() as usize;
    let (x_star, j_star) = gamma.split_at(ell * width);
    (0..width)
        .map(|i| {
            // `i` is bit position, we're
            // representing that in binary
            (0..log_w)
                .map(|k| F::from(((i >> k) & 1) as u64))
                .chain(j_star.iter().copied())
                .chain(x_star.iter().copied())
                .collect()
        })
        .collect()
}

/// Verify one PCS opening, turning a rejection into an error. Takes the
/// verifier key `vk`, the `commitment`, the `point`, the claimed `value`,
/// its `proof` and the Fiat–Shamir `transcript`. Returns `Ok(())` if
/// the opening verifies, [`MulPermError::PcsVerifyFailed`] otherwise.
fn check_opening<F: PrimeField, P: PolynomialCommitment<F>>(
    vk: &P::VerifierKey,
    commitment: &P::Commitment,
    point: &[F],
    value: F,
    proof: &P::Proof,
    transcript: &mut Transcript,
) -> Result<(), MulPermError<P::Error>> {
    if P::verify(vk, commitment, point, value, proof, transcript)
        .map_err(MulPermError::Pcs)?
    {
        Ok(())
    } else {
        // `Ok(false)`, the PCS verify ran
        Err(MulPermError::PcsVerifyFailed)
    }
}

/// Algorithm 3's prover. Commits to $f, g$ and absorbs every commitment
/// before squeezing $\alpha$. Opens $g$ at $\alpha$, runs sumcheck 1 to get
/// $\beta$ and the $v_j$, opens $f$ at $\beta$, then absorbs the $v_j$ before
/// squeezing $t$, so $\beta' = (\beta, t)$ depends on them. Sumcheck 2 proves
/// the $v_j$ and ends at $\gamma$, where $\tilde{\sigma}$ is opened $w$ times.
///
/// Takes the PCS prover key `pk`, the prover half of the [`index`],
/// $f$ and $g$ as `f` and `g`, and the Fiat–Shamir `transcript`.
/// Returns the proof, or an error if a PCS operation fails.
///
/// # Panics
///
/// Panics if `f` or `g` isn't over $\mu$ variables.
pub fn prove<F: PrimeField, P: PolynomialCommitment<F>>(
    pk: &P::ProverKey,
    index: &MulPermProverIndex<F, P>,
    f: &DenseMultilinearExtension<F>,
    g: &DenseMultilinearExtension<F>,
    transcript: &mut Transcript,
) -> Result<MulPermProof<F, P>, MulPermError<P::Error>> {
    // All `num_vars` are the same across polys
    let num_vars = index.perm.num_vars();
    assert_eq!(num_vars, f.num_vars, "f num_vars must match μ");
    assert_eq!(num_vars, g.num_vars, "g num_vars must match μ");
    let groups = index.groups;
    let f_commit = P::commit(pk, f.into()).map_err(MulPermError::Pcs)?;
    let g_commit = P::commit(pk, g.into()).map_err(MulPermError::Pcs)?;
    // Remember, we absorb all public messages, everything the
    // prover could potentially pick needs to be used to
    // compute a challenge so there is no freedom.
    transcript.append(b"sigma_commit", &index.sigma_commit);
    transcript.append(b"f_commit", &f_commit);
    transcript.append(b"g_commit", &g_commit);
    let alpha: Vec<F> = transcript.challenge_vec(b"alpha", num_vars);
    let (g_at_alpha, g_opening) =
        P::open(pk, g.into(), &alpha, transcript).map_err(MulPermError::Pcs)?;
    let parts = partial_products(&index.perm, groups, &alpha)?;
    let (out1, p_at_beta) = sumcheck1_prove(f, parts, transcript);
    // Mapping to what we call $\beta$
    let beta = out1.challenges;
    let (f_at_beta, f_opening) =
        P::open(pk, f.into(), &beta, transcript).map_err(MulPermError::Pcs)?;
    transcript.append_slice(b"p_at_beta", &p_at_beta);
    let log_ell = groups.ell.trailing_zeros() as usize;
    let t: Vec<F> = transcript.challenge_vec(b"t", log_ell);
    let beta_t = [beta, t].concat();
    let out2 =
        sumcheck2_prove(&index.sigma, groups, &alpha, &beta_t, transcript);
    let (sigma_at_gamma, sigma_openings) =
        sigma_points(groups, &out2.challenges)
            .iter()
            // Each point to evaluation/proof pairs that we need later
            .map(|point| P::open(pk, (&index.sigma).into(), point, transcript))
            .collect::<Result<Vec<_>, _>>()
            .map_err(MulPermError::Pcs)?
            .into_iter()
            .unzip();
    Ok(MulPermProof {
        f_commit,
        g_commit,
        g_at_alpha,
        g_opening,
        sumcheck1: out1.proof,
        f_at_beta,
        f_opening,
        p_at_beta,
        sumcheck2: out2.proof,
        sigma_at_gamma,
        sigma_openings,
    })
}

/// Algorithm 3's verifier. Checks the $g$ opening, sumcheck 1 against
/// $g(\alpha)$ at degree $\ell + 1$, the $f$ opening and $C_1$. Then
/// sumcheck 2 against $S_p = \sum_j \mathrm{eq}(t, \langle j \rangle) v_j$
/// at degree $2w + 1$, the $w$ $\tilde{\sigma}$ openings $V_i$ and $C_2$.
/// The verifier computes $\tilde{\alpha}(\langle i \rangle, j^*) =
/// \sum_j \mathrm{eq}(j^*, \langle j \rangle) \alpha_{j w + i}$
/// itself, $O(\ell)$ per $i$ and $O(\mu)$ in total.
///
/// Takes the PCS verifier key `vk`, the verifier half of the [`index`],
/// the `proof` and the Fiat–Shamir `transcript`. Returns `Ok(())` if the
/// proof verifies, or the first check that failed.
pub fn verify<F: PrimeField, P: PolynomialCommitment<F>>(
    vk: &P::VerifierKey,
    index: &MulPermVerifierIndex<F, P>,
    proof: &MulPermProof<F, P>,
    transcript: &mut Transcript,
) -> Result<(), MulPermError<P::Error>> {
    let num_vars = index.num_vars;
    let groups = index.groups;
    let Groups { ell, width } = groups;
    if proof.p_at_beta.len() != ell
        || proof.sigma_at_gamma.len() != width
        || proof.sigma_openings.len() != width
    {
        return Err(MulPermError::MalformedProof);
    }
    // Just like the prover, before squeezing $\alpha$
    transcript.append(b"sigma_commit", &index.sigma_commit);
    transcript.append(b"f_commit", &proof.f_commit);
    transcript.append(b"g_commit", &proof.g_commit);
    let alpha: Vec<F> = transcript.challenge_vec(b"alpha", num_vars);
    check_opening::<F, P>(
        vk,
        &proof.g_commit,
        &alpha,
        proof.g_at_alpha,
        &proof.g_opening,
        transcript,
    )?;
    let out1 = sumcheck::verify(
        proof.g_at_alpha,
        num_vars,
        // Because $v$s and $f$
        ell + 1,
        &proof.sumcheck1,
        transcript,
    )
    .map_err(MulPermError::Sumcheck1)?;
    let beta = out1.challenges;
    check_opening::<F, P>(
        vk,
        &proof.f_commit,
        &beta,
        proof.f_at_beta,
        &proof.f_opening,
        transcript,
    )?;
    // $C_1 = f(\beta) \prod_j v_j$, final claim 1
    if proof.f_at_beta * proof.p_at_beta.iter().product::<F>()
        != out1.final_claim
    {
        return Err(MulPermError::C1Failed);
    }
    transcript.append_slice(b"p_at_beta", &proof.p_at_beta);
    let log_ell = ell.trailing_zeros() as usize;
    let t: Vec<F> = transcript.challenge_vec(b"t", log_ell);
    // $S_p = \tilde{p}(\beta, t)$, interpolating the $v_j$ in $j$
    // The claimed sum for sumcheck 2, folds the $\ell$ claims
    let s_p: F = eq_eval_table(&t)
        .iter()
        .zip(&proof.p_at_beta)
        .map(|(e, v)| *e * v)
        .sum();
    // $\mu + \log \ell$ entries
    let beta_t = [beta, t].concat();
    let out2 = sumcheck::verify(
        s_p,
        num_vars + log_ell,
        2 * width + 1,
        &proof.sumcheck2,
        transcript,
    )
    .map_err(MulPermError::Sumcheck2)?;
    // $\gamma$ is the random point Sumcheck 2 ends at. Remember
    // we narrow at each round, so  one challeng per round,
    // where at the end we get the entirety of $\gamma$.
    let gamma = out2.challenges;
    // Nested pairs, because we zip twice
    for ((point, &value), opening) in sigma_points(groups, &gamma)
        .iter()
        .zip(&proof.sigma_at_gamma)
        .zip(&proof.sigma_openings)
    {
        check_opening::<F, P>(
            vk,
            &index.sigma_commit,
            point,
            value,
            opening,
            transcript,
        )?;
    }
    let eq_j = eq_eval_table(&gamma[num_vars..]);
    let prod: F = proof
        .sigma_at_gamma
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let a: F = (0..ell).map(|j| eq_j[j] * alpha[j * width + i]).sum();
            // $E(a, s)$ is the one-variable eq
            eq(&[a], &[v])
        })
        .product();
    // $C_2 = \mathrm{eq}(\beta', \gamma) \prod_i
    // E(\tilde{\alpha}(\langle i \rangle, j^*), V_i)$
    // This is the second, and final sumcheck claim.
    if eq(&beta_t, &gamma) * prod != out2.final_claim {
        return Err(MulPermError::C2Failed);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_ff::{Field, UniformRand};
    use ark_std::test_rng;
    use core::convert::Infallible;
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

    /// Prove $(f, g)$ against `perm`'s index at `ell` on
    /// MockPcs, apply `tamper` to the proof, then verify against
    /// `v_perm`'s index. Returns the verifier's result. This
    /// helper provides flexibility for different test cases.
    fn prove_then_verify(
        perm: &Permutation,
        // Swaps to test tampering
        v_perm: &Permutation,
        ell: usize,
        f: &DenseMultilinearExtension<Fr>,
        g: &DenseMultilinearExtension<Fr>,
        tamper: impl FnOnce(&mut MulPermProof<Fr, MockPcs<Fr>>),
    ) -> Result<(), MulPermError<Infallible>> {
        let (p_idx, _) = index_with::<Fr, MockPcs<Fr>>(&(), perm, ell).unwrap();
        let (_, v_idx) =
            index_with::<Fr, MockPcs<Fr>>(&(), v_perm, ell).unwrap();
        let mut p_t = Transcript::new(b"mulperm");
        let mut proof = prove(&(), &p_idx, f, g, &mut p_t).unwrap();
        // Can also be a noop,
        // for non-tampered calls
        tamper(&mut proof);
        let mut v_t = Transcript::new(b"mulperm");
        verify(&(), &v_idx, &proof, &mut v_t)
    }

    /// [`prove_then_verify`] on a consistent pair, with the same
    /// `perm` on both sides, so only `tamper` can make it fail.
    fn consistent_round_trip(
        perm: &Permutation,
        ell: usize,
        tamper: impl FnOnce(&mut MulPermProof<Fr, MockPcs<Fr>>),
    ) -> Result<(), MulPermError<Infallible>> {
        let (f, g) = consistent_pair(perm, &mut test_rng());
        prove_then_verify(perm, perm, ell, &f, &g, tamper)
    }

    #[test]
    fn honest_round_trip() {
        for (perm, ells) in [
            // Turning both into slices
            // gives the same type `&[usize]`.
            (sample_perm(), &[1, 2, 4][..]),
            (shift_perm(), &[1, 2, 4, 8][..]),
        ] {
            for &ell in ells {
                assert_eq!(consistent_round_trip(&perm, ell, |_| {}), Ok(()));
            }
        }
    }

    #[test]
    fn rejects_inconsistent_pair() {
        let mut rng = test_rng();
        let perm = sample_perm();
        let (f, _g_good) = consistent_pair(&perm, &mut rng);
        // Random $g$ unrelated to $f$ and $\sigma$.
        let g_bad = DenseMultilinearExtension::from_evaluations_vec(
            perm.num_vars(),
            (0..perm.size()).map(|_| Fr::rand(&mut rng)).collect(),
        );
        // The honest round 0 sums to the true $\sum_x f(x) \cdot
        // eq(\alpha, \sigma(x))$, which isn't `g_bad`'s $g(\alpha)$.
        assert_eq!(
            prove_then_verify(&perm, &perm, 2, &f, &g_bad, |_| {}),
            Err(MulPermError::Sumcheck1(SumcheckError::RoundCheckFailed {
                round: 0
            }))
        );
    }

    #[test]
    // Tamper (minimal) a single round polynomial of each sumcheck w/
    // a single field element change, for a single case of $\ell = 2$.
    fn rejects_tampered_sumcheck() {
        let perm = sample_perm();
        assert_eq!(
            consistent_round_trip(&perm, 2, |p| {
                p.sumcheck1.round_polys[1][0] += Fr::from(1)
            }),
            Err(MulPermError::Sumcheck1(SumcheckError::RoundCheckFailed {
                round: 1
            }))
        );
        assert_eq!(
            consistent_round_trip(&perm, 2, |p| {
                p.sumcheck2.round_polys[1][0] += Fr::from(1)
            }),
            Err(MulPermError::Sumcheck2(SumcheckError::RoundCheckFailed {
                round: 1
            }))
        );
    }

    #[test]
    fn rejects_mismatched_index() {
        // $\tilde{\sigma}$'s commitment is absorbed into the transcript, so
        // a verifier holding a different $\sigma$'s index derives a
        // different $\alpha$ and the $g$-opening check fails immediately.
        let mut rng = test_rng();
        let perm = sample_perm();
        let (f, g) = consistent_pair(&perm, &mut rng);
        let other_perm = Permutation::identity(perm.num_vars());
        assert_eq!(
            prove_then_verify(&perm, &other_perm, 2, &f, &g, |_| {}),
            Err(MulPermError::PcsVerifyFailed)
        );
    }

    #[test]
    // A forged $\tilde{\sigma}$ value must fail against the
    // preprocessed commitment, not be taken on faith.
    fn rejects_tampered_sigma_opening() {
        assert_eq!(
            consistent_round_trip(&sample_perm(), 2, |p| {
                p.sigma_at_gamma[0] += Fr::from(1)
            }),
            Err(MulPermError::PcsVerifyFailed)
        );
    }

    #[test]
    fn rejects_forged_p_at_beta() {
        let perm = sample_perm();
        // Changing the product of the $v_j$ breaks $C_1$
        assert_eq!(
            consistent_round_trip(&perm, 2, |p| p.p_at_beta[0] += Fr::from(1)),
            Err(MulPermError::C1Failed)
        );
        // $(v_0 c, v_1 / c)$ keeps the product, so $C_1$ passes, but
        // $t$ and $S_p$ change, so sumcheck 2 fails at round 0.
        let c = Fr::from(3);
        assert_eq!(
            consistent_round_trip(&perm, 2, |p| {
                p.p_at_beta[0] *= c;
                p.p_at_beta[1] *= c.inverse().unwrap();
            }),
            Err(MulPermError::Sumcheck2(SumcheckError::RoundCheckFailed {
                round: 0
            }))
        );
    }

    #[test]
    fn rejects_malformed_proof() {
        let perm = sample_perm();
        assert_eq!(
            consistent_round_trip(&perm, 2, |p| {
                p.p_at_beta.pop();
            }),
            Err(MulPermError::MalformedProof)
        );
        assert_eq!(
            consistent_round_trip(&perm, 2, |p| {
                p.sigma_openings.pop();
            }),
            Err(MulPermError::MalformedProof)
        );
    }

    #[test]
    // A prover that runs sumcheck 2 on a different polynomial with the
    // same hypercube sum passes every round and every opening, so only
    // $C_2$ can catch it. $\ell = \mu$ makes $w = 1$, so the summand
    // $\mathrm{eq} \cdot E(\tilde{\alpha}_0, \tilde{\sigma}_0)$
    // has a single $E$, linear in $\tilde{\sigma}_0$.
    fn rejects_cheating_sumcheck2() {
        let mut rng = test_rng();
        let perm = sample_perm();
        let (f, g) = consistent_pair(&perm, &mut rng);
        let num_vars = perm.num_vars();
        let ell = num_vars;
        let (p_idx, v_idx) =
            index_with::<Fr, MockPcs<Fr>>(&(), &perm, ell).unwrap();
        let sigma = &p_idx.sigma;
        let one = Fr::from(1);
        // An honest proof with sumcheck 2 redone on a forged $\tilde{\sigma}'$
        // with the same $S_p$, then the real $\tilde{\sigma}$ opened at its
        // $\gamma$. The two differ at $\gamma$ when $c \ne 0$.
        let forge = |c: Fr| {
            let mut proof =
                prove(&(), &p_idx, &f, &g, &mut Transcript::new(b"mulperm"))
                    .unwrap();
            // Replay `prove`'s transcript up to sumcheck 2
            let mut tr = Transcript::new(b"mulperm");
            tr.append(b"sigma_commit", &p_idx.sigma_commit);
            tr.append(b"f_commit", &proof.f_commit);
            tr.append(b"g_commit", &proof.g_commit);
            let alpha: Vec<Fr> = tr.challenge_vec(b"alpha", num_vars);
            MockPcs::<Fr>::open(&(), (&g).into(), &alpha, &mut tr).unwrap();
            let parts = partial_products(&perm, p_idx.groups, &alpha).unwrap();
            let beta = sumcheck1_prove(&f, parts, &mut tr).0.challenges;
            MockPcs::<Fr>::open(&(), (&f).into(), &beta, &mut tr).unwrap();
            tr.append_slice(b"p_at_beta", &proof.p_at_beta);
            let log_ell = ell.trailing_zeros() as usize;
            let t: Vec<Fr> = tr.challenge_vec(b"t", log_ell);
            let beta_t = [beta, t].concat();
            // $E(a, s) = (1 - a) + s (2a - 1)$, so shifting
            // $\tilde{\sigma}_0$ by $c\, w_1$ at $(x, j) = (0, 0)$
            // and $-c\, w_0$ at $(1, 0)$, $w_x = \mathrm{eq}(\beta',
            // (x, 0)) (2 \alpha_0 - 1)$, changes $S_p$ by
            // $c\, w_1 w_0 - c\, w_0 w_1 = 0$.
            let eq_bt = eq_eval_table(&beta_t);
            let w = |x: usize| eq_bt[x] * (alpha[0] + alpha[0] - one);
            // On a forged sigma, almost the same. Using `w` this allows us to
            // keep the sum the same but $\sigma$ and $\sigma'$ still differ.
            let mut forged = sigma.clone();
            forged.evaluations[0] += c * w(1);
            forged.evaluations[num_vars] -= c * w(0);
            let out = sumcheck2_prove(
                &forged,
                p_idx.groups,
                &alpha,
                &beta_t,
                &mut tr,
            );
            let (values, openings): (Vec<Fr>, Vec<()>) =
                sigma_points(p_idx.groups, &out.challenges)
                    .iter()
                    .map(|pt| {
                        MockPcs::<Fr>::open(&(), sigma.into(), pt, &mut tr)
                            .unwrap()
                    })
                    .unzip();
            proof.sumcheck2 = out.proof;
            proof.sigma_at_gamma = values;
            proof.sigma_openings = openings;
            proof
        };
        let check = |proof| {
            verify(&(), &v_idx, &proof, &mut Transcript::new(b"mulperm"))
        };
        // At $c = 0$ the forgery is honest, and replay ok
        assert_eq!(check(forge(Fr::from(0))), Ok(()));
        assert_eq!(check(forge(one)), Err(MulPermError::C2Failed));
    }
}
