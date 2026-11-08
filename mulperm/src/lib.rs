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
use ark_poly::DenseMultilinearExtension;
use permcore::{Permutation, PolynomialCommitment};

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
    /// PCS operation produced an error
    Pcs(E),
}

impl<E> From<ParamsError> for MulPermError<E> {
    fn from(e: ParamsError) -> Self {
        Self::Params(e)
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

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_ff::UniformRand;
    use ark_poly::Polynomial;
    use ark_std::test_rng;
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
    fn index_rejects_non_power_of_two_mu() {
        let perm = Permutation::identity(3);
        assert!(matches!(
            index::<Fr, MockPcs<Fr>>(&(), &perm),
            Err(MulPermError::Params(ParamsError::NumVarsNotPowerOfTwo {
                num_vars: 3
            }))
        ));
    }
}
