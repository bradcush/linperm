//! Permutations $\sigma : B_\mu \rightarrow B_\mu$ on the boolean hypercube.
//!
//! Both BiPerm and MulPerm work with a permutation $\sigma$
//! represented as the image of every boolean point.

use alloc::vec::Vec;

use crate::error::CoreError;

/// A permutation $\sigma : B_\mu → B_\mu$,
/// stored as $image\[x\] = \sigma(x)$.
///
/// * [`Permutation::new`] - validating constructor.
/// * [`Permutation::bit`] - the $\sigma_i(x)$ bit-decomposition
/// * [`Permutation::halves`] - the ($\sigma_L$, $\sigma_R$) split for BiPerm
/// * [`Permutation::group`] - the $\mu/\ell$-bit group decomposition for MulPerm
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Permutation {
    image: Vec<usize>,
    // Just $\mu$, the bit-width
    num_vars: usize,
}

impl Permutation {
    /// Build a permutation from its image vector.
    ///
    /// `image.len()` must be a power of two (since $\mu$ bits, $2^\mu$
    /// elements) and `image` must be a bijection of `[0, image.len())`.
    pub fn new(image: Vec<usize>) -> Result<Self, CoreError> {
        let len = image.len();
        if !len.is_power_of_two() {
            return Err(CoreError::NotPowerOfTwo { len });
        }
        // Compute once on init, set, then trust
        let num_vars = len.trailing_zeros() as usize;
        let mut seen = alloc::vec![false; len];
        for (index, &value) in image.iter().enumerate() {
            // Note that we just want mapped indices and a
            // true permutation shouldn't have repeats
            if value >= len || seen[value] {
                return Err(CoreError::NotAPermutation { len, index });
            }
            seen[value] = true;
        }
        Ok(Self { image, num_vars })
    }

    /// The identity permutation on $2^{num_vars}$ elements.
    /// From `[0, image.len())` $\rightarrow$ `[0, image.len())`.
    pub fn identity(num_vars: usize) -> Self {
        let len = 1usize << num_vars;
        Self {
            image: (0..len).collect(),
            num_vars,
        }
    }

    /// $\mu$, the number of boolean variables.
    /// The hypercube has $n = 2^\mu$ points.
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// $n = 2^\mu$, the number of permuted elements.
    pub fn size(&self) -> usize {
        self.image.len()
    }

    /// $\sigma(x)$ as an integer in $[0, n)$.
    ///
    /// `x` is interpreted in little-endian bit order (variable `0` is
    /// the LSB), matching the ark-poly multilinear-extension convention.
    pub fn apply(&self, x: usize) -> usize {
        self.image[x]
    }

    /// $\sigma_i(x)$, the $i$-th bit of $\sigma(x)$, with $i \in [0, \mu)$.
    /// Whether the $i$-th bit of the permutaiton is 0 (false) or 1 (true).
    pub fn bit(&self, i: usize, x: usize) -> bool {
        debug_assert!(i < self.num_vars);
        (self.image[x] >> i) & 1 == 1
    }

    /// Build the dense evaluation table for $\sigma_i$ over $B_\mu$, suitable
    /// for passing to `DenseMultilinearExtension::from_evaluations_vec`.
    /// $\sigma_i$ over $B_\mu$ is all $i$-th bits as field elements.
    pub fn bit_evaluations<F: ark_ff::Field>(&self, i: usize) -> Vec<F> {
        debug_assert!(i < self.num_vars);
        self.image
            .iter()
            .map(|&y| {
                if (y >> i) & 1 == 1 {
                    F::one()
                } else {
                    F::zero()
                }
            })
            .collect()
    }

    /// ($\sigma_L(x)$, $\sigma_R(x)$), the first $\mu/2$ and
    /// last $\mu/2$ bits of $\sigma(x)$, used by BiPerm.
    ///
    /// Returns an error if $\mu$ is odd.
    pub fn halves(&self, x: usize) -> Result<(usize, usize), CoreError> {
        if self.num_vars % 2 != 0 {
            return Err(CoreError::OddNumVars {
                num_vars: self.num_vars,
            });
        }
        let half = self.num_vars / 2;
        // The bits we want to keep
        let mask = (1usize << half) - 1;
        let y = self.image[x];
        Ok((y & mask, (y >> half) & mask))
    }

    /// $\sigma$ projected onto group $j \in [0, \ell)$ of
    /// width $\mu/\ell$, used by MulPerm. The output has
    /// $\mu/\ell$ bits packed in little-endian order.
    ///
    /// $\ell$ must divide $\mu$.
    pub fn group(
        &self,
        ell: usize,
        // Index of the group
        j: usize,
        x: usize,
    ) -> Result<usize, CoreError> {
        if ell == 0 || self.num_vars % ell != 0 {
            // Slightly confusing usage for this case
            return Err(CoreError::NumVarsMismatch {
                expected: ell,
                got: self.num_vars,
            });
        }
        let width = self.num_vars / ell;
        let mask = (1usize << width) - 1;
        Ok((self.image[x] >> (j * width)) & mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_power_of_two() {
        assert!(matches!(
            Permutation::new(alloc::vec![0, 1, 2]),
            Err(CoreError::NotPowerOfTwo { len: 3 })
        ));
    }

    #[test]
    fn rejects_duplicate_image() {
        assert!(matches!(
            Permutation::new(alloc::vec![0, 0, 2, 3]),
            Err(CoreError::NotAPermutation { .. })
        ));
    }

    #[test]
    fn identity_is_well_formed() {
        let id = Permutation::identity(3);
        assert_eq!(id.num_vars(), 3);
        assert_eq!(id.size(), 8);
        for x in 0..8 {
            assert_eq!(id.apply(x), x);
        }
    }

    #[test]
    // Round-trip test makes for a robust check what every bit decomposition
    // is what we expect. We're checking across a few different numbers.
    fn bit_decomposition_round_trips() {
        let perm = Permutation::new(alloc::vec![3, 1, 2, 0]).unwrap();
        for x in 0..4 {
            let mut reconstructed = 0usize;
            for i in 0..perm.num_vars() {
                if perm.bit(i, x) {
                    reconstructed |= 1 << i;
                }
            }
            assert_eq!(reconstructed, perm.apply(x));
        }
    }

    #[test]
    fn halves_rejects_odd_num_vars() {
        let perm =
            Permutation::new(alloc::vec![5, 3, 7, 1, 0, 6, 2, 4]).unwrap();
        assert_eq!(perm.num_vars(), 3);
        assert!(matches!(
            perm.halves(0),
            Err(CoreError::OddNumVars { num_vars: 3 })
        ));
    }

    #[test]
    fn halves_recombine_to_image() {
        let perm = Permutation::new(alloc::vec![
            5, 3, 7, 1, 0, 6, 2, 4, 9, 11, 8, 10, 13, 15, 12, 14
        ])
        .unwrap();
        for x in 0..perm.size() {
            let (lo, hi) = perm.halves(x).unwrap();
            assert_eq!(lo | (hi << (perm.num_vars() / 2)), perm.apply(x));
        }
    }

    #[test]
    fn group_rejects_zero_ell() {
        let perm = Permutation::identity(4);
        assert!(matches!(
            perm.group(0, 0, 0),
            Err(CoreError::NumVarsMismatch { .. })
        ));
    }

    #[test]
    fn group_rejects_indivisible_ell() {
        let perm = Permutation::identity(4);
        assert!(matches!(
            perm.group(3, 0, 0),
            Err(CoreError::NumVarsMismatch {
                expected: 3,
                got: 4
            })
        ));
    }

    #[test]
    fn group_recombines_to_image() {
        let perm = Permutation::new(alloc::vec![
            5, 3, 7, 1, 0, 6, 2, 4, 9, 11, 8, 10, 13, 15, 12, 14
        ])
        .unwrap();
        let ell = 2;
        let width = perm.num_vars() / ell;
        for x in 0..perm.size() {
            let mut reconstructed = 0usize;
            for j in 0..ell {
                reconstructed |= perm.group(ell, j, x).unwrap() << (j * width);
            }
            assert_eq!(reconstructed, perm.apply(x));
        }
    }
}
