//! The equality polynomial used in the protocols.
//!
//! Definition 2 (Equality Polynomial).
//!
//! $$eq(X, Y) = \prod_{i=1}^{\mu} (X_i \cdot Y_i + (1 - X_i) \cdot (1 - Y_i))$$
//!
//! For boolean inputs $eq(x, y)$ is $1$ iff $x == y$. Over the
//! field, $eq$ is the multilinear extension of that indicator.

use alloc::vec;
use alloc::vec::Vec;
use ark_ff::Field;

/// Evaluate $eq(x, y)$ at a single pair of points.
/// Both slices must have the same length $\mu$.
pub fn eq<F: Field>(x: &[F], y: &[F]) -> F {
    assert_eq!(
        x.len(),
        y.len(),
        "eq: argument lengths must match (got {} and {})",
        x.len(),
        y.len()
    );
    let mut acc = F::one();
    for (xi, yi) in x.iter().zip(y) {
        // Single iteration in defintion 2
        acc *= *xi * yi + (F::one() - xi) * (F::one() - yi);
    }
    acc
}

/// Build the evaluation table of $eq(\cdot, \alpha)$ over the
/// boolean hypercube $B_{\mu}$ in $O(2^{\mu})$ field operations.
/// Uses DP to get $O(2^{\mu})$, otherwise $O(\mu \cdot 2^{\mu})$.
///
/// The layout matches ark-poly's `DenseMultilinearExtension` little-endian
/// convention: entry `idx` corresponds to the boolean assignment whose `i`-th
/// bit is `(idx >> i) & 1`, so variable `0` is the least-significant bit.
pub fn eq_eval_table<F: Field>(alpha: &[F]) -> Vec<F> {
    let mu = alpha.len();
    let size = 1usize << mu;
    let mut table = vec![F::zero(); size];
    // Base case
    table[0] = F::one();
    for (k, a) in alpha.iter().enumerate() {
        // Not free so we want to save 2^k of them per
        let one_minus_a = F::one() - a;
        for x in (0..(1usize << k)).rev() {
            let v = table[x];
            // 2 values from one, means 2^{\mu}
            table[x | (1 << k)] = v * a;
            table[x] = v * one_minus_a;
        }
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_ff::{One, UniformRand, Zero};
    use ark_std::test_rng;

    // Transform an integer to it's bit-representation
    fn boolean_assignment<F: Field>(idx: usize, num_vars: usize) -> Vec<F> {
        (0..num_vars)
            .map(|i| {
                if (idx >> i) & 1 == 1 {
                    F::one()
                } else {
                    F::zero()
                }
            })
            .collect()
    }

    #[test]
    fn eq_indicator_on_booleans() {
        let mu = 4;
        for i in 0..(1usize << mu) {
            for j in 0..(1usize << mu) {
                let x: Vec<Fr> = boolean_assignment(i, mu);
                let y: Vec<Fr> = boolean_assignment(j, mu);
                let expected = if i == j { Fr::one() } else { Fr::zero() };
                assert_eq!(eq(&x, &y), expected);
            }
        }
    }

    #[test]
    fn eq_eval_table_matches_pointwise_at_random_alpha() {
        let mut rng = test_rng();
        let mu = 5;
        let alpha: Vec<Fr> = (0..mu).map(|_| Fr::rand(&mut rng)).collect();
        let table = eq_eval_table(&alpha);
        for (x, &cell) in table.iter().enumerate() {
            let xs: Vec<Fr> = boolean_assignment(x, mu);
            assert_eq!(cell, eq(&xs, &alpha), "mismatch at x = {x:b}");
        }
    }

    #[test]
    fn eq_eval_table_zero_vars_is_one() {
        let table = eq_eval_table::<Fr>(&[]);
        assert_eq!(table, vec![Fr::one()]);
    }

    #[test]
    // There's a single value that matches so any number of zeros + 1 = 1
    // on $B_\mu$. This extends to $F_\mu$ for the multilinear extension,
    // since it's uniquely determined by it's values on $B_\mu$.
    fn eq_eval_table_sums_to_one() {
        let mut rng = test_rng();
        for mu in [1, 4, 7] {
            // Random field elements, $\mu$ variables
            let alpha: Vec<Fr> = (0..mu).map(|_| Fr::rand(&mut rng)).collect();
            let table = eq_eval_table(&alpha);
            let sum: Fr = table.iter().copied().sum();
            assert_eq!(sum, Fr::one(), "mismatch at mu = {mu}");
        }
    }
}
