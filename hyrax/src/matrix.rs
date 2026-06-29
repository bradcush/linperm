//! Matrix reshape and tensor helpers for the Hyrax commitment.
//!
//! A $\nu$-variate multilinear polynomial is committed by reshaping its
//! $2^\nu$-entry evaluation vector into a $2^{\nu_r} \times 2^{\nu_c}$ matrix
//! $M$ and committing each row. Evaluation at a point $z$ factorizes through
//! the tensor structure of the equality polynomial:
//!
//! $$f(z) = L^\top M R, \quad L = \mathrm{eq}(\cdot, z_{\mathrm{row}}), \quad R = \mathrm{eq}(\cdot, z_{\mathrm{col}})$$
//!
//! # Layout convention
//!
//! Evaluation index $k$ has variable $i$ in bit $i$ (little-endian, matching
//! ark-poly). The low $\nu_c$ bits are the **column** index, the high $\nu_r$
//! bits the **row** index, so $M[\mathrm{row}][\mathrm{col}] = \mathrm{evals}[\mathrm{row} \cdot 2^{\nu_c} + \mathrm{col}]$.
//! Correspondingly $z$'s first $\nu_c$ coordinates are the column variables
//! and its last $\nu_r$ are the row variables.

use alloc::vec;
use alloc::vec::Vec;

use ark_ff::Field;
use permcore::eq::eq_eval_table;

/// Split $\nu$ variables into (row, column) counts. Balanced, with the extra
/// variable (odd $\nu$) going to the columns: $\nu_r = \lfloor \nu/2 \rfloor$,
/// $\nu_c = \lceil \nu/2 \rceil$. Columns drive the proof and generator-set
/// size; rows drive the commitment size.
pub fn split_vars(num_vars: usize) -> (usize, usize) {
    let n_row = num_vars / 2;
    let n_col = num_vars - n_row;
    // Number of variables still so
    // actual number is $2^{n_*}$
    (n_row, n_col)
}

/// The row tensor $L = \mathrm{eq}(\cdot, z_{\mathrm{row}})$,
/// length $2^{\nu_r}$, built from the high (row) coordinates of `point`.
pub fn row_tensor<F: Field>(point: &[F], n_col: usize) -> Vec<F> {
    eq_eval_table(&point[n_col..])
}

/// The column tensor $R = \mathrm{eq}(\cdot, z_{\mathrm{col}})$, length
/// $2^{\nu_c}$, built from the low (column) coordinates of `point`.
pub fn col_tensor<F: Field>(point: &[F], n_col: usize) -> Vec<F> {
    eq_eval_table(&point[..n_col])
}

/// Compute the combined row $w = L^\top M$ (length $2^{\nu_c}$), where
/// `evals` is the row-major flattened matrix and `l` is the row tensor.
///
/// This is the dense form of the prover's central computation; the sparse
/// backend specializes the inner loop to skip zero entries of $M$.
pub fn lt_times_m_dense<F: Field>(
    evals: &[F],
    l: &[F],
    n_col: usize,
) -> Vec<F> {
    // n_col coordinates for cols so total
    // number of columns is 2^{n_col}
    let cols = 1usize << n_col;
    let mut w = vec![F::zero(); cols];
    for (row, &l_row) in l.iter().enumerate() {
        let base = row * cols;
        for (col, w_col) in w.iter_mut().enumerate() {
            *w_col += l_row * evals[base + col];
        }
    }
    w
}

/// Sparse form of [`lt_times_m_dense`]: accumulate $w = L^\top M$ over the
/// nonzero `(index, value)` entries. Index `k` splits as `row = k >> n_col`
/// and `col = k & (2^{n_col} - 1)` (the layout above), so the result is
/// identical to densifying `M` then [`lt_times_m_dense`], `O(nnz)` work.
pub fn lt_times_m_sparse<F: Field>(
    nonzeros: impl Iterator<Item = (usize, F)>,
    l: &[F],
    n_col: usize,
) -> Vec<F> {
    let cols = 1usize << n_col;
    let mask = cols - 1;
    let mut w = vec![F::zero(); cols];
    for (k, v) in nonzeros {
        w[k & mask] += l[k >> n_col] * v;
    }
    w
}

/// DP $\langle w, R \rangle$, the final evaluation value.
pub fn dot<F: Field>(w: &[F], r: &[F]) -> F {
    w.iter().zip(r).map(|(&wi, &ri)| wi * ri).sum()
}
