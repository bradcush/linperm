//! Shared building blocks for the BiPerm and MulPerm permutation
//! arguments described in *Linear\*-Time Permutation Check*.
//!
//! This crate is intentionally small. It provides the primitives both
//! protocols build on (the equality polynomial, permutation representations,
//! transcripts, and a polynomial-commitment trait) without committing to a
//! particular sumcheck or PCS implementation.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod eq;
pub mod error;
pub mod pcs;
pub mod permutation;
pub mod transcript;

pub use error::CoreError;
pub use pcs::{MockPcs, PolynomialCommitment};
pub use permutation::Permutation;
pub use transcript::Transcript;

pub use ark_ff;
pub use ark_poly;
pub use ark_serialize;
pub use ark_std;
