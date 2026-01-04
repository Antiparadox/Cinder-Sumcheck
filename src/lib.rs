//! Cinder: A simple dense-to-sparse MLE compiler
//!
//! This crate implements the sumcheck component of the Cinder protocol
//! for sparse polynomial commitment schemes over the BLS12-381 scalar field.

pub mod field;
pub mod sparse_matrix;
pub mod cinder_mle;
pub mod eq_polynomial;
pub mod sumcheck;

pub use field::*;
pub use sparse_matrix::*;
pub use cinder_mle::*;
pub use eq_polynomial::*;
pub use sumcheck::*;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod bench_single;

