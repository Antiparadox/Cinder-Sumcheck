//! BLS12-381 scalar field wrapper and utilities

use ark_bls12_381::Fr;
use ark_ff::{Field as ArkField, UniformRand};
use ark_std::rand::RngCore;
use std::ops::{Add, Sub, Mul, Neg, AddAssign, SubAssign, MulAssign};

/// Type alias for the BLS12-381 scalar field
pub type Scalar = Fr;

/// Extension trait for field operations needed by Cinder
pub trait CinderField: 
    Sized 
    + Clone 
    + Copy
    + PartialEq
    + std::fmt::Debug
    + Add<Output = Self> 
    + Sub<Output = Self> 
    + Mul<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + Send
    + Sync
{
    fn zero() -> Self;
    fn one() -> Self;
    fn from_u64(val: u64) -> Self;
    fn random<R: RngCore>(rng: &mut R) -> Self;
    fn inverse(&self) -> Option<Self>;
    fn square(&self) -> Self;
    fn double(&self) -> Self;
}

impl CinderField for Scalar {
    #[inline]
    fn zero() -> Self {
        <Fr as ark_ff::Zero>::zero()
    }

    #[inline]
    fn one() -> Self {
        <Fr as ark_ff::One>::one()
    }

    #[inline]
    fn from_u64(val: u64) -> Self {
        Fr::from(val)
    }

    #[inline]
    fn random<R: RngCore>(rng: &mut R) -> Self {
        Fr::rand(rng)
    }

    #[inline]
    fn inverse(&self) -> Option<Self> {
        ArkField::inverse(self)
    }

    #[inline]
    fn square(&self) -> Self {
        ArkField::square(self)
    }

    #[inline]
    fn double(&self) -> Self {
        ArkField::double(self)
    }
}

/// Utility to convert a usize index to field element
#[inline]
pub fn index_to_field(idx: usize) -> Scalar {
    Scalar::from_u64(idx as u64)
}

/// Utility to extract the t-th bit of a number
#[inline]
pub fn get_bit(value: usize, bit_position: usize) -> bool {
    (value >> bit_position) & 1 == 1
}

/// Utility to convert a bit to a field element (0 or 1)
#[inline]
pub fn bit_to_field(bit: bool) -> Scalar {
    if bit {
        <Scalar as CinderField>::one()
    } else {
        <Scalar as CinderField>::zero()
    }
}

#[cfg(test)]
mod field_tests {
    use super::*;
    use ark_std::test_rng;

    #[test]
    fn test_field_basics() {
        let zero = Scalar::zero();
        let one = Scalar::one();
        let two = one + one;
        
        assert_eq!(zero + one, one);
        assert_eq!(one + one, two);
        assert_eq!(one * one, one);
        assert_eq!(zero * one, zero);
    }

    #[test]
    fn test_field_inverse() {
        let mut rng = test_rng();
        let a = Scalar::random(&mut rng);
        let a_inv = CinderField::inverse(&a).unwrap();
        
        assert_eq!(a * a_inv, <Scalar as CinderField>::one());
    }

    #[test]
    fn test_bit_extraction() {
        assert!(!get_bit(0b1010, 0));
        assert!(get_bit(0b1010, 1));
        assert!(!get_bit(0b1010, 2));
        assert!(get_bit(0b1010, 3));
    }
}

