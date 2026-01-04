//! Equality polynomial implementation
//!
//! The equality polynomial eq(x, r) evaluates to 1 iff x == r on the boolean hypercube,
//! and interpolates smoothly between hypercube points.

use crate::field::{Scalar, CinderField};

/// Helper functions to avoid ambiguity with ark_ff traits
#[inline]
fn one() -> Scalar {
    <Scalar as CinderField>::one()
}

#[inline]
fn zero() -> Scalar {
    <Scalar as CinderField>::zero()
}

/// Equality polynomial utilities
pub struct EqPolynomial;

impl EqPolynomial {
    /// Evaluate eq(x, y) for single variable
    /// eq(x, y) = x·y + (1-x)·(1-y)
    #[inline]
    pub fn eq_single(x: Scalar, y: Scalar) -> Scalar {
        x * y + (one() - x) * (one() - y)
    }

    /// Evaluate eq(x, r) for vectors x and r
    /// eq(x, r) = Π_i eq(x_i, r_i)
    pub fn eq_vec(x: &[Scalar], r: &[Scalar]) -> Scalar {
        assert_eq!(x.len(), r.len(), "Vector length mismatch");
        
        let mut result = one();
        for (xi, ri) in x.iter().zip(r.iter()) {
            result *= Self::eq_single(*xi, *ri);
        }
        result
    }

    /// Compute all evaluations of eq(·, r) over the hypercube {0,1}^n
    /// 
    /// Returns a vector of size 2^n where result[i] = eq(bits(i), r)
    pub fn evals(r: &[Scalar]) -> Vec<Scalar> {
        let n = r.len();
        if n == 0 {
            return vec![one()];
        }
        
        let size = 1 << n;
        let mut evals = vec![one(); size];
        
        // Build eq evaluations iteratively
        // At step i, we have eq evaluations for (r_0, ..., r_{i-1})
        // and we extend them to include r_i
        let mut cur_size = 1;
        for ri in r.iter() {
            for j in (0..cur_size).rev() {
                // evals[2j] = evals[j] * (1 - r_i)  (for x_i = 0)
                // evals[2j+1] = evals[j] * r_i      (for x_i = 1)
                let prev = evals[j];
                evals[2 * j] = prev * (one() - *ri);
                evals[2 * j + 1] = prev * *ri;
            }
            cur_size *= 2;
        }
        
        evals
    }

    /// Compute eq evaluations with a scaling factor
    /// 
    /// Returns result[i] = scale * eq(bits(i), r)
    pub fn scaled_evals(r: &[Scalar], scale: Scalar) -> Vec<Scalar> {
        let mut evals = Self::evals(r);
        for e in evals.iter_mut() {
            *e *= scale;
        }
        evals
    }

    /// Evaluate eq at a specific index without computing all evaluations
    /// 
    /// eq(bits(index), r) using O(n) time
    /// 
    /// Note: The bit ordering matches the evals() function, where:
    /// - bit 0 of index corresponds to r[n-1]
    /// - bit (n-1) of index corresponds to r[0]
    pub fn eval_at_index(r: &[Scalar], index: usize) -> Scalar {
        let n = r.len();
        assert!(index < (1 << n), "Index out of bounds");
        
        let mut result = one();
        for (i, ri) in r.iter().enumerate() {
            // The evals() function builds indices such that bit (n-1-i) of the index
            // corresponds to the i-th element of r
            let bit_position = n - 1 - i;
            let bit = ((index >> bit_position) & 1) == 1;
            if bit {
                result *= *ri;
            } else {
                result *= one() - *ri;
            }
        }
        result
    }
}

/// Extended eq polynomial for Cinder
/// 
/// E_x(Z) = eq_{r_x}(row_0(Z), ..., row_{s-1}(Z))
/// 
/// This is a composition of the eq polynomial with the row/col bit MLEs.
pub struct CinderEqPoly;

impl CinderEqPoly {
    /// Evaluate E_x(k) = eq(r_x, bits(row_k))
    /// 
    /// This is the eq polynomial evaluated at the bits of the row index of entry k.
    pub fn eval_at_entry(r_x: &[Scalar], row_bits: &[Scalar]) -> Scalar {
        EqPolynomial::eq_vec(row_bits, r_x)
    }
}

#[cfg(test)]
mod eq_polynomial_tests {
    use super::*;
    use ark_std::test_rng;

    #[test]
    fn test_eq_single() {
        let z = zero();
        let o = one();
        
        // eq(0, 0) = 1
        assert_eq!(EqPolynomial::eq_single(z, z), o);
        // eq(1, 1) = 1
        assert_eq!(EqPolynomial::eq_single(o, o), o);
        // eq(0, 1) = 0
        assert_eq!(EqPolynomial::eq_single(z, o), z);
        // eq(1, 0) = 0
        assert_eq!(EqPolynomial::eq_single(o, z), z);
    }

    #[test]
    fn test_eq_vec() {
        let z = zero();
        let o = one();
        
        // eq([0,0], [0,0]) = 1
        assert_eq!(EqPolynomial::eq_vec(&[z, z], &[z, z]), o);
        // eq([1,1], [1,1]) = 1
        assert_eq!(EqPolynomial::eq_vec(&[o, o], &[o, o]), o);
        // eq([0,1], [1,0]) = 0
        assert_eq!(EqPolynomial::eq_vec(&[z, o], &[o, z]), z);
    }

    #[test]
    fn test_eq_evals() {
        let mut rng = test_rng();
        let r: Vec<Scalar> = (0..3).map(|_| Scalar::random(&mut rng)).collect();
        
        let evals = EqPolynomial::evals(&r);
        
        assert_eq!(evals.len(), 8);
        
        // Verify a few evaluations
        for i in 0..8 {
            let expected = EqPolynomial::eval_at_index(&r, i);
            assert_eq!(evals[i], expected, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_eq_evals_sum_to_one() {
        let mut rng = test_rng();
        let r: Vec<Scalar> = (0..4).map(|_| Scalar::random(&mut rng)).collect();
        
        let evals = EqPolynomial::evals(&r);
        let sum: Scalar = evals.iter().copied().sum();
        
        assert_eq!(sum, Scalar::one());
    }
}

