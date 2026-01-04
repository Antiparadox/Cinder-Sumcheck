//! Cinder MLE representations
//!
//! The Cinder protocol represents a sparse matrix using 2s+1 MLEs:
//! - row_t(k) for t ∈ [s]: the t-th bit of the row index for entry k
//! - col_t(k) for t ∈ [s]: the t-th bit of the column index for entry k  
//! - val(k): the value of entry k

use crate::field::{Scalar, CinderField};
use crate::sparse_matrix::CinderMatrix;

/// Multilinear polynomial in evaluation form
/// 
/// Stores evaluations over the boolean hypercube {0,1}^n
#[derive(Clone, Debug)]
pub struct MultiLinearPoly {
    /// Evaluations at each point of the hypercube
    /// coeffs[i] = f(bits(i)) where bits(i) is the binary representation of i
    pub coeffs: Vec<Scalar>,
    /// Number of variables
    pub num_vars: usize,
}

impl MultiLinearPoly {
    /// Create a new MLE from evaluations
    pub fn new(coeffs: Vec<Scalar>) -> Self {
        let len = coeffs.len();
        assert!(len.is_power_of_two(), "Coefficients length must be a power of 2");
        let num_vars = len.trailing_zeros() as usize;
        Self { coeffs, num_vars }
    }

    /// Create a zero polynomial with given number of variables
    pub fn zero(num_vars: usize) -> Self {
        Self {
            coeffs: vec![Scalar::zero(); 1 << num_vars],
            num_vars,
        }
    }

    /// Get the number of evaluations
    pub fn len(&self) -> usize {
        self.coeffs.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.coeffs.is_empty()
    }

    /// Evaluate the polynomial at a point
    /// 
    /// Uses the standard MLE evaluation algorithm:
    /// f(r_0, ..., r_{n-1}) = Σ_i f(i) · eq(i, r)
    pub fn evaluate(&self, point: &[Scalar]) -> Scalar {
        assert_eq!(point.len(), self.num_vars, "Point dimension mismatch");
        
        if self.num_vars == 0 {
            return self.coeffs[0];
        }
        
        // Iterative evaluation using the folding technique
        let mut evals = self.coeffs.clone();
        let mut size = self.coeffs.len();
        
        for r in point.iter() {
            size /= 2;
            for i in 0..size {
                // evals[i] = evals[2i] · (1 - r) + evals[2i+1] · r
                //          = evals[2i] + r · (evals[2i+1] - evals[2i])
                evals[i] = evals[2 * i] + *r * (evals[2 * i + 1] - evals[2 * i]);
            }
        }
        
        evals[0]
    }

    /// Bind the first variable to a value r, reducing the polynomial by one variable
    /// 
    /// Returns a new polynomial f'(x_1, ..., x_{n-1}) = f(r, x_1, ..., x_{n-1})
    pub fn bind_first_variable(&self, r: Scalar) -> Self {
        if self.num_vars == 0 {
            return self.clone();
        }
        
        let new_size = self.coeffs.len() / 2;
        let mut new_coeffs = Vec::with_capacity(new_size);
        
        for i in 0..new_size {
            // f'[i] = f[2i] + r * (f[2i+1] - f[2i])
            let val = self.coeffs[2 * i] + r * (self.coeffs[2 * i + 1] - self.coeffs[2 * i]);
            new_coeffs.push(val);
        }
        
        Self {
            coeffs: new_coeffs,
            num_vars: self.num_vars - 1,
        }
    }

    /// Bind the first variable in-place
    pub fn bind_first_variable_in_place(&mut self, r: Scalar) {
        if self.num_vars == 0 {
            return;
        }
        
        let new_size = self.coeffs.len() / 2;
        
        for i in 0..new_size {
            self.coeffs[i] = self.coeffs[2 * i] + r * (self.coeffs[2 * i + 1] - self.coeffs[2 * i]);
        }
        
        self.coeffs.truncate(new_size);
        self.num_vars -= 1;
    }
}

/// The 2s+1 MLEs for the Cinder representation
/// 
/// For a sparse matrix with s = log(m) bit indices:
/// - row_bits[t] is the MLE of the t-th bit of row indices
/// - col_bits[t] is the MLE of the t-th bit of column indices
/// - val is the MLE of the values
#[derive(Clone, Debug)]
pub struct CinderMLEs {
    /// MLEs for row index bits: row_t(k) = bit_t(row_k)
    pub row_bits: Vec<MultiLinearPoly>,
    /// MLEs for column index bits: col_t(k) = bit_t(col_k)
    pub col_bits: Vec<MultiLinearPoly>,
    /// MLE for values: val(k) = V_{row_k, col_k}
    pub val: MultiLinearPoly,
    /// Values as u64 (for delayed field conversion)
    pub val_u64: Vec<u64>,
    /// Number of index bits (s = log m)
    pub s: usize,
    /// Number of variables (log n where n is padded)
    pub num_vars: usize,
}

impl CinderMLEs {
    /// Construct CinderMLEs from a sparse matrix
    /// 
    /// This builds the 2s+1 MLEs from the sparse matrix representation.
    pub fn from_matrix(matrix: &CinderMatrix) -> Self {
        let s = matrix.s;
        let num_vars = matrix.log_n;
        let padded_n = matrix.padded_n;
        
        // Build row_bits MLEs
        let mut row_bits = Vec::with_capacity(s);
        for t in 0..s {
            let mut coeffs = Vec::with_capacity(padded_n);
            for k in 0..padded_n {
                coeffs.push(matrix.row_bit(k, t));
            }
            row_bits.push(MultiLinearPoly::new(coeffs));
        }
        
        // Build col_bits MLEs
        let mut col_bits = Vec::with_capacity(s);
        for t in 0..s {
            let mut coeffs = Vec::with_capacity(padded_n);
            for k in 0..padded_n {
                coeffs.push(matrix.col_bit(k, t));
            }
            col_bits.push(MultiLinearPoly::new(coeffs));
        }
        
        // Build val MLE (both field and u64 versions)
        let mut val_coeffs = Vec::with_capacity(padded_n);
        let mut val_u64 = Vec::with_capacity(padded_n);
        for k in 0..padded_n {
            val_coeffs.push(matrix.value(k));
            val_u64.push(matrix.value_u64(k));
        }
        let val = MultiLinearPoly::new(val_coeffs);
        
        Self {
            row_bits,
            col_bits,
            val,
            val_u64,
            s,
            num_vars,
        }
    }

    /// Get the total number of MLEs (2s + 1)
    pub fn num_mles(&self) -> usize {
        2 * self.s + 1
    }
}

#[cfg(test)]
mod cinder_mle_tests {
    use super::*;
    use crate::sparse_matrix::SparseEntry;
    use ark_std::test_rng;

    #[test]
    fn test_mle_evaluation() {
        // Create a simple MLE: f(x) = 1 + 2x = [1, 3]
        // f(0) = 1, f(1) = 3
        let mle = MultiLinearPoly::new(vec![
            Scalar::from_u64(1),
            Scalar::from_u64(3),
        ]);
        
        assert_eq!(mle.evaluate(&[Scalar::zero()]), Scalar::from_u64(1));
        assert_eq!(mle.evaluate(&[Scalar::one()]), Scalar::from_u64(3));
        
        // f(0.5) = 1 + 0.5 * (3 - 1) = 2
        let half = Scalar::from_u64(2).inverse().unwrap();
        assert_eq!(mle.evaluate(&[half]), Scalar::from_u64(2));
    }

    #[test]
    fn test_mle_bind_variable() {
        // f(x0, x1) = 1 + 2*x0 + 3*x1 + 4*x0*x1
        // Evaluations: f(0,0)=1, f(1,0)=3, f(0,1)=4, f(1,1)=10
        let mle = MultiLinearPoly::new(vec![
            Scalar::from_u64(1),  // f(0,0)
            Scalar::from_u64(3),  // f(1,0)
            Scalar::from_u64(4),  // f(0,1)
            Scalar::from_u64(10), // f(1,1)
        ]);
        
        // Bind x0 = 0: f'(x1) = f(0, x1) = 1 + 3*x1 = [1, 4]
        let bound = mle.bind_first_variable(Scalar::zero());
        assert_eq!(bound.coeffs, vec![Scalar::from_u64(1), Scalar::from_u64(4)]);
        
        // Bind x0 = 1: f'(x1) = f(1, x1) = 3 + 7*x1 = [3, 10]
        let bound = mle.bind_first_variable(Scalar::one());
        assert_eq!(bound.coeffs, vec![Scalar::from_u64(3), Scalar::from_u64(10)]);
    }

    #[test]
    fn test_cinder_mles_construction() {
        let entries = vec![
            SparseEntry::new(0, 0, Scalar::from_u64(5)),
            SparseEntry::new(1, 2, Scalar::from_u64(7)),
        ];
        let matrix = CinderMatrix::new(4, entries);
        let mles = CinderMLEs::from_matrix(&matrix);
        
        assert_eq!(mles.s, 2);
        assert_eq!(mles.num_vars, 1); // log2(2) = 1, but padded to 2
        assert_eq!(mles.num_mles(), 5); // 2*2 + 1 = 5
        
        // Check row_bits[0] for entry 0 (row=0) and entry 1 (row=1)
        assert_eq!(mles.row_bits[0].coeffs[0], Scalar::zero()); // bit0 of 0
        assert_eq!(mles.row_bits[0].coeffs[1], Scalar::one());  // bit0 of 1
        
        // Check val
        assert_eq!(mles.val.coeffs[0], Scalar::from_u64(5));
        assert_eq!(mles.val.coeffs[1], Scalar::from_u64(7));
    }

    #[test]
    fn test_random_matrix_mles() {
        let mut rng = test_rng();
        let matrix = CinderMatrix::random(8, 5, &mut rng);
        let mles = CinderMLEs::from_matrix(&matrix);
        
        assert_eq!(mles.s, 3); // log2(8) = 3
        assert_eq!(mles.num_mles(), 7); // 2*3 + 1
    }
}

