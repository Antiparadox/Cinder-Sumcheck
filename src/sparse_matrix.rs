//! Sparse matrix representation for Cinder
//!
//! A sparse matrix stores only non-zero entries, which is efficient
//! for R1CS constraint matrices that are typically very sparse.

use crate::field::{Scalar, CinderField, get_bit, bit_to_field};
use ark_std::rand::RngCore;

/// A single non-zero entry in the sparse matrix
#[derive(Clone, Debug)]
pub struct SparseEntry {
    /// Row index (0-indexed)
    pub row: usize,
    /// Column index (0-indexed)
    pub col: usize,
    /// Non-zero value (as field element)
    pub value: Scalar,
    /// Non-zero value (as u64 for delayed field conversion)
    /// Only valid for small values (≤64 bits)
    pub value_u64: u64,
}

impl SparseEntry {
    /// Create a new sparse entry
    pub fn new(row: usize, col: usize, value: Scalar) -> Self {
        Self { row, col, value, value_u64: 0 }
    }
    
    /// Create a new sparse entry with small value (for delayed field conversion)
    pub fn new_small(row: usize, col: usize, value_u64: u64) -> Self {
        Self { 
            row, 
            col, 
            value: Scalar::from_u64(value_u64), 
            value_u64 
        }
    }
}

/// A sparse matrix in Cinder format
/// 
/// The matrix is m × m where m = 2^s, and has n non-zero entries.
/// This representation is optimized for the Cinder protocol.
#[derive(Clone, Debug)]
pub struct CinderMatrix {
    /// Matrix dimension (m × m matrix)
    pub m: usize,
    /// log2(m) - number of bits needed for row/col indices
    pub s: usize,
    /// Non-zero entries
    pub entries: Vec<SparseEntry>,
    /// Number of non-zero entries (convenience, same as entries.len())
    pub n: usize,
    /// log2 of the next power of 2 >= n
    pub log_n: usize,
    /// Padded size (2^log_n)
    pub padded_n: usize,
}

impl CinderMatrix {
    /// Create a new sparse matrix from a list of non-zero entries
    /// 
    /// # Arguments
    /// * `m` - Matrix dimension (must be a power of 2)
    /// * `entries` - List of non-zero entries
    /// 
    /// # Panics
    /// Panics if m is not a power of 2
    pub fn new(m: usize, entries: Vec<SparseEntry>) -> Self {
        assert!(m.is_power_of_two(), "Matrix dimension must be a power of 2");
        
        let s = m.trailing_zeros() as usize;
        let n = entries.len();
        
        // Pad n to next power of 2 for sumcheck
        let log_n = if n == 0 { 0 } else { (n.next_power_of_two()).trailing_zeros() as usize };
        let padded_n = 1 << log_n;
        
        // Validate all entries are within bounds
        for entry in &entries {
            assert!(entry.row < m, "Row index {} out of bounds (m={})", entry.row, m);
            assert!(entry.col < m, "Column index {} out of bounds (m={})", entry.col, m);
        }
        
        Self {
            m,
            s,
            entries,
            n,
            log_n,
            padded_n,
        }
    }

    /// Create a random sparse matrix for testing/benchmarking
    /// 
    /// # Arguments
    /// * `m` - Matrix dimension (must be a power of 2)
    /// * `n` - Number of non-zero entries
    /// * `rng` - Random number generator
    pub fn random<R: RngCore>(m: usize, n: usize, rng: &mut R) -> Self {
        assert!(m.is_power_of_two(), "Matrix dimension must be a power of 2");
        assert!(n <= m * m, "Cannot have more non-zero entries than matrix size");
        
        let mut entries = Vec::with_capacity(n);
        
        // Generate random entries (may have duplicates, but that's okay for testing)
        for _ in 0..n {
            let row = (rng.next_u64() as usize) % m;
            let col = (rng.next_u64() as usize) % m;
            let value = Scalar::random(rng);
            entries.push(SparseEntry::new(row, col, value));
        }
        
        Self::new(m, entries)
    }
    
    /// Create a random sparse matrix with SMALL values (≤16 bits)
    /// 
    /// This simulates realistic R1CS matrices where coefficients are small.
    /// 
    /// # Arguments
    /// * `m` - Matrix dimension (must be a power of 2)
    /// * `n` - Number of non-zero entries  
    /// * `rng` - Random number generator
    pub fn random_small<R: RngCore>(m: usize, n: usize, rng: &mut R) -> Self {
        assert!(m.is_power_of_two(), "Matrix dimension must be a power of 2");
        assert!(n <= m * m, "Cannot have more non-zero entries than matrix size");
        
        let mut entries = Vec::with_capacity(n);
        
        // Generate random entries with small (16-bit) values
        for _ in 0..n {
            let row = (rng.next_u64() as usize) % m;
            let col = (rng.next_u64() as usize) % m;
            // Use only 16 bits for the value
            let value_u64 = rng.next_u64() & 0xFFFF;
            entries.push(SparseEntry::new_small(row, col, value_u64));
        }
        
        Self::new(m, entries)
    }

    /// Create an identity matrix (sparse representation)
    pub fn identity(m: usize) -> Self {
        assert!(m.is_power_of_two(), "Matrix dimension must be a power of 2");
        
        let entries: Vec<_> = (0..m)
            .map(|i| SparseEntry::new(i, i, Scalar::one()))
            .collect();
        
        Self::new(m, entries)
    }

    /// Get the t-th bit of the row index for entry k
    #[inline]
    pub fn row_bit(&self, k: usize, t: usize) -> Scalar {
        if k >= self.n {
            Scalar::zero() // Padding
        } else {
            bit_to_field(get_bit(self.entries[k].row, t))
        }
    }

    /// Get the t-th bit of the column index for entry k
    #[inline]
    pub fn col_bit(&self, k: usize, t: usize) -> Scalar {
        if k >= self.n {
            Scalar::zero() // Padding
        } else {
            bit_to_field(get_bit(self.entries[k].col, t))
        }
    }

    /// Get the value of entry k (as field element)
    #[inline]
    pub fn value(&self, k: usize) -> Scalar {
        if k >= self.n {
            Scalar::zero() // Padding
        } else {
            self.entries[k].value
        }
    }
    
    /// Get the value of entry k (as u64, for delayed field conversion)
    #[inline]
    pub fn value_u64(&self, k: usize) -> u64 {
        if k >= self.n {
            0 // Padding
        } else {
            self.entries[k].value_u64
        }
    }

    /// Evaluate the MLE of this matrix at point (r_x, r_y)
    /// 
    /// This computes: Σ_{i,j} V_{i,j} · eq_i(r_x) · eq_j(r_y)
    /// 
    /// Used for testing to verify sumcheck correctness.
    pub fn evaluate_mle(&self, r_x: &[Scalar], r_y: &[Scalar]) -> Scalar {
        assert_eq!(r_x.len(), self.s, "r_x length must equal s");
        assert_eq!(r_y.len(), self.s, "r_y length must equal s");
        
        let mut result = Scalar::zero();
        
        for entry in &self.entries {
            // Compute eq(row, r_x) = Π_t (row_t · r_x_t + (1 - row_t)(1 - r_x_t))
            let mut eq_row = Scalar::one();
            for t in 0..self.s {
                let row_bit = bit_to_field(get_bit(entry.row, t));
                eq_row *= row_bit * r_x[t] + (Scalar::one() - row_bit) * (Scalar::one() - r_x[t]);
            }
            
            // Compute eq(col, r_y) = Π_t (col_t · r_y_t + (1 - col_t)(1 - r_y_t))
            let mut eq_col = Scalar::one();
            for t in 0..self.s {
                let col_bit = bit_to_field(get_bit(entry.col, t));
                eq_col *= col_bit * r_y[t] + (Scalar::one() - col_bit) * (Scalar::one() - r_y[t]);
            }
            
            result += entry.value * eq_row * eq_col;
        }
        
        result
    }
}

#[cfg(test)]
mod sparse_matrix_tests {
    use super::*;
    use ark_std::test_rng;

    #[test]
    fn test_sparse_matrix_creation() {
        let entries = vec![
            SparseEntry::new(0, 0, Scalar::one()),
            SparseEntry::new(1, 2, Scalar::from_u64(5)),
            SparseEntry::new(3, 1, Scalar::from_u64(7)),
        ];
        
        let matrix = CinderMatrix::new(4, entries);
        
        assert_eq!(matrix.m, 4);
        assert_eq!(matrix.s, 2);
        assert_eq!(matrix.n, 3);
        assert_eq!(matrix.log_n, 2); // ceil(log2(3)) = 2
        assert_eq!(matrix.padded_n, 4);
    }

    #[test]
    fn test_random_matrix() {
        let mut rng = test_rng();
        let matrix = CinderMatrix::random(16, 10, &mut rng);
        
        assert_eq!(matrix.m, 16);
        assert_eq!(matrix.s, 4);
        assert_eq!(matrix.n, 10);
    }

    #[test]
    fn test_identity_matrix() {
        let matrix = CinderMatrix::identity(4);
        
        assert_eq!(matrix.n, 4);
        for (i, entry) in matrix.entries.iter().enumerate() {
            assert_eq!(entry.row, i);
            assert_eq!(entry.col, i);
            assert_eq!(entry.value, Scalar::one());
        }
    }

    #[test]
    fn test_row_col_bits() {
        let entries = vec![
            SparseEntry::new(0b11, 0b01, Scalar::one()), // row=3, col=1
        ];
        let matrix = CinderMatrix::new(4, entries);
        
        // Row bits of 3 = 0b11: bit0=1, bit1=1
        assert_eq!(matrix.row_bit(0, 0), Scalar::one());
        assert_eq!(matrix.row_bit(0, 1), Scalar::one());
        
        // Col bits of 1 = 0b01: bit0=1, bit1=0
        assert_eq!(matrix.col_bit(0, 0), Scalar::one());
        assert_eq!(matrix.col_bit(0, 1), Scalar::zero());
    }

    #[test]
    fn test_identity_mle_evaluation() {
        let mut rng = test_rng();
        let matrix = CinderMatrix::identity(4);
        
        // For identity matrix, V(r_x, r_y) = eq(r_x, r_y)
        let r_x: Vec<Scalar> = (0..2).map(|_| Scalar::random(&mut rng)).collect();
        let r_y: Vec<Scalar> = (0..2).map(|_| Scalar::random(&mut rng)).collect();
        
        let mle_val = matrix.evaluate_mle(&r_x, &r_y);
        
        // Compute eq(r_x, r_y) directly
        let mut eq_val = Scalar::one();
        for t in 0..2 {
            eq_val *= r_x[t] * r_y[t] + (Scalar::one() - r_x[t]) * (Scalar::one() - r_y[t]);
        }
        
        assert_eq!(mle_val, eq_val);
    }
}

