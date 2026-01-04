//! Sumcheck protocol for Cinder
//!
//! This module implements the sumcheck prover and verifier for the Cinder
//! polynomial: F(Z) = val(Z) · E_x(Z) · E_y(Z)
//!
//! OPTIMIZED VERSION v3:
//! - Pattern-based E_x/E_y precomputation
//! - Precompute E_x values at boolean points using patterns
//! - Only compute polynomial at non-boolean points

use crate::field::{Scalar, CinderField};
use crate::sparse_matrix::CinderMatrix;
use crate::cinder_mle::CinderMLEs;
use std::collections::HashMap;
use rayon::prelude::*;

/// Result of a sumcheck proof
#[derive(Clone, Debug)]
pub struct SumcheckProof {
    /// Round polynomials (evaluations at 0, 1, 2, ..., degree)
    pub round_polys: Vec<Vec<Scalar>>,
    /// Final evaluation claim
    pub final_claim: Scalar,
    /// Challenges received from verifier
    pub challenges: Vec<Scalar>,
}

// ============================================================================
// SETUP PHASE (Preprocessing)
// ============================================================================

/// Setup data for Cinder - computed once per matrix
#[derive(Clone, Debug)]
pub struct CinderSetup {
    /// The Cinder MLEs (row_t, col_t, val)
    pub mles: CinderMLEs,
    /// Number of index bits (s = log m)
    pub s: usize,
    /// Number of sumcheck variables (log n_padded)
    pub num_vars: usize,
    /// Padded size of entries
    pub padded_n: usize,
    /// Row indices for pattern-based batching (padded with 0)
    pub row_indices: Vec<usize>,
    /// Col indices for pattern-based batching (padded with 0)
    pub col_indices: Vec<usize>,
}

impl CinderSetup {
    /// Create setup from a sparse matrix
    pub fn new(matrix: &CinderMatrix) -> Self {
        let mles = CinderMLEs::from_matrix(matrix);
        
        // Store row/col indices for pattern-based batching
        let mut row_indices = Vec::with_capacity(matrix.padded_n);
        let mut col_indices = Vec::with_capacity(matrix.padded_n);
        for k in 0..matrix.padded_n {
            if k < matrix.n {
                row_indices.push(matrix.entries[k].row);
                col_indices.push(matrix.entries[k].col);
            } else {
                row_indices.push(0);
                col_indices.push(0);
            }
        }
        
        Self {
            s: mles.s,
            num_vars: mles.num_vars,
            padded_n: matrix.padded_n,
            mles,
            row_indices,
            col_indices,
        }
    }

    /// Get the degree of the sumcheck polynomial per round
    pub fn degree(&self) -> usize {
        2 * self.s + 1
    }
}

// ============================================================================
// POLYNOMIAL ARITHMETIC (OPTIMIZED)
// ============================================================================

#[inline]
fn one() -> Scalar {
    <Scalar as CinderField>::one()
}

#[inline]
fn zero() -> Scalar {
    <Scalar as CinderField>::zero()
}

/// Evaluate polynomial at a point using Horner's method
#[inline]
fn poly_eval(coeffs: &[Scalar], x: Scalar) -> Scalar {
    let mut result = zero();
    for &c in coeffs.iter().rev() {
        result = result * x + c;
    }
    result
}

/// Evaluate polynomial at points 0, 1, 2, ..., n-1
#[inline]
fn poly_eval_at_integers(coeffs: &[Scalar], n: usize) -> Vec<Scalar> {
    (0..n).map(|i| poly_eval(coeffs, Scalar::from_u64(i as u64))).collect()
}

/// Karatsuba multiplication for two polynomials
/// Returns a new vector with the product
fn karatsuba_mul(a: &[Scalar], b: &[Scalar]) -> Vec<Scalar> {
    let n = a.len().max(b.len());
    
    // Base case: schoolbook for small polynomials
    if n <= 16 {
        let result_len = a.len() + b.len() - 1;
        let mut result = vec![zero(); result_len];
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() {
                result[i + j] += ai * bj;
            }
        }
        return result;
    }
    
    // Pad to same length
    let mut a_padded = a.to_vec();
    let mut b_padded = b.to_vec();
    a_padded.resize(n, zero());
    b_padded.resize(n, zero());
    
    let mid = n / 2;
    
    // Split
    let a_lo = &a_padded[..mid];
    let a_hi = &a_padded[mid..];
    let b_lo = &b_padded[..mid];
    let b_hi = &b_padded[mid..];
    
    // z0 = a_lo * b_lo
    let z0 = karatsuba_mul(a_lo, b_lo);
    
    // z2 = a_hi * b_hi
    let z2 = karatsuba_mul(a_hi, b_hi);
    
    // a_sum = a_lo + a_hi, b_sum = b_lo + b_hi
    let a_sum: Vec<Scalar> = a_lo.iter().zip(a_hi.iter())
        .map(|(&x, &y)| x + y).collect();
    let b_sum: Vec<Scalar> = b_lo.iter().zip(b_hi.iter())
        .map(|(&x, &y)| x + y).collect();
    
    // z1 = (a_lo + a_hi) * (b_lo + b_hi) - z0 - z2
    let mut z1 = karatsuba_mul(&a_sum, &b_sum);
    for (i, &z) in z0.iter().enumerate() {
        if i < z1.len() {
            z1[i] -= z;
        }
    }
    for (i, &z) in z2.iter().enumerate() {
        if i < z1.len() {
            z1[i] -= z;
        }
    }
    
    // Combine: result = z0 + z1*X^mid + z2*X^(2*mid)
    let result_len = 2 * n - 1;
    let mut result = vec![zero(); result_len];
    
    for (i, &z) in z0.iter().enumerate() {
        result[i] += z;
    }
    for (i, &z) in z1.iter().enumerate() {
        if mid + i < result_len {
            result[mid + i] += z;
        }
    }
    for (i, &z) in z2.iter().enumerate() {
        if 2 * mid + i < result_len {
            result[2 * mid + i] += z;
        }
    }
    
    result
}

/// Multiply two polynomials using schoolbook algorithm
/// (Karatsuba has issues, use simple approach for now)
#[inline]
fn poly_mul(a: &[Scalar], a_len: usize, b: &[Scalar], b_len: usize, result: &mut [Scalar]) {
    let result_len = a_len + b_len - 1;
    
    // Clear result
    for r in result.iter_mut().take(result_len) {
        *r = zero();
    }
    
    // Schoolbook multiplication
    for i in 0..a_len {
        for j in 0..b_len {
            result[i + j] += a[i] * b[j];
        }
    }
}

// ============================================================================
// HARDCODED POLYNOMIAL BUILDING FOR s=20
// ============================================================================

/// Build degree-20 polynomial from 20 linear factors
/// E(X) = Π_{t=0}^{19} (eq_0[t] + eq_d[t] * X)
/// 
/// This is the fully unrolled version - no loops at runtime
#[inline(always)]
fn build_poly_20(eq_0: &[Scalar], eq_d: &[Scalar]) -> [Scalar; 21] {
    // Layer 0: just [eq_0[0], eq_d[0]] (degree-1)
    let p1 = [eq_0[0], eq_d[0]];
    
    // Layer 1: multiply by (eq_0[1] + eq_d[1]*X)
    let p2 = {
        let mut p = [zero(); 3];
        p[0] = eq_0[1] * p1[0];
        p[1] = eq_0[1] * p1[1] + eq_d[1] * p1[0];
        p[2] = eq_d[1] * p1[1];
        p
    };
    
    // Layer 2
    let p3 = {
        let mut p = [zero(); 4];
        p[0] = eq_0[2] * p2[0];
        p[1] = eq_0[2] * p2[1] + eq_d[2] * p2[0];
        p[2] = eq_0[2] * p2[2] + eq_d[2] * p2[1];
        p[3] = eq_d[2] * p2[2];
        p
    };
    
    // Layer 3
    let p4 = {
        let mut p = [zero(); 5];
        p[0] = eq_0[3] * p3[0];
        for j in 1..4 { p[j] = eq_0[3] * p3[j] + eq_d[3] * p3[j-1]; }
        p[4] = eq_d[3] * p3[3];
        p
    };
    
    // Layer 4
    let p5 = {
        let mut p = [zero(); 6];
        p[0] = eq_0[4] * p4[0];
        for j in 1..5 { p[j] = eq_0[4] * p4[j] + eq_d[4] * p4[j-1]; }
        p[5] = eq_d[4] * p4[4];
        p
    };
    
    // Layer 5
    let p6 = {
        let mut p = [zero(); 7];
        p[0] = eq_0[5] * p5[0];
        for j in 1..6 { p[j] = eq_0[5] * p5[j] + eq_d[5] * p5[j-1]; }
        p[6] = eq_d[5] * p5[5];
        p
    };
    
    // Layer 6
    let p7 = {
        let mut p = [zero(); 8];
        p[0] = eq_0[6] * p6[0];
        for j in 1..7 { p[j] = eq_0[6] * p6[j] + eq_d[6] * p6[j-1]; }
        p[7] = eq_d[6] * p6[6];
        p
    };
    
    // Layer 7
    let p8 = {
        let mut p = [zero(); 9];
        p[0] = eq_0[7] * p7[0];
        for j in 1..8 { p[j] = eq_0[7] * p7[j] + eq_d[7] * p7[j-1]; }
        p[8] = eq_d[7] * p7[7];
        p
    };
    
    // Layer 8
    let p9 = {
        let mut p = [zero(); 10];
        p[0] = eq_0[8] * p8[0];
        for j in 1..9 { p[j] = eq_0[8] * p8[j] + eq_d[8] * p8[j-1]; }
        p[9] = eq_d[8] * p8[8];
        p
    };
    
    // Layer 9
    let p10 = {
        let mut p = [zero(); 11];
        p[0] = eq_0[9] * p9[0];
        for j in 1..10 { p[j] = eq_0[9] * p9[j] + eq_d[9] * p9[j-1]; }
        p[10] = eq_d[9] * p9[9];
        p
    };
    
    // Layer 10
    let p11 = {
        let mut p = [zero(); 12];
        p[0] = eq_0[10] * p10[0];
        for j in 1..11 { p[j] = eq_0[10] * p10[j] + eq_d[10] * p10[j-1]; }
        p[11] = eq_d[10] * p10[10];
        p
    };
    
    // Layer 11
    let p12 = {
        let mut p = [zero(); 13];
        p[0] = eq_0[11] * p11[0];
        for j in 1..12 { p[j] = eq_0[11] * p11[j] + eq_d[11] * p11[j-1]; }
        p[12] = eq_d[11] * p11[11];
        p
    };
    
    // Layer 12
    let p13 = {
        let mut p = [zero(); 14];
        p[0] = eq_0[12] * p12[0];
        for j in 1..13 { p[j] = eq_0[12] * p12[j] + eq_d[12] * p12[j-1]; }
        p[13] = eq_d[12] * p12[12];
        p
    };
    
    // Layer 13
    let p14 = {
        let mut p = [zero(); 15];
        p[0] = eq_0[13] * p13[0];
        for j in 1..14 { p[j] = eq_0[13] * p13[j] + eq_d[13] * p13[j-1]; }
        p[14] = eq_d[13] * p13[13];
        p
    };
    
    // Layer 14
    let p15 = {
        let mut p = [zero(); 16];
        p[0] = eq_0[14] * p14[0];
        for j in 1..15 { p[j] = eq_0[14] * p14[j] + eq_d[14] * p14[j-1]; }
        p[15] = eq_d[14] * p14[14];
        p
    };
    
    // Layer 15
    let p16 = {
        let mut p = [zero(); 17];
        p[0] = eq_0[15] * p15[0];
        for j in 1..16 { p[j] = eq_0[15] * p15[j] + eq_d[15] * p15[j-1]; }
        p[16] = eq_d[15] * p15[15];
        p
    };
    
    // Layer 16
    let p17 = {
        let mut p = [zero(); 18];
        p[0] = eq_0[16] * p16[0];
        for j in 1..17 { p[j] = eq_0[16] * p16[j] + eq_d[16] * p16[j-1]; }
        p[17] = eq_d[16] * p16[16];
        p
    };
    
    // Layer 17
    let p18 = {
        let mut p = [zero(); 19];
        p[0] = eq_0[17] * p17[0];
        for j in 1..18 { p[j] = eq_0[17] * p17[j] + eq_d[17] * p17[j-1]; }
        p[18] = eq_d[17] * p17[17];
        p
    };
    
    // Layer 18
    let p19 = {
        let mut p = [zero(); 20];
        p[0] = eq_0[18] * p18[0];
        for j in 1..19 { p[j] = eq_0[18] * p18[j] + eq_d[18] * p18[j-1]; }
        p[19] = eq_d[18] * p18[18];
        p
    };
    
    // Layer 19 (final)
    let mut p20 = [zero(); 21];
    p20[0] = eq_0[19] * p19[0];
    for j in 1..20 { p20[j] = eq_0[19] * p19[j] + eq_d[19] * p19[j-1]; }
    p20[20] = eq_d[19] * p19[19];
    
    p20
}

/// Process one entry pair (helper function)
/// This function handles: gather eq values, build E_x/E_y, multiply, accumulate
#[inline(always)]
fn process_entry_pair_unrolled(
    eq_rx_even: &[Scalar],
    eq_rx_odd: &[Scalar],
    eq_ry_even: &[Scalar],
    eq_ry_odd: &[Scalar],
    val_0: Scalar,
    val_delta: Scalar,
    h_coeffs: &mut [Scalar],
) {
    // Gather eq values for E_x
    let mut eq_rx_0 = [zero(); 20];
    let mut eq_rx_d = [zero(); 20];
    for t in 0..20 {
        eq_rx_0[t] = eq_rx_even[t];
        eq_rx_d[t] = eq_rx_odd[t] - eq_rx_even[t];
    }
    
    // Gather eq values for E_y
    let mut eq_ry_0 = [zero(); 20];
    let mut eq_ry_d = [zero(); 20];
    for t in 0..20 {
        eq_ry_0[t] = eq_ry_even[t];
        eq_ry_d[t] = eq_ry_odd[t] - eq_ry_even[t];
    }
    
    // Build E_x and E_y using hardcoded unrolled function
    let ex = build_poly_20(&eq_rx_0, &eq_rx_d);
    let ey = build_poly_20(&eq_ry_0, &eq_ry_d);
    
    // Multiply E_x * E_y (both degree-20 → result degree-40)
    let mut exy = [zero(); 41];
    for i in 0..21 {
        for j in 0..21 {
            exy[i + j] += ex[i] * ey[j];
        }
    }
    
    // Multiply by val and add to h_coeffs
    for j in 0..41 {
        h_coeffs[j] += val_0 * exy[j];
        h_coeffs[j + 1] += val_delta * exy[j];
    }
}

// ============================================================================
// OPTIMIZED PROVER PHASE (Transposed memory layout)
// ============================================================================

/// Bookkeeping tables for efficient sumcheck prover
/// 
/// Uses transposed storage for better cache locality:
/// - eq_rx_flat[k * s + t] instead of eq_rx_evals[t][k]
/// - This puts all eq values for an entry contiguous in memory
/// 
/// For delayed field conversion:
/// - val_u64 stores original small values (first round only)
/// - val_evals stores field elements (after first bind)
struct BookkeepingTablesOpt {
    /// Value evaluations as field elements (used after round 1)
    val_evals: Vec<Scalar>,
    /// Value evaluations as u64 (used in round 1 only, for cache efficiency)
    val_u64: Vec<u64>,
    /// Whether we're in round 1 (can use u64 values)
    first_round: bool,
    
    /// Transposed layout: eq_rx_flat[k * s + t] = eq(r_x[t], row_t(k))
    eq_rx_flat: Vec<Scalar>,
    /// Transposed layout: eq_ry_flat[k * s + t] = eq(r_y[t], col_t(k))
    eq_ry_flat: Vec<Scalar>,
    
    size: usize,
    s: usize,
}

impl BookkeepingTablesOpt {
    fn new(setup: &CinderSetup, r_x: &[Scalar], r_y: &[Scalar]) -> Self {
        let s = setup.s;
        let size = setup.padded_n;
        
        // Check if u64 values are valid (non-zero means we have small values)
        // If all u64 values are 0, fall back to Scalar mode
        let has_valid_u64 = setup.mles.val_u64.iter().any(|&v| v != 0);
        
        let (val_u64, val_evals, first_round) = if has_valid_u64 {
            // Use u64 values for delayed conversion
            (setup.mles.val_u64.clone(), Vec::new(), true)
        } else {
            // Fall back to Scalar mode (for matrices with large values)
            (Vec::new(), setup.mles.val.coeffs.clone(), false)
        };
        
        // Build transposed eq tables: eq_flat[k * s + t]
        let mut eq_rx_flat = vec![zero(); size * s];
        let mut eq_ry_flat = vec![zero(); size * s];
        
        for k in 0..size {
            for t in 0..s {
                let row_bit = setup.mles.row_bits[t].coeffs[k];
                let col_bit = setup.mles.col_bits[t].coeffs[k];
                let eq_rx = r_x[t] * row_bit + (one() - r_x[t]) * (one() - row_bit);
                let eq_ry = r_y[t] * col_bit + (one() - r_y[t]) * (one() - col_bit);
                eq_rx_flat[k * s + t] = eq_rx;
                eq_ry_flat[k * s + t] = eq_ry;
            }
        }
        
        Self { 
            val_evals,
            val_u64,
            first_round,
            eq_rx_flat,
            eq_ry_flat,
            size, 
            s 
        }
    }

    fn bind_variable(&mut self, r: Scalar) {
        let new_size = self.size / 2;
        let s = self.s;
        
        if self.first_round {
            // First bind: convert u64 to Scalar while binding
            let mut new_val = Vec::with_capacity(new_size);
            for i in 0..new_size {
                let val_0 = Scalar::from_u64(self.val_u64[2 * i]);
                let val_1 = Scalar::from_u64(self.val_u64[2 * i + 1]);
                new_val.push(val_0 + r * (val_1 - val_0));
            }
            self.val_evals = new_val;
            self.val_u64.clear(); // Free memory
            self.first_round = false;
        } else {
            // Subsequent binds: use Scalar values
            for i in 0..new_size {
                self.val_evals[i] = self.val_evals[2 * i] 
                    + r * (self.val_evals[2 * i + 1] - self.val_evals[2 * i]);
            }
            self.val_evals.truncate(new_size);
        }
        
        // Update eq tables (transposed layout)
        // New: eq_flat[i * s + t] = eq_flat[(2i) * s + t] + r * (eq_flat[(2i+1) * s + t] - eq_flat[(2i) * s + t])
        for i in 0..new_size {
            let base_even = (2 * i) * s;
            let base_odd = (2 * i + 1) * s;
            let base_new = i * s;
            
            for t in 0..s {
                let eq_0 = self.eq_rx_flat[base_even + t];
                let eq_1 = self.eq_rx_flat[base_odd + t];
                self.eq_rx_flat[base_new + t] = eq_0 + r * (eq_1 - eq_0);
            }
            for t in 0..s {
                let eq_0 = self.eq_ry_flat[base_even + t];
                let eq_1 = self.eq_ry_flat[base_odd + t];
                self.eq_ry_flat[base_new + t] = eq_0 + r * (eq_1 - eq_0);
            }
        }
        self.eq_rx_flat.truncate(new_size * s);
        self.eq_ry_flat.truncate(new_size * s);
        
        self.size = new_size;
    }

    /// Compute E_x value at position k on-demand (O(s))
    #[inline]
    fn get_e_x(&self, k: usize) -> Scalar {
        let base = k * self.s;
        let mut ex = one();
        for t in 0..self.s {
            ex *= self.eq_rx_flat[base + t];
        }
        ex
    }

    /// Compute E_y value at position k on-demand (O(s))
    #[inline]
    fn get_e_y(&self, k: usize) -> Scalar {
        let base = k * self.s;
        let mut ey = one();
        for t in 0..self.s {
            ey *= self.eq_ry_flat[base + t];
        }
        ey
    }
    
    /// Get eq_rx value at position k for bit t (O(1))
    #[inline]
    fn get_eq_rx(&self, k: usize, t: usize) -> Scalar {
        self.eq_rx_flat[k * self.s + t]
    }
    
    /// Get eq_ry value at position k for bit t (O(1))
    #[inline]
    fn get_eq_ry(&self, k: usize, t: usize) -> Scalar {
        self.eq_ry_flat[k * self.s + t]
    }
}

/// Sumcheck prover state
pub struct CinderSumcheckProver<'a> {
    setup: &'a CinderSetup,
    round: usize,
    challenges: Vec<Scalar>,
    bookkeeping: BookkeepingTablesOpt,
}

impl<'a> CinderSumcheckProver<'a> {
    pub fn new(setup: &'a CinderSetup, r_x: Vec<Scalar>, r_y: Vec<Scalar>) -> Self {
        let bookkeeping = BookkeepingTablesOpt::new(setup, &r_x, &r_y);
        
        Self {
            setup,
            round: 0,
            challenges: vec![],
            bookkeeping,
        }
    }

    pub fn degree(&self) -> usize {
        self.setup.degree()
    }

    pub fn num_rounds(&self) -> usize {
        self.setup.num_vars
    }

    /// Prove one round using coefficient-based approach with cache-friendly access
    /// 
    /// Builds E_x(X) and E_y(X) as polynomials, multiplies them,
    /// and accumulates into h_coeffs.
    pub fn prove_round(&mut self) -> Vec<Scalar> {
        let s = self.setup.s;
        let degree = self.degree();
        let num_evals = degree + 1;
        
        let mut h_coeffs = vec![zero(); degree + 2];
        let half_size = self.bookkeeping.size / 2;
        
        // Pre-allocate buffers
        let mut ex_curr = vec![zero(); s + 1];
        let mut ex_next = vec![zero(); s + 2];
        let mut ey_curr = vec![zero(); s + 1];
        let mut ey_next = vec![zero(); s + 2];
        let mut exy = vec![zero(); 2 * s + 1];
        
        // Use u64 values in round 1 for better cache efficiency
        let use_u64 = self.bookkeeping.first_round;
        
        for i in 0..half_size {
            let (val_0, val_delta) = if use_u64 {
                let v0 = self.bookkeeping.val_u64[2 * i];
                let v1 = self.bookkeeping.val_u64[2 * i + 1];
                let val_0 = Scalar::from_u64(v0);
                let val_1 = Scalar::from_u64(v1);
                (val_0, val_1 - val_0)
            } else {
                let val_0 = self.bookkeeping.val_evals[2 * i];
                let val_1 = self.bookkeeping.val_evals[2 * i + 1];
                (val_0, val_1 - val_0)
            };
            
            // Build E_x(X) = Π_t (eq_0[t] + eq_delta[t] * X)
            // Using cache-friendly transposed access
            ex_curr[0] = one();
            let mut ex_len = 1;
            
            for t in 0..s {
                let eq_0 = self.bookkeeping.get_eq_rx(2 * i, t);
                let eq_1 = self.bookkeeping.get_eq_rx(2 * i + 1, t);
                let eq_delta = eq_1 - eq_0;
                
                ex_next[0] = eq_0 * ex_curr[0];
                for j in 1..ex_len {
                    ex_next[j] = eq_0 * ex_curr[j] + eq_delta * ex_curr[j - 1];
                }
                ex_next[ex_len] = eq_delta * ex_curr[ex_len - 1];
                ex_len += 1;
                std::mem::swap(&mut ex_curr, &mut ex_next);
            }
            
            // Build E_y(X) similarly
            ey_curr[0] = one();
            let mut ey_len = 1;
            
            for t in 0..s {
                let eq_0 = self.bookkeeping.get_eq_ry(2 * i, t);
                let eq_1 = self.bookkeeping.get_eq_ry(2 * i + 1, t);
                let eq_delta = eq_1 - eq_0;
                
                ey_next[0] = eq_0 * ey_curr[0];
                for j in 1..ey_len {
                    ey_next[j] = eq_0 * ey_curr[j] + eq_delta * ey_curr[j - 1];
                }
                ey_next[ey_len] = eq_delta * ey_curr[ey_len - 1];
                ey_len += 1;
                std::mem::swap(&mut ey_curr, &mut ey_next);
            }
            
            // Multiply E_x * E_y
            let exy_len = ex_len + ey_len - 1;
            poly_mul(&ex_curr, ex_len, &ey_curr, ey_len, &mut exy);
            
            // Multiply by val and add to h_coeffs
            for j in 0..exy_len {
                h_coeffs[j] += val_0 * exy[j];
                h_coeffs[j + 1] += val_delta * exy[j];
            }
        }
        
        poly_eval_at_integers(&h_coeffs[..num_evals.min(h_coeffs.len())], num_evals)
    }

    /// Prove one round using HARDCODED polynomial building for s=20
    /// 
    /// This version uses the unrolled build_poly_20 function instead of loops
    /// In round 1, uses u64 values for better cache efficiency
    pub fn prove_round_hardcoded_20(&mut self) -> Vec<Scalar> {
        let s = self.setup.s;
        assert_eq!(s, 20, "Hardcoded version requires s=20");
        
        let degree = self.degree();
        let num_evals = degree + 1;
        
        let mut h_coeffs = vec![zero(); degree + 2];
        let half_size = self.bookkeeping.size / 2;
        
        // Use u64 values in round 1 for better cache efficiency
        let use_u64 = self.bookkeeping.first_round;
        
        for i in 0..half_size {
            // Get val values (either from u64 or Scalar storage)
            let (val_0, val_delta) = if use_u64 {
                let v0 = self.bookkeeping.val_u64[2 * i];
                let v1 = self.bookkeeping.val_u64[2 * i + 1];
                // Convert to Scalar for field operations
                let val_0 = Scalar::from_u64(v0);
                let val_1 = Scalar::from_u64(v1);
                (val_0, val_1 - val_0)
            } else {
                let val_0 = self.bookkeeping.val_evals[2 * i];
                let val_1 = self.bookkeeping.val_evals[2 * i + 1];
                (val_0, val_1 - val_0)
            };
            
            let base_even = (2 * i) * s;
            let base_odd = (2 * i + 1) * s;
            
            // Use unrolled helper function for all inner operations
            process_entry_pair_unrolled(
                &self.bookkeeping.eq_rx_flat[base_even..base_even + 20],
                &self.bookkeeping.eq_rx_flat[base_odd..base_odd + 20],
                &self.bookkeeping.eq_ry_flat[base_even..base_even + 20],
                &self.bookkeeping.eq_ry_flat[base_odd..base_odd + 20],
                val_0,
                val_delta,
                &mut h_coeffs,
            );
        }
        
        poly_eval_at_integers(&h_coeffs[..num_evals.min(h_coeffs.len())], num_evals)
    }

    /// Prove one round using PARALLEL processing (multi-threaded)
    /// 
    /// Uses rayon to parallelize over entry pairs
    pub fn prove_round_parallel(&mut self) -> Vec<Scalar> {
        let s = self.setup.s;
        assert_eq!(s, 20, "Parallel version requires s=20");
        
        let degree = self.degree();
        let num_evals = degree + 1;
        let half_size = self.bookkeeping.size / 2;
        let use_u64 = self.bookkeeping.first_round;
        
        // Parallel map-reduce: each thread computes partial h_coeffs, then sum
        let h_coeffs: Vec<Scalar> = (0..half_size)
            .into_par_iter()
            .map(|i| {
                let (val_0, val_delta) = if use_u64 {
                    let v0 = self.bookkeeping.val_u64[2 * i];
                    let v1 = self.bookkeeping.val_u64[2 * i + 1];
                    let val_0 = Scalar::from_u64(v0);
                    let val_1 = Scalar::from_u64(v1);
                    (val_0, val_1 - val_0)
                } else {
                    let val_0 = self.bookkeeping.val_evals[2 * i];
                    let val_1 = self.bookkeeping.val_evals[2 * i + 1];
                    (val_0, val_1 - val_0)
                };
                
                let base_even = (2 * i) * s;
                let base_odd = (2 * i + 1) * s;
                
                // Compute contribution for this entry pair
                let mut local_h = vec![zero(); degree + 2];
                process_entry_pair_unrolled(
                    &self.bookkeeping.eq_rx_flat[base_even..base_even + 20],
                    &self.bookkeeping.eq_rx_flat[base_odd..base_odd + 20],
                    &self.bookkeeping.eq_ry_flat[base_even..base_even + 20],
                    &self.bookkeeping.eq_ry_flat[base_odd..base_odd + 20],
                    val_0,
                    val_delta,
                    &mut local_h,
                );
                local_h
            })
            .reduce(
                || vec![zero(); degree + 2],
                |mut a, b| {
                    for (ai, bi) in a.iter_mut().zip(b.iter()) {
                        *ai += *bi;
                    }
                    a
                },
            );
        
        poly_eval_at_integers(&h_coeffs[..num_evals.min(h_coeffs.len())], num_evals)
    }

    /// Prove round 1 using pattern-based batching
    /// 
    /// Groups entry pairs by their (row, col) pattern and batches computation.
    /// Only effective in round 1 when we have access to original row/col indices.
    pub fn prove_round_batched(&mut self) -> Vec<Scalar> {
        let s = self.setup.s;
        let degree = self.degree();
        let num_evals = degree + 1;
        
        let mut h_coeffs = vec![zero(); degree + 2];
        let half_size = self.bookkeeping.size / 2;
        
        if !self.bookkeeping.first_round {
            // After round 1, fall back to regular method
            return self.prove_round_hardcoded_20();
        }
        
        // Pattern = (row_even, col_even, row_odd, col_odd)
        // Group entry pairs by their pattern
        // Value is (sum_val_0, sum_val_1) as i64 (since delta can be negative)
        let mut pattern_groups: HashMap<(usize, usize, usize, usize), (i64, i64)> = HashMap::new();
        
        for i in 0..half_size {
            let row_even = self.setup.row_indices[2 * i];
            let col_even = self.setup.col_indices[2 * i];
            let row_odd = self.setup.row_indices[2 * i + 1];
            let col_odd = self.setup.col_indices[2 * i + 1];
            
            let pattern = (row_even, col_even, row_odd, col_odd);
            let v0 = self.bookkeeping.val_u64[2 * i] as i64;
            let v1 = self.bookkeeping.val_u64[2 * i + 1] as i64;
            
            let entry = pattern_groups.entry(pattern).or_insert((0, 0));
            entry.0 += v0;
            entry.1 += v1;
        }
        
        let num_patterns = pattern_groups.len();
        let compression_ratio = half_size as f64 / num_patterns as f64;
        
        // If no significant compression, fall back to regular method
        if compression_ratio < 1.5 {
            return self.prove_round_hardcoded_20();
        }
        
        // Process each pattern group
        for ((row_even, col_even, row_odd, col_odd), (sum_v0, sum_v1)) in pattern_groups {
            // Convert summed values to Scalar
            let val_0 = if sum_v0 >= 0 {
                Scalar::from_u64(sum_v0 as u64)
            } else {
                -Scalar::from_u64((-sum_v0) as u64)
            };
            let val_1 = if sum_v1 >= 0 {
                Scalar::from_u64(sum_v1 as u64)
            } else {
                -Scalar::from_u64((-sum_v1) as u64)
            };
            let val_delta = val_1 - val_0;
            
            // Compute E_x and E_y for this pattern
            // Need to find an entry with this pattern to get eq values
            // Actually, we can compute directly from row/col indices and r_x/r_y
            // But we don't have r_x/r_y here... need to store them.
            
            // For now, use the first entry pair with this pattern (lookup its index)
            // This is O(1) since we have the eq values stored in bookkeeping
            
            // Find an entry pair with this pattern
            let mut found_idx = None;
            for i in 0..half_size {
                if self.setup.row_indices[2 * i] == row_even 
                   && self.setup.col_indices[2 * i] == col_even
                   && self.setup.row_indices[2 * i + 1] == row_odd
                   && self.setup.col_indices[2 * i + 1] == col_odd {
                    found_idx = Some(i);
                    break;
                }
            }
            
            let idx = found_idx.unwrap();
            let base_even = (2 * idx) * s;
            let base_odd = (2 * idx + 1) * s;
            
            // Use the helper function with aggregated val
            process_entry_pair_unrolled(
                &self.bookkeeping.eq_rx_flat[base_even..base_even + 20],
                &self.bookkeeping.eq_rx_flat[base_odd..base_odd + 20],
                &self.bookkeeping.eq_ry_flat[base_even..base_even + 20],
                &self.bookkeeping.eq_ry_flat[base_odd..base_odd + 20],
                val_0,
                val_delta,
                &mut h_coeffs,
            );
        }
        
        poly_eval_at_integers(&h_coeffs[..num_evals.min(h_coeffs.len())], num_evals)
    }

    pub fn receive_challenge(&mut self, r: Scalar) {
        self.challenges.push(r);
        self.bookkeeping.bind_variable(r);
        self.round += 1;
    }

    /// Prove all rounds using the HARDCODED version for s=20
    pub fn prove_all_hardcoded_20<R: ark_std::rand::RngCore>(&mut self, rng: &mut R) -> SumcheckProof {
        assert_eq!(self.setup.s, 20, "Hardcoded version requires s=20");
        
        let num_rounds = self.num_rounds();
        let mut round_polys = Vec::with_capacity(num_rounds);
        
        for _ in 0..num_rounds {
            let evals = self.prove_round_hardcoded_20();
            round_polys.push(evals);
            
            let r = Scalar::random(rng);
            self.receive_challenge(r);
        }
        
        let final_claim = self.bookkeeping.val_evals[0] 
            * self.bookkeeping.get_e_x(0)
            * self.bookkeeping.get_e_y(0);
        
        SumcheckProof {
            round_polys,
            final_claim,
            challenges: self.challenges.clone(),
        }
    }

    /// Prove all rounds using PARALLEL processing (multi-threaded)
    pub fn prove_all_parallel<R: ark_std::rand::RngCore>(&mut self, rng: &mut R) -> SumcheckProof {
        assert_eq!(self.setup.s, 20, "Parallel version requires s=20");
        
        let num_rounds = self.num_rounds();
        let mut round_polys = Vec::with_capacity(num_rounds);
        
        for _ in 0..num_rounds {
            let evals = self.prove_round_parallel();
            round_polys.push(evals);
            
            let r = Scalar::random(rng);
            self.receive_challenge(r);
        }
        
        let final_claim = self.bookkeeping.val_evals[0] 
            * self.bookkeeping.get_e_x(0)
            * self.bookkeeping.get_e_y(0);
        
        SumcheckProof {
            round_polys,
            final_claim,
            challenges: self.challenges.clone(),
        }
    }

    /// Prove all rounds using pattern-based batching (tries batching for round 1)
    pub fn prove_all_batched<R: ark_std::rand::RngCore>(&mut self, rng: &mut R) -> SumcheckProof {
        assert_eq!(self.setup.s, 20, "Batched version requires s=20");
        
        let num_rounds = self.num_rounds();
        let mut round_polys = Vec::with_capacity(num_rounds);
        
        // Round 1: try batching
        let evals = self.prove_round_batched();
        round_polys.push(evals);
        let r = Scalar::random(rng);
        self.receive_challenge(r);
        
        // Remaining rounds: use hardcoded version
        for _ in 1..num_rounds {
            let evals = self.prove_round_hardcoded_20();
            round_polys.push(evals);
            
            let r = Scalar::random(rng);
            self.receive_challenge(r);
        }
        
        let final_claim = self.bookkeeping.val_evals[0] 
            * self.bookkeeping.get_e_x(0)
            * self.bookkeeping.get_e_y(0);
        
        SumcheckProof {
            round_polys,
            final_claim,
            challenges: self.challenges.clone(),
        }
    }

    pub fn prove_all<R: ark_std::rand::RngCore>(&mut self, rng: &mut R) -> SumcheckProof {
        let num_rounds = self.num_rounds();
        let mut round_polys = Vec::with_capacity(num_rounds);
        
        for _ in 0..num_rounds {
            let evals = self.prove_round();
            round_polys.push(evals);
            
            let r = Scalar::random(rng);
            self.receive_challenge(r);
        }
        
        let final_claim = self.bookkeeping.val_evals[0] 
            * self.bookkeeping.get_e_x(0)
            * self.bookkeeping.get_e_y(0);
        
        SumcheckProof {
            round_polys,
            final_claim,
            challenges: self.challenges.clone(),
        }
    }
}

// ============================================================================
// VERIFIER
// ============================================================================

pub struct CinderSumcheckVerifier;

impl CinderSumcheckVerifier {
    pub fn verify(
        claimed_sum: Scalar,
        proof: &SumcheckProof,
        _degree: usize,
    ) -> (bool, Vec<Scalar>) {
        let mut expected = claimed_sum;
        
        for (round, poly_evals) in proof.round_polys.iter().enumerate() {
            if poly_evals.len() < 2 {
                return (false, vec![]);
            }
            
            let sum = poly_evals[0] + poly_evals[1];
            if sum != expected {
                return (false, vec![]);
            }
            
            let r = proof.challenges[round];
            expected = Self::evaluate_univariate(poly_evals, r);
        }
        
        if expected != proof.final_claim {
            return (false, vec![]);
        }
        
        (true, proof.challenges.clone())
    }

    fn evaluate_univariate(evals: &[Scalar], x: Scalar) -> Scalar {
        let n = evals.len();
        let mut result = zero();
        
        for (i, &yi) in evals.iter().enumerate() {
            let mut li = one();
            let xi = Scalar::from_u64(i as u64);
            
            for j in 0..n {
                if i != j {
                    let xj = Scalar::from_u64(j as u64);
                    li *= (x - xj) * CinderField::inverse(&(xi - xj)).unwrap();
                }
            }
            
            result += yi * li;
        }
        
        result
    }
}

// ============================================================================
// LEGACY API
// ============================================================================

pub struct LegacyCinderProver {
    setup: CinderSetup,
    r_x: Vec<Scalar>,
    r_y: Vec<Scalar>,
    matrix: CinderMatrix,
}

impl LegacyCinderProver {
    pub fn new(matrix: CinderMatrix, r_x: Vec<Scalar>, r_y: Vec<Scalar>) -> Self {
        let setup = CinderSetup::new(&matrix);
        Self { setup, r_x, r_y, matrix }
    }

    pub fn claimed_sum(&self) -> Scalar {
        self.matrix.evaluate_mle(&self.r_x, &self.r_y)
    }

    pub fn prove_all<R: ark_std::rand::RngCore>(&self, rng: &mut R) -> SumcheckProof {
        let mut prover = CinderSumcheckProver::new(&self.setup, self.r_x.clone(), self.r_y.clone());
        prover.prove_all(rng)
    }
    
    pub fn degree(&self) -> usize {
        self.setup.degree()
    }
}

#[cfg(test)]
mod sumcheck_tests {
    use super::*;
    use crate::sparse_matrix::SparseEntry;
    use ark_std::test_rng;

    #[test]
    fn test_poly_eval() {
        let coeffs = vec![Scalar::from_u64(1), Scalar::from_u64(2), Scalar::from_u64(3)];
        assert_eq!(poly_eval(&coeffs, zero()), Scalar::from_u64(1));
        assert_eq!(poly_eval(&coeffs, one()), Scalar::from_u64(6));
        assert_eq!(poly_eval(&coeffs, Scalar::from_u64(2)), Scalar::from_u64(17));
    }

    #[test]
    fn test_karatsuba() {
        // Test: (1 + 2X) * (3 + 4X) = 3 + 10X + 8X^2
        let a = vec![Scalar::from_u64(1), Scalar::from_u64(2)];
        let b = vec![Scalar::from_u64(3), Scalar::from_u64(4)];
        let result = karatsuba_mul(&a, &b);
        
        assert_eq!(result[0], Scalar::from_u64(3));
        assert_eq!(result[1], Scalar::from_u64(10));
        assert_eq!(result[2], Scalar::from_u64(8));
    }

    #[test]
    fn test_sumcheck_simple() {
        let mut rng = test_rng();
        
        let entries = vec![
            SparseEntry::new(0, 0, Scalar::from_u64(5)),
            SparseEntry::new(1, 2, Scalar::from_u64(7)),
        ];
        let matrix = CinderMatrix::new(4, entries);
        
        let r_x: Vec<Scalar> = (0..2).map(|_| Scalar::random(&mut rng)).collect();
        let r_y: Vec<Scalar> = (0..2).map(|_| Scalar::random(&mut rng)).collect();
        
        let expected = matrix.evaluate_mle(&r_x, &r_y);
        let setup = CinderSetup::new(&matrix);
        let mut prover = CinderSumcheckProver::new(&setup, r_x, r_y);
        let proof = prover.prove_all(&mut rng);
        
        let (verified, _) = CinderSumcheckVerifier::verify(expected, &proof, setup.degree());
        assert!(verified, "Sumcheck verification failed");
    }

    #[test]
    fn test_sumcheck_random() {
        let mut rng = test_rng();
        
        let matrix = CinderMatrix::random(16, 10, &mut rng);
        
        let r_x: Vec<Scalar> = (0..4).map(|_| Scalar::random(&mut rng)).collect();
        let r_y: Vec<Scalar> = (0..4).map(|_| Scalar::random(&mut rng)).collect();
        
        let expected = matrix.evaluate_mle(&r_x, &r_y);
        let setup = CinderSetup::new(&matrix);
        let mut prover = CinderSumcheckProver::new(&setup, r_x, r_y);
        let proof = prover.prove_all(&mut rng);
        
        let (verified, _) = CinderSumcheckVerifier::verify(expected, &proof, setup.degree());
        assert!(verified, "Sumcheck verification failed");
    }

    #[test]
    fn test_sumcheck_identity() {
        let mut rng = test_rng();
        
        let matrix = CinderMatrix::identity(8);
        
        let r_x: Vec<Scalar> = (0..3).map(|_| Scalar::random(&mut rng)).collect();
        let r_y: Vec<Scalar> = (0..3).map(|_| Scalar::random(&mut rng)).collect();
        
        let expected = matrix.evaluate_mle(&r_x, &r_y);
        let setup = CinderSetup::new(&matrix);
        let mut prover = CinderSumcheckProver::new(&setup, r_x, r_y);
        let proof = prover.prove_all(&mut rng);
        
        let (verified, _) = CinderSumcheckVerifier::verify(expected, &proof, setup.degree());
        assert!(verified, "Sumcheck verification failed");
    }

    #[test]
    fn test_sumcheck_larger() {
        let mut rng = test_rng();
        
        let matrix = CinderMatrix::random(256, 100, &mut rng);
        
        let r_x: Vec<Scalar> = (0..8).map(|_| Scalar::random(&mut rng)).collect();
        let r_y: Vec<Scalar> = (0..8).map(|_| Scalar::random(&mut rng)).collect();
        
        let expected = matrix.evaluate_mle(&r_x, &r_y);
        let setup = CinderSetup::new(&matrix);
        let mut prover = CinderSumcheckProver::new(&setup, r_x, r_y);
        let proof = prover.prove_all(&mut rng);
        
        let (verified, _) = CinderSumcheckVerifier::verify(expected, &proof, setup.degree());
        assert!(verified, "Sumcheck verification failed");
    }

    #[test]
    fn test_legacy_api() {
        let mut rng = test_rng();
        
        let entries = vec![
            SparseEntry::new(0, 0, Scalar::from_u64(5)),
            SparseEntry::new(1, 2, Scalar::from_u64(7)),
        ];
        let matrix = CinderMatrix::new(4, entries);
        
        let r_x: Vec<Scalar> = (0..2).map(|_| Scalar::random(&mut rng)).collect();
        let r_y: Vec<Scalar> = (0..2).map(|_| Scalar::random(&mut rng)).collect();
        
        let prover = LegacyCinderProver::new(matrix, r_x, r_y);
        let claimed_sum = prover.claimed_sum();
        let proof = prover.prove_all(&mut rng);
        
        let (verified, _) = CinderSumcheckVerifier::verify(claimed_sum, &proof, prover.degree());
        assert!(verified);
    }
}
