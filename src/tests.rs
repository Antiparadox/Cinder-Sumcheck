//! Integration tests for the Cinder sumcheck

use crate::field::{Scalar, CinderField};
use crate::sparse_matrix::{CinderMatrix, SparseEntry};
use crate::sumcheck::{CinderSetup, CinderSumcheckProver, CinderSumcheckVerifier};
use ark_std::test_rng;

#[test]
fn test_end_to_end_small() {
    let mut rng = test_rng();
    
    // 4x4 matrix, 3 non-zero entries
    let entries = vec![
        SparseEntry::new(0, 0, Scalar::from_u64(1)),
        SparseEntry::new(1, 1, Scalar::from_u64(2)),
        SparseEntry::new(2, 3, Scalar::from_u64(3)),
    ];
    let matrix = CinderMatrix::new(4, entries);
    
    let r_x: Vec<Scalar> = (0..2).map(|_| Scalar::random(&mut rng)).collect();
    let r_y: Vec<Scalar> = (0..2).map(|_| Scalar::random(&mut rng)).collect();
    
    let expected = matrix.evaluate_mle(&r_x, &r_y);
    
    // Setup phase (preprocessing)
    let setup = CinderSetup::new(&matrix);
    
    // Prover phase
    let mut prover = CinderSumcheckProver::new(&setup, r_x, r_y);
    let proof = prover.prove_all(&mut rng);
    
    // Verify
    let (verified, _) = CinderSumcheckVerifier::verify(expected, &proof, setup.degree());
    
    assert!(verified, "Verification failed");
}

#[test]
fn test_end_to_end_medium() {
    let mut rng = test_rng();
    
    // 64x64 matrix, 50 non-zero entries
    let matrix = CinderMatrix::random(64, 50, &mut rng);
    
    let r_x: Vec<Scalar> = (0..6).map(|_| Scalar::random(&mut rng)).collect();
    let r_y: Vec<Scalar> = (0..6).map(|_| Scalar::random(&mut rng)).collect();
    
    let expected = matrix.evaluate_mle(&r_x, &r_y);
    
    // Setup phase
    let setup = CinderSetup::new(&matrix);
    
    // Prover phase
    let mut prover = CinderSumcheckProver::new(&setup, r_x, r_y);
    let proof = prover.prove_all(&mut rng);
    
    // Verify
    let (verified, _) = CinderSumcheckVerifier::verify(expected, &proof, setup.degree());
    
    assert!(verified, "Verification failed");
}

#[test]
fn test_varying_sizes() {
    let mut rng = test_rng();
    
    for log_m in 2..=6 {
        let m = 1 << log_m;
        let n = m / 2; // Half-full sparse matrix
        
        let matrix = CinderMatrix::random(m, n, &mut rng);
        
        let r_x: Vec<Scalar> = (0..log_m).map(|_| Scalar::random(&mut rng)).collect();
        let r_y: Vec<Scalar> = (0..log_m).map(|_| Scalar::random(&mut rng)).collect();
        
        let expected = matrix.evaluate_mle(&r_x, &r_y);
        
        // Setup phase
        let setup = CinderSetup::new(&matrix);
        
        // Prover phase
        let mut prover = CinderSumcheckProver::new(&setup, r_x, r_y);
        let proof = prover.prove_all(&mut rng);
        
        // Verify
        let (verified, _) = CinderSumcheckVerifier::verify(expected, &proof, setup.degree());
        
        assert!(verified, "Verification failed for m={}", m);
    }
}
