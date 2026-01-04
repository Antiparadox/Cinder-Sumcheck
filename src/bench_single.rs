//! Benchmark for real-world R1CS matrix
//!
//! Target matrix:
//! - Rows (constraints m): 1,040,083
//! - Cols (variables n): 1,016,724  
//! - Non-zero entries: 3,151,183

use crate::field::{Scalar, CinderField};
use crate::sparse_matrix::CinderMatrix;
use crate::sumcheck::{CinderSetup, CinderSumcheckProver, CinderSumcheckVerifier};
use std::time::Instant;
use rayon;

/// Benchmark helper - runs single-threaded
pub fn run_real_world_benchmark() {
    println!("=== Cinder Single-Threaded Benchmark ===\n");
    
    // Real-world R1CS matrix dimensions
    let rows: usize = 1_040_083;
    let cols: usize = 1_016_724;
    let nnz: usize = 3_151_183;
    
    // For Cinder, m = max(rows, cols) rounded up to power of 2
    let m_raw = rows.max(cols);
    let log_m = (m_raw as f64).log2().ceil() as usize;
    let m = 1 << log_m;
    
    println!("Real-world R1CS matrix:");
    println!("  Rows (constraints): {:>12}", rows);
    println!("  Cols (variables):   {:>12}", cols);
    println!("  Non-zero entries:   {:>12}", nnz);
    println!();
    println!("Cinder parameters:");
    println!("  m (padded dim):     {:>12} = 2^{}", m, log_m);
    println!("  s = log(m):         {:>12}", log_m);
    println!("  n (entries):        {:>12}", nnz);
    println!("  Degree (2s+1):      {:>12}", 2 * log_m + 1);
    println!();
    
    // Generate random matrix with SMALL (16-bit) values
    println!("Generating random matrix with {} non-zero entries (16-bit values)...", nnz);
    let start = Instant::now();
    
    let mut rng = ark_std::test_rng();
    let matrix = CinderMatrix::random_small(m, nnz, &mut rng);
    
    let gen_time = start.elapsed();
    println!("  Matrix generation: {:.2?}", gen_time);
    
    // Setup phase
    println!("\n--- Setup Phase (MLE Construction) ---");
    let start = Instant::now();
    let setup = CinderSetup::new(&matrix);
    let setup_time = start.elapsed();
    
    println!("  Setup time: {:.2?}", setup_time);
    println!("  Throughput: {:.2} M entries/sec", 
             nnz as f64 / setup_time.as_secs_f64() / 1_000_000.0);
    
    // Generate random challenge points
    let r_x: Vec<Scalar> = (0..log_m).map(|_| Scalar::random(&mut rng)).collect();
    let r_y: Vec<Scalar> = (0..log_m).map(|_| Scalar::random(&mut rng)).collect();
    
    // Prover phase
    let num_threads = rayon::current_num_threads();
    println!("\n--- Prover Phase ---");
    println!("  Number of rounds: {}", setup.num_vars);
    println!("  Degree per round: {}", setup.degree());
    println!("  Threads (RAYON_NUM_THREADS): {}", num_threads);
    
    let start = Instant::now();
    let mut prover = CinderSumcheckProver::new(&setup, r_x.clone(), r_y.clone());
    let proof = if num_threads == 1 {
        // Use single-threaded hardcoded version
        prover.prove_all_hardcoded_20(&mut rng)
    } else {
        // Use parallel version
        prover.prove_all_parallel(&mut rng)
    };
    let prover_time = start.elapsed();
    
    println!("  Prover time: {:.2?}", prover_time);
    println!("  Throughput: {:.2} K entries/sec",
             nnz as f64 / prover_time.as_secs_f64() / 1_000.0);
    
    // Verify
    println!("\n--- Verification ---");
    let expected = matrix.evaluate_mle(&r_x, &r_y);
    let start = Instant::now();
    let (verified, _) = CinderSumcheckVerifier::verify(expected, &proof, setup.degree());
    let verify_time = start.elapsed();
    
    println!("  Verification: {}", if verified { "PASSED" } else { "FAILED" });
    println!("  Verify time: {:.2?}", verify_time);
    
    // Proof size
    let proof_elements = proof.round_polys.iter().map(|p| p.len()).sum::<usize>() + 1;
    let proof_bytes = proof_elements * 32; // 32 bytes per BLS12-381 scalar
    println!("\n--- Proof Statistics ---");
    println!("  Rounds: {}", proof.round_polys.len());
    println!("  Proof elements: {}", proof_elements);
    println!("  Proof size: {} bytes ({:.2} KB)", proof_bytes, proof_bytes as f64 / 1024.0);
    
    // Summary
    println!("\n=== Summary ===");
    println!("  Matrix:    {} x {} with {} non-zeros", m, m, nnz);
    println!("  Setup:     {:.2?}", setup_time);
    println!("  Prover:    {:.2?}", prover_time);
    println!("  Verifier:  {:.2?}", verify_time);
    println!("  Total:     {:.2?}", setup_time + prover_time + verify_time);
}

/// Benchmark with different thread counts
pub fn run_scaling_benchmark() {
    println!("=== Cinder Multi-Threaded Scaling Benchmark ===\n");
    
    // Real-world R1CS matrix dimensions
    let rows: usize = 1_040_083;
    let cols: usize = 1_016_724;
    let nnz: usize = 3_151_183;
    
    let m_raw = rows.max(cols);
    let log_m = (m_raw as f64).log2().ceil() as usize;
    let m = 1 << log_m;
    
    println!("Matrix: {} x {} with {} non-zeros", m, m, nnz);
    println!("s = {}, degree = {}\n", log_m, 2 * log_m + 1);
    
    // Generate matrix once
    let mut rng = ark_std::test_rng();
    let matrix = CinderMatrix::random_small(m, nnz, &mut rng);
    let setup = CinderSetup::new(&matrix);
    
    let r_x: Vec<Scalar> = (0..log_m).map(|_| Scalar::random(&mut rng)).collect();
    let r_y: Vec<Scalar> = (0..log_m).map(|_| Scalar::random(&mut rng)).collect();
    
    println!("{:>8} {:>12} {:>12} {:>10}", "Threads", "Time (s)", "Throughput", "Speedup");
    println!("{:-<46}", "");
    
    let thread_counts = [1, 2, 4, 8, 16];
    let mut baseline_time = 0.0f64;
    
    for &num_threads in &thread_counts {
        // Set thread count
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .ok(); // Ignore error if pool already built
        
        // Run benchmark
        let start = Instant::now();
        let mut prover = CinderSumcheckProver::new(&setup, r_x.clone(), r_y.clone());
        let proof = prover.prove_all_parallel(&mut rng);
        let elapsed = start.elapsed().as_secs_f64();
        
        // Verify
        let expected = matrix.evaluate_mle(&r_x, &r_y);
        let (verified, _) = CinderSumcheckVerifier::verify(expected, &proof, setup.degree());
        assert!(verified, "Verification failed for {} threads", num_threads);
        
        if num_threads == 1 {
            baseline_time = elapsed;
        }
        
        let throughput = nnz as f64 / elapsed / 1000.0;
        let speedup = baseline_time / elapsed;
        
        println!("{:>8} {:>12.2} {:>10.1} K/s {:>8.2}x", 
                 num_threads, elapsed, throughput, speedup);
    }
    
    println!("\n=== Done ===");
}

#[cfg(test)]
mod benchmark_tests {
    use super::*;

    #[test]
    #[ignore] // Run with: cargo test bench_real_world --release -- --ignored --nocapture
    fn bench_real_world() {
        run_real_world_benchmark();
    }
    
    #[test]
    #[ignore] // Run with: cargo test bench_scaling --release -- --ignored --nocapture
    fn bench_scaling() {
        run_scaling_benchmark();
    }
    
    #[test]
    fn bench_smaller_scale() {
        // 1/1000 scale test to verify correctness quickly
        println!("=== Smaller Scale Test (1/1000 of real-world, 16-bit values) ===\n");
        
        let nnz = 3_151; // 1/1000 of real
        let m = 1 << 11; // 2048
        
        let mut rng = ark_std::test_rng();
        let matrix = CinderMatrix::random_small(m, nnz, &mut rng);
        
        println!("Matrix: {} x {} with {} non-zeros", m, m, nnz);
        
        let start = Instant::now();
        let setup = CinderSetup::new(&matrix);
        let setup_time = start.elapsed();
        println!("Setup: {:.2?}", setup_time);
        
        let log_m = 11;
        let r_x: Vec<Scalar> = (0..log_m).map(|_| Scalar::random(&mut rng)).collect();
        let r_y: Vec<Scalar> = (0..log_m).map(|_| Scalar::random(&mut rng)).collect();
        
        let start = Instant::now();
        let mut prover = CinderSumcheckProver::new(&setup, r_x.clone(), r_y.clone());
        let proof = prover.prove_all(&mut rng);
        let prover_time = start.elapsed();
        println!("Prover: {:.2?}", prover_time);
        
        let expected = matrix.evaluate_mle(&r_x, &r_y);
        let (verified, _) = CinderSumcheckVerifier::verify(expected, &proof, setup.degree());
        assert!(verified, "Verification failed");
        println!("Verified: PASSED");
    }
}

