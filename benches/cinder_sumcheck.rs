//! Benchmarks for Cinder sumcheck
//!
//! This benchmarks:
//! 1. Setup time (MLE construction from sparse matrix)
//! 2. Prover time (sumcheck proving, excluding setup)

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use cinder::field::{Scalar, CinderField};
use cinder::sparse_matrix::CinderMatrix;
use cinder::sumcheck::{CinderSetup, CinderSumcheckProver};
use ark_std::test_rng;

/// Benchmark the setup phase (MLE construction)
fn bench_setup(c: &mut Criterion) {
    let mut group = c.benchmark_group("cinder_setup");
    
    // Test various (n, m) combinations
    // n = number of non-zero entries
    // m = matrix dimension (m x m)
    let configs = [
        // (log_n, log_m)
        (10, 8),   // n=1024, m=256
        (12, 10),  // n=4096, m=1024
        (14, 12),  // n=16384, m=4096
        (16, 14),  // n=65536, m=16384
        (18, 16),  // n=262144, m=65536
        (20, 18),  // n=1M, m=262K
    ];
    
    for (log_n, log_m) in configs {
        let n = 1 << log_n;
        let m = 1 << log_m;
        
        let mut rng = test_rng();
        let matrix = CinderMatrix::random(m, n, &mut rng);
        
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("n=2^{},m=2^{}", log_n, log_m), n),
            &matrix,
            |b, matrix| {
                b.iter(|| {
                    CinderSetup::new(matrix)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark the prover phase (sumcheck only, excluding setup)
fn bench_prover(c: &mut Criterion) {
    let mut group = c.benchmark_group("cinder_prover");
    group.sample_size(10); // Reduce samples for larger tests
    
    // Test various (n, m) combinations
    let configs = [
        // (log_n, log_m)
        (10, 8),   // n=1024, m=256
        (12, 10),  // n=4096, m=1024
        (14, 12),  // n=16384, m=4096
        (16, 14),  // n=65536, m=16384
        (18, 16),  // n=262144, m=65536
    ];
    
    for (log_n, log_m) in configs {
        let n = 1 << log_n;
        let m = 1 << log_m;
        let s = log_m;
        
        let mut rng = test_rng();
        let matrix = CinderMatrix::random(m, n, &mut rng);
        
        // Pre-compute setup (not part of prover benchmark)
        let setup = CinderSetup::new(&matrix);
        
        // Generate random challenge points
        let r_x: Vec<Scalar> = (0..s).map(|_| Scalar::random(&mut rng)).collect();
        let r_y: Vec<Scalar> = (0..s).map(|_| Scalar::random(&mut rng)).collect();
        
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("n=2^{},m=2^{},s={}", log_n, log_m, s), n),
            &(&setup, &r_x, &r_y),
            |b, (setup, r_x, r_y)| {
                b.iter(|| {
                    let mut rng = test_rng();
                    let mut prover = CinderSumcheckProver::new(setup, (*r_x).clone(), (*r_y).clone());
                    prover.prove_all(&mut rng)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark prover time for fixed n, varying m (to see effect of degree s = log m)
fn bench_prover_varying_m(c: &mut Criterion) {
    let mut group = c.benchmark_group("cinder_prover_vary_m");
    group.sample_size(10);
    
    let log_n = 16; // Fixed n = 65536
    let n = 1 << log_n;
    
    for log_m in [10, 12, 14, 16, 18, 20] {
        let m = 1 << log_m;
        let s = log_m;
        
        let mut rng = test_rng();
        let matrix = CinderMatrix::random(m, n, &mut rng);
        
        let setup = CinderSetup::new(&matrix);
        
        let r_x: Vec<Scalar> = (0..s).map(|_| Scalar::random(&mut rng)).collect();
        let r_y: Vec<Scalar> = (0..s).map(|_| Scalar::random(&mut rng)).collect();
        
        group.bench_with_input(
            BenchmarkId::new(format!("s={}", s), s),
            &(&setup, &r_x, &r_y),
            |b, (setup, r_x, r_y)| {
                b.iter(|| {
                    let mut rng = test_rng();
                    let mut prover = CinderSumcheckProver::new(setup, (*r_x).clone(), (*r_y).clone());
                    prover.prove_all(&mut rng)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark prover time for fixed m, varying n (to see linear scaling)
fn bench_prover_varying_n(c: &mut Criterion) {
    let mut group = c.benchmark_group("cinder_prover_vary_n");
    group.sample_size(10);
    
    let log_m = 14; // Fixed m = 16384
    let m = 1 << log_m;
    let s = log_m;
    
    for log_n in [10, 12, 14, 16, 18] {
        let n = 1 << log_n;
        
        let mut rng = test_rng();
        let matrix = CinderMatrix::random(m, n, &mut rng);
        
        let setup = CinderSetup::new(&matrix);
        
        let r_x: Vec<Scalar> = (0..s).map(|_| Scalar::random(&mut rng)).collect();
        let r_y: Vec<Scalar> = (0..s).map(|_| Scalar::random(&mut rng)).collect();
        
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("n=2^{}", log_n), n),
            &(&setup, &r_x, &r_y),
            |b, (setup, r_x, r_y)| {
                b.iter(|| {
                    let mut rng = test_rng();
                    let mut prover = CinderSumcheckProver::new(setup, (*r_x).clone(), (*r_y).clone());
                    prover.prove_all(&mut rng)
                });
            },
        );
    }
    
    group.finish();
}

/// Detailed breakdown: benchmark each round separately
fn bench_prover_per_round(c: &mut Criterion) {
    let mut group = c.benchmark_group("cinder_prover_per_round");
    
    let log_n = 14;
    let log_m = 10;
    let n = 1 << log_n;
    let m = 1 << log_m;
    let s = log_m;
    
    let mut rng = test_rng();
    let matrix = CinderMatrix::random(m, n, &mut rng);
    let setup = CinderSetup::new(&matrix);
    
    let r_x: Vec<Scalar> = (0..s).map(|_| Scalar::random(&mut rng)).collect();
    let r_y: Vec<Scalar> = (0..s).map(|_| Scalar::random(&mut rng)).collect();
    
    // Benchmark a single round (first round is the most expensive)
    group.bench_function("first_round", |b| {
        b.iter(|| {
            let mut prover = CinderSumcheckProver::new(&setup, r_x.clone(), r_y.clone());
            prover.prove_round()
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_setup,
    bench_prover,
    bench_prover_varying_m,
    bench_prover_varying_n,
    bench_prover_per_round,
);
criterion_main!(benches);

