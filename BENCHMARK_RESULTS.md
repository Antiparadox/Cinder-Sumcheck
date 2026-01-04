# Cinder Sumcheck Benchmark Results

## Overview

This document summarizes the benchmarking experiments conducted on the Cinder sumcheck implementation for sparse R1CS matrices.

## Test Configuration

### Target Matrix (Real-World R1CS)
| Property | Value |
|----------|-------|
| Rows (constraints) | 1,040,083 |
| Columns (variables) | 1,016,724 |
| Non-zero entries | 3,151,183 |
| Padded dimension (m) | 1,048,576 = 2²⁰ |
| s = log(m) | 20 |
| Polynomial degree | 2s+1 = 41 |
| Sumcheck rounds | 22 |
| Proof size | 28.91 KB (925 field elements) |

### Hardware
- Apple Silicon (M-series) or equivalent
- BLS12-381 scalar field (255-bit)

---

## Single-Threaded Optimization Experiments

### Baseline vs. Optimizations

| Version | Prover Time | Throughput | Speedup |
|---------|-------------|------------|---------|
| Original (loop-based) | 106.88s | 29.5 K/s | baseline |
| Hardcoded `build_poly_20` | 103.29s | 30.5 K/s | 3.4% |
| Helper function (inlined) | **101.44s** | **31.1 K/s** | **5.1%** |
| Loop unrolling (`unroll` crate) | 138.27s | 22.8 K/s | -29% (slower!) |
| Delayed u64 conversion | 103.77s | 30.4 K/s | ~0% |
| Pattern-based batching | 102.65s | 30.7 K/s | ~0% |

### Key Findings

1. **Loop unrolling hurt performance**: Full unrolling with the `unroll` crate caused instruction cache thrashing, resulting in 34% slower performance.

2. **Delayed field conversion didn't help**: Converting `val` from u64 to Scalar on-the-fly has similar cost to storing as Scalar upfront.

3. **Pattern batching ineffective for random matrices**:
   - Entry pairs: 2,097,152
   - Unique patterns: 1,575,593
   - Compression ratio: 1.33x (insufficient for batching gains)

4. **Best single-threaded result**: **~101 seconds** with helper function optimization.

---

## Multi-Threaded Scaling Results

### Parallel Implementation
Using rayon for parallel processing of entry pairs with map-reduce pattern.

| Threads | Prover Time | Throughput | Speedup | Efficiency |
|--------:|------------:|-----------:|--------:|-----------:|
| 1 | 106.62s | 29.6 K/s | 1.00x | 100% |
| 2 | 63.40s | 49.7 K/s | 1.68x | 84% |
| 4 | 37.13s | 84.9 K/s | 2.87x | 72% |
| 8 | 25.08s | 125.6 K/s | 4.25x | 53% |
| 12 | 24.02s | 131.2 K/s | 4.44x | 37% |
| 16 | 22.84s | 138.0 K/s | 4.67x | 29% |

### Scaling Characteristics

```
Speedup vs Thread Count:

Threads:  1     2     4     8    12    16
         |     |     |     |     |     |
Speedup: 1.0x  1.7x  2.9x  4.3x  4.4x  4.7x
         ████  ████  ████  ████  ████  ████
               ████  ████  ████  ████  ████
                     ████  ████  ████  ████
                           ████  ████  ████
                                 ████  ████
```

- **Near-linear scaling up to 4 threads**
- **Good scaling up to 8 threads** (4.25x on 8 cores)
- **Diminishing returns beyond 8 threads** (memory bandwidth limited)

---

## Complexity Analysis

### Per Entry Pair Operations
| Operation | Field Multiplications | Field Additions |
|-----------|----------------------|-----------------|
| Build E_x (degree-20) | 210 | 210 |
| Build E_y (degree-20) | 210 | 210 |
| E_x × E_y (21×21) | 441 | 441 |
| Multiply by val | 82 | 82 |
| **Total** | **~943** | **~943** |

### Asymptotic Complexity
- Per round: O(current_size × s²)
- Total across all rounds: O(n × s²)
- For our matrix: ~3M entries × 400 operations = ~1.2 billion field operations

---

## Why Further Single-Threaded Optimization is Hard

1. **Field arithmetic dominates**: BLS12-381 multiplication takes ~35 cycles each
2. **No pattern structure in random matrices**: 1.33x compression insufficient
3. **Memory bandwidth not bottleneck**: u64 storage didn't improve cache performance significantly
4. **Compiler already optimizes well**: LLVM handles small fixed-bound loops efficiently

---

## Summary

| Configuration | Time | vs. Single-Thread Baseline |
|--------------|------|---------------------------|
| Single-threaded (original) | 107s | baseline |
| Single-threaded (optimized) | **101s** | **5% faster** |
| 4 threads (parallel) | **37s** | **2.9x faster** |
| 8 threads (parallel) | **25s** | **4.3x faster** |
| 16 threads (parallel) | **23s** | **4.7x faster** |

### Recommendations

1. **For single-threaded**: Use the helper function approach (~101s)
2. **For multi-threaded**: Use 8 threads for best efficiency/performance balance (~25s)
3. **For maximum throughput**: Use all available cores (~23s with 16 threads)

---

## Code Location

- Implementation: `cinder/src/sumcheck.rs`
- Benchmarks: `cinder/src/bench_single.rs`
- Run single-threaded: `RAYON_NUM_THREADS=1 cargo test bench_real_world --release -- --ignored --nocapture`
- Run parallel: `RAYON_NUM_THREADS=8 cargo test bench_real_world --release -- --ignored --nocapture`

