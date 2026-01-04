# Cinder Sumcheck Implementation

Implementation of the core **sumcheck prover** for the [Cinder](https://alinush.github.io/cinder) sparse MLE compiler.

Cinder is a simple dense-to-sparse MLE compiler described by Alin Tomescu. It's suitable for circuits with `n ≤ 2²⁰` non-zero entries.

---

## What This Implements

We implement **only the sumcheck component** of Cinder — the most computationally expensive part.

### The Sumcheck Statement

Given a sparse matrix with `n` non-zero entries and random evaluation points `(r_x, r_y)`, we prove:

```
Ṽ(r_x, r_y) = Σ_{k ∈ [n]} val(k) · E_x(k) · E_y(k)
```

Where:
- `val(k)` = value of k-th non-zero entry
- `E_x(k) = eq_{r_x}(row_0(k), ..., row_{s-1}(k))` — equality polynomial over row bits
- `E_y(k) = eq_{r_y}(col_0(k), ..., col_{s-1}(k))` — equality polynomial over col bits
- `s = log(m)` where `m` is the matrix dimension

### Why It's Expensive

Each round of sumcheck requires computing a **degree-(2s+1)** univariate polynomial. Since `E_x` and `E_y` are each products of `s` linear terms, the polynomial:

```
h(X) = Σ_k val(k, X) · E_x(k, X) · E_y(k, X)
```

has degree `1 + s + s = 2s + 1` in each variable. For `s = 20`, this is degree **41**.

---

## Our Prover Algorithm

### Per-Round Algorithm

For each round of sumcheck over `log(n)` rounds:

```
1. Initialize h_coeffs[0..42] = 0  (degree-42 polynomial coefficients)

2. For each entry pair (2i, 2i+1):
   a. Look up val[2i], val[2i+1] from bookkeeping table
      → val(X) = val[2i] + (val[2i+1] - val[2i]) · X  (degree-1)
   
   b. For each bit t ∈ [s]:
      - Look up row_t[2i], row_t[2i+1] 
      - Compute linear factor: (1-r_x[t])·(1-row_t(X)) + r_x[t]·row_t(X)
   
   c. Multiply all s linear factors → E_x(X), degree-s polynomial
   
   d. Similarly compute E_y(X), degree-s polynomial
   
   e. Multiply E_x(X) · E_y(X) → degree-2s polynomial (schoolbook O(s²))
   
   f. Multiply by val(X) → degree-(2s+1) polynomial
   
   g. Accumulate into h_coeffs

3. Send h_coeffs to verifier (or hash for Fiat-Shamir)

4. Receive challenge r, fold all tables:
   table[i] = table[2i] · (1-r) + table[2i+1] · r
```

### Complexity

- **Per round**: `O(n/2^round · s²)` — process n/2^round entry pairs, each costs O(s²)
- **Total**: `O(n · s²)` field operations

### Key Optimizations

| Optimization | Description |
|--------------|-------------|
| **Coefficient arithmetic** | Work with polynomial coefficients, not Lagrange evaluations |
| **Hardcoded s=20** | Unroll the degree-20 polynomial building loop |
| **Transposed memory** | Store eq tables in cache-friendly layout |
| **Multi-threading** | Parallelize entry pair processing with rayon |

---

## Benchmark Results

**Test matrix**: `m = 1,040,083` rows, `nnz = 3,151,183` non-zero entries, `s = 20`

| Threads | Prover Time | Throughput |
|---------|-------------|------------|
| 1 | 26.0s | 121 K entries/s |
| 2 | 13.5s | 233 K entries/s |
| 4 | 7.0s | 450 K entries/s |
| 8 | 4.4s | 716 K entries/s |
| 12 | 3.5s | 900 K entries/s |
| 16 | 3.1s | 1,016 K entries/s |

---

## Quick Start

### Prerequisites
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Run Benchmark

Single-threaded:
```bash
cargo test bench_real_world --release -- --ignored --nocapture
```

Multi-threaded scaling:
```bash
for t in 1 2 4 8 12 16; do
  echo "=== Threads: $t ==="
  RAYON_NUM_THREADS=$t cargo test bench_real_world --release -- --ignored --nocapture 2>&1 | grep -E "Prover time|Throughput"
done
```

### Run Unit Tests
```bash
cargo test --release
```

---

## References

1. [Cinder: A simple-but-not-so-efficient dense-to-sparse MLE compiler](https://alinush.github.io/cinder) — Alin Tomescu, June 2025
2. [Spartan: Efficient and general-purpose zkSNARKs without trusted setup](https://eprint.iacr.org/2019/550)
