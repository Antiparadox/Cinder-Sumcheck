# Cinder Sumcheck Implementation

Implementation of the core **sumcheck prover** for the [Cinder](https://alinush.github.io/cinder) sparse MLE compiler by Alin Tomescu.

Cinder is a simple dense-to-sparse MLE compiler, suitable for circuits with ≤ 2²⁰ non-zero entries.

---

## What This Implements

We implement **only the sumcheck component** of Cinder — the most computationally expensive part.

### The Sumcheck Statement

Given a sparse matrix with `n` non-zero entries and random evaluation points `(rₓ, rᵧ)`, we prove:

```
Ṽ(rₓ, rᵧ) = Σ_{k ∈ [n]} val(k) · Eₓ(k) · Eᵧ(k)
```

Where (following [Cinder notation](https://alinush.github.io/cinder)):
- `n` = number of non-zero entries in the sparse matrix
- `m = 2ˢ` = matrix dimension  
- `s = log(m)` = number of bits for row/column indices
- `val(k)` = value of k-th non-zero entry (MLE over `log(n)` variables)
- `rowₜ(k)`, `colₜ(k)` = t-th bit of row/column index for entry k
- `Eₓ(k) = eq_{rₓ}(row₀(k), ..., rowₛ₋₁(k))` — equality polynomial over row bits
- `Eᵧ(k) = eq_{rᵧ}(col₀(k), ..., colₛ₋₁(k))` — equality polynomial over col bits

### Why It's Expensive

Each round of sumcheck requires computing a **degree-(2s+1)** univariate polynomial. Since `Eₓ` and `Eᵧ` are each products of `s` linear terms:

```
h(X) = Σ_k val(k, X) · Eₓ(k, X) · Eᵧ(k, X)
```

has degree `1 + s + s = 2s + 1` in each variable. For `s = 20`, this is degree **41**.

---

## Our Prover Algorithm

### Per-Round Algorithm

For each of the `log(n)` rounds:

```
1. Initialize h_coeffs[0..42] = 0  (degree-42 polynomial coefficients)

2. For each entry pair (2i, 2i+1):
   a. Look up val[2i], val[2i+1] from bookkeeping table
      → val(X) = val[2i] + (val[2i+1] - val[2i]) · X  (degree-1)
   
   b. For each bit t ∈ [s]:
      - Look up rowₜ[2i], rowₜ[2i+1] 
      - Compute linear factor: (1-rₓ[t])·(1-rowₜ(X)) + rₓ[t]·rowₜ(X)
   
   c. Multiply all s linear factors → Eₓ(X), degree-s polynomial
   
   d. Similarly compute Eᵧ(X), degree-s polynomial
   
   e. Multiply Eₓ(X) · Eᵧ(X) → degree-2s polynomial (schoolbook O(s²))
   
   f. Multiply by val(X) → degree-(2s+1) polynomial
   
   g. Accumulate into h_coeffs

3. Send h(X) evaluations to verifier (or hash for Fiat-Shamir)

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

**Hardware:** Apple M2 Max (12 cores), 32 GB RAM

**Test matrix:** `m = 2²⁰ = 1,048,576`, `n = 3,151,183` non-zero entries, `s = 20`

| Threads | Prover Time | Throughput | Speedup |
|---------|-------------|------------|---------|
| 1 | 101.7s | 31 K entries/s | 1.0x |
| 4 | 36.5s | 86 K entries/s | 2.8x |
| 8 | 24.6s | 128 K entries/s | 4.1x |
| 12 | 23.2s | 136 K entries/s | 4.4x |

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
RAYON_NUM_THREADS=1 cargo test bench_real_world --release -- --ignored --nocapture
```

Multi-threaded (e.g., 8 threads):
```bash
RAYON_NUM_THREADS=8 cargo test bench_real_world --release -- --ignored --nocapture
```

### Run Unit Tests
```bash
cargo test --release
```

---

## References

1. [Cinder: A simple-but-not-so-efficient dense-to-sparse MLE compiler](https://alinush.github.io/cinder) — Alin Tomescu, June 2025
2. [Spartan: Efficient and general-purpose zkSNARKs without trusted setup](https://eprint.iacr.org/2019/550)
