# Cinder: A Simple Dense-to-Sparse MLE Compiler

## Overview

**Cinder** is a simple (but not-so-efficient) compiler that transforms a **dense MLE polynomial commitment scheme (PCS)** into a **sparse MLE PCS**. This is useful for systems like [Spartan](https://eprint.iacr.org/2019/550), which require sparse MLE commitments to efficiently handle R1CS constraint systems.

### Context: Why Sparse MLE PCS?

In zkSNARK systems like Spartan, we need to commit to sparse R1CS matrices. A typical R1CS instance has:
- Matrix size: `m × m` (so `m²` entries)
- Non-zero entries: only `n ≈ m` (very sparse!)

Committing to the full `m²`-sized matrix as a dense polynomial is wasteful. We want a PCS that only pays for the `n` non-zero entries.

### Spark vs Cinder

The Spartan paper introduced **Spark**, an efficient compiler that:
- Takes a dense MLE PCS for size-`n` MLEs
- Produces a sparse MLE PCS for size-`m²` MLEs with `n ≈ m` non-zero entries

**Cinder** is a simpler alternative to Spark:
- Easier to implement
- Slightly more expensive (proof size and verifier time)
- Good enough for moderately-sized circuits (`n ≤ 2²³`)

---

## Preliminaries

### Notation

| Symbol | Meaning |
|--------|---------|
| `[s]` | The set `{0, 1, ..., s-1}` |
| `{0,1}^s` | Boolean hypercube of dimension `s` |
| `m = 2^s` | Matrix dimension |
| `n` | Number of non-zero entries in the sparse matrix |
| `𝔽` | Finite field |
| `eq_i(X)` | Equality polynomial: `∏_j (X_j · i_j + (1-X_j)(1-i_j))` |

### MLE of a Sparse Matrix

Given a sparse R1CS matrix `V = (V_{i,j})` with `n` non-zero entries, its **multilinear extension (MLE)** is:

```
Ṽ(X, Y) = Σ_{i,j ∈ {0,1}^s} V_{i,j} · eq_i(X) · eq_j(Y)
```

**Goal:** Efficiently commit to `Ṽ` and open it at a random point `(r_x, r_y)`.

---

## The Cinder Approach

### Key Insight: Bit Decomposition

Instead of representing row/column indices as single values, Cinder decomposes them into **individual bits**.

For the `n` non-zero entries, define:
- `(i_k, j_k, V_{i_k,j_k})` for `k ∈ [n]` — the row index, column index, and value of the k-th non-zero entry

Cinder commits to **`2s + 1` MLEs**:

```
For all k ∈ [n]:
  row_t(k) = bit_t(i_k),  for t ∈ [s]     (s row-bit MLEs)
  col_t(k) = bit_t(j_k),  for t ∈ [s]     (s col-bit MLEs)
  val(k)   = V_{i_k,j_k}                  (1 value MLE)
```

where `bit_t(b)` is the t-th bit of b's binary representation.

### Reconstructing Row/Column Indices

The full row index can be reconstructed as:

```
row(k) = Σ_{t ∈ [s]} 2^t · row_t(k)
```

### Rewriting the Sumcheck

Define helper polynomials:

```
E_x(Z) := eq_{r_x}(row_0(Z), ..., row_{s-1}(Z))
E_y(Z) := eq_{r_y}(col_0(Z), ..., col_{s-1}(Z))
```

The matrix MLE evaluation becomes:

```
Ṽ(r_x, r_y) = Σ_{k ∈ [n]} val(k) · E_x(k) · E_y(k)
```

This is now a **sumcheck** over `log(n)` variables!

---

## Cinder Algorithms

### 1. Setup

**`Cinder_D.Setup(n, m) → (ck, ok)`**

```
Input: n (number of non-zero entries), m (matrix dimension)
Output: commitment key (ck), opening key (ok)

1. (ck_D, ok_D) ← D.Setup(n)           // Setup underlying dense PCS
2. s ← log(m)                          // Number of bits for indices
3. ck ← (ck_D, s, n)
4. ok ← (ok_D, s, n)
5. return (ck, ok)
```

### 2. Commit

**`Cinder_D.Commit(ck, Ṽ) → (c, aux)`**

```
Input: commitment key (ck), sparse matrix MLE (Ṽ)
Output: commitment (c), auxiliary data (aux)

1. Parse (ck_D, s, ·) ← ck
2. For t ∈ [s]:
   - c_row,t ← D.Commit(ck_D, row_t)    // Commit to each row bit MLE
   - c_col,t ← D.Commit(ck_D, col_t)    // Commit to each col bit MLE
3. c_val ← D.Commit(ck_D, val)          // Commit to value MLE
4. c ← ((c_row,t, c_col,t)_{t∈[s]}, c_val)
5. aux ← ((row_t(Z), col_t(Z))_{t∈[s]}, val(Z))
6. return (c, aux)
```

**Note:** This produces `2s + 1` dense MLE commitments.

### 3. Open (Prover)

**`Cinder_D.Open(ck, aux, c, Ṽ, (r_x, r_y)) → (v, π)`**

```
Input: ck, aux, commitment c, matrix Ṽ, evaluation point (r_x, r_y)
Output: claimed value v, proof π

// Phase 1: Run Sumcheck
1. Parse (·, s, n) ← ck
2. v ← Ṽ(r_x, r_y)                     // Claimed evaluation
3. Define the sumcheck polynomial:
   F(Z) = val(Z) · E_x(Z) · E_y(Z)
4. (e_Σ, π_Σ; r_k) ← SumCheck.Prove(F, v, log(n), s)
   // e_Σ: final evaluation claim
   // π_Σ: sumcheck proof
   // r_k: random point from Fiat-Shamir

// Phase 2: Open Dense Polynomials at r_k
5. (ρ_t, γ_t)_{t∈[s]} ← FS ← F^{2s}   // Sample batching coefficients
6. P(Z) ← val(Z) + Σ_{t∈[s]} ρ_t · row_t(Z) + γ_t · col_t(Z)
7. c_P ← c_val + Σ_{t∈[s]} ρ_t · c_row,t + γ_t · c_col,t
   // Note: Requires homomorphic commitments!
8. (·, π_P) ← D.Open(ck_D, c_P, P, r_k)

// Phase 3: Assemble Proof
9. v_val ← val(r_k)
10. v_row,t ← row_t(r_k), ∀t ∈ [s]
11. v_col,t ← col_t(r_k), ∀t ∈ [s]
12. π ← (((v_row,t, v_col,t)_{t∈[s]}, v_val), π_P, e_Σ, π_Σ)
13. return (v, π)
```

### 4. Verify

**`Cinder_D.Verify(ok, c, v, (r_x, r_y); π) → {0, 1}`**

```
Input: opening key ok, commitment c, claimed value v, point (r_x, r_y), proof π
Output: accept (1) or reject (0)

// Phase 1: Verify Sumcheck
1. Parse (·, s, ·) ← ok
2. Add c to the FS transcript
3. Parse (·, ·, e_Σ, π_Σ) ← π
4. (b; r_k) ← SumCheck.Verify(v, e_Σ, s; π_Σ)
5. assert b = 1

// Phase 2: Verify Evaluation Claim
6. Parse (((v_row,t, v_col,t)_{t∈[s]}, v_val), ·, ·, ·) ← π
7. Reconstruct E_x(r_k) and E_y(r_k):
   E_x ← eq_{r_x}(v_row,0, ..., v_row,s-1)
   E_y ← eq_{r_y}(v_col,0, ..., v_col,s-1)
8. assert e_Σ = v_val · E_x · E_y

// Phase 3: Verify Dense MLE Opening
9. Parse ((c_row,t, c_col,t)_{t∈[s]}, c_val) ← c
10. (ρ_t, γ_t)_{t∈[s]} ← FS ← F^{2s}
11. c_P ← c_val + Σ_{t∈[s]} ρ_t · c_row,t + γ_t · c_col,t
12. v_P ← v_val + Σ_{t∈[s]} ρ_t · v_row,t + γ_t · v_col,t
13. Parse (ok_D, ·, ·) ← ok
14. Parse (·, π_P, ·, ·) ← π
15. assert D.Verify(ok_D, c_P, v_P, r_k; π_P)

// Success
16. return 1
```

---

## Proof Structure

The Cinder proof `π` consists of:

| Component | Size | Description |
|-----------|------|-------------|
| `e_Σ` | 32 bytes | Sumcheck evaluation claim |
| `π_Σ` | `(log(m) + 1) · log(n) · |𝔽|` | Sumcheck proof |
| `(v_row,t, v_col,t)` | `2s · |𝔽|` | Row/col bit evaluations |
| `v_val` | `|𝔽|` | Value evaluation |
| `π_P` | Depends on D | Dense MLE PCS opening proof |

### Example Size Calculation

For `n = 2²³` (8M non-zero entries), `m = 2²⁰` (1M × 1M matrix), `|𝔽| = 32` bytes:

- Sumcheck proof: `(20 + 1) × 23 × 32 = 15,456` bytes ≈ **15.1 KiB**
- Evaluations: `(2 × 20 + 1) × 32 = 1,312` bytes ≈ **1.3 KiB**
- Plus dense PCS opening proof

---

## Verifier Time Complexity

1. **Sumcheck verification**: `O(log(n) · log(m))` — verify degree-`log(m)` polynomial over `log(n)` variables
2. **Compute e_Σ**: `O(log(m))` — evaluate `eq_{r_x}` and `eq_{r_y}`
3. **Fiat-Shamir hashing**: `O(s)` — derive `ρ_t, γ_t` coefficients
4. **Derive c_P**: `O(s)` — MSM of size `2·log(m)` (assuming group commitments)
5. **Derive v_P**: `O(s)` — `2·log(m)` field multiplications
6. **Dense PCS verification**: Depends on underlying D

---

## Inefficiencies Compared to Spark

### 1. Higher-Degree Sumcheck

The polynomial `E_x(Z)` has:
- `log(n)` variables
- Degree `log(m) = s` in each variable

This results in:
- **Prover time**: `O(s·n) = O(n·log(m))`
- **Proof size**: `O(s·log(n)) = O(log(n)·log(m))`

### 2. More Polynomial Evaluations

Cinder requires `2s + 1` evaluations to be included in the proof, whereas Spark uses a more compact representation.

### 3. Batched Evaluation Proof

Computing the batched evaluation requires combining `2s + 1` MLEs, each of size `n`:
- Work: `O(n·log(m))`

---

## When to Use Cinder

**Cinder is a good choice when:**
- Simplicity is more important than optimal efficiency
- Circuit size is moderate (`n ≤ 2²³`)
- You already have a dense MLE PCS implementation
- You want a transparent setup (if using a transparent dense PCS)

**Consider Spark instead when:**
- You need optimal asymptotic efficiency
- Working with very large circuits
- Proof size is critical

---

## Implementation Notes

### Requirements for Dense PCS D

Cinder requires the underlying dense MLE PCS to support:
1. **Homomorphic commitments**: `c_P = c_val + Σ ρ_t · c_row,t + γ_t · c_col,t`
2. **Batched openings**: Open a linear combination of polynomials

Examples of suitable dense PCS:
- **KZG** (requires trusted setup)
- **Hyrax** (transparent, based on Pedersen)
- **PST** (polynomial commitment from pairing)

### Sumcheck Implementation

The sumcheck in Cinder is over a polynomial of the form:

```
F(Z) = val(Z) · E_x(Z) · E_y(Z)
```

where `E_x` and `E_y` are products of `s` linear terms each. This means:
- `F` has degree `1 + s + s = 2s + 1` in each variable (after expansion)
- Actually, since `eq` is multilinear, the degree is `s` in each variable

The prover needs to evaluate `F` at `s + 1` points per round.

---

## Quick Start

### Prerequisites
- Rust (1.75+): `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`

### Run Real-World Benchmark (m=1M, nnz=3.15M)

Single-threaded:
```bash
cargo test bench_real_world --release -- --ignored --nocapture
```

Multi-threaded scaling (1, 2, 4, 8, 12, 16 threads):
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

1. [Spartan: Efficient and general-purpose zkSNARKs without trusted setup](https://eprint.iacr.org/2019/550)
2. [HyperPlonk: Plonk with Linear-Time Prover](https://eprint.iacr.org/2022/1355)
3. Cinder blog post by Alin Tomescu (June 2025)
