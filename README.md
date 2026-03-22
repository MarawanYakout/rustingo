# About Rustingo

Why = "Well I really felt that i needed somthing faster than CUDA and that was it and also because i needed it for a bigger project"

This library is designed to be the fastest ML framework in existence. I used AI intensively and abusively to create it, and I verified it using my knowledge of AI and ML from my studies and research.

Rustingo is designed to leverage GPU power to perform ML and AI complex operations with extremely memory‑efficient and optimized timing.

Everything is designed to be memory‑safe down to the bit. I don’t expect it to be perfect—there may be many issues that arise over time—but for a start, this is Rustingo v1.0.0.

## rustingo — Test Results

**Last Check Date**: 2026-03-22
**Rust**: Cargo 2021 edition
**Command**: `cargo test -- --nocapture`
**Build**: `cargo clippy -- -D warnings` → **0 warnings, 0 errors**
**Test Time**: 0.30s

---

## Tests Summary

| Suite | Tests | Passed | Failed | Ignored | Time |
|-------|-------|--------|--------|---------|------|
| `tests/autograd_tests.rs` | 15 | **15** | 0 | 0 | 0.00s |
| `tests/nn_tests.rs` | 12 | **12** | 0 | 0 | 0.02s |
| `tests/tensor_tests.rs` | 26 | **26** | 0 | 0 | 0.00s |
| Doc tests (`src/`) | 20 | **19** | 0 | 1 | 0.31s |
| Lib unit tests | 0 | 0 | 0 | 0 | — |
| **TOTAL** | **73** | **72** | **0** | **1** | |

> **Ignored**: `src/optim/mod.rs` (line 8) — usage example uses `/// ```ignore` block
> (intentional — the snippet requires a full training context not available in isolation)

---

## Phase 1 — Tensor Core & Math Module

**Suite**: `tests/tensor_tests.rs` — **26/26 PASSED**

### Construction

| Test | Description | Result |
|------|-------------|--------|
| `test_zeros` | Tensor::zeros([2,3]) shape and all-zero data | ✅ |
| `test_ones` | Tensor::ones([3]) all-ones data | ✅ |
| `test_scalar` | Scalar tensor shape=[], item() accessor | ✅ |
| `test_from_vec_ok` | from_vec([4], shape=[2,2]) → correct shape & data | ✅ |
| `test_from_vec_size_mismatch` | from_vec length mismatch → Err(SizeMismatch) | ✅ |
| `test_rand_shape` | rand([4,5]) → correct shape, values in [0,1) | ✅ |
| `test_arange` | arange(0,5,1) → [0,1,2,3,4] | ✅ |
| `test_xavier_uniform_shape` | xavier_uniform([4,8]) → values within ±limit | ✅ |

### Gradient Tracking

| Test | Description | Result |
|------|-------------|--------|
| `test_requires_grad_propagates` | add(a,b) with a.requires_grad=true → output.requires_grad=true | ✅ |
| `test_no_grad_when_neither_requires_it` | Both false → output.requires_grad=false | ✅ |

### Element-wise Ops (forward)

| Test | Description | Result |
|------|-------------|--------|
| `test_add` | [1,2] + [3,4] = [4,6] | ✅ |
| `test_sub` | [5,3] - [2,1] = [3,2] | ✅ |
| `test_mul` | [2,3] * [4,5] = [8,15] | ✅ |
| `test_div` | [6,9] / [2,3] = [3,3] | ✅ |
| `test_pow` | [2,3]^2 = [4,9] | ✅ |
| `test_relu` | [-2,-0.5,0,1,3] → [0,0,0,1,3] | ✅ |
| `test_sigmoid_range` | sigmoid(-10)≈0, sigmoid(0)=0.5, sigmoid(10)≈1 | ✅ |
| `test_tanh_range` | tanh(-5)<-0.99, tanh(0)=0, tanh(5)>0.99 | ✅ |

### Matmul & Reductions

| Test | Description | Result |
|------|-------------|--------|
| `test_matmul_2x2` | [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]] | ✅ |
| `test_matmul_shape_error` | Incompatible shapes → Err(MatmulIncompatible) | ✅ |
| `test_transpose_2d` | [2,3]^T → [3,2] correct values | ✅ |
| `test_sum_scalar` | sum([1,2,3]) = 6.0 scalar | ✅ |
| `test_mean_scalar` | mean([2,4,6]) = 4.0 scalar | ✅ |

### Low-level Math Module

| Test | Description | Result |
|------|-------------|--------|
| `test_matrix_dot` | 2×2 matmul correct values | ✅ |
| `test_matrix_identity` | 3×3 identity diagonals | ✅ |
| `test_vector_dot` | [1,2,3]·[4,5,6] = 32 | ✅ |

---

## Phase 2 — Autograd Engine

**Suite**: `tests/autograd_tests.rs` — **15/15 PASSED**

### Scalar Op Gradients (analytically verified)

| Test | Formula | Expected Gradient | Result |
|------|---------|-------------------|--------|
| `test_grad_pow_squared` | y=x², x=[2,3] | [4.0, 6.0] | ✅ |
| `test_grad_add` | z=a+b, sum | da=db=[1,1] | ✅ |
| `test_grad_sub` | z=a-b, sum | da=[1], db=[-1] | ✅ |
| `test_grad_mul` | z=a*b, sum | da=b=[4,5], db=a=[2,3] | ✅ |
| `test_grad_div` | z=6/2, sum | da=0.5, db=-1.5 | ✅ |
| `test_grad_log` | y=ln(x), sum | [0.5, 0.25] | ✅ |
| `test_grad_exp` | y=e^x, sum | [1.0, e] | ✅ |

### Activation Gradients

| Test | Formula | Expected Gradient | Result |
|------|---------|-------------------|--------|
| `test_grad_relu` | y=relu([-1,0,2]) | [0.0, 0.0, 1.0] | ✅ |
| `test_grad_sigmoid` | y=σ(0) | 0.25 (= 0.5×0.5) | ✅ |
| `test_grad_tanh` | y=tanh(0) | 1.0 (= 1-0²) | ✅ |

### Chain Rule

| Test | Formula | Expected Gradient | Result |
|------|---------|-------------------|--------|
| `test_chain_rule_pow_relu` | y=relu(x²-2), x=[2,1] | [4.0, 0.0] | ✅ |
| `test_gradient_accumulation_x_plus_x` | y=x+x → dy/dx=2 | [2.0] | ✅ |
| `test_gradient_accumulation_x_times_x` | y=x*x, x=3 → 2x=6 | [6.0] | ✅ |

### Matmul + Mean Gradient

| Test | Description | Result |
|------|-------------|--------|
| `test_grad_matmul` | ∂L/∂A=grad@B^T, ∂L/∂B=A^T@grad | ✅ |
| `test_grad_mean` | dy/dx_i = 1/n | ✅ |

---

## Phase 3 — Neural Network Modules

**Suite**: `tests/nn_tests.rs` — **12/12 PASSED**

### Linear Layer

| Test | Description | Result |
|------|-------------|--------|
| `test_linear_forward_shape` | Linear(4,8) forward [3,4] → [3,8] | ✅ |
| `test_linear_parameters_count` | With bias=2 params, without=1 | ✅ |
| `test_linear_grad_flow` | weight.grad and bias.grad populated after backward | ✅ |

### Activations

| Test | Description | Result |
|------|-------------|--------|
| `test_relu_zero_pass` | Module wrapper correct output | ✅ |
| `test_leaky_relu` | α=0.1, input=-2 → -0.2 | ✅ |

### Loss Functions

| Test | Description | Result |
|------|-------------|--------|
| `test_mse_zero_loss` | pred==target → loss=0 | ✅ |
| `test_mse_known_value` | pred=[1,3], target=[2,2] → loss=1.0 | ✅ |
| `test_bce_known_value` | pred=0.8, target=1 → -ln(0.8) | ✅ |

### Sequential & Utils

| Test | Description | Result |
|------|-------------|--------|
| `test_sequential_forward` | 3→5→ReLU→2, input [4,3] → [4,2] | ✅ |
| `test_sequential_depth` | 2 layers → depth()==2 | ✅ |
| `test_zero_grad_clears_gradients` | After zero_grad, grad values are 0 | ✅ |

### XOR Convergence (end-to-end training)

| Test | Description | Result |
|------|-------------|--------|
| `test_xor_loss_decreases` | 2→8→ReLU→1 MLP, Adam lr=0.05, 500 steps | ✅ |

> **XOR result**: Loss drops from ~0.25 to <0.10 within 500 Adam steps.
> Model learns all 4 XOR cases correctly.

---

## Doc Tests

**20 doc test blocks across all modules** — **19 passed, 1 ignored**

| Module | Test | Result |
|--------|------|--------|
| `src/lib.rs` | Quick-start example | ✅ |
| `src/tensor/mod.rs` | `from_vec`, `scalar`, `backward` | ✅ ✅ ✅ |
| `src/tensor/init.rs` | `zeros` example | ✅ |
| `src/math/matrix.rs` | `Matrix`, `Matrix::new`, `identity` | ✅ ✅ ✅ |
| `src/math/vector.rs` | `Vector` dot product | ✅ |
| `src/nn/linear.rs` | `Linear` forward | ✅ |
| `src/nn/activations.rs` | `Relu`, `LeakyRelu` | ✅ ✅ |
| `src/nn/sequential.rs` | `Sequential` forward | ✅ |
| `src/nn/loss.rs` | `MseLoss`, `BceLoss` | ✅ ✅ |
| `src/optim/adam.rs` | `Adam` example | ✅ |
| `src/optim/sgd.rs` | `Sgd` example | ✅ |
| `src/gpu/mod.rs` | `Device` methods | ✅ |
| `src/optim/mod.rs` | Usage example | ⊘ ignored |

---
