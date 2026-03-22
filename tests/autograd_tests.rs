//! # Autograd Tests
//!
//! Verifies gradient correctness through the backward pass.
//! All expected values are analytically derived.

use rustingo::Tensor;

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn assert_close(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "gradient length mismatch");
    for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (av - bv).abs() < tol,
            "index {}: expected {} got {} (diff {})",
            i, bv, av, (av - bv).abs()
        );
    }
}

// ─── Scalar ops ──────────────────────────────────────────────────────────────

#[test]
fn test_grad_pow_squared() {
    // y = x², dy/dx = 2x
    // x = [2.0, 3.0] → grad = [4.0, 6.0]
    let x = Tensor::from_vec(vec![2.0, 3.0], vec![2], true).unwrap();
    let y = x.pow(2.0).sum();
    y.backward();
    assert_close(&x.grad().unwrap(), &[4.0, 6.0], 1e-5);
}

#[test]
fn test_grad_add() {
    // z = a + b, dz/da = 1, dz/db = 1
    let a = Tensor::from_vec(vec![1.0, 2.0], vec![2], true).unwrap();
    let b = Tensor::from_vec(vec![3.0, 4.0], vec![2], true).unwrap();
    let z = a.add(&b).unwrap().sum();
    z.backward();
    assert_close(&a.grad().unwrap(), &[1.0, 1.0], 1e-5);
    assert_close(&b.grad().unwrap(), &[1.0, 1.0], 1e-5);
}

#[test]
fn test_grad_sub() {
    // z = a - b, dz/da = 1, dz/db = -1
    let a = Tensor::from_vec(vec![5.0], vec![1], true).unwrap();
    let b = Tensor::from_vec(vec![3.0], vec![1], true).unwrap();
    let z = a.sub(&b).unwrap().sum();
    z.backward();
    assert_close(&a.grad().unwrap(), &[1.0], 1e-5);
    assert_close(&b.grad().unwrap(), &[-1.0], 1e-5);
}

#[test]
fn test_grad_mul() {
    // z = a * b, dz/da = b, dz/db = a
    let a = Tensor::from_vec(vec![2.0, 3.0], vec![2], true).unwrap();
    let b = Tensor::from_vec(vec![4.0, 5.0], vec![2], true).unwrap();
    let z = a.mul(&b).unwrap().sum();
    z.backward();
    assert_close(&a.grad().unwrap(), &[4.0, 5.0], 1e-5);
    assert_close(&b.grad().unwrap(), &[2.0, 3.0], 1e-5);
}

#[test]
fn test_grad_div() {
    // z = a / b, dz/da = 1/b, dz/db = -a/b²
    // a=6, b=2 → dz/da=0.5, dz/db=-6/4=-1.5
    let a = Tensor::from_vec(vec![6.0], vec![1], true).unwrap();
    let b = Tensor::from_vec(vec![2.0], vec![1], true).unwrap();
    let z = a.div(&b).unwrap().sum();
    z.backward();
    assert_close(&a.grad().unwrap(), &[0.5], 1e-5);
    assert_close(&b.grad().unwrap(), &[-1.5], 1e-5);
}

#[test]
fn test_grad_log() {
    // y = ln(x), dy/dx = 1/x
    // x=2.0 → grad=0.5
    let x = Tensor::from_vec(vec![2.0, 4.0], vec![2], true).unwrap();
    let y = x.log().sum();
    y.backward();
    assert_close(&x.grad().unwrap(), &[0.5, 0.25], 1e-5);
}

#[test]
fn test_grad_exp() {
    // y = e^x, dy/dx = e^x
    let x = Tensor::from_vec(vec![0.0, 1.0], vec![2], true).unwrap();
    let y = x.exp().sum();
    y.backward();
    // grad at x=0 is e^0=1.0, at x=1 is e^1≈2.71828
    let grad = x.grad().unwrap();
    assert!((grad[0] - 1.0).abs() < 1e-5);
    assert!((grad[1] - std::f32::consts::E).abs() < 1e-4);
}

// ─── Activation gradients ────────────────────────────────────────────────────

#[test]
fn test_grad_relu() {
    // y = relu(x), dy/dx = 1 if x>0, else 0
    let x = Tensor::from_vec(vec![-1.0, 0.0, 2.0], vec![3], true).unwrap();
    let y = x.relu().sum();
    y.backward();
    assert_close(&x.grad().unwrap(), &[0.0, 0.0, 1.0], 1e-5);
}

#[test]
fn test_grad_sigmoid() {
    // dy/dx = σ(x) * (1 - σ(x))
    // At x=0: σ=0.5, grad = 0.5 * 0.5 = 0.25
    let x = Tensor::from_vec(vec![0.0], vec![1], true).unwrap();
    let y = x.sigmoid().sum();
    y.backward();
    assert_close(&x.grad().unwrap(), &[0.25], 1e-5);
}

#[test]
fn test_grad_tanh() {
    // dy/dx = 1 - tanh²(x)
    // At x=0: tanh=0, grad = 1
    let x = Tensor::from_vec(vec![0.0], vec![1], true).unwrap();
    let y = x.tanh().sum();
    y.backward();
    assert_close(&x.grad().unwrap(), &[1.0], 1e-5);
}

// ─── Chain rule ───────────────────────────────────────────────────────────────

#[test]
fn test_chain_rule_pow_relu() {
    // y = relu(x² - 2)
    // At x=2: x²=4, relu(4-2)=2, dy/dx = 1 * 2*2 = 4
    // At x=1: x²=1, relu(1-2)=0 (negative), dy/dx = 0
    let x = Tensor::from_vec(vec![2.0, 1.0], vec![2], true).unwrap();
    let sq = x.pow(2.0);
    let shifted = sq.add_scalar(-2.0);
    let y = shifted.relu().sum();
    y.backward();
    assert_close(&x.grad().unwrap(), &[4.0, 0.0], 1e-5);
}

#[test]
fn test_gradient_accumulation_x_plus_x() {
    // y = x + x  → dy/dx = 2
    let x = Tensor::from_vec(vec![3.0], vec![1], true).unwrap();
    let y = x.add(&x).unwrap().sum();
    y.backward();
    assert_close(&x.grad().unwrap(), &[2.0], 1e-5);
}

#[test]
fn test_gradient_accumulation_x_times_x() {
    // y = x * x = x²  → dy/dx = 2x = 6 (at x=3)
    let x = Tensor::from_vec(vec![3.0], vec![1], true).unwrap();
    let y = x.mul(&x).unwrap().sum();
    y.backward();
    assert_close(&x.grad().unwrap(), &[6.0], 1e-5);
}

// ─── Matmul gradient ─────────────────────────────────────────────────────────

#[test]
fn test_grad_matmul() {
    // A = [[1,2]], B = [[3],[4]]
    // C = A @ B = [[1*3 + 2*4]] = [[11]]
    // ∂L/∂A = ∂L/∂C @ B^T = [[1]] @ [[3,4]] = [[3,4]]
    // ∂L/∂B = A^T @ ∂L/∂C = [[1],[2]] @ [[1]] = [[1],[2]]
    let a = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2], true).unwrap();
    let b = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1], true).unwrap();
    let c = a.matmul(&b).unwrap().sum();
    c.backward();
    assert_close(&a.grad().unwrap(), &[3.0, 4.0], 1e-5);
    assert_close(&b.grad().unwrap(), &[1.0, 2.0], 1e-5);
}

// ─── Mean gradient ───────────────────────────────────────────────────────────

#[test]
fn test_grad_mean() {
    // y = mean([a, b, c]) → dy/da = dy/db = dy/dc = 1/3
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3], true).unwrap();
    let y = x.mean();
    y.backward();
    assert_close(&x.grad().unwrap(), &[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], 1e-5);
}
