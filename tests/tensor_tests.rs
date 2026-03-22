//! # Tensor Tests
//!
//! Tests for tensor creation, shape, element access, and basic ops.

use rustingo::Tensor;

// ─── Construction ─────────────────────────────────────────────────────────────

#[test]
fn test_zeros() {
    let t = Tensor::zeros(&[2, 3]);
    assert_eq!(t.shape(), vec![2, 3]);
    assert_eq!(t.numel(), 6);
    assert!(t.data().iter().all(|&x| x == 0.0));
}

#[test]
fn test_ones() {
    let t = Tensor::ones(&[3]);
    assert_eq!(t.numel(), 3);
    assert!(t.data().iter().all(|&x| x == 1.0));
}

#[test]
fn test_scalar() {
    let s = Tensor::scalar(42.0, false);
    assert_eq!(s.shape(), vec![]);
    assert_eq!(s.item(), 42.0);
    assert_eq!(s.numel(), 1);
}

#[test]
fn test_from_vec_ok() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false).unwrap();
    assert_eq!(t.shape(), vec![2, 2]);
    assert_eq!(t.data(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_from_vec_size_mismatch() {
    let result = Tensor::from_vec(vec![1.0, 2.0], vec![3], false);
    assert!(result.is_err());
}

#[test]
fn test_rand_shape() {
    let t = Tensor::rand(&[4, 5], 99);
    assert_eq!(t.shape(), vec![4, 5]);
    assert_eq!(t.numel(), 20);
    // All values should be in [0, 1)
    assert!(t.data().iter().all(|&x| (0.0..1.0).contains(&x)));
}

#[test]
fn test_arange() {
    let t = Tensor::arange(0.0, 5.0, 1.0).unwrap();
    assert_eq!(t.shape(), vec![5]);
    assert_eq!(t.data(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_xavier_uniform_shape() {
    let t = Tensor::xavier_uniform(&[4, 8], 7).unwrap();
    assert_eq!(t.shape(), vec![4, 8]);
    let limit = (6.0_f32 / 12.0_f32).sqrt();
    assert!(t.data().iter().all(|&x| x >= -limit && x <= limit));
}

// ─── Requires-grad flag ───────────────────────────────────────────────────────

#[test]
fn test_requires_grad_propagates() {
    let a = Tensor::from_vec(vec![1.0, 2.0], vec![2], true).unwrap();
    let b = Tensor::from_vec(vec![3.0, 4.0], vec![2], false).unwrap();
    let c = a.add(&b).unwrap();
    assert!(c.requires_grad(), "output should require grad because a does");
}

#[test]
fn test_no_grad_when_neither_requires_it() {
    let a = Tensor::from_vec(vec![1.0], vec![1], false).unwrap();
    let b = Tensor::from_vec(vec![2.0], vec![1], false).unwrap();
    let c = a.add(&b).unwrap();
    assert!(!c.requires_grad());
}

// ─── Element-wise ops (forward values) ────────────────────────────────────────

#[test]
fn test_add() {
    let a = Tensor::from_vec(vec![1.0, 2.0], vec![2], false).unwrap();
    let b = Tensor::from_vec(vec![3.0, 4.0], vec![2], false).unwrap();
    let c = a.add(&b).unwrap();
    assert_eq!(c.data(), vec![4.0, 6.0]);
}

#[test]
fn test_sub() {
    let a = Tensor::from_vec(vec![5.0, 3.0], vec![2], false).unwrap();
    let b = Tensor::from_vec(vec![2.0, 1.0], vec![2], false).unwrap();
    let c = a.sub(&b).unwrap();
    assert_eq!(c.data(), vec![3.0, 2.0]);
}

#[test]
fn test_mul() {
    let a = Tensor::from_vec(vec![2.0, 3.0], vec![2], false).unwrap();
    let b = Tensor::from_vec(vec![4.0, 5.0], vec![2], false).unwrap();
    let c = a.mul(&b).unwrap();
    assert_eq!(c.data(), vec![8.0, 15.0]);
}

#[test]
fn test_div() {
    let a = Tensor::from_vec(vec![6.0, 9.0], vec![2], false).unwrap();
    let b = Tensor::from_vec(vec![2.0, 3.0], vec![2], false).unwrap();
    let c = a.div(&b).unwrap();
    assert_eq!(c.data(), vec![3.0, 3.0]);
}

#[test]
fn test_pow() {
    let x = Tensor::from_vec(vec![2.0, 3.0], vec![2], false).unwrap();
    let y = x.pow(2.0);
    assert_eq!(y.data(), vec![4.0, 9.0]);
}

#[test]
fn test_relu() {
    let x = Tensor::from_vec(vec![-2.0, -0.5, 0.0, 1.0, 3.0], vec![5], false).unwrap();
    let y = x.relu();
    assert_eq!(y.data(), vec![0.0, 0.0, 0.0, 1.0, 3.0]);
}

#[test]
fn test_sigmoid_range() {
    let x = Tensor::from_vec(vec![-10.0, 0.0, 10.0], vec![3], false).unwrap();
    let y = x.sigmoid();
    let d = y.data();
    assert!(d[0] > 0.0 && d[0] < 0.01); // near 0
    assert!((d[1] - 0.5).abs() < 1e-6); // exactly 0.5 at x=0
    assert!(d[2] > 0.99);               // near 1
}

#[test]
fn test_tanh_range() {
    let x = Tensor::from_vec(vec![-5.0, 0.0, 5.0], vec![3], false).unwrap();
    let y = x.tanh();
    let d = y.data();
    assert!(d[0] < -0.99);
    assert!(d[1].abs() < 1e-6);
    assert!(d[2] > 0.99);
}

// ─── Matmul ───────────────────────────────────────────────────────────────────

#[test]
fn test_matmul_2x2() {
    // [1 2] @ [5 6] = [1*5+2*7, 1*6+2*8] = [19, 22]
    // [3 4]   [7 8]   [3*5+4*7, 3*6+4*8]   [43, 50]
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false).unwrap();
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], false).unwrap();
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.shape(), vec![2, 2]);
    assert_eq!(c.data(), vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn test_matmul_shape_error() {
    let a = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2], false).unwrap();
    let b = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2], false).unwrap();
    assert!(a.matmul(&b).is_err());
}

#[test]
fn test_transpose_2d() {
    // [1 2 3]^T = [1 4]
    // [4 5 6]     [2 5]
    //             [3 6]
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false).unwrap();
    let t = a.transpose().unwrap();
    assert_eq!(t.shape(), vec![3, 2]);
    assert_eq!(t.data(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

// ─── Reductions ───────────────────────────────────────────────────────────────

#[test]
fn test_sum_scalar() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3], false).unwrap();
    let s = x.sum();
    assert_eq!(s.shape(), vec![]);
    assert_eq!(s.item(), 6.0);
}

#[test]
fn test_mean_scalar() {
    let x = Tensor::from_vec(vec![2.0, 4.0, 6.0], vec![3], false).unwrap();
    let m = x.mean();
    assert_eq!(m.item(), 4.0);
}

// ─── Math module (no autograd) ────────────────────────────────────────────────

#[test]
fn test_matrix_dot() {
    use rustingo::math::Matrix;
    let a = Matrix::new(2, 2, [1, 2, 3, 4]).unwrap();
    let b = Matrix::new(2, 2, [5, 6, 7, 8]).unwrap();
    let c = a.dot(&b).unwrap();
    assert_eq!(c.get(0, 0), 19.0);
    assert_eq!(c.get(1, 1), 50.0);
}

#[test]
fn test_matrix_identity() {
    use rustingo::math::Matrix;
    let eye = Matrix::identity(3);
    assert_eq!(eye.get(0, 0), 1.0);
    assert_eq!(eye.get(0, 1), 0.0);
    assert_eq!(eye.get(1, 1), 1.0);
}

#[test]
fn test_vector_dot() {
    use rustingo::math::Vector;
    let a = Vector::from_vec(vec![1.0, 2.0, 3.0]);
    let b = Vector::from_vec(vec![4.0, 5.0, 6.0]);
    assert_eq!(a.dot(&b).unwrap(), 32.0);
}
