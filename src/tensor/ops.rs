//! # Tensor Operations
//!
//! All mathematical operations on `Tensor`, each with a correct backward pass.
//!
//! ## Autograd Contract
//!
//! Every operation follows this pattern:
//! 1. **Forward**: compute the output data
//! 2. **Capture**: clone input tensor handles and any data needed for backward
//! 3. **Backward closure**: compute ∂loss/∂inputs given ∂loss/∂output
//! 4. **Return**: `Tensor::from_op(result, shape, parents, Some(backward_fn))`

use crate::error::{Error, Result};
use super::Tensor;

impl Tensor {
    // ─── Element-wise binary ops ─────────────────────────────────────────────

    /// Element-wise addition: `self + other`.
    ///
    /// Both tensors must have the same shape.
    /// ∂/∂a = 1, ∂/∂b = 1
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        let (a_data, a_shape) = read_data(self);
        let (b_data, b_shape) = read_data(other);

        check_shape(&a_shape, &b_shape)?;

        let result: Vec<f32> = a_data.iter().zip(&b_data).map(|(a, b)| a + b).collect();

        let need_grad = self.requires_grad() || other.requires_grad();
        if !need_grad {
            return Ok(Tensor::from_op(result, a_shape, vec![], None));
        }

        let a = self.clone();
        let b = other.clone();
        let bwd = Box::new(move |grad: &[f32]| {
            a.accumulate_grad(grad);
            b.accumulate_grad(grad);
        });

        Ok(Tensor::from_op(
            result,
            a_shape,
            vec![self.clone(), other.clone()],
            Some(bwd),
        ))
    }

    /// Element-wise subtraction: `self - other`.
    ///
    /// ∂/∂a = 1, ∂/∂b = -1
    pub fn sub(&self, other: &Tensor) -> Result<Tensor> {
        let (a_data, a_shape) = read_data(self);
        let (b_data, b_shape) = read_data(other);

        check_shape(&a_shape, &b_shape)?;

        let result: Vec<f32> = a_data.iter().zip(&b_data).map(|(a, b)| a - b).collect();

        let need_grad = self.requires_grad() || other.requires_grad();
        if !need_grad {
            return Ok(Tensor::from_op(result, a_shape, vec![], None));
        }

        let a = self.clone();
        let b = other.clone();
        let bwd = Box::new(move |grad: &[f32]| {
            a.accumulate_grad(grad);
            let neg: Vec<f32> = grad.iter().map(|g| -g).collect();
            b.accumulate_grad(&neg);
        });

        Ok(Tensor::from_op(
            result,
            a_shape,
            vec![self.clone(), other.clone()],
            Some(bwd),
        ))
    }

    /// Element-wise multiplication (Hadamard product): `self * other`.
    ///
    /// ∂/∂a = b, ∂/∂b = a
    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        let (a_data, a_shape) = read_data(self);
        let (b_data, b_shape) = read_data(other);

        check_shape(&a_shape, &b_shape)?;

        let result: Vec<f32> = a_data.iter().zip(&b_data).map(|(a, b)| a * b).collect();

        let need_grad = self.requires_grad() || other.requires_grad();
        if !need_grad {
            return Ok(Tensor::from_op(result, a_shape, vec![], None));
        }

        let a = self.clone();
        let b = other.clone();
        let a_cap = a_data;
        let b_cap = b_data;
        let bwd = Box::new(move |grad: &[f32]| {
            let ga: Vec<f32> = grad.iter().zip(&b_cap).map(|(g, bv)| g * bv).collect();
            let gb: Vec<f32> = grad.iter().zip(&a_cap).map(|(g, av)| g * av).collect();
            a.accumulate_grad(&ga);
            b.accumulate_grad(&gb);
        });

        Ok(Tensor::from_op(
            result,
            a_shape,
            vec![self.clone(), other.clone()],
            Some(bwd),
        ))
    }

    /// Element-wise division: `self / other`.
    ///
    /// ∂/∂a = 1/b, ∂/∂b = -a/b²
    pub fn div(&self, other: &Tensor) -> Result<Tensor> {
        let (a_data, a_shape) = read_data(self);
        let (b_data, b_shape) = read_data(other);

        check_shape(&a_shape, &b_shape)?;

        if b_data.contains(&0.0) {
            return Err(Error::DivisionByZero);
        }

        let result: Vec<f32> = a_data.iter().zip(&b_data).map(|(a, b)| a / b).collect();

        let need_grad = self.requires_grad() || other.requires_grad();
        if !need_grad {
            return Ok(Tensor::from_op(result, a_shape, vec![], None));
        }

        let a = self.clone();
        let b = other.clone();
        let a_cap = a_data;
        let b_cap = b_data;
        let bwd = Box::new(move |grad: &[f32]| {
            // ∂/∂a = 1/b
            let ga: Vec<f32> = grad.iter().zip(&b_cap).map(|(g, bv)| g / bv).collect();
            // ∂/∂b = -a / b²
            let gb: Vec<f32> = grad
                .iter()
                .zip(a_cap.iter().zip(&b_cap))
                .map(|(g, (av, bv))| -g * av / (bv * bv))
                .collect();
            a.accumulate_grad(&ga);
            b.accumulate_grad(&gb);
        });

        Ok(Tensor::from_op(
            result,
            a_shape,
            vec![self.clone(), other.clone()],
            Some(bwd),
        ))
    }

    // ─── Matrix multiplication ───────────────────────────────────────────────

    /// Matrix multiplication: `[m, k] @ [k, n] → [m, n]`.
    ///
    /// Both tensors must be exactly 2-D.
    ///
    /// Backward:
    /// - ∂L/∂A = ∂L/∂C @ B^T
    /// - ∂L/∂B = A^T @ ∂L/∂C
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        let (a_data, a_shape) = read_data(self);
        let (b_data, b_shape) = read_data(other);

        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(Error::InvalidInput(
                "matmul requires exactly 2-D tensors".into(),
            ));
        }

        let (m, k, n) = (a_shape[0], a_shape[1], b_shape[1]);
        if k != b_shape[0] {
            return Err(Error::MatmulIncompatible {
                a_cols: k,
                b_rows: b_shape[0],
            });
        }

        // Naive O(m·k·n) — readable and correct; can be replaced with BLAS later.
        let mut result = vec![0.0_f32; m * n];
        for i in 0..m {
            for l in 0..k {
                let a_il = a_data[i * k + l];
                for j in 0..n {
                    result[i * n + j] += a_il * b_data[l * n + j];
                }
            }
        }

        let need_grad = self.requires_grad() || other.requires_grad();
        if !need_grad {
            return Ok(Tensor::from_op(result, vec![m, n], vec![], None));
        }

        let a = self.clone();
        let b = other.clone();
        let a_cap = a_data;
        let b_cap = b_data;

        let bwd = Box::new(move |grad: &[f32]| {
            // grad_a[i,l] = Σ_j grad[i,j] * B[l,j]  (grad @ B^T)
            let mut grad_a = vec![0.0_f32; m * k];
            for i in 0..m {
                for l in 0..k {
                    let mut s = 0.0_f32;
                    for j in 0..n {
                        s += grad[i * n + j] * b_cap[l * n + j];
                    }
                    grad_a[i * k + l] = s;
                }
            }

            // grad_b[l,j] = Σ_i A[i,l] * grad[i,j]  (A^T @ grad)
            let mut grad_b = vec![0.0_f32; k * n];
            for l in 0..k {
                for j in 0..n {
                    let mut s = 0.0_f32;
                    for i in 0..m {
                        s += a_cap[i * k + l] * grad[i * n + j];
                    }
                    grad_b[l * n + j] = s;
                }
            }

            a.accumulate_grad(&grad_a);
            b.accumulate_grad(&grad_b);
        });

        Ok(Tensor::from_op(
            result,
            vec![m, n],
            vec![self.clone(), other.clone()],
            Some(bwd),
        ))
    }

    /// Transpose a 2-D tensor: `[m, n] → [n, m]`.
    ///
    /// ∂L/∂A = (∂L/∂A^T)^T
    pub fn transpose(&self) -> Result<Tensor> {
        let (data, shape) = read_data(self);

        if shape.len() != 2 {
            return Err(Error::InvalidInput(
                "transpose requires a 2-D tensor".into(),
            ));
        }

        let (m, n) = (shape[0], shape[1]);
        let mut result = vec![0.0_f32; m * n];
        for i in 0..m {
            for j in 0..n {
                result[j * m + i] = data[i * n + j];
            }
        }

        let need_grad = self.requires_grad();
        if !need_grad {
            return Ok(Tensor::from_op(result, vec![n, m], vec![], None));
        }

        let a = self.clone();
        let bwd = Box::new(move |grad: &[f32]| {
            // grad of output is [n,m]; we need to transpose back to [m,n]
            let mut grad_a = vec![0.0_f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    grad_a[i * n + j] = grad[j * m + i];
                }
            }
            a.accumulate_grad(&grad_a);
        });

        Ok(Tensor::from_op(
            result,
            vec![n, m],
            vec![self.clone()],
            Some(bwd),
        ))
    }

    /// Adds a 1-D bias to each row of a 2-D tensor.
    ///
    /// `self` must be `[batch, features]`, `bias` must be `[features]`.
    ///
    /// Used internally by `nn::Linear`. Accumulates the bias gradient by
    /// summing `grad` over the batch dimension.
    pub fn add_bias(&self, bias: &Tensor) -> Result<Tensor> {
        let (data, shape) = read_data(self);
        let (b_data, b_shape) = read_data(bias);

        if shape.len() != 2 {
            return Err(Error::InvalidInput(
                "add_bias requires a 2-D input tensor".into(),
            ));
        }
        if b_shape.len() != 1 || b_shape[0] != shape[1] {
            return Err(Error::ShapeMismatch {
                expected: vec![shape[1]],
                got: b_shape,
            });
        }

        let (batch, feats) = (shape[0], shape[1]);
        let mut result = vec![0.0_f32; batch * feats];
        for i in 0..batch {
            for j in 0..feats {
                result[i * feats + j] = data[i * feats + j] + b_data[j];
            }
        }

        let need_grad = self.requires_grad() || bias.requires_grad();
        if !need_grad {
            return Ok(Tensor::from_op(result, vec![batch, feats], vec![], None));
        }

        let a = self.clone();
        let b = bias.clone();
        let bwd = Box::new(move |grad: &[f32]| {
            // ∂L/∂input = grad (pass-through)
            a.accumulate_grad(grad);

            // ∂L/∂bias = Σ_batch grad   (reduce over batch dim)
            let mut bias_grad = vec![0.0_f32; feats];
            for i in 0..batch {
                for j in 0..feats {
                    bias_grad[j] += grad[i * feats + j];
                }
            }
            b.accumulate_grad(&bias_grad);
        });

        Ok(Tensor::from_op(
            result,
            vec![batch, feats],
            vec![self.clone(), bias.clone()],
            Some(bwd),
        ))
    }

    // ─── Scalar ops ──────────────────────────────────────────────────────────

    /// Adds a scalar to every element: `x + scalar`.
    pub fn add_scalar(&self, scalar: f32) -> Tensor {
        let (data, shape) = read_data(self);
        let result: Vec<f32> = data.iter().map(|x| x + scalar).collect();

        if !self.requires_grad() {
            return Tensor::from_op(result, shape, vec![], None);
        }

        let a = self.clone();
        let bwd = Box::new(move |grad: &[f32]| a.accumulate_grad(grad));

        Tensor::from_op(result, shape, vec![self.clone()], Some(bwd))
    }

    /// Multiplies every element by a scalar: `x * scalar`.
    pub fn mul_scalar(&self, scalar: f32) -> Tensor {
        let (data, shape) = read_data(self);
        let result: Vec<f32> = data.iter().map(|x| x * scalar).collect();

        if !self.requires_grad() {
            return Tensor::from_op(result, shape, vec![], None);
        }

        let a = self.clone();
        let bwd = Box::new(move |grad: &[f32]| {
            let g: Vec<f32> = grad.iter().map(|gv| gv * scalar).collect();
            a.accumulate_grad(&g);
        });

        Tensor::from_op(result, shape, vec![self.clone()], Some(bwd))
    }

    /// Negates every element: `-x`.
    pub fn neg(&self) -> Tensor {
        self.mul_scalar(-1.0)
    }

    // ─── Element-wise unary ops ──────────────────────────────────────────────

    /// Element-wise power: `x ^ exp`.
    ///
    /// ∂/∂x = exp · x^(exp-1)
    pub fn pow(&self, exp: f32) -> Tensor {
        let (data, shape) = read_data(self);
        let result: Vec<f32> = data.iter().map(|x| x.powf(exp)).collect();

        if !self.requires_grad() {
            return Tensor::from_op(result, shape, vec![], None);
        }

        let a = self.clone();
        let data_cap = data;
        let bwd = Box::new(move |grad: &[f32]| {
            let g: Vec<f32> = grad
                .iter()
                .zip(&data_cap)
                .map(|(gv, x)| gv * exp * x.powf(exp - 1.0))
                .collect();
            a.accumulate_grad(&g);
        });

        Tensor::from_op(result, shape, vec![self.clone()], Some(bwd))
    }

    /// Element-wise natural logarithm: `ln(x)`.
    ///
    /// ∂/∂x = 1/x
    pub fn log(&self) -> Tensor {
        let (data, shape) = read_data(self);
        let result: Vec<f32> = data.iter().map(|x| x.ln()).collect();

        if !self.requires_grad() {
            return Tensor::from_op(result, shape, vec![], None);
        }

        let a = self.clone();
        let data_cap = data;
        let bwd = Box::new(move |grad: &[f32]| {
            let g: Vec<f32> = grad.iter().zip(&data_cap).map(|(gv, x)| gv / x).collect();
            a.accumulate_grad(&g);
        });

        Tensor::from_op(result, shape, vec![self.clone()], Some(bwd))
    }

    /// Element-wise exponential: `e^x`.
    ///
    /// ∂/∂x = e^x (same as output)
    pub fn exp(&self) -> Tensor {
        let (data, shape) = read_data(self);
        let out: Vec<f32> = data.iter().map(|x| x.exp()).collect();

        if !self.requires_grad() {
            return Tensor::from_op(out, shape, vec![], None);
        }

        let a = self.clone();
        let out_cap = out.clone();
        let bwd = Box::new(move |grad: &[f32]| {
            let g: Vec<f32> = grad.iter().zip(&out_cap).map(|(gv, ev)| gv * ev).collect();
            a.accumulate_grad(&g);
        });

        Tensor::from_op(out, shape, vec![self.clone()], Some(bwd))
    }

    // ─── Activation functions ────────────────────────────────────────────────

    /// Rectified Linear Unit: `max(0, x)`.
    ///
    /// ∂/∂x = 1 if x > 0, else 0
    pub fn relu(&self) -> Tensor {
        let (data, shape) = read_data(self);
        let out: Vec<f32> = data.iter().map(|x| x.max(0.0)).collect();

        if !self.requires_grad() {
            return Tensor::from_op(out, shape, vec![], None);
        }

        let a = self.clone();
        // Mask: 1.0 where x > 0, else 0.0
        let mask: Vec<f32> = data.iter().map(|x| if *x > 0.0 { 1.0 } else { 0.0 }).collect();
        let bwd = Box::new(move |grad: &[f32]| {
            let g: Vec<f32> = grad.iter().zip(&mask).map(|(gv, m)| gv * m).collect();
            a.accumulate_grad(&g);
        });

        Tensor::from_op(out, shape, vec![self.clone()], Some(bwd))
    }

    /// Sigmoid: `1 / (1 + e^{-x})`.
    ///
    /// ∂/∂x = σ(x) · (1 - σ(x))
    pub fn sigmoid(&self) -> Tensor {
        let (data, shape) = read_data(self);
        let out: Vec<f32> = data.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();

        if !self.requires_grad() {
            return Tensor::from_op(out, shape, vec![], None);
        }

        let a = self.clone();
        let out_cap = out.clone();
        let bwd = Box::new(move |grad: &[f32]| {
            let g: Vec<f32> = grad
                .iter()
                .zip(&out_cap)
                .map(|(gv, s)| gv * s * (1.0 - s))
                .collect();
            a.accumulate_grad(&g);
        });

        Tensor::from_op(out, shape, vec![self.clone()], Some(bwd))
    }

    /// Hyperbolic tangent: `tanh(x)`.
    ///
    /// ∂/∂x = 1 - tanh²(x)
    pub fn tanh(&self) -> Tensor {
        let (data, shape) = read_data(self);
        let out: Vec<f32> = data.iter().map(|x| x.tanh()).collect();

        if !self.requires_grad() {
            return Tensor::from_op(out, shape, vec![], None);
        }

        let a = self.clone();
        let out_cap = out.clone();
        let bwd = Box::new(move |grad: &[f32]| {
            let g: Vec<f32> = grad
                .iter()
                .zip(&out_cap)
                .map(|(gv, t)| gv * (1.0 - t * t))
                .collect();
            a.accumulate_grad(&g);
        });

        Tensor::from_op(out, shape, vec![self.clone()], Some(bwd))
    }

    /// Leaky ReLU: `max(α·x, x)`.
    ///
    /// ∂/∂x = 1 if x > 0, else α
    pub fn leaky_relu(&self, alpha: f32) -> Tensor {
        let (data, shape) = read_data(self);
        let out: Vec<f32> = data
            .iter()
            .map(|x| if *x > 0.0 { *x } else { alpha * x })
            .collect();

        if !self.requires_grad() {
            return Tensor::from_op(out, shape, vec![], None);
        }

        let a = self.clone();
        let mask: Vec<f32> = data
            .iter()
            .map(|x| if *x > 0.0 { 1.0 } else { alpha })
            .collect();
        let bwd = Box::new(move |grad: &[f32]| {
            let g: Vec<f32> = grad.iter().zip(&mask).map(|(gv, m)| gv * m).collect();
            a.accumulate_grad(&g);
        });

        Tensor::from_op(out, shape, vec![self.clone()], Some(bwd))
    }

    // ─── Reduction ops ───────────────────────────────────────────────────────

    /// Sums all elements → scalar tensor.
    ///
    /// ∂/∂x_i = 1 for all i
    pub fn sum(&self) -> Tensor {
        let (data, _shape) = read_data(self);
        let total: f32 = data.iter().sum();
        let n = data.len();

        if !self.requires_grad() {
            return Tensor::from_op(vec![total], vec![], vec![], None);
        }

        let a = self.clone();
        let bwd = Box::new(move |grad: &[f32]| {
            let g = vec![grad[0]; n];
            a.accumulate_grad(&g);
        });

        Tensor::from_op(vec![total], vec![], vec![self.clone()], Some(bwd))
    }

    /// Mean of all elements → scalar tensor.
    ///
    /// ∂/∂x_i = 1/n
    pub fn mean(&self) -> Tensor {
        let (data, _shape) = read_data(self);
        let n = data.len();
        let mean_val: f32 = data.iter().sum::<f32>() / n as f32;

        if !self.requires_grad() {
            return Tensor::from_op(vec![mean_val], vec![], vec![], None);
        }

        let a = self.clone();
        let bwd = Box::new(move |grad: &[f32]| {
            let g = vec![grad[0] / n as f32; n];
            a.accumulate_grad(&g);
        });

        Tensor::from_op(vec![mean_val], vec![], vec![self.clone()], Some(bwd))
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Read a snapshot of a tensor's data and shape under a read lock.
#[inline]
fn read_data(t: &Tensor) -> (Vec<f32>, Vec<usize>) {
    let inner = t.inner.read().unwrap();
    (inner.data.clone(), inner.shape.clone())
}

/// Validate that two shapes are identical.
#[inline]
fn check_shape(a: &[usize], b: &[usize]) -> crate::error::Result<()> {
    if a != b {
        return Err(Error::ShapeMismatch {
            expected: a.to_vec(),
            got: b.to_vec(),
        });
    }
    Ok(())
}
