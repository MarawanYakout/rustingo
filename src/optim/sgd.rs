//! # Stochastic Gradient Descent (SGD)
//!
//! Classic SGD with optional Nesterov-style momentum.
//!
//! ## Update Rule (with momentum)
//!
//! ```text
//! v ← momentum · v − lr · ∇param
//! param ← param + v
//! ```
//!
//! With `momentum = 0.0` this reduces to vanilla gradient descent.

use crate::tensor::Tensor;
use super::Optimizer;

/// SGD optimizer with optional momentum.
///
/// # Examples
/// ```
/// use rustingo::optim::{Sgd, Optimizer};
/// use rustingo::Tensor;
///
/// let w = Tensor::from_vec(vec![1.0], vec![], true).unwrap();
/// // pretend backward has run and set grad
/// let mut opt = Sgd::new(vec![w.clone()], 0.1, 0.0);
/// opt.step();
/// ```
pub struct Sgd {
    params: Vec<Tensor>,
    /// Learning rate.
    pub lr: f32,
    /// Momentum coefficient. `0.0` = vanilla SGD.
    pub momentum: f32,
    /// Velocity vectors — one per parameter, same shape as parameter data.
    velocity: Vec<Vec<f32>>,
}

impl Sgd {
    /// Creates a new `SGD` optimizer.
    ///
    /// # Parameters
    /// - `params`: tensors to optimise (typically `model.parameters()`)
    /// - `lr`: learning rate (e.g. `0.01`)
    /// - `momentum`: momentum factor, `0.0` for vanilla SGD
    pub fn new(params: Vec<Tensor>, lr: f32, momentum: f32) -> Self {
        let velocity: Vec<Vec<f32>> = params.iter().map(|p| vec![0.0; p.numel()]).collect();
        Sgd { params, lr, momentum, velocity }
    }
}

impl Optimizer for Sgd {
    fn step(&mut self) {
        for (i, param) in self.params.iter().enumerate() {
            if let Some(grad) = param.grad() {
                let mut inner = param.inner.write().unwrap();
                for (j, &g) in grad.iter().enumerate() {
                    // v = momentum * v - lr * g
                    self.velocity[i][j] = self.momentum * self.velocity[i][j] - self.lr * g;
                    // param += v
                    inner.data[j] += self.velocity[i][j];
                }
            }
        }
    }

    fn zero_grad(&mut self) {
        for param in &self.params {
            param.zero_grad();
        }
    }
}
