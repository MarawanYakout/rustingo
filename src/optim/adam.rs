//! # Adam Optimizer
//!
//! Adaptive Moment Estimation (Adam) — Kingma & Ba, 2014.
//!
//! ## Update Rule
//!
//! ```text
//! m  ← β₁ · m  + (1 − β₁) · g
//! v  ← β₂ · v  + (1 − β₂) · g²
//! m̂  = m  / (1 − β₁ᵗ)          ← bias-corrected first moment
//! v̂  = v  / (1 − β₂ᵗ)          ← bias-corrected second moment
//! param ← param − α · m̂ / (√v̂ + ε)
//! ```
//!
//! Default hyperparameters match the paper: `β₁=0.9, β₂=0.999, ε=1e-8`.

use crate::tensor::Tensor;
use super::Optimizer;

/// Adam optimizer with bias correction.
///
/// # Examples
/// ```
/// use rustingo::optim::{Adam, Optimizer};
/// use rustingo::Tensor;
///
/// let w = Tensor::from_vec(vec![0.5], vec![], true).unwrap();
/// let mut opt = Adam::new(vec![w.clone()], 1e-3);
/// // ... forward / backward ...
/// opt.step();
/// ```
pub struct Adam {
    params: Vec<Tensor>,
    /// Learning rate α.
    pub lr: f32,
    /// First-moment decay (default 0.9).
    pub beta1: f32,
    /// Second-moment decay (default 0.999).
    pub beta2: f32,
    /// Numerical stability constant ε (default 1e-8).
    pub eps: f32,
    /// Step counter — used for bias correction.
    t: u64,
    /// First moment estimates — one `Vec<f32>` per parameter.
    m: Vec<Vec<f32>>,
    /// Second moment estimates — one `Vec<f32>` per parameter.
    v: Vec<Vec<f32>>,
}

impl Adam {
    /// Creates an `Adam` optimizer with default hyperparameters.
    ///
    /// # Parameters
    /// - `params`: tensors to optimise
    /// - `lr`: learning rate (e.g. `1e-3`)
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        let m: Vec<Vec<f32>> = params.iter().map(|p| vec![0.0; p.numel()]).collect();
        let v: Vec<Vec<f32>> = params.iter().map(|p| vec![0.0; p.numel()]).collect();
        Adam {
            params,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            t: 0,
            m,
            v,
        }
    }

    /// Overrides the default β₁ and β₂ decay rates.
    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    /// Overrides the default ε stability constant.
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }
}

impl Optimizer for Adam {
    fn step(&mut self) {
        self.t += 1;
        let t = self.t as f32;

        // Bias-correction denominators
        let bc1 = 1.0 - self.beta1.powf(t);
        let bc2 = 1.0 - self.beta2.powf(t);

        for (i, param) in self.params.iter().enumerate() {
            if let Some(grad) = param.grad() {
                let mut inner = param.inner.write().unwrap();
                for (j, &g) in grad.iter().enumerate() {
                    // Update biased moment estimates
                    self.m[i][j] = self.beta1 * self.m[i][j] + (1.0 - self.beta1) * g;
                    self.v[i][j] = self.beta2 * self.v[i][j] + (1.0 - self.beta2) * g * g;

                    // Bias-corrected estimates
                    let m_hat = self.m[i][j] / bc1;
                    let v_hat = self.v[i][j] / bc2;

                    // Parameter update
                    inner.data[j] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
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
