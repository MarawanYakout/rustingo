//! # Activation Functions
//!
//! Stateless `Module` wrappers around the `Tensor` activation ops.
//! All activations have no trainable parameters.

use crate::error::Result;
use crate::tensor::Tensor;
use super::Module;

/// Rectified Linear Unit: `max(0, x)`.
///
/// # Examples
/// ```
/// use rustingo::nn::{Relu, Module};
/// use rustingo::Tensor;
///
/// let relu = Relu;
/// let x = Tensor::from_vec(vec![-1.0, 0.0, 2.0], vec![3], false).unwrap();
/// let y = relu.forward(&x).unwrap();
/// assert_eq!(y.data(), vec![0.0, 0.0, 2.0]);
/// ```
pub struct Relu;

impl Module for Relu {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(input.relu())
    }
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

/// Logistic sigmoid: `1 / (1 + e^{-x})`.
pub struct Sigmoid;

impl Module for Sigmoid {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(input.sigmoid())
    }
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

/// Hyperbolic tangent: `tanh(x)`.
pub struct Tanh;

impl Module for Tanh {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(input.tanh())
    }
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

/// Leaky ReLU: `max(α · x, x)`.
///
/// Unlike standard ReLU, Leaky ReLU has a non-zero slope for negative inputs,
/// which avoids the "dying ReLU" problem.
///
/// # Examples
/// ```
/// use rustingo::nn::{LeakyRelu, Module};
/// use rustingo::Tensor;
///
/// let act = LeakyRelu::new(0.01);
/// let x = Tensor::from_vec(vec![-2.0, 1.0], vec![2], false).unwrap();
/// let y = act.forward(&x).unwrap();
/// assert!((y.data()[0] - (-0.02)).abs() < 1e-6);
/// ```
pub struct LeakyRelu {
    /// Slope for negative inputs (typically a small positive value like 0.01).
    pub alpha: f32,
}

impl LeakyRelu {
    /// Creates a `LeakyRelu` with slope `alpha` for negative inputs.
    pub fn new(alpha: f32) -> Self {
        LeakyRelu { alpha }
    }
}

impl Module for LeakyRelu {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(input.leaky_relu(self.alpha))
    }
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}
