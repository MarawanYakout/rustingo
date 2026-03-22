//! # Linear (Dense) Layer
//!
//! Implements `y = x @ W + b` where:
//! - `x` is the input  `[batch, in_features]`
//! - `W` is the weight `[in_features, out_features]`
//! - `b` is the bias   `[out_features]`
//!
//! Weight initialisation uses **Kaiming uniform** by default (good for ReLU).

use crate::error::Result;
use crate::tensor::Tensor;
use super::Module;

/// A fully-connected linear transformation layer.
///
/// # Examples
/// ```
/// use rustingo::nn::{Linear, Module};
/// use rustingo::Tensor;
///
/// let layer = Linear::new(4, 8, true, 42).unwrap();
/// let x = Tensor::randn(&[2, 4], 1);
/// let y = layer.forward(&x).unwrap();
/// assert_eq!(y.shape(), vec![2, 8]);
/// ```
pub struct Linear {
    /// Weight matrix — shape `[in_features, out_features]`.
    pub weight: Tensor,
    /// Optional bias vector — shape `[out_features]`.
    pub bias: Option<Tensor>,
    /// Number of input features.
    pub in_features: usize,
    /// Number of output features.
    pub out_features: usize,
}

impl Linear {
    /// Creates a `Linear` layer with **Kaiming uniform** weight init.
    ///
    /// # Parameters
    /// - `in_features`: size of each input sample
    /// - `out_features`: size of each output sample
    /// - `bias`: whether to add a learnable bias term
    /// - `seed`: RNG seed for weight initialisation
    ///
    /// # Errors
    /// Propagates errors from `kaiming_uniform` (shape must be ≥ 1-D).
    pub fn new(
        in_features: usize,
        out_features: usize,
        bias: bool,
        seed: u64,
    ) -> Result<Self> {
        // Weight: [in_features, out_features]
        let weight = Tensor::kaiming_uniform(&[in_features, out_features], seed)?;
        weight.set_requires_grad(true);

        let bias_tensor = if bias {
            // Bias initialised to zeros (common default)
            let b = Tensor::zeros(&[out_features]);
            b.set_requires_grad(true);
            Some(b)
        } else {
            None
        };

        Ok(Linear {
            weight,
            bias: bias_tensor,
            in_features,
            out_features,
        })
    }
}

impl Module for Linear {
    /// Forward pass: `y = x @ W + b`.
    ///
    /// - Input `x` must be `[batch, in_features]` or `[in_features]` (1 sample).
    /// - Output shape is `[batch, out_features]`.
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Ensure input is 2-D by treating a 1-D input as a single-row batch.
        let x = if input.ndim() == 1 {
            let n = input.numel();
            let data = input.data();
            Tensor::from_vec(data, vec![1, n], input.requires_grad())?
        } else {
            input.clone()
        };

        // x: [batch, in]  @  W: [in, out]  →  [batch, out]
        let out = x.matmul(&self.weight)?;

        // Optionally add bias with proper gradient flow.
        if let Some(ref b) = self.bias {
            out.add_bias(b)
        } else {
            Ok(out)
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref b) = self.bias {
            params.push(b.clone());
        }
        params
    }
}
