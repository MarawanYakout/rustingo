//! # Neural Network Modules (`nn`)
//!
//! PyTorch-style composable building blocks for deep learning models.
//!
//! ## The `Module` Trait
//!
//! Every layer implements `Module`, giving a uniform interface for:
//! - `forward(&self, input) -> Result<Tensor>` — the forward pass
//! - `parameters() -> Vec<Tensor>` — all trainable weights
//! - `zero_grad()` — zeroes gradients before each backward pass
//!
//! ## Available Layers
//!
//! | Type | Description |
//! |------|-------------|
//! | [`Linear`] | Fully-connected layer: `y = xW + b` |
//! | [`Relu`] | Rectified linear unit |
//! | [`Sigmoid`] | Logistic sigmoid |
//! | [`Tanh`] | Hyperbolic tangent |
//! | [`LeakyRelu`] | Leaky ReLU with configurable slope |
//! | [`Sequential`] | Chains layers in order |
//! | [`MseLoss`] | Mean squared error |
//! | [`BceLoss`] | Binary cross-entropy |

pub mod activations;
pub mod linear;
pub mod loss;
pub mod sequential;

pub use activations::{LeakyRelu, Relu, Sigmoid, Tanh};
pub use linear::Linear;
pub use loss::{BceLoss, MseLoss};
pub use sequential::Sequential;

use crate::error::Result;
use crate::tensor::Tensor;

// ─── Module trait ─────────────────────────────────────────────────────────────

/// Base trait for all neural-network modules.
///
/// Implement this trait to make your type usable inside a [`Sequential`]
/// container and compatible with all optimizers.
///
/// # Object Safety
///
/// `Module` is object-safe — you can box it as `Box<dyn Module>`.
pub trait Module {
    /// Runs the forward pass and returns an output tensor.
    ///
    /// # Errors
    /// Propagates shape or numeric errors from underlying tensor operations.
    fn forward(&self, input: &Tensor) -> Result<Tensor>;

    /// Returns all trainable parameter tensors (weights, biases…).
    ///
    /// Used by optimizers to enumerate what to update.
    fn parameters(&self) -> Vec<Tensor>;

    /// Zeroes the `.grad` field of every parameter.
    ///
    /// Call this at the start of each training step, before `backward()`.
    fn zero_grad(&self) {
        for p in self.parameters() {
            p.zero_grad();
        }
    }
}
