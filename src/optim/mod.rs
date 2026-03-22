//! # Optimizers
//!
//! Gradient-descent optimizers that update trainable parameters in-place
//! after `backward()` has populated `.grad` on each parameter.
//!
//! ## Usage Pattern
//!
//! ```ignore
//! let mut opt = Adam::new(model.parameters(), 1e-3);
//!
//! for (x, y) in dataset {
//!     opt.zero_grad();          // 1. clear old gradients
//!     let pred = model.forward(&x)?;
//!     let loss = MseLoss::forward(&pred, &y)?;
//!     loss.backward();          // 2. compute gradients
//!     opt.step();               // 3. update parameters
//! }
//! ```

pub mod adam;
pub mod sgd;

pub use adam::Adam;
pub use sgd::Sgd;

/// Common interface for all optimizers.
pub trait Optimizer {
    /// Updates all parameters using their accumulated gradients.
    ///
    /// Call after `backward()`.
    fn step(&mut self);

    /// Zeroes the `.grad` field of all managed parameters.
    ///
    /// Call at the **start** of each training step (before `forward`).
    fn zero_grad(&mut self);
}
