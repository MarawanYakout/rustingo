//! # rustingo — A Pure-Rust Deep Learning Framework
//!
//! `rustingo` is a lightweight, educational deep learning library built from
//! scratch in Rust with **zero external runtime dependencies**.
//!
//! Inspired by PyTorch's dynamic computation graph and autograd engine.
//!
//! ## Quick Start
//!
//! ```rust
//! use rustingo::Tensor;
//! use rustingo::nn::{Linear, Sequential, Relu, MseLoss, Module};
//! use rustingo::optim::{Adam, Optimizer};
//!
//! // Build a 2-layer MLP
//! let mut model = Sequential::new();
//! model.add(Linear::new(2, 4, true, 1).unwrap());
//! model.add(Relu);
//! model.add(Linear::new(4, 1, true, 2).unwrap());
//!
//! let mut opt = Adam::new(model.parameters(), 1e-3);
//!
//! // Training step
//! let x = Tensor::randn(&[8, 2], 42);
//! let y = Tensor::zeros(&[8, 1]);
//!
//! opt.zero_grad();
//! let pred = model.forward(&x).unwrap();
//! let loss = MseLoss::forward(&pred, &y).unwrap();
//! loss.backward();
//! opt.step();
//! ```
//!
//! ## Module Overview
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`tensor`] | `Tensor` type, all math ops, autograd backward |
//! | [`nn`] | `Linear`, `Sequential`, activations, loss functions |
//! | [`optim`] | `Sgd`, `Adam` optimizers |
//! | [`gpu`] | `Device` enum, `KernelDispatch` trait |
//! | [`math`] | Low-level `Matrix` and `Vector` (no autograd) |

// ─── Sub-modules ──────────────────────────────────────────────────────────────

pub mod autograd;
pub mod error;
pub mod gpu;
pub mod math;
pub mod nn;
pub mod optim;
pub mod tensor;

// ─── Top-level re-exports ─────────────────────────────────────────────────────

/// The core data structure — re-exported for ergonomic `use rustingo::Tensor`.
pub use tensor::Tensor;

/// Error and result types.
pub use error::{Error, Result};
