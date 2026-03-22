//! # Autograd Engine
//!
//! Reverse-mode automatic differentiation (backpropagation).
//!
//! The autograd engine is **not** a separate runtime — it is woven directly
//! into the `Tensor` type. When you perform operations on tensors that have
//! `requires_grad = true`, a dynamic computation graph (DAG) is built
//! transparently. Calling `.backward()` on the scalar loss node then walks
//! this graph in reverse-topological order and accumulates gradients.
//!
//! ## Key Design Points
//!
//! - **Dynamic graph**: built on the fly during the forward pass — no `Graph`
//!   object or `Session` required (eager execution like PyTorch).
//! - **Shared ownership**: `Tensor` is a `Arc<RwLock<TensorData>>` wrapper,
//!   so the backward closure can capture input tensors cheaply.
//! - **Topological sort**: implemented with DFS in `tensor::build_topo`.
//! - **No `unsafe`**: the design relies on Rust's ownership and locking
//!   guarantees — no raw pointers or `unsafe` blocks.
//!
//! The actual implementation lives in `src/tensor/mod.rs` and `src/tensor/ops.rs`.
//! This module re-exports the relevant items for convenience.

// Re-export the Tensor backward entry point.
pub use crate::tensor::Tensor;
