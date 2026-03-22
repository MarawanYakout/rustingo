//! # Math Module
//!
//! Low-level linear algebra types (`Matrix`, `Vector`) that underpin the
//! higher-level `Tensor` API. These types are useful when you want a
//! dependency-free, allocation-efficient matrix type without gradient tracking.

pub mod activations;
pub mod arithmetic;
pub mod geometry;
pub mod matrix;
pub mod stats;
pub mod vector;

pub use matrix::Matrix;
pub use vector::Vector;
