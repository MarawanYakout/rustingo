//! # Error Types
//!
//! All errors that can arise in the `rustingo` library.
//! Every fallible operation returns `Result<T>` (this module's alias).

use std::fmt;

/// The unified error type for all `rustingo` operations.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// Two tensors have incompatible shapes for the requested operation.
    ///
    /// # Example
    /// Adding a `[2, 3]` tensor to a `[3, 2]` tensor.
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    /// The flat data length does not match the product of the shape.
    SizeMismatch { expected: usize, got: usize },

    /// An operation was attempted on an empty (zero-element) tensor.
    EmptyTensor,

    /// Generic invalid input (bad arguments, zero dimensions, etc.).
    InvalidInput(String),

    /// Division or modulo by zero detected.
    DivisionByZero,

    /// Gradient was requested but is not available.
    ///
    /// Either `backward()` was not called, or the tensor was created with
    /// `requires_grad = false`.
    NoGradient,

    /// Two shapes cannot be broadcast together.
    BroadcastError { a: Vec<usize>, b: Vec<usize> },

    /// Matrix multiplication dimension mismatch: A.cols ≠ B.rows.
    MatmulIncompatible { a_cols: usize, b_rows: usize },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::ShapeMismatch { expected, got } => {
                write!(f, "Shape mismatch: expected {:?}, got {:?}", expected, got)
            }
            Error::SizeMismatch { expected, got } => {
                write!(f, "Size mismatch: expected {} elements, got {}", expected, got)
            }
            Error::EmptyTensor => write!(f, "Tensor is empty (zero elements)"),
            Error::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            Error::DivisionByZero => write!(f, "Division by zero"),
            Error::NoGradient => write!(
                f,
                "No gradient available — call backward() first or set requires_grad=true"
            ),
            Error::BroadcastError { a, b } => {
                write!(f, "Cannot broadcast shapes {:?} and {:?}", a, b)
            }
            Error::MatmulIncompatible { a_cols, b_rows } => {
                write!(
                    f,
                    "Matmul shape error: A.cols={} ≠ B.rows={}",
                    a_cols, b_rows
                )
            }
        }
    }
}

impl std::error::Error for Error {}

/// Convenience alias — every fallible function returns `Result<T>`.
pub type Result<T> = std::result::Result<T, Error>;
