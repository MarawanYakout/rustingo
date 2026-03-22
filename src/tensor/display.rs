//! # Tensor Display
//!
//! Human-readable formatting for tensors, inspired by NumPy's `__repr__`.

use std::fmt;
use super::Tensor;

impl fmt::Display for Tensor {
    /// Formats the tensor as:
    ///
    /// ```text
    /// Tensor([1.0000, 2.0000, 3.0000], shape=[3], dtype=f32)
    /// Tensor([[1.0000, 2.0000],
    ///         [3.0000, 4.0000]], shape=[2, 2], dtype=f32)
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = self.inner.read().unwrap();
        let shape = &inner.shape;
        let data = &inner.data;

        write!(f, "Tensor(")?;

        match shape.len() {
            // Scalar
            0 => {
                write!(f, "{:.4}", data[0])?;
            }
            // 1-D vector
            1 => {
                write!(f, "[")?;
                for (i, v) in data.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{:.4}", v)?;
                }
                write!(f, "]")?;
            }
            // 2-D matrix
            2 => {
                let (rows, cols) = (shape[0], shape[1]);
                write!(f, "[")?;
                for i in 0..rows {
                    if i > 0 {
                        write!(f, ",\n        ")?;
                    }
                    write!(f, "[")?;
                    for j in 0..cols {
                        if j > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{:.4}", data[i * cols + j])?;
                    }
                    write!(f, "]")?;
                }
                write!(f, "]")?;
            }
            // N-D: show flat data with shape annotation
            _ => {
                write!(f, "[")?;
                for (i, v) in data.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{:.4}", v)?;
                }
                write!(f, "]")?;
            }
        }

        write!(f, ", shape={:?}, dtype=f32", shape)?;

        if inner.requires_grad {
            write!(f, ", requires_grad=true")?;
        }
        if inner.grad.is_some() {
            write!(f, ", grad=Some")?;
        }

        write!(f, ")")
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
