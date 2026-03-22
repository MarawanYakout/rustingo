//! # Vector
//!
//! A 1-D array of `f64` with common linear algebra operations.

use crate::error::{Error, Result};

/// A heap-allocated 1-D vector of `f64`.
///
/// # Examples
/// ```
/// use rustingo::math::Vector;
///
/// let v = Vector::from_vec(vec![1.0, 2.0, 3.0]);
/// assert_eq!(v.dot(&v).unwrap(), 14.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Vector {
    /// Element storage.
    pub data: Vec<f64>,
}

impl Vector {
    /// Creates a vector from a `Vec<f64>`.
    pub fn from_vec(data: Vec<f64>) -> Self {
        Vector { data }
    }

    /// Creates a vector of `n` zeros.
    pub fn zeros(n: usize) -> Self {
        Vector { data: vec![0.0; n] }
    }

    /// Creates a vector of `n` ones.
    pub fn ones(n: usize) -> Self {
        Vector { data: vec![1.0; n] }
    }

    /// Creates a vector filled with `value`.
    pub fn full(n: usize, value: f64) -> Self {
        Vector { data: vec![value; n] }
    }

    /// Returns the number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the vector has no elements.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns element at index `i`.
    #[inline]
    pub fn get(&self, i: usize) -> f64 {
        self.data[i]
    }

    /// Sets element at index `i`.
    #[inline]
    pub fn set(&mut self, i: usize, value: f64) {
        self.data[i] = value;
    }

    /// Dot product (inner product): `Σ self[i] * other[i]`.
    ///
    /// # Errors
    /// Returns `ShapeMismatch` if lengths differ.
    pub fn dot(&self, other: &Vector) -> Result<f64> {
        if self.len() != other.len() {
            return Err(Error::ShapeMismatch {
                expected: vec![self.len()],
                got: vec![other.len()],
            });
        }
        Ok(self.data.iter().zip(&other.data).map(|(a, b)| a * b).sum())
    }

    /// Euclidean (L2) norm: `sqrt(Σ x_i²)`.
    pub fn norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Returns a normalised copy (unit vector). Returns `None` if the norm is zero.
    pub fn normalise(&self) -> Option<Vector> {
        let n = self.norm();
        if n == 0.0 {
            return None;
        }
        Some(Vector {
            data: self.data.iter().map(|x| x / n).collect(),
        })
    }

    /// Element-wise addition.
    ///
    /// # Errors
    /// Returns `ShapeMismatch` if lengths differ.
    pub fn add(&self, other: &Vector) -> Result<Vector> {
        self.check_same_len(other)?;
        Ok(Vector {
            data: self.data.iter().zip(&other.data).map(|(a, b)| a + b).collect(),
        })
    }

    /// Element-wise subtraction.
    ///
    /// # Errors
    /// Returns `ShapeMismatch` if lengths differ.
    pub fn sub(&self, other: &Vector) -> Result<Vector> {
        self.check_same_len(other)?;
        Ok(Vector {
            data: self.data.iter().zip(&other.data).map(|(a, b)| a - b).collect(),
        })
    }

    /// Scales all elements by `scalar`.
    pub fn scale(&self, scalar: f64) -> Vector {
        Vector {
            data: self.data.iter().map(|x| x * scalar).collect(),
        }
    }

    /// Sum of all elements.
    pub fn sum(&self) -> f64 {
        self.data.iter().sum()
    }

    /// Mean of all elements.
    pub fn mean(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }
        self.sum() / self.data.len() as f64
    }

    /// Returns the maximum element value.
    pub fn max(&self) -> Option<f64> {
        self.data.iter().copied().reduce(f64::max)
    }

    /// Returns the minimum element value.
    pub fn min(&self) -> Option<f64> {
        self.data.iter().copied().reduce(f64::min)
    }
}

impl std::fmt::Display for Vector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Vector([")?;
        for (i, v) in self.data.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:.4}", v)?;
        }
        write!(f, "])")
    }
}

// ─── Private helpers ─────────────────────────────────────────────────────────

impl Vector {
    fn check_same_len(&self, other: &Vector) -> Result<()> {
        if self.len() != other.len() {
            return Err(Error::ShapeMismatch {
                expected: vec![self.len()],
                got: vec![other.len()],
            });
        }
        Ok(())
    }
}
