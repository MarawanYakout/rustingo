// File Name: matrix.rs
// File Purpose: Low-level Matrix type — flat row-major storage, no autograd.
// Author: Marawan Yakout (M.Y)
// Date Created: 2026-03-13

//! # Matrix
//!
//! A row-major, heap-allocated 2-D numeric array.
//! All element storage is a single `Vec<f64>` for cache efficiency.
//!
//! For gradient-tracked operations use the `Tensor` type instead.

use crate::error::{Error, Result};

/// A row-major 2-D matrix of `f64` values.
///
/// Element at row `r`, column `c` is stored at `data[r * cols + c]`.
///
/// # Examples
/// ```
/// use rustingo::math::Matrix;
///
/// let m = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
/// assert_eq!(m.get(1, 2), 6.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Flat row-major element storage: `data[row * cols + col]`.
    pub data: Vec<f64>,
}

// ─── Constructors ────────────────────────────────────────────────────────────

impl Matrix {
    /// Creates a `Matrix` from a flat iterator of values.
    ///
    /// Accepts any iterator whose items convert into `f64`, so integers work too.
    ///
    /// # Errors
    /// - `InvalidInput` if `rows` or `cols` is zero.
    /// - `SizeMismatch` if the iterator yields a different number of elements
    ///   than `rows * cols`.
    ///
    /// # Examples
    /// ```
    /// use rustingo::math::Matrix;
    ///
    /// // Integer literals work because i32 implements Into<f64>
    /// let m = Matrix::new(2, 3, [1, 2, 3, 4, 5, 6]).unwrap();
    /// ```
    pub fn new<I, T>(rows: usize, cols: usize, data: I) -> Result<Self>
    where
        I: IntoIterator<Item = T>,
        T: Into<f64>,
    {
        if rows == 0 || cols == 0 {
            return Err(Error::InvalidInput(
                "Matrix dimensions must be non-zero".into(),
            ));
        }

        let data: Vec<f64> = data.into_iter().map(Into::into).collect();

        if data.len() != rows * cols {
            return Err(Error::SizeMismatch {
                expected: rows * cols,
                got: data.len(),
            });
        }

        Ok(Matrix { rows, cols, data })
    }

    /// Creates a `Matrix` filled with zeros.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Creates a `Matrix` filled with ones.
    pub fn ones(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![1.0; rows * cols],
        }
    }

    /// Creates a `Matrix` where every element equals `value`.
    pub fn fill(rows: usize, cols: usize, value: f64) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![value; rows * cols],
        }
    }

    /// Creates an identity matrix (1s on diagonal, 0s elsewhere).
    ///
    /// Only valid for square matrices.
    ///
    /// # Examples
    /// ```
    /// use rustingo::math::Matrix;
    ///
    /// let eye = Matrix::identity(3);
    /// assert_eq!(eye.get(0, 0), 1.0);
    /// assert_eq!(eye.get(0, 1), 0.0);
    /// ```
    pub fn identity(size: usize) -> Self {
        let mut m = Matrix::zeros(size, size);
        for i in 0..size {
            m.set(i, i, 1.0);
        }
        m
    }

    /// Creates a `Matrix` from a 2-D `Vec<Vec<f64>>`.
    ///
    /// # Errors
    /// - `InvalidInput` if the outer vec is empty.
    /// - `InvalidInput` if rows have unequal lengths.
    pub fn from_2d(data: Vec<Vec<f64>>) -> Result<Self> {
        if data.is_empty() {
            return Err(Error::InvalidInput(
                "Matrix data must not be empty".into(),
            ));
        }
        let rows = data.len();
        let cols = data[0].len();
        if data.iter().any(|row| row.len() != cols) {
            return Err(Error::InvalidInput(
                "All rows must have the same number of columns".into(),
            ));
        }
        let flat: Vec<f64> = data.into_iter().flatten().collect();
        Ok(Matrix { rows, cols, data: flat })
    }
}

// ─── Element access ──────────────────────────────────────────────────────────

impl Matrix {
    /// Returns the element at `(row, col)`.
    ///
    /// # Panics
    /// Panics if `row >= self.rows` or `col >= self.cols`.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row * self.cols + col]
    }

    /// Sets the element at `(row, col)`.
    ///
    /// # Panics
    /// Panics if `row >= self.rows` or `col >= self.cols`.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        self.data[row * self.cols + col] = value;
    }

    /// Returns an iterator over the elements of a given row.
    pub fn row(&self, r: usize) -> &[f64] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }
}

// ─── Linear algebra ──────────────────────────────────────────────────────────

impl Matrix {
    /// Matrix multiplication: `self × other` → `[m, n]`.
    ///
    /// `self` is `[m, k]`, `other` is `[k, n]`.
    ///
    /// # Errors
    /// Returns `MatmulIncompatible` if `self.cols != other.rows`.
    pub fn dot(&self, other: &Matrix) -> Result<Matrix> {
        if self.cols != other.rows {
            return Err(Error::MatmulIncompatible {
                a_cols: self.cols,
                b_rows: other.rows,
            });
        }

        let (m, k, n) = (self.rows, self.cols, other.cols);
        let mut result = Matrix::zeros(m, n);

        for i in 0..m {
            for l in 0..k {
                let a_il = self.get(i, l);
                for j in 0..n {
                    let prev = result.get(i, j);
                    result.set(i, j, prev + a_il * other.get(l, j));
                }
            }
        }

        Ok(result)
    }

    /// Transposes the matrix: `[m, n]` → `[n, m]`.
    pub fn transpose(&self) -> Matrix {
        let mut result = Matrix::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(j, i, self.get(i, j));
            }
        }
        result
    }

    /// Element-wise addition: `self + other`.
    ///
    /// # Errors
    /// Returns `ShapeMismatch` if dimensions differ.
    pub fn add(&self, other: &Matrix) -> Result<Matrix> {
        self.check_same_shape(other)?;
        let data: Vec<f64> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a + b)
            .collect();
        Ok(Matrix { rows: self.rows, cols: self.cols, data })
    }

    /// Element-wise subtraction: `self - other`.
    ///
    /// # Errors
    /// Returns `ShapeMismatch` if dimensions differ.
    pub fn sub(&self, other: &Matrix) -> Result<Matrix> {
        self.check_same_shape(other)?;
        let data: Vec<f64> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a - b)
            .collect();
        Ok(Matrix { rows: self.rows, cols: self.cols, data })
    }

    /// Multiplies every element by a scalar.
    pub fn scale(&self, scalar: f64) -> Matrix {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|x| x * scalar).collect(),
        }
    }

    /// Returns the sum of all elements.
    pub fn sum(&self) -> f64 {
        self.data.iter().sum()
    }

    /// Returns the mean of all elements.
    pub fn mean(&self) -> f64 {
        self.sum() / (self.rows * self.cols) as f64
    }
}

// ─── Display ─────────────────────────────────────────────────────────────────

impl std::fmt::Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Matrix({}×{}) [", self.rows, self.cols)?;
        for i in 0..self.rows {
            write!(f, "  [")?;
            for j in 0..self.cols {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:.4}", self.get(i, j))?;
            }
            write!(f, "]")?;
            if i < self.rows - 1 {
                write!(f, ",")?;
            }
            writeln!(f)?;
        }
        write!(f, "])")
    }
}

// ─── Private helpers ─────────────────────────────────────────────────────────

impl Matrix {
    fn check_same_shape(&self, other: &Matrix) -> Result<()> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(Error::ShapeMismatch {
                expected: vec![self.rows, self.cols],
                got: vec![other.rows, other.cols],
            });
        }
        Ok(())
    }
}
