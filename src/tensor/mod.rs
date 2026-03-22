//! # Tensor — Core Data Structure
//!
//! A `Tensor` is the fundamental unit of this framework, analogous to
//! `torch.Tensor` in PyTorch. It wraps:
//!
//! - **Data** — flat `Vec<f32>` stored in row-major order
//! - **Shape** — `Vec<usize>` describing dimensions. Scalar = `vec![]`.
//! - **Gradient** — optional `Vec<f32>` accumulated during `backward()`
//! - **Compute graph node** — stores parent tensors and a backward function
//!
//! ## Thread Safety
//!
//! `Tensor` is cheaply cloneable (it's an `Arc` wrapper) and `Send + Sync`,
//! allowing use across threads. Gradient accumulation is guarded by `RwLock`.
//!
//! ## Usage
//! ```
//! use rustingo::Tensor;
//!
//! let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3], true).unwrap();
//! let y = x.pow(2.0);
//! y.sum().backward();
//! // x.grad() == Some([2.0, 4.0, 6.0])
//! ```

pub mod display;
pub mod init;
pub mod ops;

use std::collections::HashSet;
use std::sync::{Arc, RwLock};

use crate::error::{Error, Result};

/// Type alias for the backward function stored in each tensor node.
///
/// Given the output gradient `&[f32]`, the closure accumulates gradients
/// into the input tensors that were captured at operation time.
pub(crate) type BackwardFn = Box<dyn Fn(&[f32]) + Send + Sync>;

// ─── Internal data ─────────────────────────────────────────────────────────

/// The inner storage of a `Tensor`, shared via `Arc<RwLock<>>`.
///
/// Never access this directly from outside the `tensor` module —
/// use the public `Tensor` API instead.
pub(crate) struct TensorData {
    /// Flat, row-major element storage.
    pub(crate) data: Vec<f32>,

    /// Logical shape. Empty vec = scalar (1 element).
    pub(crate) shape: Vec<usize>,

    /// Accumulated gradient (same length as `data`).
    /// `None` until the first gradient contribution arrives.
    pub(crate) grad: Option<Vec<f32>>,

    /// Whether this tensor participates in gradient computation.
    pub(crate) requires_grad: bool,

    /// The backward function for this tensor's creation op.
    ///
    /// Given the output gradient (`&[f32]`), it accumulates gradients
    /// into the input tensors captured in the closure.
    pub(crate) backward_fn: Option<BackwardFn>,

    /// Parent tensors in the DAG — used solely for topological sort.
    pub(crate) parents: Vec<Tensor>,
}

// ─── Public handle ──────────────────────────────────────────────────────────

/// A multi-dimensional array with optional automatic differentiation.
///
/// Cheap to clone — each clone shares the same underlying data.
///
/// # Gradient tracking
///
/// Set `requires_grad = true` on leaf tensors (weights, inputs you want
/// gradients for). Call `.backward()` on the scalar loss to propagate.
pub struct Tensor {
    pub(crate) inner: Arc<RwLock<TensorData>>,
}

// Safety: TensorData is guarded by RwLock and all contained types are Send+Sync.
unsafe impl Send for Tensor {}
unsafe impl Sync for Tensor {}

impl Clone for Tensor {
    /// Clones the handle — shares the **same** underlying data (like `Arc::clone`).
    fn clone(&self) -> Self {
        Tensor {
            inner: Arc::clone(&self.inner),
        }
    }
}

// ─── Construction ───────────────────────────────────────────────────────────

impl Tensor {
    /// Creates a **leaf tensor** from a flat `Vec<f32>` and a shape.
    ///
    /// A leaf tensor has no parent operation. It is the entry point for
    /// user-provided data and trainable parameters.
    ///
    /// # Errors
    /// Returns `SizeMismatch` if `data.len() != shape.iter().product()`.
    ///
    /// # Examples
    /// ```
    /// use rustingo::Tensor;
    ///
    /// // 2×3 matrix with gradient tracking
    /// let t = Tensor::from_vec(vec![1.0; 6], vec![2, 3], true).unwrap();
    /// assert_eq!(t.shape(), vec![2, 3]);
    /// ```
    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> Result<Self> {
        let expected = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };
        if data.len() != expected {
            return Err(Error::SizeMismatch {
                expected,
                got: data.len(),
            });
        }
        Ok(Tensor {
            inner: Arc::new(RwLock::new(TensorData {
                data,
                shape,
                grad: None,
                requires_grad,
                backward_fn: None,
                parents: Vec::new(),
            })),
        })
    }

    /// Creates a **scalar** leaf tensor from a single `f32`.
    ///
    /// # Examples
    /// ```
    /// use rustingo::Tensor;
    /// let s = Tensor::scalar(3.14, false);
    /// assert_eq!(s.shape(), vec![]);
    /// assert_eq!(s.item(), 3.14_f32);
    /// ```
    pub fn scalar(value: f32, requires_grad: bool) -> Self {
        Tensor {
            inner: Arc::new(RwLock::new(TensorData {
                data: vec![value],
                shape: vec![],
                grad: None,
                requires_grad,
                backward_fn: None,
                parents: Vec::new(),
            })),
        }
    }

    /// Internal: creates a tensor produced by an operation.
    ///
    /// `requires_grad` is automatically set to `true` if any parent requires it.
    pub(crate) fn from_op(
        data: Vec<f32>,
        shape: Vec<usize>,
        parents: Vec<Tensor>,
        backward_fn: Option<BackwardFn>,
    ) -> Self {
        let requires_grad = parents.iter().any(|p| p.requires_grad());
        Tensor {
            inner: Arc::new(RwLock::new(TensorData {
                data,
                shape,
                grad: None,
                requires_grad,
                backward_fn,
                parents,
            })),
        }
    }
}

// ─── Accessors ──────────────────────────────────────────────────────────────

impl Tensor {
    /// Returns the shape as a `Vec<usize>`. Empty = scalar.
    pub fn shape(&self) -> Vec<usize> {
        self.inner.read().unwrap().shape.clone()
    }

    /// Returns the number of elements (product of shape dimensions).
    pub fn numel(&self) -> usize {
        let inner = self.inner.read().unwrap();
        inner.data.len()
    }

    /// Returns the number of dimensions (`0` for scalars).
    pub fn ndim(&self) -> usize {
        self.inner.read().unwrap().shape.len()
    }

    /// Returns a flat clone of the data.
    pub fn data(&self) -> Vec<f32> {
        self.inner.read().unwrap().data.clone()
    }

    /// Returns the scalar value. Panics if this is not a scalar.
    pub fn item(&self) -> f32 {
        let inner = self.inner.read().unwrap();
        assert!(
            inner.shape.is_empty() || inner.data.len() == 1,
            "item() called on non-scalar tensor with shape {:?}",
            inner.shape
        );
        inner.data[0]
    }

    /// Returns a clone of the gradient, or `None` if no gradient has been accumulated.
    pub fn grad(&self) -> Option<Vec<f32>> {
        self.inner.read().unwrap().grad.clone()
    }

    /// Whether this tensor participates in gradient computation.
    pub fn requires_grad(&self) -> bool {
        self.inner.read().unwrap().requires_grad
    }

    /// Enables or disables gradient tracking for this tensor.
    ///
    /// Use to convert a parameter to a non-trainable buffer, or vice versa.
    pub fn set_requires_grad(&self, flag: bool) {
        self.inner.write().unwrap().requires_grad = flag;
    }

    /// Zeroes the accumulated gradient (sets it to all zeros).
    ///
    /// Call on all parameters before each backward pass to avoid accumulation
    /// across steps. `Optimizer::zero_grad()` does this automatically.
    pub fn zero_grad(&self) {
        let mut inner = self.inner.write().unwrap();
        if let Some(ref mut g) = inner.grad {
            for x in g.iter_mut() {
                *x = 0.0;
            }
        }
        // If grad was None, leave it None — nothing to zero.
    }

    /// Accumulates an incoming gradient into this tensor's `.grad` field.
    ///
    /// If `.grad` is `None`, initialises it with `incoming`.
    /// If `.grad` is `Some(g)`, adds `incoming` element-wise.
    /// Silently does nothing if `requires_grad = false`.
    pub(crate) fn accumulate_grad(&self, incoming: &[f32]) {
        let mut inner = self.inner.write().unwrap();
        if !inner.requires_grad {
            return;
        }
        match &mut inner.grad {
            Some(g) => {
                for (a, b) in g.iter_mut().zip(incoming.iter()) {
                    *a += b;
                }
            }
            None => {
                inner.grad = Some(incoming.to_vec());
            }
        }
    }
}

// ─── Backward pass ──────────────────────────────────────────────────────────

impl Tensor {
    /// Runs **reverse-mode automatic differentiation** from this tensor.
    ///
    /// 1. Builds a topological ordering of the computation graph (DFS from `self`)
    /// 2. Seeds this tensor's gradient with all-ones (or all-ones for scalars)
    /// 3. Walks in reverse order, calling each node's backward function
    ///
    /// After `backward()`:
    /// - Every leaf tensor with `requires_grad = true` that contributed to
    ///   `self` has its `.grad` populated.
    ///
    /// # Examples
    /// ```
    /// use rustingo::Tensor;
    ///
    /// let x = Tensor::from_vec(vec![2.0], vec![], true).unwrap();
    /// let y = x.pow(2.0);   // y = x²
    /// y.backward();
    /// assert_eq!(x.grad().unwrap()[0], 4.0); // dy/dx = 2x = 4
    /// ```
    pub fn backward(&self) {
        // ── Step 1: Build topological order (post-order DFS) ──────────────────
        let mut topo: Vec<Tensor> = Vec::new();
        let mut visited: HashSet<usize> = HashSet::new();

        build_topo(self, &mut topo, &mut visited);

        // ── Step 2: Seed the output gradient with 1s ──────────────────────────
        {
            let mut inner = self.inner.write().unwrap();
            let n = inner.data.len();
            inner.grad = Some(vec![1.0_f32; n]);
        }

        // ── Step 3: Walk in reverse topological order ─────────────────────────
        for t in topo.iter().rev() {
            // Snapshot the current gradient for this node.
            let grad_opt = {
                let inner = t.inner.read().unwrap();
                inner.grad.clone()
            };

            if let Some(g) = grad_opt {
                // Call the backward function while holding the read lock.
                // The function writes ONLY to parent tensors, not to `t`,
                // so there is no deadlock.
                let inner = t.inner.read().unwrap();
                if let Some(ref bf) = inner.backward_fn {
                    bf(&g);
                }
            }
        }
    }
}

/// Recursive DFS post-order traversal to build topological order.
///
/// Each unique tensor (identified by its `Arc` pointer) is visited once.
/// A tensor appears in `topo` *after* all of its parents.
fn build_topo(t: &Tensor, topo: &mut Vec<Tensor>, visited: &mut HashSet<usize>) {
    // Use the raw pointer of the Arc as a unique identity.
    let id = Arc::as_ptr(&t.inner) as usize;
    if visited.contains(&id) {
        return;
    }
    visited.insert(id);

    // Visit parents first (depth-first).
    let parents = t.inner.read().unwrap().parents.clone();
    for p in &parents {
        build_topo(p, topo, visited);
    }

    // Push self AFTER all parents (post-order = correct topological order).
    topo.push(t.clone());
}
