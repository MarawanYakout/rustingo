//! # Tensor Initialisers & Constructors
//!
//! Factory functions for creating tensors with common fill patterns,
//! and weight-initialisation schemes used in neural network training.
//!
//! ## Random Number Generation
//!
//! All random initialisers use a built-in **Xorshift64** generator — no
//! external crate required.  Pass a `seed` for reproducibility, or use
//! `0` to get a default seed.

use crate::error::{Error, Result};
use super::Tensor;

// ─── Pseudo-random number generator ─────────────────────────────────────────

/// A fast, deterministic Xorshift64 PRNG.
///
/// Produces statistically good pseudo-random numbers with no external deps.
/// Not cryptographically secure — for ML weight init only.
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Creates a new RNG with the given seed.
    ///
    /// A seed of `0` is replaced with a non-zero constant to avoid the
    /// degenerate all-zeros state of the Xorshift algorithm.
    pub fn new(seed: u64) -> Self {
        Rng {
            state: if seed == 0 { 0x853c_49e6_748f_ea9b } else { seed },
        }
    }

    /// Returns the next pseudo-random `u64`.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        // Xorshift64 — Marsaglia 2003
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Returns a uniform float in `[0.0, 1.0)`.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        // Map the top 23 mantissa bits to [1.0, 2.0) then subtract 1.
        let bits = 0x3F80_0000u32 | ((self.next_u64() >> 41) as u32);
        f32::from_bits(bits) - 1.0
    }

    /// Returns a standard-normal `N(0,1)` float via the Box–Muller transform.
    pub fn next_normal_f32(&mut self) -> f32 {
        loop {
            let u1 = self.next_f32();
            let u2 = self.next_f32();
            if u1 > 0.0 {
                let r = (-2.0 * u1.ln()).sqrt();
                return r * (2.0 * std::f32::consts::PI * u2).cos();
            }
        }
    }

    /// Returns a float in `[lo, hi)`.
    #[inline]
    pub fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_f32() * (hi - lo)
    }
}

// ─── Constructors ────────────────────────────────────────────────────────────

impl Tensor {
    /// Creates a tensor filled with **zeros**.
    ///
    /// # Examples
    /// ```
    /// use rustingo::Tensor;
    /// let t = Tensor::zeros(&[2, 3]);
    /// assert_eq!(t.data(), vec![0.0; 6]);
    /// ```
    pub fn zeros(shape: &[usize]) -> Self {
        let n = size_of(shape);
        Tensor::from_op(vec![0.0_f32; n], shape.to_vec(), vec![], None)
    }

    /// Creates a tensor filled with **ones**.
    pub fn ones(shape: &[usize]) -> Self {
        let n = size_of(shape);
        Tensor::from_op(vec![1.0_f32; n], shape.to_vec(), vec![], None)
    }

    /// Creates a tensor where every element equals `value`.
    pub fn full(shape: &[usize], value: f32) -> Self {
        let n = size_of(shape);
        Tensor::from_op(vec![value; n], shape.to_vec(), vec![], None)
    }

    /// Creates a tensor of **uniform random** values in `[0.0, 1.0)`.
    ///
    /// # Parameters
    /// - `seed`: Xorshift seed. Use `0` for a default non-zero seed.
    pub fn rand(shape: &[usize], seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let n = size_of(shape);
        let data: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
        Tensor::from_op(data, shape.to_vec(), vec![], None)
    }

    /// Creates a tensor of **standard-normal** random values `N(0, 1)`.
    ///
    /// # Parameters
    /// - `seed`: Xorshift seed.
    pub fn randn(shape: &[usize], seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let n = size_of(shape);
        let data: Vec<f32> = (0..n).map(|_| rng.next_normal_f32()).collect();
        Tensor::from_op(data, shape.to_vec(), vec![], None)
    }

    /// **Xavier / Glorot uniform** initialisation.
    ///
    /// Fills with values from `U[ -limit, +limit ]` where
    /// `limit = sqrt(6 / (fan_in + fan_out))`.
    ///
    /// Designed for layers with `tanh` or `sigmoid` activations.
    ///
    /// # Parameters
    /// - `shape`: must have ≥ 2 dimensions. `shape[-2]` = fan_in, `shape[-1]` = fan_out.
    /// - `seed`: RNG seed.
    ///
    /// # Errors
    /// Returns `InvalidInput` if `shape.len() < 2`.
    pub fn xavier_uniform(shape: &[usize], seed: u64) -> Result<Self> {
        if shape.len() < 2 {
            return Err(Error::InvalidInput(
                "xavier_uniform requires at least a 2-D shape [fan_in, fan_out]".into(),
            ));
        }
        let fan_in = shape[shape.len() - 2];
        let fan_out = shape[shape.len() - 1];
        let limit = (6.0_f32 / (fan_in + fan_out) as f32).sqrt();

        let mut rng = Rng::new(seed);
        let n = size_of(shape);
        let data: Vec<f32> = (0..n).map(|_| rng.uniform(-limit, limit)).collect();
        Ok(Tensor::from_op(data, shape.to_vec(), vec![], None))
    }

    /// **Kaiming / He uniform** initialisation.
    ///
    /// Fills with values from `U[ -limit, +limit ]` where
    /// `limit = sqrt(3 / fan_in)`.
    ///
    /// Designed for layers with `ReLU` activations.
    ///
    /// # Parameters
    /// - `shape`: `shape[0]` is treated as `fan_in`.
    /// - `seed`: RNG seed.
    ///
    /// # Errors
    /// Returns `InvalidInput` if `shape` is empty.
    pub fn kaiming_uniform(shape: &[usize], seed: u64) -> Result<Self> {
        if shape.is_empty() {
            return Err(Error::InvalidInput(
                "kaiming_uniform requires at least a 1-D shape".into(),
            ));
        }
        let fan_in = shape[0];
        let limit = (3.0_f32 / fan_in as f32).sqrt();

        let mut rng = Rng::new(seed);
        let n = size_of(shape);
        let data: Vec<f32> = (0..n).map(|_| rng.uniform(-limit, limit)).collect();
        Ok(Tensor::from_op(data, shape.to_vec(), vec![], None))
    }

    /// Creates a 1-D tensor: `[start, start+step, ..., < end]`.
    ///
    /// # Errors
    /// Returns `InvalidInput` if `step == 0.0`.
    pub fn arange(start: f32, end: f32, step: f32) -> Result<Self> {
        if step == 0.0 {
            return Err(Error::InvalidInput("arange: step must be non-zero".into()));
        }
        let n = ((end - start) / step).ceil().max(0.0) as usize;
        let data: Vec<f32> = (0..n).map(|i| start + i as f32 * step).collect();
        Ok(Tensor::from_op(data, vec![n], vec![], None))
    }

    /// Creates a 1-D tensor of `n` evenly-spaced values in `[start, end]` (inclusive).
    ///
    /// # Errors
    /// Returns `InvalidInput` if `n < 2`.
    pub fn linspace(start: f32, end: f32, n: usize) -> Result<Self> {
        if n < 2 {
            return Err(Error::InvalidInput(
                "linspace: n must be ≥ 2".into(),
            ));
        }
        let step = (end - start) / (n - 1) as f32;
        let data: Vec<f32> = (0..n).map(|i| start + i as f32 * step).collect();
        Ok(Tensor::from_op(data, vec![n], vec![], None))
    }
}

// ─── Internal helper ─────────────────────────────────────────────────────────

/// Returns the total number of elements for a shape.
/// An empty shape (scalar) has 1 element.
#[inline]
fn size_of(shape: &[usize]) -> usize {
    if shape.is_empty() {
        1
    } else {
        shape.iter().product()
    }
}
