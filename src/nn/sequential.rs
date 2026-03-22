//! # Sequential Container
//!
//! A simple ordered container that applies layers one after another.
//! Equivalent to `torch.nn.Sequential`.

use crate::error::Result;
use crate::tensor::Tensor;
use super::Module;

/// Applies a list of modules sequentially: `output = layerN(... layer2(layer1(input)) ...)`.
///
/// # Examples
/// ```
/// use rustingo::nn::{Sequential, Linear, Relu, Module};
/// use rustingo::Tensor;
///
/// let mut model = Sequential::new();
/// model.add(Linear::new(2, 4, true, 1).unwrap());
/// model.add(Relu);
/// model.add(Linear::new(4, 1, true, 2).unwrap());
///
/// let x = Tensor::randn(&[3, 2], 42);
/// let y = model.forward(&x).unwrap();
/// assert_eq!(y.shape(), vec![3, 1]);
/// ```
pub struct Sequential {
    layers: Vec<Box<dyn Module + Send + Sync>>,
}

impl Sequential {
    /// Creates an empty `Sequential` model.
    pub fn new() -> Self {
        Sequential { layers: Vec::new() }
    }

    /// Appends a layer to the end of the sequence.
    pub fn add<M: Module + Send + Sync + 'static>(&mut self, layer: M) {
        self.layers.push(Box::new(layer));
    }

    /// Returns the number of layers.
    pub fn depth(&self) -> usize {
        self.layers.len()
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Sequential::new()
    }
}

impl Module for Sequential {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = input.clone();
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}
