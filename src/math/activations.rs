//! Activation functions operating on plain `f64` (no autograd).

/// Rectified Linear Unit: `max(0, x)`.
#[inline]
pub fn relu(x: f64) -> f64 {
    x.max(0.0)
}

/// Sigmoid: `1 / (1 + e^{-x})`.
#[inline]
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Hyperbolic tangent.
#[inline]
pub fn tanh(x: f64) -> f64 {
    x.tanh()
}

/// Leaky ReLU: `max(alpha * x, x)`.
#[inline]
pub fn leaky_relu(x: f64, alpha: f64) -> f64 {
    if x > 0.0 { x } else { alpha * x }
}

/// Softmax over a slice — returns a new `Vec<f64>` summing to 1.
pub fn softmax(xs: &[f64]) -> Vec<f64> {
    let max = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = xs.iter().map(|x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}
