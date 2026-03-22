//! Scalar arithmetic utilities.

/// Clamps `x` to the range `[lo, hi]`.
#[inline]
pub fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    x.max(lo).min(hi)
}

/// Linear interpolation: `(1 - t) * a + t * b`.
#[inline]
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}
