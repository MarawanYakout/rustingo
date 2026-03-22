//! Basic statistical functions over slices.

/// Returns the mean of a slice.
pub fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() { return 0.0; }
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// Population variance.
pub fn variance(xs: &[f64]) -> f64 {
    if xs.is_empty() { return 0.0; }
    let m = mean(xs);
    xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64
}

/// Population standard deviation.
pub fn std_dev(xs: &[f64]) -> f64 {
    variance(xs).sqrt()
}
