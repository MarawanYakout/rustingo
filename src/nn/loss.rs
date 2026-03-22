//! # Loss Functions
//!
//! All loss functions return a scalar `Tensor` with a correct backward pass.
//!
//! ## Available Losses
//!
//! | Loss | Formula | Use Case |
//! |------|---------|----------|
//! | [`MseLoss`] | `mean((pred − target)²)` | Regression |
//! | [`BceLoss`] | `−mean(y·log(p) + (1−y)·log(1−p))` | Binary classification |

use crate::error::Result;
use crate::tensor::Tensor;

/// Mean Squared Error: `loss = mean((pred − target)²)`.
///
/// # Examples
/// ```
/// use rustingo::nn::MseLoss;
/// use rustingo::Tensor;
///
/// let pred   = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3], true).unwrap();
/// let target = Tensor::from_vec(vec![1.5, 2.0, 2.5], vec![3], false).unwrap();
/// let loss = MseLoss::forward(&pred, &target).unwrap();
/// // loss ≈ (0.25 + 0 + 0.25) / 3 ≈ 0.1667
/// ```
pub struct MseLoss;

impl MseLoss {
    /// Computes `mean((pred − target)²)`.
    ///
    /// # Errors
    /// Propagates shape errors from `sub`.
    pub fn forward(pred: &Tensor, target: &Tensor) -> Result<Tensor> {
        let diff = pred.sub(target)?;
        let sq = diff.pow(2.0);
        Ok(sq.mean())
    }
}

/// Binary Cross-Entropy: `loss = −mean(y·log(p) + (1−y)·log(1−p))`.
///
/// Predictions are **not** passed through sigmoid inside this loss —
/// apply `sigmoid` yourself before calling `forward` if needed.
/// A small ε is used to clamp predictions and avoid `log(0)`.
///
/// # Examples
/// ```
/// use rustingo::nn::BceLoss;
/// use rustingo::Tensor;
///
/// let pred   = Tensor::from_vec(vec![0.8, 0.3], vec![2], true).unwrap();
/// let target = Tensor::from_vec(vec![1.0, 0.0], vec![2], false).unwrap();
/// let loss = BceLoss::forward(&pred, &target).unwrap();
/// ```
pub struct BceLoss;

impl BceLoss {
    /// Computes `−mean(y·log(p) + (1−y)·log(1−p))`.
    ///
    /// Clamps predictions to `[ε, 1−ε]` to prevent `log(0)`.
    ///
    /// # Errors
    /// Propagates shape errors.
    pub fn forward(pred: &Tensor, target: &Tensor) -> Result<Tensor> {
        const EPS: f32 = 1e-7;

        let pred_data = pred.data();
        let target_data = target.data();
        let n = pred_data.len();

        // Clamped predictions
        let clamped: Vec<f32> = pred_data.iter().map(|p| p.clamp(EPS, 1.0 - EPS)).collect();

        // Forward value
        let loss_val: f32 = clamped
            .iter()
            .zip(&target_data)
            .map(|(p, y)| -(y * p.ln() + (1.0 - y) * (1.0 - p).ln()))
            .sum::<f32>()
            / n as f32;

        // We don't need grad if pred doesn't require it.
        if !pred.requires_grad() {
            return Ok(Tensor::from_op(vec![loss_val], vec![], vec![], None));
        }

        // ∂BCE/∂p = (1/n) · (-y/p + (1-y)/(1-p))
        let a = pred.clone();
        let t_cap = target_data;
        let bwd = Box::new(move |grad: &[f32]| {
            let g = grad[0] / n as f32;
            let grad_pred: Vec<f32> = clamped
                .iter()
                .zip(&t_cap)
                .map(|(p, y)| g * (-y / p + (1.0 - y) / (1.0 - p)))
                .collect();
            a.accumulate_grad(&grad_pred);
        });

        Ok(Tensor::from_op(
            vec![loss_val],
            vec![],
            vec![pred.clone()],
            Some(bwd),
        ))
    }
}
