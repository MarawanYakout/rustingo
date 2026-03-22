//! # Linear Regression — Full Training Loop Demo
//!
//! Learns `y = 2x + 1` from noisy samples using a single Linear layer.
//!
//! Run with: `cargo run --example linear_regression`

use rustingo::nn::{Linear, MseLoss, Module};
use rustingo::optim::{Adam, Optimizer};
use rustingo::tensor::init::Rng;
use rustingo::Tensor;

fn main() {
    // ── Generate dataset: y = 2x + 1 + noise ────────────────────────────────
    let n_samples = 20;
    let mut rng = Rng::new(123);

    let mut x_data = Vec::with_capacity(n_samples);
    let mut y_data = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let x: f32 = rng.uniform(-2.0, 2.0);
        let noise: f32 = rng.next_normal_f32() * 0.1;
        x_data.push(x);
        y_data.push(2.0 * x + 1.0 + noise);
    }

    let x = Tensor::from_vec(x_data, vec![n_samples, 1], false).unwrap();
    let y_target = Tensor::from_vec(y_data, vec![n_samples, 1], false).unwrap();

    // ── Model: 1-feature → 1-output (effectively learns W and b) ────────────
    let layer = Linear::new(1, 1, true, 42).unwrap();
    let mut opt = Adam::new(layer.parameters(), 0.1);

    // ── Training loop ────────────────────────────────────────────────────────
    println!("{:<8} {:<12}", "Step", "MSE Loss");
    println!("{}", "─".repeat(22));

    for step in 0..=500 {
        opt.zero_grad();
        let pred = layer.forward(&x).unwrap();
        let loss = MseLoss::forward(&pred, &y_target).unwrap();

        if step % 50 == 0 {
            println!("{:<8} {:.6}", step, loss.item());
        }

        loss.backward();
        opt.step();
    }

    // ── Learned parameters ───────────────────────────────────────────────────
    let w = layer.weight.data()[0];
    let b = layer.bias.as_ref().unwrap().data()[0];
    println!("\nLearned: y = {:.4}x + {:.4}", w, b);
    println!("True:    y = 2.0000x + 1.0000");
    println!(
        "Weight error: {:.4}, Bias error: {:.4}",
        (w - 2.0).abs(),
        (b - 1.0).abs()
    );
}
