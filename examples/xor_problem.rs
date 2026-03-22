//! # XOR Problem — Forward + Backward Demo
//!
//! Trains a 2-layer MLP to learn the XOR function using the rustingo framework.
//!
//! Run with: `cargo run --example xor_problem`

use rustingo::nn::{Linear, MseLoss, Module, Relu, Sequential};
use rustingo::optim::{Adam, Optimizer};
use rustingo::Tensor;

fn main() {
    // ── Dataset: XOR truth table ─────────────────────────────────────────────
    // Input:  (0,0), (1,0), (0,1), (1,1)
    // Output: 0,     1,     1,     0
    let x = Tensor::from_vec(
        vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vec![4, 2],
        false,
    )
    .unwrap();

    let y_target =
        Tensor::from_vec(vec![0.0, 1.0, 1.0, 0.0], vec![4, 1], false).unwrap();

    // ── Model: 2→8→ReLU→1 ───────────────────────────────────────────────────
    let mut model = Sequential::new();
    model.add(Linear::new(2, 8, true, 1).unwrap());
    model.add(Relu);
    model.add(Linear::new(8, 1, true, 2).unwrap());

    println!("Model parameters: {}", model.parameters().len());

    // ── Optimizer: Adam ──────────────────────────────────────────────────────
    let mut opt = Adam::new(model.parameters(), 0.05);

    // ── Training loop ────────────────────────────────────────────────────────
    println!("\n{:<8} {:<12}", "Step", "MSE Loss");
    println!("{}", "─".repeat(22));

    for step in 0..=1000 {
        opt.zero_grad();

        let pred = model.forward(&x).unwrap();
        let loss = MseLoss::forward(&pred, &y_target).unwrap();

        if step % 100 == 0 {
            println!("{:<8} {:.6}", step, loss.item());
        }

        loss.backward();
        opt.step();
    }

    // ── Evaluation ───────────────────────────────────────────────────────────
    println!("\nFinal predictions:");
    println!("{:<15} {:<10} {:<10}", "Input", "Target", "Predicted");
    println!("{}", "─".repeat(37));

    let inputs = [(0.0_f32, 0.0_f32), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)];
    let targets = [0.0_f32, 1.0, 1.0, 0.0];

    for (&(a, b), &t) in inputs.iter().zip(targets.iter()) {
        let xi = Tensor::from_vec(vec![a, b], vec![1, 2], false).unwrap();
        let pred = model.forward(&xi).unwrap();
        println!("({}, {})          {:<10.1} {:.4}", a, b, t, pred.item());
    }
}
