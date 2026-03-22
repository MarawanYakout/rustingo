//! # Neural Network Tests
//!
//! Tests for layers, activations, loss functions, and a mini training loop.

use rustingo::Tensor;
use rustingo::nn::{Linear, MseLoss, Module, Relu, Sequential};
use rustingo::optim::{Adam, Optimizer};

// ─── Linear layer ─────────────────────────────────────────────────────────────

#[test]
fn test_linear_forward_shape() {
    let layer = Linear::new(4, 8, true, 1).unwrap();
    let x = Tensor::randn(&[3, 4], 42);
    let y = layer.forward(&x).unwrap();
    assert_eq!(y.shape(), vec![3, 8]);
}

#[test]
fn test_linear_parameters_count() {
    // With bias: weight [4,8] + bias [8] = 2 parameter tensors
    let layer = Linear::new(4, 8, true, 2).unwrap();
    assert_eq!(layer.parameters().len(), 2);

    // Without bias: just weight
    let layer_nb = Linear::new(4, 8, false, 3).unwrap();
    assert_eq!(layer_nb.parameters().len(), 1);
}

#[test]
fn test_linear_grad_flow() {
    // Check that gradients reach the weight after backward.
    let layer = Linear::new(2, 3, true, 10).unwrap();
    let x = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2], false).unwrap();

    let out = layer.forward(&x).unwrap();
    let loss = MseLoss::forward(&out, &Tensor::zeros(&[1, 3])).unwrap();
    loss.backward();

    // Weight and bias should have gradients now
    assert!(
        layer.weight.grad().is_some(),
        "weight gradient should be populated"
    );
    assert!(
        layer.bias.as_ref().unwrap().grad().is_some(),
        "bias gradient should be populated"
    );
}

// ─── Activations ─────────────────────────────────────────────────────────────

#[test]
fn test_relu_zero_pass() {
    let relu = Relu;
    let x = Tensor::from_vec(vec![-3.0, 0.0, 5.0], vec![3], false).unwrap();
    let y = relu.forward(&x).unwrap();
    assert_eq!(y.data(), vec![0.0, 0.0, 5.0]);
}

#[test]
fn test_leaky_relu() {
    use rustingo::nn::LeakyRelu;
    let act = LeakyRelu::new(0.1);
    let x = Tensor::from_vec(vec![-2.0, 1.0], vec![2], false).unwrap();
    let y = act.forward(&x).unwrap();
    let d = y.data();
    assert!((d[0] - (-0.2)).abs() < 1e-6);
    assert!((d[1] - 1.0).abs() < 1e-6);
}

// ─── Loss functions ──────────────────────────────────────────────────────────

#[test]
fn test_mse_zero_loss() {
    let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3], false).unwrap();
    let target = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3], false).unwrap();
    let loss = MseLoss::forward(&pred, &target).unwrap();
    assert!(loss.item().abs() < 1e-6);
}

#[test]
fn test_mse_known_value() {
    // pred=[1,3], target=[2,2] → diff=[-1,1] → sq=[1,1] → mean=1.0
    let pred = Tensor::from_vec(vec![1.0, 3.0], vec![2], false).unwrap();
    let target = Tensor::from_vec(vec![2.0, 2.0], vec![2], false).unwrap();
    let loss = MseLoss::forward(&pred, &target).unwrap();
    assert!((loss.item() - 1.0).abs() < 1e-5);
}

#[test]
fn test_bce_known_value() {
    use rustingo::nn::BceLoss;
    // pred=0.8, target=1.0 → -ln(0.8) ≈ 0.2231
    let pred = Tensor::from_vec(vec![0.8], vec![1], false).unwrap();
    let target = Tensor::from_vec(vec![1.0], vec![1], false).unwrap();
    let loss = BceLoss::forward(&pred, &target).unwrap();
    assert!((loss.item() - (-0.8_f32.ln())).abs() < 1e-4);
}

// ─── Sequential model ─────────────────────────────────────────────────────────

#[test]
fn test_sequential_forward() {
    let mut model = Sequential::new();
    model.add(Linear::new(3, 5, true, 1).unwrap());
    model.add(Relu);
    model.add(Linear::new(5, 2, true, 2).unwrap());

    let x = Tensor::randn(&[4, 3], 7);
    let y = model.forward(&x).unwrap();
    assert_eq!(y.shape(), vec![4, 2]);
}

#[test]
fn test_sequential_depth() {
    let mut model = Sequential::new();
    model.add(Linear::new(2, 4, true, 1).unwrap());
    model.add(Relu);
    assert_eq!(model.depth(), 2);
}

#[test]
fn test_zero_grad_clears_gradients() {
    let layer = Linear::new(2, 2, true, 42).unwrap();
    let x = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2], false).unwrap();
    let out = layer.forward(&x).unwrap();
    let loss = MseLoss::forward(&out, &Tensor::zeros(&[1, 2])).unwrap();
    loss.backward();

    // Gradients exist after backward
    assert!(layer.weight.grad().is_some());

    // Zero them out
    layer.zero_grad();
    let grad = layer.weight.grad().unwrap();
    assert!(grad.iter().all(|&g| g == 0.0));
}

// ─── Mini training loop (XOR sanity check) ───────────────────────────────────

#[test]
fn test_xor_loss_decreases() {
    // A simple MLP should be able to reduce MSE loss on XOR in < 500 steps.
    // XOR truth table: (0,0)→0, (1,0)→1, (0,1)→1, (1,1)→0
    let x = Tensor::from_vec(
        vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vec![4, 2],
        false,
    )
    .unwrap();
    let y_target = Tensor::from_vec(vec![0.0, 1.0, 1.0, 0.0], vec![4, 1], false).unwrap();

    let mut model = Sequential::new();
    model.add(Linear::new(2, 8, true, 1).unwrap());
    model.add(Relu);
    model.add(Linear::new(8, 1, true, 2).unwrap());

    let mut opt = Adam::new(model.parameters(), 0.05);

    // Record initial loss
    let pred0 = model.forward(&x).unwrap();
    let initial_loss = MseLoss::forward(&pred0, &y_target).unwrap().item();

    // Train for 500 steps
    for _ in 0..500 {
        opt.zero_grad();
        let pred = model.forward(&x).unwrap();
        let loss = MseLoss::forward(&pred, &y_target).unwrap();
        loss.backward();
        opt.step();
    }

    let pred_final = model.forward(&x).unwrap();
    let final_loss = MseLoss::forward(&pred_final, &y_target).unwrap().item();

    assert!(
        final_loss < initial_loss,
        "XOR loss did not decrease: initial={:.4}, final={:.4}",
        initial_loss,
        final_loss
    );
    // Should converge reasonably well
    assert!(
        final_loss < 0.1,
        "XOR loss too high after training: {:.4}",
        final_loss
    );
}
