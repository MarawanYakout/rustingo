// src/math/math.rs

// Declaring  all sub-modules - simply tells Rust "these files exist"

// TODO:
// 1- Find a way to make declaration easier for the user.
// 2- Implement that way. :)

pub mod matrix;
pub mod vector;
pub mod stats;
pub mod activations;

// Implementaion Help
//    `use rustingo::math::Matrix` instead of
//    `use rustingo::math::matrix::Matrix`

pub use matrix::Matrix;
pub use vector::Vector;
