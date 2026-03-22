//! # GPU / Device Abstraction Layer
//!
//! Defines the `Device` enum — the single source of truth for where tensor
//! data physically resides.
//!
//! ## Device Hierarchy
//!
//! ```text
//! Device
//! ├── Cpu            — Host RAM, always available
//! └── Gpu(usize)     — GPU ordinal (0 = first GPU, 1 = second, …)
//! ```
//!
//! ## Phase-1 Status
//!
//! All operations currently run on CPU. The `Device` type and `KernelDispatch`
//! trait are in place as hooks — a CUDA/ROCm/Metal backend can be dropped in
//! without changing any public API.

use std::fmt;

// ─── Device enum ─────────────────────────────────────────────────────────────

/// Identifies where a tensor's data physically resides.
///
/// # Examples
/// ```
/// use rustingo::gpu::Device;
///
/// assert!(Device::Cpu.is_cpu());
/// assert!(Device::Gpu(0).is_gpu());
/// assert_eq!(Device::Gpu(2).gpu_index(), Some(2));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Device {
    /// Host (CPU) memory — always supported.
    #[default]
    Cpu,
    /// GPU device with zero-based ordinal index.
    Gpu(usize),
}

impl Device {
    /// Returns `true` if this device is the CPU.
    #[inline(always)]
    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::Cpu)
    }

    /// Returns `true` if this device is any GPU.
    #[inline(always)]
    pub fn is_gpu(&self) -> bool {
        matches!(self, Device::Gpu(_))
    }

    /// Returns the GPU ordinal if this is a GPU device, else `None`.
    pub fn gpu_index(&self) -> Option<usize> {
        match self {
            Device::Gpu(idx) => Some(*idx),
            Device::Cpu => None,
        }
    }

    /// Convenience constructor for CPU.
    pub fn cpu() -> Self {
        Device::Cpu
    }

    /// Convenience constructor for a specific GPU.
    pub fn gpu(idx: usize) -> Self {
        Device::Gpu(idx)
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Gpu(idx) => write!(f, "gpu:{}", idx),
        }
    }
}

// ─── KernelDispatch trait (future GPU hook) ──────────────────────────────────

/// Hook trait for plugging in a hardware-accelerated kernel backend.
///
/// Phase-1: only the CPU reference implementation exists.
/// To add GPU support, implement this trait for a CUDA/ROCm/Metal backend
/// and wire it into the `Tensor` operation dispatch.
///
/// All method signatures receive flat `f32` slices for maximum portability.
pub trait KernelDispatch: Send + Sync {
    /// Element-wise addition: `out[i] = a[i] + b[i]`.
    fn add(&self, a: &[f32], b: &[f32], out: &mut [f32]);

    /// Element-wise multiplication: `out[i] = a[i] * b[i]`.
    fn mul(&self, a: &[f32], b: &[f32], out: &mut [f32]);

    /// Matrix multiplication: `C[m×n] = A[m×k] @ B[k×n]`.
    fn matmul(&self, a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize);
}

/// CPU reference implementation of `KernelDispatch`.
pub struct CpuKernel;

impl KernelDispatch for CpuKernel {
    fn add(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        for ((av, bv), ov) in a.iter().zip(b.iter()).zip(out.iter_mut()) {
            *ov = av + bv;
        }
    }

    fn mul(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        for ((av, bv), ov) in a.iter().zip(b.iter()).zip(out.iter_mut()) {
            *ov = av * bv;
        }
    }

    fn matmul(&self, a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
        for i in 0..m {
            for l in 0..k {
                let a_il = a[i * k + l];
                for j in 0..n {
                    out[i * n + j] += a_il * b[l * n + j];
                }
            }
        }
    }
}
