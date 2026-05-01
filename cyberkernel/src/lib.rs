//! # CyberKernel — Control Plane Entry Point
//!
//! CyberCore AIOS Rust Microkernel (`cyberkernel`).
//!
//! ## Architecture overview
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────┐
//! │                    Data Plane (Mojo)                       │
//! │  Matrix ops · AI kernels · Heterogeneous compute dispatch  │
//! └──────────────────────┬────────────────────────────────────┘
//!                        │  C ABI FFI + shared pinned memory
//! ┌──────────────────────▼────────────────────────────────────┐
//! │               Control Plane (cyberkernel / Rust)           │
//! │                                                            │
//! │  ┌──────────┐  ┌────────────┐  ┌──────────┐  ┌────────┐  │
//! │  │ memory/  │  │ capability/│  │ fabric/  │  │ tvdp/  │  │
//! │  │ allocator│  │ cap system │  │RDMA/NIC  │  │ TVDP   │  │
//! │  │ iommu    │  │ auth DMA   │  │ abstract.│  │ verify │  │
//! │  └──────────┘  └────────────┘  └──────────┘  └────────┘  │
//! └───────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Control Plane responsibilities
//! - Memory safety & ownership (allocator, IOMMU mapping)
//! - Capability-based security (no ACL; everything is a capability token)
//! - Hardware Proof-of-Trust signatures (TVDP)
//! - Zero-copy DMA buffer lifecycle management
//! - Safe exposure of compute regions to the Mojo Data Plane via FFI
//!
//! ## `no_std` note
//! This crate is `#![no_std]` by default.  Enable the `std` feature flag for
//! host-side unit / integration tests.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_op_in_unsafe_fn)]
#![deny(missing_docs)]
#![deny(clippy::undocumented_unsafe_blocks)]

// ── In no_std mode we rely on the `alloc` crate for heap types. ──────────────
// A global allocator must be registered by the platform-specific boot shim.
#[cfg(not(feature = "std"))]
extern crate alloc;

// ── Public sub-modules ────────────────────────────────────────────────────────

/// Memory management: physical allocator, virtual address space, IOMMU.
pub mod memory;

/// Capability system: unforgeable tokens that gate access to hardware resources.
pub mod capability;

/// CyberFabric: RDMA / SmartNIC network-as-a-bus abstraction.
pub mod fabric;

/// C ABI FFI surface exposed to the Mojo Data Plane.
pub mod ffi;

/// Trust-Verify Data Path: hardware signature & zero-copy verification.
pub mod tvdp;

// ── Re-exports for convenience ────────────────────────────────────────────────
pub use capability::{Capability, CapabilityError};
pub use memory::iommu::{IommuMapping, IommuError};

/// Initialise all kernel subsystems in dependency order.
///
/// # Safety
/// Must be called exactly **once**, from the boot context, before any other
/// kernel API is used.  Calling it more than once or from multiple concurrent
/// contexts is undefined behaviour.
///
/// # Panics
/// Panics if any mandatory subsystem fails to initialise (e.g. the IOMMU is
/// unreachable).  In a production kernel this would halt with a machine check.
pub fn init(memory_map: &memory::MemoryMap) {
    // 1. Physical memory allocator — must come first so later subsystems can
    //    allocate.
    memory::init(memory_map);

    // 2. IOMMU — must be up before any DMA-capable device is allowed to transfer.
    memory::iommu::init();

    // 3. Capability table — depends on the allocator being ready.
    capability::init();

    // 4. CyberFabric network interface.
    fabric::init();

    // 5. Trust-Verify Data Path.
    tvdp::init();
}
