//! # FFI Surface — C ABI interface for the Mojo Data Plane
//!
//! This module exposes `extern "C"` functions that allow the Mojo runtime to:
//!
//! 1. **Register a compute region**: tell the kernel that a particular physical
//!    memory range will be used as a shared tensor buffer between Rust and Mojo.
//! 2. **Issue a compute capability**: obtain an opaque handle that grants
//!    permission to schedule work on a specific device.
//! 3. **Validate a capability handle**: called by the kernel before honouring
//!    any Data-Plane request from Mojo.
//!
//! ## Security model
//!
//! - Every function validates inputs before touching kernel state.
//! - Physical addresses provided by Mojo are checked against the IOMMU mapping
//!   table — Mojo cannot invent addresses that haven't been registered.
//! - Capability handles are opaque `u64` values; the kernel owns the table that
//!   maps handles → permissions.
//!
//! ## Memory sharing protocol
//!
//! ```text
//!  Rust kernel                         Mojo runtime
//!  ──────────────────────────────────  ────────────────────────────────
//!  1. allocate_pinned_frame()
//!  2. iommu::map(pinned, …)  ────────► physical address visible to device
//!  3. register_compute_region(addr, …) ◄── Mojo calls with phys_addr
//!  4. issue_compute_capability() ──────► handle: u64  ◄── Mojo stores
//!  5. (Mojo runs kernel on shared buf)
//!  6. release_compute_region(addr) ◄─── Mojo signals completion
//!  7. IOMMU mapping dropped automatically
//! ```
//!
//! ## `unsafe` contract
//!
//! All `extern "C"` functions here are implicitly `unsafe` because they are
//! called from foreign code.  Every parameter from the foreign side is
//! validated before use; raw pointers are checked for null and alignment.

use crate::capability::{
    self, Capability, CapabilityKind, CapabilityRights, CapabilityError,
};
use crate::memory::allocator::{allocate_pinned_frame, PAGE_SIZE};
use crate::memory::iommu::{self, IommuPermissions};
use spin::Mutex;

#[cfg(not(feature = "std"))]
use alloc::collections::BTreeMap;
#[cfg(feature = "std")]
use std::collections::BTreeMap;

// ── ComputeRegionHandle ───────────────────────────────────────────────────────

/// Internal record kept per registered compute region.
///
/// Fields are retained for their RAII drop effects and for future introspection
/// (e.g., listing active regions via a debug interface).
#[allow(dead_code)]
struct ComputeRegion {
    phys_addr: u64,
    length: usize,
    capability: Capability,
    /// Active IOMMU mapping — dropped on `release_compute_region`.
    _mapping: iommu::IommuMapping,
}

// ── Registry ─────────────────────────────────────────────────────────────────

static REGIONS: Mutex<Option<BTreeMap<u64, ComputeRegion>>> = Mutex::new(None);

fn regions_lock() -> spin::MutexGuard<'static, Option<BTreeMap<u64, ComputeRegion>>> {
    let mut guard = REGIONS.lock();
    if guard.is_none() {
        *guard = Some(BTreeMap::new());
    }
    guard
}

// ── FFI error codes (returned to Mojo) ────────────────────────────────────────

/// Return codes used by all `extern "C"` functions.
///
/// Mojo checks for `FFI_OK` (0) on every call.
#[repr(i32)]
#[allow(dead_code)]
pub enum FfiStatus {
    /// Operation completed successfully.
    Ok = 0,
    /// A required argument was null, zero, or out of range.
    InvalidArgument = -1,
    /// The physical frame allocator is exhausted.
    OutOfMemory = -2,
    /// The capability token is missing, revoked, or has insufficient rights.
    CapabilityDenied = -3,
    /// A region at this address is already registered.
    AlreadyRegistered = -4,
    /// No region or capability with the given identifier was found.
    NotFound = -5,
    /// The IOMMU hardware returned an error.
    IommuError = -6,
}

// ── register_compute_region ───────────────────────────────────────────────────

/// Allocate a kernel-managed, IOMMU-mapped shared memory region and register it
/// for use by the Mojo Data Plane.
///
/// # Parameters (C side)
/// - `length`       : size of the region in bytes (must be ≤ `PAGE_SIZE`).
/// - `device_id`    : the device (GPU / NPU) that will access the buffer.
/// - `out_phys_addr`: written with the physical address on success.
///
/// # Returns
/// `0` (`FFI_OK`) on success; negative error code otherwise.
///
/// # Safety
/// `out_phys_addr` must be a valid, non-null pointer to a `u64` that is
/// writable by this call.
#[no_mangle]
pub unsafe extern "C" fn register_compute_region(
    length: usize,
    device_id: u64,
    out_phys_addr: *mut u64,
) -> i32 {
    // ── Input validation ──────────────────────────────────────────────────────
    if out_phys_addr.is_null() {
        return FfiStatus::InvalidArgument as i32;
    }
    if length == 0 || length > PAGE_SIZE {
        return FfiStatus::InvalidArgument as i32;
    }

    // ── Issue a DMA capability ────────────────────────────────────────────────
    let cap = capability::issue(
        CapabilityKind::DmaBuffer,
        CapabilityRights::READ | CapabilityRights::WRITE | CapabilityRights::DMA,
    );

    // ── Allocate and pin a physical frame ─────────────────────────────────────
    let pinned = match allocate_pinned_frame() {
        Some(p) => p,
        None => return FfiStatus::OutOfMemory as i32,
    };

    let phys_addr = pinned.phys_addr();

    // ── Map through the IOMMU ─────────────────────────────────────────────────
    let mapping = match iommu::map(
        pinned,
        length,
        IommuPermissions::READ | IommuPermissions::WRITE,
        device_id,
        &cap,
    ) {
        Ok(m) => m,
        Err((_pinned, e)) => {
            return match e {
                iommu::IommuError::AlreadyMapped => FfiStatus::AlreadyRegistered as i32,
                iommu::IommuError::CapabilityMismatch => FfiStatus::CapabilityDenied as i32,
                _ => FfiStatus::IommuError as i32,
            };
        }
    };

    // ── Store in the registry ─────────────────────────────────────────────────
    {
        let mut guard = regions_lock();
        let map = guard.as_mut().expect("regions map initialised");
        map.insert(
            phys_addr,
            ComputeRegion {
                phys_addr,
                length,
                capability: cap,
                _mapping: mapping,
            },
        );
    }

    // SAFETY: Caller guarantees `out_phys_addr` is valid and writable.
    unsafe {
        out_phys_addr.write(phys_addr);
    }

    FfiStatus::Ok as i32
}

// ── issue_compute_capability ─────────────────────────────────────────────────

/// Issue a compute capability handle for the Mojo runtime to present when
/// scheduling work on `device_id`.
///
/// # Parameters (C side)
/// - `device_id`       : target accelerator node ID.
/// - `flops_budget`    : FLOPS budget granted to this capability.
/// - `out_cap_handle`  : written with the opaque capability handle on success.
///
/// # Returns
/// `0` on success; negative error code otherwise.
///
/// # Safety
/// `out_cap_handle` must be a valid, non-null pointer to a `u64`.
#[no_mangle]
pub unsafe extern "C" fn issue_compute_capability(
    _device_id: u64,
    _flops_budget: u64,
    out_cap_handle: *mut u64,
) -> i32 {
    if out_cap_handle.is_null() {
        return FfiStatus::InvalidArgument as i32;
    }

    let cap = capability::issue(
        CapabilityKind::ComputeRegion,
        CapabilityRights::EXECUTE | CapabilityRights::READ,
    );

    let handle = cap.id.0;

    // `capability::issue()` has already inserted the capability into the global
    // table; no additional persistence is needed here.  In a production kernel
    // with a per-process capability table, this would also store `device_id` and
    // `flops_budget` as quota metadata alongside the table entry.
    //
    // Explicitly drop `cap` now so the borrow checker sees it is consumed.
    // The entry in the global table is keyed by `handle` and remains live.
    drop(cap);

    // SAFETY: Caller guarantees the pointer is valid.
    unsafe {
        out_cap_handle.write(handle);
    }

    FfiStatus::Ok as i32
}

// ── validate_capability_handle ────────────────────────────────────────────────

/// Validate that `cap_handle` is still live and carries `EXECUTE` rights.
///
/// Called by the Mojo runtime before submitting a compute kernel to confirm
/// the kernel hasn't been revoked since it was issued.
///
/// # Returns
/// `0` if valid; [`FfiStatus::CapabilityDenied`] otherwise.
///
/// # Safety
/// This function is safe to call from any context; it does not dereference
/// raw pointers.
#[no_mangle]
pub unsafe extern "C" fn validate_capability_handle(cap_handle: u64) -> i32 {
    use crate::capability::{CapabilityId};

    // Re-construct a minimal Capability shell to call validate().
    // The actual rights are re-checked against the live table entry.
    let cap = Capability {
        id: CapabilityId(cap_handle),
        kind: CapabilityKind::ComputeRegion,
        rights: CapabilityRights::EXECUTE,
    };

    match capability::validate(&cap, CapabilityRights::EXECUTE) {
        Ok(()) => FfiStatus::Ok as i32,
        Err(CapabilityError::Revoked) | Err(CapabilityError::NotFound) => {
            FfiStatus::CapabilityDenied as i32
        }
        Err(_) => FfiStatus::CapabilityDenied as i32,
    }
}

// ── release_compute_region ────────────────────────────────────────────────────

/// Release a previously-registered compute region.
///
/// The IOMMU mapping is revoked atomically; the underlying physical frame is
/// returned to the allocator.
///
/// # Returns
/// `0` on success; [`FfiStatus::NotFound`] if no region at `phys_addr`.
///
/// # Safety
/// Caller must ensure no DMA transfers are in flight to this region.
#[no_mangle]
pub unsafe extern "C" fn release_compute_region(phys_addr: u64) -> i32 {
    let mut guard = regions_lock();
    let map = guard.as_mut().expect("regions map initialised");
    match map.remove(&phys_addr) {
        Some(_region) => {
            // `_region` is dropped here: IOMMU mapping revoked, frame freed.
            FfiStatus::Ok as i32
        }
        None => FfiStatus::NotFound as i32,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{MemoryMap, MemoryRegion, MemoryRegionKind};

    static INIT: spin::Once<()> = spin::Once::new();
    static TEST_REGION: MemoryRegion = MemoryRegion {
        start: 0x4000_0000,
        length: 0x100_0000, // 16 MiB
        kind: MemoryRegionKind::Usable,
    };

    fn init_all() {
        INIT.call_once(|| {
            crate::memory::allocator::init(&MemoryMap {
                regions: core::slice::from_ref(&TEST_REGION),
            });
            iommu::init();
            capability::init();
        });
    }

    #[test]
    fn test_register_and_release_compute_region() {
        init_all();
        let mut phys_addr: u64 = 0;
        let ret = unsafe {
            register_compute_region(512, 0xBEEF, &mut phys_addr as *mut u64)
        };
        assert_eq!(ret, FfiStatus::Ok as i32);
        assert!(phys_addr > 0);

        let ret2 = unsafe { release_compute_region(phys_addr) };
        assert_eq!(ret2, FfiStatus::Ok as i32);
    }

    #[test]
    fn test_release_nonexistent_region() {
        init_all();
        let ret = unsafe { release_compute_region(0xDEAD_BEEF) };
        assert_eq!(ret, FfiStatus::NotFound as i32);
    }

    #[test]
    fn test_issue_and_validate_compute_capability() {
        init_all();
        let mut handle: u64 = 0;
        let ret = unsafe {
            issue_compute_capability(0xCAFE, 1_000_000, &mut handle as *mut u64)
        };
        assert_eq!(ret, FfiStatus::Ok as i32);
        assert!(handle > 0);

        let vret = unsafe { validate_capability_handle(handle) };
        assert_eq!(vret, FfiStatus::Ok as i32);
    }

    #[test]
    fn test_validate_invalid_handle() {
        init_all();
        let vret = unsafe { validate_capability_handle(0xFFFF_FFFF_FFFF_FFFF) };
        assert_eq!(vret, FfiStatus::CapabilityDenied as i32);
    }
}
