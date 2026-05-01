//! # IOMMU Abstraction Layer
//!
//! The IOMMU (Input-Output Memory Management Unit) is the hardware component
//! responsible for restricting which physical memory addresses a DMA-capable
//! device (NIC, GPU, NVMe …) is allowed to access.
//!
//! ## Role in the zero-copy path
//!
//! ```text
//!  ┌────────────────────┐   iommu::map()   ┌──────────────────────────┐
//!  │ PhysicallyPinned   │ ────────────────► │ IommuMapping             │
//!  │ frame              │                   │ (device-accessible range) │
//!  └────────────────────┘                   └──────────────────────────┘
//!                                                    │ DMA transfer
//!                                           ┌────────▼─────────────────┐
//!                                           │ NIC / GPU / NVMe (device) │
//!                                           └──────────────────────────┘
//! ```
//!
//! ## Capability integration
//!
//! Mapping a frame requires a valid [`crate::capability::Capability`] of kind
//! `DmaBuffer` or `ComputeRegion`.  The capability kind is verified before any
//! hardware page-table modification.
//!
//! ## Platform note
//!
//! This module contains a **portable abstract interface**.  The actual hardware
//! programming (Intel VT-d DMAR, AMD-Vi, ARM SMMU) lives in platform-specific
//! HAL modules that are feature-gated by target architecture.

use spin::Mutex;

use crate::capability::{Capability, CapabilityKind};
use super::allocator::PhysicallyPinned;

// ── IommuPermissions ─────────────────────────────────────────────────────────

bitflags::bitflags! {
    /// Permissions granted to the device for a given IOMMU mapping.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct IommuPermissions: u8 {
        /// Device may read from the mapped physical range.
        const READ  = 0b0000_0001;
        /// Device may write to the mapped physical range.
        const WRITE = 0b0000_0010;
        /// Device may execute from the mapped physical range (rare; GPU shaders).
        const EXEC  = 0b0000_0100;
    }
}

// ── IommuError ───────────────────────────────────────────────────────────────

/// Errors that can occur during IOMMU operations.
#[derive(Debug, PartialEq, Eq)]
pub enum IommuError {
    /// The IOMMU hardware is not present or failed to initialise.
    NotInitialised,
    /// The provided capability does not authorise DMA to this address range.
    CapabilityMismatch,
    /// The physical address range is already mapped (double-map attempt).
    AlreadyMapped,
    /// Hardware returned an error during page-table update.
    HardwareFault,
}

// ── IommuMapping ─────────────────────────────────────────────────────────────

/// An active IOMMU mapping.
///
/// While this value exists, the device identified by `device_id` can access
/// the physical memory region specified at construction time, subject to
/// [`IommuPermissions`].
///
/// # RAII contract
/// Dropping an `IommuMapping` **atomically** revokes the device's access
/// and un-pins the underlying physical frame.
#[derive(Debug)]
pub struct IommuMapping {
    /// Physical start address of the mapped region.
    pub phys_start: u64,
    /// Length of the mapped region in bytes.
    pub length: usize,
    /// Permissions granted to the device.
    pub permissions: IommuPermissions,
    /// Opaque device identifier (PCI BDF, SMMU stream ID, …).
    pub device_id: u64,
    /// The pinned frame backing this mapping (kept alive until unmap).
    _pinned: PhysicallyPinned,
}

impl Drop for IommuMapping {
    fn drop(&mut self) {
        // Revoke the mapping in hardware before releasing the pinned frame.
        // SAFETY: We hold the only outstanding reference to this mapping; the
        //         hardware tables are updated atomically under the IOMMU lock.
        if let Err(_e) = unmap_internal(self.device_id, self.phys_start, self.length) {
            // In production this would trigger a kernel panic / machine check.
            #[cfg(feature = "std")]
            eprintln!(
                "[iommu] WARNING: failed to unmap 0x{:x}: {:?}",
                self.phys_start, _e
            );
        }
        // `_pinned` is dropped here, returning the frame to the allocator.
    }
}

// ── IOMMU state ──────────────────────────────────────────────────────────────

struct IommuState {
    initialised: bool,
    /// Simple accounting of active mappings: phys_start → device_id.
    /// In production, this is backed by hardware page tables.
    active_mappings: hashbrown::HashMap<u64, u64>,
}

impl IommuState {
    fn new() -> Self {
        Self {
            initialised: false,
            active_mappings: hashbrown::HashMap::new(),
        }
    }
}

// `hashbrown::HashMap::new()` is not a `const fn`, so we use a wrapper that
// lazily initialises the map on first lock acquisition.
static IOMMU: Mutex<Option<IommuState>> = Mutex::new(None);

/// Return a locked reference, initialising the inner state on first call.
fn iommu_lock() -> spin::MutexGuard<'static, Option<IommuState>> {
    let mut guard = IOMMU.lock();
    if guard.is_none() {
        *guard = Some(IommuState::new());
    }
    guard
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Initialise the IOMMU subsystem.
///
/// In a real kernel this would:
/// 1. Parse ACPI DMAR/IVRS tables to discover IOMMU units.
/// 2. Enable DMA remapping for all PCIe devices.
/// 3. Set the default deny-all policy.
pub fn init() {
    let mut guard = iommu_lock();
    let state = guard.as_mut().expect("iommu state initialised above");
    // TODO: probe hardware DMAR/IVRS tables via ACPI.
    state.initialised = true;
}

/// Map a pinned physical frame for DMA access by `device_id`.
///
/// # Arguments
/// - `pinned` — The physical frame to expose.  Ownership is transferred into
///   the returned [`IommuMapping`].
/// - `length` — How many bytes of the frame to expose (≤ 4 KiB for a single
///   frame).
/// - `permissions` — Read/write/execute permissions for the device.
/// - `device_id` — Opaque device identifier (e.g., PCI BDF encoded as `u64`).
/// - `capability` — A kernel capability token that authorises this mapping.
///
/// # Errors
/// Returns [`IommuError`] if the IOMMU is not initialised, if the capability
/// is invalid, or if a hardware fault occurs.
pub fn map(
    pinned: PhysicallyPinned,
    length: usize,
    permissions: IommuPermissions,
    device_id: u64,
    capability: &Capability,
) -> Result<IommuMapping, (PhysicallyPinned, IommuError)> {
    // ── 1. Verify the capability authorises DMA access. ───────────────────────
    match capability.kind {
        CapabilityKind::DmaBuffer | CapabilityKind::ComputeRegion => {}
        _ => return Err((pinned, IommuError::CapabilityMismatch)),
    }

    let mut guard = iommu_lock();
    let state = guard.as_mut().expect("iommu state initialised");

    // ── 2. Check the IOMMU is ready. ─────────────────────────────────────────
    if !state.initialised {
        return Err((pinned, IommuError::NotInitialised));
    }

    let phys_start = pinned.phys_addr();

    // ── 3. Detect double-map attempts. ───────────────────────────────────────
    if state.active_mappings.contains_key(&phys_start) {
        return Err((pinned, IommuError::AlreadyMapped));
    }

    // ── 4. Program the hardware page table (stub; replace with real HAL call).─
    // SAFETY: We hold the IOMMU lock and have verified the capability and
    //         the absence of existing mappings.  The pinned frame will not move
    //         or be freed while this mapping is live.
    unsafe {
        hal_iommu_map(device_id, phys_start, length, permissions);
    }

    state.active_mappings.insert(phys_start, device_id);

    Ok(IommuMapping {
        phys_start,
        length,
        permissions,
        device_id,
        _pinned: pinned,
    })
}

// ── Internal helpers ──────────────────────────────────────────────────────────

fn unmap_internal(device_id: u64, phys_start: u64, length: usize) -> Result<(), IommuError> {
    let mut guard = iommu_lock();
    let state = guard.as_mut().expect("iommu state initialised");
    if !state.initialised {
        return Err(IommuError::NotInitialised);
    }
    state.active_mappings.remove(&phys_start);

    // SAFETY: We are the only caller for this mapping (enforced by RAII Drop);
    //         the hardware tables are updated atomically under the IOMMU lock.
    unsafe {
        hal_iommu_unmap(device_id, phys_start, length);
    }
    Ok(())
}

// ── HAL stubs (platform-specific implementations replace these) ───────────────

/// Program the hardware IOMMU to allow `device_id` to access `[phys, phys+len)`.
///
/// # Safety
/// Caller must:
/// - Hold the IOMMU lock.
/// - Ensure `phys_start` is page-aligned.
/// - Ensure `length` ≤ the size of the pinned frame.
/// - Ensure no conflicting mapping exists for this device/address pair.
#[allow(unused_variables)]
unsafe fn hal_iommu_map(
    device_id: u64,
    phys_start: u64,
    length: usize,
    permissions: IommuPermissions,
) {
    // TODO: replace with platform-specific DMAR / SMMU register writes.
    // Example for Intel VT-d:
    //   let dmar = VtdUnit::for_device(device_id);
    //   dmar.add_mapping(phys_start, phys_start, length, permissions.bits());
    //   dmar.flush_iotlb();
}

/// Remove the IOMMU mapping for `device_id` covering `[phys, phys+len)`.
///
/// # Safety
/// Caller must hold the IOMMU lock and ensure the mapping was previously
/// established by [`hal_iommu_map`].
#[allow(unused_variables)]
unsafe fn hal_iommu_unmap(device_id: u64, phys_start: u64, length: usize) {
    // TODO: platform-specific DMAR / SMMU TLB invalidation.
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iommu_state_starts_uninitialised() {
        let state = IommuState::new();
        assert!(!state.initialised);
        assert!(state.active_mappings.is_empty());
    }

    #[test]
    fn test_permissions_flags() {
        let rw = IommuPermissions::READ | IommuPermissions::WRITE;
        assert!(rw.contains(IommuPermissions::READ));
        assert!(rw.contains(IommuPermissions::WRITE));
        assert!(!rw.contains(IommuPermissions::EXEC));
    }
}
