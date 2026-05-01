//! # High-level capability wrappers (type-state pattern)
//!
//! These wrappers use Rust's **type-state** pattern to make incorrect use of
//! hardware resources a compile-time error.
//!
//! ## Example: DMA buffer lifecycle
//!
//! ```text
//!  allocate_pinned_frame()
//!       │
//!       ▼
//!  AuthorizedDmaBuffer::new(frame, cap)   ← requires DmaBuffer capability
//!       │
//!       ▼
//!  &AuthorizedDmaBuffer → phys_addr, len  ← safe to pass to hardware
//!       │
//!       ▼
//!  drop(buffer)                           ← IOMMU unmapped, frame returned
//! ```
//!
//! Without a valid [`super::Capability`] of kind `DmaBuffer`, the
//! `AuthorizedDmaBuffer` cannot be constructed.

use crate::memory::allocator::PhysicallyPinned;
use super::{Capability, CapabilityKind, CapabilityRights, CapabilityError};

// ── AuthorizedDmaBuffer ───────────────────────────────────────────────────────

/// A physically-pinned memory buffer that has been **authorised** for DMA use.
///
/// Construction requires a [`Capability`] of kind [`CapabilityKind::DmaBuffer`]
/// with at least the [`CapabilityRights::DMA`] right.  This type-state ensures
/// that no unauthenticated buffer can reach the hardware DMA path.
///
/// # RAII contract
/// Dropping an `AuthorizedDmaBuffer` returns the pinned frame to the allocator.
pub struct AuthorizedDmaBuffer {
    /// The capability that authorises this buffer.
    pub capability: Capability,
    /// The underlying pinned physical frame.
    _pinned: PhysicallyPinned,
    /// Length of the accessible region in bytes.
    pub length: usize,
}

impl AuthorizedDmaBuffer {
    /// Construct an `AuthorizedDmaBuffer` from a pinned frame and a capability.
    ///
    /// # Errors
    /// Returns [`CapabilityError::InsufficientRights`] if `cap` does not carry
    /// [`CapabilityRights::DMA`], or [`CapabilityError::Revoked`] if the
    /// capability has been invalidated.
    pub fn new(
        pinned: PhysicallyPinned,
        length: usize,
        cap: Capability,
    ) -> Result<Self, (PhysicallyPinned, CapabilityError)> {
        // Verify kind.
        if cap.kind != CapabilityKind::DmaBuffer {
            return Err((pinned, CapabilityError::InsufficientRights));
        }
        // Verify rights via the global capability table.
        if let Err(e) = super::validate(&cap, CapabilityRights::DMA) {
            return Err((pinned, e));
        }
        Ok(Self {
            capability: cap,
            _pinned: pinned,
            length,
        })
    }

    /// Return the physical address of this buffer.
    ///
    /// # Safety
    /// The physical address is only valid as long as the `AuthorizedDmaBuffer`
    /// is live.  Stashing the address beyond the lifetime of this struct leads
    /// to use-after-free.
    pub fn phys_addr(&self) -> u64 {
        self._pinned.phys_addr()
    }
}

impl core::fmt::Debug for AuthorizedDmaBuffer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AuthorizedDmaBuffer")
            .field("phys_addr", &format_args!("0x{:x}", self.phys_addr()))
            .field("length", &self.length)
            .field("capability_id", &self.capability.id)
            .finish()
    }
}

// ── ComputeCapability ─────────────────────────────────────────────────────────

/// A capability that grants the Mojo Data Plane permission to schedule compute
/// work on a specific GPU / NPU / heterogeneous execution unit.
///
/// Issued by [`crate::ffi::issue_compute_capability`] and passed back to the
/// Mojo runtime as an opaque `u64` handle.
#[derive(Debug, Clone)]
pub struct ComputeCapability {
    /// The underlying kernel capability token.
    pub capability: Capability,
    /// Opaque device identifier (matches the fabric topology node ID).
    pub device_id: u64,
    /// Maximum FLOPS budget the holder is allowed to consume.
    pub flops_budget: u64,
}

impl ComputeCapability {
    /// Construct a `ComputeCapability`.
    ///
    /// # Errors
    /// Returns [`CapabilityError`] if `cap` is not of kind `ComputeRegion` or
    /// does not carry `EXECUTE` rights.
    pub fn new(
        cap: Capability,
        device_id: u64,
        flops_budget: u64,
    ) -> Result<Self, CapabilityError> {
        if cap.kind != CapabilityKind::ComputeRegion {
            return Err(CapabilityError::InsufficientRights);
        }
        super::validate(&cap, CapabilityRights::EXECUTE)?;
        Ok(Self { capability: cap, device_id, flops_budget })
    }

    /// Return the capability ID as a raw `u64` handle for use across the FFI.
    ///
    /// The Mojo runtime stores this handle and presents it back when submitting
    /// compute work.  The kernel validates the handle on every submission.
    pub fn as_raw_handle(&self) -> u64 {
        self.capability.id.0
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability;

    fn init() {
        capability::init();
    }

    /// Allocate a real pinned frame from the physical allocator for use in tests.
    fn make_pinned() -> crate::memory::allocator::PhysicallyPinned {
        use crate::memory::{MemoryMap, MemoryRegion, MemoryRegionKind};
        use crate::memory::allocator;
        static INIT: spin::Once<()> = spin::Once::new();
        static REGION: MemoryRegion = MemoryRegion {
            start: 0x1000_0000,
            length: 0x10_0000, // 1 MiB
            kind: MemoryRegionKind::Usable,
        };
        INIT.call_once(|| {
            allocator::init(&MemoryMap { regions: core::slice::from_ref(&REGION) });
        });
        allocator::allocate_pinned_frame().expect("allocator has frames")
    }

    #[test]
    fn test_authorized_dma_buffer_happy_path() {
        init();
        let cap = capability::issue(
            capability::CapabilityKind::DmaBuffer,
            capability::CapabilityRights::READ
                | capability::CapabilityRights::WRITE
                | capability::CapabilityRights::DMA,
        );
        let pinned = make_pinned();
        let buf = AuthorizedDmaBuffer::new(pinned, 4096, cap)
            .expect("construction should succeed");
        assert!(buf.phys_addr() > 0);
        assert_eq!(buf.length, 4096);
    }

    #[test]
    fn test_authorized_dma_buffer_wrong_kind() {
        init();
        let cap = capability::issue(
            capability::CapabilityKind::ComputeRegion, // wrong kind
            capability::CapabilityRights::DMA,
        );
        let pinned = make_pinned();
        let result = AuthorizedDmaBuffer::new(pinned, 4096, cap);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_capability_happy_path() {
        init();
        let cap = capability::issue(
            capability::CapabilityKind::ComputeRegion,
            capability::CapabilityRights::EXECUTE,
        );
        let cc = ComputeCapability::new(cap, 0xDEAD, 1_000_000)
            .expect("should succeed");
        assert!(cc.as_raw_handle() > 0);
    }
}
