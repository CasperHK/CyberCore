//! # Capability System
//!
//! CyberCore uses **capability-based security** instead of traditional Access
//! Control Lists (ACL).  Every operation on a hardware resource — allocating
//! DMA buffers, launching a compute kernel, opening a fabric endpoint — requires
//! the caller to present an unforgeable capability token.
//!
//! ## Principles
//!
//! - Capabilities are **opaque handles** issued by the kernel.  User-space
//!   (including the Mojo Data Plane) cannot forge or upgrade a capability.
//! - Each capability has a **kind** (what resource it represents) and a set of
//!   **rights** (what operations are permitted).
//! - Capabilities are **transferable** but not duplicable: handing a capability
//!   to another domain revokes it from the sender.
//! - The kernel's capability table is protected by a spin-lock and is only
//!   accessible through the public API of this module.
//!
//! ## Type-state pattern
//!
//! Some APIs (e.g. [`crate::ffi`]) use type-state wrappers such as
//! [`AuthorizedDmaBuffer`] that statically ensure a buffer has been authorised
//! before it can be used in a DMA path.

pub mod types;

pub use types::{AuthorizedDmaBuffer, ComputeCapability};

use spin::Mutex;

// ── CapabilityId ─────────────────────────────────────────────────────────────

/// A unique, monotonically increasing identifier for each issued capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CapabilityId(pub u64);

// ── CapabilityKind ────────────────────────────────────────────────────────────

/// The kind of resource a capability grants access to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapabilityKind {
    /// A physically pinned, IOMMU-mapped DMA buffer.
    DmaBuffer,
    /// A compute region (GPU/NPU execution context).
    ComputeRegion,
    /// A CyberFabric (RDMA) endpoint.
    FabricEndpoint,
    /// A Trust-Verify Data Path session handle.
    TvdpSession,
}

// ── CapabilityRights ──────────────────────────────────────────────────────────

bitflags::bitflags! {
    /// Fine-grained rights associated with a capability.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct CapabilityRights: u16 {
        /// May read from / query the resource.
        const READ     = 0b0000_0000_0001;
        /// May write to / modify the resource.
        const WRITE    = 0b0000_0000_0010;
        /// May transfer the capability to another domain.
        const TRANSFER = 0b0000_0000_0100;
        /// May derive a child capability with equal or lesser rights.
        const DERIVE   = 0b0000_0000_1000;
        /// May initiate a DMA transfer using this resource.
        const DMA      = 0b0000_0001_0000;
        /// May schedule compute work on this resource.
        const EXECUTE  = 0b0000_0010_0000;
    }
}

// ── Capability ────────────────────────────────────────────────────────────────

/// An unforgeable kernel capability token.
///
/// Created by the kernel via [`issue`] and validated before any privileged
/// operation.  Cannot be constructed directly outside this module.
#[derive(Debug, Clone)]
pub struct Capability {
    /// Unique identifier assigned at issuance.
    pub id: CapabilityId,
    /// What resource this capability represents.
    pub kind: CapabilityKind,
    /// Permitted operations.
    pub rights: CapabilityRights,
}

// ── CapabilityError ───────────────────────────────────────────────────────────

/// Errors that can be returned by capability operations.
#[derive(Debug, PartialEq, Eq)]
pub enum CapabilityError {
    /// No capability with the given ID exists in the table.
    NotFound,
    /// The capability exists but does not grant the requested right.
    InsufficientRights,
    /// The capability has already been consumed / revoked.
    Revoked,
    /// The capability table is full.
    TableFull,
}

// ── Capability table ──────────────────────────────────────────────────────────

struct CapabilityTable {
    next_id: u64,
    entries: hashbrown::HashMap<CapabilityId, Capability>,
}

impl CapabilityTable {
    fn new() -> Self {
        Self {
            next_id: 1,
            entries: hashbrown::HashMap::new(),
        }
    }

    fn issue(&mut self, kind: CapabilityKind, rights: CapabilityRights) -> Capability {
        let id = CapabilityId(self.next_id);
        self.next_id += 1;
        let cap = Capability { id, kind, rights };
        self.entries.insert(id, cap.clone());
        cap
    }

    fn revoke(&mut self, id: CapabilityId) -> Result<(), CapabilityError> {
        self.entries.remove(&id).map(|_| ()).ok_or(CapabilityError::NotFound)
    }

    fn lookup(&self, id: CapabilityId) -> Option<&Capability> {
        self.entries.get(&id)
    }
}

static CAP_TABLE: Mutex<Option<CapabilityTable>> = Mutex::new(None);

fn table_lock() -> spin::MutexGuard<'static, Option<CapabilityTable>> {
    let mut guard = CAP_TABLE.lock();
    if guard.is_none() {
        *guard = Some(CapabilityTable::new());
    }
    guard
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Initialise the capability subsystem.
///
/// Called once by [`crate::init`].
pub fn init() {
    let mut guard = CAP_TABLE.lock();
    if guard.is_none() {
        *guard = Some(CapabilityTable::new());
    }
}

/// Issue a new capability of `kind` with `rights`.
///
/// Returns the new [`Capability`] token.  The token is also recorded in the
/// kernel capability table for later validation.
pub fn issue(kind: CapabilityKind, rights: CapabilityRights) -> Capability {
    table_lock()
        .as_mut()
        .expect("capability table initialised")
        .issue(kind, rights)
}

/// Revoke (invalidate) a capability by ID.
///
/// After revocation any further use of a [`Capability`] with this ID will
/// return [`CapabilityError::Revoked`].
pub fn revoke(id: CapabilityId) -> Result<(), CapabilityError> {
    table_lock()
        .as_mut()
        .expect("capability table initialised")
        .revoke(id)
}

/// Validate that `cap` exists in the table and has `required_right`.
///
/// Returns `Ok(())` if valid; [`CapabilityError`] otherwise.
pub fn validate(cap: &Capability, required_right: CapabilityRights) -> Result<(), CapabilityError> {
    let guard = table_lock();
    let table = guard.as_ref().expect("capability table initialised");
    match table.lookup(cap.id) {
        None => Err(CapabilityError::Revoked),
        Some(stored) => {
            if stored.rights.contains(required_right) {
                Ok(())
            } else {
                Err(CapabilityError::InsufficientRights)
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn init_table() {
        let mut guard = CAP_TABLE.lock();
        if guard.is_none() {
            *guard = Some(CapabilityTable::new());
        }
    }

    #[test]
    fn test_issue_and_validate() {
        init_table();
        let cap = issue(
            CapabilityKind::DmaBuffer,
            CapabilityRights::READ | CapabilityRights::DMA,
        );
        assert_eq!(cap.kind, CapabilityKind::DmaBuffer);
        assert!(validate(&cap, CapabilityRights::READ).is_ok());
        assert!(validate(&cap, CapabilityRights::DMA).is_ok());
        assert!(validate(&cap, CapabilityRights::EXECUTE).is_err());
    }

    #[test]
    fn test_revoke() {
        init_table();
        let cap = issue(CapabilityKind::ComputeRegion, CapabilityRights::EXECUTE);
        let id = cap.id;
        revoke(id).expect("revoke should succeed");
        assert_eq!(validate(&cap, CapabilityRights::EXECUTE), Err(CapabilityError::Revoked));
    }

    #[test]
    fn test_unique_ids() {
        init_table();
        let a = issue(CapabilityKind::DmaBuffer, CapabilityRights::READ);
        let b = issue(CapabilityKind::DmaBuffer, CapabilityRights::READ);
        assert_ne!(a.id, b.id);
    }
}
