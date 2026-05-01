//! # CyberFabric — Network-as-a-Bus abstraction
//!
//! CyberFabric is the control-plane view of the high-speed network fabric that
//! connects CPUs, GPUs, NPUs, SmartNICs and storage nodes.  The key insight is:
//!
//! > *The network is the system bus.*
//!
//! Traditional OS designs treat the network as an I/O peripheral.  CyberCore
//! inverts this: every compute and storage node is a peer on the fabric, and
//! memory can flow between them **without CPU involvement** (RDMA / P2P DMA).
//!
//! ## Responsibilities of this module (Control Plane)
//!
//! - Maintain a topology map of fabric nodes.
//! - Issue and validate [`crate::capability::Capability`] tokens for RDMA
//!   endpoints.
//! - Set up QP (Queue Pair) descriptors that the SmartNIC can use autonomously.
//! - Enforce IOMMU policies on fabric buffers.
//!
//! ## What this module does NOT do (Data Plane — Mojo side)
//!
//! - Execute the actual RDMA read/write operations.
//! - Handle packet scheduling or congestion control.
//! - Drive tensor data between GPU HBM and remote storage.

use spin::Mutex;

use crate::capability::{Capability, CapabilityKind, CapabilityRights};

// ── NodeId ────────────────────────────────────────────────────────────────────

/// Opaque identifier for a fabric node (CPU socket, GPU, NVMe controller, …).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u64);

// ── NodeKind ──────────────────────────────────────────────────────────────────

/// The role a node plays in the fabric.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    /// Host CPU (runs the Rust microkernel).
    Cpu,
    /// GPU / NPU accelerator.
    Accelerator,
    /// SmartNIC with onboard compute (e.g. BlueField DPU).
    SmartNic,
    /// Non-volatile memory target (NVMe-oF / CXL).
    Storage,
}

// ── FabricEndpoint ────────────────────────────────────────────────────────────

/// A registered fabric endpoint: a node that can participate in zero-copy
/// P2P DMA transfers.
#[derive(Debug, Clone)]
pub struct FabricEndpoint {
    /// The node this endpoint belongs to.
    pub node_id: NodeId,
    /// Kind of device at the other end.
    pub node_kind: NodeKind,
    /// Base physical address of the node's registered memory region.
    pub mr_base: u64,
    /// Length of the memory region in bytes.
    pub mr_length: usize,
    /// Capability token gating access to this endpoint.
    pub capability: Capability,
}

// ── FabricError ───────────────────────────────────────────────────────────────

/// Errors from fabric operations.
#[derive(Debug, PartialEq, Eq)]
pub enum FabricError {
    /// CyberFabric subsystem is not initialised.
    NotInitialised,
    /// The given node ID is not known to the fabric topology.
    UnknownNode,
    /// The endpoint is already registered.
    AlreadyRegistered,
    /// Capability verification failed.
    CapabilityError(crate::capability::CapabilityError),
}

impl From<crate::capability::CapabilityError> for FabricError {
    fn from(e: crate::capability::CapabilityError) -> Self {
        FabricError::CapabilityError(e)
    }
}

// ── Fabric state ──────────────────────────────────────────────────────────────

struct FabricState {
    initialised: bool,
    endpoints: hashbrown::HashMap<NodeId, FabricEndpoint>,
}

impl FabricState {
    fn new() -> Self {
        Self {
            initialised: false,
            endpoints: hashbrown::HashMap::new(),
        }
    }
}

static FABRIC: Mutex<Option<FabricState>> = Mutex::new(None);

fn fabric_lock() -> spin::MutexGuard<'static, Option<FabricState>> {
    let mut guard = FABRIC.lock();
    if guard.is_none() {
        *guard = Some(FabricState::new());
    }
    guard
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Initialise the CyberFabric subsystem.
///
/// In production this would enumerate PCIe fabric adapters (RoCE / InfiniBand /
/// CXL) and build an initial topology.
pub fn init() {
    let mut guard = fabric_lock();
    let state = guard.as_mut().expect("fabric state initialised");
    // TODO: enumerate fabric adapters (RDMA verbs / BlueField DMA engine).
    state.initialised = true;
}

/// Register a node as a fabric endpoint, enabling P2P DMA to/from it.
///
/// Issues a [`CapabilityKind::FabricEndpoint`] capability for the node and
/// stores the endpoint descriptor.
///
/// # Errors
/// Returns [`FabricError::AlreadyRegistered`] if the node is already present.
pub fn register_endpoint(
    node_id: NodeId,
    node_kind: NodeKind,
    mr_base: u64,
    mr_length: usize,
) -> Result<FabricEndpoint, FabricError> {
    let mut guard = fabric_lock();
    let state = guard.as_mut().expect("fabric state initialised");

    if !state.initialised {
        return Err(FabricError::NotInitialised);
    }
    if state.endpoints.contains_key(&node_id) {
        return Err(FabricError::AlreadyRegistered);
    }

    let cap = crate::capability::issue(
        CapabilityKind::FabricEndpoint,
        CapabilityRights::READ | CapabilityRights::WRITE | CapabilityRights::DMA,
    );

    let endpoint = FabricEndpoint {
        node_id,
        node_kind,
        mr_base,
        mr_length,
        capability: cap,
    };
    state.endpoints.insert(node_id, endpoint.clone());
    Ok(endpoint)
}

/// Look up a registered fabric endpoint by node ID.
///
/// Returns a clone of the endpoint descriptor, or [`FabricError::UnknownNode`].
pub fn lookup_endpoint(node_id: NodeId) -> Result<FabricEndpoint, FabricError> {
    let guard = fabric_lock();
    let state = guard.as_ref().expect("fabric state initialised");
    state
        .endpoints
        .get(&node_id)
        .cloned()
        .ok_or(FabricError::UnknownNode)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn init_subsystems() {
        crate::capability::init();
        init();
    }

    #[test]
    fn test_register_and_lookup() {
        init_subsystems();
        let node = NodeId(42);
        let ep = register_endpoint(node, NodeKind::Accelerator, 0x8000_0000, 1 << 20)
            .expect("registration should succeed");
        assert_eq!(ep.node_id, node);
        assert_eq!(ep.node_kind, NodeKind::Accelerator);

        let found = lookup_endpoint(node).expect("lookup should succeed");
        assert_eq!(found.mr_base, 0x8000_0000);
    }

    #[test]
    fn test_lookup_unknown_node() {
        init_subsystems();
        let result = lookup_endpoint(NodeId(999));
        assert!(matches!(result, Err(FabricError::UnknownNode)));
    }
}
