//! # Memory Management Subsystem
//!
//! Provides:
//! - Physical frame allocator ([`allocator`])
//! - IOMMU / DMA mapping layer ([`iommu`])
//! - A thin [`MemoryMap`] abstraction used during boot
//!
//! ## Design principles
//! - All allocations are tracked with explicit lifetimes or RAII wrappers.
//! - The IOMMU layer ensures that no device can access host memory that has not
//!   been explicitly mapped with the correct permissions.
//! - Zero-copy paths pin memory in place (`PhysicallyPinned`) so that the OS
//!   cannot reclaim or move a page while a DMA transfer is in flight.

pub mod allocator;
pub mod iommu;

// ── MemoryMap ─────────────────────────────────────────────────────────────────

/// A descriptor for a single contiguous region of physical memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryRegion {
    /// Start address (physical, byte-aligned).
    pub start: u64,
    /// Length in bytes.
    pub length: u64,
    /// Kind of region (usable RAM, reserved, MMIO …).
    pub kind: MemoryRegionKind,
}

/// Classification of a [`MemoryRegion`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryRegionKind {
    /// General-purpose RAM available for kernel use.
    Usable,
    /// Reserved by firmware or hardware — must not be allocated.
    Reserved,
    /// Memory-mapped I/O — access only through volatile reads/writes.
    Mmio,
    /// Framebuffer / video memory.
    Framebuffer,
    /// ACPI reclaimable memory.
    AcpiReclaimable,
}

/// The firmware-supplied physical memory map handed to [`init`].
///
/// In a real boot scenario this would come from the bootloader (e.g. UEFI
/// `GetMemoryMap`).  For tests it can be constructed manually.
pub struct MemoryMap<'a> {
    /// Slice of memory region descriptors, ordered by `start` address.
    pub regions: &'a [MemoryRegion],
}

// ── Subsystem init ────────────────────────────────────────────────────────────

/// Initialise the memory subsystem from the boot-time memory map.
///
/// Walks all `Usable` regions and hands them to the frame allocator.
pub fn init(map: &MemoryMap<'_>) {
    allocator::init(map);
}
