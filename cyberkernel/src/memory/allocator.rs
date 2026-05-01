//! # Physical Frame Allocator
//!
//! A two-phase allocator:
//!
//! 1. **Boot phase** — a simple bump allocator that hands out 4 KiB frames in
//!    order.  It is initialised from the boot-time [`MemoryMap`] and requires
//!    no heap.
//!
//! 2. **Runtime phase** (future) — a buddy allocator or slab allocator for
//!    sub-page allocations once the virtual address space is set up.
//!
//! ## Frame lifecycle
//!
//! ```text
//!  allocate_frame()  →  PhysFrame (owned, Drop → return to free list)
//!       │
//!       ▼
//!  pin_for_dma()  →  PhysicallyPinned<T> (cannot be moved / reclaimed)
//!       │
//!       ▼
//!  iommu::map()  →  IommuMapping (device can now access the buffer)
//! ```

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

use spin::Mutex;

use super::{MemoryMap, MemoryRegionKind};

// ── Constants ─────────────────────────────────────────────────────────────────

/// System page size in bytes (4 KiB).
pub const PAGE_SIZE: usize = 4096;

/// Alignment required for DMA buffers (64-byte cache line).
pub const DMA_ALIGNMENT: usize = 64;

// ── PhysFrame ────────────────────────────────────────────────────────────────

/// An owned physical page frame.
///
/// Dropping a `PhysFrame` returns the frame to the global free list.
/// Frames that are in use for DMA must be wrapped in [`PhysicallyPinned`]
/// before mapping, which prevents accidental drops.
#[derive(Debug)]
pub struct PhysFrame {
    /// Physical base address of this 4 KiB frame.
    pub phys_addr: u64,
}

impl Drop for PhysFrame {
    fn drop(&mut self) {
        // Return the frame to the global allocator.
        FRAME_ALLOCATOR
            .lock()
            .free(self.phys_addr);
    }
}

// ── PhysicallyPinned ─────────────────────────────────────────────────────────

/// A frame that has been **pinned** for DMA use.
///
/// Pinning prevents the frame from being reclaimed until the [`PhysicallyPinned`]
/// wrapper is explicitly un-pinned via [`PhysicallyPinned::unpin`].
///
/// # Type-state pattern
/// `PhysicallyPinned` is *not* `Send` + `Sync` in isolation; the IOMMU layer
/// wraps it in an [`crate::memory::iommu::IommuMapping`] which carries the
/// device-side permission proof.
#[derive(Debug)]
pub struct PhysicallyPinned {
    frame: PhysFrame,
}

impl PhysicallyPinned {
    /// Consume the `PhysicallyPinned` wrapper and recover the underlying frame.
    ///
    /// # Safety
    /// The caller **must** ensure no DMA transfer is in flight when this is
    /// called.  Typically, this is enforced by first dropping the
    /// `IommuMapping` that wraps this pinned frame.
    pub unsafe fn unpin(self) -> PhysFrame {
        self.frame
    }

    /// Physical address of the pinned frame.
    #[inline]
    pub fn phys_addr(&self) -> u64 {
        self.frame.phys_addr
    }
}

// ── BumpAllocator ─────────────────────────────────────────────────────────────

struct BumpAllocator {
    regions: Vec<(u64, u64)>, // (start, end) of usable regions
    current_region: usize,
    next: u64,
    free_list: Vec<u64>, // returned frames
}

impl BumpAllocator {
    const fn empty() -> Self {
        Self {
            regions: Vec::new(),
            current_region: 0,
            next: 0,
            free_list: Vec::new(),
        }
    }

    fn add_region(&mut self, start: u64, end: u64) {
        self.regions.push((start, end));
        if self.regions.len() == 1 {
            self.next = start;
        }
    }

    /// Allocate one 4 KiB frame; returns `None` if OOM.
    fn allocate(&mut self) -> Option<u64> {
        // Prefer frames from the free list (from prior drops).
        if let Some(addr) = self.free_list.pop() {
            return Some(addr);
        }

        loop {
            if self.current_region >= self.regions.len() {
                return None; // OOM
            }
            let (_, end) = self.regions[self.current_region];
            let aligned = align_up(self.next, PAGE_SIZE as u64);
            if aligned + PAGE_SIZE as u64 <= end {
                self.next = aligned + PAGE_SIZE as u64;
                return Some(aligned);
            }
            // Advance to the next region.
            self.current_region += 1;
            if self.current_region < self.regions.len() {
                self.next = self.regions[self.current_region].0;
            }
        }
    }

    fn free(&mut self, addr: u64) {
        self.free_list.push(addr);
    }
}

// ── Global allocator instance ─────────────────────────────────────────────────

static FRAME_ALLOCATOR: Mutex<BumpAllocator> = Mutex::new(BumpAllocator::empty());

/// Initialise the frame allocator from the boot-time memory map.
///
/// Called once by [`crate::memory::init`].
pub fn init(map: &MemoryMap<'_>) {
    let mut alloc = FRAME_ALLOCATOR.lock();
    for region in map.regions {
        if region.kind == MemoryRegionKind::Usable {
            alloc.add_region(region.start, region.start + region.length);
        }
    }
}

/// Allocate a single 4 KiB physical frame.
///
/// Returns `None` if no physical memory is available.
pub fn allocate_frame() -> Option<PhysFrame> {
    let addr = FRAME_ALLOCATOR.lock().allocate()?;
    Some(PhysFrame { phys_addr: addr })
}

/// Allocate a frame and immediately pin it for DMA use.
///
/// Equivalent to `allocate_frame()` followed by pinning, but avoids any
/// window in which the frame could be accidentally dropped.
pub fn allocate_pinned_frame() -> Option<PhysicallyPinned> {
    let frame = allocate_frame()?;
    Some(PhysicallyPinned { frame })
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Round `addr` up to the nearest multiple of `align` (must be a power of 2).
#[inline]
pub(crate) fn align_up(addr: u64, align: u64) -> u64 {
    debug_assert!(align.is_power_of_two());
    (addr + align - 1) & !(align - 1)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 4096), 0);
        assert_eq!(align_up(1, 4096), 4096);
        assert_eq!(align_up(4096, 4096), 4096);
        assert_eq!(align_up(4097, 4096), 8192);
    }

    #[test]
    fn test_bump_allocator_basic() {
        let mut alloc = BumpAllocator::empty();
        // 8 KiB region → should yield exactly 2 frames
        alloc.add_region(0x1000_0000, 0x1000_0000 + 2 * PAGE_SIZE as u64);
        assert!(alloc.allocate().is_some());
        assert!(alloc.allocate().is_some());
        assert!(alloc.allocate().is_none()); // OOM
    }

    #[test]
    fn test_bump_allocator_free_list() {
        let mut alloc = BumpAllocator::empty();
        alloc.add_region(0x2000_0000, 0x2000_0000 + PAGE_SIZE as u64);
        let addr = alloc.allocate().expect("first allocation");
        alloc.free(addr);
        // Should get the same address back from the free list.
        let addr2 = alloc.allocate().expect("second allocation after free");
        assert_eq!(addr, addr2);
    }
}
