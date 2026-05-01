# CyberCore AIOS

CyberCore 是一個實驗性的 **AI-Native Matrix OS**，目標實現「Datacenter-as-a-Computer」與「Data-Centric Architecture」。

它摒棄傳統以 CPU 為中心的架構，以 Rust 的嚴格記憶體安全與所有權模型作為安全基礎，結合 Mojo 的異構運算效能，實現數據在網路、儲存與 GPU/NPU 之間的零拷貝超導流動。

---

## Architecture Overview

```
┌───────────────────────────────────────────────────────────────┐
│                    Data Plane (Mojo)                           │
│  Matrix ops · AI kernels · Heterogeneous compute dispatch      │
│  mojo_runtime/compute.mojo                                     │
└──────────────────────────┬────────────────────────────────────┘
                           │  C ABI FFI + shared pinned memory
                           │  (register_compute_region,
                           │   issue_compute_capability,
                           │   release_compute_region)
┌──────────────────────────▼────────────────────────────────────┐
│               Control Plane (cyberkernel / Rust)               │
│                                                                │
│  ┌──────────┐  ┌────────────┐  ┌──────────┐  ┌────────────┐  │
│  │ memory/  │  │ capability/│  │ fabric/  │  │  tvdp/     │  │
│  │ allocator│  │ cap system │  │RDMA/NIC  │  │ Trust-     │  │
│  │ iommu    │  │ DMA auth   │  │ abstract.│  │ Verify DP  │  │
│  └──────────┘  └────────────┘  └──────────┘  └────────────┘  │
│                                                                │
│  cyberkernel/src/lib.rs  (init, subsystem orchestration)       │
└───────────────────────────────────────────────────────────────┘
```

### Core Principles

| Concern | Owner | Rationale |
|---|---|---|
| Memory safety, ownership | **Rust** | Compile-time guarantees; no GC pauses |
| Capability-based security | **Rust** | No ACLs; unforgeable tokens gate all hardware access |
| IOMMU / DMA isolation | **Rust** | Devices can only access explicitly-mapped buffers |
| Proof-of-Trust (TVDP) | **Rust** | Hardware-rooted signatures on every data transfer |
| Matrix / AI kernels | **Mojo** | MLIR-compiled, heterogeneous, GPU/NPU-native |
| Zero-copy scheduling | **Mojo** | Data stays on-device; kernel only tracks DMA descriptors |

---

## Repository Layout

```
CyberCore/
├── Cargo.toml                    # Workspace root
├── cyberkernel/                  # Rust microkernel (Control Plane)
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                # Entry point, init() orchestration
│       ├── memory/
│       │   ├── mod.rs            # MemoryMap, subsystem init
│       │   ├── allocator.rs      # Bump frame allocator, PhysicallyPinned
│       │   └── iommu.rs          # IOMMU mapping, IommuMapping RAII
│       ├── capability/
│       │   ├── mod.rs            # Capability table, issue/revoke/validate
│       │   └── types.rs          # AuthorizedDmaBuffer, ComputeCapability
│       ├── fabric/
│       │   └── mod.rs            # CyberFabric endpoint registry (RDMA/SmartNIC)
│       ├── ffi/
│       │   └── mod.rs            # extern "C" surface for Mojo FFI
│       └── tvdp/
│           └── mod.rs            # Trust-Verify Data Path (signature & replay guard)
└── mojo_runtime/
    └── compute.mojo              # Mojo Data Plane stub with FFI calls & GEMM kernel
```

---

## Key Subsystems

### `memory/` — Physical Allocator + IOMMU

- **`allocator.rs`**: bump allocator that hands out 4 KiB `PhysFrame` tokens.
  Frames can be promoted to `PhysicallyPinned` to prevent reclamation during DMA.
- **`iommu.rs`**: IOMMU abstraction layer. Calling `iommu::map()` programs the
  hardware DMAR/SMMU page table and returns an `IommuMapping` RAII handle.
  Dropping the handle atomically revokes device access.

### `capability/` — Capability-Based Security

Every hardware operation requires an unforgeable `Capability` token:

| Kind | Grants |
|---|---|
| `DmaBuffer` | Physically-pinned buffer accessible to a named device |
| `ComputeRegion` | Permission to schedule kernels on a GPU/NPU |
| `FabricEndpoint` | RDMA endpoint for P2P zero-copy transfers |
| `TvdpSession` | Trust-Verify Data Path session |

Type-state wrappers (`AuthorizedDmaBuffer`, `ComputeCapability`) enforce at
compile time that unauthorised buffers can never reach the hardware path.

### `fabric/` — CyberFabric (Network-as-a-Bus)

Registers accelerators, SmartNICs, and storage nodes as fabric endpoints.
Issues `FabricEndpoint` capabilities, sets up queue-pair descriptors, and
enforces IOMMU policies so P2P DMA transfers never touch the CPU data path.

### `ffi/` — Rust ↔ Mojo FFI Surface

Exposes `extern "C"` functions for the Mojo runtime:

```c
// Allocate a kernel-managed, IOMMU-mapped tensor buffer
i32 register_compute_region(usize length, u64 device_id, u64 *out_phys_addr);

// Issue an opaque capability handle for compute scheduling
i32 issue_compute_capability(u64 device_id, u64 flops_budget, u64 *out_handle);

// Validate a previously-issued capability handle
i32 validate_capability_handle(u64 cap_handle);

// Release the buffer and revoke the IOMMU mapping
i32 release_compute_region(u64 phys_addr);
```

### `tvdp/` — Trust-Verify Data Path

Provides hardware-rooted authentication and integrity checking for zero-copy
data transfers:

1. **Authenticity** — device certificate chain verification (stubbed; integrate TEE)
2. **Integrity** — HMAC over payload + nonce (stubbed; integrate SHA-256)
3. **Freshness** — monotonic nonce replay guard
4. **Isolation** — IOMMU mapping verified before any CPU access

---

## Zero-Copy Data Flow

```
Sender (GPU / SmartNIC / NVMe)
  │
  │ 1. Rust kernel: allocate_pinned_frame() → PhysicallyPinned
  │ 2. Rust kernel: iommu::map(pinned, ...) → IommuMapping
  │ 3. Rust kernel: register_compute_region() → phys_addr (via FFI)
  │
  ▼
Shared DMA buffer [phys_addr, phys_addr + length)
  │
  │ 4. Mojo: populate tensor A, B via device DMA (no CPU copies)
  │ 5. Mojo: run matmul_kernel() entirely on-device
  │ 6. TVDP: verify signature + nonce → TvdpProof
  │
  ▼
Receiver (another GPU / storage node)
  │
  │ 7. Mojo: release_compute_region(phys_addr) → IOMMU unmapped
  │ 8. PhysicallyPinned dropped → frame returned to allocator
```

---

## Building & Testing

```bash
# Build (std feature enables host-side unit tests)
cargo build --features std

# Run all unit tests
cargo test --features std

# Build for bare-metal (no_std) — requires a target spec
cargo build --target x86_64-unknown-none
```

### Test coverage

| Module | Tests |
|---|---|
| `memory::allocator` | align_up, bump allocator basic, free list |
| `memory::iommu` | state init, permission flags |
| `capability` | issue+validate, revoke, unique IDs |
| `capability::types` | AuthorizedDmaBuffer happy/error paths, ComputeCapability |
| `fabric` | register+lookup, unknown node |
| `ffi` | register+release region, issue+validate cap, invalid handle |
| `tvdp` | happy path, replay detection, invalid identity, bad HMAC |

---

## Roadmap

- [ ] **Intel VT-d / AMD-Vi HAL**: replace `hal_iommu_map` stubs with real DMAR register writes
- [ ] **ACPI DMAR/IVRS parser**: auto-discover IOMMU units at boot
- [ ] **Buddy allocator**: replace bump allocator for runtime sub-page allocations
- [ ] **TVDP crypto**: integrate SHA-256 HMAC and device attestation certificates (AMD SEV-SNP / Intel TDX)
- [ ] **CXL / RDMA fabric driver**: SmartNIC queue-pair setup and BlueField DMA engine integration
- [ ] **Mojo async scheduler**: replace synchronous stub with MLIR-compiled async dispatch
- [ ] **x86_64 / AArch64 boot shim**: UEFI entry point, page table setup, global allocator registration
- [ ] **Process / domain model**: per-domain capability tables and resource quotas
