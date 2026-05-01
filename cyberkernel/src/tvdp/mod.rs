//! # Trust-Verify Data Path (TVDP)
//!
//! The Trust-Verify Data Path is CyberCore's answer to the question:
//!
//! > *"How do we know that data arriving from a remote node hasn't been tampered
//!   with — and that the device sending it is who it claims to be?"*
//!
//! ## Architecture
//!
//! ```text
//!  Sender (GPU / SmartNIC / Storage)
//!  ┌──────────────────────────────────────────┐
//!  │ 1. Compute HMAC over data + nonce        │
//!  │ 2. Attach proof to DMA descriptor        │
//!  └──────────────────────┬───────────────────┘
//!                         │  Zero-copy DMA (RDMA)
//!  Receiver (CyberKernel Control Plane)
//!  ┌──────────────────────▼───────────────────┐
//!  │ 3. Verify HMAC against device public key │
//!  │ 4. Check nonce freshness (replay guard)  │
//!  │ 5. Validate IOMMU mapping is still live  │
//!  │ 6. Stamp `TvdpProof` onto data buffer    │
//!  └──────────────────────────────────────────┘
//! ```
//!
//! ## Security properties
//!
//! - **Authenticity**: every incoming data transfer carries a hardware-rooted
//!   signature that proves the sender identity.
//! - **Integrity**: the signature covers the payload, so tampering is detected.
//! - **Freshness**: a monotonic nonce prevents replay attacks.
//! - **Least privilege**: IOMMU mapping is verified before any CPU access.
//!
//! ## Current status
//!
//! This is a **concept implementation** / skeleton.  The cryptographic
//! primitives (e.g. SHA-256 HMAC, device attestation certificates) are stubbed
//! out with `TODO` markers; a production implementation would integrate with a
//! TEE (e.g. Intel TDX, AMD SEV-SNP) or a dedicated security co-processor.

use spin::Mutex;

use crate::capability::{Capability, CapabilityKind, CapabilityRights};

// ── DeviceIdentity ────────────────────────────────────────────────────────────

/// A device's identity claim, provided as part of the DMA transfer descriptor.
#[derive(Debug, Clone)]
pub struct DeviceIdentity {
    /// Opaque device certificate / endorsement key (DER-encoded in production).
    /// Here represented as a fixed-size byte array for simplicity.
    pub cert_bytes: [u8; 32],
    /// Monotonic nonce to prevent replay attacks.
    pub nonce: u64,
}

// ── TvdpProof ─────────────────────────────────────────────────────────────────

/// Attestation proof stamped onto a data buffer after TVDP verification.
///
/// Any downstream consumer that sees a `TvdpProof` wrapper knows that:
/// - The data came from a verified device.
/// - The IOMMU mapping was live at the time of verification.
/// - The nonce was fresh (no replay detected).
#[derive(Debug)]
pub struct TvdpProof {
    /// The capability that was used to authorise this data path.
    pub capability: Capability,
    /// Physical address of the verified buffer.
    pub phys_addr: u64,
    /// Length of the verified region.
    pub length: usize,
    /// The nonce value used in the verification.
    pub nonce: u64,
}

// ── TvdpError ─────────────────────────────────────────────────────────────────

/// Errors from TVDP verification.
#[derive(Debug, PartialEq, Eq)]
pub enum TvdpError {
    /// Subsystem not initialised.
    NotInitialised,
    /// The device certificate is invalid or unknown.
    InvalidIdentity,
    /// The HMAC signature does not match the data.
    SignatureInvalid,
    /// The nonce has already been seen (replay attack).
    NonceReplay,
    /// The IOMMU mapping for this buffer is not active.
    IommuNotMapped,
    /// The capability presented does not grant TVDP access.
    CapabilityError(crate::capability::CapabilityError),
}

impl From<crate::capability::CapabilityError> for TvdpError {
    fn from(e: crate::capability::CapabilityError) -> Self {
        TvdpError::CapabilityError(e)
    }
}

// ── TVDP state ────────────────────────────────────────────────────────────────

struct TvdpState {
    initialised: bool,
    /// Set of seen nonces for replay detection.
    seen_nonces: hashbrown::HashSet<u64>,
}

impl TvdpState {
    fn new() -> Self {
        Self {
            initialised: false,
            seen_nonces: hashbrown::HashSet::new(),
        }
    }
}

static TVDP: Mutex<Option<TvdpState>> = Mutex::new(None);

fn tvdp_lock() -> spin::MutexGuard<'static, Option<TvdpState>> {
    let mut guard = TVDP.lock();
    if guard.is_none() {
        *guard = Some(TvdpState::new());
    }
    guard
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Initialise the TVDP subsystem.
///
/// In production this would:
/// - Load device certificate roots from a TPM / secure enclave.
/// - Initialise the nonce counter from a hardware RNG.
pub fn init() {
    let mut guard = tvdp_lock();
    let state = guard.as_mut().expect("tvdp state initialised");
    // TODO: load trust anchors from secure storage / TPM NV index.
    state.initialised = true;
}

/// Verify a data transfer's identity proof and stamp a [`TvdpProof`].
///
/// This is the **hot path** called for every incoming zero-copy DMA transfer.
///
/// ## Steps
/// 1. Check the `TvdpSession` capability is valid.
/// 2. Verify the device identity certificate chain (stubbed).
/// 3. Check nonce freshness.
/// 4. Verify the HMAC over `[phys_addr, phys_addr + length)` (stubbed).
/// 5. Mark the nonce as seen.
/// 6. Return a [`TvdpProof`] that downstream consumers can rely on.
///
/// # Errors
/// Returns [`TvdpError`] on any verification failure.  The caller must **not**
/// use the data if verification fails.
pub fn verify_transfer(
    identity: &DeviceIdentity,
    phys_addr: u64,
    length: usize,
    hmac: &[u8; 32],
    capability: Capability,
) -> Result<TvdpProof, TvdpError> {
    // ── 1. Validate the capability. ───────────────────────────────────────────
    if capability.kind != CapabilityKind::TvdpSession {
        return Err(TvdpError::CapabilityError(
            crate::capability::CapabilityError::InsufficientRights,
        ));
    }
    crate::capability::validate(&capability, CapabilityRights::READ)?;

    let mut guard = tvdp_lock();
    let state = guard.as_mut().expect("tvdp state initialised");

    if !state.initialised {
        return Err(TvdpError::NotInitialised);
    }

    // ── 2. Verify device identity (stub). ────────────────────────────────────
    // TODO: verify `identity.cert_bytes` against a trust anchor.
    //       e.g. sha2::Sha256::verify(cert, root_ca_pubkey)
    if identity.cert_bytes == [0u8; 32] {
        return Err(TvdpError::InvalidIdentity);
    }

    // ── 3. Nonce freshness check. ─────────────────────────────────────────────
    if state.seen_nonces.contains(&identity.nonce) {
        return Err(TvdpError::NonceReplay);
    }

    // ── 4. Verify HMAC over the payload (stub). ───────────────────────────────
    // TODO: compute expected_hmac = HMAC-SHA256(device_key, phys_addr || length || nonce)
    //       and compare with `hmac` in constant time.
    //
    // SAFETY: In production, the physical memory range [phys_addr, phys_addr+length)
    //         must have an active IOMMU mapping before we access it.  The IOMMU
    //         module guarantees this if iommu::map() was called first.
    let _ = (phys_addr, length, hmac); // silence unused warnings in stub
    // Stub: accept any non-zero HMAC.
    if hmac == &[0u8; 32] {
        return Err(TvdpError::SignatureInvalid);
    }

    // ── 5. Record nonce to prevent future replays. ────────────────────────────
    state.seen_nonces.insert(identity.nonce);

    // ── 6. Return the proof. ──────────────────────────────────────────────────
    Ok(TvdpProof {
        capability,
        phys_addr,
        length,
        nonce: identity.nonce,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability;

    // Test-only nonce constants.  These are not real cryptographic nonces — the
    // TVDP crypto is stubbed out for this skeleton.  Each test uses a distinct
    // value to avoid cross-test interference with the global `seen_nonces` set.
    const NONCE_HAPPY: u64 = 1001;
    const NONCE_REPLAY: u64 = 1002;
    const NONCE_BAD_ID: u64 = 1003;
    const NONCE_BAD_HMAC: u64 = 1004;

    fn init_all() {
        capability::init();
        init();
    }

    fn make_tvdp_cap() -> Capability {
        capability::issue(
            CapabilityKind::TvdpSession,
            CapabilityRights::READ | CapabilityRights::WRITE,
        )
    }

    fn identity_with_nonce(nonce: u64) -> DeviceIdentity {
        let mut cert = [0u8; 32];
        cert[0] = 0xAB; // non-zero stub marker for a valid certificate
        DeviceIdentity { cert_bytes: cert, nonce }
    }

    fn good_hmac() -> [u8; 32] {
        let mut h = [0u8; 32];
        h[0] = 0xFF; // non-zero stub marker for a valid HMAC
        h
    }

    #[test]
    fn test_verify_transfer_happy_path() {
        init_all();
        let cap = make_tvdp_cap();
        let identity = identity_with_nonce(NONCE_HAPPY);
        let proof = verify_transfer(&identity, 0x1000, 4096, &good_hmac(), cap)
            .expect("should succeed");
        assert_eq!(proof.phys_addr, 0x1000);
        assert_eq!(proof.nonce, NONCE_HAPPY);
    }

    #[test]
    fn test_replay_detected() {
        init_all();
        let cap1 = make_tvdp_cap();
        let cap2 = make_tvdp_cap();
        let identity = identity_with_nonce(NONCE_REPLAY);
        verify_transfer(&identity, 0x2000, 4096, &good_hmac(), cap1)
            .expect("first transfer ok");
        // Same nonce → replay
        let result = verify_transfer(&identity, 0x2000, 4096, &good_hmac(), cap2);
        assert!(matches!(result, Err(TvdpError::NonceReplay)));
    }

    #[test]
    fn test_invalid_identity_rejected() {
        init_all();
        let cap = make_tvdp_cap();
        let bad_identity = DeviceIdentity {
            cert_bytes: [0u8; 32], // all-zero = invalid stub
            nonce: NONCE_BAD_ID,
        };
        let result = verify_transfer(&bad_identity, 0x3000, 4096, &good_hmac(), cap);
        assert!(matches!(result, Err(TvdpError::InvalidIdentity)));
    }

    #[test]
    fn test_bad_hmac_rejected() {
        init_all();
        let cap = make_tvdp_cap();
        let result = verify_transfer(
            &identity_with_nonce(NONCE_BAD_HMAC),
            0x4000,
            4096,
            &[0u8; 32], // all-zero = invalid stub
            cap,
        );
        assert!(matches!(result, Err(TvdpError::SignatureInvalid)));
    }
}
