//! The cross-process glass protocol: named objects, the manifest layout and
//! the shared-memory accessors.
//!
//! One broker process owns the Windows Graphics Capture session and the
//! blur chain, and publishes every fresh backdrop into a pair of shared,
//! double-buffered keyed-mutex D3D11 textures. Every consumer app maps
//! those textures on its own device (same GPU adapter) and composites
//! locally, so the per-app frame path stays exactly as fast as the
//! in-process engine while the capture + blur work happens once for
//! everybody.
//!
//! The manifest is a small file mapping (`Local\LiquidGlass.Manifest.v1`)
//! describing the shared textures and consumer slots. Consumers register
//! themselves by PID in a free slot; the broker then duplicates the
//! textures' NT handles into those processes (the only cross-process way to
//! transfer a D3D11 NT-handle resource) and flags the slot as supplied.
//! Every field is read/written atomically through `std::sync::atomic` on
//! the mapped memory: there is exactly one writer per field class (the
//! broker owns the header, each consumer owns exactly one slot), so
//! acquires/releases (strong ordering on x64) keep the pages coherent
//! across processes.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Duration;

use windows::core::w;
use windows::Win32::Foundation::{CloseHandle, GetLastError, HANDLE, ERROR_ALREADY_EXISTS};
use windows::Win32::System::Memory::{
    CreateFileMappingW, FILE_MAP, FILE_MAP_ALL_ACCESS, MapViewOfFile, MEMORY_MAPPED_VIEW_ADDRESS,
    OpenFileMappingW, PAGE_READWRITE, UnmapViewOfFile,
};

/// The manifest's file mapping (session-local).
pub const MANIFEST_NAME: windows::core::PCWSTR = w!("Local\\LiquidGlass.Manifest.v1");
/// Magic/protocol guards: a mismatch means an old broker process or a
/// foreign build is running; consumers refuse to attach and use their
/// in-process engine.
pub const PROTOCOL_MAGIC: u32 = 0x4C47_53B1;
pub const PROTOCOL_VERSION: u32 = 1;
/// Consumer slots in the manifest. Small on purpose: the local-engine
/// fallback covers more, and empty-slot lookup stays trivial.
pub const CONSUMER_SLOTS: usize = 16;
/// The broker binary's name; auto-start spawns it next to the host apps'
/// own executable (installers colocate the two).
pub const BROKER_EXE_NAME: &str = "glass-broker.exe";
/// Consumer states for [`ConsumerSlot::state`].
pub const SLOT_UNUSED: u32 = 0;
pub const SLOT_REQUESTED: u32 = 1;
pub const SLOT_SUPPLIED: u32 = 2;
/// How long a consumer waits (polling non-blockingly) for the broker to
/// duplicate handles into its slot before giving up and falling back to
/// the local engine.
pub const ATTACH_TIMEOUT: Duration = Duration::from_millis(3000);

/// One consumer app's registration in the manifest.
#[repr(C)]
#[derive(Debug)]
pub struct ConsumerSlot {
    /// The consumer's process id, 0 while unused. Written once (and cleared
    /// on detach) by the consumer itself.
    pub pid: AtomicU32,
    /// STATE constants above: the slot's handshake state. The consumer
    /// advances REGISTERED -> REQUESTED, the broker advances to SUPPLIED
    /// last, so `SLOT_SUPPLIED` verifiably carries valid duplicated
    /// handles (request and pid are written before it).
    pub state: AtomicU32,
    /// The handle values the broker duplicated into the consumer's process
    /// for the two shared backdrop textures, in buffer order. Valid only
    /// in the consumer's process; the consumer opens them with
    /// `OpenSharedResource1` (which adds its own reference) and then
    /// closes the duplicated handles.
    pub handles: [AtomicU64; 2],
    /// The broker's texture epoch these handles were supplied for; the
    /// broker invalidates supplied state on a texture recreate (monitor
    /// resolution changes), and consumers re-register when theirs differs.
    pub handles_epoch: AtomicU32,
    /// 1 while this consumer's glass is actively rendering. The broker's
    /// capture callback skips its full-monitor frame copy when every
    /// consumer's slot reads 0 — identical to the in-process engine's
    /// parked-surface logic.
    pub active: AtomicU32,
    pub reserved: AtomicU32,
}

/// The manifest: the broker's descriptor of the shared backdrop pair and
/// the consumer registry.
#[repr(C)]
pub struct Manifest {
    pub magic: AtomicU32,
    pub protocol: AtomicU32,
    /// The broker's process id; consumers probe it (a cheap OpenProcess
    /// existence check) to detect broker death.
    pub broker_pid: AtomicU32,
    /// Bumped every time the shared backdrop textures are (re)created
    /// (monitor resolution change, first frame). Consumers whose mapped
    /// epoch differs re-open the resources.
    pub texture_epoch: AtomicU32,
    /// The shared textures' pixel size — the monitor's size, the never
    /// changing publication resolution (see the broker host's docs).
    pub frame_width: AtomicU32,
    pub frame_height: AtomicU32,
    /// Which of the two shared textures holds the newest backdrop (0/1),
    /// written by the broker after its publish release.
    pub front: AtomicU32,
    pub reserved: AtomicU32,
    /// Monotonic, bumped for every published backdrop. Consumers wake on
    /// change; frozen while the desktop is static (no re-publish).
    pub frame_version: AtomicU64,
    /// The publisher's adapter LUID; consumers on a different adapter fall
    /// back to the in-process engine (same-adapter shares are required).
    pub adapter_luid: AtomicU64,
    /// The keyed-mutex protocol's current key per shared texture: the key
    /// any next acquirer must use (see the acquire discipline in
    /// [`crate::broker::client`]).
    pub keys: [AtomicU64; 2],
    pub slots: [ConsumerSlot; CONSUMER_SLOTS],
}

impl Manifest {
    pub fn slot(&self, index: usize) -> &ConsumerSlot {
        &self.slots[index]
    }
}

const _: () = {
    assert!(std::mem::size_of::<Manifest>() <= 4096);
    assert!(std::mem::size_of::<ConsumerSlot>() == 40);
    assert!(std::mem::size_of::<Manifest>() == 64 + CONSUMER_SLOTS * 40);
};

/// A mapped manifest view, owned by either the broker or a consumer.
pub struct ManifestView {
    mapping: HANDLE,
    view: *mut Manifest,
}

unsafe impl Send for ManifestView {}

impl ManifestView {
    /// Maps the broker's manifest; `Ok(None)` when none exists yet (no
    /// broker running).
    pub fn open() -> Result<Option<Self>, String> {
        let Some(mapping) = Self::existing_mapping() else {
            return Ok(None);
        };
        let view = Self::map(mapping)?;
        Ok(Some(Self { mapping, view }))
    }

    /// Creates a fresh manifest for a starting broker. `Err` when another
    /// broker already owns the mapping (deduplicated broker launches) or
    /// the kernel refuses.
    pub fn create() -> Result<Self, String> {
        if Self::existing_mapping().is_some() {
            return Err("Another glass broker already owns the manifest".to_owned());
        }
        unsafe {
            let mapping = CreateFileMappingW(
                HANDLE::default(),
                None,
                PAGE_READWRITE,
                0,
                Manifest::size_bytes() as u32,
                MANIFEST_NAME,
            )
            .map_err(|error| format!("Could not create the glass manifest mapping: {error}"))?;
            let existed = GetLastError() == ERROR_ALREADY_EXISTS;
            if existed {
                let _ = CloseHandle(mapping);
                return Err("Another glass broker already owns the manifest".to_owned());
            }
            let view = Self::map(mapping)?;
            // The zero-page image doubles as the zeroed manifest: an
            // all-zero byte range is exactly the unused-slot state every
            // reader expects (SLOT_UNUSED = 0, no frames, no broker pid),
            // so there is nothing struct-wise to write field by field.
            std::ptr::write_bytes(view.cast::<u8>(), 0, Manifest::size_bytes());
            let manifest = &*view;
            manifest.magic.store(PROTOCOL_MAGIC, Ordering::Release);
            manifest.protocol.store(PROTOCOL_VERSION, Ordering::Release);
            Ok(Self { mapping, view })
        }
    }

    fn existing_mapping() -> Option<HANDLE> {
        unsafe {
            OpenFileMappingW(
                FILE_MAP_ALL_ACCESS.0,
                false,
                MANIFEST_NAME,
            )
            .ok()
        }
    }

    fn map(mapping: HANDLE) -> Result<*mut Manifest, String> {
        unsafe {
            let view: MEMORY_MAPPED_VIEW_ADDRESS =
                MapViewOfFile(mapping, FILE_MAP(FILE_MAP_ALL_ACCESS.0), 0, 0, Manifest::size_bytes());
            if view.Value.is_null() {
                Err("Could not map the glass manifest view".to_owned())
            } else {
                Ok(view.Value.cast())
            }
        }
    }

    /// The manifest sharing this process's view.
    pub fn manifest(&self) -> &Manifest {
        unsafe { &*self.view }
    }
}

impl Drop for ManifestView {
    fn drop(&mut self) {
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: self.view.cast(),
            })
            .ok();
            CloseHandle(self.mapping).ok();
        }
    }
}

/// A publish-consistent snapshot of the manifest's frame state. The broker
/// writes `frame_version` last (release store) after `front` and the frame
/// size, so peeking the version twice brackets an internally consistent
/// read; `None` means the fields moved underneath (the caller retries next
/// tick).
pub fn snapshot(manifest: &Manifest) -> Option<(u64, u32, u64)> {
    let v0 = manifest.frame_version.load(Ordering::Acquire);
    if v0 == 0 {
        return None;
    }
    let front = manifest.front.load(Ordering::Acquire);
    let keys = (
        manifest.keys[0].load(Ordering::Acquire),
        manifest.keys[1].load(Ordering::Acquire),
    );
    let v1 = manifest.frame_version.load(Ordering::Acquire);
    if v0 != v1 {
        return None;
    }
    Some((v0, front, if front == 0 { keys.0 } else { keys.1 }))
}

impl Manifest {
    fn size_bytes() -> usize {
        std::mem::size_of::<Manifest>()
    }
}

/// The LUID-carrying adapter description clients match against (from
/// `IDXGIDevice::GetAdapterLuid`): a stable two-u64 identity of the GPU.
#[inline]
pub fn adapter_luid(luid: windows::Win32::Foundation::LUID) -> u64 {
    (luid.LowPart as u64) | ((luid.HighPart as u64) << 32)
}
