//! The consumer side of the glass protocol: attaching an app's engine to
//! the running broker and pulling published backdrops into local private
//! textures.
//!
//! Attaching is a non-blocking state machine (engine-loop friendly — no
//! step holds a lock for long):
//!
//! `Idle` — the broker's manifest is opened (the broker is spawned first
//! when it seems absent); a free slot is claimed under this process's PID
//! and handle duplication is requested (`SLOT_REQUESTED`).
//! `Registered` — the broker duplicates the shared textures' NT handles
//! into this process (`SLOT_SUPPLIED`); after an overall attach timeout
//! the machine gives up and the engine falls back to the in-process
//! engine.
//! `Attached` — every published backdrop is copied out of the shared
//! textures under the keyed mutex (brief acquire retries; a contested
//! copy skips the frame) into a private texture the local compositors
//! bind. From here the frame path is identical to the in-process engine —
//! the publish's `frame_version` bump is the only cross-process input.
//!
//! A dead broker (PID probe), a texture-epoch change (the broker rebuilt
//! the pair for a monitor change) and a protocol mismatch all reset the
//! machine; [`BrokerClient::tick`] reports the failure and the engine
//! transparently switches back to its local engine.

use std::os::windows::process::CommandExt;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use windows::core::Interface;
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::Graphics::Direct3D11::{
    ID3D11Device, ID3D11Device1, ID3D11DeviceContext, ID3D11ShaderResourceView, ID3D11Texture2D,
    D3D11_BIND_SHADER_RESOURCE, D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT,
};
use windows::Win32::Graphics::Dxgi::IDXGIKeyedMutex;
use windows::Win32::System::Threading::{OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION};

use super::protocol::{
    snapshot, ConsumerSlot, CONSUMER_SLOTS, PROTOCOL_MAGIC, PROTOCOL_VERSION,
    SLOT_REQUESTED, SLOT_SUPPLIED, SLOT_UNUSED, ManifestView, ATTACH_TIMEOUT, BROKER_EXE_NAME,
};
use crate::shared::Backdrop;
use super::device_adapter_luid;

/// The attach/presence poll cadence.
const PRESENCE_POLL: Duration = Duration::from_millis(500);
/// The broker-liveness probe cadence while attached.
const ALIVE_PROBE: Duration = Duration::from_millis(1000);
/// Keyed-mutex retries when copying a published frame; a competing copy
/// lasts microseconds and a skipped frame costs nothing.
const COPY_RETRIES: u32 = 4;
const COPY_TIMEOUT_MS: u32 = 25;
/// The constant keyed-mutex key — identical to the broker's discipline.
const MUTEX_KEY: u64 = 0;
/// How long after a spawn attempt before another auto-start attempt.
const RESPAWN_AFTER: Duration = Duration::from_millis(15_000);
/// `CREATE_NO_WINDOW`: a spawned broker must never flash a console.
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

enum Phase {
    /// Nothing is registered; the manifest may not even exist yet.
    Idle,
    /// A slot is claimed and handle duplication is requested.
    Registered { slot: usize, since: Instant },
    /// The shared textures are mapped and frames are pulled.
    Attached { slot: usize },
}

/// The mapped shared pair: the opened NT-handle textures on this
/// process's device and their keyed mutexes.
struct Mapped {
    textures: [ID3D11Texture2D; 2],
    keyed: [IDXGIKeyedMutex; 2],
    epoch: u32,
}

/// The private copy target: the local compositors bind this texture's
/// view, exactly like the in-process engine's own backdrop.
struct PrivateBackdrop {
    texture: ID3D11Texture2D,
    view: ID3D11ShaderResourceView,
    width: u32,
    height: u32,
}

/// What a pulled frame hands to the engine loop.
pub(crate) struct PulledFrame {
    pub(crate) backdrop: Backdrop,
}

pub(crate) struct BrokerClient {
    phase: Phase,
    /// The broker's manifest view, `None` until its mapping is found.
    view: Option<Arc<ManifestView>>,
    my_pid: u32,
    mapped: Option<Mapped>,
    private: Option<PrivateBackdrop>,
    /// The last published version copied into the private backdrop.
    last_version: u64,
    /// The slot's active flag as last written (stored only on change).
    last_active: bool,
    /// Auto-start bookkeeping (spawn attempts space themselves out).
    last_spawn: Option<Instant>,
    last_presence: Option<Instant>,
    last_probe: Option<Instant>,
    /// The last failure surfaced to the engine; identical follow-ups are
    /// suppressed (see [`BrokerClient::report_once`]) so a persistent
    /// condition logs once, not per tick.
    last_error: Option<String>,
}

// The client lives under the engine's own `Mutex` and its mapped pointers
// (D3D11 objects and the manifest view) only serve the render loop's
// thread — the same confinement as the existing compositors.
unsafe impl Send for BrokerClient {}

impl BrokerClient {
    pub(crate) fn new() -> Self {
        Self {
            phase: Phase::Idle,
            view: None,
            my_pid: std::process::id(),
            mapped: None,
            private: None,
            last_version: 0,
            last_active: false,
            last_spawn: None,
            last_presence: None,
            last_probe: None,
            last_error: None,
        }
    }

    pub(crate) fn attached(&self) -> bool {
        matches!(self.phase, Phase::Attached { .. })
    }

    /// Advances the attach state machine and (once attached) pulls any
    /// freshly published backdrop into the private copy texture.
    /// `Ok(Some(frame))` means the engine's compositors must re-bind the
    /// new backdrop; `Ok(None)` means everything is unchanged.
    pub(crate) fn tick(
        &mut self,
        device: &ID3D11Device,
        context: &ID3D11DeviceContext,
        active: bool,
    ) -> Result<Option<PulledFrame>, String> {
        match self.phase {
            Phase::Idle => {
                self.progress_presence(device)?;
                Ok(None)
            }
            Phase::Registered { slot, since } => self.progress_registration(device, slot, since),
            Phase::Attached { slot } => self.pull_frame(device, context, active, slot),
        }
    }

    fn progress_presence(&mut self, device: &ID3D11Device) -> Result<(), String> {
        let now = Instant::now();
        if self
            .last_presence
            .map(|at| now.duration_since(at) < PRESENCE_POLL)
            .unwrap_or(false)
        {
            return Ok(());
        }
        self.last_presence = Some(now);
        match ManifestView::open() {
            Ok(Some(view)) => {
                self.last_error = None;
                self.admit(device, view)
            }
            Ok(None) => {
                // No broker's mapping: spawn one — unless a spawn just ran
                // (the respawn grace avoids hammering a failing launch).
                if self
                    .last_spawn
                    .map(|at| now.duration_since(at) >= RESPAWN_AFTER)
                    .unwrap_or(true)
                {
                    self.spawn_broker()
                } else {
                    Ok(())
                }
            }
            Err(error) => self.report_once(error),
        }
    }

    /// Surfaces a repeating failure exactly once: the first occurrence
    /// returns `Err` (the engine logs it); identical follow-ups stay
    /// silent until a different failure or any success resets the slate.
    fn report_once(&mut self, error: String) -> Result<(), String> {
        if self.last_error.as_deref() == Some(&error) {
            return Ok(());
        }
        self.last_error = Some(error.clone());
        Err(error)
    }

    /// Validates a found broker manifest (protocol version + same GPU
    /// adapter) and claims this process's consumer slot.
    fn admit(&mut self, device: &ID3D11Device, view: ManifestView) -> Result<(), String> {
        let manifest = view.manifest();
        if manifest.magic.load(Ordering::Acquire) != PROTOCOL_MAGIC
            || manifest.protocol.load(Ordering::Acquire) != PROTOCOL_VERSION
        {
            let error = "A glass broker with an incompatible protocol is running".to_owned();
            self.last_error = Some(error.clone());
            return Err(error);
        }
        let manifest_luid = manifest.adapter_luid.load(Ordering::Acquire);
        if manifest_luid != 0 && manifest_luid != device_adapter_luid(device)? {
            let error = "The glass broker runs on a different GPU adapter".to_owned();
            self.last_error = Some(error.clone());
            return Err(error);
        }
        // Claim the slot: reuse this process's own slot on a re-attach,
        // otherwise the first unused one.
        let mut chosen: Option<usize> = None;
        for index in 0..CONSUMER_SLOTS {
            let slot = manifest.slot(index);
            let pid = slot.pid.load(Ordering::Acquire);
            if pid == self.my_pid && slot.state.load(Ordering::Acquire) != SLOT_UNUSED {
                chosen = Some(index);
                break;
            }
            if pid == 0 && slot.state.load(Ordering::Acquire) == SLOT_UNUSED && chosen.is_none() {
                chosen = Some(index);
            }
        }
        let Some(slot) = chosen else {
            let error = "No free glass consumer slot in the manifest".to_owned();
            self.last_error = Some(error.clone());
            return Err(error);
        };
        // The consumer's handshake writes: any stale supplied data resets
        // first, the pid second, and the requested state LAST — the broker
        // only ever acts on fully-written slots.
        let slot_ref = manifest.slot(slot);
        slot_ref.handles[0].store(0, Ordering::Release);
        slot_ref.handles[1].store(0, Ordering::Release);
        slot_ref.handles_epoch.store(0, Ordering::Release);
        slot_ref.active.store(0, Ordering::Release);
        if slot_ref.pid.load(Ordering::Acquire) != self.my_pid {
            slot_ref.pid.store(self.my_pid, Ordering::Release);
        }
        slot_ref.state.store(SLOT_REQUESTED, Ordering::Release);
        self.view = Some(Arc::new(view));
        self.phase = Phase::Registered {
            slot,
            since: Instant::now(),
        };
        Ok(())
    }

    /// Waits for the broker's supplied handles and maps the shared pair.
    fn progress_registration(
        &mut self,
        device: &ID3D11Device,
        slot: usize,
        since: Instant,
    ) -> Result<Option<PulledFrame>, String> {
        let now = Instant::now();
        if now.duration_since(since) >= ATTACH_TIMEOUT {
            let reason = "The glass broker never supplied the shared textures".to_owned();
            return Err(self.detach(reason));
        }
        let Some(view) = self.view.as_ref() else {
            self.phase = Phase::Idle;
            return Ok(None);
        };
        let manifest = view.manifest();
        let slot_ref = manifest.slot(slot);
        if slot_ref.state.load(Ordering::Acquire) != SLOT_SUPPLIED {
            return Ok(None);
        }
        let epoch = manifest.texture_epoch.load(Ordering::Acquire);
        // A map failure poisons the supplied slot's stale handles: detach
        // (releasing the slot) rather than retrying the same broken map.
        let mapped = match BrokerClient::map_shared(device, slot_ref, epoch) {
            Ok(mapped) => mapped,
            Err(reason) => return Err(self.detach(reason)),
        };
        self.mapped = Some(mapped);
        self.private = None;
        self.last_version = 0;
        self.phase = Phase::Attached { slot };
        Ok(None)
    }

    fn map_shared(
        device: &ID3D11Device,
        slot: &ConsumerSlot,
        epoch: u32,
    ) -> Result<Mapped, String> {
        let handles = [
            slot.handles[0].load(Ordering::Acquire),
            slot.handles[1].load(Ordering::Acquire),
        ];
        if handles.iter().any(|handle| *handle == 0) {
            return Err("The broker supplied zero-valued glass handles".to_owned());
        }
        let device1: ID3D11Device1 = device
            .cast()
            .map_err(|error| format!("Could not cast the glass device to ID3D11Device1: {error}"))?;
        let mut textures: Vec<ID3D11Texture2D> = Vec::with_capacity(2);
        for raw_value in handles {
            let raw = HANDLE(raw_value as usize as *mut std::ffi::c_void);
            let texture: ID3D11Texture2D = unsafe {
                device1
                    .OpenSharedResource1(raw)
                    .map_err(|error| format!("Could not open the shared glass texture: {error}"))?
            };
            // OpenSharedResource1 added the process's own reference; the
            // duplicated handle's transfer lifetime ends here (closing it
            // keeps the texture alive).
            unsafe {
                CloseHandle(raw).ok();
            }
            textures.push(texture);
        }
        let mut keyed: Vec<IDXGIKeyedMutex> = Vec::with_capacity(2);
        for texture in &textures {
            let mutex = texture
                .cast::<IDXGIKeyedMutex>()
                .map_err(|error| format!("Could not obtain the shared glass mutex: {error}"))?;
            keyed.push(mutex);
        }
        Ok(Mapped {
            textures: textures.try_into().expect("two shared textures"),
            keyed: keyed.try_into().expect("two keyed mutexes"),
            epoch,
        })
    }

    /// The attached tick: broker liveness + epoch watch, the active flag,
    /// and the fresh-backdrop copy when the manifest's version moved.
    fn pull_frame(
        &mut self,
        device: &ID3D11Device,
        context: &ID3D11DeviceContext,
        active: bool,
        slot: usize,
    ) -> Result<Option<PulledFrame>, String> {
        let now = Instant::now();
        let Some(view) = self.view.as_ref() else {
            let reason = "The glass manifest vanished".to_owned();
            return Err(self.detach(reason));
        };
        let manifest = view.manifest();
        if self
            .last_probe
            .map(|at| now.duration_since(at) >= ALIVE_PROBE)
            .unwrap_or(true)
        {
            self.last_probe = Some(now);
            let broker_pid = manifest.broker_pid.load(Ordering::Acquire);
            let alive = broker_pid != 0
                && unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, broker_pid) }
                    .is_ok();
            if !alive {
                let reason = "The glass broker process exited".to_owned();
                return Err(self.detach(reason));
            }
        }
        let epoch = manifest.texture_epoch.load(Ordering::Acquire);
        if let Some(mapped) = self.mapped.as_ref() {
            if epoch != mapped.epoch {
                // The pair was rebuilt: re-map through a fresh request on
                // the same slot.
                self.re_request(slot);
                return Ok(None);
            }
        }
        // The active flag — brokers skip their full-monitor frame copy
        // while every consumer parks.
        if active != self.last_active {
            self.last_active = active;
            let slot_ref = manifest.slot(slot);
            slot_ref.active.store(u32::from(active), Ordering::Release);
        }
        let Some((version, front, _key)) = snapshot(manifest) else {
            return Ok(None);
        };
        if version == self.last_version {
            return Ok(None);
        }
        // The mapped back buffer's pieces (cloned: the COM references just
        // AddRef) so the mutation below can run without borrowing `self`.
        let (keyed, shared) = {
            let Some(mapped) = self.mapped.as_ref() else {
                return Ok(None);
            };
            (
                mapped.keyed[front as usize].clone(),
                mapped.textures[front as usize].clone(),
            )
        };
        let width = manifest.frame_width.load(Ordering::Acquire).max(1);
        let height = manifest.frame_height.load(Ordering::Acquire).max(1);
        let (private_texture, private_view) = self.private_for(device, width, height)?;
        let mut acquired = false;
        for _ in 0..COPY_RETRIES {
            if unsafe { keyed.AcquireSync(MUTEX_KEY, COPY_TIMEOUT_MS) }.is_ok() {
                acquired = true;
                break;
            }
        }
        if !acquired {
            // A competing copy still holds the buffer: skip this tick.
            return Ok(None);
        }
        unsafe {
            context.CopyResource(&private_texture, &shared);
            keyed.ReleaseSync(MUTEX_KEY).ok();
        }
        self.last_version = version;
        Ok(Some(PulledFrame {
            backdrop: Backdrop {
                view: private_view,
            },
        }))
    }

    /// The private backdrop (copy target), recreated on a frame-size
    /// change; the compositors otherwise keep binding its view. Returns
    /// cloned COM references so the callers never borrow `self`.
    fn private_for(
        &mut self,
        device: &ID3D11Device,
        width: u32,
        height: u32,
    ) -> Result<(ID3D11Texture2D, ID3D11ShaderResourceView), String> {
        if let Some(private) = self.private.as_ref() {
            if private.width == width && private.height == height {
                return Ok((private.texture.clone(), private.view.clone()));
            }
        }
        let mut texture = None;
        unsafe {
            device.CreateTexture2D(
                &D3D11_TEXTURE2D_DESC {
                    Width: width,
                    Height: height,
                    MipLevels: 1,
                    ArraySize: 1,
                    Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM,
                    SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    Usage: D3D11_USAGE_DEFAULT,
                    BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
                    CPUAccessFlags: 0,
                    MiscFlags: 0,
                },
                None,
                Some(&mut texture),
            )
        }
        .map_err(|error| format!("Could not create the glass copy backdrop: {error}"))?;
        let texture = texture.expect("the glass copy backdrop is missing");
        let mut view = None;
        unsafe {
            device.CreateShaderResourceView(&texture, None, Some(&mut view))
        }
        .map_err(|error| format!("Could not create the glass copy view: {error}"))?;
        self.private = Some(PrivateBackdrop {
            texture,
            view: view.expect("the glass copy view is missing"),
            width,
            height,
        });
        let private = self.private.as_ref().expect("just created");
        Ok((private.texture.clone(), private.view.clone()))
    }

    /// Resets the mappings after a texture-epoch change and re-requests
    /// duplication on the same slot.
    fn re_request(&mut self, slot: usize) {
        self.mapped = None;
        self.last_version = 0;
        if let Some(view) = self.view.as_ref() {
            let slot_ref = view.manifest().slot(slot);
            slot_ref.handles[0].store(0, Ordering::Release);
            slot_ref.handles[1].store(0, Ordering::Release);
            slot_ref.handles_epoch.store(0, Ordering::Release);
            slot_ref.state.store(SLOT_REQUESTED, Ordering::Release);
        }
        self.phase = Phase::Registered {
            slot,
            since: Instant::now(),
        };
    }

    /// Tears the client down after a failure and returns the reason (the
    /// caller surfaces it): the slot is released and every mapped
    /// resource dropped, resetting the machine to `Idle`.
    fn detach(&mut self, reason: String) -> String {
        if let Some(view) = self.view.as_ref() {
            let slot = match self.phase {
                Phase::Registered { slot, .. } | Phase::Attached { slot } => Some(slot),
                Phase::Idle => None,
            };
            if let Some(slot) = slot {
                let slot_ref = view.manifest().slot(slot);
                // Order matters: the reader (the broker's maintenance and
                // other consumers' scans) treats pid == 0 as the final
                // release, so the pid write comes last.
                slot_ref.state.store(SLOT_UNUSED, Ordering::Release);
                slot_ref.active.store(0, Ordering::Release);
                slot_ref.handles[0].store(0, Ordering::Release);
                slot_ref.handles[1].store(0, Ordering::Release);
                slot_ref.handles_epoch.store(0, Ordering::Release);
                slot_ref.pid.store(0, Ordering::Release);
            }
        }
        self.view = None;
        self.mapped = None;
        self.private = None;
        self.phase = Phase::Idle;
        self.last_version = 0;
        self.last_active = false;
        self.last_probe = None;
        reason
    }

    /// Auto-start: launches the broker binary, searching (in order) the
    /// directory colocated with this process's own executable, the shared
    /// LiquidGlass data directory (where an installer may have placed a
    /// session-wide copy) and finally the PATH — so multiple glass apps can
    /// share one broker even when only one of them ships the binary.
    fn spawn_broker(&mut self) -> Result<(), String> {
        self.last_spawn = Some(Instant::now());
        let mut candidates: Vec<PathBuf> = Vec::new();
        if let Some(directory) = std::env::current_exe()
            .ok()
            .and_then(|path| path.parent().map(|dir| dir.to_path_buf()))
        {
            candidates.push(directory.join(BROKER_EXE_NAME));
        }
        if let Some(mut shared) = super::material_path().and_then(|path| path.parent().map(|dir| dir.to_path_buf())) {
            shared.push(BROKER_EXE_NAME);
            candidates.push(shared);
        }
        let found = candidates
            .iter()
            .find(|path| path.exists())
            .cloned();
        match found {
            Some(path) => {
                if let Err(error) = std::process::Command::new(&path)
                    .creation_flags(CREATE_NO_WINDOW)
                    .spawn()
                {
                    return Err(format!("Could not spawn the glass broker: {error}"));
                }
                Ok(())
            }
            // No colocated copy: a bare-name attempt lets CreateProcess's
            // PATH search find one (a broker on PATH serves every app).
            None => std::process::Command::new(BROKER_EXE_NAME)
                .creation_flags(CREATE_NO_WINDOW)
                .spawn()
                .map(|_| ())
                .map_err(|error| format!("Could not spawn the glass broker: {error}")),
        }
    }
}
