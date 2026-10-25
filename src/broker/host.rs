//! The broker process's engine: it owns the one WGC capture session and
//! the one blur chain for every glass app in the session and publishes
//! every fresh backdrop into the shared double-buffered keyed-mutex
//! texture pair (see `protocol`).
//!
//! Publication per fresh frame:
//! 1. the capture copy already landed in the host's stable texture
//!    (skipped entirely while every consumer is parked);
//! 2. the backdrop is composited exactly like the in-process engine
//!    (dual-Kawase blur, or the raw capture when blur is off), from the
//!    global material's params read from the broker-owned material file;
//! 3. the pair's back buffer is acquired (keyed mutex, constant key), the
//!    backdrop is written in — always at the monitor's full publication
//!    resolution, so consumers' shader geometry never depends on the
//!    material — and released;
//! 4. `front` is stored, then `frame_version` bumps last (release store)
//!    — that final bump is the signal every consumer's engine loop sits
//!    on.
//!
//! The process is deliberately short-lived: with no consumer registered
//! for the exit grace it quits (the next glass app re-spawns it), and
//! while consumers exist but none is active it tears the capture feed
//! down, exactly like the in-process engine parks its pipeline.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use windows::core::{Interface, HRESULT};
use windows::Graphics::SizeInt32;
use windows::Win32::Foundation::{
    CloseHandle, DuplicateHandle, HANDLE, DUPLICATE_SAME_ACCESS,
};
use windows::Win32::Graphics::Direct3D11::{
    ID3D11Device, ID3D11DeviceContext, ID3D11RenderTargetView, ID3D11ShaderResourceView,
    ID3D11Texture2D, D3D11_BIND_RENDER_TARGET, D3D11_BIND_SHADER_RESOURCE,
    D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX, D3D11_RESOURCE_MISC_SHARED_NTHANDLE,
    D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT,
};
use windows::Win32::Graphics::Dxgi::{
    IDXGIKeyedMutex, IDXGIResource1, DXGI_SHARED_RESOURCE_READ, DXGI_SHARED_RESOURCE_WRITE,
};
use windows::Win32::System::Threading::{
    GetCurrentProcess, OpenProcess, PROCESS_DUP_HANDLE, PROCESS_QUERY_LIMITED_INFORMATION,
};

use super::device_adapter_luid;
use super::protocol::{
    ConsumerSlot, CONSUMER_SLOTS, Manifest, ManifestView, SLOT_REQUESTED, SLOT_SUPPLIED,
    SLOT_UNUSED,
};
use crate::capture::CaptureFeed;
use crate::params::GlassShaderParams;
use crate::shared::{blur_plan, Backdrop, BlurEngine};
use crate::surface::create_d3d11_device;

/// The idle feed grace while consumers still exist but no one is active
/// (mirrors the in-process engine's hysteresis so a preview or a quick
/// hide never churns the capture pipeline).
const IDLE_FEED_TICKS: u64 = 120;
/// The process-exit grace with no registered consumer at all; the next
/// glass app re-spawns the broker.
const EXIT_IDLE_TICKS: u64 = 640;
/// Consumer maintenance cadence: prune dead processes, duplicate handles
/// into requests.
const MAINTENANCE_INTERVAL: Duration = Duration::from_millis(250);
/// Ticks to wait before re-attempting a failed capture start (a persistent
/// failure must not retry at the loop's tick rate).
const CAPTURE_RETRY_BACKOFF: u64 = 120;
/// Consecutive failed publication cycles before the pair is rebuilt — a
/// consumer that died while holding a keyed mutex would otherwise poison
/// it permanently (a recreate invalidates every mapping and forces a
/// clean re-handshake).
const PUBLISH_RECOVERY_TICKS: u32 = 300;
/// Keyed-mutex acquire attempts before a publication skips a frame
/// (another party's copy lasts microseconds; skipping costs nothing).
const ACQUIRE_RETRIES: u32 = 8;
/// Per-attempt keyed-mutex timeout.
const ACQUIRE_TIMEOUT_MS: u32 = 25;
/// The constant keyed-mutex key every acquire/release pair uses — the
/// plain-mutex discipline: with one writer and serialized copyers the key
/// never moves, so no inter-key choreography exists to get wrong.
const MUTEX_KEY: u64 = 0;
/// `HRESULT_FROM_WIN32(ERROR_INVALID_PARAMETER)` — the kernel's answer
/// when `OpenProcess` names a pid that no longer exists (any other
/// failure means the process is alive but out of reach).
const DEAD_PID_HRESULT: HRESULT = HRESULT(0x8007_0057u32 as i32);

/// The broker's entry point: owns the shared manifest, the capture feed
/// and the publication; exits when the last consumer releases. Non-zero
/// on startup failure.
pub fn run() -> i32 {
    // The mapping create is the single-broker gate (the kernel marks the
    // pre-existing name, and a second broker exits without touching any
    // consumer state).
    let manifest_view = match ManifestView::create() {
        Ok(view) => Arc::new(view),
        Err(error) => {
            eprintln!("glass broker exiting: {error}");
            return 2;
        }
    };
    let manifest = manifest_view.manifest();
    let (device, context) = match create_d3d11_device() {
        Ok(pair) => pair,
        Err(error) => {
            // Leaving the failure here still drops the mapping view, so
            // the name frees and a later broker can claim it.
            eprintln!("glass broker exiting: {error}");
            return 1;
        }
    };
    // The publisher's adapter identity: consumers on another adapter must
    // never attach (NT-handle shares are same-adapter only).
    match device_adapter_luid(&device) {
        Ok(luid) => manifest.adapter_luid.store(luid, Ordering::Release),
        Err(error) => {
            eprintln!("glass broker exiting: {error}");
            return 1;
        }
    }
    manifest.broker_pid.store(std::process::id(), Ordering::Release);

    // The shared engine's parts, mirroring the in-process engine's setup:
    // one host (capture delivery), one feed (restarted on demand), one
    // publication engine.
    let registry = Arc::new(ConsumerRegistry {
        shared: Mutex::new(Some(Arc::clone(&manifest_view))),
    });
    let host: Arc<Mutex<BrokerHost>> = Arc::new(Mutex::new(BrokerHost::new(
        device.clone(),
        context.clone(),
        Arc::clone(&registry),
    )));
    let version = host
        .lock()
        .expect("broker host lock poisoned")
        .version_counter();
    let host_dyn: Arc<Mutex<dyn crate::capture::FrameHost>> = Arc::clone(&host) as _;
    let feed: Arc<Mutex<Option<CaptureFeed>>> = Arc::new(Mutex::new(None));

    let mut engine = PublishEngine::new(&device);
    let mut material = MaterialWatcher::new();
    let mut idle_ticks = 0u64;
    let mut restart_backoff = 0u64;
    // Publication failures log once per distinct reason: a persistent
    // failure repeats every tick, and the recovery recreate addresses it
    // without a log storm.
    let mut last_publish_error = None;
    let mut last_maintenance = Instant::now()
        .checked_sub(MAINTENANCE_INTERVAL)
        .unwrap_or_else(Instant::now);
    let my_pid = std::process::id();

    loop {
        std::thread::sleep(Duration::from_millis(16));

        // The material's params poll (its revision drives the blur
        // refresh) and the frame/backdrop refresh + publication cycle.
        material.poll(&mut engine.params);
        let material_revision = material.revision();
        {
            let host_guard = host.lock().expect("broker host lock poisoned");
            let host_version = host_guard.frame_version.load(Ordering::Acquire);
            engine.refresh(&context, &host_guard, host_version, material_revision);
            if let Err(error) =
                engine.publish_inner(&device, &context, &host_guard, manifest, host_version)
            {
                if last_publish_error.as_deref() != Some(error.as_str()) {
                    eprintln!("glass broker publish failed: {error}");
                    last_publish_error = Some(error);
                }
            } else {
                last_publish_error = None;
            }
        }

        // Consumer maintenance on its own cadence.
        if last_maintenance.elapsed() >= MAINTENANCE_INTERVAL {
            last_maintenance = Instant::now();
            maintain(manifest, engine.pair.as_ref(), my_pid);
        }

        // Capture feed lifecycle: start it for active consumers, park it
        // once everyone sleeps, and exit once nobody is registered.
        let any_active = registry.any_active();
        let some_consumer = manifest_any_consumer(manifest);
        if some_consumer && any_active {
            idle_ticks = 0;
            if restart_backoff > 0 {
                restart_backoff -= 1;
            } else if feed.lock().expect("broker feed lock poisoned").is_none() {
                match CaptureFeed::start(&host_dyn, &version) {
                    Ok(new_feed) => {
                        *feed.lock().expect("broker feed lock poisoned") = Some(new_feed);
                    }
                    Err(error) => {
                        eprintln!("glass broker capture failed to start: {error}");
                        restart_backoff = CAPTURE_RETRY_BACKOFF;
                    }
                }
            }
        } else {
            idle_ticks += 1;
            if !some_consumer {
                if idle_ticks >= EXIT_IDLE_TICKS {
                    break;
                }
            } else if idle_ticks >= IDLE_FEED_TICKS {
                // Consumers are registered but every one is parked: park
                // the capture with the same hysteresis as the in-process
                // engine (the first slot to raise its active flag restarts
                // the feed above).
                idle_ticks = 0;
                *feed.lock().expect("broker feed lock poisoned") = None;
            }
        }
    }

    // Teardown ordering matters: the capture stops and the host drops
    // before the manifest view does (the capture callback reads the slots
    // through the registry).
    *feed.lock().expect("broker feed lock poisoned") = None;
    drop(host);
    drop(registry);
    0
}

/// The consumer registry the capture callback reads for its any-active
/// answer: does any registered slot claim active rendering right now?
///
/// The manifest pointer is stored as a view through a plain mutex (only
/// ever read from), with all real access going through the mapped atomics
/// — no TLS pins or unsync weak spots remain anywhere.
pub(crate) struct ConsumerRegistry {
    shared: Mutex<Option<Arc<ManifestView>>>,
}

impl ConsumerRegistry {
    fn any_active(&self) -> bool {
        let guard = self.shared.lock().expect("consumer registry lock poisoned");
        let Some(view) = guard.as_ref() else {
            return false;
        };
        let manifest = view.manifest();
        for index in 0..CONSUMER_SLOTS {
            let slot = manifest.slot(index);
            if slot.pid.load(Ordering::Acquire) != 0
                && slot.active.load(Ordering::Acquire) != 0
            {
                return true;
            }
        }
        false
    }
}

fn manifest_any_consumer(manifest: &Manifest) -> bool {
    for index in 0..CONSUMER_SLOTS {
        if manifest.slot(index).pid.load(Ordering::Acquire) != 0 {
            return true;
        }
    }
    false
}

/// Per-consumer maintenance: prune dead processes (freeing their slots)
/// and duplicate the shared textures into the requested ones. Non-supplied
/// slots stay pending until the shared pair exists (the first publish's
/// pair creation precedes any handle duplication).
fn maintain(manifest: &Manifest, pair: Option<&SharedPair>, my_pid: u32) {
    for index in 0..CONSUMER_SLOTS {
        let slot = manifest.slot(index);
        let pid = slot.pid.load(Ordering::Acquire);
        if pid == 0 || pid == my_pid {
            continue;
        }
        let process = match unsafe {
            OpenProcess(
                PROCESS_DUP_HANDLE | PROCESS_QUERY_LIMITED_INFORMATION,
                false,
                pid,
            )
        } {
            Ok(handle) => handle,
            Err(error) => {
                // Only a dead pid frees its slot. An access failure (an
                // elevated consumer, a protected process) keeps the slot:
                // the consumer's own attach timeout releases it if the
                // share never lands.
                if error.code() == DEAD_PID_HRESULT {
                    release_slot(slot);
                }
                continue;
            }
        };
        if slot.state.load(Ordering::Acquire) == SLOT_REQUESTED {
            if let Some(pair) = pair {
                match duplicate_pair(process, &pair.cells) {
                    Ok(handles) => {
                        slot.handles[0].store(handles[0], Ordering::Release);
                        slot.handles[1].store(handles[1], Ordering::Release);
                        slot.handles_epoch.store(
                            manifest.texture_epoch.load(Ordering::Acquire),
                            Ordering::Release,
                        );
                        slot.state.store(SLOT_SUPPLIED, Ordering::Release);
                    }
                    Err(error) => {
                        eprintln!(
                            "glass broker could not share with consumer pid {pid}: {error}"
                        );
                    }
                }
            }
        }
        unsafe {
            CloseHandle(process).ok();
        }
    }
}

/// Slot release for a dead consumer: state then pid, in that order (pid==0
/// is the readers' "free slot" marker).
fn release_slot(slot: &ConsumerSlot) {
    slot.state.store(SLOT_UNUSED, Ordering::Release);
    slot.active.store(0, Ordering::Release);
    slot.pid.store(0, Ordering::Release);
}

/// Duplicates one process's pair handles for one slot.
fn duplicate_pair(process: HANDLE, cells: &[SharedCell; 2]) -> Result<[u64; 2], String> {
    let mut out = [0u64; 2];
    for (index, cell) in cells.iter().enumerate() {
        let mut dup = HANDLE(std::ptr::null_mut());
        unsafe {
            DuplicateHandle(
                GetCurrentProcess(),
                cell.handle,
                process,
                &mut dup,
                0,
                false,
                DUPLICATE_SAME_ACCESS,
            )
        }
        .map_err(|error| format!("Could not duplicate the glass handle: {error}"))?;
        out[index] = dup.0 as usize as u64;
    }
    Ok(out)
}

/// One shared backdrop buffer: the texture on its keyed mutex, the
/// render-target view the publication path writes into, and the resource
/// carrying the NT handle the broker duplicates into consumers.
pub(crate) struct SharedCell {
    texture: ID3D11Texture2D,
    target: ID3D11RenderTargetView,
    resource: IDXGIResource1,
    handle: HANDLE,
}

impl SharedCell {
    fn keyed(&self) -> IDXGIKeyedMutex {
        self.resource
            .cast::<IDXGIKeyedMutex>()
            .expect("keyed-mutex textures provide IDXGIKeyedMutex")
    }
}

/// The two-buffer backdrop pair at the monitor's full publication size.
pub(crate) struct SharedPair {
    cells: [SharedCell; 2],
    width: u32,
    height: u32,
}

/// The publication machinery: the material's params (read from the shared
/// settings file), the blur chain, and the shared pair's lifecycle.
struct PublishEngine {
    blur: BlurEngine,
    params: GlassShaderParams,
    /// The material revision the current params were read from (any
    /// change forces a re-blur; the manifest's versions stay monotonic).
    backdrop: Option<Backdrop>,
    backdrop_version: u64,
    backdrop_material: u64,
    pair: Option<SharedPair>,
    /// The host frame version already published; publications are skipped
    /// while the backdrop has not moved.
    published_version: u64,
    publish_failures: u32,
}

impl PublishEngine {
    fn new(device: &ID3D11Device) -> Self {
        Self {
            blur: BlurEngine::new(device),
            params: GlassShaderParams::default(),
            backdrop: None,
            backdrop_version: 0,
            backdrop_material: 0,
            pair: None,
            published_version: 0,
            publish_failures: 0,
        }
    }

    /// Refreshes the shared backdrop when a fresh capture frame or a
    /// material revision landed; static desktops re-use the last one.
    fn refresh(
        &mut self,
        context: &ID3D11DeviceContext,
        host: &BrokerHost,
        host_version: u64,
        material_revision: u64,
    ) {
        if self.backdrop_version == host_version && self.backdrop_material == material_revision {
            return;
        }
        let Some((_texture, view)) = host.stable.as_ref() else {
            return;
        };
        let size = [
            host.frame_size.Width.max(1) as u32,
            host.frame_size.Height.max(1) as u32,
        ];
        match self.blur.render(context, view, size, &self.params) {
            Ok(backdrop) => {
                self.backdrop = Some(backdrop);
                self.backdrop_version = host_version;
                self.backdrop_material = material_revision;
            }
            Err(error) => eprintln!("glass broker blur failed: {error}"),
        }
    }

    /// Creates the shared pair at the monitor's size, stamps the manifest
    /// (sizes, bumped epoch) and invalidates consumer handles.
    fn recreate(
        &mut self,
        device: &ID3D11Device,
        manifest: &Manifest,
        width: u32,
        height: u32,
    ) -> Result<(), String> {
        let make_cell = || -> Result<SharedCell, String> {
            let mut texture = None;
            unsafe {
                device.CreateTexture2D(
                    &D3D11_TEXTURE2D_DESC {
                        Width: width,
                        Height: height,
                        MipLevels: 1,
                        ArraySize: 1,
                        BindFlags: (D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE).0
                            as u32,
                        Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM,
                        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                            Count: 1,
                            Quality: 0,
                        },
                        Usage: D3D11_USAGE_DEFAULT,
                        MiscFlags: (D3D11_RESOURCE_MISC_SHARED_NTHANDLE
                            | D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX)
                            .0 as u32,
                        CPUAccessFlags: 0,
                    },
                    None,
                    Some(&mut texture),
                )
            }
            .map_err(|error| {
                format!("Could not create the shared glass backdrop texture: {error}")
            })?;
            let texture = texture.expect("shared glass texture missing");
            let mut target = None;
            unsafe {
                device.CreateRenderTargetView(&texture, None, Some(&mut target))
            }
            .map_err(|error| {
                format!("Could not create the shared glass target view: {error}")
            })?;
            let target = target.expect("shared glass target view missing");
            let resource = texture
                .cast::<IDXGIResource1>()
                .map_err(|error| format!("Could not obtain the shared glass resource: {error}"))?;
            let handle = unsafe {
                resource
                    .CreateSharedHandle(
                        None,
                        DXGI_SHARED_RESOURCE_READ.0 | DXGI_SHARED_RESOURCE_WRITE.0,
                        windows::core::w!(""),
                    )
                    .map_err(|error| {
                        format!("Could not create the shared glass NT handle: {error}")
                    })?
            };
            Ok(SharedCell {
                texture,
                target,
                resource,
                handle,
            })
        };
        let first = make_cell()?;
        let second = make_cell()?;
        let epoch = manifest
            .texture_epoch
            .load(Ordering::Acquire)
            .wrapping_add(1)
            .max(1);
        self.pair = Some(SharedPair {
            cells: [first, second],
            width,
            height,
        });
        manifest.frame_width.store(width, Ordering::Release);
        manifest.frame_height.store(height, Ordering::Release);
        manifest.texture_epoch.store(epoch, Ordering::Release);
        // The pair's texture handles changed: every supplied slot's map is
        // invalid — those slots round-trip through the request handshake.
        for index in 0..CONSUMER_SLOTS {
            let slot = manifest.slot(index);
            if slot.state.load(Ordering::Acquire) == SLOT_SUPPLIED {
                slot.state.store(SLOT_REQUESTED, Ordering::Release);
            }
        }
        self.published_version = 0;
        Ok(())
    }

    /// Ensures a pair at the current frame size exists (recreated on a
    /// monitor change).
    fn ensure_pair(
        &mut self,
        device: &ID3D11Device,
        manifest: &Manifest,
        host: &BrokerHost,
    ) -> Result<(), String> {
        let width = host.frame_size.Width.max(1) as u32;
        let height = host.frame_size.Height.max(1) as u32;
        match &self.pair {
            Some(pair) if pair.width == width && pair.height == height => Ok(()),
            _ => self.recreate(device, manifest, width, height),
        }
    }

    /// Publishes the latest backdrop into the pair's back buffer and
    /// flips the manifest. A contended mutex skips the tick; an `Err` is
    /// the caller's q to log (repeating failures trigger the pair's
    /// recreation through the failure counter).
    fn publish_inner(
        &mut self,
        device: &ID3D11Device,
        context: &ID3D11DeviceContext,
        host: &BrokerHost,
        manifest: &Manifest,
        version: u64,
    ) -> Result<(), String> {
        // No backdrop yet, or already published: nothing to do.
        let backdrop_view = match self.backdrop.as_ref() {
            Some(backdrop) => backdrop.view.clone(),
            None => return Ok(()),
        };
        if self.published_version >= version {
            return Ok(());
        }
        // The sharp-path source is resolved BEFORE the pair's work (and
        // crucially before the keyed-mutex acquire): an early return after
        // the acquire would have to release the mutex first.
        let sharp = blur_plan(&self.params).sharp;
        let sharp_source = if sharp {
            match host.stable.as_ref() {
                Some((stable, _)) => Some(stable.clone()),
                None => return Ok(()),
            }
        } else {
            None
        };
        self.ensure_pair(device, manifest, host)?;
        let (pair_width, pair_height) = {
            let pair = self.pair.as_ref().expect("the pair exists at publish");
            (pair.width, pair.height)
        };
        let back = ((manifest.front.load(Ordering::Acquire) as usize) ^ 1) & 1;
        let (keyed, texture, target) = {
            let pair = self.pair.as_ref().expect("the pair exists at publish");
            let cell = &pair.cells[back];
            (cell.keyed(), cell.texture.clone(), cell.target.clone())
        };
        // The acquire: a brief retry window handles the momentary race
        // with a consumer still copying the previous frame out of this
        // same buffer (the pair's two buffers alternate, so a competing
        // copy is rare and short).
        let mut acquired = false;
        for _ in 0..ACQUIRE_RETRIES {
            if unsafe { keyed.AcquireSync(MUTEX_KEY, ACQUIRE_TIMEOUT_MS) }.is_ok() {
                acquired = true;
                break;
            }
        }
        if !acquired {
            self.publish_failures += 1;
            if self.publish_failures >= PUBLISH_RECOVERY_TICKS {
                self.publish_failures = 0;
                let (width, height) = (
                    host.frame_size.Width.max(1) as u32,
                    host.frame_size.Height.max(1) as u32,
                );
                self.recreate(device, manifest, width, height)?;
            }
            return Ok(());
        }
        let result = if sharp {
            // The sharp backdrop IS the capture, already at the publication
            // size: a direct GPU copy suffices. The stable texture and the
            // pair share the format; the copy needs no bindings.
            let stable = sharp_source.expect("the sharp publication source exists");
            unsafe {
                context.CopyResource(&texture, &stable)
            };
            Ok(())
        } else {
            // A blur-reduced backdrop: bilinear-stretch it into the
            // monitor-sized shared buffer.
            self.blur
                .render_publication(context, &backdrop_view, &target, pair_width, pair_height)
        };
        // The release happens on EVERY path past the acquire — an
        // unreleased keyed mutex would lock every future publication out
        // of this buffer until the recovery recreate.
        unsafe {
            keyed.ReleaseSync(MUTEX_KEY).ok();
        }
        match result {
            Ok(()) => {
                // The manifest's publisher-side order: front first (the
                // consumers' snapshot brackets read it before the version
                // that unlocks it), version LAST with a release store.
                manifest.front.store(back as u32, Ordering::Release);
                manifest.frame_version.store(version, Ordering::Release);
                self.published_version = version;
                self.publish_failures = 0;
                Ok(())
            }
            Err(error) => Err(error),
        }
    }
}

/// The capture host the shared feed delivers frames into — identical
/// plumbing to the in-process engine's (a stable copy of each frame, a
/// version counter, and the any-active answer that suppresses copies for
/// parked glass).
pub(crate) struct BrokerHost {
    device: ID3D11Device,
    context: ID3D11DeviceContext,
    stable: Option<(ID3D11Texture2D, ID3D11ShaderResourceView)>,
    frame_size: SizeInt32,
    frame_version: Arc<AtomicU64>,
    registry: Arc<ConsumerRegistry>,
}

unsafe impl Send for BrokerHost {}

impl crate::capture::FrameHost for BrokerHost {
    fn device(&self) -> &ID3D11Device {
        &self.device
    }

    fn copy_context(&self) -> ID3D11DeviceContext {
        self.context.clone()
    }

    fn set_frame(
        &mut self,
        frame: (ID3D11Texture2D, ID3D11ShaderResourceView),
        size: SizeInt32,
    ) {
        self.stable = Some(frame);
        self.frame_size = size;
        self.frame_version.fetch_add(1, Ordering::Release);
    }

    fn frame_texture(&self) -> Option<&ID3D11Texture2D> {
        self.stable.as_ref().map(|(texture, _)| texture)
    }

    fn frame_size(&self) -> SizeInt32 {
        self.frame_size
    }

    fn any_active(&self) -> bool {
        self.registry.any_active()
    }
}

impl BrokerHost {
    fn new(
        device: ID3D11Device,
        context: ID3D11DeviceContext,
        registry: Arc<ConsumerRegistry>,
    ) -> Self {
        Self {
            device,
            context,
            stable: None,
            frame_size: SizeInt32::default(),
            frame_version: Arc::new(AtomicU64::new(0)),
            registry,
        }
    }

    fn version_counter(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.frame_version)
    }
}

/// The global material watcher: reads the shared material from the
/// broker-owned material file (`%LOCALAPPDATA%\LiquidGlass\material.json`),
/// re-reading on mtime change. Missing or malformed files fall back
/// silently to the library defaults.
struct MaterialWatcher {
    path: Option<PathBuf>,
    last_mtime: Option<SystemTime>,
    /// A monotonically increasing revision; any change bumps it, and the
    /// publication's backdrop keys off it.
    revision: u64,
}

impl MaterialWatcher {
    fn new() -> Self {
        let path = super::material_path();
        if let Err(error) = super::seed_material_if_missing() {
            eprintln!("glass broker material seed failed: {error}");
        }
        Self {
            path,
            last_mtime: None,
            revision: 1,
        }
    }

    fn revision(&self) -> u64 {
        self.revision
    }

    /// Polls the file (bounded by the poll cadence the caller chooses to
    /// respect); `true` when the material changed and params were updated.
    fn poll(&mut self, params: &mut GlassShaderParams) -> bool {
        let Some(path) = self.path.as_ref() else {
            return false;
        };
        let mtime = std::fs::metadata(path)
            .and_then(|metadata| metadata.modified())
            .ok();
        if mtime == self.last_mtime {
            return false;
        }
        self.last_mtime = mtime;
        let material = super::read_material();
        let mut updated = *params;
        // The broker's capture and publication run at full monitor
        // resolution, so no texel scaling applies here (texel scale 1.0).
        material.apply_onto(&mut updated, 1.0);
        if updated != *params {
            *params = updated;
            self.revision += 1;
            return true;
        }
        false
    }
}
