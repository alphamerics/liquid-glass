//! The shared glass engine: ONE D3D11 device, ONE Windows Graphics Capture
//! session, ONE frame texture, ONE blur, ONE render loop — for every glass
//! surface in the process.
//!
//! Previously every surface ran its own capture of the entire monitor plus a
//! presentation tick that re-rendered at ~60Hz; a handful of open panels
//! meant a dozen full-screen captures and hundreds of renders per second.
//! Now the capture delivers one frame, the newest frame is copied into a
//! single shared texture, the engine blurs it ONCE per fresh frame (the
//! dual-Kawase chain below), and one render thread composites every
//! registered surface from the shared blurred backdrop — only surfaces that
//! actually need a frame (content changed, animating, params dirty, not
//! paused). When every surface is parked, the capture session itself is
//! torn down after a short idle delay: the OS capture pipeline then idles
//! with the glass instead of capturing the monitor forever for no one.

use std::collections::HashMap;
use std::ffi::CString;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime};

use windows::core::PCSTR;
use windows::Graphics::SizeInt32;
use windows::Win32::Graphics::{
    Direct3D::{
        Fxc::{D3DCompile, D3DCOMPILE_ENABLE_STRICTNESS},
        ID3DBlob, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
    },
    Direct3D11::{
        ID3D11Buffer, ID3D11Device, ID3D11DeviceContext, ID3D11PixelShader, ID3D11RenderTargetView,
        ID3D11SamplerState, ID3D11ShaderResourceView, ID3D11Texture2D, ID3D11VertexShader,
        D3D11_BIND_CONSTANT_BUFFER, D3D11_BIND_RENDER_TARGET, D3D11_BIND_SHADER_RESOURCE,
        D3D11_BUFFER_DESC, D3D11_COMPARISON_ALWAYS, D3D11_FILTER_MIN_MAG_MIP_LINEAR,
        D3D11_SAMPLER_DESC, D3D11_SUBRESOURCE_DATA, D3D11_TEXTURE2D_DESC,
        D3D11_TEXTURE_ADDRESS_CLAMP, D3D11_USAGE_DEFAULT, D3D11_VIEWPORT,
    },
    Dxgi::{DXGI_ERROR_DEVICE_HUNG, DXGI_ERROR_DEVICE_REMOVED, DXGI_ERROR_DEVICE_RESET},
};

use crate::backdrop::{
    local_backdrop, refresh_local_backdrop, start_local_feed, LocalFeedState, SurfaceRoute,
};
use crate::broker::client::BrokerClient;
use crate::capture::{CaptureFeed, FrameHost};
use crate::params::{BackdropSource, GlassShaderParams};
use crate::quality::{detect_gpu_class, fullscreen_foreground, resolve, GpuClass, QualityCaps};
use crate::surface::{create_d3d11_device, SharedCompositor};
use crate::window_capture::WindowCaptureFeed;

/// Whether the D3D11 device is gone (driver reset/TDR). All GPU work fails
/// after this; continuing to submit would crash-loop the driver instead of
/// letting it recover.
fn device_lost(device: &ID3D11Device) -> bool {
    match unsafe { device.GetDeviceRemovedReason() } {
        Ok(()) => false,
        Err(error) => matches!(
            error.code(),
            DXGI_ERROR_DEVICE_REMOVED | DXGI_ERROR_DEVICE_RESET | DXGI_ERROR_DEVICE_HUNG
        ),
    }
}

/// The capture pool size multiplier. WGC renders content into the pool at
/// native scale (a smaller pool CROPS and zooms, it does not downscale), so
/// this must stay 1.0 unless the frame is actively downscaled during the
/// copy. Kept as a knob: the sharing, throttling and pause wins are
/// independent of it.
pub(crate) const CAPTURE_SCALE: f32 = 1.0;

/// How often the shared material watch stats the material file. Settings
/// edits land within this delay; the stat is one syscall per interval, so
/// this is cheap to keep aggressive.
const MATERIAL_POLL: Duration = Duration::from_millis(250);

/// Insurance against a misbehaving capture driver: with dirty-region skips
/// active, the backdrop still force-refreshes at least this often.
const FORCE_BACKDROP_SECS: f32 = 2.0;

/// How often the render loop re-checks the fullscreen-foreground state that
/// throttles the backdrop rate under games/presentations.
const FULLSCREEN_POLL: Duration = Duration::from_millis(500);

/// How often the render loop re-checks the battery state (Auto tiers only).
const POWER_POLL: Duration = Duration::from_secs(10);

/// The shared material watch: polls the broker-owned material file's mtime
/// on a bounded cadence and, on change, pushes the material onto every
/// registered compositor in this process. Surface-specific state (reveal
/// progress, capsule radius, pill scale, animation time, pointer
/// influence, glass fade, reveal origin) is preserved: the merge starts
/// from each compositor's current params.
#[derive(Default)]
struct MaterialWatch {
    last_mtime: Option<SystemTime>,
    last_check: Option<Instant>,
}

impl MaterialWatch {
    fn poll(&mut self, state: &mut EngineState) {
        // Bounded cadence: the file is only statted (and parsed) once per
        // interval, not once per render tick.
        let now = Instant::now();
        if self
            .last_check
            .is_some_and(|at| now.duration_since(at) < MATERIAL_POLL)
        {
            return;
        }
        self.last_check = Some(now);
        let Some(path) = crate::broker::material_path() else {
            return;
        };
        let mtime = std::fs::metadata(path)
            .and_then(|metadata| metadata.modified())
            .ok();
        if mtime == self.last_mtime {
            return;
        }
        self.last_mtime = mtime;
        let material = crate::broker::read_material();
        state.apply_quality(material.quality);
        for (_, compositor, _route) in &state.surfaces {
            let Ok(mut compositor) = compositor.lock() else {
                continue;
            };
            // The lite composite shader follows the tier; flipping it is
            // idempotent per compositor (the flag is compared before use).
            compositor.set_lite_shader(state.quality.lite_shader);
            // Pinned surfaces derive their params from the material with
            // app-side rules; the app re-applies the material itself.
            if compositor.material_locked() {
                continue;
            }
            let mut merged = compositor.active_params();
            material.apply_onto(&mut merged, CAPTURE_SCALE);
            let _ = compositor.set_params(&state.context, merged);
        }
    }
}

pub(crate) const KAWASE_HLSL: &str = include_str!("kawase.hlsl");

/// What a composite pass samples: the shared blurred backdrop, or (blur
/// off) the sharp capture itself. `crop` records the capture region the
/// backdrop was built over (identity for the full-frame sharp path and the
/// broker's full-monitor publication): the composite's capture→backdrop
/// uv mapping is stamped from it at present time, so a backdrop blurred
/// over a sub-region of the capture stays aligned with the screen.
#[derive(Clone)]
pub(crate) struct Backdrop {
    pub(crate) view: ID3D11ShaderResourceView,
    pub(crate) crop: CropConstants,
}

pub(crate) struct EngineState {
    pub device: ID3D11Device,
    pub context: ID3D11DeviceContext,
    /// The newest GLOBAL capture frame, copied out of the WGC frame pool
    /// (whose surfaces are recycled by the system) into a stable texture,
    /// plus the shader view created alongside it.
    pub frame_texture: Option<ID3D11Texture2D>,
    pub capture_view: Option<ID3D11ShaderResourceView>,
    /// Current GLOBAL frame (and pool) size in pixels.
    pub frame_size: SizeInt32,
    /// Registered surfaces: (id, compositor, backdrop route). The route
    /// says which feed paints the surface: the GLOBAL monitor capture or a
    /// LOCAL per-window feed.
    pub surfaces: Vec<(u64, SharedCompositor, SurfaceRoute)>,
    pub next_surface_id: u64,
    pub render_running: Arc<AtomicBool>,
    /// The engine-level blur: one dual-Kawase chain over the GLOBAL
    /// capture's union-of-active-surfaces region (or the whole capture when
    /// that cannot be placed), rendered once per fresh frame and sampled by
    /// every Global-sourced surface's composite. `None` until the first
    /// blurred frame is needed.
    blur: Option<BlurEngine>,
    /// The last rendered GLOBAL backdrop, the capture version it was built
    /// from and the material params it was built for; reused by every
    /// animation-driven frame in between.
    backdrop: Option<Backdrop>,
    backdrop_version: u64,
    backdrop_inputs: Option<BlurInputs>,
    /// The capture-pixel rect (x0, y0, x1, y1) the last GLOBAL backdrop was
    /// blurred over (`None` = the full frame). A change forces one rebuild:
    /// the crop is the union of the awake surfaces' regions, and a surface
    /// must never sample a backdrop whose region no longer contains it.
    backdrop_crop: Option<[u32; 4]>,
    /// The resolved quality tier: the caps every backdrop build obeys.
    pub(crate) quality: QualityCaps,
    /// The material's quality mode (kept so power-state transitions can
    /// re-resolve the tier).
    quality_mode: crate::quality::QualityMode,
    /// The GPU class the device runs on (tier resolution input).
    gpu_class: GpuClass,
    /// The backdrop rate cap currently in force (tier × fullscreen state);
    /// the capture callbacks read it through the capture-due gates.
    pub(crate) backdrop_interval: Option<Duration>,
    /// When the last GLOBAL capture blit ran (rate-cap bookkeeping).
    last_copy: Option<Instant>,
    /// When the GLOBAL backdrop last rebuilt (force-refresh cadence).
    last_backdrop_build: Option<Instant>,
    /// The last battery state seen by the tier resolution.
    last_battery: Option<bool>,
    /// The LOCAL feeds: one entry per captured app window (the pool of
    /// windows with Local-sourced surfaces over them). Each entry carries
    /// its own capture texture, its own small Kawase chain and its own
    /// cached backdrop. `BackdropSource::Local` surfaces route here
    /// through `backdrop::local_backdrop`; the GLOBAL feed above stays
    /// untouched — an app with only Local surfaces never starts a monitor
    /// capture at all.
    pub(crate) local: HashMap<isize, LocalFeedState>,
}

impl EngineState {
    /// Replaces the shared capture frame and its view (called by the
    /// capture feed when a frame lands or the monitor's size changes).
    pub(crate) fn set_frame(&mut self, frame: (ID3D11Texture2D, ID3D11ShaderResourceView)) {
        (self.frame_texture, self.capture_view) = (Some(frame.0), Some(frame.1));
    }

    /// Resolves the tier from the material's quality mode on the current
    /// hardware and battery state, applying it to engine-level state (the
    /// per-surface lite-shader flag is pushed by the material watch).
    pub(crate) fn apply_quality(&mut self, mode: crate::quality::QualityMode) {
        self.quality_mode = mode;
        self.quality = resolve(mode, self.gpu_class);
        self.backdrop_interval = self.quality.backdrop_interval(false);
    }
}

impl FrameHost for EngineState {
    fn device(&self) -> &ID3D11Device {
        &self.device
    }

    fn copy_context(&self) -> ID3D11DeviceContext {
        self.context.clone()
    }

    fn set_frame(&mut self, frame: (ID3D11Texture2D, ID3D11ShaderResourceView), size: SizeInt32) {
        self.frame_size = size;
        EngineState::set_frame(self, frame);
    }

    fn frame_texture(&self) -> Option<&ID3D11Texture2D> {
        self.frame_texture.as_ref()
    }

    fn frame_size(&self) -> SizeInt32 {
        self.frame_size
    }

    fn any_active(&self) -> bool {
        // Only the GLOBAL feed's consumers count: Local-only glass must
        // not keep a monitor capture (or its blit) alive for nothing.
        self.surfaces.iter().any(|(_, compositor, route)| {
            route.source == BackdropSource::Global
                && compositor
                    .lock()
                    .map(|compositor| !compositor.is_paused())
                    .unwrap_or(true)
        })
    }

    fn capture_due(&mut self) -> bool {
        // The render loop owns `backdrop_interval` (it polls the fullscreen
        // state); the capture callback only consults it against the last
        // blit time. This keeps the shell query off the WGC worker thread.
        let due = self.backdrop_interval.map_or(true, |interval| {
            self.last_copy
                .map_or(true, |last| last.elapsed() >= interval)
        });
        if due {
            self.last_copy = Some(Instant::now());
        }
        due
    }
}

pub(crate) struct SharedEngine {
    pub state: Arc<Mutex<EngineState>>,
    /// The single GLOBAL capture session, owned by the render loop (started
    /// with the first awake Global surface, stopped when the last one
    /// unregisters — or when every one parks, so the OS capture pipeline
    /// idles with the glass). Registration and teardown reach it through
    /// this handle. A process with only Local surfaces never has one.
    feed: Arc<Mutex<Option<CaptureFeed>>>,
    /// The LOCAL capture sessions, one per captured window, keyed by the
    /// host HWND value. Also owned by the render loop; kept outside the
    /// engine state so a session teardown (`Drop` → OS calls) never runs
    /// while the engine mutex is held.
    local_feeds: Arc<Mutex<HashMap<isize, WindowCaptureFeed>>>,
}

static ENGINE: OnceLock<SharedEngine> = OnceLock::new();

pub(crate) fn engine() -> &'static SharedEngine {
    ENGINE.get_or_init(|| {
        let (device, context) = create_d3d11_device().expect("Glassbar shared glass device failed");
        let gpu_class = detect_gpu_class(&device);
        let mut state = EngineState {
            device: device.clone(),
            context,
            frame_texture: None,
            capture_view: None,
            frame_size: SizeInt32::default(),
            surfaces: Vec::new(),
            next_surface_id: 0,
            render_running: Arc::new(AtomicBool::new(true)),
            blur: None,
            backdrop: None,
            backdrop_version: 0,
            backdrop_inputs: None,
            backdrop_crop: None,
            quality: QualityCaps::HIGH,
            quality_mode: crate::quality::QualityMode::Auto,
            gpu_class,
            backdrop_interval: None,
            last_copy: None,
            last_backdrop_build: None,
            last_battery: None,
            local: HashMap::new(),
        };
        // The initial tier resolves from the material's mode (default auto)
        // on this machine's GPU class and battery state.
        state.apply_quality(crate::broker::read_material().quality);
        let state = Arc::new(Mutex::new(state));
        let version = Arc::new(AtomicU64::new(0));
        let feed = Arc::new(Mutex::new(None));
        let local_feeds = Arc::new(Mutex::new(HashMap::new()));
        spawn_render_loop(
            Arc::clone(&state),
            Arc::clone(&version),
            Arc::clone(&feed),
            Arc::clone(&local_feeds),
        );
        SharedEngine {
            state,
            feed,
            local_feeds,
        }
    })
}

impl SharedEngine {
    /// Registers a surface's compositor together with its backdrop route
    /// (which feed paints it, and for Local surfaces over which host
    /// window). The captures themselves are the render loop's call: its
    /// very next tick sees the awake surface and starts the feed (the
    /// GLOBAL monitor capture or the LOCAL window capture — or attaches to
    /// the broker, which replaces the global pipeline entirely). One owner
    /// for the whole lifecycle.
    pub(crate) fn register(&self, compositor: SharedCompositor, route: SurfaceRoute) -> u64 {
        let mut state = self.state.lock().expect("shared glass lock poisoned");
        let id = state.next_surface_id;
        state.next_surface_id += 1;
        state.surfaces.push((id, Arc::clone(&compositor), route));
        // A Local surface's window needs its feed entry from now on (the
        // render loop starts the capture for it); the entry persists after
        // unregister until this call removes it again, so its cached
        // backdrop survives brief park cycles.
        if let Some(window) = route.window {
            if matches!(route.source, crate::params::BackdropSource::Local) {
                state
                    .local
                    .entry(window)
                    .or_insert_with(LocalFeedState::new);
            }
        }
        // The fresh surface adopts the shared material: the merge starts
        // from what the caller configured (surface-specific fields like the
        // capsule radius survive), so a new glass window is styled
        // consistently with every other glass app in the session. Pinned
        // surfaces (the app re-derives params itself) are skipped: the app
        // seeds them right after creation.
        let material = crate::broker::read_material();
        if let Ok(mut compositor) = compositor.lock() {
            // A surface registered after the last material change still
            // needs the tier's composite shader variant.
            compositor.set_lite_shader(state.quality.lite_shader);
            if !compositor.material_locked() {
                let mut merged = compositor.active_params();
                material.apply_onto(&mut merged, CAPTURE_SCALE);
                let _ = compositor.set_params(&state.context, merged);
            }
        }
        id
    }

    /// Unregisters a surface; stops the GLOBAL capture with the last
    /// Global surface and tears down LOCAL feeds whose last consumer is
    /// gone (their cached backdrops go with them — nothing can present
    /// them anymore).
    pub(crate) fn unregister(&self, id: u64) {
        let dropped_window_backdrops: Vec<isize>;
        let global_empty = {
            let mut state = self.state.lock().expect("shared glass lock poisoned");
            state.surfaces.retain(|(existing, _, _)| *existing != id);
            dropped_window_backdrops = state
                .local
                .keys()
                .copied()
                .filter(|window| {
                    !state
                        .surfaces
                        .iter()
                        .any(|(_, _, route)| route.consumes_window(*window))
                })
                .collect();
            for window in &dropped_window_backdrops {
                state.local.remove(window);
            }
            !state
                .surfaces
                .iter()
                .any(|(_, _, route)| route.source == BackdropSource::Global)
        };
        if global_empty {
            // No global glass left: stop the monitor capture entirely (the
            // broker client parks itself on its next tick — see the loop's
            // route conditions).
            *self.feed.lock().expect("shared glass feed lock poisoned") = None;
        }
        // Session teardowns must never run under a lock the frame callbacks
        // may be waiting on.
        if !dropped_window_backdrops.is_empty() {
            let mut feeds = self
                .local_feeds
                .lock()
                .expect("shared glass local feed lock poisoned");
            for window in dropped_window_backdrops {
                feeds.remove(&window);
            }
        }
    }

    /// The capture downscale factor, for compensating texel-unit params.
    pub(crate) fn capture_scale(&self) -> f32 {
        CAPTURE_SCALE
    }
}

/// One snapshot of what the blur needs from the live surfaces: the active
/// (unpaused) surfaces' shared material params.
#[derive(PartialEq)]
pub(crate) struct BlurInputs {
    pub(crate) params: GlassShaderParams,
}

impl BlurInputs {
    /// Collects from the registered surfaces. Every surface derives its
    /// material from the same saved settings, so the blur-relevant params
    /// (radius, samples) are identical across surfaces; the first active
    /// surface's copy is authoritative.
    pub(crate) fn collect(state: &EngineState) -> Option<Self> {
        Self::collect_matching(state, |_| true)
    }

    /// Like [`BlurInputs::collect`], but only from the surfaces the given
    /// route filter selects (the GLOBAL feed ignores Local surfaces and
    /// each window's LOCAL feed only reads its own consumers).
    pub(crate) fn collect_matching(
        state: &EngineState,
        select: impl Fn(&SurfaceRoute) -> bool,
    ) -> Option<Self> {
        let mut params = None;
        for (_, compositor, route) in &state.surfaces {
            if !select(route) {
                continue;
            }
            let Ok(compositor) = compositor.lock() else {
                continue;
            };
            if compositor.is_paused() {
                continue;
            }
            if params.is_none() {
                params = Some(compositor.active_params());
            }
        }
        params.map(|params| Self { params })
    }
}

/// The maximum sample displacement the composite can apply beyond a
/// surface's bounds, in capture texels: edge refraction, chromatic
/// dispersion, the pointer-driven bend (its velocity input is clamped to
/// ±36 screen px), the reveal turbulence and wobble, plus a small constant
/// so bilinear interpolation at the crop's rim always has real texels.
fn displacement_reach_texels(params: &GlassShaderParams) -> f32 {
    params.refraction_strength
        + params.chromatic_offset_texels
        + params.turbulence_strength
        + params.reveal_wobble * 5.0
        + params.pointer_influence * 36.0
        + 8.0
}

/// Rounds a pixel rect outward to a multiple of `grid` px, clamped into the
/// frame. Rounding keeps small surface moves from churning Kawase chain
/// rebuilds (whose textures are sized to the crop) at the cost of a few
/// dozen wasted texel rows.
fn grid_aligned_crop(rect: [f32; 4], frame: [u32; 2], grid: f32) -> Option<[u32; 4]> {
    let width = frame[0].max(1) as f32;
    let height = frame[1].max(1) as f32;
    let floor = |value: f32| (value / grid).floor() * grid;
    let ceil = |value: f32| (value / grid).ceil() * grid;
    let x0 = floor(rect[0].max(0.0)).clamp(0.0, width);
    let y0 = floor(rect[1].max(0.0)).clamp(0.0, height);
    let x1 = ceil(rect[2].min(width)).max(x0).min(width);
    let y1 = ceil(rect[3].min(height)).max(y0).min(height);
    if x1 - x0 < 1.0 || y1 - y0 < 1.0 {
        return None;
    }
    Some([x0 as u32, y0 as u32, x1 as u32, y1 as u32])
}

/// The region of the capture the shared GLOBAL chain should blur: the union
/// of the awake Global surfaces' sampling rects grown by the composite's
/// displacement reach, rounded to a 64 px grid. A taskbar pinned to one
/// monitor edge turns the whole-monitor blur into a thin-strip blur.
/// `None` = the full frame (also the fallback when an awake Global
/// surface's sampling rect cannot be placed in monitor space).
fn backdrop_crop_rect(
    state: &EngineState,
    frame: [u32; 2],
    params: &GlassShaderParams,
) -> Option<[u32; 4]> {
    let mut union: Option<[f32; 4]> = None;
    for (_, compositor, route) in &state.surfaces {
        if route.source != BackdropSource::Global {
            continue;
        }
        let Ok(compositor) = compositor.lock() else {
            continue;
        };
        if compositor.is_paused() {
            continue;
        }
        let Some(rect) = compositor.monitor_uv_rect() else {
            return None;
        };
        union = Some(match union {
            None => rect,
            Some(current) => [
                current[0].min(rect[0]),
                current[1].min(rect[1]),
                current[2].max(rect[2]),
                current[3].max(rect[3]),
            ],
        });
    }
    let union = union?;
    let width = frame[0].max(1) as f32;
    let height = frame[1].max(1) as f32;
    let reach = displacement_reach_texels(params);
    grid_aligned_crop(
        [
            union[0] * width - reach,
            union[1] * height - reach,
            union[2] * width + reach,
            union[3] * height + reach,
        ],
        frame,
        64.0,
    )
}

/// Renders the shared backdrop once per tick when the capture has changed.
/// With a static desktop (and during animations, which re-render only the
/// composite) the previous backdrop is reused untouched. Returns whether the
/// blur failed (the caller checks for device loss on failure).
fn refresh_backdrop(state: &mut EngineState, version: u64, force: bool) -> bool {
    let Some(capture) = state.capture_view.clone() else {
        return false;
    };
    let Some(inputs) = BlurInputs::collect(state) else {
        // Nothing is awake to consume a backdrop; keep the last one so a
        // surface resuming over a static desktop has material immediately.
        return false;
    };
    // Insurance against drivers that report dirty regions inaccurately: if
    // unchanged-frame skips somehow starve the backdrop, it still
    // force-refreshes on this cadence (two seconds of worst-case lag).
    let stale = state.last_backdrop_build.map_or(true, |last| {
        last.elapsed().as_secs_f32() >= FORCE_BACKDROP_SECS
    });
    // The blur only ever covers the awake surfaces' region (the union crop,
    // None = the full frame); when the glass moves, the crop moves with it
    // and the rebuild happens this tick rather than waiting for a fresh
    // capture frame.
    let frame = [
        state.frame_size.Width.max(1) as u32,
        state.frame_size.Height.max(1) as u32,
    ];
    let crop = backdrop_crop_rect(state, frame, &inputs.params);
    let crop_changed = crop != state.backdrop_crop;
    // Idle tick fast path: the backdrop already matches the newest capture
    // frame and the current material: the passes only run when one of those
    // moved (a settings change re-blurs even on a static desktop).
    if !force
        && !stale
        && !crop_changed
        && state.backdrop_version == version
        && state
            .backdrop_inputs
            .as_ref()
            .is_some_and(|last| *last == inputs)
    {
        return false;
    }
    state.backdrop_crop = crop;
    let blur = state
        .blur
        .get_or_insert_with(|| BlurEngine::new(&state.device));
    match blur.render(
        &state.context,
        &capture,
        frame,
        crop,
        &inputs.params,
        &state.quality,
    ) {
        Ok(backdrop) => {
            state.backdrop = Some(backdrop);
            state.backdrop_version = version;
            state.backdrop_inputs = Some(inputs);
            state.last_backdrop_build = Some(Instant::now());
            false
        }
        Err(error) => {
            eprintln!("Glassbar shared blur failed: {error}");
            true
        }
    }
}

/// Hysteresis for the feed lifecycle (about two seconds at the loop's ~60Hz
/// tick): brief all-parked moments (a settings preview, a quick hide/show)
/// must not churn the capture session.
const IDLE_FEED_TICKS: u32 = 120;

/// Escalating restart backoff for a capture feed that keeps failing to
/// start. The first failure waits one hysteresis window (~2s); every
/// further consecutive failure quadruples the wait (capped at ~2 minutes),
/// so a permanently rejected capture degrades into a quiet retry every
/// couple of minutes instead of hammering the capture service — and
/// flooding the log — every two seconds.
struct RestartBackoff {
    consecutive_failures: u32,
    ticks: u32,
}

impl RestartBackoff {
    fn new() -> Self {
        Self {
            consecutive_failures: 0,
            ticks: 0,
        }
    }

    /// Whether a start attempt may run now; consumes one waited tick
    /// otherwise (the loop's own cadence drives the countdown).
    fn ready(&mut self) -> bool {
        if self.ticks > 0 {
            self.ticks -= 1;
            false
        } else {
            true
        }
    }

    fn note_success(&mut self) {
        self.consecutive_failures = 0;
        self.ticks = 0;
    }

    /// Records a failed start: escalates the wait and logs — the first
    /// failure with the bare error, the second with the recovery posture
    /// and the known refusal causes, further ones silently.
    fn note_failure(&mut self, scope: &str, error: &str) {
        self.consecutive_failures += 1;
        let level = self.consecutive_failures.min(4);
        self.ticks = IDLE_FEED_TICKS
            .saturating_mul(1u32 << (2 * (level - 1)))
            .min(IDLE_FEED_TICKS.saturating_mul(64));
        match self.consecutive_failures {
            1 => eprintln!("Glassbar {scope} failed to start: {error}"),
            2 => eprintln!(
                "Glassbar {scope} keeps failing to start ({error}); retrying at a slower \
                 cadence. Window capture is refused for some hosts: windows hidden from \
                 the taskbar (tauri's skipTaskbar calls ITaskbarList::DeleteTab), \
                 WS_EX_NOACTIVATE / WS_EX_TOOLWINDOW windows, minimized windows, and \
                 windows protected with SetWindowDisplayAffinity."
            ),
            _ => {}
        }
    }
}

fn spawn_render_loop(
    state: Arc<Mutex<EngineState>>,
    version: Arc<AtomicU64>,
    feed: Arc<Mutex<Option<CaptureFeed>>>,
    local_feeds: Arc<Mutex<HashMap<isize, WindowCaptureFeed>>>,
) {
    std::thread::spawn(move || {
        // The feed's host view of the engine state: the capture pipeline
        // is host-agnostic (the broker runs the exact same code over its
        // own host), so this loop owns the trait object locally.
        let host: Arc<Mutex<dyn FrameHost>> = Arc::clone(&state) as _;
        let mut last_version = 0u64;
        let mut idle_ticks = 0u32;
        // After a failed capture start, wait a full hysteresis window before
        // retrying so a persistent failure does not spam the session at the
        // loop's tick rate.
        let mut restart_backoff = RestartBackoff::new();
        // The broker client machine (see `broker::client`): every tick
        // advances attach/pull; a silent failure falls back to the local
        // pipeline by returning an `Err` once and resetting to `Idle`.
        let mut broker = BrokerClient::new();
        // The shared material watch (see [`MaterialWatch`]): file changes
        // restyle every registered compositor within one poll cadence.
        let mut material = MaterialWatch::default();
        // Fullscreen foreground state (throttled shell query) that drops
        // the backdrop rate to the tier's fullscreen cap.
        let mut fullscreen = false;
        let mut last_fullscreen_check: Option<Instant> = None;
        // Battery transitions re-resolve the Auto tier (throttled query).
        let mut last_power_check: Option<Instant> = None;
        // The caps the current backdrop was built under: a tier change
        // (settings edit or battery transition) must rebuild it.
        let mut last_caps: Option<QualityCaps> = None;
        // The interval already applied to the capture session (applied only
        // on change, outside the state lock).
        let mut applied_session_interval: Option<Duration> = None;
        // Per-LOCAL-window bookkeeping: the frame version still owed to
        // surfaces (dispatched after each pass), the parked-idle counters
        // and per-window start backoff.
        let mut local_signal: HashMap<isize, u64> = HashMap::new();
        let mut local_seen_version: HashMap<isize, u64> = HashMap::new();
        let mut local_idle_ticks: HashMap<isize, u32> = HashMap::new();
        let mut local_restarts: HashMap<isize, RestartBackoff> = HashMap::new();
        'main: while state
            .lock()
            .expect("shared glass lock poisoned")
            .render_running
            .load(Ordering::Relaxed)
        {
            std::thread::sleep(Duration::from_millis(16));
            let mut engine = state.lock().expect("shared glass lock poisoned");

            // The shared material may have changed on disk (a settings UI
            // in any glass app wrote it); push it onto the compositors.
            material.poll(&mut engine);

            // Slow-cadence environment re-resolution: the fullscreen and
            // power state gate the tier's backdrop rate; both shell calls
            // are cheap but not free, and neither changes per frame.
            let now = Instant::now();
            if last_fullscreen_check.map_or(true, |at| now.duration_since(at) >= FULLSCREEN_POLL) {
                last_fullscreen_check = Some(now);
                fullscreen = fullscreen_foreground();
            }
            if last_power_check.map_or(true, |at| now.duration_since(at) >= POWER_POLL) {
                last_power_check = Some(now);
                let battery = crate::quality::on_battery();
                if engine.last_battery != Some(battery) {
                    engine.last_battery = Some(battery);
                    let mode = engine.quality_mode;
                    engine.apply_quality(mode);
                    let lite = engine.quality.lite_shader;
                    for (_, compositor, _route) in &engine.surfaces {
                        if let Ok(mut compositor) = compositor.lock() {
                            compositor.set_lite_shader(lite);
                        }
                    }
                }
            }
            engine.backdrop_interval = engine.quality.backdrop_interval(fullscreen);
            let effective_interval = engine.backdrop_interval;

            // The awake-GLOBAL-surface answer drives the global pipelines:
            // the broker's slot flag and the monitor feed's lifecycle.
            // LOCAL feeds are gated per capturing window further down. A
            // compositor whose lock is poisoned counts as awake — a stuck
            // lock must never park the glass.
            let any_global_active = engine.surfaces.iter().any(|(_, compositor, route)| {
                route.source == BackdropSource::Global
                    && compositor.lock().map(|c| !c.is_paused()).unwrap_or(true)
            });
            // A tier change (settings edit, battery transition) changes the
            // chain caps of EVERY feed: force one rebuild even without a
            // fresh capture frame.
            let tier_changed = last_caps.map_or(false, |caps| caps != engine.quality);
            last_caps = Some(engine.quality);

            // Broker attempt first: when it hands over a fresh backdrop the
            // global pipeline is skipped entirely this tick. Only Global
            // consumers participate — the broker publishes the shared
            // screen material, which Local glass must not adopt. A process
            // with no Global surfaces (and no attachment yet) never starts
            // the broker machine at all: a local-only app neither spawns
            // nor contacts a broker.
            let mut frame_changed = false;
            let mut broker_attached = false;
            if any_global_active || broker.attached() {
                match broker.tick(&engine.device, &engine.context, any_global_active) {
                    Ok(Some(frame)) => {
                        engine.backdrop = Some(frame.backdrop);
                        frame_changed = true;
                        broker_attached = true;
                    }
                    Ok(None) => broker_attached = broker.attached(),
                    Err(error) => {
                        eprintln!(
                            "glass broker unavailable, running the in-process engine instead: \
                             {error}"
                        );
                        // Forget whatever the broker published: the global
                        // pipeline must build the next backdrop fresh.
                        engine.backdrop = None;
                    }
                }
            }

            let current = version.load(Ordering::Relaxed);
            let version_changed = current != last_version;
            last_version = current;
            // One blur per fresh capture frame — never per surface, never
            // per animation tick. The first tick after startup forces one
            // render even before a capture frame lands, so freshly spawned
            // surfaces composite from the very first frame. In broker mode
            // the global chain never runs: the published backdrop replaces
            // its output wholesale.
            if !broker_attached {
                let first_backdrop = engine.backdrop.is_none();
                let blur_failed =
                    refresh_backdrop(&mut engine, current, first_backdrop || tier_changed);
                if blur_failed && device_lost(&engine.device) {
                    eprintln!(
                        "Glassbar glass device removed (driver reset); \
                         stopping the glass engine"
                    );
                    engine.render_running.store(false, Ordering::Relaxed);
                    break 'main;
                }
            }

            // Each LOCAL feed blurs its own window capture independently of
            // the broker above; refreshes follow the same
            // content-force/tier-force cadence. The window's present
            // signal — a fresh frame surfaces have not seen yet — is
            // snapshot per window and consumed by the dispatch below.
            for window in engine.local.keys().copied().collect::<Vec<_>>() {
                let first_backdrop = engine
                    .local
                    .get(&window)
                    .map_or(false, |entry| entry.backdrop.is_none());
                let blur_failed =
                    refresh_local_backdrop(&mut engine, window, first_backdrop || tier_changed);
                if blur_failed && device_lost(&engine.device) {
                    eprintln!(
                        "Glassbar glass device removed (driver reset); \
                         stopping the glass engine"
                    );
                    engine.render_running.store(false, Ordering::Relaxed);
                    break 'main;
                }
                let fresh_version = engine
                    .local
                    .get(&window)
                    .map_or(0, |entry| entry.version.load(Ordering::Relaxed));
                local_signal.insert(window, fresh_version);
            }

            // One drawer, routed by each surface's backdrop feed: a Global
            // surface composites the shared monitor backdrop (driven by the
            // monitor's frame version, or the broker's pull once attached),
            // a Local surface its own window's backdrop (driven by that
            // window's frame version). Animations and dirty params still
            // request frames through `wants_frame` regardless.
            for (_, compositor, route) in &engine.surfaces {
                let Ok(mut compositor) = compositor.lock() else {
                    continue;
                };
                let device_lost = match route.source {
                    BackdropSource::Global => {
                        let present_signal = if broker_attached {
                            frame_changed
                        } else {
                            version_changed
                        };
                        if compositor.wants_frame(present_signal) {
                            engine.backdrop.as_ref().map_or(false, |backdrop| {
                                failed_present(&mut compositor, &engine.context, backdrop)
                            })
                        } else {
                            false
                        }
                    }
                    BackdropSource::Local => {
                        let Some(window) = route.window else {
                            continue;
                        };
                        let fresh = local_signal.get(&window).copied().unwrap_or(0);
                        let changed =
                            fresh != local_seen_version.get(&window).copied().unwrap_or(0);
                        if compositor.wants_frame(changed) {
                            match local_backdrop(&engine, window) {
                                Some(backdrop) => {
                                    // `failed_present` needs a `&mut
                                    // compositor` while `backdrop` borrows
                                    // `engine`: clone the cheap COM view
                                    // out of the engine borrow first.
                                    let backdrop = backdrop.clone();
                                    failed_present(&mut compositor, &engine.context, &backdrop)
                                }
                                None => false,
                            }
                        } else {
                            false
                        }
                    }
                };
                if device_lost {
                    eprintln!(
                        "Glassbar glass device removed (driver reset); \
                         stopping the glass engine"
                    );
                    engine.render_running.store(false, Ordering::Relaxed);
                    break 'main;
                }
            }
            // Everything that needed the newest frame version saw it; the
            // next versions become the following tick's fresh signal.
            local_seen_version.extend(local_signal.drain());
            drop(engine);
            // The session-level rate cap (Windows 11 22H2+): WGC coalesces
            // its frame deliveries to the interval, so the capture callback
            // — and its blit — never even run above the cap. Applied
            // outside the state lock, only when the interval moved (tier
            // change or fullscreen transition), to BOTH the monitor feed
            // and every live local feed. Older systems ignore this and
            // rely on the callbacks' capture-due gates.
            if effective_interval != applied_session_interval {
                applied_session_interval = effective_interval;
                if let Some(active_feed) = feed
                    .lock()
                    .expect("shared glass feed lock poisoned")
                    .as_ref()
                {
                    active_feed.set_min_update_interval(effective_interval);
                }
                for local_feed in local_feeds
                    .lock()
                    .expect("shared glass local feed lock poisoned")
                    .values()
                {
                    local_feed.set_min_update_interval(effective_interval);
                }
            }
            // GLOBAL feed lifecycle, strictly outside the state lock:
            // starting the feed re-locks it, and dropping a feed while its
            // own frame callback blocks on that lock could deadlock the
            // teardown.
            if broker_attached {
                // The broker publishes the backdrop: the monitor capture
                // and blur pipeline is pure duplicate cost — tear it down.
                // The broker's own feed is parked through the slot's active
                // flag (the client wrote it) whenever every global surface
                // sleeps.
                idle_ticks = 0;
                *feed.lock().expect("shared glass feed lock poisoned") = None;
            } else if any_global_active {
                idle_ticks = 0;
                if restart_backoff.ready()
                    && !feed
                        .lock()
                        .expect("shared glass feed lock poisoned")
                        .is_some()
                {
                    match CaptureFeed::start(&host, &version) {
                        Ok(new_feed) => {
                            restart_backoff.note_success();
                            *feed.lock().expect("shared glass feed lock poisoned") = Some(new_feed)
                        }
                        Err(error) => restart_backoff.note_failure("shared capture", &error),
                    }
                }
            } else {
                idle_ticks += 1;
                if idle_ticks >= IDLE_FEED_TICKS {
                    idle_ticks = 0;
                    // Every Global surface is parked: the monitor capture is
                    // torn down so the OS capture pipeline idles with it.
                    // The last backdrop stays cached; the first surface to
                    // wake restarts the feed above and composites
                    // immediately from the cached backdrop while the first
                    // frame flies.
                    *feed.lock().expect("shared glass feed lock poisoned") = None;
                }
            }
            // LOCAL feed lifecycles, also strictly outside the state lock:
            // one capture session per consumed window, started with the
            // first awake surface over it and idled out on the same
            // hysteresis. The per-window awake flags are snapshotted under
            // a quick lock, then acted on with the state lock released.
            let local_awake: HashMap<isize, bool> = {
                let engine = state.lock().expect("shared glass lock poisoned");
                engine
                    .local
                    .keys()
                    .copied()
                    .map(|window| {
                        let awake = engine.surfaces.iter().any(|(_, compositor, route)| {
                            route.consumes_window(window)
                                && compositor.lock().map(|c| !c.is_paused()).unwrap_or(true)
                        });
                        (window, awake)
                    })
                    .collect()
            };
            let mut feeds = local_feeds
                .lock()
                .expect("shared glass local feed lock poisoned");
            // Windows whose last surface unregistered: drop their retry
            // bookkeeping so a later re-registration starts fresh.
            local_restarts.retain(|window, _| local_awake.contains_key(window));
            for (window, awake) in local_awake {
                if awake {
                    if let Some(ticks) = local_idle_ticks.get_mut(&window) {
                        *ticks = 0;
                    }
                    if !feeds.contains_key(&window) {
                        let backoff = local_restarts.entry(window).or_insert_with(RestartBackoff::new);
                        if !backoff.ready() {
                            continue;
                        }
                        // Feeds are started (and their deadlocked-teardown
                        // hazard managed) only ever outside the engine lock.
                        match start_local_feed(&state, window) {
                            Ok(new_feed) => {
                                backoff.note_success();
                                feeds.insert(window, new_feed);
                            }
                            Err(error) => backoff.note_failure(
                                &format!("the local capture of window {window}"),
                                &error,
                            ),
                        }
                    }
                } else if let Some(ticks) = local_idle_ticks.get_mut(&window) {
                    *ticks += 1;
                    if *ticks >= IDLE_FEED_TICKS {
                        *ticks = 0;
                        // The window's glass is parked: its capture session
                        // is torn down (the cached backdrop stays alive for
                        // the wake-up composite).
                        feeds.remove(&window);
                    }
                }
            }
        }
    });
}

/// Runs one compositor present for the routed backdrop and reports whether
/// the failure signals a lost device (a driver reset — common on saturated
/// integrated GPUs — fails every later call: stopping the loop lets the
/// driver recover instead of crash-looping it. The glass freezes on its
/// last frame until the app restarts).
fn failed_present(
    compositor: &mut crate::surface::D3D11Compositor,
    context: &ID3D11DeviceContext,
    backdrop: &Backdrop,
) -> bool {
    if let Err(error) = compositor.present(context, backdrop) {
        // A failed present means the surface silently never appears (hosts
        // gate reveals on `has_presented`); surfacing the error here is
        // the only way to see it.
        eprintln!("Glassbar surface render failed: {error}");
        return true;
    }
    false
}

/// How the Kawase chain should run for the current params.
#[derive(Clone, Copy, PartialEq)]
pub(crate) struct BlurPlan {
    /// Blur is effectively off: composites sample the sharp capture
    /// directly, skipping the chain entirely.
    pub(crate) sharp: bool,
    /// Kawase levels actually used (1..=5). Capped by the radius: the
    /// chain's own down/upsampling imposes a minimum blur per level, so a
    /// small radius must use fewer levels no matter the samples setting.
    /// One more level than a pure-hop estimate: the chain's shallowest
    /// buffer runs at base resolution (its hops are undone by the up side
    /// there), so each band needs that extra level to reach the same blur
    /// at the same slider position.
    pub(crate) levels: usize,
    /// Resolution of the chain's base (down0) level relative to the capture
    /// (0.9..1.0). Less blur hides less resampling, so lower radii run the
    /// chain at a higher resolution to avoid visible pixelation.
    pub(crate) base_scale: f32,
    /// Kawase tap-offset scale.
    pub(crate) offset_scale: f32,
}

/// Derives the chain configuration from the shader params. The samples
/// setting is an upper bound on levels; the radius picks the actual depth.
/// The chain's own down/upsampling carries most of the blur, so the level
/// depth stays minimal and the base resolution near the capture's own.
///
/// The per-band ramps make the slider's response CONTINUOUS: within each
/// level band the tap offsets ramp from their start value to 2.0 while the
/// base resolution eases down to 0.9, and each deeper band starts at
/// offset 0.9 / base 1.0 — which, in the `sigma ≈ offsets · 2^levels ·
/// base` model, lands exactly on the previous band's end. Adding a level
/// therefore never snaps: the new level begins at nearly full resolution
/// with tiny taps (a gentle addition) and grows gradually from there.
pub(crate) fn blur_plan_with_caps(params: &GlassShaderParams, caps: &QualityCaps) -> BlurPlan {
    let radius = params.blur_radius_texels;
    if radius < 0.5 {
        return BlurPlan {
            sharp: true,
            levels: 1,
            base_scale: 1.0,
            offset_scale: 0.0,
        };
    }
    let samples = (params.blur_samples.round() as usize).clamp(1, 5);
    // One extra level covers radii up to ~12px; each extra level roughly
    // doubles the reach (bands 12-28, 28-64, 64+). `levels` counts the
    // chain's buffers, so a band of `band_levels` 2x-ish hops runs
    // `band_levels + 1` buffers (down0 crops at the base scale, not at 2x).
    let (band_levels, band_start, band_end) = if radius < 12.0 {
        (1, 1.0, 12.0)
    } else if radius < 28.0 {
        (2, 12.0, 28.0)
    } else if radius < 64.0 {
        (3, 28.0, 64.0)
    } else {
        (4, 64.0, 128.0)
    };
    let position = ((radius - band_start) / (band_end - band_start)).clamp(0.0, 1.0);
    // Band 1 ramps from the sharp-side minimum (0.25); deeper bands start at
    // the base-scale floor (0.9) to match the previous band's end.
    let offsets_start = if band_levels == 1 { 0.25 } else { 0.9 };
    let offset_scale = (offsets_start + (2.0 - offsets_start) * position).clamp(0.25, 2.0);
    // Quality tier: the chain runs at a reduced internal resolution (the
    // composite upsamples it anyway — a blurred backdrop has no fine detail
    // to lose). The reduction ramps in over radius 12..24 so a slider sweep
    // never jumps when it engages, and small radii keep full resolution
    // where resampling would be visible.
    let ramp = ((radius - 12.0) / 12.0).clamp(0.0, 1.0);
    let tier_scale = 1.0 - ramp * (1.0 - caps.blur_internal_scale);
    let base_scale = ((1.0 - 0.1 * position) * tier_scale).clamp(0.25, 1.0);
    let levels = samples.min(band_levels + 1);
    BlurPlan {
        sharp: false,
        levels,
        base_scale,
        offset_scale,
    }
}

/// Mirrors the shader's `CropParams` constant buffer (register b2): the
/// capture-uv geometry of the region the Kawase chain blurred, plus the
/// capture texel size. The composite maps a capture uv through
/// `crop_origin_uv`/`crop_size_uv` into the backdrop's own uv space, so a
/// backdrop blurred over a sub-region of the capture stays aligned with the
/// screen. The sharp path and the broker's publication carry the identity
/// crop (the backdrop covers the whole capture).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CropConstants {
    pub(crate) crop_origin_uv: [f32; 2],
    pub(crate) crop_size_uv: [f32; 2],
    pub(crate) capture_texel_uv: [f32; 2],
    pub(crate) kawase_scale: f32,
    /// Constant buffers must be a multiple of 16 bytes: 12 floats of data
    /// per 16-byte row = 48 bytes total.
    padding: [f32; 5],
}

impl CropConstants {
    /// The identity crop: the backdrop covers the whole capture.
    pub(crate) fn identity(texel_uv: [f32; 2]) -> Self {
        Self {
            crop_origin_uv: [0.0, 0.0],
            crop_size_uv: [1.0, 1.0],
            capture_texel_uv: texel_uv,
            kawase_scale: 0.0,
            padding: [0.0; 5],
        }
    }

    /// The crop for a backdrop blurred over the capture's sub-rectangle
    /// `px` (x0, y0, x1, y1 in capture pixels).
    pub(crate) fn from_pixel_rect(px: [u32; 4], frame: [u32; 2], texel_uv: [f32; 2]) -> Self {
        let width = frame[0].max(1) as f32;
        let height = frame[1].max(1) as f32;
        Self {
            crop_origin_uv: [px[0] as f32 / width, px[1] as f32 / height],
            crop_size_uv: [
                (px[2] - px[0]) as f32 / width,
                (px[3] - px[1]) as f32 / height,
            ],
            capture_texel_uv: texel_uv,
            kawase_scale: 0.0,
            padding: [0.0; 5],
        }
    }
}

/// One mip of the Kawase chain: the texture plus the views its pass and the
/// next pass need.
struct KawaseLevel {
    #[allow(dead_code)]
    texture: ID3D11Texture2D,
    srv: ID3D11ShaderResourceView,
    rtv: ID3D11RenderTargetView,
}

/// The intermediate buffers of the dual-Kawase blur over the (possibly
/// cropped) capture. The down/up halves run at successively halved
/// resolutions with the shallowest pair (down0/up0) at the full base
/// resolution — the chain's output, which the composites sample directly.
/// Recreated whenever the blurred region's size or the level count changes
/// (monitor change, a moved surface union, or params that move the level
/// count).
struct KawaseChain {
    /// Downsampling targets: down[i] is the source at 1/2^i resolution.
    down: Vec<KawaseLevel>,
    /// Upsampling targets: up[i] matches down[i]'s size; up[i] blends
    /// down[i+1] (deepest) or up[i+1] back toward down[i]'s resolution.
    up: Vec<KawaseLevel>,
    /// down0/up0 size in pixels: the blurred region scaled by the plan's
    /// base scale.
    width: u32,
    height: u32,
    /// Configured level count (1..=5).
    levels: usize,
}

/// The engine-level Kawase machinery: shaders, sampler, chain and the two
/// constant buffers its passes read. Created once for the whole process;
/// every surface's composite then samples its output for free. The broker
/// additionally renders this same engine's output into the shared pair
/// (its publication pass below).
pub(crate) struct BlurEngine {
    device: ID3D11Device,
    vertex_shader: ID3D11VertexShader,
    crop_shader: ID3D11PixelShader,
    down_shader: ID3D11PixelShader,
    up_shader: ID3D11PixelShader,
    /// The identity resample used to write a (lower-resolution) backdrop
    /// into the publication target at the monitor's full resolution.
    stretch_shader: ID3D11PixelShader,
    sampler: ID3D11SamplerState,
    crop: ID3D11Buffer,
    crop_current: CropConstants,
    chain: Option<KawaseChain>,
    /// The frame size and plan the current chain was built for.
    chain_frame: [u32; 2],
    chain_plan: Option<BlurPlan>,
}

impl BlurEngine {
    pub(crate) fn new(device: &ID3D11Device) -> Self {
        let compile = |entry: &str| -> ID3D11PixelShader {
            let mut shader = None;
            // SAFETY: the bytecode comes from the cached compile below; a
            // failure here is a shader/toolchain defect surfaced at first
            // use, and the shared engine is created before any surface.
            unsafe {
                device
                    .CreatePixelShader(
                        &compiled_shader(KAWASE_HLSL, entry, "ps_5_0")
                            .expect("the Kawase shader should compile"),
                        None,
                        Some(&mut shader),
                    )
                    .expect("Could not create the Kawase pixel shader");
            }
            shader.expect("Kawase pixel shader missing")
        };
        let mut vertex_shader = None;
        unsafe {
            device
                .CreateVertexShader(
                    &compiled_shader(KAWASE_HLSL, "vs_main", "vs_5_0")
                        .expect("the Kawase vertex shader should compile"),
                    None,
                    Some(&mut vertex_shader),
                )
                .expect("Could not create the Kawase vertex shader");
        }
        let sampler = create_linear_clamp_sampler(device);
        let crop_current = CropConstants::identity([0.0, 0.0]);
        let crop =
            constant_buffer(device, &crop_current).expect("Could not create the blur crop buffer");
        Self {
            device: device.clone(),
            vertex_shader: vertex_shader.expect("Kawase vertex shader missing"),
            crop_shader: compile("ps_kawase_crop"),
            down_shader: compile("ps_kawase_down"),
            up_shader: compile("ps_kawase_up"),
            stretch_shader: compile("ps_stretch"),
            sampler,
            crop,
            crop_current,
            chain: None,
            chain_frame: [0, 0],
            chain_plan: None,
        }
    }

    /// Renders (or refreshes) the shared blurred backdrop for one capture
    /// frame. Cheap when nothing changed: the plan and the chain are
    /// fingerprinted and the passes only run on change. `crop_px` (x0, y0,
    /// x1, y1 in capture pixels, `None` = the full frame) restricts both the
    /// chain's textures and its sampling to that region — a crop is a
    /// region, not a scale, so tap offsets keep their capture-texel
    /// semantics — and the returned backdrop carries the matching
    /// `CropConstants`.
    pub(crate) fn render(
        &mut self,
        context: &ID3D11DeviceContext,
        capture: &ID3D11ShaderResourceView,
        frame: [u32; 2],
        crop_px: Option<[u32; 4]>,
        params: &GlassShaderParams,
        caps: &QualityCaps,
    ) -> Result<Backdrop, String> {
        let texel_uv = [1.0 / frame[0].max(1) as f32, 1.0 / frame[1].max(1) as f32];
        let plan = blur_plan_with_caps(params, caps);
        if plan.sharp {
            return Ok(Backdrop {
                view: capture.clone(),
                crop: CropConstants::identity(texel_uv),
            });
        }
        // The chain's base resolution scales the CROPPED region's pixel
        // extent; a crop only discards texels the composites cannot sample
        // (they were outside every surface's reach).
        let crop_rect = match crop_px {
            Some(px @ [x0, y0, x1, y1]) if x1 > x0 && y1 > y0 => Some(px),
            _ => None,
        };
        let region = crop_rect.map_or(frame, |[x0, y0, x1, y1]| [x1 - x0, y1 - y0]);
        let width = (region[0] as f32 * plan.base_scale).ceil().max(1.0) as u32;
        let height = (region[1] as f32 * plan.base_scale).ceil().max(1.0) as u32;
        let mut crop_current = crop_rect.map_or_else(
            || CropConstants::identity(texel_uv),
            |px| CropConstants::from_pixel_rect(px, frame, texel_uv),
        );
        crop_current.kawase_scale = plan.offset_scale;
        if crop_current != self.crop_current {
            unsafe {
                context.UpdateSubresource(
                    &self.crop,
                    0,
                    None,
                    (&crop_current as *const CropConstants).cast(),
                    0,
                    0,
                );
            }
            self.crop_current = crop_current;
        }
        if self.chain.is_none()
            || self.chain_frame != [width, height]
            || self.chain_plan.as_ref() != Some(&plan)
        {
            self.chain = Some(build_kawase_chain(
                &self.device,
                width,
                height,
                plan.levels,
            )?);
            self.chain_frame = [width, height];
            self.chain_plan = Some(plan);
        }
        let chain = self.chain.as_ref().expect("kawase chain missing");
        unsafe {
            context.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            context.VSSetShader(&self.vertex_shader, None);
            context.PSSetSamplers(0, Some(&[Some(self.sampler.clone())]));
            context.PSSetConstantBuffers(2, Some(&[Some(self.crop.clone())]));
            for (level, target) in chain.down.iter().enumerate() {
                let (width, height) = chain_target_size(chain, level);
                if level == 0 {
                    draw_kawase_pass(context, &self.crop_shader, target, capture, width, height);
                } else {
                    let source = &chain.down[level - 1].srv;
                    draw_kawase_pass(context, &self.down_shader, target, source, width, height);
                }
            }
            // Up side: expand from the deepest level back to down0's
            // resolution. down0/up0 run at the chain's full base
            // resolution, so the chain's output is up0 — the composites
            // sample it directly, and a single-level chain (no up passes)
            // hands them down0 instead.
            for level in (0..chain.up.len()).rev() {
                let source = if level + 1 == chain.up.len() {
                    &chain.down[chain.levels - 1].srv
                } else {
                    &chain.up[level + 1].srv
                };
                let (width, height) = chain_target_size(chain, level);
                draw_kawase_pass(
                    context,
                    &self.up_shader,
                    &chain.up[level],
                    source,
                    width,
                    height,
                );
            }
        }
        let view = match chain.up.first() {
            Some(shallowest) => &shallowest.srv,
            // A single level has no up passes; down0 is the output.
            None => &chain.down[0].srv,
        };
        Ok(Backdrop {
            view: view.clone(),
            crop: crop_current,
        })
    }

    /// The broker's publication pass: writes a (possibly blur-reduced)
    /// backdrop into the monitor-sized shared target through a plain
    /// bilinear resample. The shared fullscreen pipeline `vs_main` +
    /// `ps_stretch` — identical geometry to the chain's own passes, with
    /// no crop constants of interest to it.
    pub(crate) fn render_publication(
        &self,
        context: &ID3D11DeviceContext,
        backdrop: &ID3D11ShaderResourceView,
        target: &ID3D11RenderTargetView,
        width: u32,
        height: u32,
    ) -> Result<(), String> {
        if width == 0 || height == 0 {
            return Err("The publication target has no extent".to_owned());
        }
        unsafe {
            context.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            context.VSSetShader(&self.vertex_shader, None);
            context.PSSetShader(&self.stretch_shader, None);
            context.PSSetSamplers(0, Some(&[Some(self.sampler.clone())]));
            context.OMSetRenderTargets(Some(&[Some(target.clone())]), None);
            context.RSSetViewports(Some(&[D3D11_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: width as f32,
                Height: height as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]));
            context.PSSetShaderResources(0, Some(&[Some(backdrop.clone())]));
            context.Draw(3, 0);
            // Unbind the backdrop's view: the pipeline may render it again
            // later as a render target in the same command stream.
            context.PSSetShaderResources(0, Some(&[None]));
        }
        Ok(())
    }
}

/// The pixel size of Kawase chain level `level` (down and up buffers share
/// sizes): down0 runs at the chain's full base resolution and every deeper
/// level halves it.
fn chain_target_size(chain: &KawaseChain, level: usize) -> (u32, u32) {
    let shift = level.min(16) as u32;
    (
        chain.width.div_ceil(1u32 << shift),
        chain.height.div_ceil(1u32 << shift),
    )
}

/// One fullscreen Kawase pass: binds `target` as the render target at its
/// own viewport and `source` at t0, then draws the shared fullscreen
/// triangle.
fn draw_kawase_pass(
    context: &ID3D11DeviceContext,
    pixel_shader: &ID3D11PixelShader,
    target: &KawaseLevel,
    source: &ID3D11ShaderResourceView,
    width: u32,
    height: u32,
) {
    unsafe {
        context.OMSetRenderTargets(Some(&[Some(target.rtv.clone())]), None);
        context.RSSetViewports(Some(&[D3D11_VIEWPORT {
            TopLeftX: 0.0,
            TopLeftY: 0.0,
            Width: width as f32,
            Height: height as f32,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        }]));
        context.PSSetShader(pixel_shader, None);
        context.PSSetShaderResources(0, Some(&[Some(source.clone())]));
        context.Draw(3, 0);
    }
}

fn create_kawase_level(
    device: &ID3D11Device,
    width: u32,
    height: u32,
) -> Result<KawaseLevel, String> {
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
                BindFlags: (D3D11_BIND_SHADER_RESOURCE.0 | D3D11_BIND_RENDER_TARGET.0) as u32,
                CPUAccessFlags: 0,
                MiscFlags: 0,
            },
            None,
            Some(&mut texture),
        )
    }
    .map_err(|error| format!("Could not create the Kawase chain texture: {error}"))?;
    let texture = texture.expect("Kawase chain texture missing");
    let mut srv = None;
    unsafe {
        device
            .CreateShaderResourceView(&texture, None, Some(&mut srv))
            .map_err(|error| format!("Could not create the Kawase chain SRV: {error}"))?;
    }
    let mut rtv = None;
    unsafe {
        device
            .CreateRenderTargetView(&texture, None, Some(&mut rtv))
            .map_err(|error| format!("Could not create the Kawase chain RTV: {error}"))?;
    }
    Ok(KawaseLevel {
        texture,
        srv: srv.expect("Kawase chain SRV missing"),
        rtv: rtv.expect("Kawase chain RTV missing"),
    })
}

fn build_kawase_chain(
    device: &ID3D11Device,
    width: u32,
    height: u32,
    levels: usize,
) -> Result<KawaseChain, String> {
    // down[i] is the source at 1/2^i; up[i] re-expands back to down[i]'s
    // size, sourced from down[i+1] (the first up) or up[i+1]. down0/up0 run
    // at the full base resolution, so the chain's output is up0 (down0 for
    // a single level) and no separate finishing pass is needed.
    let mut down = Vec::with_capacity(levels);
    let mut up = Vec::with_capacity(levels.saturating_sub(1));
    for level in 0..levels {
        let shift = level.min(16) as u32;
        down.push(create_kawase_level(
            device,
            width.div_ceil(1u32 << shift),
            height.div_ceil(1u32 << shift),
        )?);
    }
    for level in 0..levels.saturating_sub(1) {
        let shift = level.min(16) as u32;
        up.push(create_kawase_level(
            device,
            width.div_ceil(1u32 << shift),
            height.div_ceil(1u32 << shift),
        )?);
    }
    Ok(KawaseChain {
        down,
        up,
        width,
        height,
        levels,
    })
}

pub(crate) fn constant_buffer<T>(device: &ID3D11Device, value: &T) -> Result<ID3D11Buffer, String> {
    let mut buffer = None;
    unsafe {
        device.CreateBuffer(
            &D3D11_BUFFER_DESC {
                ByteWidth: std::mem::size_of::<T>() as u32,
                Usage: D3D11_USAGE_DEFAULT,
                BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32,
                CPUAccessFlags: 0,
                MiscFlags: 0,
                StructureByteStride: 0,
            },
            Some(&D3D11_SUBRESOURCE_DATA {
                pSysMem: (value as *const T).cast(),
                SysMemPitch: 0,
                SysMemSlicePitch: 0,
            }),
            Some(&mut buffer),
        )
    }
    .map_err(|error| format!("Could not create a constant buffer: {error}"))?;
    Ok(buffer.expect("D3D11 constant buffer missing"))
}

pub(crate) fn create_linear_clamp_sampler(device: &ID3D11Device) -> ID3D11SamplerState {
    let mut sampler = None;
    unsafe {
        device
            .CreateSamplerState(
                &D3D11_SAMPLER_DESC {
                    Filter: D3D11_FILTER_MIN_MAG_MIP_LINEAR,
                    AddressU: D3D11_TEXTURE_ADDRESS_CLAMP,
                    AddressV: D3D11_TEXTURE_ADDRESS_CLAMP,
                    AddressW: D3D11_TEXTURE_ADDRESS_CLAMP,
                    MipLODBias: 0.0,
                    MaxAnisotropy: 1,
                    ComparisonFunc: D3D11_COMPARISON_ALWAYS,
                    BorderColor: [0.0; 4],
                    MinLOD: 0.0,
                    MaxLOD: f32::MAX,
                },
                Some(&mut sampler),
            )
            .expect("Could not create the D3D11 sampler");
    }
    sampler.expect("D3D11 sampler missing")
}

/// Compiles a shader program once per (source, entry) pair for the whole
/// process: spawning a surface must never re-run the HLSL compiler.
pub(crate) fn compiled_shader(
    source: &'static str,
    entry: &str,
    target: &str,
) -> Result<Vec<u8>, String> {
    static CACHE: OnceLock<Mutex<HashMap<(&'static str, String), Vec<u8>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().expect("shader cache lock poisoned");
    if let Some(blob) = cache.get(&(source, entry.to_owned())) {
        return Ok(blob.clone());
    }
    let blob = compile_shader_from(source, entry, target)?;
    cache.insert((source, entry.to_owned()), blob.clone());
    Ok(blob)
}

/// Compiles HLSL source to shader bytecode via D3DCompile.
pub(crate) fn compile_shader_from(
    source: &str,
    entry: &str,
    target: &str,
) -> Result<Vec<u8>, String> {
    let entry = CString::new(entry).expect("shader entry contains no NUL");
    let target = CString::new(target).expect("shader target contains no NUL");
    let mut blob: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            PCSTR::null(),
            None,
            None::<&windows::Win32::Graphics::Direct3D::ID3DInclude>,
            PCSTR(entry.as_ptr() as _),
            PCSTR(target.as_ptr() as _),
            D3DCOMPILE_ENABLE_STRICTNESS,
            0,
            &mut blob,
            Some(&mut errors),
        )
        .map_err(|error| {
            let details = errors
                .as_ref()
                .map(|blob| {
                    let bytes = std::slice::from_raw_parts(
                        blob.GetBufferPointer() as *const u8,
                        blob.GetBufferSize(),
                    );
                    String::from_utf8_lossy(bytes).into_owned()
                })
                .unwrap_or_default();
            format!("Could not compile the D3D11 liquid shader: {error}\n{details}")
        })?;
        let blob = blob.expect("D3D compiler returned no bytecode");
        Ok(
            std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize())
                .to_vec(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{blur_plan_with_caps, grid_aligned_crop, CropConstants};
    use crate::params::GlassShaderParams;
    use crate::quality::QualityCaps;

    const HIGH: QualityCaps = QualityCaps::HIGH;

    #[test]
    fn blur_plan_scales_resolution_and_levels_with_radius() {
        let mut params = GlassShaderParams::default();
        // Blur off: sharp capture, no chain.
        params.blur_radius_texels = 0.0;
        assert!(blur_plan_with_caps(&params, &HIGH).sharp);
        // Default-ish blur (8): a shallow chain at high resolution.
        params.blur_radius_texels = 8.0;
        let plan = blur_plan_with_caps(&params, &HIGH);
        assert!(!plan.sharp);
        assert_eq!(plan.levels, 2);
        assert!(plan.base_scale > 0.9);
        // Level bands at 12 / 28 / 64 (one buffer per hop, plus the crop).
        params.blur_radius_texels = 16.0;
        assert_eq!(blur_plan_with_caps(&params, &HIGH).levels, 3);
        params.blur_radius_texels = 32.0;
        assert_eq!(blur_plan_with_caps(&params, &HIGH).levels, 4);
        // Subtle blur: still a shallow chain at ~full resolution.
        params.blur_radius_texels = 2.0;
        let plan = blur_plan_with_caps(&params, &HIGH);
        assert_eq!(plan.levels, 2);
        assert!(plan.base_scale > 0.98);
        // Samples setting only ever caps levels, never raises them.
        params.blur_samples = 1.0;
        params.blur_radius_texels = 32.0;
        assert_eq!(blur_plan_with_caps(&params, &HIGH).levels, 1);
    }

    #[test]
    fn blur_plan_is_continuous_across_level_bands() {
        // In the sigma ≈ offsets · 2^levels · base model, each band must END
        // where the next one STARTS — otherwise dragging the radius slider
        // snaps harshly when a level is added.
        let mut params = GlassShaderParams::default();
        // The full 5-level headroom: the 64-boundary region needs the fifth
        // level, and a smaller samples cap would truncate it.
        params.blur_samples = 5.0;
        let reach = |radius: f32| {
            let plan = blur_plan_with_caps(
                &{
                    let mut p = params;
                    p.blur_radius_texels = radius;
                    p
                },
                &HIGH,
            );
            plan.offset_scale * (1usize << plan.levels) as f32 * plan.base_scale
        };
        for boundary in [12.0_f32, 28.0, 64.0] {
            let before = reach(boundary - 0.01);
            let after = reach(boundary + 0.01);
            let jump = (after - before).abs() / before.abs().max(1e-4);
            assert!(
                jump < 0.1,
                "reach jumped {before} -> {after} at the level boundary {boundary}"
            );
        }
    }

    #[test]
    fn efficiency_caps_cut_internal_resolution() {
        let mut params = GlassShaderParams::default();
        params.blur_samples = 5.0;
        params.blur_radius_texels = 32.0;
        let plan = blur_plan_with_caps(&params, &QualityCaps::EFFICIENCY);
        // The internal scale ramp is fully engaged at radius 24+.
        assert!(plan.base_scale <= QualityCaps::EFFICIENCY.blur_internal_scale);
        // Subtle blur (below the ramp) keeps full resolution even on the
        // efficient tier: resampling would be visible there. (Radius 8
        // sits in the first band whose own ramp reaches base 0.936.)
        params.blur_radius_texels = 8.0;
        let plan = blur_plan_with_caps(&params, &QualityCaps::EFFICIENCY);
        assert!(plan.base_scale > 0.9);
        // The level count is untouched by the tier (only the user's
        // samples setting caps it).
        params.blur_radius_texels = 32.0;
        assert_eq!(
            blur_plan_with_caps(&params, &QualityCaps::EFFICIENCY).levels,
            4
        );
    }

    #[test]
    fn blur_plan_stays_continuous_under_tier_caps() {
        use crate::quality::QualityCaps;
        let mut params = GlassShaderParams::default();
        params.blur_samples = 5.0;
        let reach = |radius: f32| {
            let plan = blur_plan_with_caps(
                &{
                    let mut p = params;
                    p.blur_radius_texels = radius;
                    p
                },
                &QualityCaps::EFFICIENCY,
            );
            plan.offset_scale * (1usize << plan.levels) as f32 * plan.base_scale
        };
        for boundary in [12.0_f32, 24.0, 28.0, 64.0] {
            let before = reach(boundary - 0.01);
            let after = reach(boundary + 0.01);
            let jump = (after - before).abs() / before.abs().max(1e-4);
            assert!(
                jump < 0.1,
                "efficiency reach jumped {before} -> {after} at {boundary}"
            );
        }
    }

    #[test]
    fn grid_aligned_crop_rounds_outward_and_clamps() {
        // A rect strictly inside the frame grows to the 64 px grid in both
        // directions: the rounded rect must CONTAIN the input.
        let frame = [1920, 1080];
        let rect = grid_aligned_crop([100.3, 200.6, 500.2, 700.9], frame, 64.0).expect("rect");
        assert!(rect[0] <= 100 && rect[1] <= 200);
        assert!(rect[2] >= 501 && rect[3] >= 701);
        assert_eq!(rect[0] % 64, 0);
        assert_eq!(rect[1] % 64, 0);
        assert_eq!(rect[2] % 64, 0);
        assert_eq!(rect[3] % 64, 0);
        // Inputs inside the same grid cells snap to the same rect: small
        // surface moves must not churn chain rebuilds.
        let nudged = grid_aligned_crop([103.0, 205.0, 505.0, 704.0], frame, 64.0).expect("rect");
        assert_eq!(rect, nudged);
        // Rects reaching past the frame clamp into it.
        let clamped = grid_aligned_crop([-50.0, -50.0, 5000.0, 5000.0], frame, 64.0)
            .expect("rect");
        assert_eq!(clamped, [0, 0, 1920, 1080]);
        // Degenerate rects (past the frame edge, empty after clamping)
        // fall back to no crop.
        assert_eq!(
            grid_aligned_crop([2000.0, 500.0, 2100.0, 600.0], frame, 64.0),
            None
        );
    }

    #[test]
    fn crop_constants_map_capture_uvs_into_the_backdrop() {
        let frame = [1920, 1080];
        let texel = [1.0 / 1920.0, 1.0 / 1080.0];
        // The identity crop maps every capture uv to itself.
        let identity = CropConstants::identity(texel);
        assert_eq!(
            backdrop_uv_of(&identity, [0.25, 0.75]),
            [0.25, 0.75]
        );
        // A sub-rect crop maps the region's corners to (0, 0) and (1, 1)
        // and its center to (0.5, 0.5) — exactly what the composite's
        // `backdropUv` does in the shader.
        let rect = [640, 360, 1280, 720];
        let crop = CropConstants::from_pixel_rect(rect, frame, texel);
        let map = |uv: [f32; 2]| backdrop_uv_of(&crop, uv);
        assert_eq!(map([rect[0] as f32 / 1920.0, rect[1] as f32 / 1080.0]), [0.0, 0.0]);
        assert_eq!(map([rect[2] as f32 / 1920.0, rect[3] as f32 / 1080.0]), [1.0, 1.0]);
        let center = map([960.0 / 1920.0, 540.0 / 1080.0]);
        assert!((center[0] - 0.5).abs() < 1e-6 && (center[1] - 0.5).abs() < 1e-6);
    }

    /// The composite's capture→backdrop mapping (`backdropUv` in
    /// liquid_composition.hlsl), mirrored here so the Rust-provided crop
    /// constants are checked against the shader's exact math.
    fn backdrop_uv_of(crop: &CropConstants, capture_uv: [f32; 2]) -> [f32; 2] {
        [
            (capture_uv[0] - crop.crop_origin_uv[0]) / crop.crop_size_uv[0],
            (capture_uv[1] - crop.crop_origin_uv[1]) / crop.crop_size_uv[1],
        ]
    }
}
