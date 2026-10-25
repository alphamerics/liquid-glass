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
use windows::Win32::Graphics::{
    Direct3D::{
        Fxc::{D3DCompile, D3DCOMPILE_ENABLE_STRICTNESS},
        ID3DBlob, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
    },
    Direct3D11::{
        ID3D11Buffer, ID3D11Device, ID3D11DeviceContext, ID3D11PixelShader,
        ID3D11RenderTargetView, ID3D11SamplerState, ID3D11ShaderResourceView, ID3D11Texture2D,
        ID3D11VertexShader, D3D11_BIND_CONSTANT_BUFFER, D3D11_BIND_RENDER_TARGET,
        D3D11_BIND_SHADER_RESOURCE, D3D11_BUFFER_DESC, D3D11_COMPARISON_ALWAYS,
        D3D11_FILTER_MIN_MAG_MIP_LINEAR, D3D11_SAMPLER_DESC, D3D11_SUBRESOURCE_DATA,
        D3D11_TEXTURE_ADDRESS_CLAMP, D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT, D3D11_VIEWPORT,
    },
};
use windows::Graphics::SizeInt32;

use crate::broker::client::BrokerClient;
use crate::capture::{CaptureFeed, FrameHost};
use crate::params::GlassShaderParams;
use crate::surface::{create_d3d11_device, SharedCompositor};

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
        for (_, compositor) in &state.surfaces {
            let Ok(mut compositor) = compositor.lock() else {
                continue;
            };
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
/// off) the sharp capture itself.
#[derive(Clone)]
pub(crate) struct Backdrop {
    pub(crate) view: ID3D11ShaderResourceView,
}

pub(crate) struct EngineState {
    pub device: ID3D11Device,
    pub context: ID3D11DeviceContext,
    /// The newest captured frame, copied out of the WGC frame pool (whose
    /// surfaces are recycled by the system) into a stable texture, plus the
    /// shader view created alongside it.
    pub frame_texture: Option<ID3D11Texture2D>,
    pub capture_view: Option<ID3D11ShaderResourceView>,
    /// Current frame (and pool) size in pixels.
    pub frame_size: SizeInt32,
    /// Registered surfaces: (id, compositor).
    pub surfaces: Vec<(u64, SharedCompositor)>,
    pub next_surface_id: u64,
    pub render_running: Arc<AtomicBool>,
    /// The engine-level blur: one dual-Kawase chain over the whole capture,
    /// rendered once per fresh frame and sampled by every surface's
    /// composite. `None` until the first blurred frame is needed.
    blur: Option<BlurEngine>,
    /// The last rendered backdrop, the capture version it was built from
    /// and the material params it was built for; reused by every
    /// animation-driven frame in between.
    backdrop: Option<Backdrop>,
    backdrop_version: u64,
    backdrop_inputs: Option<BlurInputs>,
}

impl EngineState {
    /// Replaces the shared capture frame and its view (called by the
    /// capture feed when a frame lands or the monitor's size changes).
    pub(crate) fn set_frame(
        &mut self,
        frame: (ID3D11Texture2D, ID3D11ShaderResourceView),
    ) {
        (self.frame_texture, self.capture_view) = (Some(frame.0), Some(frame.1));
    }
}

impl FrameHost for EngineState {
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
        self.surfaces.iter().any(|(_, compositor)| {
            compositor.lock().map(|compositor| !compositor.is_paused()).unwrap_or(true)
        })
    }
}

pub(crate) struct SharedEngine {
    pub state: Arc<Mutex<EngineState>>,
    /// The single capture session, owned by the render loop (started with
    /// the first awake surface, stopped when the last one unregisters — or
    /// when every surface parks, so the OS capture pipeline idles with the
    /// glass). Registration and teardown reach it through this handle.
    feed: Arc<Mutex<Option<CaptureFeed>>>,
}

static ENGINE: OnceLock<SharedEngine> = OnceLock::new();

pub(crate) fn engine() -> &'static SharedEngine {
    ENGINE.get_or_init(|| {
        let (device, context) =
            create_d3d11_device().expect("Glassbar shared glass device failed");
        let state = Arc::new(Mutex::new(EngineState {
            device,
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
        }));
        let version = Arc::new(AtomicU64::new(0));
        let feed = Arc::new(Mutex::new(None));
        spawn_render_loop(
            Arc::clone(&state),
            Arc::clone(&version),
            Arc::clone(&feed),
        );
        SharedEngine {
            state,
            feed,
        }
    })
}

impl SharedEngine {
    /// Registers a surface's compositor. The capture itself is the render
    /// loop's call: its very next tick sees the awake surface and starts
    /// the feed (or attaches to the broker and skips a local feed
    /// entirely) — one owner for the whole lifecycle.
    pub(crate) fn register(&self, compositor: SharedCompositor) -> u64 {
        let mut state = self.state.lock().expect("shared glass lock poisoned");
        let id = state.next_surface_id;
        state.next_surface_id += 1;
        state.surfaces.push((id, Arc::clone(&compositor)));
        // The fresh surface adopts the shared material: the merge starts
        // from what the caller configured (surface-specific fields like the
        // capsule radius survive), so a new glass window is styled
        // consistently with every other glass app in the session. Pinned
        // surfaces (the app re-derives params itself) are skipped: the app
        // seeds them right after creation.
        let material = crate::broker::read_material();
        if let Ok(mut compositor) = compositor.lock() {
            if !compositor.material_locked() {
                let mut merged = compositor.active_params();
                material.apply_onto(&mut merged, CAPTURE_SCALE);
                let _ = compositor.set_params(&state.context, merged);
            }
        }
        id
    }

    /// Unregisters a surface; stops the shared capture with the last one.
    pub(crate) fn unregister(&self, id: u64) {
        let empty = {
            let mut state = self.state.lock().expect("shared glass lock poisoned");
            state.surfaces.retain(|(existing, _)| *existing != id);
            state.surfaces.is_empty()
        };
        if empty {
            // No glass left anywhere: stop the capture entirely.
            *self.feed.lock().expect("shared glass feed lock poisoned") = None;
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
struct BlurInputs {
    params: GlassShaderParams,
}

impl BlurInputs {
    /// Collects from the registered surfaces. Every surface derives its
    /// material from the same saved settings, so the blur-relevant params
    /// (radius, samples) are identical across surfaces; the first active
    /// surface's copy is authoritative.
    fn collect(state: &EngineState) -> Option<Self> {
        let mut params = None;
        for (_, compositor) in &state.surfaces {
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

/// Renders the shared backdrop once per tick when the capture has changed.
/// With a static desktop (and during animations, which re-render only the
/// composite) the previous backdrop is reused untouched.
fn refresh_backdrop(state: &mut EngineState, version: u64, force: bool) {
    let Some(capture) = state.capture_view.clone() else {
        return;
    };
    let Some(inputs) = BlurInputs::collect(state) else {
        // Nothing is awake to consume a backdrop; keep the last one so a
        // surface resuming over a static desktop has material immediately.
        return;
    };
    // Idle tick fast path: the backdrop already matches the newest capture
    // frame and the current material: the passes only run when one of those
    // moved (a settings change re-blurs even on a static desktop).
    if !force
        && state.backdrop_version == version
        && state
            .backdrop_inputs
            .as_ref()
            .is_some_and(|last| *last == inputs)
    {
        return;
    }
    let frame = [state.frame_size.Width.max(1) as u32, state.frame_size.Height.max(1) as u32];
    let blur = state.blur.get_or_insert_with(|| BlurEngine::new(&state.device));
    match blur.render(
        &state.context,
        &capture,
        frame,
        &inputs.params,
    ) {
        Ok(backdrop) => {
            state.backdrop = Some(backdrop);
            state.backdrop_version = version;
            state.backdrop_inputs = Some(inputs);
        }
        Err(error) => eprintln!("Glassbar shared blur failed: {error}"),
    }
}

/// Hysteresis for the feed lifecycle (about two seconds at the loop's ~60Hz
/// tick): brief all-parked moments (a settings preview, a quick hide/show)
/// must not churn the capture session.
const IDLE_FEED_TICKS: u32 = 120;

fn spawn_render_loop(
    state: Arc<Mutex<EngineState>>,
    version: Arc<AtomicU64>,
    feed: Arc<Mutex<Option<CaptureFeed>>>,
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
        let mut restart_backoff = 0u32;
        // The broker client machine (see `broker::client`): every tick
        // advances attach/pull; a silent failure falls back to the local
        // pipeline by returning an `Err` once and resetting to `Idle`.
        let mut broker = BrokerClient::new();
        // The shared material watch (see [`MaterialWatch`]): file changes
        // restyle every registered compositor within one poll cadence.
        let mut material = MaterialWatch::default();
        while state
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

            // The awake-surface answer drives both pipelines: the broker's
            // slot flag and the local feed's lifecycle. A compositor whose
            // lock is poisoned counts as awake — a stuck lock must never
            // park the glass.
            let any_active = engine.surfaces.iter().any(|(_, compositor)| {
                compositor.lock().map(|c| !c.is_paused()).unwrap_or(true)
            });

            // Broker attempt first: when it hands over a fresh backdrop the
            // local pipeline is skipped entirely this tick.
            let mut frame_changed = false;
            let mut broker_attached = false;
            match broker.tick(&engine.device, &engine.context, any_active) {
                Ok(Some(frame)) => {
                    engine.backdrop = Some(frame.backdrop);
                    frame_changed = true;
                    broker_attached = true;
                }
                Ok(None) => broker_attached = broker.attached(),
                Err(error) => {
                    eprintln!(
                        "glass broker unavailable, running the in-process engine instead: {error}"
                    );
                    // Forget whatever the broker published: the local
                    // pipeline must build the next backdrop fresh.
                    engine.backdrop = None;
                }
            }

            let current = version.load(Ordering::Relaxed);
            let version_changed = current != last_version;
            last_version = current;
            // One blur per fresh capture frame — never per surface, never
            // per animation tick. The first tick after startup forces one
            // render even before a capture frame lands, so freshly spawned
            // surfaces composite from the very first frame. In broker mode
            // the local chain never runs: the published backdrop replaces
            // its output wholesale.
            if !broker_attached {
                let first_backdrop = engine.backdrop.is_none();
                refresh_backdrop(&mut engine, current, first_backdrop);
            }
            // What tells the compositors a FRESH backdrop arrived: the
            // capture's version in local mode, the broker's pull in
            // attached mode. Animations and dirty params still request
            // frames through `wants_frame` regardless.
            let present_signal = if broker_attached { frame_changed } else { version_changed };
            for (_, compositor) in &engine.surfaces {
                let Ok(mut compositor) = compositor.lock() else {
                    continue;
                };
                if compositor.wants_frame(present_signal) {
                    if let Some(backdrop) = engine.backdrop.as_ref() {
                        if let Err(error) = compositor.present(&engine.context, backdrop) {
                            // A failed present means the surface silently never
                            // appears (hosts gate reveals on `has_presented`);
                            // surfacing the error here is the only way to see it.
                            eprintln!("Glassbar surface render failed: {error}");
                        }
                    }
                }
            }
            drop(engine);
            // Feed lifecycle, strictly outside the state lock: starting the
            // feed re-locks it, and dropping a feed while its own frame
            // callback blocks on that lock could deadlock the teardown.
            if broker_attached {
                // The broker publishes the backdrop: the local capture and
                // blur pipeline is pure duplicate cost — tear it down. The
                // broker's own feed is parked through the slot's active flag
                // (the client wrote it) whenever every local surface sleeps.
                idle_ticks = 0;
                *feed.lock().expect("shared glass feed lock poisoned") = None;
            } else if any_active {
                idle_ticks = 0;
                if restart_backoff > 0 {
                    restart_backoff -= 1;
                } else if !feed
                    .lock()
                    .expect("shared glass feed lock poisoned")
                    .is_some()
                {
                    match CaptureFeed::start(&host, &version) {
                        Ok(new_feed) => {
                            *feed.lock().expect("shared glass feed lock poisoned") = Some(new_feed)
                        }
                        Err(error) => {
                            eprintln!("Glassbar shared capture failed to start: {error}");
                            restart_backoff = IDLE_FEED_TICKS;
                        }
                    }
                }
            } else {
                idle_ticks += 1;
                if idle_ticks >= IDLE_FEED_TICKS {
                    idle_ticks = 0;
                    // Every surface is parked: the capture session is torn
                    // down so the OS capture pipeline idles with it. The
                    // last backdrop stays cached; the first surface to wake
                    // restarts the feed above and composites immediately
                    // from the cached backdrop while the first frame flies.
                    *feed.lock().expect("shared glass feed lock poisoned") = None;
                }
            }
        }
    });
}

/// How the Kawase chain should run for the current params.
#[derive(Clone, Copy, PartialEq)]
pub(crate) struct BlurPlan {
    /// Blur is effectively off: composites sample the sharp capture
    /// directly, skipping the chain entirely.
    pub(crate) sharp: bool,
    /// Kawase levels actually used (1..=4). Capped by the radius: the
    /// chain's own down/upsampling imposes a minimum blur per level, so a
    /// small radius must use fewer levels no matter the samples setting.
    pub(crate) levels: usize,
    /// Resolution of the chain's base (down0) level relative to the capture
    /// (0.5..1.0). Less blur hides less resampling, so lower radii run the
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
/// base resolution eases down to 0.75, and each deeper band starts at
/// offset 0.75 / base 1.0 — which, in the `sigma ≈ offsets · 2^levels ·
/// base` model, lands exactly on the previous band's end. Adding a level
/// therefore never snaps: the new level begins at nearly full resolution
/// with tiny taps (a gentle addition) and grows gradually from there.
pub(crate) fn blur_plan(params: &GlassShaderParams) -> BlurPlan {
    let radius = params.blur_radius_texels;
    if radius < 0.5 {
        return BlurPlan {
            sharp: true,
            levels: 1,
            base_scale: 1.0,
            offset_scale: 0.0,
        };
    }
    let samples = (params.blur_samples.round() as usize).clamp(1, 4);
    // One level covers radii up to ~12px; each extra level roughly doubles
    // the reach (bands 12-28, 28-64, 64+).
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
    // 0.75 to match the previous band's end.
    let offsets_start = if band_levels == 1 { 0.25 } else { 0.75 };
    let offset_scale = (offsets_start + (2.0 - offsets_start) * position).clamp(0.25, 2.0);
    let base_scale = 1.0 - 0.25 * position;
    let levels = samples.min(band_levels);
    BlurPlan {
        sharp: false,
        levels,
        base_scale,
        offset_scale,
    }
}

/// Mirrors the shader's `CropParams` constant buffer (register b2): the
/// capture-uv geometry of the region the Kawase chain blurs, plus the
/// capture texel size. The shared chain blurs the WHOLE capture, so the
/// composite's backdrop uv equals the capture uv (identity crop); only the
/// capture texel size carries information.
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
    /// The identity crop the composites bind for the shared full-capture
    /// backdrop.
    pub(crate) fn identity(texel_uv: [f32; 2]) -> Self {
        Self {
            crop_origin_uv: [0.0, 0.0],
            crop_size_uv: [1.0, 1.0],
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

/// The intermediate buffers of the dual-Kawase blur over the whole capture,
/// at successively halved resolutions. Recreated whenever the frame size or
/// the level count changes (monitor change, or params that move the level
/// count).
struct KawaseChain {
    /// Downsampling targets: down[i] is the capture at 1/2^(i+1) resolution.
    down: Vec<KawaseLevel>,
    /// Upsampling targets: up[i] matches down[i]'s size; up[i] blends
    /// down[i+1] (deepest) or up[i+1] back toward down[i]'s resolution.
    up: Vec<KawaseLevel>,
    /// down0/up0 size in pixels: the capture at half resolution scaled by
    /// the plan's base scale.
    width: u32,
    height: u32,
    /// Configured level count (1..=4).
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
        let crop = constant_buffer(device, &crop_current)
            .expect("Could not create the blur crop buffer");
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
    /// fingerprinted and the passes only run on change.
    pub(crate) fn render(
        &mut self,
        context: &ID3D11DeviceContext,
        capture: &ID3D11ShaderResourceView,
        frame: [u32; 2],
        params: &GlassShaderParams,
    ) -> Result<Backdrop, String> {
        let plan = blur_plan(params);
        if plan.sharp {
            return Ok(Backdrop {
                view: capture.clone(),
            });
        }
        let width = (frame[0] as f32 * plan.base_scale).ceil().max(1.0) as u32;
        let height = (frame[1] as f32 * plan.base_scale).ceil().max(1.0) as u32;
        let crop_current = CropConstants {
            crop_origin_uv: [0.0, 0.0],
            crop_size_uv: [1.0, 1.0],
            capture_texel_uv: [1.0 / frame[0] as f32, 1.0 / frame[1] as f32],
            kawase_scale: plan.offset_scale,
            padding: [0.0; 5],
        };
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
            self.chain = Some(build_kawase_chain(&self.device, width, height, plan.levels)?);
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
                    draw_kawase_pass(
                        context,
                        &self.crop_shader,
                        target,
                        capture,
                        width,
                        height,
                    );
                } else {
                    let source = &chain.down[level - 1].srv;
                    draw_kawase_pass(
                        context,
                        &self.down_shader,
                        target,
                        source,
                        width,
                        height,
                    );
                }
            }
            // Up side: expand from the deepest level back to down0's
            // resolution; the composites' bilinear sampler finishes the
            // final 2x upsample.
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
        let backdrop = match chain.up.first() {
            Some(shallowest) => shallowest.srv.clone(),
            // A single level has no up passes; composites sample down0.
            None => chain.down[0].srv.clone(),
        };
        Ok(Backdrop {
            view: backdrop,
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
/// sizes): the frame at 1/2^(level+1) resolution.
fn chain_target_size(chain: &KawaseChain, level: usize) -> (u32, u32) {
    let shift = (level + 1).min(16) as u32;
    (
        chain.width.div_ceil(1 << shift),
        chain.height.div_ceil(1 << shift),
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
    // down[i] is the capture at 1/2^(i+1); up[i] re-expands back to down[i]'s
    // size, sourced from down[i+1] (the first up) or up[i+1].
    let mut down = Vec::with_capacity(levels);
    let mut up = Vec::with_capacity(levels.saturating_sub(1));
    for level in 0..levels {
        let shift = (level + 1).min(16) as u32;
        down.push(create_kawase_level(
            device,
            width.div_ceil(1 << shift),
            height.div_ceil(1 << shift),
        )?);
    }
    for level in 0..levels.saturating_sub(1) {
        let shift = (level + 1).min(16) as u32;
        up.push(create_kawase_level(
            device,
            width.div_ceil(1 << shift),
            height.div_ceil(1 << shift),
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
    use super::blur_plan;
    use crate::params::GlassShaderParams;

    #[test]
    fn blur_plan_scales_resolution_and_levels_with_radius() {
        let mut params = GlassShaderParams::default();
        // Blur off: sharp capture, no chain.
        params.blur_radius_texels = 0.0;
        assert!(blur_plan(&params).sharp);
        // Default-ish blur (8): a single level at high resolution.
        params.blur_radius_texels = 8.0;
        let plan = blur_plan(&params);
        assert!(!plan.sharp);
        assert_eq!(plan.levels, 1);
        assert!(plan.base_scale > 0.8);
        // Level bands at 12 / 28 / 64.
        params.blur_radius_texels = 16.0;
        assert_eq!(blur_plan(&params).levels, 2);
        params.blur_radius_texels = 32.0;
        assert_eq!(blur_plan(&params).levels, 3);
        // Subtle blur: still a single level at ~full resolution.
        params.blur_radius_texels = 2.0;
        let plan = blur_plan(&params);
        assert_eq!(plan.levels, 1);
        assert!(plan.base_scale > 0.95);
        // Samples setting only ever caps levels, never raises them.
        params.blur_samples = 1.0;
        params.blur_radius_texels = 32.0;
        assert_eq!(blur_plan(&params).levels, 1);
    }

    #[test]
    fn blur_plan_is_continuous_across_level_bands() {
        // In the sigma ≈ offsets · 2^levels · base model, each band must END
        // where the next one STARTS — otherwise dragging the radius slider
        // snaps harshly when a level is added.
        let mut params = GlassShaderParams::default();
        params.blur_samples = 4.0;
        let reach = |radius: f32| {
            let plan = blur_plan(&{
                let mut p = params;
                p.blur_radius_texels = radius;
                p
            });
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
}
