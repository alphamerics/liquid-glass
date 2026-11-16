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

mod backdrop;
mod blur;
mod d3d;
pub(crate) mod reaper;
mod render_loop;
#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) use backdrop::grid_aligned_crop;
pub(crate) use backdrop::{Backdrop, BlurInputs, CropConstants};
pub(crate) use blur::{blur_plan_with_caps, BlurEngine};
#[cfg(test)]
pub(crate) use blur::{BlurPlan, KAWASE_HLSL};
#[cfg(test)]
pub(crate) use d3d::compile_shader_from;
pub(crate) use d3d::{compiled_shader, constant_buffer, create_linear_clamp_sampler};
use render_loop::spawn_render_loop;

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime};

use windows::Graphics::SizeInt32;
use windows::Win32::Graphics::Direct3D11::{
    ID3D11Device, ID3D11DeviceContext, ID3D11ShaderResourceView, ID3D11Texture2D,
};
use windows::Win32::Graphics::Dxgi::{
    DXGI_ERROR_DEVICE_HUNG, DXGI_ERROR_DEVICE_REMOVED, DXGI_ERROR_DEVICE_RESET,
};

use crate::backdrop::{local_backdrop, LocalFeedState, SurfaceRoute};
use crate::capture::{CaptureFeed, FrameHost};
use crate::params::BackdropSource;
use crate::quality::{detect_gpu_class, resolve, GpuClass, QualityCaps};
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

    /// Composites and presents one registered surface's current field right
    /// now, on the caller's thread — the in-step present behind
    /// [`GlassSurface::present_now`]. The lock order mirrors the render
    /// loop's (engine state, then the surface's compositor; the compositor
    /// lock nests INSIDE `state`, whose references feed it). Backdrop
    /// routing mirrors the loop's dispatch: the registered route picks the
    /// feed, and a not-yet-warm backdrop (Global feed still spinning up, or
    /// a Local window whose first frame has not landed) skips the present —
    /// the shared loop composites when it exists.
    pub(crate) fn present_surface_now(
        &self,
        compositor: &SharedCompositor,
        engine_id: u64,
    ) -> Result<(), String> {
        let state = self.state.lock().expect("shared glass lock poisoned");
        let Some((_, _, route)) = state
            .surfaces
            .iter()
            .find(|(registered, _, _)| *registered == engine_id)
        else {
            return Ok(());
        };
        let mut compositor = compositor.lock().expect("D3D11 compositor lock poisoned");
        if compositor.is_paused() {
            return Ok(());
        }
        let backdrop = match route.source {
            BackdropSource::Global => state.backdrop.as_ref(),
            BackdropSource::Local => route
                .window
                .and_then(|window| local_backdrop(&state, window)),
        };
        let Some(backdrop) = backdrop else {
            return Ok(());
        };
        compositor.present_forced(&state.context, backdrop)
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
