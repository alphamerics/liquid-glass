//! The two backdrop feeds behind the one drawer.
//!
//! Every glass surface is painted by the same compositor and the same
//! shader (`ps_main`): one drawer, many surfaces. What differs is only the
//! data handed to that drawer, and this module owns the decision:
//!
//! - **Global** feed: the whole primary monitor, captured once by
//!   [`crate::capture::CaptureFeed`], blurred once by the shared Kawase
//!   chain (see `shared.rs`), sampled by every surface. This is what makes
//!   glass `Global`: it refracts the entire screen, at the price of one
//!   screen-duplication capture.
//!
//! - **Local** feed: one app window, captured by
//!   [`crate::window_capture::WindowCaptureFeed`] and blurred by its own
//!   small Kawase chain, sampled only by the glass elements that live over
//!   that window. Glass that sits on top of pixels its app already owns —
//!   a container, button or slider inside the app — has no need of the
//!   screen: the local feed can source it entirely from the app window,
//!   so an app with only Local surfaces runs with **no monitor capture
//!   and no broker involvement**.
//!
//! The engine state holds one persisted entry per captured window
//! ([`LocalFeedState`]); the render loop in `shared.rs` refreshes and
//! routes. Surfaces declare which entry they consume at creation through
//! [`GlassConfig::backdrop_source`].

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};
use std::time::Instant;

use windows::Graphics::SizeInt32;
use windows::Win32::Graphics::Direct3D11::{
    ID3D11Device, ID3D11DeviceContext, ID3D11ShaderResourceView, ID3D11Texture2D,
};

use crate::shared::{Backdrop, BlurEngine, BlurInputs};
use crate::window_capture::{WindowCaptureFeed, WindowFrameHost};

/// The engine's force-refresh cadence, shared with the global pipeline
/// (same insurance: a driver that misreports dirty regions can still never
/// starve a local backdrop for more than this).
const FORCE_BACKDROP_SECS: f32 = 2.0;

/// Per-surface routing: which feed hands the drawer its data, and for
/// Local surfaces which window's feed that is (the surface's host window).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SurfaceRoute {
    pub(crate) source: crate::params::BackdropSource,
    /// The host window for `BackdropSource::Local` surfaces; `None` for
    /// Global (which routes through the process-wide monitor feed).
    pub(crate) window: Option<isize>,
}

impl SurfaceRoute {
    pub(crate) fn global() -> Self {
        Self {
            source: crate::params::BackdropSource::Global,
            window: None,
        }
    }

    pub(crate) fn local(window: isize) -> Self {
        Self {
            source: crate::params::BackdropSource::Local,
            window: Some(window),
        }
    }

    /// Whether this route consumes the feed of `hwnd` (Local surfaces) or
    /// the process-wide monitor feed (Global surfaces).
    pub(crate) fn consumes_window(&self, hwnd: isize) -> bool {
        matches!(self.source, crate::params::BackdropSource::Local) && self.window == Some(hwnd)
    }
}

/// Everything the local feed of one window persists across ticks: the
/// stable capture texture, the window's own Kawase chain and its last
/// backdrop. Created when the first Local surface for the window
/// registers; the last backdrop stays cached so a surface resuming over a
/// paused video composites immediately from the previous frame.
pub(crate) struct LocalFeedState {
    /// The newest frame, copied out of the WGC frame pool (whose surfaces
    /// are recycled by the system) into a stable texture, plus the shader
    /// view created alongside it.
    pub(crate) frame: Option<(ID3D11Texture2D, ID3D11ShaderResourceView)>,
    /// Current frame (and pool) size in pixels.
    pub(crate) frame_size: SizeInt32,
    /// Bumped by the capture callback whenever a frame lands; the present
    /// signal for this window's surfaces.
    pub(crate) version: Arc<AtomicU64>,
    /// When the last capture blit ran (rate-cap bookkeeping).
    pub(crate) last_copy: Option<Instant>,
    /// The window's own blur: one dual-Kawase chain over the window
    /// capture, rendered once per fresh frame and sampled by every Local
    /// surface over this window. `None` until the first blurred frame is
    /// needed.
    pub(crate) blur: Option<BlurEngine>,
    /// The last rendered backdrop, the capture version it was built from
    /// and the material params it was built for; reused by every
    /// animation-driven frame in between.
    pub(crate) backdrop: Option<Backdrop>,
    pub(crate) backdrop_version: u64,
    pub(crate) backdrop_inputs: Option<BlurInputs>,
    /// When the backdrop last rebuilt (force-refresh cadence).
    pub(crate) last_backdrop_build: Option<Instant>,
}

impl LocalFeedState {
    pub(crate) fn new() -> Self {
        Self {
            frame: None,
            frame_size: SizeInt32::default(),
            version: Arc::new(AtomicU64::new(0)),
            last_copy: None,
            blur: None,
            backdrop: None,
            backdrop_version: 0,
            backdrop_inputs: None,
            last_backdrop_build: None,
        }
    }
}

/// The capture host view over one window's local feed: tunnels the feed
/// callback's frame handling into the shared engine state, scoped to this
/// window's entry. Members hand back clones because the shared state lives
/// behind a mutex.
struct LocalFeedHost {
    state: Arc<Mutex<crate::shared::EngineState>>,
    hwnd: isize,
}

// The mutex serializes every D3D11 touch (the immediate context is not
// thread-safe; the capture callback locks it for its blit exactly like the
// monitor feed's callback does).
unsafe impl Send for LocalFeedHost {}

impl WindowFrameHost for LocalFeedHost {
    fn device(&self) -> ID3D11Device {
        self.state
            .lock()
            .expect("shared glass lock poisoned")
            .device
            .clone()
    }

    fn set_frame(&mut self, frame: (ID3D11Texture2D, ID3D11ShaderResourceView), size: SizeInt32) {
        let mut state = self.state.lock().expect("shared glass lock poisoned");
        if let Some(entry) = state.local.get_mut(&self.hwnd) {
            entry.frame = Some(frame);
            entry.frame_size = size;
        }
        // Otherwise the entry was deleted out from under a live feed (its
        // last surface unregistered while a frame raced in): the frame
        // simply drops.
    }

    fn frame_texture(&self) -> Option<ID3D11Texture2D> {
        let state = self.state.lock().expect("shared glass lock poisoned");
        state
            .local
            .get(&self.hwnd)
            .and_then(|entry| entry.frame.as_ref().map(|(texture, _)| texture.clone()))
    }

    fn copy_context(&self) -> ID3D11DeviceContext {
        self.state
            .lock()
            .expect("shared glass lock poisoned")
            .context
            .clone()
    }

    fn frame_size(&self) -> SizeInt32 {
        self.state
            .lock()
            .expect("shared glass lock poisoned")
            .local
            .get(&self.hwnd)
            .map(|entry| entry.frame_size)
            .unwrap_or_default()
    }

    fn version(&self) -> Arc<AtomicU64> {
        self.state
            .lock()
            .expect("shared glass lock poisoned")
            .local
            .get(&self.hwnd)
            .map(|entry| entry.version.clone())
            .unwrap_or_else(|| Arc::new(AtomicU64::new(0)))
    }

    fn any_active(&self) -> bool {
        let state = self.state.lock().expect("shared glass lock poisoned");
        state.surfaces.iter().any(|(_id, compositor, route)| {
            route.consumes_window(self.hwnd)
                && compositor
                    .lock()
                    .map(|compositor| !compositor.is_paused())
                    .unwrap_or(true)
        })
    }

    fn capture_due(&mut self) -> bool {
        let mut state = self.state.lock().expect("shared glass lock poisoned");
        // Copy the interval out first: the entry borrow below is mutable,
        // and the render loop owns `backdrop_interval` (it polls the
        // fullscreen state); the capture callback only consults it against
        // the last blit time, keeping the shell query off the WGC worker
        // thread.
        let interval = state.backdrop_interval;
        let Some(entry) = state.local.get_mut(&self.hwnd) else {
            // No entry (its last surface unregistered mid-frame): nothing
            // is consuming feeds for this window anymore.
            return false;
        };
        let due = interval.map_or(true, |interval| {
            entry
                .last_copy
                .map_or(true, |last| last.elapsed() >= interval)
        });
        if due {
            entry.last_copy = Some(Instant::now());
        }
        due
    }
}

/// Starts the window capture for `hwnd`'s local feed. The session handle
/// is returned to the caller (the render loop manages the lifecycle
/// strictly outside the state lock).
pub(crate) fn start_local_feed(
    state: &Arc<Mutex<crate::shared::EngineState>>,
    hwnd: isize,
) -> Result<WindowCaptureFeed, String> {
    let host: Arc<Mutex<dyn WindowFrameHost>> = Arc::new(Mutex::new(LocalFeedHost {
        state: Arc::clone(state),
        hwnd,
    }));
    WindowCaptureFeed::start(&host, hwnd)
}

/// Refreshes one window's local backdrop: re-runs its Kawase chain when a
/// fresh capture frame, a material/params change or the force cadence
/// demands it; returns whether the blur failed (the caller checks for
/// device loss on failure). Mirrors the global `refresh_backdrop`
/// semantics, scoped to a single window feed.
pub(crate) fn refresh_local_backdrop(
    state: &mut crate::shared::EngineState,
    hwnd: isize,
    force: bool,
) -> bool {
    let Some(inputs) = BlurInputs::collect_matching(state, |route| route.consumes_window(hwnd))
    else {
        // Nothing on this window is awake to consume a backdrop; keep the
        // last one so a surface resuming over a paused video has material
        // immediately.
        return false;
    };
    let Some(entry) = state.local.get_mut(&hwnd) else {
        return false;
    };
    let Some(capture) = entry.frame.as_ref().map(|(_, view)| view.clone()) else {
        return false;
    };
    // Insurance against drivers that report dirty regions inaccurately.
    let stale = entry.last_backdrop_build.map_or(true, |last| {
        last.elapsed().as_secs_f32() >= FORCE_BACKDROP_SECS
    });
    // Idle tick fast path: the backdrop already matches the newest capture
    // frame and the current material: the passes only run when one of
    // those moved.
    let current_version = entry.version.load(Ordering::Relaxed);
    if !force
        && !stale
        && entry.backdrop_version == current_version
        && entry
            .backdrop_inputs
            .as_ref()
            .is_some_and(|last| *last == inputs)
    {
        return false;
    }
    let frame = [
        entry.frame_size.Width.max(1) as u32,
        entry.frame_size.Height.max(1) as u32,
    ];
    let context = state.context.clone();
    let blur = entry
        .blur
        .get_or_insert_with(|| BlurEngine::new(&state.device));
    match blur.render(
        &context,
        &capture,
        frame,
        // Local glass covers its host window's whole frame; the full-frame
        // blur over the window capture is already small.
        None,
        &inputs.params,
        &state.quality,
    ) {
        Ok(backdrop) => {
            entry.backdrop = Some(backdrop);
            entry.backdrop_version = current_version;
            entry.backdrop_inputs = Some(inputs);
            entry.last_backdrop_build = Some(Instant::now());
            false
        }
        Err(error) => {
            eprintln!("Glassbar local blur failed (window {hwnd}): {error}");
            true
        }
    }
}

/// Convenience accessor for the render loop: one window's current backdrop,
/// if its feed has produced one yet.
pub(crate) fn local_backdrop(state: &crate::shared::EngineState, hwnd: isize) -> Option<&Backdrop> {
    state
        .local
        .get(&hwnd)
        .and_then(|entry| entry.backdrop.as_ref())
}
