//! The shared backdrop: what one composite pass samples.
//!
//! [`Backdrop`] is the pair the drawer needs — the shader view plus the
//! [`CropConstants`] recording the capture region the blur covered. The
//! helpers here decide WHAT region of the capture the shared chain should
//! blur over (the awake surfaces' union, grown by the composite's
//! displacement reach and rounded to a 64 px grid) and drive the per-tick
//! refresh (`refresh_backdrop`): one blur per fresh frame, reused by every
//! animation tick in between.

use std::time::Instant;

use windows::Win32::Graphics::Direct3D11::ID3D11ShaderResourceView;

use crate::backdrop::SurfaceRoute;
use crate::params::{BackdropSource, GlassShaderParams};

use super::blur::BlurEngine;
use super::EngineState;

/// Insurance against a misbehaving capture driver: with dirty-region skips
/// active, the backdrop still force-refreshes at least this often.
const FORCE_BACKDROP_SECS: f32 = 2.0;

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
pub(crate) fn grid_aligned_crop(rect: [f32; 4], frame: [u32; 2], grid: f32) -> Option<[u32; 4]> {
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
pub(crate) fn refresh_backdrop(state: &mut EngineState, version: u64, force: bool) -> bool {
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
