//! The per-surface D3D11 compositor: ONE swap chain, ONE composite draw
//! call per present, driven by the shared render loop.
//!
//! [`D3D11Compositor`] owns the DirectComposition visual tree bound to the
//! surface window, the compiled composite shaders (full and lite), the
//! three constant buffers (source rect, glass params, backdrop crop) and
//! the compositor-owned animation state. The implementation is split by
//! concern: [`present`] draws, [`gate`] schedules, [`animate`] moves,
//! [`spring`] is the damped-spring math the animations integrate with.
//!
//! [`present`]: self::present
//! [`gate`]: self::gate
//! [`animate`]: self::animate
//! [`spring`]: self::spring

use std::time::{Duration, Instant};

use windows::{
    core::Interface,
    Win32::{
        Foundation::{BOOL, HWND, POINT, RECT},
        Graphics::{
            Direct3D11::{
                ID3D11Buffer, ID3D11Device, ID3D11DeviceContext, ID3D11PixelShader,
                ID3D11RenderTargetView, ID3D11SamplerState, ID3D11Texture2D, ID3D11VertexShader,
            },
            DirectComposition::{
                DCompositionCreateDevice, IDCompositionDevice, IDCompositionTarget,
                IDCompositionVisual,
            },
            Dxgi::{
                Common::{
                    DXGI_ALPHA_MODE_PREMULTIPLIED, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_SAMPLE_DESC,
                },
                CreateDXGIFactory2, IDXGIDevice, IDXGIFactory2, IDXGISwapChain1,
                DXGI_CREATE_FACTORY_FLAGS, DXGI_PRESENT, DXGI_SCALING_STRETCH,
                DXGI_SWAP_CHAIN_DESC1, DXGI_SWAP_EFFECT_FLIP_DISCARD,
                DXGI_USAGE_RENDER_TARGET_OUTPUT,
            },
        },
    },
};

use super::geometry::{source_rect_constants, source_rect_constants_texel};
use super::SharedCompositor;
use crate::shared::{compiled_shader, create_linear_clamp_sampler, CropConstants};
use crate::{
    params::{GlassShaderParams, RevealAnimation},
    SourceRectConstants,
};

use spring::TransformSpring;

mod animate;
mod gate;
mod present;
mod spring;

const LIQUID_COMPOSITION_HLSL: &str = include_str!("../liquid_composition.hlsl");

/// How long after the last fresh capture frame an adaptively-paced surface
/// keeps rendering at the shared loop's full rate before falling back to its
/// idle frame cap. Long enough to ride out capture hiccups, short enough
/// that a one-off update (a clock tick) doesn't hold the fast rate.
const ACTIVE_CONTENT_WINDOW: Duration = Duration::from_millis(250);

/// Shifts one compositor's capture-sampling origin by `dy` physical pixels
/// (the D3D half of [`crate::GlassSurface::translate`]). Only engine-internal
/// locks are taken, so this is safe to call from any thread.
pub fn translate_compositor(compositor: &SharedCompositor, dy: i32) {
    if dy == 0 {
        return;
    }
    let shared = crate::shared::engine();
    let state = shared.state.lock().expect("shared glass lock poisoned");
    if let Ok(mut compositor) = compositor.lock() {
        compositor.translate(&state.context, dy);
    }
}

pub struct D3D11Compositor {
    device: ID3D11Device,
    // The composition objects must outlive the swap chain to keep the visual
    // tree bound to the surface window.
    swap_chain: IDXGISwapChain1,
    #[allow(dead_code)]
    composition_device: IDCompositionDevice,
    #[allow(dead_code)]
    composition_target: IDCompositionTarget,
    #[allow(dead_code)]
    composition_visual: IDCompositionVisual,
    vertex_shader: ID3D11VertexShader,
    pixel_shader: ID3D11PixelShader,
    /// The tier's lite composite shader variant (fewer noise octaves,
    /// single dispersion tap). Compiled once per process alongside the
    /// full shader; selected per present by `lite_shader`.
    pixel_shader_lite: ID3D11PixelShader,
    /// Whether `present` uses the lite shader variant (quality tier).
    lite_shader: bool,
    sampler: ID3D11SamplerState,
    source_rect: ID3D11Buffer,
    source_rect_constants: SourceRectConstants,
    glass_params: ID3D11Buffer,
    glass_params_current: GlassShaderParams,
    /// Compositor-owned anisotropic pill scale (the resize spring's live
    /// squash/stretch), stamped into every params upload.
    pill_scale_x: f32,
    pill_scale_y: f32,
    /// The compositor-owned SHAPE TRANSFORM (the cbuffer's `shapeTransform`
    /// row): (scale_x, scale_y, offset_x, offset_y) in shape-region units.
    /// At rest it is the identity (1, 1, 0, 0); while a shape animation
    /// runs it carries the spring's live value and is stamped into every
    /// params upload, exactly like the pill scale and `animation_time`.
    shape_transform: [f32; 4],
    /// The shape transform's spring — the animated retarget of the drawn
    /// shape (a resize pursuit). `None` = settled or absent.
    shape_spring: Option<TransformSpring>,
    /// The pill squash/stretch spring. Ported from the host's animation
    /// thread into the compositor: integrating here keeps the animation
    /// phase-aligned with the render loop (and paced to composition via the
    /// loop's DWM wait) instead of quantized to a sleep timer's granularity.
    pill_spring: Option<TransformSpring>,
    /// Wall clock for the springs' real-dt integration.
    spring_last_frame: Instant,
    crop_constants: ID3D11Buffer,
    crop_constants_current: CropConstants,
    /// The swap chain's back buffer and its render-target view, acquired
    /// once and reused every frame (refreshed after a resize invalidates
    /// them). Recreating both per frame used to churn thousands of D3D
    /// objects per second across the open surfaces.
    back_buffer: Option<(ID3D11Texture2D, ID3D11RenderTargetView)>,
    previous_cursor_position: Option<POINT>,
    width: u32,
    height: u32,
    #[allow(dead_code)]
    hwnd: HWND,
    #[allow(dead_code)]
    overlay: HWND,
    reveal_current: f32,
    /// Set once the first frame has been presented. Surfaces await this
    /// before being revealed, so a group of freshly created surfaces can pop
    /// in together instead of each appearing when its capture session warms
    /// up.
    pub(crate) has_presented: bool,
    reveal_target: f32,
    reveal: RevealAnimation,
    reveal_dirty: bool,
    reveal_last_frame: Instant,
    /// The surface's on-screen bounds (monitor-absolute pixels), needed to
    /// rebuild the source-rect constants when the surface moves.
    surface_bounds: RECT,
    /// The rect of "space" the backdrop texture covers: the primary
    /// monitor for the GLOBAL feed, the host app window's frame for the
    /// LOCAL feed. The source-rect constants map the surface into it.
    backdrop_space: RECT,
    /// Whether the backdrop space is the monitor (GLOBAL feed). LOCAL
    /// surfaces map into their host window's frame — samples must never be
    /// re-anchored to the monitor (see `translate`).
    monitor_space: bool,
    /// When true, `present` is a no-op: parked (hidden) surfaces render
    /// nothing at all, and their presentation tick only burns a boolean
    /// check.
    paused: bool,
    /// Render exactly one more frame, then pause. Surfaces that are hidden
    /// but must still prove their capture session is warm (group reveals,
    /// pre-warmed pool members) use this: the first frame lands, then all
    /// rendering stops.
    pause_after_present: bool,
    /// Pause as soon as the reveal animation settles at 0 (fully collapsed).
    /// Parking a surface must still RENDER its collapse animation — pausing
    /// immediately would freeze the last visible frame on screen — but once
    /// the glass is fully transparent, rendering stops for good.
    pause_when_collapsed: bool,
    /// Minimum spacing between renders while nothing is animating. This caps
    /// the effective render rate for content-driven frames (a surface can be
    /// driven by capture at the monitor's refresh rate, e.g. 144Hz, which is
    /// wasteful for a blurred backdrop). `zero` = unthrottled (the taskbar).
    min_present_interval: Duration,
    last_present: Option<Instant>,
    /// Adaptive pacing (opt-in, the taskbar pill): while fresh capture
    /// frames keep arriving, content-driven frames render at the shared
    /// render loop's full rate; once the content has been quiet for
    /// [`ACTIVE_CONTENT_WINDOW`], pacing falls back to
    /// `min_present_interval`.
    adaptive_rate: bool,
    /// The last time a fresh capture frame arrived, recorded by
    /// `wants_frame`. Drives the adaptive pacing decision above.
    last_content_frame: Option<Instant>,
    /// Seconds of animation time, ticking while the reveal animates; fed to
    /// the shader so the wobble/turbulence flows instead of being a static
    /// warp.
    animation_time: f32,
    /// Set when params changed outside a render (settings, reveal config) so
    /// the presentation tick renders one fresh frame; cleared by `present`.
    params_dirty: bool,
    /// Shared-field present gate (metaball group sync): while set, ticking
    /// never presents — the field the compositor holds must not show on
    /// screen until its sibling surfaces have caught up with the same field
    /// generation (the manager releases the gate, or the deadline passes).
    /// Without it a render-loop tick landing between two sibling surfaces'
    /// field pushes composes one surface's NEW blob half against the other's
    /// old one — the merged neck opens a gap for exactly that beat, which
    /// heals when the blobs settle.
    present_gate_until: Option<Instant>,
    /// When true, the shared material watch skips this surface: the app
    /// derives its params from the material with its own per-surface rules
    /// (e.g. size-relative rim scaling) and re-applies them itself, so a
    /// global material push must not flatten that derivation.
    material_locked: bool,
}

unsafe impl Send for D3D11Compositor {}

impl D3D11Compositor {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn create(
        device: &ID3D11Device,
        hwnd: HWND,
        overlay: HWND,
        width: u32,
        height: u32,
        surface_bounds: RECT,
        backdrop_space: RECT,
        monitor_space: bool,
        shader_params: GlassShaderParams,
    ) -> Result<Self, String> {
        let factory: IDXGIFactory2 = unsafe { CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)) }
            .map_err(|error| format!("Could not create the DXGI factory: {error}"))?;
        let description = DXGI_SWAP_CHAIN_DESC1 {
            Width: width,
            Height: height,
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            Stereo: BOOL(0),
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
            BufferCount: 2,
            Scaling: DXGI_SCALING_STRETCH,
            SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
            // Premultiplied alpha lets DWM composite the shader's soft SDF
            // coverage, giving antialiased pill edges without a window region.
            AlphaMode: DXGI_ALPHA_MODE_PREMULTIPLIED,
            Flags: 0,
        };
        let swap_chain =
            unsafe { factory.CreateSwapChainForComposition(device, &description, None) }.map_err(
                |error| format!("Could not create the D3D11 composition swap chain: {error}"),
            )?;
        let dxgi_device = device
            .cast::<IDXGIDevice>()
            .map_err(|error| format!("Could not obtain the D3D11 DXGI device: {error}"))?;
        let composition_device: IDCompositionDevice = unsafe {
            DCompositionCreateDevice(&dxgi_device)
        }
        .map_err(|error| format!("Could not create the DirectComposition device: {error}"))?;
        let composition_target: IDCompositionTarget = unsafe {
            composition_device.CreateTargetForHwnd(hwnd, BOOL(1))
        }
        .map_err(|error| format!("Could not create the DirectComposition target: {error}"))?;
        let composition_visual: IDCompositionVisual = unsafe { composition_device.CreateVisual() }
            .map_err(|error| format!("Could not create the composition visual: {error}"))?;
        unsafe {
            composition_visual
                .SetContent(&swap_chain)
                .map_err(|error| {
                    format!("Could not bind the swap chain to the composition visual: {error}")
                })?;
            composition_target
                .SetRoot(&composition_visual)
                .map_err(|error| format!("Could not set the composition root visual: {error}"))?;
            composition_device
                .Commit()
                .map_err(|error| format!("Could not commit the composition tree: {error}"))?;
        }
        // The bytecode is compiled once per process and cached; spawning a
        // surface only instantiates the GPU shader objects (microseconds).
        let mut vertex_shader = None;
        unsafe {
            device.CreateVertexShader(
                &compiled_shader(LIQUID_COMPOSITION_HLSL, "vs_main", "vs_5_0")?,
                None,
                Some(&mut vertex_shader),
            )
        }
        .map_err(|error| format!("Could not create the D3D11 vertex shader: {error}"))?;
        let mut pixel_shader = None;
        unsafe {
            device.CreatePixelShader(
                &compiled_shader(LIQUID_COMPOSITION_HLSL, "ps_main", "ps_5_0")?,
                None,
                Some(&mut pixel_shader),
            )
        }
        .map_err(|error| format!("Could not create the D3D11 pixel shader: {error}"))?;
        let pixel_shader = pixel_shader.expect("D3D11 pixel shader missing");
        // The lite variant compiles alongside the full one (the shader
        // cache makes the second compile cheap after the first surface).
        let mut pixel_shader_lite = None;
        unsafe {
            device.CreatePixelShader(
                &compiled_shader(LIQUID_COMPOSITION_HLSL, "ps_main_lite", "ps_5_0")?,
                None,
                Some(&mut pixel_shader_lite),
            )
        }
        .map_err(|error| format!("Could not create the lite pixel shader: {error}"))?;
        let pixel_shader_lite = pixel_shader_lite.expect("lite pixel shader missing");
        let sampler = create_linear_clamp_sampler(device);
        let source_rect_constants = source_rect_constants(surface_bounds, backdrop_space)?;
        let source_rect = crate::shared::constant_buffer(device, &source_rect_constants)?;
        let glass_params = crate::shared::constant_buffer(device, &shader_params)?;
        // The composite starts with the identity crop (backdrop = the whole
        // capture); `present` re-stamps the buffer from the backdrop's own
        // crop constants whenever the engine blurred a sub-region, and
        // `resize` keeps the capture texel size fresh in between.
        let crop_constants_current =
            CropConstants::identity(source_rect_constants_texel(&backdrop_space));
        let crop_constants = crate::shared::constant_buffer(device, &crop_constants_current)?;
        Ok(Self {
            device: device.clone(),
            swap_chain,
            composition_device,
            composition_target,
            composition_visual,
            vertex_shader: vertex_shader.expect("D3D11 vertex shader missing"),
            pixel_shader,
            pixel_shader_lite,
            lite_shader: false,
            sampler,
            source_rect,
            source_rect_constants,
            glass_params,
            glass_params_current: shader_params,
            pill_scale_x: 1.0,
            pill_scale_y: 1.0,
            shape_transform: [1.0, 1.0, 0.0, 0.0],
            shape_spring: None,
            pill_spring: None,
            spring_last_frame: Instant::now(),
            crop_constants,
            crop_constants_current,
            back_buffer: None,
            previous_cursor_position: None,
            width,
            height,
            hwnd,
            overlay,
            reveal_current: 0.0,
            reveal_target: 1.0,
            reveal: RevealAnimation::default(),
            reveal_dirty: false,
            reveal_last_frame: Instant::now(),
            has_presented: false,
            surface_bounds,
            backdrop_space,
            monitor_space,
            paused: false,
            pause_after_present: false,
            pause_when_collapsed: false,
            min_present_interval: Duration::ZERO,
            last_present: None,
            adaptive_rate: false,
            last_content_frame: None,
            animation_time: 0.0,
            params_dirty: false,
            present_gate_until: None,
            material_locked: false,
        })
    }
}

impl D3D11Compositor {
    pub(super) fn resize(
        &mut self,
        context: &ID3D11DeviceContext,
        width: u32,
        height: u32,
        surface_bounds: RECT,
        backdrop_space: RECT,
    ) -> Result<(), String> {
        self.surface_bounds = surface_bounds;
        let source_rect_constants = source_rect_constants(surface_bounds, backdrop_space)?;
        if source_rect_constants != self.source_rect_constants {
            unsafe {
                context.UpdateSubresource(
                    &self.source_rect,
                    0,
                    None,
                    (&source_rect_constants as *const SourceRectConstants).cast(),
                    0,
                    0,
                );
            }
            self.source_rect_constants = source_rect_constants;
        }
        // The capture texel size feeds the composite's displacement math in
        // capture-uv units; a monitor change (DPI, resolution) moves it —
        // and for a LOCAL feed, a host window resize does.
        self.backdrop_space = backdrop_space;
        let texel = source_rect_constants_texel(&backdrop_space);
        if texel != self.crop_constants_current.capture_texel_uv {
            self.crop_constants_current.capture_texel_uv = texel;
            unsafe {
                context.UpdateSubresource(
                    &self.crop_constants,
                    0,
                    None,
                    (&self.crop_constants_current as *const CropConstants).cast(),
                    0,
                    0,
                );
            }
        }
        if width == self.width && height == self.height {
            return Ok(());
        }
        // Resizing releases the swap chain's buffers: drop the cached view
        // so the next frame re-acquires a fresh back buffer.
        self.back_buffer = None;
        unsafe {
            self.swap_chain.ResizeBuffers(
                0,
                width,
                height,
                DXGI_FORMAT_B8G8R8A8_UNORM,
                Default::default(),
            )
        }
        .map_err(|error| format!("Could not resize the D3D11 swap chain: {error}"))?;
        self.width = width;
        self.height = height;
        // A resized composition swap chain's buffers are UNINITIALIZED:
        // DWM composites garbage (a black box) until the next present —
        // visible as a black flash on every resize of visible glass. Clear
        // and present one fully transparent frame immediately, so the
        // window never shows the reallocated buffers' contents.
        let (_, target) = self.back_buffer_view(context)?;
        unsafe {
            context.ClearRenderTargetView(&target, &[0.0, 0.0, 0.0, 0.0]);
            self.swap_chain
                .Present(0, DXGI_PRESENT(0))
                .ok()
                .map_err(|error| format!("Could not present the resize clear frame: {error}"))?;
        }
        Ok(())
    }
}

#[cfg(test)]
pub(super) fn compile_shader(entry: &str, target: &str) -> Result<Vec<u8>, String> {
    crate::shared::compile_shader_from(LIQUID_COMPOSITION_HLSL, entry, target)
}
