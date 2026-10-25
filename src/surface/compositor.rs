use std::time::{Duration, Instant};

use windows::{
    core::Interface,
    Win32::{
        Foundation::{BOOL, HWND, POINT, RECT},
        Graphics::{
            Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            Direct3D11::{
                ID3D11Buffer, ID3D11Device, ID3D11DeviceContext, ID3D11PixelShader,
                ID3D11RenderTargetView, ID3D11SamplerState, ID3D11Texture2D, ID3D11VertexShader,
                D3D11_CPU_ACCESS_READ, D3D11_MAP_READ, D3D11_MAPPED_SUBRESOURCE,
                D3D11_TEXTURE2D_DESC, D3D11_USAGE_STAGING, D3D11_VIEWPORT,
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
        UI::WindowsAndMessaging::GetCursorPos,
    },
};

use super::SharedCompositor;
use super::geometry::{primary_monitor_bounds, source_rect_constants, source_rect_constants_texel};
use crate::shared::{Backdrop, CropConstants, compiled_shader, create_linear_clamp_sampler};
use crate::{
    params::{GlassShaderParams, RevealAnimation},
    SourceRectConstants,
};

const LIQUID_COMPOSITION_HLSL: &str = include_str!("liquid_composition.hlsl");

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
    sampler: ID3D11SamplerState,
    source_rect: ID3D11Buffer,
    source_rect_constants: SourceRectConstants,
    glass_params: ID3D11Buffer,
    glass_params_current: GlassShaderParams,
    /// Compositor-owned anisotropic pill scale (the resize spring's live
    /// squash/stretch), stamped into every params upload.
    pill_scale_x: f32,
    pill_scale_y: f32,
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
        primary_monitor_bounds: RECT,
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
        let sampler = create_linear_clamp_sampler(device);
        let source_rect_constants =
            source_rect_constants(surface_bounds, primary_monitor_bounds)?;
        let source_rect = crate::shared::constant_buffer(device, &source_rect_constants)?;
        let glass_params = crate::shared::constant_buffer(device, &shader_params)?;
        // The shared engine blurs the WHOLE capture, so the composite's
        // backdrop uv equals the capture uv (identity crop); the buffer only
        // carries the capture texel size and the sharp-backdrop flag, both
        // refreshed by `present`/`resize`.
        let crop_constants_current =
            CropConstants::identity(source_rect_constants_texel(&primary_monitor_bounds));
        let crop_constants = crate::shared::constant_buffer(device, &crop_constants_current)?;
        Ok(Self {
            device: device.clone(),
            swap_chain,
            composition_device,
            composition_target,
            composition_visual,
            vertex_shader: vertex_shader.expect("D3D11 vertex shader missing"),
            pixel_shader: pixel_shader.expect("D3D11 pixel shader missing"),
            sampler,
            source_rect,
            source_rect_constants,
            glass_params,
            glass_params_current: shader_params,
            pill_scale_x: 1.0,
            pill_scale_y: 1.0,
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
            paused: false,
            pause_after_present: false,
            pause_when_collapsed: false,
            min_present_interval: Duration::ZERO,
            last_present: None,
            adaptive_rate: false,
            last_content_frame: None,
            animation_time: 0.0,
            params_dirty: false,
            material_locked: false,
        })
    }

    /// Composites this surface's frame from the shared blurred backdrop.
    /// The heavy blur runs once per capture frame in the engine; a present
    /// is one draw call (displace, tint, mask) plus the flip-model present.
    pub(crate) fn present(
        &mut self,
        context: &ID3D11DeviceContext,
        backdrop: &Backdrop,
    ) -> Result<(), String> {
        if self.paused {
            return Ok(());
        }
        let pause_now = self.pause_after_present;
        // While the reveal animation runs, render at the caller's rate (the
        // presentation tick keeps it smooth). When idle, throttle
        // content-driven renders to the surface's frame cap — capture on a
        // high-refresh monitor would otherwise drive every surface at full
        // refresh rate.
        let animating = (self.reveal_current - self.reveal_target).abs() > 0.001;
        if !animating && !pause_now {
            let now = Instant::now();
            // Adaptive surfaces (the taskbar pill) render at the shared
            // loop's full rate while fresh capture frames keep arriving —
            // content is moving somewhere, so the glass tracks it smoothly —
            // and fall back to the idle frame cap once the content has been
            // quiet past the grace window.
            let interval = if self.adaptive_rate
                && self
                    .last_content_frame
                    .map(|fresh| now.duration_since(fresh) < ACTIVE_CONTENT_WINDOW)
                    .unwrap_or(false)
            {
                Duration::ZERO
            } else {
                self.min_present_interval
            };
            if let Some(last) = self.last_present {
                if now.duration_since(last) < interval {
                    return Ok(());
                }
            }
            self.last_present = Some(now);
        }
        self.params_dirty = false;
        self.update_pointer_velocity(context);
        self.update_reveal(context);
        let (_, target) = self.back_buffer_view(context)?;
        unsafe {
            context.OMSetRenderTargets(Some(&[Some(target.clone())]), None);
            context.RSSetViewports(Some(&[D3D11_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: self.width as f32,
                Height: self.height as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]));
            context.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            context.VSSetShader(&self.vertex_shader, None);
            context.PSSetShader(&self.pixel_shader, None);
            context.PSSetSamplers(0, Some(&[Some(self.sampler.clone())]));
            context.PSSetShaderResources(0, Some(&[Some(backdrop.view.clone())]));
            context.PSSetConstantBuffers(0, Some(&[Some(self.source_rect.clone())]));
            context.PSSetConstantBuffers(1, Some(&[Some(self.glass_params.clone())]));
            context.PSSetConstantBuffers(2, Some(&[Some(self.crop_constants.clone())]));
            context.Draw(3, 0);
            // Sync interval 0: a composition swap chain cannot tear (DWM
            // composites it), and waiting for the vblank here used to hold
            // the shared engine's lock for a vblank per surface — with a
            // panel of glass open that serialized the whole render loop,
            // starved the capture callback and blocked every main-thread
            // glass call behind it.
            self.swap_chain
                .Present(0, DXGI_PRESENT(0))
                .ok()
                .map_err(|error| format!("Could not present the D3D11 frame: {error}"))?;
            self.has_presented = true;
            context.PSSetShaderResources(0, Some(&[None]));
        }
        if pause_now {
            self.paused = true;
            self.pause_after_present = false;
        }
        // The surface window is shown by `stack_beneath_host` on the thread
        // that created it; never touch window visibility from the capture
        // callback — ShowWindow/SetWindowPos are synchronous across threads
        // and deadlocked the compositor against the window's owner.
        Ok(())
    }

    /// The swap chain's back buffer and its render-target view, acquired
    /// once and reused until a resize invalidates them.
    fn back_buffer_view(
        &mut self,
        context: &ID3D11DeviceContext,
    ) -> Result<(ID3D11Texture2D, ID3D11RenderTargetView), String> {
        if let Some((_, view)) = &self.back_buffer {
            return Ok((
                self.back_buffer.as_ref().expect("checked").0.clone(),
                view.clone(),
            ));
        }
        let back_buffer: ID3D11Texture2D = unsafe { self.swap_chain.GetBuffer(0) }
            .map_err(|error| format!("Could not acquire the D3D11 swap-chain buffer: {error}"))?;
        let mut target = None;
        unsafe {
            self.device
                .CreateRenderTargetView(&back_buffer, None, Some(&mut target))
        }
        .map_err(|error| format!("Could not create the D3D11 render target view: {error}"))?;
        let pair = (
            back_buffer,
            target.expect("D3D11 render target view missing"),
        );
        self.back_buffer = Some(pair.clone());
        // A fresh (or resized) buffer can hold garbage from its last life:
        // clear it once so the surface never flashes a black box.
        unsafe { context.ClearRenderTargetView(&pair.1, &[0.0, 0.0, 0.0, 0.0]) };
        Ok(pair)
    }

    /// Stops (or resumes) all rendering for this surface. Parked surfaces are
    /// paused: their presentation tick and capture callbacks then do nothing
    /// beyond a boolean check, so invisible glass costs no GPU time.
    pub fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
        if !paused {
            self.pause_after_present = false;
            self.pause_when_collapsed = false;
        }
        // Force one fresh frame with the current state on resume.
        self.params_dirty = true;
    }

    /// Whether all rendering is stopped for this surface. The capture feed
    /// uses this to skip the full-screen frame copy while every surface is
    /// parked.
    pub fn is_paused(&self) -> bool {
        self.paused
    }

    /// The surface's current material params. All surfaces derive from the
    /// same saved settings, so the engine uses the first active surface's
    /// copy to plan the shared blur (blur radius and samples are identical
    /// across surfaces).
    pub(crate) fn active_params(&self) -> GlassShaderParams {
        self.glass_params_current
    }

    /// Whether the shared material watch must skip this surface (see the
    /// `material_locked` field).
    pub(crate) fn material_locked(&self) -> bool {
        self.material_locked
    }

    /// Pins (or unpins) the surface against the shared material watch: a
    /// pinned surface keeps its app-derived params; the app re-applies the
    /// material itself with its own per-surface rules.
    pub fn set_material_locked(&mut self, locked: bool) {
        self.material_locked = locked;
    }

    /// Lets exactly one more frame render, then pauses. Hidden surfaces use
    /// this to warm their capture session (and become "presented") without
    /// rendering continuously.
    pub fn pause_after_next_present(&mut self) {
        self.paused = false;
        self.pause_after_present = true;
        self.pause_when_collapsed = false;
        self.params_dirty = true;
    }

    /// Renders the collapse animation to the end, then pauses. Used when
    /// parking a surface: the close transition must play out (pausing
    /// immediately freezes the last visible frame on screen), and once the
    /// glass is fully transparent no more rendering is needed.
    pub fn collapse_then_pause(&mut self) {
        self.pause_when_collapsed = true;
        // An already-collapsed surface needs no collapse render: its last
        // frame is already transparent, so park it immediately.
        if self.reveal_current <= 0.001 && self.reveal_target <= 0.001 {
            self.paused = true;
            self.pause_when_collapsed = false;
            return;
        }
        self.params_dirty = true;
    }

    /// Caps the idle render rate (content-driven frames); zero disables the
    /// cap. Reveal animations always run at the caller's tick rate.
    pub fn set_min_present_interval(&mut self, interval: Duration) {
        self.min_present_interval = interval;
    }

    /// Enables adaptive pacing: content-driven frames render at the shared
    /// loop's full rate while fresh capture frames keep arriving, then fall
    /// back to the idle frame cap after the content goes quiet.
    pub fn set_adaptive_rate(&mut self, enabled: bool) {
        self.adaptive_rate = enabled;
    }

    /// Whether the presentation tick must render right now: the reveal
    /// animation is running, or the params changed and need a frame to land.
    pub fn needs_present(&self) -> bool {
        if self.paused {
            return false;
        }
        if self.pause_after_present {
            return true;
        }
        let animating = (self.reveal_current - self.reveal_target).abs() > 0.001;
        animating || self.reveal_dirty || self.params_dirty
    }

    /// Whether the render loop should render for this surface given the
    /// shared capture state. Surface-local needs (animation, dirty params)
    /// always render; content-driven needs additionally respect the frame
    /// cap via `present`'s own throttle. Fresh capture frames also refresh
    /// the adaptive pacing's content timer.
    pub fn wants_frame(&mut self, frame_changed: bool) -> bool {
        if self.paused {
            return false;
        }
        if self.pause_after_present {
            return true;
        }
        if frame_changed {
            self.last_content_frame = Some(Instant::now());
        }
        let animating = (self.reveal_current - self.reveal_target).abs() > 0.001;
        if animating || self.reveal_dirty || self.params_dirty {
            return true;
        }
        frame_changed
    }

    /// Starts (or reverses) the reveal animation with the given configuration.
    pub fn set_reveal(&mut self, open: bool, reveal: RevealAnimation) {
        let target = if open { 1.0 } else { 0.0 };
        // Re-asserting an unchanged, settled state (every tracked push
        // re-asserts the open reveal) must not dirty the surface: that would
        // force a fresh frame forever and pin the shared render loop at its
        // full vsync rate for every open surface.
        if !self.paused
            && target == self.reveal_target
            && reveal == self.reveal
            && (self.reveal_current - target).abs() <= 0.001
        {
            return;
        }
        self.reveal_target = target;
        self.reveal = reveal;
        self.glass_params_current.reveal_wobble = reveal.wobble;
        self.glass_params_current.turbulence_strength = reveal.turbulence_strength;
        self.glass_params_current.turbulence_scale = reveal.turbulence_scale;
        self.glass_params_current.reveal_origin = reveal.origin;
        if open {
            self.pause_when_collapsed = false;
            self.paused = false;
        }
        self.reveal_dirty = true;
    }

    fn update_reveal(&mut self, context: &ID3D11DeviceContext) {
        let now = Instant::now();
        let dt = now
            .duration_since(self.reveal_last_frame)
            .as_secs_f32()
            .min(0.05);
        self.reveal_last_frame = now;
        // Time only advances while frames are being produced (i.e. while
        // animating), which is all the fluid motion needs.
        self.animation_time = (self.animation_time + dt) % 3600.0;
        // Consume a pending collapse pause even when nothing is animating:
        // `collapse_then_pause` on an already-collapsed surface would
        // otherwise never reach this state through the animation path (the
        // early return below skips it) and the surface would linger unpaused.
        if self.pause_when_collapsed && self.reveal_current <= 0.001 && self.reveal_target <= 0.001
        {
            self.pause_when_collapsed = false;
            self.paused = true;
        }
        let animating = (self.reveal_current - self.reveal_target).abs() > 0.001;
        if !animating && !self.reveal_dirty {
            return;
        }
        if animating {
            let direction = (self.reveal_target - self.reveal_current).signum();
            self.reveal_current =
                (self.reveal_current + direction * dt / self.reveal.duration_secs).clamp(0.0, 1.0);
        } else {
            self.reveal_current = self.reveal_target;
        }
        self.reveal_dirty = false;
        // The collapse finished: the surface is fully transparent, so stop
        // rendering it entirely (the parked surface keeps presenting its
        // last — invisible — frame until it is re-anchored).
        if self.pause_when_collapsed && !animating && self.reveal_current <= 0.001 {
            self.pause_when_collapsed = false;
            self.paused = true;
        }
        let mut params = self.glass_params_current;
        params.reveal_progress = self.reveal_current;
        params.animation_time = self.animation_time;
        self.stamp_pill_scale_xy(&mut params);
        unsafe {
            context.UpdateSubresource(
                &self.glass_params,
                0,
                None,
                (&params as *const GlassShaderParams).cast(),
                0,
                0,
            );
        }
    }

    pub fn set_params(
        &mut self,
        context: &ID3D11DeviceContext,
        params: GlassShaderParams,
    ) -> Result<(), String> {
        // Callers pass params without animation time (it is compositor-owned);
        // compare with time zeroed so the ticking clock never counts as a
        // param change.
        let mut incoming_no_time = params;
        incoming_no_time.animation_time = 0.0;
        let mut current_no_time = self.glass_params_current;
        current_no_time.animation_time = 0.0;
        if incoming_no_time == current_no_time {
            return Ok(());
        }
        let mut incoming = params;
        incoming.animation_time = self.animation_time;
        self.stamp_pill_scale_xy(&mut incoming);
        unsafe {
            context.UpdateSubresource(
                &self.glass_params,
                0,
                None,
                (&incoming as *const GlassShaderParams).cast(),
                0,
                0,
            );
        }
        self.glass_params_current = incoming;
        // If the surface is parked (paused, or fully collapsed awaiting its
        // collapse render), a params change must NOT wake it: the params are
        // stored and `set_paused(false)` re-dirties on wake, so parked glass
        // can never flash back on screen from a settings sweep.
        if self.paused || (self.reveal_target <= 0.001 && self.reveal_current <= 0.001) {
            return Ok(());
        }
        self.params_dirty = true;
        Ok(())
    }

    pub(super) fn resize(
        &mut self,
        context: &ID3D11DeviceContext,
        width: u32,
        height: u32,
        surface_bounds: RECT,
        primary_monitor_bounds: RECT,
    ) -> Result<(), String> {
        self.surface_bounds = surface_bounds;
        let source_rect_constants =
            source_rect_constants(surface_bounds, primary_monitor_bounds)?;
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
        // capture-uv units; a monitor change (DPI, resolution) moves it.
        let texel = source_rect_constants_texel(&primary_monitor_bounds);
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
        Ok(())
    }

    /// Presents one fully transparent frame so freshly (re)allocated swap chain
    /// buffers never show as a black box before the next capture frame lands.
    pub fn clear_to_transparent(&mut self, context: &ID3D11DeviceContext) -> Result<(), String> {
        let (_, target) = self.back_buffer_view(context)?;
        unsafe {
            context.ClearRenderTargetView(&target, &[0.0, 0.0, 0.0, 0.0]);
            self.swap_chain
                .Present(0, DXGI_PRESENT(0))
                .ok()
                .map_err(|error| format!("Could not present the clear frame: {error}"))?;
        }
        Ok(())
    }

    /// Diagnostic: copies the last rendered frame (taken before Present) to
    /// the CPU and returns the mean BGRA (0..255 per channel) of one sub-rect,
    /// given as normalized fractions (x0, y0, x1, y1); `None` covers the whole
    /// buffer.
    /// Diagnostic: copies the back buffer to the CPU and returns its mean BGRA
    /// (0..255 per channel). Reveals whether the surface is transparent.
    pub fn backbuffer_mean(&mut self, context: &ID3D11DeviceContext) -> Result<[f32; 4], String> {
        let back_buffer: ID3D11Texture2D = unsafe { self.swap_chain.GetBuffer(0) }
            .map_err(|error| format!("Could not acquire the D3D11 swap-chain buffer: {error}"))?;
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        unsafe { back_buffer.GetDesc(&mut desc) };
        let staging_desc = D3D11_TEXTURE2D_DESC {
            Usage: D3D11_USAGE_STAGING,
            BindFlags: 0,
            CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
            MiscFlags: 0,
            ..desc
        };
        let mut staging: Option<ID3D11Texture2D> = None;
        unsafe {
            self.device
                .CreateTexture2D(&staging_desc, None, Some(&mut staging))
        }
        .map_err(|error| format!("Could not create the staging texture: {error}"))?;
        let staging = staging.expect("D3D11 staging texture missing");
        unsafe { context.CopyResource(&staging, &back_buffer) };
        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        unsafe { context.Map(&staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped)) }
            .map_err(|error| format!("Could not map the staging texture: {error}"))?;
        let row = unsafe {
            std::slice::from_raw_parts(
                mapped.pData as *const u8,
                mapped.RowPitch as usize * desc.Height as usize,
            )
        };
        let (mut b, mut g, mut r, mut a, mut count) = (0f64, 0f64, 0f64, 0f64, 0u64);
        for y in 0..desc.Height as usize {
            let step = (desc.Width as usize / 64).max(1);
            for x in (0..desc.Width as usize).step_by(step) {
                let offset = y * mapped.RowPitch as usize + x * 4;
                b += row[offset] as f64;
                g += row[offset + 1] as f64;
                r += row[offset + 2] as f64;
                a += row[offset + 3] as f64;
                count += 1;
            }
        }
        unsafe { context.Unmap(&staging, 0) };
        let total = count.max(1) as f64;
        Ok([
            (b / total) as f32,
            (g / total) as f32,
            (r / total) as f32,
            (a / total) as f32,
        ])
    }

    fn update_pointer_velocity(&mut self, context: &ID3D11DeviceContext) {
        let mut cursor = POINT::default();
        let velocity = match unsafe { GetCursorPos(&mut cursor) } {
            Ok(()) => {
                let velocity = self
                    .previous_cursor_position
                    .map(|previous| {
                        [
                            (cursor.x - previous.x).clamp(-36, 36) as f32,
                            (cursor.y - previous.y).clamp(-36, 36) as f32,
                        ]
                    })
                    .unwrap_or([0.0, 0.0]);
                self.previous_cursor_position = Some(cursor);
                velocity
            }
            Err(_) => {
                self.previous_cursor_position = None;
                [0.0, 0.0]
            }
        };
        if self.source_rect_constants.pointer_velocity == velocity {
            return;
        }
        self.source_rect_constants.pointer_velocity = velocity;
        unsafe {
            context.UpdateSubresource(
                &self.source_rect,
                0,
                None,
                (&self.source_rect_constants as *const SourceRectConstants).cast(),
                0,
                0,
            );
        }
    }

    /// Sets the pill's uniform shape scale, re-uploading the params only when
    /// the value changes (called every animation tick).
    pub fn set_pill_scale(&mut self, context: &ID3D11DeviceContext, scale: f32) {
        if (self.glass_params_current.pill_scale - scale).abs() < 0.001 {
            return;
        }
        self.glass_params_current.pill_scale = scale;
        let mut params = self.glass_params_current;
        self.stamp_pill_scale_xy(&mut params);
        unsafe {
            context.UpdateSubresource(
                &self.glass_params,
                0,
                None,
                (&params as *const GlassShaderParams).cast(),
                0,
                0,
            );
        }
    }

    /// Sets the pill's anisotropic squash/stretch factors (1 = undistorted),
    /// re-uploading the params only when the values change (called every
    /// spring tick by the taskbar's resize animation).
    pub fn set_pill_scale_xy(&mut self, context: &ID3D11DeviceContext, scale_x: f32, scale_y: f32) {
        if (self.pill_scale_x - scale_x).abs() < 0.001
            && (self.pill_scale_y - scale_y).abs() < 0.001
        {
            return;
        }
        self.pill_scale_x = scale_x;
        self.pill_scale_y = scale_y;
        let mut params = self.glass_params_current;
        self.stamp_pill_scale_xy(&mut params);
        unsafe {
            context.UpdateSubresource(
                &self.glass_params,
                0,
                None,
                (&params as *const GlassShaderParams).cast(),
                0,
                0,
            );
        }
    }

    /// Stamps the compositor-owned anisotropic scale into a params copy
    /// before upload (the same way `animation_time` is compositor-owned).
    fn stamp_pill_scale_xy(&self, params: &mut GlassShaderParams) {
        params.pill_scale_x = self.pill_scale_x;
        params.pill_scale_y = self.pill_scale_y;
    }

    /// Shifts the sampling origin down with the surface window (auto-hide slide).
    pub fn translate(&mut self, context: &ID3D11DeviceContext, dy: i32) {
        if dy == 0 {
            return;
        }
        self.surface_bounds.top += dy;
        self.surface_bounds.bottom += dy;
        let Ok(monitor) = primary_monitor_bounds() else {
            return;
        };
        if let Ok(updated) =
            source_rect_constants(self.surface_bounds, monitor)
        {
            unsafe {
                context.UpdateSubresource(
                    &self.source_rect,
                    0,
                    None,
                    (&updated as *const SourceRectConstants).cast(),
                    0,
                    0,
                );
            }
            self.source_rect_constants = updated;
        }
    }
}

#[cfg(test)]
pub(super) fn compile_shader(entry: &str, target: &str) -> Result<Vec<u8>, String> {
    crate::shared::compile_shader_from(LIQUID_COMPOSITION_HLSL, entry, target)
}