use std::sync::{Arc, Mutex};

use windows::Win32::{
    Foundation::HWND,
    Graphics::Direct3D11::ID3D11DeviceContext,
    UI::WindowsAndMessaging::{DestroyWindow, SetWindowDisplayAffinity, WDA_EXCLUDEFROMCAPTURE},
};

use super::compositor::D3D11Compositor;
use super::geometry::{anchor_rect, height, overlay_bounds, primary_monitor_bounds, width};
use super::window::{
    create_surface_window, hide_surface_window, stack_beneath_host, translate_surface_window,
};
use super::SharedCompositor;
use crate::params::{GlassConfig, GlassShaderParams, RevealAnimation, SurfaceAnchor};
use crate::shared::engine;

/// DXGI D3D11 presentation target positioned beneath a host overlay window.
/// Create one per overlay with [`GlassSurface::create`]; keep it alive while
/// the glass should render. Dropping it stops the capture and destroys the
/// native surface window.
///
/// All surfaces share one D3D11 device, one capture session and one render
/// loop (see the shared engine); a surface only owns its window, swap chain
/// and compositor state.
pub struct GlassSurface {
    host: isize,
    hwnd: HWND,
    engine_id: u64,
    compositor: SharedCompositor,
    shader_params: GlassShaderParams,
    corner_radius_px: f32,
    surface_anchor: SurfaceAnchor,
    surface_height_px: f32,
    style_host_window: bool,
}

unsafe impl Send for GlassSurface {}

impl GlassSurface {
    /// Creates the GPU surface beneath `host` (an HWND value) and joins the
    /// shared capture.
    pub fn create(host: isize, config: GlassConfig) -> Result<Self, String> {
        let shared = engine();
        let overlay = HWND(host as *mut _);
        let bounds = anchor_rect(overlay_bounds(overlay)?, &config.surface_anchor);
        let primary_monitor_bounds = primary_monitor_bounds()?;
        let hwnd = create_surface_window(bounds)?;
        let compositor = {
            let device = shared
                .state
                .lock()
                .expect("shared glass lock poisoned")
                .device
                .clone();
            match D3D11Compositor::create(
                &device,
                hwnd,
                overlay,
                width(&bounds)?,
                height(&bounds)?,
                bounds,
                primary_monitor_bounds,
                {
                    // The configured corner radius is expressed in surface
                    // pixels and translated into the SDF's normalized capsule
                    // radius. Texel-unit params are compensated for the
                    // downscaled shared capture here, so callers keep passing
                    // full-resolution values.
                    let mut shader_params = scale_texel_params(config.shader, shared.capture_scale());
                    shader_params.capsule_radius =
                        (config.corner_radius_px / height(&bounds)? as f32).clamp(0.0, 0.5);
                    shader_params
                },
            ) {
                Ok(compositor) => Arc::new(Mutex::new(compositor)),
                Err(error) => {
                    unsafe {
                        let _ = DestroyWindow(hwnd);
                    }
                    return Err(error);
                }
            }
        };

        let mut surface = Self {
            host,
            hwnd,
            engine_id: 0,
            compositor,
            shader_params: config.shader,
            corner_radius_px: config.corner_radius_px,
            surface_anchor: config.surface_anchor,
            surface_height_px: height(&bounds)? as f32,
            style_host_window: config.style_host_window,
        };
        unsafe {
            // Display affinity keeps the GPU surface itself out of every
            // screen capture, including the shared WGC feed: without it the
            // glass would refract its own rendered pixels (a feedback loop).
            SetWindowDisplayAffinity(surface.hwnd, WDA_EXCLUDEFROMCAPTURE).map_err(|error| {
                format!("Could not exclude the GPU surface from capture: {error}")
            })?;
            stack_beneath_host(
                surface.host,
                surface.hwnd,
                &surface.surface_anchor,
                surface.style_host_window,
            )?;
        }
        // Initialize the swap chain to fully transparent: the first real
        // frame can be arbitrarily late and an uninitialized buffer renders
        // as a black box.
        surface.with_context(|context| {
            surface
                .compositor
                .lock()
                .expect("D3D11 compositor lock poisoned")
                .clear_to_transparent(context)
        })?;
        surface.engine_id = shared.register(Arc::clone(&surface.compositor));
        Ok(surface)
    }

    /// Runs a closure with the shared immediate context while holding the
    /// engine lock — every D3D11 touch outside the render loop must be
    /// serialized this way (the immediate context is not thread-safe).
    fn with_context<T>(
        &self,
        f: impl FnOnce(&ID3D11DeviceContext) -> Result<T, String>,
    ) -> Result<T, String> {
        let shared = engine();
        let state = shared.state.lock().expect("shared glass lock poisoned");
        f(&state.context)
    }

    pub fn compositor(&self) -> SharedCompositor {
        Arc::clone(&self.compositor)
    }

    /// Replaces every shader float live; takes effect on the next frame.
    pub fn set_shader_params(&mut self, mut params: GlassShaderParams) -> Result<(), String> {
        params = scale_texel_params(params, engine().capture_scale());
        self.with_context(|context| {
            self.compositor
                .lock()
                .expect("D3D11 compositor lock poisoned")
                .set_params(context, params)
        })?;
        self.shader_params = params;
        Ok(())
    }
    /// Like [`GlassSurface::set_shader_params`], but keeps the surface's own corner radius:
    /// shared settings applied to several surfaces won't distort their shapes.
    pub fn set_shader_params_keeping_corner(
        &mut self,
        mut params: GlassShaderParams,
    ) -> Result<(), String> {
        params.capsule_radius =
            (self.corner_radius_px / self.surface_height_px.max(1.0)).clamp(0.0, 0.5);
        self.set_shader_params(params)
    }

    /// Pins (or unpins) this surface against the shared material watch: a
    /// pinned surface keeps its app-derived params and the app re-applies
    /// the material itself (with its own per-surface rules, e.g. size-
    /// relative rim scaling); an unpinned surface adopts every global
    /// material change automatically.
    pub fn set_material_locked(&mut self, locked: bool) {
        self.compositor
            .lock()
            .expect("D3D11 compositor lock poisoned")
            .set_material_locked(locked);
    }

    /// Diagnostic: mean BGRA of the surface's back buffer (0..255/channel).
    pub fn backbuffer_mean(&mut self) -> Option<[f32; 4]> {
        self.with_context(|context| {
            self.compositor
                .lock()
                .expect("D3D11 compositor lock poisoned")
                .backbuffer_mean(context)
        })
        .ok()
    }

    /// Changes the pill's corner radius in surface pixels. Any radius is
    /// allowed, from square corners (`0.0`) to a full capsule (half the
    /// surface height). The radius lives in the shader's SDF, so edges stay
    /// antialiased through the composition alpha.
    pub fn set_corner_radius(&mut self, corner_radius_px: f32) -> Result<(), String> {
        self.corner_radius_px = corner_radius_px;
        let radius_units = (corner_radius_px / self.surface_height_px.max(1.0)).clamp(0.0, 0.5);
        self.shader_params.capsule_radius = radius_units;
        self.with_context(|context| {
            self.compositor
                .lock()
                .expect("D3D11 compositor lock poisoned")
                .set_params(context, self.shader_params)
        })
    }

    /// Starts (or reverses) the reveal animation for this surface.
    pub fn set_reveal(&mut self, open: bool, reveal: RevealAnimation) {
        self.compositor
            .lock()
            .expect("D3D11 compositor lock poisoned")
            .set_reveal(open, reveal);
    }

    /// Hides the surface window outright. Used when parking a surface: even
    /// though the collapsed reveal renders fully transparent, the window can
    /// briefly (or permanently, if capture stalls) keep showing its last
    /// presented frame — hiding the window guarantees nothing lingers.
    pub fn hide_window(&self) -> Result<(), String> {
        hide_surface_window(self.hwnd)
    }

    /// Stops (or resumes) all rendering: paused surfaces present nothing.
    pub fn set_paused(&mut self, paused: bool) {
        self.compositor
            .lock()
            .expect("D3D11 compositor lock poisoned")
            .set_paused(paused);
    }

    /// Lets exactly one frame render, then pauses (capture stays warm).
    pub fn pause_after_next_present(&mut self) {
        self.compositor
            .lock()
            .expect("D3D11 compositor lock poisoned")
            .pause_after_next_present();
    }

    /// Renders the collapse animation to the end, then pauses.
    pub fn collapse_then_pause(&mut self) {
        self.compositor
            .lock()
            .expect("D3D11 compositor lock poisoned")
            .collapse_then_pause();
    }

    /// Caps the idle render rate in frames per second (`None` = uncapped).
    pub fn set_frame_cap(&mut self, fps: Option<f32>) {
        self.compositor
            .lock()
            .expect("D3D11 compositor lock poisoned")
            .set_min_present_interval(match fps {
                Some(fps) if fps > 0.0 => std::time::Duration::from_secs_f32(1.0 / fps),
                _ => std::time::Duration::ZERO,
            });
    }

    /// Enables adaptive pacing: content-driven frames render at the shared
    /// render loop's full rate while fresh capture frames keep arriving,
    /// then settle to the idle frame cap once the content goes quiet.
    pub fn set_adaptive_rate(&mut self, enabled: bool) {
        self.compositor
            .lock()
            .expect("D3D11 compositor lock poisoned")
            .set_adaptive_rate(enabled);
    }

    /// Whether at least one frame has been presented (the capture session is
    /// warm and the shader has produced output).
    pub fn has_presented(&mut self) -> bool {
        self.compositor
            .lock()
            .expect("D3D11 compositor lock poisoned")
            .has_presented
    }

    /// Sets the pill's uniform shape scale (1.0 = full size). Used by the
    /// host's auto-hide reveal to scale the pill in from slightly smaller.
    pub fn set_pill_scale(&mut self, scale: f32) {
        let _ = self.with_context(|context| {
            self.compositor
                .lock()
                .expect("D3D11 compositor lock poisoned")
                .set_pill_scale(context, scale);
            Ok(())
        });
    }

    /// Sets the pill's anisotropic squash/stretch factors (1.0 = undistorted),
    /// anchored to the pill's bottom-center. Used by the host's resize spring:
    /// width changes overshoot along x while y counter-squashes.
    pub fn set_pill_scale_xy(&mut self, scale_x: f32, scale_y: f32) {
        let _ = self.with_context(|context| {
            self.compositor
                .lock()
                .expect("D3D11 compositor lock poisoned")
                .set_pill_scale_xy(context, scale_x, scale_y);
            Ok(())
        });
    }

    /// Aligns the native surface with its host window while retaining its
    /// lower Z order. Call whenever the host moves or resizes.
    pub fn sync_bounds(&mut self, host: isize) -> Result<(), String> {
        if host != self.host {
            return Err("Host window handle changed; create a new GlassSurface".to_owned());
        }
        unsafe { stack_beneath_host(host, self.hwnd, &self.surface_anchor, self.style_host_window)? }
        let bounds = anchor_rect(overlay_bounds(HWND(host as *mut _))?, &self.surface_anchor);
        let primary_monitor_bounds = primary_monitor_bounds()?;
        self.surface_height_px = height(&bounds)? as f32;
        self.with_context(|context| {
            self.compositor
                .lock()
                .expect("D3D11 compositor lock poisoned")
                .resize(
                    context,
                    width(&bounds)?,
                    height(&bounds)?,
                    bounds,
                    primary_monitor_bounds,
                )
        })
    }

    /// Retargets the surface to a new anchor without recreating it — cheap
    /// for surfaces that follow moving or resizing elements.
    pub fn set_anchor(&mut self, anchor: SurfaceAnchor) -> Result<(), String> {
        self.surface_anchor = anchor;
        self.sync_bounds(self.host)
    }

    /// The surface window's raw handle. Callers that must move the window
    /// while another thread could be waiting on their locks should drop those
    /// locks first and use [`translate_surface_window`] — see the renderer's
    /// auto-hide slide path.
    pub fn hwnd_addr(&self) -> isize {
        self.hwnd.0 as isize
    }

    /// Shifts the surface window and its capture-sampling origin vertically
    /// by `dy` physical pixels. Used by the host's auto-hide slide: the shader
    /// refracts the desktop at the surface's own screen position, so the
    /// window and the source-rect constants must move together.
    pub fn translate(&mut self, dy: i32) -> Result<(), String> {
        if dy == 0 {
            return Ok(());
        }
        translate_surface_window(self.hwnd_addr(), dy)?;
        super::translate_compositor(&self.compositor, dy);
        Ok(())
    }
}

impl Drop for GlassSurface {
    fn drop(&mut self) {
        engine().unregister(self.engine_id);
        unsafe {
            let _ = DestroyWindow(self.hwnd);
        }
    }
}

/// Compensates texel-unit shader params for the downscaled shared capture:
/// blur, refraction and chromatic displacement are expressed in capture
/// texels, and a half-resolution capture has double-size texels.
fn scale_texel_params(
    mut params: GlassShaderParams,
    scale: f32,
) -> GlassShaderParams {
    if (scale - 1.0).abs() < f32::EPSILON {
        return params;
    }
    params.blur_radius_texels *= scale;
    params.refraction_strength *= scale;
    params.chromatic_offset_texels *= scale;
    params
}