//! Constructing a [`GlassSurface`]: the native surface window, the shared
//! compositor join and the one context helper every callback goes through.
use std::sync::{Arc, Mutex};

use windows::Win32::{
    Foundation::HWND,
    Graphics::Direct3D11::ID3D11DeviceContext,
    UI::WindowsAndMessaging::{DestroyWindow, SetWindowDisplayAffinity, WDA_EXCLUDEFROMCAPTURE},
};

use crate::surface::compositor::D3D11Compositor;
use crate::surface::geometry::{
    anchor_rect, height, overlay_bounds, primary_monitor_bounds, width, window_frame_rect,
};
use crate::surface::window::{
    create_surface_window,
    stack_above_host, stack_beneath_host,
};
use crate::surface::SharedCompositor;
use crate::backdrop::SurfaceRoute;
use crate::params::{
    BackdropSource, GlassConfig, MetaballField,
};
use crate::shared::engine;

use super::{GlassSurface, scale_texel_params};
impl GlassSurface {
    /// Creates the GPU surface for `host` (an HWND value) and joins the
    /// shared engine's feeds.
    pub fn create(host: isize, config: GlassConfig) -> Result<Self, String> {
        let shared = engine();
        let overlay = HWND(host as *mut _);
        let bounds = anchor_rect(overlay_bounds(overlay)?, &config.surface_anchor);
        // The backdrop's uv space: the monitor for Global glass, the host
        // window's own frame for Local glass.
        let backdrop_space = match config.backdrop_source {
            BackdropSource::Global => primary_monitor_bounds()?,
            BackdropSource::Local => window_frame_rect(host)?,
        };
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
                backdrop_space,
                config.backdrop_source == BackdropSource::Global,
                {
                    // The configured corner radius is expressed in surface
                    // pixels and translated into the SDF's normalized capsule
                    // radius. Texel-unit params are compensated for the
                    // downscaled shared capture here, so callers keep passing
                    // full-resolution values.
                    let mut shader_params =
                        scale_texel_params(config.shader, shared.capture_scale());
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
            shape: config.shape,
            shape_region: [0.0, 0.0, 1.0, 1.0],
            shape_rect_px: None,
            metaballs: MetaballField::default(),
            metaball_own_index: None,
            surface_anchor: config.surface_anchor,
            surface_width_px: width(&bounds)? as f32,
            surface_height_px: height(&bounds)? as f32,
            style_host_window: config.style_host_window,
            backdrop_source: config.backdrop_source,
            placement: config.placement,
        };
        unsafe {
            // Display affinity keeps the GPU surface itself out of every
            // screen capture, including the shared WGC feed: without it the
            // glass would refract its own rendered pixels (a feedback
            // loop). Local glass can opt out — its per-window capture
            // excludes overlapping windows on its own.
            if config.exclude_from_capture {
                SetWindowDisplayAffinity(surface.hwnd, WDA_EXCLUDEFROMCAPTURE).map_err(
                    |error| format!("Could not exclude the GPU surface from capture: {error}"),
                )?;
            }
            match surface.placement {
                crate::params::SurfacePlacement::BeneathHost => {
                    stack_beneath_host(
                        surface.host,
                        surface.hwnd,
                        &surface.surface_anchor,
                        surface.style_host_window,
                    )?;
                }
                crate::params::SurfacePlacement::AboveHost => {
                    stack_above_host(
                        surface.host,
                        surface.hwnd,
                        &surface.surface_anchor,
                        surface.style_host_window,
                    )?;
                }
            }
        }
        // Initialize the swap chain to fully transparent: the first real
        // frame can be arbitrarily late and an uninitialized buffer renders
        // as a black box.
        GlassSurface::with_context(|context| {
            surface
                .compositor
                .lock()
                .expect("D3D11 compositor lock poisoned")
                .clear_to_transparent(context)
        })?;
        surface.engine_id = shared.register(
            Arc::clone(&surface.compositor),
            match surface.backdrop_source {
                BackdropSource::Global => SurfaceRoute::global(),
                BackdropSource::Local => SurfaceRoute::local(host),
            },
        );
        Ok(surface)
    }

    /// Runs a closure with the shared immediate context while holding the
    /// engine lock — every D3D11 touch outside the render loop must be
    /// serialized this way (the immediate context is not thread-safe).
    /// Static so the closure can freely borrow the surface it works on.
    pub(super) fn with_context<T>(
        f: impl FnOnce(&ID3D11DeviceContext) -> Result<T, String>,
    ) -> Result<T, String> {
        let shared = engine();
        let state = shared.state.lock().expect("shared glass lock poisoned");
        f(&state.context)
    }

    pub fn compositor(&self) -> SharedCompositor {
        Arc::clone(&self.compositor)
    }

}
