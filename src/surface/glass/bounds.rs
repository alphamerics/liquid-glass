//! Keeping the native surface aligned with its host window: bounds sync,
//! anchors, and the translation path the window manager drives.


use windows::Win32::Foundation::HWND;

use crate::surface::geometry::{
    anchor_rect, height, overlay_bounds, primary_monitor_bounds, width, window_frame_rect,
};
use crate::surface::window::{
    move_above_host, move_beneath_host,
    stack_above_host, stack_beneath_host, translate_surface_window,
};
use crate::params::{
    BackdropSource,
    SurfaceAnchor,
};

use super::GlassSurface;
impl GlassSurface {
    /// Aligns the native surface with its host window (keeping its
    /// placement's Z relation). Call whenever the host moves or resizes —
    /// for a Local surface this also re-maps the samples into the host's
    /// fresh window frame.
    pub fn sync_bounds(&mut self, host: isize) -> Result<(), String> {
        if host != self.host {
            return Err("Host window handle changed; create a new GlassSurface".to_owned());
        }
        unsafe {
            match self.placement {
                crate::params::SurfacePlacement::BeneathHost => stack_beneath_host(
                    host,
                    self.hwnd,
                    &self.surface_anchor,
                    self.style_host_window,
                )?,
                crate::params::SurfacePlacement::AboveHost => stack_above_host(
                    host,
                    self.hwnd,
                    &self.surface_anchor,
                    self.style_host_window,
                )?,
            }
        }
        self.sync_geometry(host)
    }

    /// Re-anchors the surface to `anchor` while KEEPING its z slot: the
    /// window moves/resizes but never re-stacks. Per-frame re-anchors — a
    /// dragged metaball following its pointer — need this: the full re-pin
    /// ([`GlassSurface::sync_bounds`]) inserts the surface at the bottom of
    /// the host's surface stack, which buries the moved glass beneath
    /// whatever else the host owns mid-drag. Z repair is the host-event
    /// sync path's job (which re-pins every surface in a deliberate order).
    pub fn set_anchor_keep_z(&mut self, anchor: SurfaceAnchor) -> Result<(), String> {
        self.surface_anchor = anchor;
        unsafe {
            match self.placement {
                crate::params::SurfacePlacement::BeneathHost => {
                    move_beneath_host(self.host, self.hwnd, &self.surface_anchor)?
                }
                crate::params::SurfacePlacement::AboveHost => {
                    move_above_host(self.host, self.hwnd, &self.surface_anchor)?
                }
            }
        }
        self.sync_geometry(self.host)
    }

    /// Re-derives the backdrop mapping and swap-chain size from the stored
    /// anchor against the host's CURRENT bounds — the half of a bounds sync
    /// that does not touch the window's z order.
    fn sync_geometry(&mut self, host: isize) -> Result<(), String> {
        let bounds = anchor_rect(overlay_bounds(HWND(host as *mut _))?, &self.surface_anchor);
        // The backdrop's uv space follows the source: the monitor frame for
        // Global glass, the host window's own frame for Local glass.
        let backdrop_space = match self.backdrop_source {
            BackdropSource::Global => primary_monitor_bounds()?,
            BackdropSource::Local => window_frame_rect(host)?,
        };
        self.surface_width_px = width(&bounds)? as f32;
        self.surface_height_px = height(&bounds)? as f32;
        Self::with_context(|context| {
            self.compositor
                .lock()
                .expect("D3D11 compositor lock poisoned")
                .resize(
                    context,
                    width(&bounds)?,
                    height(&bounds)?,
                    bounds,
                    backdrop_space,
                )
        })
    }

    /// Retargets the surface to a new anchor without recreating it — cheap
    /// for surfaces that follow moving or resizing elements.
    pub fn set_anchor(&mut self, anchor: SurfaceAnchor) -> Result<(), String> {
        self.surface_anchor = anchor;
        self.sync_bounds(self.host)
    }

    /// Stores a new anchor WITHOUT re-syncing the surface. Use when a host
    /// window resize lands immediately afterwards: the resize's own sync
    /// pass then applies the new anchor against the fresh bounds in the
    /// same window operation, instead of the old anchor against the new
    /// bounds (a one-frame wrong-geometry flash).
    pub fn set_anchor_quiet(&mut self, anchor: SurfaceAnchor) {
        self.surface_anchor = anchor;
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
        crate::surface::translate_compositor(&self.compositor, dy);
        Ok(())
    }
}
