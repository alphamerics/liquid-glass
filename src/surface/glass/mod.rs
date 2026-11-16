use windows::Win32::{Foundation::HWND, UI::WindowsAndMessaging::DestroyWindow};

use super::SharedCompositor;
use crate::params::{BackdropSource, GlassShaderParams, GlassShape, MetaballField, SurfaceAnchor};
use crate::shared::engine;

/// DXGI D3D11 presentation target positioned beneath (or above) a host
/// window. Create one per glass element with [`GlassSurface::create`]; keep
/// it alive while the glass should render. Dropping it stops the window's
/// capture (if Local) and destroys the native surface window, and the
/// GLOBAL monitor capture with the last Global surface's drop.
///
/// All surfaces share one D3D11 device and one render loop (see the shared
/// engine). Each surface is fed by its declared backdrop source
/// ([`GlassConfig::backdrop_source`]): Global surfaces sample the shared
/// monitor capture; Local surfaces a per-window capture of their host app.
pub struct GlassSurface {
    host: isize,
    hwnd: HWND,
    engine_id: u64,
    compositor: SharedCompositor,
    shader_params: GlassShaderParams,
    corner_radius_px: f32,
    /// The surface's own shape preset (the shape engine's rounded rect /
    /// superellipse). Drives the shader's silhouette, refraction normals
    /// and bevel.
    shape: GlassShape,
    /// The shape region: the rect (in surface uv, origin top-left) the
    /// shape field lives in. Identity spans the whole surface.
    shape_region: [f32; 4],
    /// The drawn shape's rect in surface-relative physical pixels
    /// (x, y, width, height) when the shape is driven inside a fixed-size
    /// canvas ([`GlassSurface::set_shape_rect_px`]); `None` keeps the
    /// identity (the shape spans the whole surface).
    shape_rect_px: Option<[f32; 4]>,
    /// The metaball field replacing the drawn shape when active (a count
    /// of 0 keeps the surface's own shape preset).
    metaballs: MetaballField,
    /// The surface's own ball inside a SHARED field (see
    /// [`GlassSurface::set_metaballs_field`]); `None` renders the whole
    /// field.
    metaball_own_index: Option<usize>,
    surface_anchor: SurfaceAnchor,
    surface_width_px: f32,
    surface_height_px: f32,
    style_host_window: bool,
    backdrop_source: BackdropSource,
    placement: crate::params::SurfacePlacement,
}

unsafe impl Send for GlassSurface {}

/// Folds each active ball's weight into its influence radius and silhouette
/// extents, and splits the field into the shader's two stamped row sets:
/// the potential rows (`(cx, cy, effectiveRadius, weight)`, slot 3 inert
/// layout parity) and the shape rows (`(exponent, extentX, extentY, 0)`).
/// The field's contribution uses the effective radius
/// `min(extents) * weight^(1/tension)` and the folded extents — per-ball
/// CONSTANTS that would otherwise be recomputed per pixel inside the
/// shader's ball loop. An exponent-2 ball with equal extents stamps shape
/// rows the shader dispatches through its circle fast path; a shaped ball
/// stamps its superellipse (exponent 2 with unequal extents is an ellipse).
/// Inactive rows stamp zero (never read — the ball count bounds the loop).
pub(crate) fn metaball_effective_rows(
    balls: [[f32; 6]; 6],
    count: usize,
    tension: f32,
) -> ([[f32; 4]; 6], [[f32; 4]; 6]) {
    let inv_tension = 1.0 / tension.max(1.0);
    let mut rows = [[0.0; 4]; 6];
    let mut shapes = [[0.0; 4]; 6];
    for (index, ball) in balls.iter().enumerate().take(count) {
        let fold = ball[4].max(0.0).powf(inv_tension);
        let extent_x = ball[2] * fold;
        let extent_y = ball[3] * fold;
        rows[index] = [ball[0], ball[1], extent_x.min(extent_y), ball[4]];
        shapes[index] = [ball[5], extent_x, extent_y, 0.0];
    }
    (rows, shapes)
}

mod bounds;
mod create;
mod pill;
mod present;
mod shape;

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
fn scale_texel_params(mut params: GlassShaderParams, scale: f32) -> GlassShaderParams {
    if (scale - 1.0).abs() < f32::EPSILON {
        return params;
    }
    params.blur_radius_texels *= scale;
    params.refraction_strength *= scale;
    params.chromatic_offset_texels *= scale;
    params
}
