//! The surface's own visual identity: shader params, corner radius, the
//! shape preset and region, shape-rect canvases, metaball fields, and the
//! params stamping/uploading all of that rides on.

use std::time::Duration;

use crate::params::{GlassShaderParams, GlassShape, MetaballField};
use crate::shared::engine;
use crate::surface::D3D11Compositor;

use super::{metaball_effective_rows, scale_texel_params, GlassSurface};
impl GlassSurface {
    /// Replaces every shader float live; takes effect on the next frame.
    /// The surface's own shape stamp (shape region + shape preset) rides on
    /// top, so a settings push never clobbers the surface's shape state.
    pub fn set_shader_params(&mut self, mut params: GlassShaderParams) -> Result<(), String> {
        params = scale_texel_params(params, engine().capture_scale());
        self.shader_params = params;
        self.upload_params()
    }
    /// Like [`GlassSurface::set_shader_params`], but keeps the surface's own corner radius:
    /// shared settings applied to several surfaces won't distort their shapes.
    pub fn set_shader_params_keeping_corner(
        &mut self,
        mut params: GlassShaderParams,
    ) -> Result<(), String> {
        params.capsule_radius =
            (self.corner_radius_px / self.shape_height_px().max(1.0)).clamp(0.0, 0.5);
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
        Self::with_context(|context| {
            self.compositor
                .lock()
                .expect("D3D11 compositor lock poisoned")
                .backbuffer_mean(context)
        })
        .ok()
    }

    /// Changes the pill's corner radius in surface pixels. Any radius is
    /// allowed, from square corners (`0.0`) to a full capsule (half the
    /// shape's height). The radius lives in the shader's SDF, so edges stay
    /// antialiased through the composition alpha.
    pub fn set_corner_radius(&mut self, corner_radius_px: f32) -> Result<(), String> {
        self.corner_radius_px = corner_radius_px;
        let radius_units = (corner_radius_px / self.shape_height_px().max(1.0)).clamp(0.0, 0.5);
        self.shader_params.capsule_radius = radius_units;
        self.upload_params()
    }

    /// The height (physical pixels) the shape's units normalize against:
    /// the drawn shape rect's height when a shape rect is set (canvas
    /// mode), the surface window's own height otherwise.
    pub(super) fn shape_height_px(&self) -> f32 {
        self.shape_rect_px
            .map(|rect| rect[3])
            .unwrap_or(self.surface_height_px)
    }

    /// Clamps a shape rect against the canvas and stores it (plus the
    /// corner radius and its normalized capsule ratio) without uploading —
    /// the shared body of [`Self::set_shape_rect_px`] and
    /// [`Self::set_shape_rect_px_animated`].
    pub(super) fn store_shape_rect(&mut self, rect: [f32; 4], corner_radius_px: f32) {
        let (x, y, w, h) = (rect[0], rect[1], rect[2].max(1.0), rect[3].max(1.0));
        // The shape can never exceed the canvas it is drawn on.
        let w = w.min(self.surface_width_px.max(1.0));
        let h = h.min(self.surface_height_px.max(1.0));
        let x = x.clamp(0.0, (self.surface_width_px - w).max(0.0));
        let y = y.clamp(0.0, (self.surface_height_px - h).max(0.0));
        self.shape_rect_px = Some([x, y, w, h]);
        self.shape_region = [
            x / self.surface_width_px.max(1.0),
            y / self.surface_height_px.max(1.0),
            w / self.surface_width_px.max(1.0),
            h / self.surface_height_px.max(1.0),
        ];
        self.corner_radius_px = corner_radius_px;
        self.shader_params.capsule_radius = (corner_radius_px / h.max(1.0)).clamp(0.0, 0.5);
    }

    /// Moves and resizes the drawn shape WITHIN the surface window, in
    /// surface-relative physical pixels (x, y, width, height), and sets the
    /// pill's corner radius in the same step (normalized against the rect's
    /// height). The surface window and its swap chain are NOT touched: the
    /// resize lands as one constant-buffer update on the next frame, so the
    /// shape can be retargeted every animation tick and resizes play back
    /// at the render loop's full rate — no window resizes, no swap-chain
    /// reallocation, no transparent flash.
    ///
    /// This is the smooth counterpart to resizing the surface itself (see
    /// [`GlassSurface::sync_bounds`], which must reallocate the swap chain's
    /// buffers): create the surface once at a fixed, generous canvas (a
    /// [`SurfaceAnchor::Rect`] anchor with headroom) and drive the drawn
    /// shape inside it.
    pub fn set_shape_rect_px(
        &mut self,
        rect: [f32; 4],
        corner_radius_px: f32,
    ) -> Result<(), String> {
        // An instant set is the shape's truth now: any running shape
        // animation targets a rect that no longer exists, so it is
        // cancelled and the transform snaps back to its identity.
        self.compositor
            .lock()
            .expect("D3D11 compositor lock poisoned")
            .cancel_shape_transform();
        self.store_shape_rect(rect, corner_radius_px);
        self.upload_params()
    }

    /// The animated counterpart to [`GlassSurface::set_shape_rect_px`]:
    /// the drawn shape GLIDES from wherever it visually is right now into
    /// the new rect, integrated by the compositor at the render loop's rate
    /// (a constant-buffer update per frame — no window operations, no
    /// swap-chain resizes). Consecutive calls retarget the pursuit: each
    /// step recomputes the spring's start from the live visual rect and
    /// carries the velocity over, so a drag's step stream reads as one
    /// smooth continuous motion.
    ///
    /// The spring constants follow the shared glass spring convention
    /// (ω = 2π/duration, ζ = 1 − bounce·0.75); the scale axes ceiling-clamp
    /// at the target so the drawn shape can never overshoot its canvas.
    /// At rest the animation's transform is the identity and the stored
    /// shape rect IS the drawn shape, exactly like the instant path.
    pub fn set_shape_rect_px_animated(
        &mut self,
        rect: [f32; 4],
        corner_radius_px: f32,
        omega: f64,
        zeta: f64,
        restitution: f32,
    ) -> Result<(), String> {
        self.retarget_shape_rect(rect, corner_radius_px, |compositor, initial| {
            compositor.begin_shape_transform(initial, omega, zeta, restitution);
        })
    }

    /// The bezier-eased counterpart to
    /// [`GlassSurface::set_shape_rect_px_animated`]: the drawn shape glides
    /// into the new rect paced by a CSS cubic-bezier curve
    /// (`[x1, y1, x2, y2]` control points) over `duration_secs`, after
    /// `delay_secs`. Deterministic endpoints instead of a spring: the
    /// motion lands exactly on the target when the clock runs out, and
    /// mid-flight retargets restart from the live visual value (CSS
    /// transition semantics) with no velocity carry-over.
    pub fn set_shape_rect_px_eased(
        &mut self,
        rect: [f32; 4],
        corner_radius_px: f32,
        duration_secs: f32,
        easing: [f32; 4],
        delay_secs: f32,
    ) -> Result<(), String> {
        self.retarget_shape_rect(rect, corner_radius_px, |compositor, initial| {
            compositor.begin_shape_tween(
                initial,
                [1.0, 1.0, 0.0, 0.0],
                Duration::from_secs_f32(delay_secs.max(0.0)),
                Duration::from_secs_f32(duration_secs.max(1e-3)),
                easing,
            );
        })
    }

    /// Sets the SHAPE TRANSFORM instantly, with no animation: the drawn
    /// shape sits exactly at `(scale_x, scale_y, offset_x, offset_y)`
    /// (shape-region units, bottom-center anchor; `(1, 1, 0, 0)` is the
    /// identity) from the next frame on. Cancels any running shape
    /// animation. The instant counterpart of
    /// [`GlassSurface::set_shape_transform_eased`].
    pub fn set_shape_transform(&mut self, value: [f32; 4]) -> Result<(), String> {
        Self::with_context(|context| {
            let mut compositor = self
                .compositor
                .lock()
                .expect("D3D11 compositor lock poisoned");
            compositor.set_shape_transform_direct(value);
            let mut params = self.shader_params;
            self.stamp_shape(&mut params);
            compositor.set_params(context, params)
        })
    }

    /// Animates the SHAPE TRANSFORM itself to an arbitrary target — the
    /// CSS `transform` + `transition` model: the drawn shape eases from
    /// wherever it visually is into `(scale_x, scale_y, offset_x,
    /// offset_y)` (shape-region units, anchored to the shape's
    /// bottom-center; `(1, 1, 0, 0)` is the identity) over `duration_secs`
    /// paced by the cubic-bezier control points `easing` (`[x1, y1, x2,
    /// y2]`), after `delay_secs`. The target PERSISTS: at rest the
    /// transform holds `to`, unlike the rect-glide variants whose
    /// animations dissolve into the identity. Retargets restart from the
    /// live value; a running spring is replaced.
    pub fn set_shape_transform_eased(
        &mut self,
        to: [f32; 4],
        duration_secs: f32,
        easing: [f32; 4],
        delay_secs: f32,
    ) -> Result<(), String> {
        Self::with_context(|context| {
            let mut compositor = self
                .compositor
                .lock()
                .expect("D3D11 compositor lock poisoned");
            // CSS retarget semantics: the tween starts from wherever the
            // transform stands right now (the previous animation's live
            // value, or the identity at rest).
            let from = compositor.shape_transform();
            compositor.begin_shape_tween(
                from,
                to,
                Duration::from_secs_f32(delay_secs.max(0.0)),
                Duration::from_secs_f32(duration_secs.max(1e-3)),
                easing,
            );
            // The transform is compositor-owned (stamped into every
            // upload), so the params re-push only re-arms the surface's
            // dirty state; the tween's own steps carry the values from
            // here on.
            let mut params = self.shader_params;
            self.stamp_shape(&mut params);
            compositor.set_params(context, params)
        })
    }

    /// The shared body of the two animated rect setters: stores the target
    /// rect (clamped exactly like the instant path), recomputes the
    /// current visual rect's initial conditions relative to the NEW region
    /// and hands them to `begin` on the compositor (the spring's or the
    /// tween's start).
    fn retarget_shape_rect(
        &mut self,
        rect: [f32; 4],
        corner_radius_px: f32,
        begin: impl FnOnce(&mut D3D11Compositor, [f32; 4]),
    ) -> Result<(), String> {
        let surface_w = self.surface_width_px.max(1.0);
        let surface_h = self.surface_height_px.max(1.0);
        let previous = self
            .shape_rect_px
            .unwrap_or([0.0, 0.0, surface_w, surface_h]);
        Self::with_context(|context| {
            // The current VISUAL rect: the stored shape rect with the live
            // transform applied. The transform's anchor is the shape's
            // bottom-center (in region units: the region center offset by
            // (0, +0.5); one unit = the region's height in px), displaced
            // by the transform's offset. The ENGINE lock held across this
            // whole closure blocks the render loop's presents, so the read,
            // the store and the retarget below compose into one atomic
            // geometry step even though the compositor guard is scoped.
            let transform = {
                self.compositor
                    .lock()
                    .expect("D3D11 compositor lock poisoned")
                    .shape_transform()
            };
            let visual_w = previous[2] * transform[0];
            let visual_h = previous[3] * transform[1];
            let anchor_x = previous[0] + previous[2] / 2.0 + transform[2] * previous[3];
            let anchor_y = previous[1] + previous[3] + transform[3] * previous[3];
            // Store the target rect (clamped exactly like the instant path).
            self.store_shape_rect(rect, corner_radius_px);
            let target = self.shape_rect_px.expect("shape rect was just stored");
            // The animation's start, expressed relative to the NEW region:
            // the visual size as a fraction of the target's, and the anchor
            // displacement in target-region units.
            let region_h = target[3].max(1.0);
            let initial = [
                visual_w / target[2].max(1.0),
                visual_h / region_h,
                (anchor_x - (target[0] + target[2] / 2.0)) / region_h,
                (anchor_y - (target[1] + target[3] / 2.0)) / region_h - 0.5,
            ];
            let mut compositor = self
                .compositor
                .lock()
                .expect("D3D11 compositor lock poisoned");
            begin(&mut compositor, initial);
            // The params carry the new shape region; the compositor stamps
            // the just-set transform over its stored value, so the upload
            // and the animation start agree exactly.
            let mut params = self.shader_params;
            self.stamp_shape(&mut params);
            compositor.set_params(context, params)
        })
    }

    /// Changes the surface's own shape preset (the shape engine's rounded
    /// rect / superellipse). The rounded kind keeps the surface's corner
    /// radius; the superellipse's exponent comes from the preset.
    pub fn set_shape(&mut self, shape: GlassShape) -> Result<(), String> {
        if shape == self.shape {
            return Ok(());
        }
        self.shape = shape;
        self.upload_params()
    }

    /// Toggles the surface's drawn shape to (or from) a liquid metaball
    /// field. Each ball is `(cx, cy, extentX, extentY, weight, exponent)` in
    /// shape-region units — the same space [`GlassShaderParams::reveal_origin`]
    /// uses (origin at the region's center, +y down, one unit = the region's
    /// height). Exponent 2 with equal extents is the classic circle ball;
    /// any other exponent or unequal extents shapes the ball's silhouette
    /// (an ellipse, squircle or box-like superellipse that blends into the
    /// field exactly like a circle). An empty slice turns the field off and
    /// restores the surface's own shape preset.
    ///
    /// The field REPLACES the drawn silhouette while active: only surfaces
    /// running their own metaball fields look blob-like; a plain shape and
    /// a field never merge across surfaces (every surface evaluates its
    /// own field in its own region). All downstream effects — edge
    /// refraction, chromatic dispersion, bevel and rim light — follow the
    /// blended field, so the liquid necks refract like the rest of the
    /// silhouette. The shape region bounds the field, so give it headroom
    /// when balls sit near (or beyond) its edges.
    ///
    /// `tension` is the field's exponent: 2 is the classic metaball blend,
    /// higher values merge tighter (rounder necks), lower values stretch
    /// merges into long threads. This is the SINGLE-SURFACE form: the whole
    /// field renders. For a surface that shares the field with other
    /// metaball surfaces, use [`GlassSurface::set_metaballs_field`], which
    /// also tiles the merged blob across the surfaces.
    pub fn set_metaballs(&mut self, balls: &[[f32; 6]], tension: f32) -> Result<(), String> {
        self.set_metaballs_field(balls, tension, None)
    }

    /// Like [`GlassSurface::set_metaballs`], but marks the surface as ONE
    /// container inside a field SHARED with other metaball surfaces:
    /// `own_index` selects this surface's own ball, and the field renders
    /// only where that ball's contribution dominates the FOREIGN balls
    /// (index > own index) — the surfaces tile the merged blob exactly,
    /// without double-compositing their canvases' overlap into
    /// straight-edge seams. `None` renders the whole field (single-surface
    /// form; a foreign-dominated pixel would otherwise go dark).
    pub fn set_metaballs_field(
        &mut self,
        balls: &[[f32; 6]],
        tension: f32,
        own_index: Option<usize>,
    ) -> Result<(), String> {
        let mut field = MetaballField {
            tension: tension.max(1.0),
            ..Default::default()
        };
        for ball in balls.iter().take(field.balls.len()) {
            // Degenerate balls (no extent or no weight) never contribute to
            // the potential; skip them instead of counting hollow balls.
            if ball[2] <= 0.0 || ball[3] <= 0.0 || ball[4] <= 0.0 {
                continue;
            }
            field.balls[field.count] = *ball;
            field.count += 1;
        }
        // The own index must point INTO the field for the clip to mean
        // anything; a stale index (ball list changed) disables the clip
        // rather than blanking the surface.
        let own_index = own_index.filter(|own| *own < field.count);
        if field == self.metaballs && own_index == self.metaball_own_index {
            return Ok(());
        }
        self.metaballs = field;
        self.metaball_own_index = own_index;
        self.upload_params()
    }

    /// Like [`GlassSurface::set_metaballs`], but the balls arrive in
    /// surface-relative physical pixels — `(x, y, extentX, extentY, weight,
    /// exponent)`, origin top-left — the natural unit for hosts that
    /// position balls from DOM geometry. Converted into shape-region units
    /// through the active shape rect (the whole surface when it is the
    /// identity).
    pub fn set_metaballs_px(&mut self, balls_px: &[[f32; 6]], tension: f32) -> Result<(), String> {
        self.set_metaballs_field_px(balls_px, tension, None)
    }

    /// The shared-field + pixel-units form of the metaball setters: see
    /// [`GlassSurface::set_metaballs_field`] and
    /// [`GlassSurface::set_metaballs_px`].
    pub fn set_metaballs_field_px(
        &mut self,
        balls_px: &[[f32; 6]],
        tension: f32,
        own_index: Option<usize>,
    ) -> Result<(), String> {
        let (region_x, region_y, region_w, region_h) = match self.shape_rect_px {
            Some(rect) => (rect[0], rect[1], rect[2], rect[3]),
            None => (0.0, 0.0, self.surface_width_px, self.surface_height_px),
        };
        let region_h = region_h.max(1.0);
        let center_x = region_x + region_w * 0.5;
        let center_y = region_y + region_h * 0.5;
        let balls: Vec<[f32; 6]> = balls_px
            .iter()
            .map(|ball| {
                [
                    (ball[0] - center_x) / region_h,
                    (ball[1] - center_y) / region_h,
                    ball[2] / region_h,
                    ball[3] / region_h,
                    ball[4],
                    ball[5],
                ]
            })
            .collect();
        self.set_metaballs_field(&balls, tension, own_index)
    }

    /// Uploads the stored shader params (the compositor no-ops when nothing
    /// changed).
    pub(super) fn upload_params(&mut self) -> Result<(), String> {
        let mut params = self.shader_params;
        self.stamp_shape(&mut params);
        Self::with_context(|context| {
            self.compositor
                .lock()
                .expect("D3D11 compositor lock poisoned")
                .set_params(context, params)
        })
    }

    /// Writes the surface's shape state into a params copy (the cbuffer's
    /// shape rows: the shape region, the own shape preset and the metaball
    /// field). An active metaball field replaces the preset: kind 2, with
    /// the ball count and tension riding in the meta's spare slots and the
    /// potential + shape rows stamped by [`metaball_effective_rows`] (each
    /// ball's weight pre-folded into its effective radius and silhouette
    /// extents).
    pub(super) fn stamp_shape(&self, params: &mut GlassShaderParams) {
        params.shape_region = self.shape_region;
        params.shape_meta = if self.metaballs.count > 0 {
            [
                2.0,
                2.0,
                self.metaballs.count as f32,
                self.metaballs.tension,
            ]
        } else {
            match self.shape {
                GlassShape::Rounded => [0.0, 2.0, 0.0, 0.0],
                GlassShape::Superellipse { exponent } => [1.0, exponent.max(1.0), 0.0, 0.0],
            }
        };
        let (rows, shapes) = metaball_effective_rows(
            self.metaballs.balls,
            self.metaballs.count,
            self.metaballs.tension,
        );
        params.metaballs = rows;
        params.metaball_shapes = shapes;
        // The territory clip arms only when there are foreign balls after
        // the own one (a single-ball or whole-field surface renders all).
        params.metaball_own = match self.metaball_own_index {
            Some(own) if own + 1 < self.metaballs.count => [own as f32, 1.0, (own + 1) as f32, 0.0],
            _ => [0.0; 4],
        };
    }
}
