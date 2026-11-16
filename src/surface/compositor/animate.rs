//! Everything the compositor ANIMATES on its own: the reveal progress, the
//! spring-integrated pill squash/stretch and shape transform, the pointer
//! velocity input and the params stamping/uploading all of it rides on.
use super::super::geometry::{primary_monitor_bounds, source_rect_constants};
use super::easing::{CubicBezier, ShapeTween};
use super::spring::TransformSpring;
use crate::params::{GlassShaderParams, RevealAnimation};
use crate::SourceRectConstants;
use std::time::{Duration, Instant};
use windows::Win32::Foundation::POINT;
use windows::Win32::Graphics::Direct3D11::ID3D11DeviceContext;
use windows::Win32::UI::WindowsAndMessaging::GetCursorPos;

use super::D3D11Compositor;

impl D3D11Compositor {
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

    pub(super) fn update_reveal(&mut self, context: &ID3D11DeviceContext) {
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
        let params = self.current_params();
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

    /// Builds the params copy for a cbuffer upload: the stored params plus
    /// everything the compositor owns (reveal progress, animation time and
    /// the stamped springs' live values).
    fn current_params(&self) -> GlassShaderParams {
        let mut params = self.glass_params_current;
        params.reveal_progress = self.reveal_current;
        params.animation_time = self.animation_time;
        self.stamp_compositor_owned(&mut params);
        params
    }

    /// Uploads the current params (the stored ones with every
    /// compositor-owned value stamped in).
    fn upload_compositor_params(&mut self, context: &ID3D11DeviceContext) {
        let params = self.current_params();
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

    /// Integrates the compositor-owned animations (the pill squash/stretch
    /// spring, the shape transform's pursuit spring and its bezier tween)
    /// with the frame's real dt, then uploads the params once when any
    /// value moved this frame. Runs every present; with no active
    /// animation it only advances the shared wall clock.
    pub(super) fn update_springs(&mut self, context: &ID3D11DeviceContext) {
        let now = Instant::now();
        let dt = now
            .duration_since(self.spring_last_frame)
            .as_secs_f32()
            .min(0.05);
        self.spring_last_frame = now;
        let mut changed = false;
        if let Some(spring) = &mut self.pill_spring {
            let (previous_x, previous_y) = (self.pill_scale_x, self.pill_scale_y);
            let settled = spring.step(dt);
            self.pill_scale_x = spring.value[0];
            self.pill_scale_y = spring.value[1];
            if settled {
                self.pill_spring = None;
            }
            changed |= (self.pill_scale_x - previous_x).abs() > 1e-5
                || (self.pill_scale_y - previous_y).abs() > 1e-5;
        }
        if let Some(spring) = &mut self.shape_spring {
            let previous = self.shape_transform;
            let settled = spring.step(dt);
            self.shape_transform = spring.value;
            if settled {
                self.shape_spring = None;
            }
            changed |= self
                .shape_transform
                .iter()
                .zip(previous.iter())
                .any(|(a, b)| (a - b).abs() > 1e-5);
        }
        // The time-based tween shares the spring's upload budget: both
        // integrate against the same frame clock and stamp the same
        // compositor-owned transform.
        if let Some(tween) = &self.shape_tween {
            let previous = self.shape_transform;
            self.shape_transform = tween.value_at(now);
            if tween.progress(now) >= 1.0 {
                self.shape_tween = None;
            }
            changed |= self
                .shape_transform
                .iter()
                .zip(previous.iter())
                .any(|(a, b)| (a - b).abs() > 1e-5);
        }
        if changed {
            self.upload_compositor_params(context);
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
        self.stamp_compositor_owned(&mut incoming);
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

    pub(super) fn update_pointer_velocity(&mut self, context: &ID3D11DeviceContext) {
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
        self.stamp_compositor_owned(&mut params);
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
    /// re-uploading the params only when the values change. A direct set
    /// overrides the running squash spring (it is cancelled — the values are
    /// the caller's truth until the next kick).
    pub fn set_pill_scale_xy(&mut self, context: &ID3D11DeviceContext, scale_x: f32, scale_y: f32) {
        if self.pill_spring.is_some() {
            self.pill_spring = None;
        }
        if (self.pill_scale_x - scale_x).abs() < 0.001
            && (self.pill_scale_y - scale_y).abs() < 0.001
        {
            return;
        }
        self.pill_scale_x = scale_x;
        self.pill_scale_y = scale_y;
        let mut params = self.glass_params_current;
        self.stamp_compositor_owned(&mut params);
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

    /// The compositor-owned shape transform's current value. Hosts read it
    /// to compute a continuous retarget's initial conditions (see
    /// [`GlassSurface::set_shape_rect_px_animated`]).
    pub fn shape_transform(&self) -> [f32; 4] {
        self.shape_transform
    }

    /// Starts (or retargets) the shape transform's spring: `initial`
    /// becomes the transform's live value — expressed relative to the shape
    /// region applied in the same step — and the spring pursues the
    /// identity, so the drawn shape glides from its previous visual rect
    /// into the new one. A running spring's velocity carries over, rescaled
    /// by the value remap, so a stream of retargets (a drag's step sequence)
    /// pursues smoothly instead of restarting from rest each step.
    pub fn begin_shape_transform(
        &mut self,
        initial: [f32; 4],
        omega: f64,
        zeta: f64,
        restitution: f32,
    ) {
        // At most one shape animation runs at a time: a spring start
        // replaces a running tween (the spring is the stateful pursuit;
        // the tween the deterministic glide).
        self.shape_tween = None;
        let mut velocity = [0.0f32; 4];
        if let Some(spring) = &self.shape_spring {
            for axis in 0..4 {
                let previous = spring.value[axis];
                velocity[axis] = if previous.abs() > 1e-3 {
                    spring.velocity[axis] * (initial[axis] / previous)
                } else {
                    0.0
                };
            }
        }
        self.shape_transform = initial;
        self.shape_spring = Some(TransformSpring::new(
            initial,
            velocity,
            // The spring always pursues the identity: the applied shape
            // region IS the target, and the transform dissolves into it.
            [1.0, 1.0, 0.0, 0.0],
            // Scale axes ceiling-clamp at the target (overshoot past the
            // drawn shape's rect would clip at the canvas edge); offsets
            // overshoot freely.
            2,
            omega,
            zeta,
            restitution,
        ));
    }

    /// Starts (or retargets) the shape transform's TIME-BASED tween: the
    /// transform glides from `from` to `to` over `duration` after `delay`,
    /// paced by a cubic-bezier curve (`easing` = the `[x1, y1, x2, y2]`
    /// control points) — CSS `transition` semantics. `from` is normally
    /// the live transform (a retarget mid-flight restarts from wherever
    /// the previous animation stands, so a stream of targets reads as one
    /// continuous motion); a running spring is replaced.
    pub fn begin_shape_tween(
        &mut self,
        from: [f32; 4],
        to: [f32; 4],
        delay: Duration,
        duration: Duration,
        easing: [f32; 4],
    ) {
        // At most one shape animation runs at a time: a tween start
        // replaces a running spring.
        self.shape_spring = None;
        self.shape_tween = Some(ShapeTween::new(
            from,
            to,
            delay,
            duration,
            CubicBezier::from_points(easing),
        ));
    }

    /// Cancels the shape transform's spring and snaps the transform to its
    /// identity: the drawn shape sits exactly where the shape region puts
    /// it (the instant-resize path). A running tween is cancelled too.
    pub fn cancel_shape_transform(&mut self) {
        self.shape_spring = None;
        self.shape_tween = None;
        self.shape_transform = [1.0, 1.0, 0.0, 0.0];
    }

    /// Sets the shape transform directly, with no animation: cancels any
    /// running spring or tween. The instant counterpart of
    /// [`Self::begin_shape_tween`] (a freshly created surface snapping to
    /// its menu's held transform).
    pub fn set_shape_transform_direct(&mut self, value: [f32; 4]) {
        self.shape_spring = None;
        self.shape_tween = None;
        self.shape_transform = value;
    }

    /// Kicks the pill squash/stretch spring with explicit initial
    /// conditions — the host's resize animation computes them from the size
    /// change (compressed start with a velocity kick, see the renderer).
    /// The spring integrates inside the render loop from here on, replacing
    /// the legacy per-tick host thread.
    pub fn begin_pill_spring(
        &mut self,
        x: f32,
        y: f32,
        vx: f32,
        vy: f32,
        omega: f64,
        zeta: f64,
        restitution: f32,
    ) {
        self.pill_spring = Some(TransformSpring::new(
            [x, y, 0.0, 0.0],
            [vx, vy, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            2,
            omega,
            zeta,
            restitution,
        ));
    }

    /// Stamps the compositor-owned anisotropic scale into a params copy
    /// before upload (the same way `animation_time` is compositor-owned).
    fn stamp_compositor_owned(&self, params: &mut GlassShaderParams) {
        params.pill_scale_x = self.pill_scale_x;
        params.pill_scale_y = self.pill_scale_y;
        params.shape_transform = self.shape_transform;
    }

    /// Shifts the sampling origin down with the surface window (auto-hide
    /// slide). A GLOBAL surface's samples hang in monitor space and follow
    /// the slide; a LOCAL surface's samples hang in its host window's
    /// frame, which does not move relative to the glass when the host's own
    /// chrome slides — and Glassbar's auto-hide slide is a monitor-space
    /// feature anyway, so never re-anchors there.
    pub fn translate(&mut self, context: &ID3D11DeviceContext, dy: i32) {
        if dy == 0 || !self.monitor_space {
            return;
        }
        self.surface_bounds.top += dy;
        self.surface_bounds.bottom += dy;
        let Ok(monitor) = primary_monitor_bounds() else {
            return;
        };
        if let Ok(updated) = source_rect_constants(self.surface_bounds, monitor) {
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
