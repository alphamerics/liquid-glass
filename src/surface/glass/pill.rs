//! The pill's squash/stretch spring control.




use super::GlassSurface;
impl GlassSurface {
    /// Sets the pill's uniform shape scale (1.0 = full size). Used by the
    /// host's auto-hide reveal to scale the pill in from slightly smaller.
    pub fn set_pill_scale(&mut self, scale: f32) {
        let _ = Self::with_context(|context| {
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
        let _ = Self::with_context(|context| {
            self.compositor
                .lock()
                .expect("D3D11 compositor lock poisoned")
                .set_pill_scale_xy(context, scale_x, scale_y);
            Ok(())
        });
    }

    /// Kicks the pill squash/stretch spring with explicit initial
    /// conditions; the compositor integrates it at the render loop's rate
    /// (phase-aligned with composition — no host-side animation thread).
    /// `omega`/`zeta`/`restitution` follow the shared glass spring
    /// convention (ω = 2π/duration, ζ = 1 − bounce·0.75, restitution
    /// bounce·0.6); both axes ceiling-clamp at 1.0 so the squash can never
    /// stretch the shape beyond its surface.
    #[allow(clippy::too_many_arguments)]
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
        let _ = Self::with_context(|_| {
            self.compositor
                .lock()
                .expect("D3D11 compositor lock poisoned")
                .begin_pill_spring(x, y, vx, vy, omega, zeta, restitution);
            Ok(())
        });
    }
}
