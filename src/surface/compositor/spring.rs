//! The damped-spring system behind the compositor's shape and pill
//! animations.

/// One damped-spring system over up to four axes (the shape transform's
/// scale x/y and offset x/y, or the pill squash/stretch's two axes). The
/// integration mirrors glass-ui's spring: ω = 2π/duration, damping ratio
/// ζ = 1 − bounce·0.75. `ceiling` axes clamp at their target — overshoot
/// past the drawn shape's rect would clip at the canvas edge, so reaching
/// the target reflects the velocity with `restitution` instead (the "squash
/// into place" bounce); non-ceiling axes (offsets) overshoot freely.
pub(super) struct TransformSpring {
    pub(super) value: [f32; 4],
    pub(super) velocity: [f32; 4],
    target: [f32; 4],
    /// How many leading axes are ceiling-clamped at their target (the rest
    /// overshoot freely).
    ceiling_axes: usize,
    omega: f64,
    zeta: f64,
    restitution: f32,
}

impl TransformSpring {
    pub(super) fn new(
        value: [f32; 4],
        velocity: [f32; 4],
        target: [f32; 4],
        ceiling_axes: usize,
        omega: f64,
        zeta: f64,
        restitution: f32,
    ) -> Self {
        Self {
            value,
            velocity,
            target,
            ceiling_axes,
            omega,
            zeta,
            restitution,
        }
    }

    /// Integrates one step with a real `dt` (seconds). Returns whether the
    /// whole system has settled (every axis at its target and at rest).
    pub(super) fn step(&mut self, dt: f32) -> bool {
        let mut settled = true;
        for axis in 0..4 {
            let value = self.value[axis];
            let velocity = self.velocity[axis];
            let target = self.target[axis];
            let acceleration = (self.omega * self.omega) * ((target - value) as f64)
                - (2.0 * self.zeta * self.omega) * (velocity as f64);
            let mut velocity = velocity + (acceleration as f32) * dt;
            let mut value = value + velocity * dt;
            if axis < self.ceiling_axes && value > target && velocity > 0.0 {
                // Only upward overshoot is clamped: a shrink legitimately
                // starts ABOVE its target (the previous visual rect was
                // larger) and must glide down through it — the ceiling
                // exists to keep a growing/bouncing shape from stretching
                // past its canvas.
                value = target;
                velocity = -velocity * self.restitution;
                // Micro-bounce deadzone: reflecting a tiny velocity at the
                // ceiling produces sub-pixel vibration (visible jitter).
                // Below the threshold the axis simply settles at the target.
                if velocity.abs() < 0.08 {
                    velocity = 0.0;
                }
            }
            if (value - target).abs() >= 0.002 || velocity.abs() >= 0.01 {
                settled = false;
            } else {
                value = target;
                velocity = 0.0;
            }
            self.value[axis] = value;
            self.velocity[axis] = velocity;
        }
        settled
    }
}
