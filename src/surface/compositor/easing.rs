//! The time-based easing system behind the compositor's shape tweens: the
//! CSS timing-function family (cubic-bezier control points), the named
//! presets, and the tween state that rides next to the damped springs.
//!
//! The bezier solver follows the CSS timing-function algorithm (the same
//! one WebKit ships): the x polynomial is inverted for the curve parameter
//! with Newton-Raphson, falling back to a bisection when Newton stalls,
//! then the y polynomial is evaluated at that parameter. Unlike the
//! springs (state: value + velocity), a tween is fully described by its
//! endpoints and clock, so a retarget simply restarts from the live value —
//! CSS transition semantics.

use std::time::{Duration, Instant};

/// One cubic-bezier timing function: `cubic-bezier(x1, y1, x2, y2)` —
/// the CSS `transition-timing-function`. The control points' x coordinates
/// must stay in [0, 1] (CSS rejects the rest); the y coordinates may
/// overshoot the unit interval, which is what produces the "back-out"
/// spring-like feel (e.g. `cubic-bezier(0.34, 1.56, 0.64, 1)`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct CubicBezier {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

/// The named CSS easing presets, mirroring `transition-timing-function`'s
/// keywords (the wire APIs speak control points; these exist for callers
/// that want a preset without parsing).
// The presets are the canonical reference values and the tests exercise
// them; production callers pass explicit control points.
#[allow(dead_code)]
impl CubicBezier {
    pub(super) const LINEAR: Self = Self::new(0.0, 0.0, 1.0, 1.0);
    pub(super) const EASE: Self = Self::new(0.25, 0.1, 0.25, 1.0);
    pub(super) const EASE_IN: Self = Self::new(0.42, 0.0, 1.0, 1.0);
    pub(super) const EASE_OUT: Self = Self::new(0.0, 0.0, 0.58, 1.0);
    pub(super) const EASE_IN_OUT: Self = Self::new(0.42, 0.0, 0.58, 1.0);
}

impl CubicBezier {
    pub(super) const fn new(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Self { x1, y1, x2, y2 }
    }

    /// Builds the curve from a `[x1, y1, x2, y2]` control-point array (the
    /// wire format every host API speaks).
    pub(super) const fn from_points(points: [f32; 4]) -> Self {
        Self::new(points[0], points[1], points[2], points[3])
    }

    /// Evaluates the easing at normalized progress `x` (0 = start, 1 = end).
    /// Returns exactly 0 and 1 at the endpoints; the output may overshoot
    /// [0, 1] for curves whose y control points leave the unit interval.
    pub(super) fn solve(&self, x: f32) -> f32 {
        // The timing function's domain is the unit interval; clamp inputs
        // arriving from a stepping clock that overshot by a hair.
        let x = x.clamp(0.0, 1.0);
        if x <= 0.0 {
            return 0.0;
        }
        if x >= 1.0 {
            return 1.0;
        }
        self.sample_y(self.solve_parameter(x))
    }

    /// The Bézier's power-basis coefficients (P0 = (0, 0), P3 = (1, 1)):
    /// `p(t) = ((a*t + b)*t + c)*t` per axis.
    fn coefficients(&self) -> ([f32; 3], [f32; 3]) {
        let cx = 3.0 * self.x1;
        let bx = 3.0 * (self.x2 - self.x1) - cx;
        let ax = 1.0 - cx - bx;
        let cy = 3.0 * self.y1;
        let by = 3.0 * (self.y2 - self.y1) - cy;
        let ay = 1.0 - cy - by;
        ([ax, bx, cx], [ay, by, cy])
    }

    fn sample_x(&self, t: f32, x: [f32; 3]) -> f32 {
        ((x[0] * t + x[1]) * t + x[2]) * t
    }

    fn sample_y(&self, t: f32) -> f32 {
        let (_, y) = self.coefficients();
        ((y[0] * t + y[1]) * t + y[2]) * t
    }

    fn sample_x_derivative(&self, t: f32, x: [f32; 3]) -> f32 {
        (3.0 * x[0] * t + 2.0 * x[1]) * t + x[2]
    }

    /// Inverts the x polynomial for the curve parameter: Newton-Raphson
    /// from a linear guess (the curve is close to linear for most presets,
    /// so this converges in a handful of steps), with a bisection fallback
    /// for the cases Newton cannot bracket (horizontal-tangent segments).
    fn solve_parameter(&self, goal_x: f32) -> f32 {
        let x = self.coefficients().0;
        // Newton-Raphson: 8 iterations, the classic iteration budget.
        let mut t = goal_x;
        for _ in 0..8 {
            let error = self.sample_x(t, x) - goal_x;
            if error.abs() < 1e-6 {
                return t;
            }
            let derivative = self.sample_x_derivative(t, x);
            if derivative.abs() < 1e-6 {
                break;
            }
            t -= error / derivative;
            if !(0.0..=1.0).contains(&t) {
                break;
            }
        }
        // Bisection fallback: the x polynomial is monotonic for every
        // valid CSS curve (x control points in [0, 1]), so the root is
        // always bracketed between the endpoints.
        let (mut low, mut high) = (0.0_f32, 1.0_f32);
        let mut t = goal_x;
        for _ in 0..20 {
            let sampled = self.sample_x(t, x);
            if (sampled - goal_x).abs() < 1e-6 {
                return t;
            }
            if sampled < goal_x {
                low = t;
            } else {
                high = t;
            }
            t = (low + high) * 0.5;
        }
        t
    }
}

/// The compositor's time-based shape tween: the shape transform glides
/// from `from` (the live value when the tween started) to `to` over
/// `duration` after `delay`, paced by a [`CubicBezier`] curve. The
/// counterpart of the damped [`super::spring::TransformSpring`]: a tween
/// is deterministic (endpoints + clock) instead of stateful, so a retarget
/// restarts from the live value with no velocity to carry over.
pub(super) struct ShapeTween {
    pub(super) from: [f32; 4],
    pub(super) to: [f32; 4],
    /// When the animation began. The `delay` folds into this instant: the
    /// tween's clock reads zero until `start` arrives.
    start: Instant,
    pub(super) duration: Duration,
    pub(super) easing: CubicBezier,
}

impl ShapeTween {
    pub(super) fn new(
        from: [f32; 4],
        to: [f32; 4],
        delay: Duration,
        duration: Duration,
        easing: CubicBezier,
    ) -> Self {
        Self {
            from,
            to,
            start: Instant::now() + delay,
            duration,
            easing,
        }
    }

    /// The tween's normalized progress at `now` (0 while the delay holds,
    /// 1 once the duration has fully elapsed).
    pub(super) fn progress(&self, now: Instant) -> f32 {
        let elapsed = now.checked_duration_since(self.start).unwrap_or_default();
        (elapsed.as_secs_f32() / self.duration.as_secs_f32().max(1e-4)).clamp(0.0, 1.0)
    }

    /// The transform value at `now`, interpolating `from` → `to` through
    /// the easing curve.
    pub(super) fn value_at(&self, now: Instant) -> [f32; 4] {
        let eased = self.easing.solve(self.progress(now));
        std::array::from_fn(|axis| self.from[axis] + (self.to[axis] - self.from[axis]) * eased)
    }
}

#[cfg(test)]
mod tests {
    use super::CubicBezier;

    /// The x polynomial's power-basis coefficients, rebuilt from the
    /// curve's fields (the tests access the private fields directly).
    fn x_coefficients(curve: &CubicBezier) -> [f32; 3] {
        let cx = 3.0 * curve.x1;
        let bx = 3.0 * (curve.x2 - curve.x1) - cx;
        let ax = 1.0 - cx - bx;
        [ax, bx, cx]
    }

    fn sample_x(t: f32, x: [f32; 3]) -> f32 {
        ((x[0] * t + x[1]) * t + x[2]) * t
    }

    /// A brute-force reference inverter: plain binary search on the x
    /// polynomial (a different algorithm path than the production
    /// solver), then the y evaluation through the production sampler.
    /// Agreement between the two validates the Newton-Raphson path.
    fn reference_solve(curve: &CubicBezier, x: f32) -> f32 {
        let x_coefficients = x_coefficients(curve);
        let (mut low, mut high) = (0.0_f32, 1.0_f32);
        for _ in 0..60 {
            let middle = (low + high) * 0.5;
            if sample_x(middle, x_coefficients) < x {
                low = middle;
            } else {
                high = middle;
            }
        }
        curve.sample_y((low + high) * 0.5)
    }

    #[test]
    fn endpoints_are_exact() {
        for curve in [
            CubicBezier::LINEAR,
            CubicBezier::EASE,
            CubicBezier::EASE_IN,
            CubicBezier::EASE_OUT,
            CubicBezier::EASE_IN_OUT,
            CubicBezier::new(0.34, 1.56, 0.64, 1.0),
        ] {
            assert_eq!(curve.solve(0.0), 0.0);
            assert_eq!(curve.solve(1.0), 1.0);
        }
    }

    #[test]
    fn linear_is_identity() {
        for i in 0..=20 {
            let x = i as f32 / 20.0;
            assert!((CubicBezier::LINEAR.solve(x) - x).abs() < 1e-5);
        }
    }

    #[test]
    fn ease_in_out_is_symmetric() {
        // A symmetric curve (x1 + x2 = 1, y1 + y2 = 1) satisfies
        // y(1 - x) = 1 - y(x); the halfway point is exactly one half.
        assert!((CubicBezier::EASE_IN_OUT.solve(0.5) - 0.5).abs() < 1e-4);
        for i in 1..10 {
            let x = i as f32 / 10.0;
            let mirrored = CubicBezier::EASE_IN_OUT.solve(1.0 - x);
            assert!((mirrored - (1.0 - CubicBezier::EASE_IN_OUT.solve(x))).abs() < 1e-3);
        }
    }

    #[test]
    fn overshooting_curves_exceed_the_unit_interval() {
        // The back-out curve (y1 > 1) must rise past 1 before settling.
        let back_out = CubicBezier::new(0.34, 1.56, 0.64, 1.0);
        let peak = (0..=100)
            .map(|i| back_out.solve(i as f32 / 100.0))
            .fold(f32::MIN, f32::max);
        assert!(peak > 1.05, "back-out peak {peak} never overshot");
        assert_eq!(back_out.solve(1.0), 1.0);
    }

    #[test]
    fn solver_matches_the_bisection_reference() {
        for (curve, name) in [
            (CubicBezier::EASE, "ease"),
            (CubicBezier::EASE_IN, "ease-in"),
            (CubicBezier::EASE_OUT, "ease-out"),
            (CubicBezier::new(0.19, 1.0, 0.22, 1.0), "expo-out"),
            (CubicBezier::new(0.34, 1.56, 0.64, 1.0), "back-out"),
        ] {
            for i in 1..20 {
                let x = i as f32 / 20.0;
                let solved = curve.solve(x);
                let reference = reference_solve(&curve, x);
                assert!(
                    (solved - reference).abs() < 2e-4,
                    "{name} at {x}: {solved} vs reference {reference}"
                );
            }
        }
    }

    #[test]
    fn monotone_curves_solve_monotonically() {
        // Every valid CSS curve is non-decreasing in y; sample coarsely
        // and assert the sequence never steps back.
        for curve in [
            CubicBezier::EASE,
            CubicBezier::EASE_IN,
            CubicBezier::EASE_OUT,
            CubicBezier::EASE_IN_OUT,
            CubicBezier::LINEAR,
        ] {
            let mut previous = -1.0_f32;
            for i in 0..=50 {
                let y = curve.solve(i as f32 / 50.0);
                assert!(y >= previous - 1e-5, "curve regressed at sample {i}");
                previous = y;
            }
        }
    }

    #[test]
    fn inputs_outside_the_unit_interval_clamp() {
        assert_eq!(CubicBezier::EASE.solve(-0.5), 0.0);
        assert_eq!(CubicBezier::EASE.solve(1.5), 1.0);
    }

    #[test]
    fn from_points_matches_new() {
        let built = CubicBezier::from_points([0.19, 1.0, 0.22, 1.0]);
        assert_eq!(built, CubicBezier::new(0.19, 1.0, 0.22, 1.0));
    }
}
