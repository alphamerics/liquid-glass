//! The metaball field's contract, mirrored from
//! `liquid_composition.hlsl`'s `metaballFieldSd` in Rust.
//!
//! The mirror consumes the STAMPED rows exactly as the shader reads them:
//! potential rows `(cx, cy, influence, weight)` (the weight already folded
//! into the influence radius by `metaball_effective_rows`) and shape rows
//! `(exponent, extentX, extentY, spare)`. A circle ball (exponent 2, equal
//! extents) must behave exactly like the classic field; a shaped ball
//! surfaces on its own superellipse silhouette and merges like a circle.
#[test]
fn metaball_field_reduces_to_a_circle_for_one_ball_and_merges_for_two() {
    let smoothstep = |edge0: f64, edge1: f64, x: f64| -> f64 {
        let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };
    // The fused superellipse pseudo-SDF (superellipseSdNormal): signed
    // distance with the exact boundary slope, outward normal riding out.
    let superellipse_sd_normal = |p: [f64; 2], ext: [f64; 2], exponent: f64| -> (f64, [f64; 2]) {
        let n = exponent.max(1.0);
        let safe = [ext[0].max(1e-5), ext[1].max(1e-5)];
        let q = [p[0].abs() / safe[0], p[1].abs() / safe[1]];
        let g = q[0].powf(n) + q[1].powf(n);
        let f = g.max(0.0).powf(1.0 / n) - 1.0;
        let common = g.max(1e-6).powf(1.0 / n - 1.0);
        let grad = [
            common * p[0].signum() * q[0].powf(n - 1.0) / safe[0],
            common * p[1].signum() * q[1].powf(n - 1.0) / safe[1],
        ];
        let grad_length = (grad[0] * grad[0] + grad[1] * grad[1]).sqrt().max(1e-6);
        (
            f / grad_length,
            [grad[0] / grad_length, grad[1] / grad_length],
        )
    };
    let is_circle =
        |shape: &[f64; 4]| (shape[0] - 2.0).abs() < 1e-4 && (shape[1] - shape[2]).abs() < 1e-4;
    // The field loop, per ball: contribution (r / (r + sd))ⁿ over the
    // ball's own signed distance, the 1..2 influence-radius falloff window
    // (see METABALL_FALLOFF_* — er + sd IS the circle's center distance),
    // and the interior metric falling back to the nearest ball's signed
    // distance clamped at the influence radius.
    let field_sd = |p: [f64; 2], rows: &[[f64; 4]], shapes: &[[f64; 4]], n: f64| -> f64 {
        let quadratic = (n - 2.0).abs() < 1e-4;
        let mut field = 0.0;
        let mut gradient = [0.0, 0.0];
        let mut nearest_disk = f64::INFINITY;
        let mut max_influence = 0.0f64;
        for (ball, shape) in rows.iter().zip(shapes.iter()) {
            let effective_radius = ball[2];
            let (sd, sd_grad, u);
            if is_circle(shape) {
                let offset = [p[0] - ball[0], p[1] - ball[1]];
                let dist = (offset[0] * offset[0] + offset[1] * offset[1] + 1e-5).sqrt();
                sd = dist - effective_radius;
                sd_grad = [offset[0] / dist, offset[1] / dist];
                u = dist;
            } else {
                let (sd_value, normal) = superellipse_sd_normal(
                    [p[0] - ball[0], p[1] - ball[1]],
                    [shape[1], shape[2]],
                    shape[0],
                );
                sd = sd_value;
                sd_grad = normal;
                u = (effective_radius + sd).max(1e-5);
            }
            let raw = if quadratic {
                effective_radius * effective_radius / (u * u)
            } else {
                (effective_radius / u).powf(n)
            };
            let window = 1.0 - smoothstep(effective_radius, 2.0 * effective_radius, u);
            let contribution = raw * window;
            field += contribution;
            gradient[0] += -n * contribution * sd_grad[0] / u;
            gradient[1] += -n * contribution * sd_grad[1] / u;
            nearest_disk = nearest_disk.min(sd);
            max_influence = max_influence.max(effective_radius);
        }
        let safe_field = field.max(1e-6);
        let f = 1.0
            - if quadratic {
                safe_field.sqrt()
            } else {
                safe_field.powf(1.0 / n)
            };
        let gradient_magnitude = (gradient[0] * gradient[0] + gradient[1] * gradient[1]).sqrt()
            * if quadratic {
                0.5 * safe_field.sqrt().recip()
            } else {
                safe_field.powf(1.0 / n - 1.0) / n
            };
        let potential_sd = f / gradient_magnitude.max(0.3);
        nearest_disk.min(potential_sd).max(-max_influence)
    };
    // ---- The classic circle field (exponent 2, equal extents) ----
    let circle_rows = |balls: &[[[f64; 2]; 2]]| -> (Vec<[f64; 4]>, Vec<[f64; 4]>) {
        (
            balls
                .iter()
                .map(|b| [b[0][0], b[0][1], b[1][0], 1.0])
                .collect(),
            balls.iter().map(|b| [2.0, b[1][0], b[1][0], 0.0]).collect(),
        )
    };
    // Weight 1 leaves the radius unchanged (the fold is an identity here).
    let (one_rows, one_shapes) = circle_rows(&[[[0.0, 0.0], [0.5, 0.5]]]);
    // Exactly on the ball's boundary: the surface.
    assert!(
        field_sd([0.5, 0.0], &one_rows, &one_shapes, 2.0).abs() < 1e-3,
        "an isolated weight-1 ball must surface exactly at its radius"
    );
    // The interior depth is metric (the disk distance, not the potential's
    // collapsed estimate): half a radius in reads as half a radius deep.
    let inside = field_sd([0.25, 0.0], &one_rows, &one_shapes, 2.0);
    assert!(
        (inside + 0.25).abs() < 1e-3,
        "interior depth must stay metric at the ball's center line, got {inside}"
    );
    assert!(field_sd([0.7, 0.0], &one_rows, &one_shapes, 2.0) > 0.0);
    // Two balls whose plain circles do NOT touch still merge across the
    // gap (the metaball neck) — the merge threshold sits at ~2.5 influence
    // radii center-to-center with the 1..2-radius falloff window.
    let (two_rows, two_shapes) = circle_rows(&[
        [[-0.3125, 0.0], [0.25, 0.25]],
        [[0.3125, 0.0], [0.25, 0.25]],
    ]);
    let neck = field_sd([0.0, 0.0], &two_rows, &two_shapes, 2.0);
    assert!(
        neck < 0.0,
        "separated balls must merge across the gap between them, got {neck}"
    );
    // The neck's depth stays sane at the symmetric saddle (where the
    // field's gradient vanishes and the raw estimate would explode).
    assert!(
        neck > -0.2,
        "the merge saddle must not explode the interior depth, got {neck}"
    );
    // The falloff: balls separated past each other's influence window
    // (centers 8 radii apart here) contribute nothing at the midpoint, so
    // each stays its own round circle and no neck forms.
    // Isolated-circle exactness is the single-ball test above.
    let (two_far_rows, two_far_shapes) =
        circle_rows(&[[[-1.0, 0.0], [0.25, 0.25]], [[1.0, 0.0], [0.25, 0.25]]]);
    let midpoint = field_sd([0.0, 0.0], &two_far_rows, &two_far_shapes, 2.0);
    assert!(
        midpoint > 0.0,
        "balls past the falloff range must not merge into a neck, got {midpoint}"
    );
    // And the point between them is at least as far from each full
    // interior disk as the raw gap implies (the disks stay round).
    assert!(
        midpoint > 0.5,
        "the falloff must leave distant balls' disks intact, got {midpoint}"
    );
    let (empty_rows, empty_shapes): (Vec<[f64; 4]>, Vec<[f64; 4]>) = (Vec::new(), Vec::new());
    assert!(field_sd([0.0, 0.0], &empty_rows, &empty_shapes, 2.0) > 0.0);

    // ---- Shaped balls ----
    // A single squircle ball (extents 0.5 x 0.3, exponent 4) surfaces
    // exactly on its own superellipse silhouette: on both axis ends, on
    // the 45-degree-ish boundary point (where |x/a|⁴ + |y/b|⁴ = 1 with
    // equal fractions), inside, and outside.
    let squircle_rows = [[0.0, 0.0, 0.3, 1.0]];
    let squircle_shapes = [[4.0, 0.5, 0.3, 0.0]];
    let diag = 2.0f64.powf(-0.25);
    for point in [
        [0.5, 0.0],
        [-0.5, 0.0],
        [0.0, 0.3],
        [0.5 * diag, 0.3 * diag],
        [-0.5 * diag, -0.3 * diag],
    ] {
        let sd = field_sd(point, &squircle_rows, &squircle_shapes, 2.0);
        assert!(
            sd.abs() < 2e-3,
            "an isolated shaped ball must surface on its superellipse at {point:?}, got {sd}"
        );
    }
    assert!(
        field_sd([0.2, 0.1], &squircle_rows, &squircle_shapes, 2.0) < 0.0,
        "a point inside the squircle silhouette must read interior"
    );
    assert!(
        field_sd([0.6, 0.0], &squircle_rows, &squircle_shapes, 2.0) > 0.0,
        "a point outside the squircle silhouette must read exterior"
    );
    // The interior metric stays bounded even where the pseudo-SDF
    // collapses (the superellipse's center, where its gradient dies):
    // clamped at the influence radius, never exploding.
    let center = field_sd([0.0, 0.0], &squircle_rows, &squircle_shapes, 2.0);
    assert!(
        center >= -0.3,
        "the shaped interior depth must clamp at the influence radius, got {center}"
    );
    // A squircle and a circle merge across the gap like two circles: the
    // neck forms once the silhouettes come within the merge threshold of
    // the influence radii (centers 2.4 influence radii apart here).
    let mixed_rows = [
        [-0.3, 0.0, 0.25, 1.0], // circle: influence 0.25
        [0.3, 0.0, 0.25, 1.0],  // squircle: influence = min extents
    ];
    let mixed_shapes = [[2.0, 0.25, 0.25, 0.0], [4.0, 0.25, 0.25, 0.0]];
    let mixed_neck = field_sd([0.0, 0.0], &mixed_rows, &mixed_shapes, 2.0);
    assert!(
        mixed_neck < 0.0,
        "a squircle and a circle must merge into one blob, got {mixed_neck}"
    );
    assert!(
        mixed_neck >= -0.25,
        "the mixed merge saddle must saturate at the influence cap (a \
         symmetric saddle's field is nearly flat, so the metric reads the \
         capped bound rather than the potential's estimate), got {mixed_neck}"
    );
    // ...and stay separate once pulled apart past the falloff window.
    let mixed_far_rows = [[-0.9, 0.0, 0.25, 1.0], [0.9, 0.0, 0.25, 1.0]];
    let mixed_far = field_sd([0.0, 0.0], &mixed_far_rows, &mixed_shapes, 2.0);
    assert!(
        mixed_far > 0.0,
        "a far squircle and circle must not merge, got {mixed_far}"
    );

    // The cross-surface territory clip: a shared field renders only where
    // the OWN ball dominates the foreign ones, so two surfaces tile the
    // merged blob without rendering it twice. Own ball = index 0. The
    // contribution formula is shape-agnostic (each ball's own), so the
    // clip works unchanged for mixed fields.
    let own_dominant = |p: [f64; 2]| -> bool {
        let mut own = -1.0f64;
        let mut max_foreign = -1.0f64;
        for (index, ball) in two_rows.iter().enumerate() {
            let offset = [p[0] - ball[0], p[1] - ball[1]];
            let dist = (offset[0] * offset[0] + offset[1] * offset[1] + 1e-5).sqrt();
            let contribution = (ball[2] * dist.sqrt().recip()).powf(2.0);
            if index == 0 {
                own = contribution;
            } else {
                max_foreign = max_foreign.max(contribution);
            }
        }
        own >= max_foreign
    };
    // Left of the bisector the own ball dominates (visible); right of it a
    // foreign ball dominates (clipped away — the other surface draws it).
    assert!(own_dominant([-0.3, 0.0]));
    assert!(!own_dominant([0.3, 0.0]));
}
