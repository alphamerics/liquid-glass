//! The metaball field's contract, mirrored from
//! `liquid_composition.hlsl`'s `metaballFieldSd` in Rust.
#[test]
fn metaball_field_reduces_to_a_circle_for_one_ball_and_merges_for_two() {
    // Mirrors metaballFieldSd (liquid_composition.hlsl) so the field's
    // contract is pinned in Rust: a single weight-1 ball is the exact
    // circle of its radius at the boundary, the interior depth stays
    // metric (the potential's first-order estimate collapses toward the
    // ball centers), two separated balls merge into one blob across
    // the gap between them as long as their falloff windows overlap, and
    // balls whose centers lie past each other's influence falloff stop
    // interacting entirely (each stays a round circle — no neck). The
    // rows arrive the way the compositor stamps
    // them (metaball_effective_radii): the weight is already folded into
    // the radius in slot 2, and the field reads it directly.
    let smoothstep = |edge0: f64, edge1: f64, x: f64| -> f64 {
        let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };
    let field_sd = |p: [f64; 2], balls: &[[f64; 4]], n: f64| -> f64 {
        let mut field = 0.0;
        let mut gradient = [0.0, 0.0];
        let mut nearest_disk = f64::INFINITY;
        for ball in balls {
            let offset = [p[0] - ball[0], p[1] - ball[1]];
            let dist = (offset[0] * offset[0] + offset[1] * offset[1] + 1e-5).sqrt();
            let effective_radius = ball[2];
            let raw = (effective_radius / dist).powf(n);
            let window = 1.0 - smoothstep(3.0 * effective_radius, 4.5 * effective_radius, dist);
            let contribution = raw * window;
            field += contribution;
            gradient[0] += -n * contribution * offset[0] / (dist * dist);
            gradient[1] += -n * contribution * offset[1] / (dist * dist);
            nearest_disk = nearest_disk.min(dist - effective_radius);
        }
        let safe_field = field.max(1e-6);
        let f = 1.0 - safe_field.powf(1.0 / n);
        let gradient_magnitude = (gradient[0] * gradient[0] + gradient[1] * gradient[1]).sqrt()
            * safe_field.powf(1.0 / n - 1.0)
            / n;
        let potential_sd = f / gradient_magnitude.max(0.3);
        nearest_disk.min(potential_sd)
    };
    // Weight 1 leaves the radius unchanged (the fold is an identity here).
    let one = [[0.0, 0.0, 0.5, 1.0]];
    // Exactly on the ball's boundary: the surface.
    assert!(
        field_sd([0.5, 0.0], &one, 2.0).abs() < 1e-3,
        "an isolated weight-1 ball must surface exactly at its radius"
    );
    // The interior depth is metric (the disk distance, not the potential's
    // collapsed estimate): half a radius in reads as half a radius deep.
    let inside = field_sd([0.25, 0.0], &one, 2.0);
    assert!(
        (inside + 0.25).abs() < 1e-3,
        "interior depth must stay metric at the ball's center line, got {inside}"
    );
    assert!(field_sd([0.7, 0.0], &one, 2.0) > 0.0);
    // Two balls whose plain circles do NOT touch still merge across the
    // gap (the metaball neck) — and the field stays empty without balls.
    let two = [[-0.35, 0.0, 0.25, 1.0], [0.35, 0.0, 0.25, 1.0]];
    let neck = field_sd([0.0, 0.0], &two, 2.0);
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
    // (start 3.0 radii, end 4.5 — centers 8 radii apart here) contribute
    // nothing at the midpoint, so each stays its own round circle and no
    // neck forms. Isolated-circle exactness is the single-ball test above.
    let two_far = [[-1.0, 0.0, 0.25, 1.0], [1.0, 0.0, 0.25, 1.0]];
    let midpoint = field_sd([0.0, 0.0], &two_far, 2.0);
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
    assert!(field_sd([0.0, 0.0], &[], 2.0) > 0.0);

    // The cross-surface territory clip: a shared field renders only where
    // the OWN ball dominates the foreign ones, so two surfaces tile the
    // merged blob without rendering it twice. Own ball = index 0.
    let own_dominant = |p: [f64; 2]| -> bool {
        let mut own = -1.0f64;
        let mut max_foreign = -1.0f64;
        for (index, ball) in two.iter().enumerate() {
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
