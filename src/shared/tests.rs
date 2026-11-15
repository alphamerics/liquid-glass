    use super::{blur_plan_with_caps, grid_aligned_crop, CropConstants};
    use crate::params::GlassShaderParams;
    use crate::quality::QualityCaps;

    const HIGH: QualityCaps = QualityCaps::HIGH;

    #[test]
    fn blur_plan_scales_resolution_and_levels_with_radius() {
        let mut params = GlassShaderParams::default();
        // Blur off: sharp capture, no chain.
        params.blur_radius_texels = 0.0;
        assert!(blur_plan_with_caps(&params, &HIGH).sharp);
        // Default-ish blur (8): a shallow chain at high resolution.
        params.blur_radius_texels = 8.0;
        let plan = blur_plan_with_caps(&params, &HIGH);
        assert!(!plan.sharp);
        assert_eq!(plan.levels, 2);
        assert!(plan.base_scale > 0.9);
        // Level bands at 12 / 28 / 64 (one buffer per hop, plus the crop).
        params.blur_radius_texels = 16.0;
        assert_eq!(blur_plan_with_caps(&params, &HIGH).levels, 3);
        params.blur_radius_texels = 32.0;
        assert_eq!(blur_plan_with_caps(&params, &HIGH).levels, 4);
        // Subtle blur: still a shallow chain at ~full resolution.
        params.blur_radius_texels = 2.0;
        let plan = blur_plan_with_caps(&params, &HIGH);
        assert_eq!(plan.levels, 2);
        assert!(plan.base_scale > 0.98);
        // Samples setting only ever caps levels, never raises them.
        params.blur_samples = 1.0;
        params.blur_radius_texels = 32.0;
        assert_eq!(blur_plan_with_caps(&params, &HIGH).levels, 1);
    }

    #[test]
    fn blur_plan_is_continuous_across_level_bands() {
        // In the sigma ≈ offsets · 2^levels · base model, each band must END
        // where the next one STARTS — otherwise dragging the radius slider
        // snaps harshly when a level is added.
        let mut params = GlassShaderParams::default();
        // The full 5-level headroom: the 64-boundary region needs the fifth
        // level, and a smaller samples cap would truncate it.
        params.blur_samples = 5.0;
        let reach = |radius: f32| {
            let plan = blur_plan_with_caps(
                &{
                    let mut p = params;
                    p.blur_radius_texels = radius;
                    p
                },
                &HIGH,
            );
            plan.offset_scale * (1usize << plan.levels) as f32 * plan.base_scale
        };
        for boundary in [12.0_f32, 28.0, 64.0] {
            let before = reach(boundary - 0.01);
            let after = reach(boundary + 0.01);
            let jump = (after - before).abs() / before.abs().max(1e-4);
            assert!(
                jump < 0.1,
                "reach jumped {before} -> {after} at the level boundary {boundary}"
            );
        }
    }

    #[test]
    fn efficiency_caps_cut_internal_resolution() {
        let mut params = GlassShaderParams::default();
        params.blur_samples = 5.0;
        params.blur_radius_texels = 32.0;
        let plan = blur_plan_with_caps(&params, &QualityCaps::EFFICIENCY);
        // The internal scale ramp is fully engaged at radius 24+.
        assert!(plan.base_scale <= QualityCaps::EFFICIENCY.blur_internal_scale);
        // Subtle blur (below the ramp) keeps full resolution even on the
        // efficient tier: resampling would be visible there. (Radius 8
        // sits in the first band whose own ramp reaches base 0.936.)
        params.blur_radius_texels = 8.0;
        let plan = blur_plan_with_caps(&params, &QualityCaps::EFFICIENCY);
        assert!(plan.base_scale > 0.9);
        // The level count is untouched by the tier (only the user's
        // samples setting caps it).
        params.blur_radius_texels = 32.0;
        assert_eq!(
            blur_plan_with_caps(&params, &QualityCaps::EFFICIENCY).levels,
            4
        );
    }

    #[test]
    fn blur_plan_stays_continuous_under_tier_caps() {
        use crate::quality::QualityCaps;
        let mut params = GlassShaderParams::default();
        params.blur_samples = 5.0;
        let reach = |radius: f32| {
            let plan = blur_plan_with_caps(
                &{
                    let mut p = params;
                    p.blur_radius_texels = radius;
                    p
                },
                &QualityCaps::EFFICIENCY,
            );
            plan.offset_scale * (1usize << plan.levels) as f32 * plan.base_scale
        };
        for boundary in [12.0_f32, 24.0, 28.0, 64.0] {
            let before = reach(boundary - 0.01);
            let after = reach(boundary + 0.01);
            let jump = (after - before).abs() / before.abs().max(1e-4);
            assert!(
                jump < 0.1,
                "efficiency reach jumped {before} -> {after} at {boundary}"
            );
        }
    }

    #[test]
    fn grid_aligned_crop_rounds_outward_and_clamps() {
        // A rect strictly inside the frame grows to the 64 px grid in both
        // directions: the rounded rect must CONTAIN the input.
        let frame = [1920, 1080];
        let rect = grid_aligned_crop([100.3, 200.6, 500.2, 700.9], frame, 64.0).expect("rect");
        assert!(rect[0] <= 100 && rect[1] <= 200);
        assert!(rect[2] >= 501 && rect[3] >= 701);
        assert_eq!(rect[0] % 64, 0);
        assert_eq!(rect[1] % 64, 0);
        assert_eq!(rect[2] % 64, 0);
        assert_eq!(rect[3] % 64, 0);
        // Inputs inside the same grid cells snap to the same rect: small
        // surface moves must not churn chain rebuilds.
        let nudged = grid_aligned_crop([103.0, 205.0, 505.0, 704.0], frame, 64.0).expect("rect");
        assert_eq!(rect, nudged);
        // Rects reaching past the frame clamp into it.
        let clamped = grid_aligned_crop([-50.0, -50.0, 5000.0, 5000.0], frame, 64.0).expect("rect");
        assert_eq!(clamped, [0, 0, 1920, 1080]);
        // Degenerate rects (past the frame edge, empty after clamping)
        // fall back to no crop.
        assert_eq!(
            grid_aligned_crop([2000.0, 500.0, 2100.0, 600.0], frame, 64.0),
            None
        );
    }

    #[test]
    fn crop_constants_map_capture_uvs_into_the_backdrop() {
        let frame = [1920, 1080];
        let texel = [1.0 / 1920.0, 1.0 / 1080.0];
        // The identity crop maps every capture uv to itself.
        let identity = CropConstants::identity(texel);
        assert_eq!(backdrop_uv_of(&identity, [0.25, 0.75]), [0.25, 0.75]);
        // A sub-rect crop maps the region's corners to (0, 0) and (1, 1)
        // and its center to (0.5, 0.5) — exactly what the composite's
        // `backdropUv` does in the shader.
        let rect = [640, 360, 1280, 720];
        let crop = CropConstants::from_pixel_rect(rect, frame, texel);
        let map = |uv: [f32; 2]| backdrop_uv_of(&crop, uv);
        assert_eq!(
            map([rect[0] as f32 / 1920.0, rect[1] as f32 / 1080.0]),
            [0.0, 0.0]
        );
        assert_eq!(
            map([rect[2] as f32 / 1920.0, rect[3] as f32 / 1080.0]),
            [1.0, 1.0]
        );
        let center = map([960.0 / 1920.0, 540.0 / 1080.0]);
        assert!((center[0] - 0.5).abs() < 1e-6 && (center[1] - 0.5).abs() < 1e-6);
    }

    /// The composite's capture→backdrop mapping (`backdropUv` in
    /// liquid_composition.hlsl), mirrored here so the Rust-provided crop
    /// constants are checked against the shader's exact math.
    fn backdrop_uv_of(crop: &CropConstants, capture_uv: [f32; 2]) -> [f32; 2] {
        [
            (capture_uv[0] - crop.crop_origin_uv[0]) / crop.crop_size_uv[0],
            (capture_uv[1] - crop.crop_origin_uv[1]) / crop.crop_size_uv[1],
        ]
    }
