use std::mem::size_of;

use super::compositor::compile_shader;
use super::geometry::source_rect_constants;
use crate::params::GlassShaderParams;
use crate::shared::{compile_shader_from, KAWASE_HLSL};
use windows::Win32::Foundation::RECT;

#[test]
fn liquid_composition_hlsl_compiles_for_d3d11() {
    compile_shader("vs_main", "vs_5_0").expect("D3D11 liquid vertex shader should compile");
    compile_shader("ps_main", "ps_5_0").expect("D3D11 liquid pixel shader should compile");
    // The Efficiency tier's lite variant: both entries live in the same
    // source, so a syntax error in either fails here at build time.
    compile_shader("ps_main_lite", "ps_5_0")
        .expect("D3D11 lite liquid pixel shader should compile");
}

#[test]
fn kawase_hlsl_compiles_for_d3d11() {
    compile_shader_from(KAWASE_HLSL, "vs_main", "vs_5_0")
        .expect("Kawase vertex shader should compile");
    for entry in ["ps_kawase_crop", "ps_kawase_down", "ps_kawase_up"] {
        compile_shader_from(KAWASE_HLSL, entry, "ps_5_0")
            .unwrap_or_else(|error| panic!("{entry} should compile: {error}"));
    }
}

#[test]
fn source_rectangle_maps_overlay_into_primary_capture() {
    let constants = source_rect_constants(
        RECT {
            left: 640,
            top: 300,
            right: 1920,
            bottom: 400,
        },
        RECT {
            left: 0,
            top: 0,
            right: 2560,
            bottom: 1440,
        },
    )
    .expect("valid bounds should map to source UVs");

    assert_eq!(constants.source_uv_origin, [0.25, 300.0 / 1440.0]);
    assert_eq!(constants.source_uv_size, [0.5, 100.0 / 1440.0]);
}

#[test]
fn glass_shader_params_layout_matches_16_byte_constant_rows() {
    assert_eq!(
        size_of::<GlassShaderParams>(),
        size_of::<[f32; 76]>(),
        "GlassShaderParams must stay exactly nineteen 16-byte constant rows \
         (35 core floats + one alignment filler + shape region + shape \
         transform + shape meta + six metaball rows + the territory clip row)"
    );
}

mod metaball_field;
mod metaball_seam_harness;

#[test]
fn metaball_effective_radii_fold_weight_into_radius_and_skip_inactive_rows() {
    // The stamp the shader relies on: slot 2 becomes radius * weight^(1/n),
    // inactive rows pass through untouched, and a sub-1 tension clamps to
    // the field's >= 1 contract (weight^(1/1) = weight).
    let balls = [[1.0, 2.0, 8.0, 4.0], [3.0, 4.0, 5.0, 1.0], [0.0; 4], [0.0; 4], [0.0; 4], [0.0; 4]];
    let stamped = super::glass::metaball_effective_radii(balls, 2, 2.0);
    assert_eq!(stamped[0][2], 16.0, "weight 4 at tension 2 doubles the radius");
    assert_eq!(stamped[1][2], 5.0, "weight 1 is an identity fold");
    let clamped = super::glass::metaball_effective_radii(balls, 1, 0.5);
    assert_eq!(clamped[0][2], 32.0, "tension clamps to 1 before the fold");
    let untouched = super::glass::metaball_effective_radii(balls, 1, 2.0);
    assert_eq!(untouched[1], balls[1], "rows past the count stay raw");
}
