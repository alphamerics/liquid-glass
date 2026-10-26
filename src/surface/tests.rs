use std::mem::size_of;

use super::compositor::compile_shader;
use super::geometry::source_rect_constants;
use crate::params::GlassShaderParams;
use crate::shared::{KAWASE_HLSL, compile_shader_from};
use windows::Win32::Foundation::RECT;

#[test]
fn liquid_composition_hlsl_compiles_for_d3d11() {
    compile_shader("vs_main", "vs_5_0").expect("D3D11 liquid vertex shader should compile");
    compile_shader("ps_main", "ps_5_0").expect("D3D11 liquid pixel shader should compile");
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
        size_of::<[f32; 36]>(),
        "GlassShaderParams must stay exactly nine 16-byte constant rows"
    );
}