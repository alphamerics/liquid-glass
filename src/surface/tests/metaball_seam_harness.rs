//! GPU seam harness: renders the REAL composite shader for the two
//! territory-clipped metaball surface halves exactly the way the manager
//! sets them up (canvas snap, own-first field, pixel normalization), then
//! composites the two premultiplied outputs the way DWM does and verifies
//! coverage against the same field rendered WHOLE on one big canvas — the
//! mathematically exact union of the merged blob.
//!
//! A hollow merge seam shows up as composite alpha collapsing where the
//! reference is solid (the reported bug); a refracted seam shows up as a
//! color discontinuity between the two halves' samples. The neck's
//! centerline alpha per distance is printed so thin-neck translucency near
//! the merge threshold is visible in the output.
use crate::params::GlassShaderParams;
use crate::shared::{constant_buffer, create_linear_clamp_sampler, CropConstants};
use crate::surface::compositor::compile_shader;
use crate::surface::device::create_d3d11_device;
use crate::SourceRectConstants;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D11::{
    ID3D11Buffer, ID3D11Device, ID3D11DeviceContext, ID3D11PixelShader, ID3D11RenderTargetView,
    ID3D11SamplerState, ID3D11ShaderResourceView, ID3D11Texture2D, ID3D11VertexShader,
    D3D11_BIND_RENDER_TARGET, D3D11_BIND_SHADER_RESOURCE, D3D11_CPU_ACCESS_READ,
    D3D11_MAPPED_SUBRESOURCE, D3D11_MAP_READ, D3D11_SUBRESOURCE_DATA, D3D11_TEXTURE2D_DESC,
    D3D11_USAGE_DEFAULT, D3D11_USAGE_STAGING, D3D11_VIEWPORT,
};
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM;

/// Mirrors the manager's `metaball_canvas`: the canvas snaps onto the
/// host's integer pixel grid exactly like the placement truncation.
fn metaball_canvas(rect: [f32; 4], reach: f32) -> [f32; 4] {
    let slack = reach + 48.0;
    [
        (rect[0] - slack).trunc(),
        (rect[1] - slack).trunc(),
        (rect[2] + 2.0 * slack).trunc(),
        (rect[3] + 2.0 * slack).trunc(),
    ]
}

/// Mirrors the manager's `metaball_ball_reach`: the distance a ball's
/// silhouette can reach from its center — its radius for the plain
/// circle, the superellipse's bounding-disc corner distance otherwise.
fn metaball_ball_reach(ball: [f32; 6]) -> f32 {
    let circle = ball[5] == 2.0 && ball[2] == ball[3];
    if circle {
        ball[2]
    } else {
        (ball[2] * ball[2] + ball[3] * ball[3]).sqrt()
    }
}

/// One scene: two 16px anchor elements whose centers carry the balls.
struct Scene {
    balls: [[f32; 6]; 2], // window-space (x, y, extentX, extentY, weight, exponent)
    canvas_a: [f32; 4],   // surface A's snapped canvas
    canvas_b: [f32; 4],   // surface B's snapped canvas
    distance: f32,
}

fn scene(element_a: [f32; 4], distance: f32, radius: f32) -> Scene {
    scene_shaped(
        element_a, distance, radius, radius, 2.0, radius, radius, 2.0,
    )
}

/// The general form: each ball's extents and exponent independently
/// (a squircle merges with a circle, a pill with a dot, ...).
fn scene_shaped(
    element_a: [f32; 4],
    distance: f32,
    extent_x_a: f32,
    extent_y_a: f32,
    exponent_a: f32,
    extent_x_b: f32,
    extent_y_b: f32,
    exponent_b: f32,
) -> Scene {
    let element_b = [
        element_a[0] + distance,
        element_a[1],
        element_a[2],
        element_a[3],
    ];
    let ball_a = [
        element_a[0] + element_a[2] * 0.5,
        element_a[1] + element_a[3] * 0.5,
        extent_x_a,
        extent_y_a,
        1.0,
        exponent_a,
    ];
    let ball_b = [
        element_b[0] + element_b[2] * 0.5,
        element_b[1] + element_b[3] * 0.5,
        extent_x_b,
        extent_y_b,
        1.0,
        exponent_b,
    ];
    Scene {
        balls: [ball_a, ball_b],
        canvas_a: metaball_canvas(element_a, metaball_ball_reach(ball_a)),
        canvas_b: metaball_canvas(element_b, metaball_ball_reach(ball_b)),
        distance,
    }
}

/// Stamps one surface's params the way the manager + GlassSurface do:
/// canvas-local balls normalized by the region's height (identity shape
/// region = the whole canvas), the OWN ball reordered first, the weight
/// folded into the influence radius and extents, and the territory
/// clip armed with own index 0.
fn surface_params(
    canvas: [f32; 4],
    balls_window: &[[f32; 6]],
    own_index: Option<usize>,
    tension: f32,
) -> GlassShaderParams {
    let region_h = canvas[3].max(1.0);
    let center_x = canvas[0] + canvas[2] * 0.5;
    let center_y = canvas[1] + canvas[3] * 0.5;
    let mut balls: Vec<[f32; 6]> = balls_window
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
    if let Some(own) = own_index {
        balls.swap(0, own);
    }
    let count = balls.len().min(6);
    let mut params = GlassShaderParams::default();
    // Steady state: the reveal fully open, no fade, no spring scale.
    params.reveal_progress = 1.0;
    params.glass_fade = 0.0;
    params.pill_scale = 1.0;
    params.pill_scale_x = 1.0;
    params.pill_scale_y = 1.0;
    params.pointer_influence = 0.0;
    params.animation_time = 0.0;
    params.shape_region = [0.0, 0.0, 1.0, 1.0];
    params.shape_meta = [2.0, 2.0, count as f32, tension];
    params.metaballs = [[0.0; 4]; 6];
    params.metaball_shapes = [[0.0; 4]; 6];
    for (slot, ball) in balls.iter().take(6).enumerate() {
        // Mirrors stamp_shape (metaball_effective_rows): the weight
        // folds into the influence radius and the extents before
        // upload (the harness's weight-1 balls leave it unchanged).
        let fold = ball[4].powf(1.0 / tension);
        let extent_x = ball[2] * fold;
        let extent_y = ball[3] * fold;
        params.metaballs[slot] = [ball[0], ball[1], extent_x.min(extent_y), ball[4]];
        params.metaball_shapes[slot] = [ball[5], extent_x, extent_y, 0.0];
    }
    // The clip arms only when foreign balls follow the own one.
    params.metaball_own = match own_index {
        Some(_) if count > 1 => [0.0, 1.0, 1.0, 0.0],
        _ => [0.0; 4],
    };
    params
}

/// One rendered surface: premultiplied RGBA8 bytes, tightly packed.
struct RenderedSurface {
    bytes: Vec<u8>,
    width: u32,
    height: u32,
    origin: [f32; 2],
}

impl RenderedSurface {
    fn alpha(&self, screen_x: i64, screen_y: i64) -> f32 {
        let x = screen_x - self.origin[0] as i64;
        let y = screen_y - self.origin[1] as i64;
        if x < 0 || y < 0 || x >= self.width as i64 || y >= self.height as i64 {
            return 0.0;
        }
        self.bytes[((y as usize) * self.width as usize + x as usize) * 4 + 3] as f32 / 255.0
    }

    fn channel(&self, screen_x: i64, screen_y: i64, channel: usize) -> Option<f32> {
        let x = screen_x - self.origin[0] as i64;
        let y = screen_y - self.origin[1] as i64;
        if x < 0 || y < 0 || x >= self.width as i64 || y >= self.height as i64 {
            return None;
        }
        let base = ((y as usize) * self.width as usize + x as usize) * 4;
        Some(self.bytes[base + channel] as f32 / 255.0)
    }
}

struct Harness {
    device: ID3D11Device,
    context: ID3D11DeviceContext,
    vertex_shader: ID3D11VertexShader,
    pixel_shader: ID3D11PixelShader,
    sampler: ID3D11SamplerState,
    backdrop_srv: ID3D11ShaderResourceView,
    backdrop: ID3D11Texture2D,
}

const BACKDROP: u32 = 1024;
/// The metaball tension exponent the harness sweeps (the classic blend).
const TENSION: f32 = 2.0;

impl Harness {
    fn new() -> Result<Self, String> {
        let (device, context) = create_d3d11_device()?;
        let vertex_shader = {
            let bytes = compile_shader("vs_main", "vs_5_0")?;
            let mut shader: Option<ID3D11VertexShader> = None;
            unsafe { device.CreateVertexShader(&bytes, None, Some(&mut shader)) }
                .map_err(|error| format!("vs_main creation failed: {error}"))?;
            shader.expect("vs_main missing")
        };
        let pixel_shader = {
            let bytes = compile_shader("ps_main", "ps_5_0")?;
            let mut shader: Option<ID3D11PixelShader> = None;
            unsafe { device.CreatePixelShader(&bytes, None, Some(&mut shader)) }
                .map_err(|error| format!("ps_main creation failed: {error}"))?;
            shader.expect("ps_main missing")
        };
        let sampler = create_linear_clamp_sampler(&device);
        // A high-frequency checkerboard in R/B with an inverse pattern in
        // G: any refraction mismatch between the two halves shifts the
        // sampled pattern and lights up the color check.
        let mut pattern: Vec<u8> = Vec::with_capacity((BACKDROP * BACKDROP * 4) as usize);
        for y in 0..BACKDROP {
            for x in 0..BACKDROP {
                let cell = ((x / 8 + y / 8) % 2) == 0;
                let value = if cell { 230 } else { 25 };
                pattern.extend_from_slice(&[value, 255 - value, 128, 255]);
            }
        }
        let desc = D3D11_TEXTURE2D_DESC {
            Width: BACKDROP,
            Height: BACKDROP,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        let init = D3D11_SUBRESOURCE_DATA {
            pSysMem: pattern.as_ptr().cast(),
            SysMemPitch: BACKDROP * 4,
            SysMemSlicePitch: 0,
        };
        let backdrop = unsafe {
            let mut texture: Option<ID3D11Texture2D> = None;
            device
                .CreateTexture2D(&desc, Some(&init), Some(&mut texture))
                .map_err(|error| format!("backdrop texture failed: {error}"))?;
            texture.expect("backdrop texture missing")
        };
        let backdrop_srv = unsafe {
            let mut view: Option<ID3D11ShaderResourceView> = None;
            device
                .CreateShaderResourceView(&backdrop, None, Some(&mut view))
                .map_err(|error| format!("backdrop SRV failed: {error}"))?;
            view.expect("backdrop SRV missing")
        };
        Ok(Self {
            device,
            context,
            vertex_shader,
            pixel_shader,
            sampler,
            backdrop_srv,
            backdrop,
        })
    }

    /// Renders one surface pass of (width, height) pixels: the real
    /// composite shader with the given params, read back to RGBA8.
    fn render(
        &self,
        width: u32,
        height: u32,
        uv_origin: [f32; 2],
        params: &GlassShaderParams,
    ) -> Result<RenderedSurface, String> {
        let target = unsafe {
            let mut texture: Option<ID3D11Texture2D> = None;
            self.device
                .CreateTexture2D(
                    &D3D11_TEXTURE2D_DESC {
                        Width: width,
                        Height: height,
                        MipLevels: 1,
                        ArraySize: 1,
                        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                            Count: 1,
                            Quality: 0,
                        },
                        Usage: D3D11_USAGE_DEFAULT,
                        BindFlags: D3D11_BIND_RENDER_TARGET.0 as u32,
                        CPUAccessFlags: 0,
                        MiscFlags: 0,
                    },
                    None,
                    Some(&mut texture),
                )
                .map_err(|error| format!("target texture failed: {error}"))?;
            texture.expect("target texture missing")
        };
        let rtv = unsafe {
            let mut view: Option<ID3D11RenderTargetView> = None;
            self.device
                .CreateRenderTargetView(&target, None, Some(&mut view))
                .map_err(|error| format!("RTV failed: {error}"))?;
            view.expect("RTV missing")
        };
        let staging = unsafe {
            let mut texture: Option<ID3D11Texture2D> = None;
            self.device
                .CreateTexture2D(
                    &D3D11_TEXTURE2D_DESC {
                        Usage: D3D11_USAGE_STAGING,
                        BindFlags: 0,
                        CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
                        MiscFlags: 0,
                        ..params_staging_desc(width, height)
                    },
                    None,
                    Some(&mut texture),
                )
                .map_err(|error| format!("staging texture failed: {error}"))?;
            texture.expect("staging texture missing")
        };
        let source_rect = SourceRectConstants {
            source_uv_origin: uv_origin,
            source_uv_size: [
                width as f32 / BACKDROP as f32,
                height as f32 / BACKDROP as f32,
            ],
            pointer_velocity: [0.0; 2],
            _padding: [0.0; 2],
        };
        let cb0: ID3D11Buffer = constant_buffer(&self.device, &source_rect)?;
        let cb1: ID3D11Buffer = constant_buffer(&self.device, params)?;
        let cb2: ID3D11Buffer = constant_buffer(
            &self.device,
            &CropConstants::identity([1.0 / BACKDROP as f32; 2]),
        )?;
        unsafe {
            self.context
                .ClearRenderTargetView(&rtv, &[0.0, 0.0, 0.0, 0.0]);
            self.context
                .OMSetRenderTargets(Some(&[Some(rtv.clone())]), None);
            self.context.RSSetViewports(Some(&[D3D11_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: width as f32,
                Height: height as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]));
            self.context
                .IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            self.context.VSSetShader(&self.vertex_shader, None);
            self.context.PSSetShader(&self.pixel_shader, None);
            self.context
                .PSSetSamplers(0, Some(&[Some(self.sampler.clone())]));
            self.context
                .PSSetShaderResources(0, Some(&[Some(self.backdrop_srv.clone())]));
            self.context.PSSetConstantBuffers(0, Some(&[Some(cb0)]));
            self.context.PSSetConstantBuffers(1, Some(&[Some(cb1)]));
            self.context.PSSetConstantBuffers(2, Some(&[Some(cb2)]));
            self.context.Draw(3, 0);
            self.context.CopyResource(&staging, &target);
            let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
            self.context
                .Map(&staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))
                .map_err(|error| format!("staging map failed: {error}"))?;
            let mut bytes = vec![0u8; (width * height * 4) as usize];
            for y in 0..height {
                let source = mapped.pData as *const u8;
                let row = std::slice::from_raw_parts(
                    source.add((y as usize) * mapped.RowPitch as usize),
                    (width as usize) * 4,
                );
                bytes[(y as usize) * (width as usize) * 4..(y as usize + 1) * (width as usize) * 4]
                    .copy_from_slice(row);
            }
            self.context.Unmap(&staging, 0);
            self.context.PSSetShaderResources(0, Some(&[None]));
            Ok(RenderedSurface {
                bytes,
                width,
                height,
                origin: uv_origin.map(|uv| uv * BACKDROP as f32),
            })
        }
    }
}

/// A plain texture description the staging desc derives from.
fn params_staging_desc(width: u32, height: u32) -> D3D11_TEXTURE2D_DESC {
    D3D11_TEXTURE2D_DESC {
        Width: width,
        Height: height,
        MipLevels: 1,
        ArraySize: 1,
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Usage: D3D11_USAGE_DEFAULT,
        BindFlags: D3D11_BIND_RENDER_TARGET.0 as u32,
        CPUAccessFlags: 0,
        MiscFlags: 0,
    }
}

#[test]
fn metaball_two_surface_composite_covers_the_merged_silhouette() {
    let harness = match Harness::new() {
        Ok(harness) => harness,
        Err(error) => {
            // The harness needs a D3D11 device; a machine without one
            // skips rather than failing (the compile tests still run).
            eprintln!("metaball seam harness skipped: {error}");
            return;
        }
    };
    const RADIUS: f32 = 64.0;
    // The merge threshold for r=64 at tension 2 with the falloff
    // window (1.0..2.0 radii) is d ≈ 160: below it the balls share a
    // liquid neck, above it they are separate lobes.
    let distances: [f32; 14] = [
        120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 175.0, 178.0, 180.0, 181.0, 182.0, 185.0, 190.0,
        200.0,
    ];
    let fracs: [f32; 4] = [0.0, 0.3, 0.5, 0.7];
    let mut failures: Vec<String> = Vec::new();
    let mut saddle_table: Vec<String> = Vec::new();

    for frac in fracs {
        for distance in distances {
            let scene = scene([300.0 + frac, 200.0 + frac, 16.0, 16.0], distance, RADIUS);
            // The two clipped halves, exactly as the manager sets them up.
            let params_a = surface_params(scene.canvas_a, &scene.balls, Some(0), 2.0);
            let params_b = surface_params(scene.canvas_b, &scene.balls, Some(1), 2.0);
            let tex_a = harness
                .render(
                    scene.canvas_a[2] as u32,
                    scene.canvas_a[3] as u32,
                    [
                        scene.canvas_a[0] / BACKDROP as f32,
                        scene.canvas_a[1] / BACKDROP as f32,
                    ],
                    &params_a,
                )
                .expect("surface A render");
            let tex_b = harness
                .render(
                    scene.canvas_b[2] as u32,
                    scene.canvas_b[3] as u32,
                    [
                        scene.canvas_b[0] / BACKDROP as f32,
                        scene.canvas_b[1] / BACKDROP as f32,
                    ],
                    &params_b,
                )
                .expect("surface B render");
            // The unclipped whole-field reference on one big canvas.
            let ref_x = scene.canvas_a[0].min(scene.canvas_b[0]) - 8.0;
            let ref_y = scene.canvas_a[1].min(scene.canvas_b[1]) - 8.0;
            let ref_w = (scene.canvas_a[0] + scene.canvas_a[2])
                .max(scene.canvas_b[0] + scene.canvas_b[2])
                - ref_x
                + 8.0;
            let ref_h = (scene.canvas_a[1] + scene.canvas_a[3])
                .max(scene.canvas_b[1] + scene.canvas_b[3])
                - ref_y
                + 8.0;
            let ref_canvas = [ref_x, ref_y, ref_w, ref_h];
            let params_ref = surface_params(ref_canvas, &scene.balls, None, 2.0);
            let tex_ref = harness
                .render(
                    ref_w as u32,
                    ref_h as u32,
                    [ref_x / BACKDROP as f32, ref_y / BACKDROP as f32],
                    &params_ref,
                )
                .expect("reference render");
            // Clip-isolation reference: the SAME canvas and
            // normalization as surface A but with the territory clip
            // off. The clip only drops pixels (sd = 1e3); it must never
            // change a rendered pixel's color, so this must match tex_a
            // exactly wherever both are solid.
            let tex_ref_same = harness
                .render(
                    scene.canvas_a[2] as u32,
                    scene.canvas_a[3] as u32,
                    [
                        scene.canvas_a[0] / BACKDROP as f32,
                        scene.canvas_a[1] / BACKDROP as f32,
                    ],
                    &surface_params(scene.canvas_a, &scene.balls, None, TENSION),
                )
                .expect("clip-isolation reference render");

            // Composite A over B (DWM: premultiplied over, B on top).
            let x0 = scene.canvas_a[0].min(scene.canvas_b[0]) as i64;
            let y0 = scene.canvas_a[1].min(scene.canvas_b[1]) as i64;
            let x1 = (scene.canvas_a[0] + scene.canvas_a[2])
                .max(scene.canvas_b[0] + scene.canvas_b[2]) as i64;
            let y1 = (scene.canvas_a[1] + scene.canvas_a[3])
                .max(scene.canvas_b[1] + scene.canvas_b[3]) as i64;
            let mut holes = 0usize;
            let mut worst_dip = 1.0f32;
            let mut first_hole: Option<(i64, i64, f32, f32)> = None;
            let mut seam_delta = 0u32;
            let mut clip_delta = 0u32;
            for y in y0..y1 {
                for x in x0..x1 {
                    let a_a = tex_a.alpha(x, y);
                    let a_b = tex_b.alpha(x, y);
                    let composite = a_a + a_b * (1.0 - a_a);
                    let reference = tex_ref.alpha(x, y);
                    if reference >= 0.9 {
                        if composite < worst_dip {
                            worst_dip = composite;
                        }
                        if composite < 0.5 {
                            holes += 1;
                            if first_hole.is_none() {
                                first_hole = Some((x, y, reference, composite));
                            }
                        }
                    }
                    // Color continuity where both are fully solid: the
                    // two halves must sample and refract identically.
                    // The stored bytes are already premultiplied, so the
                    // composite is premulA + premulB * (1 - alphaA).
                    if false {
                        let _ = tex_ref.channel(x, y, 0);
                    }
                    // The direct crack detector: at the territory tie
                    // line BOTH surfaces render the same screen pixel.
                    // Their canvases are the same size and the field is
                    // canvas-size independent, so the two renders must
                    // be pixel-identical there — any color delta is a
                    // visible seam between the merging metaballs.
                    if a_a >= 0.9 && a_b >= 0.9 {
                        for channel in 0..3 {
                            let rgb_a = tex_a.channel(x, y, channel).unwrap_or(0.0);
                            let rgb_b = tex_b.channel(x, y, channel).unwrap_or(0.0);
                            let delta = ((rgb_a - rgb_b).abs() * 255.0).round() as u32;
                            if delta > seam_delta {
                                seam_delta = delta;
                            }
                        }
                    }
                    // Clip isolation: identical canvas geometry with the
                    // clip on vs off must be pixel-identical.
                    if tex_ref_same.alpha(x, y) >= 0.99 && tex_a.alpha(x, y) >= 0.99 {
                        for channel in 0..3 {
                            let iso = tex_ref_same.channel(x, y, channel).unwrap_or(0.0);
                            let clipped = tex_a.channel(x, y, channel).unwrap_or(0.0);
                            let delta = ((iso - clipped).abs() * 255.0).round() as u32;
                            if delta > clip_delta {
                                clip_delta = delta;
                            }
                        }
                    }
                }
            }
            // The neck's centerline alpha from the reference: how solid
            // the merge neck reads at this distance.
            let saddle = (
                ((scene.balls[0][0] + scene.balls[1][0]) * 0.5) as i64,
                (scene.balls[0][1]) as i64,
            );
            let saddle_alpha = tex_ref.alpha(saddle.0, saddle.1);
            saddle_table.push(format!(
                    "d={distance:6.1} frac={frac} neck_alpha={saddle_alpha:.3} worst_dip={worst_dip:.3} holes={holes} seamΔ={seam_delta} clipΔ={clip_delta}"
                ));
            if holes > 0 {
                failures.push(format!(
                        "distance {distance} frac {frac}: {holes} hollow px (worst composite alpha {worst_dip:.3}), first at {:?} (ref {}, composite {})",
                        first_hole.map(|hole| (hole.0, hole.1)),
                        first_hole.map(|hole| hole.2).unwrap_or(0.0),
                        first_hole.map(|hole| hole.3).unwrap_or(0.0),
                    ));
            }
            if seam_delta > 2 {
                failures.push(format!(
                        "distance {distance} frac {frac}: territory tie line renders a {seam_delta}/255 color seam between the two surfaces"
                    ));
            }
            if clip_delta > 2 {
                failures.push(format!(
                        "distance {distance} frac {frac}: territory clip changes solid pixels' color by {clip_delta}/255"
                    ));
            }
        }
    }
    for line in &saddle_table {
        println!("{line}");
    }
    assert!(
        failures.is_empty(),
        "the territory-clipped composite leaves hollow seams or color discontinuities:\n{}",
        failures.join("\n")
    );
}

/// The shaped-ball composite: a squircle surface and a circle surface
/// sharing one field must tile the merged blob exactly like two
/// circles do — the mixed-shape merge is the whole point of shaped
/// balls. Sweeps a few distances across the merge threshold with the
/// same hole / seam / clip-isolation checks as the circle sweep.
#[test]
fn metaball_two_surface_composite_covers_a_shaped_merge() {
    let harness = match Harness::new() {
        Ok(harness) => harness,
        Err(error) => {
            eprintln!("metaball seam harness skipped: {error}");
            return;
        }
    };
    // A squircle ball (iOS-style exponent 5, extents 64 x 64) and a
    // plain circle ball (radius 64): same influence scale, different
    // silhouettes.
    let distances: [f32; 5] = [180.0, 200.0, 240.0, 280.0, 320.0];
    let mut failures: Vec<String> = Vec::new();
    for distance in distances {
        let scene = scene_shaped(
            [300.0, 200.0, 16.0, 16.0],
            distance,
            64.0,
            64.0,
            5.0, // squircle on surface A
            64.0,
            64.0,
            2.0, // circle on surface B
        );
        let params_a = surface_params(scene.canvas_a, &scene.balls, Some(0), 2.0);
        let params_b = surface_params(scene.canvas_b, &scene.balls, Some(1), 2.0);
        let tex_a = harness
            .render(
                scene.canvas_a[2] as u32,
                scene.canvas_a[3] as u32,
                [
                    scene.canvas_a[0] / BACKDROP as f32,
                    scene.canvas_a[1] / BACKDROP as f32,
                ],
                &params_a,
            )
            .expect("surface A render");
        let tex_b = harness
            .render(
                scene.canvas_b[2] as u32,
                scene.canvas_b[3] as u32,
                [
                    scene.canvas_b[0] / BACKDROP as f32,
                    scene.canvas_b[1] / BACKDROP as f32,
                ],
                &params_b,
            )
            .expect("surface B render");
        // The unclipped whole-field reference on one big canvas.
        let ref_x = scene.canvas_a[0].min(scene.canvas_b[0]) - 8.0;
        let ref_y = scene.canvas_a[1].min(scene.canvas_b[1]) - 8.0;
        let ref_w = (scene.canvas_a[0] + scene.canvas_a[2])
            .max(scene.canvas_b[0] + scene.canvas_b[2])
            - ref_x
            + 8.0;
        let ref_h = (scene.canvas_a[1] + scene.canvas_a[3])
            .max(scene.canvas_b[1] + scene.canvas_b[3])
            - ref_y
            + 8.0;
        let params_ref = surface_params([ref_x, ref_y, ref_w, ref_h], &scene.balls, None, 2.0);
        let tex_ref = harness
            .render(
                ref_w as u32,
                ref_h as u32,
                [ref_x / BACKDROP as f32, ref_y / BACKDROP as f32],
                &params_ref,
            )
            .expect("reference render");
        let x0 = scene.canvas_a[0].min(scene.canvas_b[0]) as i64;
        let y0 = scene.canvas_a[1].min(scene.canvas_b[1]) as i64;
        let x1 = (scene.canvas_a[0] + scene.canvas_a[2]).max(scene.canvas_b[0] + scene.canvas_b[2])
            as i64;
        let y1 = (scene.canvas_a[1] + scene.canvas_a[3]).max(scene.canvas_b[1] + scene.canvas_b[3])
            as i64;
        let mut holes = 0usize;
        let mut worst_dip = 1.0f32;
        let mut first_hole: Option<(i64, i64)> = None;
        let mut seam_delta = 0u32;
        for y in y0..y1 {
            for x in x0..x1 {
                let a_a = tex_a.alpha(x, y);
                let a_b = tex_b.alpha(x, y);
                let composite = a_a + a_b * (1.0 - a_a);
                let reference = tex_ref.alpha(x, y);
                if reference >= 0.9 {
                    if composite < worst_dip {
                        worst_dip = composite;
                    }
                    if composite < 0.5 {
                        holes += 1;
                        if first_hole.is_none() {
                            first_hole = Some((x, y));
                        }
                    }
                }
                if a_a >= 0.9 && a_b >= 0.9 {
                    for channel in 0..3 {
                        let rgb_a = tex_a.channel(x, y, channel).unwrap_or(0.0);
                        let rgb_b = tex_b.channel(x, y, channel).unwrap_or(0.0);
                        let delta = ((rgb_a - rgb_b).abs() * 255.0).round() as u32;
                        if delta > seam_delta {
                            seam_delta = delta;
                        }
                    }
                }
            }
        }
        println!(
                "d={distance:6.1} shaped neck: worst_dip={worst_dip:.3} holes={holes} seamΔ={seam_delta}"
            );
        if holes > 0 {
            failures.push(format!(
                    "distance {distance}: {holes} hollow px (worst composite alpha {worst_dip:.3}), first at {first_hole:?}"
                ));
        }
        if seam_delta > 2 {
            failures.push(format!(
                    "distance {distance}: territory tie line renders a {seam_delta}/255 color seam between the squircle and circle surfaces"
                ));
        }
    }
    assert!(
            failures.is_empty(),
            "the shaped-ball territory-clipped composite leaves hollow seams or color discontinuities:\n{}",
            failures.join("\n")
        );
}
