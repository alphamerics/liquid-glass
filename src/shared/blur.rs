//! The dual-Kawase blur chain: the engine-level blur that runs ONCE per
//! fresh capture frame and is sampled by every composite for free.
//!
//! [`BlurEngine`] owns the shaders, the sampler and the mip chain; the
//! plan ([`BlurPlan`]) derives the chain's depth and resolution from the
//! material params and the quality tier. The chain's own down/upsampling
//! carries most of the blur, so the level depth stays minimal and the
//! base resolution stays near the capture's own.

use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D11::{
    ID3D11Buffer, ID3D11Device, ID3D11DeviceContext, ID3D11PixelShader, ID3D11RenderTargetView,
    ID3D11ShaderResourceView, ID3D11Texture2D, ID3D11VertexShader, D3D11_BIND_RENDER_TARGET,
    D3D11_BIND_SHADER_RESOURCE, D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT, D3D11_VIEWPORT,
};

use crate::params::GlassShaderParams;
use crate::quality::QualityCaps;

use super::backdrop::{Backdrop, CropConstants};
use super::d3d::{compiled_shader, constant_buffer, create_linear_clamp_sampler};

pub(crate) const KAWASE_HLSL: &str = include_str!("../kawase.hlsl");

/// How the Kawase chain should run for the current params.
#[derive(Clone, Copy, PartialEq)]
pub(crate) struct BlurPlan {
    /// Blur is effectively off: composites sample the sharp capture
    /// directly, skipping the chain entirely.
    pub(crate) sharp: bool,
    /// Kawase levels actually used (1..=5). Capped by the radius: the
    /// chain's own down/upsampling imposes a minimum blur per level, so a
    /// small radius must use fewer levels no matter the samples setting.
    /// One more level than a pure-hop estimate: the chain's shallowest
    /// buffer runs at base resolution (its hops are undone by the up side
    /// there), so each band needs that extra level to reach the same blur
    /// at the same slider position.
    pub(crate) levels: usize,
    /// Resolution of the chain's base (down0) level relative to the capture
    /// (0.9..1.0). Less blur hides less resampling, so lower radii run the
    /// chain at a higher resolution to avoid visible pixelation.
    pub(crate) base_scale: f32,
    /// Kawase tap-offset scale.
    pub(crate) offset_scale: f32,
}

/// Derives the chain configuration from the shader params. The samples
/// setting is an upper bound on levels; the radius picks the actual depth.
/// The chain's own down/upsampling carries most of the blur, so the level
/// depth stays minimal and the base resolution near the capture's own.
///
/// The per-band ramps make the slider's response CONTINUOUS: within each
/// level band the tap offsets ramp from their start value to 2.0 while the
/// base resolution eases down to 0.9, and each deeper band starts at
/// offset 0.9 / base 1.0 — which, in the `sigma ≈ offsets · 2^levels ·
/// base` model, lands exactly on the previous band's end. Adding a level
/// therefore never snaps: the new level begins at nearly full resolution
/// with tiny taps (a gentle addition) and grows gradually from there.
pub(crate) fn blur_plan_with_caps(params: &GlassShaderParams, caps: &QualityCaps) -> BlurPlan {
    let radius = params.blur_radius_texels;
    if radius < 0.5 {
        return BlurPlan {
            sharp: true,
            levels: 1,
            base_scale: 1.0,
            offset_scale: 0.0,
        };
    }
    let samples = (params.blur_samples.round() as usize).clamp(1, 5);
    // One extra level covers radii up to ~12px; each extra level roughly
    // doubles the reach (bands 12-28, 28-64, 64+). `levels` counts the
    // chain's buffers, so a band of `band_levels` 2x-ish hops runs
    // `band_levels + 1` buffers (down0 crops at the base scale, not at 2x).
    let (band_levels, band_start, band_end) = if radius < 12.0 {
        (1, 1.0, 12.0)
    } else if radius < 28.0 {
        (2, 12.0, 28.0)
    } else if radius < 64.0 {
        (3, 28.0, 64.0)
    } else {
        (4, 64.0, 128.0)
    };
    let position = ((radius - band_start) / (band_end - band_start)).clamp(0.0, 1.0);
    // Band 1 ramps from the sharp-side minimum (0.25); deeper bands start at
    // the base-scale floor (0.9) to match the previous band's end.
    let offsets_start = if band_levels == 1 { 0.25 } else { 0.9 };
    let offset_scale = (offsets_start + (2.0 - offsets_start) * position).clamp(0.25, 2.0);
    // Quality tier: the chain runs at a reduced internal resolution (the
    // composite upsamples it anyway — a blurred backdrop has no fine detail
    // to lose). The reduction ramps in over radius 12..24 so a slider sweep
    // never jumps when it engages, and small radii keep full resolution
    // where resampling would be visible.
    let ramp = ((radius - 12.0) / 12.0).clamp(0.0, 1.0);
    let tier_scale = 1.0 - ramp * (1.0 - caps.blur_internal_scale);
    let base_scale = ((1.0 - 0.1 * position) * tier_scale).clamp(0.25, 1.0);
    let levels = samples.min(band_levels + 1);
    BlurPlan {
        sharp: false,
        levels,
        base_scale,
        offset_scale,
    }
}

/// One mip of the Kawase chain: the texture plus the views its pass and the
/// next pass need.
struct KawaseLevel {
    #[allow(dead_code)]
    texture: ID3D11Texture2D,
    srv: ID3D11ShaderResourceView,
    rtv: ID3D11RenderTargetView,
}

/// The intermediate buffers of the dual-Kawase blur over the (possibly
/// cropped) capture. The down/up halves run at successively halved
/// resolutions with the shallowest pair (down0/up0) at the full base
/// resolution — the chain's output, which the composites sample directly.
/// Recreated whenever the blurred region's size or the level count changes
/// (monitor change, a moved surface union, or params that move the level
/// count).
struct KawaseChain {
    /// Downsampling targets: down[i] is the source at 1/2^i resolution.
    down: Vec<KawaseLevel>,
    /// Upsampling targets: up[i] matches down[i]'s size; up[i] blends
    /// down[i+1] (deepest) or up[i+1] back toward down[i]'s resolution.
    up: Vec<KawaseLevel>,
    /// down0/up0 size in pixels: the blurred region scaled by the plan's
    /// base scale.
    width: u32,
    height: u32,
    /// Configured level count (1..=5).
    levels: usize,
}

/// The engine-level Kawase machinery: shaders, sampler, chain and the two
/// constant buffers its passes read. Created once for the whole process;
/// every surface's composite then samples its output for free. The broker
/// additionally renders this same engine's output into the shared pair
/// (its publication pass below).
pub(crate) struct BlurEngine {
    device: ID3D11Device,
    vertex_shader: ID3D11VertexShader,
    crop_shader: ID3D11PixelShader,
    down_shader: ID3D11PixelShader,
    up_shader: ID3D11PixelShader,
    /// The identity resample used to write a (lower-resolution) backdrop
    /// into the publication target at the monitor's full resolution.
    stretch_shader: ID3D11PixelShader,
    sampler: windows::Win32::Graphics::Direct3D11::ID3D11SamplerState,
    crop: ID3D11Buffer,
    crop_current: CropConstants,
    chain: Option<KawaseChain>,
    /// The frame size and plan the current chain was built for.
    chain_frame: [u32; 2],
    chain_plan: Option<BlurPlan>,
}

impl BlurEngine {
    pub(crate) fn new(device: &ID3D11Device) -> Self {
        let compile = |entry: &str| -> ID3D11PixelShader {
            let mut shader = None;
            // SAFETY: the bytecode comes from the cached compile below; a
            // failure here is a shader/toolchain defect surfaced at first
            // use, and the shared engine is created before any surface.
            unsafe {
                device
                    .CreatePixelShader(
                        &compiled_shader(KAWASE_HLSL, entry, "ps_5_0")
                            .expect("the Kawase shader should compile"),
                        None,
                        Some(&mut shader),
                    )
                    .expect("Could not create the Kawase pixel shader");
            }
            shader.expect("Kawase pixel shader missing")
        };
        let mut vertex_shader = None;
        unsafe {
            device
                .CreateVertexShader(
                    &compiled_shader(KAWASE_HLSL, "vs_main", "vs_5_0")
                        .expect("the Kawase vertex shader should compile"),
                    None,
                    Some(&mut vertex_shader),
                )
                .expect("Could not create the Kawase vertex shader");
        }
        let sampler = create_linear_clamp_sampler(device);
        let crop_current = CropConstants::identity([0.0, 0.0]);
        let crop =
            constant_buffer(device, &crop_current).expect("Could not create the blur crop buffer");
        Self {
            device: device.clone(),
            vertex_shader: vertex_shader.expect("Kawase vertex shader missing"),
            crop_shader: compile("ps_kawase_crop"),
            down_shader: compile("ps_kawase_down"),
            up_shader: compile("ps_kawase_up"),
            stretch_shader: compile("ps_stretch"),
            sampler,
            crop,
            crop_current,
            chain: None,
            chain_frame: [0, 0],
            chain_plan: None,
        }
    }

    /// Renders (or refreshes) the shared blurred backdrop for one capture
    /// frame. Cheap when nothing changed: the plan and the chain are
    /// fingerprinted and the passes only run on change. `crop_px` (x0, y0,
    /// x1, y1 in capture pixels, `None` = the full frame) restricts both the
    /// chain's textures and its sampling to that region — a crop is a
    /// region, not a scale, so tap offsets keep their capture-texel
    /// semantics — and the returned backdrop carries the matching
    /// `CropConstants`.
    pub(crate) fn render(
        &mut self,
        context: &ID3D11DeviceContext,
        capture: &ID3D11ShaderResourceView,
        frame: [u32; 2],
        crop_px: Option<[u32; 4]>,
        params: &GlassShaderParams,
        caps: &QualityCaps,
    ) -> Result<Backdrop, String> {
        let texel_uv = [1.0 / frame[0].max(1) as f32, 1.0 / frame[1].max(1) as f32];
        let plan = blur_plan_with_caps(params, caps);
        if plan.sharp {
            return Ok(Backdrop {
                view: capture.clone(),
                crop: CropConstants::identity(texel_uv),
            });
        }
        // The chain's base resolution scales the CROPPED region's pixel
        // extent; a crop only discards texels the composites cannot sample
        // (they were outside every surface's reach).
        let crop_rect = match crop_px {
            Some(px @ [x0, y0, x1, y1]) if x1 > x0 && y1 > y0 => Some(px),
            _ => None,
        };
        let region = crop_rect.map_or(frame, |[x0, y0, x1, y1]| [x1 - x0, y1 - y0]);
        let width = (region[0] as f32 * plan.base_scale).ceil().max(1.0) as u32;
        let height = (region[1] as f32 * plan.base_scale).ceil().max(1.0) as u32;
        let mut crop_current = crop_rect.map_or_else(
            || CropConstants::identity(texel_uv),
            |px| CropConstants::from_pixel_rect(px, frame, texel_uv),
        );
        crop_current.kawase_scale = plan.offset_scale;
        if crop_current != self.crop_current {
            unsafe {
                context.UpdateSubresource(
                    &self.crop,
                    0,
                    None,
                    (&crop_current as *const CropConstants).cast(),
                    0,
                    0,
                );
            }
            self.crop_current = crop_current;
        }
        if self.chain.is_none()
            || self.chain_frame != [width, height]
            || self.chain_plan.as_ref() != Some(&plan)
        {
            self.chain = Some(build_kawase_chain(
                &self.device,
                width,
                height,
                plan.levels,
            )?);
            self.chain_frame = [width, height];
            self.chain_plan = Some(plan);
        }
        let chain = self.chain.as_ref().expect("kawase chain missing");
        unsafe {
            context.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            context.VSSetShader(&self.vertex_shader, None);
            context.PSSetSamplers(0, Some(&[Some(self.sampler.clone())]));
            context.PSSetConstantBuffers(2, Some(&[Some(self.crop.clone())]));
            for (level, target) in chain.down.iter().enumerate() {
                let (width, height) = chain_target_size(chain, level);
                if level == 0 {
                    draw_kawase_pass(context, &self.crop_shader, target, capture, width, height);
                } else {
                    let source = &chain.down[level - 1].srv;
                    draw_kawase_pass(context, &self.down_shader, target, source, width, height);
                }
            }
            // Up side: expand from the deepest level back to down0's
            // resolution. down0/up0 run at the chain's full base
            // resolution, so the chain's output is up0 — the composites
            // sample it directly, and a single-level chain (no up passes)
            // hands them down0 instead.
            for level in (0..chain.up.len()).rev() {
                let source = if level + 1 == chain.up.len() {
                    &chain.down[chain.levels - 1].srv
                } else {
                    &chain.up[level + 1].srv
                };
                let (width, height) = chain_target_size(chain, level);
                draw_kawase_pass(
                    context,
                    &self.up_shader,
                    &chain.up[level],
                    source,
                    width,
                    height,
                );
            }
        }
        let view = match chain.up.first() {
            Some(shallowest) => &shallowest.srv,
            // A single level has no up passes; down0 is the output.
            None => &chain.down[0].srv,
        };
        Ok(Backdrop {
            view: view.clone(),
            crop: crop_current,
        })
    }

    /// The broker's publication pass: writes a (possibly blur-reduced)
    /// backdrop into the monitor-sized shared target through a plain
    /// bilinear resample. The shared fullscreen pipeline `vs_main` +
    /// `ps_stretch` — identical geometry to the chain's own passes, with
    /// no crop constants of interest to it.
    pub(crate) fn render_publication(
        &self,
        context: &ID3D11DeviceContext,
        backdrop: &ID3D11ShaderResourceView,
        target: &ID3D11RenderTargetView,
        width: u32,
        height: u32,
    ) -> Result<(), String> {
        if width == 0 || height == 0 {
            return Err("The publication target has no extent".to_owned());
        }
        unsafe {
            context.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            context.VSSetShader(&self.vertex_shader, None);
            context.PSSetShader(&self.stretch_shader, None);
            context.PSSetSamplers(0, Some(&[Some(self.sampler.clone())]));
            context.OMSetRenderTargets(Some(&[Some(target.clone())]), None);
            context.RSSetViewports(Some(&[D3D11_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: width as f32,
                Height: height as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]));
            context.PSSetShaderResources(0, Some(&[Some(backdrop.clone())]));
            context.Draw(3, 0);
            // Unbind the backdrop's view: the pipeline may render it again
            // later as a render target in the same command stream.
            context.PSSetShaderResources(0, Some(&[None]));
        }
        Ok(())
    }
}

/// The pixel size of Kawase chain level `level` (down and up buffers share
/// sizes): down0 runs at the chain's full base resolution and every deeper
/// level halves it.
fn chain_target_size(chain: &KawaseChain, level: usize) -> (u32, u32) {
    let shift = level.min(16) as u32;
    (
        chain.width.div_ceil(1u32 << shift),
        chain.height.div_ceil(1u32 << shift),
    )
}

/// One fullscreen Kawase pass: binds `target` as the render target at its
/// own viewport and `source` at t0, then draws the shared fullscreen
/// triangle.
fn draw_kawase_pass(
    context: &ID3D11DeviceContext,
    pixel_shader: &ID3D11PixelShader,
    target: &KawaseLevel,
    source: &ID3D11ShaderResourceView,
    width: u32,
    height: u32,
) {
    unsafe {
        context.OMSetRenderTargets(Some(&[Some(target.rtv.clone())]), None);
        context.RSSetViewports(Some(&[D3D11_VIEWPORT {
            TopLeftX: 0.0,
            TopLeftY: 0.0,
            Width: width as f32,
            Height: height as f32,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        }]));
        context.PSSetShader(pixel_shader, None);
        context.PSSetShaderResources(0, Some(&[Some(source.clone())]));
        context.Draw(3, 0);
    }
}

fn create_kawase_level(
    device: &ID3D11Device,
    width: u32,
    height: u32,
) -> Result<KawaseLevel, String> {
    let mut texture = None;
    unsafe {
        device.CreateTexture2D(
            &D3D11_TEXTURE2D_DESC {
                Width: width,
                Height: height,
                MipLevels: 1,
                ArraySize: 1,
                Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Usage: D3D11_USAGE_DEFAULT,
                BindFlags: (D3D11_BIND_SHADER_RESOURCE.0 | D3D11_BIND_RENDER_TARGET.0) as u32,
                CPUAccessFlags: 0,
                MiscFlags: 0,
            },
            None,
            Some(&mut texture),
        )
    }
    .map_err(|error| format!("Could not create the Kawase chain texture: {error}"))?;
    let texture = texture.expect("Kawase chain texture missing");
    let mut srv = None;
    unsafe {
        device
            .CreateShaderResourceView(&texture, None, Some(&mut srv))
            .map_err(|error| format!("Could not create the Kawase chain SRV: {error}"))?;
    }
    let mut rtv = None;
    unsafe {
        device
            .CreateRenderTargetView(&texture, None, Some(&mut rtv))
            .map_err(|error| format!("Could not create the Kawase chain RTV: {error}"))?;
    }
    Ok(KawaseLevel {
        texture,
        srv: srv.expect("Kawase chain SRV missing"),
        rtv: rtv.expect("Kawase chain RTV missing"),
    })
}

fn build_kawase_chain(
    device: &ID3D11Device,
    width: u32,
    height: u32,
    levels: usize,
) -> Result<KawaseChain, String> {
    // down[i] is the source at 1/2^i; up[i] re-expands back to down[i]'s
    // size, sourced from down[i+1] (the first up) or up[i+1]. down0/up0 run
    // at the full base resolution, so the chain's output is up0 (down0 for
    // a single level) and no separate finishing pass is needed.
    let mut down = Vec::with_capacity(levels);
    let mut up = Vec::with_capacity(levels.saturating_sub(1));
    for level in 0..levels {
        let shift = level.min(16) as u32;
        down.push(create_kawase_level(
            device,
            width.div_ceil(1u32 << shift),
            height.div_ceil(1u32 << shift),
        )?);
    }
    for level in 0..levels.saturating_sub(1) {
        let shift = level.min(16) as u32;
        up.push(create_kawase_level(
            device,
            width.div_ceil(1u32 << shift),
            height.div_ceil(1u32 << shift),
        )?);
    }
    Ok(KawaseChain {
        down,
        up,
        width,
        height,
        levels,
    })
}
