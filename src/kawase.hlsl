// Dual-Kawase blur chain, rendered before the liquid composite:
//
//   capture --ps_kawase_crop--> down0 (base crop) --ps_kawase_down--> down1
//   (half) --ps_kawase_down--> down2 (quarter) --ps_kawase_up--> up1
//   (half) --ps_kawase_up--> up0 (base) --ps_kawase_up--> polish (full
//   base) --ps_stretch (publication)--> back buffer.
//
// Every pass touches 5 (down) or 8 (up) taps instead of the 27 the
// composite used to spend per pixel on its Vogel-disk blur, and the taps run
// at a fraction of the pixel count so the whole blur costs a fraction of one
// full-resolution fill. The final polish pass re-expands the shallowest
// chain level back to the base resolution: without it the composite's
// backdrop would be the half-resolution up0 magnified 2x by a plain bilinear
// stretch, which shows up as blocky, pixelated content behind the glass.
//

Texture2D<float4> sourceTexture : register(t0);
SamplerState linearClamp : register(s0);

// Mirrors the Rust CropConstants struct: geometry of the crop region this
// chain blurs, expressed in capture UV space.
cbuffer CropParams : register(b2) {
    float2 cropOriginUv;
    float2 cropSizeUv;
    float2 captureTexelUv;
    // Scales every Kawase tap offset; derived from blur_radius_texels.
    float kawaseScale;
    // Unused padding, kept for the Rust struct's 48-byte buffer layout.
    float4 cropPadding;
};

struct VertexOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

VertexOutput vs_main(uint vertexId : SV_VertexID) {
    const float2 positions[3] = {
        float2(-1.0, -1.0),
        float2(-1.0,  3.0),
        float2( 3.0, -1.0),
    };
    const float2 uvs[3] = {
        float2(0.0, 1.0),
        float2(0.0, -1.0),
        float2(2.0, 1.0),
    };

    VertexOutput output;
    output.position = float4(positions[vertexId], 0.0, 1.0);
    output.uv = uvs[vertexId];
    return output;
}

// First pass: crops the capture down to the chain's base resolution over
// the crop region. Corner taps sit at the tent-downsample position (half a
// destination texel; two source texels for a 2x hop) but never tighter than
// one source texel: a narrower spread leaves the halving without a
// prefilter and it aliases into blocky, pixelated output.
// weights 4/1/1/1/1.
float4 ps_kawase_crop(VertexOutput input) : SV_Target {
    const float2 captureUv = cropOriginUv + input.uv * cropSizeUv;
    const float2 offset = captureTexelUv * max(kawaseScale, 1.0);
    float4 sum = sourceTexture.Sample(linearClamp, captureUv) * 4.0;
    sum += sourceTexture.Sample(linearClamp, captureUv + float2(-offset.x, -offset.y));
    sum += sourceTexture.Sample(linearClamp, captureUv + float2( offset.x, -offset.y));
    sum += sourceTexture.Sample(linearClamp, captureUv + float2(-offset.x,  offset.y));
    sum += sourceTexture.Sample(linearClamp, captureUv + float2( offset.x,  offset.y));
    return sum / 8.0;
}

// Intermediate down pass: source and destination differ by 2x, so corner
// taps at one source texel land half a destination texel out. As in the
// crop pass, the taps never collapse tighter than one source texel or the
// halving aliases.
float4 ps_kawase_down(VertexOutput input) : SV_Target {
    uint width;
    uint height;
    sourceTexture.GetDimensions(width, height);
    const float2 offset = max(kawaseScale, 1.0) / float2(width, height);
    float4 sum = sourceTexture.Sample(linearClamp, input.uv) * 4.0;
    sum += sourceTexture.Sample(linearClamp, input.uv + float2(-offset.x, -offset.y));
    sum += sourceTexture.Sample(linearClamp, input.uv + float2( offset.x, -offset.y));
    sum += sourceTexture.Sample(linearClamp, input.uv + float2(-offset.x,  offset.y));
    sum += sourceTexture.Sample(linearClamp, input.uv + float2( offset.x,  offset.y));
    return sum / 8.0;
}

// Up pass: the tent-weighted dual-filter upsample, offsets in source texel
// units (the source here is the smaller texture), weights 1/2 alternating.
float4 ps_kawase_up(VertexOutput input) : SV_Target {
    uint width;
    uint height;
    sourceTexture.GetDimensions(width, height);
    const float2 texel = kawaseScale / float2(width, height);
    float4 sum = sourceTexture.Sample(linearClamp, input.uv + float2(-2.0,  0.0) * texel);
    sum += sourceTexture.Sample(linearClamp, input.uv + float2(-1.0,  1.0) * texel) * 2.0;
    sum += sourceTexture.Sample(linearClamp, input.uv + float2( 0.0,  2.0) * texel);
    sum += sourceTexture.Sample(linearClamp, input.uv + float2( 1.0,  1.0) * texel) * 2.0;
    sum += sourceTexture.Sample(linearClamp, input.uv + float2( 2.0,  0.0) * texel);
    sum += sourceTexture.Sample(linearClamp, input.uv + float2( 1.0, -1.0) * texel) * 2.0;
    sum += sourceTexture.Sample(linearClamp, input.uv + float2( 0.0, -2.0) * texel);
    sum += sourceTexture.Sample(linearClamp, input.uv + float2(-1.0, -1.0) * texel) * 2.0;
    return sum / 12.0;
}

// Identity stretch: resamples the source across the full viewport with the
// bilinear sampler. The broker uses it to publish the (possibly
// blur-reduced) backdrop at the fixed full-monitor shared resolution so
// consumers' shared-texture geometry never depends on the material.
float4 ps_stretch(VertexOutput input) : SV_Target {
    return sourceTexture.Sample(linearClamp, input.uv);
}

