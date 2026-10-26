// Window preview thumbnail: the captured app window rendered as an
// alpha-masked rounded rect, replacing the DWM thumbnail sprite (whose
// corners are locked to the system corner radius and ignore window
// regions). The mask is an in-shader SDF, so any radius works and edges
// stay antialiased through premultiplied composition alpha.

Texture2D<float4> sourceTexture : register(t0);
SamplerState linearClamp : register(s0);

// Mirrors the Rust ThumbConstants struct (register b0).
cbuffer ThumbParams : register(b0) {
    // The client-area crop, in capture UV space: the WGC per-window item
    // covers the whole window frame (title bar included), which is cropped
    // out here like the old thumbnail's rcSource.
    float2 cropOriginUv;
    float2 cropSizeUv;
    // Corner radius normalized to the quad's height (the same units the
    // glass surface uses), and the quad's width/height aspect.
    float cornerRadius;
    float quadAspect;
    float2 thumbPadding;
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

float4 ps_main(VertexOutput input) : SV_Target {
    // Rounded-rect SDF over the quad, aspect-corrected so the corners are
    // circular in screen space. d runs negative inside, 0 on the edge.
    const float2 p = (input.uv - 0.5) * float2(quadAspect, 1.0);
    const float radius = clamp(cornerRadius, 0.0, 0.5);
    const float2 halfExtent = float2(quadAspect, 1.0) * 0.5 - radius;
    const float2 q = abs(p) - halfExtent;
    const float d = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - radius;
    // One fwidth of d antialiases the edge through premultiplied alpha.
    const float aa = length(float2(ddx(d), ddy(d)));
    const float alpha = saturate(0.5 - d / max(aa, 1e-6));

    const float2 captureUv = cropOriginUv + input.uv * cropSizeUv;
    const float4 color = sourceTexture.Sample(linearClamp, captureUv);
    // Premultiplied alpha: the swap chain composites with
    // DXGI_ALPHA_MODE_PREMULTIPLIED, so the mask must scale the color too.
    return float4(color.rgb * alpha, alpha);
}
