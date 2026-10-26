// The pre-blurred backdrop produced by the dual-Kawase chain (kawase.hlsl):
// the capture cropped to this surface's surroundings and blurred at the
// chain's base resolution. The composite only ever samples this texture.
Texture2D<float4> glassBackdrop : register(t0);
SamplerState linearClamp : register(s0);

cbuffer SourceRect : register(b0) {
    float2 sourceUvOrigin;
    float2 sourceUvSize;
    // Screen-pixel delta collected with GetCursorPos for the current frame.
    float2 pointerVelocity;
    float2 sourceRectPadding;
};

// All tunable values arrive from Rust via GlassShaderParams.
cbuffer GlassParams : register(b1) {
    float refractionStrength;
    // Kept for constant-buffer layout parity with GlassShaderParams; the
    // blur itself now runs in the Kawase chain (kawaseScale in b2).
    float blurRadiusTexels;
    float tintStrength;
    float revealProgress;
    float3 glassTint;
    float chromaticOffsetTexels;
    float chromaticBlend;
    float capsuleRadius;
    float pillSoftness;
    float edgeBand;
    float pointerInfluence;
    float outlineStrength;
    float outlineAngleDegrees;
    float outlineMirror;
    float outlineThicknessPx;
    float outlineFadePx;
    float outlineFalloff;
    float revealWobble;
    float turbulenceStrength;
    float turbulenceScale;
    float2 revealOrigin;
    // Uniform shape scale anchored to the surface's bottom-center (1 = full
    // size); the auto-hide reveal scales the pill in from ~85%.
    float pillScale;
    // Anisotropic squash/stretch factors per axis on top of pillScale
    // (1.0 = undistorted); the taskbar's resize spring drives these.
    float pillScaleX;
    float pillScaleY;
    // Seconds of animation time, ticking while the reveal animates; drives
    // the flowing wobble/turbulence.
    float animationTime;
    // Fade-style reveal: 1.0 makes the reveal a pure opacity dissolve (the
    // shape is fully open from the first frame and the reveal progress ramps
    // the glass alpha); 0.0 keeps the bubble/shape reveal.
    float glassFade;
    // Kawase chain level count (1..4); read by the compositor in Rust, not
    // by this shader. Kept in the buffer for GlassShaderParams layout parity.
    float blurSamples;
    // Uniform black border around the silhouette: no light response, full
    // opacity at the rim fading inward over its thickness (0 thickness or
    // 0 opacity disables it). Occupies the former padding floats so the
    // buffer stays eight 16-byte rows.
    float borderThicknessPx;
    float borderOpacity;
    // Adaptive luminance-aware blend (vibrancy): how strongly the tint
    // strength adapts per pixel to the contrast between the backdrop's
    // luminance and the tint's (0 = uniform tint, the legacy alpha blend).
    float vibrancy;
    // Chroma boost applied to the backdrop before the tint blend (0..1):
    // the glass re-saturates the background by its own chroma.
    float vibrancySaturation;
    // Strength of the rim's inner contact shadow on the side turned away
    // from the light (0 disables it).
    float rimShadowStrength;
    // Layout parity with GlassShaderParams' reserved padding.
    float vibrancyPadding;
};

// Geometry of the Kawase chain's crop region, uploaded by the compositor
// alongside the per-pass constants in kawase.hlsl. Displacement math stays
// in capture-texel units (captureTexelUv) and only the final sample is
// mapped into the crop's uv space.
cbuffer CropParams : register(b2) {
    float2 cropOriginUv;
    float2 cropSizeUv;
    float2 captureTexelUv;
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

// Maps a capture-space uv into the blurred backdrop's own uv space.

float2 backdropUv(float2 captureUv) {
    return (captureUv - cropOriginUv) / cropSizeUv;
}

// One tap into the backdrop: the pre-blurred Kawase texture (whose clamped
// sampler finishes the final 2x upsample), or the sharp capture when the
// blur is off.
float3 sampleBackdrop(float2 captureUv) {
    return glassBackdrop.Sample(linearClamp, backdropUv(captureUv)).rgb;
}

float hash21(float2 p) {
    return frac(sin(dot(p, float2(127.1, 311.7))) * 43758.5453123);
}

float valueNoise(float2 p) {
    const float2 cell = floor(p);
    const float2 f = frac(p);
    const float2 smooth = f * f * (3.0 - 2.0 * f);
    return lerp(
        lerp(hash21(cell), hash21(cell + float2(1.0, 0.0)), smooth.x),
        lerp(hash21(cell + float2(0.0, 1.0)), hash21(cell + float2(1.0, 1.0)), smooth.x),
        smooth.y);
}

float fbm(float2 p) {
    float sum = 0.0;
    float amplitude = 0.55;
    [unroll]
    for (int octave = 0; octave < 3; octave++) {
        sum += valueNoise(p) * amplitude;
        p = p * 2.03 + float2(19.7, 7.3);
        amplitude *= 0.5;
    }
    return sum;
}

// Rounded-rect signed distance for a point in capsule space.
float roundedRectSd(float2 position, float aspect, float radius) {
    const float2 bounds = float2(aspect * 0.5 - radius, 0.5 - radius);
    const float2 q = abs(position) - bounds;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - radius;
}

// The rounded-rect SDF's own gradient: the outward normal of the edge the
// point is nearest. Perpendicular to a straight edge everywhere, turning
// radial only on the rounded end caps — unlike a center-out direction,
// which on a wide pill runs nearly parallel to the long edges.
float2 roundedRectNormal(float2 position, float aspect, float radius) {
    const float2 bounds = float2(aspect * 0.5 - radius, 0.5 - radius);
    const float2 q = abs(position) - bounds;
    if (q.x > 0.0 && q.y > 0.0) {
        return normalize(sign(position) * q);
    }
    if (q.x > q.y) {
        return float2(sign(position.x), 0.0);
    }
    return float2(0.0, sign(position.y));
}

float4 ps_main(VertexOutput input) : SV_Target {
    // Capture-texel units: all displacement math stays in the capture's own
    // uv space and is mapped into the crop only at the final sample.
    const float2 texel = captureTexelUv;
    // Surface size in capture pixels. The capsule SDF must be isotropic in
    // surface pixels, so the aspect ratio comes from the surface itself —
    // never from the capture-to-surface uv scale, which would bake the
    // monitor's aspect ratio into the shape.
    const float surfaceWidthPx = sourceUvSize.x / captureTexelUv.x;
    const float surfaceHeightPx = sourceUvSize.y / captureTexelUv.y;
    const float aspect = surfaceWidthPx / surfaceHeightPx;
    const float2 baseUv = sourceUvOrigin + input.uv * sourceUvSize;
    const float2 capsulePosition = (input.uv - 0.5) * float2(aspect, 1.0);
    // Scale the glass shape uniformly around its bottom-center so the reveal
    // can grow the pill from slightly smaller, with anisotropic squash and
    // stretch on top (the resize spring). Scaling the position (rather than
    // the SDF result) keeps every downstream effect — refraction, outline,
    // reveal warp — consistent with the scaled shape.
    const float2 scaleOrigin = float2(0.0, 0.5);
    const float2 pillScaleVec = float2(pillScale * pillScaleX, pillScale * pillScaleY);
    const float2 scaledPosition = (capsulePosition - scaleOrigin) / pillScaleVec + scaleOrigin;
    // Standard rounded-rect signed distance (valid for both rounded rects
    // and stadiums, where the y bounds collapse to zero).
    const float roundedRectDistance = roundedRectSd(scaledPosition, aspect, capsuleRadius);
    // Reveal: the glass bubbles out of its origin point and grows into the
    // full shape. Three ingredients build the liquid feel:
    //   1. a noise-lobed blob expanding from the origin — liquid mass that
    //      surges out ahead of the shape,
    //   2. satellite bubbles that inflate on staggered timers, drift outward
    //      and pop into the main shape,
    //   3. the uniformly scaled rounded rect, whose rim is warped by
    //      screen-space fbm: as it grows, the boundary crawls through the
    //      fixed noise field, so the lobes visibly travel, split and merge
    //      instead of riding along with the scale.
    // Everything is intersected with the final rounded rect (max of the
    // distances), so the warp can never exceed the border radius at any
    // point of the animation. The revealMask additionally guarantees zero
    // alpha (no pixels written at all) once the glass is fully collapsed.
    const float eased = revealProgress * revealProgress * (3.0 - 2.0 * revealProgress);
    const float revealEnergy = 1.0 - eased;
    // Fade-style reveal: the shape stays fully open and the reveal progress
    // ramps the glass alpha instead (a plain dissolve, no bubble).
    const float shapeEased = lerp(eased, 1.0, glassFade);
    const float revealAlpha = lerp(1.0, eased, glassFade);
    const float2 revealOriginPx = float2(
        (revealOrigin.x - 0.5) * aspect,
        revealOrigin.y - 0.5
    );
    const float2 revealRelative = scaledPosition - revealOriginPx;
    const float revealDistance = length(revealRelative);
    // Rim turbulence, sampled in screen space (surface-normalized units) so
    // the growing boundary sweeps through a fixed noise field.
    const float turbulenceFrequency = turbulenceScale * 3.5;
    const float lobeAmplitude =
        (turbulenceStrength / surfaceHeightPx) * revealEnergy * 0.8;
    const float lobe = (fbm(
        scaledPosition * turbulenceFrequency * 1.7
        + float2(animationTime * 1.1, animationTime * 0.7)) - 0.5) * lobeAmplitude;
    // Expanding blob: a liquid mass surging out of the origin, slightly
    // off-axis so the growth reads as a surge rather than a balloon.
    const float blobRadius = (0.30 + 0.14 * fbm(float2(animationTime * 0.9, 4.2)))
        * smoothstep(0.0, 0.55, eased);
    const float2 blobCenter = revealOriginPx
        + normalize(revealRelative + 1e-5) * blobRadius * 0.25;
    const float blobSd = length(scaledPosition - blobCenter) - blobRadius - lobe;
    // Satellite bubbles: each inflates on a staggered slice of the timeline
    // (the sine window grows it, then pops it), drifting outward from the
    // origin in a pseudo-random direction before merging into the shape.
    float bubblesSd = 1e3;
    [unroll]
    for (int bubble = 0; bubble < 3; bubble++) {
        const float seed = float(bubble) * 2.7 + 1.0;
        const float local = saturate((eased - 0.08 - float(bubble) * 0.16) / 0.45);
        const float arc = sin(3.14159265 * local);
        const float angle = hash21(float2(seed, 3.7)) * 6.2831853;
        const float radius = (0.05 + 0.05 * hash21(float2(seed, 9.1))) * arc;
        const float2 center = revealOriginPx
            + float2(cos(angle), sin(angle))
            * local * (0.16 + 0.14 * hash21(float2(seed, 5.3)));
        bubblesSd = min(bubblesSd, length(scaledPosition - center) - radius);
    }
    // In fade mode the bubble and satellites are disabled entirely: their
    // distances are pushed beyond the surface so the intersection below
    // reduces to the scaled rounded rect alone.
    const float fadeBlobSd = lerp(blobSd, 1e3, glassFade);
    const float fadeBubblesSd = lerp(bubblesSd, 1e3, glassFade);
    // The uniformly scaled rounded rect the blob morphs into as the reveal
    // completes; its rim gets the same turbulence lobes.
    const float revealScale = max(shapeEased, 0.02);
    const float2 revealedPosition =
        (scaledPosition - revealOriginPx) / revealScale + revealOriginPx;
    const float scaledSd = roundedRectSd(revealedPosition, aspect, capsuleRadius) - lobe;
    // Intersecting with the unscaled rounded rect keeps the animation from
    // ever spilling outside the surface's final bounds.
    const float revealShape = min(min(scaledSd, fadeBlobSd), fadeBubblesSd);
    const float signedDistance = max(revealShape, roundedRectDistance);
    const float revealMask = smoothstep(0.0, 0.1, eased);
    // Shape coverage. The AA softness is clamped to a constant ~1.25px: a
    // normalized softness stretches the silhouette fringe (and the outline's
    // white riding it) in proportion to the surface height, which made the
    // rim read soft and fat on large containers and crisp on small ones.
    const float edgeSoftness = min(pillSoftness, 1.25 / surfaceHeightPx);
    const float pillMask =
        (1.0 - smoothstep(0.0, edgeSoftness, signedDistance))
        * lerp(revealMask, revealAlpha, glassFade);
    const float distanceToBorder = max(-signedDistance, 0.0);
    // Edge refraction profile: displacement is strongest at the rim and
    // falls off quadratically inward, so its slope (what visibly bends
    // content) is steepest right at the edge rather than mid-band.
    const float edgeT = saturate(1.0 - distanceToBorder / max(edgeBand, 1e-5));
    const float edgeStrength = edgeT * edgeT;
    // Refraction follows the true edge normal: content near an edge is
    // stretched straight away from that edge (magnifying toward the rim),
    // fading to zero displacement toward the center via edgeStrength.
    const float2 edgeNormal = roundedRectNormal(scaledPosition, aspect, capsuleRadius);
    float2 displacement = edgeNormal * texel * refractionStrength * edgeStrength;
    displacement += pointerVelocity * texel * pointerInfluence * edgeStrength;
    // Liquid content warp during the reveal: a traveling radial pulse
    // emanating from the reveal origin (the captured content ripples
    // outward like a droplet hitting water) plus domain-scrolled fractal
    // shimmer, both fading out as the glass finishes opening.
    displacement += normalize(revealRelative + 1e-5) * texel * revealWobble
        * revealEnergy * sin(revealDistance * 34.0 - animationTime * 9.0) * 5.0;
    displacement += float2(
        fbm(baseUv * turbulenceScale * 12.0 + animationTime * 0.6) - 0.5,
        fbm(baseUv * turbulenceScale * 12.0 + 11.3 - animationTime * 0.5) - 0.5
    ) * texel * turbulenceStrength * revealEnergy;
    const float2 refractedUv = baseUv - displacement;
    // One tap into the pre-blurred Kawase backdrop — the heavy lifting (24
    // Vogel taps per pixel) now happens in the chain's low-resolution
    // passes; the composite only displaces, tints and masks.
    const float3 blurred = sampleBackdrop(refractedUv);
    // Dispersion is intentionally almost imperceptible and exists only at the edge.
    const float2 chromatic = edgeNormal * texel * edgeStrength * chromaticOffsetTexels;
    const float red = sampleBackdrop(refractedUv - chromatic).r;
    const float blue = sampleBackdrop(refractedUv + chromatic).b;
    const float3 dispersed = lerp(blurred, float3(red, blurred.g, blue), edgeStrength * chromaticBlend);
    // ---- Adaptive Luminance-Aware Blend (vibrancy) ----
    // A uniform tint ignores what is behind the glass: a bright window washes
    // the glass out and a dark one turns it into a flat black block. Instead,
    // the backdrop sample that is already in `dispersed` is worked per pixel:
    //   1. Vibrancy: re-saturate the backdrop by its own chroma, so the glass
    //      keeps the colorful character of what is behind it.
    //   2. Luminance-aware strength: the more a pixel's luminance disagrees
    //      with the tint's luminance (contrast), the more tint is pulled in
    //      for that pixel — bright content behind a dark glass darkens the
    //      glass there, dark content lets it stay clear. Every pixel adapts
    //      toward the legible band, so foreground elements stay legible.
    // With vibrancy = 0 and vibrancySaturation = 0 this collapses to the
    // legacy uniform `lerp(dispersed, glassTint, tintStrength)`.
    const float backdropLuma = dot(dispersed, float3(0.2126, 0.7152, 0.0722));
    const float tintLuma = dot(glassTint, float3(0.2126, 0.7152, 0.0722));
    const float3 vibrant = dispersed
        + (dispersed - backdropLuma) * vibrancySaturation;
    const float lumaContrast = abs(backdropLuma - tintLuma);
    const float adaptiveStrength = saturate(
        tintStrength * lerp(1.0, 1.0 + lumaContrast * 2.0, vibrancy));
    const float3 tinted = lerp(vibrant, glassTint, adaptiveStrength);
    // ---- Specular rim lighting ----
    // The border is a raised, polished ridge: a quarter-round bevel filling
    // the outermost outlineThicknessPx of the glass. Its surface normal
    // tilts from fully outward at the crest (the rim) to fully up where it
    // merges into the flat glass, so a Phong lobe against the light lights
    // the ridge the way light catches a physical polished edge: a glint
    // whose core sits just below the crest, whose brightness turns smoothly
    // around corners as the edge normal rotates away from the light, and
    // which slides slightly down the slope on oblique edges. The light is
    // added over the glass (never blended toward flat white), so it reads as
    // illumination landing on the surface rather than a drawn border line.
    const float signedDistancePx = signedDistance * surfaceHeightPx;
    const float distanceToBorderPx = max(-signedDistancePx, 0.0);
    const float lightAngle = radians(outlineAngleDegrees);
    const float2 lightDir = float2(cos(lightAngle), sin(lightAngle));
    // The light sits 30 degrees off the glass plane: low enough that grazing
    // light catches the polished crest hardest, high enough that the bevel
    // slope still answers on oblique edges.
    const float3 frontLight = normalize(float3(lightDir, 0.5));
    const float3 backLight = normalize(float3(-lightDir, 0.5));
    // Quarter-round bevel normal. At bevel depth u (0 = the crest, 1 = the
    // bevel's inner edge) the normal is (1 - u) outward along the edge
    // normal and sqrt(2u - u^2) up — components that already square-sum to
    // one, so no re-normalization is needed.
    const float bevelWidth = max(outlineThicknessPx, 1e-4);
    const float bevelU = saturate(distanceToBorderPx / bevelWidth);
    const float3 bevelNormal = float3(
        edgeNormal * (1.0 - bevelU),
        sqrt(max(2.0 * bevelU - bevelU * bevelU, 0.0)));
    // The border point facing the light: the corner-circle center in the
    // light's quadrant, pushed out along the light direction onto the rim.
    const float2 capsuleBounds = float2(aspect * 0.5 - capsuleRadius, 0.5 - capsuleRadius);
    const float2 anchor = sign(lightDir) * capsuleBounds + lightDir * capsuleRadius;
    // Fade distance along the border from the lit corner. Auto = a quarter
    // of the half-perimeter, so the same share of the border is lit on any
    // surface size; an explicit pixel fade is capped there so the front and
    // mirrored lobes can never overlap and sum into double brightness on
    // small surfaces.
    const float autoFadePx = (surfaceWidthPx + surfaceHeightPx) * 0.25;
    const float fadePx = outlineFadePx > 0.0
        ? min(outlineFadePx, autoFadePx)
        : autoFadePx;
    const float frontFade = 1.0 - smoothstep(0.0, fadePx, length(scaledPosition - anchor) * surfaceHeightPx);
    const float backFade = 1.0 - smoothstep(0.0, fadePx, length(scaledPosition + anchor) * surfaceHeightPx);
    // The polish: a tight specular glint over a soft broad sheen. The glint
    // carries the "light catching the ridge" read; the sheen keeps the whole
    // lit slope visibly raised between glints.
    const float gloss = lerp(12.0, 64.0, saturate(outlineFalloff));
    const float frontFacing = saturate(dot(bevelNormal, frontLight));
    const float backFacing = saturate(dot(bevelNormal, backLight));
    const float frontGlint = pow(frontFacing, gloss) * 1.15;
    const float backGlint = pow(backFacing, gloss * 2.0) * 0.55;
    const float sheen = 0.25 * (pow(frontFacing, 3.0) + pow(backFacing, 3.0));
    // The lobes only exist on the bevel itself: bandMask eases them out
    // across the bevel width so the flat glass interior stays unlit.
    const float bandMask = 1.0 - smoothstep(0.0, bevelWidth, distanceToBorderPx);
    const float3 rimLight = float3(1.0, 1.0, 1.0)
        * saturate(((frontGlint + sheen) * frontFade
            + (outlineMirror * backGlint + sheen) * backFade)
            * bandMask * outlineStrength);
    // A uniform black border owning the outermost band of the silhouette:
    // the same SDF band mechanics but with no light response — full border
    // opacity at the rim, easing inward over its thickness. It reads as the
    // ridge's shaded outer wall.
    const float borderMask =
        1.0 - smoothstep(0.0, max(borderThicknessPx, 1e-4), distanceToBorderPx);
    const float3 bordered = lerp(tinted, float3(0.0, 0.0, 0.0), saturate(borderMask * borderOpacity));
    // Contact shadow: the bevel slope turned away from the light falls into
    // shade, deepening toward the crest and vanishing as the normal tilts
    // back up toward the glass plane. This is what sells the ridge as raised.
    const float rimShadow = pow(saturate(-dot(bevelNormal, frontLight)), 2.0)
        * bandMask * rimShadowStrength;
    // Rim lighting applies after the contour so the glint rides over the
    // shaded outer wall exactly where a polished edge catches light.
    const float3 shaded = bordered * (1.0 - rimShadow) + rimLight;
    // The SDF controls coverage and distortion only; it never adds an edge
    // color. The composition swap chain expects premultiplied alpha, so the
    // color is scaled by the capsule coverage for smooth antialiased edges.
    return float4(saturate(shaded) * pillMask, pillMask);
}
