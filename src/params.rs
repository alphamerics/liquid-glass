//! All tunable values for the liquid glass material and surface sizing.
/// Shader floats uploaded to the pixel shader's `GlassParams` constant
/// buffer (register b1). Every field maps 1:1 to a uniform in
/// `liquid_composition.hlsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlassShaderParams {
    /// Strength of the edge refraction offset, in texels.
    pub refraction_strength: f32,
    /// Gaussian blur sigma, in captured texels. The kernel reaches one sigma
    /// from the center, so higher values blur deeper and cleaner.
    pub blur_radius_texels: f32,
    /// How strongly the tint color is mixed in (0..1).
    pub tint_strength: f32,
    /// Reveal animation progress (0 = collapsed bubble, 1 = fully open).
    /// Overridden per frame by the compositor while animating.
    pub reveal_progress: f32,
    /// RGB tint color mixed over the glass.
    pub glass_tint: [f32; 3],
    /// Chromatic aberration offset at the edge, in texels.
    pub chromatic_offset_texels: f32,
    /// How far chromatic aberration reaches inward from the edge.
    pub chromatic_blend: f32,
    /// Capsule SDF radius in normalized surface units (0.5 = full pill).
    pub capsule_radius: f32,
    /// Anti-aliased softness of the capsule edge, in normalized units. The
    /// shader clamps the effective softness to a constant ~1.25px so the rim
    /// (and the outline riding it) stays equally crisp on any surface size.
    pub pill_softness: f32,
    /// Width of the refracting edge band, in normalized units.
    pub edge_band: f32,
    /// How strongly cursor movement bends the glass (texel-uv scale per
    /// screen-pixel of cursor delta). 0 disables pointer-driven refraction.
    pub pointer_influence: f32,
    /// Brightness of the angular gradient outline (0 disables it).
    pub outline_strength: f32,
    /// Light direction for the outline, in degrees. 0 = right, 90 = down,
    /// 180 = left. Acts as an opacity falloff: border edges facing the light
    /// (within a 60 degree flat cone) are fully opaque, easing to invisible
    /// across the terminator; the line's width never varies.
    pub outline_angle_degrees: f32,
    /// Strength of the mirrored outline lobe on the opposite side, as a
    /// fraction of `outline_strength` (1.0 = a full mirrored line).
    pub outline_mirror: f32,
    /// Outline line thickness in surface pixels.
    pub outline_thickness_px: f32,
    /// Distance in surface pixels over which the outline's opacity fades out
    /// from its lit corner anchor along the border. `0.0` auto-sizes to a
    /// quarter of the half-perimeter (the same border share lit on any
    /// surface size); explicit values are capped there so the front and
    /// mirrored lobes never overlap.
    pub outline_fade_px: f32,
    /// Sharpness of the outline's angular terminator: higher values ease
    /// the opacity falloff to zero more tightly around the light direction.
    pub outline_falloff: f32,
    /// Amplitude of the sine wobble during the reveal, in texels.
    pub reveal_wobble: f32,
    /// Turbulence warp amplitude during the reveal, in surface pixels.
    pub turbulence_strength: f32,
    /// Turbulence noise frequency (higher = smaller, busier bubbles).
    pub turbulence_scale: f32,
    /// Reveal origin in normalized surface coordinates (0..1). The bubble
    /// grows out of this point; defaults to the bottom-right corner.
    pub reveal_origin: [f32; 2],
    /// Uniform scale of the glass shape, anchored to its bottom-center.
    /// 1.0 = normal; 0.85 = 15% smaller (the auto-hide reveal scales in).
    pub pill_scale: f32,
    /// Anisotropic squash/stretch factors on top of `pill_scale`, per axis
    /// (1.0 = undistorted). Drives the taskbar's resize spring: width
    /// changes overshoot along x while y counter-squashes to preserve the
    /// pill's apparent volume.
    pub pill_scale_x: f32,
    pub pill_scale_y: f32,
    /// Seconds of animation time, ticking while the reveal animates. Drives
    /// the flowing wobble/turbulence; overridden by the compositor.
    pub animation_time: f32,
    /// Fade-style reveal: 1.0 turns the reveal into a pure opacity dissolve
    /// (the shape is fully open from the first frame and the reveal progress
    /// ramps the glass alpha), 0.0 keeps the bubble/shape reveal.
    pub glass_fade: f32,
    /// Number of Kawase blur levels (1..=4). Each level halves the
    /// resolution and roughly doubles the blur reach; more levels give a
    /// smoother, deeper blur for the same tap count. Rounded and clamped by
    /// the compositor.
    pub blur_samples: f32,
    /// Uniform black border around the silhouette: thickness in surface
    /// pixels (0 disables it) and constant opacity (0..1). Drawn on top of
    /// the angular outline with no light response. Occupies the struct's
    /// former reserved padding so the constant buffer stays eight 16-byte
    /// rows.
    pub border_thickness_px: f32,
    pub border_opacity: f32,
}

/// Configurable reveal animation, reusable by any liquid glass surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RevealAnimation {
    /// Seconds for the glass to fully open (or close).
    pub duration_secs: f32,
    /// Sine wobble amplitude on the captured content, in texels.
    pub wobble: f32,
    /// Turbulence warp amplitude on the shape and content, in surface pixels.
    pub turbulence_strength: f32,
    /// Turbulence noise frequency.
    pub turbulence_scale: f32,
    /// Where the bubble grows from, in normalized surface coordinates.
    pub origin: [f32; 2],
}

impl Default for RevealAnimation {
    fn default() -> Self {
        Self {
            duration_secs: 0.34,
            wobble: 1.4,
            turbulence_strength: 12.0,
            turbulence_scale: 2.6,
            origin: [1.0, 1.0],
        }
    }
}

impl Default for GlassShaderParams {
    fn default() -> Self {
        Self {
            refraction_strength: 9.0,
            blur_radius_texels: 8.0,
            tint_strength: 0.24,
            reveal_progress: 1.0,
            glass_tint: [1.0, 0.25, 0.25],
            chromatic_offset_texels: 1.15,
            chromatic_blend: 1.0,
            capsule_radius: 0.5,
            pill_softness: 0.02,
            edge_band: 0.35,
            pointer_influence: 0.0,
            outline_strength: 1.75,
            outline_angle_degrees: 280.0,
            outline_mirror: 1.0,
            outline_thickness_px: 2.0,
            outline_fade_px: 0.0,
            outline_falloff: 1.0,
            reveal_wobble: 1.4,
            turbulence_strength: 12.0,
            turbulence_scale: 2.6,
            reveal_origin: [1.0, 1.0],
            pill_scale: 1.0,
            pill_scale_x: 1.0,
            pill_scale_y: 1.0,
            animation_time: 0.0,
            glass_fade: 0.0,
            blur_samples: 4.0,
            border_thickness_px: 1.0,
            border_opacity: 0.85,
        }
    }
}

/// The shared glass material: every appearance value that is global to the
/// session rather than per-surface. One broker-owned `material.json` holds
/// it; every glass app's engine watches that file and pushes it onto its
/// compositors, so a settings edit in one app restyles every app's glass.
///
/// Surface-specific state (reveal progress, capsule radius, pill scale,
/// animation time, pointer influence, glass fade, reveal origin) is NOT
/// part of the material: [`Material::apply_onto`] preserves it.
///
/// The JSON keys are camelCase and deliberately mirror the legacy Glassbar
/// settings keys, so `from_legacy_glassbar` is a straight key-for-key map.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Material {
    /// Strength of the edge refraction offset, in texels.
    pub refraction_strength: f32,
    /// Gaussian blur sigma, in captured texels.
    pub blur_radius: f32,
    /// Kawase blur chain level count (1..=4).
    pub blur_samples: f32,
    /// How strongly the tint color is mixed in (0..1).
    pub tint_strength: f32,
    /// RGB tint color mixed over the glass.
    #[serde(rename = "tintColor")]
    pub glass_tint: [f32; 3],
    /// Width of the refracting edge band, in normalized units.
    pub edge_band: f32,
    /// Chromatic aberration offset at the edge, in texels.
    pub chromatic_offset: f32,
    /// How far chromatic aberration reaches inward from the edge.
    pub chromatic_blend: f32,
    /// Brightness of the angular gradient outline (0 disables it).
    pub outline_strength: f32,
    /// Outline line thickness in surface pixels.
    pub outline_thickness: f32,
    /// Light direction for the outline, in degrees.
    pub outline_angle: f32,
    /// Strength of the mirrored outline lobe, as a fraction of strength.
    pub outline_mirror: f32,
    /// Distance in surface pixels over which the outline's opacity fades
    /// along the border.
    pub outline_fade: f32,
    /// Uniform black border thickness in surface pixels (0 disables it).
    pub border_thickness: f32,
    /// Uniform black border opacity (0..1).
    pub border_opacity: f32,
    /// Turbulence warp amplitude, in surface pixels.
    pub turbulence_strength: f32,
    /// Turbulence noise frequency (higher = smaller, busier bubbles).
    pub turbulence_scale: f32,
    /// Amplitude of the sine wobble during the reveal, in texels.
    #[serde(rename = "wobble")]
    pub reveal_wobble: f32,
}

impl Default for Material {
    fn default() -> Self {
        Self::from_params(&GlassShaderParams::default())
    }
}

impl Material {
    /// Extracts the material fields from a params set (the caller's values
    /// are expected unscaled — full-resolution texel units).
    pub fn from_params(params: &GlassShaderParams) -> Self {
        Self {
            refraction_strength: params.refraction_strength,
            blur_radius: params.blur_radius_texels,
            blur_samples: params.blur_samples,
            tint_strength: params.tint_strength,
            glass_tint: params.glass_tint,
            edge_band: params.edge_band,
            chromatic_offset: params.chromatic_offset_texels,
            chromatic_blend: params.chromatic_blend,
            outline_strength: params.outline_strength,
            outline_thickness: params.outline_thickness_px,
            outline_angle: params.outline_angle_degrees,
            outline_mirror: params.outline_mirror,
            outline_fade: params.outline_fade_px,
            border_thickness: params.border_thickness_px,
            border_opacity: params.border_opacity,
            turbulence_strength: params.turbulence_strength,
            turbulence_scale: params.turbulence_scale,
            reveal_wobble: params.reveal_wobble,
        }
    }

    /// Writes the material fields into a params set, preserving everything
    /// surface-specific (reveal progress, capsule radius, pill scale,
    /// animation time, pointer influence, glass fade, reveal origin, the
    /// outline terminator sharpness). `texel_scale` compensates texel-unit
    /// fields for a downscaled shared capture, exactly like
    /// `GlassSurface::set_shader_params` does for caller-provided params.
    pub fn apply_onto(&self, params: &mut GlassShaderParams, texel_scale: f32) {
        params.refraction_strength = self.refraction_strength * texel_scale;
        params.blur_radius_texels = self.blur_radius * texel_scale;
        params.chromatic_offset_texels = self.chromatic_offset * texel_scale;
        params.blur_samples = self.blur_samples;
        params.tint_strength = self.tint_strength;
        params.glass_tint = self.glass_tint;
        params.edge_band = self.edge_band;
        params.chromatic_blend = self.chromatic_blend;
        params.outline_strength = self.outline_strength;
        params.outline_thickness_px = self.outline_thickness;
        params.outline_angle_degrees = self.outline_angle;
        params.outline_mirror = self.outline_mirror;
        params.outline_fade_px = self.outline_fade;
        params.border_thickness_px = self.border_thickness;
        params.border_opacity = self.border_opacity;
        params.turbulence_strength = self.turbulence_strength;
        params.turbulence_scale = self.turbulence_scale;
        params.reveal_wobble = self.reveal_wobble;
    }

    /// Maps a legacy Glassbar `settings.json` onto the material. Missing or
    /// malformed keys fall back to the material defaults, exactly like the
    /// old per-app settings load did.
    pub fn from_legacy_glassbar(value: &serde_json::Value) -> Self {
        let defaults = Self::default();
        let number = |key: &str, fallback: f32| -> f32 {
            value
                .get(key)
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .unwrap_or(fallback)
        };
        Self {
            refraction_strength: number("refractionStrength", defaults.refraction_strength),
            blur_radius: number("blurRadius", defaults.blur_radius),
            blur_samples: number("blurSamples", defaults.blur_samples),
            tint_strength: number("tintStrength", defaults.tint_strength),
            glass_tint: value
                .get("tintColor")
                .and_then(|v| serde_json::from_value::<[f64; 3]>(v.clone()).ok())
                .map(|rgb| [rgb[0] as f32, rgb[1] as f32, rgb[2] as f32])
                .unwrap_or(defaults.glass_tint),
            edge_band: number("edgeBand", defaults.edge_band),
            chromatic_offset: number("chromaticOffset", defaults.chromatic_offset),
            chromatic_blend: number("chromaticBlend", defaults.chromatic_blend),
            outline_strength: number("outlineStrength", defaults.outline_strength),
            outline_thickness: number("outlineThickness", defaults.outline_thickness),
            outline_angle: number("outlineAngle", defaults.outline_angle),
            outline_mirror: number("outlineMirror", defaults.outline_mirror),
            outline_fade: number("outlineFade", defaults.outline_fade),
            border_thickness: number("borderThickness", defaults.border_thickness),
            border_opacity: number("borderOpacity", defaults.border_opacity),
            turbulence_strength: number("turbulenceStrength", defaults.turbulence_strength),
            turbulence_scale: number("turbulenceScale", defaults.turbulence_scale),
            reveal_wobble: number("wobble", defaults.reveal_wobble),
        }
    }
}

/// Which part of the host window the glass surface covers.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum SurfaceAnchor {
    /// The full host window rectangle.
    #[default]
    HostBounds,
    /// Only the bottom strip of the host window, `height_px` tall. Useful
    /// when the host window is taller than the glass bar (e.g. menu popup
    /// space above a taskbar pill).
    BottomStrip { height_px: f32 },
    /// An explicit rectangle, in physical pixels relative to the host
    /// window's top-left corner. Lets a host render glass behind arbitrary
    /// floating surfaces such as menus.
    Rect {
        offset_x: f32,
        offset_y: f32,
        width: f32,
        height: f32,
    },
}

/// Creation-time options for a [`crate::GlassSurface`].
#[derive(Clone, Debug)]
pub struct GlassConfig {
    /// Pill corner radius in surface pixels, applied inside the shader's SDF
    /// so the edge stays antialiased through the composition alpha. Any
    /// radius is allowed (not just full capsules); `0.0` produces square
    /// corners.
    pub corner_radius_px: f32,
    /// Which part of the host window the surface covers.
    pub surface_anchor: SurfaceAnchor,
    /// Apply borderless styling (`configure_borderless_host`) to the host
    /// window every time the surface stacks beneath it. Leave on for Tauri
    /// overlays; disable if you already style the host yourself.
    pub style_host_window: bool,
    /// Initial shader parameters; can be changed later with
    /// [`crate::GlassSurface::set_shader_params`].
    pub shader: GlassShaderParams,
}

impl Default for GlassConfig {
    fn default() -> Self {
        Self {
            corner_radius_px: 0.0,
            surface_anchor: SurfaceAnchor::default(),
            style_host_window: true,
            shader: GlassShaderParams::default(),
        }
    }
}
