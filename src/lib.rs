//! # liquid-glass
//!
//! A reusable, GPU-composited "liquid glass" backdrop for Windows overlay
//! windows — designed to drop into Tauri apps with one call.
//!
//! The crate creates a native D3D11 surface window positioned directly
//! beneath your (transparent) overlay window, captures the primary monitor
//! with Windows Graphics Capture, and composites the captured pixels through
//! a liquid-glass shader (refraction, blur, chromatic dispersion, tint). The
//! shader is fully customizable at runtime via [`GlassShaderParams`], and the
//! native corner shape via [`GlassConfig::corner_radius_px`].
//!
//! ## Tauri usage
//!
//! ```ignore
//! use liquid_glass::{GlassConfig, GlassSurface};
//!
//! let window = tauri_webview_window();
//! let hwnd = window.hwnd()?.0 as isize;
//!
//! // Recommended for tao/tauri hosts: suppress classic frame repaints on
//! // focus changes (tao always keeps WS_CAPTION in the style).
//! liquid_glass::install_activation_subclass(hwnd)?;
//!
//! let surface = GlassSurface::create(hwnd, GlassConfig::default())?;
//!
//! // Live-tune the material:
//! surface.set_shader_params(liquid_glass::GlassShaderParams {
//!     refraction_strength: 14.0,
//!     glass_tint: [0.9, 0.75, 1.0],
//!     ..Default::default()
//! })?;
//!
//! // On window move/resize events: surface.sync_bounds(hwnd)?;
//! // Dropping `surface` stops the capture and destroys the surface window.
//! ```
//!
//! The crate is intentionally Tauri-free: it only needs an `isize` HWND, so
//! it works with tao, winit, raw Win32, or any framework.

mod borderless;
pub mod params;
pub mod broker;
mod shared;
mod surface;
pub mod thumbnail;

pub(crate) mod capture;

pub use borderless::{configure_borderless_host, install_activation_subclass};
pub use params::{GlassConfig, GlassShaderParams, Material, RevealAnimation, SurfaceAnchor};
pub use surface::{
    translate_compositor, translate_surface_window, D3D11Compositor, GlassSurface,
    SharedCompositor,
};
pub use thumbnail::ThumbnailSurface;

/// Mirrors the shader's `SourceRect` constant buffer (register b0).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SourceRectConstants {
    pub(crate) source_uv_origin: [f32; 2],
    pub(crate) source_uv_size: [f32; 2],
    pub(crate) pointer_velocity: [f32; 2],
    /// Constant buffers must be a multiple of 16 bytes: 8 floats = 32 bytes.
    pub(crate) _padding: [f32; 2],
}
