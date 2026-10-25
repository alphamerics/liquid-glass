//! GPU glass surfaces: the public [`GlassSurface`] handle, its native
//! window, the per-surface D3D11 compositor and the shared D3D11 device,
//! split into focused submodules.

mod compositor;
mod device;
mod geometry;
mod glass;
mod window;

#[cfg(test)]
mod tests;

pub use compositor::{translate_compositor, D3D11Compositor};
pub(crate) use device::create_d3d11_device;
pub use glass::GlassSurface;
pub use window::translate_surface_window;

use std::sync::{Arc, Mutex};

pub type SharedCompositor = Arc<Mutex<D3D11Compositor>>;