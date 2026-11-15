//! The thumbnail's core state: the D3D/WinRT devices, the composition
//! swap chain, the captured-frame slot and the capture session registry.
//! Everything the surface API, the render loop and the capture callbacks
//! share lives here, behind one mutex.

use std::sync::{Arc, Mutex};

use windows::Graphics::{
    Capture::{Direct3D11CaptureFramePool, GraphicsCaptureSession},
    DirectX::Direct3D11::IDirect3DDevice,
    SizeInt32,
};
use windows::Win32::Graphics::Direct3D11::{
    ID3D11Buffer, ID3D11Device, ID3D11DeviceContext, ID3D11PixelShader, ID3D11RenderTargetView,
    ID3D11SamplerState, ID3D11ShaderResourceView, ID3D11Texture2D, ID3D11VertexShader,
};
use windows::Win32::Graphics::DirectComposition::{
    IDCompositionDevice, IDCompositionTarget, IDCompositionVisual,
};
use windows::Win32::Graphics::Dxgi::IDXGISwapChain1;
/// Mirrors the shader's `ThumbParams` constant buffer (register b0).
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct ThumbConstants {
    /// The client-area crop in frame UV (the WGC window item covers the
    /// whole title-barred frame; the old thumbnail cropped it via rcSource).
    pub(super) crop_origin_uv: [f32; 2],
    pub(super) crop_size_uv: [f32; 2],
    /// Corner radius normalized to the quad's height (0..0.5).
    pub(super) corner_radius: f32,
    /// Quad width / height, for aspect-corrected SDF corners.
    pub(super) quad_aspect: f32,
    pub(super) padding: [f32; 2],
}

/// The WGC per-window capture session. `epoch` is the capture state epoch
/// the session was started under (see `ThumbCore::capture_epoch`): a session
/// whose epoch no longer matches the core's is superseded, and its callback
/// must not touch the core or the pool.
pub(super) struct Capture {
    pub(super) frame_pool: Direct3D11CaptureFramePool,
    pub(super) session: GraphicsCaptureSession,
    pub(super) frame_arrived_token: windows::Foundation::EventRegistrationToken,
    /// The capture epoch this session started under; debugging aid (the
    /// ground truth lives in the frame callback's closure).
    #[allow(dead_code)]
    pub(super) epoch: u64,
}

impl Drop for Capture {
    fn drop(&mut self) {
        // Remember: never run this while holding the core mutex. `Close`
        // synchronizes with the free-threaded frame callback, and that
        // callback needs the core lock to run — tearing down under the lock
        // deadlocks (or races the pool's close against an in-flight
        // `TryGetNextFrame`, which is an access violation). Callers keep the
        // session out of the core and drop it after releasing the lock.
        let _ = self.frame_pool.RemoveFrameArrived(self.frame_arrived_token);
        let _ = self.session.Close();
        let _ = self.frame_pool.Close();
    }
}

/// `CreateDirect3D11DeviceFromDXGIDevice` produces an agile WinRT wrapper
/// around the free-threaded D3D11 device; windows-rs does not mark it `Send`
/// although WGC invokes the frame callback on a worker thread.
#[derive(Clone)]
pub(super) struct AgileWinRTDevice(pub(super) Arc<IDirect3DDevice>);

unsafe impl Send for AgileWinRTDevice {}

pub(super) struct ThumbCore {
    pub(super) device: ID3D11Device,
    pub(super) context: ID3D11DeviceContext,
    pub(super) winrt_device: AgileWinRTDevice,
    pub(super) swap_chain: IDXGISwapChain1,
    #[allow(dead_code)]
    pub(super) composition_device: IDCompositionDevice,
    #[allow(dead_code)]
    pub(super) composition_target: IDCompositionTarget,
    #[allow(dead_code)]
    pub(super) composition_visual: IDCompositionVisual,
    pub(super) vertex_shader: ID3D11VertexShader,
    pub(super) pixel_shader: ID3D11PixelShader,
    pub(super) sampler: ID3D11SamplerState,
    pub(super) constants: ID3D11Buffer,
    pub(super) back_buffer: Option<(ID3D11Texture2D, ID3D11RenderTargetView)>,
    pub(super) swap_chain_size: [u32; 2],
    /// The newest captured frame, copied out of the WGC pool (whose surfaces
    /// are recycled) into a stable texture the render loop samples.
    pub(super) frame_texture: Option<ID3D11Texture2D>,
    pub(super) frame_view: Option<ID3D11ShaderResourceView>,
    pub(super) frame_size: SizeInt32,
    /// Bumped by the capture callback when a fresh frame has been copied in.
    pub(super) version: u64,
    /// The client-area crop in frame UV space, computed at set_source.
    pub(super) crop: Option<([f32; 2], [f32; 2])>,
    pub(super) corner_radius_px: f32,
    /// The window being captured (0 = none).
    pub(super) source: isize,
    /// The composition host (the preview popup's hwnd as an isize).
    pub(super) window: isize,
    pub(super) feed: Option<Capture>,
    pub(super) visible: bool,
    pub(super) has_presented: bool,
    pub(super) dirty: bool,
    /// Bumped by every state change that supersedes a capture (retarget,
    /// hide). Sessions capture the epoch they started under; a callback from
    /// an older epoch must not touch the core or the pool. Also lets a
    /// capture started outside the lock (the WinRT session-start call
    /// re-enters the callback immediately) be discarded when the desired
    /// state has since changed.
    pub(super) capture_epoch: u64,
}

impl ThumbCore {
    pub(super) fn compute_constants(&self) -> ThumbConstants {
        let width = self.swap_chain_size[0].max(1) as f32;
        let height = self.swap_chain_size[1].max(1) as f32;
        let aspect = width / height;
        let (origin, mut size) = self.crop.unwrap_or(([0.0, 0.0], [1.0, 1.0]));
        let mut origin = origin;
        // Center-crop the client area to the quad's aspect (source wider
        // than the box crops horizontally, taller crops vertically) so the
        // content fills the box without stretching, like the previous DWM
        // thumbnail's aspect-matched rcSource.
        let frame_width = self.frame_size.Width.max(1) as f32;
        let frame_height = self.frame_size.Height.max(1) as f32;
        let source_aspect = (size[0] * frame_width) / (size[1] * frame_height);
        if size[1].max(1e-6) > 0.0 && source_aspect.is_finite() && source_aspect > 0.0 {
            if source_aspect > aspect {
                let frac = (aspect / source_aspect).clamp(0.0, 1.0);
                origin[0] += size[0] * (1.0 - frac) / 2.0;
                size[0] *= frac;
            } else {
                let frac = (source_aspect / aspect).clamp(0.01, 1.0);
                origin[1] += size[1] * (1.0 - frac) / 2.0;
                size[1] *= frac;
            }
        }
        ThumbConstants {
            crop_origin_uv: origin,
            crop_size_uv: size,
            corner_radius: (self.corner_radius_px / height).clamp(0.0, 0.5),
            quad_aspect: aspect,
            padding: [0.0; 2],
        }
    }
}

/// `Arc<Mutex<ThumbCore>>` is moved across threads (render loop, WGC
/// callback): the D3D immediate context it holds is not `Send`, but all its
/// uses are serialized behind the mutex — this wrapper carries that
/// invariant (the same promise `FrameHost: Send` makes in `capture.rs`).
#[derive(Clone)]
pub(super) struct SendCore(pub(super) Arc<Mutex<ThumbCore>>);

unsafe impl Send for SendCore {}
unsafe impl Sync for SendCore {}
