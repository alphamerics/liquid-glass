//! The shared Windows Graphics Capture feed.
//!
//! ONE session for every glass host in the process (the in-process engine
//! or the broker): frames arrive scaled down to `CAPTURE_SCALE` of the
//! monitor (the frame pool is simply created at the smaller size — WGC
//! scales the content), and the newest frame is copied into a single
//! stable texture that the host's render loop samples. The callback does
//! no rendering; it only refreshes the shared texture and bumps the
//! version counter.
//!
//! The feed is generic over its host through [`FrameHost`]: the library's
//! engine (`EngineState`) and the broker's publisher (`BrokerHost`) both
//! implement it, so both drive the exact same pipeline.

use windows::Graphics::SizeInt32;
use windows::Win32::Graphics::Direct3D11::{
    ID3D11Device, ID3D11DeviceContext, ID3D11ShaderResourceView, ID3D11Texture2D,
};

mod feed;
mod frame;
mod monitor;

pub(crate) use feed::CaptureFeed;
pub(crate) use monitor::request_borderless_capture;

const FRAME_POOL_BUFFERS: i32 = 2;

/// The capture host's delivery interface: the feed copies each fresh
/// frame into the host's texture and asks the host who is actively
/// rendering (parked glass suppresses the copy entirely).
pub(crate) trait FrameHost: Send {
    /// The device all feed textures must be created on.
    fn device(&self) -> &ID3D11Device;
    /// Delivers the stable texture + shader view for the current pool
    /// size (also bumps the host's frame version).
    fn set_frame(&mut self, frame: (ID3D11Texture2D, ID3D11ShaderResourceView), size: SizeInt32);
    /// The stable texture to copy frames into, if one exists yet.
    fn frame_texture(&self) -> Option<&ID3D11Texture2D>;
    /// The immediate context clone for the callback's `CopyResource`
    /// (contexts are not `Send`, so the trait tunnels a clone out from
    /// behind the host's mutex).
    fn copy_context(&self) -> ID3D11DeviceContext;
    /// The scaled size the pool and texture were created at.
    fn frame_size(&self) -> SizeInt32;
    /// Whether any consumer of this host is actively rendering (windows
    /// visible); `false` lets the feed skip its per-frame GPU blit.
    fn any_active(&self) -> bool;
    /// Whether a frame copy may run now (the host owns the quality
    /// tier's backdrop rate cap). Returns `true` and stamps the copy
    /// time when the copy is allowed.
    fn capture_due(&mut self) -> bool;
}
