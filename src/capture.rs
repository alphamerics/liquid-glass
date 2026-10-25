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

use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, Mutex,
};

use windows::{
    core::Interface,
    Graphics::{
        Capture::{Direct3D11CaptureFramePool, GraphicsCaptureItem, GraphicsCaptureSession},
        DirectX::{Direct3D11::IDirect3DDevice, DirectXPixelFormat},
        SizeInt32,
    },
    Win32::{
        Foundation::POINT,
        Graphics::{
            Direct3D11::{
                ID3D11Device, ID3D11DeviceContext, ID3D11ShaderResourceView, ID3D11Texture2D,
                D3D11_BIND_SHADER_RESOURCE, D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT,
            },
            Dxgi::IDXGIDevice,
            Gdi::{MonitorFromPoint, MONITOR_DEFAULTTOPRIMARY},
        },
        System::WinRT::{
            Direct3D11::{CreateDirect3D11DeviceFromDXGIDevice, IDirect3DDxgiInterfaceAccess},
            Graphics::Capture::IGraphicsCaptureItemInterop,
        },
    },
};

use crate::shared::CAPTURE_SCALE;

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
}

/// `CreateDirect3D11DeviceFromDXGIDevice` produces an agile WinRT wrapper
/// around our free-threaded D3D11 device. windows-rs does not mark this
/// particular WinRT interface as `Send`, though WGC invokes its free-threaded
/// frame callback on a worker thread.
struct AgileDirect3DDevice(IDirect3DDevice);

unsafe impl Send for AgileDirect3DDevice {}

impl AgileDirect3DDevice {
    fn as_device(&self) -> &IDirect3DDevice {
        &self.0
    }
}

/// Dropping the feed stops WGC and releases the capture session.
pub(crate) struct CaptureFeed {
    frame_pool: Direct3D11CaptureFramePool,
    session: GraphicsCaptureSession,
    frame_arrived_token: windows::Foundation::EventRegistrationToken,
    #[allow(dead_code)]
    render_running: Arc<AtomicBool>,
}

impl CaptureFeed {
    pub(crate) fn start(
        host: &Arc<Mutex<dyn FrameHost>>,
        version: &Arc<AtomicU64>,
    ) -> Result<Self, String> {
        // Windows 11: request the borderless-capture permission once, so the
        // session below can disable the yellow capture border. This requires
        // the packaged graphicsCaptureProgrammatic capability; unpackaged
        // builds simply skip it silently.
        request_borderless_capture();
        if !GraphicsCaptureSession::IsSupported()
            .map_err(|error| format!("Could not query Windows Graphics Capture support: {error}"))?
        {
            return Err("Windows Graphics Capture is not supported on this system".to_owned());
        }

        let (device, _) = {
            let host = host.lock().expect("glass host lock poisoned");
            (host.device().clone(), ())
        };
        let dxgi_device = device
            .cast::<IDXGIDevice>()
            .map_err(|error| format!("Could not obtain the D3D11 DXGI device: {error}"))?;
        let direct3d_device: IDirect3DDevice = unsafe {
            CreateDirect3D11DeviceFromDXGIDevice(&dxgi_device)
                .and_then(|device| device.cast())
                .map_err(|error| format!("Could not create the WinRT D3D11 device: {error}"))?
        };
        let item = primary_monitor_item()?;
        let monitor_size = item
            .Size()
            .map_err(|error| format!("Could not read the primary monitor size: {error}"))?;
        // The pool size is the CAPTURE size: WGC scales the whole monitor
        // into it. All surface shaders sample the capture with normalized
        // full-monitor UVs, so the reduced resolution is invisible to them.
        let scaled = scale_size(monitor_size);
        let frame_pool = Direct3D11CaptureFramePool::CreateFreeThreaded(
            &direct3d_device,
            DirectXPixelFormat::B8G8R8A8UIntNormalized,
            FRAME_POOL_BUFFERS,
            scaled,
        )
        .map_err(|error| {
            format!("Could not create the Windows Graphics Capture frame pool: {error}")
        })?;
        let session = frame_pool
            .CreateCaptureSession(&item)
            .map_err(|error| format!("Could not create the primary-monitor capture session: {error}"))?;
        // Windows 11: hide the yellow capture border (packaged capability
        // required; unpackaged builds ignore this).
        if let Err(error) = session.SetIsBorderRequired(false) {
            eprintln!("glass capture border stays on (needs the packaged graphicsCaptureProgrammatic capability): {error}");
        }

        {
            let mut host = host.lock().expect("glass host lock poisoned");
            let frame = create_frame_texture(&device, scaled);
            host.set_frame(frame, scaled);
        }

        let callback_host = Arc::clone(host);
        let callback_version = Arc::clone(version);
        let callback_device = AgileDirect3DDevice(direct3d_device.clone());
        let frame_arrived_handler = windows::Foundation::TypedEventHandler::<
            Direct3D11CaptureFramePool,
            windows::core::IInspectable,
        >::new(move |pool, _| {
            let Some(pool) = pool.as_ref() else {
                return Ok(());
            };
            let frame = match pool.TryGetNextFrame() {
                Ok(frame) => frame,
                Err(_) => return Ok(()),
            };
            let size = match frame.ContentSize() {
                Ok(size) => size,
                Err(_) => return Ok(()),
            };
            if size.Width <= 0 || size.Height <= 0 {
                return Ok(());
            }
            let scaled = scale_size(size);
            let mut host = callback_host.lock().expect("glass host lock poisoned");
            if host.frame_size() != scaled {
                // Monitor changed (resolution, DPI): restart the pool and the
                // shared texture at the new scaled size.
                let _ = pool.Recreate(
                    callback_device.as_device(),
                    DirectXPixelFormat::B8G8R8A8UIntNormalized,
                    FRAME_POOL_BUFFERS,
                    scaled,
                );
                let frame = create_frame_texture(host.device(), scaled);
                host.set_frame(frame, scaled);
                return Ok(());
            }
            let texture = match unsafe {
                frame
                    .Surface()
                    .and_then(|surface| surface.cast::<IDirect3DDxgiInterfaceAccess>())
                    .and_then(|access| access.GetInterface::<ID3D11Texture2D>())
            } {
                Ok(texture) => texture,
                Err(_) => return Ok(()),
            };
            if let Some(stable) = host.frame_texture() {
                // WGC recycles its frame surfaces, so the newest frame must
                // be copied out before the callback returns. But the copy is
                // a full-monitor GPU blit — skip it entirely when every
                // surface is parked: invisible glass must not keep burning
                // GPU on frames it will never render (the first frame after
                // any surface resumes refreshes the texture).
                if host.any_active() {
                    let context = host.copy_context();
                    unsafe {
                        context.CopyResource(stable, &texture);
                    }
                    callback_version.fetch_add(1, Ordering::Relaxed);
                }
            }
            Ok(())
        });
        let frame_arrived_token = frame_pool
            .FrameArrived(&frame_arrived_handler)
            .map_err(|error| format!("Could not register the capture frame callback: {error}"))?;

        session
            .StartCapture()
            .map_err(|error| format!("Could not start primary-monitor capture: {error}"))?;

        Ok(Self {
            frame_pool,
            session,
            frame_arrived_token,
            render_running: Arc::new(AtomicBool::new(true)),
        })
    }
}

impl Drop for CaptureFeed {
    fn drop(&mut self) {
        let _ = self.frame_pool.RemoveFrameArrived(self.frame_arrived_token);
        let _ = self.session.Close();
        let _ = self.frame_pool.Close();
    }
}

fn scale_size(size: SizeInt32) -> SizeInt32 {
    SizeInt32 {
        Width: ((size.Width as f32 * CAPTURE_SCALE).round() as i32).max(1),
        Height: ((size.Height as f32 * CAPTURE_SCALE).round() as i32).max(1),
    }
}

/// The shared capture frame texture plus its shader view, created together:
/// every consumer samples the same view for the texture's whole lifetime
/// (recreated only when the monitor's size changes).
pub(crate) fn create_frame_texture(
    device: &ID3D11Device,
    size: SizeInt32,
) -> (ID3D11Texture2D, ID3D11ShaderResourceView) {
    let mut texture = None;
    unsafe {
        device.CreateTexture2D(
            &D3D11_TEXTURE2D_DESC {
                Width: size.Width as u32,
                Height: size.Height as u32,
                MipLevels: 1,
                ArraySize: 1,
                Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Usage: D3D11_USAGE_DEFAULT,
                BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
                CPUAccessFlags: 0,
                MiscFlags: 0,
            },
            None,
            Some(&mut texture),
        )
    }
    .map_err(|error| format!("Could not create the shared capture frame texture: {error}"))
    .expect("Could not create the shared capture frame texture");
    let texture = texture.expect("D3D11 shared capture frame texture missing");
    let mut view = None;
    unsafe {
        device
            .CreateShaderResourceView(&texture, None, Some(&mut view))
            .expect("Could not create the shared capture frame view");
    }
    (texture, view.expect("D3D11 shared capture frame view missing"))
}

fn primary_monitor_item() -> Result<GraphicsCaptureItem, String> {
    let monitor = unsafe { MonitorFromPoint(POINT::default(), MONITOR_DEFAULTTOPRIMARY) };
    if monitor.0.is_null() {
        return Err("Could not locate the primary monitor".to_owned());
    }
    let interop: IGraphicsCaptureItemInterop =
        windows::core::factory::<GraphicsCaptureItem, IGraphicsCaptureItemInterop>()
            .map_err(|error| {
                format!("Could not obtain the GraphicsCaptureItem interop factory: {error}")
            })?;
    unsafe {
        interop
            .CreateForMonitor(monitor)
            .map_err(|error| format!("Could not create the primary-monitor capture item: {error}"))
    }
}

/// Windows 11: grants the borderless-capture permission so
/// GraphicsCaptureSession::SetBorderRequired(false) is accepted. Requires the
/// packaged graphicsCaptureProgrammatic capability; no-op elsewhere.
fn request_borderless_capture() {
    use windows::Graphics::Capture::{GraphicsCaptureAccess, GraphicsCaptureAccessKind};
    let _ = GraphicsCaptureAccess::RequestAccessAsync(GraphicsCaptureAccessKind::Borderless)
        .and_then(|operation| operation.get());
}
