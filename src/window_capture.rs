//! The local Windows Graphics Capture feed: one `CreateForWindow` session
//! per captured window, feeding glass whose backdrop is the app's own
//! content.
//!
//! This is the local sibling of [`crate::capture::CaptureFeed`]: the
//! monitor feed brings the WHOLE screen to global glass; this feed brings
//! one window's pixels to local glass, so a glass element that only ever
//! sits over its app's own UI runs with no screen duplication at all. The
//! session, frame pool, dirty-region skip, rate cap and stable-texture
//! copy mirror the monitor feed's semantics 1:1 — the only differences are
//! the capture item (a window instead of the primary monitor) and the
//! cursor capture (disabled: a blurred cursor inside the glass would
//! double the real one).
//!
//! Hosts tunnel their state through [`WindowFrameHost`], the local
//! counterpart of [`crate::capture::FrameHost`]: the engine's
//! [`crate::backdrop::LocalFeedHost`] proxy locks the shared engine state
//! and delegates to the captured window's own feed entry.

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};
use std::time::Duration;

use windows::{
    core::Interface,
    Foundation::TimeSpan,
    Graphics::{
        Capture::{Direct3D11CaptureFramePool, GraphicsCaptureItem, GraphicsCaptureSession},
        DirectX::{Direct3D11::IDirect3DDevice, DirectXPixelFormat},
        SizeInt32,
    },
    Win32::{
        Foundation::HWND,
        Graphics::{
            Direct3D11::{
                ID3D11Device, ID3D11DeviceContext, ID3D11ShaderResourceView, ID3D11Texture2D,
            },
            Dxgi::IDXGIDevice,
        },
        System::WinRT::{
            Direct3D11::{CreateDirect3D11DeviceFromDXGIDevice, IDirect3DDxgiInterfaceAccess},
            Graphics::Capture::IGraphicsCaptureItemInterop,
        },
    },
};

const FRAME_POOL_BUFFERS: i32 = 2;

/// The local capture host's delivery interface: the feed copies each fresh
/// frame into the host window's stable texture and asks the host who is
/// actively rendering (parked glass suppresses the copy entirely).
///
/// Unlike [`crate::capture::FrameHost`], the accessors hand back clones:
/// the engine's per-window proxy may only borrow the shared engine state
/// from behind a mutex, so nothing here can return a reference into it.
pub(crate) trait WindowFrameHost: Send {
    /// A clone of the device all feed textures must be created on.
    fn device(&self) -> ID3D11Device;
    /// Delivers the stable texture + shader view for the current pool
    /// size (bumps the host's frame version).
    fn set_frame(&mut self, frame: (ID3D11Texture2D, ID3D11ShaderResourceView), size: SizeInt32);
    /// The stable texture to copy frames into, if one exists yet.
    fn frame_texture(&self) -> Option<ID3D11Texture2D>;
    /// A clone of the immediate context for the callback's `CopyResource`.
    fn copy_context(&self) -> ID3D11DeviceContext;
    /// The current frame (and pool) size in pixels.
    fn frame_size(&self) -> SizeInt32;
    /// The version counter the callback bumps when a frame is copied in.
    fn version(&self) -> Arc<AtomicU64>;
    /// Whether any consumer of this host is actively rendering; `false`
    /// lets the feed skip its per-frame GPU blit.
    fn any_active(&self) -> bool;
    /// Whether a frame copy may run now (the host owns the quality tier's
    /// backdrop rate cap). Returns `true` and stamps the copy time when
    /// the copy is allowed.
    fn capture_due(&mut self) -> bool;
}

/// `CreateDirect3D11DeviceFromDXGIDevice` produces an agile WinRT wrapper
/// around our free-threaded D3D11 device. windows-rs does not mark this
/// particular WinRT interface as `Send`, though WGC invokes its
/// free-threaded frame callback on a worker thread.
struct AgileDirect3DDevice(IDirect3DDevice);

unsafe impl Send for AgileDirect3DDevice {}

impl AgileDirect3DDevice {
    fn as_device(&self) -> &IDirect3DDevice {
        &self.0
    }
}

/// Dropping the feed stops WGC and releases the capture session.
pub(crate) struct WindowCaptureFeed {
    frame_pool: Direct3D11CaptureFramePool,
    session: GraphicsCaptureSession,
    frame_arrived_token: windows::Foundation::EventRegistrationToken,
}

impl WindowCaptureFeed {
    /// Applies the quality tier's backdrop rate as WGC's own minimum frame
    /// interval (Windows 11 22H2+): the OS coalesces frame deliveries, so
    /// the callback — and its window blit — do not even run above the cap.
    /// Older systems ignore this and rely on the callback's `capture_due`
    /// gate instead.
    pub(crate) fn set_min_update_interval(&self, interval: Option<Duration>) {
        let Some(interval) = interval else {
            return;
        };
        // WinRT TimeSpan ticks are 100-ns units.
        let ticks = (interval.as_nanos().min(i64::MAX as u128) / 100) as i64;
        let _ = self
            .session
            .SetMinUpdateInterval(TimeSpan { Duration: ticks });
    }
}

impl WindowCaptureFeed {
    /// Starts capturing `target` (an HWND value) on the host's device.
    pub(crate) fn start(
        host: &Arc<Mutex<dyn WindowFrameHost>>,
        target: isize,
    ) -> Result<Self, String> {
        // Windows 11: request the borderless-capture permission once, so the
        // session below can disable the yellow capture border. Requires the
        // packaged graphicsCaptureProgrammatic capability; unpackaged builds
        // and older systems skip it silently.
        request_borderless_capture();
        if !GraphicsCaptureSession::IsSupported()
            .map_err(|error| format!("Could not query Windows Graphics Capture support: {error}"))?
        {
            return Err("Windows Graphics Capture is not supported on this system".to_owned());
        }

        let device = {
            let host = host.lock().expect("local glass host lock poisoned");
            host.device()
        };
        let dxgi_device = device
            .cast::<IDXGIDevice>()
            .map_err(|error| format!("Could not obtain the D3D11 DXGI device: {error}"))?;
        let direct3d_device: IDirect3DDevice = unsafe {
            CreateDirect3D11DeviceFromDXGIDevice(&dxgi_device)
                .and_then(|device| device.cast())
                .map_err(|error| format!("Could not create the WinRT D3D11 device: {error}"))?
        };
        let item = window_item(target)?;
        let window_size = item
            .Size()
            .map_err(|error| format!("Could not read the captured window size: {error}"))?;
        if window_size.Width <= 0 || window_size.Height <= 0 {
            return Err("The captured window has no extent (is it minimized?)".to_owned());
        }
        let frame_pool = Direct3D11CaptureFramePool::CreateFreeThreaded(
            &direct3d_device,
            DirectXPixelFormat::B8G8R8A8UIntNormalized,
            FRAME_POOL_BUFFERS,
            window_size,
        )
        .map_err(|error| format!("Could not create the local glass capture frame pool: {error}"))?;
        let session = frame_pool.CreateCaptureSession(&item).map_err(|error| {
            format!("Could not create the local glass capture session: {error}")
        })?;
        // Windows 11: hide the yellow capture border (packaged capability
        // required; unpackaged builds ignore this).
        if let Err(error) = session.SetIsBorderRequired(false) {
            eprintln!("local glass capture border stays on (needs the packaged graphicsCaptureProgrammatic capability): {error}");
        }
        // The captured cursor must not land in the backdrop: the glass would
        // blur (and refract) a frozen copy of the pointer that hovers on top.
        if let Err(error) = session.SetIsCursorCaptureEnabled(false) {
            eprintln!("local glass capture keeps the cursor: {error}");
        }

        // NOTE: no dirty-region mode here. The monitor feed skips frames
        // whose whole content is unchanged, but WGC does not populate dirty
        // regions for window captures — trusting an empty report there
        // would skip every frame and silently freeze the local backdrop.
        // The rate caps alone bound this feed's cost.

        {
            let mut host = host.lock().expect("local glass host lock poisoned");
            let frame = create_frame_texture(&device, window_size);
            host.set_frame(frame, window_size);
        }

        let callback_host = Arc::clone(host);
        let callback_device = AgileDirect3DDevice(direct3d_device);
        let frame_arrived_handler = windows::Foundation::TypedEventHandler::<
            Direct3D11CaptureFramePool,
            windows::core::IInspectable,
        >::new(move |pool, _| {
            let Some(pool) = pool.as_ref() else {
                return Ok(());
            };
            frame_arrived(&callback_host, &callback_device, pool)
        });
        let frame_arrived_token =
            frame_pool
                .FrameArrived(&frame_arrived_handler)
                .map_err(|error| {
                    format!("Could not register the local glass frame callback: {error}")
                })?;

        session
            .StartCapture()
            .map_err(|error| format!("Could not start the local glass capture: {error}"))?;

        Ok(Self {
            frame_pool,
            session,
            frame_arrived_token,
        })
    }
}

impl Drop for WindowCaptureFeed {
    fn drop(&mut self) {
        let _ = self.frame_pool.RemoveFrameArrived(self.frame_arrived_token);
        let _ = self.session.Close();
        let _ = self.frame_pool.Close();
    }
}

/// The frame callback (runs on a WGC worker thread): copies the newest
/// frame into the stable texture, rebuilding the pool when the window
/// changes size.
fn frame_arrived(
    host: &Arc<Mutex<dyn WindowFrameHost>>,
    device: &AgileDirect3DDevice,
    pool: &Direct3D11CaptureFramePool,
) -> windows::core::Result<()> {
    let Ok(frame) = pool.TryGetNextFrame() else {
        return Ok(());
    };
    let Ok(size) = frame.ContentSize() else {
        return Ok(());
    };
    if size.Width <= 0 || size.Height <= 0 {
        return Ok(());
    }
    let mut host = host.lock().expect("local glass host lock poisoned");
    if host.frame_size() != size {
        // The window resized (or DPI changed): reroute the pool and
        // rebuild the stable texture at the new size.
        let _ = pool.Recreate(
            device.as_device(),
            DirectXPixelFormat::B8G8R8A8UIntNormalized,
            FRAME_POOL_BUFFERS,
            size,
        );
        let fresh = create_frame_texture(&host.device(), size);
        host.set_frame(fresh, size);
        return Ok(());
    }
    let Some(stable) = host.frame_texture() else {
        return Ok(());
    };
    // Parked glass must not burn GPU on frames it will never render.
    if !host.any_active() {
        return Ok(());
    }
    // Quality-tier backdrop cap: at most one blit (and thus one blur) per
    // interval regardless of how fast frames arrive.
    if host.capture_due() {
        let Ok(texture) = (unsafe {
            frame
                .Surface()
                .and_then(|surface| surface.cast::<IDirect3DDxgiInterfaceAccess>())
                .and_then(|access| access.GetInterface::<ID3D11Texture2D>())
        }) else {
            return Ok(());
        };
        let context = host.copy_context();
        unsafe {
            context.CopyResource(&stable, &texture);
        }
        host.version().fetch_add(1, Ordering::Relaxed);
    }
    Ok(())
}

fn create_frame_texture(
    device: &ID3D11Device,
    size: SizeInt32,
) -> (ID3D11Texture2D, ID3D11ShaderResourceView) {
    let mut texture = None;
    unsafe {
        device.CreateTexture2D(
            &windows::Win32::Graphics::Direct3D11::D3D11_TEXTURE2D_DESC {
                Width: size.Width.max(1) as u32,
                Height: size.Height.max(1) as u32,
                MipLevels: 1,
                ArraySize: 1,
                Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Usage: windows::Win32::Graphics::Direct3D11::D3D11_USAGE_DEFAULT,
                BindFlags: windows::Win32::Graphics::Direct3D11::D3D11_BIND_SHADER_RESOURCE.0
                    as u32,
                CPUAccessFlags: 0,
                MiscFlags: 0,
            },
            None,
            Some(&mut texture),
        )
    }
    .map_err(|error| format!("Could not create the local glass frame texture: {error}"))
    .expect("Could not create the local glass frame texture");
    let texture = texture.expect("local glass frame texture missing");
    let mut view = None;
    unsafe {
        device
            .CreateShaderResourceView(&texture, None, Some(&mut view))
            .expect("Could not create the local glass frame view");
    }
    (texture, view.expect("local glass frame view missing"))
}

fn window_item(target: isize) -> Result<GraphicsCaptureItem, String> {
    let interop: IGraphicsCaptureItemInterop =
        windows::core::factory::<GraphicsCaptureItem, IGraphicsCaptureItemInterop>().map_err(
            |error| format!("Could not obtain the GraphicsCaptureItem interop factory: {error}"),
        )?;
    unsafe {
        interop
            .CreateForWindow(HWND(target as *mut _))
            .map_err(|error| format!("Could not create the local glass capture item: {error}"))
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
