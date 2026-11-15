//! The monitor capture session: pool creation, the frame-arrived callback
//! (dirty-region skips, rate caps, the stable-texture copy) and teardown.

use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, Mutex,
};
use std::time::Duration;

use windows::{
    core::Interface,
    Foundation::TimeSpan,
    Graphics::{
        Capture::{
            Direct3D11CaptureFramePool, GraphicsCaptureDirtyRegionMode,
            GraphicsCaptureSession,
        },
        DirectX::{Direct3D11::IDirect3DDevice, DirectXPixelFormat},
    },
    Win32::Graphics::{
        Direct3D11::ID3D11Texture2D,
        Dxgi::IDXGIDevice,
    },
    Win32::System::WinRT::Direct3D11::{CreateDirect3D11DeviceFromDXGIDevice, IDirect3DDxgiInterfaceAccess},
};

use super::frame::{create_frame_texture, scale_size};
use super::{FrameHost, FRAME_POOL_BUFFERS};
use super::monitor::{primary_monitor_item, request_borderless_capture};
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
    /// Applies the quality tier's backdrop rate as WGC's own minimum frame
    /// interval (Windows 11 22H2+): the OS coalesces frame deliveries, so
    /// the callback — and its full-monitor blit — do not even run above
    /// the cap. Older systems ignore this and rely on the callback's
    /// `capture_due` gate instead.
    pub(crate) fn set_min_update_interval(&self, interval: Option<Duration>) {
        let Some(interval) = interval else {
            return;
        };
        // WinRT TimeSpan ticks are 100-ns units. The class wrapper casts
        // to IGraphicsCaptureSession5 internally (a no-op failure on
        // systems without it).
        let ticks = (interval.as_nanos().min(i64::MAX as u128) / 100) as i64;
        let _ = self
            .session
            .SetMinUpdateInterval(TimeSpan { Duration: ticks });
    }
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
        let session = frame_pool.CreateCaptureSession(&item).map_err(|error| {
            format!("Could not create the primary-monitor capture session: {error}")
        })?;
        // Windows 11: hide the yellow capture border (packaged capability
        // required; unpackaged builds ignore this).
        if let Err(error) = session.SetIsBorderRequired(false) {
            eprintln!("glass capture border stays on (needs the packaged graphicsCaptureProgrammatic capability): {error}");
        }

        // Windows 11 24H2: ask WGC to report dirty regions per frame. The
        // callback then skips frames whose content is unchanged entirely —
        // without this, every DWM recomposition (any window's caret blink,
        // clock tick, or the glass app's own DOM update) re-runs the
        // full-monitor blit and blur even though nothing visible changed.
        // Older systems fail the call silently and fall back to the rate
        // cap alone.
        let dirty_regions = session
            .SetDirtyRegionMode(GraphicsCaptureDirtyRegionMode::ReportOnly)
            .is_ok();

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
            if let Some(stable) = host.frame_texture().cloned() {
                // WGC recycles its frame surfaces, so the newest frame must
                // be copied out before the callback returns. But the copy is
                // a full-monitor GPU blit — skip it entirely when every
                // surface is parked: invisible glass must not keep burning
                // GPU on frames it will never render (the first frame after
                // any surface resumes refreshes the texture).
                if host.any_active() {
                    // Unchanged-content skip (24H2 dirty regions): a frame
                    // with no dirty regions is a recomposition with identical
                    // pixels — copying it would re-run the whole blur chain
                    // for nothing. The render loop force-refreshes the
                    // backdrop on a slow cadence, so a misreported frame at
                    // worst costs two seconds of staleness.
                    if dirty_regions
                        && frame
                            .DirtyRegions()
                            .map_or(false, |regions| regions.Size() == Ok(0))
                    {
                        return Ok(());
                    }
                    // Quality-tier backdrop cap: at most one blit (and thus
                    // one blur) per interval regardless of how fast frames
                    // arrive — animations on a fast-changing desktop are
                    // sampled at the cap, not at the monitor's frame rate.
                    if host.capture_due() {
                        let context = host.copy_context();
                        unsafe {
                            context.CopyResource(&stable, &texture);
                        }
                        callback_version.fetch_add(1, Ordering::Relaxed);
                    }
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
