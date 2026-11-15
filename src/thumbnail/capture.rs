use std::sync::{Arc, Mutex};

use windows::core::Interface;
use windows::Graphics::{
    Capture::{Direct3D11CaptureFramePool, GraphicsCaptureItem},
    DirectX::DirectXPixelFormat,
    SizeInt32,
};
use windows::Win32::Foundation::{HWND, POINT, RECT};
use windows::Win32::Graphics::Gdi::ClientToScreen;
use windows::Win32::Graphics::Direct3D11::{
    ID3D11Device, ID3D11ShaderResourceView, ID3D11Texture2D,
    D3D11_BIND_RENDER_TARGET,
    D3D11_BIND_SHADER_RESOURCE, D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT,
};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_SAMPLE_DESC};
use windows::Win32::System::WinRT::{
    Direct3D11::IDirect3DDxgiInterfaceAccess, Graphics::Capture::IGraphicsCaptureItemInterop,
};
use windows::Win32::UI::WindowsAndMessaging::{GetClientRect, GetWindowRect};


use super::FRAME_POOL_BUFFERS;
use super::state::{AgileWinRTDevice, Capture, SendCore, ThumbCore};
pub(super) fn measure_window(window: isize) -> Result<(u32, u32), String> {
    let mut client = RECT::default();
    let hwnd = HWND(window as *mut _);
    unsafe {
        GetClientRect(hwnd, &mut client)
            .map_err(|error| format!("Could not measure the preview window: {error}"))?;
    }
    Ok((
        (client.right - client.left).max(1) as u32,
        (client.bottom - client.top).max(1) as u32,
    ))
}

/// Starts the per-window WGC capture of `target`, feeding the core's stable
/// frame texture through the free-threaded frame callback. Runs without the
/// core lock (the callback re-enters it as soon as the session starts), so
/// shared D3D handles are cloned out under a brief lock first. `epoch` is the
/// capture epoch the caller started under; the returned session is tagged
/// with it so superseded sessions can be recognized (and discarded).
pub(super) fn start_capture(
    core: &Arc<Mutex<ThumbCore>>,
    target: isize,
    epoch: u64,
) -> Result<Capture, String> {
    // Windows 11: request the borderless-capture permission once so the
    // session can disable the yellow capture border (no-op unpackaged).
    request_borderless_capture();

    let interop: IGraphicsCaptureItemInterop =
        windows::core::factory::<GraphicsCaptureItem, IGraphicsCaptureItemInterop>().map_err(
            |error| format!("Could not obtain the GraphicsCaptureItem interop factory: {error}"),
        )?;
    let item: GraphicsCaptureItem = unsafe { interop.CreateForWindow(HWND(target as *mut _)) }
        .map_err(|error| format!("Could not capture the target window: {error}"))?;
    let size = item
        .Size()
        .map_err(|error| format!("Could not read the target window size: {error}"))?;

    let winrt_device = {
        let core = core
            .lock()
            .map_err(|_| "thumbnail lock poisoned".to_owned())?;
        AgileWinRTDevice(Arc::clone(&core.winrt_device.0))
    };
    let frame_pool = Direct3D11CaptureFramePool::CreateFreeThreaded(
        winrt_device.0.as_ref(),
        DirectXPixelFormat::B8G8R8A8UIntNormalized,
        FRAME_POOL_BUFFERS,
        size,
    )
    .map_err(|error| format!("Could not create the thumbnail frame pool: {error}"))?;
    let session = frame_pool
        .CreateCaptureSession(&item)
        .map_err(|error| format!("Could not create the thumbnail capture session: {error}"))?;
    if let Err(error) = session.SetIsCursorCaptureEnabled(false) {
        eprintln!("Thumbnail cursor capture stays on: {error}");
    }
    if let Err(error) = session.SetIsBorderRequired(false) {
        eprintln!("Thumbnail capture border stays on (needs packaged capability): {error}");
    }

    let handler_core = SendCore(Arc::clone(core));
    let handler_epoch = epoch;
    let frame_arrived_handler = windows::Foundation::TypedEventHandler::<
        Direct3D11CaptureFramePool,
        windows::core::IInspectable,
    >::new(move |pool, _| {
        let Some(pool) = pool.as_ref() else {
            return Ok(());
        };
        frame_arrived(&handler_core, &winrt_device, pool, handler_epoch)
    });
    let frame_arrived_token = frame_pool
        .FrameArrived(&frame_arrived_handler)
        .map_err(|error| format!("Could not register the thumbnail frame callback: {error}"))?;
    session
        .StartCapture()
        .map_err(|error| format!("Could not start the window capture: {error}"))?;

    // Client-area inset measured once: a moving window keeps the same
    // frame-relative crop; a resizing one recomputes on the next retarget.
    let mut core = core
        .lock()
        .map_err(|_| "thumbnail lock poisoned".to_owned())?;
    if core.capture_epoch == epoch {
        core.crop = client_crop_uv(target);
    }

    Ok(Capture {
        frame_pool,
        session,
        frame_arrived_token,
        epoch,
    })
}

/// The frame callback (runs on a WGC worker thread): copies the newest
/// frame into the stable texture, rebuilding the pool when the window
/// changes size. Takes the core lock BEFORE touching the pool and bails on a
/// stale epoch: after a retarget or hide the core moves on first, and this
/// session's pool is torn down outside the lock — a superseded callback must
/// never recreate or read from a pool that is being closed.
fn frame_arrived(
    core: &SendCore,
    device: &AgileWinRTDevice,
    pool: &Direct3D11CaptureFramePool,
    epoch: u64,
) -> windows::core::Result<()> {
    let Ok(mut core) = core.0.lock() else {
        return Ok(());
    };
    if core.capture_epoch != epoch {
        // Superseded (retargeted or hidden): leave the dying pool alone.
        return Ok(());
    }
    let Ok(frame) = pool.TryGetNextFrame() else {
        return Ok(());
    };
    let Ok(size) = frame.ContentSize() else {
        return Ok(());
    };
    if size.Width <= 0 || size.Height <= 0 {
        return Ok(());
    }
    if core.frame_size != size {
        // The window resized (or DPI changed): reroute the pool and
        // rebuild the stable texture at the new size.
        let _ = pool.Recreate(
            device.0.as_ref(),
            DirectXPixelFormat::B8G8R8A8UIntNormalized,
            FRAME_POOL_BUFFERS,
            size,
        );
        match create_frame_texture(&core.device, size) {
            Ok((texture, view)) => {
                core.frame_texture = Some(texture);
                core.frame_view = Some(view);
                core.frame_size = size;
            }
            Err(error) => eprintln!("Glassbar thumbnail frame restart failed: {error}"),
        }
        return Ok(());
    }
    let Some(stable) = core.frame_texture.as_ref() else {
        return Ok(());
    };
    if let Ok(texture) = unsafe {
        frame
            .Surface()
            .and_then(|surface| surface.cast::<IDirect3DDxgiInterfaceAccess>())
            .and_then(|access| access.GetInterface::<ID3D11Texture2D>())
    } {
        unsafe {
            core.context.CopyResource(stable, &texture);
        }
        core.version += 1;
    }
    Ok(())
}

/// Computes the client-area crop of `target` in the WGC window item's frame
/// UV space: the item covers the whole window frame, and we sample only the
/// client rect inside it (insets measured in screen-space physical pixels).
fn client_crop_uv(target: isize) -> Option<([f32; 2], [f32; 2])> {
    let hwnd = HWND(target as *mut _);
    let mut window = RECT::default();
    let mut client = RECT::default();
    unsafe {
        GetWindowRect(hwnd, &mut window).ok()?;
        GetClientRect(hwnd, &mut client).ok()?;
    }
    let mut origin = POINT::default();
    unsafe {
        if !ClientToScreen(hwnd, &mut origin).as_bool() {
            return None;
        }
    }
    let width = window.right - window.left;
    let height = window.bottom - window.top;
    let client_width = client.right - client.left;
    let client_height = client.bottom - client.top;
    if width < 1 || height < 1 {
        return None;
    }
    Some((
        [
            (origin.x - window.left) as f32 / width as f32,
            (origin.y - window.top) as f32 / height as f32,
        ],
        [
            client_width.max(1) as f32 / width as f32,
            client_height.max(1) as f32 / height as f32,
        ],
    ))
}

fn create_frame_texture(
    device: &ID3D11Device,
    size: SizeInt32,
) -> Result<(ID3D11Texture2D, ID3D11ShaderResourceView), String> {
    let mut texture = None;
    unsafe {
        device.CreateTexture2D(
            &D3D11_TEXTURE2D_DESC {
                Width: size.Width as u32,
                Height: size.Height as u32,
                MipLevels: 1,
                ArraySize: 1,
                Format: DXGI_FORMAT_B8G8R8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Usage: D3D11_USAGE_DEFAULT,
                // It is sampled by the render loop and (deprecated) sized
                // as a render target for the initial transparent clear.
                BindFlags: (D3D11_BIND_SHADER_RESOURCE.0 | D3D11_BIND_RENDER_TARGET.0) as u32,
                CPUAccessFlags: 0,
                MiscFlags: 0,
            },
            None,
            Some(&mut texture),
        )
    }
    .map_err(|error| format!("Could not create the thumbnail frame texture: {error}"))?;
    let texture = texture.expect("thumbnail texture missing");
    let mut view = None;
    unsafe {
        device
            .CreateShaderResourceView(&texture, None, Some(&mut view))
            .map_err(|error| format!("Could not create the thumbnail frame view: {error}"))?;
    }
    Ok((texture, view.expect("thumbnail view missing")))
}

fn request_borderless_capture() {
    use windows::Graphics::Capture::{GraphicsCaptureAccess, GraphicsCaptureAccessKind};
    let _ = GraphicsCaptureAccess::RequestAccessAsync(GraphicsCaptureAccessKind::Borderless)
        .and_then(|operation| operation.get());
}
