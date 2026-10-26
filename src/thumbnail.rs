//! A live window thumbnail, rendered by our GPU rather than the DWM
//! thumbnail compositor.
//!
//! The DWM thumbnail sprite composites at the Windows 11 system corner
//! radius only — it ignores window regions entirely, so its corners can
//! never follow a caller's radius. This module instead captures the target
//! window with Windows Graphics Capture (`CreateForWindow`) and draws the
//! newest frame into a DirectComposition swap chain bound to the preview
//! popup's window, masked by an in-shader rounded-rect SDF. The corner
//! radius is whatever the caller passes, in physical pixels.
//!
//! Each surface owns a private D3D11 device and render thread — separate
//! from the shared glass engine, because this is a content feed (the target
//! window's own pixels), not a backdrop filter, and it only runs while a
//! hover preview is open. Hiding the surface tears the WGC session down so
//! the capture pipeline idles with the preview.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use windows::core::Interface;
use windows::Graphics::{
    Capture::{Direct3D11CaptureFramePool, GraphicsCaptureItem, GraphicsCaptureSession},
    DirectX::{Direct3D11::IDirect3DDevice, DirectXPixelFormat},
    SizeInt32,
};
use windows::Win32::Foundation::{BOOL, HWND, POINT, RECT};
use windows::Win32::Graphics::Gdi::ClientToScreen;
use windows::Win32::Graphics::{
    Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
    Direct3D11::{
        ID3D11Buffer, ID3D11Device, ID3D11DeviceContext, ID3D11PixelShader,
        ID3D11RenderTargetView, ID3D11SamplerState, ID3D11ShaderResourceView, ID3D11Texture2D,
        ID3D11VertexShader, D3D11_BIND_RENDER_TARGET, D3D11_BIND_SHADER_RESOURCE,
        D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT, D3D11_VIEWPORT,
    },
    DirectComposition::{
        DCompositionCreateDevice, IDCompositionDevice, IDCompositionTarget, IDCompositionVisual,
    },
    Dxgi::{
        Common::{DXGI_ALPHA_MODE_PREMULTIPLIED, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_SAMPLE_DESC},
        CreateDXGIFactory2, IDXGIDevice, IDXGIFactory2, IDXGISwapChain1,
        DXGI_CREATE_FACTORY_FLAGS, DXGI_PRESENT, DXGI_SCALING_STRETCH, DXGI_SWAP_CHAIN_DESC1,
        DXGI_SWAP_EFFECT_FLIP_DISCARD, DXGI_USAGE_RENDER_TARGET_OUTPUT,
    },
};
use windows::Win32::System::WinRT::{
    Direct3D11::{CreateDirect3D11DeviceFromDXGIDevice, IDirect3DDxgiInterfaceAccess},
    Graphics::Capture::IGraphicsCaptureItemInterop,
};
use windows::Win32::UI::WindowsAndMessaging::{GetClientRect, GetWindowRect};

use crate::shared::{compiled_shader, constant_buffer, create_linear_clamp_sampler};
use crate::surface::create_d3d11_device;

const THUMBNAIL_HLSL: &str = include_str!("thumbnail.hlsl");
const FRAME_POOL_BUFFERS: i32 = 2;
/// The render tick. Frames are only presented when the capture moves or a
/// parameter changes, so an idle (but visible) preview draws nothing.
const TICK: Duration = Duration::from_millis(16);

/// Mirrors the shader's `ThumbParams` constant buffer (register b0).
#[repr(C)]
#[derive(Clone, Copy)]
struct ThumbConstants {
    /// The client-area crop in frame UV (the WGC window item covers the
    /// whole title-barred frame; the old thumbnail cropped it via rcSource).
    crop_origin_uv: [f32; 2],
    crop_size_uv: [f32; 2],
    /// Corner radius normalized to the quad's height (0..0.5).
    corner_radius: f32,
    /// Quad width / height, for aspect-corrected SDF corners.
    quad_aspect: f32,
    padding: [f32; 2],
}

/// The WGC per-window capture session.
struct Capture {
    frame_pool: Direct3D11CaptureFramePool,
    session: GraphicsCaptureSession,
    frame_arrived_token: windows::Foundation::EventRegistrationToken,
}

impl Drop for Capture {
    fn drop(&mut self) {
        let _ = self.frame_pool.RemoveFrameArrived(self.frame_arrived_token);
        let _ = self.session.Close();
        let _ = self.frame_pool.Close();
    }
}

/// `CreateDirect3D11DeviceFromDXGIDevice` produces an agile WinRT wrapper
/// around the free-threaded D3D11 device; windows-rs does not mark it `Send`
/// although WGC invokes the frame callback on a worker thread.
#[derive(Clone)]
struct AgileWinRTDevice(Arc<IDirect3DDevice>);

unsafe impl Send for AgileWinRTDevice {}

struct ThumbCore {
    device: ID3D11Device,
    context: ID3D11DeviceContext,
    winrt_device: AgileWinRTDevice,
    swap_chain: IDXGISwapChain1,
    #[allow(dead_code)]
    composition_device: IDCompositionDevice,
    #[allow(dead_code)]
    composition_target: IDCompositionTarget,
    #[allow(dead_code)]
    composition_visual: IDCompositionVisual,
    vertex_shader: ID3D11VertexShader,
    pixel_shader: ID3D11PixelShader,
    sampler: ID3D11SamplerState,
    constants: ID3D11Buffer,
    back_buffer: Option<(ID3D11Texture2D, ID3D11RenderTargetView)>,
    swap_chain_size: [u32; 2],
    /// The newest captured frame, copied out of the WGC pool (whose surfaces
    /// are recycled) into a stable texture the render loop samples.
    frame_texture: Option<ID3D11Texture2D>,
    frame_view: Option<ID3D11ShaderResourceView>,
    frame_size: SizeInt32,
    /// Bumped by the capture callback when a fresh frame has been copied in.
    version: u64,
    /// The client-area crop in frame UV space, computed at set_source.
    crop: Option<([f32; 2], [f32; 2])>,
    corner_radius_px: f32,
    /// The window being captured (0 = none).
    source: isize,
    /// The composition host (the preview popup's hwnd as an isize).
    window: isize,
    feed: Option<Capture>,
    visible: bool,
    has_presented: bool,
    dirty: bool,
}

impl ThumbCore {
    fn compute_constants(&self) -> ThumbConstants {
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
struct SendCore(Arc<Mutex<ThumbCore>>);

unsafe impl Send for SendCore {}
unsafe impl Sync for SendCore {}

/// A live window thumbnail bound to `window` (the preview popup's HWND, as
/// an isize): its composition swap chain renders into that window.
pub struct ThumbnailSurface {
    core: Arc<Mutex<ThumbCore>>,
    render_running: Arc<AtomicBool>,
    render_thread: Option<std::thread::JoinHandle<()>>,
}

unsafe impl Send for ThumbnailSurface {}
unsafe impl Sync for ThumbnailSurface {}

impl ThumbnailSurface {
    /// Creates the surface on `window`. The window's current size is adopt-
    /// ed; call [`ThumbnailSurface::sync_window_size`] after moving it.
    pub fn create(window: isize) -> Result<Self, String> {
        let (device, context) = create_d3d11_device()?;
        let dxgi_device = device
            .cast::<IDXGIDevice>()
            .map_err(|error| format!("Could not obtain the thumbnail DXGI device: {error}"))?;
        let winrt_device = AgileWinRTDevice(Arc::new(unsafe {
            CreateDirect3D11DeviceFromDXGIDevice(&dxgi_device)
                .and_then(|wrapped| wrapped.cast::<IDirect3DDevice>())
                .map_err(|error| format!("Could not create the WinRT D3D11 device: {error}"))?
        }));

        let size = measure_window(window)?;
        let swap_chain_size = [size.0, size.1];
        let factory: IDXGIFactory2 = unsafe { CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)) }
            .map_err(|error| format!("Could not create the thumbnail DXGI factory: {error}"))?;
        let description = DXGI_SWAP_CHAIN_DESC1 {
            Width: swap_chain_size[0],
            Height: swap_chain_size[1],
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            Stereo: BOOL(0),
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
            BufferCount: 2,
            Scaling: DXGI_SCALING_STRETCH,
            SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
            // Premultiplied alpha: the shader's SDF mask owns the soft
            // edges, no window region involved.
            AlphaMode: DXGI_ALPHA_MODE_PREMULTIPLIED,
            Flags: 0,
        };
        let swap_chain = unsafe {
            factory.CreateSwapChainForComposition(&device, &description, None)
        }
        .map_err(|error| format!("Could not create the thumbnail swap chain: {error}"))?;
        let composition_device: IDCompositionDevice =
            unsafe { DCompositionCreateDevice(&dxgi_device) }
                .map_err(|error| {
                    format!("Could not create the thumbnail composition device: {error}")
                })?;
        let composition_target: IDCompositionTarget = unsafe {
            composition_device.CreateTargetForHwnd(HWND(window as *mut _), BOOL(1))
        }
        .map_err(|error| format!("Could not create the thumbnail composition target: {error}"))?;
        let composition_visual: IDCompositionVisual = unsafe { composition_device.CreateVisual() }
            .map_err(|error| format!("Could not create the thumbnail visual: {error}"))?;
        unsafe {
            composition_visual
                .SetContent(&swap_chain)
                .map_err(|error| format!("Could not bind the thumbnail swap chain: {error}"))?;
            composition_target
                .SetRoot(&composition_visual)
                .map_err(|error| format!("Could not root the thumbnail visual: {error}"))?;
            composition_device
                .Commit()
                .map_err(|error| format!("Could not commit the thumbnail visual tree: {error}"))?;
        }

        let mut vertex_shader = None;
        let mut pixel_shader = None;
        unsafe {
            device
                .CreateVertexShader(
                    &compiled_shader(THUMBNAIL_HLSL, "vs_main", "vs_5_0")?,
                    None,
                    Some(&mut vertex_shader),
                )
                .map_err(|error| {
                    format!("Could not create the thumbnail vertex shader: {error}")
                })?;
            device
                .CreatePixelShader(
                    &compiled_shader(THUMBNAIL_HLSL, "ps_main", "ps_5_0")?,
                    None,
                    Some(&mut pixel_shader),
                )
                .map_err(|error| {
                    format!("Could not create the thumbnail pixel shader: {error}")
                })?;
        }
        let initial_constants = ThumbConstants {
            crop_origin_uv: [0.0; 2],
            crop_size_uv: [1.0, 1.0],
            corner_radius: 0.25,
            quad_aspect: swap_chain_size[0] as f32 / swap_chain_size[1].max(1) as f32,
            padding: [0.0; 2],
        };
        let constants = constant_buffer(&device, &initial_constants)?;

        let core = Arc::new(Mutex::new(ThumbCore {
            device: device.clone(),
            context: context.clone(),
            winrt_device,
            swap_chain,
            composition_device,
            composition_target,
            composition_visual,
            vertex_shader: vertex_shader.expect("thumbnail vertex shader missing"),
            pixel_shader: pixel_shader.expect("thumbnail pixel shader missing"),
            sampler: create_linear_clamp_sampler(&device),
            constants,
            back_buffer: None,
            swap_chain_size,
            frame_texture: None,
            frame_view: None,
            frame_size: SizeInt32::default(),
            version: 0,
            crop: None,
            corner_radius_px: 13.0,
            source: 0,
            window,
            feed: None,
            visible: false,
            has_presented: false,
            dirty: false,
        }));

        let render_running = Arc::new(AtomicBool::new(true));
        let thread_core = SendCore(Arc::clone(&core));
        let thread_running = Arc::clone(&render_running);
        let render_thread = std::thread::spawn(move || render_loop(thread_core, thread_running));

        Ok(Self {
            core,
            render_running,
            render_thread: Some(render_thread),
        })
    }

    /// Captures `target` (an HWND value; 0 = none). Restarts the capture
    /// session when the source changes and (re)computes the client-area
    /// crop that cuts the window's title frame out of the capture.
    pub fn set_source(&self, target: isize) -> Result<(), String> {
        let start = {
            let mut core = self
                .core
                .lock()
                .map_err(|_| "thumbnail lock poisoned".to_owned())?;
            if core.source == target {
                return Ok(());
            }
            core.source = target;
            core.feed = None;
            core.frame_texture = None;
            core.frame_view = None;
            core.frame_size = SizeInt32::default();
            core.version = 0;
            core.has_presented = false;
            core.crop = None;
            core.dirty = true;
            target != 0 && core.visible
        };
        // The capture is created without the core lock: the frame callback
        // re-enters it as soon as the session runs.
        let capture = start_capture(&self.core, target)?;
        let mut core = self
            .core
            .lock()
            .map_err(|_| "thumbnail lock poisoned".to_owned())?;
        core.feed = Some(capture);
        let _ = start;
        Ok(())
    }

    /// Sets the corner radius in physical pixels.
    pub fn set_radius(&self, radius_px: f32) {
        let Ok(mut core) = self.core.lock() else {
            return;
        };
        let radius = radius_px.max(0.0);
        if (core.corner_radius_px - radius).abs() > f32::EPSILON {
            core.corner_radius_px = radius;
            core.dirty = true;
        }
    }

    /// Re-measures the bound window and resizes the swap chain to match.
    /// Call whenever the popup moved or resized.
    pub fn sync_window_size(&self) -> Result<(), String> {
        let size = {
            let core = self
                .core
                .lock()
                .map_err(|_| "thumbnail lock poisoned".to_owned())?;
            measure_window(core.window)?
        };
        let mut core = self
            .core
            .lock()
            .map_err(|_| "thumbnail lock poisoned".to_owned())?;
        // A frame may arrive between the two locks and disagree with the
        // fresh measurement; that frame is dropped on the next draw match,
        // and the capture pool itself does not depend on the popup size.
        if core.swap_chain_size[0] != size.0 || core.swap_chain_size[1] != size.1 {
            return Ok(());
        }
        core.back_buffer = None;
        unsafe {
            core.swap_chain
                .ResizeBuffers(0, size.0, size.1, DXGI_FORMAT_B8G8R8A8_UNORM, Default::default())
                .map_err(|error| format!("Could not resize the thumbnail swap chain: {error}"))?;
        }
        core.swap_chain_size = [size.0, size.1];
        core.dirty = true;
        Ok(())
    }

    /// Shows or hides the thumbnail. Showing starts the WGC capture session
    /// (if not already running); hiding stops it, so the capture pipeline
    /// idles whenever the preview popup is hidden.
    pub fn set_visible(&self, visible: bool) {
        let start_target = {
            let Ok(mut core) = self.core.lock() else {
                return;
            };
            if core.visible == visible {
                return;
            }
            core.visible = visible;
            if visible {
                core.dirty = true;
                // 0 = nothing to capture; the caller sees an empty (fully
                // transparent) popup until a source is set.
                core.source
            } else {
                // Tear the capture down: the popup is hidden, and a running
                // WGC session would keep the OS capture pipeline alive for
                // nobody.
                core.feed = None;
                0
            }
        };
        if start_target == 0 {
            return;
        }
        // The capture is created without the core lock (the frame callback
        // re-enters it as soon as the session runs — see set_source).
        match start_capture(&self.core, start_target) {
            Ok(capture) => {
                let Ok(mut core) = self.core.lock() else {
                    return;
                };
                core.feed = Some(capture);
            }
            Err(error) => eprintln!("Glassbar thumbnail capture failed: {error}"),
        }
    }
}

impl Drop for ThumbnailSurface {
    fn drop(&mut self) {
        self.render_running.store(false, Ordering::Relaxed);
        if let Some(thread) = self.render_thread.take() {
            let _ = thread.join();
        }
    }
}

/// The render loop: presents one frame whenever the capture moved or a
/// parameter changed; sleeps at `TICK` otherwise (and while hidden).
fn render_loop(core: SendCore, running: Arc<AtomicBool>) {
    let mut seen_version = 0u64;
    while running.load(Ordering::Relaxed) {
        std::thread::sleep(TICK);
        let Ok(mut core) = core.0.lock() else {
            return;
        };
        if !core.visible {
            seen_version = 0;
            continue;
        }
        if !core.dirty && core.has_presented && core.version == seen_version {
            continue;
        }
        core.dirty = false;
        seen_version = core.version;
        let Some(view) = core.frame_view.clone() else {
            continue;
        };
        match draw(&mut core, view) {
            Ok(()) => core.has_presented = true,
            Err(error) if !core.has_presented => {
                eprintln!("Glassbar thumbnail render failed: {error}")
            }
            Err(_) => {}
        }
    }
}

fn draw(core: &mut ThumbCore, view: ID3D11ShaderResourceView) -> Result<(), String> {
    unsafe {
        if core.back_buffer.is_none() {
            let back_buffer: ID3D11Texture2D = core
                .swap_chain
                .GetBuffer(0)
                .map_err(|error| format!("Could not acquire the thumbnail buffer: {error}"))?;
            let mut target = None;
            core.device
                .CreateRenderTargetView(&back_buffer, None, Some(&mut target))
                .map_err(|error| {
                    format!("Could not create the thumbnail render target: {error}")
                })?;
            let pair = (
                back_buffer,
                target.expect("thumbnail render target missing"),
            );
            core.context.ClearRenderTargetView(&pair.1, &[0.0, 0.0, 0.0, 0.0]);
            core.back_buffer = Some(pair);
        }
        let (_, target) = core
            .back_buffer
            .as_ref()
            .expect("thumbnail back buffer missing");
        let [width, height] = core.swap_chain_size;
        let context = &core.context;
        // The frame may have a different size than the popup (the capture
        // pool lags a resize beat): the constants run in normalized UVs, so
        // the quad maps the newest frame onto the popup's shape untouched.
        let constants = core.compute_constants();
        context.UpdateSubresource(
            &core.constants,
            0,
            None,
            (&constants as *const ThumbConstants).cast(),
            0,
            0,
        );
        context.OMSetRenderTargets(Some(&[Some(target.clone())]), None);
        context.RSSetViewports(Some(&[D3D11_VIEWPORT {
            TopLeftX: 0.0,
            TopLeftY: 0.0,
            Width: width.max(1) as f32,
            Height: height.max(1) as f32,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        }]));
        context.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context.VSSetShader(&core.vertex_shader, None);
        context.PSSetShader(&core.pixel_shader, None);
        context.PSSetConstantBuffers(0, Some(&[Some(core.constants.clone())]));
        context.PSSetSamplers(0, Some(&[Some(core.sampler.clone())]));
        context.PSSetShaderResources(0, Some(&[Some(view)]));
        context.Draw(3, 0);
        // Unbind the frame view: it may be rewritten by the capture callback
        // between draws.
        context.PSSetShaderResources(0, Some(&[None]));
        core.swap_chain
            .Present(0, DXGI_PRESENT(0))
            .ok()
            .map_err(|error| format!("Could not present the thumbnail: {error}"))?;
    }
    Ok(())
}

fn measure_window(window: isize) -> Result<(u32, u32), String> {
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
/// shared D3D handles are cloned out under a brief lock first.
fn start_capture(core: &Arc<Mutex<ThumbCore>>, target: isize) -> Result<Capture, String> {
    // Windows 11: request the borderless-capture permission once so the
    // session can disable the yellow capture border (no-op unpackaged).
    request_borderless_capture();

    let interop: IGraphicsCaptureItemInterop = windows::core::factory::<
        GraphicsCaptureItem,
        IGraphicsCaptureItemInterop,
    >()
    .map_err(|error| {
        format!("Could not obtain the GraphicsCaptureItem interop factory: {error}")
    })?;
    let item: GraphicsCaptureItem = unsafe { interop.CreateForWindow(HWND(target as *mut _)) }
        .map_err(|error| format!("Could not capture the target window: {error}"))?;
    let size = item
        .Size()
        .map_err(|error| format!("Could not read the target window size: {error}"))?;

    let winrt_device = {
        let core = core.lock().map_err(|_| "thumbnail lock poisoned".to_owned())?;
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
    let frame_arrived_handler = windows::Foundation::TypedEventHandler::<
        Direct3D11CaptureFramePool,
        windows::core::IInspectable,
    >::new(move |pool, _| {
        let Some(pool) = pool.as_ref() else {
            return Ok(());
        };
        frame_arrived(&handler_core, &winrt_device, pool)
    });
    let frame_arrived_token = frame_pool
        .FrameArrived(&frame_arrived_handler)
        .map_err(|error| format!("Could not register the thumbnail frame callback: {error}"))?;
    session
        .StartCapture()
        .map_err(|error| format!("Could not start the window capture: {error}"))?;

    // Client-area inset measured once: a moving window keeps the same
    // frame-relative crop; a resizing one recomputes on the next retarget.
    let mut core = core.lock().map_err(|_| "thumbnail lock poisoned".to_owned())?;
    core.crop = client_crop_uv(target);

    Ok(Capture {
        frame_pool,
        session,
        frame_arrived_token,
    })
}

/// The frame callback (runs on a WGC worker thread): copies the newest
/// frame into the stable texture, rebuilding the pool when the window
/// changes size. Iterating `TryGetNextFrame` drains the pool so a burst of
/// frames never leaves a stale one behind.
fn frame_arrived(
    core: &SendCore,
    device: &AgileWinRTDevice,
    pool: &Direct3D11CaptureFramePool,
) -> windows::core::Result<()> {
    loop {
        let Ok(frame) = pool.TryGetNextFrame() else {
            return Ok(());
        };
        let Ok(size) = frame.ContentSize() else {
            return Ok(());
        };
        if size.Width <= 0 || size.Height <= 0 {
            return Ok(());
        }
        let Ok(mut core) = core.0.lock() else {
            return Ok(());
        };
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
        return Ok(());
    }
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
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
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
