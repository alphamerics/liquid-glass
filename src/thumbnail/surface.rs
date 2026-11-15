use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use windows::core::Interface;
use windows::Graphics::{
    DirectX::Direct3D11::IDirect3DDevice,
    SizeInt32,
};
use windows::Win32::Foundation::{BOOL, HWND};
use windows::Win32::Graphics::{
    DirectComposition::{
        DCompositionCreateDevice, IDCompositionDevice, IDCompositionTarget, IDCompositionVisual,
    },
    Dxgi::{
        Common::{DXGI_ALPHA_MODE_PREMULTIPLIED, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_SAMPLE_DESC},
        CreateDXGIFactory2, IDXGIDevice, IDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, DXGI_SCALING_STRETCH, DXGI_SWAP_CHAIN_DESC1, DXGI_SWAP_EFFECT_FLIP_DISCARD,
        DXGI_USAGE_RENDER_TARGET_OUTPUT,
    },
};
use windows::Win32::System::WinRT::Direct3D11::CreateDirect3D11DeviceFromDXGIDevice;

use crate::shared::{compiled_shader, constant_buffer, create_linear_clamp_sampler};
use crate::surface::create_d3d11_device;

use super::render::render_loop;
use super::THUMBNAIL_HLSL;
use super::ThumbnailSurface;
use super::capture::{measure_window, start_capture};
use super::state::{AgileWinRTDevice, SendCore, ThumbConstants, ThumbCore};
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
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
            BufferCount: 2,
            Scaling: DXGI_SCALING_STRETCH,
            SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
            // Premultiplied alpha: the shader's SDF mask owns the soft
            // edges, no window region involved.
            AlphaMode: DXGI_ALPHA_MODE_PREMULTIPLIED,
            Flags: 0,
        };
        let swap_chain =
            unsafe { factory.CreateSwapChainForComposition(&device, &description, None) }
                .map_err(|error| format!("Could not create the thumbnail swap chain: {error}"))?;
        let composition_device: IDCompositionDevice = unsafe {
            DCompositionCreateDevice(&dxgi_device)
        }
        .map_err(|error| format!("Could not create the thumbnail composition device: {error}"))?;
        let composition_target: IDCompositionTarget =
            unsafe { composition_device.CreateTargetForHwnd(HWND(window as *mut _), BOOL(1)) }
                .map_err(|error| {
                    format!("Could not create the thumbnail composition target: {error}")
                })?;
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
                .map_err(|error| format!("Could not create the thumbnail pixel shader: {error}"))?;
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
            capture_epoch: 0,
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
    /// crop that cuts the window's title frame out of the capture. A capture
    /// is only started for a visible source; a hidden surface records the
    /// source and sees `set_visible(true)` start the session.
    pub fn set_source(&self, target: isize) -> Result<(), String> {
        let (old, want, epoch) = {
            let mut core = self
                .core
                .lock()
                .map_err(|_| "thumbnail lock poisoned".to_owned())?;
            // Nothing to do when the source is already live (feed attached,
            // or the surface is hidden and will start on show).
            if core.source == target && (target == 0 || core.feed.is_some() || !core.visible) {
                return Ok(());
            }
            core.capture_epoch += 1;
            let old = core.feed.take();
            core.source = target;
            core.frame_texture = None;
            core.frame_view = None;
            core.frame_size = SizeInt32::default();
            core.version = 0;
            core.has_presented = false;
            core.crop = None;
            core.dirty = true;
            (old, target != 0 && core.visible, core.capture_epoch)
        };
        // The old session's WinRT teardown must NOT run under the core lock:
        // the frame callback needs that lock, and Close() synchronizes with
        // it (deadlock / race, see Capture::drop).
        drop(old);
        if want {
            self.attach_capture(target, epoch)?;
        }
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
        // Only resize when the swap chain actually disagrees with the
        // window: resizing in place repeatedly churns the DComp chain.
        if core.swap_chain_size[0] == size.0 && core.swap_chain_size[1] == size.1 {
            return Ok(());
        }
        core.back_buffer = None;
        unsafe {
            core.swap_chain
                .ResizeBuffers(
                    0,
                    size.0,
                    size.1,
                    DXGI_FORMAT_B8G8R8A8_UNORM,
                    Default::default(),
                )
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
        let (old, want) = {
            let Ok(mut core) = self.core.lock() else {
                return;
            };
            if core.visible == visible {
                return;
            }
            core.visible = visible;
            if visible {
                core.dirty = true;
                // A feed already attached (session still live) keeps it; a
                // bare source (visible toggled without a retarget) starts a
                // fresh session. 0 = nothing to capture; the caller sees an
                // empty (fully transparent) popup until a source is set.
                if core.feed.is_none() && core.source != 0 {
                    (None, Some(core.source))
                } else {
                    (None, None)
                }
            } else {
                // Tear the capture down: the popup is hidden, and a running
                // WGC session would keep the OS capture pipeline alive for
                // nobody. Bump the epoch first so the dying session's frame
                // callback bails before touching anything, then take the
                // session out — its teardown runs outside the lock.
                core.capture_epoch += 1;
                (core.feed.take(), None)
            }
        };
        // WinRT teardown outside the core lock (see Capture::drop).
        drop(old);
        if let Some(target) = want {
            // The capture is created without the core lock (the frame callback
            // re-enters it as soon as the session runs — see set_source).
            if let Err(error) = self.attach_capture(target, self.current_epoch()) {
                eprintln!("Glassbar thumbnail capture failed: {error}");
            }
        }
    }

    /// The core lock's capture epoch at this instant. Only safe on the thread
    /// that serializes `set_source`/`set_visible` (no bump can interleave).
    fn current_epoch(&self) -> u64 {
        self.core
            .lock()
            .map(|core| core.capture_epoch)
            .unwrap_or(u64::MAX)
    }

    /// Starts a WGC capture of `target` under `epoch` and attaches it only
    /// if the core's state has not moved on (a newer retarget or hide wins).
    fn attach_capture(&self, target: isize, epoch: u64) -> Result<(), String> {
        let capture = start_capture(&self.core, target, epoch)?;
        let mut core = self
            .core
            .lock()
            .map_err(|_| "thumbnail lock poisoned".to_owned())?;
        if core.capture_epoch == epoch && core.visible && core.source == target {
            core.feed = Some(capture);
            Ok(())
        } else {
            // Superseded while the session was starting (or the surface hid):
            // discard. The teardown must not run under the lock.
            drop(core);
            drop(capture);
            Ok(())
        }
    }
}
