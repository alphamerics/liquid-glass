use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use windows::Win32::Foundation::HWND;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D11::{
    ID3D11ShaderResourceView, ID3D11Texture2D, D3D11_VIEWPORT,
};
use windows::Win32::Graphics::Dxgi::DXGI_PRESENT;
use windows::Win32::UI::WindowsAndMessaging::IsWindowVisible;

use super::state::{SendCore, ThumbConstants, ThumbCore};
use super::TICK;
/// The render loop: presents one frame whenever the capture moved or a
/// parameter changed; sleeps at `TICK` otherwise (and while hidden).
pub(super) fn render_loop(core: SendCore, running: Arc<AtomicBool>) {
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
        // NEVER present into a hidden window. The host hides the popup
        // window BEFORE the surface worker marks the core invisible (the
        // hide's window ops and the teardown are deliberately decoupled), so
        // a capture frame arriving in that gap would otherwise be presented
        // into an already-hidden window. A flip-model composition swap
        // chain's Present BACKPRESSURES once the compositor stops consuming
        // (a hidden window composes nothing): with only two buffers, a
        // couple of un-consumed presents block `Present` forever — and this
        // loop holds the core lock across it, wedging the worker's teardown
        // (and every later hover) behind it. `IsWindowVisible` reads the
        // style bits without sending messages, so it is safe cross-thread.
        if !unsafe { IsWindowVisible(HWND(core.window as *mut _)) }.as_bool() {
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
            core.context
                .ClearRenderTargetView(&pair.1, &[0.0, 0.0, 0.0, 0.0]);
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
