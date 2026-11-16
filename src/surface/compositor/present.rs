//! The present pipeline: compositing one frame from the shared backdrop
//! into the swap chain and flipping it, plus the GPU readback helpers
//! (back-buffer acquisition, transparent clears, the CPU-side mean
//! diagnostic).
use std::time::{Duration, Instant};

use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D11::{
    ID3D11DeviceContext, ID3D11RenderTargetView, ID3D11Texture2D, D3D11_CPU_ACCESS_READ,
    D3D11_MAPPED_SUBRESOURCE, D3D11_MAP_READ, D3D11_TEXTURE2D_DESC, D3D11_USAGE_STAGING,
    D3D11_VIEWPORT,
};
use windows::Win32::Graphics::Dxgi::DXGI_PRESENT;

use super::ACTIVE_CONTENT_WINDOW;
use crate::shared::{Backdrop, CropConstants};

use super::D3D11Compositor;

impl D3D11Compositor {
    /// Composites this surface's frame from the shared blurred backdrop.
    /// The heavy blur runs once per capture frame in the engine; a present
    /// is one draw call (displace, tint, mask) plus the flip-model present.
    pub(crate) fn present(
        &mut self,
        context: &ID3D11DeviceContext,
        backdrop: &Backdrop,
    ) -> Result<(), String> {
        self.present_impl(context, backdrop, false)
    }

    /// [`D3D11Compositor::present`], bypassing the idle frame cap: the
    /// shared-field present sync renders a dragged metaball surface in the
    /// same op that stamped its field (and moved its window), so blob
    /// content and window position land on one composition beat instead of
    /// deferring to the loop's throttled tick.
    pub(crate) fn present_forced(
        &mut self,
        context: &ID3D11DeviceContext,
        backdrop: &Backdrop,
    ) -> Result<(), String> {
        self.present_impl(context, backdrop, true)
    }

    fn present_impl(
        &mut self,
        context: &ID3D11DeviceContext,
        backdrop: &Backdrop,
        forced: bool,
    ) -> Result<(), String> {
        if self.paused {
            return Ok(());
        }
        let pause_now = self.pause_after_present;
        // While an animation runs (reveal, shape pursuit or squash spring),
        // render at the caller's rate (the presentation tick keeps it
        // smooth). When idle or paused-gated, throttle content-driven
        // renders to the surface's frame cap — capture on a high-refresh
        // monitor would otherwise drive every surface at full refresh rate.
        // A forced present bypasses the cap: it must render RIGHT NOW (the
        // shared-field present sync renders a dragged metaball surface in
        // the same op that stamped its field and moved its window), and it
        // resets the cap's clock so the next loop tick does not
        // double-present the same beat.
        let animating = self.is_animating();
        if !animating && !pause_now && !forced {
            let now = Instant::now();
            // Adaptive surfaces (the taskbar pill) render at the shared
            // loop's full rate while fresh capture frames keep arriving —
            // content is moving somewhere, so the glass tracks it smoothly —
            // and fall back to the idle frame cap once the content has been
            // quiet past the grace window.
            let interval = if self.adaptive_rate
                && self
                    .last_content_frame
                    .map(|fresh| now.duration_since(fresh) < ACTIVE_CONTENT_WINDOW)
                    .unwrap_or(false)
            {
                Duration::ZERO
            } else {
                self.min_present_interval
            };
            if let Some(last) = self.last_present {
                if now.duration_since(last) < interval {
                    return Ok(());
                }
            }
            self.last_present = Some(now);
        } else if forced {
            self.last_present = Some(Instant::now());
        }
        self.params_dirty = false;
        self.update_pointer_velocity(context);
        self.update_reveal(context);
        // The compositor-owned springs integrate with the frame's real dt;
        // the combined update uploads the params itself when any value
        // moved this frame.
        self.update_springs(context);
        let (_, target) = self.back_buffer_view(context)?;
        // The backdrop carries the crop it was blurred over; the composite's
        // capture→backdrop mapping must follow it (identity for the
        // full-frame paths). Updated only on change, like the other buffers.
        if self.crop_constants_current != backdrop.crop {
            unsafe {
                context.UpdateSubresource(
                    &self.crop_constants,
                    0,
                    None,
                    (&backdrop.crop as *const CropConstants).cast(),
                    0,
                    0,
                );
            }
            self.crop_constants_current = backdrop.crop;
        }
        unsafe {
            context.OMSetRenderTargets(Some(&[Some(target.clone())]), None);
            context.RSSetViewports(Some(&[D3D11_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: self.width as f32,
                Height: self.height as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]));
            context.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            context.VSSetShader(&self.vertex_shader, None);
            let pixel_shader = if self.lite_shader {
                &self.pixel_shader_lite
            } else {
                &self.pixel_shader
            };
            context.PSSetShader(pixel_shader, None);
            context.PSSetSamplers(0, Some(&[Some(self.sampler.clone())]));
            context.PSSetShaderResources(0, Some(&[Some(backdrop.view.clone())]));
            context.PSSetConstantBuffers(0, Some(&[Some(self.source_rect.clone())]));
            context.PSSetConstantBuffers(1, Some(&[Some(self.glass_params.clone())]));
            context.PSSetConstantBuffers(2, Some(&[Some(self.crop_constants.clone())]));
            context.Draw(3, 0);
            // Sync interval 0: a composition swap chain cannot tear (DWM
            // composites it), and waiting for the vblank here used to hold
            // the shared engine's lock for a vblank per surface — with a
            // panel of glass open that serialized the whole render loop,
            // starved the capture callback and blocked every main-thread
            // glass call behind it.
            self.swap_chain
                .Present(0, DXGI_PRESENT(0))
                .ok()
                .map_err(|error| format!("Could not present the D3D11 frame: {error}"))?;
            self.has_presented = true;
            context.PSSetShaderResources(0, Some(&[None]));
        }
        if pause_now {
            self.paused = true;
            self.pause_after_present = false;
        }
        // The surface window is shown by `stack_beneath_host` on the thread
        // that created it; never touch window visibility from the capture
        // callback — ShowWindow/SetWindowPos are synchronous across threads
        // and deadlocked the compositor against the window's owner.
        Ok(())
    }

    /// The swap chain's back buffer and its render-target view, acquired
    /// once and reused until a resize invalidates them.
    pub(super) fn back_buffer_view(
        &mut self,
        context: &ID3D11DeviceContext,
    ) -> Result<(ID3D11Texture2D, ID3D11RenderTargetView), String> {
        if let Some((_, view)) = &self.back_buffer {
            return Ok((
                self.back_buffer.as_ref().expect("checked").0.clone(),
                view.clone(),
            ));
        }
        let back_buffer: ID3D11Texture2D = unsafe { self.swap_chain.GetBuffer(0) }
            .map_err(|error| format!("Could not acquire the D3D11 swap-chain buffer: {error}"))?;
        let mut target = None;
        unsafe {
            self.device
                .CreateRenderTargetView(&back_buffer, None, Some(&mut target))
        }
        .map_err(|error| format!("Could not create the D3D11 render target view: {error}"))?;
        let pair = (
            back_buffer,
            target.expect("D3D11 render target view missing"),
        );
        self.back_buffer = Some(pair.clone());
        // A fresh (or resized) buffer can hold garbage from its last life:
        // clear it once so the surface never flashes a black box.
        unsafe { context.ClearRenderTargetView(&pair.1, &[0.0, 0.0, 0.0, 0.0]) };
        Ok(pair)
    }

    /// Presents one fully transparent frame so freshly (re)allocated swap chain
    /// buffers never show as a black box before the next capture frame lands.
    pub fn clear_to_transparent(&mut self, context: &ID3D11DeviceContext) -> Result<(), String> {
        let (_, target) = self.back_buffer_view(context)?;
        unsafe {
            context.ClearRenderTargetView(&target, &[0.0, 0.0, 0.0, 0.0]);
            self.swap_chain
                .Present(0, DXGI_PRESENT(0))
                .ok()
                .map_err(|error| format!("Could not present the clear frame: {error}"))?;
        }
        Ok(())
    }

    /// Diagnostic: copies the last rendered frame (taken before Present) to
    /// the CPU and returns the mean BGRA (0..255 per channel) of one sub-rect,
    /// given as normalized fractions (x0, y0, x1, y1); `None` covers the whole
    /// buffer.
    /// Diagnostic: copies the back buffer to the CPU and returns its mean BGRA
    /// (0..255 per channel). Reveals whether the surface is transparent.
    pub fn backbuffer_mean(&mut self, context: &ID3D11DeviceContext) -> Result<[f32; 4], String> {
        let back_buffer: ID3D11Texture2D = unsafe { self.swap_chain.GetBuffer(0) }
            .map_err(|error| format!("Could not acquire the D3D11 swap-chain buffer: {error}"))?;
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        unsafe { back_buffer.GetDesc(&mut desc) };
        let staging_desc = D3D11_TEXTURE2D_DESC {
            Usage: D3D11_USAGE_STAGING,
            BindFlags: 0,
            CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
            MiscFlags: 0,
            ..desc
        };
        let mut staging: Option<ID3D11Texture2D> = None;
        unsafe {
            self.device
                .CreateTexture2D(&staging_desc, None, Some(&mut staging))
        }
        .map_err(|error| format!("Could not create the staging texture: {error}"))?;
        let staging = staging.expect("D3D11 staging texture missing");
        unsafe { context.CopyResource(&staging, &back_buffer) };
        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        unsafe { context.Map(&staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped)) }
            .map_err(|error| format!("Could not map the staging texture: {error}"))?;
        let row = unsafe {
            std::slice::from_raw_parts(
                mapped.pData as *const u8,
                mapped.RowPitch as usize * desc.Height as usize,
            )
        };
        let (mut b, mut g, mut r, mut a, mut count) = (0f64, 0f64, 0f64, 0f64, 0u64);
        for y in 0..desc.Height as usize {
            let step = (desc.Width as usize / 64).max(1);
            for x in (0..desc.Width as usize).step_by(step) {
                let offset = y * mapped.RowPitch as usize + x * 4;
                b += row[offset] as f64;
                g += row[offset + 1] as f64;
                r += row[offset + 2] as f64;
                a += row[offset + 3] as f64;
                count += 1;
            }
        }
        unsafe { context.Unmap(&staging, 0) };
        let total = count.max(1) as f64;
        Ok([
            (b / total) as f32,
            (g / total) as f32,
            (r / total) as f32,
            (a / total) as f32,
        ])
    }
}
