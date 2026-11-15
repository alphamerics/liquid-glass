//! The capture host the shared feed delivers frames into — identical
//! plumbing to the in-process engine's (a stable copy of each frame, a
//! version counter, and the any-active answer that suppresses copies for
//! parked glass).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use windows::Graphics::SizeInt32;
use windows::Win32::Graphics::Direct3D11::{
    ID3D11Device, ID3D11DeviceContext, ID3D11ShaderResourceView, ID3D11Texture2D,
};

use super::consumers::ConsumerRegistry;
pub(crate) struct BrokerHost {
    device: ID3D11Device,
    context: ID3D11DeviceContext,
    pub(super) stable: Option<(ID3D11Texture2D, ID3D11ShaderResourceView)>,
    pub(super) frame_size: SizeInt32,
    pub(super) frame_version: Arc<AtomicU64>,
    registry: Arc<ConsumerRegistry>,
    /// The quality tier's backdrop rate cap (the publish loop writes it;
    /// the capture callback reads it through `capture_due`).
    backdrop_interval: Option<Duration>,
    /// When the last capture blit ran (rate-cap bookkeeping).
    last_copy: Option<Instant>,
}

unsafe impl Send for BrokerHost {}

impl crate::capture::FrameHost for BrokerHost {
    fn device(&self) -> &ID3D11Device {
        &self.device
    }

    fn copy_context(&self) -> ID3D11DeviceContext {
        self.context.clone()
    }

    fn set_frame(&mut self, frame: (ID3D11Texture2D, ID3D11ShaderResourceView), size: SizeInt32) {
        self.stable = Some(frame);
        self.frame_size = size;
        self.frame_version.fetch_add(1, Ordering::Release);
    }

    fn frame_texture(&self) -> Option<&ID3D11Texture2D> {
        self.stable.as_ref().map(|(texture, _)| texture)
    }

    fn frame_size(&self) -> SizeInt32 {
        self.frame_size
    }

    fn any_active(&self) -> bool {
        self.registry.any_active()
    }

    fn capture_due(&mut self) -> bool {
        let due = self.backdrop_interval.map_or(true, |interval| {
            self.last_copy
                .map_or(true, |last| last.elapsed() >= interval)
        });
        if due {
            self.last_copy = Some(Instant::now());
        }
        due
    }
}

impl BrokerHost {
    pub(super) fn new(
        device: ID3D11Device,
        context: ID3D11DeviceContext,
        registry: Arc<ConsumerRegistry>,
    ) -> Self {
        Self {
            device,
            context,
            stable: None,
            frame_size: SizeInt32::default(),
            frame_version: Arc::new(AtomicU64::new(0)),
            registry,
            backdrop_interval: None,
            last_copy: None,
        }
    }

    pub(super) fn set_backdrop_interval(&mut self, interval: Option<Duration>) {
        self.backdrop_interval = interval;
    }

    pub(super) fn version_counter(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.frame_version)
    }
}
