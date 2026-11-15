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



const THUMBNAIL_HLSL: &str = include_str!("../thumbnail.hlsl");
const FRAME_POOL_BUFFERS: i32 = 2;
/// The render tick. Frames are only presented when the capture moves or a
/// parameter changes, so an idle (but visible) preview draws nothing.
const TICK: Duration = Duration::from_millis(16);

mod capture;
mod state;

use state::ThumbCore;
mod render;
mod surface;

/// A live window thumbnail bound to `window` (the preview popup's HWND, as
/// an isize): its composition swap chain renders into that window.
pub struct ThumbnailSurface {
    core: Arc<Mutex<ThumbCore>>,
    render_running: Arc<AtomicBool>,
    render_thread: Option<std::thread::JoinHandle<()>>,
}

unsafe impl Send for ThumbnailSurface {}
unsafe impl Sync for ThumbnailSurface {}

impl Drop for ThumbnailSurface {
    fn drop(&mut self) {
        self.render_running.store(false, Ordering::Relaxed);
        if let Some(thread) = self.render_thread.take() {
            let _ = thread.join();
        }
    }
}
