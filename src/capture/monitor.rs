//! Locating the capture item (the primary monitor) and the Windows 11
//! borderless-capture permission request.

use std::sync::OnceLock;

use windows::Graphics::Capture::GraphicsCaptureItem;
use windows::Win32::Foundation::POINT;
use windows::Win32::Graphics::Gdi::{MonitorFromPoint, MONITOR_DEFAULTTOPRIMARY};
use windows::Win32::System::WinRT::Graphics::Capture::IGraphicsCaptureItemInterop;
pub(super) fn primary_monitor_item() -> Result<GraphicsCaptureItem, String> {
    let monitor = unsafe { MonitorFromPoint(POINT::default(), MONITOR_DEFAULTTOPRIMARY) };
    if monitor.0.is_null() {
        return Err("Could not locate the primary monitor".to_owned());
    }
    let interop: IGraphicsCaptureItemInterop =
        windows::core::factory::<GraphicsCaptureItem, IGraphicsCaptureItemInterop>().map_err(
            |error| format!("Could not obtain the GraphicsCaptureItem interop factory: {error}"),
        )?;
    unsafe {
        interop
            .CreateForMonitor(monitor)
            .map_err(|error| format!("Could not create the primary-monitor capture item: {error}"))
    }
}

static BORDERLESS_REQUESTED: OnceLock<()> = OnceLock::new();

/// Windows 11: grants the borderless-capture permission so
/// GraphicsCaptureSession::SetBorderRequired(false) is accepted. Requires the
/// packaged graphicsCaptureProgrammatic capability; no-op elsewhere.
///
/// Requested ONCE per process, on a worker thread: awaiting the async
/// operation with `get()` on an STA thread pumps that thread's message loop
/// (COM reentrancy) — and the thumbnail's capture session starts on the
/// host app's MAIN thread (inside a `run_on_main_thread` closure), where a
/// pumped message can re-enter the caller's state machine mid-start (a
/// queued hide/show runs re-entrantly while the caller still holds its
/// locks). The request only needs to complete eventually: without the grant
/// the sessions degrade to a logged `SetIsBorderRequired` failure.
pub(crate) fn request_borderless_capture() {
    if BORDERLESS_REQUESTED.set(()).is_err() {
        return;
    }
    std::thread::spawn(|| {
        use windows::Graphics::Capture::{GraphicsCaptureAccess, GraphicsCaptureAccessKind};
        let _ = GraphicsCaptureAccess::RequestAccessAsync(GraphicsCaptureAccessKind::Borderless)
            .and_then(|operation| operation.get());
    });
}
