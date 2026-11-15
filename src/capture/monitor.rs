//! Locating the capture item (the primary monitor) and the Windows 11
//! borderless-capture permission request.

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

/// Windows 11: grants the borderless-capture permission so
/// GraphicsCaptureSession::SetBorderRequired(false) is accepted. Requires the
/// packaged graphicsCaptureProgrammatic capability; no-op elsewhere.
pub(super) fn request_borderless_capture() {
    use windows::Graphics::Capture::{GraphicsCaptureAccess, GraphicsCaptureAccessKind};
    let _ = GraphicsCaptureAccess::RequestAccessAsync(GraphicsCaptureAccessKind::Borderless)
        .and_then(|operation| operation.get());
}
