use std::mem::size_of;

use windows::Win32::{
    Foundation::{HWND, RECT},
    Graphics::Gdi::{GetMonitorInfoW, MonitorFromPoint, MONITORINFO, MONITOR_DEFAULTTOPRIMARY},
    UI::WindowsAndMessaging::GetWindowRect,
};

use crate::params::SurfaceAnchor;
use crate::SourceRectConstants;

/// The capture texel size (in uv units) the composite's displacement math
/// is expressed in: one over the primary monitor's pixel size.
pub(super) fn source_rect_constants_texel(monitor: &RECT) -> [f32; 2] {
    [
        1.0 / (monitor.right - monitor.left).max(1) as f32,
        1.0 / (monitor.bottom - monitor.top).max(1) as f32,
    ]
}

/// Shrinks the host rect to the part of it the surface should cover.
pub(super) fn anchor_rect(bounds: RECT, anchor: &SurfaceAnchor) -> RECT {
    match anchor {
        SurfaceAnchor::HostBounds => bounds,
        SurfaceAnchor::BottomStrip { height_px } => RECT {
            left: bounds.left,
            top: (bounds.bottom - (*height_px as i32)).max(bounds.top),
            right: bounds.right,
            bottom: bounds.bottom,
        },
        SurfaceAnchor::Rect {
            offset_x,
            offset_y,
            width,
            height,
        } => {
            let left = bounds.left + (*offset_x as i32);
            let top = bounds.top + (*offset_y as i32);
            RECT {
                left,
                top,
                right: left + (*width as i32),
                bottom: top + (*height as i32),
            }
        }
    }
}

pub(super) fn overlay_bounds(overlay: HWND) -> Result<RECT, String> {
    unsafe {
        let mut bounds = RECT::default();
        GetWindowRect(overlay, &mut bounds)
            .map_err(|error| format!("Could not read overlay bounds: {error}"))?;
        if bounds.right <= bounds.left || bounds.bottom <= bounds.top {
            return Err("Overlay bounds are empty".to_owned());
        }
        Ok(bounds)
    }
}

pub(crate) fn primary_monitor_bounds() -> Result<RECT, String> {
    unsafe {
        let monitor = MonitorFromPoint(Default::default(), MONITOR_DEFAULTTOPRIMARY);
        if monitor.0.is_null() {
            return Err("Could not locate the primary monitor".to_owned());
        }
        let mut info = MONITORINFO {
            cbSize: size_of::<MONITORINFO>() as u32,
            ..Default::default()
        };
        if !GetMonitorInfoW(monitor, &mut info).as_bool() {
            return Err(format!(
                "Could not read primary monitor bounds: {}",
                windows::core::Error::from_win32()
            ));
        }
        if info.rcMonitor.right <= info.rcMonitor.left
            || info.rcMonitor.bottom <= info.rcMonitor.top
        {
            return Err("Primary monitor bounds are empty".to_owned());
        }
        Ok(info.rcMonitor)
    }
}

pub(super) fn source_rect_constants(
    surface_bounds: RECT,
    primary_monitor_bounds: RECT,
) -> Result<SourceRectConstants, String> {
    let monitor_width = primary_monitor_bounds.right - primary_monitor_bounds.left;
    let monitor_height = primary_monitor_bounds.bottom - primary_monitor_bounds.top;
    let surface_width = surface_bounds.right - surface_bounds.left;
    let surface_height = surface_bounds.bottom - surface_bounds.top;
    if monitor_width <= 0 || monitor_height <= 0 || surface_width <= 0 || surface_height <= 0 {
        return Err("Source rectangle bounds are empty".to_owned());
    }
    Ok(SourceRectConstants {
        source_uv_origin: [
            (surface_bounds.left - primary_monitor_bounds.left) as f32 / monitor_width as f32,
            (surface_bounds.top - primary_monitor_bounds.top) as f32 / monitor_height as f32,
        ],
        source_uv_size: [
            surface_width as f32 / monitor_width as f32,
            surface_height as f32 / monitor_height as f32,
        ],
        pointer_velocity: [0.0, 0.0],
        _padding: [0.0, 0.0],
    })
}

pub(super) fn width(bounds: &RECT) -> Result<u32, String> {
    u32::try_from(bounds.right - bounds.left).map_err(|_| "Overlay width is invalid".to_owned())
}

pub(super) fn height(bounds: &RECT) -> Result<u32, String> {
    u32::try_from(bounds.bottom - bounds.top).map_err(|_| "Overlay height is invalid".to_owned())
}