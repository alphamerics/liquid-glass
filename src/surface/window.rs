use windows::Win32::{
    Foundation::{HWND, RECT, LPARAM, LRESULT, WPARAM},
    UI::WindowsAndMessaging::{
        CreateWindowExW, DefWindowProcW, GetWindowRect, RegisterClassW, SetLayeredWindowAttributes,
        SetWindowLongW, SetWindowPos, ShowWindow, GWL_EXSTYLE, HTTRANSPARENT, HWND_TOPMOST,
        LWA_ALPHA, SWP_NOACTIVATE, SWP_NOMOVE, SWP_NOSIZE, SWP_NOZORDER, SWP_SHOWWINDOW, SW_HIDE,
        WINDOW_EX_STYLE, WINDOW_STYLE, WM_NCHITTEST, WNDCLASSW, WS_EX_LAYERED, WS_EX_NOACTIVATE,
        WS_EX_TOOLWINDOW, WS_EX_TRANSPARENT, WS_POPUP,
    },
};

use crate::params::SurfaceAnchor;

use super::geometry::{anchor_rect, overlay_bounds};

const RENDER_SURFACE_CLASS: windows::core::PCWSTR = windows::core::w!("LiquidGlassRenderer");

unsafe extern "system" fn render_surface_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // Surfaces never take mouse input: the host overlay above them handles
    // interaction, and collapsed/recycled surfaces must not block clicks.
    if message == WM_NCHITTEST {
        return LRESULT(HTTRANSPARENT as isize);
    }
    DefWindowProcW(hwnd, message, wparam, lparam)
}

/// Creates the transparent, click-through surface window covering `bounds`.
/// The layered attribute is what makes WS_EX_TRANSPARENT actually pass mouse
/// hits through cross-thread (the style alone still routes clicks to this
/// window); the DComp target binds fine to layered windows.
pub(super) fn create_surface_window(bounds: RECT) -> Result<HWND, String> {
    unsafe {
        let _ = RegisterClassW(&WNDCLASSW {
            lpfnWndProc: Some(render_surface_proc),
            lpszClassName: RENDER_SURFACE_CLASS,
            ..Default::default()
        });
        let created = CreateWindowExW(
            // Transparent to mouse hits: every surface's interaction goes
            // through the host overlay above it, and collapsed/recycled
            // surfaces must never swallow clicks over their old rects.
            WINDOW_EX_STYLE(WS_EX_NOACTIVATE.0 | WS_EX_TOOLWINDOW.0 | WS_EX_TRANSPARENT.0),
            RENDER_SURFACE_CLASS,
            windows::core::w!("Liquid Glass GPU Surface"),
            WINDOW_STYLE(WS_POPUP.0),
            bounds.left,
            bounds.top,
            bounds.right - bounds.left,
            bounds.bottom - bounds.top,
            None,
            None,
            None,
            None,
        )
        .map_err(|error| format!("Could not create the GPU render surface: {error}"))?;
        let _ = SetWindowLongW(
            created,
            GWL_EXSTYLE,
            (WS_EX_NOACTIVATE.0 | WS_EX_TOOLWINDOW.0 | WS_EX_TRANSPARENT.0 | WS_EX_LAYERED.0)
                as i32,
        );
        let _ = SetLayeredWindowAttributes(created, None, 255, LWA_ALPHA);
        Ok(created)
    }
}

/// Hides the surface window outright. Used when parking a surface: even
/// though the collapsed reveal renders fully transparent, the window can
/// briefly (or permanently, if capture stalls) keep showing its last
/// presented frame — hiding the window guarantees nothing lingers.
pub(super) fn hide_surface_window(hwnd: HWND) -> Result<(), String> {
    unsafe {
        ShowWindow(hwnd, SW_HIDE)
            .ok()
            .map_err(|error| format!("Could not hide the glass surface: {error}"))
    }
}

/// Aligns the native surface with its host window while retaining its lower
/// Z order. Call whenever the host moves or resizes.
pub(super) unsafe fn stack_beneath_host(
    host: isize,
    hwnd: HWND,
    anchor: &SurfaceAnchor,
    style_host_window: bool,
) -> Result<(), String> {
    let overlay = HWND(host as *mut _);
    let bounds = anchor_rect(overlay_bounds(overlay)?, anchor);
    SetWindowPos(
        hwnd,
        HWND_TOPMOST,
        bounds.left,
        bounds.top,
        bounds.right - bounds.left,
        bounds.bottom - bounds.top,
        SWP_NOACTIVATE | SWP_SHOWWINDOW,
    )
    .map_err(|error| format!("Could not position the GPU surface: {error}"))?;
    if style_host_window {
        crate::borderless::configure_borderless_host(host)?;
    }
    SetWindowPos(
        overlay,
        HWND_TOPMOST,
        0,
        0,
        0,
        0,
        SWP_NOACTIVATE | SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW,
    )
    .map_err(|error| format!("Could not keep the host above the GPU surface: {error}"))
}

/// Moves one glass surface window vertically by `dy` physical pixels (the
/// window half of [`crate::GlassSurface::translate`]).
///
/// # Deadlock hazard
/// `SetWindowPos` targeting a window owned by another thread blocks until
/// that thread pumps messages — and that thread (the host's command
/// handler) may itself be waiting on a lock the caller holds. Collect the
/// targets, DROP every lock, then move the windows.
pub fn translate_surface_window(hwnd_addr: isize, dy: i32) -> Result<(), String> {
    if dy == 0 {
        return Ok(());
    }
    let hwnd = HWND(hwnd_addr as *mut _);
    let mut rect = RECT::default();
    unsafe {
        GetWindowRect(hwnd, &mut rect)
            .map_err(|error| format!("Could not read the surface rect: {error}"))?;
        SetWindowPos(
            hwnd,
            HWND_TOPMOST,
            rect.left,
            rect.top + dy,
            0,
            0,
            SWP_NOACTIVATE | SWP_NOSIZE | SWP_NOZORDER,
        )
        .map_err(|error| format!("Could not slide the GPU surface: {error}"))?;
    }
    Ok(())
}