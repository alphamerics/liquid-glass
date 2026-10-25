//! Borderless styling for host overlay windows, tuned for Tauri/tao windows.
//!
//! tao keeps WS_CAPTION in the overlay's style even though its WM_NCCALCSIZE
//! handler removes the frame, so hosts need three guards: never rewrite
//! GWL_STYLE (that fights tao and can hide the window), strip the frame
//! extended styles, and suppress the WM_NCACTIVATE non-client repaint that
//! would otherwise paint a classic frame on every focus change.

use std::ffi::c_void;
use windows::Win32::{
    Foundation::{HWND, LPARAM, LRESULT, WPARAM},
    Graphics::Dwm::{
        DwmSetWindowAttribute, DWMWA_WINDOW_CORNER_PREFERENCE, DWMWCP_DONOTROUND,
        DWM_WINDOW_CORNER_PREFERENCE,
    },
    UI::{
        Shell::{DefSubclassProc, SetWindowSubclass, SUBCLASSPROC},
        WindowsAndMessaging::{
            GetWindowLongW, SetWindowLongW, SetWindowPos, GWL_EXSTYLE, HWND_TOPMOST,
            SWP_FRAMECHANGED, SWP_NOACTIVATE, SWP_NOMOVE, SWP_NOSIZE, SWP_NOZORDER, WM_NCACTIVATE,
            WS_EX_CLIENTEDGE, WS_EX_DLGMODALFRAME, WS_EX_STATICEDGE, WS_EX_WINDOWEDGE,
        },
    },
};

const HOST_SUBCLASS_ID: usize = 1;

unsafe extern "system" fn host_subclass_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
    _subclass_id: usize,
    _ref_data: usize,
) -> LRESULT {
    if message == WM_NCACTIVATE {
        return DefSubclassProc(hwnd, WM_NCACTIVATE, wparam, LPARAM(-1));
    }
    DefSubclassProc(hwnd, message, wparam, lparam)
}

/// Installs a subclass on the host window that forwards WM_NCACTIVATE with
/// lParam(-1), suppressing the classic frame repaint on activation changes.
pub fn install_activation_subclass(hwnd: isize) -> Result<(), String> {
    let subclass: SUBCLASSPROC = Some(host_subclass_proc);
    unsafe {
        SetWindowSubclass(HWND(hwnd as *mut _), subclass, HOST_SUBCLASS_ID, 0)
            .ok()
            .map_err(|error| format!("Could not subclass the host window: {error}"))
    }
}

/// Strips the frame extended styles and disables rounded corners on the host
/// window. Must be called on the thread that owns the window.
pub fn configure_borderless_host(hwnd: isize) -> Result<(), String> {
    let hwnd = HWND(hwnd as *mut _);
    unsafe {
        let current_ex_style = GetWindowLongW(hwnd, GWL_EXSTYLE) as u32;
        let frame_ex_style =
            WS_EX_DLGMODALFRAME.0 | WS_EX_WINDOWEDGE.0 | WS_EX_CLIENTEDGE.0 | WS_EX_STATICEDGE.0;
        let _ = SetWindowLongW(
            hwnd,
            GWL_EXSTYLE,
            (current_ex_style & !frame_ex_style) as i32,
        );

        let corner_preference = DWMWCP_DONOTROUND;
        DwmSetWindowAttribute(
            hwnd,
            DWMWA_WINDOW_CORNER_PREFERENCE,
            (&corner_preference as *const DWM_WINDOW_CORNER_PREFERENCE).cast::<c_void>(),
            size_of_val(&corner_preference) as u32,
        )
        .map_err(|error| format!("Could not disable Windows corner rendering: {error}"))?;

        SetWindowPos(
            hwnd,
            HWND_TOPMOST,
            0,
            0,
            0,
            0,
            SWP_FRAMECHANGED | SWP_NOACTIVATE | SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER,
        )
        .map_err(|error| format!("Could not refresh the host frame: {error}"))
    }
}
