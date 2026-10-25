//! glass-demo — a minimal second liquid-glass consumer.
//!
//! Run it alongside Glassbar (or any other liquid-glass app): both claim
//! consumer slots in the same broker manifest and share one capture + one
//! blur pipeline. A floating rounded glass panel appears; drag it by any
//! pixel, press ESC to quit.

use windows::core::w;
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
use windows::Win32::Graphics::Gdi::{BeginPaint, EndPaint, PAINTSTRUCT};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::Input::KeyboardAndMouse::VK_ESCAPE;
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DefWindowProcW, DispatchMessageW, GetMessageW, PostQuitMessage,
    RegisterClassW, SetLayeredWindowAttributes, ShowWindow, TranslateMessage, CS_HREDRAW,
    CS_VREDRAW, CW_USEDEFAULT, HTCAPTION, LWA_ALPHA, MSG, SW_SHOW, WM_DESTROY, WM_KEYDOWN,
    WM_NCHITTEST, WM_PAINT, WNDCLASSW, WS_EX_LAYERED, WS_POPUP, WS_VISIBLE,
};
use windows::Win32::Foundation::COLORREF;

use liquid_glass::{GlassConfig, GlassSurface, SurfaceAnchor};

fn main() {
    unsafe {
        let hinstance = GetModuleHandleW(None).expect("module handle");
        let class_name = w!("GlassDemo");

        let class = WNDCLASSW {
            lpfnWndProc: Some(wnd_proc),
            hInstance: hinstance.into(),
            lpszClassName: class_name,
            style: CS_HREDRAW | CS_VREDRAW,
            ..Default::default()
        };
        RegisterClassW(&class);

        // The host window is fully transparent (layered, alpha 0): the only
        // thing visible is the glass surface the library stacks beneath it.
        // The surface is hit-test transparent, so this window receives every
        // click — its whole client area answers HTCAPTION, making the panel
        // draggable.
        let host = CreateWindowExW(
            WS_EX_LAYERED,
            class_name,
            w!("Glass Demo"),
            WS_POPUP | WS_VISIBLE,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            460,
            300,
            None,
            None,
            hinstance,
            None,
        )
        .expect("host window");
        SetLayeredWindowAttributes(host, COLORREF(0), 0, LWA_ALPHA)
            .expect("layered transparency");
        let _ = ShowWindow(host, SW_SHOW);

        let surface = GlassSurface::create(
            host.0 as isize,
            GlassConfig {
                corner_radius_px: 28.0,
                surface_anchor: SurfaceAnchor::HostBounds,
                ..Default::default()
            },
        )
        .expect("glass surface");
        std::mem::forget(surface);

        let mut message = MSG::default();
        while GetMessageW(&mut message, None, 0, 0).as_bool() {
            let _ = TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }
}

unsafe extern "system" fn wnd_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    match message {
        // Drag by any pixel of the panel.
        WM_NCHITTEST => LRESULT(HTCAPTION as isize),
        WM_KEYDOWN => {
            if wparam.0 == VK_ESCAPE.0 as usize {
                PostQuitMessage(0);
            }
            LRESULT(0)
        }
        WM_PAINT => {
            let mut paint = PAINTSTRUCT::default();
            BeginPaint(hwnd, &mut paint);
            let _ = EndPaint(hwnd, &paint);
            LRESULT(0)
        }
        WM_DESTROY => {
            PostQuitMessage(0);
            LRESULT(0)
        }
        _ => DefWindowProcW(hwnd, message, wparam, lparam),
    }
}
