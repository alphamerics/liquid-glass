# liquid-glass

A reusable, GPU-composited **liquid glass** backdrop for Windows overlay
windows — designed to drop into [Tauri 2](https://tauri.app) apps with one
call, and to work with any Win32 host framework (tao, winit, raw Win32).

> **Windows-only.** The crate is built directly on Win32, D3D11 and Windows
> Graphics Capture and does not compile on any other platform.

A glass app renders its UI into a transparent overlay window; this crate
stacks a native D3D11 surface window directly beneath it, captures the
primary monitor with Windows Graphics Capture, and composites the captured
pixels through a liquid-glass shader (refraction, dual-Kawase blur, chromatic
dispersion, tint, a capsule SDF and reveal animation).

## How it works

- **In-process skin** — `GlassSurface` owns a DirectComposition D3D11 child
  window positioned under your overlay. The pixels come from the shared
  backdrop below; your DOM floats on top of real glass.
- **The broker** — one `glass-broker.exe` per user session owns the
  (expensive) Windows Graphics Capture of the monitor and the blur pipeline,
  publishing the shared backdrop to every glass app through NT-handles. New
  glass apps add **zero** extra capture or blur work; two glass apps running
  side by side share a single session. Clients spawn the broker on demand,
  looking for it next to their own executable.
- **The shared material** — material values (refraction, tint, blur, ...)
  live in one broker-owned file
  (`%LOCALAPPDATA%\LiquidGlass\material.json`) that every glass app reads
  (or writes, from a settings UI), so all apps in the session render the
  same material.

## Usage (Tauri)

```rust
use liquid_glass::{GlassConfig, GlassSurface};

let window = tauri_webview_window();
let hwnd = window.hwnd()?.0 as isize;

// Recommended for tao/tauri hosts: suppress classic frame repaints on
// focus changes (tao always keeps WS_CAPTION in the style).
liquid_glass::install_activation_subclass(hwnd)?;

let surface = GlassSurface::create(hwnd, GlassConfig::default())?;

// Live-tune the material:
surface.set_shader_params(liquid_glass::GlassShaderParams {
    refraction_strength: 14.0,
    glass_tint: [0.9, 0.75, 1.0],
    ..Default::default()
})?;

// On window move/resize events: surface.sync_bounds(hwnd)?;
// Dropping `surface` stops the capture and destroys the surface window.
```

The crate is intentionally Tauri-free: it only needs an `isize` HWND.

### App setup requirements

An app rendering liquid glass must:

1. Use a **transparent, decorationless overlay window** (`transparent: true`,
   `decorations: false`, `backgroundColor: "#00000000"`, `shadow: false` in
   Tauri).
2. Force `--force-device-scale-factor=1` in the window's
   `additionalBrowserArgs` (rects are pushed in physical pixels; a
   DPI-scaled DOM would misplace every surface).
3. Exclude the web UI from the glass's own capture with
   `SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)`, and set
   `WS_EX_LAYERED` on the window.
4. Ship `glass-broker.exe` beside the executable (see below).

## Glass surface tuning

- **`GlassConfig`** — corner radius and the surface's anchoring geometry
  (**`SurfaceAnchor`**: bottom strip, rect, or full host).
- **`GlassShaderParams`** — ~30 live shader parameters (refraction, blur,
  tint, chromatic dispersion, turbulence, outline, reveal wobble, pill
  scaling, ...), changeable every frame.
- **`RevealAnimation`** — the reveal/scale animation stack.
- **`Material`** — the shared, serializable material preset (the one that
  lives in `material.json`).

## The broker

The engine lives in the crate (`liquid_glass::broker`); the executable is a
trampoline that calls `liquid_glass::broker::run()`:

- `glass-broker/` — the standalone bin crate (this workspace). Ship its exe
  beside your app: `cargo build -p glass-broker`.
- Apps embedding the crate (like Glassbar) usually compile their **own**
  trampoline binary instead, so the broker shares the app's exact protocol
  build.

The broker is a session-wide singleton: the first glass app spawns it, later
apps attach, and it exits when the session ends.

## Workspace members

- `glass-demo/` — a minimal second liquid-glass consumer: a floating,
  draggable glass panel that shares the session's broker with other glass
  apps. Run it alongside Glassbar to see the sharing in action:

  ```powershell
  cargo run -p glass-demo
  ```

## Requirements

- Windows 10 (1903+) or Windows 11
- Rust stable with the MSVC toolchain (Windows only)
- Windows Graphics Capture (built into Windows 10+)

## Related repos

- [`glass-ui`](https://github.com/alphamerics/glass-ui) — the reusable UI
  half: React components that drive native glass surfaces behind the DOM,
  plus the native `MenuGlassManager`.
- [`liquidbar` (Glassbar)](https://github.com/alphamerics/liquidbar) — a
  glass taskbar replacement built with this crate.

## License

MIT — see [LICENSE](LICENSE).
