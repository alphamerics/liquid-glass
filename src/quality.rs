//! The quality tier system: presets that trade a little visual headroom for
//! large reductions in GPU work, mirroring how iOS gates effects on device
//! capability and power state.
//!
//! A tier resolves into concrete engine caps:
//! - the backdrop refresh cap (how often the full-monitor blit + Kawase blur
//!   + broker publication may run, regardless of capture frame rate),
//! - the blur chain's internal resolution and level depth,
//! - the composite shader variant (lite = fewer noise octaves, single
//!   dispersion tap).
//!
//! `QualityMode::Auto` picks the tier from the GPU class (integrated vs
//! discrete) and the battery state, so an integrated-Graphics laptop lands
//! on the efficient path without any user configuration.

use std::time::Duration;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D11::ID3D11Device;
use windows::Win32::Graphics::Dxgi::{IDXGIDevice, IDXGIAdapter1};

/// The user-facing performance preset.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QualityMode {
    /// Hardware-detected: integrated GPUs (and battery) choose the
    /// efficient tier, discrete GPUs the balanced one.
    #[default]
    Auto,
    /// Full fidelity: uncapped backdrop rate, deep blur chain, full shader.
    High,
    /// Modest reductions, tuned for dGPUs and 1080p/1440p.
    Balanced,
    /// Integrated-graphics / battery tier: half-resolution blur chain,
    /// capped backdrop rate, lite composite shader.
    Efficiency,
}

impl QualityMode {
    /// Parses the material's `quality` string ("auto" | "high" |
    /// "balanced" | "efficiency"); unknown values fall back to auto.
    pub fn parse(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "high" => Self::High,
            "balanced" => Self::Balanced,
            "efficiency" => Self::Efficiency,
            _ => Self::Auto,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::High => "high",
            Self::Balanced => "balanced",
            Self::Efficiency => "efficiency",
        }
    }
}

/// The engine-level caps a resolved tier imposes.
///
/// The tiers reduce blur cost through the chain's internal resolution and
/// the backdrop rate rather than by truncating the Kawase level count:
/// truncating levels snaps the radius slider at the band boundaries (each
/// band's continuity assumes the extra level exists), while a resolution
/// reduction is invisible under a heavy blur and continuous by design.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QualityCaps {
    /// Multiplier on the chain's base (down0) resolution, applied with a
    /// radius ramp (fully engaged at blur radii >= 24 texels, off below
    /// 12) so the radius slider never jumps when the reduction engages.
    pub blur_internal_scale: f32,
    /// Maximum backdrop refresh rate (capture blit + blur + publication).
    pub backdrop_fps: u32,
    /// Backdrop refresh cap while a fullscreen app is foreground — glass
    /// sampling a game's frames at full rate is wasted GPU.
    pub fullscreen_backdrop_fps: u32,
    /// Use the lite composite shader variant.
    pub lite_shader: bool,
}

impl QualityCaps {
    pub const HIGH: Self = Self {
        blur_internal_scale: 1.0,
        backdrop_fps: 60,
        fullscreen_backdrop_fps: 30,
        lite_shader: false,
    };
    pub const BALANCED: Self = Self {
        blur_internal_scale: 0.75,
        backdrop_fps: 30,
        fullscreen_backdrop_fps: 15,
        lite_shader: false,
    };
    pub const EFFICIENCY: Self = Self {
        blur_internal_scale: 0.5,
        backdrop_fps: 24,
        fullscreen_backdrop_fps: 10,
        lite_shader: true,
    };

    /// The rate cap as a minimum interval (`None` = uncapped).
    pub fn backdrop_interval(&self, fullscreen: bool) -> Option<Duration> {
        let fps = if fullscreen {
            self.fullscreen_backdrop_fps
        } else {
            self.backdrop_fps
        };
        if fps == 0 {
            return None;
        }
        Some(Duration::from_millis((1000 / fps) as u64))
    }
}

/// The GPU class the engine's device runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuClass {
    Integrated,
    Discrete,
    Software,
}

/// Classifies the adapter the device was created on. Integrated GPUs report
/// no dedicated video memory (the Intel/AMD iGPU case); Intel parts are
/// always integrated; AMD parts with a small dedicated pool are APUs.
pub fn detect_gpu_class(device: &ID3D11Device) -> GpuClass {
    let Ok(dxgi_device) = device.cast::<IDXGIDevice>() else {
        return GpuClass::Integrated;
    };
    let Ok(adapter) = (unsafe { dxgi_device.GetAdapter() }) else {
        return GpuClass::Integrated;
    };
    let Ok(adapter1) = adapter.cast::<IDXGIAdapter1>() else {
        return GpuClass::Integrated;
    };
    describe_adapter(&adapter1)
}

fn describe_adapter(adapter: &IDXGIAdapter1) -> GpuClass {
    let Ok(desc) = (unsafe { adapter.GetDesc1() }) else {
        return GpuClass::Integrated;
    };
    const INTEL: u32 = 0x8086;
    const AMD: u32 = 0x1002;
    const AMD_ALT: u32 = 0x1022;
    const NVIDIA: u32 = 0x10DE;
    const MICROSOFT: u32 = 0x1414;
    match desc.VendorId {
        MICROSOFT => GpuClass::Software,
        INTEL => GpuClass::Integrated,
        NVIDIA => GpuClass::Discrete,
        AMD | AMD_ALT => {
            // Sub-1GB dedicated pools are APUs/iGPUs (the Llava-era
            // convention where the "dedicated" carve-out is small).
            if desc.DedicatedVideoMemory < 1024 * 1024 * 1024 {
                GpuClass::Integrated
            } else {
                GpuClass::Discrete
            }
        }
        // Unknown vendors: no dedicated memory means shared memory.
        _ if desc.DedicatedVideoMemory == 0 => GpuClass::Integrated,
        _ => GpuClass::Discrete,
    }
}

/// Whether the machine is running on battery power (`false` on failure —
/// desktops and query errors both resolve to "plugged in", the higher tier).
pub fn on_battery() -> bool {
    use windows::Win32::System::Power::{GetSystemPowerStatus, SYSTEM_POWER_STATUS};
    let mut status = SYSTEM_POWER_STATUS::default();
    if unsafe { GetSystemPowerStatus(&mut status) }.is_err() {
        return false;
    }
    // BatteryFlag 128 = no system battery (desktop), 255 = unknown;
    // ACLineStatus 1 = online, 255 = unknown (treated as plugged in).
    status.BatteryFlag != 128
        && status.BatteryFlag != 255
        && status.ACLineStatus != 1
        && status.ACLineStatus != 255
}

/// Whether a fullscreen (exclusive or F11) window is foreground — the state
/// where the backdrop rate drops to the tier's fullscreen cap.
pub fn fullscreen_foreground() -> bool {
    use windows::Win32::UI::Shell::{
        QUNS_BUSY, QUNS_PRESENTATION_MODE, QUNS_RUNNING_D3D_FULL_SCREEN,
        SHQueryUserNotificationState,
    };
    let Ok(state) = (unsafe { SHQueryUserNotificationState() }) else {
        return false;
    };
    // D3D exclusive fullscreen, presentation mode, and F11-style busy
    // fullscreen all cover "a fullscreen thing owns the screen".
    matches!(
        state,
        QUNS_RUNNING_D3D_FULL_SCREEN | QUNS_PRESENTATION_MODE | QUNS_BUSY
    )
}

/// Resolves the caps for the material's mode on the current hardware.
/// Battery power pulls Auto down a tier: a discrete GPU still has headroom,
/// but its laptop's battery does not.
pub fn resolve(mode: QualityMode, gpu: GpuClass) -> QualityCaps {
    match mode {
        QualityMode::High => QualityCaps::HIGH,
        QualityMode::Balanced => QualityCaps::BALANCED,
        QualityMode::Efficiency => QualityCaps::EFFICIENCY,
        QualityMode::Auto => match gpu {
            GpuClass::Discrete if !on_battery() => QualityCaps::BALANCED,
            _ => QualityCaps::EFFICIENCY,
        },
    }
}
