//! The broker service: one process that owns the WGC capture and blur
//! for every glass app in the session and publishes the shared backdrop
//! through an NT-handle/manifest protocol, so compositors get glass with
//! zero extra capture or blur work.
//!
//! Submodules:
//! - `protocol` — the manifest layout, slot states and naming, shared by
//!   the broker (publisher) and the library's clients;
//! - `host` — the broker-side engine (only this crate binary runs it);
//! - `client` — the consumer-side state machine the library's engine
//!   drives per tick (also what spawns the broker process itself).

pub(crate) mod client;
pub(crate) mod host;
pub(crate) mod protocol;

use std::path::PathBuf;

/// The material file's name inside its directory.
const MATERIAL_FILE_NAME: &str = "material.json";
/// The material schema version, written into the file. A file without it is
/// a partial (pre-`Material`) file and gets re-seeded from the legacy
/// glassbar settings on the next start.
const MATERIAL_VERSION: u32 = 1;
/// The legacy home of the material values (glassbar's settings file). Only
/// read once, to seed the broker-owned material file on first run.
const LEGACY_GLASSBAR_SETTINGS: &str = "org.newmoonworks.glassbar";

/// The material file's canonical location: `%LOCALAPPDATA%\LiquidGlass\\
/// material.json`. The broker owns this file — every glass app reads (or,
/// with a settings UI, writes) the shared material through it instead of
/// keeping a private copy of the material's config.
pub fn material_path() -> Option<PathBuf> {
    let mut path = PathBuf::from(std::env::var_os("LOCALAPPDATA")?);
    path.push("LiquidGlass");
    path.push(MATERIAL_FILE_NAME);
    Some(path)
}

/// Reads the shared material, falling back to the library defaults for a
/// missing or unparsable file (a partial file parses with defaults for its
/// missing fields, courtesy of `Material`'s serde default).
pub fn read_material() -> crate::params::Material {
    let fallback = crate::params::Material::default();
    let Some(path) = material_path() else {
        return fallback;
    };
    let Ok(raw) = std::fs::read_to_string(path) else {
        return fallback;
    };
    serde_json::from_str(&raw).unwrap_or(fallback)
}

/// Writes the shared material (plus a schema version marker) to
/// `material.json`, creating the directory as needed.
pub fn write_material(material: &crate::params::Material) -> Result<(), String> {
    let Some(path) = material_path() else {
        return Err("Could not resolve the material directory (LOCALAPPDATA missing)".to_owned());
    };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("Could not create the material directory: {error}"))?;
    }
    let mut value = serde_json::to_value(material)
        .map_err(|error| format!("Could not serialize the material: {error}"))?;
    if let Some(object) = value.as_object_mut() {
        object.insert(
            "materialVersion".to_owned(),
            serde_json::Value::from(MATERIAL_VERSION),
        );
    }
    let raw = serde_json::to_string_pretty(&value)
        .map_err(|error| format!("Could not serialize the material: {error}"))?;
    std::fs::write(&path, raw)
        .map_err(|error| format!("Could not write {}: {error}", path.display()))
}

/// One-time material seeding: when `material.json` does not exist yet (or
/// predates the full `Material` schema), it is written from the legacy
/// Glassbar settings file if that is readable — so an upgrade keeps every
/// user-tuned material value — otherwise from the library defaults. From
/// then on the material file is the only home of the shared material.
/// Returns `true` when a file was written.
pub fn seed_material_if_missing() -> Result<bool, String> {
    let Some(path) = material_path() else {
        return Err("Could not resolve the material directory (LOCALAPPDATA missing)".to_owned());
    };
    let fresh = std::fs::read_to_string(&path)
        .ok()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
        .and_then(|value| {
            value
                .get("materialVersion")
                .and_then(|v| v.as_u64())
                .map(|v| v >= u64::from(MATERIAL_VERSION))
        })
        .unwrap_or(false);
    if fresh {
        return Ok(false);
    }
    let legacy = std::env::var_os("APPDATA").map(|root| {
        let mut path = PathBuf::from(root);
        path.push(LEGACY_GLASSBAR_SETTINGS);
        path.push("settings.json");
        path
    });
    let legacy_value = legacy
        .as_deref()
        .and_then(|path| std::fs::read_to_string(path).ok())
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok());
    let material = match legacy_value {
        Some(value) => crate::params::Material::from_legacy_glassbar(&value),
        None => crate::params::Material::default(),
    };
    write_material(&material)?;
    Ok(true)
}

/// The broker process entry point (`glass-broker.exe`'s main).
pub fn run() -> i32 {
    host::run()
}

/// The adapter LUID of a device, as the manifest centers it (both the
/// broker's stamp and every client's attach check use this).
pub(crate) fn device_adapter_luid(
    device: &windows::Win32::Graphics::Direct3D11::ID3D11Device,
) -> Result<u64, String> {
    use windows::core::Interface;
    use windows::Win32::Graphics::Dxgi::IDXGIDevice;
    let dxgi: IDXGIDevice = device
        .cast()
        .map_err(|error| format!("Could not obtain the DXGI device: {error}"))?;
    let adapter = unsafe {
        dxgi.GetAdapter()
    }
    .map_err(|error| format!("Could not obtain the glass adapter: {error}"))?;
    let desc = unsafe {
        adapter.GetDesc()
    }
    .map_err(|error| format!("Could not describe the glass adapter: {error}"))?;
    Ok(protocol::adapter_luid(desc.AdapterLuid))
}
