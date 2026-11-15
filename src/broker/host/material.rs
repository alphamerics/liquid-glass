//! The global material watcher: reads the shared material from the
//! broker-owned material file (`%LOCALAPPDATA%\LiquidGlass\material.json`),
//! re-reading on mtime change. Missing or malformed files fall back
//! silently to the library defaults.

use std::path::PathBuf;
use std::time::SystemTime;

use crate::params::GlassShaderParams;
pub(super) struct MaterialWatcher {
    path: Option<PathBuf>,
    last_mtime: Option<SystemTime>,
    /// A monotonically increasing revision; any change bumps it, and the
    /// publication's backdrop keys off it.
    revision: u64,
    /// The last material's quality mode (the publish loop resolves the
    /// tier from it).
    pub(super) quality: crate::quality::QualityMode,
    /// The last battery state (Auto tiers re-resolve on transitions).
    pub(super) battery_seen: Option<bool>,
}

impl MaterialWatcher {
    pub(super) fn new() -> Self {
        let path = crate::broker::material_path();
        if let Err(error) = crate::broker::seed_material_if_missing() {
            eprintln!("glass broker material seed failed: {error}");
        }
        let quality = crate::broker::read_material().quality;
        Self {
            path,
            last_mtime: None,
            revision: 1,
            quality,
            battery_seen: None,
        }
    }

    pub(super) fn revision(&self) -> u64 {
        self.revision
    }

    /// Polls the file (bounded by the poll cadence the caller chooses to
    /// respect); `true` when the material changed and params were updated.
    pub(super) fn poll(&mut self, params: &mut GlassShaderParams) -> bool {
        let Some(path) = self.path.as_ref() else {
            return false;
        };
        let mtime = std::fs::metadata(path)
            .and_then(|metadata| metadata.modified())
            .ok();
        if mtime == self.last_mtime {
            return false;
        }
        self.last_mtime = mtime;
        let material = crate::broker::read_material();
        self.quality = material.quality;
        let mut updated = *params;
        // The broker's capture and publication run at full monitor
        // resolution, so no texel scaling applies here (texel scale 1.0).
        material.apply_onto(&mut updated, 1.0);
        if updated != *params {
            *params = updated;
            self.revision += 1;
            return true;
        }
        false
    }
}
