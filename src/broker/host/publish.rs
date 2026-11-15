//! The publication machinery: the shared keyed-mutex backdrop pair, the
//! material's params, the blur chain and the refresh/publish cycle the
//! broker's main loop drives every tick.

use std::sync::atomic::Ordering;

use windows::core::Interface;
use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D11::{
    ID3D11Device, ID3D11DeviceContext, ID3D11RenderTargetView, ID3D11Texture2D,
    D3D11_BIND_RENDER_TARGET,
    D3D11_BIND_SHADER_RESOURCE, D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX,
    D3D11_RESOURCE_MISC_SHARED_NTHANDLE, D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT,
};
use windows::Win32::Graphics::Dxgi::{
    IDXGIKeyedMutex, IDXGIResource1, DXGI_SHARED_RESOURCE_READ, DXGI_SHARED_RESOURCE_WRITE,
};

use super::frame_host::BrokerHost;
use crate::broker::protocol::{Manifest, CONSUMER_SLOTS, SLOT_REQUESTED, SLOT_SUPPLIED};
use crate::params::GlassShaderParams;
use crate::quality::QualityCaps;
use crate::shared::{blur_plan_with_caps, Backdrop, BlurEngine};

/// Consecutive failed publication cycles before the pair is rebuilt — a
/// consumer that died while holding a keyed mutex would otherwise poison
/// it permanently (a recreate invalidates every mapping and forces a
/// clean re-handshake).
const PUBLISH_RECOVERY_TICKS: u32 = 300;
/// Keyed-mutex acquire attempts before a publication skips a frame
/// (another party's copy lasts microseconds; skipping costs nothing).
const ACQUIRE_RETRIES: u32 = 8;
/// Per-attempt keyed-mutex timeout.
const ACQUIRE_TIMEOUT_MS: u32 = 25;
/// The constant keyed-mutex key every acquire/release pair uses — the
/// plain-mutex discipline: with one writer and serialized copyers the key
/// never moves, so no inter-key choreography exists to get wrong.
const MUTEX_KEY: u64 = 0;
/// One shared backdrop buffer: the texture on its keyed mutex, the
/// render-target view the publication path writes into, and the resource
/// carrying the NT handle the broker duplicates into consumers.
pub(crate) struct SharedCell {
    texture: ID3D11Texture2D,
    target: ID3D11RenderTargetView,
    resource: IDXGIResource1,
    pub(super) handle: HANDLE,
}

impl SharedCell {
    fn keyed(&self) -> IDXGIKeyedMutex {
        self.resource
            .cast::<IDXGIKeyedMutex>()
            .expect("keyed-mutex textures provide IDXGIKeyedMutex")
    }
}

/// The two-buffer backdrop pair at the monitor's full publication size.
pub(crate) struct SharedPair {
    pub(super) cells: [SharedCell; 2],
    width: u32,
    height: u32,
}

/// The publication machinery: the material's params (read from the shared
/// settings file), the blur chain, and the shared pair's lifecycle.
pub(super) struct PublishEngine {
    blur: BlurEngine,
    pub(super) params: GlassShaderParams,
    /// The material revision the current params were read from (any
    /// change forces a re-blur; the manifest's versions stay monotonic).
    backdrop: Option<Backdrop>,
    backdrop_version: u64,
    backdrop_material: u64,
    pub(super) pair: Option<SharedPair>,
    /// The host frame version already published; publications are skipped
    /// while the backdrop has not moved.
    published_version: u64,
    publish_failures: u32,
}

impl PublishEngine {
    pub(super) fn new(device: &ID3D11Device) -> Self {
        Self {
            blur: BlurEngine::new(device),
            params: GlassShaderParams::default(),
            backdrop: None,
            backdrop_version: 0,
            backdrop_material: 0,
            pair: None,
            published_version: 0,
            publish_failures: 0,
        }
    }

    /// Drops the cached backdrop so the next refresh rebuilds it (a tier
    /// change alters the chain caps, which the version/revision
    /// fingerprint alone cannot see).
    pub(super) fn invalidate_backdrop(&mut self) {
        self.backdrop = None;
    }

    /// Refreshes the shared backdrop when a fresh capture frame or a
    /// material revision landed; static desktops re-use the last one.
    pub(super) fn refresh(
        &mut self,
        context: &ID3D11DeviceContext,
        host: &BrokerHost,
        host_version: u64,
        material_revision: u64,
        caps: &QualityCaps,
    ) {
        if self.backdrop_version == host_version && self.backdrop_material == material_revision {
            return;
        }
        let Some((_texture, view)) = host.stable.as_ref() else {
            return;
        };
        let size = [
            host.frame_size.Width.max(1) as u32,
            host.frame_size.Height.max(1) as u32,
        ];
        // Full-frame blur: the publication pass stretches the backdrop into
        // the monitor-sized shared pair, so consumers' geometry must not
        // depend on which region the chain covered.
        match self
            .blur
            .render(context, view, size, None, &self.params, caps)
        {
            Ok(backdrop) => {
                self.backdrop = Some(backdrop);
                self.backdrop_version = host_version;
                self.backdrop_material = material_revision;
            }
            Err(error) => eprintln!("glass broker blur failed: {error}"),
        }
    }

    /// Creates the shared pair at the monitor's size, stamps the manifest
    /// (sizes, bumped epoch) and invalidates consumer handles.
    fn recreate(
        &mut self,
        device: &ID3D11Device,
        manifest: &Manifest,
        width: u32,
        height: u32,
    ) -> Result<(), String> {
        let make_cell = || -> Result<SharedCell, String> {
            let mut texture = None;
            unsafe {
                device.CreateTexture2D(
                    &D3D11_TEXTURE2D_DESC {
                        Width: width,
                        Height: height,
                        MipLevels: 1,
                        ArraySize: 1,
                        BindFlags: (D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE).0 as u32,
                        Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM,
                        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                            Count: 1,
                            Quality: 0,
                        },
                        Usage: D3D11_USAGE_DEFAULT,
                        MiscFlags: (D3D11_RESOURCE_MISC_SHARED_NTHANDLE
                            | D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX)
                            .0 as u32,
                        CPUAccessFlags: 0,
                    },
                    None,
                    Some(&mut texture),
                )
            }
            .map_err(|error| {
                format!("Could not create the shared glass backdrop texture: {error}")
            })?;
            let texture = texture.expect("shared glass texture missing");
            let mut target = None;
            unsafe { device.CreateRenderTargetView(&texture, None, Some(&mut target)) }.map_err(
                |error| format!("Could not create the shared glass target view: {error}"),
            )?;
            let target = target.expect("shared glass target view missing");
            let resource = texture
                .cast::<IDXGIResource1>()
                .map_err(|error| format!("Could not obtain the shared glass resource: {error}"))?;
            let handle = unsafe {
                resource
                    .CreateSharedHandle(
                        None,
                        DXGI_SHARED_RESOURCE_READ.0 | DXGI_SHARED_RESOURCE_WRITE.0,
                        windows::core::w!(""),
                    )
                    .map_err(|error| {
                        format!("Could not create the shared glass NT handle: {error}")
                    })?
            };
            Ok(SharedCell {
                texture,
                target,
                resource,
                handle,
            })
        };
        let first = make_cell()?;
        let second = make_cell()?;
        let epoch = manifest
            .texture_epoch
            .load(Ordering::Acquire)
            .wrapping_add(1)
            .max(1);
        self.pair = Some(SharedPair {
            cells: [first, second],
            width,
            height,
        });
        manifest.frame_width.store(width, Ordering::Release);
        manifest.frame_height.store(height, Ordering::Release);
        manifest.texture_epoch.store(epoch, Ordering::Release);
        // The pair's texture handles changed: every supplied slot's map is
        // invalid — those slots round-trip through the request handshake.
        for index in 0..CONSUMER_SLOTS {
            let slot = manifest.slot(index);
            if slot.state.load(Ordering::Acquire) == SLOT_SUPPLIED {
                slot.state.store(SLOT_REQUESTED, Ordering::Release);
            }
        }
        self.published_version = 0;
        Ok(())
    }

    /// Ensures a pair at the current frame size exists (recreated on a
    /// monitor change).
    fn ensure_pair(
        &mut self,
        device: &ID3D11Device,
        manifest: &Manifest,
        host: &BrokerHost,
    ) -> Result<(), String> {
        let width = host.frame_size.Width.max(1) as u32;
        let height = host.frame_size.Height.max(1) as u32;
        match &self.pair {
            Some(pair) if pair.width == width && pair.height == height => Ok(()),
            _ => self.recreate(device, manifest, width, height),
        }
    }

    /// Publishes the latest backdrop into the pair's back buffer and
    /// flips the manifest. A contended mutex skips the tick; an `Err` is
    /// the caller's q to log (repeating failures trigger the pair's
    /// recreation through the failure counter).
    pub(super) fn publish_inner(
        &mut self,
        device: &ID3D11Device,
        context: &ID3D11DeviceContext,
        host: &BrokerHost,
        manifest: &Manifest,
        version: u64,
        caps: &QualityCaps,
    ) -> Result<(), String> {
        // No backdrop yet, or already published: nothing to do.
        let backdrop_view = match self.backdrop.as_ref() {
            Some(backdrop) => backdrop.view.clone(),
            None => return Ok(()),
        };
        if self.published_version >= version {
            return Ok(());
        }
        // The sharp-path source is resolved BEFORE the pair's work (and
        // crucially before the keyed-mutex acquire): an early return after
        // the acquire would have to release the mutex first.
        let sharp = blur_plan_with_caps(&self.params, caps).sharp;
        let sharp_source = if sharp {
            match host.stable.as_ref() {
                Some((stable, _)) => Some(stable.clone()),
                None => return Ok(()),
            }
        } else {
            None
        };
        self.ensure_pair(device, manifest, host)?;
        let (pair_width, pair_height) = {
            let pair = self.pair.as_ref().expect("the pair exists at publish");
            (pair.width, pair.height)
        };
        let back = ((manifest.front.load(Ordering::Acquire) as usize) ^ 1) & 1;
        let (keyed, texture, target) = {
            let pair = self.pair.as_ref().expect("the pair exists at publish");
            let cell = &pair.cells[back];
            (cell.keyed(), cell.texture.clone(), cell.target.clone())
        };
        // The acquire: a brief retry window handles the momentary race
        // with a consumer still copying the previous frame out of this
        // same buffer (the pair's two buffers alternate, so a competing
        // copy is rare and short).
        let mut acquired = false;
        for _ in 0..ACQUIRE_RETRIES {
            if unsafe { keyed.AcquireSync(MUTEX_KEY, ACQUIRE_TIMEOUT_MS) }.is_ok() {
                acquired = true;
                break;
            }
        }
        if !acquired {
            self.publish_failures += 1;
            if self.publish_failures >= PUBLISH_RECOVERY_TICKS {
                self.publish_failures = 0;
                let (width, height) = (
                    host.frame_size.Width.max(1) as u32,
                    host.frame_size.Height.max(1) as u32,
                );
                self.recreate(device, manifest, width, height)?;
            }
            return Ok(());
        }
        let result = if sharp {
            // The sharp backdrop IS the capture, already at the publication
            // size: a direct GPU copy suffices. The stable texture and the
            // pair share the format; the copy needs no bindings.
            let stable = sharp_source.expect("the sharp publication source exists");
            unsafe { context.CopyResource(&texture, &stable) };
            Ok(())
        } else {
            // A blur-reduced backdrop: bilinear-stretch it into the
            // monitor-sized shared buffer.
            self.blur
                .render_publication(context, &backdrop_view, &target, pair_width, pair_height)
        };
        // The release happens on EVERY path past the acquire — an
        // unreleased keyed mutex would lock every future publication out
        // of this buffer until the recovery recreate.
        unsafe {
            keyed.ReleaseSync(MUTEX_KEY).ok();
        }
        match result {
            Ok(()) => {
                // The manifest's publisher-side order: front first (the
                // consumers' snapshot brackets read it before the version
                // that unlocks it), version LAST with a release store.
                manifest.front.store(back as u32, Ordering::Release);
                manifest.frame_version.store(version, Ordering::Release);
                self.published_version = version;
                self.publish_failures = 0;
                Ok(())
            }
            Err(error) => Err(error),
        }
    }
}
