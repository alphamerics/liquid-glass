//! The broker process's engine: it owns the one WGC capture session and
//! the one blur chain for every glass app in the session and publishes
//! every fresh backdrop into the shared double-buffered keyed-mutex
//! texture pair (see `protocol`).
//!
//! Publication per fresh frame:
//! 1. the capture copy already landed in the host's stable texture
//!    (skipped entirely while every consumer is parked);
//! 2. the backdrop is composited exactly like the in-process engine
//!    (dual-Kawase blur, or the raw capture when blur is off), from the
//!    global material's params read from the broker-owned material file;
//! 3. the pair's back buffer is acquired (keyed mutex, constant key), the
//!    backdrop is written in — always at the monitor's full publication
//!    resolution, so consumers' shader geometry never depends on the
//!    material — and released;
//! 4. `front` is stored, then `frame_version` bumps last (release store)
//!    — that final bump is the signal every consumer's engine loop sits
//!    on.
//!
//! The process is deliberately short-lived: with no consumer registered
//! for the exit grace it quits (the next glass app re-spawns it), and
//! while consumers exist but none is active it tears the capture feed
//! down, exactly like the in-process engine parks its pipeline.
//! Consumer registration is pruned by process liveness — one waitable
//! process handle per registered pid — so once every consumer app has
//! exited (by any means) the registry empties and the exit grace runs.

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::device_adapter_luid;
use super::protocol::ManifestView;
use crate::capture::CaptureFeed;
use crate::quality::{detect_gpu_class, fullscreen_foreground, resolve};
use crate::surface::create_d3d11_device;

mod consumers;
mod frame_host;
mod material;
mod publish;

use consumers::{maintain, manifest_any_consumer, ConsumerLiveness, ConsumerRegistry};
use frame_host::BrokerHost;
use material::MaterialWatcher;
use publish::PublishEngine;
/// The idle feed grace while consumers still exist but no one is active
/// (mirrors the in-process engine's hysteresis so a preview or a quick
/// hide never churns the capture pipeline).
const IDLE_FEED_TICKS: u64 = 120;
/// The process-exit grace with no registered consumer at all; the next
/// glass app re-spawns the broker.
const EXIT_IDLE_TICKS: u64 = 640;
/// Consumer maintenance cadence: prune dead processes, duplicate handles
/// into requests.
const MAINTENANCE_INTERVAL: Duration = Duration::from_millis(250);
/// Ticks to wait before re-attempting a failed capture start (a persistent
/// failure must not retry at the loop's tick rate).
const CAPTURE_RETRY_BACKOFF: u64 = 120;
/// How often the publish loop re-checks the fullscreen-foreground state
/// (the shell query gates the backdrop rate under fullscreen apps).
const FULLSCREEN_POLL: Duration = Duration::from_millis(500);
/// How often the publish loop re-checks battery (Auto tiers only).
const POWER_POLL: Duration = Duration::from_secs(10);
/// The broker's entry point: owns the shared manifest, the capture feed
/// and the publication; exits when the last consumer releases. Non-zero
/// on startup failure.
pub fn run() -> i32 {
    // The mapping create is the single-broker gate (the kernel marks the
    // pre-existing name, and a second broker exits without touching any
    // consumer state).
    let manifest_view = match ManifestView::create() {
        Ok(view) => Arc::new(view),
        Err(error) => {
            eprintln!("glass broker exiting: {error}");
            return 2;
        }
    };
    let manifest = manifest_view.manifest();
    let (device, context) = match create_d3d11_device() {
        Ok(pair) => pair,
        Err(error) => {
            // Leaving the failure here still drops the mapping view, so
            // the name frees and a later broker can claim it.
            eprintln!("glass broker exiting: {error}");
            return 1;
        }
    };
    // The publisher's adapter identity: consumers on another adapter must
    // never attach (NT-handle shares are same-adapter only).
    match device_adapter_luid(&device) {
        Ok(luid) => manifest.adapter_luid.store(luid, Ordering::Release),
        Err(error) => {
            eprintln!("glass broker exiting: {error}");
            return 1;
        }
    }
    manifest
        .broker_pid
        .store(std::process::id(), Ordering::Release);

    // The shared engine's parts, mirroring the in-process engine's setup:
    // one host (capture delivery), one feed (restarted on demand), one
    // publication engine.
    let registry = Arc::new(ConsumerRegistry {
        shared: Mutex::new(Some(Arc::clone(&manifest_view))),
    });
    let host: Arc<Mutex<BrokerHost>> = Arc::new(Mutex::new(BrokerHost::new(
        device.clone(),
        context.clone(),
        Arc::clone(&registry),
    )));
    let version = host
        .lock()
        .expect("broker host lock poisoned")
        .version_counter();
    let host_dyn: Arc<Mutex<dyn crate::capture::FrameHost>> = Arc::clone(&host) as _;
    let feed: Arc<Mutex<Option<CaptureFeed>>> = Arc::new(Mutex::new(None));

    let mut engine = PublishEngine::new(&device);
    let mut material = MaterialWatcher::new();
    let mut idle_ticks = 0u64;
    let mut restart_backoff = 0u64;
    // Publication failures log once per distinct reason: a persistent
    // failure repeats every tick, and the recovery recreate addresses it
    // without a log storm.
    let mut last_publish_error = None;
    let mut last_maintenance = Instant::now()
        .checked_sub(MAINTENANCE_INTERVAL)
        .unwrap_or_else(Instant::now);
    let my_pid = std::process::id();
    // The consumer-liveness table (see [`ConsumerLiveness`]): the handle
    // per registered pid that makes slot pruning authoritative.
    let mut liveness = ConsumerLiveness::default();
    // The quality tier: resolved from the material's mode on this
    // machine's GPU class (and battery), re-checked on material change,
    // battery transition and fullscreen-foreground transitions.
    let gpu_class = detect_gpu_class(&device);
    let mut quality_mode = material.quality;
    let mut quality_caps = resolve(quality_mode, gpu_class);
    let mut fullscreen = false;
    let mut last_fullscreen_check: Option<Instant> = None;
    let mut last_power_check: Option<Instant> = None;

    loop {
        std::thread::sleep(Duration::from_millis(16));

        // The material's params poll (its revision drives the blur
        // refresh) and the frame/backdrop refresh + publication cycle.
        material.poll(&mut engine.params);
        let material_revision = material.revision();
        if material.quality != quality_mode {
            quality_mode = material.quality;
            quality_caps = resolve(quality_mode, gpu_class);
            // The chain caps live outside the params fingerprint: the
            // backdrop must rebuild even if the material revision didn't
            // move.
            engine.invalidate_backdrop();
        }
        let now = Instant::now();
        if last_fullscreen_check.map_or(true, |at| now.duration_since(at) >= FULLSCREEN_POLL) {
            last_fullscreen_check = Some(now);
            fullscreen = fullscreen_foreground();
        }
        if last_power_check.map_or(true, |at| now.duration_since(at) >= POWER_POLL) {
            last_power_check = Some(now);
            let battery = crate::quality::on_battery();
            if material.battery_seen != Some(battery) {
                material.battery_seen = Some(battery);
                quality_caps = resolve(quality_mode, gpu_class);
                engine.invalidate_backdrop();
            }
        }
        // The capture callback's copy gate reads the interval through the
        // host; the publication rate follows the copy rate automatically.
        host.lock()
            .expect("broker host lock poisoned")
            .set_backdrop_interval(quality_caps.backdrop_interval(fullscreen));
        {
            let host_guard = host.lock().expect("broker host lock poisoned");
            let host_version = host_guard.frame_version.load(Ordering::Acquire);
            engine.refresh(
                &context,
                &host_guard,
                host_version,
                material_revision,
                &quality_caps,
            );
            if let Err(error) = engine.publish_inner(
                &device,
                &context,
                &host_guard,
                manifest,
                host_version,
                &quality_caps,
            ) {
                if last_publish_error.as_deref() != Some(error.as_str()) {
                    eprintln!("glass broker publish failed: {error}");
                    last_publish_error = Some(error);
                }
            } else {
                last_publish_error = None;
            }
        }

        // Consumer maintenance on its own cadence.
        if last_maintenance.elapsed() >= MAINTENANCE_INTERVAL {
            last_maintenance = Instant::now();
            maintain(manifest, engine.pair.as_ref(), my_pid, &mut liveness);
        }

        // Capture feed lifecycle: start it for active consumers, park it
        // once everyone sleeps, and exit once nobody is registered.
        let any_active = registry.any_active();
        let some_consumer = manifest_any_consumer(manifest);
        if some_consumer && any_active {
            idle_ticks = 0;
            if restart_backoff > 0 {
                restart_backoff -= 1;
            } else if feed.lock().expect("broker feed lock poisoned").is_none() {
                match CaptureFeed::start(&host_dyn, &version) {
                    Ok(new_feed) => {
                        *feed.lock().expect("broker feed lock poisoned") = Some(new_feed);
                    }
                    Err(error) => {
                        eprintln!("glass broker capture failed to start: {error}");
                        restart_backoff = CAPTURE_RETRY_BACKOFF;
                    }
                }
            }
        } else {
            idle_ticks += 1;
            if !some_consumer {
                if idle_ticks >= EXIT_IDLE_TICKS {
                    break;
                }
            } else if idle_ticks >= IDLE_FEED_TICKS {
                // Consumers are registered but every one is parked: park
                // the capture with the same hysteresis as the in-process
                // engine (the first slot to raise its active flag restarts
                // the feed above).
                idle_ticks = 0;
                *feed.lock().expect("broker feed lock poisoned") = None;
            }
        }
    }

    // Teardown ordering matters: the capture stops and the host drops
    // before the manifest view does (the capture callback reads the slots
    // through the registry).
    *feed.lock().expect("broker feed lock poisoned") = None;
    drop(host);
    drop(registry);
    0
}
