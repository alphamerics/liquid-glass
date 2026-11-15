//! The shared render loop: ONE thread that composites every registered
//! glass surface.
//!
//! Per tick the loop polls the material file, the fullscreen/battery state
//! and the broker; refreshes the GLOBAL and LOCAL backdrops (one blur per
//! fresh capture frame); dispatches one present per surface that wants a
//! frame; and drives the capture feeds' lifecycle (start with the first
//! awake surface, idle out with the last, escalating backoff on failures).
//! Feed teardowns run strictly outside the engine state lock.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use windows::Win32::Graphics::Direct3D11::ID3D11DeviceContext;
use windows::Win32::Graphics::Dwm::DwmFlush;

use crate::backdrop::{local_backdrop, refresh_local_backdrop, start_local_feed};
use crate::broker::client::BrokerClient;
use crate::capture::{CaptureFeed, FrameHost};
use crate::params::BackdropSource;
use crate::quality::{fullscreen_foreground, QualityCaps};
use crate::window_capture::WindowCaptureFeed;

use super::backdrop::{refresh_backdrop, Backdrop};
use super::{device_lost, EngineState, MaterialWatch};

/// How often the render loop re-checks the fullscreen-foreground state that
/// throttles the backdrop rate under games/presentations.
const FULLSCREEN_POLL: Duration = Duration::from_millis(500);

/// How often the render loop re-checks the battery state (Auto tiers only).
const POWER_POLL: Duration = Duration::from_secs(10);

/// Hysteresis for the feed lifecycle (about two seconds at the loop's ~60Hz
/// tick): brief all-parked moments (a settings preview, a quick hide/show)
/// must not churn the capture session.
const IDLE_FEED_TICKS: u32 = 120;

/// Escalating restart backoff for a capture feed that keeps failing to
/// start. The first failure waits one hysteresis window (~2s); every
/// further consecutive failure quadruples the wait (capped at ~2 minutes),
/// so a permanently rejected capture degrades into a quiet retry every
/// couple of minutes instead of hammering the capture service — and
/// flooding the log — every two seconds.
struct RestartBackoff {
    consecutive_failures: u32,
    ticks: u32,
}

impl RestartBackoff {
    fn new() -> Self {
        Self {
            consecutive_failures: 0,
            ticks: 0,
        }
    }

    /// Whether a start attempt may run now; consumes one waited tick
    /// otherwise (the loop's own cadence drives the countdown).
    fn ready(&mut self) -> bool {
        if self.ticks > 0 {
            self.ticks -= 1;
            false
        } else {
            true
        }
    }

    fn note_success(&mut self) {
        self.consecutive_failures = 0;
        self.ticks = 0;
    }

    /// Records a failed start: escalates the wait and logs — the first
    /// failure with the bare error, the second with the recovery posture
    /// and the known refusal causes, further ones silently.
    fn note_failure(&mut self, scope: &str, error: &str) {
        self.consecutive_failures += 1;
        let level = self.consecutive_failures.min(4);
        self.ticks = IDLE_FEED_TICKS
            .saturating_mul(1u32 << (2 * (level - 1)))
            .min(IDLE_FEED_TICKS.saturating_mul(64));
        match self.consecutive_failures {
            1 => eprintln!("Glassbar {scope} failed to start: {error}"),
            2 => eprintln!(
                "Glassbar {scope} keeps failing to start ({error}); retrying at a slower \
                 cadence. Window capture is refused for some hosts: windows hidden from \
                 the taskbar (tauri's skipTaskbar calls ITaskbarList::DeleteTab), \
                 WS_EX_NOACTIVATE / WS_EX_TOOLWINDOW windows, minimized windows, and \
                 windows protected with SetWindowDisplayAffinity."
            ),
            _ => {}
        }
    }
}

pub(crate) fn spawn_render_loop(
    state: Arc<Mutex<EngineState>>,
    version: Arc<AtomicU64>,
    feed: Arc<Mutex<Option<CaptureFeed>>>,
    local_feeds: Arc<Mutex<HashMap<isize, WindowCaptureFeed>>>,
) {
    std::thread::spawn(move || {
        // The feed's host view of the engine state: the capture pipeline
        // is host-agnostic (the broker runs the exact same code over its
        // own host), so this loop owns the trait object locally.
        let host: Arc<Mutex<dyn FrameHost>> = Arc::clone(&state) as _;
        let mut last_version = 0u64;
        let mut idle_ticks = 0u32;
        // After a failed capture start, wait a full hysteresis window before
        // retrying so a persistent failure does not spam the session at the
        // loop's tick rate.
        let mut restart_backoff = RestartBackoff::new();
        // The broker client machine (see `broker::client`): every tick
        // advances attach/pull; a silent failure falls back to the local
        // pipeline by returning an `Err` once and resetting to `Idle`.
        let mut broker = BrokerClient::new();
        // The shared material watch (see [`MaterialWatch`]): file changes
        // restyle every registered compositor within one poll cadence.
        let mut material = MaterialWatch::default();
        // Fullscreen foreground state (throttled shell query) that drops
        // the backdrop rate to the tier's fullscreen cap.
        let mut fullscreen = false;
        let mut last_fullscreen_check: Option<Instant> = None;
        // Battery transitions re-resolve the Auto tier (throttled query).
        let mut last_power_check: Option<Instant> = None;
        // The caps the current backdrop was built under: a tier change
        // (settings edit or battery transition) must rebuild it.
        let mut last_caps: Option<QualityCaps> = None;
        // The interval already applied to the capture session (applied only
        // on change, outside the state lock).
        let mut applied_session_interval: Option<Duration> = None;
        // Per-LOCAL-window bookkeeping: the frame version still owed to
        // surfaces (dispatched after each pass), the parked-idle counters
        // and per-window start backoff.
        let mut local_signal: HashMap<isize, u64> = HashMap::new();
        let mut local_seen_version: HashMap<isize, u64> = HashMap::new();
        let mut local_idle_ticks: HashMap<isize, u32> = HashMap::new();
        let mut local_restarts: HashMap<isize, RestartBackoff> = HashMap::new();
        // When any surface animates, the tick waits for the desktop
        // compositor (DwmFlush) instead of the fixed idle interval, so
        // animation presents land phase-aligned with composition — the
        // difference between a spring reading as one smooth motion and as
        // micro-stutter. Computed during the previous tick's surface pass.
        let mut pace_with_dwm = false;
        'main: while state
            .lock()
            .expect("shared glass lock poisoned")
            .render_running
            .load(Ordering::Relaxed)
        {
            if pace_with_dwm {
                // Sync interval 0 presents (see `D3D11Compositor::present`)
                // cannot wait here — the loop itself aligns to composition.
                // A failed flush (no DWM, headless session) falls back to a
                // short sleep instead of spinning.
                if unsafe { DwmFlush() }.is_err() {
                    std::thread::sleep(Duration::from_millis(8));
                }
            } else {
                std::thread::sleep(Duration::from_millis(16));
            }
            let mut engine = state.lock().expect("shared glass lock poisoned");

            // The shared material may have changed on disk (a settings UI
            // in any glass app wrote it); push it onto the compositors.
            material.poll(&mut engine);

            // Slow-cadence environment re-resolution: the fullscreen and
            // power state gate the tier's backdrop rate; both shell calls
            // are cheap but not free, and neither changes per frame.
            let now = Instant::now();
            if last_fullscreen_check.map_or(true, |at| now.duration_since(at) >= FULLSCREEN_POLL) {
                last_fullscreen_check = Some(now);
                fullscreen = fullscreen_foreground();
            }
            if last_power_check.map_or(true, |at| now.duration_since(at) >= POWER_POLL) {
                last_power_check = Some(now);
                let battery = crate::quality::on_battery();
                if engine.last_battery != Some(battery) {
                    engine.last_battery = Some(battery);
                    let mode = engine.quality_mode;
                    engine.apply_quality(mode);
                    let lite = engine.quality.lite_shader;
                    for (_, compositor, _route) in &engine.surfaces {
                        if let Ok(mut compositor) = compositor.lock() {
                            compositor.set_lite_shader(lite);
                        }
                    }
                }
            }
            engine.backdrop_interval = engine.quality.backdrop_interval(fullscreen);
            let effective_interval = engine.backdrop_interval;

            // The awake-GLOBAL-surface answer drives the global pipelines:
            // the broker's slot flag and the monitor feed's lifecycle.
            // LOCAL feeds are gated per capturing window further down. A
            // compositor whose lock is poisoned counts as awake — a stuck
            // lock must never park the glass.
            let any_global_active = engine.surfaces.iter().any(|(_, compositor, route)| {
                route.source == BackdropSource::Global
                    && compositor.lock().map(|c| !c.is_paused()).unwrap_or(true)
            });
            // A tier change (settings edit, battery transition) changes the
            // chain caps of EVERY feed: force one rebuild even without a
            // fresh capture frame.
            let tier_changed = last_caps.map_or(false, |caps| caps != engine.quality);
            last_caps = Some(engine.quality);

            // Broker attempt first: when it hands over a fresh backdrop the
            // global pipeline is skipped entirely this tick. Only Global
            // consumers participate — the broker publishes the shared
            // screen material, which Local glass must not adopt. A process
            // with no Global surfaces (and no attachment yet) never starts
            // the broker machine at all: a local-only app neither spawns
            // nor contacts a broker.
            let mut frame_changed = false;
            let mut broker_attached = false;
            if any_global_active || broker.attached() {
                match broker.tick(&engine.device, &engine.context, any_global_active) {
                    Ok(Some(frame)) => {
                        engine.backdrop = Some(frame.backdrop);
                        frame_changed = true;
                        broker_attached = true;
                    }
                    Ok(None) => broker_attached = broker.attached(),
                    Err(error) => {
                        eprintln!(
                            "glass broker unavailable, running the in-process engine instead: \
                             {error}"
                        );
                        // Forget whatever the broker published: the global
                        // pipeline must build the next backdrop fresh.
                        engine.backdrop = None;
                    }
                }
            }

            let current = version.load(Ordering::Relaxed);
            let version_changed = current != last_version;
            last_version = current;
            // One blur per fresh capture frame — never per surface, never
            // per animation tick. The first tick after startup forces one
            // render even before a capture frame lands, so freshly spawned
            // surfaces composite from the very first frame. In broker mode
            // the global chain never runs: the published backdrop replaces
            // its output wholesale.
            if !broker_attached {
                let first_backdrop = engine.backdrop.is_none();
                let blur_failed =
                    refresh_backdrop(&mut engine, current, first_backdrop || tier_changed);
                if blur_failed && device_lost(&engine.device) {
                    eprintln!(
                        "Glassbar glass device removed (driver reset); \
                         stopping the glass engine"
                    );
                    engine.render_running.store(false, Ordering::Relaxed);
                    break 'main;
                }
            }

            // Each LOCAL feed blurs its own window capture independently of
            // the broker above; refreshes follow the same
            // content-force/tier-force cadence. The window's present
            // signal — a fresh frame surfaces have not seen yet — is
            // snapshot per window and consumed by the dispatch below.
            for window in engine.local.keys().copied().collect::<Vec<_>>() {
                let first_backdrop = engine
                    .local
                    .get(&window)
                    .map_or(false, |entry| entry.backdrop.is_none());
                let blur_failed =
                    refresh_local_backdrop(&mut engine, window, first_backdrop || tier_changed);
                if blur_failed && device_lost(&engine.device) {
                    eprintln!(
                        "Glassbar glass device removed (driver reset); \
                         stopping the glass engine"
                    );
                    engine.render_running.store(false, Ordering::Relaxed);
                    break 'main;
                }
                let fresh_version = engine
                    .local
                    .get(&window)
                    .map_or(0, |entry| entry.version.load(Ordering::Relaxed));
                local_signal.insert(window, fresh_version);
            }

            // One drawer, routed by each surface's backdrop feed: a Global
            // surface composites the shared monitor backdrop (driven by the
            // monitor's frame version, or the broker's pull once attached),
            // a Local surface its own window's backdrop (driven by that
            // window's frame version). Animations and dirty params still
            // request frames through `wants_frame` regardless.
            let mut any_animating = false;
            for (_, compositor, route) in &engine.surfaces {
                let Ok(mut compositor) = compositor.lock() else {
                    continue;
                };
                any_animating |= compositor.is_animating();
                let device_lost = match route.source {
                    BackdropSource::Global => {
                        let present_signal = if broker_attached {
                            frame_changed
                        } else {
                            version_changed
                        };
                        if compositor.wants_frame(present_signal) {
                            engine.backdrop.as_ref().map_or(false, |backdrop| {
                                failed_present(&mut compositor, &engine.context, backdrop)
                            })
                        } else {
                            false
                        }
                    }
                    BackdropSource::Local => {
                        let Some(window) = route.window else {
                            continue;
                        };
                        let fresh = local_signal.get(&window).copied().unwrap_or(0);
                        let changed =
                            fresh != local_seen_version.get(&window).copied().unwrap_or(0);
                        if compositor.wants_frame(changed) {
                            match local_backdrop(&engine, window) {
                                Some(backdrop) => {
                                    // `failed_present` needs a `&mut
                                    // compositor` while `backdrop` borrows
                                    // `engine`: clone the cheap COM view
                                    // out of the engine borrow first.
                                    let backdrop = backdrop.clone();
                                    failed_present(&mut compositor, &engine.context, &backdrop)
                                }
                                None => false,
                            }
                        } else {
                            false
                        }
                    }
                };
                if device_lost {
                    eprintln!(
                        "Glassbar glass device removed (driver reset); \
                         stopping the glass engine"
                    );
                    engine.render_running.store(false, Ordering::Relaxed);
                    break 'main;
                }
            }
            // Everything that needed the newest frame version saw it; the
            // next versions become the following tick's fresh signal.
            local_seen_version.extend(local_signal.drain());
            // The next tick paces to composition while any animation runs,
            // and falls back to the idle interval when everything settled.
            pace_with_dwm = any_animating;
            drop(engine);
            // The session-level rate cap (Windows 11 22H2+): WGC coalesces
            // its frame deliveries to the interval, so the capture callback
            // — and its blit — never even run above the cap. Applied
            // outside the state lock, only when the interval moved (tier
            // change or fullscreen transition), to BOTH the monitor feed
            // and every live local feed. Older systems ignore this and
            // rely on the callbacks' capture-due gates.
            if effective_interval != applied_session_interval {
                applied_session_interval = effective_interval;
                if let Some(active_feed) = feed
                    .lock()
                    .expect("shared glass feed lock poisoned")
                    .as_ref()
                {
                    active_feed.set_min_update_interval(effective_interval);
                }
                for local_feed in local_feeds
                    .lock()
                    .expect("shared glass local feed lock poisoned")
                    .values()
                {
                    local_feed.set_min_update_interval(effective_interval);
                }
            }
            // GLOBAL feed lifecycle, strictly outside the state lock:
            // starting the feed re-locks it, and dropping a feed while its
            // own frame callback blocks on that lock could deadlock the
            // teardown.
            if broker_attached {
                // The broker publishes the backdrop: the monitor capture
                // and blur pipeline is pure duplicate cost — tear it down.
                // The broker's own feed is parked through the slot's active
                // flag (the client wrote it) whenever every global surface
                // sleeps.
                idle_ticks = 0;
                *feed.lock().expect("shared glass feed lock poisoned") = None;
            } else if any_global_active {
                idle_ticks = 0;
                if restart_backoff.ready()
                    && !feed
                        .lock()
                        .expect("shared glass feed lock poisoned")
                        .is_some()
                {
                    match CaptureFeed::start(&host, &version) {
                        Ok(new_feed) => {
                            restart_backoff.note_success();
                            *feed.lock().expect("shared glass feed lock poisoned") = Some(new_feed)
                        }
                        Err(error) => restart_backoff.note_failure("shared capture", &error),
                    }
                }
            } else {
                idle_ticks += 1;
                if idle_ticks >= IDLE_FEED_TICKS {
                    idle_ticks = 0;
                    // Every Global surface is parked: the monitor capture is
                    // torn down so the OS capture pipeline idles with it.
                    // The last backdrop stays cached; the first surface to
                    // wake restarts the feed above and composites
                    // immediately from the cached backdrop while the first
                    // frame flies.
                    *feed.lock().expect("shared glass feed lock poisoned") = None;
                }
            }
            // LOCAL feed lifecycles, also strictly outside the state lock:
            // one capture session per consumed window, started with the
            // first awake surface over it and idled out on the same
            // hysteresis. The per-window awake flags are snapshotted under
            // a quick lock, then acted on with the state lock released.
            let local_awake: HashMap<isize, bool> = {
                let engine = state.lock().expect("shared glass lock poisoned");
                engine
                    .local
                    .keys()
                    .copied()
                    .map(|window| {
                        let awake = engine.surfaces.iter().any(|(_, compositor, route)| {
                            route.consumes_window(window)
                                && compositor.lock().map(|c| !c.is_paused()).unwrap_or(true)
                        });
                        (window, awake)
                    })
                    .collect()
            };
            let mut feeds = local_feeds
                .lock()
                .expect("shared glass local feed lock poisoned");
            // Windows whose last surface unregistered: drop their retry
            // bookkeeping so a later re-registration starts fresh.
            local_restarts.retain(|window, _| local_awake.contains_key(window));
            for (window, awake) in local_awake {
                if awake {
                    if let Some(ticks) = local_idle_ticks.get_mut(&window) {
                        *ticks = 0;
                    }
                    if !feeds.contains_key(&window) {
                        let backoff = local_restarts
                            .entry(window)
                            .or_insert_with(RestartBackoff::new);
                        if !backoff.ready() {
                            continue;
                        }
                        // Feeds are started (and their deadlocked-teardown
                        // hazard managed) only ever outside the engine lock.
                        match start_local_feed(&state, window) {
                            Ok(new_feed) => {
                                backoff.note_success();
                                feeds.insert(window, new_feed);
                            }
                            Err(error) => backoff.note_failure(
                                &format!("the local capture of window {window}"),
                                &error,
                            ),
                        }
                    }
                } else if let Some(ticks) = local_idle_ticks.get_mut(&window) {
                    *ticks += 1;
                    if *ticks >= IDLE_FEED_TICKS {
                        *ticks = 0;
                        // The window's glass is parked: its capture session
                        // is torn down (the cached backdrop stays alive for
                        // the wake-up composite).
                        feeds.remove(&window);
                    }
                }
            }
        }
    });
}

/// Runs one compositor present for the routed backdrop and reports whether
/// the failure signals a lost device (a driver reset — common on saturated
/// integrated GPUs — fails every later call: stopping the loop lets the
/// driver recover instead of crash-looping it. The glass freezes on its
/// last frame until the app restarts).
fn failed_present(
    compositor: &mut crate::surface::D3D11Compositor,
    context: &ID3D11DeviceContext,
    backdrop: &Backdrop,
) -> bool {
    if let Err(error) = compositor.present(context, backdrop) {
        // A failed present means the surface silently never appears (hosts
        // gate reveals on `has_presented`); surfacing the error here is
        // the only way to see it.
        eprintln!("Glassbar surface render failed: {error}");
        return true;
    }
    false
}
