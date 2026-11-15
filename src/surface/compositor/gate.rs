//! The compositor's scheduling state machine: parking (paused surfaces),
//! the idle frame cap, adaptive pacing, the shared-field present gate and
//! the two frame-request predicates (`needs_present` / `wants_frame`) the
//! render loop drives every tick with.

use std::time::{Duration, Instant};

use crate::params::GlassShaderParams;

use super::D3D11Compositor;

impl D3D11Compositor {

    /// Stops (or resumes) all rendering for this surface. Parked surfaces are
    /// paused: their presentation tick and capture callbacks then do nothing
    /// beyond a boolean check, so invisible glass costs no GPU time.
    pub fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
        if !paused {
            self.pause_after_present = false;
            self.pause_when_collapsed = false;
        }
        // Force one fresh frame with the current state on resume.
        self.params_dirty = true;
    }

    /// Whether all rendering is stopped for this surface. The capture feed
    /// uses this to skip the full-screen frame copy while every surface is
    /// parked.
    pub fn is_paused(&self) -> bool {
        self.paused
    }

    /// Switches the composite between the full and lite shader variants
    /// (the quality tier's choice). Marks the surface dirty so the next
    /// presentation tick renders one frame with the new variant.
    pub(crate) fn set_lite_shader(&mut self, lite: bool) {
        if self.lite_shader == lite {
            return;
        }
        self.lite_shader = lite;
        self.params_dirty = true;
    }

    /// The surface's current material params. All surfaces derive from the
    /// same saved settings, so the engine uses the first active surface's
    /// copy to plan the shared blur (blur radius and samples are identical
    /// across surfaces).
    pub(crate) fn active_params(&self) -> GlassShaderParams {
        self.glass_params_current
    }

    /// The surface's sampling rect in monitor space as uv (x0, y0, x1, y1),
    /// for the shared GLOBAL blur's union crop: the engine blurs only the
    /// region the awake Global surfaces actually sample. `None` while the
    /// surface is parked, or when its samples hang in a space other than
    /// the monitor (a Local feed's window frame) — the engine falls back to
    /// the full-frame blur in the latter case.
    pub(crate) fn monitor_uv_rect(&self) -> Option<[f32; 4]> {
        if self.paused || !self.monitor_space {
            return None;
        }
        let origin = self.source_rect_constants.source_uv_origin;
        let size = self.source_rect_constants.source_uv_size;
        Some([
            origin[0],
            origin[1],
            origin[0] + size[0],
            origin[1] + size[1],
        ])
    }

    /// Whether the shared material watch must skip this surface (see the
    /// `material_locked` field).
    pub(crate) fn material_locked(&self) -> bool {
        self.material_locked
    }

    /// Pins (or unpins) the surface against the shared material watch: a
    /// pinned surface keeps its app-derived params; the app re-applies the
    /// material itself with its own per-surface rules.
    pub fn set_material_locked(&mut self, locked: bool) {
        self.material_locked = locked;
    }

    /// Lets exactly one more frame render, then pauses. Hidden surfaces use
    /// this to warm their capture session (and become "presented") without
    /// rendering continuously.
    pub fn pause_after_next_present(&mut self) {
        self.paused = false;
        self.pause_after_present = true;
        self.pause_when_collapsed = false;
        self.params_dirty = true;
    }

    /// Renders the collapse animation to the end, then pauses. Used when
    /// parking a surface: the close transition must play out (pausing
    /// immediately freezes the last visible frame on screen), and once the
    /// glass is fully transparent no more rendering is needed.
    pub fn collapse_then_pause(&mut self) {
        self.pause_when_collapsed = true;
        // An already-collapsed surface needs no collapse render: its last
        // frame is already transparent, so park it immediately.
        if self.reveal_current <= 0.001 && self.reveal_target <= 0.001 {
            self.paused = true;
            self.pause_when_collapsed = false;
            return;
        }
        self.params_dirty = true;
    }

    /// Caps the idle render rate (content-driven frames); zero disables the
    /// cap. Reveal animations always run at the caller's tick rate.
    pub fn set_min_present_interval(&mut self, interval: Duration) {
        self.min_present_interval = interval;
    }

    /// Enables adaptive pacing: content-driven frames render at the shared
    /// loop's full rate while fresh capture frames keep arriving, then fall
    /// back to the idle frame cap after the content goes quiet.
    pub fn set_adaptive_rate(&mut self, enabled: bool) {
        self.adaptive_rate = enabled;
    }

    /// Enforces the present gate: once the deadline has passed the gate
    /// lifts itself (marking the held params dirty so the next tick renders
    /// them), giving back `true` only while the gate is still active.
    fn gate_blocks(&mut self) -> bool {
        let Some(deadline) = self.present_gate_until else {
            return false;
        };
        if Instant::now() < deadline {
            return true;
        }
        // The gate expired unfilled (a surfaced sibling never caught up):
        // render the held field rather than stall the surface forever.
        self.present_gate_until = None;
        self.params_dirty = true;
        false
    }

    /// Holds the compositor back from presenting until `deadline` (the
    /// present gate above): the field it holds must not compose against
    /// siblings' older field halves. Released automatically at the deadline
    /// or explicitly when the whole field group has caught up.
    pub fn gate_frame(&mut self, deadline: Instant) {
        self.present_gate_until = Some(deadline);
    }

    /// Lifts the present gate early (the field group caught up): the held
    /// field is marked dirty so the very next tick renders it, back to back
    /// with the siblings' same-generation fields.
    pub fn release_frame_gate(&mut self) {
        if self.present_gate_until.take().is_some() {
            self.params_dirty = true;
        }
    }

    /// Whether the presentation tick must render right now: the reveal
    /// animation is running, or the params changed and need a frame to land.
    pub fn needs_present(&self) -> bool {
        if self.paused {
            return false;
        }
        if self
            .present_gate_until
            .is_some_and(|deadline| Instant::now() < deadline)
        {
            return false;
        }
        if self.pause_after_present {
            return true;
        }
        self.is_animating() || self.reveal_dirty || self.params_dirty
    }

    /// Whether the render loop should render for this surface given the
    /// shared capture state. Surface-local needs (animation, dirty params)
    /// always render; content-driven needs additionally respect the frame
    /// cap via `present`'s own throttle. Fresh capture frames also refresh
    /// the adaptive pacing's content timer.
    pub fn wants_frame(&mut self, frame_changed: bool) -> bool {
        if self.paused {
            return false;
        }
        if self.gate_blocks() {
            return false;
        }
        if self.pause_after_present {
            return true;
        }
        if frame_changed {
            self.last_content_frame = Some(Instant::now());
        }
        if self.is_animating() || self.reveal_dirty || self.params_dirty {
            return true;
        }
        frame_changed
    }

    /// Whether a continuous compositor-owned animation is running (the
    /// reveal, the shape transform's pursuit spring or the pill squash
    /// spring). The shared render loop uses this to pace its ticks to the
    /// desktop compositor instead of the fixed idle interval.
    pub fn is_animating(&self) -> bool {
        if self.paused {
            return false;
        }
        (self.reveal_current - self.reveal_target).abs() > 0.001
            || self.shape_spring.is_some()
            || self.pill_spring.is_some()
    }
}
