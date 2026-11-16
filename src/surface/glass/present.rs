//! Presentation control: the reveal animation, pausing and frame gating,
//! the rate caps and the one-shot present.

use crate::params::RevealAnimation;
use crate::shared::engine;
use crate::surface::window::hide_surface_window;

use super::GlassSurface;
impl GlassSurface {
    /// Starts (or reverses) the reveal animation for this surface.
    pub fn set_reveal(&mut self, open: bool, reveal: RevealAnimation) {
        self.compositor
            .lock()
            .expect("D3D11 compositor lock poisoned")
            .set_reveal(open, reveal);
    }

    /// Hides the surface window outright. Used when parking a surface: even
    /// though the collapsed reveal renders fully transparent, the window can
    /// briefly (or permanently, if capture stalls) keep showing its last
    /// presented frame — hiding the window guarantees nothing lingers.
    pub fn hide_window(&self) -> Result<(), String> {
        hide_surface_window(self.hwnd)
    }

    /// Stops (or resumes) all rendering: paused surfaces present nothing.
    pub fn set_paused(&mut self, paused: bool) {
        self.compositor
            .lock()
            .expect("D3D11 compositor lock poisoned")
            .set_paused(paused);
    }

    /// Whether all rendering is stopped for this surface.
    pub fn is_paused(&self) -> bool {
        self.compositor
            .lock()
            .map(|compositor| compositor.is_paused())
            .unwrap_or(true)
    }

    /// Composites and presents this surface's CURRENT field right now, on
    /// the caller's thread (the engine and compositor locks serialize it
    /// with the render loop). The shared metaball field's present sync needs
    /// this: a dragged blob's surface moves with `set_anchor_keep_z` (the
    /// window position applies on the next composition beat), while the
    /// render loop's throttled tick can land 1-2 beats later — the moved
    /// window would then compose its OLD field's blob, and the gaps between
    /// the sibling surfaces' halves stretch with drag speed. Presenting in
    /// the same op as the field stamp (and the window move) folds both into
    /// one composition beat; the last member's push presents the whole field
    /// group back to back (see the glass-ui manager's present-sync gate).
    pub fn present_now(&mut self) -> Result<(), String> {
        engine().present_surface_now(&self.compositor, self.engine_id)
    }

    /// Lets exactly one frame render, then pauses (capture stays warm).
    pub fn pause_after_next_present(&mut self) {
        self.compositor
            .lock()
            .expect("D3D11 compositor lock poisoned")
            .pause_after_next_present();
    }

    /// Holds this surface's present back until `deadline`: the field it
    /// holds must not compose on screen while sibling surfaces sharing the
    /// same metaball field generation still show their previous field. The
    /// shared-field manager gates a surface the instant it applies a new
    /// field and releases it the moment the group has caught up (or the
    /// deadline lapses as a safety net), so a dragged blob's halves always
    /// compose from the same field — the merged neck never opens a
    /// mid-frame gap.
    pub fn gate_frame(&mut self, deadline: std::time::Instant) {
        self.compositor
            .lock()
            .expect("D3D11 compositor lock poisoned")
            .gate_frame(deadline);
    }

    /// Lifts a [`GlassSurface::gate_frame`] hold early (the field group
    /// caught up): the held field renders on the next presentation tick,
    /// back to back with the siblings' same-generation fields.
    pub fn release_frame_gate(&mut self) {
        self.compositor
            .lock()
            .expect("D3D11 compositor lock poisoned")
            .release_frame_gate();
    }

    /// Renders the collapse animation to the end, then pauses.
    pub fn collapse_then_pause(&mut self) {
        self.compositor
            .lock()
            .expect("D3D11 compositor lock poisoned")
            .collapse_then_pause();
    }

    /// Caps the idle render rate in frames per second (`None` = uncapped).
    pub fn set_frame_cap(&mut self, fps: Option<f32>) {
        self.compositor
            .lock()
            .expect("D3D11 compositor lock poisoned")
            .set_min_present_interval(match fps {
                Some(fps) if fps > 0.0 => std::time::Duration::from_secs_f32(1.0 / fps),
                _ => std::time::Duration::ZERO,
            });
    }

    /// Enables adaptive pacing: content-driven frames render at the shared
    /// render loop's full rate while fresh capture frames keep arriving,
    /// then settle to the idle frame cap once the content goes quiet.
    pub fn set_adaptive_rate(&mut self, enabled: bool) {
        self.compositor
            .lock()
            .expect("D3D11 compositor lock poisoned")
            .set_adaptive_rate(enabled);
    }

    /// Whether at least one frame has been presented (the capture session is
    /// warm and the shader has produced output).
    pub fn has_presented(&mut self) -> bool {
        self.compositor
            .lock()
            .expect("D3D11 compositor lock poisoned")
            .has_presented
    }
}
