//! The WGC teardown reaper: ONE worker thread that drains every capture
//! session's WinRT teardown, for all three feed kinds (the global monitor
//! feed, the per-window local feed, and the hover-preview thumbnail).
//!
//! Why a single worker instead of a spawn per teardown: WGC's `Close`
//! synchronizes with the capture pipeline, and a wedged pipeline (or a
//! teardown racing another one) must never accumulate blocked threads —
//! the unbounded pile of detached reapers is exactly what the first hang
//! dump showed (~100 threads parked inside `Close`, eventually wedging the
//! loader lock against the main thread). Serializing teardowns also keeps
//! WGC's internal handshakes from racing each other. Callers never wait:
//! the fence (epoch/generation bump under the caller's lock) stays
//! synchronous at the call site; only the WinRT teardown is queued here.

use std::sync::mpsc::{channel, Sender};
use std::sync::{Mutex, OnceLock};
use windows::Graphics::Capture::{Direct3D11CaptureFramePool, GraphicsCaptureSession};

/// One session's teardown: the pool (whose `RemoveFrameArrived` must run
/// before `Close`), the session, and the event token — `None` when the
/// callback was never registered.
pub(crate) struct Teardown {
    pub(crate) pool: Option<Direct3D11CaptureFramePool>,
    pub(crate) session: Option<GraphicsCaptureSession>,
    pub(crate) token: Option<windows::Foundation::EventRegistrationToken>,
}

// The pool/session are agile free-threaded WinRT objects; the reaper worker
// is their only remaining user after the caller's fence. windows-rs does not
// mark them `Send` on its own.
unsafe impl Send for Teardown {}

static SENDER: OnceLock<Mutex<Sender<Teardown>>> = OnceLock::new();

/// Queues `teardown` onto the reaper worker. Never blocks the caller on WGC
/// (the only synchronization is the enqueue, which cannot fail meaningfully:
/// a wedged queue drops the teardown — the process is wedged anyway).
pub(crate) fn reap(teardown: Teardown) {
    let sender = SENDER.get_or_init(|| {
        let (tx, rx) = channel::<Teardown>();
        std::thread::Builder::new()
            .name("liquid-glass-wgc-reaper".to_owned())
            .spawn(move || {
                while let Ok(job) = rx.recv() {
                    if let (Some(pool), Some(token)) = (job.pool.as_ref(), job.token) {
                        let _ = pool.RemoveFrameArrived(token);
                    }
                    if let Some(session) = job.session.as_ref() {
                        let _ = session.Close();
                    }
                    if let Some(pool) = job.pool {
                        let _ = pool.Close();
                    }
                    // The COM references release here, on the worker.
                }
            })
            .expect("could not spawn the WGC teardown reaper");
        Mutex::new(tx)
    });
    if let Ok(sender) = sender.lock() {
        let _ = sender.send(teardown);
    }
}
