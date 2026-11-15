//! The consumer registry and its upkeep: the any-active answer the
//! capture copy gate reads, the kernel-object liveness table that makes
//! slot pruning authoritative, and the per-tick maintenance that hands
//! the shared pair's handles to requested consumers.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use windows::core::HRESULT;
use windows::Win32::Foundation::{
    CloseHandle, DuplicateHandle, DUPLICATE_SAME_ACCESS, HANDLE, WAIT_OBJECT_0, WAIT_TIMEOUT,
};
use windows::Win32::System::Threading::{
    GetCurrentProcess, OpenProcess, WaitForSingleObject, PROCESS_ACCESS_RIGHTS, PROCESS_DUP_HANDLE,
    PROCESS_QUERY_LIMITED_INFORMATION,
};

use crate::broker::protocol::{
    ConsumerSlot, Manifest, ManifestView, CONSUMER_SLOTS, SLOT_REQUESTED, SLOT_SUPPLIED,
    SLOT_UNUSED,
};
use super::publish::{SharedCell, SharedPair};

/// `HRESULT_FROM_WIN32(ERROR_INVALID_PARAMETER)` — the kernel's answer
/// when `OpenProcess` names a pid that no longer exists (any other
/// failure means the process is alive but out of reach).
const DEAD_PID_HRESULT: HRESULT = HRESULT(0x8007_0057u32 as i32);
/// `SYNCHRONIZE` (0x00100000): the process-object right the liveness
/// table's waits need. The `windows` crate does not generate the generic
/// kernel-object access rights, so it is spelled out here.
const SYNCHRONIZE: PROCESS_ACCESS_RIGHTS = PROCESS_ACCESS_RIGHTS(0x0010_0000);
/// The consumer registry the capture callback reads for its any-active
/// answer: does any registered slot claim active rendering right now?
///
/// The manifest pointer is stored as a view through a plain mutex (only
/// ever read from), with all real access going through the mapped atomics
/// — no TLS pins or unsync weak spots remain anywhere.
pub(crate) struct ConsumerRegistry {
    pub(super) shared: Mutex<Option<Arc<ManifestView>>>,
}

impl ConsumerRegistry {
    pub(super) fn any_active(&self) -> bool {
        let guard = self.shared.lock().expect("consumer registry lock poisoned");
        let Some(view) = guard.as_ref() else {
            return false;
        };
        let manifest = view.manifest();
        for index in 0..CONSUMER_SLOTS {
            let slot = manifest.slot(index);
            if slot.pid.load(Ordering::Acquire) == 0 {
                continue;
            }
            // A waiting handshake counts as active too: the shared pair the
            // handshake supplies is built by the publish path, which needs
            // capture frames — and the capture's copy gate reads THIS
            // answer. The consumer's own active flag only lands after the
            // attach completes, so without this the handshake can never
            // start (the broker waits for an active consumer, the consumer
            // waits for the broker's share).
            if slot.active.load(Ordering::Acquire) != 0
                || slot.state.load(Ordering::Acquire) == SLOT_REQUESTED
            {
                return true;
            }
        }
        false
    }
}

pub(super) fn manifest_any_consumer(manifest: &Manifest) -> bool {
    for index in 0..CONSUMER_SLOTS {
        if manifest.slot(index).pid.load(Ordering::Acquire) != 0 {
            return true;
        }
    }
    false
}

/// The broker's consumer-liveness table: one waitable process handle per
/// registered consumer pid, so death is observed through the kernel's own
/// process object. A pid probe (re-opening by id each pass) cannot do
/// this reliably: a dead consumer's pid reused by any unrelated process
/// would masquerade as a live consumer forever, keeping its slot — and
/// with it the broker — open indefinitely.
#[derive(Default)]
pub(super) struct ConsumerLiveness {
    handles: HashMap<u32, HANDLE>,
}

impl ConsumerLiveness {
    /// Whether `pid` names a living process. Opens (and caches) a
    /// SYNCHRONIZE handle on first sight; later passes wait on the cached
    /// handle with a zero timeout — the kernel signals a terminated
    /// process's object even if another process has since reused the pid.
    fn is_alive(&mut self, pid: u32) -> bool {
        if let Some(&handle) = self.handles.get(&pid) {
            match unsafe { WaitForSingleObject(handle, 0) } {
                WAIT_OBJECT_0 => return false,
                WAIT_TIMEOUT => return true,
                // A failed wait (an unusable handle) falls through to a
                // fresh open below; the stale entry is dropped first.
                _ => {
                    self.handles.remove(&pid);
                }
            }
        }
        match unsafe { OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, false, pid) } {
            Ok(handle) => {
                self.handles.insert(pid, handle);
                true
            }
            Err(error) if error.code() == DEAD_PID_HRESULT => false,
            // An access failure (an elevated consumer, a protected
            // process) keeps the slot: the consumer's own attach timeout
            // releases it if the share never lands.
            Err(_) => true,
        }
    }

    /// Closes and drops every handle whose pid is no longer registered (a
    /// released slot must not pin its process object open).
    fn retain_registered(&mut self, registered: &HashSet<u32>) {
        self.handles.retain(|pid, handle| {
            let keep = registered.contains(pid);
            if !keep {
                unsafe {
                    CloseHandle(*handle).ok();
                }
            }
            keep
        });
    }
}

impl Drop for ConsumerLiveness {
    fn drop(&mut self) {
        for handle in self.handles.values() {
            unsafe {
                CloseHandle(*handle).ok();
            }
        }
    }
}

/// Per-consumer maintenance: prune dead processes (freeing their slots,
/// through the handle-based liveness table), duplicate the shared
/// textures into the requested ones, and drop the handles of pids that
/// no longer hold a slot. Non-supplied slots stay pending until the
/// shared pair exists (the first publish's pair creation precedes any
/// handle duplication).
pub(super) fn maintain(
    manifest: &Manifest,
    pair: Option<&SharedPair>,
    my_pid: u32,
    liveness: &mut ConsumerLiveness,
) {
    let mut registered = HashSet::new();
    for index in 0..CONSUMER_SLOTS {
        let slot = manifest.slot(index);
        let pid = slot.pid.load(Ordering::Acquire);
        if pid == 0 || pid == my_pid {
            continue;
        }
        if !liveness.is_alive(pid) {
            release_slot(slot);
            continue;
        }
        registered.insert(pid);
        if slot.state.load(Ordering::Acquire) == SLOT_REQUESTED {
            if let Some(pair) = pair {
                match duplicate_pair_process(pid, &pair.cells) {
                    Ok(handles) => {
                        slot.handles[0].store(handles[0], Ordering::Release);
                        slot.handles[1].store(handles[1], Ordering::Release);
                        slot.handles_epoch.store(
                            manifest.texture_epoch.load(Ordering::Acquire),
                            Ordering::Release,
                        );
                        slot.state.store(SLOT_SUPPLIED, Ordering::Release);
                    }
                    Err(error) => {
                        eprintln!("glass broker could not share with consumer pid {pid}: {error}");
                    }
                }
            }
        }
    }
    liveness.retain_registered(&registered);
}

/// Slot release for a dead consumer: state then pid, in that order (pid==0
/// is the readers' "free slot" marker).
fn release_slot(slot: &ConsumerSlot) {
    slot.state.store(SLOT_UNUSED, Ordering::Release);
    slot.active.store(0, Ordering::Release);
    slot.pid.store(0, Ordering::Release);
}

/// Duplicates one consumer's pair handles into its process: the target is
/// opened with the duplication right fresh per request (the liveness
/// table's cached handles carry SYNCHRONIZE instead).
fn duplicate_pair_process(pid: u32, cells: &[SharedCell; 2]) -> Result<[u64; 2], String> {
    let process = match unsafe {
        OpenProcess(
            PROCESS_DUP_HANDLE | PROCESS_QUERY_LIMITED_INFORMATION,
            false,
            pid,
        )
    } {
        Ok(handle) => handle,
        Err(error) => {
            return Err(format!(
                "Could not open the glass consumer process: {error}"
            ));
        }
    };
    let result = duplicate_pair(process, cells);
    unsafe {
        CloseHandle(process).ok();
    }
    result
}

/// Duplicates one process's pair handles for one slot.
fn duplicate_pair(process: HANDLE, cells: &[SharedCell; 2]) -> Result<[u64; 2], String> {
    let mut out = [0u64; 2];
    for (index, cell) in cells.iter().enumerate() {
        let mut dup = HANDLE(std::ptr::null_mut());
        unsafe {
            DuplicateHandle(
                GetCurrentProcess(),
                cell.handle,
                process,
                &mut dup,
                0,
                false,
                DUPLICATE_SAME_ACCESS,
            )
        }
        .map_err(|error| format!("Could not duplicate the glass handle: {error}"))?;
        out[index] = dup.0 as usize as u64;
    }
    Ok(out)
}
