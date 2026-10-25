//! The glass broker process: one capture + blur engine for every glass
//! app in the session (the library's clients spawn it beside their own
//! executable on demand). See the `liquid-glass::broker` docs for the
//! full protocol.

fn main() {
    // The library configures no console output; the broker logs failure
    // reasons to stderr (visible when launched manually) and exits with
    // a non-zero code on startup failure.
    let code = liquid_glass::broker::run();
    std::process::exit(code);
}
