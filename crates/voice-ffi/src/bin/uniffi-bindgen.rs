// UniFFI binding generator entrypoint (proc-macro `--library` mode).
// Invoked by scripts/regen-ffi.sh to emit Swift bindings from the built library.
fn main() {
    uniffi::uniffi_bindgen_main()
}
