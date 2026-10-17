//! Linker settings required to relocate the shared library into app bundles.

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        // Dart and other packagers rewrite Mach-O load commands when bundling.
        println!("cargo::rustc-link-arg-cdylib=-Wl,-headerpad_max_install_names");
        println!("cargo::rustc-link-arg-cdylib=-Wl,-install_name,@rpath/libmlsr_ffi.dylib");
    }
}
