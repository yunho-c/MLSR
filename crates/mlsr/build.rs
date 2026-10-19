//! Link only the explicitly prepared native QNN bridge on its supported target.
fn main() {
    println!("cargo:rerun-if-env-changed=MLSR_QNN_BRIDGE_DIR");
    if std::env::var_os("CARGO_FEATURE_QNN").is_some()
        && std::env::var("TARGET").as_deref() == Ok("aarch64-pc-windows-msvc")
    {
        let directory = std::env::var_os("MLSR_QNN_BRIDGE_DIR").expect(
            "qnn requires MLSR_QNN_BRIDGE_DIR; build native/qnn first (docs/qnn-runtime.md)",
        );
        let directory = std::path::PathBuf::from(directory);
        assert!(
            directory.join("mlsr_qnn_bridge.lib").is_file(),
            "missing QNN bridge import library"
        );
        println!("cargo:rustc-link-search=native={}", directory.display());
        println!("cargo:rustc-link-lib=dylib=mlsr_qnn_bridge");
        println!(
            "cargo:rerun-if-changed={}",
            directory.join("mlsr_qnn_bridge.lib").display()
        );
    }
}
