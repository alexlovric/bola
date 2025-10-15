/// This build script is responsible for configuring the crate's build process for the local machine.
///
/// It checks for the following features:
/// - AVX2: if the host machine supports AVX2 instructions
/// - FMA: if the host machine supports FMA instructions
/// - NEON: if the host machine is an AArch64 (Apple Silicon, ARMv8+) processor
///
/// If any of these features are detected, the corresponding feature flag is enabled.
///
/// For example, if the host machine supports AVX2, the `has_avx2` feature flag is enabled, and code
/// inside `#[cfg(has_avx2)]` blocks is compiled.
///
/// If cross-compiling, the feature flags are not chang ed.
///
/// You can test this script by running `cargo build --verbose` in the terminal.
fn main() {
    // for testing
    // println!("cargo:rustc-link-lib=static=openblas");
    // println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu"); // or your OpenBLAS path

    println!("cargo:rustc-check-cfg=cfg(has_fma)");
    println!("cargo:rustc-check-cfg=cfg(has_avx2)");
    println!("cargo:rustc-check-cfg=cfg(has_neon)");

    let target = std::env::var("TARGET").unwrap();
    let host = std::env::var("HOST").unwrap();

    // Detect FMA and AVX2 only if compiling for the host machine (not cross-compiling)
    if target == host && target.contains("x86_64") {
        if is_x86_feature_detected!("fma") {
            println!("cargo:rustc-cfg=has_fma");
        }
        if is_x86_feature_detected!("avx2") {
            println!("cargo:rustc-cfg=has_avx2");
        }
    }

    // Unconditionally enable NEON if targeting AArch64 (Apple Silicon, ARMv8+)
    if target.contains("aarch64") {
        println!("cargo:rustc-cfg=has_neon");
    }

    println!("cargo:rerun-if-changed=build.rs");
}
