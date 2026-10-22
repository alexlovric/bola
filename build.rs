/// Configure optional native libraries for external BLAS/LAPACK comparisons.
fn main() {
    // External libraries are only needed by the BLAS/LAPACK comparisons.
    // Accept Cargo's link syntax, e.g. BOLA_BLAS_LIB=static=openblas.
    if std::env::var_os("CARGO_FEATURE_EXTERNAL_BLAS").is_some() {
        if let Ok(directory) = std::env::var("BOLA_BLAS_DIR") {
            println!("cargo:rustc-link-search=native={directory}");
        }
        // LAPACK depends on BLAS, so preserve this order for static linking
        // and prefer a separately requested LAPACK over bundled symbols.
        if let Ok(lapack) = std::env::var("BOLA_LAPACK_LIB") {
            println!("cargo:rustc-link-lib={lapack}");
        }
        let blas = std::env::var("BOLA_BLAS_LIB").unwrap_or_else(|_| "openblas".into());
        println!("cargo:rustc-link-lib={blas}");
    }
    println!("cargo:rerun-if-env-changed=BOLA_BLAS_LIB");
    println!("cargo:rerun-if-env-changed=BOLA_LAPACK_LIB");
    println!("cargo:rerun-if-env-changed=BOLA_BLAS_DIR");

    println!("cargo:rerun-if-changed=build.rs");
}
