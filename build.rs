fn main() {
    #[cfg(feature = "gpu")]
    if std::env::var("TARGET").as_deref() == Ok("aarch64-apple-darwin") {
        let source = "src/backend/gpu/metal_frame_presence.m";
        println!("cargo:rerun-if-changed={source}");
        cc::Build::new()
            .file(source)
            .flag("-fobjc-arc")
            .compile("subfast_metal_frame_presence");
        for framework in ["Foundation", "Metal", "MetalPerformanceShadersGraph"] {
            println!("cargo:rustc-link-lib=framework={framework}");
        }
    }
}
