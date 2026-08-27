#![cfg_attr(windows, windows_subsystem = "windows")]

#[cfg(target_os = "macos")]
fn main() {
    use cef::{args::Args, *};

    let args = Args::new();
    let executable = std::env::current_exe().expect("CEF helper executable path");
    let framework = executable
        .parent()
        .expect("CEF helper executable parent")
        .join("../../../Chromium Embedded Framework.framework/Chromium Embedded Framework");
    assert!(
        framework.is_file(),
        "CEF helper could not find Chromium at {}",
        framework.display()
    );

    let mut sandbox = cef::sandbox::Sandbox::new();
    sandbox.initialize(args.as_main_args());

    let loader = cef::library_loader::LibraryLoader::new(&executable, true);
    assert!(loader.load(), "CEF helper could not load Chromium");
    let _ = api_hash(sys::CEF_API_VERSION_LAST, 0);
    let code = execute_process(
        Some(args.as_main_args()),
        None::<&mut App>,
        std::ptr::null_mut(),
    );
    assert!(
        code >= 0,
        "CEF helper was launched without a subprocess type"
    );
    std::process::exit(code);
}

#[cfg(not(target_os = "macos"))]
fn main() {}
