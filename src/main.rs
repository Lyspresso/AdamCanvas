#![cfg_attr(windows, windows_subsystem = "windows")]

fn main() -> eframe::Result {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    let mut wgpu_options = eframe::WgpuConfiguration {
        surface: eframe::SurfaceConfig::LOW_LATENCY,
        ..Default::default()
    };
    if let eframe::egui_wgpu::WgpuSetup::CreateNew(setup) = &mut wgpu_options.wgpu_setup {
        setup.power_preference = eframe::wgpu::PowerPreference::LowPower;
        // PRIMARY resolves to Metal on macOS and DX12/Vulkan on Windows;
        // pinning Metal here left Windows with zero adapters.
        setup.instance_descriptor.backends = eframe::wgpu::Backends::PRIMARY;
    }

    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        multisampling: 0,
        depth_buffer: 0,
        stencil_buffer: 0,
        dithering: true,
        wgpu_options,
        viewport: egui::ViewportBuilder::default()
            .with_title("Adam")
            .with_inner_size([1380.0, 860.0])
            .with_min_inner_size([900.0, 600.0])
            // Two-layer live web tiles: the window surface must be able to go
            // transparent so Adam can punch a hole exactly at each live page's
            // rect and let the WKWebView (composited below the egui Metal layer)
            // show through. Everywhere else Adam paints an opaque desk, so the
            // desktop is never visible. See `webview_host` / `web_hole`.
            .with_transparent(true),
        ..Default::default()
    };

    eframe::run_native(
        "Adam",
        options,
        Box::new(|creation| Ok(Box::new(adam_canvas::app::AdamApp::new(creation)))),
    )
}
