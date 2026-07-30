fn main() -> eframe::Result {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    let mut wgpu_options = eframe::WgpuConfiguration {
        surface: eframe::SurfaceConfig::LOW_LATENCY,
        ..Default::default()
    };
    if let eframe::egui_wgpu::WgpuSetup::CreateNew(setup) = &mut wgpu_options.wgpu_setup {
        setup.power_preference = eframe::wgpu::PowerPreference::LowPower;
        setup.instance_descriptor.backends = eframe::wgpu::Backends::METAL;
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
            .with_min_inner_size([900.0, 600.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Adam",
        options,
        Box::new(|creation| Ok(Box::new(adam_canvas::app::AdamApp::new(creation)))),
    )
}
