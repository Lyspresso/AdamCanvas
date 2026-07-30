use bytemuck::{Pod, Zeroable};
use eframe::egui_wgpu::{
    self,
    wgpu::{self, util::DeviceExt as _},
};
use egui::{PaintCallback, PaintCallbackInfo, Rect};
use std::num::NonZeroU64;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DotsParams {
    pub style: u32,
    pub speed: f32,
    pub brightness: f32,
    pub tint: u32,
    pub background: u32,
    pub dot_size: f32,
    pub grid_density: f32,
    pub pattern_scale: f32,
    pub vignette: f32,
    pub horizon: f32,
    pub amplitude: f32,
    pub depth_fade: f32,
}

impl DotsParams {
    /// The Flow preset captured in the selected MetalForge Dots reference.
    pub const fn chrome_flow(tint: u32, background: u32) -> Self {
        Self {
            style: 4,
            speed: 0.65,
            brightness: 1.80,
            tint,
            background,
            dot_size: 0.70,
            grid_density: 1.57,
            pattern_scale: 0.75,
            vignette: 1.00,
            horizon: -0.45,
            amplitude: 1.00,
            depth_fade: 1.00,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChromeRects {
    pub toolbar: Rect,
    pub sidebar: Rect,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
struct DotsUniform {
    size: [f32; 2],
    time: f32,
    style: f32,
    speed: f32,
    brightness: f32,
    _padding0: [f32; 2],
    tint: [f32; 4],
    background: [f32; 4],
    dot_size: f32,
    grid_density: f32,
    pattern_scale: f32,
    vignette: f32,
    horizon: f32,
    amplitude: f32,
    depth_fade: f32,
    _padding1: f32,
}

impl DotsUniform {
    fn new(seconds: f32, screen: &egui_wgpu::ScreenDescriptor, tint: u32, background: u32) -> Self {
        let params = DotsParams::chrome_flow(tint, background);
        Self {
            size: [
                screen.size_in_pixels[0] as f32,
                screen.size_in_pixels[1] as f32,
            ],
            time: seconds,
            style: params.style as f32,
            speed: params.speed,
            brightness: params.brightness,
            _padding0: [0.0; 2],
            tint: rgbaf(params.tint),
            background: rgbaf(params.background),
            dot_size: params.dot_size,
            grid_density: params.grid_density,
            pattern_scale: params.pattern_scale,
            vignette: params.vignette,
            horizon: params.horizon,
            amplitude: params.amplitude,
            depth_fade: params.depth_fade,
            _padding1: 0.0,
        }
    }
}

struct DotsRenderResources {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
}

impl DotsRenderResources {
    fn new(render_state: &egui_wgpu::RenderState) -> Self {
        let device = &render_state.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("adam-dots-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("dots.wgsl").into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("adam-dots-bind-group-layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(std::mem::size_of::<DotsUniform>() as u64),
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("adam-dots-pipeline-layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("adam-dots-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: render_state.target_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("adam-dots-uniform"),
            contents: bytemuck::bytes_of(&DotsUniform::zeroed()),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("adam-dots-bind-group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        Self {
            pipeline,
            bind_group,
            uniform_buffer,
        }
    }

    fn prepare(
        &self,
        queue: &wgpu::Queue,
        screen: &egui_wgpu::ScreenDescriptor,
        seconds: f32,
        tint: u32,
        background: u32,
    ) {
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&DotsUniform::new(seconds, screen, tint, background)),
        );
    }

    fn paint(
        &self,
        info: PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'_>,
        rects: ChromeRects,
    ) {
        let [screen_width, screen_height] = info.screen_size_px;
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.set_viewport(
            0.0,
            0.0,
            screen_width as f32,
            screen_height as f32,
            0.0,
            1.0,
        );

        for scissor in chrome_scissors(rects, info.pixels_per_point, [screen_width, screen_height])
            .into_iter()
            .flatten()
        {
            render_pass.set_scissor_rect(scissor.x, scissor.y, scissor.width, scissor.height);
            render_pass.draw(0..3, 0..1);
        }
    }
}

#[derive(Clone, Copy)]
struct DotsCallback {
    rects: ChromeRects,
    seconds: f32,
    tint: u32,
    background: u32,
}

impl egui_wgpu::CallbackTrait for DotsCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(resources) = resources.get::<DotsRenderResources>() {
            resources.prepare(queue, screen, self.seconds, self.tint, self.background);
        }
        Vec::new()
    }

    fn paint(
        &self,
        info: PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        if let Some(resources) = resources.get::<DotsRenderResources>() {
            resources.paint(info, render_pass, self.rects);
        }
    }
}

pub fn install(creation: &eframe::CreationContext<'_>) -> bool {
    let Some(render_state) = creation.wgpu_render_state.as_ref() else {
        return false;
    };
    let resources = DotsRenderResources::new(render_state);
    render_state
        .renderer
        .write()
        .callback_resources
        .insert(resources);
    true
}

pub fn paint_callback(
    rect: Rect,
    rects: ChromeRects,
    seconds: f32,
    tint: u32,
    background: u32,
) -> PaintCallback {
    egui_wgpu::Callback::new_paint_callback(
        rect,
        DotsCallback {
            rects,
            seconds,
            tint,
            background,
        },
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ScissorRect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

fn chrome_scissors(
    rects: ChromeRects,
    pixels_per_point: f32,
    screen: [u32; 2],
) -> [Option<ScissorRect>; 2] {
    let toolbar = rect_to_scissor(rects.toolbar, pixels_per_point, screen);
    let mut sidebar = rect_to_scissor(rects.sidebar, pixels_per_point, screen);
    if let (Some(toolbar), Some(sidebar)) = (toolbar, sidebar.as_mut()) {
        let seam = toolbar.y.saturating_add(toolbar.height).min(screen[1]);
        let sidebar_bottom = sidebar.y.saturating_add(sidebar.height).min(screen[1]);
        sidebar.y = seam;
        sidebar.height = sidebar_bottom.saturating_sub(seam);
        if sidebar.height == 0 {
            return [Some(toolbar), None];
        }
    }
    [toolbar, sidebar]
}

fn rect_to_scissor(rect: Rect, pixels_per_point: f32, screen: [u32; 2]) -> Option<ScissorRect> {
    if !rect.is_positive() || !pixels_per_point.is_finite() || pixels_per_point <= 0.0 {
        return None;
    }
    let left = (rect.left() * pixels_per_point)
        .floor()
        .clamp(0.0, screen[0] as f32) as u32;
    let top = (rect.top() * pixels_per_point)
        .floor()
        .clamp(0.0, screen[1] as f32) as u32;
    let right = (rect.right() * pixels_per_point)
        .ceil()
        .clamp(0.0, screen[0] as f32) as u32;
    let bottom = (rect.bottom() * pixels_per_point)
        .ceil()
        .clamp(0.0, screen[1] as f32) as u32;
    (right > left && bottom > top).then_some(ScissorRect {
        x: left,
        y: top,
        width: right - left,
        height: bottom - top,
    })
}

fn rgbaf(hex: u32) -> [f32; 4] {
    [
        ((hex >> 16) & 0xFF) as f32 / 255.0,
        ((hex >> 8) & 0xFF) as f32 / 255.0,
        (hex & 0xFF) as f32 / 255.0,
        1.0,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captured_flow_preset_is_exact_and_theme_aware() {
        let dark = DotsParams::chrome_flow(0xFFFFFF, 0x000000);
        let light = DotsParams::chrome_flow(0x000000, 0xFFFFFF);

        assert_eq!(dark.style, 4);
        assert_eq!(dark.speed, 0.65);
        assert_eq!(dark.brightness, 1.80);
        assert_eq!(dark.tint, 0xFFFFFF);
        assert_eq!(dark.background, 0x000000);
        assert_eq!(dark.dot_size, 0.70);
        assert_eq!(dark.grid_density, 1.57);
        assert_eq!(dark.pattern_scale, 0.75);
        assert_eq!(dark.vignette, 1.00);
        assert_eq!(dark.horizon, -0.45);
        assert_eq!(dark.amplitude, 1.00);
        assert_eq!(dark.depth_fade, 1.00);

        assert_eq!(light.tint, 0x000000);
        assert_eq!(light.background, 0xFFFFFF);
        assert_eq!(
            DotsParams {
                tint: dark.tint,
                background: dark.background,
                ..light
            },
            dark
        );
    }

    #[test]
    fn uniform_layout_matches_metalforge_wgsl() {
        assert_eq!(std::mem::size_of::<DotsUniform>(), 96);
        assert_eq!(std::mem::size_of::<DotsUniform>() % 16, 0);
    }

    #[test]
    fn uniform_palette_follows_the_resolved_theme() {
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [1920, 1080],
            pixels_per_point: 2.0,
        };
        let dark = DotsUniform::new(2.5, &screen, 0xFFFFFF, 0x000000);
        let light = DotsUniform::new(2.5, &screen, 0x000000, 0xFFFFFF);
        let beach = DotsUniform::new(2.5, &screen, 0x1B2A25, 0xFFEEAD);

        assert_eq!(dark.tint, rgbaf(0xFFFFFF));
        assert_eq!(dark.background, rgbaf(0x000000));
        assert_eq!(light.tint, rgbaf(0x000000));
        assert_eq!(light.background, rgbaf(0xFFFFFF));
        assert_eq!(
            DotsUniform {
                tint: dark.tint,
                background: dark.background,
                ..light
            },
            dark
        );
        assert_eq!(beach.tint, rgbaf(0x1B2A25));
        assert_eq!(beach.background, rgbaf(0xFFEEAD));
    }

    #[test]
    fn one_scissor_union_covers_only_the_connected_chrome() {
        let rects = ChromeRects {
            toolbar: Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1000.0, 58.0)),
            sidebar: Rect::from_min_max(egui::pos2(0.0, 58.0), egui::pos2(224.0, 800.0)),
        };
        let scissors = chrome_scissors(rects, 2.0, [2000, 1600]);
        assert_eq!(
            scissors,
            [
                Some(ScissorRect {
                    x: 0,
                    y: 0,
                    width: 2000,
                    height: 116,
                }),
                Some(ScissorRect {
                    x: 0,
                    y: 116,
                    width: 448,
                    height: 1484,
                }),
            ]
        );
    }

    #[test]
    fn fractional_scale_uses_one_shared_toolbar_sidebar_seam() {
        let rects = ChromeRects {
            toolbar: Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1000.0, 58.0)),
            sidebar: Rect::from_min_max(egui::pos2(0.0, 58.0), egui::pos2(224.0, 800.0)),
        };
        let [Some(toolbar), Some(sidebar)] = chrome_scissors(rects, 1.25, [1250, 1000]) else {
            panic!("expected both chrome scissors");
        };
        assert_eq!(toolbar.y + toolbar.height, sidebar.y);
        assert_eq!(toolbar.height, 73);
        assert_eq!(sidebar.height, 927);
    }
}
