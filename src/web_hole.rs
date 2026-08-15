//! The GPU half of the two-layer live web tile: it punches transparent holes
//! in the otherwise-opaque egui surface, exactly where each live page sits.
//!
//! The WKWebView is composited *below* the egui Metal layer (see
//! [`crate::webview_host`]). Adam paints an opaque "desk" over the whole
//! window, so by default nothing behind the surface is visible. This callback
//! runs in the middle of the canvas paint order — after the desk and tiles,
//! before the quick bar, minimap, and every overlay Area — and writes fully
//! transparent pixels (premultiplied `(0,0,0,0)`, blend `REPLACE`) inside each
//! page's clip rectangle. That is the only place the desktop layer beneath the
//! surface — i.e. the web view — is allowed to show through. Chrome drawn after
//! the punch composites back on top and stays opaque.
//!
//! This mirrors the [`crate::dots`] callback shape: a self-contained pipeline
//! stashed in the wgpu callback resources at startup, invoked per frame.

use eframe::egui_wgpu::{
    self,
    wgpu::{self},
};
use egui::{PaintCallback, PaintCallbackInfo, Rect};

/// A fullscreen triangle whose fragment shader always emits transparent black.
/// With `BlendState::REPLACE` this overwrites the target's colour AND alpha to
/// zero inside the active scissor rectangle — a clean cut, not a blend.
const HOLE_WGSL: &str = "\
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  let x = f32((vi << 1u) & 2u);
  let y = f32(vi & 2u);
  return vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
  return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}
";

struct WebHoleResources {
    pipeline: wgpu::RenderPipeline,
}

impl WebHoleResources {
    fn new(render_state: &egui_wgpu::RenderState) -> Self {
        let device = &render_state.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("adam-web-hole-shader"),
            source: wgpu::ShaderSource::Wgsl(HOLE_WGSL.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("adam-web-hole-pipeline-layout"),
            bind_group_layouts: &[],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("adam-web-hole-pipeline"),
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
                    // REPLACE writes src (0,0,0,0) straight through, clearing the
                    // alpha the opaque desk laid down so the surface is see-through
                    // exactly here and nowhere else.
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
        Self { pipeline }
    }

    fn paint(&self, info: PaintCallbackInfo, render_pass: &mut wgpu::RenderPass<'_>, holes: &[Rect]) {
        let [screen_width, screen_height] = info.screen_size_px;
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_viewport(
            0.0,
            0.0,
            screen_width as f32,
            screen_height as f32,
            0.0,
            1.0,
        );
        for hole in holes {
            if let Some(scissor) =
                rect_to_scissor(*hole, info.pixels_per_point, [screen_width, screen_height])
            {
                render_pass.set_scissor_rect(scissor.x, scissor.y, scissor.width, scissor.height);
                render_pass.draw(0..3, 0..1);
            }
        }
    }
}

struct WebHoleCallback {
    holes: Vec<Rect>,
}

impl egui_wgpu::CallbackTrait for WebHoleCallback {
    fn paint(
        &self,
        info: PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        if let Some(resources) = resources.get::<WebHoleResources>() {
            resources.paint(info, render_pass, &self.holes);
        }
    }
}

/// Registers the hole-punch pipeline in the wgpu callback resources. Returns
/// false when there is no wgpu render state (a non-wgpu backend), in which case
/// callers simply never punch — the live-web feature is macOS/wgpu only anyway.
pub fn install(creation: &eframe::CreationContext<'_>) -> bool {
    let Some(render_state) = creation.wgpu_render_state.as_ref() else {
        return false;
    };
    let resources = WebHoleResources::new(render_state);
    render_state
        .renderer
        .write()
        .callback_resources
        .insert(resources);
    true
}

/// One paint shape that clears `holes` (screen-point rects) to transparent.
/// `rect` bounds the callback for egui; pass the canvas view.
pub fn paint_callback(rect: Rect, holes: Vec<Rect>) -> PaintCallback {
    egui_wgpu::Callback::new_paint_callback(rect, WebHoleCallback { holes })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ScissorRect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The shader is otherwise only compiled when a GPU surface is created, so
    /// a malformed edit would ship as a hole that never opens (or a validation
    /// abort at runtime) rather than a build failure. naga rides in via wgpu.
    #[test]
    fn hole_shader_parses_and_validates() {
        let module = match wgpu::naga::front::wgsl::parse_str(HOLE_WGSL) {
            Ok(module) => module,
            Err(error) => panic!(
                "web_hole shader failed to parse:\n{}",
                error.emit_to_string(HOLE_WGSL)
            ),
        };
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("web_hole shader failed validation");
    }

    #[test]
    fn a_hole_maps_to_a_scaled_scissor() {
        let scissor = rect_to_scissor(
            Rect::from_min_size(egui::pos2(100.0, 50.0), egui::vec2(200.0, 120.0)),
            2.0,
            [2000, 1600],
        )
        .expect("positive rect yields a scissor");
        assert_eq!(scissor, ScissorRect { x: 200, y: 100, width: 400, height: 240 });
    }

    #[test]
    fn a_degenerate_or_offscreen_hole_yields_nothing() {
        assert_eq!(
            rect_to_scissor(
                Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(0.0, 100.0)),
                2.0,
                [2000, 1600],
            ),
            None
        );
    }

    #[test]
    fn a_hole_is_clamped_to_the_surface() {
        let scissor = rect_to_scissor(
            Rect::from_min_size(egui::pos2(-40.0, -40.0), egui::vec2(80.0, 80.0)),
            1.0,
            [100, 100],
        )
        .expect("overlap is still positive");
        assert_eq!(scissor.x, 0);
        assert_eq!(scissor.y, 0);
        assert_eq!(scissor.width, 40);
        assert_eq!(scissor.height, 40);
    }
}
