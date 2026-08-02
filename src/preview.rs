use crate::persistence::AppPaths;
use crossbeam_channel::{Receiver, Sender, TrySendError, bounded};
use egui::{ColorImage, Context, TextureHandle, TextureOptions};
use image::{DynamicImage, GenericImageView, ImageDecoder, ImageReader, metadata::Orientation};
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant, SystemTime},
};
use uuid::Uuid;

/// Photo previews are quantized so a continuously changing canvas zoom does
/// not create a new decode request every frame.
const IMAGE_PREVIEW_EDGES: [u32; 5] = [256, 512, 1_024, 2_048, 4_096];
const IMAGE_BASE_EDGE: u32 = 256;
const DETAIL_OVERSCAN: f32 = 1.05;
// v3 is produced by the color-managed, orientation-aware Image I/O pipeline.
// Older tiers may seed the tiny progressive base, but must never become final
// detail because v1 lacked orientation and v2 predated native color handling.
const PHOTO_CACHE_VERSION: &str = "photo-v3";
const ORIENTED_LEGACY_PHOTO_CACHE_VERSION: &str = "photo-v2";
const UNORIENTED_LEGACY_PHOTO_CACHE_VERSION: &str = "photo-v1";
const QUICK_LOOK_EDGE: u32 = 512;
const GPU_CACHE_BUDGET: usize = 128 * 1024 * 1024;
const UNUSED_TTL: Duration = Duration::from_secs(45);
const FULL_QUEUE_RETRY: Duration = Duration::from_millis(125);
const FAILURE_RETRY_MAX: Duration = Duration::from_secs(30);
const MAX_PREVIEW_FAILURES: u8 = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PreviewMode {
    Image,
    QuickLook,
}

#[derive(Debug)]
struct PreviewJob {
    id: Uuid,
    request_id: u64,
    source: PathBuf,
    cache_directory: PathBuf,
    mode: PreviewMode,
    edge: u32,
}

#[derive(Debug)]
struct PreviewPixels {
    id: Uuid,
    request_id: u64,
    edge: u32,
    mode: PreviewMode,
    size: [usize; 2],
    intrinsic_size: Option<[u32; 2]>,
    rgba: Vec<u8>,
}

#[derive(Debug)]
enum PreviewResult {
    Ready(PreviewPixels),
    Failed { id: Uuid, request_id: u64 },
}

#[derive(Clone, Copy, Debug)]
struct PendingRequest {
    id: u64,
    edge: u32,
    mode: PreviewMode,
    estimated_bytes: usize,
}

#[derive(Default)]
struct Entry {
    /// Small progressive texture that is never evicted while the tile is
    /// visible. This avoids a detail-budget eviction turning into a pink card.
    base_texture: Option<TextureHandle>,
    base_edge: u32,
    base_mode: Option<PreviewMode>,
    base_bytes: usize,
    /// Optional Retina detail. This is the only layer sacrificed under GPU
    /// pressure, so a dense page stays complete and interactive.
    detail_texture: Option<TextureHandle>,
    detail_edge: u32,
    detail_bytes: usize,
    intrinsic_size: Option<[u32; 2]>,
    pending: Option<PendingRequest>,
    failure_count: u8,
    last_used: Option<Instant>,
    last_visible_frame: Option<u64>,
    desired_mode: Option<PreviewMode>,
    desired_edge: u32,
    retry_after: Option<Instant>,
    /// The source file changed on disk after this texture was decoded. The
    /// old picture keeps showing — a placeholder flash would be worse than a
    /// briefly outdated image — while a fresh base decode is scheduled.
    stale: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct WorkerDemand {
    request_id: u64,
}

pub struct PreviewCache {
    jobs: Sender<PreviewJob>,
    results: Receiver<PreviewResult>,
    entries: HashMap<Uuid, Entry>,
    worker_demands: Arc<Mutex<HashMap<Uuid, WorkerDemand>>>,
    paths: AppPaths,
    context: Context,
    next_request_id: u64,
    last_eviction: Instant,
}

impl PreviewCache {
    pub fn start(paths: AppPaths, context: Context) -> Self {
        // One slot per expected dense-page tile lets every base preview enter
        // the progressive queue once. A shared demand map below makes old
        // entries cheap to skip after a pan instead of decoding stale files.
        let (job_sender, job_receiver) = bounded::<PreviewJob>(128);
        // A completed 4K RGBA preview is about 64 MiB. Keep at most two
        // waiting for the UI so a paused/minimized window cannot accumulate a
        // gigabyte-scale decoded backlog.
        let (result_sender, result_receiver) = bounded::<PreviewResult>(2);
        let worker_demands = Arc::new(Mutex::new(HashMap::new()));
        let worker_demand_reader = Arc::clone(&worker_demands);
        let worker_context = context.clone();
        thread::Builder::new()
            .name("adam-preview".into())
            .spawn(move || {
                preview_loop(
                    job_receiver,
                    result_sender,
                    worker_context,
                    worker_demand_reader,
                )
            })
            .expect("failed to start preview worker");

        Self {
            jobs: job_sender,
            results: result_receiver,
            entries: HashMap::new(),
            worker_demands,
            paths,
            context,
            next_request_id: 1,
            last_eviction: Instant::now(),
        }
    }

    /// Uploads completed previews and returns newly discovered source
    /// dimensions. Decode and resize work stays on the preview worker; only
    /// the final GPU upload happens on the UI thread.
    pub fn poll(&mut self, context: &Context) -> Vec<(Uuid, [u32; 2])> {
        let mut received = false;
        let mut resolved_sizes = Vec::new();
        while let Ok(result) = self.results.try_recv() {
            match result {
                PreviewResult::Ready(result) => {
                    self.clear_worker_demand(result.id, result.request_id);
                    let Some(entry) = self.entries.get_mut(&result.id) else {
                        // The tile was invalidated or deleted while its decode
                        // was in flight.
                        continue;
                    };
                    if entry.pending.is_none_or(|pending| {
                        pending.id != result.request_id
                            || pending.edge != result.edge
                            || pending.mode != result.mode
                    }) {
                        continue;
                    }

                    if let Some(intrinsic_size) = result.intrinsic_size
                        && entry.intrinsic_size != Some(intrinsic_size)
                    {
                        entry.intrinsic_size = Some(intrinsic_size);
                        resolved_sizes.push((result.id, intrinsic_size));
                    }

                    let is_detail =
                        result.mode == PreviewMode::Image && result.edge > IMAGE_BASE_EDGE;
                    let resident_edge = if is_detail {
                        entry.detail_edge
                    } else {
                        entry.base_edge
                    };
                    // Never replace a sharper resident layer with a stale or
                    // smaller result. Base and detail are separate so detail
                    // eviction can never leave a visible photo blank.
                    if resident_edge <= result.edge {
                        let color_image =
                            ColorImage::from_rgba_unmultiplied(result.size, &result.rgba);
                        let texture = context.load_texture(
                            format!(
                                "preview-{}-{}-{}",
                                result.id, result.edge, result.request_id
                            ),
                            color_image,
                            TextureOptions::LINEAR,
                        );
                        if is_detail {
                            entry.detail_bytes = result.rgba.len();
                            entry.detail_texture = Some(texture);
                            entry.detail_edge = result.edge;
                        } else {
                            entry.base_bytes = result.rgba.len();
                            entry.base_texture = Some(texture);
                            entry.base_edge = result.edge;
                            entry.base_mode = Some(result.mode);
                            if entry.stale {
                                // The replacement arrived. The old Retina
                                // detail is of the old content and must not
                                // keep covering the new base; it re-decodes
                                // at the desired edge as usual.
                                entry.stale = false;
                                entry.detail_texture = None;
                                entry.detail_edge = 0;
                                entry.detail_bytes = 0;
                            }
                        }
                    }
                    entry.pending = None;
                    entry.failure_count = 0;
                    entry.retry_after = None;
                }
                PreviewResult::Failed { id, request_id } => {
                    self.clear_worker_demand(id, request_id);
                    let Some(entry) = self.entries.get_mut(&id) else {
                        continue;
                    };
                    if entry
                        .pending
                        .is_some_and(|pending| pending.id == request_id)
                    {
                        entry.pending = None;
                        entry.failure_count = entry.failure_count.saturating_add(1);
                        if let Some(delay) = retry_delay_after_failure(entry.failure_count) {
                            entry.retry_after = Some(Instant::now() + delay);
                            // A transient decode or Quick Look failure must
                            // not stay a placeholder until the next retry.
                            context.request_repaint_after(delay);
                        } else {
                            // Corrupt/unsupported files must not wake
                            // qlmanage forever while Adam sits idle.
                            entry.retry_after = None;
                        }
                    }
                }
            }
            received = true;
        }
        if received {
            context.request_repaint();
        }

        let over_budget = self.resident_bytes() > GPU_CACHE_BUDGET;
        if over_budget || self.last_eviction.elapsed() >= Duration::from_secs(5) {
            self.evict_unused(context.cumulative_frame_nr());
            self.last_eviction = Instant::now();
        }
        resolved_sizes
    }

    /// Returns the best resident photo texture and, when necessary, queues a
    /// background decode sized for the photo's projected *physical* pixels.
    ///
    /// `projected_pixels` must already include the window's Retina scale.
    pub fn image_texture(
        &mut self,
        id: Uuid,
        source: &Path,
        projected_pixels: [f32; 2],
    ) -> Option<&TextureHandle> {
        let edge = quantized_image_edge(projected_pixels);
        self.request_texture(id, source, PreviewMode::Image, edge)
    }

    /// Keeps the established macOS Quick Look behavior for documents, video,
    /// and other non-image files. These do not need zoom-dependent photo
    /// resolution.
    pub fn quick_look_texture(&mut self, id: Uuid, source: &Path) -> Option<&TextureHandle> {
        self.request_texture(id, source, PreviewMode::QuickLook, QUICK_LOOK_EDGE)
    }

    fn request_texture(
        &mut self,
        id: Uuid,
        source: &Path,
        mode: PreviewMode,
        desired_edge: u32,
    ) -> Option<&TextureHandle> {
        let now = Instant::now();
        let frame = self.context.cumulative_frame_nr();
        let cache_directory = self.paths.thumbnail_dir(id);
        let mut cancelled_request = None;
        {
            let entry = self.entries.entry(id).or_default();
            let mode_changed = entry
                .base_mode
                .or(entry.desired_mode)
                .is_some_and(|resident_mode| resident_mode != mode)
                || (mode != PreviewMode::Image && entry.detail_texture.is_some());
            if mode_changed {
                cancelled_request = entry.pending.map(|pending| pending.id);
                *entry = Entry {
                    last_used: Some(now),
                    last_visible_frame: Some(frame),
                    desired_mode: Some(mode),
                    desired_edge,
                    ..Default::default()
                };
            } else {
                entry.last_used = Some(now);
                entry.last_visible_frame = Some(frame);
                entry.desired_mode = Some(mode);
                entry.desired_edge = desired_edge;
            }
        }
        if let Some(request_id) = cancelled_request {
            self.clear_worker_demand(id, request_id);
        }

        let target_edge = {
            let entry = self
                .entries
                .get(&id)
                .expect("preview entry must exist after visibility update");
            let effective_desired_edge = entry
                .intrinsic_size
                .map(|[width, height]| desired_edge.min(width.max(height)))
                .unwrap_or(desired_edge);
            if entry.stale || entry.base_texture.is_none() || entry.base_mode != Some(mode) {
                Some(match mode {
                    PreviewMode::Image => IMAGE_BASE_EDGE,
                    PreviewMode::QuickLook => QUICK_LOOK_EDGE,
                })
            } else if mode == PreviewMode::Image
                && effective_desired_edge > IMAGE_BASE_EDGE
                && entry.detail_edge < effective_desired_edge
            {
                Some(effective_desired_edge)
            } else {
                None
            }
        };

        let pending = self.entries.get(&id).and_then(|entry| entry.pending);
        if pending.is_some_and(|pending| Some(pending.edge) != target_edge || pending.mode != mode)
        {
            let pending = pending.expect("checked pending request");
            self.clear_worker_demand(id, pending.id);
            if let Some(entry) = self.entries.get_mut(&id) {
                entry.pending = None;
            }
        }

        let should_request = target_edge.is_some_and(|target_edge| {
            let entry = self
                .entries
                .get(&id)
                .expect("preview entry must exist before request");
            entry
                .pending
                .is_none_or(|pending| pending.edge != target_edge || pending.mode != mode)
                && entry.retry_after.is_none_or(|retry| now >= retry)
                && entry.failure_count < MAX_PREVIEW_FAILURES
                && source.exists()
        });

        if should_request {
            let edge = target_edge.expect("request target checked above");
            let estimated_bytes = self
                .entries
                .get(&id)
                .and_then(|entry| entry.intrinsic_size)
                .map_or_else(
                    || square_rgba_bytes(edge),
                    |dimensions| estimated_image_bytes(edge, dimensions),
                );
            let is_detail = mode == PreviewMode::Image && edge > IMAGE_BASE_EDGE;
            if is_detail && !self.can_admit_detail(id, estimated_bytes, frame) {
                return self.best_texture(id);
            }

            let request_id = self.next_request_id;
            self.next_request_id = self.next_request_id.wrapping_add(1).max(1);
            let job = PreviewJob {
                id,
                request_id,
                source: source.to_owned(),
                cache_directory,
                mode,
                edge,
            };
            self.set_worker_demand(id, request_id);
            let send_result = self.jobs.try_send(job);
            match send_result {
                Ok(()) => {
                    let entry = self
                        .entries
                        .get_mut(&id)
                        .expect("preview entry must exist after request preparation");
                    entry.pending = Some(PendingRequest {
                        id: request_id,
                        edge,
                        mode,
                        estimated_bytes,
                    });
                    entry.retry_after = None;
                }
                Err(TrySendError::Full(_)) => {
                    self.clear_worker_demand(id, request_id);
                    // Retry at a modest cadence instead of trying on every
                    // animation frame while the bounded worker is saturated.
                    let entry = self
                        .entries
                        .get_mut(&id)
                        .expect("preview entry must exist after request preparation");
                    entry.retry_after = Some(now + FULL_QUEUE_RETRY);
                    self.context.request_repaint_after(FULL_QUEUE_RETRY);
                }
                Err(TrySendError::Disconnected(_)) => {
                    self.clear_worker_demand(id, request_id);
                    let entry = self
                        .entries
                        .get_mut(&id)
                        .expect("preview entry must exist after request preparation");
                    entry.failure_count = entry.failure_count.saturating_add(1);
                    if let Some(delay) = retry_delay_after_failure(entry.failure_count) {
                        entry.retry_after = Some(now + delay);
                        self.context.request_repaint_after(delay);
                    } else {
                        entry.retry_after = None;
                    }
                }
            }
        }

        self.best_texture(id)
    }

    fn best_texture(&self, id: Uuid) -> Option<&TextureHandle> {
        let entry = self.entries.get(&id)?;
        entry
            .detail_texture
            .as_ref()
            .or(entry.base_texture.as_ref())
    }

    /// Ends the visibility epoch after the canvas has rendered. Any queued
    /// work for tiles that were not actually painted this frame becomes stale
    /// immediately; the worker can skip it before opening the source file.
    pub fn finish_frame(&mut self) {
        let frame = self.context.cumulative_frame_nr();
        let stale: Vec<(Uuid, u64)> = self
            .entries
            .iter_mut()
            .filter_map(|(id, entry)| {
                if entry.last_visible_frame == Some(frame) {
                    return None;
                }
                entry.desired_edge = 0;
                entry.desired_mode = None;
                entry.pending.take().map(|pending| (*id, pending.id))
            })
            .collect();
        for (id, request_id) in stale {
            self.clear_worker_demand(id, request_id);
        }
    }

    /// Cancels every queued preview demand when the native window is hidden
    /// or occluded and eframe intentionally skips the canvas UI pass.
    pub fn cancel_pending(&mut self) {
        let stale: Vec<(Uuid, u64)> = self
            .entries
            .iter_mut()
            .filter_map(|(id, entry)| {
                entry.desired_edge = 0;
                entry.desired_mode = None;
                entry.pending.take().map(|pending| (*id, pending.id))
            })
            .collect();
        for (id, request_id) in stale {
            self.clear_worker_demand(id, request_id);
        }
    }

    /// The file behind a tile changed on disk. Softer than
    /// [`invalidate`](Self::invalidate): the resident texture keeps showing
    /// while its replacement is decoded, so an external save never flashes
    /// the default placeholder — the old picture simply becomes the new one.
    ///
    /// The on-disk thumbnail cache needs no attention here: it is validated
    /// against the source's modification time at decode, so a newer source
    /// regenerates it.
    pub fn mark_stale(&mut self, id: Uuid) {
        let has_texture = self
            .entries
            .get(&id)
            .is_some_and(|entry| entry.base_texture.is_some() || entry.detail_texture.is_some());
        if !has_texture {
            // Nothing on screen to preserve; the hard invalidation is
            // strictly better bookkeeping.
            self.invalidate(id);
            return;
        }
        // A decode already in flight may have read the old content; cancel
        // it rather than let it deliver outdated pixels that clear the flag.
        if let Some(pending) = self.entries.get(&id).and_then(|entry| entry.pending) {
            self.clear_worker_demand(id, pending.id);
        }
        if let Some(entry) = self.entries.get_mut(&id) {
            entry.pending = None;
            entry.stale = true;
            // A file that failed to decode before may well decode now.
            entry.failure_count = 0;
            entry.retry_after = None;
        }
    }

    pub fn invalidate(&mut self, id: Uuid) {
        if let Some(pending) = self.entries.get(&id).and_then(|entry| entry.pending) {
            self.clear_worker_demand(id, pending.id);
        }
        self.entries.remove(&id);
        let _ = fs::remove_dir_all(self.paths.thumbnail_dir(id));
    }

    pub fn retain_only(&mut self, valid_ids: &HashSet<Uuid>) {
        let stale: Vec<(Uuid, u64)> = self
            .entries
            .iter()
            .filter(|(id, _)| !valid_ids.contains(id))
            .filter_map(|(id, entry)| entry.pending.map(|pending| (*id, pending.id)))
            .collect();
        for (id, request_id) in stale {
            self.clear_worker_demand(id, request_id);
        }
        self.entries.retain(|id, _| valid_ids.contains(id));
    }

    fn resident_bytes(&self) -> usize {
        self.entries
            .values()
            .map(|entry| entry.base_bytes.saturating_add(entry.detail_bytes))
            .sum()
    }

    fn can_admit_detail(&self, id: Uuid, requested_bytes: usize, frame: u64) -> bool {
        let mut committed_base = 0usize;
        let mut committed_detail = 0usize;
        for (entry_id, entry) in &self.entries {
            if !was_recently_visible(entry, frame) {
                continue;
            }
            let base_edge = match entry.base_mode.or(entry.desired_mode) {
                Some(PreviewMode::QuickLook) => QUICK_LOOK_EDGE,
                Some(PreviewMode::Image) | None => IMAGE_BASE_EDGE,
            };
            let expected_base = entry.intrinsic_size.map_or_else(
                || square_rgba_bytes(base_edge),
                |dimensions| estimated_image_bytes(base_edge, dimensions),
            );
            committed_base = committed_base.saturating_add(entry.base_bytes.max(expected_base));
            if *entry_id != id {
                committed_detail = committed_detail.saturating_add(entry.detail_bytes);
                if let Some(pending) = entry.pending
                    && pending.mode == PreviewMode::Image
                    && pending.edge > IMAGE_BASE_EDGE
                {
                    committed_detail = committed_detail.saturating_add(pending.estimated_bytes);
                }
            }
        }

        committed_base
            .saturating_add(committed_detail)
            .saturating_add(requested_bytes)
            <= GPU_CACHE_BUDGET
    }

    fn evict_unused(&mut self, frame: u64) {
        let now = Instant::now();
        self.entries.retain(|_, entry| {
            entry.pending.is_some()
                || entry
                    .last_used
                    .is_none_or(|used| now.duration_since(used) <= UNUSED_TTL)
        });

        let mut total = self.resident_bytes();
        if total <= GPU_CACHE_BUDGET {
            return;
        }

        // Detail for offscreen tiles is the cheapest first sacrifice.
        let mut offscreen_detail: Vec<(Uuid, Option<Instant>, usize)> = self
            .entries
            .iter()
            .filter(|(_, entry)| {
                entry.detail_texture.is_some() && !was_recently_visible(entry, frame)
            })
            .map(|(id, entry)| (*id, entry.last_used, entry.detail_bytes))
            .collect();
        offscreen_detail.sort_by_key(|(_, used, _)| *used);

        for (id, _, bytes) in offscreen_detail {
            if total <= GPU_CACHE_BUDGET {
                break;
            }
            let Some(entry) = self.entries.get_mut(&id) else {
                continue;
            };
            entry.detail_texture = None;
            entry.detail_edge = 0;
            entry.detail_bytes = 0;
            total = total.saturating_sub(bytes);
        }

        if total <= GPU_CACHE_BUDGET {
            return;
        }

        // Drop offscreen bases before compromising current Retina detail.
        // This ordering is important: otherwise a visible detail could be
        // evicted and immediately re-admitted against a budget calculation
        // that correctly ignores those disposable offscreen bases.
        let mut base_candidates: Vec<(Uuid, Option<Instant>, usize)> = self
            .entries
            .iter()
            .filter(|(_, entry)| {
                entry.base_texture.is_some() && !was_recently_visible(entry, frame)
            })
            .map(|(id, entry)| (*id, entry.last_used, entry.base_bytes))
            .collect();
        base_candidates.sort_by_key(|(_, used, _)| *used);
        for (id, _, bytes) in base_candidates {
            if total <= GPU_CACHE_BUDGET {
                break;
            }
            let Some(entry) = self.entries.get_mut(&id) else {
                continue;
            };
            entry.base_texture = None;
            entry.base_edge = 0;
            entry.base_mode = None;
            entry.base_bytes = 0;
            total = total.saturating_sub(bytes);
        }

        if total <= GPU_CACHE_BUDGET {
            return;
        }

        // Admission normally makes this unnecessary. It is a final safety
        // valve for unexpected driver accounting or metadata-free formats;
        // even here the visible base remains intact.
        let mut visible_detail: Vec<(Uuid, Option<Instant>, usize)> = self
            .entries
            .iter()
            .filter(|(_, entry)| {
                entry.detail_texture.is_some() && was_recently_visible(entry, frame)
            })
            .map(|(id, entry)| (*id, entry.last_used, entry.detail_bytes))
            .collect();
        visible_detail.sort_by_key(|(_, used, _)| *used);
        for (id, _, bytes) in visible_detail {
            if total <= GPU_CACHE_BUDGET {
                break;
            }
            let Some(entry) = self.entries.get_mut(&id) else {
                continue;
            };
            entry.detail_texture = None;
            entry.detail_edge = 0;
            entry.detail_bytes = 0;
            total = total.saturating_sub(bytes);
        }
    }

    fn set_worker_demand(&self, id: Uuid, request_id: u64) {
        let mut demands = self
            .worker_demands
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        demands.insert(id, WorkerDemand { request_id });
    }

    fn clear_worker_demand(&self, id: Uuid, request_id: u64) {
        let mut demands = self
            .worker_demands
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if demands
            .get(&id)
            .is_some_and(|demand| demand.request_id == request_id)
        {
            demands.remove(&id);
        }
    }
}

fn was_recently_visible(entry: &Entry, frame: u64) -> bool {
    entry
        .last_visible_frame
        .is_some_and(|visible_frame| frame.saturating_sub(visible_frame) <= 1)
}

fn square_rgba_bytes(edge: u32) -> usize {
    (edge as usize)
        .saturating_mul(edge as usize)
        .saturating_mul(4)
}

fn estimated_image_bytes(edge: u32, dimensions: [u32; 2]) -> usize {
    let [width, height] = dimensions;
    if width == 0 || height == 0 {
        return square_rgba_bytes(edge);
    }
    let source_edge = width.max(height);
    let target_edge = edge.min(source_edge);
    let scale = target_edge as f64 / source_edge as f64;
    let target_width = ((width as f64 * scale).round() as usize).max(1);
    let target_height = ((height as f64 * scale).round() as usize).max(1);
    target_width.saturating_mul(target_height).saturating_mul(4)
}

fn retry_delay_after_failure(failure_count: u8) -> Option<Duration> {
    if failure_count >= MAX_PREVIEW_FAILURES {
        return None;
    }
    let exponent = failure_count.saturating_sub(1).min(7) as u32;
    Some(Duration::from_millis(250u64.saturating_mul(1u64 << exponent)).min(FAILURE_RETRY_MAX))
}

fn quantized_image_edge(projected_pixels: [f32; 2]) -> u32 {
    let projected_edge = projected_pixels[0].max(projected_pixels[1]);
    if !projected_edge.is_finite() {
        return *IMAGE_PREVIEW_EDGES
            .last()
            .expect("preview tiers are nonempty");
    }
    // A small amount of headroom keeps text crisp at fractional transforms
    // and avoids hovering directly on a tier boundary. Once loaded, sharper
    // resident textures are never downgraded, providing the other half of the
    // zoom hysteresis without issuing per-frame work.
    let projected_edge = projected_edge.max(1.0) * DETAIL_OVERSCAN;
    IMAGE_PREVIEW_EDGES
        .iter()
        .copied()
        .find(|edge| *edge as f32 >= projected_edge)
        .unwrap_or_else(|| {
            *IMAGE_PREVIEW_EDGES
                .last()
                .expect("preview tiers are nonempty")
        })
}

fn preview_loop(
    receiver: Receiver<PreviewJob>,
    sender: Sender<PreviewResult>,
    context: Context,
    demands: Arc<Mutex<HashMap<Uuid, WorkerDemand>>>,
) {
    while let Ok(job) = receiver.recv() {
        if !worker_demand_is_current(&demands, &job) {
            // Superseded zoom tiers and tiles panned out of view stay in the
            // bounded channel only as tiny path records; never open/decode
            // their source.
            context.request_repaint();
            continue;
        }
        let result = produce_preview(&job);
        if !worker_demand_is_current(&demands, &job) {
            // A view change may arrive during a decode. The tier can remain in
            // the disk cache, but do not occupy the result channel or upload
            // obsolete pixels to the GPU.
            context.request_repaint();
            continue;
        }
        let result = match result {
            Ok(pixels) => PreviewResult::Ready(pixels),
            Err(error) => {
                log::debug!(
                    "preview unavailable for {} at {} px: {error:#}",
                    job.source.display(),
                    job.edge
                );
                PreviewResult::Failed {
                    id: job.id,
                    request_id: job.request_id,
                }
            }
        };
        if sender.send(result).is_err() {
            break;
        }
        context.request_repaint();
    }
}

fn worker_demand_is_current(
    demands: &Mutex<HashMap<Uuid, WorkerDemand>>,
    job: &PreviewJob,
) -> bool {
    let demands = demands
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    demands
        .get(&job.id)
        .is_some_and(|demand| demand.request_id == job.request_id)
}

fn produce_preview(job: &PreviewJob) -> anyhow::Result<PreviewPixels> {
    let (image, intrinsic_size) = match job.mode {
        PreviewMode::Image => load_photo(job)?,
        PreviewMode::QuickLook => (load_quick_look_preview(job)?, None),
    };
    let image = match job.mode {
        // The versioned photo cache already contains exactly this tier.
        PreviewMode::Image => image.into_rgba8(),
        PreviewMode::QuickLook => thumbnail_without_upscale(image, job.edge).into_rgba8(),
    };
    let (width, height) = image.dimensions();
    Ok(PreviewPixels {
        id: job.id,
        request_id: job.request_id,
        edge: job.edge,
        mode: job.mode,
        size: [width as usize, height as usize],
        intrinsic_size,
        rgba: image.into_raw(),
    })
}

fn load_photo(job: &PreviewJob) -> anyhow::Result<(DynamicImage, Option<[u32; 2]>)> {
    let source_metadata = probe_source_image(&job.source).ok();
    let intrinsic_size = source_metadata.map(|metadata| metadata.oriented_size);
    let tier_directory = job
        .cache_directory
        .join(PHOTO_CACHE_VERSION)
        .join(job.edge.to_string());
    let cached = tier_directory.join("preview.png");

    if cache_is_fresh(&job.source, &cached) {
        match image::open(&cached) {
            Ok(image) if current_photo_cache_is_valid(&image, job.edge, source_metadata) => {
                return Ok((image, intrinsic_size));
            }
            Ok(_) | Err(_) => {
                // A crash during a prior write must not permanently poison a
                // tier. Likewise, discard any oversized v2 file left by an
                // interrupted development build.
                let _ = fs::remove_file(&cached);
            }
        }
    } else {
        let _ = fs::remove_file(&cached);
    }

    // A dense existing library already has 512px Quick Look/photo-v1 files.
    // Derive the progressive base from those when they are trustworthy, while
    // never accepting the old upscaled high tiers that caused GPU thrash.
    if let Some(image) = reusable_cached_photo(job, source_metadata) {
        let image = thumbnail_without_upscale(image, job.edge);
        save_photo_cache_best_effort(&image, &cached);
        return Ok((image, intrinsic_size));
    }

    let (image, intrinsic_size) = decode_photo_source(job, source_metadata)?;
    let image = thumbnail_without_upscale(image, job.edge);
    save_photo_cache_best_effort(&image, &cached);
    Ok((image, intrinsic_size))
}

fn decode_photo_source(
    job: &PreviewJob,
    source_metadata: Option<SourceImageMetadata>,
) -> anyhow::Result<(DynamicImage, Option<[u32; 2]>)> {
    #[cfg(target_os = "macos")]
    if let Some(metadata) = source_metadata {
        match crate::macos_image_io::decode_thumbnail(&job.source, job.edge) {
            Ok(image) => return Ok((image, Some(metadata.oriented_size))),
            Err(error) => {
                // Image I/O is the efficient macOS path because it can
                // downsample JPEG/HEIC while decoding. The portable decoder
                // remains a real fallback for formats or files the system
                // framework rejects.
                log::debug!(
                    "using Rust photo fallback for {} after Image I/O failed: {error:#}",
                    job.source.display()
                );
            }
        }
    }

    match decode_oriented_image(&job.source) {
        Ok(image) => {
            let (width, height) = image.dimensions();
            Ok((
                image,
                source_metadata
                    .map(|metadata| metadata.oriented_size)
                    .or(Some([width, height])),
            ))
        }
        Err(error) => {
            // HEIC, SVG, and future system-supported formats may not be
            // decoded by the Rust image crate. Quick Look remains the native
            // fallback, but now it is generated at the requested tier.
            log::debug!(
                "using Quick Look photo fallback for {}: {error}",
                job.source.display()
            );
            let image = load_quick_look_preview(job)?;
            Ok((
                image,
                source_metadata.map(|metadata| metadata.oriented_size),
            ))
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SourceImageMetadata {
    oriented_size: [u32; 2],
    orientation: Orientation,
}

/// Reads only decoder metadata and reports the pixel dimensions after EXIF
/// orientation. Import workers use this to shape a tile correctly without a
/// full image decode.
pub(crate) fn oriented_image_dimensions(path: &Path) -> Option<[u32; 2]> {
    probe_source_image(path)
        .ok()
        .map(|metadata| metadata.oriented_size)
}

fn probe_source_image(path: &Path) -> anyhow::Result<SourceImageMetadata> {
    let rust_probe = probe_rust_image_metadata(path);
    if let Ok(metadata) = rust_probe {
        return Ok(metadata);
    }
    #[cfg(target_os = "macos")]
    if let Some(metadata) = probe_macos_image_metadata(path) {
        return Ok(metadata);
    }
    rust_probe
}

fn probe_rust_image_metadata(path: &Path) -> anyhow::Result<SourceImageMetadata> {
    let reader = ImageReader::open(path)?.with_guessed_format()?;
    let mut decoder = reader.into_decoder()?;
    let encoded_size = decoder.dimensions();
    let orientation = decoder.orientation().unwrap_or(Orientation::NoTransforms);
    Ok(SourceImageMetadata {
        oriented_size: oriented_dimensions(encoded_size, orientation),
        orientation,
    })
}

#[cfg(target_os = "macos")]
fn probe_macos_image_metadata(path: &Path) -> Option<SourceImageMetadata> {
    probe_image_io_metadata(path).or_else(|| probe_sips_image_metadata(path))
}

#[cfg(target_os = "macos")]
fn probe_image_io_metadata(path: &Path) -> Option<SourceImageMetadata> {
    use objc2_core_foundation::{CFDictionary, CFNumber, CFRetained, CFString, CFType, CFURL};
    use objc2_image_io::{
        CGImageSource, kCGImagePropertyOrientation, kCGImagePropertyPixelHeight,
        kCGImagePropertyPixelWidth,
    };

    fn number_property(properties: &CFDictionary<CFString, CFType>, key: &CFString) -> Option<u32> {
        properties
            .get(key)?
            .downcast::<CFNumber>()
            .ok()?
            .as_i64()?
            .try_into()
            .ok()
    }

    let url = CFURL::from_file_path(path)?;
    // SAFETY: Both optional dictionaries are None, so there are no generic
    // key/value invariants for the caller to uphold.
    let source = unsafe { CGImageSource::with_url(&url, None) }?;
    // SAFETY: Index zero exists for a valid image source and options is None.
    let properties = unsafe { source.properties_at_index(0, None) }?;
    // SAFETY: ImageIO documents this as a CFString-keyed property dictionary
    // whose values are CFType instances. Individual values are downcast
    // before use below.
    let properties: CFRetained<CFDictionary<CFString, CFType>> =
        unsafe { CFRetained::cast_unchecked(properties) };
    // SAFETY: These are process-lifetime ImageIO CFString constants.
    let width = number_property(&properties, unsafe { kCGImagePropertyPixelWidth })?;
    // SAFETY: These are process-lifetime ImageIO CFString constants.
    let height = number_property(&properties, unsafe { kCGImagePropertyPixelHeight })?;
    // SAFETY: This is a process-lifetime ImageIO CFString constant.
    let orientation = number_property(&properties, unsafe { kCGImagePropertyOrientation })
        .and_then(|value| u8::try_from(value).ok())
        .and_then(Orientation::from_exif)
        .unwrap_or(Orientation::NoTransforms);
    (width > 0 && height > 0).then(|| SourceImageMetadata {
        oriented_size: oriented_dimensions((width, height), orientation),
        orientation,
    })
}

#[cfg(target_os = "macos")]
fn probe_sips_image_metadata(path: &Path) -> Option<SourceImageMetadata> {
    let output = Command::new("/usr/bin/sips")
        .arg("-g")
        .arg("pixelWidth")
        .arg("-g")
        .arg("pixelHeight")
        .arg("-g")
        .arg("orientation")
        .arg(path)
        .env("LC_ALL", "C")
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| parse_sips_image_metadata(&String::from_utf8_lossy(&output.stdout)))
        .flatten()
}

#[cfg(target_os = "macos")]
fn parse_sips_image_metadata(output: &str) -> Option<SourceImageMetadata> {
    fn positive_dimension(value: &str) -> Option<u32> {
        let value = value.trim().parse::<f64>().ok()?;
        (value.is_finite() && value >= 1.0 && value <= u32::MAX as f64)
            .then_some(value.round() as u32)
    }

    let mut width = None;
    let mut height = None;
    let mut orientation = Orientation::NoTransforms;
    for line in output.lines().map(str::trim) {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        match key.trim() {
            "pixelWidth" => width = positive_dimension(value),
            "pixelHeight" => height = positive_dimension(value),
            "orientation" => {
                if let Ok(value) = value.trim().parse::<u8>()
                    && let Some(parsed) = Orientation::from_exif(value)
                {
                    orientation = parsed;
                }
            }
            _ => {}
        }
    }
    let dimensions = (width?, height?);
    Some(SourceImageMetadata {
        oriented_size: oriented_dimensions(dimensions, orientation),
        orientation,
    })
}

fn decode_oriented_image(path: &Path) -> anyhow::Result<DynamicImage> {
    let reader = ImageReader::open(path)?.with_guessed_format()?;
    let mut decoder = reader.into_decoder()?;
    let orientation = decoder.orientation().unwrap_or(Orientation::NoTransforms);
    let mut image = DynamicImage::from_decoder(decoder)?;
    image.apply_orientation(orientation);
    Ok(image)
}

fn oriented_dimensions(dimensions: (u32, u32), orientation: Orientation) -> [u32; 2] {
    let (width, height) = dimensions;
    match orientation {
        Orientation::Rotate90
        | Orientation::Rotate270
        | Orientation::Rotate90FlipH
        | Orientation::Rotate270FlipH => [height, width],
        Orientation::NoTransforms
        | Orientation::Rotate180
        | Orientation::FlipHorizontal
        | Orientation::FlipVertical => [width, height],
    }
}

fn thumbnail_without_upscale(image: DynamicImage, edge: u32) -> DynamicImage {
    let (width, height) = image.dimensions();
    if width.max(height) <= edge {
        image
    } else {
        image.thumbnail(edge, edge)
    }
}

fn reusable_cached_photo(
    job: &PreviewJob,
    source_metadata: Option<SourceImageMetadata>,
) -> Option<DynamicImage> {
    let mut candidates = Vec::new();

    // Current-version larger tiers are always oriented and can cheaply seed a
    // lower base after a relaunch.
    for edge in IMAGE_PREVIEW_EDGES {
        if edge > job.edge {
            candidates.push((
                job.cache_directory
                    .join(PHOTO_CACHE_VERSION)
                    .join(edge.to_string())
                    .join("preview.png"),
                false,
            ));
        }
    }

    // Pre-v3 pixels are useful only as a temporary 256px base. Detail must
    // always come from the current color-managed Image I/O pipeline.
    if job.edge == IMAGE_BASE_EDGE {
        for edge in IMAGE_PREVIEW_EDGES {
            if edge >= job.edge {
                candidates.push((
                    job.cache_directory
                        .join(ORIENTED_LEGACY_PHOTO_CACHE_VERSION)
                        .join(edge.to_string())
                        .join("preview.png"),
                    false,
                ));
            }
        }

        // v1 was not orientation-aware. It remains safe for screenshots and
        // other no-transform sources, subject to the size/aspect checks below.
        if source_metadata.is_some_and(|metadata| metadata.orientation == Orientation::NoTransforms)
        {
            for edge in IMAGE_PREVIEW_EDGES {
                if edge >= job.edge {
                    candidates.push((
                        job.cache_directory
                            .join(UNORIENTED_LEGACY_PHOTO_CACHE_VERSION)
                            .join(edge.to_string())
                            .join("preview.png"),
                        true,
                    ));
                }
            }
        }

        // The pre-adaptive root preview was generated through macOS Quick
        // Look, which already applies orientation. It is an excellent instant
        // base, but never final detail.
        candidates.push((job.cache_directory.join("preview.png"), false));
    }

    for (candidate, orientation_was_checked) in candidates {
        if !cache_is_fresh(&job.source, &candidate) {
            continue;
        }
        let Ok(image) = image::open(&candidate) else {
            continue;
        };
        if reusable_cache_is_valid(&image, job.edge, source_metadata, orientation_was_checked) {
            return Some(image);
        }
    }
    None
}

fn reusable_cache_is_valid(
    image: &DynamicImage,
    requested_edge: u32,
    source_metadata: Option<SourceImageMetadata>,
    orientation_was_checked: bool,
) -> bool {
    let (width, height) = image.dimensions();
    let cached_edge = width.max(height);
    let Some(metadata) = source_metadata else {
        // Only Quick Look caches are offered for formats the image crate
        // cannot probe. They are safe as progressive bases, never as an
        // invented high-resolution tier.
        return !orientation_was_checked && cached_edge >= requested_edge;
    };
    let [source_width, source_height] = metadata.oriented_size;
    let source_edge = source_width.max(source_height);
    let needed_edge = requested_edge.min(source_edge);
    if cached_edge < needed_edge || cached_edge > source_edge {
        return false;
    }

    // A rotated-but-unoriented v1 cache has the reciprocal aspect. Quick Look
    // and v2 should agree closely with the oriented source.
    let cached_aspect = width as f64 / height.max(1) as f64;
    let source_aspect = source_width as f64 / source_height.max(1) as f64;
    ((cached_aspect / source_aspect) - 1.0).abs() <= 0.02
}

fn current_photo_cache_is_valid(
    image: &DynamicImage,
    requested_edge: u32,
    source_metadata: Option<SourceImageMetadata>,
) -> bool {
    let (width, height) = image.dimensions();
    if let Some(metadata) = source_metadata {
        reusable_cache_is_valid(image, requested_edge, Some(metadata), true)
    } else {
        // The Rust decoder cannot probe formats such as HEIC. These pixels
        // came from this orientation-aware cache version's Quick Look path;
        // ensure only that the tier itself was not enlarged past its request.
        width.max(height) <= requested_edge
    }
}

fn save_photo_cache_best_effort(image: &DynamicImage, destination: &Path) {
    if let Err(error) = save_png_atomically(image, destination) {
        // Cache availability must never decide whether already-decoded pixels
        // can be shown. A later request can retry the cache write.
        log::debug!(
            "could not cache photo preview at {}: {error:#}",
            destination.display()
        );
    }
}

fn load_quick_look_preview(job: &PreviewJob) -> anyhow::Result<DynamicImage> {
    let tier_directory = job
        .cache_directory
        .join("quick-look")
        .join(job.edge.to_string());
    fs::create_dir_all(&tier_directory)?;
    let cached = tier_directory.join("preview.png");

    load_or_regenerate_cached_preview(&job.source, &cached, || {
        generate_quick_look_image(&job.source, &tier_directory, job.edge)
    })
}

fn load_or_regenerate_cached_preview<F>(
    source: &Path,
    cached: &Path,
    generate: F,
) -> anyhow::Result<DynamicImage>
where
    F: FnOnce() -> anyhow::Result<DynamicImage>,
{
    if cache_is_fresh(source, cached) {
        if let Ok(image) = image::open(cached) {
            return Ok(image);
        }
        // A fresh timestamp does not imply valid pixels: crashes and aborted
        // writes can leave a corrupt PNG. Remove it before regenerating once.
        let _ = fs::remove_file(cached);
    } else {
        let _ = fs::remove_file(cached);
    }

    let image = generate()?;
    if let Err(error) = save_png_atomically(&image, cached) {
        // The generated pixels are still usable this frame. Avoid turning a
        // cache permission/full-disk problem into a visible placeholder.
        log::debug!(
            "could not atomically cache Quick Look preview at {}: {error:#}",
            cached.display()
        );
    }
    Ok(image)
}

fn generate_quick_look_image(
    source: &Path,
    tier_directory: &Path,
    edge: u32,
) -> anyhow::Result<DynamicImage> {
    let staging = tier_directory.join(format!(".quick-look-{}", Uuid::new_v4()));
    fs::create_dir_all(&staging)?;
    let result = (|| {
        generate_quick_look_thumbnail(source, &staging, edge)?;
        let generated = first_generated_png(&staging, &staging.join("preview.png"))
            .ok_or_else(|| anyhow::anyhow!("Quick Look did not produce a PNG"))?;
        Ok(image::open(generated)?)
    })();
    let _ = fs::remove_dir_all(&staging);
    result
}

fn cache_is_fresh(source: &Path, cached: &Path) -> bool {
    let source_modified = fs::metadata(source)
        .and_then(|metadata| metadata.modified())
        .unwrap_or(SystemTime::UNIX_EPOCH);
    fs::metadata(cached)
        .and_then(|metadata| metadata.modified())
        .is_ok_and(|modified| modified >= source_modified)
}

fn save_png_atomically(image: &DynamicImage, destination: &Path) -> anyhow::Result<()> {
    let parent = destination
        .parent()
        .ok_or_else(|| anyhow::anyhow!("preview cache destination has no parent"))?;
    fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(".{}.partial.png", Uuid::new_v4()));
    if let Err(error) = image.save_with_format(&temporary, image::ImageFormat::Png) {
        let _ = fs::remove_file(&temporary);
        return Err(error.into());
    }
    if let Err(error) = fs::rename(&temporary, destination) {
        let _ = fs::remove_file(&temporary);
        return Err(error.into());
    }
    Ok(())
}

fn generate_quick_look_thumbnail(
    source: &Path,
    destination: &Path,
    edge: u32,
) -> anyhow::Result<()> {
    let status = Command::new("/usr/bin/qlmanage")
        .arg("-t")
        .arg("-s")
        .arg(edge.to_string())
        .arg("-o")
        .arg(destination)
        .arg(source)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()?;
    anyhow::ensure!(status.success(), "Quick Look thumbnail generation failed");
    Ok(())
}

fn first_generated_png(directory: &Path, canonical: &Path) -> Option<PathBuf> {
    if canonical.exists() {
        return Some(canonical.to_owned());
    }
    fs::read_dir(directory)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| {
            path.extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("png"))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    fn preview_job(source: PathBuf, cache_directory: PathBuf, edge: u32) -> PreviewJob {
        PreviewJob {
            id: Uuid::new_v4(),
            request_id: 7,
            source,
            cache_directory,
            mode: PreviewMode::Image,
            edge,
        }
    }

    fn write_oriented_jpeg(path: &Path, width: u32, height: u32, orientation: u8) {
        image::RgbImage::from_pixel(width, height, image::Rgb([20, 40, 60]))
            .save_with_format(path, image::ImageFormat::Jpeg)
            .expect("save JPEG source");
        let encoded = fs::read(path).expect("read JPEG source");
        assert_eq!(&encoded[..2], &[0xff, 0xd8]);

        let mut exif = vec![
            b'E',
            b'x',
            b'i',
            b'f',
            0,
            0, // Exif header
            b'M',
            b'M',
            0,
            42,
            0,
            0,
            0,
            8, // big-endian TIFF header
            0,
            1, // one IFD entry
            0x01,
            0x12, // Orientation tag
            0,
            3, // SHORT
            0,
            0,
            0,
            1, // one value
            0,
            orientation,
            0,
            0, // inline SHORT value
            0,
            0,
            0,
            0, // no next IFD
        ];
        assert_eq!(exif.len(), 32);
        let segment_length = (exif.len() + 2) as u16;
        let mut oriented = Vec::with_capacity(encoded.len() + exif.len() + 4);
        oriented.extend_from_slice(&encoded[..2]);
        oriented.extend_from_slice(&[0xff, 0xe1]);
        oriented.extend_from_slice(&segment_length.to_be_bytes());
        oriented.append(&mut exif);
        oriented.extend_from_slice(&encoded[2..]);
        fs::write(path, oriented).expect("write oriented JPEG");
    }

    #[test]
    fn image_resolution_is_quantized_to_stable_tiers() {
        assert_eq!(quantized_image_edge([1.0, 1.0]), 256);
        assert_eq!(quantized_image_edge([240.0, 120.0]), 256);
        assert_eq!(quantized_image_edge([244.0, 200.0]), 512);
        assert_eq!(quantized_image_edge([488.0, 300.0]), 1_024);
        assert_eq!(quantized_image_edge([1_200.0, 700.0]), 2_048);
        assert_eq!(quantized_image_edge([9_000.0, 7_000.0]), 4_096);
        assert_eq!(quantized_image_edge([f32::INFINITY, 100.0]), 4_096);
    }

    #[test]
    fn photo_decode_preserves_intrinsic_size_and_honors_tier() {
        let temporary = tempfile::tempdir().expect("temporary directory");
        let source = temporary.path().join("source.png");
        let pixels = image::RgbaImage::from_pixel(800, 400, image::Rgba([20, 40, 60, 255]));
        pixels.save(&source).expect("save source image");
        let job = preview_job(source, temporary.path().join("cache"), 256);

        let preview = produce_preview(&job).expect("produce photo preview");

        assert_eq!(preview.intrinsic_size, Some([800, 400]));
        assert_eq!(preview.size, [256, 128]);
        assert_eq!(preview.edge, 256);
        assert!(
            temporary
                .path()
                .join("cache")
                .join(PHOTO_CACHE_VERSION)
                .join("256")
                .join("preview.png")
                .is_file()
        );
    }

    #[test]
    fn legacy_preview_never_satisfies_an_adaptive_photo_tier() {
        let temporary = tempfile::tempdir().expect("temporary directory");
        let source = temporary.path().join("source.png");
        let cache = temporary.path().join("cache");
        fs::create_dir_all(&cache).expect("create cache");
        image::RgbaImage::from_pixel(1, 1, image::Rgba([255, 0, 0, 255]))
            .save(cache.join("preview.png"))
            .expect("save legacy preview");
        image::RgbaImage::from_pixel(800, 400, image::Rgba([20, 40, 60, 255]))
            .save(&source)
            .expect("save source image");
        let job = PreviewJob {
            id: Uuid::new_v4(),
            request_id: 9,
            source,
            cache_directory: cache.clone(),
            mode: PreviewMode::Image,
            edge: 512,
        };

        let preview = produce_preview(&job).expect("produce adaptive preview");

        assert_eq!(preview.size, [512, 256]);
        assert!(cache.join(PHOTO_CACHE_VERSION).join("512").is_dir());
        assert!(cache.join("preview.png").is_file());
    }

    #[test]
    fn pre_v3_tier_only_seeds_the_progressive_base() {
        let temporary = tempfile::tempdir().expect("temporary directory");
        let source = temporary.path().join("source.png");
        let cache = temporary.path().join("cache");
        image::RgbaImage::from_pixel(800, 400, image::Rgba([20, 40, 60, 255]))
            .save(&source)
            .expect("save source image");

        let legacy_tier = cache.join(ORIENTED_LEGACY_PHOTO_CACHE_VERSION).join("512");
        fs::create_dir_all(&legacy_tier).expect("create legacy tier");
        image::RgbaImage::from_pixel(512, 256, image::Rgba([255, 0, 0, 255]))
            .save(legacy_tier.join("preview.png"))
            .expect("save legacy tier");

        let base = produce_preview(&preview_job(source.clone(), cache.clone(), 256))
            .expect("produce progressive base");
        assert_eq!(&base.rgba[..4], &[255, 0, 0, 255]);

        let detail = produce_preview(&preview_job(source, cache.clone(), 512))
            .expect("produce current detail");
        assert_eq!(&detail.rgba[..4], &[20, 40, 60, 255]);
        assert!(
            cache
                .join(PHOTO_CACHE_VERSION)
                .join("512")
                .join("preview.png")
                .is_file()
        );
    }

    #[test]
    fn photo_preview_never_upscales_beyond_source_pixels() {
        let temporary = tempfile::tempdir().expect("temporary directory");
        let source = temporary.path().join("source.png");
        image::RgbaImage::from_pixel(752, 728, image::Rgba([20, 40, 60, 255]))
            .save(&source)
            .expect("save source image");
        let job = preview_job(source, temporary.path().join("cache"), 2_048);

        let preview = produce_preview(&job).expect("produce photo preview");

        assert_eq!(preview.intrinsic_size, Some([752, 728]));
        assert_eq!(preview.size, [752, 728]);
        assert_eq!(
            image::image_dimensions(
                temporary
                    .path()
                    .join("cache")
                    .join(PHOTO_CACHE_VERSION)
                    .join("2048")
                    .join("preview.png")
            )
            .expect("cached dimensions"),
            (752, 728)
        );
    }

    #[test]
    fn exif_orientation_controls_intrinsic_and_preview_dimensions() {
        let temporary = tempfile::tempdir().expect("temporary directory");
        let source = temporary.path().join("portrait.jpg");
        write_oriented_jpeg(&source, 40, 80, 6);
        let job = preview_job(source.clone(), temporary.path().join("cache"), 256);

        assert_eq!(oriented_image_dimensions(&source), Some([80, 40]));
        let preview = produce_preview(&job).expect("produce oriented photo preview");

        assert_eq!(preview.intrinsic_size, Some([80, 40]));
        assert_eq!(preview.size, [80, 40]);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn image_io_metadata_reports_oriented_dimensions() {
        let temporary = tempfile::tempdir().expect("temporary directory");
        let source = temporary.path().join("portrait.jpg");
        write_oriented_jpeg(&source, 40, 80, 6);

        assert_eq!(
            probe_image_io_metadata(&source),
            Some(SourceImageMetadata {
                oriented_size: [80, 40],
                orientation: Orientation::Rotate90,
            })
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn system_metadata_fallback_supports_svg_dimensions() {
        let temporary = tempfile::tempdir().expect("temporary directory");
        let source = temporary.path().join("diagram.svg");
        fs::write(
            &source,
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480"></svg>"#,
        )
        .expect("save SVG");

        assert_eq!(oriented_image_dimensions(&source), Some([640, 480]));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn sips_metadata_parser_accepts_decimal_dimensions_and_orientation() {
        let metadata = parse_sips_image_metadata(
            "/tmp/photo.heic\n  pixelWidth: 3024.000\n  pixelHeight: 4032.000\n  orientation: 6\n",
        )
        .expect("parse sips metadata");

        assert_eq!(metadata.oriented_size, [4_032, 3_024]);
        assert_eq!(metadata.orientation, Orientation::Rotate90);
    }

    #[test]
    fn fresh_corrupt_quick_look_cache_regenerates_exactly_once() {
        let temporary = tempfile::tempdir().expect("temporary directory");
        let source = temporary.path().join("source.dat");
        let cached = temporary.path().join("cache").join("preview.png");
        fs::write(&source, b"source").expect("save source");
        fs::create_dir_all(cached.parent().expect("cache parent")).expect("create cache");
        fs::write(&cached, b"not a png").expect("save corrupt cache");
        assert!(cache_is_fresh(&source, &cached));
        let generations = Cell::new(0);

        let regenerated = load_or_regenerate_cached_preview(&source, &cached, || {
            generations.set(generations.get() + 1);
            Ok(DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
                8,
                4,
                image::Rgba([20, 40, 60, 255]),
            )))
        })
        .expect("regenerate corrupt cache");
        assert_eq!(regenerated.dimensions(), (8, 4));
        assert_eq!(generations.get(), 1);
        assert_eq!(
            image::image_dimensions(&cached).expect("valid replacement cache"),
            (8, 4)
        );

        let reused = load_or_regenerate_cached_preview(&source, &cached, || {
            generations.set(generations.get() + 1);
            anyhow::bail!("fresh valid cache should not regenerate")
        })
        .expect("reuse repaired cache");
        assert_eq!(reused.dimensions(), (8, 4));
        assert_eq!(generations.get(), 1);
    }

    #[test]
    fn permanent_failures_have_a_finite_retry_budget() {
        assert_eq!(
            retry_delay_after_failure(1),
            Some(Duration::from_millis(250))
        );
        assert!(retry_delay_after_failure(MAX_PREVIEW_FAILURES - 1).is_some());
        assert_eq!(retry_delay_after_failure(MAX_PREVIEW_FAILURES), None);
        assert_eq!(retry_delay_after_failure(u8::MAX), None);
    }

    fn test_texture(context: &Context, name: &str) -> TextureHandle {
        context.load_texture(
            name,
            ColorImage::from_rgba_unmultiplied([1, 1], &[10, 20, 30, 255]),
            TextureOptions::LINEAR,
        )
    }

    #[test]
    fn a_stale_entry_keeps_serving_its_texture_while_a_fresh_decode_queues() {
        let temporary = tempfile::tempdir().expect("temporary directory");
        let source = temporary.path().join("photo.png");
        image::RgbaImage::from_pixel(64, 64, image::Rgba([200, 10, 10, 255]))
            .save(&source)
            .expect("save source");
        let (jobs, job_receiver) = bounded(4);
        let (_result_sender, results) = bounded(4);
        let context = Context::default();
        let id = Uuid::new_v4();
        let mut cache = PreviewCache {
            jobs,
            results,
            entries: HashMap::from([(
                id,
                Entry {
                    base_texture: Some(test_texture(&context, "old-base")),
                    base_edge: QUICK_LOOK_EDGE,
                    base_mode: Some(PreviewMode::QuickLook),
                    ..Default::default()
                },
            )]),
            worker_demands: Arc::new(Mutex::new(HashMap::new())),
            paths: AppPaths::at(temporary.path()),
            context: context.clone(),
            next_request_id: 1,
            last_eviction: Instant::now(),
        };

        cache.mark_stale(id);
        assert!(cache.entries[&id].stale);
        assert!(
            cache.entries[&id].base_texture.is_some(),
            "the old picture must survive going stale — that is the point"
        );

        // The next paint keeps its texture AND schedules a fresh base decode.
        let texture = cache.quick_look_texture(id, &source);
        assert!(texture.is_some(), "no placeholder frame is ever shown");
        let entry = &cache.entries[&id];
        assert!(
            entry.pending.is_some(),
            "a revalidating decode must have been scheduled"
        );
        let queued = job_receiver.try_recv().expect("job reaches the worker");
        assert_eq!(queued.mode, PreviewMode::QuickLook);
        assert_eq!(queued.edge, QUICK_LOOK_EDGE);
    }

    #[test]
    fn a_fresh_base_clears_staleness_and_drops_the_outdated_detail_layer() {
        let temporary = tempfile::tempdir().expect("temporary directory");
        let (jobs, _job_receiver) = bounded(4);
        let (result_sender, results) = bounded(4);
        let context = Context::default();
        let id = Uuid::new_v4();
        let old_base = test_texture(&context, "old-base");
        let old_base_id = old_base.id();
        let mut cache = PreviewCache {
            jobs,
            results,
            entries: HashMap::from([(
                id,
                Entry {
                    base_texture: Some(old_base),
                    base_edge: IMAGE_BASE_EDGE,
                    base_mode: Some(PreviewMode::Image),
                    detail_texture: Some(test_texture(&context, "old-detail")),
                    detail_edge: 1_024,
                    detail_bytes: 4,
                    stale: true,
                    pending: Some(PendingRequest {
                        id: 7,
                        edge: IMAGE_BASE_EDGE,
                        mode: PreviewMode::Image,
                        estimated_bytes: 4,
                    }),
                    ..Default::default()
                },
            )]),
            worker_demands: Arc::new(Mutex::new(HashMap::new())),
            paths: AppPaths::at(temporary.path()),
            context: context.clone(),
            next_request_id: 8,
            last_eviction: Instant::now(),
        };

        result_sender
            .send(PreviewResult::Ready(PreviewPixels {
                id,
                request_id: 7,
                mode: PreviewMode::Image,
                edge: IMAGE_BASE_EDGE,
                size: [1, 1],
                rgba: vec![5, 6, 7, 255],
                intrinsic_size: None,
            }))
            .expect("deliver replacement");
        cache.poll(&context);

        let entry = &cache.entries[&id];
        assert!(!entry.stale, "the replacement clears staleness");
        let new_base = entry.base_texture.as_ref().expect("fresh base resident");
        assert_ne!(new_base.id(), old_base_id, "the texture was replaced");
        assert!(
            entry.detail_texture.is_none() && entry.detail_edge == 0 && entry.detail_bytes == 0,
            "the outdated detail layer must not keep covering the new base"
        );
    }

    #[test]
    fn marking_a_textureless_entry_stale_is_a_hard_invalidate() {
        let temporary = tempfile::tempdir().expect("temporary directory");
        let (jobs, _job_receiver) = bounded(1);
        let (_result_sender, results) = bounded(1);
        let id = Uuid::new_v4();
        let mut cache = PreviewCache {
            jobs,
            results,
            entries: HashMap::from([(id, Entry::default())]),
            worker_demands: Arc::new(Mutex::new(HashMap::new())),
            paths: AppPaths::at(temporary.path()),
            context: Context::default(),
            next_request_id: 1,
            last_eviction: Instant::now(),
        };
        cache.mark_stale(id);
        assert!(
            !cache.entries.contains_key(&id),
            "nothing to preserve means nothing to keep half-alive"
        );
        // And a tile never seen at all is a no-op, not a panic.
        cache.mark_stale(Uuid::new_v4());
    }

    #[test]
    fn invalidate_resets_a_permanently_failed_entry() {
        let temporary = tempfile::tempdir().expect("temporary directory");
        let (jobs, _job_receiver) = bounded(1);
        let (_result_sender, results) = bounded(1);
        let id = Uuid::new_v4();
        let mut cache = PreviewCache {
            jobs,
            results,
            entries: HashMap::from([(
                id,
                Entry {
                    failure_count: MAX_PREVIEW_FAILURES,
                    ..Default::default()
                },
            )]),
            worker_demands: Arc::new(Mutex::new(HashMap::new())),
            paths: AppPaths::at(temporary.path()),
            context: Context::default(),
            next_request_id: 1,
            last_eviction: Instant::now(),
        };

        cache.invalidate(id);
        assert!(!cache.entries.contains_key(&id));
        assert_eq!(cache.entries.entry(id).or_default().failure_count, 0);
    }

    #[test]
    fn hidden_window_cancellation_clears_pending_worker_demand() {
        let temporary = tempfile::tempdir().expect("temporary directory");
        let (jobs, _job_receiver) = bounded(1);
        let (_result_sender, results) = bounded(1);
        let id = Uuid::new_v4();
        let worker_demands = Arc::new(Mutex::new(HashMap::from([(
            id,
            WorkerDemand { request_id: 7 },
        )])));
        let mut cache = PreviewCache {
            jobs,
            results,
            entries: HashMap::from([(
                id,
                Entry {
                    pending: Some(PendingRequest {
                        id: 7,
                        edge: 512,
                        mode: PreviewMode::Image,
                        estimated_bytes: square_rgba_bytes(512),
                    }),
                    desired_mode: Some(PreviewMode::Image),
                    desired_edge: 512,
                    ..Default::default()
                },
            )]),
            worker_demands: Arc::clone(&worker_demands),
            paths: AppPaths::at(temporary.path()),
            context: Context::default(),
            next_request_id: 1,
            last_eviction: Instant::now(),
        };

        cache.cancel_pending();

        let entry = cache.entries.get(&id).expect("entry remains cached");
        assert!(entry.pending.is_none());
        assert_eq!(entry.desired_mode, None);
        assert_eq!(entry.desired_edge, 0);
        assert!(
            worker_demands
                .lock()
                .expect("worker demands")
                .get(&id)
                .is_none()
        );
    }

    #[test]
    fn actual_source_size_caps_detail_memory_estimate() {
        assert_eq!(estimated_image_bytes(2_048, [752, 728]), 752 * 728 * 4);
        assert_eq!(estimated_image_bytes(256, [800, 400]), 256 * 128 * 4);
    }

    #[test]
    fn detail_admission_reserves_every_visible_base_and_pending_detail() {
        let temporary = tempfile::tempdir().expect("temporary directory");
        let (jobs, _job_receiver) = bounded(1);
        let (_result_sender, results) = bounded(1);
        let context = Context::default();
        let mut cache = PreviewCache {
            jobs,
            results,
            entries: HashMap::new(),
            worker_demands: Arc::new(Mutex::new(HashMap::new())),
            paths: AppPaths::at(temporary.path()),
            context,
            next_request_id: 1,
            last_eviction: Instant::now(),
        };
        let frame = 9;
        let ids: Vec<Uuid> = (0..100)
            .map(|_| {
                let id = Uuid::new_v4();
                cache.entries.insert(
                    id,
                    Entry {
                        base_mode: Some(PreviewMode::Image),
                        base_bytes: square_rgba_bytes(IMAGE_BASE_EDGE),
                        intrinsic_size: Some([4_096, 4_096]),
                        last_visible_frame: Some(frame),
                        ..Default::default()
                    },
                );
                id
            })
            .collect();

        assert!(cache.can_admit_detail(ids[0], 64 * 1024 * 1024, frame));
        cache.entries.get_mut(&ids[0]).expect("first entry").pending = Some(PendingRequest {
            id: 1,
            edge: 4_096,
            mode: PreviewMode::Image,
            estimated_bytes: 64 * 1024 * 1024,
        });
        assert!(!cache.can_admit_detail(ids[1], 64 * 1024 * 1024, frame));
    }

    #[test]
    fn pressure_drops_offscreen_base_before_visible_layers() {
        let temporary = tempfile::tempdir().expect("temporary directory");
        let (jobs, _job_receiver) = bounded(1);
        let (_result_sender, results) = bounded(1);
        let context = Context::default();
        let visible_id = Uuid::new_v4();
        let offscreen_id = Uuid::new_v4();
        let mut cache = PreviewCache {
            jobs,
            results,
            entries: HashMap::from([
                (
                    visible_id,
                    Entry {
                        base_texture: Some(context.load_texture(
                            "visible-base",
                            ColorImage::filled([1, 1], egui::Color32::WHITE),
                            TextureOptions::LINEAR,
                        )),
                        base_mode: Some(PreviewMode::Image),
                        base_bytes: 90 * 1024 * 1024,
                        detail_texture: Some(context.load_texture(
                            "visible-detail",
                            ColorImage::filled([1, 1], egui::Color32::WHITE),
                            TextureOptions::LINEAR,
                        )),
                        detail_edge: 1_024,
                        detail_bytes: 20 * 1024 * 1024,
                        last_used: Some(Instant::now()),
                        last_visible_frame: Some(9),
                        ..Default::default()
                    },
                ),
                (
                    offscreen_id,
                    Entry {
                        base_texture: Some(context.load_texture(
                            "offscreen-base",
                            ColorImage::filled([1, 1], egui::Color32::WHITE),
                            TextureOptions::LINEAR,
                        )),
                        base_mode: Some(PreviewMode::Image),
                        base_bytes: 30 * 1024 * 1024,
                        last_used: Some(Instant::now()),
                        last_visible_frame: Some(1),
                        ..Default::default()
                    },
                ),
            ]),
            worker_demands: Arc::new(Mutex::new(HashMap::new())),
            paths: AppPaths::at(temporary.path()),
            context,
            next_request_id: 1,
            last_eviction: Instant::now(),
        };

        cache.evict_unused(9);

        let visible = cache.entries.get(&visible_id).expect("visible entry");
        assert!(visible.base_texture.is_some());
        assert!(visible.detail_texture.is_some());
        assert!(
            cache
                .entries
                .get(&offscreen_id)
                .expect("offscreen entry")
                .base_texture
                .is_none()
        );
    }

    #[test]
    fn worker_demand_rejects_superseded_jobs() {
        let id = Uuid::new_v4();
        let demands = Mutex::new(HashMap::from([(id, WorkerDemand { request_id: 2 })]));
        let job = PreviewJob {
            id,
            request_id: 1,
            source: PathBuf::from("source.png"),
            cache_directory: PathBuf::from("cache"),
            mode: PreviewMode::Image,
            edge: 512,
        };

        assert!(!worker_demand_is_current(&demands, &job));
    }
}
