//! Bounded, on-demand OCR scheduling.
//!
//! No model or image work starts until the user explicitly asks for text in a
//! photo's Details window. One worker keeps difficult documents from
//! competing with Adam's render, import, or preview workers.

use crate::{
    domain::UnixMillis,
    photo_details::{PhotoOcrArtifact, PhotoVisualLabel},
};
use crossbeam_channel::{Receiver, Sender, TrySendError, bounded};
use egui::Context;
use std::{
    path::{Path, PathBuf},
    sync::{Arc, atomic::AtomicUsize},
    thread,
    time::{SystemTime, UNIX_EPOCH},
};
#[cfg(any(target_os = "macos", test))]
use std::{
    sync::{
        atomic::Ordering,
        mpsc::{RecvTimeoutError, sync_channel},
    },
    time::Duration,
};
use uuid::Uuid;

const OCR_QUEUE_CAPACITY: usize = 2;
const OCR_RESULT_CAPACITY: usize = 4;
#[cfg(target_os = "macos")]
const OCR_MAX_ACTIVE_RECOGNITIONS: usize = 2;
#[cfg(target_os = "macos")]
const OCR_RECOGNITION_TIMEOUT: Duration = Duration::from_secs(90);
#[cfg(any(target_os = "macos", test))]
const OCR_CAPACITY_MESSAGE: &str =
    "Text recognition is still finishing an earlier photo. Please try again in a moment.";
#[cfg(any(target_os = "macos", test))]
const OCR_TIMEOUT_MESSAGE: &str = "Text recognition took too long. Please try again.";
#[cfg(any(target_os = "macos", test))]
const OCR_STOPPED_MESSAGE: &str = "Text recognition stopped unexpectedly. Please try again.";

#[derive(Clone, Debug)]
pub struct PhotoOcrRequest {
    pub request_id: Uuid,
    pub tile_id: Uuid,
    pub path: PathBuf,
    pub source_fingerprint: String,
    pub media_revision: u64,
}

#[derive(Debug)]
pub struct PhotoOcrCompletion {
    pub request_id: Uuid,
    pub tile_id: Uuid,
    pub path: PathBuf,
    pub source_fingerprint: String,
    pub media_revision: u64,
    pub outcome: Result<PhotoOcrArtifact, String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OcrQueueError {
    Busy,
    Unavailable,
}

pub struct PhotoOcrWorker {
    jobs: Sender<PhotoOcrRequest>,
    results: Receiver<PhotoOcrCompletion>,
}

impl PhotoOcrWorker {
    pub fn start(context: Context) -> Self {
        let (jobs, job_receiver) = bounded::<PhotoOcrRequest>(OCR_QUEUE_CAPACITY);
        let (result_sender, results) = bounded::<PhotoOcrCompletion>(OCR_RESULT_CAPACITY);
        let active_recognitions = Arc::new(AtomicUsize::new(0));
        thread::Builder::new()
            .name("adam-photo-ocr".into())
            .spawn(move || {
                while let Ok(request) = job_receiver.recv() {
                    let outcome = run_request(&request, &active_recognitions);
                    let completion = PhotoOcrCompletion {
                        request_id: request.request_id,
                        tile_id: request.tile_id,
                        path: request.path,
                        source_fingerprint: request.source_fingerprint,
                        media_revision: request.media_revision,
                        outcome,
                    };
                    if result_sender.send(completion).is_err() {
                        break;
                    }
                    context.request_repaint();
                }
            })
            .expect("failed to start photo OCR worker");
        Self { jobs, results }
    }

    pub fn try_request(&self, request: PhotoOcrRequest) -> Result<(), OcrQueueError> {
        self.jobs.try_send(request).map_err(|error| match error {
            TrySendError::Full(_) => OcrQueueError::Busy,
            TrySendError::Disconnected(_) => OcrQueueError::Unavailable,
        })
    }

    pub fn poll(&self) -> impl Iterator<Item = PhotoOcrCompletion> + '_ {
        self.results.try_iter()
    }
}

pub fn source_fingerprint(path: &Path) -> std::io::Result<String> {
    let metadata = std::fs::metadata(path)?;
    let modified = metadata
        .modified()
        .map(system_time_fingerprint)
        .unwrap_or_else(|_| "unknown".into());
    let fingerprint = format!("v2:size={}:mtime={modified}", metadata.len());

    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt as _;

        return Ok(format!(
            "{fingerprint}:dev={}:ino={}:ctime={}.{:09}",
            metadata.dev(),
            metadata.ino(),
            metadata.ctime(),
            metadata.ctime_nsec()
        ));
    }

    #[cfg(not(unix))]
    Ok(fingerprint)
}

fn system_time_fingerprint(time: SystemTime) -> String {
    match time.duration_since(UNIX_EPOCH) {
        Ok(duration) => format!("{}.{:09}", duration.as_secs(), duration.subsec_nanos()),
        Err(error) => {
            let duration = error.duration();
            format!(
                "before-{}.{:09}",
                duration.as_secs(),
                duration.subsec_nanos()
            )
        }
    }
}

fn run_request(
    request: &PhotoOcrRequest,
    active_recognitions: &Arc<AtomicUsize>,
) -> Result<PhotoOcrArtifact, String> {
    let current_fingerprint =
        source_fingerprint(&request.path).map_err(|error| error.to_string())?;
    if current_fingerprint != request.source_fingerprint {
        return Err("The photo changed before text recognition began.".into());
    }

    #[cfg(target_os = "macos")]
    let (output, classification) = {
        let path = request.path.clone();
        run_supervised(
            active_recognitions,
            OCR_MAX_ACTIVE_RECOGNITIONS,
            OCR_RECOGNITION_TIMEOUT,
            move || {
                let text = crate::macos_vision::recognize_text(&path)
                    .map_err(|error| error.to_string())?;
                // Classification improves the visual prose, but OCR remains
                // useful even if the system taxonomy is unavailable.
                let classification = crate::macos_vision::classify_image(&path).ok();
                Ok((text, classification))
            },
        )?
    };

    #[cfg(not(target_os = "macos"))]
    {
        let _ = active_recognitions;
        return Err("No OCR engine is installed for this platform.".into());
    }

    #[cfg(target_os = "macos")]
    {
        let completed_fingerprint =
            source_fingerprint(&request.path).map_err(|error| error.to_string())?;
        if completed_fingerprint != request.source_fingerprint {
            return Err("The photo changed while text recognition was running.".into());
        }
        let visual_labels = classification
            .as_ref()
            .map(|output| {
                output
                    .classifications
                    .iter()
                    .map(|classification| PhotoVisualLabel {
                        identifier: classification.identifier.clone(),
                        confidence: classification.confidence,
                    })
                    .collect()
            })
            .unwrap_or_default();
        let classifier_revision = classification
            .as_ref()
            .map(|output| format!(" · VNClassifyImageRequest r{}", output.revision))
            .unwrap_or_default();
        let text = Arc::new(output.text);
        Ok(PhotoOcrArtifact {
            text: Arc::clone(&text),
            raw_text: Some(text),
            user_edited: false,
            engine: "Apple Vision".into(),
            engine_version: format!("VNRecognizeTextRequest · accurate{classifier_revision}"),
            recognized_at: now(),
            source_fingerprint: request.source_fingerprint.clone(),
            media_revision: request.media_revision,
            mean_confidence: output.mean_confidence,
            line_count: output.lines.len(),
            visual_labels,
        })
    }
}

#[cfg(any(target_os = "macos", test))]
struct ActiveRecognitionGuard(Arc<AtomicUsize>);

#[cfg(any(target_os = "macos", test))]
impl Drop for ActiveRecognitionGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

#[cfg(any(target_os = "macos", test))]
fn run_supervised<T, F>(
    active_recognitions: &Arc<AtomicUsize>,
    max_active: usize,
    timeout: Duration,
    task: F,
) -> Result<T, String>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, String> + Send + 'static,
{
    if active_recognitions
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |active| {
            (active < max_active).then_some(active + 1)
        })
        .is_err()
    {
        return Err(OCR_CAPACITY_MESSAGE.into());
    }

    let active_for_task = Arc::clone(active_recognitions);
    let (result_sender, result_receiver) = sync_channel(1);
    if thread::Builder::new()
        .name("adam-vision-ocr".into())
        .spawn(move || {
            let active_guard = ActiveRecognitionGuard(active_for_task);
            let result = task();
            drop(active_guard);
            let _ = result_sender.send(result);
        })
        .is_err()
    {
        active_recognitions.fetch_sub(1, Ordering::AcqRel);
        return Err(OCR_STOPPED_MESSAGE.into());
    }

    match result_receiver.recv_timeout(timeout) {
        Ok(result) => result,
        Err(RecvTimeoutError::Timeout) => Err(OCR_TIMEOUT_MESSAGE.into()),
        Err(RecvTimeoutError::Disconnected) => Err(OCR_STOPPED_MESSAGE.into()),
    }
}

fn now() -> UnixMillis {
    let milliseconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(i64::MAX as u128) as i64)
        .unwrap_or(0);
    UnixMillis(milliseconds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    #[test]
    fn source_fingerprint_changes_with_file_content_metadata() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(b"first").unwrap();
        file.flush().unwrap();
        let first = source_fingerprint(file.path()).unwrap();
        file.write_all(b"-second").unwrap();
        file.flush().unwrap();
        let second = source_fingerprint(file.path()).unwrap();
        assert_ne!(first, second);
    }

    #[cfg(unix)]
    #[test]
    fn source_fingerprint_includes_unix_file_identity() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(b"photo").unwrap();
        file.flush().unwrap();

        let fingerprint = source_fingerprint(file.path()).unwrap();

        assert!(fingerprint.starts_with("v2:size=5:mtime="));
        assert!(fingerprint.contains(":dev="));
        assert!(fingerprint.contains(":ino="));
        assert!(fingerprint.contains(":ctime="));
    }

    #[cfg(unix)]
    #[test]
    fn source_fingerprint_detects_same_size_same_mtime_replacement() {
        use std::{
            fs::{File, FileTimes},
            os::unix::fs::MetadataExt as _,
        };

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("photo.png");
        let replacement = directory.path().join("replacement.png");
        std::fs::write(&path, b"before").unwrap();
        let original_metadata = std::fs::metadata(&path).unwrap();
        let original_modified = original_metadata.modified().unwrap();
        let original_fingerprint = source_fingerprint(&path).unwrap();

        std::fs::write(&replacement, b"after!").unwrap();
        File::options()
            .write(true)
            .open(&replacement)
            .unwrap()
            .set_times(FileTimes::new().set_modified(original_modified))
            .unwrap();
        std::fs::rename(&replacement, &path).unwrap();

        let replacement_metadata = std::fs::metadata(&path).unwrap();
        let replacement_fingerprint = source_fingerprint(&path).unwrap();
        assert_eq!(original_metadata.len(), replacement_metadata.len());
        assert_eq!(
            system_time_fingerprint(original_modified),
            system_time_fingerprint(replacement_metadata.modified().unwrap())
        );
        assert_ne!(original_metadata.ino(), replacement_metadata.ino());
        assert_ne!(original_fingerprint, replacement_fingerprint);
    }

    #[test]
    fn queue_is_bounded_when_worker_is_not_consuming_fast_enough() {
        let (sender, _receiver) = bounded::<PhotoOcrRequest>(OCR_QUEUE_CAPACITY);
        let request = PhotoOcrRequest {
            request_id: Uuid::new_v4(),
            tile_id: Uuid::nil(),
            path: PathBuf::from("photo.png"),
            source_fingerprint: "v2:size=1:mtime=0.000000000".into(),
            media_revision: 1,
        };
        assert!(sender.try_send(request.clone()).is_ok());
        assert!(sender.try_send(request.clone()).is_ok());
        assert!(matches!(
            sender.try_send(request),
            Err(TrySendError::Full(_))
        ));
    }

    #[test]
    fn supervised_recognition_times_out_without_exceeding_thread_cap() {
        let active = Arc::new(AtomicUsize::new(0));
        let (release_sender, release_receiver) = sync_channel::<()>(0);

        let first = run_supervised(&active, 1, Duration::from_millis(10), move || {
            release_receiver.recv().unwrap();
            Ok(())
        });
        assert_eq!(first.unwrap_err(), OCR_TIMEOUT_MESSAGE);
        assert_eq!(active.load(Ordering::Acquire), 1);

        let second = run_supervised(&active, 1, Duration::from_secs(1), || Ok(()));
        assert_eq!(second.unwrap_err(), OCR_CAPACITY_MESSAGE);

        release_sender.send(()).unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(1);
        while active.load(Ordering::Acquire) != 0 && std::time::Instant::now() < deadline {
            thread::yield_now();
        }
        assert_eq!(active.load(Ordering::Acquire), 0);

        assert!(run_supervised(&active, 1, Duration::from_secs(1), || Ok(())).is_ok());
    }
}
