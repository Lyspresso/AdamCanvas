//! On-demand, on-device photo analysis through Apple's Vision framework.
//!
//! This module is deliberately synchronous. Adam calls it only from a bounded
//! analysis worker, never from the canvas/UI thread. Keeping the native
//! boundary this small also leaves room for another engine without coupling
//! persisted photo details to a particular recognizer.

use anyhow::{Context as _, anyhow, ensure};
use objc2::{
    AnyThread,
    rc::{Retained, autoreleasepool},
};
use objc2_foundation::{NSArray, NSDictionary, NSString, NSURL};
use objc2_vision::{
    VNClassifyImageRequest, VNImageRequestHandler, VNRecognizeTextRequest, VNRequest,
    VNRequestTextRecognitionLevel,
};
use std::path::Path;

const CLASSIFICATION_MIN_CONFIDENCE: f32 = 0.05;
const MAX_RETURNED_CLASSIFICATIONS: usize = 12;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct VisionTextLine {
    pub text: String,
    pub confidence: f32,
    /// Normalized Vision coordinates, with the origin at the lower-left.
    pub bounds: [f32; 4],
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct VisionOcrOutput {
    pub text: String,
    pub lines: Vec<VisionTextLine>,
    pub mean_confidence: Option<f32>,
}

/// A raw, nonlocalized identifier from Apple's built-in image taxonomy.
///
/// Callers should map identifiers to user-facing prose rather than displaying
/// these technical labels directly.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct VisionImageClassification {
    pub identifier: String,
    pub confidence: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct VisionClassificationOutput {
    pub classifications: Vec<VisionImageClassification>,
    /// The implementation revision selected by Vision for this request.
    pub revision: usize,
}

pub(crate) fn recognize_text(path: &Path) -> anyhow::Result<VisionOcrOutput> {
    ensure!(path.is_file(), "photo is not available");

    autoreleasepool(|_| {
        let path_string = NSString::from_str(&path.to_string_lossy());
        let image_url = NSURL::fileURLWithPath(&path_string);
        let options = NSDictionary::new();
        let handler = unsafe {
            // SAFETY: `image_url` points at an immutable file for the lifetime
            // of this synchronous request and the options dictionary is empty.
            VNImageRequestHandler::initWithURL_options(
                VNImageRequestHandler::alloc(),
                &image_url,
                &options,
            )
        };

        let request = VNRecognizeTextRequest::new();
        request.setRecognitionLevel(VNRequestTextRecognitionLevel::Accurate);
        request.setUsesLanguageCorrection(true);
        request.setAutomaticallyDetectsLanguage(true);

        let erased: Retained<VNRequest> = request.clone().into_super().into_super();
        let requests = NSArray::from_retained_slice(&[erased]);
        handler
            .performRequests_error(&requests)
            .map_err(|error| anyhow!("{}", error.localizedDescription()))
            .with_context(|| format!("Vision could not read {}", path.display()))?;

        let observations = request
            .results()
            .ok_or_else(|| anyhow!("Vision returned no text result"))?;
        let mut lines = Vec::with_capacity(observations.len());
        for observation in observations.iter() {
            let Some(candidate) = observation.topCandidates(1).firstObject() else {
                continue;
            };
            let text = candidate.string().to_string();
            if text.trim().is_empty() {
                continue;
            }
            let rect = unsafe {
                // SAFETY: The observation is retained by `observations`, and
                // Vision documents this rectangle as a value-type property.
                observation.boundingBox()
            };
            lines.push(VisionTextLine {
                text,
                confidence: candidate.confidence(),
                bounds: [
                    rect.origin.x as f32,
                    rect.origin.y as f32,
                    rect.size.width as f32,
                    rect.size.height as f32,
                ],
            });
        }

        // Preserve Vision's observation order. A global row sort interleaves
        // unrelated columns in multi-column documents.
        Ok(output_from_lines(lines))
    })
}

/// Classifies a photo with Apple's built-in, on-device taxonomy.
///
/// Vision returns its entire taxonomy. Adam retains only the twelve strongest
/// positive results at or above 5% confidence so callers cannot accidentally
/// persist hundreds of effectively-zero observations.
pub(crate) fn classify_image(path: &Path) -> anyhow::Result<VisionClassificationOutput> {
    ensure!(path.is_file(), "photo is not available");

    autoreleasepool(|_| {
        let path_string = NSString::from_str(&path.to_string_lossy());
        let image_url = NSURL::fileURLWithPath(&path_string);
        let options = NSDictionary::new();
        let handler = unsafe {
            // SAFETY: `image_url` points at an immutable file for the lifetime
            // of this synchronous request and the options dictionary is empty.
            VNImageRequestHandler::initWithURL_options(
                VNImageRequestHandler::alloc(),
                &image_url,
                &options,
            )
        };

        let request = unsafe {
            // SAFETY: VNClassifyImageRequest is available on Adam's macOS 13
            // deployment target.
            VNClassifyImageRequest::new()
        };

        let erased: Retained<VNRequest> = request.clone().into_super().into_super();
        let requests = NSArray::from_retained_slice(&[erased]);
        handler
            .performRequests_error(&requests)
            .map_err(|error| anyhow!("{}", error.localizedDescription()))
            .with_context(|| format!("Vision could not classify {}", path.display()))?;

        let observations = unsafe {
            // SAFETY: The synchronous handler completed successfully and
            // `request` remains retained for the lifetime of its results.
            request.results()
        }
        .ok_or_else(|| anyhow!("Vision returned no classification result"))?;
        let revision = unsafe {
            // SAFETY: Vision owns this value-type request property, and the
            // request remains retained here.
            request.revision()
        };
        let classifications = observations
            .iter()
            .map(|observation| VisionImageClassification {
                identifier: unsafe {
                    // SAFETY: The observation is retained by `observations`.
                    observation.identifier()
                }
                .to_string(),
                confidence: unsafe {
                    // SAFETY: `confidence` is a value-type property inherited
                    // from VNObservation.
                    observation.confidence()
                },
            })
            .collect();

        Ok(VisionClassificationOutput {
            classifications: strongest_classifications(classifications),
            revision,
        })
    })
}

fn output_from_lines(lines: Vec<VisionTextLine>) -> VisionOcrOutput {
    let text = lines
        .iter()
        .map(|line| line.text.trim())
        .collect::<Vec<_>>()
        .join("\n");
    let mean_confidence = (!lines.is_empty())
        .then(|| lines.iter().map(|line| line.confidence).sum::<f32>() / lines.len() as f32);
    VisionOcrOutput {
        text,
        lines,
        mean_confidence,
    }
}

fn strongest_classifications(
    mut classifications: Vec<VisionImageClassification>,
) -> Vec<VisionImageClassification> {
    classifications.retain(|classification| {
        !classification.identifier.trim().is_empty()
            && classification.confidence.is_finite()
            && classification.confidence >= CLASSIFICATION_MIN_CONFIDENCE
    });
    classifications.sort_by(|left, right| {
        right
            .confidence
            .total_cmp(&left.confidence)
            .then_with(|| left.identifier.cmp(&right.identifier))
    });
    classifications.truncate(MAX_RETURNED_CLASSIFICATIONS);
    classifications
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_vision_observation_order_for_multi_column_text() {
        let output = output_from_lines(vec![
            VisionTextLine {
                text: "left first".into(),
                confidence: 0.9,
                bounds: [0.1, 0.8, 0.3, 0.1],
            },
            VisionTextLine {
                text: "left second".into(),
                confidence: 0.8,
                bounds: [0.1, 0.6, 0.3, 0.1],
            },
            VisionTextLine {
                text: "right first".into(),
                confidence: 0.7,
                bounds: [0.6, 0.8, 0.3, 0.1],
            },
        ]);

        assert_eq!(
            output.text, "left first\nleft second\nright first",
            "assembling OCR output must not row-sort separate columns"
        );
        assert_eq!(output.lines[1].text, "left second");
    }

    #[test]
    fn classification_output_is_filtered_sorted_and_bounded() {
        let mut candidates = vec![
            VisionImageClassification {
                identifier: "too-weak".into(),
                confidence: 0.049,
            },
            VisionImageClassification {
                identifier: "strongest".into(),
                confidence: 0.92,
            },
            VisionImageClassification {
                identifier: "not-a-number".into(),
                confidence: f32::NAN,
            },
        ];
        candidates.extend(
            (0..20)
                .map(|index| VisionImageClassification {
                    identifier: format!("candidate-{index:02}"),
                    confidence: 0.5 - index as f32 * 0.01,
                })
                .collect::<Vec<_>>(),
        );

        let output = strongest_classifications(candidates);

        assert_eq!(output.len(), MAX_RETURNED_CLASSIFICATIONS);
        assert_eq!(output[0].identifier, "strongest");
        assert!(
            output
                .windows(2)
                .all(|pair| pair[0].confidence >= pair[1].confidence)
        );
        assert!(
            output
                .iter()
                .all(|item| item.confidence >= CLASSIFICATION_MIN_CONFIDENCE)
        );
        assert!(!output.iter().any(|item| item.identifier == "too-weak"));
        assert!(!output.iter().any(|item| item.identifier == "not-a-number"));
    }

    /// Manual probe for real-world document photos without making the normal
    /// test suite depend on a machine-local fixture.
    #[test]
    #[ignore = "set ADAM_OCR_TEST_IMAGE to exercise Vision on a real photo"]
    fn recognizes_requested_document_photo() {
        let path = std::env::var_os("ADAM_OCR_TEST_IMAGE")
            .map(std::path::PathBuf::from)
            .expect("ADAM_OCR_TEST_IMAGE must name a local photo");
        let output = recognize_text(&path).expect("recognize document photo");
        assert!(!output.text.trim().is_empty());
        assert!(!output.lines.is_empty());
        eprintln!("{}", output.text);
    }

    /// Manual probe for Apple's built-in classification taxonomy.
    #[test]
    #[ignore = "set ADAM_CLASSIFICATION_TEST_IMAGE to classify a real photo"]
    fn classifies_requested_photo() {
        let path = std::env::var_os("ADAM_CLASSIFICATION_TEST_IMAGE")
            .map(std::path::PathBuf::from)
            .expect("ADAM_CLASSIFICATION_TEST_IMAGE must name a local photo");
        let output = classify_image(&path).expect("classify photo");
        assert!(output.classifications.len() <= MAX_RETURNED_CLASSIFICATIONS);
        eprintln!("Vision classification revision {}", output.revision);
        for classification in &output.classifications {
            eprintln!(
                "{} {:.3}",
                classification.identifier, classification.confidence
            );
        }
    }
}
