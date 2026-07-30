//! Efficient, color-managed photo downsampling through macOS Image I/O.
//!
//! `image` remains the portable decoder and fallback. This module exists for
//! the common macOS cold-cache path: Image I/O can decode a JPEG or HEIC
//! directly near the requested display size instead of first allocating the
//! full-resolution raster.

use anyhow::{Context as _, anyhow, ensure};
use image::{DynamicImage, RgbaImage};
use objc2_core_foundation::{
    CFBoolean, CFDictionary, CFNumber, CFType, CFURL, CGPoint, CGRect, CGSize,
};
use objc2_core_graphics::{
    CGBitmapContextCreate, CGColorSpace, CGContext, CGImage, CGImageAlphaInfo,
    CGImageByteOrderInfo, kCGColorSpaceSRGB,
};
use objc2_image_io::{
    CGImageSource, kCGImageSourceCreateThumbnailFromImageAlways,
    kCGImageSourceCreateThumbnailWithTransform, kCGImageSourceShouldCache,
    kCGImageSourceShouldCacheImmediately, kCGImageSourceThumbnailMaxPixelSize,
};
use std::{ffi::c_void, os::unix::ffi::OsStrExt, path::Path};

/// Decodes the first image in `path` into an orientation-correct, sRGB RGBA
/// thumbnail whose longest side is at most `max_pixel_size`.
///
/// Image I/O does not enlarge a source that is already smaller than the
/// requested size. Core Graphics returns premultiplied pixels, so this
/// converts them back to the straight-alpha representation expected by the
/// app's `image::DynamicImage` and disk-cache pipeline.
pub(crate) fn decode_thumbnail(path: &Path, max_pixel_size: u32) -> anyhow::Result<DynamicImage> {
    ensure!(max_pixel_size > 0, "thumbnail edge must be nonzero");

    let path_bytes = path.as_os_str().as_bytes();
    let url = unsafe {
        // SAFETY: `path_bytes` remains alive for this call and its exact byte
        // length is supplied. POSIX paths on macOS may contain non-UTF-8
        // bytes, which is why this avoids constructing a CFString.
        CFURL::from_file_system_representation(
            None,
            path_bytes.as_ptr(),
            path_bytes
                .len()
                .try_into()
                .context("image path is too long for Core Foundation")?,
            false,
        )
    }
    .ok_or_else(|| anyhow!("could not create an Image I/O file URL"))?;

    let source_cache_key = unsafe {
        // SAFETY: This process-lifetime framework constant is a valid
        // CFString option key.
        kCGImageSourceShouldCache
    };
    let source_options = CFDictionary::from_slices(&[source_cache_key], &[CFBoolean::new(false)]);
    let source = unsafe {
        // SAFETY: A CFURL is a valid CGImageSource input and the dictionary
        // contains the documented CFBoolean value. Disabling the source cache
        // prevents an accidental full-resolution raster from being retained;
        // the requested thumbnail itself is decoded immediately below.
        CGImageSource::with_url(&url, Some(source_options.as_opaque()))
    }
    .ok_or_else(|| anyhow!("Image I/O could not open {}", path.display()))?;

    let max_edge = CFNumber::new_i64(max_pixel_size.into());
    let keys: [&CFType; 4] = unsafe {
        // SAFETY: These process-lifetime framework constants are valid
        // CFString option keys.
        [
            kCGImageSourceCreateThumbnailFromImageAlways.as_ref(),
            kCGImageSourceCreateThumbnailWithTransform.as_ref(),
            kCGImageSourceShouldCacheImmediately.as_ref(),
            kCGImageSourceThumbnailMaxPixelSize.as_ref(),
        ]
    };
    let values: [&CFType; 4] = [
        CFBoolean::new(true).as_ref(),
        CFBoolean::new(true).as_ref(),
        CFBoolean::new(true).as_ref(),
        max_edge.as_ref(),
    ];
    let options = CFDictionary::<CFType, CFType>::from_slices(&keys, &values);
    let thumbnail = unsafe {
        // SAFETY: The dictionary values match the documented Image I/O option
        // types: three CFBooleans and one CFNumber.
        source.thumbnail_at_index(0, Some(options.as_opaque()))
    }
    .ok_or_else(|| anyhow!("Image I/O could not decode {}", path.display()))?;

    cg_image_to_srgb_rgba(&thumbnail)
}

fn cg_image_to_srgb_rgba(image: &CGImage) -> anyhow::Result<DynamicImage> {
    let width = CGImage::width(Some(image));
    let height = CGImage::height(Some(image));
    ensure!(width > 0 && height > 0, "Image I/O returned an empty image");
    ensure!(
        width <= u32::MAX as usize && height <= u32::MAX as usize,
        "thumbnail dimensions exceed the image crate limits"
    );

    let bytes_per_row = width.checked_mul(4).context("thumbnail row is too large")?;
    let byte_count = bytes_per_row
        .checked_mul(height)
        .context("thumbnail is too large")?;
    let mut rgba = vec![0_u8; byte_count];

    let color_space = CGColorSpace::with_name(Some(unsafe {
        // SAFETY: This is a process-lifetime Core Graphics CFString.
        kCGColorSpaceSRGB
    }))
    .ok_or_else(|| anyhow!("Core Graphics could not create an sRGB color space"))?;
    let bitmap_info = CGImageAlphaInfo::PremultipliedLast.0 | CGImageByteOrderInfo::Order32Big.0;
    let context = unsafe {
        // SAFETY: `rgba` owns `byte_count` writable bytes and is not moved,
        // resized, or read until the synchronous draw completes and the
        // context is dropped. Its row stride and bitmap description agree.
        CGBitmapContextCreate(
            rgba.as_mut_ptr().cast::<c_void>(),
            width,
            height,
            8,
            bytes_per_row,
            Some(&color_space),
            bitmap_info,
        )
    }
    .ok_or_else(|| anyhow!("Core Graphics could not allocate an RGBA context"))?;

    CGContext::draw_image(
        Some(&context),
        CGRect::new(CGPoint::ZERO, CGSize::new(width as f64, height as f64)),
        Some(image),
    );
    drop(context);

    unpremultiply_rgba(&mut rgba);
    let pixels = RgbaImage::from_raw(width as u32, height as u32, rgba)
        .ok_or_else(|| anyhow!("Core Graphics returned an invalid RGBA buffer"))?;
    Ok(DynamicImage::ImageRgba8(pixels))
}

fn unpremultiply_rgba(rgba: &mut [u8]) {
    for pixel in rgba.chunks_exact_mut(4) {
        let alpha = u32::from(pixel[3]);
        match alpha {
            0 => pixel[..3].fill(0),
            255 => {}
            _ => {
                for channel in &mut pixel[..3] {
                    let straight = (u32::from(*channel) * 255 + alpha / 2) / alpha;
                    *channel = straight.min(255) as u8;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GenericImageView as _, Rgb, RgbImage, Rgba};
    use std::fs;

    fn insert_exif_orientation(path: &Path, orientation: u8) {
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
        let segment_length = (exif.len() + 2) as u16;
        let mut oriented = Vec::with_capacity(encoded.len() + exif.len() + 4);
        oriented.extend_from_slice(&encoded[..2]);
        oriented.extend_from_slice(&[0xff, 0xe1]);
        oriented.extend_from_slice(&segment_length.to_be_bytes());
        oriented.append(&mut exif);
        oriented.extend_from_slice(&encoded[2..]);
        fs::write(path, oriented).expect("write oriented JPEG");
    }

    fn assert_rgb_near(actual: Rgba<u8>, expected: [u8; 3]) {
        for (actual, expected) in actual.0[..3].iter().zip(expected) {
            assert!(
                actual.abs_diff(expected) <= 35,
                "channel {actual} was not near {expected}"
            );
        }
    }

    #[test]
    fn applies_exif_rotation_and_preserves_corner_order() {
        let temporary = tempfile::tempdir().expect("temporary directory");
        let path = temporary.path().join("rotated.jpg");
        let mut source = RgbImage::from_pixel(80, 40, Rgb([0, 0, 0]));
        for y in 0..20 {
            for x in 0..40 {
                source.put_pixel(x, y, Rgb([255, 0, 0]));
                source.put_pixel(x + 40, y, Rgb([0, 255, 0]));
                source.put_pixel(x, y + 20, Rgb([0, 0, 255]));
                source.put_pixel(x + 40, y + 20, Rgb([255, 255, 255]));
            }
        }
        source
            .save_with_format(&path, image::ImageFormat::Jpeg)
            .expect("save JPEG source");
        insert_exif_orientation(&path, 6);

        let decoded = decode_thumbnail(&path, 200).expect("native thumbnail");

        assert_eq!(decoded.dimensions(), (40, 80));
        assert_rgb_near(decoded.get_pixel(5, 5), [0, 0, 255]);
        assert_rgb_near(decoded.get_pixel(34, 5), [255, 0, 0]);
        assert_rgb_near(decoded.get_pixel(5, 74), [255, 255, 255]);
        assert_rgb_near(decoded.get_pixel(34, 74), [0, 255, 0]);
    }

    #[test]
    fn preserves_top_to_bottom_rgba_order_and_does_not_upscale() {
        let temporary = tempfile::tempdir().expect("temporary directory");
        let path = temporary.path().join("corners.png");
        let mut source = image::RgbaImage::from_pixel(40, 20, Rgba([0, 0, 0, 255]));
        source.put_pixel(0, 0, Rgba([255, 0, 0, 255]));
        source.put_pixel(39, 0, Rgba([0, 255, 0, 255]));
        source.put_pixel(0, 19, Rgba([0, 0, 255, 255]));
        source.put_pixel(39, 19, Rgba([255, 255, 255, 255]));
        source.save(&path).expect("save PNG source");

        let decoded = decode_thumbnail(&path, 256).expect("native thumbnail");

        assert_eq!(decoded.dimensions(), (40, 20));
        assert_eq!(decoded.get_pixel(0, 0), Rgba([255, 0, 0, 255]));
        assert_eq!(decoded.get_pixel(39, 0), Rgba([0, 255, 0, 255]));
        assert_eq!(decoded.get_pixel(0, 19), Rgba([0, 0, 255, 255]));
        assert_eq!(decoded.get_pixel(39, 19), Rgba([255, 255, 255, 255]));
    }

    #[test]
    fn unpremultiplies_translucent_pixels_for_the_image_pipeline() {
        let mut pixels = [64, 32, 16, 128, 0, 0, 0, 0, 10, 20, 30, 255];
        unpremultiply_rgba(&mut pixels);
        assert_eq!(pixels, [128, 64, 32, 128, 0, 0, 0, 0, 10, 20, 30, 255]);
    }
}
