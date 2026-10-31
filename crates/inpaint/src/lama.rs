//! Local LaMa (Large Mask Inpainting) ONNX Runtime (`ort`) inference + Frequency-Separated
//! detail synthesis with 512×512 / 1024×1024 context rescaling and ISO-adaptive sensor grain.
//!
//! Loads `lama_fp32.onnx` lazily into a persistent thread-safe `ort::session::Session` on first use
//! and runs Fast Fourier Convolution neural inpainting on `[1, 3, 512, 512]` crops, combined with
//! high-frequency boundary texture continuity and ISO sensor grain injection.

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use crate::{CompleteParams, complete_guided_nnf, membrane_fill};

/// Configuration for a local LaMa inpainting pass on a cropped context bounding box.
#[derive(Clone, Debug)]
pub struct LamaConfig {
    /// Target working resolution cap (512 or 1024). Crops larger than `max_side` are downsampled
    /// to `max_side` for fast inference and guided-upsampled back to native crop resolution.
    pub max_side: usize,
    /// Deterministic seed for texture & ISO grain synthesis.
    pub seed: u64,
    /// Optional explicit ISO sensitivity (e.g., 100..12800). When `0`, ISO sensor grain is
    /// estimated automatically from the unmasked context ring's micro-variance.
    pub iso: u32,
}

impl Default for LamaConfig {
    fn default() -> Self {
        Self {
            max_side: 512,
            seed: 0x4C41_4D41_2026,
            iso: 0,
        }
    }
}

/// Checks if a local ONNX LaMa model file exists on disk.
pub fn local_onnx_model_path() -> Option<PathBuf> {
    if let Ok(custom) = std::env::var("LIGHTCRAFT_LAMA_MODEL") {
        let p = PathBuf::from(custom);
        if p.is_file() {
            return Some(p);
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        for rel in [
            ".mm-studio/models/lama_fp32.onnx",
            ".mm-studio/models/lama.onnx",
            ".lightcraft/models/lama_fp32.onnx",
        ] {
            let p = PathBuf::from(&home).join(rel);
            if p.is_file() {
                return Some(p);
            }
        }
    }
    let ws_asset = PathBuf::from("/Users/mcruz/lightcraft/assets/models/lama_fp32.onnx");
    if ws_asset.is_file() {
        return Some(ws_asset);
    }
    if let Ok(exe) = std::env::current_exe() {
        for anc in exe.ancestors().take(6) {
            let candidate = anc.join("assets/models/lama_fp32.onnx");
            if candidate.is_file() {
                return Some(candidate);
            }
            let res_candidate = anc.join("Resources/models/lama_fp32.onnx");
            if res_candidate.is_file() {
                return Some(res_candidate);
            }
        }
    }
    None
}

#[cfg(target_os = "macos")]
static LAMA_ORT_SESSION: OnceLock<Mutex<Option<ort::session::Session>>> = OnceLock::new();

#[cfg(target_os = "macos")]
fn get_or_init_lama_session() -> &'static Mutex<Option<ort::session::Session>> {
    LAMA_ORT_SESSION.get_or_init(|| {
        let Some(model_path) = local_onnx_model_path() else {
            return Mutex::new(None);
        };
        let session = ort::session::Session::builder()
            .and_then(|b| b.with_intra_threads(4))
            .and_then(|b| b.commit_from_file(&model_path))
            .ok();
        Mutex::new(session)
    })
}

/// Runs neural inference on `rgb` (`w × h × 3`) and `hole` (`w × h`) using the ONNX Runtime
/// `lama_fp32.onnx` model (`[1, 3, 512, 512]` + `[1, 1, 512, 512]` -> `[1, 3, 512, 512]`).
/// Converts linear scene values to perceptual sRGB `[0, 1]` for the neural net and maps back
/// to the crop's native linear exposure scale.
#[cfg(target_os = "macos")]
pub fn run_lama_onnx_512(rgb: &[f32], hole: &[bool], w: usize, h: usize) -> Option<Vec<f32>> {
    let mtx = get_or_init_lama_session();
    let mut guard = mtx.lock().ok()?;
    let session = guard.as_mut()?;

    const SIDE: usize = 512;
    let n = SIDE * SIDE;

    // Determine scene-linear 99.5th-percentile peak outside the hole so dark unexposed RAW crops
    // utilize the full [0, 1] sRGB range expected by LaMa while preserving bright highlights
    let mut max_samples = Vec::with_capacity(1024);
    let step = ((w * h) / 1024).max(1);
    let mut abs_max = 1e-4f32;
    for i in (0..(w * h)).step_by(step) {
        if !hole[i] {
            let idx = i * 3;
            let m = rgb[idx].max(rgb[idx + 1]).max(rgb[idx + 2]);
            if m.is_finite() && m > 0.0 {
                max_samples.push(m);
                if m > abs_max {
                    abs_max = m;
                }
            }
        }
    }
    let peak = if max_samples.len() >= 16 {
        max_samples.sort_by(|a, b| a.total_cmp(b));
        let p995 = max_samples[(max_samples.len() * 995 / 1000).min(max_samples.len() - 1)];
        (p995 * 1.03).max(abs_max.min(1.2)).clamp(0.08, 16.0)
    } else {
        abs_max.max(1.0)
    };
    let inv_peak = 1.0 / peak;

    // Reflect-pad rectangular (w, h) to a square (sq × sq) so aspect ratio is never distorted in 512×512
    let sq = w.max(h).max(1);
    let pad_x = (sq - w) / 2;
    let pad_y = (sq - h) / 2;
    let reflect_idx = |c: isize, len: usize| -> usize {
        if len <= 1 {
            return 0;
        }
        let mut v = c;
        if v < 0 {
            v = -v;
        }
        let period = (len * 2 - 2).max(1) as isize;
        let m = v.rem_euclid(period) as usize;
        if m >= len { period as usize - m } else { m }
    };

    let (sq_rgb, sq_mask) = if w == sq && h == sq {
        (rgb.to_vec(), hole.to_vec())
    } else {
        let mut s_rgb = vec![0.0f32; sq * sq * 3];
        let mut s_msk = vec![false; sq * sq];
        for sy in 0..sq {
            let src_y = reflect_idx(sy as isize - pad_y as isize, h);
            for sx in 0..sq {
                let src_x = reflect_idx(sx as isize - pad_x as isize, w);
                let si = src_y * w + src_x;
                let di = sy * sq + sx;
                s_rgb[di * 3] = rgb[si * 3];
                s_rgb[di * 3 + 1] = rgb[si * 3 + 1];
                s_rgb[di * 3 + 2] = rgb[si * 3 + 2];
                // Critical: if a reflected pixel originates from inside `hole`, mark it masked too
                // so reflected copies of border-touching objects never leak as unmasked context!
                s_msk[di] = hole[si];
            }
        }
        (s_rgb, s_msk)
    };

    // Rescale square crop to 512x512
    let rgb_512 = if sq == SIDE {
        sq_rgb
    } else {
        resize_rgb(&sq_rgb, sq, sq, SIDE, SIDE)
    };
    let mask_512 = if sq == SIDE {
        sq_mask
    } else {
        resize_mask(&sq_mask, sq, sq, SIDE, SIDE)
    };

    // Dilate the binary mask by 2px at 512x512 so resize anti-aliasing never leaks into LaMa's encoder
    let mut mask_dilated = mask_512.clone();
    for y in 0..SIDE {
        for x in 0..SIDE {
            if mask_512[y * SIDE + x] {
                for dy in -2isize..=2 {
                    for dx in -2isize..=2 {
                        if dx * dx + dy * dy <= 5 {
                            let ny = (y as isize + dy).clamp(0, (SIDE - 1) as isize) as usize;
                            let nx = (x as isize + dx).clamp(0, (SIDE - 1) as isize) as usize;
                            mask_dilated[ny * SIDE + nx] = true;
                        }
                    }
                }
            }
        }
    }

    // Build NCHW tensors: image [1, 3, 512, 512] in perceptual sRGB [0, 1], mask [1, 1, 512, 512] in {0.0, 1.0}
    let mut img_nchw = vec![0.0f32; 3 * n];
    let mut mask_nchw = vec![0.0f32; n];
    for i in 0..n {
        let m = if mask_dilated[i] { 1.0f32 } else { 0.0f32 };
        mask_nchw[i] = m;
        // LaMa expects unmasked pixels in [0, 1] sRGB; zeroing masked pixels explicitly
        if m < 0.5 {
            let r_srgb = (rgb_512[i * 3] * inv_peak).clamp(0.0, 1.0).powf(1.0 / 2.2);
            let g_srgb = (rgb_512[i * 3 + 1] * inv_peak).clamp(0.0, 1.0).powf(1.0 / 2.2);
            let b_srgb = (rgb_512[i * 3 + 2] * inv_peak).clamp(0.0, 1.0).powf(1.0 / 2.2);
            img_nchw[i] = r_srgb;
            img_nchw[n + i] = g_srgb;
            img_nchw[2 * n + i] = b_srgb;
        }
    }

    let img_tensor = ort::value::Tensor::from_array(([1usize, 3, SIDE, SIDE], img_nchw.into_boxed_slice())).ok()?;
    let mask_tensor = ort::value::Tensor::from_array(([1usize, 1, SIDE, SIDE], mask_nchw.into_boxed_slice())).ok()?;

    let outputs = session
        .run(ort::inputs![
            "image" => img_tensor,
            "mask" => mask_tensor,
        ])
        .ok()?;

    let (_, out_slice) = outputs[0].try_extract_tensor::<f32>().ok()?;
    if out_slice.len() < 3 * n {
        return None;
    }

    // Detect if model outputs [0, 255] (standard Carve/LaMa-ONNX export) or [0, 1]
    let mut max_out = 0.0f32;
    for &v in out_slice.iter().step_by(17) {
        if v.is_finite() && v > max_out {
            max_out = v;
        }
    }
    let out_norm = if max_out > 2.0 { 1.0 / 255.0 } else { 1.0 };

    // Convert NCHW sRGB back to interleaved linear RGB [512 × 512 × 3]
    let mut out_512 = vec![0.0f32; n * 3];
    for i in 0..n {
        let r_s = (out_slice[i] * out_norm).clamp(0.0, 1.0);
        let g_s = (out_slice[n + i] * out_norm).clamp(0.0, 1.0);
        let b_s = (out_slice[2 * n + i] * out_norm).clamp(0.0, 1.0);
        out_512[i * 3] = r_s.powf(2.2) * peak;
        out_512[i * 3 + 1] = g_s.powf(2.2) * peak;
        out_512[i * 3 + 2] = b_s.powf(2.2) * peak;
    }

    // Upsample 512×512 back to sq×sq with bicubic Catmull-Rom interpolation and extract (w, h)
    let out_sq = if sq == SIDE {
        out_512
    } else {
        resize_rgb_bicubic(&out_512, SIDE, SIDE, sq, sq)
    };
    let out_native = if w == sq && h == sq {
        out_sq
    } else {
        let mut sub = vec![0.0f32; w * h * 3];
        for y in 0..h {
            let sy = y + pad_y;
            for x in 0..w {
                let sx = x + pad_x;
                let si = (sy * sq + sx) * 3;
                let di = (y * w + x) * 3;
                sub[di] = out_sq[si];
                sub[di + 1] = out_sq[si + 1];
                sub[di + 2] = out_sq[si + 2];
            }
        }
        sub
    };
    Some(out_native)
}

#[cfg(not(target_os = "macos"))]
pub fn run_lama_onnx_512(_rgb: &[f32], _hole: &[bool], _w: usize, _h: usize) -> Option<Vec<f32>> {
    None
}

/// Bicubic (Catmull-Rom) resize of an interleaved 3-channel float buffer `src` (`sw × sh`) to `dw × dh`.
fn resize_rgb_bicubic(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    if sw < 4 || sh < 4 || dw == 0 || dh == 0 || src.len() < sw.saturating_mul(sh).saturating_mul(3) {
        return resize_rgb(src, sw, sh, dw, dh);
    }
    if sw == dw && sh == dh {
        return src.to_vec();
    }
    let catmull = |t: f32| -> [f32; 4] {
        let t2 = t * t;
        let t3 = t2 * t;
        [
            -0.5 * t + t2 - 0.5 * t3,
            1.0 - 2.5 * t2 + 1.5 * t3,
            0.5 * t + 2.0 * t2 - 1.5 * t3,
            -0.5 * t2 + 0.5 * t3,
        ]
    };
    let mut dst = vec![0.0f32; dw * dh * 3];
    let sx = sw as f32 / dw as f32;
    let sy = sh as f32 / dh as f32;
    let max_x = (sw - 1) as isize;
    let max_y = (sh - 1) as isize;
    for y in 0..dh {
        let fy = ((y as f32 + 0.5) * sy - 0.5).clamp(0.0, max_y as f32);
        let y1 = fy.floor() as isize;
        let ty = fy - y1 as f32;
        let wy = catmull(ty);
        let ys = [
            (y1 - 1).clamp(0, max_y) as usize,
            y1.clamp(0, max_y) as usize,
            (y1 + 1).clamp(0, max_y) as usize,
            (y1 + 2).clamp(0, max_y) as usize,
        ];
        for x in 0..dw {
            let fx = ((x as f32 + 0.5) * sx - 0.5).clamp(0.0, max_x as f32);
            let x1 = fx.floor() as isize;
            let tx = fx - x1 as f32;
            let wx = catmull(tx);
            let xs = [
                (x1 - 1).clamp(0, max_x) as usize,
                x1.clamp(0, max_x) as usize,
                (x1 + 1).clamp(0, max_x) as usize,
                (x1 + 2).clamp(0, max_x) as usize,
            ];
            let mut acc = [0.0f32; 3];
            for (j, &yy) in ys.iter().enumerate() {
                let row_off = yy * sw;
                let wj = wy[j];
                for (i, &xx) in xs.iter().enumerate() {
                    let w = wj * wx[i];
                    let idx = (row_off + xx) * 3;
                    acc[0] += src[idx] * w;
                    acc[1] += src[idx + 1] * w;
                    acc[2] += src[idx + 2] * w;
                }
            }
            let out_idx = (y * dw + x) * 3;
            dst[out_idx] = acc[0].max(0.0);
            dst[out_idx + 1] = acc[1].max(0.0);
            dst[out_idx + 2] = acc[2].max(0.0);
        }
    }
    dst
}

/// Bilinearly resizes an interleaved 3-channel float buffer `src` (`sw × sh`) to `dw × dh`.
fn resize_rgb(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    if sw == 0 || sh == 0 || dw == 0 || dh == 0 || src.len() < sw.saturating_mul(sh).saturating_mul(3) {
        return vec![0.0; dw.saturating_mul(dh).saturating_mul(3)];
    }
    if sw == dw && sh == dh {
        return src.to_vec();
    }
    let mut dst = vec![0.0f32; dw * dh * 3];
    let sx = sw as f32 / dw as f32;
    let sy = sh as f32 / dh as f32;
    for y in 0..dh {
        let fy = ((y as f32 + 0.5) * sy - 0.5).clamp(0.0, (sh - 1) as f32);
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let ty = fy - y0 as f32;
        for x in 0..dw {
            let fx = ((x as f32 + 0.5) * sx - 0.5).clamp(0.0, (sw - 1) as f32);
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let tx = fx - x0 as f32;
            let i00 = (y0 * sw + x0) * 3;
            let i10 = (y0 * sw + x1) * 3;
            let i01 = (y1 * sw + x0) * 3;
            let i11 = (y1 * sw + x1) * 3;
            let out_idx = (y * dw + x) * 3;
            for c in 0..3 {
                let top = src[i00 + c] * (1.0 - tx) + src[i10 + c] * tx;
                let bot = src[i01 + c] * (1.0 - tx) + src[i11 + c] * tx;
                dst[out_idx + c] = top * (1.0 - ty) + bot * ty;
            }
        }
    }
    dst
}

/// Resizes a binary mask (`sw × sh`) to `dw × dh` using area/bilinear thresholding.
fn resize_mask(mask: &[bool], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<bool> {
    if sw == 0 || sh == 0 || dw == 0 || dh == 0 || mask.len() < sw.saturating_mul(sh) {
        return vec![false; dw.saturating_mul(dh)];
    }
    if sw == dw && sh == dh {
        return mask.to_vec();
    }
    let mut dst = vec![false; dw * dh];
    let sx = sw as f32 / dw as f32;
    let sy = sh as f32 / dh as f32;
    for y in 0..dh {
        let fy = ((y as f32 + 0.5) * sy - 0.5).clamp(0.0, (sh - 1) as f32);
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let ty = fy - y0 as f32;
        for x in 0..dw {
            let fx = ((x as f32 + 0.5) * sx - 0.5).clamp(0.0, (sw - 1) as f32);
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let tx = fx - x0 as f32;
            let v00 = if mask[y0 * sw + x0] { 1.0 } else { 0.0 };
            let v10 = if mask[y0 * sw + x1] { 1.0 } else { 0.0 };
            let v01 = if mask[y1 * sw + x0] { 1.0 } else { 0.0 };
            let v11 = if mask[y1 * sw + x1] { 1.0 } else { 0.0 };
            let v = (v00 * (1.0 - tx) + v10 * tx) * (1.0 - ty) + (v01 * (1.0 - tx) + v11 * tx) * ty;
            dst[y * dw + x] = v >= 0.35;
        }
    }
    dst
}

/// Two-pass separable tent/Gaussian smoothing on a 3-channel interleaved buffer (`w × h × 3`)
/// using $O(1)$-per-pixel sliding window accumulation (zero box-filter ringing).
fn blur_rgb3(buf: &[f32], w: usize, h: usize, radius: usize) -> Vec<f32> {
    if radius == 0 || w <= 1 || h <= 1 || buf.len() < w.saturating_mul(h).saturating_mul(3) {
        return buf.to_vec();
    }
    let r = (radius * 2 / 3).min(w.max(h) / 2).max(1);
    let mut cur = buf.to_vec();
    let mut tmp = vec![0.0f32; w * h * 3];
    for _pass in 0..2 {
        // Horizontal pass
        for y in 0..h {
            let row_off = y * w;
            for x in 0..w {
                let x0 = x.saturating_sub(r);
                let x1 = (x + r).min(w - 1);
                let inv_c = 1.0 / ((x1 - x0 + 1) as f32);
                let mut sum = [0.0f32; 3];
                for xx in x0..=x1 {
                    let idx = (row_off + xx) * 3;
                    sum[0] += cur[idx];
                    sum[1] += cur[idx + 1];
                    sum[2] += cur[idx + 2];
                }
                let out_idx = (row_off + x) * 3;
                tmp[out_idx] = sum[0] * inv_c;
                tmp[out_idx + 1] = sum[1] * inv_c;
                tmp[out_idx + 2] = sum[2] * inv_c;
            }
        }
        // Vertical pass
        for y in 0..h {
            let y0 = y.saturating_sub(r);
            let y1 = (y + r).min(h - 1);
            let inv_c = 1.0 / ((y1 - y0 + 1) as f32);
            for x in 0..w {
                let mut sum = [0.0f32; 3];
                for yy in y0..=y1 {
                    let idx = (yy * w + x) * 3;
                    sum[0] += tmp[idx];
                    sum[1] += tmp[idx + 1];
                    sum[2] += tmp[idx + 2];
                }
                let out_idx = (y * w + x) * 3;
                cur[out_idx] = sum[0] * inv_c;
                cur[out_idx + 1] = sum[1] * inv_c;
                cur[out_idx + 2] = sum[2] * inv_c;
            }
        }
    }
    cur
}

/// Synthesizes 1:1 native-resolution (`w × h`) high- and mid-frequency texture across `hole`
/// using boundary-outward (onion-peel) overlapping patch quilting (Efros & Freeman) on the
/// detail band `rgb - base_low`, preserving the smooth illumination/structure of `base_low`.
pub fn quilt_native_detail_in_hole(
    rgb: &[f32],
    base: &[f32],
    hole: &[bool],
    w: usize,
    h: usize,
    r_detail: usize,
    is_outpaint: bool,
    seed: u64,
) -> Vec<f32> {
    if w < 24 || h < 24 || rgb.len() < w * h * 3 || base.len() < w * h * 3 || hole.len() < w * h {
        return base.to_vec();
    }
    let b_sz = ((r_detail * 2).clamp(12, 28)) & !1;
    let stride = (b_sz / 2).max(6);
    if w < b_sz * 2 || h < b_sz * 2 {
        return base.to_vec();
    }

    // 1. Decompose `base` (which equals `rgb` on `!hole` and is C1-continuous across the boundary)
    //    into smooth macro illumination `base_low` and native detail `src_detail = base - base_low`.
    let base_low = blur_rgb3(base, w, h, r_detail.max(3));
    let mut src_detail = vec![0.0f32; w * h * 3];
    for i in 0..(w * h) {
        let idx = i * 3;
        if !hole[i] {
            src_detail[idx] = rgb[idx] - base_low[idx];
            src_detail[idx + 1] = rgb[idx + 1] - base_low[idx + 1];
            src_detail[idx + 2] = rgb[idx + 2] - base_low[idx + 2];
        } else {
            src_detail[idx] = base[idx] - base_low[idx];
            src_detail[idx + 1] = base[idx + 1] - base_low[idx + 1];
            src_detail[idx + 2] = base[idx + 2] - base_low[idx + 2];
        }
    }

    // 2. Dilate `hole` by 2px and build a summed-area table so candidate source patches
    //    are strictly inside the clean unmasked photo and never touch the hole seam.
    let mut hole_dil = hole.to_vec();
    let mut vx0 = w;
    let mut vy0 = h;
    let mut vx1 = 0usize;
    let mut vy1 = 0usize;
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if !hole[i] {
                vx0 = vx0.min(x);
                vy0 = vy0.min(y);
                vx1 = vx1.max(x);
                vy1 = vy1.max(y);
            } else {
                for dy in -2isize..=2 {
                    for dx in -2isize..=2 {
                        let nx = (x as isize + dx).clamp(0, (w - 1) as isize) as usize;
                        let ny = (y as isize + dy).clamp(0, (h - 1) as isize) as usize;
                        hole_dil[ny * w + nx] = true;
                    }
                }
            }
        }
    }
    if vx0 >= vx1 || vy0 >= vy1 {
        return base.to_vec();
    }

    let mut sat = vec![0u32; (w + 1) * (h + 1)];
    for y in 0..h {
        let mut row = 0u32;
        for x in 0..w {
            row += u32::from(hole_dil[y * w + x]);
            sat[(y + 1) * (w + 1) + x + 1] = sat[y * (w + 1) + x + 1] + row;
        }
    }
    let rect_sum = |x0: usize, y0: usize, x1: usize, y1: usize| -> u32 {
        let x1 = x1.min(w);
        let y1 = y1.min(h);
        if x1 <= x0 || y1 <= y0 {
            return 1;
        }
        let w1 = w + 1;
        sat[y1 * w1 + x1] + sat[y0 * w1 + x0] - sat[y0 * w1 + x1] - sat[y1 * w1 + x0]
    };

    // Collect valid unmasked `b_sz × b_sz` source patch top-left coordinates
    let step_cand = (stride / 2).max(4);
    let mut valid_sources = Vec::new();
    for sy in (0..=h.saturating_sub(b_sz)).step_by(step_cand) {
        for sx in (0..=w.saturating_sub(b_sz)).step_by(step_cand) {
            if rect_sum(sx, sy, sx + b_sz, sy + b_sz) == 0 {
                valid_sources.push((sx, sy));
            }
        }
    }
    if valid_sources.is_empty() {
        return base.to_vec();
    }

    // 3. Pre-seed `acc_detail` and `acc_weight` on unmasked pixels (`!hole`) so the first
    //    ring of tiles stitches directly onto the real photo's native texture across the boundary.
    let mut dist_in = vec![0u8; w * h];
    for i in 0..(w * h) {
        if !hole[i] {
            dist_in[i] = stride as u8;
        }
    }
    for pass in 1..=(stride as u8) {
        let prev = pass - 1;
        for y in 1..(h - 1) {
            for x in 1..(w - 1) {
                let i = y * w + x;
                if dist_in[i] == stride as u8
                    && (dist_in[i - 1] == prev
                        || dist_in[i + 1] == prev
                        || dist_in[i - w] == prev
                        || dist_in[i + w] == prev)
                {
                    dist_in[i] = pass;
                }
            }
        }
    }

    let mut acc_detail = vec![0.0f32; w * h * 3];
    let mut acc_weight = vec![0.0f32; w * h];
    let mut acc_weight2 = vec![0.0f32; w * h];
    for i in 0..(w * h) {
        if !hole[i] {
            let t = (dist_in[i] as f32 / stride as f32).clamp(0.05, 1.0);
            let w_pre = t * t * (3.0 - 2.0 * t);
            acc_detail[i * 3] = src_detail[i * 3] * w_pre;
            acc_detail[i * 3 + 1] = src_detail[i * 3 + 1] * w_pre;
            acc_detail[i * 3 + 2] = src_detail[i * 3 + 2] * w_pre;
            acc_weight[i] = w_pre;
            acc_weight2[i] = w_pre * w_pre;
        }
    }

    // 4. Generate overlapping target tiles `(tx, ty)` and sort them in Boundary-Outward (Onion-Peel) order
    let mut xs: Vec<usize> = (0..=w.saturating_sub(b_sz)).step_by(stride).collect();
    if xs.last().copied() != Some(w - b_sz) {
        xs.push(w - b_sz);
    }
    let mut ys: Vec<usize> = (0..=h.saturating_sub(b_sz)).step_by(stride).collect();
    if ys.last().copied() != Some(h - b_sz) {
        ys.push(h - b_sz);
    }

    let mut tiles: Vec<(u32, usize, usize)> = Vec::new();
    for &ty in &ys {
        for &tx in &xs {
            let mut hole_cnt = 0u32;
            for v in 0..b_sz {
                let row = (ty + v) * w + tx;
                for u in 0..b_sz {
                    if hole[row + u] {
                        hole_cnt += 1;
                    }
                }
            }
            if hole_cnt == 0 {
                continue;
            }
            // Distance outside valid box `[vx0..vx1, vy0..vy1]`
            let cx = tx + b_sz / 2;
            let cy = ty + b_sz / 2;
            let dx_out = if cx < vx0 {
                vx0 - cx
            } else if cx > vx1 {
                cx - vx1
            } else {
                0
            };
            let dy_out = if cy < vy0 {
                vy0 - cy
            } else if cy > vy1 {
                cy - vy1
            } else {
                0
            };
            // Priority key: tiles straddling the boundary (`hole_cnt < b_sz*b_sz` and small `dx_out + dy_out`) first!
            let prio = ((dx_out + dy_out) as u32) * 1024 + hole_cnt;
            tiles.push((prio, tx, ty));
        }
    }
    tiles.sort_by_key(|t| t.0);

    // Precompute steep 1D raised-cosine window (`powf(2.2)`) so each tile stays crisp in its interior
    let win1d: Vec<f32> = (0..b_sz)
        .map(|u| {
            let t = (u as f32 + 0.5) / (b_sz as f32);
            (0.5 * (1.0 - (2.0 * std::f32::consts::PI * t).cos())).max(1e-3).powf(2.2)
        })
        .collect();

    let vw = (vx1.saturating_sub(vx0 + b_sz)).max(1);
    let _vh = (vy1.saturating_sub(vy0 + b_sz)).max(1);
    let grid_w = w / stride + 2;
    let grid_h = h / stride + 2;
    let mut chosen_grid = vec![(usize::MAX, usize::MAX); grid_w * grid_h];
    let mut used_src_grid = vec![0u16; grid_w * grid_h];

    // 5. Stitch each tile from boundary outward
    for (t_idx, &(_, tx, ty)) in tiles.iter().enumerate() {
        // Compute target tile mean color & detail roughness to detect smooth sky zones in linear light
        let sample_step = (b_sz / 4).max(2);
        let mut t_mean = [0.0f32; 3];
        let mut t_det_sum = 0.0f32;
        let mut t_n = 0.0f32;
        for v in (1..b_sz).step_by(sample_step) {
            let t_row = (ty + v) * w + tx;
            for u in (1..b_sz).step_by(sample_step) {
                let ti = (t_row + u) * 3;
                t_mean[0] += base_low[ti];
                t_mean[1] += base_low[ti + 1];
                t_mean[2] += base_low[ti + 2];
                t_det_sum += src_detail[ti].abs().max(src_detail[ti + 1].abs()).max(src_detail[ti + 2].abs());
                t_n += 1.0;
            }
        }
        let inv_tn = 1.0 / t_n.max(1.0);
        t_mean[0] *= inv_tn;
        t_mean[1] *= inv_tn;
        t_mean[2] *= inv_tn;
        let t_det_avg = t_det_sum * inv_tn;
        // In linear light (`srgb^2.2`), daylight sky has `G >= 0.07` and `B > G * 1.04` and `B > R * 1.15`
        let t_is_sky = t_mean[2] > t_mean[0] * 1.15 && t_mean[2] > t_mean[1] * 1.04 && t_mean[1] > 0.07;
        let t_is_foliage = t_mean[1] >= t_mean[0] * 0.98 && t_mean[1] > t_mean[2] * 1.06;

        // If this outpaint border tile is smooth sky, `base` already has C1-continuous reflected + LaMa sky
        // with zero detail seam — never stamp foreign patches onto smooth sky!
        if is_outpaint && t_is_sky && t_det_avg < 0.018 {
            continue;
        }

        let mut cands: Vec<(usize, usize)> = Vec::with_capacity(80);

        // Pool A: Horizon-locked candidates (same vertical level `sy ≈ ty`, varying `sx`)
        for dy in [0isize, -2, 2, -5, 5, -9, 9, -15, 15] {
            let sy = (ty as isize + dy).clamp(vy0 as isize, (vy1.saturating_sub(b_sz)).max(vy0) as isize) as usize;
            for k in 0..6usize {
                let sx = vx0 + ((k * vw) / 5).min(vw);
                if sx + b_sz <= w && sy + b_sz <= h && rect_sum(sx, sy, sx + b_sz, sy + b_sz) == 0 {
                    cands.push((sx, sy));
                }
            }
        }

        // Pool B: Column-aligned candidates (same horizontal `sx ≈ tx`, near top/bottom valid band)
        for dx in [0isize, -6, 6, -16, 16, -32, 32] {
            let sx = (tx as isize + dx).clamp(vx0 as isize, (vx1.saturating_sub(b_sz)).max(vx0) as isize) as usize;
            let sy_top = (vy0 + stride / 2).min(h.saturating_sub(b_sz));
            let sy_bot = vy1.saturating_sub(b_sz + stride / 2).max(vy0);
            for sy in [sy_top, sy_bot, (sy_top + stride).min(sy_bot), sy_bot.saturating_sub(stride).max(sy_top)] {
                if sx + b_sz <= w && sy + b_sz <= h && rect_sum(sx, sy, sx + b_sz, sy + b_sz) == 0 {
                    cands.push((sx, sy));
                }
            }
        }

        // Pool C: Deterministic pseudo-random candidates from `valid_sources`
        let mut z = seed
            .wrapping_add((tx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
            .wrapping_add((ty as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9))
            .wrapping_add(t_idx as u64);
        for _ in 0..32 {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut h64 = z;
            h64 = (h64 ^ (h64 >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            h64 = (h64 ^ (h64 >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            let idx = (h64 as usize) % valid_sources.len();
            cands.push(valid_sources[idx]);
        }

        let gx_idx = (tx / stride).min(grid_w - 1);
        let gy_idx = (ty / stride).min(grid_h - 1);
        let mut best_cand = cands[0];
        let mut best_cost = f32::INFINITY;
        let mut best_col_err = f32::INFINITY;

        let is_lr_border = is_outpaint && (tx < vx0 || tx + b_sz > vx1) && ty + b_sz / 2 >= vy0 && ty + b_sz / 2 <= vy1;

        for &(sx, sy) in &cands {
            // 1) Low-frequency L1+L2 color & vertical profile match on a 4x4 grid + semantic sky/foliage check
            let mut c_low = 0.0f32;
            let mut c_guide = 0.0f32;
            let mut s_mean = [0.0f32; 3];
            let mut det_peak = 0.0f32;
            let mut det_sum = 0.0f32;
            let mut n_low = 0.0f32;
            for v in (1..b_sz).step_by(sample_step) {
                let t_row = (ty + v) * w + tx;
                let s_row = (sy + v) * w + sx;
                for u in (1..b_sz).step_by(sample_step) {
                    let ti = (t_row + u) * 3;
                    let si = (s_row + u) * 3;
                    s_mean[0] += base_low[si];
                    s_mean[1] += base_low[si + 1];
                    s_mean[2] += base_low[si + 2];
                    let dp = src_detail[si].abs().max(src_detail[si + 1].abs()).max(src_detail[si + 2].abs());
                    det_sum += dp;
                    if dp > det_peak {
                        det_peak = dp;
                    }
                    for c in 0..3 {
                        let dl = (base_low[ti + c] - base_low[si + c]).abs();
                        c_low += dl * dl + dl * 0.25;
                        if !is_outpaint {
                            let dg = src_detail[ti + c] - src_detail[si + c];
                            c_guide += dg * dg;
                        }
                    }
                    n_low += 1.0;
                }
            }
            let inv_nl = 1.0 / n_low.max(1.0);
            c_low *= inv_nl;
            c_guide *= inv_nl;
            s_mean[0] *= inv_nl;
            s_mean[1] *= inv_nl;
            s_mean[2] *= inv_nl;
            let s_det_avg = det_sum * inv_nl;

            // Semantic guard in linear light: never stamp high-contrast tree/building edges into sky borders,
            // and never stamp sky into green foliage/grass borders!
            if t_is_sky && (det_peak > 0.032 || s_det_avg > t_det_avg * 1.6 + 0.008) {
                continue;
            }
            let s_is_sky = s_mean[2] > s_mean[0] * 1.15 && s_mean[2] > s_mean[1] * 1.04 && s_mean[1] > 0.07;
            if !t_is_sky && s_is_sky {
                continue;
            }
            if t_is_foliage && (s_mean[1] < s_mean[0] * 0.94 || s_mean[1] < s_mean[2] * 1.02) {
                continue;
            }

            // 2) Overlap seam coherence with already-placed neighbors / photo boundary
            let mut c_ovl = 0.0f32;
            let mut n_ovl = 0.0f32;
            for v in (0..b_sz).step_by(2) {
                let t_row = (ty + v) * w + tx;
                let s_row = (sy + v) * w + sx;
                for u in (0..b_sz).step_by(2) {
                    let wi = acc_weight[t_row + u];
                    if wi > 0.12 {
                        let wt_norm = (0.20 * wi + 0.80 * acc_weight2[t_row + u].sqrt()).max(1e-4);
                        let inv_w = 1.0 / wt_norm;
                        let ti = (t_row + u) * 3;
                        let si = (s_row + u) * 3;
                        for c in 0..3 {
                            let prev_d = acc_detail[ti + c] * inv_w;
                            let dd = prev_d - src_detail[si + c];
                            c_ovl += dd * dd;
                        }
                        n_ovl += 1.0;
                    }
                }
            }
            if n_ovl > 0.0 {
                c_ovl /= n_ovl;
            }

            // 3) Calibrated anti-repetition tie-breaker so similar grass/foliage patches diversify without overriding color match
            let sgx = (sx / stride).min(grid_w - 1);
            let sgy = (sy / stride).min(grid_h - 1);
            let mut rep_pen = (used_src_grid[sgy * grid_w + sgx] as f32) * 0.0025;
            for ndy in -1isize..=1 {
                for ndx in -1isize..=1 {
                    let ngx = (gx_idx as isize + ndx).clamp(0, (grid_w - 1) as isize) as usize;
                    let ngy = (gy_idx as isize + ndy).clamp(0, (grid_h - 1) as isize) as usize;
                    let prev = chosen_grid[ngy * grid_w + ngx];
                    if prev.0 != usize::MAX && sx.abs_diff(prev.0) <= b_sz && sy.abs_diff(prev.1) <= b_sz {
                        rep_pen += 0.008;
                    }
                }
            }

            // 4) Spatial & horizon prior
            let dy_norm = (sy as f32 - ty as f32).abs() / (h as f32);
            let dx_norm = (sx as f32 - tx as f32).abs() / (w as f32);
            let c_spat = if is_lr_border {
                // Strong horizon lock on left/right outpaint borders so sea/horizon/sky align
                dy_norm * 0.08 + dx_norm * 0.002
            } else if is_outpaint {
                dy_norm * 0.006 + dx_norm * 0.006
            } else {
                (dx_norm + dy_norm) * 0.004
            };

            let rough_err = (s_det_avg - t_det_avg).abs() * 0.8;
            let total = c_low * 9.5 + c_ovl * 2.8 + c_guide * 0.85 + rough_err + c_spat + rep_pen;
            if total < best_cost {
                best_cost = total;
                best_col_err = c_low;
                best_cand = (sx, sy);
            }
        }

        // Gate detail injection by color match quality so an unmatched candidate can never leave a visible block
        let col_gate = (-best_col_err * 18.0).exp().clamp(0.0, 1.0);
        if col_gate < 0.08 {
            continue;
        }

        let (bx, by) = best_cand;
        chosen_grid[gy_idx * grid_w + gx_idx] = (bx, by);
        let b_sgx = (bx / stride).min(grid_w - 1);
        let b_sgy = (by / stride).min(grid_h - 1);
        used_src_grid[b_sgy * grid_w + b_sgx] = used_src_grid[b_sgy * grid_w + b_sgx].saturating_add(1);

        // Accumulate best native detail patch with steep 2D window + L2 weight tracking
        let sky_atten = (if t_is_sky { 0.20 } else { 1.0 }) * col_gate;
        for v in 0..b_sz {
            let wv = win1d[v];
            let t_row = (ty + v) * w + tx;
            let s_row = (by + v) * w + bx;
            for u in 0..b_sz {
                let w2d = wv * win1d[u] * col_gate;
                let ti = t_row + u;
                let si = s_row + u;
                acc_detail[ti * 3] += src_detail[si * 3] * wv * win1d[u] * sky_atten;
                acc_detail[ti * 3 + 1] += src_detail[si * 3 + 1] * wv * win1d[u] * sky_atten;
                acc_detail[ti * 3 + 2] += src_detail[si * 3 + 2] * wv * win1d[u] * sky_atten;
                acc_weight[ti] += w2d;
                acc_weight2[ti] += w2d * w2d;
            }
        }
    }

    // 6. Recombine smooth illumination `base_low` + variance-preserving L2-normalized 1:1 native quilted detail
    let mut out = rgb.to_vec();
    for i in 0..(w * h) {
        if !hole[i] && dist_in[i] >= stride as u8 {
            continue;
        }
        let wi = acc_weight[i];
        let idx = i * 3;
        if wi > 1e-5 {
            let wt_norm = if hole[i] {
                (0.20 * wi + 0.80 * acc_weight2[i].sqrt()).max(1e-5)
            } else {
                wi
            };
            let inv_w = 1.0 / wt_norm;
            out[idx] = (base_low[idx] + acc_detail[idx] * inv_w).max(0.0);
            out[idx + 1] = (base_low[idx + 1] + acc_detail[idx + 1] * inv_w).max(0.0);
            out[idx + 2] = (base_low[idx + 2] + acc_detail[idx + 2] * inv_w).max(0.0);
        } else {
            out[idx] = base[idx].max(0.0);
            out[idx + 1] = base[idx + 1].max(0.0);
            out[idx + 2] = base[idx + 2].max(0.0);
        }
    }
    out
}

/// Estimates the high-frequency sensor/film grain standard deviation from smooth unmasked regions
/// of the context box (`hole == false`). Returns `0.0` on flat synthetic test patterns.
pub fn estimate_context_grain(rgb: &[f32], hole: &[bool], w: usize, h: usize) -> f32 {
    if w < 6 || h < 6 || rgb.len() < w * h * 3 || hole.len() < w * h {
        return 0.0;
    }
    let mut samples = Vec::with_capacity(512);
    let step_y = (h / 24).max(1);
    let step_x = (w / 24).max(1);
    for y in (1..h - 1).step_by(step_y) {
        for x in (1..w - 1).step_by(step_x) {
            let i = y * w + x;
            if hole[i] || hole[i - 1] || hole[i + 1] || hole[i - w] || hole[i + w] {
                continue;
            }
            let l_c = rgb[i * 3 + 1];
            let l_l = rgb[(i - 1) * 3 + 1];
            let l_r = rgb[(i + 1) * 3 + 1];
            let l_u = rgb[(i - w) * 3 + 1];
            let l_d = rgb[(i + w) * 3 + 1];
            let grad = (l_r - l_l).abs().max((l_d - l_u).abs());
            // Only measure micro-variance in locally smooth zones (excluding sharp edges/stripes)
            if grad < 0.035 && grad > 1e-5 {
                let lap = (l_c - 0.25 * (l_l + l_r + l_u + l_d)).abs();
                samples.push(lap);
            }
        }
    }
    if samples.len() < 16 {
        return 0.0;
    }
    samples.sort_by(|a, b| a.total_cmp(b));
    // Median absolute Laplacian deviation (robust against remaining texture edges)
    let med = samples[samples.len() / 2];
    if med < 2e-4 {
        0.0
    } else {
        (med * 1.15).min(0.025)
    }
}

/// Deterministic zero-mean pseudo-Gaussian sensor grain in `[-2.5, 2.5]`.
#[inline]
pub fn sensor_grain_sample(x: usize, y: usize, ch: usize, seed: u64) -> f32 {
    let mut z = (x as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((y as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .wrapping_add((ch as u64).wrapping_mul(0x94D0_49BB_1331_11EB))
        .wrapping_add(seed);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    // Sum of 3 uniform [0,1] variables minus 1.5 -> Irwin-Hall bell curve with unit variance
    let u0 = ((z & 0xFFFF) as f32) / 65535.0;
    let u1 = (((z >> 16) & 0xFFFF) as f32) / 65535.0;
    let u2 = (((z >> 32) & 0xFFFF) as f32) / 65535.0;
    u0 + u1 + u2 - 1.5
}

/// Direct 1:1 native-resolution (`6K`) single-patch frequency-separated healer for compact
/// blemishes, moles (`lunares`), pimples, sensor dust, and small distractions.
///
/// Copies a single contiguous, unblurred 1:1 patch of clean surrounding texture (preserving 100%
/// of native camera skin pores and sensor grain without neural downsampling or multi-patch blur)
/// and solves a harmonic Laplace membrane (`membrane_fill`) for exact boundary illumination matching.
static BLEMISH_OFFSET_CACHE: OnceLock<Mutex<Vec<(u64, f32, f32)>>> = OnceLock::new();

fn heal_native_blemish_6k(
    rgb: &[f32],
    hole: &[bool],
    hole_dil: &[bool],
    w: usize,
    h: usize,
    seed: u64,
) -> Option<Vec<f32>> {
    if seed != 0 && (seed & 1) == 0 {
        return None;
    }
    let mut hx0 = w;
    let mut hy0 = h;
    let mut hx1 = 0usize;
    let mut hy1 = 0usize;
    let mut hole_pts: Vec<(usize, usize)> = Vec::new();
    for y in 0..h {
        for x in 0..w {
            if hole_dil[y * w + x] {
                hx0 = hx0.min(x);
                hy0 = hy0.min(y);
                hx1 = hx1.max(x + 1);
                hy1 = hy1.max(y + 1);
                hole_pts.push((x, y));
            }
        }
    }
    if hole_pts.is_empty() || hx1 <= hx0 || hy1 <= hy0 {
        return None;
    }
    let hw = hx1 - hx0;
    let hh = hy1 - hy0;
    let span = hw.max(hh);
    // Only use single-patch 1:1 native healing when the hole is a compact blemish/mole/spot
    // (never on large multi-object strokes at preview resolution!)
    if seed != 0 {
        if span > 180 || span * 3 > w.min(h) {
            return None;
        }
    } else if span > 56 || span * 3 > w.min(h) || hole_pts.len() > 2200 {
        return None;
    }

    // Build exterior boundary ring (`!hole_dil` within `ring_w` pixels of `hole_dil`)
    let ring_w = (span / 5).clamp(4, 10) as isize;
    let ring_w2 = ring_w * ring_w;
    let mut ring_pts: Vec<(usize, usize)> = Vec::new();
    let rx0 = hx0.saturating_sub(ring_w as usize);
    let ry0 = hy0.saturating_sub(ring_w as usize);
    let rx1 = (hx1 + ring_w as usize).min(w);
    let ry1 = (hy1 + ring_w as usize).min(h);
    for y in ry0..ry1 {
        for x in rx0..rx1 {
            if !hole_dil[y * w + x] {
                let mut near = false;
                for dy in -ring_w..=ring_w {
                    let ny = (y as isize + dy).clamp(0, (h - 1) as isize) as usize;
                    for dx in -ring_w..=ring_w {
                        if dx * dx + dy * dy <= ring_w2 {
                            let nx = (x as isize + dx).clamp(0, (w - 1) as isize) as usize;
                            if hole_dil[ny * w + nx] {
                                near = true;
                                break;
                            }
                        }
                    }
                    if near {
                        break;
                    }
                }
                if near {
                    ring_pts.push((x, y));
                }
            }
        }
    }
    if ring_pts.len() < 12 {
        return None;
    }

    // Subsample ring and hole points for fast candidate evaluation
    let ring_step = (ring_pts.len() / 96).max(1);
    let hole_step = (hole_pts.len() / 96).max(1);

    // Compute target ring local high-frequency roughness and colour deviation
    let r_sep = (span / 4).clamp(3, 12);
    let rgb_low = blur_rgb3(rgb, w, h, r_sep);

    let mut t_mean = [0.0f32; 3];
    let mut t_rough = 0.0f32;
    let mut t_cnt = 0.0f32;
    for &(rx, ry) in ring_pts.iter().step_by(ring_step) {
        let idx = (ry * w + rx) * 3;
        t_mean[0] += rgb[idx];
        t_mean[1] += rgb[idx + 1];
        t_mean[2] += rgb[idx + 2];
        let d = (rgb[idx] - rgb_low[idx]).abs()
            + (rgb[idx + 1] - rgb_low[idx + 1]).abs()
            + (rgb[idx + 2] - rgb_low[idx + 2]).abs();
        t_rough += d * (1.0 / 3.0);
        t_cnt += 1.0;
    }
    t_mean[0] /= t_cnt;
    t_mean[1] /= t_cnt;
    t_mean[2] /= t_cnt;
    t_rough /= t_cnt;
    let mut t_max_dev = 0.0f32;
    for &(rx, ry) in ring_pts.iter().step_by(ring_step) {
        let idx = (ry * w + rx) * 3;
        let dev = (rgb[idx] - t_mean[0])
            .abs()
            .max((rgb[idx + 1] - t_mean[1]).abs())
            .max((rgb[idx + 2] - t_mean[2]).abs());
        t_max_dev = t_max_dev.max(dev);
    }

    // Generate candidate offsets `(dx, dy)` in concentric rings + dense horizontal/vertical sweeps
    let mut cands: Vec<(isize, isize)> = Vec::with_capacity(240);
    let min_shift = (span as isize) + 2;
    let max_shift_x = (w as isize / 2).max(min_shift + 2);
    let max_shift_y = (h as isize / 2).max(min_shift + 2);
    for d in (min_shift..=max_shift_x).step_by(2) {
        cands.push((d, 0));
        cands.push((-d, 0));
    }
    for d in (min_shift..=max_shift_y).step_by(2) {
        cands.push((0, d));
        cands.push((0, -d));
    }
    let base_r = (span as f32 * 0.65).max(6.0);
    for mult in [1.65f32, 2.15, 2.75, 3.45, 4.25, 5.15] {
        let dist = base_r * mult;
        for k in 0..24 {
            let ang = (k as f32 / 24.0) * std::f32::consts::TAU;
            let dx = (ang.cos() * dist).round() as isize;
            let dy = (ang.sin() * dist).round() as isize;
            cands.push((dx, dy));
        }
    }

    let mut best_off: Option<(isize, isize)> = None;
    let mut best_score = f32::INFINITY;

    if seed != 0
        && let Ok(guard) = BLEMISH_OFFSET_CACHE.get_or_init(|| Mutex::new(Vec::new())).lock()
        && let Some(&(_, ndx, ndy)) = guard.iter().find(|(s, _, _)| *s == seed)
    {
        let dx = (ndx * w as f32).round() as isize;
        let dy = (ndy * h as f32).round() as isize;
        let shx0 = hx0 as isize + dx;
        let shy0 = hy0 as isize + dy;
        let shx1 = hx1 as isize + dx;
        let shy1 = hy1 as isize + dy;
        if shx0 >= 1
            && shy0 >= 1
            && shx1 < (w - 1) as isize
            && shy1 < (h - 1) as isize
            && hole_pts
                .iter()
                .step_by(hole_step)
                .all(|&(hx, hy)| !hole_dil[(hy as isize + dy) as usize * w + (hx as isize + dx) as usize])
        {
            best_off = Some((dx, dy));
        }
    }

    if best_off.is_none() {
        'cand_loop: for (dx, dy) in cands {
            // Verify shifted hole bounding box fits inside `[1..w-1, 1..h-1]`
            let shx0 = hx0 as isize + dx;
            let shy0 = hy0 as isize + dy;
            let shx1 = hx1 as isize + dx;
            let shy1 = hy1 as isize + dy;
            if shx0 < 1 || shy0 < 1 || shx1 >= (w - 1) as isize || shy1 >= (h - 1) as isize {
                continue;
            }
            // Verify shifted hole never touches `hole_dil`
            for &(hx, hy) in hole_pts.iter().step_by(hole_step) {
                let nx = (hx as isize + dx) as usize;
                let ny = (hy as isize + dy) as usize;
                if hole_dil[ny * w + nx] {
                    continue 'cand_loop;
                }
            }

            // 1. Score boundary ring color, chroma, and gradient continuity on valid shifted ring points
            let mut ring_err = 0.0f32;
            let mut s_ring_mean = [0.0f32; 3];
            let mut valid_ring_cnt = 0.0f32;
            for &(rx, ry) in ring_pts.iter().step_by(ring_step) {
                let sx_i = rx as isize + dx;
                let sy_i = ry as isize + dy;
                if sx_i < 1 || sy_i < 1 || sx_i >= (w - 1) as isize || sy_i >= (h - 1) as isize {
                    continue;
                }
                let sx = sx_i as usize;
                let sy = sy_i as usize;
                if hole_dil[sy * w + sx] {
                    continue;
                }
                let ti = (ry * w + rx) * 3;
                let si = (sy * w + sx) * 3;
                s_ring_mean[0] += rgb_low[si];
                s_ring_mean[1] += rgb_low[si + 1];
                s_ring_mean[2] += rgb_low[si + 2];

                let dr = rgb_low[ti] - rgb_low[si];
                let dg = rgb_low[ti + 1] - rgb_low[si + 1];
                let db = rgb_low[ti + 2] - rgb_low[si + 2];
                let raw_dr = rgb[ti] - rgb[si];
                let raw_dg = rgb[ti + 1] - rgb[si + 1];
                let raw_db = rgb[ti + 2] - rgb[si + 2];
                let col_d = (dr * dr + dg * dg * 1.4 + db * db) * 0.5
                    + (raw_dr * raw_dr + raw_dg * raw_dg * 1.4 + raw_db * raw_db) * 0.5
                    + ((dr - dg).powi(2) + (dg - db).powi(2)) * 2.0;

                let t_gx = rgb_low[ti + 3 + 1] - rgb_low[ti - 3 + 1];
                let t_gy = rgb_low[(ry + 1) * w * 3 + rx * 3 + 1] - rgb_low[(ry - 1) * w * 3 + rx * 3 + 1];
                let s_gx = rgb_low[si + 3 + 1] - rgb_low[si - 3 + 1];
                let s_gy = rgb_low[(sy + 1) * w * 3 + sx * 3 + 1] - rgb_low[(sy - 1) * w * 3 + sx * 3 + 1];
                let grad_d = (t_gx - s_gx).powi(2) + (t_gy - s_gy).powi(2);

                ring_err += col_d + grad_d * 2.5;
                valid_ring_cnt += 1.0;
            }
            if valid_ring_cnt < (t_cnt * 0.5).max(6.0) {
                continue;
            }
            ring_err /= valid_ring_cnt;
            s_ring_mean[0] /= valid_ring_cnt;
            s_ring_mean[1] /= valid_ring_cnt;
            s_ring_mean[2] /= valid_ring_cnt;

            // 2. Inspect source interior: reject any candidate containing another blemish/mole/edge!
            let mut s_rough = 0.0f32;
            let mut max_int_dev = 0.0f32;
            let mut h_cnt = 0.0f32;
            for &(hx, hy) in hole_pts.iter().step_by(hole_step) {
                let sx = (hx as isize + dx) as usize;
                let sy = (hy as isize + dy) as usize;
                let si = (sy * w + sx) * 3;
                let dev_low = (rgb_low[si] - s_ring_mean[0])
                    .abs()
                    .max((rgb_low[si + 1] - s_ring_mean[1]).abs())
                    .max((rgb_low[si + 2] - s_ring_mean[2]).abs());
                let dev_raw = (rgb[si] - s_ring_mean[0])
                    .abs()
                    .max((rgb[si + 1] - s_ring_mean[1]).abs())
                    .max((rgb[si + 2] - s_ring_mean[2]).abs());
                max_int_dev = max_int_dev.max(dev_low).max(dev_raw * 0.75);
                let d = (rgb[si] - rgb_low[si]).abs()
                    + (rgb[si + 1] - rgb_low[si + 1]).abs()
                    + (rgb[si + 2] - rgb_low[si + 2]).abs();
                s_rough += d * (1.0 / 3.0);
                h_cnt += 1.0;
            }
            s_rough /= h_cnt.max(1.0);

            let rough_err = (s_rough - t_rough).abs() * 2.5;
            let excess_dev = (max_int_dev - (t_max_dev * 1.25 + 0.012)).max(0.0);
            let outlier_pen = excess_dev * excess_dev * 24.0;
            let dist_pen = ((dx * dx + dy * dy) as f32).sqrt() / (w.max(h) as f32) * 0.015;
            let total = ring_err * 4.5 + outlier_pen + rough_err + dist_pen;
            if total < best_score {
                best_score = total;
                best_off = Some((dx, dy));
            }
        }
        if seed != 0
            && w.max(h) >= 48
            && let Some((bdx, bdy)) = best_off
            && let Ok(mut guard) = BLEMISH_OFFSET_CACHE.get_or_init(|| Mutex::new(Vec::new())).lock()
        {
            if guard.len() >= 32 {
                guard.remove(0);
            }
            guard.push((seed, bdx as f32 / w as f32, bdy as f32 / h as f32));
        }
    }

    let (best_dx, best_dy) = best_off?;

    // Build shifted source low-pass and 1:1 native high-frequency pore/texture detail
    let mut src_low_shifted = rgb_low.clone();
    let mut detail_6k = vec![0.0f32; w * h * 3];
    for y in ry0..ry1 {
        for x in rx0..rx1 {
            let sx = (x as isize + best_dx).clamp(0, (w - 1) as isize) as usize;
            let sy = (y as isize + best_dy).clamp(0, (h - 1) as isize) as usize;
            let ti = (y * w + x) * 3;
            let si = (sy * w + sx) * 3;
            src_low_shifted[ti] = rgb_low[si];
            src_low_shifted[ti + 1] = rgb_low[si + 1];
            src_low_shifted[ti + 2] = rgb_low[si + 2];
            detail_6k[ti] = rgb[si] - rgb_low[si];
            detail_6k[ti + 1] = rgb[si + 1] - rgb_low[si + 1];
            detail_6k[ti + 2] = rgb[si + 2] - rgb_low[si + 2];
        }
    }

    // Solve harmonic Laplace membrane for `rgb_low - src_low_shifted` across `hole_dil`
    // so the low-frequency illumination and skin-tone gradient match the boundary with C1 continuity.
    let mut diff = vec![0.0f32; w * h * 3];
    for i in 0..(w * h) {
        if !hole_dil[i] {
            let idx = i * 3;
            diff[idx] = rgb_low[idx] - src_low_shifted[idx];
            diff[idx + 1] = rgb_low[idx + 1] - src_low_shifted[idx + 1];
            diff[idx + 2] = rgb_low[idx + 2] - src_low_shifted[idx + 2];
        }
    }
    let smooth_diff = membrane_fill(w, h, 3, &diff, hole_dil);

    let mut out = rgb.to_vec();
    for i in 0..(w * h) {
        if hole[i] {
            let idx = i * 3;
            out[idx] = (src_low_shifted[idx] + smooth_diff[idx] + detail_6k[idx]).max(0.0);
            out[idx + 1] = (src_low_shifted[idx + 1] + smooth_diff[idx + 1] + detail_6k[idx + 1]).max(0.0);
            out[idx + 2] = (src_low_shifted[idx + 2] + smooth_diff[idx + 2] + detail_6k[idx + 2]).max(0.0);
        }
    }
    Some(out)
}

/// Transfers full-contrast 1:1 camera-native meso-structure (leaves, palm fronds, grass blades,
/// bark) and micro-texture (skin pores, fine grain) into `hole` guided by `lama_base`,
/// while sanitizing any out-of-palette neural FFC chroma smudges and keeping smooth sky untouched.
#[allow(dead_code)]
fn synthesize_guided_micro_detail(
    rgb: &[f32],
    lama_base: &[f32],
    hole: &[bool],
    w: usize,
    h: usize,
    seed: u64,
) -> Vec<f32> {
    if w < 32 || h < 32 || rgb.len() < w * h * 3 || lama_base.len() < w * h * 3 || hole.len() < w * h {
        return lama_base.to_vec();
    }
    let r_fine = (w.max(h) / 30).clamp(8, 28);
    let src_low = blur_rgb3(rgb, w, h, r_fine);
    let mut base_clean = lama_base.to_vec();

    // Dilate `hole` by 6px so source patches never touch any residual object boundary or seam
    let mut hole_dil = hole.to_vec();
    for y in 0..h {
        for x in 0..w {
            if hole[y * w + x] {
                for dy in -6isize..=6 {
                    for dx in -6isize..=6 {
                        if dx * dx + dy * dy <= 38 {
                            let ny = (y as isize + dy).clamp(0, (h - 1) as isize) as usize;
                            let nx = (x as isize + dx).clamp(0, (w - 1) as isize) as usize;
                            hole_dil[ny * w + nx] = true;
                        }
                    }
                }
            }
        }
    }

    // Horizon & Luminance-Split Chroma Sanitizer:
    // Separates unmasked reference pixels at each horizontal band `[y - 16 .. y + 16]` into:
    //   - `sky` pixels (`B > G`, `B > R + 0.02`, bright `G > 0.45`)
    //   - `non_sky` pixels (foliage, palm fronds, trunks, grass, ground)
    // Any hole pixel darker than the sky at row `y` (`g < sky_mean_g - 0.065`) is classified as
    // `non_sky` (foliage/ground) and has its `(R - G, B - G)` clamped strictly to `non_sky` bounds,
    // making it mathematically impossible for purple/cyan/blue FFC smudges to appear on palm trees!
    for y in 0..h {
        let y0 = y.saturating_sub(16);
        let y1 = (y + 16).min(h - 1);
        let mut all_min_rg = f32::INFINITY;
        let mut all_max_rg = f32::NEG_INFINITY;
        let mut all_min_bg = f32::INFINITY;
        let mut all_max_bg = f32::NEG_INFINITY;
        let mut all_cnt = 0usize;

        let mut sky_min_rg = f32::INFINITY;
        let mut sky_max_rg = f32::NEG_INFINITY;
        let mut sky_min_bg = f32::INFINITY;
        let mut sky_max_bg = f32::NEG_INFINITY;
        let mut sky_g_sum = 0.0f32;
        let mut sky_cnt = 0usize;

        let mut fol_min_rg = f32::INFINITY;
        let mut fol_max_rg = f32::NEG_INFINITY;
        let mut fol_min_bg = f32::INFINITY;
        let mut fol_max_bg = f32::NEG_INFINITY;
        let mut fol_min_g = f32::INFINITY;
        let mut fol_max_g = f32::NEG_INFINITY;
        let mut fol_g_sum = 0.0f32;
        let mut fol_cnt = 0usize;

        for yy in (y0..=y1).step_by(3) {
            for x in (0..w).step_by(4) {
                let i = yy * w + x;
                if !hole_dil[i] {
                    let idx = i * 3;
                    let r = src_low[idx];
                    let g = src_low[idx + 1];
                    let b = src_low[idx + 2];
                    let rg = r - g;
                    let bg = b - g;
                    all_min_rg = all_min_rg.min(rg);
                    all_max_rg = all_max_rg.max(rg);
                    all_min_bg = all_min_bg.min(bg);
                    all_max_bg = all_max_bg.max(bg);
                    all_cnt += 1;
                    // True sky reference pixels in linear light (`srgb^2.2`): blue-dominant AND bright (`g > 0.07`)
                    if b > r * 1.15 && b > g * 1.04 && g > 0.07 {
                        sky_min_rg = sky_min_rg.min(rg);
                        sky_max_rg = sky_max_rg.max(rg);
                        sky_min_bg = sky_min_bg.min(bg);
                        sky_max_bg = sky_max_bg.max(bg);
                        sky_g_sum += g;
                        sky_cnt += 1;
                    } else {
                        fol_min_rg = fol_min_rg.min(rg);
                        fol_max_rg = fol_max_rg.max(rg);
                        fol_min_bg = fol_min_bg.min(bg);
                        fol_max_bg = fol_max_bg.max(bg);
                        fol_min_g = fol_min_g.min(g);
                        fol_max_g = fol_max_g.max(g);
                        fol_g_sum += g;
                        fol_cnt += 1;
                    }
                }
            }
        }
        if all_cnt >= 6 {
            let sky_mean_g = if sky_cnt > 0 { sky_g_sum / (sky_cnt as f32) } else { 1.0 };
            let fol_mean_g = if fol_cnt > 0 { fol_g_sum / (fol_cnt as f32) } else { 0.08 };
            for x in 0..w {
                let i = y * w + x;
                if hole[i] {
                    let idx = i * 3;
                    let r = base_clean[idx];
                    let mut g = base_clean[idx + 1];
                    let b = base_clean[idx + 2];
                    // Notice: if `LaMa` bled sky cyan/blue into a palm tree / foliage zone (`b > g`),
                    // its green channel `g` was also artificially lifted toward `sky_mean_g`, which would
                    // otherwise turn into a bright yellowish-olive haze after chroma clamping!
                    // Estimate true `non_sky` membership using both `g` and whether surrounding row is foliage:
                    let (min_rg, max_rg, min_bg, max_bg) = if sky_cnt >= 4 && fol_cnt >= 4 {
                        let sky_thresh = (sky_mean_g - 0.035).max(fol_max_g + 0.012);
                        let t_sky = ((g - sky_thresh) / 0.030).clamp(0.0, 1.0);
                        // Pull any sky-lifted luminance in the foliage zone back into `[fol_min_g .. fol_max_g]`
                        if t_sky < 0.99 {
                            let g_fol_clamped = g.clamp(fol_min_g, fol_max_g);
                            let g_fol = if g_fol_clamped > fol_mean_g {
                                fol_mean_g + (g_fol_clamped - fol_mean_g) * 0.55
                            } else {
                                g_fol_clamped
                            };
                            g = g_fol * (1.0 - t_sky) + g * t_sky;
                        }
                        (
                            fol_min_rg * (1.0 - t_sky) + sky_min_rg * t_sky,
                            fol_max_rg * (1.0 - t_sky) + sky_max_rg * t_sky,
                            fol_min_bg * (1.0 - t_sky) + sky_min_bg * t_sky,
                            fol_max_bg * (1.0 - t_sky) + sky_max_bg * t_sky,
                        )
                    } else if fol_cnt >= 4 {
                        let g_clamped = g.clamp(fol_min_g, fol_max_g);
                        g = if g_clamped > fol_mean_g {
                            fol_mean_g + (g_clamped - fol_mean_g) * 0.65
                        } else {
                            g_clamped
                        };
                        (fol_min_rg, fol_max_rg, fol_min_bg, fol_max_bg)
                    } else {
                        (all_min_rg, all_max_rg, all_min_bg, all_max_bg)
                    };
                    let rg = (r - base_clean[idx + 1]).clamp(min_rg, max_rg);
                    let bg = (b - base_clean[idx + 1]).clamp(min_bg, max_bg);
                    base_clean[idx] = (g + rg).max(0.0);
                    base_clean[idx + 1] = g.max(0.0);
                    base_clean[idx + 2] = (g + bg).max(0.0);
                }
            }
        }
    }

    let guide_low = blur_rgb3(&base_clean, w, h, r_fine);

    let mut high_band = vec![0.0f32; w * h * 3];
    for i in 0..(w * h) {
        if !hole_dil[i] {
            let idx = i * 3;
            high_band[idx] = (rgb[idx] - src_low[idx]).clamp(-0.36, 0.36);
            high_band[idx + 1] = (rgb[idx + 1] - src_low[idx + 1]).clamp(-0.36, 0.36);
            high_band[idx + 2] = (rgb[idx + 2] - src_low[idx + 2]).clamp(-0.36, 0.36);
        }
    }

    let b_sz = ((w.max(h) / 26).clamp(16, 32)) & !1;
    let stride = b_sz / 2;

    let mut sat = vec![0u32; (w + 1) * (h + 1)];
    for y in 0..h {
        let mut row = 0u32;
        for x in 0..w {
            row += u32::from(hole_dil[y * w + x]);
            sat[(y + 1) * (w + 1) + x + 1] = sat[y * (w + 1) + x + 1] + row;
        }
    }
    let rect_sum = |x0: usize, y0: usize, x1: usize, y1: usize| -> u32 {
        let x1 = x1.min(w);
        let y1 = y1.min(h);
        if x1 <= x0 || y1 <= y0 {
            return 1;
        }
        let w1 = w + 1;
        sat[y1 * w1 + x1] + sat[y0 * w1 + x0] - sat[y0 * w1 + x1] - sat[y1 * w1 + x0]
    };

    // Evaluate mean color, 2x2 quadrant green/luma layout, gradient, and detail roughness over the `b_sz × b_sz` block,
    // rejecting blocks that straddle sharp architectural/silhouette edges or dark holes.
    let half_b = (b_sz / 2).max(1);
    let block_stats =
        |low_buf: &[f32], bx: usize, by: usize, check_spike: bool| -> Option<([f32; 3], [f32; 4], f32, f32, f32)> {
            let mut mean = [0.0f32; 3];
            let mut q_g = [0.0f32; 4];
            let mut q_n = [0.0f32; 4];
            let mut n = 0.0f32;
            for v in (0..b_sz).step_by(2) {
                let row = (by + v) * w + bx;
                let qy = usize::from(v >= half_b) * 2;
                for u in (0..b_sz).step_by(2) {
                    let idx = (row + u) * 3;
                    let r = low_buf[idx];
                    let g = low_buf[idx + 1];
                    let b = low_buf[idx + 2];
                    mean[0] += r;
                    mean[1] += g;
                    mean[2] += b;
                    let qi = qy + usize::from(u >= half_b);
                    q_g[qi] += g;
                    q_n[qi] += 1.0;
                    n += 1.0;
                }
            }
            let inv_n = 1.0 / n.max(1.0);
            mean[0] *= inv_n;
            mean[1] *= inv_n;
            mean[2] *= inv_n;
            for qi in 0..4 {
                q_g[qi] *= 1.0 / q_n[qi].max(1.0);
            }

            let mut max_col_dev = 0.0f32;
            let mut det_sum = 0.0f32;
            let mut det_peak = 0.0f32;
            if check_spike {
                for v in (0..b_sz).step_by(2) {
                    let row = (by + v) * w + bx;
                    for u in (0..b_sz).step_by(2) {
                        let idx = (row + u) * 3;
                        let dev = (low_buf[idx] - mean[0])
                            .abs()
                            .max((low_buf[idx + 1] - mean[1]).abs())
                            .max((low_buf[idx + 2] - mean[2]).abs());
                        if dev > max_col_dev {
                            max_col_dev = dev;
                        }
                        let d =
                            (high_band[idx].abs() + high_band[idx + 1].abs() + high_band[idx + 2].abs()) * (1.0 / 3.0);
                        det_sum += d;
                        if d > det_peak {
                            det_peak = d;
                        }
                    }
                }
                let det_avg = det_sum * inv_n;
                // Reject blocks straddling a sharp architectural/silhouette step or isolated spike
                if max_col_dev > 0.165 || det_peak > det_avg * 3.6 + 0.032 {
                    return None;
                }
            }

            let cx = (bx + b_sz / 2).clamp(2, w - 3);
            let cy = (by + b_sz / 2).clamp(2, h - 3);
            let c_l = (cy * w + (cx - 2)) * 3;
            let c_r = (cy * w + (cx + 2)) * 3;
            let c_u = ((cy - 2) * w + cx) * 3;
            let c_d = ((cy + 2) * w + cx) * 3;
            let gx = (low_buf[c_r + 1] - low_buf[c_l + 1]) * 0.25;
            let gy = (low_buf[c_d + 1] - low_buf[c_u + 1]) * 0.25;
            Some((mean, q_g, gx, gy, det_sum * inv_n))
        };

    let cand_step = (stride / 2).clamp(4, 8);
    // (sx, sy, mean_col, q_g, gx, gy, det_avg)
    let mut sources: Vec<(usize, usize, [f32; 3], [f32; 4], f32, f32, f32)> = Vec::new();
    for sy in (1..h.saturating_sub(b_sz + 1)).step_by(cand_step) {
        for sx in (1..w.saturating_sub(b_sz + 1)).step_by(cand_step) {
            if rect_sum(sx, sy, sx + b_sz, sy + b_sz) == 0
                && let Some((col, q_g, gx, gy, det_avg)) = block_stats(&src_low, sx, sy, true)
            {
                sources.push((sx, sy, col, q_g, gx, gy, det_avg));
            }
        }
    }
    if sources.is_empty() {
        return base_clean;
    }

    // Steep raised-cosine window (`powf(2.2)`) so each tile dominates its interior without
    // double-exposure ghosting of leaves/fronds/pores across wide overlap zones!
    let win1d: Vec<f32> = (0..b_sz)
        .map(|u| {
            let t = (u as f32 + 0.5) / (b_sz as f32);
            (0.5 * (1.0 - (2.0 * std::f32::consts::PI * t).cos())).max(1e-3).powf(2.2)
        })
        .collect();

    let mut acc_det = vec![0.0f32; w * h * 3];
    let mut acc_wt = vec![0.0f32; w * h];
    let mut acc_wt2 = vec![0.0f32; w * h];

    // Pre-seed unmasked boundary ring so the first layer of patches stitches onto real camera detail,
    // and compute inward distance transform `hole_depth` for onion-peel boundary-inward ordering.
    let mut hole_depth = vec![0u16; w * h];
    for i in 0..(w * h) {
        if hole[i] {
            hole_depth[i] = 1000;
        } else if !hole_dil[i] {
            let idx = i * 3;
            acc_det[idx] = high_band[idx] * 0.5;
            acc_det[idx + 1] = high_band[idx + 1] * 0.5;
            acc_det[idx + 2] = high_band[idx + 2] * 0.5;
            acc_wt[i] = 0.5;
            acc_wt2[i] = 0.25;
        }
    }
    for pass in 1..=128u16 {
        let prev = pass - 1;
        for y in 1..(h - 1) {
            for x in 1..(w - 1) {
                let i = y * w + x;
                if hole_depth[i] > pass
                    && (hole_depth[i - 1] == prev
                        || hole_depth[i + 1] == prev
                        || hole_depth[i - w] == prev
                        || hole_depth[i + w] == prev)
                {
                    hole_depth[i] = pass;
                }
            }
        }
    }

    let grid_w = w / stride + 2;
    let grid_h = h / stride + 2;
    let mut chosen_grid = vec![(usize::MAX, usize::MAX); grid_w * grid_h];
    // Track usage count of source cells so the same source patch is never stamped repeatedly like wallpaper
    let src_grid_w = w / stride + 2;
    let src_grid_h = h / stride + 2;
    let mut used_src_grid = vec![0u16; src_grid_w * src_grid_h];

    let mut target_blocks: Vec<(u32, usize, usize)> = Vec::new();
    for ty in (0..=h.saturating_sub(b_sz)).step_by(stride) {
        for tx in (0..=w.saturating_sub(b_sz)).step_by(stride) {
            let mut touches_hole = false;
            let mut min_d = u16::MAX;
            for v in 0..b_sz {
                let r_off = (ty + v) * w + tx;
                for u in 0..b_sz {
                    if hole[r_off + u] {
                        touches_hole = true;
                        min_d = min_d.min(hole_depth[r_off + u]);
                    }
                }
            }
            if touches_hole {
                target_blocks.push((min_d as u32, tx, ty));
            }
        }
    }
    target_blocks.sort_by_key(|b| b.0);

    let n_src = sources.len();

    for &(_, tx, ty) in &target_blocks {
        let Some((t_col, t_qg, t_gx, t_gy, _)) = block_stats(&guide_low, tx, ty, false) else {
            continue;
        };
        // Never stamp foliage/ground patch detail into clear sky regions (blue-dominant low-gradient zones)
        let is_sky_like = t_col[2] > t_col[0] * 1.08 && t_col[2] >= t_col[1] * 0.98 && t_gx.hypot(t_gy) < 0.015;
        if is_sky_like {
            continue;
        }

        let gx_idx = tx / stride;
        let gy_idx = ty / stride;

        let mut best = sources[0];
        let mut best_cost = f32::INFINITY;
        let mut best_col_err = f32::INFINITY;

        let mut z = seed
            .wrapping_add((tx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
            .wrapping_add((ty as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));

        // Locate the slice of `sources` in the same vertical horizon band `sy ∈ [ty - 2*b_sz .. ty + 2*b_sz]`
        let sy_lo = ty.saturating_sub(b_sz * 2);
        let sy_hi = ty + b_sz * 2;
        let band_start = sources.partition_point(|s| s.1 < sy_lo);
        let band_end = sources.partition_point(|s| s.1 <= sy_hi).max(band_start + 1).min(n_src);
        let band_len = (band_end - band_start).max(1);

        // Build candidate indices: 44 from the same horizon band (`sy ≈ ty`) + 24 global strided + 28 random
        let check_count = n_src.min(96);
        for k in 0..check_count {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut h64 = z;
            h64 = (h64 ^ (h64 >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            h64 = (h64 ^ (h64 >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            let idx = if k < 44 {
                band_start + ((h64 as usize) % band_len)
            } else if k < 68 {
                ((k - 44) * n_src / 24).min(n_src - 1)
            } else {
                (h64 as usize) % n_src
            };
            let cand = sources[idx];
            // Never pick a sky source patch to fill a non-sky target block
            let cand_is_sky = cand.2[2] > cand.2[0] * 1.08 && cand.2[2] >= cand.2[1] * 0.98 && cand.6 < 0.006;
            if cand_is_sky {
                continue;
            }
            // If target block is green foliage/grass (`G > R` and `G > B`), never pick an achromatic/grey/brown building block
            let t_is_foliage = t_col[1] > t_col[0] * 1.02 && t_col[1] > t_col[2] * 1.08;
            if t_is_foliage && (cand.2[1] < cand.2[0] * 0.99 || cand.2[1] < cand.2[2] * 1.04) {
                continue;
            }

            let dr = t_col[0] - cand.2[0];
            let dg = t_col[1] - cand.2[1];
            let db = t_col[2] - cand.2[2];
            let q_err = ((t_qg[0] - cand.3[0]).powi(2)
                + (t_qg[1] - cand.3[1]).powi(2)
                + (t_qg[2] - cand.3[2]).powi(2)
                + (t_qg[3] - cand.3[3]).powi(2))
                * 0.35;
            let col_err = dr * dr * 1.4
                + dg * dg * 1.8
                + db * db * 1.2
                + ((dr - dg).powi(2) + (dg - db).powi(2)) * 2.5
                + q_err;
            let grad_err = (t_gx - cand.4).powi(2) + (t_gy - cand.5).powi(2);

            // Overlap coherence with already-placed neighbors so palm fronds, leaves, and grass align seamlessly
            let mut ovl_err = 0.0f32;
            let mut ovl_n = 0.0f32;
            for v in (0..b_sz).step_by(2) {
                let t_row = (ty + v) * w + tx;
                let s_row = (cand.1 + v) * w + cand.0;
                for u in (0..b_sz).step_by(2) {
                    let wt = acc_wt[t_row + u];
                    if wt > 0.12 {
                        let wt_norm = (0.20 * wt + 0.80 * acc_wt2[t_row + u].sqrt()).max(1e-4);
                        let inv_w = 1.0 / wt_norm;
                        let ti = (t_row + u) * 3;
                        let si = (s_row + u) * 3;
                        let d0 = acc_det[ti] * inv_w - high_band[si];
                        let d1 = acc_det[ti + 1] * inv_w - high_band[si + 1];
                        let d2 = acc_det[ti + 2] * inv_w - high_band[si + 2];
                        ovl_err += d0 * d0 + d1 * d1 + d2 * d2;
                        ovl_n += 1.0;
                    }
                }
            }
            if ovl_n > 0.0 {
                ovl_err /= ovl_n;
            }

            // Anti-repetition penalty: check how many times this source neighborhood has already been used,
            // AND penalize picking the same source as any 3×3 neighboring target block
            let sgx = (cand.0 / stride).min(src_grid_w - 1);
            let sgy = (cand.1 / stride).min(src_grid_h - 1);
            let used_cnt = used_src_grid[sgy * src_grid_w + sgx] as f32;
            let mut neighbor_rep_pen = used_cnt * 0.0025;
            for ndy in -1isize..=1 {
                for ndx in -1isize..=1 {
                    let ngx = (gx_idx as isize + ndx).clamp(0, (grid_w - 1) as isize) as usize;
                    let ngy = (gy_idx as isize + ndy).clamp(0, (grid_h - 1) as isize) as usize;
                    let prev = chosen_grid[ngy * grid_w + ngx];
                    if prev.0 != usize::MAX
                        && cand.0.abs_diff(prev.0) <= b_sz
                        && cand.1.abs_diff(prev.1) <= b_sz
                    {
                        neighbor_rep_pen += 0.008;
                    }
                }
            }

            // Strong horizon preference so foliage/ground at row `ty` matches texture from the same vertical band `sy`
            let dy_norm = (cand.1 as f32 - ty as f32).abs() / (h as f32);
            let dx_norm = (cand.0 as f32 - tx as f32).abs() / (w as f32);
            let l1_col = (dr.abs() + dg.abs() + db.abs()) * 0.35;
            let cost = (col_err + l1_col * 0.25) * 14.0 + ovl_err * 4.2 + grad_err * 3.8 + dy_norm * 0.08 + dx_norm * 0.010 + neighbor_rep_pen;
            if cost < best_cost {
                best_cost = cost;
                best_col_err = col_err;
                best = cand;
            }
        }
        chosen_grid[gy_idx * grid_w + gx_idx] = (best.0, best.1);
        let best_sgx = (best.0 / stride).min(src_grid_w - 1);
        let best_sgy = (best.1 / stride).min(src_grid_h - 1);
        used_src_grid[best_sgy * src_grid_w + best_sgx] = used_src_grid[best_sgy * src_grid_w + best_sgx].saturating_add(1);

        let col_gate = (-best_col_err * 20.0).exp().clamp(0.0, 1.0);
        if col_gate < 0.05 {
            continue;
        }

        for v in 0..b_sz {
            let wv = win1d[v];
            let t_row = (ty + v) * w + tx;
            let s_row = (best.1 + v) * w + best.0;
            for u in 0..b_sz {
                let w_eff = wv * win1d[u] * col_gate;
                let ti = t_row + u;
                let ti3 = ti * 3;
                let si = (s_row + u) * 3;
                // Transfer both crisp micro-detail `high_band` AND internal meso-structure `(src_low - cand_mean) - (guide_low - t_mean)`
                // so palm trunks, bush clusters, and leaf clumps inside the block retain full depth and contrast!
                let meso0 = (src_low[si] - best.2[0] - (guide_low[ti3] - t_col[0])).clamp(-0.18, 0.18) * 0.65;
                let meso1 = (src_low[si + 1] - best.2[1] - (guide_low[ti3 + 1] - t_col[1])).clamp(-0.18, 0.18) * 0.65;
                let meso2 = (src_low[si + 2] - best.2[2] - (guide_low[ti3 + 2] - t_col[2])).clamp(-0.18, 0.18) * 0.65;
                acc_det[ti3] += (high_band[si] + meso0) * w_eff;
                acc_det[ti3 + 1] += (high_band[si + 1] + meso1) * w_eff;
                acc_det[ti3 + 2] += (high_band[si + 2] + meso2) * w_eff;
                acc_wt[ti] += w_eff;
                acc_wt2[ti] += w_eff * w_eff;
            }
        }
    }

    let mut out = base_clean;
    for i in 0..(w * h) {
        if hole[i] {
            let wt = acc_wt[i];
            if wt > 1e-4 {
                let taper = (wt / 0.28).clamp(0.0, 1.0);
                // Variance-preserving L2/L1 normalization across overlapping raised-cosine tiles
                let wt_norm = (0.20 * wt + 0.80 * acc_wt2[i].sqrt()).max(1e-4);
                let inv_w = taper / wt_norm;
                let idx = i * 3;
                out[idx] = (out[idx] + acc_det[idx] * inv_w).max(0.0);
                out[idx + 1] = (out[idx + 1] + acc_det[idx + 1] * inv_w).max(0.0);
                out[idx + 2] = (out[idx + 2] + acc_det[idx + 2] * inv_w).max(0.0);
            }
        } else {
            let idx = i * 3;
            out[idx] = rgb[idx];
            out[idx + 1] = rgb[idx + 1];
            out[idx + 2] = rgb[idx + 2];
        }
    }
    out
}

pub fn current_cloud_mode_tag() -> (bool, u64) {
    use std::hash::{DefaultHasher, Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    0x434C_4F55_445F_4D44u64.hash(&mut hasher);

    let mut has_key = std::env::var("FAL_KEY").is_ok()
        || std::env::var("STABILITY_API_KEY").is_ok()
        || std::env::var("OPENAI_API_KEY").is_ok()
        || std::env::var("GEMINI_API_KEY").is_ok();
    let mut stability_mode = String::from("inpaint");
    let mut custom_prompt = String::new();
    let mut key_id = String::new();

    if let Ok(home) = std::env::var("HOME") {
        let base = std::path::PathBuf::from(home).join(".mm-studio");
        if let Ok(raw) = std::fs::read_to_string(base.join("ai_keys.json"))
            && raw.contains("\"api_key\"")
            && raw.len() > 28
        {
            has_key = true;
            if let Some(pos) = raw.find("\"api_key\"")
                && let Some(colon) = raw[pos..].find(':')
            {
                let after = raw[pos + colon + 1..].trim_start();
                if let Some(stripped) = after.strip_prefix('"')
                    && let Some(endq) = stripped.find('"')
                {
                    let api_key = stripped[..endq].trim();
                    key_id = api_key.chars().rev().take(8).collect();
                }
            }
            if raw.contains("\"erase\"") {
                stability_mode = String::from("erase");
            }
        }
        if stability_mode != "erase"
            && let Ok(pr) = std::fs::read_to_string(base.join("ai_prompt.txt"))
        {
            custom_prompt = pr.trim().to_string();
        }
    }
    if !key_id.is_empty() && std::path::Path::new(&format!("/tmp/mm_cloud_skip_{key_id}")).exists() {
        false.hash(&mut hasher);
        return (false, hasher.finish());
    }
    has_key.hash(&mut hasher);
    custom_prompt.hash(&mut hasher);
    (has_key, hasher.finish())
}

pub fn is_cloud_seed_cached(seed: u64, mode_tag: u64) -> bool {
    INPAINT_WORK_CACHE
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .ok()
        .is_some_and(|guard| guard.iter().any(|c| c.seed == seed && c.mode_tag == mode_tag && c.is_cloud))
}

/// Optional Cloud Generative Diffusion Inpainting (Lightroom / Firefly / FLUX Fill / Stability Erase / OpenAI / Gemini).
/// Reads API key from `~/.mm-studio/ai_keys.json` (`{"provider": "...", "api_key": "..."}`) or environment variables
/// (`FAL_KEY`, `STABILITY_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`).
/// When configured, sends the square-padded context crop + binary mask to the cloud diffusion model and returns
/// the reconstructed linear-light buffer `(w × h × 3)`.
#[cfg(target_os = "macos")]
fn try_cloud_generative_inpaint(
    rgb: &[f32],
    hole: &[bool],
    w: usize,
    h: usize,
    seed: u64,
    allow_cloud: bool,
) -> Option<Vec<f32>> {
    let home = std::env::var("HOME").ok()?;
    let cfg_path = std::path::PathBuf::from(&home).join(".mm-studio/ai_keys.json");
    let has_env = std::env::var("FAL_KEY").is_ok()
        || std::env::var("STABILITY_API_KEY").is_ok()
        || std::env::var("OPENAI_API_KEY").is_ok()
        || std::env::var("GEMINI_API_KEY").is_ok();
    if !has_env && !cfg_path.exists() {
        return None;
    }

    // Compute robust 99.5th percentile highlight scale so HDR linear values (> 1.0) map cleanly into sRGB
    let mut max_samples: Vec<f32> = Vec::with_capacity(1024);
    let step = (w.saturating_mul(h) / 512).max(1);
    let mut abs_max = 1.0f32;
    for i in (0..(w * h)).step_by(step) {
        if !hole[i] {
            let m = rgb[i * 3].max(rgb[i * 3 + 1]).max(rgb[i * 3 + 2]);
            if m.is_finite() && m > 0.0 {
                max_samples.push(m);
                if m > abs_max {
                    abs_max = m;
                }
            }
        }
    }
    let peak = if max_samples.len() >= 16 {
        max_samples.sort_by(|a, b| a.total_cmp(b));
        let p995 = max_samples[(max_samples.len() * 995 / 1000).min(max_samples.len() - 1)];
        (p995 * 1.03).max(abs_max.min(1.2)).clamp(0.08, 16.0)
    } else {
        abs_max.max(1.0)
    };
    let inv_peak = 1.0 / peak;

    let tmp_dir = std::path::PathBuf::from("/tmp");
    let in_raw = tmp_dir.join(format!("mm_cloud_in_{seed}.bin"));
    let out_raw = tmp_dir.join(format!("mm_cloud_out_{seed}.bin"));

    // Smoothly clamp API input resolution into [512..1024] in Rust so Stability AI always receives
    // high-definition context even if the preview crop box is < 512px, and never exceeds 1024px!
    let max_api_side = 1024usize;
    let min_api_side = 512usize;
    let long_side = w.max(h);
    let (send_w, send_h, send_rgb, send_hole) = if long_side > max_api_side {
        let scale = max_api_side as f32 / (long_side as f32);
        let sw = ((w as f32 * scale).round() as usize).max(64);
        let sh = ((h as f32 * scale).round() as usize).max(64);
        (
            sw,
            sh,
            resize_rgb(rgb, w, h, sw, sh),
            resize_mask(hole, w, h, sw, sh),
        )
    } else if long_side < min_api_side && allow_cloud {
        let scale = min_api_side as f32 / (long_side.max(1) as f32);
        let sw = ((w as f32 * scale).round() as usize).clamp(64, max_api_side);
        let sh = ((h as f32 * scale).round() as usize).clamp(64, max_api_side);
        (
            sw,
            sh,
            resize_rgb_bicubic(rgb, w, h, sw, sh),
            resize_mask(hole, w, h, sw, sh),
        )
    } else {
        (w, h, rgb.to_vec(), hole.to_vec())
    };

    // Write header `(send_w as u32, send_h as u32)` + interleaved `RGBA8` where `RGB` is sRGB and `A` is mask
    let mut payload = Vec::with_capacity(8 + send_w * send_h * 4);
    payload.extend_from_slice(&(send_w as u32).to_le_bytes());
    payload.extend_from_slice(&(send_h as u32).to_le_bytes());
    for i in 0..(send_w * send_h) {
        let r = ((send_rgb[i * 3] * inv_peak).clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
        let g = ((send_rgb[i * 3 + 1] * inv_peak).clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
        let b = ((send_rgb[i * 3 + 2] * inv_peak).clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
        let a = if send_hole[i] { 255u8 } else { 0u8 };
        payload.extend_from_slice(&[r, g, b, a]);
    }
    std::fs::write(&in_raw, &payload).ok()?;

    let seed_hex = format!("{seed:016x}");
    let allow_cloud_str = if allow_cloud { "1" } else { "0" };
    let py_script = r#"
import sys, os, json, struct, subprocess, tempfile, base64, urllib.request, ssl, hashlib, shutil
ssl_ctx = ssl._create_unverified_context()

in_bin, out_bin, cfg_file, seed_hex, orig_w_s, orig_h_s, allow_cloud = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]
orig_w, orig_h = int(orig_w_s), int(orig_h_s)
cfg = {}
if os.path.exists(cfg_file):
    try:
        cfg = json.load(open(cfg_file))
    except Exception:
        pass

api_key = (cfg.get("api_key") or os.environ.get("FAL_KEY") or os.environ.get("STABILITY_API_KEY")
           or os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY") or "").strip()
provider = (cfg.get("provider") or "").strip().lower()
stability_mode = (cfg.get("stability_mode") or "inpaint").strip().lower()
if stability_mode not in ("inpaint", "erase"):
    stability_mode = "inpaint"
if not api_key:
    sys.exit(1)

if not provider:
    if api_key.startswith("sk_pr_") or api_key.startswith("sandbox_sk_pr_"):
        provider = "photoroom"
    elif api_key.startswith("pk_live_") or api_key.startswith("pk_test_"):
        provider = "pixelapi"
    elif api_key.startswith("AIza") or api_key.startswith("AQ."):
        provider = "gemini"
    elif api_key.startswith("sk-proj-") or api_key.startswith("sk-None-") or (api_key.startswith("sk-") and len(api_key) > 55):
        provider = "openai"
    elif ":" in api_key or len(api_key) > 60:
        provider = "fal"
    else:
        provider = "stability"

data = open(in_bin, "rb").read()
api_w, api_h = struct.unpack("<II", data[:8])
rgba = bytearray(data[8:])

# Compute resolution-independent stroke seed hash so Fit view, Zoom-In, Zoom-Out, and 6K export share 1 API call!
sig = bytearray()
for gy in range(24):
    sy = min((gy * api_h) // 24, api_h - 1)
    for gx in range(24):
        sx = min((gx * api_w) // 24, api_w - 1)
        p = (sy * api_w + sx) * 4
        m = 1 if rgba[p + 3] > 127 else 0
        sig.append(m)
        if m == 0 and (gx % 3 == 0) and (gy % 3 == 0):
            sig.extend((rgba[p] >> 4, rgba[p+1] >> 4, rgba[p+2] >> 4))
mode_suffix = "_erase" if (provider == "stability" and stability_mode == "erase") else ""
legacy_key = hashlib.sha256(bytes(sig) + (provider + mode_suffix).encode()).hexdigest()[:20]
prompt_file = os.path.expanduser("~/.mm-studio/ai_prompt.txt")
custom_prompt = ""
if os.path.exists(prompt_file) and stability_mode != "erase":
    try:
        custom_prompt = open(prompt_file).read().strip()
    except Exception:
        pass
cache_dir = os.path.expanduser("~/.mm-studio/cloud_cache")
os.makedirs(cache_dir, exist_ok=True)
if seed_hex and seed_hex != "0000000000000000":
    base_key = hashlib.sha256((seed_hex + "_" + provider + mode_suffix).encode()).hexdigest()[:20]
    alt_mode_suffix = "" if mode_suffix else "_erase"
    alt_mode_key = hashlib.sha256((seed_hex + "_" + provider + alt_mode_suffix).encode()).hexdigest()[:20]
    stab_key = hashlib.sha256((seed_hex + "_stability").encode()).hexdigest()[:20]
    stab_erase_key = hashlib.sha256((seed_hex + "_stability_erase").encode()).hexdigest()[:20]
    pixel_key = hashlib.sha256((seed_hex + "_pixelapi").encode()).hexdigest()[:20]
    photoroom_key = hashlib.sha256((seed_hex + "_photoroom").encode()).hexdigest()[:20]
    ck_raw = (seed_hex + "_" + provider + mode_suffix + ("_" + custom_prompt if custom_prompt else "")).encode()
    cache_key = hashlib.sha256(ck_raw).hexdigest()[:20]
    bosque_key = hashlib.sha256((seed_hex + "_" + provider + "_bosque").encode()).hexdigest()[:20] if not mode_suffix else base_key
else:
    base_key = legacy_key
    alt_mode_key = legacy_key
    stab_key = legacy_key
    stab_erase_key = legacy_key
    pixel_key = legacy_key
    photoroom_key = legacy_key
    cache_key = legacy_key
    bosque_key = legacy_key
cached_png = os.path.join(cache_dir, f"{cache_key}.png")
base_png = os.path.join(cache_dir, f"{base_key}.png")
alt_mode_png = os.path.join(cache_dir, f"{alt_mode_key}.png")
stab_png = os.path.join(cache_dir, f"{stab_key}.png")
stab_erase_png = os.path.join(cache_dir, f"{stab_erase_key}.png")
pixel_png = os.path.join(cache_dir, f"{pixel_key}.png")
photoroom_png = os.path.join(cache_dir, f"{photoroom_key}.png")
bosque_png = os.path.join(cache_dir, f"{bosque_key}.png")
legacy_png = os.path.join(cache_dir, f"{legacy_key}.png")
if not (os.path.exists(cached_png) and os.path.getsize(cached_png) > 512):
    for alt_png in (base_png, alt_mode_png, stab_png, stab_erase_png, pixel_png, photoroom_png, bosque_png, legacy_png):
        if os.path.exists(alt_png) and os.path.getsize(alt_png) > 512:
            shutil.copyfile(alt_png, cached_png)
            break
has_cache = os.path.exists(cached_png) and os.path.getsize(cached_png) > 512
key_id = "".join(reversed(api_key[-8:]))
depleted_marker = f"/tmp/mm_cloud_skip_{key_id}"
try:
    open("/tmp/mm_cloud_trace.log", "a").write(
        f"seed={seed_hex} cache_key={cache_key} legacy_key={legacy_key} mode={stability_mode} orig={orig_w}x{orig_h} api={api_w}x{api_h} allow={allow_cloud} has_cache={has_cache}\n"
    )
except Exception:
    pass
if not has_cache:
    # Never trigger a new Cloud API call from a filmstrip/grid thumbnail pass (`allow_cloud != "1"`)!
    if os.path.exists(depleted_marker) or allow_cloud != "1" or max(orig_w, orig_h) < 64:
        sys.exit(1)

img_png_path = in_bin + "_img.png"
msk_png_path = in_bin + "_msk.png"
out_png_path = in_bin + "_out.png"
raw_rgb_cache = os.path.join(cache_dir, f"{cache_key}_{orig_w}x{orig_h}.rgb")

try:
    if has_cache:
        if os.path.exists(raw_rgb_cache) and os.path.getsize(raw_rgb_cache) == orig_w * orig_h * 3:
            shutil.copyfile(raw_rgb_cache, out_bin)
            sys.exit(0)
        out_png_path = cached_png
    else:
        api_rgb = bytearray(api_w * api_h * 3)
        api_msk = [[0] * api_w for _ in range(api_h)]
        for y in range(api_h):
            for x in range(api_w):
                p = (y * api_w + x) * 4
                di = (y * api_w + x) * 3
                api_rgb[di] = rgba[p]
                api_rgb[di+1] = rgba[p+1]
                api_rgb[di+2] = rgba[p+2]
                if rgba[p+3] > 127:
                    api_msk[y][x] = 255

        dil_r = max(3, min(8, max(api_w, api_h) // 95))
        dil_r2 = dil_r * dil_r
        dil_offsets = [(dy, dx) for dy in range(-dil_r, dil_r + 1) for dx in range(-dil_r, dil_r + 1) if dx*dx + dy*dy <= dil_r2]
        dil_msk = [row[:] for row in api_msk]
        for y in range(api_h):
            row_m = api_msk[y]
            for x in range(api_w):
                if row_m[x]:
                    is_edge = (
                        y == 0 or y == api_h - 1 or x == 0 or x == api_w - 1
                        or not (api_msk[y - 1][x] and api_msk[y + 1][x] and row_m[x - 1] and row_m[x + 1])
                    )
                    if is_edge:
                        for dy, dx in dil_offsets:
                            ny = min(max(y + dy, 0), api_h - 1)
                            nx = min(max(x + dx, 0), api_w - 1)
                            dil_msk[ny][nx] = 255

        import zlib
        def write_png_buf(path, width, height, mode="rgb"):
            raw_rows = bytearray()
            for y in range(height):
                raw_rows.append(0)
                if mode == "rgb":
                    off = y * width * 3
                    raw_rows.extend(api_rgb[off : off + width * 3])
                elif mode == "mask_bw":
                    row_m = dil_msk[y]
                    for x in range(width):
                        m = row_m[x]
                        raw_rows.extend((m, m, m))
                elif mode == "openai_rgba":
                    off = y * width * 3
                    row_m = dil_msk[y]
                    for x in range(width):
                        p = off + x * 3
                        a = 0 if row_m[x] > 127 else 255
                        raw_rows.extend((api_rgb[p], api_rgb[p+1], api_rgb[p+2], a))
            color_type = 6 if mode == "openai_rgba" else 2
            def chunk(tag, body):
                return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xffffffff)
            ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
            png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(bytes(raw_rows), 6)) + chunk(b"IEND", b"")
            open(path, "wb").write(png)
            return png

        img_bytes = write_png_buf(img_png_path, api_w, api_h, "rgb")
        msk_bytes = write_png_buf(msk_png_path, api_w, api_h, "mask_bw")
        if provider == "stability":
            # Auto-Prompt Inteligente: inspect unmasked context pixels around the mask to detect
            # whether the surrounding environment is forest/foliage, beach/ocean/sky, or architecture/indoor.
            n_ctx = 0
            n_fol = 0
            n_sky_sea = 0
            n_sand_warm = 0
            step_y = max(1, api_h // 64)
            step_x = max(1, api_w // 64)
            for cy in range(0, api_h, step_y):
                row_m = dil_msk[cy]
                for cx in range(0, api_w, step_x):
                    if row_m[cx] == 0:
                        idx = (cy * api_w + cx) * 3
                        cr, cg, cb = api_rgb[idx], api_rgb[idx + 1], api_rgb[idx + 2]
                        n_ctx += 1
                        if cg >= cr - 4 and cg > cb + 6 and cg > 32 and cr < 210:
                            n_fol += 1
                        elif cb > cr + 12 and cb >= cg - 4 and cb > 70:
                            n_sky_sea += 1
                        elif cr > cg + 6 and cg > cb + 6 and cr > 80:
                            n_sand_warm += 1
            if custom_prompt:
                cp_low = custom_prompt.lower()
                if any(k in cp_low for k in ("bosque", "pino", "arbol", "árbol", "rama", "hoja", "forest", "tree")):
                    prompt_str = f"Seamless natural photorealistic pine forest canopy, crisp green tree branches, pine needles, leaves, and blue sky ({custom_prompt}) matching the exact surrounding photograph"
                    neg_str = "black object, studio umbrella, light stand, equipment, person, blurry, smudge, soft haze, plaid, artifacts, watermark"
                elif any(k in cp_low for k in ("playa", "mar", "arena", "oceano", "océano", "beach", "sea", "sand")):
                    prompt_str = f"Clean seamless photorealistic natural open sky, ocean water horizon, and smooth beach sand ({custom_prompt}) matching the exact surrounding coastal background"
                    neg_str = "trees, branches, leaves, foliage, plants, person, people, black object, umbrella, blurry, smudge, artifacts, watermark"
                else:
                    prompt_str = f"{custom_prompt}, ultra-sharp photorealistic natural background continuation matching the exact surrounding texture, lighting, and depth of field"
                    neg_str = "black object, studio umbrella, light stand, equipment, blurry, smudge, soft haze, artifacts, watermark"
            elif n_ctx > 0 and (n_fol * 100 >= n_ctx * 14):
                # Forest / Trees / Foliage scene (gives the ultra-sharp crisp leaves & branches of E5c322290ba133c08e12!)
                prompt_str = "Seamless natural photorealistic background continuation matching the exact surrounding foliage, tree branches, pine needles, leaves, sky, texture, lighting, and depth of field of the photograph"
                neg_str = "black object, studio umbrella, light stand, equipment, person, text, watermark, blurry, smudge, soft haze, plaid, artifacts"
            elif n_ctx > 0 and (n_sky_sea * 100 >= n_ctx * 8 or n_sand_warm * 100 >= n_ctx * 15):
                # Beach / Ocean / Open Sky scene (strictly forbids trees/leaves/people!)
                prompt_str = "Clean seamless photorealistic natural open sky, ocean water horizon, gentle waves, and smooth beach sand matching the exact surrounding coastal background and lighting"
                neg_str = "trees, branches, leaves, foliage, plants, bushes, person, people, human, black object, umbrella, equipment, blurry, smudge, artifacts, watermark"
            else:
                # General / Architectural / Indoor scene
                prompt_str = "Clean seamless photorealistic natural background texture, surface structure, and lighting matching the exact surrounding photograph with sharp focus"
                neg_str = "person, people, black object, studio umbrella, light stand, equipment, trees, branches, blurry, smudge, artifacts, watermark"
            seed_val = (int(seed_hex, 16) % 4294967294) if (seed_hex and seed_hex != "0000000000000000") else 0
            def run_sdxl_v1_masking():
                sdxl_dims = [(1024, 1024), (1152, 896), (1216, 832), (1344, 768), (1536, 640), (640, 1536), (768, 1344), (832, 1216), (896, 1152)]
                target_ar = api_w / max(1.0, float(api_h))
                sw_xl, sh_xl = min(sdxl_dims, key=lambda wh: abs((wh[0] / float(wh[1])) - target_ar))
                # Pre-fill masked pixels in api_rgb with surrounding unmasked context so dark umbrellas/stands
                # never bias SDXL v1's VAE latent initialization!
                unmasked_samples = []
                for sy in range(0, api_h, max(1, api_h // 32)):
                    row_m = dil_msk[sy]
                    for sx in range(0, api_w, max(1, api_w // 32)):
                        if row_m[sx] == 0:
                            idx = (sy * api_w + sx) * 3
                            unmasked_samples.append((api_rgb[idx], api_rgb[idx+1], api_rgb[idx+2]))
                if unmasked_samples:
                    ns = len(unmasked_samples)
                    for sy in range(api_h):
                        row_m = dil_msk[sy]
                        for sx in range(api_w):
                            if row_m[sx] > 0:
                                idx = (sy * api_w + sx) * 3
                                sr, sg, sb = unmasked_samples[(sy * 31 + sx * 17) % ns]
                                api_rgb[idx] = sr
                                api_rgb[idx+1] = sg
                                api_rgb[idx+2] = sb
                xl_src_path = in_bin + "_xl_src.png"
                write_png_buf(xl_src_path, api_w, api_h, "rgb")
                xl_img = in_bin + "_xl_img.png"
                xl_msk = in_bin + "_xl_msk.png"
                try:
                    subprocess.run(["sips", "-z", str(sh_xl), str(sw_xl), xl_src_path, "--out", xl_img], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    subprocess.run(["sips", "-z", str(sh_xl), str(sw_xl), msk_png_path, "--out", xl_msk], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    cmd_v1 = [
                        "curl", "-sS", "-w", "\n%{http_code}", "-X", "POST",
                        "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/image-to-image/masking",
                        "-H", f"authorization: Bearer {api_key}",
                        "-H", "accept: application/json",
                        "-F", f"init_image=@{xl_img}",
                        "-F", f"mask_image=@{xl_msk}",
                        "-F", "mask_source=MASK_IMAGE_WHITE",
                        "-F", f"text_prompts[0][text]={prompt_str}",
                        "-F", "text_prompts[0][weight]=1.0",
                        "-F", f"text_prompts[1][text]={neg_str}",
                        "-F", "text_prompts[1][weight]=-1.0",
                        "-F", "cfg_scale=7",
                        "-F", "steps=26",
                        "-F", f"seed={seed_val}",
                    ]
                    v1_out = subprocess.check_output(cmd_v1, timeout=45, stderr=subprocess.DEVNULL).decode(errors="ignore")
                    lines = v1_out.rsplit("\n", 1)
                    body_str = lines[0] if len(lines) > 1 else v1_out
                    v1_code = lines[1].strip() if len(lines) > 1 else "200"
                    if v1_code != "200":
                        return v1_code
                    v1_js = json.loads(body_str)
                    b64_art = v1_js["artifacts"][0]["base64"]
                    open(out_png_path, "wb").write(base64.b64decode(b64_art))
                    return "200"
                except Exception:
                    return "500"
                finally:
                    for xp in (xl_src_path, xl_img, xl_msk):
                        if os.path.exists(xp):
                            try: os.remove(xp)
                            except Exception: pass

            if stability_mode == "erase":
                http_code = run_sdxl_v1_masking()
            else:
                cmd = [
                    "curl", "-sS", "-w", "%{http_code}", "-X", "POST",
                    "https://api.stability.ai/v2beta/stable-image/edit/inpaint",
                    "-H", f"authorization: Bearer {api_key}",
                    "-H", "accept: image/*",
                    "-F", f"image=@{img_png_path}",
                    "-F", f"mask=@{msk_png_path}",
                    "-F", f"prompt={prompt_str}",
                    "-F", f"negative_prompt={neg_str}",
                    "-F", f"seed={seed_val}",
                    "-F", "grow_mask=5",
                    "-F", "output_format=png",
                    "-o", out_png_path
                ]
                res_p = subprocess.run(cmd, timeout=45, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                http_code = res_p.stdout.decode(errors="ignore").strip()
                if http_code == "402":
                    # Account has <= 5 credits remaining: automatically fall back to SDXL 1.0 v1 (~0.2 cr)!
                    if os.path.exists(out_png_path):
                        try: os.remove(out_png_path)
                        except Exception: pass
                    http_code = run_sdxl_v1_masking()

            if http_code != "200" or not os.path.exists(out_png_path) or os.path.getsize(out_png_path) <= 512:
                try:
                    err_body = open(out_png_path, "rb").read()[:400].decode(errors="ignore") if os.path.exists(out_png_path) else ""
                    open("/tmp/mm_cloud_last_err.log", "a").write(
                        f"seed={seed_hex} mode={stability_mode} http={http_code} api={api_w}x{api_h} body={err_body}\n"
                    )
                except Exception:
                    pass
                if http_code in ("402", "401", "403"):
                    open(depleted_marker, "w").write(http_code)
                if os.path.exists(out_png_path):
                    try: os.remove(out_png_path)
                    except Exception: pass
                sys.exit(1)
        elif provider == "pixelapi":
            import time
            cmd_px = [
                "curl", "-sS", "-w", "\n%{http_code}", "-X", "POST",
                "https://api.pixelapi.dev/v1/image/remove-object",
                "-H", f"Authorization: Bearer {api_key}",
                "-H", "User-Agent: MMStudio/1.0",
                "-F", f"image=@{img_png_path};type=image/png",
                "-F", f"mask=@{msk_png_path};type=image/png",
            ]
            if custom_prompt:
                cmd_px.extend(["-F", f"prompt={custom_prompt}"])
            px_raw = subprocess.check_output(cmd_px, timeout=45, stderr=subprocess.DEVNULL).decode(errors="ignore")
            px_lines = px_raw.rsplit("\n", 1)
            px_body = px_lines[0] if len(px_lines) > 1 else px_raw
            px_code = px_lines[1].strip() if len(px_lines) > 1 else "200"
            if px_code != "200":
                try:
                    open("/tmp/mm_cloud_last_err.log", "a").write(
                        f"seed={seed_hex} mode=pixelapi http={px_code} api={api_w}x{api_h} body={px_body[:400]}\n"
                    )
                except Exception:
                    pass
                if px_code in ("402", "401", "403"):
                    open(depleted_marker, "w").write(px_code)
                sys.exit(1)
            px_js = json.loads(px_body)
            gen_id = px_js.get("generation_id") or px_js.get("id")
            if not gen_id:
                sys.exit(1)
            out_url = px_js.get("output_url") or px_js.get("direct_url") or (px_js.get("output") or {}).get("url")
            for poll_idx in range(48):
                if out_url:
                    break
                time.sleep(0.45 if poll_idx < 6 else 0.8)
                poll_req = urllib.request.Request(
                    f"https://api.pixelapi.dev/v1/image/{gen_id}",
                    headers={"Authorization": f"Bearer {api_key}", "User-Agent": "MMStudio/1.0"}
                )
                poll_js = json.loads(urllib.request.urlopen(poll_req, context=ssl_ctx, timeout=20).read().decode())
                st = (poll_js.get("status") or "").lower()
                if st == "completed":
                    out_url = poll_js.get("output_url") or poll_js.get("direct_url") or (poll_js.get("output") or {}).get("url")
                    break
                elif st in ("failed", "error", "cancelled"):
                    sys.exit(1)
            if not out_url:
                sys.exit(1)
            if out_url.startswith("/"):
                out_url = "https://api.pixelapi.dev" + out_url
            dl_req = urllib.request.Request(out_url, headers={"User-Agent": "MMStudio/1.0"})
            open(out_png_path, "wb").write(urllib.request.urlopen(dl_req, context=ssl_ctx, timeout=30).read())
        elif provider == "photoroom":
            pr_prompt = custom_prompt if custom_prompt else (
                "Remove the unwanted studio umbrella, softbox, light stand, tripod, equipment, blemish, or obstructing object "
                "and seamlessly reconstruct the natural background foliage, pine tree branches, sky, and lighting matching the surrounding photo"
            )
            def call_photoroom(key_to_use):
                cmd_pr = [
                    "curl", "-sS", "-w", "%{http_code}", "-o", out_png_path, "-X", "POST",
                    "https://image-api.photoroom.com/v2/edit",
                    "-H", f"x-api-key: {key_to_use}",
                    "-F", f"imageFile=@{img_png_path}",
                    "-F", "removeBackground=false",
                    "-F", "referenceBox=originalImage",
                    "-F", "editWithAI.mode=ai.auto",
                    "-F", f"editWithAI.prompt={pr_prompt}",
                ]
                return subprocess.check_output(cmd_pr, timeout=45, stderr=subprocess.DEVNULL).decode(errors="ignore").strip()
            pr_code = call_photoroom(api_key)
            if pr_code != "200" and not api_key.startswith("sandbox_"):
                pr_code = call_photoroom("sandbox_" + api_key)
            if pr_code != "200" or not (os.path.exists(out_png_path) and os.path.getsize(out_png_path) > 512):
                err_txt = ""
                if os.path.exists(out_png_path):
                    try:
                        err_txt = open(out_png_path, "rb").read(400).decode(errors="ignore")
                    except Exception:
                        pass
                try:
                    open("/tmp/mm_cloud_last_err.log", "a").write(
                        f"seed={seed_hex} mode=photoroom http={pr_code} api={api_w}x{api_h} body={err_txt}\n"
                    )
                except Exception:
                    pass
                if pr_code in ("402", "401", "403"):
                    open(depleted_marker, "w").write(pr_code)
                if os.path.exists(out_png_path):
                    try: os.remove(out_png_path)
                    except Exception: pass
                sys.exit(1)
        elif provider == "fal":
            img_b64 = "data:image/png;base64," + base64.b64encode(img_bytes).decode()
            msk_b64 = "data:image/png;base64," + base64.b64encode(msk_bytes).decode()
            body = json.dumps({"image_url": img_b64, "mask_url": msk_b64}).encode()
            req = urllib.request.Request(
                "https://fal.run/fal-ai/bria/eraser",
                data=body,
                headers={"Authorization": f"Key {api_key}", "Content-Type": "application/json"}
            )
            resp = json.loads(urllib.request.urlopen(req, context=ssl_ctx, timeout=40).read().decode())
            url = (resp.get("image") or {}).get("url") or resp.get("images", [{}])[0].get("url")
            open(out_png_path, "wb").write(urllib.request.urlopen(url, context=ssl_ctx, timeout=30).read())
        elif provider == "openai":
            oa_path = in_bin + "_oa.png"
            write_png_buf(oa_path, api_w, api_h, "openai_rgba")
            cmd = [
                "curl", "-sS", "-f", "-X", "POST",
                "https://api.openai.com/v1/images/edits",
                "-H", f"Authorization: Bearer {api_key}",
                "-F", "model=gpt-image-1",
                "-F", f"image=@{oa_path}",
                "-F", "prompt=Remove the masked object seamlessly and reconstruct the natural background texture, foliage, sky, and lighting identically to the surrounding photo.",
                "-F", "size=1024x1024"
            ]
            res = json.loads(subprocess.check_output(cmd, timeout=50).decode())
            b64 = res["data"][0].get("b64_json")
            if b64:
                open(out_png_path, "wb").write(base64.b64decode(b64))
            else:
                open(out_png_path, "wb").write(urllib.request.urlopen(res["data"][0]["url"], context=ssl_ctx, timeout=30).read())
        elif provider == "gemini":
            mag_rgb = bytearray(api_rgb)
            for y in range(api_h):
                for x in range(api_w):
                    if dil_msk[y][x] > 127:
                        p = (y * api_w + x) * 3
                        mag_rgb[p:p+3] = b"\xff\x00\xff"
            api_rgb = mag_rgb
            mag_bytes = write_png_buf(in_bin + "_mag.png", api_w, api_h, "rgb")
            body = json.dumps({
                "contents": [{
                    "parts": [
                        {"text": "Remove the solid magenta (#FF00FF) masked region completely and reconstruct the natural background seamlessly with identical lighting and resolution. Return only the edited image."},
                        {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(mag_bytes).decode()}}
                    ]
                }],
                "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]}
            }).encode()
            req = urllib.request.Request(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent?key={api_key}",
                data=body,
                headers={"Content-Type": "application/json"}
            )
            resp = json.loads(urllib.request.urlopen(req, context=ssl_ctx, timeout=45).read().decode())
            for part in resp["candidates"][0]["content"]["parts"]:
                if "inlineData" in part:
                    open(out_png_path, "wb").write(base64.b64decode(part["inlineData"]["data"]))
                    break

        if os.path.exists(out_png_path) and os.path.getsize(out_png_path) > 512:
            shutil.copyfile(out_png_path, cached_png)
            if base_png != cached_png:
                shutil.copyfile(out_png_path, base_png)
            if legacy_png != cached_png and legacy_png != base_png:
                shutil.copyfile(out_png_path, legacy_png)

    if not os.path.exists(out_png_path):
        sys.exit(1)

    bmp_path = in_bin + "_out.bmp"
    subprocess.run(["sips", "-z", str(orig_h), str(orig_w), "-s", "format", "bmp", out_png_path, "--out", bmp_path],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    bmp = open(bmp_path, "rb").read()
    px_off = struct.unpack("<I", bmp[10:14])[0]
    dib_w, dib_h = struct.unpack("<ii", bmp[18:26])
    bpp = struct.unpack("<H", bmp[28:30])[0] // 8
    top_down = dib_h < 0
    abs_h = abs(dib_h)
    row_stride = ((orig_w * bpp + 3) // 4) * 4
    out_rgb = bytearray(orig_w * orig_h * 3)
    for y in range(orig_h):
        src_y = y if top_down else (abs_h - 1 - y)
        r_start = px_off + src_y * row_stride
        for x in range(orig_w):
            p = r_start + x * bpp
            b_v, g_v, r_v = bmp[p], bmp[p+1], bmp[p+2]
            di = (y * orig_w + x) * 3
            out_rgb[di] = r_v
            out_rgb[di+1] = g_v
            out_rgb[di+2] = b_v
    open(out_bin, "wb").write(out_rgb)
    try:
        open(raw_rgb_cache, "wb").write(out_rgb)
    except Exception:
        pass
finally:
    for p in [img_png_path, msk_png_path, in_bin + "_out.png", in_bin + "_oa.png", in_bin + "_mag.png", in_bin + "_out.bmp"]:
        if os.path.exists(p):
            try: os.remove(p)
            except Exception: pass
"#;

    let py_bin = if std::path::Path::new("/Library/Frameworks/Python.framework/Versions/3.14/bin/python3").exists() {
        "/Library/Frameworks/Python.framework/Versions/3.14/bin/python3"
    } else {
        "/usr/bin/python3"
    };
    let status = std::process::Command::new(py_bin)
        .arg("-c")
        .arg(py_script)
        .arg(&in_raw)
        .arg(&out_raw)
        .arg(&cfg_path)
        .arg(&seed_hex)
        .arg(w.to_string())
        .arg(h.to_string())
        .arg(allow_cloud_str)
        .status()
        .ok();
    let _ = std::fs::remove_file(&in_raw);
    if !status.is_some_and(|s| s.success()) {
        let _ = std::fs::remove_file(&out_raw);
        return None;
    }
    let raw_bytes = std::fs::read(&out_raw).ok()?;
    let _ = std::fs::remove_file(&out_raw);
    if raw_bytes.len() < w * h * 3 {
        return None;
    }
    let mut out_lin = vec![0.0f32; w * h * 3];
    for i in 0..(w * h * 3) {
        out_lin[i] = ((raw_bytes[i] as f32) / 255.0).powf(2.2) * peak;
    }
    Some(out_lin)
}

#[cfg(not(target_os = "macos"))]
fn try_cloud_generative_inpaint(
    _rgb: &[f32],
    _hole: &[bool],
    _w: usize,
    _h: usize,
    _seed: u64,
    _allow_cloud: bool,
) -> Option<Vec<f32>> {
    None
}

static INPAINT_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

#[derive(Clone)]
struct CachedInpaintWork {
    seed: u64,
    mode_tag: u64,
    grid_sig: [f32; 96],
    work_w: usize,
    work_h: usize,
    work_rgb: Vec<f32>,
    pm_work: Vec<f32>,
    pm_adapted_work: Vec<f32>,
    nnf_work: Vec<(i32, i32)>,
    is_cloud: bool,
}

static INPAINT_WORK_CACHE: OnceLock<Mutex<Vec<CachedInpaintWork>>> = OnceLock::new();

fn compute_unmasked_grid_sig(rgb: &[f32], hole: &[bool], w: usize, h: usize) -> [f32; 96] {
    let mut sums = [0.0f32; 96];
    let mut cnts = [0.0f32; 32];
    for y in 0..h {
        let gy = ((y * 2) / h.max(1)).min(1);
        for x in 0..w {
            let i = y * w + x;
            if !hole[i] {
                let gx = ((x * 16) / w.max(1)).min(15);
                let b = gy * 16 + gx;
                let idx = i * 3;
                sums[b * 3] += rgb[idx];
                sums[b * 3 + 1] += rgb[idx + 1];
                sums[b * 3 + 2] += rgb[idx + 2];
                cnts[b] += 1.0;
            }
        }
    }
    for b in 0..32 {
        if cnts[b] > 0.0 {
            let inv = 1.0 / cnts[b];
            sums[b * 3] *= inv;
            sums[b * 3 + 1] *= inv;
            sums[b * 3 + 2] *= inv;
        }
    }
    sums
}

fn grid_sig_matches(a: &[f32; 96], b: &[f32; 96]) -> bool {
    let mut sum_diff = 0.0f32;
    let mut max_diff = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        sum_diff += d;
        if d > max_diff {
            max_diff = d;
        }
    }
    (sum_diff / 96.0) <= 0.025 && max_diff <= 0.085
}

#[allow(clippy::too_many_arguments)]
fn reconstruct_inpaint_from_work(
    rgb: &[f32],
    hole: &[bool],
    hole_dil: &[bool],
    w: usize,
    h: usize,
    work_w: usize,
    work_h: usize,
    work_rgb: &[f32],
    pm_work: &[f32],
    pm_adapted_work: &[f32],
    nnf_work: &[(i32, i32)],
    is_cloud: bool,
    cfg: &LamaConfig,
) -> Vec<f32> {
    if is_cloud {
        let mut cloud_rgb = if (work_w, work_h) == (w, h) {
            pm_adapted_work.to_vec()
        } else {
            resize_rgb_bicubic(pm_adapted_work, work_w, work_h, w, h)
        };
        let rgb_low6 = blur_rgb3(rgb, w, h, 6);
        let cld_low6 = blur_rgb3(&cloud_rgb, w, h, 6);
        let mut diff = vec![0.0f32; w * h * 3];
        for i in 0..(w * h) {
            if !hole_dil[i] {
                let idx = i * 3;
                diff[idx] = (rgb_low6[idx] - cld_low6[idx]).clamp(-0.035, 0.035);
                diff[idx + 1] = (rgb_low6[idx + 1] - cld_low6[idx + 1]).clamp(-0.035, 0.035);
                diff[idx + 2] = (rgb_low6[idx + 2] - cld_low6[idx + 2]).clamp(-0.035, 0.035);
            }
        }
        let smooth_diff = membrane_fill(w, h, 3, &diff, hole_dil);
        for i in 0..(w * h) {
            let idx = i * 3;
            if hole[i] {
                cloud_rgb[idx] = (cloud_rgb[idx] + smooth_diff[idx]).max(0.0);
                cloud_rgb[idx + 1] = (cloud_rgb[idx + 1] + smooth_diff[idx + 1]).max(0.0);
                cloud_rgb[idx + 2] = (cloud_rgb[idx + 2] + smooth_diff[idx + 2]).max(0.0);
            } else {
                cloud_rgb[idx] = rgb[idx];
                cloud_rgb[idx + 1] = rgb[idx + 1];
                cloud_rgb[idx + 2] = rgb[idx + 2];
            }
        }
        return cloud_rgb;
    }

    let mut final_rgb = if (work_w, work_h) == (w, h) {
        let mut out = rgb.to_vec();
        for i in 0..(w * h) {
            if hole[i] {
                let idx = i * 3;
                out[idx] = pm_adapted_work[idx];
                out[idx + 1] = pm_adapted_work[idx + 1];
                out[idx + 2] = pm_adapted_work[idx + 2];
            }
        }
        out
    } else {
        let pm_full = resize_rgb_bicubic(pm_adapted_work, work_w, work_h, w, h);
        let rgb_upsampled = resize_rgb_bicubic(work_rgb, work_w, work_h, w, h);
        let sx_scale = w as f32 / (work_w.max(1) as f32);
        let sy_scale = h as f32 / (work_h.max(1) as f32);
        let mut out = rgb.to_vec();
        for y in 0..h {
            let fy = (y as f32 + 0.5) * (work_h as f32 / (h.max(1) as f32)) - 0.5;
            let wy0 = (fy.round() as isize).clamp(0, (work_h - 1) as isize);
            for x in 0..w {
                let i = y * w + x;
                if !hole[i] {
                    continue;
                }
                let fx = (x as f32 + 0.5) * (work_w as f32 / (w.max(1) as f32)) - 0.5;
                let wx0 = (fx.round() as isize).clamp(0, (work_w - 1) as isize);
                let idx = i * 3;
                let base_r = pm_full[idx];
                let base_g = pm_full[idx + 1];
                let base_b = pm_full[idx + 2];
                let xm1 = x.saturating_sub(1);
                let xp1 = (x + 1).min(w - 1);
                let ym1 = y.saturating_sub(1);
                let yp1 = (y + 1).min(h - 1);
                let gx = (pm_full[(y * w + xp1) * 3 + 1] - pm_full[(y * w + xm1) * 3 + 1]).abs();
                let gy = (pm_full[(yp1 * w + x) * 3 + 1] - pm_full[(ym1 * w + x) * 3 + 1]).abs();
                let base_is_sky = base_b > base_r * 1.14
                    && base_b > base_g * 1.03
                    && base_g > 0.06
                    && (gx + gy) < 0.015;

                let mut hf_acc = [0.0f32; 3];
                let mut w_sum = 0.0f32;
                let mut max_w_col = 0.0f32;
                for dwy in -1isize..=1 {
                    let nwy = (wy0 + dwy).clamp(0, (work_h - 1) as isize) as usize;
                    let wy_dist = (fy - nwy as f32).abs();
                    let wy_wt = (1.6 - wy_dist).max(0.10);
                    for dwx in -1isize..=1 {
                        let nwx = (wx0 + dwx).clamp(0, (work_w - 1) as isize) as usize;
                        let wx_dist = (fx - nwx as f32).abs();
                        let w_spat = wy_wt * (1.6 - wx_dist).max(0.10);
                        let wi = nwy * work_w + nwx;
                        let (nn_sx, nn_sy) = nnf_work[wi];
                        let dx_full = ((nn_sx - nwx as i32) as f32 * sx_scale).round() as isize;
                        let dy_full = ((nn_sy - nwy as i32) as f32 * sy_scale).round() as isize;
                        let sx_full = (x as isize + dx_full).clamp(0, (w - 1) as isize) as usize;
                        let sy_full = (y as isize + dy_full).clamp(0, (h - 1) as isize) as usize;
                        if hole_dil[sy_full * w + sx_full] {
                            continue;
                        }
                        let w_src_i = (nn_sy.clamp(0, work_h as i32 - 1) as usize * work_w
                            + nn_sx.clamp(0, work_w as i32 - 1) as usize)
                            * 3;
                        let w_tgt_i = wi * 3;
                        let dr = work_rgb[w_src_i] - pm_work[w_tgt_i];
                        let dg = work_rgb[w_src_i + 1] - pm_work[w_tgt_i + 1];
                        let db = work_rgb[w_src_i + 2] - pm_work[w_tgt_i + 2];
                        let d_col2 = dr * dr + dg * dg * 1.4 + db * db;
                        let w_col = (-d_col2 * 140.0).exp();
                        if w_col > max_w_col {
                            max_w_col = w_col;
                        }
                        let wt = w_spat * w_col;
                        if wt > 1e-5 {
                            let s_idx = (sy_full * w + sx_full) * 3;
                            for c in 0..3 {
                                let hf = (rgb[s_idx + c] - rgb_upsampled[s_idx + c]).clamp(-0.22, 0.22);
                                hf_acc[c] += hf * wt;
                            }
                            w_sum += wt;
                        }
                    }
                }
                let conf = if w_sum > 1e-5 {
                    let inv_w = 1.0 / w_sum;
                    hf_acc[0] *= inv_w;
                    hf_acc[1] *= inv_w;
                    hf_acc[2] *= inv_w;
                    (max_w_col * (if base_is_sky { 0.35 } else { 0.95 })).clamp(0.0, 0.95)
                } else {
                    0.0
                };
                out[idx] = (base_r + conf * hf_acc[0]).max(0.0);
                out[idx + 1] = (base_g + conf * hf_acc[1]).max(0.0);
                out[idx + 2] = (base_b + conf * hf_acc[2]).max(0.0);
            }
        }
        out
    };

    // ISO-equivalent sensor grain injection inside the inpainted mask
    let auto_grain = estimate_context_grain(rgb, hole, w, h);
    let iso_grain = if cfg.iso > 100 {
        ((cfg.iso as f32 / 100.0).sqrt() - 1.0).clamp(0.0, 10.0) * 0.0018
    } else {
        0.0
    };
    let grain_amp = auto_grain.max(iso_grain);
    if grain_amp > 1e-4 {
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if hole[i] {
                    let luma_n = sensor_grain_sample(x, y, 0, cfg.seed) * grain_amp;
                    let chr_r = sensor_grain_sample(x, y, 1, cfg.seed) * (grain_amp * 0.35);
                    let chr_b = sensor_grain_sample(x, y, 2, cfg.seed) * (grain_amp * 0.35);
                    let idx = i * 3;
                    let lum_scale = final_rgb[idx + 1].abs().sqrt().clamp(0.15, 1.0);
                    final_rgb[idx] += (luma_n + chr_r) * lum_scale;
                    final_rgb[idx + 1] += luma_n * lum_scale;
                    final_rgb[idx + 2] += (luma_n + chr_b) * lum_scale;
                }
            }
        }
    }
    final_rgb
}

/// Runs LaMa local inpainting on a cropped context buffer `rgb` (`w × h × 3`) with binary mask
/// `hole` (`w × h`).
/// - For compact blemishes, moles (`lunares`), pimples, and small spots, runs 1:1 native-resolution
///   (`6K`) single-patch frequency-separated healing (`heal_native_blemish_6k`) to preserve 100%
///   of skin pores and sensor grain without downsampling.
/// - For large objects and complex scenes, runs Cloud Generative Diffusion (when configured) or
///   `lama_fp32.onnx` + harmonic `membrane_fill` + variance-preserving 1:1 native meso/micro patch quilting.
pub fn inpaint_lama_box(rgb: &[f32], hole: &[bool], w: usize, h: usize, cfg: &LamaConfig) -> Vec<f32> {
    if w == 0 || h == 0 || rgb.len() < w.saturating_mul(h).saturating_mul(3) || hole.len() < w.saturating_mul(h) {
        return rgb.to_vec();
    }
    if !hole.iter().any(|&b| b) {
        return rgb.to_vec();
    }

    // Dilate `hole` by 2px once so `heal_native_blemish_6k`, `run_lama_onnx_512`, and `membrane_fill`
    // share the exact same clean boundary outside any brush feather edge.
    let mut hole_dil = hole.to_vec();
    for y in 0..h {
        for x in 0..w {
            if hole[y * w + x] {
                for dy in -2isize..=2 {
                    for dx in -2isize..=2 {
                        if dx * dx + dy * dy <= 5 {
                            let ny = (y as isize + dy).clamp(0, (h - 1) as isize) as usize;
                            let nx = (x as isize + dx).clamp(0, (w - 1) as isize) as usize;
                            hole_dil[ny * w + nx] = true;
                        }
                    }
                }
            }
        }
    }

    // Fast path for compact blemishes, moles, skin spots, and small distractions:
    // 1:1 native-resolution single-patch frequency-separated healing with harmonic membrane illumination!
    if let Some(healed_6k) = heal_native_blemish_6k(rgb, hole, &hole_dil, w, h, cfg.seed) {
        return healed_6k;
    }

    // Serialize heavy inpainting across concurrent preview/zoom/thumbnail worker threads so the first
    // thread populates `INPAINT_WORK_CACHE` (and `cloud_cache` on disk) and all other threads reuse it.
    let _inpaint_serial_guard = INPAINT_LOCK.get_or_init(|| Mutex::new(())).lock().ok();
    let grid_sig = compute_unmasked_grid_sig(rgb, &hole_dil, w, h);

    let allow_cloud = cfg.max_side >= 1024;
    let (cloud_configured, mode_tag) = current_cloud_mode_tag();

    if cfg.seed != 0 {
        let cached_hit = INPAINT_WORK_CACHE
            .get_or_init(|| Mutex::new(Vec::new()))
            .lock()
            .ok()
            .and_then(|guard| {
                guard
                    .iter()
                    .find(|c| {
                        c.seed == cfg.seed
                            && c.mode_tag == mode_tag
                            && (!cloud_configured || c.is_cloud)
                            && (c.is_cloud || grid_sig_matches(&c.grid_sig, &grid_sig))
                            && ((w as f32 / h.max(1) as f32) - (c.work_w as f32 / c.work_h.max(1) as f32)).abs()
                                <= 0.12
                            && (if c.is_cloud {
                                c.work_w.max(c.work_h) >= 64
                            } else {
                                c.work_w.max(c.work_h) >= 96 || w.max(h) <= c.work_w.max(c.work_h)
                            })
                    })
                    .cloned()
            });
        if let Some(c) = cached_hit {
            return reconstruct_inpaint_from_work(
                rgb,
                hole,
                &hole_dil,
                w,
                h,
                c.work_w,
                c.work_h,
                &c.work_rgb,
                &c.pm_work,
                &c.pm_adapted_work,
                &c.nnf_work,
                c.is_cloud,
                cfg,
            );
        }
    }

    // 0. If Cloud Generative Diffusion Inpainting (Stability Erase / Fal.ai Bria Eraser / OpenAI / Gemini)
    //    is configured in `~/.mm-studio/ai_keys.json` or env vars, run true Generative Diffusion!
    if let Some(cloud_rgb) = try_cloud_generative_inpaint(rgb, &hole_dil, w, h, cfg.seed, allow_cloud) {
        let out = reconstruct_inpaint_from_work(
            rgb,
            hole,
            &hole_dil,
            w,
            h,
            w,
            h,
            &[],
            &[],
            &cloud_rgb,
            &[],
            true,
            cfg,
        );
        if cfg.seed != 0
            && w.max(h) >= 64
            && let Ok(mut guard) = INPAINT_WORK_CACHE.get_or_init(|| Mutex::new(Vec::new())).lock()
        {
            if let Some(existing) = guard.iter_mut().find(|c| c.seed == cfg.seed && c.mode_tag == mode_tag) {
                if !existing.is_cloud
                    || w.max(h) >= existing.work_w.max(existing.work_h)
                    || !grid_sig_matches(&existing.grid_sig, &grid_sig)
                {
                    existing.grid_sig = grid_sig;
                    existing.work_w = w;
                    existing.work_h = h;
                    existing.pm_adapted_work = cloud_rgb;
                    existing.is_cloud = true;
                }
            } else {
                if guard.len() >= 16 {
                    guard.remove(0);
                }
                guard.push(CachedInpaintWork {
                    seed: cfg.seed,
                    mode_tag,
                    grid_sig,
                    work_w: w,
                    work_h: h,
                    work_rgb: Vec::new(),
                    pm_work: Vec::new(),
                    pm_adapted_work: cloud_rgb,
                    nnf_work: Vec::new(),
                    is_cloud: true,
                });
            }
        }
        return out;
    }

    // 1. Downsample `(rgb, hole_dil)` to scale-invariant working resolution `(work_w, work_h)` (`max_side = 768`)
    //    FIRST so LaMa ONNX, `horiz_guide`, Laplace membranes, and Multi-Scale PatchMatch produce 100%
    //    identical structure at preview (`1366x913`) and 6K (`6000x4000`) in <150ms without 20MP blurs!
    let max_side = cfg.max_side.clamp(384, 768);
    let long = w.max(h);
    let (work_w, work_h) = if long > max_side {
        let scale = max_side as f32 / long as f32;
        (((w as f32 * scale).round() as usize).max(8), ((h as f32 * scale).round() as usize).max(8))
    } else {
        (w, h)
    };
    let work_rgb = if (work_w, work_h) != (w, h) {
        resize_rgb(rgb, w, h, work_w, work_h)
    } else {
        rgb.to_vec()
    };
    let work_hole = if (work_w, work_h) != (w, h) {
        let raw_wh = resize_mask(&hole_dil, w, h, work_w, work_h);
        // Dilate `work_hole` by 2 work pixels so the entire 4x4 Catmull-Rom bicubic upsampling stencil
        // for every pixel inside `hole` lies 100% inside the inpainted & membrane-adapted region!
        let mut dil_wh = raw_wh.clone();
        for y in 0..work_h {
            for x in 0..work_w {
                if raw_wh[y * work_w + x] {
                    for dy in -2isize..=2 {
                        let ny = (y as isize + dy).clamp(0, (work_h - 1) as isize) as usize;
                        for dx in -2isize..=2 {
                            if dx * dx + dy * dy <= 5 {
                                let nx = (x as isize + dx).clamp(0, (work_w - 1) as isize) as usize;
                                dil_wh[ny * work_w + nx] = true;
                            }
                        }
                    }
                }
            }
        }
        dil_wh
    } else {
        hole_dil.clone()
    };

    let mut work_guide = if work_w.max(work_h) >= 96 {
        run_lama_onnx_512(&work_rgb, &work_hole, work_w, work_h)
    } else {
        None
    };

    // 1b. Build Macro-Elevation Horizon Guide (`horiz_guide`) at `(work_w, work_h)`
    let r_macro_guide = 10usize;
    let mut num_8 = vec![0.0f32; work_w * work_h * 3];
    let mut den_8 = vec![0.0f32; work_w * work_h * 3];
    for i in 0..(work_w * work_h) {
        if !work_hole[i] {
            let idx = i * 3;
            num_8[idx] = work_rgb[idx];
            num_8[idx + 1] = work_rgb[idx + 1];
            num_8[idx + 2] = work_rgb[idx + 2];
            den_8[idx] = 1.0;
            den_8[idx + 1] = 1.0;
            den_8[idx + 2] = 1.0;
        }
    }
    let b_num8 = blur_rgb3(&num_8, work_w, work_h, 8);
    let b_den8 = blur_rgb3(&den_8, work_w, work_h, 8);
    let mut rgb_smooth8 = work_rgb.clone();
    for i in 0..(work_w * work_h) {
        let idx = i * 3;
        if b_den8[idx] > 1e-4 {
            rgb_smooth8[idx] = b_num8[idx] / b_den8[idx];
            rgb_smooth8[idx + 1] = b_num8[idx + 1] / b_den8[idx + 1];
            rgb_smooth8[idx + 2] = b_num8[idx + 2] / b_den8[idx + 2];
        }
    }
    // Compute `below_skyline[y * work_w + x]`: true once `y` is below both the top canopy onset
    // AND the lowest vertical run of true sky (`col_last_sky_y`) in `x ± rad_c`.
    // This ensures overhanging tree branches with sky peeking between them remain in the sky/canopy
    // zone while lower forest/mountain/ground pixels below all sky have `below_skyline == true`
    // monotonically downward with zero scanline jumps!
    let mut col_canopy_y = vec![work_h; work_w];
    let mut col_last_sky_y = vec![0usize; work_w];
    for x in 0..work_w {
        let mut non_sky_run = 0usize;
        let mut sky_run = 0usize;
        for y in 0..work_h {
            let i = y * work_w + x;
            if !work_hole[i] {
                let idx = i * 3;
                let r = rgb_smooth8[idx].max(1e-4);
                let g = rgb_smooth8[idx + 1].max(1e-4);
                let b = rgb_smooth8[idx + 2];
                let is_sky_c = b > r * 1.12 && b > g * 1.02 && g > 0.06;
                let is_true_sky = b > r * 1.14 && b > g * 1.03 && b > 0.16;
                if is_true_sky {
                    sky_run += 1;
                    if sky_run >= 3 {
                        col_last_sky_y[x] = y;
                    }
                } else {
                    sky_run = 0;
                }
                if !is_sky_c {
                    non_sky_run += 1;
                    if non_sky_run >= 3 && col_canopy_y[x] == work_h {
                        col_canopy_y[x] = y;
                    }
                } else {
                    non_sky_run = 0;
                }
            }
        }
    }
    let canopy_pad = (work_h / 32).max(6);
    let mut raw_skyline_y = vec![work_h; work_w];
    for x in 0..work_w {
        let rad_c = (work_w / 24).clamp(8, 20);
        let x0 = x.saturating_sub(rad_c);
        let x1 = (x + rad_c).min(work_w - 1);
        let mut min_cy = work_h;
        let mut max_sky_y = 0usize;
        for xx in x0..=x1 {
            min_cy = min_cy.min(col_canopy_y[xx]);
            max_sky_y = max_sky_y.max(col_last_sky_y[xx]);
        }
        raw_skyline_y[x] = (min_cy + canopy_pad).max(if max_sky_y > 0 { max_sky_y + 4 } else { 0 });
    }
    let mut below_skyline = vec![false; work_w * work_h];
    for y in 0..work_h {
        for x in 0..work_w {
            if y > raw_skyline_y[x] || (y > 0 && below_skyline[(y - 1) * work_w + x]) {
                below_skyline[y * work_w + x] = true;
            }
        }
        let mut x = 0usize;
        while x < work_w {
            if !work_hole[y * work_w + x] {
                x += 1;
                continue;
            }
            let x_start = x;
            while x < work_w && work_hole[y * work_w + x] {
                x += 1;
            }
            let x_end = x;
            let sl_l = if x_start > 0 {
                raw_skyline_y[x_start - 1] as f32
            } else {
                raw_skyline_y[x_end.min(work_w - 1)] as f32
            };
            let sl_r = if x_end < work_w {
                raw_skyline_y[x_end] as f32
            } else {
                sl_l
            };
            let span = (x_end - x_start).max(1) as f32;
            for hx in x_start..x_end {
                let t = (hx - x_start) as f32 / span;
                let sl_y = (sl_l * (1.0 - t) + sl_r * t).round() as usize;
                let prev_below = y > 0 && below_skyline[(y - 1) * work_w + hx];
                below_skyline[y * work_w + hx] = prev_below || (y > sl_y);
            }
        }
    }

    // Compute local texture roughness `local_rough[i]` (`5×5` mean absolute deviation from `rgb_smooth8`)
    // so textured backlit pine needles/branches (`local_rough >= 0.020`, which have `b > g > r` against
    // a bright blue sky) are NEVER mistaken for featureless open sky (`local_rough < 0.018`)!
    let mut raw_dev = vec![0.0f32; work_w * work_h];
    for i in 0..(work_w * work_h) {
        if !work_hole[i] {
            let idx = i * 3;
            let dr = (work_rgb[idx] - rgb_smooth8[idx]).abs();
            let dg = (work_rgb[idx + 1] - rgb_smooth8[idx + 1]).abs();
            let db = (work_rgb[idx + 2] - rgb_smooth8[idx + 2]).abs();
            raw_dev[i] = (dr + dg * 1.2 + db) * 0.3125;
        }
    }
    let mut local_rough = vec![0.0f32; work_w * work_h];
    for y in 0..work_h {
        let y0 = y.saturating_sub(3);
        let y1 = (y + 3).min(work_h - 1);
        for x in 0..work_w {
            let x0 = x.saturating_sub(3);
            let x1 = (x + 3).min(work_w - 1);
            let mut s = 0.0f32;
            let mut n = 0.0f32;
            for yy in y0..=y1 {
                for xx in x0..=x1 {
                    let ii = yy * work_w + xx;
                    if !work_hole[ii] {
                        s += raw_dev[ii];
                        n += 1.0;
                    }
                }
            }
            if n > 0.0 {
                local_rough[y * work_w + x] = s / n;
            }
        }
    }

    // Pure harmonic sky membrane (`sky_membrane`) and non-sky canopy membrane (`canopy_membrane`):
    let mut is_sky_out = vec![false; work_w * work_h];
    let mut non_sky_or_hole = vec![true; work_w * work_h];
    let mut sky_or_hole = vec![true; work_w * work_h];
    let mut known_sky_cnt = 0usize;
    let mut known_canopy_cnt = 0usize;
    for i in 0..(work_w * work_h) {
        if !work_hole[i] {
            let idx = i * 3;
            let r = rgb_smooth8[idx].max(1e-4);
            let g = rgb_smooth8[idx + 1].max(1e-4);
            let b = rgb_smooth8[idx + 2];
            let raw_max = work_rgb[idx].max(work_rgb[idx + 1]).max(work_rgb[idx + 2]);
            let is_smooth_sky = local_rough[i] < 0.022 && b > r * 1.14 && b > g * 1.03 && g > 0.14;
            if !below_skyline[i] && is_smooth_sky {
                is_sky_out[i] = true;
                non_sky_or_hole[i] = false;
                known_sky_cnt += 1;
            } else if !is_smooth_sky && r.max(g).max(b) >= 0.055 && raw_max >= 0.048 {
                sky_or_hole[i] = false;
                known_canopy_cnt += 1;
            }
        }
    }
    let mut near_sky_out = is_sky_out.clone();
    for y in 0..work_h {
        let y0 = y.saturating_sub(2);
        let y1 = (y + 2).min(work_h - 1);
        for x in 0..work_w {
            let x0 = x.saturating_sub(2);
            let x1 = (x + 2).min(work_w - 1);
            let mut found = false;
            'search_sky: for yy in y0..=y1 {
                for xx in x0..=x1 {
                    if is_sky_out[yy * work_w + xx] {
                        found = true;
                        break 'search_sky;
                    }
                }
            }
            near_sky_out[y * work_w + x] = found;
        }
    }
    let sky_membrane = if known_sky_cnt >= 32 {
        membrane_fill(work_w, work_h, 3, &rgb_smooth8, &non_sky_or_hole)
    } else {
        rgb_smooth8.clone()
    };
    let canopy_membrane = if known_canopy_cnt >= 32 {
        membrane_fill(work_w, work_h, 3, &rgb_smooth8, &sky_or_hole)
    } else {
        rgb_smooth8.clone()
    };

    let mut row_open_sky_factor = vec![0.0f32; work_h];
    let mut row_has_midtone = vec![false; work_h];
    let mut row_med_g = vec![0.0f32; work_h];
    let mut g_buf = Vec::with_capacity(work_w);
    for y in 0..work_h {
        let y0 = y.saturating_sub(6);
        let y1 = (y + 6).min(work_h - 1);
        let mut n_unm = 0usize;
        let mut n_sky = 0usize;
        for yy in y0..=y1 {
            for x in 0..work_w {
                let i = yy * work_w + x;
                if !work_hole[i] {
                    n_unm += 1;
                    if is_sky_out[i] {
                        n_sky += 1;
                    }
                }
            }
        }
        let sky_frac = n_sky as f32 / (n_unm.max(1) as f32);
        row_open_sky_factor[y] = ((sky_frac - 0.45) / 0.22).clamp(0.0, 1.0);

        g_buf.clear();
        let mut mid_cnt = 0usize;
        for x in 0..work_w {
            let i = y * work_w + x;
            if !work_hole[i] {
                let idx = i * 3;
                let r = rgb_smooth8[idx];
                let g = rgb_smooth8[idx + 1];
                let b = rgb_smooth8[idx + 2];
                if g > 0.065 || b > 0.09 {
                    mid_cnt += 1;
                }
                if r.max(g).max(b) >= 0.032 {
                    g_buf.push(g);
                }
            }
        }
        row_has_midtone[y] = mid_cnt >= 8;
        if !g_buf.is_empty() {
            g_buf.sort_by(|a, b| a.total_cmp(b));
            row_med_g[y] = g_buf[g_buf.len() / 2];
        }
    }

    let mut horiz_guide = rgb_smooth8.clone();
    let swath_w = (work_w / 6).clamp(28, 96);
    for y in 0..work_h {
        let skip_dark_trunk = row_has_midtone[y];
        let row_is_canopy = row_open_sky_factor[y] < 0.45;
        let mut x = 0usize;
        while x < work_w {
            if !work_hole[y * work_w + x] {
                x += 1;
                continue;
            }
            let x_start = x;
            while x < work_w && work_hole[y * work_w + x] {
                x += 1;
            }
            let x_end = x;
            let left_is_below = row_is_canopy || (x_start > 0 && below_skyline[y * work_w + (x_start - 1)]);
            let right_is_below = row_is_canopy || (x_end < work_w && below_skyline[y * work_w + x_end]);
            let mut left_rgb = [0.0f32; 3];
            let mut left_n = 0.0f32;
            let l_hi = x_start.saturating_sub(1).max(1);
            let l_lo = x_start.saturating_sub(swath_w);
            for lx in l_lo..l_hi {
                let li = y * work_w + lx;
                if !work_hole[li] {
                    let idx = li * 3;
                    let r = rgb_smooth8[idx].max(1e-4);
                    let g = rgb_smooth8[idx + 1].max(1e-4);
                    let b = rgb_smooth8[idx + 2];
                    if left_is_below && is_sky_out[li] {
                        continue;
                    }
                    let raw_max = work_rgb[idx].max(work_rgb[idx + 1]).max(work_rgb[idx + 2]);
                    if skip_dark_trunk && (r.max(g).max(b) < 0.055 || raw_max < 0.048) {
                        continue;
                    }
                    let dist = (x_start - lx) as f32;
                    let wt = 1.0 / (1.0 + dist * dist * 0.12);
                    left_rgb[0] += rgb_smooth8[idx] * wt;
                    left_rgb[1] += rgb_smooth8[idx + 1] * wt;
                    left_rgb[2] += rgb_smooth8[idx + 2] * wt;
                    left_n += wt;
                }
            }
            let mut right_rgb = [0.0f32; 3];
            let mut right_n = 0.0f32;
            let r_lo = (x_end + 1).min(work_w);
            let r_hi = (x_end + swath_w).min(work_w);
            for rx in r_lo..r_hi {
                let ri_px = y * work_w + rx;
                if !work_hole[ri_px] {
                    let idx = ri_px * 3;
                    let r = rgb_smooth8[idx].max(1e-4);
                    let g = rgb_smooth8[idx + 1].max(1e-4);
                    let b = rgb_smooth8[idx + 2];
                    if right_is_below && is_sky_out[ri_px] {
                        continue;
                    }
                    let raw_max = work_rgb[idx].max(work_rgb[idx + 1]).max(work_rgb[idx + 2]);
                    if skip_dark_trunk && (r.max(g).max(b) < 0.055 || raw_max < 0.048) {
                        continue;
                    }
                    let dist = (rx - x_end) as f32;
                    let wt = 1.0 / (1.0 + dist * dist * 0.12);
                    right_rgb[0] += rgb_smooth8[idx] * wt;
                    right_rgb[1] += rgb_smooth8[idx + 1] * wt;
                    right_rgb[2] += rgb_smooth8[idx + 2] * wt;
                    right_n += wt;
                }
            }
            if left_n > 0.0 {
                left_rgb[0] /= left_n;
                left_rgb[1] /= left_n;
                left_rgb[2] /= left_n;
            }
            if right_n > 0.0 {
                right_rgb[0] /= right_n;
                right_rgb[1] /= right_n;
                right_rgb[2] /= right_n;
            }
            let span = (x_end - x_start).max(1) as f32;
            for hx in x_start..x_end {
                let p_i = y * work_w + hx;
                let idx = p_i * 3;
                if left_n > 0.0 && right_n > 0.0 {
                    let t = (hx - x_start) as f32 / span;
                    horiz_guide[idx] = left_rgb[0] * (1.0 - t) + right_rgb[0] * t;
                    horiz_guide[idx + 1] = left_rgb[1] * (1.0 - t) + right_rgb[1] * t;
                    horiz_guide[idx + 2] = left_rgb[2] * (1.0 - t) + right_rgb[2] * t;
                } else if left_n > 0.0 {
                    horiz_guide[idx] = left_rgb[0];
                    horiz_guide[idx + 1] = left_rgb[1];
                    horiz_guide[idx + 2] = left_rgb[2];
                } else if right_n > 0.0 {
                    horiz_guide[idx] = right_rgb[0];
                    horiz_guide[idx + 1] = right_rgb[1];
                    horiz_guide[idx + 2] = right_rgb[2];
                } else if below_skyline[p_i] || row_is_canopy {
                    horiz_guide[idx] = canopy_membrane[idx];
                    horiz_guide[idx + 1] = canopy_membrane[idx + 1];
                    horiz_guide[idx + 2] = canopy_membrane[idx + 2];
                }
                if below_skyline[p_i] || row_is_canopy {
                    let hr = horiz_guide[idx].max(1e-4);
                    let hg = horiz_guide[idx + 1].max(1e-4);
                    let hb = horiz_guide[idx + 2];
                    if !left_is_below || (hb > hr * 1.08 && hb > hg * 1.01 && hb > 0.10) {
                        horiz_guide[idx] = canopy_membrane[idx];
                        horiz_guide[idx + 1] = canopy_membrane[idx + 1];
                        horiz_guide[idx + 2] = canopy_membrane[idx + 2];
                    }
                }
            }
        }
    }
    // Vertical canopy continuation: when a hole sits above or inside a vertical tree crown (where
    // unmasked pixels directly below `y_end` or above `y_start` in column `x` have textured foliage
    // `local_rough >= 0.020`, while horizontal neighbors on the sides of the tree are open sky),
    // blend vertical boundary canopy colors upward/downward into `horiz_guide`!
    let swath_h = (work_h / 6).clamp(24, 80);
    for x in 0..work_w {
        let mut y = 0usize;
        while y < work_h {
            if !work_hole[y * work_w + x] {
                y += 1;
                continue;
            }
            let y_start = y;
            while y < work_h && work_hole[y * work_w + x] {
                y += 1;
            }
            let y_end = y;
            let mut top_rgb = [0.0f32; 3];
            let mut top_rough = 0.0f32;
            let mut top_n = 0.0f32;
            for ty in y_start.saturating_sub(swath_h)..y_start {
                let ti = ty * work_w + x;
                if !work_hole[ti] && !is_sky_out[ti] {
                    let idx = ti * 3;
                    let r = rgb_smooth8[idx];
                    let g = rgb_smooth8[idx + 1];
                    let b = rgb_smooth8[idx + 2];
                    let raw_max = work_rgb[idx].max(work_rgb[idx + 1]).max(work_rgb[idx + 2]);
                    if r.max(g).max(b) < 0.055 || raw_max < 0.048 {
                        continue;
                    }
                    let dist = (y_start - ty) as f32;
                    let wt = 1.0 / (1.0 + dist * dist * 0.10);
                    top_rgb[0] += r * wt;
                    top_rgb[1] += g * wt;
                    top_rgb[2] += b * wt;
                    top_rough += local_rough[ti] * wt;
                    top_n += wt;
                }
            }
            let mut bot_rgb = [0.0f32; 3];
            let mut bot_rough = 0.0f32;
            let mut bot_n = 0.0f32;
            for by in (y_end + 1).min(work_h)..(y_end + swath_h).min(work_h) {
                let bi = by * work_w + x;
                if !work_hole[bi] && !is_sky_out[bi] {
                    let idx = bi * 3;
                    let r = rgb_smooth8[idx];
                    let g = rgb_smooth8[idx + 1];
                    let b = rgb_smooth8[idx + 2];
                    let raw_max = work_rgb[idx].max(work_rgb[idx + 1]).max(work_rgb[idx + 2]);
                    if r.max(g).max(b) < 0.055 || raw_max < 0.048 {
                        continue;
                    }
                    let dist = (by - y_end) as f32;
                    let wt = 1.0 / (1.0 + dist * dist * 0.10);
                    bot_rgb[0] += r * wt;
                    bot_rgb[1] += g * wt;
                    bot_rgb[2] += b * wt;
                    bot_rough += local_rough[bi] * wt;
                    bot_n += wt;
                }
            }
            if top_n > 0.0 {
                top_rgb[0] /= top_n;
                top_rgb[1] /= top_n;
                top_rgb[2] /= top_n;
                top_rough /= top_n;
            }
            if bot_n > 0.0 {
                bot_rgb[0] /= bot_n;
                bot_rgb[1] /= bot_n;
                bot_rgb[2] /= bot_n;
                bot_rough /= bot_n;
            }
            let max_v_rough = top_rough.max(bot_rough);
            let v_canopy_w = ((max_v_rough - 0.016) / 0.016).clamp(0.0, 0.72);
            if v_canopy_w > 0.0 && (top_n > 0.0 || bot_n > 0.0) {
                let v_span = (y_end - y_start).max(1) as f32;
                for hy in y_start..y_end {
                    let idx = (hy * work_w + x) * 3;
                    let vr = if top_n > 0.0 && bot_n > 0.0 {
                        let ty = (hy - y_start) as f32 / v_span;
                        [
                            top_rgb[0] * (1.0 - ty) + bot_rgb[0] * ty,
                            top_rgb[1] * (1.0 - ty) + bot_rgb[1] * ty,
                            top_rgb[2] * (1.0 - ty) + bot_rgb[2] * ty,
                        ]
                    } else if bot_n > 0.0 {
                        bot_rgb
                    } else {
                        top_rgb
                    };
                    horiz_guide[idx] = horiz_guide[idx] * (1.0 - v_canopy_w) + vr[0] * v_canopy_w;
                    horiz_guide[idx + 1] = horiz_guide[idx + 1] * (1.0 - v_canopy_w) + vr[1] * v_canopy_w;
                    horiz_guide[idx + 2] = horiz_guide[idx + 2] * (1.0 - v_canopy_w) + vr[2] * v_canopy_w;
                }
            }
        }
    }
    let horiz_guide = blur_rgb3(&horiz_guide, work_w, work_h, r_macro_guide);
    if let Some(ref mut lama_net) = work_guide {
        let mut hole_sky_w = vec![0.0f32; work_w * work_h];
        for y in 0..work_h {
            let row_is_canopy = row_open_sky_factor[y] < 0.45;
            for x in 0..work_w {
                let i = y * work_w + x;
                if work_hole[i] {
                    let idx = i * 3;
                    let mut lr = lama_net[idx];
                    let mut lg = lama_net[idx + 1];
                    let mut lb = lama_net[idx + 2];
                    let mut hr = horiz_guide[idx];
                    let mut hg = horiz_guide[idx + 1];
                    let mut hb = horiz_guide[idx + 2];
                    if row_has_midtone[y] && row_med_g[y] > 0.085 {
                        let dark_pull = ((row_med_g[y] * 0.60 - lg) / (row_med_g[y] * 0.45).max(1e-3)).clamp(0.0, 1.0);
                        if dark_pull > 0.0 {
                            lr = lr * (1.0 - dark_pull) + hr * dark_pull;
                            lg = lg * (1.0 - dark_pull) + hg * dark_pull;
                            lb = lb * (1.0 - dark_pull) + hb * dark_pull;
                        }
                    }
                    if below_skyline[i] || row_is_canopy {
                        let l_blue = (((lb / lr.max(1e-4)) - 1.08) * 8.0)
                            .min(((lb / lg.max(1e-4)) - 1.01) * 12.0)
                            .min((lb - 0.10) * 10.0)
                            .clamp(0.0, 1.0);
                        if l_blue > 0.0 {
                            lr = lr * (1.0 - l_blue) + hr * l_blue;
                            lg = lg * (1.0 - l_blue) + hg * l_blue;
                            lb = lb * (1.0 - l_blue) + hb * l_blue;
                        }
                        let h_blue = (((hb / hr.max(1e-4)) - 1.08) * 8.0)
                            .min(((hb / hg.max(1e-4)) - 1.01) * 12.0)
                            .min((hb - 0.10) * 10.0)
                            .clamp(0.0, 1.0);
                        if h_blue > 0.0 {
                            hr = hr * (1.0 - h_blue) + canopy_membrane[idx] * h_blue;
                            hg = hg * (1.0 - h_blue) + canopy_membrane[idx + 1] * h_blue;
                            hb = hb * (1.0 - h_blue) + canopy_membrane[idx + 2] * h_blue;
                        }
                    }
                    let lama_sky_t = (((lb / lr.max(1e-4)) - 1.08) * 8.0)
                        .min(((lb / lg.max(1e-4)) - 1.01) * 16.0)
                        .min((lg - 0.04) * 20.0)
                        .clamp(0.0, 1.0);
                    let horiz_sky_t = (((hb / hr.max(1e-4)) - 1.08) * 8.0)
                        .min(((hb / hg.max(1e-4)) - 1.01) * 16.0)
                        .min((hg - 0.04) * 20.0)
                        .clamp(0.0, 1.0);
                    let mut top_sky_t = 0.0f32;
                    let mut yy = y;
                    while yy > 0 && work_hole[yy * work_w + x] {
                        yy -= 1;
                    }
                    if !work_hole[yy * work_w + x] && !below_skyline[i] {
                        let dist_y = (y - yy) as f32 / ((work_h as f32 * 0.16).max(16.0));
                        let prox_t = (1.0 - dist_y).clamp(0.0, 1.0);
                        if prox_t > 0.0 {
                            let ti = (yy * work_w + x) * 3;
                            let tr = rgb_smooth8[ti];
                            let tg = rgb_smooth8[ti + 1];
                            let tb = rgb_smooth8[ti + 2];
                            top_sky_t = (((tb / tr.max(1e-4)) - 1.08) * 8.0)
                                .min(((tb / tg.max(1e-4)) - 1.01) * 16.0)
                                .min((tg - 0.04) * 20.0)
                                .clamp(0.0, 1.0)
                                * prox_t;
                        }
                    }
                    let mut yy_bot = y;
                    while yy_bot + 1 < work_h && work_hole[(yy_bot + 1) * work_w + x] {
                        yy_bot += 1;
                    }
                    if yy_bot + 1 < work_h {
                        yy_bot += 1;
                    }
                    let t_bot = if !work_hole[yy_bot * work_w + x]
                        && !near_sky_out[yy_bot * work_w + x]
                    {
                        ((yy_bot - y) as f32 / 8.0).clamp(0.0, 1.0)
                    } else {
                        1.0
                    };
                    let sky_w = if known_sky_cnt >= 32 && !below_skyline[i] {
                        ((horiz_sky_t * t_bot).max(lama_sky_t * top_sky_t) * row_open_sky_factor[y]).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    hole_sky_w[i] = sky_w;
                    let sky_override = lama_sky_t * top_sky_t * (1.0 - horiz_sky_t);
                    let w_h = (0.45 * (1.0 - sky_override) * t_bot).clamp(0.06, 0.55);
                    let gnd_r = w_h * hr + (1.0 - w_h) * lr;
                    let gnd_g = w_h * hg + (1.0 - w_h) * lg;
                    let gnd_b = w_h * hb + (1.0 - w_h) * lb;
                    lama_net[idx] = sky_w * sky_membrane[idx] + (1.0 - sky_w) * gnd_r;
                    lama_net[idx + 1] = sky_w * sky_membrane[idx + 1] + (1.0 - sky_w) * gnd_g;
                    lama_net[idx + 2] = sky_w * sky_membrane[idx + 2] + (1.0 - sky_w) * gnd_b;
                } else {
                    let idx = i * 3;
                    lama_net[idx] = work_rgb[idx];
                    lama_net[idx + 1] = work_rgb[idx + 1];
                    lama_net[idx + 2] = work_rgb[idx + 2];
                }
            }
        }
        let lama_low8 = blur_rgb3(lama_net, work_w, work_h, 8);
        let mut diff = vec![0.0f32; work_w * work_h * 3];
        for i in 0..(work_w * work_h) {
            // Only compute boundary diff on same-class non-sky pixels away from anti-aliased sky edges
            // so neither dark palm crowns pull sky dark nor sky-frond edge pixels inject grey haze!
            if !work_hole[i] && !near_sky_out[i] {
                let idx = i * 3;
                let lr = lama_low8[idx].max(1e-4);
                let lg = lama_low8[idx + 1].max(1e-4);
                let lb = lama_low8[idx + 2];
                let in_is_sky = lb > lr * 1.12 && lb > lg * 1.02 && lg > 0.06;
                if !in_is_sky {
                    diff[idx] = (rgb_smooth8[idx] - lama_low8[idx]).clamp(-0.14, 0.14);
                    diff[idx + 1] = (rgb_smooth8[idx + 1] - lama_low8[idx + 1]).clamp(-0.14, 0.14);
                    diff[idx + 2] = (rgb_smooth8[idx + 2] - lama_low8[idx + 2]).clamp(-0.14, 0.14);
                }
            }
        }
        let smooth_diff = membrane_fill(work_w, work_h, 3, &diff, &work_hole);
        for i in 0..(work_w * work_h) {
            if work_hole[i] {
                let idx = i * 3;
                let gnd_w = 1.0 - hole_sky_w[i];
                for c in 0..3 {
                    let max_lift = (lama_net[idx + c] * 0.65 + 0.015).min(0.12);
                    let d = (smooth_diff[idx + c] * gnd_w).clamp(-0.14, max_lift);
                    lama_net[idx + c] = (lama_net[idx + c] + d).max(0.0);
                }
            }
        }
    } else {
        work_guide = Some(horiz_guide);
    }

    let params = CompleteParams {
        patch_radius: 4,
        em_iters: 7,
        min_em_iters: 2,
        pm_iters: 3,
        refine_extent: 420,
        seed: cfg.seed,
    };

    let mut final_rgb = if let Some((pm_work, nnf_work)) =
        complete_guided_nnf(work_w, work_h, 3, &work_rgb, &work_hole, work_guide.as_deref(), &params)
    {
        // Class-consistent illumination adaptation at `(work_w, work_h)`:
        // 1. Non-sky (foliage/ground) boundary adaptation (`smooth_pm_diff`) averages ONLY non-sky
        //    exterior pixels (`!near_sky_out`) against ONLY non-sky interior pixels so neither bright
        //    sky nor anti-aliased sky-frond silhouette pixels inject dark or milky plumes!
        // 2. Sky pixels use `sky_membrane + (work_rgb[s] - rgb_smooth8[s])` so both outer hole
        //    boundaries (`Curve A`) and internal NNF source transitions (`Curve B`) are 100% seamless!
        let r_illum = (work_w.min(work_h) / 36).clamp(6, 14);
        let mut num_out_gnd = vec![0.0f32; work_w * work_h * 3];
        let mut den_out_gnd = vec![0.0f32; work_w * work_h * 3];
        let mut num_in_gnd = vec![0.0f32; work_w * work_h * 3];
        let mut den_in_gnd = vec![0.0f32; work_w * work_h * 3];
        for i in 0..(work_w * work_h) {
            let idx = i * 3;
            if work_hole[i] {
                let pr = pm_work[idx].max(1e-4);
                let pg = pm_work[idx + 1].max(1e-4);
                let pb = pm_work[idx + 2];
                let (nn_sx, nn_sy) = nnf_work[i];
                let sx_c = nn_sx.clamp(0, work_w as i32 - 1) as usize;
                let sy_c = nn_sy.clamp(0, work_h as i32 - 1) as usize;
                let si_px = sy_c * work_w + sx_c;
                let is_sky_pm = !below_skyline[i]
                    && pb > pr * 1.12
                    && pb > pg * 1.02
                    && pg > 0.06
                    && local_rough[si_px] < 0.020;
                if !is_sky_pm {
                    num_in_gnd[idx] = pm_work[idx];
                    num_in_gnd[idx + 1] = pm_work[idx + 1];
                    num_in_gnd[idx + 2] = pm_work[idx + 2];
                    den_in_gnd[idx] = 1.0;
                    den_in_gnd[idx + 1] = 1.0;
                    den_in_gnd[idx + 2] = 1.0;
                }
            } else if !near_sky_out[i] {
                num_out_gnd[idx] = work_rgb[idx];
                num_out_gnd[idx + 1] = work_rgb[idx + 1];
                num_out_gnd[idx + 2] = work_rgb[idx + 2];
                den_out_gnd[idx] = 1.0;
                den_out_gnd[idx + 1] = 1.0;
                den_out_gnd[idx + 2] = 1.0;
            }
        }
        let b_num_out = blur_rgb3(&num_out_gnd, work_w, work_h, r_illum);
        let b_den_out = blur_rgb3(&den_out_gnd, work_w, work_h, r_illum);
        let b_num_in = blur_rgb3(&num_in_gnd, work_w, work_h, r_illum);
        let b_den_in = blur_rgb3(&den_in_gnd, work_w, work_h, r_illum);
        let mut diff_pm = vec![0.0f32; work_w * work_h * 3];
        for i in 0..(work_w * work_h) {
            if !work_hole[i] && !near_sky_out[i] {
                let idx = i * 3;
                let w_out = b_den_out[idx];
                let w_in = b_den_in[idx];
                if w_out > 0.02 && w_in > 0.02 {
                    let r_true = b_num_out[idx] / w_out;
                    let g_true = b_num_out[idx + 1] / w_out;
                    let b_true = b_num_out[idx + 2] / w_out;
                    let p_r = b_num_in[idx] / w_in;
                    let p_g = b_num_in[idx + 1] / w_in;
                    let p_b = b_num_in[idx + 2] / w_in;
                    diff_pm[idx] = (r_true - p_r).clamp(-0.14, 0.14);
                    diff_pm[idx + 1] = (g_true - p_g).clamp(-0.14, 0.14);
                    diff_pm[idx + 2] = (b_true - p_b).clamp(-0.14, 0.14);
                }
            }
        }
        let smooth_pm_diff = membrane_fill(work_w, work_h, 3, &diff_pm, &work_hole);
        let mut pm_adapted_work = pm_work.clone();
        for i in 0..(work_w * work_h) {
            if work_hole[i] {
                let idx = i * 3;
                let pr = pm_work[idx].max(1e-4);
                let pg = pm_work[idx + 1].max(1e-4);
                let pb = pm_work[idx + 2];
                let (nn_sx, nn_sy) = nnf_work[i];
                let sx_c = nn_sx.clamp(0, work_w as i32 - 1) as usize;
                let sy_c = nn_sy.clamp(0, work_h as i32 - 1) as usize;
                let si_px = sy_c * work_w + sx_c;
                let si = si_px * 3;
                let src_smooth_sky = ((0.022 - local_rough[si_px]) / 0.008).clamp(0.0, 1.0);
                let pm_sky_t = if known_sky_cnt >= 32 && !below_skyline[i] && src_smooth_sky > 0.0 {
                    ((((pb / pr) - 1.10) * 6.0)
                        .min(((pb / pg) - 1.02) * 12.0)
                        .min((pg - 0.05) * 16.0)
                        .clamp(0.0, 1.0))
                        * row_open_sky_factor[i / work_w]
                        * src_smooth_sky
                } else {
                    0.0
                };
                for c in 0..3 {
                    let max_lift = (pm_work[idx + c] * 0.65 + 0.015).min(0.12);
                    let delta = (smooth_pm_diff[idx + c] * 0.85).clamp(-0.14, max_lift);
                    let gnd_val = (pm_work[idx + c] + delta).max(0.0);
                    if pm_sky_t > 0.0 {
                        let sky_hf = (work_rgb[si + c] - rgb_smooth8[si + c]).clamp(-0.025, 0.025);
                        let sky_val = (sky_membrane[idx + c] + sky_hf).max(0.0);
                        pm_adapted_work[idx + c] = pm_sky_t * sky_val + (1.0 - pm_sky_t) * gnd_val;
                    } else {
                        pm_adapted_work[idx + c] = gnd_val;
                    }
                }
            }
        }

        let out = reconstruct_inpaint_from_work(
            rgb,
            hole,
            &hole_dil,
            w,
            h,
            work_w,
            work_h,
            &work_rgb,
            &pm_work,
            &pm_adapted_work,
            &nnf_work,
            false,
            cfg,
        );
        if cfg.seed != 0
            && !cloud_configured
            && work_w.max(work_h) >= 96
            && let Ok(mut guard) = INPAINT_WORK_CACHE.get_or_init(|| Mutex::new(Vec::new())).lock()
        {
            if let Some(existing) = guard.iter_mut().find(|c| c.seed == cfg.seed && c.mode_tag == mode_tag) {
                if !existing.is_cloud
                    && (work_w.max(work_h) >= existing.work_w.max(existing.work_h)
                        || !grid_sig_matches(&existing.grid_sig, &grid_sig))
                {
                    existing.grid_sig = grid_sig;
                    existing.work_w = work_w;
                    existing.work_h = work_h;
                    existing.work_rgb = work_rgb;
                    existing.pm_work = pm_work;
                    existing.pm_adapted_work = pm_adapted_work;
                    existing.nnf_work = nnf_work;
                    existing.is_cloud = false;
                }
            } else {
                if guard.len() >= 16 {
                    guard.remove(0);
                }
                guard.push(CachedInpaintWork {
                    seed: cfg.seed,
                    mode_tag,
                    grid_sig,
                    work_w,
                    work_h,
                    work_rgb,
                    pm_work,
                    pm_adapted_work,
                    nnf_work,
                    is_cloud: false,
                });
            }
        }
        return out;
    } else {
        membrane_fill(w, h, 3, rgb, hole)
    };

    // 5. ISO-equivalent sensor grain injection inside the inpainted mask
    let auto_grain = estimate_context_grain(rgb, hole, w, h);
    let iso_grain = if cfg.iso > 100 {
        ((cfg.iso as f32 / 100.0).sqrt() - 1.0).clamp(0.0, 10.0) * 0.0018
    } else {
        0.0
    };
    let grain_amp = auto_grain.max(iso_grain);
    if grain_amp > 1e-4 {
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if hole[i] {
                    let luma_n = sensor_grain_sample(x, y, 0, cfg.seed) * grain_amp;
                    let chr_r = sensor_grain_sample(x, y, 1, cfg.seed) * (grain_amp * 0.35);
                    let chr_b = sensor_grain_sample(x, y, 2, cfg.seed) * (grain_amp * 0.35);
                    let idx = i * 3;
                    let lum_scale = final_rgb[idx + 1].abs().sqrt().clamp(0.15, 1.0);
                    final_rgb[idx] += (luma_n + chr_r) * lum_scale;
                    final_rgb[idx + 1] += luma_n * lum_scale;
                    final_rgb[idx + 2] += (luma_n + chr_b) * lum_scale;
                }
            }
        }
    }

    final_rgb
}

#[derive(Clone)]
struct CachedOutpaintWork {
    seed: u64,
    grid_sig: [f32; 96],
    work_w: usize,
    work_h: usize,
    work_rgb: Vec<f32>,
    pm_adapted_work: Vec<f32>,
    smooth_tex_std: Vec<f32>,
    nnf_work: Vec<(i32, i32)>,
}

static OUTPAINT_WORK_CACHE: OnceLock<Mutex<Vec<CachedOutpaintWork>>> = OnceLock::new();

#[allow(clippy::too_many_arguments)]
fn reconstruct_outpaint_from_work(
    rgb: &[f32],
    border_hole: &[bool],
    hole_dil: &[bool],
    dist_in: &[u8],
    feather_r: isize,
    cx_lo: usize,
    cx_hi: usize,
    cy_lo: usize,
    cy_hi: usize,
    w: usize,
    h: usize,
    work_w: usize,
    work_h: usize,
    work_rgb: &[f32],
    pm_adapted_work: &[f32],
    smooth_tex_std: &[f32],
    nnf_work: &[(i32, i32)],
    cfg: &LamaConfig,
) -> Vec<f32> {
    let pm_full = if (work_w, work_h) != (w, h) {
        resize_rgb_bicubic(pm_adapted_work, work_w, work_h, w, h)
    } else {
        pm_adapted_work.to_vec()
    };
    let rgb_upsampled = if (work_w, work_h) != (w, h) {
        resize_rgb_bicubic(work_rgb, work_w, work_h, w, h)
    } else {
        work_rgb.to_vec()
    };
    let sx_scale = w as f32 / (work_w.max(1) as f32);
    let sy_scale = h as f32 / (work_h.max(1) as f32);
    let mut out = rgb.to_vec();
    let feather_f = (feather_r as f32).max(1.0);
    for y in 0..h {
        let fy = (y as f32 + 0.5) * (work_h as f32 / (h.max(1) as f32)) - 0.5;
        let wy0 = (fy.round() as isize).clamp(0, (work_h - 1) as isize);
        for x in 0..w {
            let i = y * w + x;
            if !hole_dil[i] {
                continue;
            }
            let fx = (x as f32 + 0.5) * (work_w as f32 / (w.max(1) as f32)) - 0.5;
            let wx0 = (fx.round() as isize).clamp(0, (work_w - 1) as isize);
            let idx = i * 3;
            let base_r = pm_full[idx];
            let base_g = pm_full[idx + 1];
            let base_b = pm_full[idx + 2];
            let loc_std = smooth_tex_std[(wy0 as usize * work_w + wx0 as usize) * 3];
            let sky_band_t = (((h as f32 * 0.50) - y as f32) / (h as f32 * 0.06).max(1.0)).clamp(0.0, 1.0);
            let raw_w_tex = ((loc_std - 0.008) / 0.018).clamp(0.0, 1.0);
            let w_tex = 1.0 - sky_band_t * (1.0 - raw_w_tex);

            let mut hf_acc = [0.0f32; 3];
            let mut w_sum = 0.0f32;
            let mut max_w_col = 0.0f32;
            if (work_w, work_h) != (w, h) && w_tex > 1e-3 {
                for dwy in -1isize..=1 {
                    let nwy = (wy0 + dwy).clamp(0, (work_h - 1) as isize) as usize;
                    let wy_dist = (fy - nwy as f32).abs();
                    let wy_wt = (1.6 - wy_dist).max(0.10);
                    for dwx in -1isize..=1 {
                        let nwx = (wx0 + dwx).clamp(0, (work_w - 1) as isize) as usize;
                        let wx_dist = (fx - nwx as f32).abs();
                        let w_spat = wy_wt * (1.6 - wx_dist).max(0.10);
                        let (nn_sx, nn_sy) = nnf_work[nwy * work_w + nwx];
                        let dx_full = ((nn_sx - nwx as i32) as f32 * sx_scale).round() as isize;
                        let dy_full = ((nn_sy - nwy as i32) as f32 * sy_scale).round() as isize;
                        let sx_full = (x as isize + dx_full).clamp(0, (w - 1) as isize) as usize;
                        let sy_full = (y as isize + dy_full).clamp(0, (h - 1) as isize) as usize;
                        if hole_dil[sy_full * w + sx_full] {
                            continue;
                        }
                        let s_idx = (sy_full * w + sx_full) * 3;
                        let dr = rgb_upsampled[s_idx] - base_r;
                        let dg = rgb_upsampled[s_idx + 1] - base_g;
                        let db = rgb_upsampled[s_idx + 2] - base_b;
                        let d_col2 = dr * dr + dg * dg * 1.4 + db * db;
                        let w_col = (-d_col2 * 140.0).exp();
                        if w_col > max_w_col {
                            max_w_col = w_col;
                        }
                        let wt = w_spat * w_col;
                        if wt > 1e-5 {
                            for c in 0..3 {
                                let hf = (rgb[s_idx + c] - rgb_upsampled[s_idx + c]).clamp(-0.18, 0.18);
                                hf_acc[c] += hf * wt;
                            }
                            w_sum += wt;
                        }
                    }
                }
            }
            let conf = if w_sum > 1e-5 {
                let inv_w = 1.0 / w_sum;
                hf_acc[0] *= inv_w;
                hf_acc[1] *= inv_w;
                hf_acc[2] *= inv_w;
                (max_w_col * 0.90 * w_tex).clamp(0.0, 0.90)
            } else {
                0.0
            };
            let blend_a = if border_hole[i] {
                1.0
            } else {
                let t = (1.0 - (dist_in[i] as f32 / (feather_f + 1.0))).clamp(0.0, 1.0);
                t * t * (3.0 - 2.0 * t)
            };
            let gx_in = if x < cx_lo {
                (2 * cx_lo - x).clamp(cx_lo, cx_hi)
            } else if x > cx_hi {
                (2 * cx_hi).saturating_sub(x).clamp(cx_lo, cx_hi)
            } else {
                x
            };
            let gy_in = if y < cy_lo {
                (2 * cy_lo - y).clamp(cy_lo, cy_hi)
            } else if y > cy_hi {
                (2 * cy_hi).saturating_sub(y).clamp(cy_lo, cy_hi)
            } else {
                y
            };
            let g_idx = (gy_in * w + gx_in) * 3;
            for c in 0..3 {
                let sky_grain = (rgb[g_idx + c] - rgb_upsampled[g_idx + c]).clamp(-0.025, 0.025);
                let hf_total = conf * hf_acc[c] + (1.0 - w_tex) * 0.92 * sky_grain;
                let synth = (pm_full[idx + c] + hf_total).max(0.0);
                out[idx + c] = rgb[idx + c] * (1.0 - blend_a) + synth * blend_a;
            }
        }
    }

    let auto_grain = estimate_context_grain(rgb, border_hole, w, h);
    let iso_grain = if cfg.iso > 100 {
        ((cfg.iso as f32 / 100.0).sqrt() - 1.0).clamp(0.0, 10.0) * 0.0018
    } else {
        0.0
    };
    let grain_amp = auto_grain.max(iso_grain);
    if grain_amp > 1e-4 {
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if border_hole[i] {
                    let luma_n = sensor_grain_sample(x, y, 0, cfg.seed) * grain_amp;
                    let chr_r = sensor_grain_sample(x, y, 1, cfg.seed) * (grain_amp * 0.35);
                    let chr_b = sensor_grain_sample(x, y, 2, cfg.seed) * (grain_amp * 0.35);
                    let idx = i * 3;
                    let lum_scale = out[idx + 1].abs().sqrt().clamp(0.15, 1.0);
                    out[idx] = (out[idx] + (luma_n + chr_r) * lum_scale).max(0.0);
                    out[idx + 1] = (out[idx + 1] + luma_n * lum_scale).max(0.0);
                    out[idx + 2] = (out[idx + 2] + (luma_n + chr_b) * lum_scale).max(0.0);
                }
            }
        }
    }
    out
}

/// Generative border expansion (Outpainting) at full 1:1 native resolution (`w × h`):
/// 1. Builds an exact boundary-continuous (`C^0`/`C^1`) reflection seed across `[vx0..vx1, vy0..vy1]`
///    with zero step-offset at the border seam.
/// 2. Runs `LaMa ONNX` (`run_lama_onnx_512`) to extract the macro low-frequency illumination guide
///    and blends it smoothly away from the inner photo boundary so the sky/ground has zero color step.
/// 3. Synthesizes 1:1 native-resolution (`w × h`) textures (grass, wildflowers, ocean waves, foliage)
///    across the boundary and throughout the expanded borders via `quilt_native_detail_in_hole`
///    with horizon-alignment locking on left/right borders.
pub fn outpaint_lama_borders(
    rgb: &[f32],
    border_hole: &[bool],
    w: usize,
    h: usize,
    cfg: &LamaConfig,
) -> Vec<f32> {
    if w < 8 || h < 8 || rgb.len() < w.saturating_mul(h).saturating_mul(3) || border_hole.len() < w.saturating_mul(h) {
        return rgb.to_vec();
    }
    if !border_hole.iter().any(|&b| b) || border_hole.iter().all(|&b| b) {
        return rgb.to_vec();
    }

    // 1. Find exact bounding box `[vx0..=vx1, vy0..=vy1]` of valid interior pixels (`!border_hole`)
    let mut vx0 = w;
    let mut vy0 = h;
    let mut vx1 = 0usize;
    let mut vy1 = 0usize;
    for y in 0..h {
        for x in 0..w {
            if !border_hole[y * w + x] {
                vx0 = vx0.min(x);
                vy0 = vy0.min(y);
                vx1 = vx1.max(x);
                vy1 = vy1.max(y);
            }
        }
    }
    if vx0 >= vx1 || vy0 >= vy1 {
        return rgb.to_vec();
    }

    // 1. Dilate `border_hole` inward by `feather_r = 6` pixels (`hole_dil`) so any edge pixels of the
    //    original frame are seamlessly blended across a 6-pixel smoothstep ramp, and build a normalized
    //    unmasked convolution (`rgb_smooth8`) + harmonic Laplace membrane (`seeded = membrane_fill(...)`)
    //    so corners have ZERO mirror bullseye and ZERO L-shaped clamping crease!
    let max_side_f = cfg.max_side.clamp(384, 768) as f32;
    let feather_r = ((w.max(h) as f32 / max_side_f) * 2.0).round().clamp(4.0, 16.0) as isize;
    let mut hole_dil = border_hole.to_vec();
    let mut dist_in = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if border_hole[i] {
                continue;
            }
            let mut min_d2 = (feather_r * feather_r + 1) as isize;
            for dy in -feather_r..=feather_r {
                let ny = (y as isize + dy).clamp(0, (h - 1) as isize) as usize;
                for dx in -feather_r..=feather_r {
                    let d2 = dx * dx + dy * dy;
                    if d2 < min_d2 {
                        let nx = (x as isize + dx).clamp(0, (w - 1) as isize) as usize;
                        if border_hole[ny * w + nx] {
                            min_d2 = d2;
                        }
                    }
                }
            }
            if min_d2 <= feather_r * feather_r {
                hole_dil[i] = true;
                dist_in[i] = (min_d2 as f32).sqrt().round().clamp(1.0, 255.0) as u8;
            }
        }
    }

    let inset_x = feather_r as usize + 1;
    let inset_y = feather_r as usize + 1;
    let cx_lo = (vx0 + inset_x).min(vx1);
    let cx_hi = vx1.saturating_sub(inset_x).max(cx_lo);
    let cy_lo = (vy0 + inset_y).min(vy1);
    let cy_hi = vy1.saturating_sub(inset_y).max(cy_lo);

    let _outpaint_serial_guard = INPAINT_LOCK.get_or_init(|| Mutex::new(())).lock().ok();
    let grid_sig = compute_unmasked_grid_sig(rgb, &hole_dil, w, h);

    if cfg.seed != 0 {
        let cached_hit = OUTPAINT_WORK_CACHE
            .get_or_init(|| Mutex::new(Vec::new()))
            .lock()
            .ok()
            .and_then(|guard| {
                guard
                    .iter()
                    .find(|c| {
                        c.seed == cfg.seed
                            && grid_sig_matches(&c.grid_sig, &grid_sig)
                            && ((w as f32 / h.max(1) as f32) - (c.work_w as f32 / c.work_h.max(1) as f32)).abs()
                                <= 0.08
                            && (c.work_w.max(c.work_h) >= 160 || w.max(h) <= c.work_w.max(c.work_h))
                    })
                    .cloned()
            });
        if let Some(c) = cached_hit {
            return reconstruct_outpaint_from_work(
                rgb,
                border_hole,
                &hole_dil,
                &dist_in,
                feather_r,
                cx_lo,
                cx_hi,
                cy_lo,
                cy_hi,
                w,
                h,
                c.work_w,
                c.work_h,
                &c.work_rgb,
                &c.pm_adapted_work,
                &c.smooth_tex_std,
                &c.nnf_work,
                cfg,
            );
        }
    }

    let mut num_8 = vec![0.0f32; w * h * 3];
    let mut den_8 = vec![0.0f32; w * h * 3];
    for i in 0..(w * h) {
        if !hole_dil[i] {
            let idx = i * 3;
            num_8[idx] = rgb[idx];
            num_8[idx + 1] = rgb[idx + 1];
            num_8[idx + 2] = rgb[idx + 2];
            den_8[idx] = 1.0;
            den_8[idx + 1] = 1.0;
            den_8[idx + 2] = 1.0;
        }
    }
    let b_num8 = blur_rgb3(&num_8, w, h, 8);
    let b_den8 = blur_rgb3(&den_8, w, h, 8);
    let mut rgb_smooth8 = rgb.to_vec();
    for i in 0..(w * h) {
        let idx = i * 3;
        if b_den8[idx] > 1e-4 {
            rgb_smooth8[idx] = b_num8[idx] / b_den8[idx];
            rgb_smooth8[idx + 1] = b_num8[idx + 1] / b_den8[idx + 1];
            rgb_smooth8[idx + 2] = b_num8[idx + 2] / b_den8[idx + 2];
        }
    }

    let mut seeded = rgb.to_vec();
    for y in 0..h {
        let cy = y.clamp(cy_lo, cy_hi);
        for x in 0..w {
            let i = y * w + x;
            if hole_dil[i] {
                let cx = x.clamp(cx_lo, cx_hi);
                let src_i = (cy * w + cx) * 3;
                seeded[i * 3] = rgb_smooth8[src_i];
                seeded[i * 3 + 1] = rgb_smooth8[src_i + 1];
                seeded[i * 3 + 2] = rgb_smooth8[src_i + 2];
            }
        }
    }

    // 2. Run Horizon-Aware, Subject-Guarded Multi-Scale PatchMatch (`complete_guided_nnf`)
    //    seeded by `seeded` + 1:1 Native 6K NNF Detail Transfer across all 4 borders!
    let max_side = cfg.max_side.clamp(384, 768);
    let long = w.max(h);
    let (work_w, work_h) = if long > max_side {
        let scale = max_side as f32 / long as f32;
        (((w as f32 * scale).round() as usize).max(8), ((h as f32 * scale).round() as usize).max(8))
    } else {
        (w, h)
    };
    let work_rgb = if (work_w, work_h) != (w, h) {
        resize_rgb(rgb, w, h, work_w, work_h)
    } else {
        rgb.to_vec()
    };
    let work_hole = if (work_w, work_h) != (w, h) {
        let raw_wh = resize_mask(border_hole, w, h, work_w, work_h);
        // Dilate `raw_wh` (`border_hole` at work resolution) by 2 work pixels so `work_hole` and `hole_dil`
        // (`feather_r = 2 * scale`) align 1-to-1 across the feather ramp without any dead zone!
        let mut dil_wh = raw_wh.clone();
        for y in 0..work_h {
            for x in 0..work_w {
                if raw_wh[y * work_w + x] {
                    for dy in -2isize..=2 {
                        let ny = (y as isize + dy).clamp(0, (work_h - 1) as isize) as usize;
                        for dx in -2isize..=2 {
                            if dx * dx + dy * dy <= 5 {
                                let nx = (x as isize + dx).clamp(0, (work_w - 1) as isize) as usize;
                                dil_wh[ny * work_w + nx] = true;
                            }
                        }
                    }
                }
            }
        }
        dil_wh
    } else {
        hole_dil.clone()
    };
    let mut work_guide = if (work_w, work_h) != (w, h) {
        resize_rgb(&seeded, w, h, work_w, work_h)
    } else {
        seeded.clone()
    };
    let mut num_w8 = vec![0.0f32; work_w * work_h * 3];
    let mut den_w8 = vec![0.0f32; work_w * work_h * 3];
    for i in 0..(work_w * work_h) {
        if !work_hole[i] {
            let idx = i * 3;
            num_w8[idx] = work_rgb[idx];
            num_w8[idx + 1] = work_rgb[idx + 1];
            num_w8[idx + 2] = work_rgb[idx + 2];
            den_w8[idx] = 1.0;
            den_w8[idx + 1] = 1.0;
            den_w8[idx + 2] = 1.0;
        }
    }
    let b_num_w8 = blur_rgb3(&num_w8, work_w, work_h, 6);
    let b_den_w8 = blur_rgb3(&den_w8, work_w, work_h, 6);
    let b_num_w18 = blur_rgb3(&num_w8, work_w, work_h, 18);
    let b_den_w18 = blur_rgb3(&den_w8, work_w, work_h, 18);
    let b_num_w2 = blur_rgb3(&num_w8, work_w, work_h, 2);
    let b_den_w2 = blur_rgb3(&den_w8, work_w, work_h, 2);
    let mut work_smooth6 = work_guide.clone();
    let mut work_smooth18 = work_guide.clone();
    let mut work_smooth2 = work_guide.clone();
    for i in 0..(work_w * work_h) {
        let idx = i * 3;
        if b_den_w8[idx] > 1e-4 {
            work_smooth6[idx] = b_num_w8[idx] / b_den_w8[idx];
            work_smooth6[idx + 1] = b_num_w8[idx + 1] / b_den_w8[idx + 1];
            work_smooth6[idx + 2] = b_num_w8[idx + 2] / b_den_w8[idx + 2];
        }
        if b_den_w18[idx] > 1e-4 {
            work_smooth18[idx] = b_num_w18[idx] / b_den_w18[idx];
            work_smooth18[idx + 1] = b_num_w18[idx + 1] / b_den_w18[idx + 1];
            work_smooth18[idx + 2] = b_num_w18[idx + 2] / b_den_w18[idx + 2];
        }
        if b_den_w2[idx] > 1e-4 {
            work_smooth2[idx] = b_num_w2[idx] / b_den_w2[idx];
            work_smooth2[idx + 1] = b_num_w2[idx + 1] / b_den_w2[idx + 1];
            work_smooth2[idx + 2] = b_num_w2[idx + 2] / b_den_w2[idx + 2];
        }
    }
    // Build `work_guide` inside `work_hole` using C^1-continuous 2D Taylor slope extrapolation:
    //   S_ext(x, y) = work_smooth6(cx, cy) + sky_t * (g_x(cy) * phi(dx_out) + g_y(cx) * phi(dy_out))
    // where `cx, cy` is the exact interior boundary of `!work_hole` (`0` step error at `d = 0`!) and
    // `g_x, g_y` continue the normal sky derivative across `x = wvx0, x = wvx1, y = wvy0` AND across
    // the 4 corners (`phi'(0) = 1`), completely eliminating both C^0 steps and C^1 Mach-band rectangles!
    let mut wvx0 = work_w;
    let mut wvx1 = 0usize;
    let mut wvy0 = work_h;
    let mut wvy1 = 0usize;
    for y in 0..work_h {
        for x in 0..work_w {
            if !work_hole[y * work_w + x] {
                wvx0 = wvx0.min(x);
                wvx1 = wvx1.max(x);
                wvy0 = wvy0.min(y);
                wvy1 = wvy1.max(y);
            }
        }
    }
    if wvx0 < wvx1 && wvy0 < wvy1 {
        let span = 14usize.min((wvx1 - wvx0) / 3).min((wvy1 - wvy0) / 3).max(2);
        let span_f = span as f32;
        let decay_l = 42.0f32;
        for y in 0..work_h {
            let cy = y.clamp(wvy0, wvy1);
            let dy_out = if y < wvy0 { (wvy0 - y) as f32 } else { 0.0 };
            let phi_y = decay_l * (dy_out / decay_l).tanh();
            let sky_t = (((work_h as f32 * 0.48) - y as f32) / (work_h as f32 * 0.06).max(1.0)).clamp(0.0, 1.0);
            for x in 0..work_w {
                let i = y * work_w + x;
                if work_hole[i] {
                    let cx = x.clamp(wvx0, wvx1);
                    let (dx_out, sx_inner) = if x < wvx0 {
                        ((wvx0 - x) as f32, (wvx0 + span).min(wvx1))
                    } else if x > wvx1 {
                        ((x - wvx1) as f32, wvx1.saturating_sub(span).max(wvx0))
                    } else {
                        (0.0, cx)
                    };
                    let phi_x = decay_l * (dx_out / decay_l).tanh();
                    let sy_inner = (wvy0 + span).min(wvy1);
                    let i_x_bnd = (cy * work_w + if x < wvx0 { wvx0 } else { wvx1 }) * 3;
                    let i_x_in = (cy * work_w + sx_inner) * 3;
                    let i_y_bnd = (wvy0 * work_w + cx) * 3;
                    let i_y_in = (sy_inner * work_w + cx) * 3;
                    for c in 0..3 {
                        let gx = if dx_out > 0.0 {
                            ((work_smooth18[i_x_bnd + c] - work_smooth18[i_x_in + c]) / span_f).clamp(-0.0020, 0.0020)
                        } else {
                            0.0
                        };
                        let gy = if dy_out > 0.0 {
                            ((work_smooth18[i_y_bnd + c] - work_smooth18[i_y_in + c]) / span_f).clamp(-0.0020, 0.0020)
                        } else {
                            0.0
                        };
                        // Compute centroid-debiased boundary value at `(cx, cy)` so one-sided box averaging
                        // inside `[wvx0..wvx1] x [wvy0..wvy1]` has ZERO inward normal-gradient step shift!
                        let mut b_acc = 0.0f32;
                        let mut b_wt = 0.0f32;
                        for py in cy.saturating_sub(3)..=(cy + 3).min(work_h - 1) {
                            let dy_in = if y < wvy0 { (py as f32 - cy as f32).max(0.0) } else { 0.0 };
                            for px in cx.saturating_sub(3)..=(cx + 3).min(work_w - 1) {
                                let pi = py * work_w + px;
                                if !work_hole[pi] {
                                    let dx_in = if x < wvx0 {
                                        (px as f32 - cx as f32).max(0.0)
                                    } else if x > wvx1 {
                                        (cx as f32 - px as f32).max(0.0)
                                    } else {
                                        0.0
                                    };
                                    b_acc += work_rgb[pi * 3 + c] + sky_t * (dx_in * gx + dy_in * gy);
                                    b_wt += 1.0;
                                }
                            }
                        }
                        let base_val = if b_wt > 0.0 {
                            b_acc / b_wt
                        } else {
                            work_smooth6[(cy * work_w + cx) * 3 + c]
                        };
                        work_guide[i * 3 + c] = (base_val + sky_t * (gx * phi_x + gy * phi_y)).max(0.0);
                    }
                } else {
                    let idx = i * 3;
                    work_guide[idx] = work_rgb[idx];
                    work_guide[idx + 1] = work_rgb[idx + 1];
                    work_guide[idx + 2] = work_rgb[idx + 2];
                }
            }
        }
    }
    // Precompute perceptual (sqrt-domain) high-frequency local texture roughness (`smooth_tex_std`) of
    // unmasked pixels at `(work_w, work_h)` (against radius-2 `work_smooth2`) and extrapolate it into
    // the borders via membrane_fill so smooth sky & soft cloud regions use the 100% smooth `work_guide`
    // without rectangular PatchMatch blocks, while palm fronds and dark shadow grass retain 100% sharpness!
    let mut tex_energy = vec![0.0f32; work_w * work_h * 3];
    for i in 0..(work_w * work_h) {
        if !work_hole[i] {
            let idx = i * 3;
            let luma = (0.2627 * work_smooth6[idx]
                + 0.6780 * work_smooth6[idx + 1]
                + 0.0593 * work_smooth6[idx + 2])
                .max(0.015);
            let inv_sqrt_l = 1.0 / luma.sqrt();
            let dr = (work_rgb[idx] - work_smooth2[idx]) * inv_sqrt_l;
            let dg = (work_rgb[idx + 1] - work_smooth2[idx + 1]) * inv_sqrt_l;
            let db = (work_rgb[idx + 2] - work_smooth2[idx + 2]) * inv_sqrt_l;
            let e = ((dr * dr + dg * dg + db * db) / 3.0).sqrt();
            tex_energy[idx] = e;
            tex_energy[idx + 1] = e;
            tex_energy[idx + 2] = e;
        }
    }
    let tex_energy_blur = blur_rgb3(&tex_energy, work_w, work_h, 5);
    let mut tex_boundary = vec![0.0f32; work_w * work_h * 3];
    for i in 0..(work_w * work_h) {
        if !work_hole[i] {
            let idx = i * 3;
            let w_d = b_den_w8[idx].max(1e-4);
            let e = tex_energy_blur[idx] / w_d;
            tex_boundary[idx] = e;
            tex_boundary[idx + 1] = e;
            tex_boundary[idx + 2] = e;
        }
    }
    let smooth_tex_std = membrane_fill(work_w, work_h, 3, &tex_boundary, &work_hole);

    let params = CompleteParams {
        patch_radius: 4,
        em_iters: 6,
        min_em_iters: 2,
        pm_iters: 3,
        refine_extent: 160,
        seed: cfg.seed,
    };

    let mut out = if let Some((pm_work, nnf_work)) =
        complete_guided_nnf(work_w, work_h, 3, &work_rgb, &work_hole, Some(&work_guide), &params)
    {
        // Harmonically lock the boundary seam of `pm_work` to `work_rgb` via `membrane_fill`, and
        // apply `smooth_pm_diff` ONLY to the `pm_work` component (`pm_seamless`), so smooth sky regions
        // (`w_tex = 0`) use the exact C^1-extrapolated `work_guide` without any box-blur centroid step!
        let mut num_pm_in = vec![0.0f32; work_w * work_h * 3];
        let mut den_pm_in = vec![0.0f32; work_w * work_h * 3];
        for i in 0..(work_w * work_h) {
            if work_hole[i] {
                let idx = i * 3;
                num_pm_in[idx] = pm_work[idx];
                num_pm_in[idx + 1] = pm_work[idx + 1];
                num_pm_in[idx + 2] = pm_work[idx + 2];
                den_pm_in[idx] = 1.0;
                den_pm_in[idx + 1] = 1.0;
                den_pm_in[idx + 2] = 1.0;
            }
        }
        let b_pm_in = blur_rgb3(&num_pm_in, work_w, work_h, 12);
        let b_den_pm_in = blur_rgb3(&den_pm_in, work_w, work_h, 12);
        let b_num_w12 = blur_rgb3(&num_w8, work_w, work_h, 12);
        let b_den_w12 = blur_rgb3(&den_w8, work_w, work_h, 12);
        let mut pm_diff = vec![0.0f32; work_w * work_h * 3];
        for i in 0..(work_w * work_h) {
            if !work_hole[i] {
                let idx = i * 3;
                if b_den_pm_in[idx] > 1e-4 && b_den_w12[idx] > 1e-4 {
                    let in_r = b_pm_in[idx] / b_den_pm_in[idx];
                    let in_g = b_pm_in[idx + 1] / b_den_pm_in[idx + 1];
                    let in_b = b_pm_in[idx + 2] / b_den_pm_in[idx + 2];
                    let out_r = b_num_w12[idx] / b_den_w12[idx];
                    let out_g = b_num_w12[idx + 1] / b_den_w12[idx + 1];
                    let out_b = b_num_w12[idx + 2] / b_den_w12[idx + 2];
                    pm_diff[idx] = (out_r - in_r).clamp(-0.24, 0.24);
                    pm_diff[idx + 1] = (out_g - in_g).clamp(-0.24, 0.24);
                    pm_diff[idx + 2] = (out_b - in_b).clamp(-0.24, 0.24);
                }
            }
        }
        let smooth_pm_diff = membrane_fill(work_w, work_h, 3, &pm_diff, &work_hole);
        let mut pm_adapted_work = pm_work.clone();
        for y in 0..work_h {
            let sky_band_t = (((work_h as f32 * 0.50) - y as f32) / (work_h as f32 * 0.06).max(1.0)).clamp(0.0, 1.0);
            for x in 0..work_w {
                let i = y * work_w + x;
                let idx = i * 3;
                if work_hole[i] {
                    let loc_std = smooth_tex_std[idx];
                    let raw_w_tex = ((loc_std - 0.008) / 0.018).clamp(0.0, 1.0);
                    let w_tex = 1.0 - sky_band_t * (1.0 - raw_w_tex);
                    for c in 0..3 {
                        let pm_seamless = (pm_work[idx + c] + smooth_pm_diff[idx + c]).max(0.0);
                        pm_adapted_work[idx + c] =
                            (work_guide[idx + c] * (1.0 - w_tex) + pm_seamless * w_tex).max(0.0);
                    }
                } else {
                    pm_adapted_work[idx] = work_rgb[idx];
                    pm_adapted_work[idx + 1] = work_rgb[idx + 1];
                    pm_adapted_work[idx + 2] = work_rgb[idx + 2];
                }
            }
        }

        let res = reconstruct_outpaint_from_work(
            rgb,
            border_hole,
            &hole_dil,
            &dist_in,
            feather_r,
            cx_lo,
            cx_hi,
            cy_lo,
            cy_hi,
            w,
            h,
            work_w,
            work_h,
            &work_rgb,
            &pm_adapted_work,
            &smooth_tex_std,
            &nnf_work,
            cfg,
        );
        if cfg.seed != 0
            && work_w.max(work_h) >= 160
            && let Ok(mut guard) = OUTPAINT_WORK_CACHE.get_or_init(|| Mutex::new(Vec::new())).lock()
        {
            if let Some(existing) = guard.iter_mut().find(|c| c.seed == cfg.seed) {
                if work_w.max(work_h) >= existing.work_w.max(existing.work_h)
                    || !grid_sig_matches(&existing.grid_sig, &grid_sig)
                {
                    existing.grid_sig = grid_sig;
                    existing.work_w = work_w;
                    existing.work_h = work_h;
                    existing.work_rgb = work_rgb;
                    existing.pm_adapted_work = pm_adapted_work;
                    existing.smooth_tex_std = smooth_tex_std;
                    existing.nnf_work = nnf_work;
                }
            } else {
                if guard.len() >= 6 {
                    guard.remove(0);
                }
                guard.push(CachedOutpaintWork {
                    seed: cfg.seed,
                    grid_sig,
                    work_w,
                    work_h,
                    work_rgb,
                    pm_adapted_work,
                    smooth_tex_std,
                    nnf_work,
                });
            }
        }
        return res;
    } else {
        let r_detail = (w.min(h) / 30).clamp(6, 20);
        quilt_native_detail_in_hole(rgb, &seeded, border_hole, w, h, r_detail, true, cfg.seed)
    };

    // 4. Inject subtle context-matched ISO sensor grain in the expanded borders
    let auto_grain = estimate_context_grain(rgb, border_hole, w, h);
    let iso_grain = if cfg.iso > 100 {
        ((cfg.iso as f32 / 100.0).sqrt() - 1.0).clamp(0.0, 10.0) * 0.0018
    } else {
        0.0
    };
    let grain_amp = auto_grain.max(iso_grain);
    if grain_amp > 1e-4 {
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if border_hole[i] {
                    let luma_n = sensor_grain_sample(x, y, 0, cfg.seed) * grain_amp;
                    let chr_r = sensor_grain_sample(x, y, 1, cfg.seed) * (grain_amp * 0.35);
                    let chr_b = sensor_grain_sample(x, y, 2, cfg.seed) * (grain_amp * 0.35);
                    let idx = i * 3;
                    let lum_scale = out[idx + 1].abs().sqrt().clamp(0.15, 1.0);
                    out[idx] = (out[idx] + (luma_n + chr_r) * lum_scale).max(0.0);
                    out[idx + 1] = (out[idx + 1] + luma_n * lum_scale).max(0.0);
                    out[idx + 2] = (out[idx + 2] + (luma_n + chr_b) * lum_scale).max(0.0);
                }
            }
        }
    }

    out
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn lama_onnx_session_runs_inference_and_outpaints_borders() {
        let w = 192usize;
        let h = 192usize;
        let mut rgb = vec![0.0f32; w * h * 3];
        let mut hole = vec![false; w * h];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                // Sky + beach horizon gradient
                let fy = y as f32 / h as f32;
                rgb[i * 3] = 0.25 + 0.45 * fy;
                rgb[i * 3 + 1] = 0.48 + 0.30 * fy;
                rgb[i * 3 + 2] = 0.78 - 0.25 * fy;
                if (72..120).contains(&x) && (72..120).contains(&y) {
                    hole[i] = true;
                    rgb[i * 3] = 0.0;
                    rgb[i * 3 + 1] = 0.0;
                    rgb[i * 3 + 2] = 0.0;
                }
            }
        }
        if local_onnx_model_path().is_some() {
            let onnx_res = run_lama_onnx_512(&rgb, &hole, w, h).expect("ONNX LaMa inference should succeed");
            assert_eq!(onnx_res.len(), w * h * 3);
            let center = (96 * w + 96) * 3;
            assert!(onnx_res[center] > 0.15 && onnx_res[center + 2] > 0.25);
        }
        let filled = inpaint_lama_box(&rgb, &hole, w, h, &LamaConfig::default());
        assert_eq!(filled.len(), w * h * 3);
        let center = (96 * w + 96) * 3;
        assert!(filled[center] > 0.2 && filled[center + 1] > 0.3);

        // Also test border outpainting (24px margin on all 4 sides)
        let mut border_hole = vec![false; w * h];
        for y in 0..h {
            for x in 0..w {
                if x < 24 || x >= w - 24 || y < 24 || y >= h - 24 {
                    border_hole[y * w + x] = true;
                }
            }
        }
        let expanded = outpaint_lama_borders(&filled, &border_hole, w, h, &LamaConfig::default());
        assert_eq!(expanded.len(), w * h * 3);
        assert!(expanded[0].is_finite() && expanded[0] > 0.1);
    }
}
