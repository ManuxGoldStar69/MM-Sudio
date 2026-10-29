//! Local LaMa (Large Mask Inpainting) ONNX Runtime (`ort`) inference + Frequency-Separated
//! detail synthesis with 512×512 / 1024×1024 context rescaling and ISO-adaptive sensor grain.
//!
//! Loads `lama_fp32.onnx` lazily into a persistent thread-safe `ort::session::Session` on first use
//! and runs Fast Fourier Convolution neural inpainting on `[1, 3, 512, 512]` crops, combined with
//! high-frequency boundary texture continuity and ISO sensor grain injection.

use std::path::PathBuf;
#[cfg(target_os = "macos")]
use std::sync::{Mutex, OnceLock};

use crate::{CompleteParams, complete, membrane_fill};

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

    // Determine scene-linear peak outside the hole so HDR/bright highlights are preserved without clipping
    let mut peak = 1.0f32;
    for (i, &is_hole) in hole.iter().enumerate().take(w * h) {
        if !is_hole {
            let idx = i * 3;
            let m = rgb[idx].max(rgb[idx + 1]).max(rgb[idx + 2]);
            if m.is_finite() && m > peak {
                peak = m;
            }
        }
    }
    let inv_peak = 1.0 / peak;

    // Rescale crop to 512x512
    let rgb_512 = if (w, h) == (SIDE, SIDE) {
        rgb.to_vec()
    } else {
        resize_rgb(rgb, w, h, SIDE, SIDE)
    };
    let mask_512 = if (w, h) == (SIDE, SIDE) {
        hole.to_vec()
    } else {
        resize_mask(hole, w, h, SIDE, SIDE)
    };

    // Dilate the binary mask by 2px at 512x512 so brush boundary anti-aliasing never leaks into LaMa's encoder
    let mut mask_dilated = mask_512.clone();
    for y in 0..SIDE {
        for x in 0..SIDE {
            if mask_512[y * SIDE + x] {
                for dy in -2isize..=2 {
                    for dx in -2isize..=2 {
                        let ny = (y as isize + dy).clamp(0, (SIDE - 1) as isize) as usize;
                        let nx = (x as isize + dx).clamp(0, (SIDE - 1) as isize) as usize;
                        mask_dilated[ny * SIDE + nx] = true;
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

    let out_native = if (w, h) == (SIDE, SIDE) {
        out_512
    } else {
        resize_rgb(&out_512, SIDE, SIDE, w, h)
    };
    Some(out_native)
}

#[cfg(not(target_os = "macos"))]
pub fn run_lama_onnx_512(_rgb: &[f32], _hole: &[bool], _w: usize, _h: usize) -> Option<Vec<f32>> {
    None
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

/// Separable box/Gaussian approximation on a 3-channel interleaved buffer (`w × h × 3`).
fn blur_rgb3(buf: &[f32], w: usize, h: usize, radius: usize) -> Vec<f32> {
    if radius == 0 || w <= 1 || h <= 1 || buf.len() < w.saturating_mul(h).saturating_mul(3) {
        return buf.to_vec();
    }
    let r = radius.min(w.max(h) / 2).max(1);
    let mut tmp = vec![0.0f32; w * h * 3];
    let mut out = vec![0.0f32; w * h * 3];
    // Horizontal pass
    for y in 0..h {
        for x in 0..w {
            let x0 = x.saturating_sub(r);
            let x1 = (x + r).min(w - 1);
            let count = (x1 - x0 + 1) as f32;
            let mut sum = [0.0f32; 3];
            for xx in x0..=x1 {
                let idx = (y * w + xx) * 3;
                sum[0] += buf[idx];
                sum[1] += buf[idx + 1];
                sum[2] += buf[idx + 2];
            }
            let out_idx = (y * w + x) * 3;
            tmp[out_idx] = sum[0] / count;
            tmp[out_idx + 1] = sum[1] / count;
            tmp[out_idx + 2] = sum[2] / count;
        }
    }
    // Vertical pass
    for y in 0..h {
        let y0 = y.saturating_sub(r);
        let y1 = (y + r).min(h - 1);
        let count = (y1 - y0 + 1) as f32;
        for x in 0..w {
            let mut sum = [0.0f32; 3];
            for yy in y0..=y1 {
                let idx = (yy * w + x) * 3;
                sum[0] += tmp[idx];
                sum[1] += tmp[idx + 1];
                sum[2] += tmp[idx + 2];
            }
            let out_idx = (y * w + x) * 3;
            out[out_idx] = sum[0] / count;
            out[out_idx + 1] = sum[1] / count;
            out[out_idx + 2] = sum[2] / count;
        }
    }
    out
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
    for i in 0..(w * h) {
        if !hole[i] {
            let t = (dist_in[i] as f32 / stride as f32).clamp(0.05, 1.0);
            let w_pre = t * t * (3.0 - 2.0 * t);
            acc_detail[i * 3] = src_detail[i * 3] * w_pre;
            acc_detail[i * 3 + 1] = src_detail[i * 3 + 1] * w_pre;
            acc_detail[i * 3 + 2] = src_detail[i * 3 + 2] * w_pre;
            acc_weight[i] = w_pre;
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

    // Precompute 1D raised-cosine window
    let win1d: Vec<f32> = (0..b_sz)
        .map(|u| {
            let t = (u as f32 + 0.5) / (b_sz as f32);
            (0.5 * (1.0 - (2.0 * std::f32::consts::PI * t).cos())).max(1e-3)
        })
        .collect();

    let vw = (vx1.saturating_sub(vx0 + b_sz)).max(1);
    let _vh = (vy1.saturating_sub(vy0 + b_sz)).max(1);

    // 5. Stitch each tile from boundary outward
    for (t_idx, &(_, tx, ty)) in tiles.iter().enumerate() {
        let mut cands: Vec<(usize, usize)> = Vec::with_capacity(72);

        // Pool A: Horizon-locked candidates (same vertical level `sy ≈ ty`, varying `sx`)
        for dy in [0isize, -2, 2, -5, 5, -9, 9] {
            let sy = (ty as isize + dy).clamp(vy0 as isize, (vy1.saturating_sub(b_sz)).max(vy0) as isize) as usize;
            for k in 0..5usize {
                let sx = vx0 + ((k * vw) / 4).min(vw);
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
        for _ in 0..28 {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut h64 = z;
            h64 = (h64 ^ (h64 >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            h64 = (h64 ^ (h64 >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            let idx = (h64 as usize) % valid_sources.len();
            cands.push(valid_sources[idx]);
        }

        let mut best_cand = cands[0];
        let mut best_cost = f32::INFINITY;
        let sample_step = (b_sz / 4).max(2);

        let is_lr_border = is_outpaint && (tx < vx0 || tx + b_sz > vx1) && ty + b_sz / 2 >= vy0 && ty + b_sz / 2 <= vy1;

        for &(sx, sy) in &cands {
            // 1) Low-frequency color & vertical profile match on a 4x4 grid
            let mut c_low = 0.0f32;
            let mut c_guide = 0.0f32;
            let mut n_low = 0.0f32;
            for v in (1..b_sz).step_by(sample_step) {
                let t_row = (ty + v) * w + tx;
                let s_row = (sy + v) * w + sx;
                for u in (1..b_sz).step_by(sample_step) {
                    let ti = (t_row + u) * 3;
                    let si = (s_row + u) * 3;
                    for c in 0..3 {
                        let dl = base_low[ti + c] - base_low[si + c];
                        c_low += dl * dl;
                        if !is_outpaint {
                            let dg = src_detail[ti + c] - src_detail[si + c];
                            c_guide += dg * dg;
                        }
                    }
                    n_low += 1.0;
                }
            }
            c_low /= n_low.max(1.0);
            c_guide /= n_low.max(1.0);

            // 2) Overlap seam coherence with already-placed neighbors / photo boundary
            let mut c_ovl = 0.0f32;
            let mut n_ovl = 0.0f32;
            for v in (0..b_sz).step_by(2) {
                let t_row = (ty + v) * w + tx;
                let s_row = (sy + v) * w + sx;
                for u in (0..b_sz).step_by(2) {
                    let wi = acc_weight[t_row + u];
                    if wi > 0.15 {
                        let inv_w = 1.0 / wi;
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

            // 3) Spatial & horizon prior
            let dy_norm = (sy as f32 - ty as f32).abs() / (h as f32);
            let dx_norm = (sx as f32 - tx as f32).abs() / (w as f32);
            let c_spat = if is_lr_border {
                // Strong horizon lock on left/right outpaint borders so sea/horizon/sky align
                dy_norm * 0.08 + dx_norm * 0.002
            } else if is_outpaint {
                dy_norm * 0.005 + dx_norm * 0.005
            } else {
                (dx_norm + dy_norm) * 0.004
            };

            let total = c_low * 4.0 + c_ovl * 2.2 + c_guide * 0.85 + c_spat;
            if total < best_cost {
                best_cost = total;
                best_cand = (sx, sy);
            }
        }

        // Accumulate best native detail patch with 2D raised-cosine window
        let (bx, by) = best_cand;
        for v in 0..b_sz {
            let wv = win1d[v];
            let t_row = (ty + v) * w + tx;
            let s_row = (by + v) * w + bx;
            for u in 0..b_sz {
                let w2d = wv * win1d[u];
                let ti = t_row + u;
                let si = s_row + u;
                acc_detail[ti * 3] += src_detail[si * 3] * w2d;
                acc_detail[ti * 3 + 1] += src_detail[si * 3 + 1] * w2d;
                acc_detail[ti * 3 + 2] += src_detail[si * 3 + 2] * w2d;
                acc_weight[ti] += w2d;
            }
        }
    }

    // 6. Recombine smooth illumination `base_low` + 1:1 native quilted detail
    let mut out = rgb.to_vec();
    for i in 0..(w * h) {
        if !hole[i] && dist_in[i] >= stride as u8 {
            continue;
        }
        let wi = acc_weight[i];
        let idx = i * 3;
        if wi > 1e-5 {
            let inv_w = 1.0 / wi;
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

/// Runs LaMa-style local inpainting on a cropped context buffer `rgb` (`w × h × 3`) with binary mask
/// `hole` (`w × h`). Combines LaMa ONNX low-frequency illumination guidance with multi-scale
/// PatchMatch structure continuation and 1:1 native-resolution (`w × h`) patch quilting so textures
/// (grass, foliage, waves, skin, fabric) remain razor-sharp at native camera resolution.
pub fn inpaint_lama_box(rgb: &[f32], hole: &[bool], w: usize, h: usize, cfg: &LamaConfig) -> Vec<f32> {
    if w == 0 || h == 0 || rgb.len() < w.saturating_mul(h).saturating_mul(3) || hole.len() < w.saturating_mul(h) {
        return rgb.to_vec();
    }
    if !hole.iter().any(|&b| b) {
        return rgb.to_vec();
    }

    let max_side = cfg.max_side.clamp(128, 1024);
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
        resize_mask(hole, w, h, work_w, work_h)
    } else {
        hole.to_vec()
    };

    // 1. Run real LaMa ONNX neural network (`lama_fp32.onnx` via `ort`) when crop is >= 128px
    let use_neural = w.max(h) >= 128;
    let neural_out = if use_neural {
        run_lama_onnx_512(&work_rgb, &work_hole, work_w, work_h)
    } else {
        None
    };

    let params = if work_w.max(work_h) <= 512 {
        CompleteParams {
            patch_radius: 4,
            em_iters: if neural_out.is_some() { 6 } else { 10 },
            min_em_iters: 2,
            pm_iters: 3,
            refine_extent: 128,
            seed: 1,
        }
    } else {
        CompleteParams {
            patch_radius: 5,
            em_iters: if neural_out.is_some() { 7 } else { 12 },
            min_em_iters: 3,
            pm_iters: 3,
            refine_extent: 160,
            seed: 1,
        }
    };

    // 2. Multi-scale PatchMatch structural & texture synthesis
    let pm_out = complete(work_w, work_h, 3, &work_rgb, &work_hole, &params)
        .unwrap_or_else(|| membrane_fill(work_w, work_h, 3, &work_rgb, &work_hole));

    let r_low = (work_w.min(work_h) / 10).clamp(4, 26);
    let pm_low = blur_rgb3(&pm_out, work_w, work_h, r_low);

    let work_done = if let Some(mut lama_net) = neural_out {
        // Match boundary ring linear mean per channel so LaMa's sRGB->linear prediction never has a brightness step
        let mut ring_net = [0.0f64; 3];
        let mut ring_src = [0.0f64; 3];
        let mut ring_cnt = 0.0f64;
        for y in 1..(work_h.saturating_sub(1)) {
            for x in 1..(work_w.saturating_sub(1)) {
                let i = y * work_w + x;
                if !work_hole[i]
                    && (work_hole[i - 1] || work_hole[i + 1] || work_hole[i - work_w] || work_hole[i + work_w])
                {
                    for c in 0..3 {
                        ring_src[c] += work_rgb[i * 3 + c] as f64;
                        ring_net[c] += lama_net[i * 3 + c] as f64;
                    }
                    ring_cnt += 1.0;
                }
            }
        }
        let gain = if ring_cnt >= 4.0 {
            [
                ((ring_src[0] / ring_net[0].max(1e-5)).clamp(0.75, 1.35)) as f32,
                ((ring_src[1] / ring_net[1].max(1e-5)).clamp(0.75, 1.35)) as f32,
                ((ring_src[2] / ring_net[2].max(1e-5)).clamp(0.75, 1.35)) as f32,
            ]
        } else {
            [1.0, 1.0, 1.0]
        };
        for i in 0..(work_w * work_h) {
            if work_hole[i] {
                lama_net[i * 3] *= gain[0];
                lama_net[i * 3 + 1] *= gain[1];
                lama_net[i * 3 + 2] *= gain[2];
            } else {
                lama_net[i * 3] = work_rgb[i * 3];
                lama_net[i * 3 + 1] = work_rgb[i * 3 + 1];
                lama_net[i * 3 + 2] = work_rgb[i * 3 + 2];
            }
        }
        // Extract ONLY the smooth low-frequency illumination envelope from LaMa ONNX (`lama_low`),
        // keeping 100% of the crisp medium & high frequencies (`pm_out - pm_low`) from real patches!
        let lama_low = blur_rgb3(&lama_net, work_w, work_h, r_low);
        let mut out = pm_out;
        for i in 0..(work_w * work_h) {
            if work_hole[i] {
                let idx = i * 3;
                for c in 0..3 {
                    let high_and_mid = out[idx + c] - pm_low[idx + c];
                    let low = 0.72 * pm_low[idx + c] + 0.28 * lama_low[idx + c];
                    out[idx + c] = (low + high_and_mid).max(0.0);
                }
            }
        }
        out
    } else {
        let mem_out = membrane_fill(work_w, work_h, 3, &work_rgb, &work_hole);
        let mem_low = blur_rgb3(&mem_out, work_w, work_h, r_low);
        let mut out = pm_out;
        for i in 0..(work_w * work_h) {
            if work_hole[i] {
                let idx = i * 3;
                for c in 0..3 {
                    let high_and_mid = out[idx + c] - pm_low[idx + c];
                    let low = 0.82 * pm_low[idx + c] + 0.18 * mem_low[idx + c];
                    out[idx + c] = (low + high_and_mid).max(0.0);
                }
            }
        }
        out
    };

    // 3. Upsample to native crop dimensions if rescaled
    let merged = if (work_w, work_h) != (w, h) {
        let up = resize_rgb(&work_done, work_w, work_h, w, h);
        let mut m = rgb.to_vec();
        for i in 0..(w * h) {
            if hole[i] {
                m[i * 3] = up[i * 3];
                m[i * 3 + 1] = up[i * 3 + 1];
                m[i * 3 + 2] = up[i * 3 + 2];
            }
        }
        m
    } else {
        work_done
    };

    // 4. Native 1:1 resolution patch quilting on the detail band so erased strokes have
    //    crisp camera-native texture (eliminating any multi-patch vote softening or upscale blur).
    let mut final_rgb = if w.max(h) >= 128 {
        let r_detail = (w.min(h) / 42).clamp(4, 14);
        quilt_native_detail_in_hole(rgb, &merged, hole, w, h, r_detail, false, cfg.seed)
    } else {
        merged
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

    // Reflect across the EXACT boundary `[vx0..vx1, vy0..vy1]` (inset = 0) so there is zero
    // discontinuity at `x = vx0`, `x = vx1`, `y = vy0`, `y = vy1`.
    let span_x = (vx1 - vx0).max(1) as isize;
    let span_y = (vy1 - vy0).max(1) as isize;
    let reflect_exact = |c: isize, lo: isize, span: isize| -> usize {
        let mut r = c - lo;
        if r < 0 {
            r = -r;
        }
        let period = (span * 2).max(2);
        let m = r.rem_euclid(period);
        let pos = if m >= span { period - 1 - m } else { m };
        (lo + pos).clamp(0, (lo + span).max(1)) as usize
    };

    let mut seeded = rgb.to_vec();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if border_hole[i] {
                let mut rx = reflect_exact(x as isize, vx0 as isize, span_x).min(w - 1);
                let mut ry = reflect_exact(y as isize, vy0 as isize, span_y).min(h - 1);
                if border_hole[ry * w + rx] {
                    rx = (vx0 + vx1) / 2;
                    ry = (vy0 + vy1) / 2;
                }
                let src_i = (ry * w + rx) * 3;
                seeded[i * 3] = rgb[src_i];
                seeded[i * 3 + 1] = rgb[src_i + 1];
                seeded[i * 3 + 2] = rgb[src_i + 2];
            }
        }
    }

    // 2. Low-frequency LaMa ONNX guidance (faded in smoothly away from the inner photo boundary
    //    so there is ZERO rectangular color/gamma seam in smooth skies or gradients).
    let anchor_x = ((vx0.max(w - 1 - vx1)) / 4).clamp(2, 16);
    let anchor_y = ((vy0.max(h - 1 - vy1)) / 4).clamp(2, 16);
    let mut bridge_hole = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let near_outer_edge = x < anchor_x || x >= w.saturating_sub(anchor_x) || y < anchor_y || y >= h.saturating_sub(anchor_y);
            if border_hole[i] && !near_outer_edge {
                bridge_hole[i] = true;
            }
        }
    }

    let r_macro = (w.min(h) / 22).clamp(8, 36);
    if w.max(h) >= 128
        && let Some(lama_net) = run_lama_onnx_512(&seeded, &bridge_hole, w, h)
    {
        // Match mean color on the inner boundary ring
        let mut ring_net = [0.0f64; 3];
        let mut ring_src = [0.0f64; 3];
        let mut ring_cnt = 0.0f64;
        for y in vy0.min(h - 1)..=vy1.min(h - 1) {
            for x in [vx0.min(w - 1), vx1.min(w - 1)] {
                let i = y * w + x;
                for c in 0..3 {
                    ring_src[c] += seeded[i * 3 + c] as f64;
                    ring_net[c] += lama_net[i * 3 + c] as f64;
                }
                ring_cnt += 1.0;
            }
        }
        let gain = if ring_cnt >= 4.0 {
            [
                ((ring_src[0] / ring_net[0].max(1e-5)).clamp(0.8, 1.25)) as f32,
                ((ring_src[1] / ring_net[1].max(1e-5)).clamp(0.8, 1.25)) as f32,
                ((ring_src[2] / ring_net[2].max(1e-5)).clamp(0.8, 1.25)) as f32,
            ]
        } else {
            [1.0, 1.0, 1.0]
        };

        let lama_low = blur_rgb3(&lama_net, w, h, r_macro);
        let seed_low = blur_rgb3(&seeded, w, h, r_macro);
        let ramp_span = (w.min(h) as f32 / 18.0).clamp(16.0, 56.0);

        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if border_hole[i] {
                    let dx = if x < vx0 {
                        vx0 - x
                    } else if x > vx1 {
                        x - vx1
                    } else {
                        0
                    };
                    let dy = if y < vy0 {
                        vy0 - y
                    } else if y > vy1 {
                        y - vy1
                    } else {
                        0
                    };
                    let d_out = (dx.max(dy)) as f32;
                    let t = (d_out / ramp_span).clamp(0.0, 1.0);
                    let w_lama = t * t * (3.0 - 2.0 * t) * 0.22;
                    let idx = i * 3;
                    for c in 0..3 {
                        let detail = seeded[idx + c] - seed_low[idx + c];
                        let low = (1.0 - w_lama) * seed_low[idx + c] + w_lama * (lama_low[idx + c] * gain[c]);
                        seeded[idx + c] = (low + detail).max(0.0);
                    }
                }
            }
        }
    }

    // 3. Full 1:1 native-resolution (`w × h`) patch quilting across the boundary & outer borders
    let r_detail = (w.min(h) / 38).clamp(5, 16);
    let mut out = quilt_native_detail_in_hole(rgb, &seeded, border_hole, w, h, r_detail, true, cfg.seed);

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
