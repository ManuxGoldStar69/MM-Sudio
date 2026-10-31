//! Deep-learning Sky Matting (`skyseg.onnx` U-2-Net) + Semantic Scene Parsing
//! (`segformer_ade_b0.onnx` ADE20K 150-class) + 3-Channel RGB Confidence-Weighted
//! Guided Alpha Matting (`640 × 480`) and Optical Old-Sky Color Decontamination.

use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use lightcraft_color::transfer::linear_to_srgb;
use lightcraft_geom::Point;
use lightcraft_raster::Rgb32f;

use crate::geometry::Frame;

pub const SKY_MATTE_W: usize = 640;
pub const SKY_MATTE_H: usize = 480;

/// Output of the neural sky segmentation & RGB guided alpha matting pipeline for a photo.
#[derive(Clone, Debug)]
pub struct SkyMatteResult {
    /// High-resolution (`640 × 480`) alpha matte in `0.0..=1.0`.
    pub matte: Arc<Vec<f32>>,
    /// Mean sRGB `[0..1; 3]` of the original sky in the photo, used to un-premultiply
    /// old-sky glare/halos from tree leaves, hair strands, and horizon edges.
    pub orig_sky_rgb: [f32; 3],
    /// Automatically detected lowest vertical extent (`0.22..=0.92`) of the sky in the photo
    /// so the 4K replacement sky's horizon aligns naturally with the scene horizon.
    pub detected_horizon_y: f32,
}

fn find_model_file(filename: &str) -> Option<PathBuf> {
    if let Ok(home) = std::env::var("HOME") {
        for dir in [".mm-studio/models", ".lightcraft/models"] {
            let p = PathBuf::from(&home).join(dir).join(filename);
            if p.is_file() {
                return Some(p);
            }
        }
    }
    let ws = PathBuf::from("/Users/mcruz/lightcraft/assets/models").join(filename);
    if ws.is_file() {
        return Some(ws);
    }
    if let Ok(exe) = std::env::current_exe() {
        for anc in exe.ancestors().take(6) {
            for rel in ["assets/models", "Resources/models", "Resources/assets/models"] {
                let c = anc.join(rel).join(filename);
                if c.is_file() {
                    return Some(c);
                }
            }
        }
    }
    None
}

#[cfg(target_os = "macos")]
static SKYSEG_SESSION: OnceLock<Mutex<Option<ort::session::Session>>> = OnceLock::new();

#[cfg(target_os = "macos")]
static SEGFORMER_SESSION: OnceLock<Mutex<Option<ort::session::Session>>> = OnceLock::new();

#[cfg(target_os = "macos")]
fn get_skyseg_session() -> &'static Mutex<Option<ort::session::Session>> {
    SKYSEG_SESSION.get_or_init(|| {
        let Some(path) = find_model_file("skyseg.onnx") else {
            return Mutex::new(None);
        };
        let session = ort::session::Session::builder()
            .and_then(|b| b.with_intra_threads(4))
            .and_then(|b| b.commit_from_file(&path))
            .ok();
        Mutex::new(session)
    })
}

#[cfg(target_os = "macos")]
fn get_segformer_session() -> &'static Mutex<Option<ort::session::Session>> {
    SEGFORMER_SESSION.get_or_init(|| {
        let Some(path) = find_model_file("segformer_ade_b0.onnx") else {
            return Mutex::new(None);
        };
        let session = ort::session::Session::builder()
            .and_then(|b| b.with_intra_threads(4))
            .and_then(|b| b.commit_from_file(&path))
            .ok();
        Mutex::new(session)
    })
}

/// Resizes an interleaved sRGB `[0..1]` buffer (`sw × sh × 3`) to `dw × dh × 3` with area averaging
/// when downsampling, eliminating aliasing on fine foliage and hair.
fn resize_srgb_area(src: &[[f32; 3]], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<[f32; 3]> {
    let mut out = vec![[0.0f32; 3]; dw * dh];
    let sx = sw as f32 / dw as f32;
    let sy = sh as f32 / dh as f32;
    for dy in 0..dh {
        let fy0 = (dy as f32 * sy).floor() as usize;
        let fy1 = (((dy + 1) as f32 * sy).ceil() as usize).min(sh).max(fy0 + 1);
        for dx in 0..dw {
            let fx0 = (dx as f32 * sx).floor() as usize;
            let fx1 = (((dx + 1) as f32 * sx).ceil() as usize).min(sw).max(fx0 + 1);
            let mut acc = [0.0f32; 3];
            let mut cnt = 0.0f32;
            for yy in fy0..fy1 {
                for xx in fx0..fx1 {
                    let c = src[yy * sw + xx];
                    acc[0] += c[0];
                    acc[1] += c[1];
                    acc[2] += c[2];
                    cnt += 1.0;
                }
            }
            let inv = 1.0 / cnt.max(1.0);
            out[dy * dw + dx] = [acc[0] * inv, acc[1] * inv, acc[2] * inv];
        }
    }
    out
}

/// Runs `skyseg.onnx` (U-2-Net sky matting model at `320 × 320`) on `grid_rgb` (`sw × sh`).
/// Returns a `320 × 320` sky probability plane in `0.0..=1.0`.
#[cfg(target_os = "macos")]
fn run_skyseg_320(grid_rgb: &[[f32; 3]], sw: usize, sh: usize) -> Option<Vec<f32>> {
    const SIDE: usize = 320;
    let mtx = get_skyseg_session();
    let mut guard = mtx.lock().ok()?;
    let session = guard.as_mut()?;

    let rgb_320 = resize_srgb_area(grid_rgb, sw, sh, SIDE, SIDE);
    let n = SIDE * SIDE;
    let mut nchw = vec![0.0f32; 3 * n];
    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];
    for (i, c) in rgb_320.iter().enumerate() {
        nchw[i] = (c[0] - mean[0]) / std[0];
        nchw[n + i] = (c[1] - mean[1]) / std[1];
        nchw[2 * n + i] = (c[2] - mean[2]) / std[2];
    }

    let input_tensor = ort::value::Tensor::from_array(([1usize, 3, SIDE, SIDE], nchw.into_boxed_slice())).ok()?;
    let outputs = session.run(ort::inputs![input_tensor]).ok()?;
    let (_, slice) = outputs[0].try_extract_tensor::<f32>().ok()?;
    if slice.len() < n {
        return None;
    }

    let mut mn = f32::MAX;
    let mut mx = f32::MIN;
    for &v in &slice[..n] {
        if v.is_finite() {
            if v < mn {
                mn = v;
            }
            if v > mx {
                mx = v;
            }
        }
    }
    let span = (mx - mn).max(1e-5);
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let v = slice[i];
        out[i] = if v.is_finite() { ((v - mn) / span).clamp(0.0, 1.0) } else { 0.0 };
    }
    Some(out)
}

#[cfg(not(target_os = "macos"))]
fn run_skyseg_320(_grid_rgb: &[[f32; 3]], _sw: usize, _sh: usize) -> Option<Vec<f32>> {
    None
}

/// Semantic probabilities extracted from `segformer_ade_b0.onnx` (`128 × 128` grid):
/// `(sky_prob, non_sky_foreground_barrier_prob, person_prob)` at `128 × 128`.
pub struct SegformerMaps {
    pub w: usize,
    pub h: usize,
    pub sky_prob: Vec<f32>,
    pub person_prob: Vec<f32>,
    pub water_or_ground_prob: Vec<f32>,
    pub foliage_prob: Vec<f32>,
}

#[cfg(target_os = "macos")]
fn run_segformer_512(grid_rgb: &[[f32; 3]], sw: usize, sh: usize) -> Option<SegformerMaps> {
    const SIDE: usize = 512;
    const OUT_SIDE: usize = 128;
    const NUM_CLASSES: usize = 150;

    let mtx = get_segformer_session();
    let mut guard = mtx.lock().ok()?;
    let session = guard.as_mut()?;

    let rgb_512 = resize_srgb_area(grid_rgb, sw, sh, SIDE, SIDE);
    let n = SIDE * SIDE;
    let mut nchw = vec![0.0f32; 3 * n];
    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];
    for (i, c) in rgb_512.iter().enumerate() {
        nchw[i] = (c[0] - mean[0]) / std[0];
        nchw[n + i] = (c[1] - mean[1]) / std[1];
        nchw[2 * n + i] = (c[2] - mean[2]) / std[2];
    }

    let input_tensor = ort::value::Tensor::from_array(([1usize, 3, SIDE, SIDE], nchw.into_boxed_slice())).ok()?;
    let outputs = session.run(ort::inputs![input_tensor]).ok()?;
    let (_, logits) = outputs[0].try_extract_tensor::<f32>().ok()?;
    let spatial = OUT_SIDE * OUT_SIDE;
    if logits.len() < NUM_CLASSES * spatial {
        return None;
    }

    // ADE20K class indices:
    // 2 = sky
    // 12 = person
    // 4 = tree, 9 = grass, 17 = plant, 66 = flower, 72 = palm
    // 21 = water, 26 = sea, 60 = river, 128 = lake, 13 = earth, 16 = mountain, 29 = field, 34 = rock, 46 = sand, 68 = hill
    // 0 = wall, 1 = building, 25 = house
    let mut sky_prob = vec![0.0f32; spatial];
    let mut person_prob = vec![0.0f32; spatial];
    let mut water_or_ground_prob = vec![0.0f32; spatial];
    let mut foliage_prob = vec![0.0f32; spatial];

    for idx in 0..spatial {
        let mut max_l = f32::MIN;
        for c in 0..NUM_CLASSES {
            let v = logits[c * spatial + idx];
            if v.is_finite() && v > max_l {
                max_l = v;
            }
        }
        let mut sum_exp = 0.0f32;
        for c in 0..NUM_CLASSES {
            let v = logits[c * spatial + idx];
            if v.is_finite() {
                sum_exp += (v - max_l).exp();
            }
        }
        let inv_sum = 1.0 / sum_exp.max(1e-6);
        let prob_of = |cls: usize| -> f32 {
            let v = logits[cls * spatial + idx];
            if v.is_finite() { ((v - max_l).exp() * inv_sum).clamp(0.0, 1.0) } else { 0.0 }
        };

        sky_prob[idx] = prob_of(2);
        person_prob[idx] = prob_of(12);
        foliage_prob[idx] = (prob_of(4) + prob_of(9) + prob_of(17) + prob_of(66) + prob_of(72)).clamp(0.0, 1.0);
        water_or_ground_prob[idx] = (prob_of(21)
            + prob_of(26)
            + prob_of(60)
            + prob_of(128)
            + prob_of(13)
            + prob_of(16)
            + prob_of(29)
            + prob_of(34)
            + prob_of(46)
            + prob_of(68)
            + prob_of(0)
            + prob_of(1)
            + prob_of(25))
        .clamp(0.0, 1.0);
    }

    Some(SegformerMaps {
        w: OUT_SIDE,
        h: OUT_SIDE,
        sky_prob,
        person_prob,
        water_or_ground_prob,
        foliage_prob,
    })
}

#[cfg(not(target_os = "macos"))]
fn run_segformer_512(_grid_rgb: &[[f32; 3]], _sw: usize, _sh: usize) -> Option<SegformerMaps> {
    None
}

#[inline]
fn sample_bilinear(plane: &[f32], w: usize, h: usize, nx: f32, ny: f32) -> f32 {
    let fx = (nx.clamp(0.0, 1.0) * (w as f32 - 1.0)).clamp(0.0, w as f32 - 1.0001);
    let fy = (ny.clamp(0.0, 1.0) * (h as f32 - 1.0)).clamp(0.0, h as f32 - 1.0001);
    let x0 = fx as usize;
    let y0 = fy as usize;
    let tx = fx - x0 as f32;
    let ty = fy - y0 as f32;
    let top = plane[y0 * w + x0] * (1.0 - tx) + plane[y0 * w + x0 + 1] * tx;
    let bot = plane[(y0 + 1) * w + x0] * (1.0 - tx) + plane[(y0 + 1) * w + x0 + 1] * tx;
    top * (1.0 - ty) + bot * ty
}

#[inline]
fn sstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0).max(1e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Fast $O(W \cdot H)$ separable box mean filter with clamped borders.
fn box_mean(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    if r == 0 || w == 0 || h == 0 {
        return src.to_vec();
    }
    let mut tmp = vec![0.0f32; w * h];
    let mut out = vec![0.0f32; w * h];

    for y in 0..h {
        let row = &src[y * w..(y + 1) * w];
        let dst = &mut tmp[y * w..(y + 1) * w];
        let mut sum = 0.0f32;
        let mut cnt = 0usize;
        for xx in 0..=r.min(w - 1) {
            sum += row[xx];
            cnt += 1;
        }
        for x in 0..w {
            dst[x] = sum / cnt as f32;
            if x + r + 1 < w {
                sum += row[x + r + 1];
                cnt += 1;
            }
            if x >= r {
                sum -= row[x - r];
                cnt -= 1;
            }
        }
    }

    for x in 0..w {
        let mut sum = 0.0f32;
        let mut cnt = 0usize;
        for yy in 0..=r.min(h - 1) {
            sum += tmp[yy * w + x];
            cnt += 1;
        }
        for y in 0..h {
            out[y * w + x] = sum / cnt as f32;
            if y + r + 1 < h {
                sum += tmp[(y + r + 1) * w + x];
                cnt += 1;
            }
            if y >= r {
                sum -= tmp[(y - r) * w + x];
                cnt -= 1;
            }
        }
    }
    out
}

/// Solves 3-Channel RGB Confidence-Weighted Guided Alpha Matting (`He et al. / Closed-Form Matting`).
/// Fits a local affine color model $\alpha = a_R R + a_G G + a_B B + b$ in each $(2r+1)\times(2r+1)$ window,
/// anchored on high-confidence sky and foreground regions so blurry neural edges snap to exact hair,
/// leaf, dress, and horizon contours.
pub fn rgb_guided_matte(
    guide_rgb: &[[f32; 3]],
    p: &[f32],
    conf: &[f32],
    w: usize,
    h: usize,
    r: usize,
    eps: f32,
) -> Vec<f32> {
    let n = w * h;
    let omega_raw = box_mean(conf, w, h, r);
    let mut omega = Vec::with_capacity(n);
    for &ow in &omega_raw {
        omega.push(ow.max(1e-5));
    }

    let mul_c = |f: &dyn Fn(usize) -> f32| -> Vec<f32> {
        let mut v = vec![0.0f32; n];
        for i in 0..n {
            v[i] = conf[i] * f(i);
        }
        let m = box_mean(&v, w, h, r);
        let mut res = vec![0.0f32; n];
        for i in 0..n {
            res[i] = m[i] / omega[i];
        }
        res
    };

    let mean_r = mul_c(&|i| guide_rgb[i][0]);
    let mean_g = mul_c(&|i| guide_rgb[i][1]);
    let mean_b = mul_c(&|i| guide_rgb[i][2]);
    let mean_p = mul_c(&|i| p[i]);

    let corr_rr = mul_c(&|i| guide_rgb[i][0] * guide_rgb[i][0]);
    let corr_rg = mul_c(&|i| guide_rgb[i][0] * guide_rgb[i][1]);
    let corr_rb = mul_c(&|i| guide_rgb[i][0] * guide_rgb[i][2]);
    let corr_gg = mul_c(&|i| guide_rgb[i][1] * guide_rgb[i][1]);
    let corr_gb = mul_c(&|i| guide_rgb[i][1] * guide_rgb[i][2]);
    let corr_bb = mul_c(&|i| guide_rgb[i][2] * guide_rgb[i][2]);

    let corr_rp = mul_c(&|i| guide_rgb[i][0] * p[i]);
    let corr_gp = mul_c(&|i| guide_rgb[i][1] * p[i]);
    let corr_bp = mul_c(&|i| guide_rgb[i][2] * p[i]);

    let mut a_r = vec![0.0f32; n];
    let mut a_g = vec![0.0f32; n];
    let mut a_b = vec![0.0f32; n];
    let mut b_0 = vec![0.0f32; n];

    for i in 0..n {
        let s_rr = (corr_rr[i] - mean_r[i] * mean_r[i]).max(0.0) + eps;
        let s_rg = corr_rg[i] - mean_r[i] * mean_g[i];
        let s_rb = corr_rb[i] - mean_r[i] * mean_b[i];
        let s_gg = (corr_gg[i] - mean_g[i] * mean_g[i]).max(0.0) + eps;
        let s_gb = corr_gb[i] - mean_g[i] * mean_b[i];
        let s_bb = (corr_bb[i] - mean_b[i] * mean_b[i]).max(0.0) + eps;

        let cov_r = corr_rp[i] - mean_r[i] * mean_p[i];
        let cov_g = corr_gp[i] - mean_g[i] * mean_p[i];
        let cov_b = corr_bp[i] - mean_b[i] * mean_p[i];

        // Cofactors of 3x3 symmetric covariance matrix
        let c00 = s_gg * s_bb - s_gb * s_gb;
        let c01 = -(s_rg * s_bb - s_gb * s_rb);
        let c02 = s_rg * s_gb - s_gg * s_rb;
        let c11 = s_rr * s_bb - s_rb * s_rb;
        let c12 = -(s_rr * s_gb - s_rg * s_rb);
        let c22 = s_rr * s_gg - s_rg * s_rg;

        let det = s_rr * c00 + s_rg * c01 + s_rb * c02;
        if det.abs() > 1e-12 {
            let inv_det = 1.0 / det;
            let ar = (c00 * cov_r + c01 * cov_g + c02 * cov_b) * inv_det;
            let ag = (c01 * cov_r + c11 * cov_g + c12 * cov_b) * inv_det;
            let ab = (c02 * cov_r + c12 * cov_g + c22 * cov_b) * inv_det;
            a_r[i] = ar;
            a_g[i] = ag;
            a_b[i] = ab;
            b_0[i] = mean_p[i] - ar * mean_r[i] - ag * mean_g[i] - ab * mean_b[i];
        } else {
            b_0[i] = mean_p[i];
        }
    }

    let mean_ar = box_mean(&a_r, w, h, r);
    let mean_ag = box_mean(&a_g, w, h, r);
    let mean_ab = box_mean(&a_b, w, h, r);
    let mean_b0 = box_mean(&b_0, w, h, r);

    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let c = guide_rgb[i];
        let val = mean_ar[i] * c[0] + mean_ag[i] * c[1] + mean_ab[i] * c[2] + mean_b0[i];
        out[i] = if val.is_finite() { val.clamp(0.0, 1.0) } else { 0.0 };
    }
    out
}

static MATTE_CACHE: Mutex<Vec<(u64, SkyMatteResult)>> = Mutex::new(Vec::new());

/// Computes the neural sky alpha matte (`640 × 480`), original sky color (`orig_sky_rgb`), and
/// auto-horizon line (`detected_horizon_y`) for `img`.
pub fn compute_neural_sky_matte(img: &Rgb32f, frame: &Frame, ev: f32) -> SkyMatteResult {
    let (w, h) = (img.width, img.height);
    let n_grid = SKY_MATTE_W * SKY_MATTE_H;
    if w < 8 || h < 8 || img.data.len() < w * h {
        return SkyMatteResult {
            matte: Arc::new(vec![0.0; n_grid]),
            orig_sky_rgb: [0.85, 0.90, 0.96],
            detected_horizon_y: 0.55,
        };
    }

    // Build O(1) cache key from image dimensions, exposure, crop transform, and 24 sampled pixels
    let mut cache_key = (w as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (h as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    cache_key ^= (ev.to_bits() as u64).wrapping_mul(0x94D0_49BB_1331_11EB);
    for k in 0..24 {
        let idx = (k * (img.data.len().saturating_sub(1))) / 23;
        let c = img.data[idx];
        let bits = (c[0].to_bits() as u64) ^ ((c[1].to_bits() as u64).rotate_left(21)) ^ ((c[2].to_bits() as u64).rotate_left(42));
        cache_key = cache_key.wrapping_mul(0x517C_C1B7_2722_0A95) ^ bits;
    }

    if let Ok(guard) = MATTE_CACHE.lock()
        && let Some((_, cached)) = guard.iter().find(|(k, _)| *k == cache_key)
    {
        return cached.clone();
    }

    let gain = ev.exp2().clamp(0.15, 8.0);
    let to_srgb = lightcraft_color::REC2020.to_space(&lightcraft_color::SRGB);
    let norm_to_out = frame.norm_to_out(w, h);

    // 1. Render oriented display sRGB onto the `640 × 480` matte grid with 2×2 supersampling
    let mut grid_rgb = vec![[0.0f32; 3]; n_grid];
    for gy in 0..SKY_MATTE_H {
        for gx in 0..SKY_MATTE_W {
            let mut sum = [0.0f32; 3];
            for sy in [0.25f64, 0.75] {
                let ny = (gy as f64 + sy) / SKY_MATTE_H as f64;
                for sx in [0.25f64, 0.75] {
                    let nx = (gx as f64 + sx) / SKY_MATTE_W as f64;
                    let p_out = norm_to_out.apply(Point::new(nx, ny));
                    let ix = (p_out.x.floor() as isize).clamp(0, w as isize - 1) as usize;
                    let iy = (p_out.y.floor() as isize).clamp(0, h as isize - 1) as usize;
                    let raw = img.data[iy * w + ix];
                    let lin = to_srgb.apply_f32([raw[0] * gain, raw[1] * gain, raw[2] * gain]);
                    sum[0] += linear_to_srgb(lin[0].clamp(0.0, 1.0));
                    sum[1] += linear_to_srgb(lin[1].clamp(0.0, 1.0));
                    sum[2] += linear_to_srgb(lin[2].clamp(0.0, 1.0));
                }
            }
            grid_rgb[gy * SKY_MATTE_W + gx] = [sum[0] * 0.25, sum[1] * 0.25, sum[2] * 0.25];
        }
    }

    // 2. Run neural sky matting (`skyseg.onnx` U-2-Net 320×320) + semantic segmentation (`segformer_ade_b0.onnx` 512×512)
    let u2_map = run_skyseg_320(&grid_rgb, SKY_MATTE_W, SKY_MATTE_H);
    let seg_map = run_segformer_512(&grid_rgb, SKY_MATTE_W, SKY_MATTE_H);

    // 3. Fuse U-2-Net fine sky matting + eroded SegFormer interior priors on the 640 × 480 grid
    let mut init_p = vec![0.0f32; n_grid];

    if let Some(ref u2) = u2_map {
        for gy in 0..SKY_MATTE_H {
            let ny = (gy as f32 + 0.5) / SKY_MATTE_H as f32;
            for gx in 0..SKY_MATTE_W {
                let nx = (gx as f32 + 0.5) / SKY_MATTE_W as f32;
                let idx = gy * SKY_MATTE_W + gx;
                let p_u2 = sample_bilinear(u2, 320, 320, nx, ny);

                let p_fused = if let Some(ref sm) = seg_map {
                    let p_sky_sem = sample_bilinear(&sm.sky_prob, sm.w, sm.h, nx, ny);
                    let p_foliage = sample_bilinear(&sm.foliage_prob, sm.w, sm.h, nx, ny);
                    // Sample eroded person interior (min over ±2 cells in 128×128 space, ~10px erosion)
                    // so coarse SegFormer NEVER vetoes the sky outside a person's hair, face, or shoulder!
                    let step_x = 2.0 / sm.w as f32;
                    let step_y = 2.0 / sm.h as f32;
                    let p0 = sample_bilinear(&sm.person_prob, sm.w, sm.h, nx, ny);
                    let p_l = sample_bilinear(&sm.person_prob, sm.w, sm.h, (nx - step_x).max(0.0), ny);
                    let p_r = sample_bilinear(&sm.person_prob, sm.w, sm.h, (nx + step_x).min(1.0), ny);
                    let p_u = sample_bilinear(&sm.person_prob, sm.w, sm.h, nx, (ny - step_y).max(0.0));
                    let p_d = sample_bilinear(&sm.person_prob, sm.w, sm.h, nx, (ny + step_y).min(1.0));
                    let p_person_core = p0.min(p_l).min(p_r).min(p_u).min(p_d);
                    let p_person_dilated = p0.max(p_l).max(p_r).max(p_u).max(p_d);

                    let w0 = sample_bilinear(&sm.water_or_ground_prob, sm.w, sm.h, nx, ny);
                    let w_u = sample_bilinear(&sm.water_or_ground_prob, sm.w, sm.h, nx, (ny - step_y).max(0.0));
                    let p_water_core = w0.min(w_u);

                    // Veto only deep inside person core (white shirt/dress interior) or deep inside water/ground
                    let person_veto = 1.0 - sstep(0.45, 0.78, p_person_core);
                    let water_veto = 1.0 - sstep(0.48, 0.80, p_water_core) * (1.0 - sstep(0.18, 0.50, p_sky_sem));

                    // In open sky away from people (`p_person_dilated < 0.08`), foliage (`p_foliage < 0.12`),
                    // and water (`w0 < 0.15`), allow SegFormer's semantic sky (`p_sky_sem`) to fill any
                    // overexposed backlight sun-flare hole where U-2-Net dips (`P1022313.RW2`)!
                    let open_sky_boost = sstep(0.52, 0.80, p_sky_sem)
                        * (1.0 - sstep(0.04, 0.14, p_person_dilated))
                        * (1.0 - sstep(0.08, 0.22, p_foliage))
                        * (1.0 - sstep(0.08, 0.22, w0));

                    // Trust U-2-Net (`p_u2`) for all fine edges (hair, faces, pine needles, tree canopies)
                    // while filling open-sky sun flare holes via `open_sky_boost`
                    (p_u2.max(open_sky_boost) * person_veto * water_veto).clamp(0.0, 1.0)
                } else {
                    p_u2
                };

                init_p[idx] = p_fused;
            }
        }
    } else {
        // Fallback when running on non-macOS targets without ONNX Runtime
        for gy in 0..SKY_MATTE_H {
            let ny = (gy as f32 + 0.5) / SKY_MATTE_H as f32;
            for gx in 0..SKY_MATTE_W {
                let idx = gy * SKY_MATTE_W + gx;
                let [r, g, b] = grid_rgb[idx];
                let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                let not_green = 1.0 - sstep(0.02, 0.10, g - b.max(r * 0.95));
                let top = 1.0 - sstep(0.35, 0.68, ny);
                let p = (sstep(0.35, 0.65, lum) * not_green * top).clamp(0.0, 1.0);
                init_p[idx] = p;
            }
        }
    }

    // 4. Exterior Sky Connectivity Filter:
    // Ensure isolated bright pockets deep inside foreground (e.g. a white pocket on shorts/skirt
    // completely surrounded by `init_p < 0.20`) are zeroed out before guided matting.
    let mut dummy_conf = vec![1.0f32; n_grid];
    suppress_disconnected_sky_islands(&mut init_p, &mut dummy_conf, SKY_MATTE_W, SKY_MATTE_H);

    // 5. Build an Adaptive Morphological Trimap (`conf`):
    // - Around foliage/trees (`p_foliage > 0.20`), use a 4-pixel erosion radius so the 3-channel RGB Guided Filter
    //   smoothly resolves blurry bokeh green treetops and pine needles.
    // - Around people/faces/shirts (`p_foliage <= 0.20`), use a tight 2-pixel erosion radius so narrow sky gaps
    //   between faces and low-contrast white shirt edges stay razor-sharp!
    let mut conf = vec![0.005f32; n_grid];
    let mut trimap_p = init_p.clone();
    for gy in 0..SKY_MATTE_H {
        let ny = (gy as f32 + 0.5) / SKY_MATTE_H as f32;
        for gx in 0..SKY_MATTE_W {
            let nx = (gx as f32 + 0.5) / SKY_MATTE_W as f32;
            let p_fol = seg_map
                .as_ref()
                .map(|sm| sample_bilinear(&sm.foliage_prob, sm.w, sm.h, nx, ny))
                .unwrap_or(0.0);
            let er_r: isize = if p_fol > 0.22 { 4 } else { 2 };
            let y0 = (gy as isize - er_r).max(0) as usize;
            let y1 = (gy as isize + er_r).min(SKY_MATTE_H as isize - 1) as usize;
            let x0 = (gx as isize - er_r).max(0) as usize;
            let x1 = (gx as isize + er_r).min(SKY_MATTE_W as isize - 1) as usize;
            let mut p_min = 1.0f32;
            let mut p_max = 0.0f32;
            for yy in y0..=y1 {
                let row = yy * SKY_MATTE_W;
                for xx in x0..=x1 {
                    let v = init_p[row + xx];
                    if v < p_min {
                        p_min = v;
                    }
                    if v > p_max {
                        p_max = v;
                    }
                }
            }
            let idx = gy * SKY_MATTE_W + gx;
            if p_min > 0.78 {
                trimap_p[idx] = 1.0;
                conf[idx] = 1.0;
            } else if p_max < 0.10 {
                trimap_p[idx] = 0.0;
                conf[idx] = 1.0;
            } else {
                conf[idx] = if p_fol > 0.22 { 0.004 } else { 0.04 };
            }
        }
    }

    // 6. Two-Stage 3-Channel RGB Confidence-Weighted Guided Alpha Matting:
    // Stage A (radius 5, eps 2.5e-4): Resolves hair locks, blurry green treetops, and head contours from the Trimap.
    let matte_stage1 = rgb_guided_matte(&grid_rgb, &trimap_p, &conf, SKY_MATTE_W, SKY_MATTE_H, 5, 2.5e-4);

    // Stage B (radius 2, eps 8e-5): Fine sub-pixel RGB feathering for individual hair strands and pine needles.
    let mut conf_stage2 = vec![0.04f32; n_grid];
    for i in 0..n_grid {
        let m = matte_stage1[i];
        conf_stage2[i] = if m >= 0.92 || m <= 0.05 { 0.95 } else { 0.08 };
    }
    let mut matte = rgb_guided_matte(&grid_rgb, &matte_stage1, &conf_stage2, SKY_MATTE_W, SKY_MATTE_H, 2, 8e-5);

    // Clean soft floor/ceiling so pure foreground (`< 0.03`) is exact 0.0 and pure sky (`> 0.95`) is exact 1.0,
    // while keeping the entire `0.03..0.95` optical alpha ramp continuous and smooth!
    for m in &mut matte {
        *m = ((*m - 0.03) / (0.95 - 0.03)).clamp(0.0, 1.0);
    }

    // 7. Measure Original Sky Color (`orig_sky_rgb`) weighted toward the lower/boundary sky (where subjects & trees
    // meet the sky and bright horizon glare is strongest) and Auto-Detected Horizon (`detected_horizon_y`).
    let mut sum_r = 0.0f32;
    let mut sum_g = 0.0f32;
    let mut sum_b = 0.0f32;
    let mut sum_w = 0.0f32;
    let mut row_sky_counts = vec![0usize; SKY_MATTE_H];

    for gy in 0..SKY_MATTE_H {
        let ny = (gy as f32 + 0.5) / SKY_MATTE_H as f32;
        // Weight lower sky rows (`ny > 0.20`) higher because subjects, hair, and trees sit near the horizon!
        let horizon_prox_wt = 0.4 + 1.6 * sstep(0.15, 0.55, ny);
        for gx in 0..SKY_MATTE_W {
            let idx = gy * SKY_MATTE_W + gx;
            let m = matte[idx];
            if m > 0.35 {
                row_sky_counts[gy] += 1;
            }
            if m > 0.82 {
                let c = grid_rgb[idx];
                let lum = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
                // Also favor brighter sky regions so blown-out overcast/horizon sky glare is fully decontaminated
                let wt = m * horizon_prox_wt * (0.5 + lum);
                sum_r += c[0] * wt;
                sum_g += c[1] * wt;
                sum_b += c[2] * wt;
                sum_w += wt;
            }
        }
    }

    let orig_sky_rgb = if sum_w > 4.0 {
        [sum_r / sum_w, sum_g / sum_w, sum_b / sum_w]
    } else {
        [0.88, 0.92, 0.96]
    };

    // Find lowest row where at least 2.5% of the row width is sky
    let min_cols = (SKY_MATTE_W / 40).max(4);
    let mut max_sky_row = (SKY_MATTE_H * 52) / 100;
    for (gy, &cnt) in row_sky_counts.iter().enumerate() {
        if cnt >= min_cols {
            max_sky_row = gy;
        }
    }
    let detected_horizon_y = ((max_sky_row as f32 + 1.0) / SKY_MATTE_H as f32).clamp(0.25, 0.92);

    let result = SkyMatteResult {
        matte: Arc::new(matte),
        orig_sky_rgb,
        detected_horizon_y,
    };

    if let Ok(mut guard) = MATTE_CACHE.lock() {
        if guard.len() >= 8 {
            guard.remove(0);
        }
        guard.push((cache_key, result.clone()));
    }

    result
}

/// Flood-fills from the top border and upper 35% of left/right borders through cells with `p > 0.14`
/// (allowing 2-cell jumps across thin tree branches) and zeroes out disconnected lower islands.
fn suppress_disconnected_sky_islands(p: &mut [f32], conf: &mut [f32], w: usize, h: usize) {
    let mut visited = vec![false; w * h];
    let mut queue = std::collections::VecDeque::with_capacity(w * 4);

    for x in 0..w {
        for y in 0..4.min(h) {
            let idx = y * w + x;
            if p[idx] > 0.14 && !visited[idx] {
                visited[idx] = true;
                queue.push_back((x, y));
            }
        }
    }
    let side_rows = (h * 45) / 100;
    for y in 0..side_rows {
        for x in [0, 1, w.saturating_sub(2), w.saturating_sub(1)] {
            let idx = y * w + x;
            if p[idx] > 0.18 && !visited[idx] {
                visited[idx] = true;
                queue.push_back((x, y));
            }
        }
    }

    while let Some((cx, cy)) = queue.pop_front() {
        // Allow step of 1 everywhere, and step of 2..3 in the upper 55% of the frame so sky gaps
        // between tree leaves/branches stay connected!
        let max_step: isize = if cy < (h * 55) / 100 { 3 } else { 1 };
        for dy in -max_step..=max_step {
            for dx in -max_step..=max_step {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = cx as isize + dx;
                let ny = cy as isize + dy;
                if nx >= 0 && nx < w as isize && ny >= 0 && ny < h as isize {
                    let (ux, uy) = (nx as usize, ny as usize);
                    let nidx = uy * w + ux;
                    let thresh = if dx.abs() <= 1 && dy.abs() <= 1 { 0.12 } else { 0.28 };
                    if !visited[nidx] && p[nidx] > thresh {
                        visited[nidx] = true;
                        queue.push_back((ux, uy));
                    }
                }
            }
        }
    }

    for i in 0..(w * h) {
        if !visited[i] {
            p[i] = 0.0;
            conf[i] = 1.0;
        }
    }
}
