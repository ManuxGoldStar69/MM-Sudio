//! Remove / Heal / Clone spots, rendered at output resolution in scene-linear light.
//!
//! - **Clone** copies the source patch with a feathered edge.
//! - **Heal** copies the source's *detail* and keeps the target's *tone*: result = src + blur(target − src)
//!   (frequency separation — a fast, stable approximation of gradient-domain healing).
//! - **Remove** is Heal with an automatically chosen source: candidate offsets on rings around the
//!   spot are scored by how well the source's surrounding annulus matches the target's.
//! - **Fill** (content-aware) rebuilds the spot from patches of its surroundings instead of one
//!   source: PatchMatch-based completion ([`lightcraft_inpaint::complete`]), coarse to fine, so the
//!   structure is chosen at a scale set by the spot's size (the same at preview and export size).
//!   Deterministic; the source offset is ignored.

use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{Mutex, OnceLock};

use lightcraft_develop::{Spot, SpotMode};
use lightcraft_geom::Point;
use lightcraft_inpaint::LamaConfig;
use lightcraft_raster::Rgb32f;

use crate::geometry::Frame;

#[inline]
fn smooth(e0: f32, e1: f32, x: f32) -> f32 {
    // Zero feather is a hard edge, including when the pixel lies exactly on the radius.
    // Dividing 0 by 0 here would give that pixel a NaN alpha, even at zero opacity.
    if e0 >= e1 {
        return if x < e1 { 0.0 } else { 1.0 };
    }
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Most a spot's size counts for, in long edges: a size read from a sidecar isn't held to the slider's range, and one
/// far past the picture (or infinite) made the radius, the blur it heals with and the patch it reads from unbounded
/// (issue #1519). Two long edges cover the whole frame from anywhere in it.
const MAX_SPOT_SIZE: f64 = 2.0;

/// A spot's radius in output pixels, `ppl` output pixels to the long edge: at least a pixel (a size that is not a
/// number too), at most [`MAX_SPOT_SIZE`] long edges.
fn radius_px(size: f64, ppl: f64) -> f64 {
    let size = if size.is_nan() { 0.0 } else { size.min(MAX_SPOT_SIZE) };
    (size * ppl).max(1.0)
}

/// Mean colour of `img` in a disc/annulus.
fn ring_stats(img: &Rgb32f, c: (f32, f32), r0: f32, r1: f32) -> Option<Vec<[f32; 3]>> {
    let mut out = Vec::new();
    let n = 24;
    for i in 0..n {
        let a = i as f32 / n as f32 * std::f32::consts::TAU;
        for rr in [r0, (r0 + r1) / 2.0, r1] {
            let (x, y) = (c.0 + a.cos() * rr, c.1 + a.sin() * rr);
            if x < 0.0 || y < 0.0 || x >= img.width as f32 || y >= img.height as f32 {
                return None;
            }
            out.push(img.sample_bilinear(x, y));
        }
    }
    Some(out)
}

fn score(img: &Rgb32f, target: (f32, f32), source: (f32, f32), r: f32) -> f32 {
    let (Some(a), Some(b)) = (ring_stats(img, target, r * 1.05, r * 1.6), ring_stats(img, source, r * 1.05, r * 1.6)) else { return f32::MAX };
    // Compare in perceptual Oklab (luminance + chrominance) plus interior texture roughness continuity
    let inside = ring_stats(img, source, r * 0.2, r * 0.9);
    let to_lab = |rgb: &[f32; 3]| lightcraft_color::perceptual::oklab_from_2020(rgb.map(|v| v.max(0.0)));
    let mut s = 0.0;
    for (p, q) in a.iter().zip(&b) {
        let lp = to_lab(p);
        let lq = to_lab(q);
        let dl = lp[0] - lq[0];
        let da = lp[1] - lq[1];
        let db = lp[2] - lq[2];
        s += dl * dl + 2.5 * (da * da + db * db);
    }
    if let Some(ins) = inside {
        let mean_l: f32 = a.iter().map(|p| to_lab(p)[0]).sum::<f32>() / a.len() as f32;
        let var_ring: f32 = a.iter().map(|p| (to_lab(p)[0] - mean_l).powi(2)).sum::<f32>() / a.len() as f32;
        let mean_ins: f32 = ins.iter().map(|p| to_lab(p)[0]).sum::<f32>() / ins.len() as f32;
        let var_ins: f32 = ins.iter().map(|p| (to_lab(p)[0] - mean_ins).powi(2)).sum::<f32>() / ins.len() as f32;
        s += ((var_ins - var_ring).abs() + var_ins * 0.35) * a.len() as f32;
    }
    s
}

/// Candidate source offsets (output pixels) for a spot of radius `r` at `target`, best first
/// (candidates reaching outside the image are left out).
pub fn ranked_sources(img: &Rgb32f, target: (f32, f32), r: f32) -> Vec<(f32, f32)> {
    let mut c: Vec<(f32, (f32, f32))> = Vec::new();
    for ring in [1.75f32, 2.35, 3.10, 4.05, 5.15] {
        for i in 0..24 {
            let a = i as f32 / 24.0 * std::f32::consts::TAU;
            let d = (a.cos() * r * ring, a.sin() * r * ring);
            let s = score(img, target, (target.0 + d.0, target.1 + d.1), r);
            if s < f32::MAX {
                c.push((s, d));
            }
        }
    }
    c.sort_by(|a, b| a.0.total_cmp(&b.0));
    c.into_iter().map(|(_, d)| d).collect()
}

/// Pick a source offset (in output pixels) for a remove/heal spot.
pub fn auto_source(img: &Rgb32f, target: (f32, f32), r: f32) -> (f32, f32) {
    ranked_sources(img, target, r).first().copied().unwrap_or((r * 2.5, 0.0))
}

/// A source offset (normalized, as [`Spot::source_offset`]) for `spot` on `src` developed with
/// `s`: the best match, or with `avoid` (the current offset) the best one at least a spot radius
/// away from it ("refresh source"). `None` when no candidate fits in the image.
pub fn pick_source(
    src: &Rgb32f,
    info: &crate::SourceInfo,
    s: &lightcraft_develop::DevelopSettings,
    spot: &Spot,
    avoid: Option<Point>,
) -> Option<Point> {
    let frame = crate::frame_for(src, info, s, true);
    let (w, h) = frame.fit(512, 512);
    if w == 0 || h == 0 {
        return None;
    }
    let img = frame.sample(src, w, h);
    let (to_out, to_norm) = (frame.norm_to_out(w, h), frame.out_to_norm(w, h));
    let t = to_out.apply(*spot.points.first()?);
    let r = radius_px(spot.size, frame.px_per_long(w)) as f32;
    let avoid = avoid.map(|o| {
        let (a, b) = (to_out.apply(Point::new(0.0, 0.0)), to_out.apply(o));
        ((b.x - a.x) as f32, (b.y - a.y) as f32)
    });
    let far = |d: &(f32, f32)| avoid.is_none_or(|a| ((d.0 - a.0).powi(2) + (d.1 - a.1).powi(2)).sqrt() > r);
    let d = ranked_sources(&img, (t.x as f32, t.y as f32), r).into_iter().find(far)?;
    let (n0, n1) = (to_norm.apply(t), to_norm.apply(Point::new(t.x + d.0 as f64, t.y + d.1 as f64)));
    Some(Point::new(n1.x - n0.x, n1.y - n0.y))
}

/// The most a window grows to take in the spots that touch it.
const MAX_SPOT_SPAN: usize = if cfg!(target_arch = "wasm32") { 4096 } else { 8192 };
/// Strokes with more dabs than this are not examined dab by dab (the render would be too slow anyway).
const MAX_DABS_CONSIDERED: usize = 200_000;

/// `win` (a window of the `full_w × full_h` output of `frame`) grown to hold, for every spot (or
/// Auto Mask stroke) that reaches into it, the whole spot and the patch it reads from: [`apply`] works on the pixels it
/// is given, so a spot cut by the window's edge, or whose source lies outside it, would come out
/// different from the same spot in the whole render. Spots with an automatic source (`None`)
/// search up to 4.2 radii around themselves. Growth that would pass [`MAX_SPOT_SPAN`] pixels is not
/// done (the window comes back as asked; see [`window_for_reads_checked`] to know).
pub fn window_for_reads(
    s: &lightcraft_develop::DevelopSettings,
    frame: &Frame,
    full_w: usize,
    full_h: usize,
    ppl: f64,
    win: crate::PixelWindow,
) -> crate::PixelWindow {
    window_for_reads_checked(s, frame, full_w, full_h, ppl, win).unwrap_or(win)
}

/// [`window_for_reads`], or `None` when holding everything the window reads would take more than
/// [`MAX_SPOT_SPAN`] pixels along an axis: a caller that can show something else (the loupe's
/// whole-frame render) must, because the window alone would come out wrong.
pub fn window_for_reads_checked(
    s: &lightcraft_develop::DevelopSettings,
    frame: &Frame,
    full_w: usize,
    full_h: usize,
    ppl: f64,
    win: crate::PixelWindow,
) -> Option<crate::PixelWindow> {
    let spots = &s.spots;
    let autos: Vec<&lightcraft_develop::BrushStroke> = s
        .masks
        .iter()
        .filter(|m| m.visible)
        .flat_map(|m| &m.components)
        .filter_map(|c| match &c.shape {
            lightcraft_develop::MaskShape::Brush { strokes } => Some(strokes),
            _ => None,
        })
        .flatten()
        .filter(|st| st.auto_mask)
        .collect();
    if spots.is_empty() && autos.is_empty() {
        return Some(win);
    }
    let to_out = frame.norm_to_out(full_w, full_h);
    let origin = to_out.apply(Point::new(0.0, 0.0));
    // (target box, source box) of each spot, in output pixels: [x0, y0, x1, y1]
    let boxes: Vec<([f64; 4], [f64; 4])> = spots
        .iter()
        .filter_map(|spot| {
            let r = radius_px(spot.size, ppl);
            let pts: Vec<_> = spot.points.iter().map(|p| to_out.apply(*p)).collect();
            let first = pts.first()?;
            let (mut x0, mut y0, mut x1, mut y1) = (first.x, first.y, first.x, first.y);
            for p in &pts {
                (x0, y0, x1, y1) = (x0.min(p.x), y0.min(p.y), x1.max(p.x), y1.max(p.y));
            }
            // the box `apply` works in: the stroke plus a radius, plus context padding (40% of bbox or 2r for AI Fill)
            let (pad_x, pad_y) = if spot.mode == SpotMode::Fill {
                let sw = (x1 - x0 + 2.0 * r).max(1.0);
                let sh = (y1 - y0 + 2.0 * r).max(1.0);
                ((sw * 0.40).max(2.0 * r) + r, (sh * 0.40).max(2.0 * r) + r)
            } else {
                (2.0 * r, 2.0 * r)
            };
            let t = [x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y];
            let s = match spot.source_offset {
                _ if spot.mode == SpotMode::Fill => t,
                Some(o) => {
                    let b = to_out.apply(o);
                    let (dx, dy) = (b.x - origin.x, b.y - origin.y);
                    [t[0] + dx, t[1] + dy, t[2] + dx, t[3] + dy]
                }
                None => [t[0] - 4.2 * r, t[1] - 4.2 * r, t[2] + 4.2 * r, t[3] + 4.2 * r],
            };
            Some((t, s)).filter(|(t, s)| t.iter().chain(s).all(|v| v.is_finite()))
        })
        .collect();
    // Auto Mask strokes weigh each dab by its similarity to the pixel under the dab's centre and
    // refine the stroke over a couple of radii: each dab matters to pixels within 2.5 radii
    let mut dab_boxes: Vec<[f64; 4]> = Vec::new();
    for st in autos {
        let dabs = crate::masks::brush_dabs(st, frame, full_w, full_h);
        let r = dabs.r;
        if dabs.dabs.len() > MAX_DABS_CONSIDERED {
            continue;
        }
        for d in &dabs.dabs {
            let b = [d.x - 2.5 * r, d.y - 2.5 * r, d.x + 2.5 * r, d.y + 2.5 * r];
            if b.iter().all(|v| v.is_finite()) {
                dab_boxes.push(b);
            }
        }
    }
    let (fw, fh) = (full_w as f64, full_h as f64);
    let mut r = [win.x as f64, win.y as f64, (win.x + win.w) as f64, (win.y + win.h) as f64];
    let touches = |r: &[f64; 4], b: &[f64; 4]| b[0] < r[2] && b[2] > r[0] && b[1] < r[3] && b[3] > r[1];
    // a dab reaching the requested window needs its surroundings, once: the pixels that brings in
    // are only read, not drawn on (unlike spots, whose patches other spots may read)
    let asked = r;
    for b in dab_boxes.iter().filter(|b| touches(&asked, b)) {
        r = [r[0].min(b[0]), r[1].min(b[1]), r[2].max(b[2]), r[3].max(b[3])];
    }
    r = [r[0].max(0.0), r[1].max(0.0), r[2].min(fw), r[3].min(fh)];
    if r[2] - r[0] > MAX_SPOT_SPAN as f64 || r[3] - r[1] > MAX_SPOT_SPAN as f64 {
        return None;
    }
    for _ in 0..=boxes.len() {
        let mut grown = r;
        for (t, s) in &boxes {
            if touches(&r, t) {
                for b in [t, s] {
                    grown = [grown[0].min(b[0]), grown[1].min(b[1]), grown[2].max(b[2]), grown[3].max(b[3])];
                }
            }
        }
        let grown = [grown[0].max(0.0), grown[1].max(0.0), grown[2].min(fw), grown[3].min(fh)];
        if grown == r {
            break;
        }
        if grown[2] - grown[0] > MAX_SPOT_SPAN as f64 || grown[3] - grown[1] > MAX_SPOT_SPAN as f64 {
            return None;
        }
        r = grown;
    }
    let (x, y) = (r[0].floor() as usize, r[1].floor() as usize);
    Some(
        crate::PixelWindow { x, y, w: (r[2].ceil() as usize).saturating_sub(x).max(1), h: (r[3].ceil() as usize).saturating_sub(y).max(1) }
            .clamped(full_w, full_h),
    )
}

static LAMA_PATCH_CACHE: std::sync::LazyLock<std::sync::Mutex<Vec<(u64, Vec<[f32; 3]>)>>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(Vec::new()));

fn lama_cache_key(tgt: &[[f32; 3]], hole: &[bool], bw: usize, bh: usize) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    bw.hash(&mut h);
    bh.hash(&mut h);
    let step = (tgt.len() / 256).max(1);
    for i in (0..tgt.len()).step_by(step) {
        tgt[i][0].to_bits().hash(&mut h);
        tgt[i][1].to_bits().hash(&mut h);
        tgt[i][2].to_bits().hash(&mut h);
    }
    let hstep = (hole.len() / 512).max(1);
    for i in (0..hole.len()).step_by(hstep) {
        hole[i].hash(&mut h);
    }
    h.finish()
}

/// Content-aware LaMa local inpainting of the 40%-context bounding box: `src` becomes `tgt` with
/// every pixel under the binary stroke mask (`alpha > 0.05`) reconstructed via `inpaint_lama_box`
/// (rescaled to 512×512 or 1024×1024 for speed, with ISO-matched grain synthesis and LRU caching).
fn fill_box(src: &mut [[f32; 3]], tgt: &[[f32; 3]], alpha: &[f32], bw: usize, bh: usize) {
    let hole: Vec<bool> = alpha.iter().map(|a| *a > 0.05).collect();
    if !hole.iter().any(|&h| h) {
        for (d, v) in src.iter_mut().zip(tgt.iter()) {
            *d = *v;
        }
        return;
    }
    let key = lama_cache_key(tgt, &hole, bw, bh);
    if let Ok(guard) = LAMA_PATCH_CACHE.lock()
        && let Some((_, cached)) = guard.iter().find(|(k, v)| *k == key && v.len() == src.len())
    {
        src.copy_from_slice(cached);
        return;
    }
    let flat: Vec<f32> = tgt.iter().flatten().copied().collect();
    let max_side = 1024;
    let cfg = lightcraft_inpaint::LamaConfig {
        max_side,
        seed: key ^ 0x4C41_4D42,
        iso: 0,
    };
    let out = lightcraft_inpaint::inpaint_lama_box(&flat, &hole, bw, bh, &cfg);
    let (pixels, _) = out.as_chunks::<3>();
    for (d, v) in src.iter_mut().zip(pixels) {
        *d = *v;
    }
    if let Ok(mut guard) = LAMA_PATCH_CACHE.lock() {
        if guard.len() >= 24 {
            guard.remove(0);
        }
        guard.push((key, src.to_vec()));
    }
}

/// Frequency-Separated (Laplacian Pyramid High-Pass + Low-Pass) + Statistical LAB Color Transfer
/// healing in perceptual Oklab space:
/// 1. Matches mean (mu_L, mu_a, mu_b) and standard deviation (sigma_L, sigma_a, sigma_b) in LAB
///    between the target boundary ring and the source patch.
/// 2. Extracts 2-band Laplacian high-frequency texture (fine + mid) from the color-corrected source.
/// 3. Reconstructs the target zone's exact low-frequency illumination and shadow gradient via
///    normalized boundary membrane convolution and recombines High(Source') + Low(Target).
fn heal_frequency_separated_lab(src: &[[f32; 3]], tgt: &[[f32; 3]], alpha: &[f32], bw: usize, bh: usize, r: f32) -> Vec<[f32; 3]> {
    let n = bw * bh;
    let to_lab = |rgb: [f32; 3]| lightcraft_color::perceptual::oklab_from_2020(rgb.map(|c| c.max(0.0)));
    let from_lab = |lab: [f32; 3]| lightcraft_color::perceptual::oklab_to_2020([lab[0].max(0.0), lab[1], lab[2]]).map(|c| c.max(0.0));

    let src_lab: Vec<[f32; 3]> = src.iter().copied().map(to_lab).collect();
    let tgt_lab: Vec<[f32; 3]> = tgt.iter().copied().map(to_lab).collect();

    // 1. Statistical moments (mean & std) in LAB over the target boundary ring vs source patch
    let mut sum_t = [0.0f32; 3];
    let mut sum_s = [0.0f32; 3];
    let mut wt_sum = 0.0f32;
    for i in 0..n {
        let a = alpha[i];
        // Boundary ring weights: strongest right outside and along the outer edge of the stroke
        let w_ring = if a < 0.35 { 1.0 - a * 2.5 } else { 0.0 }.max(0.0);
        if w_ring > 0.0 {
            for c in 0..3 {
                sum_t[c] += tgt_lab[i][c] * w_ring;
                sum_s[c] += src_lab[i][c] * w_ring;
            }
            wt_sum += w_ring;
        }
    }
    let inv_w = if wt_sum > 1e-4 { 1.0 / wt_sum } else { 0.0 };
    let mu_t = sum_t.map(|v| v * inv_w);
    let mu_s = sum_s.map(|v| v * inv_w);

    let mut var_t = [0.0f32; 3];
    let mut var_s = [0.0f32; 3];
    if wt_sum > 1e-4 {
        for i in 0..n {
            let a = alpha[i];
            let w_ring = if a < 0.35 { 1.0 - a * 2.5 } else { 0.0 }.max(0.0);
            if w_ring > 0.0 {
                for c in 0..3 {
                    var_t[c] += (tgt_lab[i][c] - mu_t[c]).powi(2) * w_ring;
                    var_s[c] += (src_lab[i][c] - mu_s[c]).powi(2) * w_ring;
                }
            }
        }
    }
    let std_t = var_t.map(|v| (v * inv_w).max(0.0).sqrt());
    let std_s = var_s.map(|v| (v * inv_w).max(0.0).sqrt());
    let gain_l = ((std_t[0] + 1e-3) / (std_s[0] + 1e-3)).clamp(0.60, 1.65);
    let gain_a = ((std_t[1] + 5e-4) / (std_s[1] + 5e-4)).clamp(0.55, 1.55);
    let gain_b = ((std_t[2] + 5e-4) / (std_s[2] + 5e-4)).clamp(0.55, 1.55);

    // Statistically color-adapted source in LAB
    let src_adapted = Rgb32f {
        width: bw,
        height: bh,
        data: src_lab
            .iter()
            .map(|s| {
                [
                    (mu_t[0] + (s[0] - mu_s[0]) * gain_l).max(0.0),
                    mu_t[1] + (s[1] - mu_s[1]) * gain_a,
                    mu_t[2] + (s[2] - mu_s[2]) * gain_b,
                ]
            })
            .collect(),
    };

    // 2. Laplacian Pyramid Frequency Separation (Fine + Mid High-Pass vs Coarse Low-Pass)
    let sigma_fine = (r * 0.22).max(0.8);
    let sigma_coarse = (r * 0.62).max(1.6);
    let src_blur_fine = lightcraft_raster::blur::gaussian(&src_adapted, sigma_fine);
    let src_blur_coarse = lightcraft_raster::blur::gaussian(&src_adapted, sigma_coarse);

    // 3. Target Low-Pass illumination & shadow gradient via normalized exterior boundary membrane
    // of (Target_LAB - LowPass(Source'_LAB))
    let outside: Vec<f32> = alpha.iter().map(|a| if *a < 0.02 { 1.0 } else { 0.0 }).collect();
    let diff_low = Rgb32f {
        width: bw,
        height: bh,
        data: tgt_lab
            .iter()
            .zip(&src_blur_coarse.data)
            .zip(&outside)
            .map(|((t, s_low), m)| [(t[0] - s_low[0]) * m, (t[1] - s_low[1]) * m, (t[2] - s_low[2]) * m])
            .collect(),
    };
    let wts = lightcraft_raster::Plane { width: bw, height: bh, data: outside };
    let sigma_mem = (r * 0.65).max(1.2);
    let (bd, bm) = (
        lightcraft_raster::blur::gaussian(&diff_low, sigma_mem),
        lightcraft_raster::blur::gaussian(&wts, sigma_mem),
    );

    // 4. Recombine: Target Low-Pass Illumination + Source Laplacian High-Pass (Fine + Mid bands)
    let mut out_rgb = Vec::with_capacity(n);
    for i in 0..n {
        let s_full = src_adapted.data[i];
        let s_mid = src_blur_fine.data[i];
        let s_low = src_blur_coarse.data[i];
        // Laplacian pyramid detail bands of the statistically color-matched source
        let h_fine = [s_full[0] - s_mid[0], s_full[1] - s_mid[1], s_full[2] - s_mid[2]];
        let h_mid = [s_mid[0] - s_low[0], s_mid[1] - s_low[1], s_mid[2] - s_low[2]];

        // Target illumination & shadow gradient
        let m = bm.data[i].max(1e-4);
        let d_low = [bd.data[i][0] / m, bd.data[i][1] / m, bd.data[i][2] / m];
        let t_low = [s_low[0] + d_low[0], s_low[1] + d_low[1], s_low[2] + d_low[2]];

        let healed_lab = [
            t_low[0] + h_fine[0] + h_mid[0],
            t_low[1] + h_fine[1] + h_mid[1],
            t_low[2] + h_fine[2] + h_mid[2],
        ];
        out_rgb.push(from_lab(healed_lab));
    }
    out_rgb
}

/// Apply all spots to `img` (output resolution). `ppl` = output pixels per long-edge unit.
pub fn apply(img: &mut Rgb32f, spots: &[Spot], frame: &Frame, ppl: f64) {
    if spots.is_empty() {
        return;
    }
    let (w, h) = (img.width, img.height);
    let to_out = frame.norm_to_out(w, h);
    for spot in spots {
        let opacity = (spot.opacity / 100.0).clamp(0.0, 1.0) as f32;
        if opacity <= 0.0 {
            continue;
        }
        let r = radius_px(spot.size, ppl) as f32;
        let pts: Vec<(f32, f32)> = spot
            .points
            .iter()
            .map(|p| {
                let q = to_out.apply(*p);
                (q.x as f32, q.y as f32)
            })
            .collect();
        let Some(&first) = pts.first() else { continue };
        let fill = spot.mode == SpotMode::Fill;
        let offset = match spot.source_offset {
            _ if fill => (0.0, 0.0),
            Some(o) => {
                let a = to_out.apply(Point::new(0.0, 0.0));
                let b = to_out.apply(o);
                ((b.x - a.x) as f32, (b.y - a.y) as f32)
            }
            None => auto_source(img, first, r),
        };
        // Tight bounding box of the stroke.
        let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for p in &pts {
            x0 = x0.min(p.0 - r);
            y0 = y0.min(p.1 - r);
            x1 = x1.max(p.0 + r);
            y1 = y1.max(p.1 + r);
        }
        // In AI LaMa mode (`Fill`), crop with a 40% context margin around the stroke bbox (at least 2r)
        let (pad_x, pad_y) = if fill {
            let sw = (x1 - x0).max(1.0);
            let sh = (y1 - y0).max(1.0);
            ((sw * 0.40).max(2.0 * r), (sh * 0.40).max(2.0 * r))
        } else {
            (r, r)
        };
        let (bx0, by0) = (((x0 - pad_x).floor().max(0.0)) as usize, ((y0 - pad_y).floor().max(0.0)) as usize);
        let (bx1, by1) = (((x1 + pad_x).ceil() as usize).min(w), ((y1 + pad_y).ceil() as usize).min(h));
        if bx1 <= bx0 || by1 <= by0 {
            continue;
        }
        let (bw, bh) = (bx1 - bx0, by1 - by0);
        let feather = (spot.feather / 100.0).clamp(0.0, 1.0) as f32;
        // For AI Fill, when feather > 0, enforce a smooth 8px feather (scaled to output resolution)
        let feather_band = if fill && feather > 0.0 {
            let px_8 = (8.0 * (ppl as f32 / 1200.0)).max(1.5).min(r * 0.9);
            (r * feather * 0.8).max(px_8)
        } else {
            r * feather * 0.8
        };
        let inner = (r - feather_band).max(0.0);
        // alpha of the stroke and source patch in the box
        let mut alpha = vec![0.0f32; bw * bh];
        let mut src = vec![[0.0f32; 3]; bw * bh];
        let mut tgt = vec![[0.0f32; 3]; bw * bh];
        for y in 0..bh {
            for x in 0..bw {
                let (px, py) = ((bx0 + x) as f32 + 0.5, (by0 + y) as f32 + 0.5);
                let d = pts.iter().map(|p| ((px - p.0).powi(2) + (py - p.1).powi(2)).sqrt()).fold(f32::MAX, f32::min);
                alpha[y * bw + x] = (1.0 - smooth(inner, r, d)) * opacity;
                src[y * bw + x] = img.sample_bilinear(px + offset.0, py + offset.1);
                tgt[y * bw + x] = img.get(bx0 + x, by0 + y);
            }
        }
        if fill {
            fill_box(&mut src, &tgt, &alpha, bw, bh);
        }
        let heal = matches!(spot.mode, SpotMode::Remove | SpotMode::Heal);
        let healed = if heal {
            Some(heal_frequency_separated_lab(&src, &tgt, &alpha, bw, bh, r))
        } else {
            None
        };
        for y in 0..bh {
            for x in 0..bw {
                let i = y * bw + x;
                let a = alpha[i];
                if a <= 0.0 {
                    continue;
                }
                let v = match &healed {
                    Some(h_buf) => h_buf[i],
                    None => src[i],
                };
                let t = tgt[i];
                img.set(bx0 + x, by0 + y, [t[0] + (v[0] - t[0]) * a, t[1] + (v[1] - t[1]) * a, t[2] + (v[2] - t[2]) * a].map(|c| c.max(0.0)));
            }
        }
    }
}

static OUTPAINT_CACHE: OnceLock<Mutex<Vec<(u64, usize, usize, Vec<[f32; 3]>)>>> = OnceLock::new();

/// Applies generative AI border expansion (Outpainting via LaMa ONNX + multi-scale texture synthesis)
/// and/or automatic corner filling for rotated/straightened frames.
pub fn apply_outpaint(img: &mut Rgb32f, ovl: &lightcraft_develop::CreativeOverlay) {
    if !ovl.has_outpaint() || img.width < 16 || img.height < 16 {
        return;
    }
    let w = img.width;
    let h = img.height;

    let top_pct = (ovl.expand_all + ovl.expand_top).clamp(0.0, 40.0) / 100.0;
    let bot_pct = (ovl.expand_all + ovl.expand_bottom).clamp(0.0, 40.0) / 100.0;
    let left_pct = (ovl.expand_all + ovl.expand_left).clamp(0.0, 40.0) / 100.0;
    let right_pct = (ovl.expand_all + ovl.expand_right).clamp(0.0, 40.0) / 100.0;
    let fill_corners = ovl.fill_borders >= 0.5;

    // Hash parameters + sparse image content fingerprint
    let mut h_state = DefaultHasher::new();
    w.hash(&mut h_state);
    h.hash(&mut h_state);
    top_pct.to_bits().hash(&mut h_state);
    bot_pct.to_bits().hash(&mut h_state);
    left_pct.to_bits().hash(&mut h_state);
    right_pct.to_bits().hash(&mut h_state);
    fill_corners.hash(&mut h_state);
    let step = (img.data.len() / 96).max(1);
    for px in img.data.iter().step_by(step) {
        ((px[0] * 1024.0) as i32).hash(&mut h_state);
        ((px[1] * 1024.0) as i32).hash(&mut h_state);
        ((px[2] * 1024.0) as i32).hash(&mut h_state);
    }
    let cache_key = h_state.finish();

    let cache = OUTPAINT_CACHE.get_or_init(|| Mutex::new(Vec::with_capacity(6)));
    if let Ok(guard) = cache.lock()
        && let Some((_, _, _, cached)) = guard.iter().find(|(k, cw, ch, _)| *k == cache_key && *cw == w && *ch == h)
        && cached.len() == w * h
    {
        img.data.copy_from_slice(cached);
        return;
    }

    let x0 = ((w as f64 * (left_pct / (1.0 + left_pct + right_pct))).round() as usize).min(w / 3);
    let pad_r = ((w as f64 * (right_pct / (1.0 + left_pct + right_pct))).round() as usize).min(w / 3);
    let x1 = w.saturating_sub(pad_r).max(x0 + 8).min(w);

    let y0 = ((h as f64 * (top_pct / (1.0 + top_pct + bot_pct))).round() as usize).min(h / 3);
    let pad_b = ((h as f64 * (bot_pct / (1.0 + top_pct + bot_pct))).round() as usize).min(h / 3);
    let y1 = h.saturating_sub(pad_b).max(y0 + 8).min(h);

    let iw = (x1 - x0).max(1);
    let ih = (y1 - y0).max(1);

    let mut canvas = vec![0.0f32; w * h * 3];
    let mut border_hole = vec![false; w * h];

    let has_margin = x0 > 0 || x1 < w || y0 > 0 || y1 < h;
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if has_margin {
                if x >= x0 && x < x1 && y >= y0 && y < y1 {
                    let sx = ((x - x0) as f32 + 0.5) * (w as f32 / iw as f32) - 0.5;
                    let sy = ((y - y0) as f32 + 0.5) * (h as f32 / ih as f32) - 0.5;
                    let p = img.sample_bilinear(sx, sy);
                    let pl = img.sample_bilinear(sx - 0.75, sy);
                    let pr = img.sample_bilinear(sx + 0.75, sy);
                    let pu = img.sample_bilinear(sx, sy - 0.75);
                    let pd = img.sample_bilinear(sx, sy + 0.75);
                    for c in 0..3 {
                        let lap = p[c] - 0.25 * (pl[c] + pr[c] + pu[c] + pd[c]);
                        canvas[i * 3 + c] = (p[c] + 0.22 * lap).max(0.0);
                    }
                } else {
                    border_hole[i] = true;
                }
            } else {
                let p = img.get(x, y);
                canvas[i * 3] = p[0];
                canvas[i * 3 + 1] = p[1];
                canvas[i * 3 + 2] = p[2];
            }

            if fill_corners && !border_hole[i] {
                // Detect neutral BLANK canvas pixels near outer borders from rotation/perspective warps
                let near_edge = x < w / 5 || x >= w - w / 5 || y < h / 5 || y >= h - h / 5;
                if near_edge {
                    let r = canvas[i * 3];
                    let g = canvas[i * 3 + 1];
                    let b = canvas[i * 3 + 2];
                    if (r - g).abs() < 1e-4 && (g - b).abs() < 1e-4 && (r - 0.55).abs() < 0.15 {
                        border_hole[i] = true;
                    }
                }
            }
        }
    }

    if !border_hole.iter().any(|&b| b) {
        return;
    }

    let cfg = LamaConfig {
        max_side: 1024,
        seed: cache_key ^ 0x4F55_5450_2027,
        iso: 0,
    };
    let out_flat = lightcraft_inpaint::outpaint_lama_borders(&canvas, &border_hole, w, h, &cfg);
    if out_flat.len() == w * h * 3 {
        let mut out_px = Vec::with_capacity(w * h);
        for chunk in out_flat.chunks_exact(3) {
            out_px.push([chunk[0].max(0.0), chunk[1].max(0.0), chunk[2].max(0.0)]);
        }
        img.data.copy_from_slice(&out_px);
        if let Ok(mut guard) = cache.lock() {
            if guard.len() >= 6 {
                guard.remove(0);
            }
            guard.push((cache_key, w, h, out_px));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lightcraft_develop::DevelopSettings;

    #[test]
    fn hard_edge_spots_at_zero_opacity_leave_every_pixel_unchanged() {
        for mode in [SpotMode::Clone, SpotMode::Heal, SpotMode::Remove, SpotMode::Fill] {
            for (cx, cy) in [(32.5, 24.5), (0.5, 0.5)] {
                let original = Rgb32f::from_fn(64, 48, |x, y| [0.2 + x as f32 / 128.0, 0.3 + y as f32 / 128.0, 0.4]);
                let mut img = original.clone();
                let frame = Frame::new(64, 48, &DevelopSettings::default(), true);
                let spot = Spot {
                    mode,
                    points: vec![Point::new(cx / 64.0, cy / 48.0)],
                    size: 5.0 / 64.0,
                    feather: 0.0,
                    opacity: 0.0,
                    source_offset: Some(Point::new(8.0 / 64.0, 0.0)),
                };
                apply(&mut img, &[spot], &frame, frame.px_per_long(64));
                for (i, (actual, expected)) in img.data.iter().zip(&original.data).enumerate() {
                    assert_eq!(actual, expected, "zero-opacity {mode:?} at ({cx}, {cy}), pixel {i}");
                }
            }
        }
    }

    #[test]
    fn hard_edge_clone_copies_inside_and_preserves_pixels_on_the_radius() {
        for (w, h, cx, cy, radius) in [(64, 48, 32.5, 24.5, 5.0), (128, 96, 64.5, 48.5, 10.0)] {
            let original = Rgb32f::from_fn(w, h, |x, y| [0.2 + x as f32 / w as f32, y as f32 / h as f32, 0.4]);
            let mut img = original.clone();
            let frame = Frame::new(w, h, &DevelopSettings::default(), true);
            let spot = Spot {
                mode: SpotMode::Clone,
                points: vec![Point::new(cx / w as f64, cy / h as f64)],
                size: radius / w as f64,
                feather: 0.0,
                opacity: 100.0,
                source_offset: Some(Point::new(0.125, 0.0)),
            };
            apply(&mut img, &[spot], &frame, frame.px_per_long(w));
            let (x, y) = (cx as usize, cy as usize);
            assert_eq!(img.get(x, y), original.get(x + w / 8, y), "inside copies the source at {w}px");
            assert_eq!(img.get(x + radius as usize, y), original.get(x + radius as usize, y), "radius stays unchanged at {w}px");
            assert!(img.data.iter().flatten().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn remove_erases_a_dot() {
        // flat grey with a gentle gradient and a black dot in the middle
        let mut img = Rgb32f::from_fn(120, 80, |x, _| [0.2 + x as f32 * 0.001; 3]);
        for y in 36..44 {
            for x in 56..64 {
                img.set(x, y, [0.0; 3]);
            }
        }
        let frame = Frame::new(120, 80, &DevelopSettings::default(), true);
        let spot = Spot { points: vec![Point::new(0.5, 0.5)], size: 8.0 / 120.0, feather: 30.0, ..Default::default() };
        apply(&mut img, &[spot], &frame, frame.px_per_long(120));
        let c = img.get(60, 40);
        assert!((c[0] - 0.26).abs() < 0.03, "{c:?}");
    }

    /// Vertical stripes (period `period` px) with a black blot over the middle.
    fn striped_with_blot(w: usize, h: usize, period: usize) -> Rgb32f {
        let mut img = Rgb32f::from_fn(w, h, |x, _| if (x / period).is_multiple_of(2) { [0.2; 3] } else { [0.7; 3] });
        let (cx, cy, r) = (w as f32 / 2.0, h as f32 / 2.0, w as f32 * 0.04);
        for y in 0..h {
            for x in 0..w {
                if (x as f32 + 0.5 - cx).hypot(y as f32 + 0.5 - cy) < r {
                    img.set(x, y, [0.0; 3]);
                }
            }
        }
        img
    }

    fn fill_spot() -> Spot {
        // a source offset is ignored by Fill
        Spot {
            mode: SpotMode::Fill,
            points: vec![Point::new(0.5, 0.5)],
            size: 0.06,
            feather: 0.0,
            source_offset: Some(Point::new(0.3, 0.0)),
            ..Default::default()
        }
    }

    #[test]
    fn fill_rebuilds_structure_deterministically() {
        let (w, h) = (200, 160);
        let mut img = striped_with_blot(w, h, 8);
        let frame = Frame::new(w, h, &DevelopSettings::default(), true);
        apply(&mut img, &[fill_spot()], &frame, frame.px_per_long(w));
        // the blot is gone and the stripes continue through it
        let mut err = 0.0f32;
        for x in 90..110 {
            let want = if (x / 8) % 2 == 0 { 0.2 } else { 0.7 };
            err += (img.get(x, 80)[0] - want).abs();
        }
        assert!(err / 20.0 < 0.12, "mean error {}", err / 20.0);
        let mut again = striped_with_blot(w, h, 8);
        apply(&mut again, &[fill_spot()], &frame, frame.px_per_long(w));
        assert_eq!(img.data, again.data);
    }

    #[test]
    fn fill_matches_between_preview_and_export_size() {
        // the same photo at two sizes: the half-size fill matches the full-size one, downsampled
        let (mut big, mut small) = (striped_with_blot(320, 240, 16), striped_with_blot(160, 120, 8));
        let (fb, fs) = (Frame::new(320, 240, &DevelopSettings::default(), true), Frame::new(160, 120, &DevelopSettings::default(), true));
        apply(&mut big, &[fill_spot()], &fb, fb.px_per_long(320));
        apply(&mut small, &[fill_spot()], &fs, fs.px_per_long(160));
        let (mut err, mut n) = (0.0f32, 0);
        for y in 50..70 {
            for x in 70..90 {
                let b: f32 = [(0, 0), (1, 0), (0, 1), (1, 1)].iter().map(|(dx, dy)| big.get(2 * x + dx, 2 * y + dy)[0]).sum::<f32>() / 4.0;
                err += (b - small.get(x, y)[0]).abs();
                n += 1;
            }
        }
        assert!(err / n as f32 <= 0.08, "mean difference {}", err / n as f32);
    }

    #[test]
    fn fill_survives_hostile_spots() {
        let frame = Frame::new(40, 30, &DevelopSettings::default(), true);
        for (p, size) in [((0.0, 0.0), 2.0), ((-5.0, 9.0), 0.1), ((0.5, 0.5), 0.0), ((f64::NAN, 0.5), 0.1), ((0.5, 0.5), f64::INFINITY)] {
            let mut img = Rgb32f::from_fn(40, 30, |x, _| [x as f32 / 40.0; 3]);
            let spot = Spot { mode: SpotMode::Fill, points: vec![Point::new(p.0, p.1)], size, ..Default::default() };
            apply(&mut img, &[spot], &frame, frame.px_per_long(40));
            assert!(img.data.iter().flatten().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn clone_copies_source() {
        let mut img = Rgb32f::from_fn(100, 100, |x, _| if x < 50 { [0.1; 3] } else { [0.9; 3] });
        let frame = Frame::new(100, 100, &DevelopSettings::default(), true);
        let spot = Spot {
            mode: SpotMode::Clone,
            points: vec![Point::new(0.25, 0.5)],
            size: 0.05,
            feather: 0.0,
            source_offset: Some(Point::new(0.5, 0.0)),
            ..Default::default()
        };
        apply(&mut img, &[spot], &frame, frame.px_per_long(100));
        assert!(img.get(25, 50)[0] > 0.8);
        assert!(img.get(5, 50)[0] < 0.2);
    }

    fn with_spots(spots: Vec<Spot>) -> DevelopSettings {
        DevelopSettings { spots, ..Default::default() }
    }

    fn spot_at(x: f64, y: f64, size: f64, offset: Option<Point>) -> Spot {
        Spot { points: vec![Point::new(x, y)], size, source_offset: offset, ..Default::default() }
    }

    #[test]
    fn a_window_grows_to_hold_the_spots_that_touch_it_and_their_sources() {
        use crate::PixelWindow;
        let frame = Frame::new(4000, 3000, &DevelopSettings::default(), true);
        let ppl = frame.px_per_long(4000);
        let win = PixelWindow { x: 1000, y: 1000, w: 400, h: 300 };
        // none: the window is left alone
        assert_eq!(window_for_reads(&with_spots(vec![]), &frame, 4000, 3000, ppl, win), win);
        // a spot far away is no business of this window
        let far = spot_at(0.9, 0.9, 0.01, Some(Point::new(0.02, 0.0)));
        assert_eq!(window_for_reads(&with_spots(vec![far]), &frame, 4000, 3000, ppl, win), win);
        // a spot inside it, reading from 300 px to the right, widens it to the source
        let near = spot_at(0.30, 0.38, 0.01, Some(Point::new(0.075, 0.0)));
        let g = window_for_reads(&with_spots(vec![near]), &frame, 4000, 3000, ppl, win);
        assert!(g.x <= win.x && g.x + g.w >= 1200 + 300 + 80 && g.y <= win.y && g.y + g.h >= win.y + win.h, "{g:?}");
        // a spot that only reaches in with its edge counts too
        let edge = spot_at(0.24, 0.38, 0.01, Some(Point::new(-0.05, 0.0)));
        assert!(window_for_reads(&with_spots(vec![edge]), &frame, 4000, 3000, ppl, win).x < win.x);
    }

    #[test]
    fn spot_windows_are_bounded_and_survive_hostile_spots() {
        use crate::PixelWindow;
        let frame = Frame::new(40_000, 30_000, &DevelopSettings::default(), true);
        let ppl = frame.px_per_long(40_000);
        let win = PixelWindow { x: 1000, y: 1000, w: 400, h: 300 };
        // a spot whose source is a long way off would need an enormous window: not granted
        let wide = spot_at(0.03, 0.04, 0.01, Some(Point::new(0.6, 0.0)));
        let g = window_for_reads(&with_spots(vec![wide]), &frame, 40_000, 30_000, ppl, win);
        assert!(g.w <= MAX_SPOT_SPAN && g.h <= MAX_SPOT_SPAN, "{g:?}");
        for bad in [f64::NAN, f64::INFINITY, 1e300, -1e300] {
            let s = spot_at(bad, bad, bad, Some(Point::new(bad, bad)));
            let g = window_for_reads(&with_spots(vec![s]), &frame, 40_000, 30_000, ppl, win);
            assert_eq!(g, win);
        }
        let empty = Spot { points: vec![], ..Default::default() };
        assert_eq!(window_for_reads(&with_spots(vec![empty]), &frame, 40_000, 30_000, ppl, win), win);
    }

    // Given an Auto Mask stroke reaching into the window, the window grows to hold all of it
    #[test]
    fn a_window_grows_to_hold_an_auto_mask_stroke() {
        use crate::PixelWindow;
        use lightcraft_develop::{BrushStroke, Mask, MaskComponent, MaskOp, MaskShape};
        let frame = Frame::new(4000, 3000, &DevelopSettings::default(), true);
        let ppl = frame.px_per_long(4000);
        let win = PixelWindow { x: 1000, y: 1000, w: 400, h: 300 };
        let stroke =
            |auto| BrushStroke { points: vec![Point::new(0.30, 0.38), Point::new(0.5, 0.38)], size: 0.01, auto_mask: auto, ..Default::default() };
        let mask = |auto| Mask {
            components: vec![MaskComponent { name: None, op: MaskOp::Add, invert: false, shape: MaskShape::Brush { strokes: vec![stroke(auto)] } }],
            ..Default::default()
        };
        let plain = DevelopSettings { masks: vec![mask(false)], ..Default::default() };
        assert_eq!(window_for_reads(&plain, &frame, 4000, 3000, ppl, win), win, "a plain stroke reads nothing around it");
        let auto = DevelopSettings { masks: vec![mask(true)], ..Default::default() };
        let g = window_for_reads(&auto, &frame, 4000, 3000, ppl, win);
        // the dabs near the window, not the whole stroke (it runs on to x = 2000)
        assert!(g.x <= 1000 && g.x + g.w > 1400 && g.x + g.w < 1700, "{g:?}");
        // a window at the far end of a long stroke does not grow back along all of it
        let far = window_for_reads(&auto, &frame, 4000, 3000, ppl, PixelWindow { x: 1900, ..win });
        assert!(far.x > 1500, "{far:?}");
        // a hidden mask reads nothing
        let hidden = DevelopSettings { masks: vec![Mask { visible: false, ..auto.masks[0].clone() }], ..Default::default() };
        assert_eq!(window_for_reads(&hidden, &frame, 4000, 3000, ppl, win), win);
    }

    // Given a window that would have to grow past the cap, there is no answer (never a wrong one)
    #[test]
    fn a_window_that_cannot_hold_what_it_reads_is_refused() {
        use crate::PixelWindow;
        let frame = Frame::new(40_000, 30_000, &DevelopSettings::default(), true);
        let ppl = frame.px_per_long(40_000);
        let win = PixelWindow { x: 1000, y: 1000, w: 400, h: 300 };
        let far = with_spots(vec![spot_at(0.03, 0.04, 0.01, Some(Point::new(0.6, 0.0)))]);
        assert_eq!(window_for_reads_checked(&far, &frame, 40_000, 30_000, ppl, win), None);
        let near = with_spots(vec![spot_at(0.03, 0.04, 0.01, Some(Point::new(0.02, 0.0)))]);
        assert!(window_for_reads_checked(&near, &frame, 40_000, 30_000, ppl, win).is_some());
        // a spot that doesn't touch the window is no business of it, however far it reads
        let elsewhere = with_spots(vec![spot_at(0.9, 0.9, 0.01, Some(Point::new(-0.8, 0.0)))]);
        assert_eq!(window_for_reads_checked(&elsewhere, &frame, 40_000, 30_000, ppl, win), Some(win));
    }
}
