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

/// Mean colour of `img` in a disc/annulus. When `require_all_inside` is true (source candidate),
/// returns `None` if any sample falls outside the image bounds; when false (target ring near an edge),
/// collects all in-bounds samples so edge-touching spots still find valid inward source patches.
fn ring_stats(img: &Rgb32f, c: (f32, f32), r0: f32, r1: f32, require_all_inside: bool) -> Option<Vec<[f32; 3]>> {
    let mut out = Vec::new();
    let n = 24;
    for i in 0..n {
        let a = i as f32 / n as f32 * std::f32::consts::TAU;
        for rr in [r0, (r0 + r1) / 2.0, r1] {
            let (x, y) = (c.0 + a.cos() * rr, c.1 + a.sin() * rr);
            if x < 0.0 || y < 0.0 || x >= img.width as f32 || y >= img.height as f32 {
                if require_all_inside {
                    return None;
                }
                continue;
            }
            out.push(img.sample_bilinear(x, y));
        }
    }
    if out.len() < 12 {
        return None;
    }
    Some(out)
}

fn score(img: &Rgb32f, target: (f32, f32), source: (f32, f32), r: f32) -> f32 {
    let (Some(a), Some(b)) = (
        ring_stats(img, target, r * 1.05, r * 1.6, false),
        ring_stats(img, source, r * 1.05, r * 1.6, true),
    ) else {
        return f32::MAX;
    };
    // Compare in perceptual Oklab (luminance + chrominance) plus interior texture roughness continuity
    let inside = ring_stats(img, source, r * 0.2, r * 0.9, true);
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
    let fallback_dx = if target.0 > img.width as f32 * 0.5 { -r * 2.5 } else { r * 2.5 };
    ranked_sources(img, target, r).first().copied().unwrap_or((fallback_dx, 0.0))
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
    // Connected/nearby `SpotMode::Fill` strokes are merged into unified clusters first so multi-stroke
    // object removals (`Borrar Objetos IA (3)`) share a single square context window.
    let mut fill_clusters: Vec<[f64; 5]> = Vec::new(); // [x0, y0, x1, y1, max_r]
    let mut boxes: Vec<([f64; 4], [f64; 4])> = Vec::new();
    for spot in spots {
        let r = radius_px(spot.size, ppl);
        let pts: Vec<_> = spot.points.iter().map(|p| to_out.apply(*p)).collect();
        let Some(first) = pts.first() else { continue };
        let (mut x0, mut y0, mut x1, mut y1) = (first.x, first.y, first.x, first.y);
        for p in &pts {
            (x0, y0, x1, y1) = (x0.min(p.x), y0.min(p.y), x1.max(p.x), y1.max(p.y));
        }
        if spot.mode == SpotMode::Fill
            || (spot.mode == SpotMode::Remove && (spot.source_offset.is_none() || pts.len() >= 12) && pts.len() >= 2)
        {
            let b = [x0 - r, y0 - r, x1 + r, y1 + r, r];
            if b.iter().all(|v| v.is_finite()) {
                fill_clusters.push(b);
            }
        } else {
            let t = [x0 - 2.0 * r, y0 - 2.0 * r, x1 + 2.0 * r, y1 + 2.0 * r];
            let s = match spot.source_offset {
                Some(o) => {
                    let b = to_out.apply(o);
                    let (dx, dy) = (b.x - origin.x, b.y - origin.y);
                    [t[0] + dx, t[1] + dy, t[2] + dx, t[3] + dy]
                }
                None => [t[0] - 4.2 * r, t[1] - 4.2 * r, t[2] + 4.2 * r, t[3] + 4.2 * r],
            };
            if t.iter().chain(&s).all(|v| v.is_finite()) {
                boxes.push((t, s));
            }
        }
    }
    loop {
        let mut merged_any = false;
        let mut i = 0;
        while i < fill_clusters.len() {
            let mut j = i + 1;
            while j < fill_clusters.len() {
                let a = fill_clusters[i];
                let b = fill_clusters[j];
                let gap = 2.5 * a[4].max(b[4]);
                if a[0] - gap <= b[2] && a[2] + gap >= b[0] && a[1] - gap <= b[3] && a[3] + gap >= b[1] {
                    fill_clusters[i] = [
                        a[0].min(b[0]),
                        a[1].min(b[1]),
                        a[2].max(b[2]),
                        a[3].max(b[3]),
                        a[4].max(b[4]),
                    ];
                    fill_clusters.remove(j);
                    merged_any = true;
                } else {
                    j += 1;
                }
            }
            i += 1;
        }
        if !merged_any {
            break;
        }
    }
    for c in fill_clusters {
        let (x0, y0, x1, y1, r) = (c[0], c[1], c[2], c[3], c[4]);
        let sw = (x1 - x0).max(1.0);
        let sh = (y1 - y0).max(1.0);
        let cx = 0.5 * (x0 + x1);
        let cy = 0.5 * (y0 + y1);
        let wf = full_w as f64;
        let hf = full_h as f64;
        let sx = (sw * 1.85).max(sw + 6.5 * r).max(sh * 0.85).min(wf);
        let sy = (sh * 2.15).max(sh + 6.5 * r).max(sw * 0.62).min(hf);
        let mut rx0 = cx - 0.5 * sx;
        let mut rx1 = cx + 0.5 * sx;
        if rx0 < 0.0 {
            rx1 = (rx1 - rx0).min(wf);
            rx0 = 0.0;
        }
        if rx1 > wf {
            rx0 = (rx0 - (rx1 - wf)).max(0.0);
            rx1 = wf;
        }
        let mut ry0 = cy - 0.5 * sy;
        let mut ry1 = cy + 0.5 * sy;
        if ry0 < 0.0 {
            ry1 = (ry1 - ry0).min(hf);
            ry0 = 0.0;
        }
        if ry1 > hf {
            ry0 = (ry0 - (ry1 - hf)).max(0.0);
            ry1 = hf;
        }
        let t = [rx0, ry0, rx1, ry1];
        boxes.push((t, t));
    }
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

struct CachedSpotPatch {
    seed: u64,
    mode_tag: u64,
    bw: usize,
    bh: usize,
    grid_sig: [f32; 96],
    pixels: Vec<[f32; 3]>,
}

static LAMA_PATCH_CACHE: std::sync::LazyLock<std::sync::Mutex<Vec<CachedSpotPatch>>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(Vec::new()));

fn spot_grid_sig(tgt: &[[f32; 3]], hole: &[bool], bw: usize, bh: usize) -> [f32; 96] {
    let mut sums = [0.0f32; 96];
    let mut cnts = [0.0f32; 32];
    for y in 0..bh {
        let gy = ((y * 2) / bh.max(1)).min(1);
        for x in 0..bw {
            let i = y * bw + x;
            if !hole[i] {
                let gx = ((x * 16) / bw.max(1)).min(15);
                let b = gy * 16 + gx;
                sums[b * 3] += tgt[i][0];
                sums[b * 3 + 1] += tgt[i][1];
                sums[b * 3 + 2] += tgt[i][2];
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

fn spot_grid_matches(a: &[f32; 96], b: &[f32; 96]) -> bool {
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

/// Content-aware LaMa local inpainting of the 40%-context bounding box: `src` becomes `tgt` with
/// every pixel under the binary stroke mask (`alpha > 0.05`) reconstructed via `inpaint_lama_box`
/// (rescaled to 512×512 or 1024×1024 for speed, with ISO-matched grain synthesis and LRU caching).
fn fill_box(
    src: &mut [[f32; 3]],
    tgt: &[[f32; 3]],
    alpha: &[f32],
    bw: usize,
    bh: usize,
    group_seed: u64,
    is_thumb_pass: bool,
) {
    let raw_hole: Vec<bool> = alpha.iter().map(|a| *a > 0.02).collect();
    if !raw_hole.iter().any(|&h| h) {
        for (d, v) in src.iter_mut().zip(tgt.iter()) {
            *d = *v;
        }
        return;
    }
    // Dilate binary mask by 1px (inpaint_lama_box dilates by 2px more) so the outer seam is inside the reconstructed zone
    let mut hole = raw_hole.clone();
    for y in 0..bh {
        for x in 0..bw {
            if raw_hole[y * bw + x] {
                for dy in -1isize..=1 {
                    for dx in -1isize..=1 {
                        let ny = (y as isize + dy).clamp(0, (bh - 1) as isize) as usize;
                        let nx = (x as isize + dx).clamp(0, (bw - 1) as isize) as usize;
                        hole[ny * bw + nx] = true;
                    }
                }
            }
        }
    }
    let grid_sig = spot_grid_sig(tgt, &hole, bw, bh);
    let (cloud_configured, mode_tag) = lightcraft_inpaint::current_cloud_mode_tag();
    if !is_thumb_pass
        && let Ok(guard) = LAMA_PATCH_CACHE.lock()
        && let Some(cached) = guard.iter().find(|c| {
            c.seed == group_seed
                && c.mode_tag == mode_tag
                && c.bw == bw
                && c.bh == bh
                && c.pixels.len() == src.len()
                && spot_grid_matches(&c.grid_sig, &grid_sig)
        })
    {
        src.copy_from_slice(&cached.pixels);
        return;
    }
    let flat: Vec<f32> = tgt.iter().flatten().copied().collect();
    let max_side = if is_thumb_pass { 512 } else { 1024 };
    let cfg = lightcraft_inpaint::LamaConfig {
        max_side,
        seed: group_seed,
        iso: 0,
    };
    let out = lightcraft_inpaint::inpaint_lama_box(&flat, &hole, bw, bh, &cfg);
    let (pixels, _) = out.as_chunks::<3>();
    for (d, v) in src.iter_mut().zip(pixels) {
        *d = *v;
    }
    let can_cache_patch = !is_thumb_pass
        && (!cloud_configured
            || (group_seed & 1 == 1)
            || lightcraft_inpaint::is_cloud_seed_cached(group_seed, mode_tag));
    if can_cache_patch
        && let Ok(mut guard) = LAMA_PATCH_CACHE.lock()
    {
        if let Some(existing) = guard
            .iter_mut()
            .find(|c| c.seed == group_seed && c.mode_tag == mode_tag && c.bw == bw && c.bh == bh)
        {
            existing.grid_sig = grid_sig;
            existing.pixels = src.to_vec();
        } else {
            if guard.len() >= 24 {
                guard.remove(0);
            }
            guard.push(CachedSpotPatch {
                seed: group_seed,
                mode_tag,
                bw,
                bh,
                grid_sig,
                pixels: src.to_vec(),
            });
        }
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
    let seg_dist_sq = |px: f32, py: f32, a: (f32, f32), b: (f32, f32)| -> f32 {
        let dx = b.0 - a.0;
        let dy = b.1 - a.1;
        let len_sq = dx * dx + dy * dy;
        if len_sq <= 1e-6 {
            return (px - a.0).powi(2) + (py - a.1).powi(2);
        }
        let t = (((px - a.0) * dx + (py - a.1) * dy) / len_sq).clamp(0.0, 1.0);
        let qx = a.0 + t * dx;
        let qy = a.1 + t * dy;
        (px - qx).powi(2) + (py - qy).powi(2)
    };

    // 1. Collect and cluster `SpotMode::Fill` strokes so multi-stroke object removals (`Borrar Objetos IA (3)`)
    //    are combined into a single unified mask and inpainted in one single pass (never copying un-erased
    //    parts of Stroke 2/3 into Stroke 1!).
    struct FillStroke {
        pts: Vec<(f32, f32)>,
        r: f32,
        inner: f32,
        outer: f32,
        opacity: f32,
        bbox: [f32; 4],
        spot_hash: u64,
    }
    let mut fill_groups: Vec<Vec<FillStroke>> = Vec::new();
    let (photo_x0, photo_y0, photo_x1, photo_y1, photo_w, photo_h) = match frame.view {
        Some(vw) => (
            -(vw.x as f32),
            -(vw.y as f32),
            (vw.full_w as f32) - (vw.x as f32),
            (vw.full_h as f32) - (vw.y as f32),
            vw.full_w as f32,
            vw.full_h as f32,
        ),
        None => (0.0, 0.0, w as f32, h as f32, w as f32, h as f32),
    };

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
        let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for p in &pts {
            x0 = x0.min(p.0 - r);
            y0 = y0.min(p.1 - r);
            x1 = x1.max(p.0 + r);
            y1 = y1.max(p.1 + r);
        }
        let feather = (spot.feather / 100.0).clamp(0.0, 1.0) as f32;

        if spot.mode == SpotMode::Fill
            || (spot.mode == SpotMode::Remove && (spot.source_offset.is_none() || pts.len() >= 12) && pts.len() >= 2)
        {
            let mut sh = DefaultHasher::new();
            (spot.mode as u8).hash(&mut sh);
            spot.size.to_bits().hash(&mut sh);
            spot.feather.to_bits().hash(&mut sh);
            spot.opacity.to_bits().hash(&mut sh);
            if spot.mode == SpotMode::Fill
                && let Some(so) = spot.source_offset
            {
                so.x.to_bits().hash(&mut sh);
                so.y.to_bits().hash(&mut sh);
            }
            for pt in &spot.points {
                pt.x.to_bits().hash(&mut sh);
                pt.y.to_bits().hash(&mut sh);
            }
            let spot_hash = sh.finish();
            let min_seam = (r * 0.30).clamp(2.0, 9.0);
            let extra_seam = (r * feather * 0.35).clamp(0.0, 14.0);
            let seam = (min_seam + extra_seam).min(r * 0.45);
            let (inner, outer) = ((r - seam).max(0.0), r);
            fill_groups.push(vec![FillStroke {
                pts,
                r,
                inner,
                outer,
                opacity,
                bbox: [x0, y0, x1, y1],
                spot_hash,
            }]);
            continue;
        }

        // Non-Fill (`Remove`, `Heal`, `Clone`) spots run directly:
        let offset = match spot.source_offset {
            Some(o) => {
                let a = to_out.apply(Point::new(0.0, 0.0));
                let b = to_out.apply(o);
                ((b.x - a.x) as f32, (b.y - a.y) as f32)
            }
            None => auto_source(img, first, r),
        };
        let (bx0, by0, bx1, by1) = (
            ((x0 - r).floor().max(0.0)) as usize,
            ((y0 - r).floor().max(0.0)) as usize,
            ((x1 + r).ceil() as usize).min(w),
            ((y1 + r).ceil() as usize).min(h),
        );
        if bx1 <= bx0 || by1 <= by0 {
            continue;
        }
        let (bw, bh) = (bx1 - bx0, by1 - by0);
        let band = r * feather * 0.8;
        let (inner, outer) = ((r - band).max(0.0), r);
        let mut alpha = vec![0.0f32; bw * bh];
        let mut src = vec![[0.0f32; 3]; bw * bh];
        let mut tgt = vec![[0.0f32; 3]; bw * bh];
        for y in 0..bh {
            for x in 0..bw {
                let (px, py) = ((bx0 + x) as f32 + 0.5, (by0 + y) as f32 + 0.5);
                let d_sq = if pts.len() >= 2 {
                    pts.windows(2)
                        .map(|w_pts| seg_dist_sq(px, py, w_pts[0], w_pts[1]))
                        .fold(f32::MAX, f32::min)
                } else {
                    pts.iter().map(|p| (px - p.0).powi(2) + (py - p.1).powi(2)).fold(f32::MAX, f32::min)
                };
                let d = d_sq.sqrt();
                alpha[y * bw + x] = (1.0 - smooth(inner, outer, d)) * opacity;
                src[y * bw + x] = img.sample_bilinear(px + offset.0, py + offset.1);
                tgt[y * bw + x] = img.get(bx0 + x, by0 + y);
            }
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

    // 2. Merge overlapping or nearby `SpotMode::Fill` groups (`gap <= 2.5 * max_r`)
    let group_bounds = |g: &[FillStroke]| -> ([f32; 4], f32) {
        let mut b = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];
        let mut mr = 1.0f32;
        for s in g {
            b[0] = b[0].min(s.bbox[0]);
            b[1] = b[1].min(s.bbox[1]);
            b[2] = b[2].max(s.bbox[2]);
            b[3] = b[3].max(s.bbox[3]);
            mr = mr.max(s.r);
        }
        (b, mr)
    };
    if !lightcraft_inpaint::current_cloud_mode_tag().0 {
        loop {
            let mut merged_any = false;
            let mut i = 0;
            while i < fill_groups.len() {
                let mut j = i + 1;
                while j < fill_groups.len() {
                    let (ba, ra) = group_bounds(&fill_groups[i]);
                    let (bb, rb) = group_bounds(&fill_groups[j]);
                    let gap = 2.5 * ra.max(rb);
                    if ba[0] - gap <= bb[2] && ba[2] + gap >= bb[0] && ba[1] - gap <= bb[3] && ba[3] + gap >= bb[1] {
                        let removed = fill_groups.remove(j);
                        fill_groups[i].extend(removed);
                        merged_any = true;
                    } else {
                        j += 1;
                    }
                }
                i += 1;
            }
            if !merged_any {
                break;
            }
        }
    }

    // 3. Process each unified `SpotMode::Fill` group in a single combined pass
    for group in fill_groups {
        let (gb, r) = group_bounds(&group);
        let (x0, y0, x1, y1) = (gb[0], gb[1], gb[2], gb[3]);
        let wf = w as f32;
        let hf = h as f32;
        if x1 + r <= 0.0 || x0 - r >= wf || y1 + r <= 0.0 || y0 - r >= hf {
            continue;
        }
        let sw = (x1 - x0).max(1.0);
        let sh = (y1 - y0).max(1.0);
        let cx = 0.5 * (x0 + x1);
        let sx = (sw * 1.85).max(sw + 6.5 * r).max(sh * 0.85).min(photo_w);
        let sy = (sh * 2.15).max(sh + 6.5 * r).max(sw * 0.62).min(photo_h);
        let mut rx0 = (cx - 0.5 * sx).floor();
        let mut rx1 = (cx + 0.5 * sx).ceil();
        if rx0 < photo_x0 {
            rx1 = (rx1 + (photo_x0 - rx0)).min(photo_x1);
            rx0 = photo_x0;
        }
        if rx1 > photo_x1 {
            rx0 = (rx0 - (rx1 - photo_x1)).max(photo_x0);
            rx1 = photo_x1;
        }
        let cy = 0.5 * (y0 + y1);
        let mut ry0 = (cy - 0.5 * sy).floor();
        let mut ry1 = (cy + 0.5 * sy).ceil();
        if ry0 < photo_y0 {
            ry1 = (ry1 + (photo_y0 - ry0)).min(photo_y1);
            ry0 = photo_y0;
        }
        if ry1 > photo_y1 {
            ry0 = (ry0 - (ry1 - photo_y1)).max(photo_y0);
            ry1 = photo_y1;
        }
        let (bx0, by0, bx1, by1) = (
            rx0.max(0.0) as usize,
            ry0.max(0.0) as usize,
            (rx1.max(0.0) as usize).min(w),
            (ry1.max(0.0) as usize).min(h),
        );
        if bx1 <= bx0 || by1 <= by0 {
            continue;
        }
        let mut hashes: Vec<u64> = group.iter().map(|s| s.spot_hash).collect();
        hashes.sort_unstable();
        let mut gh = DefaultHasher::new();
        0x5350_4F54_4752_5031u64.hash(&mut gh);
        for h_val in hashes {
            h_val.hash(&mut gh);
        }
        frame.orient.hash(&mut gh);
        frame.flip_h.hash(&mut gh);
        frame.flip_v.hash(&mut gh);
        let raw_seed = gh.finish();
        let photo_long = photo_w.max(photo_h).max(1.0);
        let norm_group_span = sw.max(sh) / photo_long;
        let is_compact_blemish = group.len() == 1 && norm_group_span <= 0.022 && sw.max(sh) <= 3.4 * r;
        let group_seed = if is_compact_blemish {
            raw_seed | 1
        } else {
            (raw_seed & !1).max(2)
        };
        let at_left_border = (bx0 as f32 - photo_x0).abs() <= 1.5;
        let at_right_border = (bx1 as f32 - photo_x1).abs() <= 1.5;
        let at_top_border = (by0 as f32 - photo_y0).abs() <= 1.5;
        let at_bottom_border = (by1 as f32 - photo_y1).abs() <= 1.5;
        let (bw, bh) = (bx1 - bx0, by1 - by0);
        let mut alpha = vec![0.0f32; bw * bh];
        let mut src = vec![[0.0f32; 3]; bw * bh];
        let mut tgt = vec![[0.0f32; 3]; bw * bh];
        let mut max_opacity = 0.0f32;
        let mut max_seam = 4.0f32;
        let mut total_pts = 0usize;
        struct Seg {
            a: (f32, f32),
            b: (f32, f32),
            x0: f32,
            y0: f32,
            x1: f32,
            y1: f32,
            inner: f32,
            outer: f32,
            opacity: f32,
        }
        let mut all_segs: Vec<Seg> = Vec::new();
        for st in &group {
            max_opacity = max_opacity.max(st.opacity);
            max_seam = max_seam.max(st.outer - st.inner);
            total_pts += st.pts.len();
            let min_step_sq = (st.r * 0.06).max(1.5).powi(2);
            let mut dec_pts: Vec<(f32, f32)> = Vec::with_capacity(st.pts.len());
            for (idx, &p) in st.pts.iter().enumerate() {
                if idx == 0 || idx + 1 == st.pts.len() {
                    dec_pts.push(p);
                } else if let Some(&last) = dec_pts.last()
                    && (p.0 - last.0).powi(2) + (p.1 - last.1).powi(2) >= min_step_sq
                {
                    dec_pts.push(p);
                }
            }
            let pad = st.outer + 1.0;
            if dec_pts.len() >= 2 {
                for w_pts in dec_pts.windows(2) {
                    let (a, b) = (w_pts[0], w_pts[1]);
                    all_segs.push(Seg {
                        a,
                        b,
                        x0: a.0.min(b.0) - pad,
                        y0: a.1.min(b.1) - pad,
                        x1: a.0.max(b.0) + pad,
                        y1: a.1.max(b.1) + pad,
                        inner: st.inner,
                        outer: st.outer,
                        opacity: st.opacity,
                    });
                }
            } else if let Some(&a) = dec_pts.first() {
                all_segs.push(Seg {
                    a,
                    b: a,
                    x0: a.0 - pad,
                    y0: a.1 - pad,
                    x1: a.0 + pad,
                    y1: a.1 + pad,
                    inner: st.inner,
                    outer: st.outer,
                    opacity: st.opacity,
                });
            }
        }
        let mut row_segs: Vec<&Seg> = Vec::new();
        for y in 0..bh {
            let py = (by0 + y) as f32 + 0.5;
            row_segs.clear();
            let mut rx_min = f32::MAX;
            let mut rx_max = f32::MIN;
            for seg in &all_segs {
                if py >= seg.y0 && py <= seg.y1 {
                    rx_min = rx_min.min(seg.x0);
                    rx_max = rx_max.max(seg.x1);
                    row_segs.push(seg);
                }
            }
            let x_start = ((rx_min - bx0 as f32).floor().max(0.0) as usize).min(bw);
            let x_end = ((rx_max - bx0 as f32).ceil().max(0.0) as usize).min(bw);
            for x in 0..bw {
                let i = y * bw + x;
                let px_rgb = img.get(bx0 + x, by0 + y);
                src[i] = px_rgb;
                tgt[i] = px_rgb;
                if x >= x_start && x < x_end {
                    let px = (bx0 + x) as f32 + 0.5;
                    let mut a_max = 0.0f32;
                    for seg in &row_segs {
                        if px >= seg.x0 && px <= seg.x1 {
                            let d_sq = seg_dist_sq(px, py, seg.a, seg.b);
                            if d_sq < seg.outer * seg.outer {
                                let d = d_sq.sqrt();
                                let a = (1.0 - smooth(seg.inner, seg.outer, d)) * seg.opacity;
                                if a > a_max {
                                    a_max = a;
                                }
                            }
                        }
                    }
                    alpha[i] = a_max;
                }
            }
        }
        // Fast O(bw * bh) bounded squared Euclidean distance to the nearest `true` pixel in `mask`.
        let edt_sq_bounded = |mask: &[bool], max_r: usize| -> Vec<u32> {
            let inf = (max_r + 1) as u32;
            let inf_sq = inf * inf;
            let mut dx = vec![inf as u16; bw * bh];
            for y in 0..bh {
                let row_o = y * bw;
                let mut d = inf;
                for x in 0..bw {
                    d = if mask[row_o + x] { 0 } else { (d + 1).min(inf) };
                    dx[row_o + x] = d as u16;
                }
                d = inf;
                for x in (0..bw).rev() {
                    d = (d + 1).min(dx[row_o + x] as u32);
                    dx[row_o + x] = d as u16;
                }
            }
            let mut dy_active = vec![inf as u16; bw * bh];
            for y in 0..bh {
                let row_o = y * bw;
                for x in 0..bw {
                    let prev = if y > 0 { dy_active[(y - 1) * bw + x] as u32 + 1 } else { inf };
                    dy_active[row_o + x] = if (dx[row_o + x] as u32) <= max_r as u32 { 0 } else { prev.min(inf) as u16 };
                }
            }
            for y in (0..bh).rev() {
                let row_o = y * bw;
                for x in 0..bw {
                    let next = if y + 1 < bh { dy_active[(y + 1) * bw + x] as u32 + 1 } else { inf };
                    if next < dy_active[row_o + x] as u32 {
                        dy_active[row_o + x] = next as u16;
                    }
                }
            }
            let mut out = vec![inf_sq; bw * bh];
            for y in 0..bh {
                let row_o = y * bw;
                for x in 0..bw {
                    if dy_active[row_o + x] as usize > max_r {
                        continue;
                    }
                    let d0 = dx[row_o + x] as u32;
                    let mut best = (d0 * d0).min(inf_sq);
                    if best == 0 {
                        out[row_o + x] = 0;
                        continue;
                    }
                    for d in 1..=max_r {
                        let dy2 = (d * d) as u32;
                        if dy2 >= best {
                            break;
                        }
                        if y >= d {
                            let du = dx[(y - d) * bw + x] as u32;
                            best = best.min(dy2 + du * du);
                        }
                        if y + d < bh {
                            let dd = dx[(y + d) * bw + x] as u32;
                            best = best.min(dy2 + dd * dd);
                        }
                    }
                    out[row_o + x] = best;
                }
            }
            out
        };

        // For multi-point or multi-stroke AI Fill groups, seal any unbrushed pinholes/slits between passes
        // via morphological closing + enclosed-island flood-fill (including small pockets trapped against a
        // single outer photo border when erasing objects at the edge of the frame), then feather smoothly outward.
        if total_pts >= 3 || group.len() >= 2 {
            let mut core_mask: Vec<bool> = alpha.iter().map(|&a| a >= max_opacity * 0.50).collect();

            let border_snap = (r * 0.75).clamp(6.0, 28.0).round() as usize;
            if at_left_border {
                for y in 0..bh {
                    if let Some(mx) = (0..border_snap.min(bw)).rfind(|&x| core_mask[y * bw + x]) {
                        for x in 0..=mx {
                            core_mask[y * bw + x] = true;
                        }
                    }
                }
            }
            if at_right_border {
                let x_start = bw.saturating_sub(border_snap);
                for y in 0..bh {
                    if let Some(mx) = (x_start..bw).find(|&x| core_mask[y * bw + x]) {
                        for x in mx..bw {
                            core_mask[y * bw + x] = true;
                        }
                    }
                }
            }
            if at_top_border {
                for x in 0..bw {
                    if let Some(my) = (0..border_snap.min(bh)).rfind(|&y| core_mask[y * bw + x]) {
                        for y in 0..=my {
                            core_mask[y * bw + x] = true;
                        }
                    }
                }
            }
            if at_bottom_border {
                let y_start = bh.saturating_sub(border_snap);
                for x in 0..bw {
                    if let Some(my) = (y_start..bh).find(|&y| core_mask[y * bw + x]) {
                        for y in my..bh {
                            core_mask[y * bw + x] = true;
                        }
                    }
                }
            }

            let close_r = (r * 0.45).clamp(4.0, 16.0).round() as usize;
            let close_r2 = (close_r * close_r) as u32;
            let d2_core = edt_sq_bounded(&core_mask, close_r);
            let mut dilated: Vec<bool> = d2_core.iter().map(|&d2| d2 <= close_r2).collect();

            // Bridge small pockets trapped against a SINGLE outer photo border (e.g. concrete pillar at the right frame edge)
            // Only bridge gaps up to `2.2 * r` along that border so wide background expanses are never bridged!
            let max_border_gap = (r * 2.2).max(24.0) as usize;
            if at_left_border {
                let mut prev_y: Option<usize> = None;
                for y in 0..bh {
                    if dilated[y * bw] {
                        if let Some(py) = prev_y
                            && y > py + 1
                            && (y - py) <= max_border_gap
                        {
                            for yy in (py + 1)..y {
                                dilated[yy * bw] = true;
                            }
                        }
                        prev_y = Some(y);
                    }
                }
            }
            if at_right_border && bw >= 1 {
                let xr = bw - 1;
                let mut prev_y: Option<usize> = None;
                for y in 0..bh {
                    if dilated[y * bw + xr] {
                        if let Some(py) = prev_y
                            && y > py + 1
                            && (y - py) <= max_border_gap
                        {
                            for yy in (py + 1)..y {
                                dilated[yy * bw + xr] = true;
                            }
                        }
                        prev_y = Some(y);
                    }
                }
            }
            if at_top_border {
                let mut prev_x: Option<usize> = None;
                for x in 0..bw {
                    if dilated[x] {
                        if let Some(px) = prev_x
                            && x > px + 1
                            && (x - px) <= max_border_gap
                        {
                            for xx in (px + 1)..x {
                                dilated[xx] = true;
                            }
                        }
                        prev_x = Some(x);
                    }
                }
            }
            if at_bottom_border && bh >= 1 {
                let yb = bh - 1;
                let mut prev_x: Option<usize> = None;
                for x in 0..bw {
                    if dilated[yb * bw + x] {
                        if let Some(px) = prev_x
                            && x > px + 1
                            && (x - px) <= max_border_gap
                        {
                            for xx in (px + 1)..x {
                                dilated[yb * bw + xx] = true;
                            }
                        }
                        prev_x = Some(x);
                    }
                }
            }

            let mut filled = dilated.clone();
            let mut visited = dilated.clone();
            let max_pinhole_area = ((r * r * 2.2).max(800.0) as usize).min((bw * bh) / 16);
            let mut comp = Vec::new();
            let mut queue = std::collections::VecDeque::new();
            for sy in 0..bh {
                for sx in 0..bw {
                    let si = sy * bw + sx;
                    if visited[si] {
                        continue;
                    }
                    visited[si] = true;
                    comp.clear();
                    comp.push(si);
                    queue.push_back((sx, sy));
                    let mut touches_open_crop_edge = false;
                    let mut photo_borders_mask = 0u8;
                    let mut border_px_count = 0usize;
                    while let Some((cx, cy)) = queue.pop_front() {
                        if (cx == 0 && !at_left_border)
                            || (cx + 1 == bw && !at_right_border)
                            || (cy == 0 && !at_top_border)
                            || (cy + 1 == bh && !at_bottom_border)
                        {
                            touches_open_crop_edge = true;
                        }
                        if cx == 0 && at_left_border {
                            photo_borders_mask |= 1;
                            border_px_count += 1;
                        }
                        if cx + 1 == bw && at_right_border {
                            photo_borders_mask |= 2;
                            border_px_count += 1;
                        }
                        if cy == 0 && at_top_border {
                            photo_borders_mask |= 4;
                            border_px_count += 1;
                        }
                        if cy + 1 == bh && at_bottom_border {
                            photo_borders_mask |= 8;
                            border_px_count += 1;
                        }
                        for (nx, ny) in [
                            (cx.saturating_sub(1), cy),
                            ((cx + 1).min(bw - 1), cy),
                            (cx, cy.saturating_sub(1)),
                            (cx, (cy + 1).min(bh - 1)),
                        ] {
                            let ni = ny * bw + nx;
                            if !visited[ni] {
                                visited[ni] = true;
                                comp.push(ni);
                                queue.push_back((nx, ny));
                            }
                        }
                    }
                    // Never flood-fill a corner of the photo (`photo_borders_mask.count_ones() >= 2`) or a wide border span!
                    let valid_pinhole = if touches_open_crop_edge || photo_borders_mask.count_ones() >= 2 {
                        false
                    } else if photo_borders_mask.count_ones() == 1 {
                        border_px_count <= max_border_gap && comp.len() <= max_pinhole_area / 2
                    } else {
                        comp.len() <= max_pinhole_area
                    };
                    if valid_pinhole {
                        for &idx in &comp {
                            filled[idx] = true;
                        }
                    }
                }
            }
            let erode_r = close_r.saturating_sub(1).max(1);
            let erode_r2 = (erode_r * erode_r) as u32;
            let not_filled: Vec<bool> = filled.iter().map(|&b| !b).collect();
            let d2_not_filled = edt_sq_bounded(&not_filled, erode_r);
            let mut closed: Vec<bool> = (0..(bw * bh))
                .map(|i| core_mask[i] || (filled[i] && d2_not_filled[i] > erode_r2))
                .collect();
            for i in 0..(bw * bh) {
                if core_mask[i] {
                    closed[i] = true;
                }
            }

            // Smart Object-Silhouette Tightening for large brushes (`r >= 26.0`) over dark/neutral objects
            // (e.g. studio flash umbrellas, booms, or stands in outdoor foliage/sky): runs after pinhole
            // closing so multi-pass scribbles have a unified spine, then flood-fills connected object pixels
            // and trims away excess brush overhang into surrounding sky, foliage, or adjacent tree trunks.
            if !lightcraft_inpaint::current_cloud_mode_tag().0 && r >= 26.0 {
                let not_closed: Vec<bool> = closed.iter().map(|&b| !b).collect();
                let spine_depth = (r * 0.48).round().max(8.0) as usize;
                let outer_depth = (r * 0.36).round().max(5.0) as usize;
                let d2_edge = edt_sq_bounded(&not_closed, spine_depth);
                let spine_d2 = (spine_depth * spine_depth) as u32;
                let outer_d2 = (outer_depth * outer_depth) as u32;
                let is_sky_or_fol = |rgb: [f32; 3]| -> bool {
                    (rgb[2] > rgb[0] * 1.18 && rgb[2] > 0.08)
                        || (rgb[1] > rgb[0] * 1.12 && rgb[1] > 0.045)
                };
                let is_dark_obj = |rgb: [f32; 3]| -> bool {
                    rgb[0].max(rgb[1]).max(rgb[2]) < 0.068 && (rgb[1] - rgb[0]) <= 0.015
                };
                let is_warm_skin = |rgb: [f32; 3]| -> bool {
                    rgb[0] > rgb[1] * 1.15 && (rgb[0] - rgb[1]) > 0.018 && rgb[0] > 0.08
                };
                let mut n_spine = 0usize;
                let mut n_spine_dark = 0usize;
                let mut n_spine_obj = 0usize;
                let mut n_spine_skin = 0usize;
                let mut n_outer = 0usize;
                let mut n_outer_bg = 0usize;
                for i in 0..(bw * bh) {
                    if closed[i] {
                        let c = tgt[i];
                        if d2_edge[i] >= spine_d2 {
                            n_spine += 1;
                            if is_dark_obj(c) {
                                n_spine_dark += 1;
                            }
                            if !is_sky_or_fol(c) {
                                n_spine_obj += 1;
                            }
                            if is_warm_skin(c) {
                                n_spine_skin += 1;
                            }
                        } else if d2_edge[i] <= outer_d2 {
                            n_outer += 1;
                            if is_sky_or_fol(c) {
                                n_outer_bg += 1;
                            }
                        }
                    }
                }
                if n_spine >= 32
                    && n_outer >= 32
                    && n_spine_dark * 100 >= n_spine * 32
                    && n_spine_obj * 100 >= n_spine * 62
                    && n_spine_skin * 100 <= n_spine * 10
                    && n_outer_bg * 100 >= n_outer * 38
                {
                    let seed_depth = (r * 0.40).round().max(6.0) as usize;
                    let seed_d2 = (seed_depth * seed_depth) as u32;
                    let mut obj_vis = vec![false; bw * bh];
                    let mut q = std::collections::VecDeque::new();
                    for y in 0..bh {
                        for x in 0..bw {
                            let i = y * bw + x;
                            if closed[i] && d2_edge[i] >= seed_d2 && !is_sky_or_fol(tgt[i]) {
                                obj_vis[i] = true;
                                q.push_back((x, y));
                            }
                        }
                    }
                    while let Some((cx, cy)) = q.pop_front() {
                        for (nx, ny) in [
                            (cx.saturating_sub(1), cy),
                            ((cx + 1).min(bw - 1), cy),
                            (cx, cy.saturating_sub(1)),
                            (cx, (cy + 1).min(bh - 1)),
                        ] {
                            let ni = ny * bw + nx;
                            if !obj_vis[ni] && closed[ni] && !is_sky_or_fol(tgt[ni]) {
                                obj_vis[ni] = true;
                                q.push_back((nx, ny));
                            }
                        }
                    }
                    let tight_r = (r * 0.14).clamp(7.0, 15.0).round() as usize;
                    let tight_r2 = (tight_r * tight_r) as u32;
                    let d2_obj = edt_sq_bounded(&obj_vis, tight_r);
                    let mut tight_mask: Vec<bool> = (0..(bw * bh))
                        .map(|i| closed[i] && d2_obj[i] <= tight_r2)
                        .collect();
                    // Never punch interior holes inside the tightened object silhouette: only trim
                    // overhang pixels that are connected to the exterior (`!closed`).
                    let mut ext_vis = vec![false; bw * bh];
                    let mut eq = std::collections::VecDeque::new();
                    for y in 0..bh {
                        for x in 0..bw {
                            let i = y * bw + x;
                            if !closed[i] {
                                ext_vis[i] = true;
                                eq.push_back((x, y));
                            }
                        }
                    }
                    while let Some((cx, cy)) = eq.pop_front() {
                        for (nx, ny) in [
                            (cx.saturating_sub(1), cy),
                            ((cx + 1).min(bw - 1), cy),
                            (cx, cy.saturating_sub(1)),
                            (cx, (cy + 1).min(bh - 1)),
                        ] {
                            let ni = ny * bw + nx;
                            if !ext_vis[ni] && !tight_mask[ni] {
                                ext_vis[ni] = true;
                                eq.push_back((nx, ny));
                            }
                        }
                    }
                    for i in 0..(bw * bh) {
                        if !ext_vis[i] {
                            tight_mask[i] = true;
                        }
                    }
                    if tight_mask.iter().filter(|&&b| b).count() >= n_spine / 2 {
                        closed = std::mem::take(&mut tight_mask);
                        for i in 0..(bw * bh) {
                            if !closed[i] {
                                alpha[i] = 0.0;
                            }
                        }
                    }
                }
            }

            let seam_r = (max_seam.ceil() as usize).clamp(4, 16);
            let seam_r2 = (seam_r * seam_r) as u32;
            let seam_f = seam_r as f32;
            let d2_closed = edt_sq_bounded(&closed, seam_r);
            for i in 0..(bw * bh) {
                if closed[i] {
                    alpha[i] = max_opacity;
                } else if d2_closed[i] <= seam_r2 {
                    let d_out = (d2_closed[i] as f32).sqrt();
                    let closed_alpha = (1.0 - smooth(0.0, seam_f, d_out)) * max_opacity;
                    alpha[i] = alpha[i].max(closed_alpha);
                }
            }
        }
        let is_thumb_pass = photo_long <= 520.0;
        fill_box(&mut src, &tgt, &alpha, bw, bh, group_seed, is_thumb_pass);
        for y in 0..bh {
            for x in 0..bw {
                let i = y * bw + x;
                let a = alpha[i];
                if a <= 0.0 {
                    continue;
                }
                let v = src[i];
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

    let mut p_state = DefaultHasher::new();
    0x4F55_5450_5345_4544u64.hash(&mut p_state);
    top_pct.to_bits().hash(&mut p_state);
    bot_pct.to_bits().hash(&mut p_state);
    left_pct.to_bits().hash(&mut p_state);
    right_pct.to_bits().hash(&mut p_state);
    fill_corners.hash(&mut p_state);
    let param_seed = (p_state.finish() ^ 0x4F55_5450_2027).max(2);

    // Hash parameters + sparse image content fingerprint
    let mut h_state = DefaultHasher::new();
    w.hash(&mut h_state);
    h.hash(&mut h_state);
    param_seed.hash(&mut h_state);
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
        seed: param_seed,
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
    fn fill_zoom_in_and_out_reuses_inpaint_cache_across_resolutions() {
        let spot = Spot {
            mode: SpotMode::Fill,
            points: vec![Point::new(0.48, 0.52), Point::new(0.54, 0.52)],
            size: 0.09,
            feather: 0.25,
            opacity: 1.0,
            ..Default::default()
        };
        let mut preview = striped_with_blot(320, 240, 16);
        let fp = Frame::new(320, 240, &DevelopSettings::default(), true);
        apply(&mut preview, std::slice::from_ref(&spot), &fp, fp.px_per_long(320));

        // Zoom in to 2x resolution (640x480): should hit INPAINT_WORK_CACHE and match preview structure
        let mut zoomed = striped_with_blot(640, 480, 32);
        let fz = Frame::new(640, 480, &DevelopSettings::default(), true);
        let t0 = std::time::Instant::now();
        apply(&mut zoomed, std::slice::from_ref(&spot), &fz, fz.px_per_long(640));
        let elapsed_zoom = t0.elapsed();
        assert!(
            elapsed_zoom.as_millis() < 80,
            "zoomed inpaint should reuse INPAINT_WORK_CACHE quickly, took {:?}",
            elapsed_zoom
        );
        let (mut err, mut n) = (0.0f32, 0);
        for y in 110..135 {
            for x in 145..175 {
                let z_avg: f32 = [(0, 0), (1, 0), (0, 1), (1, 1)]
                    .iter()
                    .map(|(dx, dy)| zoomed.get(2 * x + dx, 2 * y + dy)[0])
                    .sum::<f32>()
                    / 4.0;
                err += (z_avg - preview.get(x, y)[0]).abs();
                n += 1;
            }
        }
        assert!(err / n as f32 <= 0.06, "zoom vs preview mean diff {}", err / n as f32);
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
