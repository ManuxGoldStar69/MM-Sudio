//! Hole filling for Spot Healing: patch-based completion, proximity match and texture synthesis.
//!
//! * [`complete`], content-aware fill: the multi-scale "space-time completion" EM scheme of
//!   Y. Wexler, E. Shechtman, M. Irani, *Space-Time Completion of Video*, IEEE TPAMI 2007, with the
//!   nearest-neighbour field computed by C. Barnes, E. Shechtman, A. Finkelstein, D. B. Goldman,
//!   *PatchMatch: A Randomized Correspondence Algorithm for Structural Image Editing*, SIGGRAPH 2009
//!   (random init, propagation, exponentially shrinking random search). Each EM iteration finds, for
//!   every patch overlapping the hole, its most similar fully-known patch, then re-estimates hole
//!   pixels by a similarity-weighted vote of the overlapping matches. Coarse-to-fine over an image
//!   pyramid (the coarsest level is initialised by membrane interpolation), so large holes stay cheap.
//! * [`best_offset`], proximity match: the single displacement whose surrounding ring best matches the
//!   ring around the hole (sum of squared differences), i.e. an automatic Healing Brush source.
//! * [`synthesize`], create texture: patch-based texture synthesis in the style of A. Efros,
//!   W. Freeman, *Image Quilting for Texture Synthesis and Transfer*, SIGGRAPH 2001 (blocks picked from
//!   random candidates by overlap error, feathered overlap instead of a min-cut seam).
//!
//! Ported from PhotoCraft's `algo::inpaint` (MIT OR Apache-2.0); its proximity match and texture
//! synthesis are left out.
//!
//! All functions take interleaved `w × h × ch` normalised floats (any colour model/depth) and a
//! `w × h` hole mask, and return a full buffer equal to the input outside the hole. They are
//! deterministic (fixed-seed hash RNG, results independent of thread count).

use crate::Cancelled;
use crate::membrane::membrane_fill;

/// Below this many grid cells a level runs single-threaded (thread hand-off costs more than it saves).
#[cfg(not(target_arch = "wasm32"))]
const PAR_MIN: usize = 128 * 128;

/// SplitMix64: tiny deterministic RNG.
#[derive(Clone)]
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform integer in `lo..=hi`.
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        if hi <= lo {
            return lo;
        }
        lo + (self.next() % (hi - lo + 1) as u64) as i32
    }
}

/// Parameters for [`complete`].
#[derive(Clone, Debug, PartialEq)]
pub struct CompleteParams {
    /// Patch radius (patches are `2r+1` square).
    pub patch_radius: usize,
    /// EM iterations at the coarsest level (finer levels use fewer, down to `min_em_iters`).
    pub em_iters: usize,
    pub min_em_iters: usize,
    /// PatchMatch passes per EM iteration.
    pub pm_iters: usize,
    /// Levels where the hole is wider than this (in that level's pixels) only *refine* the upsampled
    /// match field (one local PatchMatch pass and one vote), which keeps big holes tractable: the
    /// structure is decided at coarse scale, fine levels just add detail.
    pub refine_extent: usize,
    pub seed: u64,
}

impl Default for CompleteParams {
    fn default() -> Self {
        Self { patch_radius: 3, em_iters: 8, min_em_iters: 2, pm_iters: 2, refine_extent: 96, seed: 1 }
    }
}

/// One pyramid level.
#[derive(Clone)]
struct Level {
    w: usize,
    h: usize,
    img: Vec<f32>,
    hole: Vec<bool>,
}

fn downsample(l: &Level, ch: usize) -> Level {
    let (w, h) = (l.w.div_ceil(2), l.h.div_ceil(2));
    let mut img = vec![0.0f32; w * h * ch];
    let mut hole = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut n = 0.0f32;
            let o = (y * w + x) * ch;
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let (sx, sy) = (x * 2 + dx, y * 2 + dy);
                if sx >= l.w || sy >= l.h {
                    continue;
                }
                let si = sy * l.w + sx;
                if l.hole[si] {
                    hole[y * w + x] = true;
                } else {
                    for c in 0..ch {
                        img[o + c] += l.img[si * ch + c];
                    }
                    n += 1.0;
                }
            }
            if n > 0.0 {
                for c in 0..ch {
                    img[o + c] /= n;
                }
            }
        }
    }
    Level { w, h, img, hole }
}

/// Summed-area table of a boolean grid (`(w+1) × (h+1)`).
fn integral(w: usize, h: usize, m: &[bool]) -> Vec<u32> {
    let mut s = vec![0u32; (w + 1) * (h + 1)];
    for y in 0..h {
        let mut row = 0u32;
        for x in 0..w {
            row += u32::from(m[y * w + x]);
            s[(y + 1) * (w + 1) + x + 1] = s[y * (w + 1) + x + 1] + row;
        }
    }
    s
}

/// Count of set cells in the clipped window `[x0, x1) × [y0, y1)`.
fn window_count(s: &[u32], gw: usize, gh: usize, x0: i32, y0: i32, x1: i32, y1: i32) -> u32 {
    let (x0, y0) = (x0.max(0) as usize, y0.max(0) as usize);
    let (x1, y1) = ((x1.max(0) as usize).min(gw), (y1.max(0) as usize).min(gh));
    if x1 <= x0 || y1 <= y0 {
        return 0;
    }
    let w1 = gw + 1;
    s[y1 * w1 + x1] + s[y0 * w1 + x0] - s[y0 * w1 + x1] - s[y1 * w1 + x0]
}

fn hole_bbox(w: usize, h: usize, hole: &[bool]) -> Option<(usize, usize, usize, usize)> {
    let mut b: Option<(usize, usize, usize, usize)> = None;
    for y in 0..h {
        for x in 0..w {
            if hole[y * w + x] {
                b = Some(match b {
                    None => (x, y, x + 1, y + 1),
                    Some((x0, y0, x1, y1)) => (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1)),
                });
            }
        }
    }
    b
}

/// Per-level PatchMatch/EM state.
struct Solver<'a> {
    w: usize,
    h: usize,
    ch: usize,
    r: i32,
    horizon_weight: f32,
    guide_weight: f32,
    is_outpaint: bool,
    valid_bbox: (i32, i32, i32, i32),
    guide_low: Option<&'a [f32]>,
    img_low: Option<&'a [f32]>,
    img_var: Option<&'a [f32]>,
    row_var: Option<&'a [f32]>,
    row_has_valid: Option<&'a [bool]>,
    img: &'a mut Vec<f32>,
    hole: &'a [bool],
    /// Valid source centres: the whole patch is inside the grid and contains no hole pixel.
    valid: Vec<bool>,
    valid_list: Vec<(i32, i32)>,
    /// Target centres (patch overlaps the hole), row-major.
    targets: Vec<bool>,
    /// Cancellation: checked per row band; a cancelled solve's output is discarded.
    ctl: &'a (dyn Fn() -> bool + Sync),
}

impl Solver<'_> {
    #[inline]
    fn is_valid(&self, x: i32, y: i32) -> bool {
        x >= 0 && y >= 0 && (x as usize) < self.w && (y as usize) < self.h && self.valid[y as usize * self.w + x as usize]
    }

    #[inline]
    fn is_bottom_white_clothing(&self, bx: i32) -> bool {
        if self.ch == 3
            && let Some(i_low) = self.img_low
        {
            let (vx0, vx1, _, vy1) = self.valid_bbox;
            let mut g_sum = 0.0f32;
            let mut b_sum = 0.0f32;
            for dx in -3i32..=3 {
                let sx = (bx + dx).clamp(vx0, vx1) as usize;
                let bc = (vy1 as usize * self.w + sx) * 3;
                g_sum += i_low[bc + 1];
                b_sum += i_low[bc + 2];
            }
            // White/cream/grey clothing (both sunlit and blue-sky-shaded!) has strong blue reflectance
            // (`b > 0.075`) and neutral green-to-blue ratio (`g <= b * 1.25`), whereas both green grass
            // and golden-olive beach grass strongly absorb blue light (`b < 0.05` and `g > b * 1.35`).
            b_sum > 0.075 * 7.0 && g_sum <= b_sum * 1.25
        } else {
            false
        }
    }

    #[inline]
    fn outpaint_anchor(&self, t: (i32, i32)) -> (i32, i32) {
        let (vx0, vx1, vy0, vy1) = self.valid_bbox;
        // Full span from `(vx0, vx1, vy0, vy1)` to the outer canvas edges `(0, w - 1, 0, h - 1)`
        // WITHOUT subtracting `self.r`, so `t.0 ± span` and `t.1 ± span` never clamp to a constant
        // value along the outer `r` pixels of any pyramid level!
        let left_w = vx0.max(4);
        let right_w = (self.w as i32 - 1 - vx1).max(4);
        let top_h = vy0.max(4);
        let bot_h = (self.h as i32 - 1 - vy1).max(4);
        let want_x = if t.0 < vx0 {
            (t.0 + left_w).clamp(vx0, vx1)
        } else if t.0 > vx1 {
            (t.0 - right_w).clamp(vx0, vx1)
        } else {
            t.0
        };
        let mut want_y = if t.1 < vy0 {
            (t.1 + top_h).clamp(vy0, vy1)
        } else if t.1 > vy1 {
            (t.1 - bot_h).clamp(vy0, vy1)
        } else {
            t.1
        };
        if t.1 > vy1 && self.is_bottom_white_clothing(want_x.clamp(vx0, vx1)) {
            let frac = ((t.1 - vy1) as f32 / (bot_h as f32)).clamp(0.0, 1.0);
            let clean_h = (bot_h * 3 / 8).max(4);
            let y_start = (vy1 - clean_h).clamp(vy0, vy1);
            let y_end = (vy1 - 1).max(y_start);
            want_y = (y_start as f32 + frac * (y_end - y_start) as f32).round() as i32;
            want_y = want_y.clamp(vy0, vy1);
        }
        (want_x, want_y)
    }

    /// SSD between the (clipped) target patch at `t` and the source patch at `s`, per sample,
    /// plus variance-preserving DC mean guide anchor (`guide_low` vs `img_low`) and texture richness matching.
    fn dist(&self, t: (i32, i32), s: (i32, i32), cutoff: f32) -> f32 {
        let (w, h, ch, r) = (self.w as i32, self.h as i32, self.ch, self.r);
        let outpaint_bnd = if self.is_outpaint {
            let (vx0, vx1, vy0, vy1) = self.valid_bbox;
            Some((t.0.clamp(vx0, vx1), t.1.clamp(vy0, vy1)))
        } else {
            None
        };
        let mut spat_pen = if self.horizon_weight > 0.0 {
            if outpaint_bnd.is_some() {
                let (vx0, vx1, vy0, vy1) = self.valid_bbox;
                let (wx, wy) = self.outpaint_anchor(t);
                let left_w = vx0.max(4);
                let right_w = (w - 1 - vx1).max(4);
                let top_h = vy0.max(4);
                let bot_h = (h - 1 - vy1).max(4);
                let tol_x = (left_w.max(right_w) / 3).clamp(3, 12);
                let tol_y = (top_h.max(bot_h) / 3).clamp(3, 10);
                if (t.0 < vx0 || t.0 > vx1) && (s.0 - wx).abs() > tol_x {
                    return f32::INFINITY;
                }
                if (t.1 < vy0 || t.1 > vy1) && (s.1 - wy).abs() > tol_y {
                    return f32::INFINITY;
                }
                let dy_a = (s.1 - wy) as f32 / (h.max(1) as f32);
                let dx_a = (s.0 - wx) as f32 / (w.max(1) as f32);
                let mut pen = (dx_a * dx_a + dy_a * dy_a) * 6.0;
                if t.1 >= vy0 && t.1 <= vy1 {
                    // Left/right border (interior row): lock horizon elevation `s.1 ≈ t.1`
                    let dy = (t.1 - s.1) as f32 / (h.max(1) as f32);
                    pen += dy * dy * 28.0;
                    if dy.abs() > 0.012 {
                        let ex = dy.abs() - 0.012;
                        pen += ex * ex * 140.0;
                    }
                }
                if t.0 >= vx0 && t.0 <= vx1 {
                    // Top/bottom border (interior column): lock horizontal column `s.0 ≈ t.0`
                    let dx = (t.0 - s.0) as f32 / (w.max(1) as f32);
                    if dx.abs() > 0.022 {
                        return f32::INFINITY;
                    }
                    pen += dx * dx * 45.0;
                }
                pen
            } else {
                let dy = (t.1 - s.1) as f32 / (h.max(1) as f32);
                let dx = (t.0 - s.0) as f32 / (w.max(1) as f32);
                let has_row_v = self.row_has_valid.is_none_or(|rv| rv[t.1 as usize]);
                let row_smooth = self
                    .row_var
                    .map_or(1.0, |rv| ((0.042 - rv[t.1 as usize]) / 0.018).clamp(0.0, 1.0));
                if has_row_v {
                    let hz_w = self.horizon_weight * (0.28 + 0.72 * row_smooth);
                    let mut pen = dy * dy * hz_w + dx * dx * 0.055;
                    if dy.abs() > 0.12 {
                        let ex = dy.abs() - 0.12;
                        pen += ex * ex * (1.2 + 2.6 * row_smooth);
                    }
                    pen
                } else {
                    let mut pen = dy * dy * 0.45 + dx * dx * 1.8;
                    if dx.abs() > 0.06 {
                        let ex = dx.abs() - 0.06;
                        pen += ex * ex * 45.0;
                    }
                    pen
                }
            }
        } else {
            0.0
        };
        let t_idx = t.1 as usize * self.w + t.0 as usize;
        let s_idx = s.1 as usize * self.w + s.0 as usize;
        let mut t_sky_score = 0.0f32;
        if let (Some(g_low), Some(i_low)) = (self.guide_low, self.img_low)
            && ch == 3
        {
            let tc = t_idx * 3;
            let sc = s_idx * 3;
            let tr = g_low[tc].max(1e-4);
            let tg = g_low[tc + 1].max(1e-4);
            let tb = g_low[tc + 2];
            let sr = i_low[sc].max(1e-4);
            let sg = i_low[sc + 1].max(1e-4);
            let sb = i_low[sc + 2];
            // Continuous sky confidence in [0, 1] — attenuated when the row's unmasked patches are
            // high-variance dappled tree canopy (`r_var > 0.025`) so canopy holes are filled with
            // textured branches rather than flat smooth sky!
            let row_smooth_sky = self
                .row_var
                .map_or(1.0, |rv| ((0.042 - rv[t.1 as usize]) / 0.018).clamp(0.0, 1.0));
            t_sky_score = ((((tb / tr) - 1.10) * 6.0)
                .min(((tb / tg) - 1.02) * 12.0)
                .min((tg - 0.05) * 16.0)
                .clamp(0.0, 1.0))
                * row_smooth_sky;
            let s_sky_score = ((((sb / sr) - 1.10) * 6.0)
                .min(((sb / sg) - 1.02) * 12.0)
                .min((sg - 0.05) * 16.0)
                .clamp(0.0, 1.0))
                * row_smooth_sky;
            let d_sky = (t_sky_score - s_sky_score).abs();
            // Strongly penalize sky-vs-ground mismatches while leaving mixed palm-crown transitions (`t_sky_score ≈ 0.5`) free!
            if d_sky > 0.35 {
                spat_pen += (d_sky - 0.35) * (d_sky - 0.35) * 3.5;
            }
            // When target guide is clearly green foliage (`tg > tr * 1.02 && tg > tb * 1.08`), penalize
            // red-dominant sources (`sr > sg * 1.06`, e.g. terracotta roofs or skin) without penalizing warm sunlit olive fronds!
            if tg > tr * 1.02 && tg > tb * 1.08 && sr > sg * 1.06 {
                let drg = (sr - sg * 1.06) / sg.max(0.05);
                spat_pen += drg * drg * 2.5;
            }
            // Compare DC patch means (`g_low` vs `i_low`) OUTSIDE the pixel loop so internal
            // leaf/frond/wildflower contrast is NEVER penalized!
            let dr = g_low[tc] - i_low[sc];
            let dg = g_low[tc + 1] - i_low[sc + 1];
            let db = g_low[tc + 2] - i_low[sc + 2];
            spat_pen += (dr * dr + dg * dg * 1.4 + db * db) * self.guide_weight;
            if let Some((bx, by)) = outpaint_bnd {
                let ac = (by as usize * self.w + bx as usize) * 3;
                let ar = i_low[ac];
                let ag = i_low[ac + 1];
                let ab = i_low[ac + 2];
                // Never copy red/brown skin (e.g. forearm/hand) into a non-skin column (white dress, shirt, grass, sky)
                if ar <= ag * 1.08 && sr > sg * 1.12 && (sr - sg) > 0.018 {
                    return f32::INFINITY;
                }
                let adr = ar - sr;
                let adg = ag - sg;
                let adb = ab - sb;
                spat_pen += (adr * adr + adg * adg * 1.4 + adb * adb) * 3.5;
            }
        }
        if let (Some(i_var), Some(r_var)) = (self.img_var, self.row_var) {
            let cand_v = i_var[s_idx];
            if let Some((bx, by)) = outpaint_bnd {
                let want_v = i_var[by as usize * self.w + bx as usize];
                let dv = cand_v - want_v;
                if dv > 0.0 {
                    // Never copy a sharp high-contrast silhouette edge onto a smooth shirt, dress, or sky!
                    spat_pen += dv * dv * 14.0;
                } else {
                    spat_pen += dv * dv * 3.0;
                }
            } else if t_sky_score > 0.65 {
                // Clear sky target pixels must stay silky-smooth
                spat_pen += cand_v * cand_v * 3.5 * t_sky_score;
            } else {
                let non_sky_w = (1.0 - t_sky_score * 3.5).clamp(0.0, 1.0);
                let want_v = r_var[t.1 as usize];
                // Compensate for the Pythagoras bias-variance penalty (`+cand_v^2`) against sharp patches
                // when `self.img` inside the hole is smoother than unmasked texture, while turning off smoothly
                // at sky transitions (`t_sky_score >= 0.28`) so sky-to-crown contours never attract white edge static!
                let min_v = want_v * 0.85;
                if non_sky_w > 0.0 && cand_v < min_v {
                    let dv = min_v - cand_v;
                    spat_pen += dv * dv * 2.2 * non_sky_w;
                }
            }
        }
        if spat_pen >= cutoff {
            return f32::INFINITY;
        }
        let rem_cutoff = cutoff - spat_pen;
        let full = ((2 * r + 1) * (2 * r + 1)) as f32 * ch as f32;
        let mut sum = 0.0f32;
        let mut n = 0usize;
        for dy in -r..=r {
            let ty = t.1 + dy;
            if ty < 0 || ty >= h {
                continue;
            }
            let sy = s.1 + dy;
            for dx in -r..=r {
                let tx = t.0 + dx;
                if tx < 0 || tx >= w {
                    continue;
                }
                let ti = (ty as usize * self.w + tx as usize) * ch;
                let si = (sy as usize * self.w + (s.0 + dx) as usize) * ch;
                for c in 0..ch {
                    let d = self.img[ti + c] - self.img[si + c];
                    sum += d * d;
                }
                n += ch;
            }
            // Early out: the mean can only be ≥ sum / (full patch size).
            if sum > rem_cutoff * full {
                return f32::INFINITY;
            }
        }
        if n == 0 { f32::INFINITY } else { (sum / n as f32) + spat_pen }
    }

    /// One PatchMatch pass over all target rows. When `horizon_weight > 0.0`, runs a single unbroken
    /// vertical+horizontal propagation sweep across the full image height so tall structures (palm trees,
    /// trunks, buildings, grass fields) never suffer 16-row horizontal band seams.
    fn patchmatch(&self, nnf: &mut [(i32, i32)], cost: &mut [f32], pass: usize, seed: u64, search_radius: i32) {
        const BAND: usize = 16;
        let w = self.w;
        let reverse = pass % 2 == 1;
        let band_size = if self.horizon_weight > 0.0 { self.h.max(1) } else { BAND };
        let band_fn = |bi: usize, nn: &mut [(i32, i32)], co: &mut [f32]| {
            if (self.ctl)() {
                return;
            }
            let mut rng = Rng::new(seed ^ (bi as u64).wrapping_mul(0x2545_F491_4F6C_DD1D) ^ ((pass as u64) << 40));
            let y0 = bi * band_size;
            let rows = nn.len() / w;
            let step: i32 = if reverse { -1 } else { 1 };
            for ry in 0..rows {
                let ly = if reverse { rows - 1 - ry } else { ry };
                let y = y0 + ly;
                for rx in 0..w {
                    let x = if reverse { w - 1 - rx } else { rx };
                    let li = ly * w + x;
                    if !self.targets[y * w + x] {
                        continue;
                    }
                    let t = (x as i32, y as i32);
                    let (mut best, mut bc) = (nn[li], co[li]);
                    let nx = x as i32 - step;
                    let nly = ly as i32 - step;
                    let pred_x = if self.horizon_weight > 0.0 && nx >= 0 && (nx as usize) < w {
                        let c = nn[ly * w + nx as usize];
                        Some((c.0 + step, c.1))
                    } else {
                        None
                    };
                    let pred_y = if self.horizon_weight > 0.0 && nly >= 0 && (nly as usize) < rows {
                        let c = nn[nly as usize * w + x];
                        Some((c.0, c.1 + step))
                    } else {
                        None
                    };
                    let seam_pen = |cand: (i32, i32)| -> f32 {
                        let mut p = 0.0f32;
                        if let Some(px) = pred_x {
                            let d2 = (cand.0 - px.0) * (cand.0 - px.0) + (cand.1 - px.1) * (cand.1 - px.1);
                            p += 0.0055 * (d2 as f32 * 0.25).min(1.0);
                        }
                        if let Some(py) = pred_y {
                            let d2 = (cand.0 - py.0) * (cand.0 - py.0) + (cand.1 - py.1) * (cand.1 - py.1);
                            p += 0.0045 * (d2 as f32 * 0.25).min(1.0);
                        }
                        p
                    };
                    let mut total_bc = bc + seam_pen(best);
                    // Propagation from the already-visited neighbours.
                    if nx >= 0 && (nx as usize) < w {
                        let c = nn[ly * w + nx as usize];
                        let cand = (c.0 + step, c.1);
                        if self.is_valid(cand.0, cand.1) && cand != best {
                            let pen = seam_pen(cand);
                            let d = self.dist(t, cand, total_bc - pen);
                            if d + pen < total_bc {
                                (best, bc, total_bc) = (cand, d, d + pen);
                            }
                        } else if self.horizon_weight > 0.0 {
                            let wrap_cand = if self.is_outpaint {
                                self.outpaint_anchor(t)
                            } else {
                                let span = (w / 8).clamp(14, 48);
                                let h_col = ((c.0 as u64).wrapping_mul(0x9E37_79B9) ^ ((pass as u64).wrapping_mul(0xBF58_476D))) as usize;
                                let jump = self.r * 4 + 8 + (h_col % span) as i32;
                                (c.0 - step * jump, c.1)
                            };
                            if self.is_valid(wrap_cand.0, wrap_cand.1) && wrap_cand != best {
                                let pen = seam_pen(wrap_cand);
                                let d = self.dist(t, wrap_cand, total_bc - pen);
                                if d + pen < total_bc {
                                    (best, bc, total_bc) = (wrap_cand, d, d + pen);
                                }
                            }
                        }
                    }
                    if nly >= 0 && (nly as usize) < rows {
                        let c = nn[nly as usize * w + x];
                        let cand = (c.0, c.1 + step);
                        if self.is_valid(cand.0, cand.1) && cand != best {
                            let pen = seam_pen(cand);
                            let d = self.dist(t, cand, total_bc - pen);
                            if d + pen < total_bc {
                                (best, bc, total_bc) = (cand, d, d + pen);
                            }
                        } else if self.horizon_weight > 0.0 {
                            let wrap_cand = if self.is_outpaint {
                                self.outpaint_anchor(t)
                            } else {
                                let span = (rows / 7).clamp(14, 42);
                                let h_row = ((c.1 as u64).wrapping_mul(0x9E37_79B9) ^ ((pass as u64).wrapping_mul(0xBF58_476D))) as usize;
                                let jump = self.r * 3 + 6 + (h_row % span) as i32;
                                (c.0, c.1 - step * jump)
                            };
                            if self.is_valid(wrap_cand.0, wrap_cand.1) && wrap_cand != best {
                                let pen = seam_pen(wrap_cand);
                                let d = self.dist(t, wrap_cand, total_bc - pen);
                                if d + pen < total_bc {
                                    (best, bc, total_bc) = (wrap_cand, d, d + pen);
                                }
                            }
                        }
                    }
                    // Diagonal propagation (`(nx, nly)`) for coherent 2D foliage/grass patch continuity
                    if self.horizon_weight > 0.0 && nx >= 0 && (nx as usize) < w && nly >= 0 && (nly as usize) < rows {
                        let c = nn[nly as usize * w + nx as usize];
                        let cand = (c.0 + step, c.1 + step);
                        if self.is_valid(cand.0, cand.1) && cand != best {
                            let pen = seam_pen(cand);
                            let d = self.dist(t, cand, total_bc - pen);
                            if d + pen < total_bc {
                                (best, bc, total_bc) = (cand, d, d + pen);
                            }
                        }
                    }
                    // Random search in exponentially shrinking windows.
                    let has_row_v = self.row_has_valid.is_none_or(|rv| rv[y]);
                    let mut rad = search_radius;
                    while rad >= 1 {
                        let rx = if self.is_outpaint && !has_row_v { rad.min(14) } else { rad };
                        let ry = if self.is_outpaint && has_row_v { rad.min(2) } else { rad };
                        let cand = (best.0 + rng.range(-rx, rx), best.1 + rng.range(-ry, ry));
                        if self.is_valid(cand.0, cand.1) && cand != best {
                            let pen = seam_pen(cand);
                            if total_bc > pen {
                                let d = self.dist(t, cand, total_bc - pen);
                                if d + pen < total_bc {
                                    (best, bc, total_bc) = (cand, d, d + pen);
                                }
                            }
                        }
                        rad /= 2;
                    }
                    nn[li] = best;
                    co[li] = bc;
                }
            }
        };
        if self.horizon_weight > 0.0 {
            band_fn(0, nnf, cost);
            return;
        }
        #[cfg(not(target_arch = "wasm32"))]
        if w * self.h >= PAR_MIN {
            use rayon::prelude::*;
            nnf.par_chunks_mut(BAND * w).zip(cost.par_chunks_mut(BAND * w)).enumerate().for_each(|(bi, (nn, co))| band_fn(bi, nn, co));
            return;
        }
        for (bi, (nn, co)) in nnf.chunks_mut(BAND * w).zip(cost.chunks_mut(BAND * w)).enumerate() {
            band_fn(bi, nn, co);
        }
    }

    /// Re-estimate hole pixels by a similarity-weighted vote of all overlapping patch matches.
    fn vote(&mut self, nnf: &[(i32, i32)], cost: &[f32], sharp_scale: f32) {
        let (w, h, ch, r) = (self.w, self.h, self.ch, self.r);
        // σ² from the 75th percentile of match costs (Wexler et al. use a robust scale like this).
        let mut cs: Vec<f32> = cost.iter().zip(&self.targets).filter(|(c, t)| **t && c.is_finite()).map(|(c, _)| *c).collect();
        let sigma2 = if cs.is_empty() {
            1.0
        } else {
            let k = (cs.len() * 3 / 4).min(cs.len() - 1);
            let (_, v, _) = cs.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
            (*v * sharp_scale).max(1e-6)
        };
        let img = &*self.img;
        let use_edge_guard = self.horizon_weight > 0.0 && ch == 3;
        let row = |y: usize| -> Vec<(usize, Vec<f32>)> {
            let mut out = Vec::new();
            if (self.ctl)() {
                return out;
            }
            for x in 0..w {
                if !self.hole[y * w + x] {
                    continue;
                }
                // Identify the best-matching patch in the 3x3 core around `(x, y)` so cross-seam
                // neighbors (e.g. bright sky patch next to a dark palm frond patch) are weighted by
                // color agreement and never average sky + frond into a blurry cyan halo!
                let mut best_core_cost = f32::INFINITY;
                let mut ref_s = nnf[y * w + x];
                if use_edge_guard {
                    for cdy in -1i32..=1 {
                        let cty = y as i32 + cdy;
                        if cty < 0 || cty as usize >= h {
                            continue;
                        }
                        for cdx in -1i32..=1 {
                            let ctx = x as i32 + cdx;
                            if ctx < 0 || ctx as usize >= w {
                                continue;
                            }
                            let cti = cty as usize * w + ctx as usize;
                            if self.targets[cti] && cost[cti] < best_core_cost {
                                let cs_cand = (nnf[cti].0 - cdx, nnf[cti].1 - cdy);
                                if cs_cand.0 >= 0 && cs_cand.1 >= 0 && (cs_cand.0 as usize) < w && (cs_cand.1 as usize) < h {
                                    best_core_cost = cost[cti];
                                    ref_s = cs_cand;
                                }
                            }
                        }
                    }
                }
                let ref_si = (ref_s.1.clamp(0, h as i32 - 1) as usize * w + ref_s.0.clamp(0, w as i32 - 1) as usize) * ch;
                let mut acc = vec![0.0f32; ch];
                let mut wsum = 0.0f32;
                for dy in -r..=r {
                    let ty = y as i32 + dy;
                    if ty < 0 || ty as usize >= h {
                        continue;
                    }
                    for dx in -r..=r {
                        let tx = x as i32 + dx;
                        if tx < 0 || tx as usize >= w {
                            continue;
                        }
                        let ti = ty as usize * w + tx as usize;
                        if !self.targets[ti] || !cost[ti].is_finite() {
                            continue;
                        }
                        let s = nnf[ti];
                        let (sx, sy) = (s.0 - dx, s.1 - dy);
                        let si = (sy as usize * w + sx as usize) * ch;
                        let spat = 1.0 / (1.0 + 0.22 * (dx * dx + dy * dy) as f32);
                        let mut wt = ((-cost[ti] / (1.5 * sigma2)).exp() * spat).max(1e-8);
                        if use_edge_guard && ((sx - ref_s.0).abs() > 1 || (sy - ref_s.1).abs() > 1) {
                            let dr = img[si] - img[ref_si];
                            let dg = img[si + 1] - img[ref_si + 1];
                            let db = img[si + 2] - img[ref_si + 2];
                            let d_col2 = dr * dr + dg * dg * 1.4 + db * db;
                            wt *= (-d_col2 * 55.0).exp();
                        }
                        for c in 0..ch {
                            acc[c] += img[si + c] * wt;
                        }
                        wsum += wt;
                    }
                }
                if wsum > 0.0 {
                    acc.iter_mut().for_each(|v| *v /= wsum);
                    out.push((x, acc));
                }
            }
            out
        };
        #[cfg(not(target_arch = "wasm32"))]
        let rows: Vec<Vec<(usize, Vec<f32>)>> = if w * h >= PAR_MIN {
            use rayon::prelude::*;
            (0..h).into_par_iter().map(row).collect()
        } else {
            (0..h).map(row).collect()
        };
        #[cfg(target_arch = "wasm32")]
        let rows: Vec<Vec<(usize, Vec<f32>)>> = (0..h).map(row).collect();
        for (y, r) in rows.into_iter().enumerate() {
            for (x, v) in r {
                self.img[(y * w + x) * ch..(y * w + x + 1) * ch].copy_from_slice(&v);
            }
        }
    }
}

/// Content-aware completion of `hole` (see module docs). Returns `None` when there is no fully-known
/// patch to copy from (the caller can fall back to [`membrane_fill`]).
pub fn complete(w: usize, h: usize, ch: usize, img: &[f32], hole: &[bool], p: &CompleteParams) -> Option<Vec<f32>> {
    // Never cancelled, so never `Err`.
    complete_with(w, h, ch, img, hole, p, &|| false).unwrap_or(None)
}

/// [`complete`] that can be cancelled (checked per EM iteration, PatchMatch pass and row band)
/// (`cancelled` is polled per EM iteration, PatchMatch pass and row band): `Err(Cancelled)` once it
/// returned `true`. `Ok(None)` also for mismatched buffer sizes or `ch == 0`.
pub fn complete_with(
    w: usize,
    h: usize,
    ch: usize,
    img: &[f32],
    hole: &[bool],
    p: &CompleteParams,
    cancelled: &(dyn Fn() -> bool + Sync),
) -> Result<Option<Vec<f32>>, Cancelled> {
    let check = || if cancelled() { Err(Cancelled) } else { Ok(()) };
    if ch == 0 || Some(hole.len()) != w.checked_mul(h) || Some(img.len()) != hole.len().checked_mul(ch) {
        return Ok(None);
    }
    let Some((bx0, by0, bx1, by1)) = hole_bbox(w, h, hole) else { return Ok(Some(img.to_vec())) };
    let r = p.patch_radius.max(1);
    let psz = 2 * r + 1;
    // Pyramid: shrink until the hole is a few patches across.
    let mut levels = vec![Level { w, h, img: img.to_vec(), hole: hole.to_vec() }];
    let mut ext = (bx1 - bx0).max(by1 - by0);
    let mut exts = vec![ext];
    while let Some(l) = levels.last() {
        check()?;
        if ext <= 2 * psz || l.w / 2 < 3 * psz || l.h / 2 < 3 * psz {
            break;
        }
        let d = downsample(l, ch);
        levels.push(d);
        ext = ext.div_ceil(2);
        exts.push(ext);
    }
    let nlev = levels.len();
    let mut nnf: Vec<(i32, i32)> = Vec::new();
    let mut prev: Option<(usize, usize, Vec<f32>)> = None;
    for li in (0..nlev).rev() {
        check()?;
        let Level { w: lw, h: lh, img: mut limg, hole: lhole } = levels[li].clone();
        // Initialise the hole: membrane at the coarsest level, upsampled result elsewhere.
        match &prev {
            None => limg = membrane_fill(lw, lh, ch, &limg, &lhole),
            Some((pw, ph, pimg)) => {
                for y in 0..lh {
                    for x in 0..lw {
                        if lhole[y * lw + x] {
                            let (sx, sy) = ((x / 2).min(pw - 1), (y / 2).min(ph - 1));
                            let (d, s) = ((y * lw + x) * ch, (sy * pw + sx) * ch);
                            limg[d..d + ch].copy_from_slice(&pimg[s..s + ch]);
                        }
                    }
                }
            }
        }
        let ri = r as i32;
        let hs = integral(lw, lh, &lhole);
        let mut valid = vec![false; lw * lh];
        let mut valid_list = Vec::new();
        let mut targets = vec![false; lw * lh];
        for y in 0..lh as i32 {
            if y % 32 == 0 {
                check()?;
            }
            for x in 0..lw as i32 {
                let cnt = window_count(&hs, lw, lh, x - ri, y - ri, x + ri + 1, y + ri + 1);
                let inside = x >= ri && y >= ri && x + ri < lw as i32 && y + ri < lh as i32;
                let i = y as usize * lw + x as usize;
                if inside && cnt == 0 {
                    valid[i] = true;
                    valid_list.push((x, y));
                }
                targets[i] = cnt > 0;
            }
        }
        if valid_list.is_empty() {
            if li == 0 {
                return Ok(None);
            }
            prev = Some((lw, lh, limg));
            nnf.clear();
            continue;
        }
        let mut solver = Solver {
            w: lw,
            h: lh,
            ch,
            r: ri,
            horizon_weight: 0.0,
            guide_weight: 0.0,
            is_outpaint: false,
            valid_bbox: (ri, lw as i32 - 1 - ri, ri, lh as i32 - 1 - ri),
            guide_low: None,
            img_low: None,
            img_var: None,
            row_var: None,
            row_has_valid: None,
            img: &mut limg,
            hole: &lhole,
            valid,
            valid_list,
            targets,
            ctl: cancelled,
        };
        // NNF init: upsampled from the coarser level where possible, random otherwise.
        let mut rng = Rng::new(p.seed ^ (li as u64) << 20);
        let pw_prev = prev.as_ref().map_or(0, |p| p.0);
        let old = std::mem::take(&mut nnf);
        let mut new_nnf = vec![(0i32, 0i32); lw * lh];
        for y in 0..lh {
            if y % 32 == 0 {
                check()?;
            }
            for x in 0..lw {
                if !solver.targets[y * lw + x] {
                    continue;
                }
                let mut cand = None;
                if !old.is_empty() && pw_prev > 0 {
                    let pi = (y / 2) * pw_prev + (x / 2);
                    if let Some(&(sx, sy)) = old.get(pi) {
                        let c = (sx * 2 + (x % 2) as i32, sy * 2 + (y % 2) as i32);
                        if solver.is_valid(c.0, c.1) {
                            cand = Some(c);
                        }
                    }
                }
                new_nnf[y * lw + x] = cand.unwrap_or_else(|| solver.valid_list[(rng.next() % solver.valid_list.len() as u64) as usize]);
            }
        }
        nnf = new_nnf;
        let refine_only = li + 1 < nlev && exts[li] > p.refine_extent;
        let em = if li + 1 == nlev {
            p.em_iters
        } else if refine_only {
            1
        } else {
            (p.em_iters >> (nlev - 1 - li)).max(p.min_em_iters)
        };
        let pm_iters = if refine_only { 1 } else { p.pm_iters };
        let full_search = (lw.max(lh)) as i32;
        let mut cost = vec![f32::INFINITY; lw * lh];
        for it in 0..em {
            // Costs change as hole pixels are re-estimated: refresh before each PatchMatch.
            for (i, c) in cost.iter_mut().enumerate() {
                if i % (lw * 32).max(1) == 0 {
                    check()?;
                }
                if solver.targets[i] {
                    *c = solver.dist(((i % lw) as i32, (i / lw) as i32), nnf[i], f32::INFINITY);
                }
            }
            // Wide random search on coarse levels / first iteration, local refinement afterwards.
            let radius = if refine_only {
                2
            } else if li + 1 == nlev || it == 0 {
                full_search
            } else {
                (4 * psz as i32).min(full_search)
            };
            for pass in 0..pm_iters {
                check()?;
                solver.patchmatch(&mut nnf, &mut cost, pass + it * pm_iters, p.seed.wrapping_add((li * 1000 + it) as u64), radius);
            }
            check()?;
            solver.vote(&nnf, &cost, 1.0);
            check()?;
        }
        prev = Some((lw, lh, limg));
    }
    Ok(prev.map(|(_, _, img)| img))
}

/// Horizon-aware, boundary-palette-guarded multi-scale PatchMatch completion that optionally uses
/// a neural/coarse `guide` at the top of the pyramid and returns both `(completed_img, finest_nnf)`.
pub fn complete_guided_nnf(
    w: usize,
    h: usize,
    ch: usize,
    img: &[f32],
    hole: &[bool],
    guide: Option<&[f32]>,
    p: &CompleteParams,
) -> Option<(Vec<f32>, Vec<(i32, i32)>)> {
    if ch == 0 || Some(hole.len()) != w.checked_mul(h) || Some(img.len()) != hole.len().checked_mul(ch) {
        return None;
    }
    let Some((bx0, by0, bx1, by1)) = hole_bbox(w, h, hole) else {
        let id_nnf: Vec<(i32, i32)> = (0..(w * h)).map(|i| ((i % w) as i32, (i / w) as i32)).collect();
        return Some((img.to_vec(), id_nnf));
    };
    let r = p.patch_radius.max(1);
    let psz = 2 * r + 1;

    // Seed input `img` inside `hole` with `guide` (if provided) so downsampled pyramid levels
    // also carry the coarse guide structure if needed.
    let mut base_in = img.to_vec();
    if let Some(g) = guide
        && g.len() >= w * h * ch
    {
        for i in 0..(w * h) {
            if hole[i] {
                let idx = i * ch;
                base_in[idx..idx + ch].copy_from_slice(&g[idx..idx + ch]);
            }
        }
    }

    let mut levels = vec![Level { w, h, img: base_in, hole: hole.to_vec() }];
    let mut ext = (bx1 - bx0).max(by1 - by0);
    let mut exts = vec![ext];
    while let Some(l) = levels.last() {
        if ext <= 2 * psz || l.w / 2 < 3 * psz || l.h / 2 < 3 * psz {
            break;
        }
        let d = downsample(l, ch);
        levels.push(d);
        ext = ext.div_ceil(2);
        exts.push(ext);
    }
    let nlev = levels.len();
    let mut nnf: Vec<(i32, i32)> = Vec::new();
    let mut prev: Option<(usize, usize, Vec<f32>)> = None;

    for li in (0..nlev).rev() {
        let Level { w: lw, h: lh, img: mut limg, hole: lhole } = levels[li].clone();
        // Downsample `levels[0].img` (which contains `guide` inside `hole`) to `(lw, lh)` as persistent level guide
        let guide_lvl = if guide.is_some() {
            let mut gl = levels[0].img.clone();
            let mut gw = w;
            let mut gh = h;
            for _ in 0..li {
                let (nw, nh) = (gw.div_ceil(2), gh.div_ceil(2));
                let mut next_g = vec![0.0f32; nw * nh * ch];
                for yy in 0..nh {
                    for xx in 0..nw {
                        let (sx, sy) = ((xx * 2).min(gw - 1), (yy * 2).min(gh - 1));
                        let dst_o = (yy * nw + xx) * ch;
                        let src_o = (sy * gw + sx) * ch;
                        next_g[dst_o..dst_o + ch].copy_from_slice(&gl[src_o..src_o + ch]);
                    }
                }
                gl = next_g;
                gw = nw;
                gh = nh;
            }
            Some(gl)
        } else {
            None
        };

        match &prev {
            None => {
                if let Some(ref gl) = guide_lvl {
                    for i in 0..(lw * lh) {
                        if lhole[i] {
                            let o = i * ch;
                            limg[o..o + ch].copy_from_slice(&gl[o..o + ch]);
                        }
                    }
                } else {
                    limg = membrane_fill(lw, lh, ch, &limg, &lhole);
                }
            }
            Some((pw, ph, pimg)) => {
                for y in 0..lh {
                    for x in 0..lw {
                        if lhole[y * lw + x] {
                            let (sx, sy) = ((x / 2).min(pw - 1), (y / 2).min(ph - 1));
                            let (d, s) = ((y * lw + x) * ch, (sy * pw + sx) * ch);
                            limg[d..d + ch].copy_from_slice(&pimg[s..s + ch]);
                        }
                    }
                }
            }
        }

        // Use smaller patch radius on coarse levels so narrow unmasked sky/border strips always contain valid patches
        let ri = if lw.min(lh) < 72 {
            2
        } else if lw.min(lh) < 160 {
            3.min(r as i32)
        } else {
            r as i32
        };
        let hs = integral(lw, lh, &lhole);
        let mut valid = vec![false; lw * lh];
        let mut targets = vec![false; lw * lh];
        for y in 0..lh as i32 {
            for x in 0..lw as i32 {
                let cnt = window_count(&hs, lw, lh, x - ri, y - ri, x + ri + 1, y + ri + 1);
                let inside = x >= ri && y >= ri && x + ri < lw as i32 && y + ri < lh as i32;
                let i = y as usize * lw + x as usize;
                if inside && cnt == 0 {
                    valid[i] = true;
                }
                targets[i] = cnt > 0;
            }
        }

        // Precompute local `(2*ri+1)×(2*ri+1)` patch DC mean (`img_low`, `guide_low`) and texture standard
        // deviation (`img_var`, `row_var`) so both `Row-Band Boundary-Palette Subject Exclusion` and
        // `Solver::dist` match macro patch color and chroma without penalizing sharp internal leaf/frond detail!
        let mut img_low = vec![0.0f32; lw * lh * ch];
        let mut guide_low = vec![0.0f32; lw * lh * ch];
        let mut img_var = vec![0.0f32; lw * lh];
        for y in 0..lh {
            let y0 = y.saturating_sub(ri as usize);
            let y1 = (y + ri as usize).min(lh - 1);
            for x in 0..lw {
                let x0 = x.saturating_sub(ri as usize);
                let x1 = (x + ri as usize).min(lw - 1);
                let mut sum_i = [0.0f32; 4];
                let mut sum_g = [0.0f32; 4];
                let mut cnt = 0.0f32;
                for yy in y0..=y1 {
                    for xx in x0..=x1 {
                        let o = (yy * lw + xx) * ch;
                        for c in 0..ch.min(4) {
                            sum_i[c] += limg[o + c];
                            if let Some(ref gl) = guide_lvl {
                                sum_g[c] += gl[o + c];
                            }
                        }
                        cnt += 1.0;
                    }
                }
                let inv = 1.0 / cnt.max(1.0);
                let out_o = (y * lw + x) * ch;
                for c in 0..ch.min(4) {
                    img_low[out_o + c] = sum_i[c] * inv;
                    guide_low[out_o + c] = if guide_lvl.is_some() { sum_g[c] * inv } else { sum_i[c] * inv };
                }
                let mut v_sum = 0.0f32;
                for yy in y0..=y1 {
                    for xx in x0..=x1 {
                        let o = (yy * lw + xx) * ch;
                        for c in 0..ch.min(4) {
                            let d = limg[o + c] - img_low[out_o + c];
                            v_sum += d * d;
                        }
                    }
                }
                img_var[y * lw + x] = (v_sum / (cnt * ch as f32).max(1.0)).sqrt();
            }
        }

        // Detect whether `lhole` is an outer border expansion (`outpaint_lama_borders`):
        let top_hole_cnt = (0..lw).filter(|&x| lhole[x]).count();
        let bot_hole_cnt = (0..lw).filter(|&x| lhole[(lh - 1) * lw + x]).count();
        let left_hole_cnt = (0..lh).filter(|&y| lhole[y * lw]).count();
        let right_hole_cnt = (0..lh).filter(|&y| lhole[y * lw + (lw - 1)]).count();
        let is_outpaint = top_hole_cnt * 10 >= lw * 9
            || bot_hole_cnt * 10 >= lw * 9
            || left_hole_cnt * 10 >= lh * 9
            || right_hole_cnt * 10 >= lh * 9;

        // Row-Band Boundary-Palette Subject & Architecture Exclusion (for object removal `!is_outpaint`):
        // Compute a 3×3 local unmasked mean `img_3x3` at each pixel. On each horizontal band `y`, inspect the
        // immediate hole boundary (`near_pad = lw / 28`) to determine the local background palette (and whether
        // the boundary itself contains achromatic bright concrete/clothes or red-dominant terracotta/skin).
        // Mark any pixel that is an outlier for its row band in `outlier_px`, and exclude any patch center `(x, y)`
        // whose `(4*ri+1)×(4*ri+1)` neighborhood overlaps an outlier pixel (`window_count(&outlier_sat) > 0`),
        // so narrow white palapa pillars, terracotta roofs, concrete walls, foreground people, and distant
        // below-skyline blue mountain gaps are 100% excluded while 100% of palm trees/foliage remain valid!
        if ch == 3 && !is_outpaint {
            let mut img_3x3 = vec![0.0f32; lw * lh * 3];
            for y in 0..lh {
                let y0 = y.saturating_sub(1);
                let y1 = (y + 1).min(lh - 1);
                for x in 0..lw {
                    let x0 = x.saturating_sub(1);
                    let x1 = (x + 1).min(lw - 1);
                    let mut s = [0.0f32; 3];
                    let mut n = 0.0f32;
                    for yy in y0..=y1 {
                        for xx in x0..=x1 {
                            if !lhole[yy * lw + xx] {
                                let o = (yy * lw + xx) * 3;
                                s[0] += limg[o];
                                s[1] += limg[o + 1];
                                s[2] += limg[o + 2];
                                n += 1.0;
                            }
                        }
                    }
                    let out_o = (y * lw + x) * 3;
                    if n > 0.0 {
                        img_3x3[out_o] = s[0] / n;
                        img_3x3[out_o + 1] = s[1] / n;
                        img_3x3[out_o + 2] = s[2] / n;
                    } else {
                        img_3x3[out_o] = limg[out_o];
                        img_3x3[out_o + 1] = limg[out_o + 1];
                        img_3x3[out_o + 2] = limg[out_o + 2];
                    }
                }
            }
            let is_sky_color = |r: f32, g: f32, b: f32| -> bool {
                b > r * 1.12 && b > g * 1.02 && b > 0.14
            };
            let is_blue_haze = |r: f32, g: f32, b: f32| -> bool {
                b > r * 1.10 && b > g * 1.015 && b > 0.12
            };
            let is_achrom_bright = |r: f32, g: f32, b: f32| -> bool {
                let min_v = r.min(g).min(b);
                !is_sky_color(r, g, b)
                    && min_v > 0.19
                    && g < b * 1.15
                    && g < r * 1.05
                    && r >= b * 0.92
            };
            let is_red_dom = |r: f32, g: f32, _b: f32| -> bool {
                r > 0.13 && r > g * 1.09 && (r - g) > 0.018
            };
            let is_dark_silhouette = |r: f32, g: f32, b: f32| -> bool {
                r.max(g).max(b) < 0.068 && g <= r * 1.15
            };
            // Compute `below_skyline[y * lw + x]`: true once `y` is below both the top canopy onset
            // AND the lowest vertical run of true sky (`col_last_sky_y`) in `x ± rad_c`.
            let mut col_canopy_y = vec![lh; lw];
            let mut col_last_sky_y = vec![0usize; lw];
            for x in 0..lw {
                let mut non_sky_run = 0usize;
                let mut sky_run = 0usize;
                for y in 0..lh {
                    let i = y * lw + x;
                    if !lhole[i] {
                        let o = i * 3;
                        let r = img_3x3[o];
                        let g = img_3x3[o + 1];
                        let b = img_3x3[o + 2];
                        if b > r * 1.14 && b > g * 1.03 && b > 0.16 {
                            sky_run += 1;
                            if sky_run >= 3 {
                                col_last_sky_y[x] = y;
                            }
                        } else {
                            sky_run = 0;
                        }
                        if !is_sky_color(r, g, b) {
                            non_sky_run += 1;
                            if non_sky_run >= 3 && col_canopy_y[x] == lh {
                                col_canopy_y[x] = y;
                            }
                        } else {
                            non_sky_run = 0;
                        }
                    }
                }
            }
            let canopy_pad = (lh / 32).max(6);
            let mut raw_skyline_y = vec![lh; lw];
            for x in 0..lw {
                let rad_c = (lw / 24).clamp(8, 20);
                let x0 = x.saturating_sub(rad_c);
                let x1 = (x + rad_c).min(lw - 1);
                let mut min_cy = lh;
                let mut max_sky_y = 0usize;
                for xx in x0..=x1 {
                    min_cy = min_cy.min(col_canopy_y[xx]);
                    max_sky_y = max_sky_y.max(col_last_sky_y[xx]);
                }
                raw_skyline_y[x] = (min_cy + canopy_pad).max(if max_sky_y > 0 { max_sky_y + 4 } else { 0 });
            }
            let mut below_skyline = vec![false; lw * lh];
            for y in 0..lh {
                for x in 0..lw {
                    if y > raw_skyline_y[x] || (y > 0 && below_skyline[(y - 1) * lw + x]) {
                        below_skyline[y * lw + x] = true;
                    }
                }
                let mut x = 0usize;
                while x < lw {
                    if !lhole[y * lw + x] {
                        x += 1;
                        continue;
                    }
                    let x_start = x;
                    while x < lw && lhole[y * lw + x] {
                        x += 1;
                    }
                    let x_end = x;
                    let sl_l = if x_start > 0 {
                        raw_skyline_y[x_start - 1] as f32
                    } else {
                        raw_skyline_y[x_end.min(lw - 1)] as f32
                    };
                    let sl_r = if x_end < lw {
                        raw_skyline_y[x_end] as f32
                    } else {
                        sl_l
                    };
                    let span = (x_end - x_start).max(1) as f32;
                    for hx in x_start..x_end {
                        let t = (hx - x_start) as f32 / span;
                        let sl_y = (sl_l * (1.0 - t) + sl_r * t).round() as usize;
                        let prev_below = y > 0 && below_skyline[(y - 1) * lw + hx];
                        below_skyline[y * lw + hx] = prev_below || (y > sl_y);
                    }
                }
            }
            let mut is_sky_px = vec![false; lw * lh];
            for i in 0..(lw * lh) {
                if !lhole[i] && !below_skyline[i] {
                    let o = i * 3;
                    if is_sky_color(img_3x3[o], img_3x3[o + 1], img_3x3[o + 2]) {
                        is_sky_px[i] = true;
                    }
                }
            }
            let sky_sat = integral(lw, lh, &is_sky_px);
            let r_sky_canopy = (lh / 28).max(16) as i32;

            let mut is_dark_trunk = vec![false; lw * lh];
            for y in 0..lh {
                let vy0 = y.saturating_sub(2);
                let vy1 = (y + 2).min(lh - 1);
                for x in 0..lw {
                    let i = y * lw + x;
                    if !lhole[i] {
                        let o = i * 3;
                        let raw_m = limg[o].max(limg[o + 1]).max(limg[o + 2]);
                        let m3 = img_3x3[o].max(img_3x3[o + 1]).max(img_3x3[o + 2]);
                        let is_dark_bark = (raw_m < 0.068 && (limg[o + 1] - limg[o]) <= 0.014)
                            || (m3 < 0.062 && (img_3x3[o + 1] - img_3x3[o]) <= 0.014);
                        if raw_m < 0.045 || m3 < 0.042 || is_dark_bark {
                            let mut v_dark = 0usize;
                            let hx0 = x.saturating_sub(1);
                            let hx1 = (x + 1).min(lw - 1);
                            for yy in vy0..=vy1 {
                                for xx in hx0..=hx1 {
                                    let ii = yy * lw + xx;
                                    if !lhole[ii] {
                                        let oo = ii * 3;
                                        let rm = limg[oo].max(limg[oo + 1]).max(limg[oo + 2]);
                                        let m3_n = img_3x3[oo].max(img_3x3[oo + 1]).max(img_3x3[oo + 2]);
                                        if rm < 0.068 || m3_n < 0.062 {
                                            v_dark += 1;
                                        }
                                    }
                                }
                            }
                            if v_dark >= 3 {
                                is_dark_trunk[i] = true;
                            }
                        }
                    }
                }
            }

            let band_r = (lh / 32).max(2);
            let near_pad = (lw / 28).max(ri as usize + 2) as i32;
            let mut outlier_px = vec![false; lw * lh];
            let mut band_invalid = vec![false; lw * lh];
            let mut ref_r = Vec::new();
            let mut ref_g = Vec::new();
            let mut ref_b = Vec::new();
            let mut ref_rg = Vec::new();
            let mut ref_bg = Vec::new();
            for y in 0..lh {
                let y0 = y.saturating_sub(band_r);
                let y1 = (y + band_r).min(lh - 1);
                let mut hx_lo = lw;
                let mut hx_hi = 0usize;
                for yy in y0..=y1 {
                    for x in 0..lw {
                        if lhole[yy * lw + x] {
                            hx_lo = hx_lo.min(x);
                            hx_hi = hx_hi.max(x);
                        }
                    }
                }
                if hx_lo <= hx_hi {
                    ref_r.clear();
                    ref_g.clear();
                    ref_b.clear();
                    ref_rg.clear();
                    ref_bg.clear();
                    let mut n_achrom = 0usize;
                    let mut n_red = 0usize;
                    let mut n_dark = 0usize;
                    for yy in y0..=y1 {
                        for xx in 0..lw {
                            if valid[yy * lw + xx]
                                && window_count(
                                    &hs,
                                    lw,
                                    lh,
                                    xx as i32 - near_pad,
                                    y0 as i32,
                                    xx as i32 + near_pad + 1,
                                    y1 as i32 + 1,
                                ) > 0
                            {
                                let idx = (yy * lw + xx) * 3;
                                let r_m = img_3x3[idx];
                                let g_m = img_3x3[idx + 1];
                                let b_m = img_3x3[idx + 2];
                                if below_skyline[yy * lw + xx] && is_blue_haze(r_m, g_m, b_m) {
                                    continue;
                                }
                                ref_r.push(r_m);
                                ref_g.push(g_m);
                                ref_b.push(b_m);
                                ref_rg.push(r_m - g_m);
                                ref_bg.push(b_m - g_m);
                                if is_achrom_bright(r_m, g_m, b_m) {
                                    n_achrom += 1;
                                }
                                if is_red_dom(r_m, g_m, b_m) {
                                    n_red += 1;
                                }
                                if is_dark_silhouette(r_m, g_m, b_m) {
                                    n_dark += 1;
                                }
                            }
                        }
                    }
                    if ref_r.len() >= 4 {
                        ref_r.sort_by(|a, b| a.total_cmp(b));
                        ref_g.sort_by(|a, b| a.total_cmp(b));
                        ref_b.sort_by(|a, b| a.total_cmp(b));
                        ref_rg.sort_by(|a, b| a.total_cmp(b));
                        ref_bg.sort_by(|a, b| a.total_cmp(b));
                        let n = ref_r.len();
                        let lo_i = (n * 5 / 100).min(n - 1);
                        let hi_i = (n * 95 / 100).min(n - 1);
                        let min_c = [ref_r[lo_i], ref_g[lo_i], ref_b[lo_i]];
                        let max_c = [ref_r[hi_i], ref_g[hi_i], ref_b[hi_i]];
                        let min_rg = ref_rg[lo_i];
                        let max_rg = ref_rg[hi_i];
                        let min_bg = ref_bg[lo_i];
                        let max_bg = ref_bg[hi_i];
                        let allow_achrom = n_achrom * 5 >= n;
                        let allow_red = n_red * 5 >= n;
                        let allow_dark = n_dark * 3 >= n || ref_g[n / 2] < 0.085;
                        let tol = 0.11;
                        let tol_c = 0.08;
                        for x in 0..lw {
                            let i = y * lw + x;
                            if !lhole[i] {
                                let idx = i * 3;
                                let r_3 = img_3x3[idx];
                                let g_3 = img_3x3[idx + 1];
                                let b_3 = img_3x3[idx + 2];
                                let r_v = img_low[idx];
                                let g_v = img_low[idx + 1];
                                let b_v = img_low[idx + 2];
                                if !allow_red && is_red_dom(r_3, g_3, b_3) {
                                    outlier_px[i] = true;
                                }
                                if !allow_dark && (is_dark_trunk[i] || r_v.max(g_v).max(b_v) < 0.045) {
                                    outlier_px[i] = true;
                                }
                                let in_sky_canopy = !below_skyline[i]
                                    && window_count(
                                        &sky_sat,
                                        lw,
                                        lh,
                                        x as i32 - r_sky_canopy,
                                        y as i32 - r_sky_canopy,
                                        x as i32 + r_sky_canopy + 1,
                                        y as i32 + r_sky_canopy + 1,
                                    ) > 0;
                                if !in_sky_canopy {
                                    if (!allow_achrom && is_achrom_bright(r_3, g_3, b_3))
                                        || (below_skyline[i] && is_blue_haze(r_3, g_3, b_3))
                                    {
                                        outlier_px[i] = true;
                                    }
                                    let rg_v = r_v - g_v;
                                    let bg_v = b_v - g_v;
                                    if r_v < min_c[0] - tol
                                        || r_v > max_c[0] + tol
                                        || g_v < min_c[1] - tol
                                        || g_v > max_c[1] + tol
                                        || b_v < min_c[2] - tol
                                        || b_v > max_c[2] + tol
                                        || rg_v < min_rg - tol_c
                                        || rg_v > max_rg + tol_c
                                        || bg_v < min_bg - tol_c
                                        || bg_v > max_bg + tol_c
                                    {
                                        band_invalid[i] = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            let outlier_sat = integral(lw, lh, &outlier_px);
            let pad_out = (ri * 2).max(3);
            let mut filtered_valid = valid.clone();
            let mut kept_count = 0usize;
            for y in 0..lh {
                for x in 0..lw {
                    let i = y * lw + x;
                    if filtered_valid[i] {
                        if band_invalid[i]
                            || window_count(
                                &outlier_sat,
                                lw,
                                lh,
                                x as i32 - pad_out,
                                y as i32 - pad_out,
                                x as i32 + pad_out + 1,
                                y as i32 + pad_out + 1,
                            ) > 0
                        {
                            filtered_valid[i] = false;
                        } else {
                            kept_count += 1;
                        }
                    }
                }
            }
            if kept_count >= 32 {
                valid = filtered_valid;
            }
        }

        let mut valid_list = Vec::new();
        for y in 0..lh as i32 {
            for x in 0..lw as i32 {
                if valid[y as usize * lw + x as usize] {
                    valid_list.push((x, y));
                }
            }
        }
        if valid_list.is_empty() {
            if li == 0 {
                return None;
            }
            prev = Some((lw, lh, limg));
            nnf.clear();
            continue;
        }
        let mut row_var = vec![0.0f32; lh];
        for y in 0..lh {
            let y0 = y.saturating_sub(ri as usize * 2);
            let y1 = (y + ri as usize * 2).min(lh - 1);
            let mut vs_nonsky = Vec::new();
            let mut vs_all = Vec::new();
            for yy in y0..=y1 {
                for x in 0..lw {
                    let i = yy * lw + x;
                    if valid[i] {
                        let v = img_var[i];
                        vs_all.push(v);
                        if ch == 3 {
                            let sc = i * 3;
                            let is_sky = img_low[sc + 2] > img_low[sc] * 1.14
                                && img_low[sc + 2] > img_low[sc + 1] * 1.03
                                && img_low[sc + 1] > 0.06
                                && v < 0.025;
                            if !is_sky {
                                vs_nonsky.push(v);
                            }
                        } else {
                            vs_nonsky.push(v);
                        }
                    }
                }
            }
            let vs = if vs_nonsky.len() >= 4 { &mut vs_nonsky } else { &mut vs_all };
            if !vs.is_empty() {
                vs.sort_by(|a, b| a.total_cmp(b));
                // 48th percentile (median) roughness of non-sky valid patches on this horizon band
                row_var[y] = vs[(vs.len() * 48 / 100).min(vs.len() - 1)];
            }
        }
        for y in 1..lh {
            if row_var[y] == 0.0 {
                row_var[y] = row_var[y - 1];
            }
        }
        for y in (0..lh.saturating_sub(1)).rev() {
            if row_var[y] == 0.0 {
                row_var[y] = row_var[y + 1];
            }
        }

        let mut row_has_valid = vec![false; lh];
        for y in 0..lh {
            for x in 0..lw {
                if valid[y * lw + x] {
                    row_has_valid[y] = true;
                    break;
                }
            }
        }

        let guide_weight = if li + 2 >= nlev {
            0.22
        } else if li >= 2 {
            0.09
        } else {
            0.025
        };

        let mut vx0 = lw as i32;
        let mut vx1 = 0i32;
        let mut vy0 = lh as i32;
        let mut vy1 = 0i32;
        for &(vx, vy) in &valid_list {
            vx0 = vx0.min(vx);
            vx1 = vx1.max(vx);
            vy0 = vy0.min(vy);
            vy1 = vy1.max(vy);
        }
        let valid_bbox = if vx0 <= vx1 && vy0 <= vy1 {
            (vx0, vx1, vy0, vy1)
        } else {
            (ri, lw as i32 - 1 - ri, ri, lh as i32 - 1 - ri)
        };

        let mut solver = Solver {
            w: lw,
            h: lh,
            ch,
            r: ri,
            horizon_weight: 0.24,
            guide_weight,
            is_outpaint,
            valid_bbox,
            guide_low: Some(&guide_low),
            img_low: Some(&img_low),
            img_var: Some(&img_var),
            row_var: Some(&row_var),
            row_has_valid: Some(&row_has_valid),
            img: &mut limg,
            hole: &lhole,
            valid,
            valid_list,
            targets,
            ctl: &|| false,
        };
        let mut rng = Rng::new(p.seed ^ (li as u64) << 20);
        let pw_prev = prev.as_ref().map_or(0, |p| p.0);
        let old = std::mem::take(&mut nnf);
        let mut new_nnf = vec![(0i32, 0i32); lw * lh];
        for y in 0..lh {
            for x in 0..lw {
                if !solver.targets[y * lw + x] {
                    new_nnf[y * lw + x] = (x as i32, y as i32);
                    continue;
                }
                let mut cand = None;
                if !old.is_empty() && pw_prev > 0 {
                    let pi = (y / 2) * pw_prev + (x / 2);
                    if let Some(&(sx, sy)) = old.get(pi) {
                        let c = (sx * 2 + (x % 2) as i32, sy * 2 + (y % 2) as i32);
                        if solver.is_valid(c.0, c.1) {
                            cand = Some(c);
                        }
                    }
                }
                let has_rv = row_has_valid[y];
                new_nnf[y * lw + x] = cand.unwrap_or_else(|| {
                    if is_outpaint {
                        let (ax, ay) = solver.outpaint_anchor((x as i32, y as i32));
                        if solver.is_valid(ax, ay) {
                            return (ax, ay);
                        }
                        for attempt in 0..32 {
                            let jx = ax + ((rng.next() % 9) as i32 - 4);
                            let jy = ay + ((attempt % 5) as i32 - 2);
                            if solver.is_valid(jx, jy) {
                                return (jx, jy);
                            }
                        }
                    }
                    if !has_rv {
                        let max_dx = lw / 5 + 8;
                        for _ in 0..24 {
                            let c = solver.valid_list[(rng.next() % solver.valid_list.len() as u64) as usize];
                            if (c.0 - x as i32).unsigned_abs() as usize <= max_dx
                                && (c.1 - y as i32).unsigned_abs() as usize <= lh / 4 + 10
                            {
                                return c;
                            }
                        }
                    }
                    let max_dy = lh / 5 + 8;
                    for _ in 0..16 {
                        let c = solver.valid_list[(rng.next() % solver.valid_list.len() as u64) as usize];
                        if (c.1 - y as i32).unsigned_abs() as usize <= max_dy {
                            return c;
                        }
                    }
                    solver.valid_list[(rng.next() % solver.valid_list.len() as u64) as usize]
                });
            }
        }
        nnf = new_nnf;
        if is_outpaint && li + 1 == nlev {
            // Seed `solver.img` inside `lhole` at the coarsest level directly from the 2D `outpaint_anchor`
            // sheet (`nnf`) instead of the 1D clamped `guide_lvl` so `solver.dist` never favors 1D barcode clamping!
            for y in 0..lh {
                for x in 0..lw {
                    let i = y * lw + x;
                    if solver.hole[i] {
                        let (sx, sy) = nnf[i];
                        let si = (sy.clamp(0, lh as i32 - 1) as usize * lw + sx.clamp(0, lw as i32 - 1) as usize) * ch;
                        let di = i * ch;
                        for c in 0..ch {
                            solver.img[di + c] = solver.img[si + c];
                        }
                    }
                }
            }
        }
        let refine_only = li + 1 < nlev && exts[li] > p.refine_extent;
        let em = if li + 1 == nlev {
            p.em_iters
        } else if refine_only {
            2
        } else {
            (p.em_iters >> (nlev - 1 - li)).max(p.min_em_iters)
        };
        let pm_iters = if refine_only { 2 } else { p.pm_iters };
        let full_search = (lw.max(lh)) as i32;
        let mut cost = vec![f32::INFINITY; lw * lh];
        for it in 0..em {
            for (i, c) in cost.iter_mut().enumerate() {
                if solver.targets[i] {
                    *c = solver.dist(((i % lw) as i32, (i / lw) as i32), nnf[i], f32::INFINITY);
                }
            }
            let radius = if refine_only {
                if it == 0 { (3 * psz as i32).min(full_search) } else { 8 }
            } else if li + 1 == nlev || it == 0 {
                full_search
            } else {
                (4 * psz as i32).min(full_search)
            };
            for pass in 0..pm_iters {
                solver.patchmatch(
                    &mut nnf,
                    &mut cost,
                    pass + it * pm_iters,
                    p.seed.wrapping_add((li * 1000 + it) as u64),
                    radius,
                );
            }
            let sharp_scale = if li == 0 && it + 1 == em { 0.22 } else { 0.75 };
            solver.vote(&nnf, &cost, sharp_scale);
        }
        prev = Some((lw, lh, limg));
    }
    prev.map(|(_, _, img)| (img, nnf))
}

/// Proximity match: the displacement `(dx, dy)` (within `max_radius`) whose surroundings best match the
/// ring of `ring` pixels around the hole, with the displaced hole fully inside the grid and outside the
#[cfg(test)]
mod tests {
    use super::*;

    /// Smooth gradient plus deterministic fine noise.
    fn texture(w: usize, h: usize) -> Vec<f32> {
        let mut rng = Rng::new(7);
        (0..w * h)
            .flat_map(|i| {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                let n = (rng.next() % 1000) as f32 / 1000.0 * 0.04 - 0.02;
                [0.3 + 0.004 * x + n, 0.5 + 0.002 * y + n, 0.4 + n]
            })
            .collect()
    }

    fn disc(w: usize, h: usize, cx: f32, cy: f32, r: f32) -> Vec<bool> {
        (0..w * h).map(|i| ((i % w) as f32 - cx).hypot((i / w) as f32 - cy) < r).collect()
    }

    fn hole_rmse(a: &[f32], b: &[f32], hole: &[bool], ch: usize) -> f32 {
        let (mut e, mut n) = (0.0f32, 0);
        for (i, hh) in hole.iter().enumerate() {
            if *hh {
                for c in 0..ch {
                    e += (a[i * ch + c] - b[i * ch + c]).powi(2);
                    n += 1;
                }
            }
        }
        (e / n as f32).sqrt()
    }

    #[test]
    fn completion_fills_blemish_from_texture() {
        let (w, h, ch) = (96, 96, 3);
        let clean = texture(w, h);
        let hole = disc(w, h, 48.0, 48.0, 9.0);
        let mut dirty = clean.clone();
        for (i, hh) in hole.iter().enumerate() {
            if *hh {
                dirty[i * ch..i * ch + ch].copy_from_slice(&[0.02, 0.02, 0.02]);
            }
        }
        let out = complete(w, h, ch, &dirty, &hole, &CompleteParams::default()).unwrap();
        let before = hole_rmse(&dirty, &clean, &hole, ch);
        let after = hole_rmse(&out, &clean, &hole, ch);
        assert!(after < 0.06 && after < before * 0.15, "rmse before {before} after {after}");
        // Outside the hole untouched.
        assert!(out.iter().zip(&dirty).zip(hole.iter().flat_map(|h| [*h; 3])).all(|((a, b), h)| h || a == b));
        // Deterministic.
        assert_eq!(out, complete(w, h, ch, &dirty, &hole, &CompleteParams::default()).unwrap());
    }

    #[test]
    fn completion_continues_structure() {
        // Vertical stripes: the filled hole should keep the stripes (not a blur).
        let (w, h, ch) = (80, 80, 1);
        let clean: Vec<f32> = (0..w * h).map(|i| if (i % w / 4) % 2 == 0 { 0.2 } else { 0.8 }).collect();
        let hole = disc(w, h, 40.0, 40.0, 7.0);
        let dirty: Vec<f32> = clean.iter().zip(&hole).map(|(c, h)| if *h { 0.5 } else { *c }).collect();
        let out = complete(w, h, ch, &dirty, &hole, &CompleteParams::default()).unwrap();
        let err = hole_rmse(&out, &clean, &hole, ch);
        assert!(err < 0.15, "{err}");
    }
}
