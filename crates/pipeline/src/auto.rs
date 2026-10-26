//! Auto tone (a Lightroom-like recipe, see [`auto_tone`]) and auto white balance (grey-world
//! statistics on a proxy).

use lightcraft_color::cct::xy_to_temp_tint;
use lightcraft_color::{REC2020, Xy, bradford, luminance_2020};
use lightcraft_develop::DevelopSettings;
use lightcraft_raster::Rgb32f;
use serde::Serialize;

use crate::SourceInfo;
use crate::local::effective_wb;

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct AutoTone {
    pub exposure: f64,
    pub contrast: f64,
    pub highlights: f64,
    pub shadows: f64,
    pub whites: f64,
    pub blacks: f64,
    pub vibrance: f64,
    pub saturation: f64,
}

/// Exposure in EV that Auto adds per EV of the scene's log-average luminance (the slope is negative:
/// darker scenes get more). The slope is Lightroom's: regressing the Exposure its Auto chose on 24
/// raw photos against our own [`scene_log_mean`] of each gives -0.41 EV/EV (r = -0.92).
const EXPOSURE_SLOPE: f64 = -0.41;
/// Exposure for a scene whose log-average luminance is middle grey: fitted, with [`CONTRAST`] and
/// [`VIBRANCE`], so our renders match Lightroom's Auto results on those photos (see [`auto_tone`]).
const EXPOSURE_AT_GREY: f64 = -0.58;
/// Highlights and Shadows are Lightroom's own: the average its Auto chose on the 24 photos (our
/// sliders aim at the same as its). Freeing them as well matched no better on held-out photos and
/// left them poorly determined.
const HIGHLIGHTS: f64 = -66.0;
const SHADOWS: f64 = 52.0;
/// Contrast and Vibrance are fitted: our slider scales differ from Lightroom's, and our base tone is
/// flatter, so Contrast is far above its +6. Whites, Blacks and Saturation stay at 0 (Whites and
/// Blacks made no difference to the match; Lightroom leaves Saturation at about 0).
const CONTRAST: f64 = 68.0;
const VIBRANCE: f64 = 30.0;

/// The scene's log-average luminance under the current white balance, in EV relative to middle grey
/// (the mean of log2(Y / 0.18) over a 512 px proxy, scene-linear, before any tone curve). Pixels that
/// aren't finite (a damaged decode) are left out: they get no weight when the proxy is made, and
/// proxy pixels made only of them are skipped. `None` when no pixel is valid.
pub fn scene_log_mean(src: &Rgb32f, info: &SourceInfo, s: &DevelopSettings) -> Option<f64> {
    let (mut clean, mut weight) = (src.clone(), Rgb32f::new(src.width, src.height));
    let mut any = false;
    for (p, w) in clean.data.iter_mut().zip(weight.data.iter_mut()) {
        if p.iter().all(|v| v.is_finite()) {
            *w = [1.0; 3];
            any = true;
        } else {
            *p = [0.0; 3];
        }
    }
    if !any {
        return None;
    }
    let fit = |img: &Rgb32f| lightcraft_raster::resample::fit(img, 512, 512, lightcraft_raster::resample::Filter::Box);
    let (img, weight) = (fit(&clean), fit(&weight));
    // the mean of the valid pixels under each proxy pixel
    let kept: Vec<[f32; 3]> = img.data.iter().zip(&weight.data).filter(|(_, w)| w[0] > 1e-6).map(|(p, w)| p.map(|v| v / w[0])).collect();
    let mut kept = Rgb32f { width: kept.len(), height: 1, data: kept };
    let base = DevelopSettings { wb: s.wb, process: s.process, ..DevelopSettings::default() };
    // (white balance is per pixel: the valid pixels can be balanced in a row of their own)
    crate::local::scene_linear_pre(&mut kept, info, &base);
    let (mut sum, mut n) = (0f64, 0usize);
    for c in &kept.data {
        let ev = (luminance_2020(*c).max(1e-6) / 0.18).log2();
        if ev.is_finite() {
            sum += ev as f64;
            n += 1;
        }
    }
    (n > 0).then(|| sum / n as f64)
}

/// Auto settings for `src` under the current white balance (ignores current tone values), aiming at
/// what Lightroom's Auto does: the Exposure a scene needs follows its log-average luminance
/// ([`scene_log_mean`], scene-linear, so the rule doesn't depend on the base tone curve), and the
/// other sliders are one fixed recipe. Calibrated black-box against Lightroom's observed behaviour
/// (the values its Auto chose and its exported results) on 24 Sony raw photos, of which 16 have a
/// starting look that Sony's in-camera DRO doesn't brighten. Held out one photo at a time, those 16
/// come out at a median dE76 of 6.4 from Lightroom's Auto results (8.3 unedited; 18.8 with the
/// previous percentile rule, which ended about 16 L* too bright); all 24 at 7.7 (7.9 unedited).
/// Rendered sources (JPEGs) are already toned, so their log-average reads brighter and Auto changes
/// their exposure less. A source without a single valid pixel gets Exposure 0 (and the rest of the
/// recipe).
pub fn auto_tone(src: &Rgb32f, info: &SourceInfo, s: &DevelopSettings) -> AutoTone {
    let exposure = scene_log_mean(src, info, s).map_or(0.0, |m| (EXPOSURE_AT_GREY + EXPOSURE_SLOPE * m).clamp(-2.0, 3.0));
    AutoTone {
        exposure: (exposure * 100.0).round() / 100.0,
        contrast: CONTRAST,
        highlights: HIGHLIGHTS,
        shadows: SHADOWS,
        whites: 0.0,
        blacks: 0.0,
        vibrance: VIBRANCE,
        saturation: 0.0,
    }
}

/// An automatic black & white mix (slider values, red … magenta) for `src` under `s`'s white
/// balance and tone: each hue band's colourful pixels are pushed away from the image's mean
/// lightness — bands brighter than average get brighter, darker ones darker — so areas that
/// differ only in colour stay apart in grey. Bands with almost no colourful pixels stay at 0.
pub fn auto_bw_mix(src: &Rgb32f, info: &SourceInfo, s: &DevelopSettings) -> [f64; 8] {
    use lightcraft_color::perceptual::{lab_to_lch, oklab_from_2020};
    let mut img = lightcraft_raster::resample::fit(src, 512, 512, lightcraft_raster::resample::Filter::Box);
    let base = DevelopSettings { wb: s.wb, light: s.light, process: s.process, ..DevelopSettings::default() };
    crate::local::scene_linear_pre(&mut img, info, &base);
    let gain = 2f32.powf(base.light.exposure as f32);
    let (mut mass, mut sum_l) = ([0f64; 8], [0f64; 8]);
    let (mut all_l, mut n) = (0f64, 0usize);
    for p in &img.data {
        let lch = lab_to_lch(oklab_from_2020(p.map(|v| (v * gain).max(0.0))));
        all_l += lch[0] as f64;
        n += 1;
        let k = (lch[1] / 0.2).min(1.0) as f64; // as in the B&W conversion
        if k < 0.05 {
            continue;
        }
        let w = crate::colorops::band_weights(lch[2]);
        for i in 0..8 {
            mass[i] += w[i] as f64 * k;
            sum_l[i] += w[i] as f64 * k * lch[0] as f64;
        }
    }
    if n == 0 {
        return [0.0; 8];
    }
    let mean = all_l / n as f64;
    let total: f64 = mass.iter().sum();
    std::array::from_fn(|i| {
        if total <= 0.0 || mass[i] / total < 0.01 {
            return 0.0;
        }
        let sep = sum_l[i] / mass[i] - mean;
        (sep * 400.0).clamp(-60.0, 60.0).round()
    })
}

/// Grey-world white balance weighted towards mid-tone, low-chroma pixels. Returns (temp, tint).
pub fn auto_wb(src: &Rgb32f, info: &SourceInfo) -> (f64, f64) {
    let img = lightcraft_raster::resample::fit(src, 256, 256, lightcraft_raster::resample::Filter::Box);
    let (mut acc, mut wsum) = ([0.0f64; 3], 0.0f64);
    for c in &img.data {
        let y = luminance_2020(*c);
        if !(0.01..=2.0).contains(&y) {
            continue;
        }
        let mx = c[0].max(c[1]).max(c[2]);
        let mn = c[0].min(c[1]).min(c[2]);
        let chroma = (mx - mn) / (mx + 1e-6);
        let w = (1.0 - chroma).powi(2) as f64 * (1.0 - ((y.log2() + 2.5) / 4.0).abs().min(1.0)) as f64;
        for i in 0..3 {
            acc[i] += c[i] as f64 * w;
        }
        wsum += w;
    }
    if wsum <= 0.0 {
        return (info.as_shot_temp, info.as_shot_tint);
    }
    let avg = acc.map(|v| v / wsum);
    // the white that makes `avg` neutral, through the same model the white balance renders with
    // (`local::wb_matrix_for`): the camera's own colour model, else a Bradford adaptation
    let camera = info.camera_color.as_deref().filter(|_| info.raw && !info.relative_wb);
    let seen = match camera.and_then(|cc| lightcraft_raw::color::neutral_white(&cc.tags, cc.developed_for, avg)) {
        Some(white) => white,
        None => {
            let shot = lightcraft_color::cct::temp_tint_to_xy(info.as_shot_temp, info.as_shot_tint);
            Xy::from_xyz(bradford(REC2020.white, shot).apply(REC2020.to_xyz().apply(avg)))
        }
    };
    let (t, tint) = xy_to_temp_tint(seen);
    (t.clamp(2000.0, 50000.0).round(), tint.clamp(-150.0, 150.0).round())
}

/// Temperature/tint currently in effect (for UI display).
pub fn current_wb(info: &SourceInfo, s: &DevelopSettings) -> (f64, f64) {
    effective_wb(info, s)
}

const QUANTILE_PROBS: [f64; 7] = [0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98];
const QUANTILE_WEIGHTS: [f64; 7] = [0.9, 1.5, 1.9, 2.4, 1.9, 1.7, 1.2];
const MATCH_PROXY_EDGE: usize = 160;
const MATCH_BUCKETS: usize = 192;

/// Target tonal and colour profile extracted from a user-edited reference photo ("Anchor Photo")
/// so subsequent photos can be adaptively edited to match its *final rendered look* rather than
/// blindly copying slider numbers.
#[derive(Clone, Debug)]
pub struct StyleAnchor {
    /// Full reference settings (used to carry over creative sections: Profile, Curve, Mixer,
    /// Grading, Point Color, Effects, Calibration, Treatment).
    pub reference: DevelopSettings,
    /// Rendered perceptual lightness (`Y^(1/2.2)`) quantiles of the reference photo at
    /// [`QUANTILE_PROBS`].
    pub quantiles: [f64; 7],
    /// Subject/centre-weighted mean perceptual lightness of the rendered reference photo.
    pub subject_lightness: f64,
    /// Creative warmth offset in mireds relative to the reference photo's auto-WB neutral
    /// (`1e6 / T_auto - 1e6 / T_effective`; positive = warmer than neutral).
    pub wb_mired_offset: f64,
    /// Creative tint offset (`tint_effective - tint_auto`) relative to the reference's neutral.
    pub wb_tint_offset: f64,
    /// Whether the reference photo used an explicit custom/preset white balance or shifted away
    /// from its as-shot white.
    pub wb_customized: bool,
}

/// Summary of the adaptive adjustments chosen by [`smart_match`] for a target photo.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct SmartMatchReport {
    pub exposure: f64,
    pub contrast: f64,
    pub highlights: f64,
    pub shadows: f64,
    pub whites: f64,
    pub blacks: f64,
    pub temp: f64,
    pub tint: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct ToneBucket {
    l_pre: f32,
    base: f32,
    subj_w: f32,
}

#[inline]
fn smooth_step(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Extract representative sorted tone buckets `(l_pre, base, subj_w)` from `src` after applying
/// `wb_settings`. Non-finite pixels are ignored.
fn extract_tone_buckets(src: &Rgb32f, info: &SourceInfo, wb_settings: &DevelopSettings) -> Vec<ToneBucket> {
    use lightcraft_color::perceptual::{lab_to_lch, oklab_from_2020};
    if src.width == 0 || src.height == 0 || src.data.is_empty() {
        return Vec::new();
    }
    let (mut clean, mut mask) = (src.clone(), Rgb32f::new(src.width, src.height));
    let mut any = false;
    for (p, m) in clean.data.iter_mut().zip(mask.data.iter_mut()) {
        if p.iter().all(|v| v.is_finite()) {
            *m = [1.0; 3];
            any = true;
        } else {
            *p = [0.0; 3];
        }
    }
    if !any {
        return Vec::new();
    }
    let fit = |img: &Rgb32f| lightcraft_raster::resample::fit(img, MATCH_PROXY_EDGE, MATCH_PROXY_EDGE, lightcraft_raster::resample::Filter::Box);
    let (mut img, weight) = (fit(&clean), fit(&mask));
    for (p, w) in img.data.iter_mut().zip(&weight.data) {
        if w[0] > 1e-6 {
            *p = p.map(|v| (v / w[0]).max(0.0));
        } else {
            *p = [0.18; 3];
        }
    }
    let base_wb = DevelopSettings { wb: wb_settings.wb, process: wb_settings.process, ..DevelopSettings::default() };
    crate::local::white_balance(&mut img, info, &base_wb);
    let log_l = img.map(crate::local::log_lum);
    let ppl = img.width.max(img.height) as f64;
    let sigma = (0.015 * ppl as f32).max(1.0);
    let detail = crate::local::tone_base_detail(&base_wb, ppl);
    let base_plane = crate::local::tone_base(&log_l, sigma, detail);

    let (w, h) = (img.width.max(1) as f32, img.height.max(1) as f32);
    let mut samples: Vec<ToneBucket> = Vec::with_capacity(img.data.len());
    for (idx, ((rgb, &lp), (&bp, vw))) in img.data.iter().zip(&log_l.data).zip(base_plane.data.iter().zip(&weight.data)).enumerate() {
        if vw[0] <= 1e-6 || !lp.is_finite() || !bp.is_finite() {
            continue;
        }
        let x = (idx % img.width) as f32 / w;
        let y = (idx / img.width) as f32 / h;
        // Centre + upper-mid bias (typical portrait/subject framing)
        let dx = x - 0.5;
        let dy = y - 0.45;
        let spatial = (-2.2 * (dx * dx + dy * dy)).exp();
        // Boost pixels in plausible human skin-tone chromaticity (warm red-orange hue, moderate chroma)
        let lch = lab_to_lch(oklab_from_2020(*rgb));
        let hue_deg = lch[2].to_degrees().rem_euclid(360.0);
        let skin = if (25.0..=72.0).contains(&hue_deg) && (0.015..=0.16).contains(&lch[1]) && (0.15..=0.88).contains(&lch[0]) {
            1.6
        } else {
            1.0
        };
        samples.push(ToneBucket { l_pre: lp, base: bp, subj_w: (0.25 + 0.75 * spatial) * skin });
    }
    if samples.is_empty() {
        return Vec::new();
    }
    samples.sort_unstable_by(|a, b| a.l_pre.total_cmp(&b.l_pre));
    if samples.len() <= MATCH_BUCKETS {
        return samples;
    }
    // Condense into MATCH_BUCKETS stratified bins so evaluation is O(MATCH_BUCKETS)
    let n = samples.len();
    (0..MATCH_BUCKETS)
        .filter_map(|b| {
            let start = b * n / MATCH_BUCKETS;
            let end = ((b + 1) * n / MATCH_BUCKETS).max(start + 1).min(n);
            let slice = samples.get(start..end)?;
            let count = slice.len() as f32;
            let mut acc = ToneBucket::default();
            for s in slice {
                acc.l_pre += s.l_pre;
                acc.base += s.base;
                acc.subj_w += s.subj_w;
            }
            Some(ToneBucket { l_pre: acc.l_pre / count, base: acc.base / count, subj_w: acc.subj_w / count })
        })
        .collect()
}

/// Evaluate the perceptual lightness quantiles (`Y^(1/2.2)`) and subject-weighted mean lightness
/// for a set of sorted [`ToneBucket`]s under candidate Light sliders `[ev, contrast, hl, sh, wh, bl]`.
fn evaluate_buckets(buckets: &[ToneBucket], info: &SourceInfo, process: lightcraft_develop::ProcessVersion, params: [f64; 6]) -> ([f64; 7], f64) {
    if buckets.is_empty() {
        return ([0.5; 7], 0.5);
    }
    let [ev, contrast, highlights, shadows, whites, blacks] = params;
    let mut temp_settings = DevelopSettings { process, ..DevelopSettings::default() };
    temp_settings.light.contrast = contrast;
    temp_settings.light.whites = whites;
    temp_settings.light.blacks = blacks;
    let tone = crate::finish::base_tone(&temp_settings, info, 1.0);

    let ev_f = ev as f32;
    let hl_f = (highlights / 100.0) as f32;
    let sh_f = (shadows / 100.0) as f32;

    let mut l_vals: Vec<f32> = Vec::with_capacity(buckets.len());
    let (mut subj_sum, mut subj_wsum) = (0.0f64, 0.0f64);
    for b in buckets {
        let l0 = b.l_pre + ev_f;
        let base = b.base + ev_f;
        let ws = 1.0 - smooth_step(-4.8, 0.3, base);
        let wh = smooth_step(-1.0, 2.8, base);
        let delta = sh_f * 1.7 * ws * ws.sqrt() + hl_f * 1.7 * wh;
        let y_scene = crate::tone::GREY * (l0 + delta).exp2();
        let y_disp = tone.apply(y_scene).clamp(0.0, 1.0);
        let l_perc = y_disp.powf(1.0 / 2.2);
        l_vals.push(l_perc);
        subj_sum += (l_perc as f64) * (b.subj_w as f64);
        subj_wsum += b.subj_w as f64;
    }
    l_vals.sort_unstable_by(f32::total_cmp);
    let last = l_vals.len().saturating_sub(1) as f64;
    let quantiles = std::array::from_fn(|i| {
        let pos = (QUANTILE_PROBS[i] * last).clamp(0.0, last);
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        let frac = pos - lo as f64;
        let a = l_vals.get(lo).copied().unwrap_or(0.5) as f64;
        let b = l_vals.get(hi).copied().unwrap_or(a as f32) as f64;
        a + (b - a) * frac
    });
    let subj_mean = if subj_wsum > 1e-9 { subj_sum / subj_wsum } else { quantiles[3] };
    (quantiles, subj_mean)
}

impl StyleAnchor {
    /// Analyse an edited reference photo (`ref_src`, `ref_info`, `ref_settings`) and capture its
    /// rendered tonal profile and creative white-balance / grading style.
    pub fn from_reference(ref_src: &Rgb32f, ref_info: &SourceInfo, ref_settings: &DevelopSettings) -> Self {
        let (eff_t, eff_tint) = effective_wb(ref_info, ref_settings);
        let (auto_t, auto_tint) = auto_wb(ref_src, ref_info);
        let mired = |t: f64| 1e6 / t.clamp(1500.0, 50000.0);
        let wb_mired_offset = (mired(auto_t) - mired(eff_t)).clamp(-250.0, 250.0);
        let wb_tint_offset = (eff_tint - auto_tint).clamp(-150.0, 150.0);
        let wb_customized = ref_settings.wb.mode != lightcraft_develop::WbMode::AsShot
            || (eff_t - ref_info.as_shot_temp).abs() > 1.0
            || (eff_tint - ref_info.as_shot_tint).abs() > 0.5;

        let buckets = extract_tone_buckets(ref_src, ref_info, ref_settings);
        let l = &ref_settings.light;
        let params = [l.exposure, l.contrast, l.highlights, l.shadows, l.whites, l.blacks];
        let (quantiles, subject_lightness) = evaluate_buckets(&buckets, ref_info, ref_settings.process, params);

        Self {
            reference: ref_settings.clone(),
            quantiles,
            subject_lightness,
            wb_mired_offset,
            wb_tint_offset,
            wb_customized,
        }
    }
}

/// Adaptively compute [`DevelopSettings`] for a target photo (`src`, `info`, `current`) so its
/// rendered tonal balance, dynamic range (shadows vs. highlights), white balance and creative
/// grading match the `anchor` photo's final look.
///
/// Unlike copying slider values, if the anchor photo had dark shadows lifted (`shadows > 0`) to
/// achieve a balanced look, and the target photo instead has bright highlights and open shadows,
/// [`smart_match`] lowers `highlights` and adjusts `exposure`/`shadows` on the target photo so both
/// photos converge to the same rendered tonal distribution.
pub fn smart_match(anchor: &StyleAnchor, src: &Rgb32f, info: &SourceInfo, current: &DevelopSettings) -> (DevelopSettings, SmartMatchReport) {
    use lightcraft_develop::{SettingsGroup, WbMode};
    // 1. Copy creative style groups from the reference photo (keeping the target's own geometry/crop/spots/masks).
    let style_groups = vec![
        SettingsGroup::Profile,
        SettingsGroup::Treatment,
        SettingsGroup::ToneCurve,
        SettingsGroup::Color,
        SettingsGroup::ColorMixer,
        SettingsGroup::ColorGrading,
        SettingsGroup::Effects,
        SettingsGroup::Vignette,
        SettingsGroup::Grain,
        SettingsGroup::Detail,
        SettingsGroup::Optics,
        SettingsGroup::Calibration,
    ];
    let partial = lightcraft_develop::extract_groups(&anchor.reference, &style_groups);
    let mut out = lightcraft_develop::apply_partial(current, &partial, 1.0);
    out.treatment = anchor.reference.treatment;
    out.process = anchor.reference.process;

    // 2. Adaptive White Balance: measure the target photo's scene neutral WB and apply the
    //    reference photo's creative warmth/tint offset when WB was styled or when scene cast differs.
    crate::local::normalize_wb(&mut out, info);
    let (target_auto_t, target_auto_tint) = auto_wb(src, info);
    let mired_target = (1e6 / target_auto_t.clamp(1500.0, 50000.0) - anchor.wb_mired_offset).clamp(1e6 / 50000.0, 1e6 / 2000.0);
    let adapted_t = (1e6 / mired_target).clamp(2000.0, 50000.0).round();
    let adapted_tint = (target_auto_tint + anchor.wb_tint_offset).clamp(-150.0, 150.0).round();

    let shot_mired_diff = ((1e6 / target_auto_t.clamp(1500.0, 50000.0)) - (1e6 / info.as_shot_temp.clamp(1500.0, 50000.0))).abs();
    if anchor.wb_customized || shot_mired_diff > 12.0 || (target_auto_tint - info.as_shot_tint).abs() > 6.0 {
        out.wb.mode = WbMode::Custom;
        out.wb.temp = adapted_t;
        out.wb.tint = adapted_tint;
    } else {
        out.wb.mode = WbMode::AsShot;
        out.wb.temp = info.as_shot_temp;
        out.wb.tint = info.as_shot_tint;
    }

    // 3. Adaptive Tone & Dynamic Range Solver: optimize (exposure, contrast, highlights, shadows,
    //    whites, blacks) on the target photo's buckets to match `anchor.quantiles` and `anchor.subject_lightness`.
    let buckets = extract_tone_buckets(src, info, &out);
    if buckets.is_empty() {
        let rep = SmartMatchReport {
            exposure: out.light.exposure,
            contrast: out.light.contrast,
            highlights: out.light.highlights,
            shadows: out.light.shadows,
            whites: out.light.whites,
            blacks: out.light.blacks,
            temp: out.wb.temp,
            tint: out.wb.tint,
        };
        return (out, rep);
    }

    let ref_l = &anchor.reference.light;
    // Initial guess: estimate exposure from neutral rendering vs anchor median, start HL/SH at 0
    // so the target's own shadow/highlight distribution drives them rather than the anchor's raw sliders.
    let (neutral_q, neutral_subj) = evaluate_buckets(&buckets, info, out.process, [0.0, ref_l.contrast, 0.0, 0.0, 0.0, 0.0]);
    let target_mid = (0.6 * anchor.quantiles[3] + 0.4 * anchor.subject_lightness).clamp(0.05, 0.95);
    let cur_mid = (0.6 * neutral_q[3] + 0.4 * neutral_subj).clamp(0.05, 0.95);
    let init_ev = ((target_mid / cur_mid).powf(2.2).log2()).clamp(-3.5, 3.5);

    let clamp_params = |mut p: [f64; 6]| -> [f64; 6] {
        p[0] = p[0].clamp(-4.0, 4.0);
        p[1] = p[1].clamp(-100.0, 100.0);
        p[2] = p[2].clamp(-100.0, 100.0);
        p[3] = p[3].clamp(-100.0, 100.0);
        p[4] = p[4].clamp(-100.0, 100.0);
        p[5] = p[5].clamp(-100.0, 100.0);
        p
    };

    let cost = |p: [f64; 6]| -> f64 {
        let (q, subj) = evaluate_buckets(&buckets, info, out.process, p);
        let mut err = 0.0f64;
        for i in 0..7 {
            let d = q[i] - anchor.quantiles[i];
            err += QUANTILE_WEIGHTS[i] * d * d;
        }
        let ds = subj - anchor.subject_lightness;
        err += 2.2 * ds * ds;

        // Match overall tonal spread (P90 - P10 and P75 - P25)
        let ref_spread_outer = anchor.quantiles[5] - anchor.quantiles[1];
        let cur_spread_outer = q[5] - q[1];
        err += 1.5 * (cur_spread_outer - ref_spread_outer).powi(2);

        let ref_spread_inner = anchor.quantiles[4] - anchor.quantiles[2];
        let cur_spread_inner = q[4] - q[2];
        err += 1.2 * (cur_spread_inner - ref_spread_inner).powi(2);

        // Guardrails against highlight clipping or crushed blacks when the reference itself wasn't clipped
        let ref_hi_headroom = (0.94 - anchor.quantiles[6]).max(0.0);
        if ref_hi_headroom > 0.02 && q[6] > anchor.quantiles[6] + 0.02 {
            let over = q[6] - anchor.quantiles[6];
            err += 4.5 * over * over;
        }
        if anchor.quantiles[0] > 0.03 && q[0] < anchor.quantiles[0] - 0.02 {
            let under = anchor.quantiles[0] - q[0];
            err += 3.0 * under * under;
        }

        // Gentle regularization: prefer Exposure + Highlights + Shadows over extreme Whites/Blacks,
        // and keep Contrast near the reference's aesthetic contrast unless needed for dynamic range.
        let [ev, con, hl, sh, wh, bl] = p;
        let reg = 0.00015 * ev * ev
            + 0.00035 * ((hl / 100.0).powi(2) + (sh / 100.0).powi(2))
            + 0.0012 * ((wh / 100.0).powi(2) + (bl / 100.0).powi(2))
            + 0.0008 * ((con - ref_l.contrast) / 100.0).powi(2);
        err + reg
    };

    let mut best = clamp_params([init_ev, ref_l.contrast, 0.0, 0.0, 0.0, 0.0]);
    let mut best_cost = cost(best);

    // Also test a few smart starting seeds (e.g. highlight-recovery seed, shadow-recovery seed, and anchor-like seed)
    let seeds = [
        [init_ev + 0.4, ref_l.contrast, -65.0, 0.0, 0.0, 0.0],
        [init_ev - 0.4, ref_l.contrast, 0.0, 65.0, 0.0, 0.0],
        [init_ev, ref_l.contrast, -50.0, 50.0, 0.0, 0.0],
        [ref_l.exposure, ref_l.contrast, ref_l.highlights, ref_l.shadows, ref_l.whites, ref_l.blacks],
    ];
    for s in seeds {
        let cand = clamp_params(s);
        let c = cost(cand);
        if c < best_cost {
            best = cand;
            best_cost = c;
        }
    }

    // Coordinate + coupled-direction pattern search
    let mut steps = [0.35, 16.0, 28.0, 28.0, 16.0, 16.0];
    for _round in 0..7 {
        let mut improved = true;
        let mut guard = 0;
        while improved && guard < 12 {
            improved = false;
            guard += 1;
            // Single-axis coordinate moves
            for axis in 0..6 {
                for sign in [-1.0, 1.0] {
                    let mut cand = best;
                    cand[axis] += sign * steps[axis];
                    cand = clamp_params(cand);
                    let c = cost(cand);
                    if c + 1e-7 < best_cost {
                        best = cand;
                        best_cost = c;
                        improved = true;
                    }
                }
            }
            // Coupled moves (decouple global exposure from local highlights/shadows)
            let coupled: [[f64; 6]; 6] = [
                // Lift midtones via exposure while pulling highlights down
                [steps[0], 0.0, -steps[2], 0.0, 0.0, 0.0],
                [-steps[0], 0.0, steps[2], 0.0, 0.0, 0.0],
                // Lower exposure for bright scenes while keeping shadows open
                [-steps[0], 0.0, 0.0, steps[3], 0.0, 0.0],
                [steps[0], 0.0, 0.0, -steps[3], 0.0, 0.0],
                // Compress or expand dynamic range via Highlights + Shadows simultaneously
                [0.0, 0.0, -steps[2], steps[3], 0.0, 0.0],
                [0.0, 0.0, steps[2], -steps[3], 0.0, 0.0],
            ];
            for delta in coupled {
                let cand = clamp_params(std::array::from_fn(|i| best[i] + delta[i]));
                let c = cost(cand);
                if c + 1e-7 < best_cost {
                    best = cand;
                    best_cost = c;
                    improved = true;
                }
            }
        }
        for s in &mut steps {
            *s *= 0.5;
        }
    }

    let round2 = |v: f64| (v * 100.0).round() / 100.0;
    let round0 = |v: f64| v.round();
    out.light.exposure = round2(best[0]);
    out.light.contrast = round0(best[1]);
    out.light.highlights = round0(best[2]);
    out.light.shadows = round0(best[3]);
    out.light.whites = round0(best[4]);
    out.light.blacks = round0(best[5]);

    let report = SmartMatchReport {
        exposure: out.light.exposure,
        contrast: out.light.contrast,
        highlights: out.light.highlights,
        shadows: out.light.shadows,
        whites: out.light.whites,
        blacks: out.light.blacks,
        temp: out.wb.temp,
        tint: out.wb.tint,
    };
    (out, report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposure_follows_the_log_average_luminance() {
        let flat = |y: f32| Rgb32f::filled(64, 64, [y; 3]);
        let auto = |img: &Rgb32f| auto_tone(img, &SourceInfo::default(), &DevelopSettings::default());
        let log_mean = |img: &Rgb32f| scene_log_mean(img, &SourceInfo::default(), &DevelopSettings::default()).unwrap();
        // a scene at middle grey gets the exposure at grey; each EV darker adds 0.41 EV
        assert!(log_mean(&flat(0.18)).abs() < 1e-3);
        assert_eq!(auto(&flat(0.18)).exposure, -0.58);
        assert!((auto(&flat(0.18 / 8.0)).exposure - (-0.58 + 3.0 * 0.41)).abs() < 1e-9);
        // darker scenes get more, brighter ones less, within -2..3 EV
        let ramp = |lo: f32, step: f32| Rgb32f::from_fn(64, 64, |x, _| [lo + x as f32 * step; 3]);
        let (dark, bright) = (auto(&ramp(0.01, 0.0003)), auto(&ramp(0.8, 0.01)));
        assert!(dark.exposure > 0.5 && bright.exposure < -1.0, "{dark:?} {bright:?}");
        assert_eq!(auto(&flat(1e-9)).exposure, 3.0);
        assert_eq!(auto(&flat(1e6)).exposure, -2.0);
        // the rest is one recipe
        assert_eq!(
            (dark.contrast, dark.highlights, dark.shadows, dark.whites, dark.blacks, dark.vibrance, dark.saturation),
            (68.0, -66.0, 52.0, 0.0, 0.0, 30.0, 0.0)
        );
        assert_eq!((dark.contrast, dark.shadows), (bright.contrast, bright.shadows));
    }

    #[test]
    fn pixels_that_are_not_finite_are_left_out() {
        let auto = |img: &Rgb32f| auto_tone(img, &SourceInfo::default(), &DevelopSettings::default());
        // half middle grey, half NaN (or infinite): the grey half alone decides, as if the rest weren't there
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let half = Rgb32f::from_fn(64, 64, |x, _| if x < 32 { [0.18; 3] } else { [bad; 3] });
            assert_eq!(auto(&half).exposure, -0.58, "{bad}");
            // also through the resampling of a source larger than the proxy (proxy pixels on the border mix both halves)
            let big = Rgb32f::from_fn(1030, 700, |x, _| if x < 515 { [0.045; 3] } else { [bad; 3] });
            assert!((auto(&big).exposure - (-0.58 + 2.0 * 0.41)).abs() < 1e-9, "{bad}");
            // a few broken pixels scattered over a grey image change nothing
            let speckled = Rgb32f::from_fn(1030, 700, |x, y| if (x * 7 + y * 13) % 97 == 0 { [bad; 3] } else { [0.18; 3] });
            assert_eq!(auto(&speckled).exposure, -0.58, "{bad}");
        }
        // with no valid pixel at all (or none at all) there is nothing to measure: Exposure 0, the recipe as usual
        for img in [Rgb32f::filled(8, 8, [f32::NAN; 3]), Rgb32f::filled(600, 600, [f32::INFINITY; 3]), Rgb32f::new(0, 0)] {
            let a = auto(&img);
            assert_eq!((a.exposure, a.contrast, a.vibrance), (0.0, 68.0, 30.0));
        }
    }

    #[test]
    fn neutral_image_keeps_as_shot_wb() {
        let img = Rgb32f::from_fn(32, 32, |x, y| [0.05 + (x + y) as f32 * 0.004; 3]);
        let (t, tint) = auto_wb(&img, &SourceInfo::default());
        assert!((t - 6500.0).abs() < 150.0, "{t}");
        assert!(tint.abs() < 6.0, "{tint}");
    }

    #[test]
    fn blue_cast_is_corrected_by_higher_temp() {
        // A bluish cast should be neutralised by telling the pipeline the light was bluer (higher K).
        let img = Rgb32f::from_fn(32, 32, |_, _| [0.16, 0.18, 0.24]);
        let (t, _) = auto_wb(&img, &SourceInfo::default());
        assert!(t > 7000.0, "{t}");
        let mut s = DevelopSettings::default();
        s.wb.mode = lightcraft_develop::WbMode::Custom;
        s.wb.temp = t;
        let (t2, tint2) = auto_wb(&img, &SourceInfo::default());
        let _ = (t2, tint2);
        let mut out = img.clone();
        let (tt, ti) = auto_wb(&img, &SourceInfo::default());
        s.wb.temp = tt;
        s.wb.tint = ti;
        crate::local::scene_linear_pre(&mut out, &SourceInfo::default(), &s);
        let c = out.get(0, 0);
        assert!((c[0] - c[2]).abs() < 0.02, "{c:?}");
    }
}

#[cfg(test)]
mod bw_tests {
    use super::*;

    #[test]
    fn auto_bw_mix_pushes_light_and_dark_hues_apart() {
        // left: a light yellow, right: a dark blue (scene-linear Rec. 2020)
        let src = Rgb32f::from_fn(64, 32, |x, _| if x < 32 { [0.55, 0.5, 0.05] } else { [0.01, 0.02, 0.12] });
        let m = auto_bw_mix(&src, &SourceInfo::default(), &DevelopSettings::default());
        let (yellow, blue) = (m[2], m[5]);
        assert!(yellow > 0.0 && blue < 0.0, "{m:?}");
        // an image without colour leaves the mix alone
        let grey = Rgb32f::from_fn(16, 16, |x, _| [x as f32 / 16.0; 3]);
        assert_eq!(auto_bw_mix(&grey, &SourceInfo::default(), &DevelopSettings::default()), [0.0; 8]);
    }
}

#[cfg(test)]
mod tint_tests {
    use super::*;

    #[test]
    fn auto_wb_and_picker_correct_green_with_positive_tint() {
        // The picker delegates a sampled patch to this same auto_wb implementation.
        for (rgb, sign) in [([0.18, 0.24, 0.18], 1.0), ([0.24, 0.18, 0.24], -1.0)] {
            let img = Rgb32f::filled(16, 16, rgb);
            let info = SourceInfo { raw: true, relative_wb: true, ..Default::default() };
            let (temp, tint) = auto_wb(&img, &info);
            assert!(tint * sign > 0.0, "{rgb:?}: temp {temp}, tint {tint}");
            let mut s = DevelopSettings::default();
            s.wb.mode = lightcraft_develop::WbMode::Custom;
            s.wb.temp = temp;
            s.wb.tint = tint;
            let mut corrected = img;
            crate::local::white_balance(&mut corrected, &info, &s);
            let p = corrected.get(0, 0);
            let spread = p[0].max(p[1]).max(p[2]) - p[0].min(p[1]).min(p[2]);
            assert!(spread < 0.003, "{rgb:?} -> {p:?}");
        }
    }
}

#[cfg(test)]
mod smart_match_tests {
    use super::*;

    #[test]
    fn shadow_lifted_anchor_adapts_to_highlight_heavy_target_instead_of_copying_sliders() {
        let info = SourceInfo { raw: true, ..Default::default() };
        // Photo 1: dark shadows on the left (0.02), normal midtones on the right (0.24)
        let photo1 = Rgb32f::from_fn(96, 64, |x, y| {
            let t = x as f32 / 95.0;
            let v = if x < 48 { 0.018 + 0.02 * t } else { 0.18 + 0.08 * (y as f32 / 63.0) };
            [v, v, v]
        });
        let mut ref_edit = DevelopSettings::default();
        ref_edit.light.exposure = 0.35;
        ref_edit.light.shadows = 80.0;
        ref_edit.light.highlights = 0.0;
        ref_edit.light.contrast = 20.0;
        ref_edit.color.vibrance = 28.0;
        ref_edit.effects.clarity = 15.0;

        let anchor = StyleAnchor::from_reference(&photo1, &info, &ref_edit);

        // Photo 2: opposite lighting! Shadows are already open (0.14..0.22), and the right half
        // has harsh highlights (0.95..1.65) that need highlight recovery rather than shadow boost.
        let photo2 = Rgb32f::from_fn(96, 64, |x, y| {
            let t = y as f32 / 63.0;
            let v = if x < 48 { 0.13 + 0.07 * t } else { 0.95 + 0.70 * t };
            [v, v, v]
        });

        let (matched, report) = smart_match(&anchor, &photo2, &info, &DevelopSettings::default());

        // Unlike naive sync (which would keep exposure = +0.35, shadows = +80, highlights = 0),
        // smart_match lowers exposure, pulls highlights negative to rescue the bright half, and
        // reduces the unnecessary shadow lift.
        assert!(report.exposure < ref_edit.light.exposure, "expected lower exposure on bright target: {report:?}");
        assert!(report.highlights < -20.0, "expected negative highlights on highlight-heavy target: {report:?}");
        assert!(report.shadows < ref_edit.light.shadows - 20.0, "expected less shadow lift on target without deep shadows: {report:?}");

        // Creative style parameters transfer cleanly
        assert_eq!(matched.color.vibrance, 28.0);
        assert_eq!(matched.effects.clarity, 15.0);
    }
}
