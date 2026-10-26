//! Mask evaluation: each mask becomes an alpha plane at output resolution.
//!
//! Shapes are defined in normalized oriented-image coordinates; sizes are in "long-edge units".
//! Components combine with Add (max), Subtract (a·(1−c)) and Intersect (a·c).
//!
//! AI shapes (Sky, Subject, Background, People) use the segmentation [`Mattes`] the source carries
//! (DNG semantic masks, e.g. iPhone ProRAW's) when it has a matching one, else a heuristic.

use lightcraft_develop::{BrushStroke, LocalAdjustments, Mask, MaskOp, MaskShape, SegMask};
use lightcraft_geom::{Orientation, Point};
use lightcraft_raster::{Image, Plane, Rgb32f};

use crate::for_rows;
use crate::geometry::Frame;

/// What an embedded segmentation matte selects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatteKind {
    Sky,
    /// A whole person (all people in the photo).
    Person,
    Skin,
    Hair,
    Teeth,
    Glasses,
}

impl MatteKind {
    /// A stable code for serialization (e.g. a smart preview's stored mattes); never renumber.
    pub fn code(self) -> u8 {
        match self {
            MatteKind::Sky => 0,
            MatteKind::Person => 1,
            MatteKind::Skin => 2,
            MatteKind::Hair => 3,
            MatteKind::Teeth => 4,
            MatteKind::Glasses => 5,
        }
    }

    /// [`MatteKind::code`]'s inverse (`None` for a code this build doesn't know).
    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(MatteKind::Sky),
            1 => Some(MatteKind::Person),
            2 => Some(MatteKind::Skin),
            3 => Some(MatteKind::Hair),
            4 => Some(MatteKind::Teeth),
            5 => Some(MatteKind::Glasses),
            _ => None,
        }
    }
}

/// Segmentation mattes that came with the source: 0..255 alpha planes, each covering the whole
/// (EXIF-oriented) source image at its own resolution. Several mattes of one kind (e.g. one per
/// person) are combined with max.
#[derive(Clone, Default, PartialEq)]
pub struct Mattes {
    mattes: Vec<(MatteKind, Image<u8>)>,
}

impl std::fmt::Debug for Mattes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.mattes.iter().map(|(k, m)| (k, m.width, m.height))).finish()
    }
}

impl Mattes {
    /// Add a matte (empty ones are ignored).
    pub fn push(&mut self, kind: MatteKind, matte: Image<u8>) {
        if matte.width > 0 && matte.height > 0 && matte.data.len() == matte.width * matte.height {
            self.mattes.push((kind, matte));
        }
    }

    pub fn is_empty(&self) -> bool {
        self.mattes.is_empty()
    }

    /// Memory held (bytes).
    pub fn bytes(&self) -> usize {
        self.mattes.iter().map(|(_, m)| m.data.len()).sum()
    }

    /// The stored mattes, in insertion order (kind and plane), e.g. for storing them with a
    /// photo's proxy source.
    pub fn iter(&self) -> impl Iterator<Item = (MatteKind, &Image<u8>)> {
        self.mattes.iter().map(|(k, m)| (*k, m))
    }

    fn of(&self, kinds: &[MatteKind]) -> Vec<&Image<u8>> {
        self.mattes.iter().filter(|(k, _)| kinds.contains(k)).map(|(_, m)| m).collect()
    }

    /// The mattes a shape uses instead of its heuristic (empty: none fits).
    fn for_shape(&self, shape: &MaskShape) -> Vec<&Image<u8>> {
        use MatteKind::*;
        // the whole person: a person matte, else the union of the parts
        let person = || {
            let p = self.of(&[Person]);
            if p.is_empty() { self.of(&[Skin, Hair, Teeth, Glasses]) } else { p }
        };
        match shape {
            MaskShape::Sky => self.of(&[Sky]),
            MaskShape::Subject => person(),
            MaskShape::People { parts, .. } => {
                let wanted: Vec<MatteKind> = parts
                    .iter()
                    .filter_map(|p| {
                        let p = p.to_ascii_lowercase();
                        [("skin", Skin), ("hair", Hair), ("teeth", Teeth), ("glasses", Glasses)]
                            .into_iter()
                            .find(|(n, _)| p.contains(n))
                            .map(|(_, k)| k)
                    })
                    .collect();
                let whole =
                    parts.is_empty() || parts.iter().any(|p| matches!(p.to_ascii_lowercase().as_str(), "person" | "entire person" | "entireperson"));
                // parts without a matte (lips, clothes, …) keep the heuristic
                if whole { person() } else { self.of(&wanted) }
            }
            _ => Vec::new(),
        }
    }
}

pub struct Evaluated {
    /// The mask's id ([`Mask::id`]).
    pub id: u32,
    pub alpha: Plane,
    pub adjust: LocalAdjustments,
}

#[inline]
fn smooth(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The visible masks with components, evaluated in order (`mattes`: the source's, see [`Mattes`]).
#[allow(clippy::too_many_arguments)]
pub fn evaluate(masks: &[Mask], frame: &Frame, w: usize, h: usize, img: &Rgb32f, log_l: &Plane, ev: f32, mattes: Option<&Mattes>) -> Vec<Evaluated> {
    masks
        .iter()
        .filter(|m| m.visible && !m.components.is_empty())
        .map(|m| Evaluated { id: m.id, alpha: evaluate_one(m, frame, w, h, img, log_l, ev, mattes), adjust: m.adjust })
        .collect()
}

/// The alpha plane of mask `m` (whether visible or not): its components combined, inverted and
/// scaled by its amount.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_one(m: &Mask, frame: &Frame, w: usize, h: usize, img: &Rgb32f, log_l: &Plane, ev: f32, mattes: Option<&Mattes>) -> Plane {
    let mut alpha = Plane::new(w, h);
    let mut first = true;
    for comp in &m.components {
        let mut c = shape_alpha(&comp.shape, frame, w, h, img, log_l, ev, mattes);
        if comp.invert {
            c.data.iter_mut().for_each(|v| *v = 1.0 - *v);
        }
        if !matches!(comp.shape, MaskShape::Brush { .. })
            && (m.offset_x != 0.0 || m.offset_y != 0.0 || (m.scale - 100.0).abs() > 1e-4)
        {
            transform_plane(&mut c, m.offset_x, m.offset_y, m.scale, frame.output_long(w, h) as f32);
        }
        if first && comp.op != MaskOp::Intersect {
            alpha = c;
            first = false;
            continue;
        }
        for (a, c) in alpha.data.iter_mut().zip(&c.data) {
            *a = match comp.op {
                MaskOp::Add => a.max(*c),
                MaskOp::Subtract => *a * (1.0 - c),
                MaskOp::Intersect => *a * c,
            };
        }
        first = false;
    }
    if m.feather > 0.0 {
        let f = (m.feather / 100.0).clamp(0.0, 1.0) as f32;
        let sigma = (0.045 * frame.output_long(w, h) as f32 * f * f.sqrt()).max(0.5);
        smooth_plane(&mut alpha, sigma);
    }
    if m.refine > 0.0 {
        // Refine Edges: the mask's edges follow the photo's (window up to ~4 % of the long edge,
        // the width of a soft brush edge or gradient)
        let k = (m.refine / 100.0).clamp(0.0, 1.0) as f32;
        let sigma = (0.04 * frame.output_long(w, h) as f32 * k).max(1.0);
        let refined = guided_cross(log_l, &alpha, sigma, 0.02);
        for (a, r) in alpha.data.iter_mut().zip(&refined.data) {
            *a += (r - *a) * k.sqrt();
        }
    }
    if m.invert {
        alpha.data.iter_mut().for_each(|v| *v = 1.0 - *v);
    }
    let amt = (m.adjust.amount / 100.0) as f32;
    if (amt - 1.0).abs() > 1e-6 {
        alpha.data.iter_mut().for_each(|v| *v *= amt);
    }
    alpha
}

/// Per-pixel positions in long-edge units for output pixel centres.
fn for_each_pos(frame: &Frame, w: usize, h: usize, out: &mut Plane, f: impl Fn(Point, usize) -> f32 + Sync + Send) {
    let m = frame.out_to_norm(w, h);
    let (ow, oh, l) = (frame.ow, frame.oh, frame.ow.max(frame.oh));
    for_rows(&mut out.data, w, |y, row| {
        for (x, v) in row.iter_mut().enumerate() {
            let n = m.apply(Point::new(x as f64 + 0.5, y as f64 + 0.5));
            *v = f(Point::new(n.x * ow / l, n.y * oh / l), y * w + x);
        }
    });
}

/// The alpha plane of one component shape. AI shapes use the source's matching `mattes`, if any.
#[allow(clippy::too_many_arguments)]
pub fn shape_alpha(shape: &MaskShape, frame: &Frame, w: usize, h: usize, img: &Rgb32f, log_l: &Plane, ev: f32, mattes: Option<&Mattes>) -> Plane {
    let fitting = mattes.map(|m| m.for_shape(shape)).unwrap_or_default();
    if !fitting.is_empty() {
        return matte_alpha(&fitting, frame, w, h);
    }
    let cache_key = shape_cache_key(shape, frame, w, h, img, ev);
    if let Some(key) = cache_key
        && let Ok(guard) = SHAPE_ALPHA_CACHE.lock()
        && let Some((_, cached)) = guard.iter().find(|(k, _)| *k == key)
    {
        return cached.clone();
    }
    let mut out = Plane::new(w, h);
    let to_long = |p: Point| frame.norm_to_long(p);
    // `img`/`log_l` are before exposure: range masks select on the exposed values.
    let gain = ev.exp2();
    match shape {
        MaskShape::Linear { start, end } => {
            let (a, b) = (to_long(*start), to_long(*end));
            let d = b - a;
            let len2 = d.dot(d).max(1e-12);
            for_each_pos(frame, w, h, &mut out, |p, _| {
                let t = ((p - a).dot(d) / len2) as f32;
                1.0 - smooth(0.0, 1.0, t)
            });
        }
        MaskShape::Radial { center, rx, ry, angle, feather, invert } => {
            let c = to_long(*center);
            let (s, co) = (-angle.to_radians()).sin_cos();
            let f = (*feather / 100.0).clamp(0.0, 1.0) as f32;
            let (rx, ry) = (rx.max(1e-6), ry.max(1e-6));
            let inv = *invert;
            for_each_pos(frame, w, h, &mut out, |p, _| {
                let d = p - c;
                let (x, y) = (d.x * co - d.y * s, d.x * s + d.y * co);
                let r = ((x / rx).powi(2) + (y / ry).powi(2)).sqrt() as f32;
                let a = 1.0 - smooth(1.0 - f.max(0.01), 1.0, r);
                if inv { 1.0 - a } else { a }
            });
        }
        MaskShape::Brush { strokes } => rasterize_brush(strokes, frame, w, h, img, log_l, &mut out),
        MaskShape::LuminanceRange { lo, hi, lo_feather, hi_feather } => {
            let (lo, hi, lf, hf) = (*lo as f32, *hi as f32, (*lo_feather as f32).max(1e-3), (*hi_feather as f32).max(1e-3));
            for (v, l) in out.data.iter_mut().zip(&log_l.data) {
                // map log luminance to a 0..1 perceptual scale (−8..+4 EV)
                let y = ((l + ev + 8.0) / 12.0).clamp(0.0, 1.0);
                *v = smooth(lo - lf, lo, y) * (1.0 - smooth(hi, hi + hf, y));
            }
        }
        MaskShape::ColorRange { samples, refine } => {
            let tol = color_range_tolerance(*refine);
            let samples: Vec<[f32; 3]> = samples.iter().map(|s| [s[0] as f32, s[1] as f32, s[2] as f32]).collect();
            for (v, c) in out.data.iter_mut().zip(&img.data) {
                let lab = lightcraft_color::perceptual::oklab_from_2020(tonemap_for_select(c.map(|v| v * gain)));
                let d = samples
                    .iter()
                    .map(|s| ((lab[1] - s[1]).powi(2) + (lab[2] - s[2]).powi(2) + 0.25 * (lab[0] - s[0]).powi(2)).sqrt())
                    .fold(f32::MAX, f32::min);
                *v = 1.0 - smooth(tol * 0.5, tol, d);
            }
        }
        MaskShape::Sky => {
            // Classical sky heuristic until the segmenter lands (M12): bright, smooth, blue-ish or
            // unsaturated, and connected to the top of the frame.
            let m = frame.out_to_norm(w, h);
            for (i, v) in out.data.iter_mut().enumerate() {
                let (x, y) = (i % w, i / w);
                let n = m.apply(Point::new(x as f64 + 0.5, y as f64 + 0.5));
                let c = img.data[i].map(|v| v * gain);
                let l = log_l.data[i] + ev;
                let blue = (c[2] - c[0]).max(0.0) / (c[2] + 1e-4);
                let top = 1.0 - smooth(0.25, 0.7, n.y as f32);
                *v = top * smooth(-3.5, -1.0, l) * (0.4 + 0.6 * smooth(0.0, 0.3, blue).max(smooth(0.0, 1.5, l)));
            }
            smooth_plane(&mut out, 0.01 * frame_px(frame, w));
        }
        MaskShape::Object { seg, detail, edge, .. } | MaskShape::Prompt { seg, detail, edge, .. } if seg.is_some() || !detail.is_empty() => {
            // Edge: below 0 a steeper transition (up to 8× the logits), above 0 feathered (up to
            // 2 % of the long edge, so previews and exports match)
            let e = (*edge / 100.0).clamp(-1.0, 1.0) as f32;
            let gain = if e < 0.0 { 1.0 - 7.0 * e } else { 1.0 };
            sample_seg(seg.as_ref(), detail, gain, frame, w, h, &mut out);
            if e > 0.0 {
                smooth_plane(&mut out, e * 0.02 * frame_px(frame, w));
            }
        }
        // not computed yet (no clicks, or a build without the model): nothing selected
        MaskShape::Object { hint, .. } if hint.is_empty() => {}
        MaskShape::Prompt { .. } => {}
        MaskShape::People { parts, .. } if !parts.is_empty() && !parts.iter().any(|p| p.eq_ignore_ascii_case("entire_person") || p.eq_ignore_ascii_case("person")) => {
            let m = frame.out_to_norm(w, h);
            let has_part = |kw: &str| parts.iter().any(|p| p.to_ascii_lowercase().contains(kw));
            let is_sclera = has_part("sclera") || has_part("esclera");
            let is_eyebrows = has_part("eyebrow") || has_part("ceja") || has_part("sobrancelha");
            let is_eyes = !is_sclera && !is_eyebrows && (has_part("iris") || has_part("pupil") || has_part("eye"));
            let is_teeth = has_part("teeth") || has_part("diente") || has_part("dente");
            let is_lips = !is_teeth && has_part("lip");
            let is_beard = has_part("facial_hair") || has_part("beard") || has_part("barba");
            let is_hair = !is_beard && !is_eyebrows && (has_part("hair") || has_part("cabello") || has_part("cabelo"));
            let is_clothes = has_part("cloth") || has_part("ropa") || has_part("roupa");
            let is_face_only = has_part("face_skin") && !has_part("body_skin");
            let faces = detect_faces_cached(img, gain);
            if is_eyes {
                if !faces.is_empty() {
                    let blur_fine = lightcraft_raster::blur::gaussian(log_l, (0.0014 * frame_px(frame, w)).clamp(0.5, 1.5));
                    for face in &faces {
                        let (rx, ry) = (face.landmarks[0].0 * w as f32, face.landmarks[0].1 * h as f32);
                        let (lx, ly) = (face.landmarks[1].0 * w as f32, face.landmarks[1].1 * h as f32);
                        let eye_dist = ((lx - rx).hypot(ly - ry)).max(4.0);
                        let r_iris = (0.086 * eye_dist).max(2.0);
                        let r_search = (0.075 * eye_dist).max(2.5);
                        let (sin_t, cos_t) = (ly - ry).atan2(lx - rx).sin_cos();
                        for &(ex, ey) in &[(rx, ry), (lx, ly)] {
                            let (cx, cy, pupil_l, surround_l) = refine_eye_center(&blur_fine, w, h, ex, ey, r_iris, r_search, cos_t, sin_t);
                            let l_cutoff = (0.52 * pupil_l + 0.48 * surround_l).clamp(pupil_l + 0.65, pupil_l + 1.38);
                            let x0 = ((cx - r_iris * 1.25).floor() as isize).max(0) as usize;
                            let x1 = ((cx + r_iris * 1.25).ceil() as isize).min(w as isize - 1).max(0) as usize;
                            let y0 = ((cy - r_iris * 1.25).floor() as isize).max(0) as usize;
                            let y1 = ((cy + r_iris * 1.25).ceil() as isize).min(h as isize - 1).max(0) as usize;
                            for py in y0..=y1 {
                                for px in x0..=x1 {
                                    let idx = py * w + px;
                                    let dx = px as f32 + 0.5 - cx;
                                    let dy = py as f32 + 0.5 - cy;
                                    let ux = dx * cos_t + dy * sin_t;
                                    let uy = -dx * sin_t + dy * cos_t;
                                    let d_circle = (ux * ux + uy * uy).sqrt() / r_iris;
                                    let d_lid = ((ux * 0.96).powi(2) + (uy * 1.24).powi(2)).sqrt() / r_iris;
                                    if d_lid >= 1.10 {
                                        continue;
                                    }
                                    let pupil_ring = 0.28 + 0.72 * smooth(0.15, 0.34, d_circle);
                                    let outer_edge = 1.0 - smooth(0.70, 1.02, d_lid);
                                    let l_px = log_l.data[idx];
                                    let not_sclera = 1.0 - smooth(l_cutoff - 0.22, l_cutoff + 0.22, l_px);
                                    let w_iris = (pupil_ring * outer_edge * not_sclera).clamp(0.0, 1.0);
                                    if w_iris.is_finite() && w_iris > out.data[idx] {
                                        out.data[idx] = w_iris;
                                    }
                                }
                            }
                        }
                    }
                    smooth_plane(&mut out, (0.0010 * frame_px(frame, w)).clamp(0.4, 1.1));
                }
            } else if is_sclera {
                let blur_fine = lightcraft_raster::blur::gaussian(log_l, (0.0014 * frame_px(frame, w)).clamp(0.5, 1.5));
                for face in &faces {
                    let (rx, ry) = (face.landmarks[0].0 * w as f32, face.landmarks[0].1 * h as f32);
                    let (lx, ly) = (face.landmarks[1].0 * w as f32, face.landmarks[1].1 * h as f32);
                    let eye_dist = ((lx - rx).hypot(ly - ry)).max(4.0);
                    let r_iris = (0.086 * eye_dist).max(2.0);
                    let r_search = (0.075 * eye_dist).max(2.5);
                    let (sin_t, cos_t) = (ly - ry).atan2(lx - rx).sin_cos();
                    for &(ex, ey) in &[(rx, ry), (lx, ly)] {
                        let (cx, cy, pupil_l, _surround_l) = refine_eye_center(&blur_fine, w, h, ex, ey, r_iris, r_search, cos_t, sin_t);
                        let (rx_open, ry_open) = (r_iris * 2.05, r_iris * 0.96);
                        let x0 = ((cx - rx_open * 1.2).floor() as isize).max(0) as usize;
                        let x1 = ((cx + rx_open * 1.2).ceil() as isize).min(w as isize - 1).max(0) as usize;
                        let y0 = ((cy - ry_open * 1.2).floor() as isize).max(0) as usize;
                        let y1 = ((cy + ry_open * 1.2).ceil() as isize).min(h as isize - 1).max(0) as usize;
                        for py in y0..=y1 {
                            for px in x0..=x1 {
                                let idx = py * w + px;
                                let (dx, dy) = (px as f32 + 0.5 - cx, py as f32 + 0.5 - cy);
                                let ux = dx * cos_t + dy * sin_t;
                                let uy = -dx * sin_t + dy * cos_t;
                                let d_open = ((ux / rx_open).powi(2) + (uy / ry_open).powi(2)).sqrt();
                                if d_open >= 1.12 {
                                    continue;
                                }
                                let d_iris = (ux * ux + uy * uy).sqrt() / r_iris;
                                let outside_iris = smooth(0.84, 1.10, d_iris);
                                let inside_lid = 1.0 - smooth(0.68, 1.04, d_open);
                                let l_px = log_l.data[idx];
                                let bright = smooth(pupil_l + 0.55, pupil_l + 1.35, l_px);
                                let v = (outside_iris * inside_lid * bright).clamp(0.0, 1.0);
                                if v.is_finite() && v > out.data[idx] {
                                    out.data[idx] = v;
                                }
                            }
                        }
                    }
                }
                smooth_plane(&mut out, (0.0014 * frame_px(frame, w)).clamp(0.5, 1.4));
            } else if is_eyebrows {
                for face in &faces {
                    let (rx, ry) = (face.landmarks[0].0 * w as f32, face.landmarks[0].1 * h as f32);
                    let (lx, ly) = (face.landmarks[1].0 * w as f32, face.landmarks[1].1 * h as f32);
                    let eye_dist = ((lx - rx).hypot(ly - ry)).max(4.0);
                    let (sin_t, cos_t) = (ly - ry).atan2(lx - rx).sin_cos();
                    let (bx_r, by_r) = (0.34 * eye_dist, 0.11 * eye_dist);
                    for &(ex, ey) in &[(rx, ry), (lx, ly)] {
                        let (bcx, bcy) = (ex + sin_t * 0.24 * eye_dist, ey - cos_t * 0.24 * eye_dist);
                        let x0 = ((bcx - bx_r * 1.3).floor() as isize).max(0) as usize;
                        let x1 = ((bcx + bx_r * 1.3).ceil() as isize).min(w as isize - 1).max(0) as usize;
                        let y0 = ((bcy - bx_r * 1.3).floor() as isize).max(0) as usize;
                        let y1 = ((bcy + bx_r * 1.3).ceil() as isize).min(h as isize - 1).max(0) as usize;
                        for py in y0..=y1 {
                            for px in x0..=x1 {
                                let idx = py * w + px;
                                let (dx, dy) = (px as f32 + 0.5 - bcx, py as f32 + 0.5 - bcy);
                                let ux = (dx * cos_t + dy * sin_t) / bx_r;
                                let uy = (-dx * sin_t + dy * cos_t) / by_r;
                                let d = (ux * ux + uy * uy).sqrt();
                                if d < 1.2 {
                                    out.data[idx] = out.data[idx].max(1.0 - smooth(0.65, 1.15, d));
                                }
                            }
                        }
                    }
                }
                smooth_plane(&mut out, (0.002 * frame_px(frame, w)).max(0.6));
            } else if is_lips || is_teeth {
                for face in &faces {
                    let (mx0, my0) = (face.landmarks[3].0 * w as f32, face.landmarks[3].1 * h as f32);
                    let (mx1, my1) = (face.landmarks[4].0 * w as f32, face.landmarks[4].1 * h as f32);
                    let mw = ((mx1 - mx0).hypot(my1 - my0)).max(4.0);
                    let (sin_t, cos_t) = (my1 - my0).atan2(mx1 - mx0).sin_cos();
                    // Initial mouth center slightly below the corner line to account for smiling
                    let mut mcx = 0.5 * (mx0 + mx1) - sin_t * 0.035 * mw;
                    let mut mcy = 0.5 * (my0 + my1) + cos_t * 0.035 * mw;

                    // 1. Sample surrounding perioral skin (ring 1.15..1.65 around mouth) to get adaptive skin Lab baseline
                    let (rx_probe, ry_probe) = (0.56 * mw, 0.34 * mw);
                    let x0_p = ((mcx - rx_probe * 1.7).floor() as isize).max(0) as usize;
                    let x1_p = ((mcx + rx_probe * 1.7).ceil() as isize).min(w as isize - 1).max(0) as usize;
                    let y0_p = ((mcy - rx_probe * 1.7).floor() as isize).max(0) as usize;
                    let y1_p = ((mcy + rx_probe * 1.7).ceil() as isize).min(h as isize - 1).max(0) as usize;

                    let (mut skin_l, mut skin_a, mut skin_b, mut skin_n) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                    for py in (y0_p..=y1_p).step_by(2) {
                        for px in (x0_p..=x1_p).step_by(2) {
                            let (dx, dy) = (px as f32 + 0.5 - mcx, py as f32 + 0.5 - mcy);
                            let ux = (dx * cos_t + dy * sin_t) / rx_probe;
                            let uy = (-dx * sin_t + dy * cos_t) / ry_probe;
                            let d = (ux * ux + uy * uy).sqrt();
                            if (1.18..1.65).contains(&d) {
                                let lab = lightcraft_color::perceptual::oklab_from_2020(tonemap_for_select(img.data[py * w + px].map(|c| c * gain)));
                                skin_l += lab[0];
                                skin_a += lab[1];
                                skin_b += lab[2];
                                skin_n += 1.0;
                            }
                        }
                    }
                    if skin_n > 0.0 {
                        skin_l /= skin_n;
                        skin_a /= skin_n;
                        skin_b /= skin_n;
                    } else {
                        skin_l = 0.55;
                        skin_a = 0.032;
                        skin_b = 0.045;
                    }

                    // 2. Refine mouth center (mcx, mcy) to the actual redness-excess centroid of the lips
                    let (mut sum_wx, mut sum_wy, mut sum_w, mut max_da) = (0.0f32, 0.0f32, 0.0f32, 0.018f32);
                    for py in y0_p..=y1_p {
                        for px in x0_p..=x1_p {
                            let (dx, dy) = (px as f32 + 0.5 - mcx, py as f32 + 0.5 - mcy);
                            let ux = (dx * cos_t + dy * sin_t) / rx_probe;
                            let uy = (-dx * sin_t + dy * cos_t) / ry_probe;
                            let d = (ux * ux + uy * uy).sqrt();
                            if d < 0.95 {
                                let lab = lightcraft_color::perceptual::oklab_from_2020(tonemap_for_select(img.data[py * w + px].map(|c| c * gain)));
                                let da = (lab[1] - skin_a) - 0.20 * (lab[2] - skin_b).max(0.0);
                                if da > 0.008 {
                                    let wt = da * da * (1.0 - 0.5 * d);
                                    sum_wx += (px as f32 + 0.5) * wt;
                                    sum_wy += (py as f32 + 0.5) * wt;
                                    sum_w += wt;
                                    if da > max_da {
                                        max_da = da;
                                    }
                                }
                            }
                        }
                    }
                    if sum_w > 1e-5 {
                        mcx = 0.35 * mcx + 0.65 * (sum_wx / sum_w);
                        mcy = 0.25 * mcy + 0.75 * (sum_wy / sum_w);
                    }

                    // 3. Evaluate lips or teeth strictly inside the refined mouth contour with zero skin bleed
                    let (rx, ry) = if is_teeth { (0.36 * mw, 0.14 * mw) } else { (0.54 * mw, 0.30 * mw) };
                    let skin_chroma = skin_a.hypot(skin_b).max(0.025);
                    let x0 = ((mcx - rx * 1.25).floor() as isize).max(0) as usize;
                    let x1 = ((mcx + rx * 1.25).ceil() as isize).min(w as isize - 1).max(0) as usize;
                    let y0 = ((mcy - ry * 1.25).floor() as isize).max(0) as usize;
                    let y1 = ((mcy + ry * 1.25).ceil() as isize).min(h as isize - 1).max(0) as usize;
                    for py in y0..=y1 {
                        for px in x0..=x1 {
                            let idx = py * w + px;
                            let (dx, dy) = (px as f32 + 0.5 - mcx, py as f32 + 0.5 - mcy);
                            let ux = (dx * cos_t + dy * sin_t) / rx;
                            let uy = (-dx * sin_t + dy * cos_t) / ry;
                            let d = (ux * ux + uy * uy).sqrt();
                            if d >= 1.15 {
                                continue;
                            }
                            let spatial = 1.0 - smooth(0.68, 1.12, d);
                            let lab = lightcraft_color::perceptual::oklab_from_2020(tonemap_for_select(img.data[idx].map(|c| c * gain)));
                            if is_teeth {
                                let chroma = lab[1].hypot(lab[2]);
                                let low_red = 1.0 - smooth(skin_a * 0.55, skin_a * 0.85, lab[1]);
                                let low_chroma = 1.0 - smooth(skin_chroma * 0.58, skin_chroma * 0.82, chroma);
                                let bright = smooth(skin_l + 0.01, skin_l + 0.09, lab[0]);
                                out.data[idx] = out.data[idx].max(spatial * low_red * low_chroma * bright);
                            } else {
                                let da = (lab[1] - skin_a) - 0.20 * (lab[2] - skin_b).max(0.0);
                                let lip_pigment = smooth(0.18 * max_da, 0.56 * max_da, da);
                                out.data[idx] = out.data[idx].max(spatial * lip_pigment);
                            }
                        }
                    }
                }
                smooth_plane(&mut out, (0.0018 * frame_px(frame, w)).max(0.5));
            } else if is_beard {
                for face in &faces {
                    let (fx0, fy0, fx1, fy1) = (face.x0 * w as f32, face.y0 * h as f32, face.x1 * w as f32, face.y1 * h as f32);
                    let (fcx, fw, fh) = (0.5 * (fx0 + fx1), (fx1 - fx0).max(4.0), (fy1 - fy0).max(4.0));
                    let ny = face.landmarks[2].1 * h as f32;
                    let (mx0, my0) = (face.landmarks[3].0 * w as f32, face.landmarks[3].1 * h as f32);
                    let (mx1, my1) = (face.landmarks[4].0 * w as f32, face.landmarks[4].1 * h as f32);
                    let (mcx, mcy, mw) = (0.5 * (mx0 + mx1), 0.5 * (my0 + my1), ((mx1 - mx0).hypot(my1 - my0)).max(4.0));
                    for (i, v) in out.data.iter_mut().enumerate() {
                        let (px, py) = ((i % w) as f32 + 0.5, (i / w) as f32 + 0.5);
                        if py < ny || py > fy1 + 0.14 * fh {
                            continue;
                        }
                        let dx = (px - fcx) / (0.52 * fw);
                        let dy = (py - (ny + 0.55 * (fy1 - ny))) / (0.55 * (fy1 - ny).max(4.0));
                        let jaw = 1.0 - smooth(0.75, 1.15, (dx * dx + dy * dy).sqrt());
                        let d_mouth = (((px - mcx) / (0.52 * mw)).powi(2) + ((py - mcy) / (0.30 * mw)).powi(2)).sqrt();
                        let not_mouth = smooth(0.75, 1.15, d_mouth);
                        *v = v.max(jaw * not_mouth);
                    }
                }
                smooth_plane(&mut out, (0.004 * frame_px(frame, w)).max(0.8));
            } else if is_hair {
                for face in &faces {
                    let (fx0, fy0, fx1, fy1) = (face.x0 * w as f32, face.y0 * h as f32, face.x1 * w as f32, face.y1 * h as f32);
                    let (fcx, fcy, fw, fh) = (0.5 * (fx0 + fx1), 0.5 * (fy0 + fy1), (fx1 - fx0).max(4.0), (fy1 - fy0).max(4.0));
                    for (i, v) in out.data.iter_mut().enumerate() {
                        let (px, py) = ((i % w) as f32 + 0.5, (i / w) as f32 + 0.5);
                        let d_outer = (((px - fcx) / (0.85 * fw)).powi(2) + ((py - (fy0 + 0.25 * fh)) / (0.85 * fh)).powi(2)).sqrt();
                        if d_outer >= 1.2 {
                            continue;
                        }
                        let d_face = (((px - fcx) / (0.45 * fw)).powi(2) + ((py - (fcy + 0.06 * fh)) / (0.46 * fh)).powi(2)).sqrt();
                        let crown = (1.0 - smooth(0.75, 1.15, d_outer)) * smooth(0.78, 1.08, d_face);
                        *v = v.max(crown);
                    }
                }
                smooth_plane(&mut out, (0.005 * frame_px(frame, w)).max(1.0));
            } else if is_clothes {
                if !faces.is_empty() {
                    evaluate_portrait_subject_or_clothes(&faces, img, log_l, w, h, gain, frame, true, &mut out);
                }
            } else {
                // Skin tones (face_skin / body_skin) in OkLab, excluding detected eyes and lips
                for (i, v) in out.data.iter_mut().enumerate() {
                    let (px, py) = ((i % w) as f32 + 0.5, (i / w) as f32 + 0.5);
                    let lab = lightcraft_color::perceptual::oklab_from_2020(tonemap_for_select(img.data[i].map(|c| c * gain)));
                    let l_w = smooth(0.28, 0.42, lab[0]) * (1.0 - smooth(0.86, 0.95, lab[0]));
                    let a_w = smooth(0.006, 0.020, lab[1]) * (1.0 - smooth(0.080, 0.115, lab[1]));
                    let b_w = smooth(0.012, 0.028, lab[2]) * (1.0 - smooth(0.105, 0.140, lab[2]));
                    let mut skin_w = l_w * a_w * b_w;
                    if is_face_only && !faces.is_empty() {
                        let in_face = faces
                            .iter()
                            .map(|f| {
                                let (fx0, fy0, fx1, fy1) = (f.x0 * w as f32, f.y0 * h as f32, f.x1 * w as f32, f.y1 * h as f32);
                                let (fcx, fcy, fw, fh) = (0.5 * (fx0 + fx1), 0.5 * (fy0 + fy1), (fx1 - fx0).max(4.0), (fy1 - fy0).max(4.0));
                                let d = (((px - fcx) / (0.58 * fw)).powi(2) + ((py - fcy) / (0.62 * fh)).powi(2)).sqrt();
                                1.0 - smooth(0.78, 1.15, d)
                            })
                            .fold(0.0f32, f32::max);
                        skin_w *= in_face;
                    }
                    *v = skin_w;
                }
                for face in &faces {
                    let (rx, ry) = (face.landmarks[0].0 * w as f32, face.landmarks[0].1 * h as f32);
                    let (lx, ly) = (face.landmarks[1].0 * w as f32, face.landmarks[1].1 * h as f32);
                    let r_eye = (0.15 * ((lx - rx).hypot(ly - ry))).max(3.0);
                    for &(ex, ey) in &[(rx, ry), (lx, ly)] {
                        let x0 = ((ex - r_eye * 1.4).floor() as isize).max(0) as usize;
                        let x1 = ((ex + r_eye * 1.4).ceil() as isize).min(w as isize - 1).max(0) as usize;
                        let y0 = ((ey - r_eye * 1.4).floor() as isize).max(0) as usize;
                        let y1 = ((ey + r_eye * 1.4).ceil() as isize).min(h as isize - 1).max(0) as usize;
                        for py in y0..=y1 {
                            for px in x0..=x1 {
                                let d = (px as f32 + 0.5 - ex).hypot(py as f32 + 0.5 - ey) / r_eye;
                                out.data[py * w + px] *= smooth(0.7, 1.2, d);
                            }
                        }
                    }
                    let (mx0, my0) = (face.landmarks[3].0 * w as f32, face.landmarks[3].1 * h as f32);
                    let (mx1, my1) = (face.landmarks[4].0 * w as f32, face.landmarks[4].1 * h as f32);
                    let (mcx, mcy, mw) = (0.5 * (mx0 + mx1), 0.5 * (my0 + my1), ((mx1 - mx0).hypot(my1 - my0)).max(4.0));
                    let x0 = ((mcx - mw).floor() as isize).max(0) as usize;
                    let x1 = ((mcx + mw).ceil() as isize).min(w as isize - 1).max(0) as usize;
                    let y0 = ((mcy - mw * 0.6).floor() as isize).max(0) as usize;
                    let y1 = ((mcy + mw * 0.6).ceil() as isize).min(h as isize - 1).max(0) as usize;
                    for py in y0..=y1 {
                        for px in x0..=x1 {
                            let d = (((px as f32 + 0.5 - mcx) / (0.6 * mw)).powi(2) + ((py as f32 + 0.5 - mcy) / (0.38 * mw)).powi(2)).sqrt();
                            out.data[py * w + px] *= smooth(0.65, 1.15, d);
                        }
                    }
                }
                smooth_plane(&mut out, (0.005 * frame_px(frame, w)).max(1.0));
            }
            let _ = m;
        }
        MaskShape::Subject | MaskShape::Object { .. } | MaskShape::People { .. } => {
            let faces = detect_faces_cached(img, gain);
            if !faces.is_empty() {
                evaluate_portrait_subject_or_clothes(&faces, img, log_l, w, h, gain, frame, false, &mut out);
            } else {
                // Saliency + border background color contrast with edge-preserving guided filter
                let m = frame.out_to_norm(w, h);
                let blur = lightcraft_raster::blur::gaussian(log_l, 0.03 * frame_px(frame, w));
                let mut border_labs = Vec::with_capacity(64);
                let step_x = (w / 12).max(1);
                let step_y = (h / 12).max(1);
                for bx in (0..w).step_by(step_x) {
                    let c_top = img.data[bx].map(|v| v * gain);
                    border_labs.push(lightcraft_color::perceptual::oklab_from_2020(tonemap_for_select(c_top)));
                }
                for by in (0..h).step_by(step_y) {
                    let c_l = img.data[by * w].map(|v| v * gain);
                    let c_r = img.data[by * w + (w - 1)].map(|v| v * gain);
                    border_labs.push(lightcraft_color::perceptual::oklab_from_2020(tonemap_for_select(c_l)));
                    border_labs.push(lightcraft_color::perceptual::oklab_from_2020(tonemap_for_select(c_r)));
                }
                for (i, v) in out.data.iter_mut().enumerate() {
                    let (x, y) = (i % w, i / w);
                    let n = m.apply(Point::new(x as f64 + 0.5, y as f64 + 0.5));
                    let d = (((n.x - 0.5) / 0.36).powi(2) + ((n.y - 0.54) / 0.42).powi(2)).sqrt() as f32;
                    let contrast = (log_l.data[i] - blur.data[i]).abs();
                    let lab = lightcraft_color::perceptual::oklab_from_2020(tonemap_for_select(img.data[i].map(|c| c * gain)));
                    let d_bg = border_labs
                        .iter()
                        .map(|b| ((lab[1] - b[1]).powi(2) + (lab[2] - b[2]).powi(2) + 0.25 * (lab[0] - b[0]).powi(2)).sqrt())
                        .fold(f32::MAX, f32::min);
                    let bg_diff = smooth(0.02, 0.08, d_bg);
                    *v = (1.0 - smooth(0.62, 1.04, d)) * (0.35 + 0.65 * bg_diff.max(smooth(0.05, 0.55, contrast)));
                }
                let sigma = (0.010 * frame_px(frame, w)).clamp(2.0, 14.0);
                out = guided_cross(log_l, &out, sigma, 0.004);
                for v in &mut out.data {
                    *v = smooth(0.22, 0.78, *v);
                }
            }
        }
        MaskShape::Background => {
            let mut s = shape_alpha(&MaskShape::Subject, frame, w, h, img, log_l, ev, mattes);
            s.data.iter_mut().for_each(|v| *v = 1.0 - *v);
            out = s;
        }
        MaskShape::DepthRange { .. } | MaskShape::Landscape { .. } => {}
    }
    if let Some(key) = cache_key
        && let Ok(mut guard) = SHAPE_ALPHA_CACHE.lock()
    {
        if let Some(pos) = guard.iter().position(|(k, _)| *k == key) {
            guard[pos] = (key, out.clone());
        } else {
            if guard.len() >= 16 {
                guard.remove(0);
            }
            guard.push((key, out.clone()));
        }
    }
    out
}

static SHAPE_ALPHA_CACHE: std::sync::Mutex<Vec<(u64, Plane)>> = std::sync::Mutex::new(Vec::new());

fn shape_cache_key(shape: &MaskShape, frame: &Frame, w: usize, h: usize, img: &Rgb32f, ev: f32) -> Option<u64> {
    use std::hash::{Hash, Hasher};
    let mut hsh = std::collections::hash_map::DefaultHasher::new();
    match shape {
        MaskShape::Sky => 1u8.hash(&mut hsh),
        MaskShape::Subject => 2u8.hash(&mut hsh),
        MaskShape::Background => 3u8.hash(&mut hsh),
        MaskShape::People { person, parts } => {
            4u8.hash(&mut hsh);
            person.hash(&mut hsh);
            parts.hash(&mut hsh);
        }
        _ => return None,
    }
    w.hash(&mut hsh);
    h.hash(&mut hsh);
    ev.to_bits().hash(&mut hsh);
    for bits in frame.out_to_norm(w, h).0.map(|v| v.to_bits()) {
        bits.hash(&mut hsh);
    }
    // Sample 16 pixels across `img` so distinct photos or white-balance changes produce a distinct key in O(1)
    let n = img.data.len();
    if n > 0 {
        for k in 0..16 {
            let idx = (k * (n.saturating_sub(1))) / 15;
            let c = img.data[idx];
            c[0].to_bits().hash(&mut hsh);
            c[1].to_bits().hash(&mut hsh);
            c[2].to_bits().hash(&mut hsh);
        }
    }
    Some(hsh.finish())
}

#[derive(Clone, Copy)]
struct ColorSample {
    nx: f32,
    ny: f32,
    lab: [f32; 3],
}

#[allow(clippy::too_many_arguments)]
fn evaluate_portrait_subject_or_clothes(
    faces: &[lightcraft_faces::yunet::Face],
    img: &Rgb32f,
    log_l: &Plane,
    w: usize,
    h: usize,
    gain: f32,
    frame: &Frame,
    clothes_only: bool,
    out: &mut Plane,
) {
    if w < 4 || h < 4 || faces.is_empty() {
        return;
    }
    let wf = w as f32;
    let hf = h as f32;
    let lab_at = |px: usize, py: usize| -> [f32; 3] {
        let c = img.data[py.min(h - 1) * w + px.min(w - 1)].map(|v| v * gain);
        lightcraft_color::perceptual::oklab_from_2020(tonemap_for_select(c))
    };

    // Local high-frequency texture / detail energy (`|log_l - blur_l|`) to distinguish textured foreground
    // (silver/gray hair curls, tiaras, lace blouses, fabric folds) from smooth painted walls/curtains.
    let blur_l = lightcraft_raster::blur::gaussian(log_l, (0.0055 * frame_px(frame, w)).clamp(1.2, 5.5));

    // Illumination-aware garment/color distance in OkLab:
    // For saturated fabrics of the SAME hue (e.g. a red ballgown with bright lit tulle and deep crimson folds),
    // OkLab (a, b) scales roughly proportionally with lightness L. Only apply shading normalization when both
    // samples are distinctly chromatic (`ca > 0.068 && cb > 0.068`) and share the same hue angle (`cos_hue > 0.94`)
    // so dark brown stair railings or warm wall shadows are never confused with a red dress.
    let garment_color_dist = |a: [f32; 3], b: [f32; 3]| -> f32 {
        let dc_raw = ((a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2) + 0.22 * (a[0] - b[0]).powi(2)).sqrt();
        let ca = a[1].hypot(a[2]);
        let cb = b[1].hypot(b[2]);
        if ca > 0.068 && cb > 0.068 {
            let cos_hue = (a[1] * b[1] + a[2] * b[2]) / (ca * cb);
            if cos_hue > 0.94 {
                let la = a[0].clamp(0.26, 0.78);
                let lb = b[0].clamp(0.26, 0.78);
                let da_n = a[1] / la - b[1] / lb;
                let db_n = a[2] / la - b[2] / lb;
                let dc_norm = 0.48 * (da_n * da_n + db_n * db_n + 0.16 * (a[0] - b[0]).powi(2)).sqrt();
                return dc_raw.min(dc_norm);
            }
        }
        dc_raw
    };

    // 1. Sample upper/outer background reference colors outside all person head & shoulder shields.
    let mut bg_samples: Vec<ColorSample> = Vec::with_capacity(140);
    let max_bg_y = faces
        .iter()
        .map(|f| {
            let fy0 = f.y0 * hf;
            let fy1 = f.y1 * hf;
            let fh = (fy1 - fy0).max(4.0);
            fy1 + 0.95 * fh
        })
        .fold(0.30 * hf, f32::max)
        .min(0.62 * hf);

    let inside_any_person_shield = |px: f32, py: f32| -> bool {
        faces.iter().any(|f| {
            let (fx0, fy0, fx1, fy1) = (f.x0 * wf, f.y0 * hf, f.x1 * wf, f.y1 * hf);
            let (fcx, fcy, fw, fh) = (0.5 * (fx0 + fx1), 0.5 * (fy0 + fy1), (fx1 - fx0).max(4.0), (fy1 - fy0).max(4.0));
            let d_head = (((px - fcx) / (1.40 * fw)).powi(2) + ((py - (fcy - 0.06 * fh)) / (1.42 * fh)).powi(2)).sqrt();
            if d_head < 1.0 {
                return true;
            }
            if py >= fy1 - 0.20 * fh {
                let t_down = ((py - fy1).max(0.0) / (hf - fy1).max(fh)).clamp(0.0, 1.0);
                let half_w = fw * (2.25 + 6.50 * t_down.sqrt());
                if (px - fcx).abs() < half_w {
                    return true;
                }
            }
            false
        })
    };

    let gx_steps = 24usize;
    let gy_steps = 16usize;
    for gy in 0..gy_steps {
        let py = ((gy as f32 + 0.5) / gy_steps as f32 * hf).clamp(0.0, hf - 1.0);
        for gx in 0..gx_steps {
            let px = ((gx as f32 + 0.5) / gx_steps as f32 * wf).clamp(0.0, wf - 1.0);
            if py <= max_bg_y && !inside_any_person_shield(px, py) {
                bg_samples.push(ColorSample {
                    nx: px / wf,
                    ny: py / hf,
                    lab: lab_at(px as usize, py as usize),
                });
            }
        }
    }
    if bg_samples.len() < 6 {
        for &(nx, ny) in &[(0.02f32, 0.02f32), (0.98, 0.02), (0.02, 0.16), (0.98, 0.16), (0.50, 0.02), (0.02, 0.35), (0.98, 0.35)] {
            bg_samples.push(ColorSample {
                nx,
                ny,
                lab: lab_at((nx * (wf - 1.0)) as usize, (ny * (hf - 1.0)) as usize),
            });
        }
    }

    // Use the 2nd-smallest distance to `bg_samples` so a single outlier pixel (e.g. one red flower
    // on a background plant) never poisons the entire background distance field.
    let robust_bg_dist_list = |samples: &[ColorSample], nx: f32, ny: f32, lab: [f32; 3]| -> f32 {
        let mut best1 = f32::MAX;
        let mut best2 = f32::MAX;
        for s in samples {
            let dc = ((lab[1] - s.lab[1]).powi(2) + (lab[2] - s.lab[2]).powi(2) + 0.26 * (lab[0] - s.lab[0]).powi(2)).sqrt();
            let ds = ((nx - s.nx).powi(2) + (ny - s.ny).powi(2)).sqrt();
            let d = dc + 0.038 * ds;
            if d < best1 {
                best2 = best1;
                best1 = d;
            } else if d < best2 {
                best2 = d;
            }
        }
        if samples.len() >= 4 { 0.45 * best1 + 0.55 * best2 } else { best1 }
    };

    // 2. Stage A: Build per-person appearance model from face skin & anatomical torso/skirt column
    struct PersonAppearance {
        fcx: f32,
        fcy: f32,
        fw: f32,
        fh: f32,
        fy1: f32,
        left_mid_x: f32,
        right_mid_x: f32,
        skin_lab: [f32; 3],
        neck_skin_lab: [f32; 3],
        skin_chroma: f32,
        hair_samples: Vec<[f32; 3]>,
        cloth_samples: Vec<ColorSample>,
        dominant_saturated_cloth: bool,
    }

    // Highlight-compensated & shadow-tolerant skin probability:
    // Captures face skin, bright window-lit shoulders/collarbones/escote (`L` up to 0.965),
    // and tanned/shaded forearms (`L` down to 0.20), while strictly rejecting saturated non-skin colors.
    let person_skin_prob = |lab: [f32; 3], skin_lab: [f32; 3], neck_skin_lab: [f32; 3], skin_chroma: f32| -> f32 {
        let l_gate = smooth(0.18, 0.28, lab[0]) * (1.0 - smooth(0.955, 0.990, lab[0]));
        if l_gate <= 1e-4 {
            return 0.0;
        }
        // Reject vivid red / magenta / blue / green non-skin hues
        let a_gate = smooth(0.001, 0.007, lab[1]) * (1.0 - smooth(0.070, 0.086, lab[1]));
        let b_gate = smooth(0.003, 0.011, lab[2]) * (1.0 - smooth(0.098, 0.128, lab[2]));
        let hue_ratio_gate = smooth(0.50 * lab[1], 0.74 * lab[1], lab[2]);
        let chroma = lab[1].hypot(lab[2]);
        let hi_scale = if lab[0] > skin_lab[0] {
            ((1.01 - lab[0]) / (1.01 - skin_lab[0]).max(0.10)).clamp(0.20, 1.0)
        } else {
            1.0
        };
        let min_c = (skin_chroma * 0.32 * hi_scale).clamp(0.006, 0.024);
        let max_c = (skin_chroma * 1.85).max(0.106);
        let chroma_gate = smooth(min_c * 0.65, min_c * 1.10, chroma) * (1.0 - smooth(max_c, max_c + 0.024, chroma));
        let locus = l_gate * a_gate * b_gate * hue_ratio_gate * chroma_gate;
        if locus <= 1e-4 {
            return 0.0;
        }
        let d_face = ((lab[1] - skin_lab[1]).powi(2) + (lab[2] - skin_lab[2]).powi(2) + 0.020 * (lab[0] - skin_lab[0]).powi(2)).sqrt();
        let d_neck = ((lab[1] - neck_skin_lab[1]).powi(2) + (lab[2] - neck_skin_lab[2]).powi(2) + 0.020 * (lab[0] - neck_skin_lab[0]).powi(2)).sqrt();
        let exp_a = skin_lab[1] * hi_scale;
        let exp_b = skin_lab[2] * hi_scale;
        let d_hi = ((lab[1] - exp_a).powi(2) + (lab[2] - exp_b).powi(2) + 0.012 * (lab[0] - skin_lab[0]).max(0.0).powi(2)).sqrt();
        let d_skin = d_face.min(d_neck).min(d_hi);
        locus * (1.0 - smooth(0.036, 0.072, d_skin))
    };

    let mut people = Vec::with_capacity(faces.len());
    for (idx, f) in faces.iter().enumerate() {
        let (fx0, fy0, fx1, fy1) = (f.x0 * wf, f.y0 * hf, f.x1 * wf, f.y1 * hf);
        let (fcx, fcy, fw, fh) = (0.5 * (fx0 + fx1), 0.5 * (fy0 + fy1), (fx1 - fx0).max(4.0), (fy1 - fy0).max(4.0));

        // Compute horizontal Voronoi bounds between adjacent people so Person B's blouse/clothing
        // model does not cross over into Person A's bare chest or bare arm.
        let mut left_mid_x = 0.0f32;
        let mut right_mid_x = wf;
        for (j, other) in faces.iter().enumerate() {
            if j == idx {
                continue;
            }
            let ocx = 0.5 * (other.x0 + other.x1) * wf;
            let mid = 0.5 * (fcx + ocx);
            if ocx < fcx - 0.25 * fw {
                left_mid_x = left_mid_x.max(mid);
            } else if ocx > fcx + 0.25 * fw {
                right_mid_x = right_mid_x.min(mid);
            }
        }

        // Sample face skin on cheeks, forehead, and nose bridge (avoiding lips and eyes)
        let mut skin_pts = Vec::with_capacity(12);
        for &(dx, dy) in &[
            (-0.24f32, 0.05f32),
            (0.24, 0.05),
            (-0.18, 0.14),
            (0.18, 0.14),
            (0.0, -0.22),
            (-0.15, -0.22),
            (0.15, -0.22),
            (0.0, -0.02),
        ] {
            let px = (fcx + dx * fw).clamp(0.0, wf - 1.0) as usize;
            let py = (fcy + dy * fh).clamp(0.0, hf - 1.0) as usize;
            let lab = lab_at(px, py);
            if lab[0] > 0.28 && lab[1] > 0.004 && lab[1] < 0.075 && lab[2] > 0.008 {
                skin_pts.push(lab);
            }
        }
        let skin_lab = if !skin_pts.is_empty() {
            let n = skin_pts.len() as f32;
            let sl: f32 = skin_pts.iter().map(|c| c[0]).sum();
            let sa: f32 = skin_pts.iter().map(|c| c[1]).sum();
            let sb: f32 = skin_pts.iter().map(|c| c[2]).sum();
            [sl / n, sa / n, sb / n]
        } else {
            lab_at(fcx.clamp(0.0, wf - 1.0) as usize, fcy.clamp(0.0, hf - 1.0) as usize)
        };
        let skin_chroma = skin_lab[1].hypot(skin_lab[2]).max(0.026);

        // Also check neck/upper-chest skin directly below the chin (`fy1 + 0.15..0.55 * fh`)
        // to capture bright key-lit collarbone/escote skin tones when present.
        let mut neck_pts = Vec::with_capacity(6);
        for &(dx, dy_after_chin) in &[(0.0f32, 0.20f32), (-0.18, 0.28), (0.18, 0.28), (0.0, 0.42), (-0.24, 0.48), (0.24, 0.48)] {
            let px = (fcx + dx * fw).clamp(0.0, wf - 1.0) as usize;
            let py = (fy1 + dy_after_chin * fh).clamp(0.0, hf - 1.0) as usize;
            let lab = lab_at(px, py);
            if lab[0] > 0.35 && lab[0] < 0.96 && lab[1] > 0.004 && lab[1] < 0.068 && lab[2] >= 0.60 * lab[1] {
                neck_pts.push(lab);
            }
        }
        let neck_skin_lab = if !neck_pts.is_empty() {
            let n = neck_pts.len() as f32;
            let sl: f32 = neck_pts.iter().map(|c| c[0]).sum();
            let sa: f32 = neck_pts.iter().map(|c| c[1]).sum();
            let sb: f32 = neck_pts.iter().map(|c| c[2]).sum();
            [sl / n, sa / n, sb / n]
        } else {
            skin_lab
        };

        let mut hair_samples = Vec::with_capacity(24);
        for iy in 0..6 {
            let py = (fy0 + (-0.32 + iy as f32 * 0.18) * fh).clamp(0.0, hf - 1.0);
            for ix in 0..7 {
                let px = (fcx + (-0.66 + ix as f32 * 0.22) * fw).clamp(0.0, wf - 1.0);
                let d_face = (((px - fcx) / (0.44 * fw)).powi(2) + ((py - (fcy + 0.05 * fh)) / (0.46 * fh)).powi(2)).sqrt();
                if d_face > 0.72 {
                    let lab = lab_at(px as usize, py as usize);
                    if robust_bg_dist_list(&bg_samples, px / wf, py / hf, lab) > 0.032 {
                        hair_samples.push(lab);
                    }
                }
            }
        }

        // Sample garment fabric strictly below the décolletage/neck (`fy1 + 0.78 * fh` when torso is visible)
        // down to the lower skirt/pants column (`fy1 + 7.2 * fh`), respecting horizontal neighbor boundaries.
        let mut cloth_samples = Vec::with_capacity(96);
        let y_start = if hf > fy1 + 1.35 * fh {
            (fy1 + 0.78 * fh).min(hf - 2.0)
        } else {
            (fy1 + 0.35 * fh).min(hf - 2.0)
        };
        let y_end = (fy1 + 7.2 * fh).min(hf - 2.0);
        if y_end > y_start + 2.0 {
            for iy in 0..18 {
                let ty = (iy as f32 + 0.5) / 18.0;
                let py = y_start + ty * (y_end - y_start);
                let col_half_w = fw * (0.72 + 1.15 * ty);
                for ix in 0..9 {
                    let tx = (ix as f32 - 4.0) / 4.0;
                    let px = (fcx + tx * col_half_w).clamp(0.0, wf - 1.0);
                    if py < fy1 + 2.5 * fh && (px < left_mid_x + 0.10 * fw || px > right_mid_x - 0.10 * fw) {
                        continue;
                    }
                    let lab = lab_at(px as usize, py as usize);
                    let skin_w = person_skin_prob(lab, skin_lab, neck_skin_lab, skin_chroma);
                    if skin_w >= 0.14 {
                        continue;
                    }
                    let d_bg = robust_bg_dist_list(&bg_samples, px / wf, py / hf, lab);
                    let in_guaranteed_torso = py >= fy1 + 0.95 * fh && py <= fy1 + 4.2 * fh && tx.abs() <= 0.60;
                    if d_bg > 0.028 || (in_guaranteed_torso && d_bg > 0.014) {
                        cloth_samples.push(ColorSample { nx: px / wf, ny: py / hf, lab });
                    }
                }
            }
            if cloth_samples.len() < 4 {
                for iy in 0..5 {
                    let py = (fy1 + (1.10 + iy as f32 * 0.45) * fh).clamp(0.0, hf - 1.0);
                    for ix in 0..5 {
                        let px = (fcx + (ix as f32 - 2.0) * 0.22 * fw).clamp(0.0, wf - 1.0);
                        let lab = lab_at(px as usize, py as usize);
                        if person_skin_prob(lab, skin_lab, neck_skin_lab, skin_chroma) < 0.20 {
                            cloth_samples.push(ColorSample { nx: px / wf, ny: py / hf, lab });
                        }
                    }
                }
            }
        }

        // If the garment is predominantly a vivid/saturated color (e.g. a red ballgown where >= 40% of samples
        // have chroma > 0.072), prune minority pastel/cream samples (such as 3D floral appliqués on the bodice
        // or sheer skin-colored mesh) from `cloth_samples` so they never cause bare skin to match `cloth_samples`.
        let sat_count = cloth_samples.iter().filter(|s| s.lab[1].hypot(s.lab[2]) > 0.072).count();
        let dominant_saturated_cloth = cloth_samples.len() >= 6 && sat_count * 10 >= cloth_samples.len() * 4;
        if dominant_saturated_cloth {
            cloth_samples.retain(|s| s.lab[1].hypot(s.lab[2]) > 0.058);
            // Expand `cloth_samples` across wide ballgown trains (e.g. cascading across stairs to the frame edge)
            // so shaded folds on the far left/right of the dress have local `(nx, ny, lab)` references.
            let core_samples = cloth_samples.clone();
            for gy in 0..10 {
                let ny = ((fy1 / hf) + (gy as f32 + 0.5) / 10.0 * (1.0 - fy1 / hf)).clamp(0.05, 0.98);
                let py = (ny * (hf - 1.0)).clamp(0.0, hf - 1.0);
                for gx in 0..16 {
                    let nx = (gx as f32 + 0.5) / 16.0;
                    let px = (nx * (wf - 1.0)).clamp(0.0, wf - 1.0);
                    let lab = lab_at(px as usize, py as usize);
                    let ch = lab[1].hypot(lab[2]);
                    if ch > 0.055
                        && core_samples.iter().any(|cs| {
                            let c_cs = cs.lab[1].hypot(cs.lab[2]).max(1e-5);
                            let cos_hue = (lab[1] * cs.lab[1] + lab[2] * cs.lab[2]) / (ch * c_cs);
                            garment_color_dist(lab, cs.lab) < 0.058 || (cos_hue > 0.90 && (lab[0] - cs.lab[0]).abs() < 0.28)
                        })
                    {
                        cloth_samples.push(ColorSample { nx, ny, lab });
                    }
                }
            }
        }

        people.push(PersonAppearance {
            fcx,
            fcy,
            fw,
            fh,
            fy1,
            left_mid_x,
            right_mid_x,
            skin_lab,
            neck_skin_lab,
            skin_chroma,
            hair_samples,
            cloth_samples,
            dominant_saturated_cloth,
        });
    }

    // 2b. Stage B: Clean `bg_samples` by removing any accidental background sample (such as a single red flower
    // on a plant behind a subject or a ballgown train reaching the border) that matches a person's garment color,
    // then augment `bg_samples` with lower-left / lower-right frame border samples ONLY where the border does NOT
    // match any person's garment or skin.
    let matches_person_garment = |lab: [f32; 3], p: &PersonAppearance| -> bool {
        let ch = lab[1].hypot(lab[2]);
        p.cloth_samples.iter().any(|cs| {
            let d = garment_color_dist(lab, cs.lab);
            if p.dominant_saturated_cloth {
                let c_cs = cs.lab[1].hypot(cs.lab[2]).max(1e-5);
                let cos_hue = if ch > 1e-5 {
                    (lab[1] * cs.lab[1] + lab[2] * cs.lab[2]) / (ch * c_cs)
                } else {
                    0.0
                };
                d < 0.078 || (ch > 0.042 && cos_hue > 0.86)
            } else {
                d < 0.065
            }
        })
    };

    if bg_samples.len() > 16 {
        bg_samples.retain(|bg| !people.iter().any(|p| p.dominant_saturated_cloth && matches_person_garment(bg.lab, p)));
    }
    for gy in 5..16 {
        let py = ((gy as f32 + 0.5) / 16.0 * hf).clamp(0.0, hf - 1.0);
        for &nx in &[0.025f32, 0.06, 0.94, 0.975] {
            let px = (nx * (wf - 1.0)).clamp(0.0, wf - 1.0);
            let lab = lab_at(px as usize, py as usize);
            let matches_any_person = people.iter().any(|p| {
                person_skin_prob(lab, p.skin_lab, p.neck_skin_lab, p.skin_chroma) > 0.16
                    || matches_person_garment(lab, p)
            });
            if !matches_any_person {
                bg_samples.push(ColorSample { nx, ny: py / hf, lab });
            }
        }
    }

    let min_bg_dist = |nx: f32, ny: f32, lab: [f32; 3]| -> f32 { robust_bg_dist_list(&bg_samples, nx, ny, lab) };

    // 3. Evaluate per-pixel probability in parallel across rows
    for_rows(&mut out.data, w, |y, row| {
        let py = y as f32 + 0.5;
        let ny = py / hf;
        for (x, v) in row.iter_mut().enumerate() {
            let px = x as f32 + 0.5;
            let nx = px / wf;
            let mut best = 0.0f32;
            let mut lab_opt: Option<[f32; 3]> = None;
            let mut d_bg_opt: Option<f32> = None;

            // Compute whether this pixel is bare skin (face, neck, bare chest/escote, bare shoulder, or bare arm)
            // on ANY detected person so that in `clothes_only` mode, bare skin is strictly excluded even when
            // two people stand side-by-side.
            let mut any_person_skin = 0.0f32;
            if clothes_only {
                for p in &people {
                    let d_face_oval = (((px - p.fcx) / (0.56 * p.fw)).powi(2) + ((py - (p.fcy + 0.02 * p.fh)) / (0.62 * p.fh)).powi(2)).sqrt();
                    if d_face_oval < 0.95 {
                        any_person_skin = 1.0;
                        break;
                    }
                    let t_down = ((py - p.fy1).max(0.0) / (hf - p.fy1).max(p.fh)).clamp(0.0, 1.0);
                    let skin_reach_x = p.fw * (1.65 + 1.25 * t_down);
                    if py >= p.fcy - 0.65 * p.fh && py <= p.fy1 + 4.2 * p.fh && (px - p.fcx).abs() <= skin_reach_x {
                        let lab = *lab_opt.get_or_insert_with(|| {
                            let c = img.data[y * w + x].map(|c| c * gain);
                            lightcraft_color::perceptual::oklab_from_2020(tonemap_for_select(c))
                        });
                        let mut sp = person_skin_prob(lab, p.skin_lab, p.neck_skin_lab, p.skin_chroma);
                        // Boost skin exclusion in the anatomical neck & bare décolletage (`escote`) zone right below the chin
                        if py >= p.fy1 - 0.10 * p.fh && py <= p.fy1 + 0.82 * p.fh && (px - p.fcx).abs() <= 0.95 * p.fw && sp > 0.12 {
                            sp = (sp * 1.85).clamp(0.0, 1.0);
                        }
                        if sp > any_person_skin {
                            any_person_skin = sp;
                        }
                    }
                }
                if any_person_skin > 0.88 {
                    *v = 0.0;
                    continue;
                }
            }

            for p in &people {
                // Tight anatomical face oval (`0.42 * fw × 0.50 * fh`) so background walls next to the head are never touched
                let d_face_core = (((px - p.fcx) / (0.42 * p.fw)).powi(2) + ((py - (p.fcy + 0.02 * p.fh)) / (0.50 * p.fh)).powi(2)).sqrt();
                if !clothes_only && d_face_core < 0.80 {
                    best = 1.0;
                    break;
                }

                let t_down = ((py - p.fy1).max(0.0) / (hf - p.fy1).max(p.fh)).clamp(0.0, 1.0);
                // Allow wide ballgown trains (`dominant_saturated_cloth`) to reach across the entire lower frame
                let dress_half_w = if p.dominant_saturated_cloth {
                    (p.fw * (2.20 + 18.0 * t_down.sqrt())).max(wf * 0.95 * t_down.sqrt().max(0.35))
                } else {
                    p.fw * (1.85 + 5.80 * t_down.sqrt())
                };
                let dx_dress = (px - p.fcx).abs() / dress_half_w;
                let d_head_halo = (((px - p.fcx) / (0.96 * p.fw)).powi(2) + ((py - (p.fcy - 0.06 * p.fh)) / (1.06 * p.fh)).powi(2)).sqrt();

                let in_cloth_zone = py >= p.fy1 + 0.08 * p.fh && dx_dress < 1.08;
                let in_head_zone = !clothes_only && d_head_halo < 1.12;
                let in_skin_body_zone = !clothes_only && py >= p.fy1 - 0.14 * p.fh && (px - p.fcx).abs() < p.fw * (1.70 + 1.25 * t_down);

                if !in_cloth_zone && !in_head_zone && !in_skin_body_zone {
                    continue;
                }

                let lab = *lab_opt.get_or_insert_with(|| {
                    let c = img.data[y * w + x].map(|c| c * gain);
                    lightcraft_color::perceptual::oklab_from_2020(tonemap_for_select(c))
                });
                let d_bg = *d_bg_opt.get_or_insert_with(|| min_bg_dist(nx, ny, lab));
                let skin_w = person_skin_prob(lab, p.skin_lab, p.neck_skin_lab, p.skin_chroma);
                let tex = (log_l.data[y * w + x] - blur_l.data[y * w + x]).abs();

                // --- A) Clothing / Dress probability (`w_cloth`) ---
                let mut w_cloth = 0.0f32;
                if in_cloth_zone && !p.cloth_samples.is_empty() {
                    let not_face = smooth(0.86, 1.14, d_face_core);
                    let neighbor_gate = if clothes_only && py < p.fy1 + 2.4 * p.fh && (!p.dominant_saturated_cloth || lab[1].hypot(lab[2]) < 0.065) {
                        smooth(p.left_mid_x - 0.25 * p.fw, p.left_mid_x + 0.20 * p.fw, px)
                            * (1.0 - smooth(p.right_mid_x - 0.20 * p.fw, p.right_mid_x + 0.25 * p.fw, px))
                    } else {
                        1.0
                    };

                    let eff_skin = if clothes_only {
                        skin_w.max(any_person_skin)
                    } else {
                        skin_w
                    };
                    let not_skin = (1.0 - eff_skin * 1.35).clamp(0.0, 1.0);

                    if not_face > 1e-3 && not_skin > 1e-3 && neighbor_gate > 1e-3 {
                        let d_cloth = p
                            .cloth_samples
                            .iter()
                            .map(|s| {
                                let dc = garment_color_dist(lab, s.lab);
                                let ds = ((nx - s.nx).powi(2) + (ny - s.ny).powi(2)).sqrt();
                                dc + 0.008 * ds
                            })
                            .fold(0.14f32, f32::min);

                        let env_dress = (1.0 - smooth(0.86, 1.08, dx_dress)) * smooth(p.fy1 + 0.08 * p.fh, p.fy1 + 0.28 * p.fh, py) * not_face * neighbor_gate;
                        let cloth_match = 1.0 - smooth(0.048, 0.115, d_cloth);
                        // Also boost lace/embroidered blouses when they match `cloth_samples` and have strong weave texture (`tex`)
                        let tex_boost = if !p.dominant_saturated_cloth && d_cloth < 0.042 {
                            smooth(0.06, 0.18, tex) * 0.35
                        } else {
                            0.0
                        };
                        let fg_vs_bg = smooth(-0.012, 0.022, d_bg - d_cloth + 0.015 * tex_boost)
                            * smooth(0.018, 0.040, d_bg + 0.018 * tex_boost);

                        w_cloth = env_dress * not_skin * cloth_match * fg_vs_bg;
                    }
                }

                if clothes_only {
                    if w_cloth > best {
                        best = w_cloth;
                    }
                } else {
                    // --- B) Full Subject = Head/Hair + Bare Skin (neck, chest, shoulders, arms) + Full Clothing/Dress ---
                    let mut w_person = w_cloth;

                    let d_skin_face = ((lab[1] - p.skin_lab[1]).powi(2) + (lab[2] - p.skin_lab[2]).powi(2) + 0.025 * (lab[0] - p.skin_lab[0]).powi(2)).sqrt();
                    let d_skin_neck = ((lab[1] - p.neck_skin_lab[1]).powi(2) + (lab[2] - p.neck_skin_lab[2]).powi(2) + 0.025 * (lab[0] - p.neck_skin_lab[0]).powi(2)).sqrt();
                    let d_skin = d_skin_face.min(d_skin_neck);
                    let d_hair = p
                        .hair_samples
                        .iter()
                        .map(|h_lab| ((lab[1] - h_lab[1]).powi(2) + (lab[2] - h_lab[2]).powi(2) + 0.24 * (lab[0] - h_lab[0]).powi(2)).sqrt())
                        .fold(0.09f32, f32::min);

                    if in_head_zone {
                        let face_soft = 1.0 - smooth(0.76, 1.02, d_face_core);
                        // Crown hair (including silver/gray curls) is selected inside the crown oval ONLY where
                        // it either differs from the background wall (`d_bg`) OR has curl/strand texture (`tex`),
                        // never blindly selecting the smooth wall next to the head!
                        let d_crown = (((px - p.fcx) / (0.64 * p.fw)).powi(2) + ((py - (p.fcy - 0.10 * p.fh)) / (0.72 * p.fh)).powi(2)).sqrt();
                        let crown_env = 1.0 - smooth(0.75, 1.05, d_crown);
                        let crown_fg = smooth(0.018, 0.042, d_bg).max(smooth(0.07, 0.18, tex));
                        let halo_env = 1.0 - smooth(0.74, 1.08, d_head_halo);
                        let d_head_fg = d_skin.min(d_hair);
                        let head_fg_vs_bg = smooth(-0.008, 0.024, d_bg - d_head_fg) * smooth(0.020, 0.046, d_bg);
                        let w_head = face_soft
                            .max(crown_env * crown_fg.max(skin_w))
                            .max(halo_env * head_fg_vs_bg.max(skin_w * smooth(0.016, 0.038, d_bg)));
                        if w_head > w_person {
                            w_person = w_head;
                        }
                    }

                    if in_skin_body_zone {
                        let arm_half_w = p.fw * (1.65 + 1.25 * t_down);
                        let skin_env = (1.0 - smooth(0.78, 1.05, (px - p.fcx).abs() / arm_half_w))
                            * smooth(p.fy1 - 0.14 * p.fh, p.fy1 + 0.06 * p.fh, py);
                        // Strictly require the pixel to differ from the background wall (`d_bg`) or be inside the tight neck column
                        // so flat background walls next to the waist/shoulders NEVER get a halo!
                        let not_bg_wall = smooth(0.016, 0.038, d_bg);
                        let skin_vs_bg = smooth(-0.014, 0.020, d_bg - d_skin) * not_bg_wall;
                        let w_skin = skin_env * skin_w * skin_vs_bg;

                        // Tight anatomical neck column directly under the chin (`±0.36 * fw`), gated by `not_bg_wall`
                        let neck_col = (1.0 - smooth(0.32, 0.56, (px - p.fcx).abs() / p.fw))
                            * smooth(p.fy1 - 0.06 * p.fh, p.fy1 + 0.12 * p.fh, py)
                            * (1.0 - smooth(p.fy1 + 0.72 * p.fh, p.fy1 + 1.05 * p.fh, py))
                            * not_bg_wall;

                        w_person = w_person.max(w_skin).max(neck_col * skin_w.max(0.75));
                    }

                    if w_person > best {
                        best = w_person;
                    }
                }
            }

            *v = best.clamp(0.0, 1.0);
        }
    });

    // 3b. Topological 2D Exterior-Border Flood-Fill + Background-Safe Interior Closure:
    // Flood-fill "Exterior Background" from the top, left, and right borders of the frame through pixels
    // where `out < 0.52`. Any pocket that is completely surrounded by the subject/garment (`!exterior`,
    // such as white 3D flowers on a red bodice, a bare forearm crossing in front of the red skirt, or lace inside
    // a blouse) cannot be reached from the exterior frame border and is filled cleanly — while the exterior
    // wall right next to the neck, waist, and head (`exterior == true`) is 100% untouched (ZERO halo!).
    let step = (w.max(h) / 180).clamp(2, 8);
    let gw = w.div_ceil(step);
    let gh = h.div_ceil(step);
    if gw >= 8 && gh >= 8 {
        let mut barrier = vec![false; gw * gh];
        for gy in 0..gh {
            let y0 = gy * step;
            let y1 = ((gy + 1) * step).min(h);
            for gx in 0..gw {
                let x0 = gx * step;
                let x1 = ((gx + 1) * step).min(w);
                let mut max_a = 0.0f32;
                for py in y0..y1 {
                    for px in x0..x1 {
                        let a = out.data[py * w + px];
                        if a > max_a {
                            max_a = a;
                        }
                    }
                }
                barrier[gy * gw + gx] = max_a >= 0.54;
            }
        }

        let mut exterior = vec![false; gw * gh];
        let mut queue = std::collections::VecDeque::with_capacity(gw * 4 + gh * 4);
        let try_seed = |gx: usize, gy: usize, exterior: &mut [bool], queue: &mut std::collections::VecDeque<(usize, usize)>| {
            let idx = gy * gw + gx;
            if !barrier[idx] && !exterior[idx] {
                exterior[idx] = true;
                queue.push_back((gx, gy));
            }
        };
        for gx in 0..gw {
            try_seed(gx, 0, &mut exterior, &mut queue);
        }
        for gy in 0..gh {
            try_seed(0, gy, &mut exterior, &mut queue);
            try_seed(gw - 1, gy, &mut exterior, &mut queue);
        }
        while let Some((cx, cy)) = queue.pop_front() {
            for (dx, dy) in [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)] {
                let nx = cx as isize + dx;
                let ny = cy as isize + dy;
                if nx >= 0 && nx < gw as isize && ny >= 0 && ny < gh as isize {
                    let (ux, uy) = (nx as usize, ny as usize);
                    let nidx = uy * gw + ux;
                    if !barrier[nidx] && !exterior[nidx] {
                        exterior[nidx] = true;
                        queue.push_back((ux, uy));
                    }
                }
            }
        }

        // Fill enclosed interior islands (`!exterior`) that lie inside a person's body/garment silhouette
        for_rows(&mut out.data, w, |y, row| {
            let gy = (y / step).min(gh - 1);
            let py = y as f32 + 0.5;
            for (x, v) in row.iter_mut().enumerate() {
                if *v >= 0.88 {
                    continue;
                }
                let gx = (x / step).min(gw - 1);
                if !exterior[gy * gw + gx] && !barrier[gy * gw + gx] {
                    let px = x as f32 + 0.5;
                    if clothes_only {
                        let c = img.data[y * w + x].map(|c| c * gain);
                        let lab = lightcraft_color::perceptual::oklab_from_2020(tonemap_for_select(c));
                        let is_skin = people.iter().any(|p| {
                            py <= p.fy1 + 3.2 * p.fh
                                && (px - p.fcx).abs() <= 1.8 * p.fw
                                && person_skin_prob(lab, p.skin_lab, p.neck_skin_lab, p.skin_chroma) > 0.20
                        });
                        if !is_skin {
                            *v = (*v).max(0.94);
                        }
                    } else {
                        *v = (*v).max(0.95);
                    }
                }
            }
        });

        // Also bridge narrow vertical/horizontal foreground gaps (such as a bare forearm crossing the dress
        // or thin stair bars over the red skirt) ONLY when the pixel is NOT an exterior background wall pixel!
        let max_fw = people.iter().map(|p| p.fw).fold(12.0f32, f32::max);
        let max_fh = people.iter().map(|p| p.fh).fold(12.0f32, f32::max);
        let max_gap_x = ((if clothes_only { 0.65 } else { 1.10 }) * max_fw).round() as usize;
        let max_gap_y = ((if clothes_only { 0.65 } else { 1.35 }) * max_fh).round() as usize;

        if max_gap_y >= 3 && h > 6 {
            let prev = out.data.clone();
            for x in 0..w {
                let px = x as f32 + 0.5;
                let nx = px / wf;
                let mut y = 0usize;
                while y < h {
                    if prev[y * w + x] >= 0.72 {
                        let top_y = y;
                        let top_v = prev[y * w + x];
                        y += 1;
                        while y < h && prev[y * w + x] < 0.72 && (y - top_y) <= max_gap_y {
                            y += 1;
                        }
                        if y < h && prev[y * w + x] >= 0.72 && (y - top_y) > 1 && (y - top_y) <= max_gap_y {
                            let bot_y = y;
                            let bot_v = prev[y * w + x];
                            let fill_v = top_v.min(bot_v) * 0.94;
                            for gy_px in (top_y + 1)..bot_y {
                                let py = gy_px as f32 + 0.5;
                                let ny = py / hf;
                                let c = img.data[gy_px * w + x].map(|c| c * gain);
                                let lab = lightcraft_color::perceptual::oklab_from_2020(tonemap_for_select(c));
                                let d_bg = min_bg_dist(nx, ny, lab);
                                // Never bridge over background wall (`d_bg < 0.026` and low skin/texture)
                                let skin_any = people
                                    .iter()
                                    .map(|p| person_skin_prob(lab, p.skin_lab, p.neck_skin_lab, p.skin_chroma))
                                    .fold(0.0f32, f32::max);
                                if clothes_only {
                                    if skin_any > 0.20 || d_bg < 0.028 {
                                        continue;
                                    }
                                } else if d_bg < 0.024 && skin_any < 0.25 {
                                    continue;
                                }
                                if fill_v > out.data[gy_px * w + x] {
                                    out.data[gy_px * w + x] = fill_v;
                                }
                            }
                        }
                    } else {
                        y += 1;
                    }
                }
            }
        }

        if max_gap_x >= 3 && w > 6 {
            let prev = out.data.clone();
            for_rows(&mut out.data, w, |y, row| {
                let py = y as f32 + 0.5;
                let ny = py / hf;
                let mut x = 0usize;
                while x < w {
                    if prev[y * w + x] >= 0.72 {
                        let left_x = x;
                        let left_v = prev[y * w + x];
                        x += 1;
                        while x < w && prev[y * w + x] < 0.72 && (x - left_x) <= max_gap_x {
                            x += 1;
                        }
                        if x < w && prev[y * w + x] >= 0.72 && (x - left_x) > 1 && (x - left_x) <= max_gap_x {
                            let right_x = x;
                            let right_v = prev[y * w + x];
                            let fill_v = left_v.min(right_v) * 0.94;
                            for gx_px in (left_x + 1)..right_x {
                                let px = gx_px as f32 + 0.5;
                                let nx = px / wf;
                                let c = img.data[y * w + gx_px].map(|c| c * gain);
                                let lab = lightcraft_color::perceptual::oklab_from_2020(tonemap_for_select(c));
                                let d_bg = min_bg_dist(nx, ny, lab);
                                let skin_any = people
                                    .iter()
                                    .map(|p| person_skin_prob(lab, p.skin_lab, p.neck_skin_lab, p.skin_chroma))
                                    .fold(0.0f32, f32::max);
                                if clothes_only {
                                    if skin_any > 0.20 || d_bg < 0.028 {
                                        continue;
                                    }
                                } else if d_bg < 0.024 && skin_any < 0.25 {
                                    continue;
                                }
                                if fill_v > row[gx_px] {
                                    row[gx_px] = fill_v;
                                }
                            }
                        }
                    } else {
                        x += 1;
                    }
                }
            });
        }
    }

    // 4. Snap mask boundaries to real image edges using edge-preserving guided filtering on log_l
    let sigma = ((if clothes_only { 0.007 } else { 0.008 }) * frame_px(frame, w)).clamp(1.5, 11.0);
    *out = guided_cross(log_l, out, sigma, 0.002);
    for v in &mut out.data {
        let vc = if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 };
        *v = smooth(0.22, 0.78, vc);
    }
    smooth_plane(out, (0.0018 * frame_px(frame, w)).clamp(0.6, 1.6));
}

/// A decoded [`SegMask`]: its logits and the normalized rectangle they cover.
struct SegGrid {
    logits: Vec<f32>,
    side: usize,
    r: [f64; 4],
}

impl SegGrid {
    fn new(seg: &SegMask) -> Option<SegGrid> {
        Some(SegGrid { logits: seg.logits()?, side: seg.side as usize, r: seg.bounds()? })
    }

    fn contains(&self, n: Point) -> bool {
        n.x >= self.r[0] && n.x <= self.r[2] && n.y >= self.r[1] && n.y <= self.r[3]
    }

    /// The bilinearly sampled logit at normalized image point `n` (cell centres at
    /// `(i + 0.5) / side` of the rectangle).
    fn logit(&self, n: Point) -> f32 {
        let side = self.side;
        let at = |x: usize, y: usize| self.logits.get(y.min(side - 1) * side + x.min(side - 1)).copied().unwrap_or(-16.0);
        let u = (n.x - self.r[0]) / (self.r[2] - self.r[0]);
        let v = (n.y - self.r[1]) / (self.r[3] - self.r[1]);
        let (fx, fy) = ((u * side as f64 - 0.5) as f32, (v * side as f64 - 0.5) as f32);
        let (fx, fy) = (fx.clamp(0.0, (side - 1) as f32), fy.clamp(0.0, (side - 1) as f32));
        let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
        let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
        let top = at(x0, y0) * (1.0 - tx) + at(x0 + 1, y0) * tx;
        let bot = at(x0, y0 + 1) * (1.0 - tx) + at(x0 + 1, y0 + 1) * tx;
        top * (1.0 - ty) + bot * ty
    }
}

/// Stored model segmentations at each output pixel: the zoomed-in `detail` passes inside
/// their rectangles (the highest of them where they overlap), the whole-photo `seg`
/// elsewhere; the logit (times `gain`: > 1 is a harder edge) through a sigmoid. Damaged
/// data selects nothing.
fn sample_seg(seg: Option<&SegMask>, detail: &[SegMask], gain: f32, frame: &Frame, w: usize, h: usize, out: &mut Plane) {
    let coarse = seg.and_then(SegGrid::new);
    // (capped: a hostile document can hold many)
    let patches: Vec<SegGrid> = detail.iter().take(lightcraft_develop::segmask::MAX_DETAIL).filter_map(SegGrid::new).collect();
    if coarse.is_none() && patches.is_empty() {
        return;
    }
    let m = frame.out_to_norm(w, h);
    for_rows(&mut out.data, w, |y, row| {
        for (x, v) in row.iter_mut().enumerate() {
            let n = m.apply(Point::new(x as f64 + 0.5, y as f64 + 0.5));
            let fine = patches.iter().filter(|p| p.contains(n)).map(|p| p.logit(n)).reduce(f32::max);
            let l = fine.or_else(|| coarse.as_ref().map(|c| c.logit(n))).unwrap_or(-16.0);
            *v = 1.0 / (1.0 + (-l * gain).exp());
        }
    });
}

fn frame_px(frame: &Frame, w: usize) -> f32 {
    frame.px_per_long(w) as f32
}

fn smooth_plane(p: &mut Plane, sigma: f32) {
    *p = lightcraft_raster::blur::gaussian(p, sigma.max(0.5));
}

/// The union (max) of `mattes` (0..255, over the EXIF-oriented source) at output resolution,
/// sampled like the image: user orientation, lens / perspective warp, crop and flips; area-averaged
/// first when the output is much smaller than the matte, bilinear otherwise.
fn matte_alpha(mattes: &[&Image<u8>], frame: &Frame, w: usize, h: usize) -> Plane {
    let mut out = Plane::new(w, h);
    let o2t = frame.out_to_oriented(w, h);
    let ppl = frame.px_per_long(w).max(1e-9);
    for m in mattes {
        let oriented =
            if frame.orient == Orientation::Normal { std::borrow::Cow::Borrowed(*m) } else { std::borrow::Cow::Owned(m.oriented(frame.orient)) };
        // matte px per output px
        let k = oriented.width as f64 / frame.ow * frame.ow.max(frame.oh) / ppl;
        let base = box_down(&oriented, if k >= 2.0 { (k.floor() as usize).min(64) } else { 1 });
        let (sx, sy) = ((base.width as f64 / frame.ow) as f32, (base.height as f64 / frame.oh) as f32);
        for_rows(&mut out.data, w, |y, row| {
            for (x, v) in row.iter_mut().enumerate() {
                let s = match &frame.warp {
                    Some(wp) => {
                        let Some((_, s)) = wp.frame(&o2t, x, y) else {
                            continue;
                        };
                        s
                    }
                    None => o2t.apply(Point::new(x as f64 + 0.5, y as f64 + 0.5)),
                };
                *v = v.max(bilinear(&base, s.x as f32 * sx, s.y as f32 * sy) / 255.0);
            }
        });
    }
    out
}

/// `m` area-averaged by `f × f` (0..255 values kept); `f == 1` converts only.
fn box_down(m: &Image<u8>, f: usize) -> Plane {
    let f = f.max(1);
    let (bw, bh) = (m.width.div_ceil(f).max(1), m.height.div_ceil(f).max(1));
    let mut out = Plane::new(bw, bh);
    for_rows(&mut out.data, bw, |by, row| {
        let rows = by * f..((by + 1) * f).min(m.height);
        for (bx, v) in row.iter_mut().enumerate() {
            let cols = bx * f..((bx + 1) * f).min(m.width);
            let (mut sum, mut n) = (0u32, 0u32);
            for y in rows.clone() {
                for x in cols.clone() {
                    sum += m.data.get(y * m.width + x).copied().unwrap_or(0) as u32;
                    n += 1;
                }
            }
            *v = sum as f32 / n.max(1) as f32;
        }
    });
    out
}

/// Bilinear sample at continuous pixel coordinates (pixel centres at +0.5), edges extended.
fn bilinear(p: &Plane, x: f32, y: f32) -> f32 {
    // (held a pixel outside, which reads the same edge: an infinite coordinate reads the edge
    // instead of a NaN weight and a neighbour index past isize::MAX)
    let (fx, fy) = (x.clamp(-1.0, p.width as f32 + 1.0) - 0.5, y.clamp(-1.0, p.height as f32 + 1.0) - 0.5);
    let (x0, y0) = (fx.floor(), fy.floor());
    let (tx, ty) = (fx - x0, fy - y0);
    let (x0, y0) = (x0 as isize, y0 as isize);
    let a = p.get_clamped(x0, y0) + (p.get_clamped(x0 + 1, y0) - p.get_clamped(x0, y0)) * tx;
    let b = p.get_clamped(x0, y0 + 1) + (p.get_clamped(x0 + 1, y0 + 1) - p.get_clamped(x0, y0 + 1)) * tx;
    a + (b - a) * ty
}

/// A rough display mapping for colour picking (so samples taken on screen match).
pub(crate) fn tonemap_for_select(c: [f32; 3]) -> [f32; 3] {
    c.map(|v| v / (1.0 + v))
}

/// Colour range: the OkLab distance (a, b at full weight, L at a quarter) at which a pixel leaves
/// the selection for a Refine amount of 0..100. A pixel is fully selected up to half of it and
/// fades out over the other half (CPU here, `mask.wgsl` kind 3; both take this value).
///
/// Issue #157: `0.04 + 0.16·refine` selected the whole photo. Through [`tonemap_for_select`] a
/// slightly teal mid-grey (sRGB 120,150,155) and a neutral light grey (200,198,195) are only ~0.063
/// apart — L 0.60 vs 0.72, a/b ≤ 0.02 — so at the default Refine 50 (full selection below 0.06)
/// the grey patch was 99 % selected, and at Refine 85 / 100 still most of it. A JND is ~0.01 on
/// this scale, so Refine 50 now holds a surface up to ~2 JND from its sample and drops everything
/// beyond ~4.5 (full to 0.0225, gone at 0.045); Refine 85 is gone at 0.066, past that grey patch;
/// Refine 100 reaches 0.075 and Refine 0 is the sample's own shade (full to 0.0075, gone at 0.015).
pub fn color_range_tolerance(refine: f64) -> f32 {
    0.015 + 0.06 * (refine / 100.0).clamp(0.0, 1.0) as f32
}

fn detect_faces_cached(img: &Rgb32f, gain: f32) -> Vec<lightcraft_faces::yunet::Face> {
    use std::sync::{Mutex, OnceLock};
    static DETECTOR: OnceLock<Option<lightcraft_faces::yunet::Detector>> = OnceLock::new();
    // Cache entries: (resolution_independent_sig, detected_long_edge_px, faces)
    static CACHE: Mutex<Vec<(u64, usize, Vec<lightcraft_faces::yunet::Face>)>> = Mutex::new(Vec::new());

    let (w, h) = (img.width, img.height);
    if w < 16 || h < 16 || img.data.len() != w * h {
        return Vec::new();
    }
    let long_px = w.max(h);
    // Compute a resolution-independent signature over an 8×8 normalized patch grid so draft renders
    // during slider drags can reuse the high-resolution 2-stage landmarks without re-running YuNet,
    // while higher-resolution renders automatically upgrade any earlier draft detection.
    let aspect_q = ((w as f64 / h as f64) * 1000.0).round() as u64;
    let mut sig = aspect_q.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    for gy in 0..8 {
        let y0 = (gy * h) / 8;
        let y1 = (((gy + 1) * h) / 8).max(y0 + 1).min(h);
        let ym = (y0 + y1) / 2;
        for gx in 0..8 {
            let x0 = (gx * w) / 8;
            let x1 = (((gx + 1) * w) / 8).max(x0 + 1).min(w);
            let xm = (x0 + x1) / 2;
            let c = img.data[ym * w + xm];
            let r_q = (c[0].clamp(0.0, 4.0) * 32.0).round() as u64;
            let g_q = (c[1].clamp(0.0, 4.0) * 32.0).round() as u64;
            let b_q = (c[2].clamp(0.0, 4.0) * 32.0).round() as u64;
            let cell_bits = r_q ^ (g_q << 11) ^ (b_q << 22);
            sig = sig.wrapping_mul(0x517c_c1b7_2722_0a95) ^ cell_bits;
        }
    }
    if let Ok(guard) = CACHE.lock()
        && let Some((_, cached_long, faces)) = guard.iter().find(|(k, _, _)| *k == sig)
        && (*cached_long >= long_px || *cached_long >= 1400)
    {
        return faces.clone();
    }
    let Some(det) = DETECTOR.get_or_init(|| {
        lightcraft_faces::yunet::Detector::new(include_bytes!("../../../assets/models/face_detection_yunet_2023mar.onnx")).ok()
    }) else {
        return Vec::new();
    };

    let to_srgb = lightcraft_color::REC2020.to_space(&lightcraft_color::SRGB);
    let base_g = gain.clamp(0.25, 4.0);

    // Helper: extract sub-rect `[rx0..rx1, ry0..ry1]` from `img` into an 8-bit sRGB buffer `(dw, dh)`
    // with optional auto-exposure normalization so indoor/backlit faces have crisp contrast.
    let render_subrect_srgb = |rx0: usize, ry0: usize, rx1: usize, ry1: usize, target_edge: usize, auto_norm: bool| -> (Vec<u8>, usize, usize) {
        let rw = (rx1 - rx0).max(1);
        let rh = (ry1 - ry0).max(1);
        let sc = (target_edge as f32 / rw.max(rh) as f32).min(1.0);
        let dw = ((rw as f32 * sc).round() as usize).max(16);
        let dh = ((rh as f32 * sc).round() as usize).max(16);

        let mut eff_g = base_g;
        if auto_norm {
            // Measure mean luminance inside the central 60% of the subrect (the face)
            let ix0 = rx0 + rw / 5;
            let ix1 = (rx1 - rw / 5).max(ix0 + 1);
            let iy0 = ry0 + rh / 5;
            let iy1 = (ry1 - rh / 5).max(iy0 + 1);
            let mut sum_y = 0.0f32;
            let mut cnt = 0usize;
            let step_y = ((iy1 - iy0) / 16).max(1);
            let step_x = ((ix1 - ix0) / 16).max(1);
            for py in (iy0..iy1).step_by(step_y) {
                for px in (ix0..ix1).step_by(step_x) {
                    let c = img.data[py * w + px];
                    let y_lin = (0.2627 * c[0] + 0.6780 * c[1] + 0.0593 * c[2]).max(0.0);
                    if y_lin.is_finite() {
                        sum_y += y_lin;
                        cnt += 1;
                    }
                }
            }
            if cnt > 0 {
                let mean_y = (sum_y / cnt as f32) * base_g;
                if mean_y > 1e-4 {
                    // Target mid-tone linear luminance ~0.22 for optimal YuNet contrast
                    let corr = (0.22 / mean_y).clamp(0.45, 3.2);
                    eff_g = (base_g * corr).clamp(0.2, 8.0);
                }
            }
        }

        let mut rgb = Vec::with_capacity(dw * dh * 3);
        for y in 0..dh {
            let sy0 = ry0 + ((y as f32 + 0.25) / dh as f32 * rh as f32).floor() as usize;
            let sy1 = ry0 + ((y as f32 + 0.75) / dh as f32 * rh as f32).floor() as usize;
            let sy0 = sy0.min(h - 1);
            let sy1 = sy1.min(h - 1);
            for x in 0..dw {
                let sx0 = rx0 + ((x as f32 + 0.25) / dw as f32 * rw as f32).floor() as usize;
                let sx1 = rx0 + ((x as f32 + 0.75) / dw as f32 * rw as f32).floor() as usize;
                let sx0 = sx0.min(w - 1);
                let sx1 = sx1.min(w - 1);
                let c00 = img.data[sy0 * w + sx0];
                let c10 = img.data[sy0 * w + sx1];
                let c01 = img.data[sy1 * w + sx0];
                let c11 = img.data[sy1 * w + sx1];
                let avg = [
                    0.25 * (c00[0] + c10[0] + c01[0] + c11[0]) * eff_g,
                    0.25 * (c00[1] + c10[1] + c01[1] + c11[1]) * eff_g,
                    0.25 * (c00[2] + c10[2] + c01[2] + c11[2]) * eff_g,
                ];
                let c = tonemap_for_select(avg);
                let s = to_srgb.apply_f32(c);
                for ch in s {
                    let enc = lightcraft_color::transfer::linear_to_srgb(ch.clamp(0.0, 1.0));
                    rgb.push((enc.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
                }
            }
        }
        (rgb, dw, dh)
    };

    // Stage 1: Full-frame detection pass to find all candidate faces (even small faces in group shots)
    let (rgb_full, dw, dh) = render_subrect_srgb(0, 0, w, h, 640, false);
    let coarse_opts = lightcraft_faces::yunet::Options { score: 0.34, nms_iou: 0.3, max_faces: 12 };
    let coarse_faces = det.detect(&rgb_full, dw, dh, &coarse_opts).unwrap_or_default();

    // Stage 2: High-Resolution Per-Face Crop Refinement!
    // In group photos or medium shots, each face in Stage 1 was only 35–80px wide inside the 640×640 tensor.
    // By cropping a ~2.2× portrait window around each detected face from the full-resolution `img` and
    // running YuNet on that portrait crop, the face fills ~300px of YuNet's 640×640 input (6×–8× higher
    // landmark accuracy for eyes, nose, and mouth!).
    let mut refined_faces = Vec::with_capacity(coarse_faces.len());
    let fine_opts = lightcraft_faces::yunet::Options { score: 0.42, nms_iou: 0.3, max_faces: 4 };
    for cf in coarse_faces {
        let fw_norm = (cf.x1 - cf.x0).max(0.01);
        let fh_norm = (cf.y1 - cf.y0).max(0.01);
        if long_px >= 400 && fw_norm.max(fh_norm) < 0.65 {
            let fcx = 0.5 * (cf.x0 + cf.x1) * w as f32;
            let fcy = 0.5 * (cf.y0 + cf.y1) * h as f32;
            let f_side = ((fw_norm * w as f32).max(fh_norm * h as f32) * 2.25).max(48.0);
            let half = 0.5 * f_side;
            let cx0 = ((fcx - half).floor() as isize).clamp(0, w as isize - 16) as usize;
            let cy0 = ((fcy - half).floor() as isize).clamp(0, h as isize - 16) as usize;
            let cx1 = ((fcx + half).ceil() as isize).clamp(cx0 as isize + 16, w as isize) as usize;
            let cy1 = ((fcy + half).ceil() as isize).clamp(cy0 as isize + 16, h as isize) as usize;
            let cw = (cx1 - cx0) as f32;
            let ch = (cy1 - cy0) as f32;
            let (crop_rgb, cdw, cdh) = render_subrect_srgb(cx0, cy0, cx1, cy1, 640, true);
            if let Ok(sub_faces) = det.detect(&crop_rgb, cdw, cdh, &fine_opts) {
                let exp_cx = (fcx - cx0 as f32) / cw;
                let exp_cy = (fcy - cy0 as f32) / ch;
                if let Some(best_rf) = sub_faces
                    .into_iter()
                    .filter(|rf| {
                        let rcx = 0.5 * (rf.x0 + rf.x1);
                        let rcy = 0.5 * (rf.y0 + rf.y1);
                        (rcx - exp_cx).hypot(rcy - exp_cy) < 0.28
                    })
                    .min_by(|a, b| {
                        let da = (0.5 * (a.x0 + a.x1) - exp_cx).hypot(0.5 * (a.y0 + a.y1) - exp_cy);
                        let db = (0.5 * (b.x0 + b.x1) - exp_cx).hypot(0.5 * (b.y0 + b.y1) - exp_cy);
                        da.total_cmp(&db)
                    })
                {
                    let map_x = |nx: f32| (cx0 as f32 + nx * cw) / w as f32;
                    let map_y = |ny: f32| (cy0 as f32 + ny * ch) / h as f32;
                    refined_faces.push(lightcraft_faces::yunet::Face {
                        x0: map_x(best_rf.x0).clamp(0.0, 1.0),
                        y0: map_y(best_rf.y0).clamp(0.0, 1.0),
                        x1: map_x(best_rf.x1).clamp(0.0, 1.0),
                        y1: map_y(best_rf.y1).clamp(0.0, 1.0),
                        score: best_rf.score,
                        landmarks: best_rf.landmarks.map(|(lx, ly)| (map_x(lx), map_y(ly))),
                    });
                    continue;
                }
            }
            // Keep coarse face only if its confidence was solid
            if cf.score >= 0.45 {
                refined_faces.push(cf);
            }
        } else if cf.score >= 0.42 {
            refined_faces.push(cf);
        }
    }

    if let Ok(mut guard) = CACHE.lock() {
        if let Some(pos) = guard.iter().position(|(k, _, _)| *k == sig) {
            guard[pos] = (sig, long_px, refined_faces.clone());
        } else {
            if guard.len() >= 16 {
                guard.remove(0);
            }
            guard.push((sig, long_px, refined_faces.clone()));
        }
    }
    refined_faces
}

#[allow(clippy::too_many_arguments)]
fn refine_eye_center(
    blur_l: &Plane,
    w: usize,
    h: usize,
    ex: f32,
    ey: f32,
    r_iris: f32,
    r_search: f32,
    cos_t: f32,
    sin_t: f32,
) -> (f32, f32, f32, f32) {
    let x0 = ((ex - r_search).floor() as isize).max(0) as usize;
    let x1 = ((ex + r_search).ceil() as isize).min(w as isize - 1).max(0) as usize;
    let y0 = ((ey - r_search * 1.15).floor() as isize).max(0) as usize;
    let y1 = ((ey + r_search * 0.75).ceil() as isize).min(h as isize - 1).max(0) as usize;
    let mut best_x = ex;
    let mut best_y = ey;
    let mut best_score = f32::MAX;
    let sample_at = |x: f32, y: f32| -> f32 {
        let ix = (x.floor() as isize).clamp(0, w as isize - 1) as usize;
        let iy = (y.floor() as isize).clamp(0, h as isize - 1) as usize;
        let v = blur_l.data[iy * w + ix];
        if v.is_finite() { v } else { 0.0 }
    };
    let mut best_l = sample_at(ex, ey);
    let mut best_surround = best_l + 1.0;
    let r_probe = (0.45 * r_iris).max(1.0);
    let r_scl = (1.16 * r_iris).max(2.0);
    let r_below = (1.05 * r_iris).max(1.8);
    let up_bias = 0.16 * r_iris;
    for py in y0..=y1 {
        for px in x0..=x1 {
            let cx = px as f32 + 0.5;
            let cy = py as f32 + 0.5;
            let dx = cx - ex;
            let dy = cy - ey;
            let ux = dx * cos_t + dy * sin_t;
            let uy = -dx * sin_t + dy * cos_t;
            if ux.abs() > r_search || uy < -r_search * 1.15 || uy > r_search * 0.65 {
                continue;
            }
            // 5-point circular iris cross: all 5 points (center, left, right, top, bottom at 0.45*r_iris)
            // must lie inside the dark iris disk. If any arm pokes into bright sclera or lower-eyelid skin,
            // `max_5` increases sharply, rejecting eyelid edges and tear ducts!
            let l_c = sample_at(cx, cy);
            let l_r = sample_at(cx + r_probe * cos_t, cy + r_probe * sin_t);
            let l_l = sample_at(cx - r_probe * cos_t, cy - r_probe * sin_t);
            let l_d = sample_at(cx - r_probe * sin_t, cy + r_probe * cos_t);
            let l_u = sample_at(cx + r_probe * sin_t, cy - r_probe * cos_t);
            let mean_5 = 0.28 * l_c + 0.18 * (l_r + l_l + l_d + l_u);
            let max_5 = l_c.max(l_r).max(l_l).max(l_d).max(l_u);
            let disk_darkness = 0.55 * mean_5 + 0.45 * max_5;

            // Surround contrast: sclera flanks (left/right) and lower eyelid skin below the iris
            let l_scl_r = sample_at(cx + r_scl * cos_t, cy + r_scl * sin_t);
            let l_scl_l = sample_at(cx - r_scl * cos_t, cy - r_scl * sin_t);
            let l_low_lid = sample_at(cx - r_below * sin_t, cy + r_below * cos_t);
            let sclera_contrast = ((l_scl_r - mean_5).max(0.0) + (l_scl_l - mean_5).max(0.0)).min(2.6);
            let lid_contrast = (l_low_lid - mean_5).clamp(0.0, 1.6);

            // Gentle anchor toward YuNet landmark with slight upward bias (away from lower eyelid)
            let anchor = 0.22 * (ux / r_search.max(1.0)).powi(2) + 0.26 * ((uy + up_bias) / r_search.max(1.0)).powi(2);
            let score = disk_darkness - 0.24 * sclera_contrast - 0.18 * lid_contrast + anchor;
            if score < best_score {
                best_score = score;
                best_x = cx;
                best_y = cy;
                best_l = l_c.min(mean_5);
                best_surround = l_scl_r.max(l_scl_l).max(l_low_lid);
            }
        }
    }
    (best_x, best_y, best_l, best_surround)
}

/// Apply manual calibration (`offset_x`, `offset_y` in `-100..100`, `scale` in `50..150%`) to a mask plane.
/// Scaling expands or contracts each mask region symmetrically around its own local center (so multi-face
/// eye/lip masks never drift apart), and `offset_x` / `offset_y` shift the plane with bilinear interpolation.
pub fn transform_plane(p: &mut Plane, offset_x: f64, offset_y: f64, scale: f64, long_px: f32) {
    let (w, h) = (p.width, p.height);
    if w == 0 || h == 0 {
        return;
    }
    let s = ((scale - 100.0) / 50.0).clamp(-1.0, 1.0) as f32;
    if s.abs() > 1e-3 {
        let sigma = (0.0035 * long_px * s.abs()).clamp(0.6, 8.0);
        let blurred = lightcraft_raster::blur::gaussian(p, sigma);
        if s > 0.0 {
            let hi = (0.58 - 0.38 * s).clamp(0.12, 0.58);
            for (v, b) in p.data.iter_mut().zip(&blurred.data) {
                let vc = if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 };
                let bc = if b.is_finite() { b.clamp(0.0, 1.0) } else { 0.0 };
                let expanded = smooth(0.04, hi, bc);
                *v = vc.max(expanded);
            }
        } else {
            let k = -s;
            let lo = (0.28 * k).clamp(0.0, 0.35);
            let hi = (lo + 0.55).clamp(0.25, 0.92);
            for (v, b) in p.data.iter_mut().zip(&blurred.data) {
                let vc = if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 };
                let bc = if b.is_finite() { b.clamp(0.0, 1.0) } else { 0.0 };
                *v = (vc * smooth(lo, hi, bc)).clamp(0.0, 1.0);
            }
        }
    }
    if offset_x.abs() > 1e-4 || offset_y.abs() > 1e-4 {
        let dx = (offset_x / 100.0).clamp(-1.0, 1.0) as f32 * 0.12 * w as f32;
        let dy = (offset_y / 100.0).clamp(-1.0, 1.0) as f32 * 0.12 * h as f32;
        let src = p.clone();
        for y in 0..h {
            let sy = y as f32 + 0.5 - dy;
            for x in 0..w {
                let sx = x as f32 + 0.5 - dx;
                let val = if sx < 0.0 || sy < 0.0 || sx > w as f32 || sy > h as f32 {
                    0.0
                } else {
                    bilinear(&src, sx, sy)
                };
                p.data[y * w + x] = if val.is_finite() { val.clamp(0.0, 1.0) } else { 0.0 };
            }
        }
    }
}

/// A brush stroke resolved to output pixels: dab centres (spacing r/4 along the path), radius,
/// hard-core radius, flow and density (0..1), and whether it erases.
pub struct BrushDabs {
    pub dabs: Vec<Point>,
    pub r: f64,
    pub hard: f64,
    pub flow: f32,
    pub density: f32,
    pub erase: bool,
    /// Auto Mask: paint only pixels like the one under each dab, then snap to edges.
    pub auto: bool,
    /// Pen pressure (0..1) of each dab, parallel to `dabs`; empty when the stroke has no pressure data (full pressure).
    pub pressure: Vec<f32>,
}

impl BrushDabs {
    /// Radius, hard core and flow of dab `i`, pen pressure applied ([`pressure_radius`], [`pressure_flow`]).
    pub fn dab(&self, i: usize) -> (f64, f64, f32) {
        match self.pressure.get(i) {
            Some(&p) => {
                let r = pressure_radius(self.r, p);
                (r, if self.r > 0.0 { self.hard * r / self.r } else { self.hard }, pressure_flow(self.flow, p))
            }
            None => (self.r, self.hard, self.flow),
        }
    }
}

/// The radius of a dab of a brush of radius `r` (pixels) painted at pen pressure `p` (0..1): from a tenth of the
/// brush at the lightest touch to the full brush at full pressure, never below half a pixel.
#[inline]
pub fn pressure_radius(r: f64, p: f32) -> f64 {
    let p = if p.is_nan() { 1.0 } else { f64::from(p.clamp(0.0, 1.0)) };
    (r * (0.1 + 0.9 * p)).max(0.5).min(r.max(0.5))
}

/// The flow (0..1) of a dab painted at pen pressure `p` (0..1): proportional to the pressure.
#[inline]
pub fn pressure_flow(flow: f32, p: f32) -> f32 {
    let p = if p.is_nan() { 1.0 } else { p.clamp(0.0, 1.0) };
    flow * p
}

/// Pressures for the `added` dabs [`push_dabs`] just laid from a sample at pressure `from` to one at `to`: evenly
/// interpolated, the last one at `to`, as the dabs themselves are placed.
pub fn push_pressures(pressure: &mut Vec<f32>, added: usize, from: f32, to: f32) {
    for k in 1..=added {
        pressure.push(from + (to - from) * k as f32 / added as f32);
    }
}

/// Output pixels a brush sample must move from the last dab to count as movement: below this the
/// pointer is held still (the same screen position mapped again, up to float rounding).
const STATIONARY_PX: f64 = 1e-6;

/// Most dabs one stroke holds (32 MiB of centres). A segment is capped on its own, but a stroke of
/// many hostile segments (an agent, a crafted XMP or catalog) would otherwise add up without
/// bound; past this the rest of the stroke lays nothing (#1520).
pub const MAX_STROKE_DABS: usize = 2_000_000;

/// The dabs of stroke `s` in a `w × h` output.
pub fn brush_dabs(s: &BrushStroke, frame: &Frame, w: usize, h: usize) -> BrushDabs {
    let to_out = frame.norm_to_out(w, h);
    let ppl = frame.px_per_long(w);
    let r = (s.size * ppl).max(0.5);
    let hard = r * (1.0 - (s.feather / 100.0).clamp(0.0, 1.0));
    // Densify the path so dabs overlap (spacing r/4). Coverage builds up by passes, not by how
    // long the button is held: a sample where the pointer hasn't moved (a stationary hold records
    // one per frame) lays no dab, so waiting in place adds nothing (#518).
    let mut dabs: Vec<Point> = Vec::new();
    let mut pressure: Vec<f32> = Vec::new();
    let with_pressure = !s.pressure.is_empty();
    let mut prev = None;
    let mut prev_p = 1.0;
    for (i, p) in s.points.iter().enumerate() {
        let q = to_out.apply(*p);
        let before = dabs.len();
        if push_dabs(&mut dabs, prev, q, r) {
            let pq = s.pressure_at(i);
            if with_pressure {
                // the first dab of the stroke has nothing to interpolate from
                let from = if prev.is_some() { prev_p } else { pq };
                push_pressures(&mut pressure, dabs.len() - before, from, pq);
            }
            prev = Some(q);
            prev_p = pq;
        }
    }
    BrushDabs {
        dabs,
        r,
        hard,
        flow: (s.flow / 100.0).clamp(0.0, 1.0) as f32,
        density: (s.density / 100.0).clamp(0.0, 1.0) as f32,
        erase: s.erase,
        auto: s.auto_mask,
        pressure,
    }
}

/// Distance between consecutive dabs of a brush of radius `r` (pixels): a quarter of the radius,
/// so neighbouring dabs overlap and a stroke reads as continuous however fast it was painted.
#[inline]
pub fn dab_spacing(r: f64) -> f64 {
    (r / 4.0).max(0.5)
}

/// Append the dabs of one path segment, from the previous sample `prev` (already stamped) to `q`,
/// for a brush of radius `r`: evenly spaced at most [`dab_spacing`] apart, ending on `q`. The
/// committed stroke ([`brush_dabs`]) and the live one the UI draws while the button is held place
/// their dabs with this, so they cover the same pixels (issue #517). A sample where the pointer
/// hasn't moved (a stationary hold records one per frame) or that is not a finite point lays
/// nothing, so holding the brush still adds no coverage (#518). Returns whether `q` was stamped;
/// callers move `prev` on only then. A stroke holding [`MAX_STROKE_DABS`] takes no more.
pub fn push_dabs(dabs: &mut Vec<Point>, prev: Option<Point>, q: Point, r: f64) -> bool {
    let mut emitted = dabs.len();
    push_dabs_counted(dabs, &mut emitted, prev, q, r)
}

/// Append one segment using a persistent count of all dabs emitted for this stroke, including
/// dabs no longer held in `dabs`. Incremental callers retain `emitted` when clearing their buffer;
/// it advances only by the number actually appended. Acceptance and spacing match [`push_dabs`].
pub fn push_dabs_counted(dabs: &mut Vec<Point>, emitted: &mut usize, prev: Option<Point>, q: Point, r: f64) -> bool {
    let room = MAX_STROKE_DABS.saturating_sub(*emitted);
    if room == 0 || !(q.x.is_finite() && q.y.is_finite()) {
        return false;
    }
    let before = dabs.len();
    if let Some(prev) = prev {
        let dist = prev.dist(q);
        if dist.is_nan() || dist <= STATIONARY_PX {
            return false;
        }
        let n = (dist / dab_spacing(r)).ceil();
        // a hostile or degenerate segment (huge) gets no in-between dabs rather than a huge
        // allocation: 1 M dabs is far more than any on-screen stroke needs
        let n = if n.is_finite() { n.clamp(0.0, 1_000_000.0) as usize } else { 0 };
        // the in-between dabs and `q` together fit what is left of the stroke's budget
        let n = n.min(room);
        for k in 1..n {
            dabs.push(prev.lerp(q, k as f64 / n as f64));
        }
    }
    dabs.push(q);
    *emitted += dabs.len() - before;
    true
}

/// How much one dab paints at distance `dd` from its centre: 1 inside the hard core `hard`,
/// smoothly falling to 0 at the radius `r`.
#[inline]
pub fn dab_alpha(dd: f64, r: f64, hard: f64) -> f32 {
    if dd > r {
        0.0
    } else if dd <= hard {
        1.0
    } else {
        1.0 - smooth(hard as f32, r as f32, dd as f32)
    }
}

/// A stroke's coverage `v` after another dab paints `a` there: flow accumulates up to density.
#[inline]
pub fn accumulate(v: f32, a: f32, flow: f32, density: f32) -> f32 {
    (v + a * flow * (1.0 - v)).min(density)
}

/// Auto Mask tolerances: a pixel takes a dab's paint fully up to half of these from the colour
/// under the dab centre, none beyond them (log2 luminance, chromaticity `rgb / Y`).
pub const AUTO_TOL_EV: f32 = 0.5;
pub const AUTO_TOL_CHROMA: f32 = 0.25;

/// Chromaticity `rgb / Y` (as colour noise reduction uses it).
#[inline]
pub fn chromaticity(c: [f32; 3]) -> [f32; 3] {
    let y = lightcraft_color::luminance_2020(c).max(1e-6);
    [c[0] / y, c[1] / y, c[2] / y]
}

/// Auto Mask: how much a pixel (log luminance `l`, chromaticity `ch`) is like the reference.
#[inline]
pub fn auto_similarity(l: f32, ch: [f32; 3], rl: f32, rch: [f32; 3]) -> f32 {
    let dl = (l - rl) / AUTO_TOL_EV;
    let dc = ((ch[0] - rch[0]).powi(2) + (ch[1] - rch[1]).powi(2) + (ch[2] - rch[2]).powi(2)).sqrt() / AUTO_TOL_CHROMA;
    1.0 - smooth(0.5, 1.0, (dl * dl + dc * dc).sqrt())
}

/// The output pixel a dab at `d` samples its reference colour from. The position is taken at f32
/// precision (like the GPU kernel): a dab on a pixel boundary must not fall on either side of it
/// by an f64 rounding error, which depends on where the window of a zoomed view starts.
#[inline]
pub fn dab_pixel(d: Point, w: usize, h: usize) -> usize {
    let x = ((d.x as f32).floor().max(0.0) as usize).min(w - 1);
    let y = ((d.y as f32).floor().max(0.0) as usize).min(h - 1);
    y * w + x
}

/// Guided-filter refinement of an Auto Mask stroke of radius `r` px: (sigma px, epsilon EV²). The
/// stroke keeps the larger of its own and the refined alpha.
pub fn auto_refine(r: f64) -> (f32, f32) {
    ((0.25 * r).max(1.0) as f32, 0.02)
}

/// Guided filter of `p` steered by `guide` (He et al.), clamped to 0..1: `p`'s edges snap to the
/// guide's.
pub fn guided_cross(guide: &Plane, p: &Plane, sigma: f32, eps: f32) -> Plane {
    use lightcraft_raster::blur::gaussian;
    let mi = gaussian(guide, sigma);
    let mp = gaussian(p, sigma);
    let cip = gaussian(&guide.zip_map(p, |a, b| a * b), sigma);
    let cii = gaussian(&guide.map(|a| a * a), sigma);
    let mut a = Plane::new(p.width, p.height);
    let mut b = Plane::new(p.width, p.height);
    for i in 0..p.data.len() {
        let var = (cii.data[i] - mi.data[i] * mi.data[i]).max(0.0);
        let k = (cip.data[i] - mi.data[i] * mp.data[i]) / (var + eps);
        a.data[i] = k;
        b.data[i] = mp.data[i] - k * mi.data[i];
    }
    let (ma, mb) = (gaussian(&a, sigma), gaussian(&b, sigma));
    let mut q = Plane::new(p.width, p.height);
    for i in 0..q.data.len() {
        q.data[i] = (ma.data[i] * guide.data[i] + mb.data[i]).clamp(0.0, 1.0);
    }
    q
}

/// Stamp brush strokes into `out` (max-combine; erase strokes subtract). Auto Mask strokes weight
/// each dab by [`auto_similarity`] to the pixel under its centre and are refined with
/// [`guided_cross`] on log luminance.
fn rasterize_brush(strokes: &[BrushStroke], frame: &Frame, w: usize, h: usize, img: &Rgb32f, log_l: &Plane, out: &mut Plane) {
    if w == 0 || h == 0 {
        return;
    }
    for s in strokes {
        let bd = brush_dabs(s, frame, w, h);
        let (dens, erase, auto) = (bd.density, bd.erase, bd.auto);
        let mut stroke_alpha = Plane::new(w, h);
        for (k, d) in bd.dabs.iter().enumerate() {
            let (r, hard, flow) = bd.dab(k);
            let refc = auto.then(|| {
                let j = dab_pixel(*d, w, h);
                (log_l.data[j], chromaticity(img.data[j]))
            });
            let (x0, x1) = (((d.x - r).floor().max(0.0)) as usize, ((d.x + r).ceil().max(0.0) as usize).min(w));
            let (y0, y1) = (((d.y - r).floor().max(0.0)) as usize, ((d.y + r).ceil().max(0.0) as usize).min(h));
            for y in y0..y1 {
                for x in x0..x1 {
                    let dd = Point::new(x as f64 + 0.5, y as f64 + 0.5).dist(*d);
                    if dd > r {
                        continue;
                    }
                    let mut a = dab_alpha(dd, r, hard);
                    let i = y * w + x;
                    if let Some((rl, rch)) = refc {
                        a *= auto_similarity(log_l.data[i], chromaticity(img.data[i]), rl, rch);
                    }
                    let v = &mut stroke_alpha.data[i];
                    // flow accumulates within a stroke up to density
                    *v = accumulate(*v, a, flow, dens);
                }
            }
        }
        if auto {
            let (sigma, eps) = auto_refine(bd.r);
            let q = guided_cross(log_l, &stroke_alpha, sigma, eps);
            // the refinement fills gaps along edges; it never takes paint away
            stroke_alpha = stroke_alpha.zip_map(&q, f32::max);
        }
        for (o, a) in out.data.iter_mut().zip(&stroke_alpha.data) {
            *o = if erase { *o * (1.0 - a) } else { o.max(*a) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lightcraft_develop::{DevelopSettings, MaskComponent};

    fn frame(w: usize, h: usize) -> Frame {
        Frame::new(w, h, &DevelopSettings::default(), true)
    }

    #[test]
    fn linear_gradient_ramps() {
        let f = frame(100, 50);
        let img = Rgb32f::new(100, 50);
        let l = Plane::new(100, 50);
        let a = shape_alpha(&MaskShape::Linear { start: Point::new(0.0, 0.0), end: Point::new(1.0, 0.0) }, &f, 100, 50, &img, &l, 0.0, None);
        assert!(a.get(1, 25) > 0.99 && a.get(98, 25) < 0.01);
        assert!((a.get(50, 25) - 0.5).abs() < 0.05);
    }

    #[test]
    fn stored_segmentations_render_at_any_size_and_follow_the_crop() {
        // the left half selected, on a 16² logit grid
        let side = 16;
        let logits: Vec<f32> = (0..side * side).map(|i| if i % side < side / 2 { 12.0 } else { -12.0 }).collect();
        let seg = SegMask::from_logits(side, &logits);
        for shape in [
            MaskShape::Object { hint: vec![Point::new(0.25, 0.5)], exclude: vec![], seg: Some(seg.clone()), detail: vec![], edge: 0.0 },
            MaskShape::Prompt { text: "left".into(), seg: Some(seg.clone()), detail: vec![], edge: 0.0 },
        ] {
            for (w, h) in [(40, 20), (400, 200)] {
                let a = shape_alpha(&shape, &frame(w, h), w, h, &Rgb32f::new(w, h), &Plane::new(w, h), 0.0, None);
                assert!(a.get(1, h / 2) > 0.99 && a.get(w - 2, h / 2) < 0.01, "{w}×{h}");
                if w >= 400 {
                    let soft = (0..w).filter(|x| (0.05..0.95).contains(&a.get(*x, h / 2))).count();
                    assert!((2..30).contains(&soft), "an anti-aliased edge, not a ramp: {soft} px");
                }
            }
        }
        // cropped to the right half: nothing left selected
        let mut d = DevelopSettings::default();
        d.crop.geometry.rect = lightcraft_geom::Rect::from_center(Point::new(0.75, 0.5), 0.5, 1.0);
        let f = Frame::new(40, 20, &d, true);
        let shape = MaskShape::Prompt { text: "left".into(), seg: Some(seg), detail: vec![], edge: 0.0 };
        let a = shape_alpha(&shape, &f, 20, 20, &Rgb32f::new(20, 20), &Plane::new(20, 20), 0.0, None);
        assert!(a.data.iter().all(|v| *v < 0.05));
    }

    #[test]
    fn detail_patches_replace_the_coarse_mask_inside_their_rectangle() {
        // coarse: nothing; a patch over the right half selects its own left half (x 0.5..0.75)
        let coarse = SegMask::from_logits(4, &[-12.0; 16]);
        let side = 16;
        let l: Vec<f32> = (0..side * side).map(|i| if i % side < side / 2 { 12.0 } else { -12.0 }).collect();
        let patch = SegMask::from_logits_in(side, &l, [0.5, 0.0, 1.0, 1.0]);
        let shape = MaskShape::Object { hint: vec![Point::new(0.6, 0.5)], exclude: vec![], seg: Some(coarse), detail: vec![patch], edge: 0.0 };
        let (w, h) = (200, 100);
        let a = shape_alpha(&shape, &frame(w, h), w, h, &Rgb32f::new(w, h), &Plane::new(w, h), 0.0, None);
        assert!(a.get(20, 50) < 0.01, "outside the patch: the coarse mask");
        assert!(a.get(120, 50) > 0.99, "inside the patch, its selected half");
        assert!(a.get(190, 50) < 0.01, "inside the patch, its unselected half");
    }

    #[test]
    fn a_hostile_number_of_detail_patches_is_capped() {
        // built in memory (a deserialized document is capped already): only the first
        // MAX_DETAIL are decoded and sampled, so this needs 4 × 4 MB, not 20 000 × 4 MB
        let side = 1024;
        let l: Vec<f32> = vec![12.0; side * side];
        let patch = SegMask::from_logits_in(side, &l, [0.0, 0.0, 0.5, 1.0]);
        let shape = MaskShape::Object { hint: vec![Point::new(0.2, 0.5)], exclude: vec![], seg: None, detail: vec![patch; 20_000], edge: 0.0 };
        let (w, h) = (64, 32);
        let t = std::time::Instant::now();
        let a = shape_alpha(&shape, &frame(w, h), w, h, &Rgb32f::new(w, h), &Plane::new(w, h), 0.0, None);
        assert!(a.get(5, 16) > 0.99 && a.get(60, 16) < 0.01);
        assert!(t.elapsed() < std::time::Duration::from_secs(20), "{:?}", t.elapsed());
    }

    #[test]
    fn the_edge_setting_hardens_or_feathers_the_transition() {
        let side = 16;
        let l: Vec<f32> = (0..side * side).map(|i| ((i % side) as f32 - 7.5) * -0.6).collect(); // a gentle ramp
        let seg = SegMask::from_logits(side, &l);
        let (w, h) = (400, 40);
        let soft_px = |edge: f64| {
            let shape = MaskShape::Prompt { text: "x".into(), seg: Some(seg.clone()), detail: vec![], edge };
            let a = shape_alpha(&shape, &frame(w, h), w, h, &Rgb32f::new(w, h), &Plane::new(w, h), 0.0, None);
            assert!(a.get(2, h / 2) > 0.9 && a.get(w - 3, h / 2) < 0.1, "still the same selection at {edge}");
            (0..w).filter(|x| (0.1..0.9).contains(&a.get(*x, h / 2))).count()
        };
        let (hard, plain, soft) = (soft_px(-100.0), soft_px(0.0), soft_px(100.0));
        assert!(hard < plain && plain < soft, "transition widths {hard} < {plain} < {soft}");
    }

    #[test]
    fn ai_shapes_without_a_segmentation_select_nothing() {
        let f = frame(30, 30);
        let (img, l) = (Rgb32f::new(30, 30), Plane::new(30, 30));
        for shape in [
            MaskShape::Object { hint: vec![], exclude: vec![], seg: None, detail: vec![], edge: 0.0 },
            MaskShape::Prompt { text: "sky".into(), seg: None, detail: vec![], edge: 0.0 },
            MaskShape::Prompt { text: "sky".into(), seg: Some(SegMask { side: 4, data: "damaged!".into(), rect: None }), detail: vec![], edge: 0.0 },
        ] {
            assert!(shape_alpha(&shape, &f, 30, 30, &img, &l, 0.0, None).data.iter().all(|v| *v == 0.0));
        }
    }

    #[test]
    fn radial_inside_and_invert() {
        let f = frame(100, 100);
        let img = Rgb32f::new(100, 100);
        let l = Plane::new(100, 100);
        let shape = MaskShape::Radial { center: Point::new(0.5, 0.5), rx: 0.2, ry: 0.2, angle: 0.0, feather: 20.0, invert: false };
        let a = shape_alpha(&shape, &f, 100, 100, &img, &l, 0.0, None);
        assert!(a.get(50, 50) > 0.99 && a.get(5, 5) < 0.01);
        let m = Mask { components: vec![MaskComponent { name: None, op: MaskOp::Add, invert: true, shape }], ..Default::default() };
        let e = evaluate(&[m], &f, 100, 100, &img, &l, 0.0, None);
        assert!(e[0].alpha.get(50, 50) < 0.01);
    }

    /// Issue #517: dabs along a long segment (a fast stroke: two pointer samples far apart) are at
    /// most [`dab_spacing`] apart, painting dab by dab (the live stroke) or the whole path at once
    /// (the committed one).
    #[test]
    fn a_long_segment_gets_dabs_at_most_the_spacing_apart() {
        let r = 12.0;
        let pts = [Point::new(10.0, 20.0), Point::new(610.0, 260.0), Point::new(611.0, 260.0), Point::new(40.0, 400.0)];
        let mut live = Vec::new();
        let mut prev = None;
        for q in pts {
            if push_dabs(&mut live, prev, q, r) {
                prev = Some(q);
            }
        }
        assert!(live.len() > 3 * 600 / 12, "{} dabs", live.len());
        assert!(live.windows(2).all(|w| w[0].dist(w[1]) <= dab_spacing(r) + 1e-9));
        assert_eq!(live.first(), pts.first());
        assert_eq!(live.last(), pts.last());
        // the committed stroke places the same dabs (in output pixels; this frame maps 1:1 scaled)
        let (w, h) = (1000, 500);
        let f = frame(w, h);
        let stroke = BrushStroke { points: pts.map(|p| Point::new(p.x / w as f64, p.y / h as f64)).to_vec(), ..Default::default() };
        let committed = brush_dabs(&stroke, &f, w, h);
        let mut again = Vec::new();
        let mut prev = None;
        for p in &stroke.points {
            let q = f.norm_to_out(w, h).apply(*p);
            if push_dabs(&mut again, prev, q, committed.r) {
                prev = Some(q);
            }
        }
        assert_eq!(committed.dabs, again);
        assert!(committed.dabs.windows(2).all(|w| w[0].dist(w[1]) <= dab_spacing(committed.r) + 1e-9));
        // many huge segments stop at the stroke budget instead of adding up without bound
        let mut big = Vec::new();
        let mut at = Point::new(0.0, 0.0);
        for i in 0..5 {
            let q = Point::new(if i % 2 == 0 { 1e12 } else { 0.0 }, 0.0);
            if push_dabs(&mut big, Some(at), q, 1.0) {
                at = q;
            }
        }
        assert!(big.len() <= MAX_STROKE_DABS, "{} dabs", big.len());
        assert!(!push_dabs(&mut big, Some(at), Point::new(5.0, 5.0), 1.0), "a full stroke takes no more");
        // a hostile segment places no in-between dabs
        let mut v = Vec::new();
        assert!(!push_dabs(&mut v, Some(Point::new(f64::NAN, 0.0)), Point::new(1.0, 1.0), r));
        assert!(!push_dabs(&mut v, None, Point::new(f64::INFINITY, 1.0), r));
        push_dabs(&mut v, Some(Point::new(0.0, 0.0)), Point::new(1e300, 1.0), 0.0);
        assert!(v.len() <= 1_000_002, "{}", v.len());
    }

    /// A temporary output buffer must spend the same stroke budget as a retained one.
    #[test]
    fn incremental_dab_buffers_share_the_stroke_budget() {
        let points = [Point::new(20.0, 20.0), Point::new(80.0, 20.0), Point::new(200.0, 80.0), Point::new(21.0, 20.0)];
        let mut retained = Vec::new();
        let mut streamed = Vec::new();
        let mut retained_count = MAX_STROKE_DABS - 3;
        let mut streamed_count = retained_count;
        let mut prev = None;
        for q in points {
            let mut chunk = Vec::new();
            let accepted = push_dabs_counted(&mut retained, &mut retained_count, prev, q, 10.0);
            assert_eq!(accepted, push_dabs_counted(&mut chunk, &mut streamed_count, prev, q, 10.0));
            streamed.extend(chunk);
            if accepted {
                prev = Some(q);
            }
        }
        assert_eq!(retained, streamed);
        assert_eq!(retained.len(), 3);
        assert_eq!(retained_count, MAX_STROKE_DABS);
        assert_eq!(streamed_count, retained_count);
        assert_eq!(prev, Some(points[1]));
        assert!(!retained.contains(&points[2]));
        assert!(!retained.contains(&points[3]));
    }

    /// Pen pressure scales each dab's radius and flow; a stroke without pressure data paints as before, and hostile
    /// pressures (NaN, out of range, a short list) neither panic nor paint more than full pressure.
    #[test]
    fn pen_pressure_scales_dab_size_and_flow() {
        let (w, h) = (400, 100);
        let f = frame(w, h);
        let img = Rgb32f::new(w, h);
        let l = Plane::new(w, h);
        let line = vec![Point::new(0.1, 0.5), Point::new(0.9, 0.5)];
        let alpha = |pressure: Vec<f32>| {
            let s = BrushStroke { points: line.clone(), size: 0.05, feather: 0.0, flow: 100.0, pressure, ..Default::default() };
            shape_alpha(&MaskShape::Brush { strokes: vec![s] }, &f, w, h, &img, &l, 0.0, None)
        };
        let full = alpha(vec![]);
        assert_eq!(full.data, alpha(vec![1.0, 1.0]).data, "full pressure is no pressure");
        // light at the start, full at the end: thin and faint on the left, as wide as the brush on the right
        let ramp = alpha(vec![0.1, 1.0]);
        let width = |a: &Plane, x: usize| (0..h).filter(|y| a.get(x, *y) > 0.01).count();
        assert!(width(&ramp, 60) < width(&full, 60) / 2, "{} vs {}", width(&ramp, 60), width(&full, 60));
        assert!(width(&ramp, 350) + 2 >= width(&full, 350));
        assert!(ramp.get(60, 50) < full.get(60, 50) + 1e-6);
        let faint = alpha(vec![0.2]);
        assert!(faint.data.iter().zip(&full.data).all(|(a, b)| *a <= *b + 1e-6), "never more than full pressure");
        assert!(faint.get(200, 50) > 0.0 && faint.get(200, 50) < full.get(200, 50));
        for hostile in [vec![f32::NAN, f32::INFINITY], vec![-3.0], vec![0.5; 10_000]] {
            let a = alpha(hostile);
            assert!(a.data.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
        }
        assert_eq!(pressure_radius(10.0, f32::NAN), 10.0);
        assert_eq!(pressure_radius(10.0, 0.0), 1.0);
        assert_eq!(pressure_flow(0.8, 2.0), 0.8);
    }

    #[test]
    fn brush_and_subtract() {
        let f = frame(200, 100);
        let img = Rgb32f::new(200, 100);
        let l = Plane::new(200, 100);
        let stroke = BrushStroke { points: vec![Point::new(0.1, 0.5), Point::new(0.9, 0.5)], size: 0.05, ..Default::default() };
        let erase = BrushStroke { points: vec![Point::new(0.5, 0.5)], size: 0.05, erase: true, feather: 0.0, ..Default::default() };
        let m = Mask {
            components: vec![MaskComponent { name: None, op: MaskOp::Add, invert: false, shape: MaskShape::Brush { strokes: vec![stroke, erase] } }],
            ..Default::default()
        };
        let e = evaluate(&[m], &f, 200, 100, &img, &l, 0.0, None);
        let a = &e[0].alpha;
        assert!(a.get(40, 50) > 0.9, "{}", a.get(40, 50));
        assert!(a.get(100, 50) < 0.05, "erased centre {}", a.get(100, 50));
        assert!(a.get(40, 5) < 0.01);
    }

    /// #518: holding the brush still adds no coverage. A press plus any number of stationary
    /// pointer samples paints exactly the single dab of the press, whatever the frame rate.
    #[test]
    fn holding_the_brush_still_adds_no_coverage() {
        let (w, h) = (200, 100);
        let f = frame(w, h);
        let img = Rgb32f::new(w, h);
        let l = Plane::new(w, h);
        let p = Point::new(0.5, 0.5);
        let spot = |samples: usize, flow: f64| {
            let s = BrushStroke { points: vec![p; samples], size: 0.1, feather: 80.0, flow, ..Default::default() };
            shape_alpha(&MaskShape::Brush { strokes: vec![s] }, &f, w, h, &img, &l, 0.0, None)
        };
        for flow in [100.0, 40.0] {
            let press = spot(1, flow);
            // the feathered edge is soft to begin with
            let edge = press.get(100 + 14, 50);
            assert!(edge > 0.05 && edge < 0.6, "flow {flow}: edge {edge}");
            for n in [2, 5, 60, 600] {
                let held = spot(n, flow);
                assert!(held.data == press.data, "flow {flow}: {n} stationary samples hardened the edge {edge} → {}", held.get(100 + 14, 50));
            }
        }
        let s = BrushStroke { points: vec![p; 60], size: 0.1, ..Default::default() };
        assert_eq!(brush_dabs(&s, &f, w, h).dabs.len(), 1, "one dab for the press");
        // a stroke that pauses partway lays the same dabs as one that doesn't
        let (a, b) = (Point::new(0.2, 0.5), Point::new(0.8, 0.5));
        let paused = BrushStroke { points: vec![a, p, p, p, p, b], flow: 30.0, ..Default::default() };
        let smooth = BrushStroke { points: vec![a, p, b], flow: 30.0, ..Default::default() };
        assert_eq!(brush_dabs(&paused, &f, w, h).dabs, brush_dabs(&smooth, &f, w, h).dabs);
    }

    /// Flow builds up by passes: going back over an area in the same stroke adds coverage.
    #[test]
    fn a_second_pass_builds_up_with_low_flow() {
        let (w, h) = (200, 100);
        let f = frame(w, h);
        let img = Rgb32f::new(w, h);
        let l = Plane::new(w, h);
        let (a, b) = (Point::new(0.2, 0.5), Point::new(0.8, 0.5));
        let paint = |points: Vec<Point>| {
            let s = BrushStroke { points, size: 0.05, feather: 50.0, flow: 20.0, ..Default::default() };
            shape_alpha(&MaskShape::Brush { strokes: vec![s] }, &f, w, h, &img, &l, 0.0, None)
        };
        let once = paint(vec![a, b]);
        let twice = paint(vec![a, b, a]);
        let thrice = paint(vec![a, b, a, b]);
        let (o, t, th) = (once.get(100, 50), twice.get(100, 50), thrice.get(100, 50));
        assert!(o > 0.05 && o < 0.99, "one pass {o}");
        assert!(t > o + 0.02 && th > t + 0.01, "passes build up: {o} → {t} → {th}");
    }

    #[test]
    fn auto_mask_stops_at_edges() {
        // dark left half, bright right half; a stroke along the boundary, centred on the dark side
        let (w, h) = (200, 100);
        let f = frame(w, h);
        let img = Rgb32f::from_fn(w, h, |x, _| if x < 100 { [0.03; 3] } else { [0.6, 0.5, 0.4] });
        let l = img.map(crate::local::log_lum);
        let stroke = |auto_mask| BrushStroke {
            points: vec![Point::new(0.47, 0.2), Point::new(0.47, 0.8)],
            size: 0.08,
            feather: 0.0,
            auto_mask,
            ..Default::default()
        };
        let comp = |auto| MaskShape::Brush { strokes: vec![stroke(auto)] };
        let plain = shape_alpha(&comp(false), &f, w, h, &img, &l, 0.0, None);
        let auto = shape_alpha(&comp(true), &f, w, h, &img, &l, 0.0, None);
        // without Auto Mask the brush spills over the edge; with it, it stays on the dark side
        assert!(plain.get(105, 50) > 0.9, "{}", plain.get(105, 50));
        assert!(auto.get(105, 50) < 0.05, "spill {}", auto.get(105, 50));
        assert!(auto.get(110, 50) < 0.02);
        assert!(auto.get(90, 50) > 0.9, "painted side {}", auto.get(90, 50));
        assert!(auto.get(99, 50) > 0.8, "up to the edge {}", auto.get(99, 50));
        // on a flat area Auto Mask paints like the plain brush
        let flat = Rgb32f::from_fn(w, h, |_, _| [0.2; 3]);
        let fl = flat.map(crate::local::log_lum);
        let a = shape_alpha(&comp(true), &f, w, h, &flat, &fl, 0.0, None);
        let b = shape_alpha(&comp(false), &f, w, h, &flat, &fl, 0.0, None);
        assert!((a.get(94, 50) - b.get(94, 50)).abs() < 0.02, "{} vs {}", a.get(94, 50), b.get(94, 50));
    }

    /// Refine Edges: a soft mask over a hard edge in the photo snaps to that edge.
    #[test]
    fn refine_edges_follows_the_photo() {
        let (w, h) = (120usize, 80usize);
        // left half dark, right half bright
        let img = Rgb32f::from_fn(w, h, |x, _| if x < 60 { [0.05; 3] } else { [0.6; 3] });
        let log_l = Plane::from_fn(w, h, |x, _| if x < 60 { (0.05f32).log2() } else { (0.6f32).log2() });
        let f = Frame::new(w, h, &Default::default(), true);
        let soft = Mask {
            components: vec![MaskComponent {
                name: None,
                op: MaskOp::Add,
                invert: false,
                shape: MaskShape::Linear { start: Point::new(0.75, 0.5), end: Point::new(0.25, 0.5) },
            }],
            ..Default::default()
        };
        let refined = Mask { refine: 100.0, ..soft.clone() };
        let step = |m: &Mask| {
            let a = evaluate_one(m, &f, w, h, &img, &log_l, 0.0, None);
            a.data[40 * w + 63] - a.data[40 * w + 56]
        };
        let (plain, sharp) = (step(&soft), step(&refined));
        assert!(sharp > plain * 1.5 && sharp > 0.1, "the edge in the mask follows the photo's: {plain} → {sharp}");
    }

    /// Issue #157: a colour-range sample of one flat patch must not select a clearly different
    /// patch. The two patches are the ones from the report (sRGB 120,150,155 and 200,198,195),
    /// sampled the way `mask.sampleColor` does (3×3 mean of the selection-space OkLab).
    #[test]
    fn color_range_keeps_to_the_sampled_colour() {
        use lightcraft_color::{REC2020, SRGB, transfer::decode_srgb8};
        let (w, h) = (120usize, 40usize);
        let m = SRGB.to_space(&REC2020);
        let patch = |rgb: [u8; 3]| m.apply_f32(rgb.map(decode_srgb8));
        let (left, right) = (patch([120, 150, 155]), patch([200, 198, 195]));
        let img = Rgb32f::from_fn(w, h, |x, _| if x < 60 { left } else { right });
        let log_l = img.map(crate::local::log_lum);
        let f = Frame::new(w, h, &Default::default(), true);
        let mut sample = [0f64; 3];
        for y in 19..=21 {
            for x in 29..=31 {
                let lab = lightcraft_color::perceptual::oklab_from_2020(tonemap_for_select(img.data[y * w + x]));
                for k in 0..3 {
                    sample[k] += lab[k] as f64 / 9.0;
                }
            }
        }
        let alpha = |refine| {
            let shape = MaskShape::ColorRange { samples: vec![sample], refine };
            let a = shape_alpha(&shape, &f, w, h, &img, &log_l, 0.0, None);
            (a.get(30, 20), a.get(90, 20))
        };
        for refine in [0.0, 50.0, 85.0] {
            let (l, r) = alpha(refine);
            assert!(l > 0.95, "refine {refine}: sampled patch {l}");
            assert!(r < 0.05, "refine {refine}: other patch {r}");
        }
        // Refine 100 reaches furthest (the grey is ~0.063 away, the fade ends at 0.075), but the
        // other patch stays mostly out
        let (l, r) = alpha(100.0);
        assert!(l > 0.95 && r < 0.3, "refine 100: {l} vs {r}");
    }

    /// A half-resolution matte selecting the left `frac` of a `w × h` source.
    fn left_matte(w: usize, h: usize, frac: f64) -> Image<u8> {
        Image::from_fn(w / 2, h / 2, |x, _| if (x as f64 + 0.5) < frac * (w / 2) as f64 { 255 } else { 0 })
    }

    /// The Sky mask follows the source's sky matte where it has one (here: the bottom-left of a
    /// uniformly bright blue photo, where the heuristic sees no sky, and not its top right, where
    /// it does), through user orientation, at any output size; without one it keeps the heuristic.
    #[test]
    fn sky_uses_the_sources_matte() {
        let (w, h) = (200, 100);
        let img = Rgb32f::filled(w, h, [0.5, 0.7, 1.4]);
        let l = img.map(crate::local::log_lum);
        let mut mattes = Mattes::default();
        mattes.push(MatteKind::Sky, left_matte(w, h, 0.5));
        let f = frame(w, h);
        let heuristic = shape_alpha(&MaskShape::Sky, &f, w, h, &img, &l, 0.0, None);
        let matte = shape_alpha(&MaskShape::Sky, &f, w, h, &img, &l, 0.0, Some(&mattes));
        assert!(heuristic.get(150, 5) > 0.9 && heuristic.get(25, 90) < 0.1, "{} {}", heuristic.get(150, 5), heuristic.get(25, 90));
        assert!(matte.get(150, 5) < 0.01 && matte.get(25, 90) > 0.99, "{} {}", matte.get(150, 5), matte.get(25, 90));
        // mattes of other kinds leave the heuristic alone
        let mut hair = Mattes::default();
        hair.push(MatteKind::Hair, left_matte(w, h, 0.5));
        assert_eq!(shape_alpha(&MaskShape::Sky, &f, w, h, &img, &l, 0.0, Some(&hair)), heuristic);
        // rotated 90° clockwise, the source's left half is the top half; a preview and a larger
        // render agree
        let s = lightcraft_develop::DevelopSettings { orientation: Orientation::Rotate90, ..Default::default() };
        let f = Frame::new(w, h, &s, true);
        for (ow, oh) in [(100, 200), (25, 50)] {
            let small = Rgb32f::filled(ow, oh, [0.5, 0.7, 1.4]);
            let a = shape_alpha(&MaskShape::Sky, &f, ow, oh, &small, &small.map(crate::local::log_lum), 0.0, Some(&mattes));
            let at = |x: f64, y: f64| a.get((x * ow as f64) as usize, (y * oh as f64) as usize);
            assert!(at(0.5, 0.2) > 0.99 && at(0.5, 0.8) < 0.01, "{ow}×{oh}: {} {}", at(0.5, 0.2), at(0.5, 0.8));
        }
    }

    /// Subject, Background and People use person mattes: the whole person is the union of the
    /// parts; a People part without a matte keeps the heuristic.
    #[test]
    fn people_and_subject_use_person_mattes() {
        let (w, h) = (200, 100);
        let img = Rgb32f::filled(w, h, [0.2; 3]);
        let l = img.map(crate::local::log_lum);
        let mut mattes = Mattes::default();
        mattes.push(MatteKind::Skin, left_matte(w, h, 0.25));
        let right: Image<u8> = Image::from_fn(w / 2, h / 2, |x, _| if x >= 75 { 255 } else { 0 });
        mattes.push(MatteKind::Hair, right);
        let f = frame(w, h);
        let eval = |shape: &MaskShape| shape_alpha(shape, &f, w, h, &img, &l, 0.0, Some(&mattes));
        let subject = eval(&MaskShape::Subject);
        assert!(subject.get(10, 50) > 0.99 && subject.get(190, 50) > 0.99 && subject.get(100, 50) < 0.01);
        let background = eval(&MaskShape::Background);
        assert!(background.get(100, 50) > 0.99 && background.get(10, 50) < 0.01);
        let hair = eval(&MaskShape::People { person: 0, parts: vec!["Hair".into()] });
        assert!(hair.get(190, 50) > 0.99 && hair.get(10, 50) < 0.01);
        let lips = MaskShape::People { person: 0, parts: vec!["Lips".into()] };
        assert_eq!(eval(&lips), shape_alpha(&lips, &f, w, h, &img, &l, 0.0, None));
    }
}
