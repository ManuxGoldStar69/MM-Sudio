//! SmartAlbums-style Photobook & Pre-Press Engine:
//! - Standard 300 DPI physical formats (8×8", 10×10", 12×12" open to 12×24", 11×14", custom)
//! - Color-coded pre-press guides: Bleed (3–5 mm red), Trim (solid black/white), Safe Zone (10–15 mm green dashed),
//!   Center Gutter / Crease (Layflat 180° fold line vs Traditional bound 10–15 mm shaded gutter loss)
//! - Preventive Face & Text warning detector when a face or caption crosses the center fold or exits Safe Zone
//! - Binary Space Partitioning (BSP) + Justified Grid Auto-Layout for 1..8 photos per spread with instant
//!   mathematical permutation cycling (`variation`) and Hardcover Wrap (`18 mm` wrap + dynamic spine width)
//! - Three real 300 DPI production export pipelines:
//!   * Mode A (`LayflatSpreads`): Sequential spread JPEGs (`Pliego_01.jpg`, `Pliego_02.jpg`, ..., `Portada.jpg`) in sRGB
//!   * Mode B (`SequentialPdf`): Single-page cut sequential PDF (Page 1 starts as odd right-hand page) with bleed, gutter & crop marks
//!   * Mode C (`SaddleStitch`): Automatic multiple-of-4 booklet imposition (`Tiro` & `Retiro` using formula `N + 1`)

use serde::{Deserialize, Serialize};

use super::PageRaster;
use super::compose::{Image16, PxRect, rec2020_from_srgb8};
use super::layout::{Orientation, Rect};
use super::template::PrintSettings;

/// Binding and print-shop output mode at 300 DPI.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum BindingMode {
    /// Modo A — Laboratorio Fotográfico Layflat 180° (Spreads Secuenciales `Pliego_01.jpg`, `Pliego_02.jpg` + `Portada.jpg`).
    #[default]
    LayflatSpreads,
    /// Modo B — Imprenta Comercial PDF Secuencial (Corte página por página, Página 1 impar derecha, con marcas de corte).
    SequentialPdf,
    /// Modo C — Imposición Automática para Cuadernillo (Saddle-Stitch / Compaginado múltiplo de 4, Tiro y Retiro `N + 1`).
    SaddleStitch,
}

impl BindingMode {
    pub fn label_es(self) -> &'static str {
        match self {
            BindingMode::LayflatSpreads => "Modo A — Laboratorio Layflat 180° (Pliegos JPG)",
            BindingMode::SequentialPdf => "Modo B — Imprenta PDF Secuencial (Página por página)",
            BindingMode::SaddleStitch => "Modo C — Cuadernillo Compaginado (Saddle-Stitch Tiro/Retiro)",
        }
    }
}

/// Photobook & SmartAlbums configuration stored inside [`PrintSettings`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub struct BookSettings {
    /// Whether SmartAlbums Photobook mode is active in the Print workspace.
    pub enabled: bool,
    /// Work by 2-page spreads (Left Page + Right Page) instead of single pages.
    pub spread_mode: bool,
    /// Print-shop binding and output mode (Mode A Layflat, Mode B Sequential PDF, Mode C Saddle-Stitch).
    pub binding_mode: BindingMode,
    /// Include/preview Hardcover Wrap (`18 mm` wrap-around margin + calculated spine width).
    pub hardcover_wrap: bool,
    /// Hardcover wrap-around fold margin in millimetres (default `18.0 mm`).
    pub wrap_mm: f64,
    /// Paper thickness per sheet/spread in millimetres (default `0.38 mm`), used to calculate spine width.
    pub paper_thickness_mm: f64,
    /// Internal gap between photo cells on a spread in millimetres (default `4.0 mm`).
    pub gap_mm: f64,
    /// Target number of photos per spread when auto-paginating the photobook (`1..=8`, default `4`).
    #[serde(default = "default_photos_per_spread")]
    pub photos_per_spread: u8,
    /// Whether auto-layout cells should extend to full bleed on outer edges when `variation` selects a full-bleed design.
    pub allow_full_bleed: bool,
    /// Per-spread layout variation index (`0..`) for instant mathematical permutation cycling.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub spread_variations: Vec<usize>,
}

fn default_photos_per_spread() -> u8 {
    4
}

impl Default for BookSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            spread_mode: true,
            binding_mode: BindingMode::LayflatSpreads,
            hardcover_wrap: false,
            wrap_mm: 18.0,
            paper_thickness_mm: 0.38,
            gap_mm: 4.0,
            photos_per_spread: 4,
            allow_full_bleed: false,
            spread_variations: Vec::new(),
        }
    }
}

impl BookSettings {
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    pub fn validate(&self) -> Result<(), String> {
        let ok = |v: f64, lo: f64, hi: f64| v.is_finite() && (lo..=hi).contains(&v);
        if !ok(self.wrap_mm, 0.0, 60.0) {
            return Err("book.wrapMm must be 0 to 60 mm".into());
        }
        if !ok(self.paper_thickness_mm, 0.01, 5.0) {
            return Err("book.paperThicknessMm must be 0.01 to 5.0 mm".into());
        }
        if !ok(self.gap_mm, 0.0, 50.0) {
            return Err("book.gapMm must be 0 to 50 mm".into());
        }
        if !(1..=8).contains(&self.photos_per_spread) {
            return Err("book.photosPerSpread must be 1 to 8".into());
        }
        Ok(())
    }

    /// Calculates the physical spine width in millimetres for `spread_count` spreads (or sheets).
    pub fn spine_width_mm(&self, spread_count: usize) -> f64 {
        let sheets = spread_count.max(1) as f64;
        (sheets * self.paper_thickness_mm + 2.0).clamp(3.0, 80.0)
    }

    /// Current layout variation index for `spread` (zero-based).
    pub fn variation_for(&self, spread: usize) -> usize {
        self.spread_variations.get(spread).copied().unwrap_or(0)
    }

    /// Sets the layout variation index for `spread` (zero-based).
    pub fn set_variation(&mut self, spread: usize, variation: usize) {
        if spread >= 500 {
            return;
        }
        if self.spread_variations.len() <= spread {
            self.spread_variations.resize(spread + 1, 0);
        }
        self.spread_variations[spread] = variation;
    }
}

/// Standard Photobook album presets at 300 DPI: `(id, label, single_page_w_in, single_page_h_in)`.
/// When `spread_mode` is true, the open spread is `(2 * single_page_w_in) × single_page_h_in`.
pub const ALBUM_PRESETS: &[(&str, &str, f64, f64)] = &[
    ("8x8", "8×8\" Álbum Cuadrado (Abierto 8×16\")", 8.0, 8.0),
    ("10x10", "10×10\" Álbum Estándar (Abierto 10×20\")", 10.0, 10.0),
    ("12x12", "12×12\" Álbum Bodas Pro (Abierto 12×24\")", 12.0, 12.0),
    ("11x14", "11×14\" Álbum Editorial (Abierto 11×28\")", 14.0, 11.0),
    ("8x10", "8×10\" Retrato / Libro Vertical", 8.0, 10.0),
    ("a4", "A4 Fotolibro (21×29.7 cm)", 210.0 / 25.4, 297.0 / 25.4),
    ("custom", "Tamaño Personalizado (in / cm / mm)", 10.0, 10.0),
];

/// Applies a named photobook format preset (`8x8`, `10x10`, `12x12`, `11x14`, `8x10`, `a4`, `custom`)
/// to `settings`, configuring 300 DPI, pre-press guides, and spread dimensions.
pub fn apply_format_preset(
    settings: &mut PrintSettings,
    preset: &str,
    spread_mode: bool,
    custom_w_in: Option<f64>,
    custom_h_in: Option<f64>,
) -> Result<(), String> {
    settings.book.enabled = true;
    settings.book.spread_mode = spread_mode;
    settings.job.ppi = 300.0;
    settings.layout.style = super::layout::Style::CustomPackage;
    settings.image.zoom_to_fill = true;
    settings.guides.show_bleed = true;
    settings.guides.show_trim = true;
    settings.guides.show_safe_zone = true;
    settings.guides.show_gutter = true;

    // Small minimal margin so cells can reach the safe zone or trim edge cleanly
    settings.layout.page.margins = super::layout::Margins::uniform(0.0);
    settings.layout.spacing = (settings.book.gap_mm / 25.4).clamp(0.0, 2.0);

    let p_lower = preset.trim().to_ascii_lowercase();
    match p_lower.as_str() {
        "8x8" | "8x16" => {
            if spread_mode {
                settings.layout.page.paper = "8x16".into();
                settings.layout.page.orientation = Orientation::Landscape;
            } else {
                settings.layout.page.paper = "8x8".into();
                settings.layout.page.orientation = Orientation::Portrait;
            }
        }
        "10x10" | "10x20" => {
            if spread_mode {
                settings.layout.page.paper = "10x20".into();
                settings.layout.page.orientation = Orientation::Landscape;
            } else {
                settings.layout.page.paper = "10x10".into();
                settings.layout.page.orientation = Orientation::Portrait;
            }
        }
        "12x12" | "12x24" => {
            if spread_mode {
                settings.layout.page.paper = "12x24".into();
                settings.layout.page.orientation = Orientation::Landscape;
            } else {
                settings.layout.page.paper = "12x12".into();
                settings.layout.page.orientation = Orientation::Portrait;
            }
        }
        "11x14" | "11x28" => {
            if spread_mode {
                settings.layout.page.paper = "11x28".into();
                settings.layout.page.orientation = Orientation::Landscape;
            } else {
                settings.layout.page.paper = "11x14".into();
                settings.layout.page.orientation = Orientation::Landscape;
            }
        }
        "8x10" => {
            if spread_mode {
                settings.layout.page.paper = "custom".into();
                settings.layout.page.custom_width = 10.0;
                settings.layout.page.custom_height = 16.0;
                settings.layout.page.orientation = Orientation::Landscape;
            } else {
                settings.layout.page.paper = "8x10".into();
                settings.layout.page.orientation = Orientation::Portrait;
            }
        }
        "a4" => {
            if spread_mode {
                settings.layout.page.paper = "a3".into();
                settings.layout.page.orientation = Orientation::Landscape;
            } else {
                settings.layout.page.paper = "a4".into();
                settings.layout.page.orientation = Orientation::Portrait;
            }
        }
        "custom" => {
            let single_w = custom_w_in.unwrap_or(10.0).clamp(2.0, 30.0);
            let single_h = custom_h_in.unwrap_or(10.0).clamp(2.0, 30.0);
            settings.layout.page.paper = "custom".into();
            if spread_mode {
                let total_w = (single_w * 2.0).clamp(2.0, 60.0);
                settings.layout.page.custom_width = single_h.min(total_w);
                settings.layout.page.custom_height = single_h.max(total_w);
                settings.layout.page.orientation = if total_w >= single_h {
                    Orientation::Landscape
                } else {
                    Orientation::Portrait
                };
            } else {
                settings.layout.page.custom_width = single_w.min(single_h);
                settings.layout.page.custom_height = single_w.max(single_h);
                settings.layout.page.orientation = if single_w > single_h {
                    Orientation::Landscape
                } else {
                    Orientation::Portrait
                };
            }
        }
        other => {
            if super::layout::paper_size(other).is_some() {
                settings.layout.page.paper = other.into();
            } else {
                return Err(format!("unknown format preset `{preset}`"));
            }
        }
    }
    settings.layout.fit_cells()?;
    settings.validate()
}

// ---- SmartAlbums Mathematical Auto-Layout (BSP & Justified Grid) ----------------------------

/// Number of distinct mathematical layout variations available for `n` photos on a spread/page.
pub fn variation_count(n: usize, spread_mode: bool) -> usize {
    match n {
        0 => 1,
        1 => if spread_mode { 6 } else { 4 },
        2 => if spread_mode { 8 } else { 5 },
        3 => if spread_mode { 10 } else { 6 },
        4 => if spread_mode { 12 } else { 8 },
        _ => 14,
    }
}

/// Computes non-overlapping cell rectangles in page inches for `1..=8` photos with given
/// aspect ratios (`width / height`), respecting Safe Zone, Center Gutter, and internal `gap_in`.
///
/// Cycling `variation` (`0, 1, 2, ...`) steps through distinct mathematical permutations
/// (Left/Right page balance, Hero + Grid, Justified Rows/Columns, Binary Space Partitioning,
/// Panoramic Spreads) without deforming the photos.
pub fn smart_layout_spread(
    page_w: f64,
    page_h: f64,
    spread_mode: bool,
    safe_in: f64,
    gutter_in: f64,
    gap_in: f64,
    binding_mode: BindingMode,
    aspects: &[f64],
    variation: usize,
) -> Vec<Rect> {
    let n = aspects.len();
    if n == 0 || page_w <= 0.2 || page_h <= 0.2 {
        return Vec::new();
    }
    let safe = safe_in.clamp(0.0, (page_w.min(page_h) * 0.22).max(0.05));
    let gap = gap_in.clamp(0.0, (page_w.min(page_h) * 0.08).max(0.02));
    let gutter_half = if spread_mode {
        match binding_mode {
            BindingMode::LayflatSpreads => (gap * 0.5).max(safe * 0.5),
            BindingMode::SequentialPdf | BindingMode::SaddleStitch => (gutter_in * 0.5 + safe * 0.6).max(gap),
        }
    } else {
        0.0
    };

    let full_safe = Rect::new(
        safe,
        safe,
        (page_w - 2.0 * safe).max(0.2),
        (page_h - 2.0 * safe).max(0.2),
    );

    if !spread_mode {
        return layout_region_bsp(full_safe, aspects, gap, variation);
    }

    let mid_x = page_w * 0.5;
    let left_page = Rect::new(
        safe,
        safe,
        (mid_x - safe - gutter_half).max(0.2),
        (page_h - 2.0 * safe).max(0.2),
    );
    let right_page = Rect::new(
        mid_x + gutter_half,
        safe,
        (page_w - safe - (mid_x + gutter_half)).max(0.2),
        (page_h - 2.0 * safe).max(0.2),
    );
    let left_full_bleed = Rect::new(0.0, 0.0, (mid_x - gutter_half * 0.25).max(0.2), page_h);
    let right_full_bleed = Rect::new(
        mid_x + gutter_half * 0.25,
        0.0,
        (page_w - (mid_x + gutter_half * 0.25)).max(0.2),
        page_h,
    );

    let total_vars = variation_count(n, true);
    let v = variation % total_vars;

    match n {
        1 => {
            let ar = aspects[0].clamp(0.4, 3.0);
            match v {
                // Var 0: Fitted respecting original aspect ratio on Right Page (classic SmartAlbums opening)
                0 => vec![fit_aspect_in_box(right_page, ar)],
                // Var 1: Fitted respecting original aspect ratio on Left Page
                1 => vec![fit_aspect_in_box(left_page, ar)],
                // Var 2: Full-page right (edge to safe zone)
                2 => vec![right_page],
                // Var 3: Full-page left (edge to safe zone)
                3 => vec![left_page],
                // Var 4: Panoramic spread across both pages (ideal for Layflat 180°)
                4 => {
                    if binding_mode == BindingMode::LayflatSpreads {
                        vec![fit_aspect_in_box(full_safe, ar.max(1.5))]
                    } else {
                        vec![fit_aspect_in_box(right_page.shrink_frac(0.08), ar)]
                    }
                }
                // Var 5: Full-spread edge-to-edge hero (Layflat) or editorial inset
                _ => {
                    if binding_mode == BindingMode::LayflatSpreads {
                        vec![full_safe]
                    } else {
                        vec![left_full_bleed.intersect_or(left_page)]
                    }
                }
            }
        }
        2 => {
            let (a0, a1) = (aspects[0].clamp(0.4, 3.0), aspects[1].clamp(0.4, 3.0));
            match v {
                // Var 0: 1 on Left (aspect-fitted), 1 on Right (aspect-fitted)
                0 => vec![fit_aspect_in_box(left_page, a0), fit_aspect_in_box(right_page, a1)],
                // Var 1: Full-box Left + aspect-fitted Right
                1 => vec![left_page, fit_aspect_in_box(right_page, a1)],
                // Var 2: Aspect-fitted Left + Full-box Right
                2 => vec![fit_aspect_in_box(left_page, a0), right_page],
                // Var 3: Both fill their respective page safe zones
                3 => vec![left_page, right_page],
                // Var 4: Both side-by-side or stacked on Right page (Left page clean white space)
                4 => layout_region_bsp(right_page, &[a0, a1], gap, 0),
                // Var 5: Both side-by-side or stacked on Left page (Right page clean white space)
                5 => layout_region_bsp(left_page, &[a0, a1], gap, 0),
                // Var 6: Full-bleed Left + framed inset Right
                6 => vec![left_full_bleed.intersect_or(left_page), fit_aspect_in_box(right_page.shrink_frac(0.10), a1)],
                // Var 7: Framed inset Left + Full-bleed Right
                _ => vec![fit_aspect_in_box(left_page.shrink_frac(0.10), a0), right_full_bleed.intersect_or(right_page)],
            }
        }
        _ => {
            // For 3..=8+ photos: generate diverse splits between Left Page (`k` photos) and Right Page (`n - k` photos)
            let splits: &[(usize, usize)] = match n {
                3 => &[
                    (1, 2),
                    (2, 1),
                    (1, 2),
                    (2, 1),
                    (0, 3),
                    (3, 0),
                    (1, 2),
                    (2, 1),
                    (1, 2),
                    (2, 1),
                ],
                4 => &[
                    (2, 2),
                    (1, 3),
                    (3, 1),
                    (2, 2),
                    (1, 3),
                    (3, 1),
                    (2, 2),
                    (0, 4),
                    (4, 0),
                    (2, 2),
                    (1, 3),
                    (3, 1),
                ],
                _ => &[],
            };

            let (left_n, _right_n) = if let Some(&(l, r)) = splits.get(v) {
                (l, r)
            } else {
                // Cycle through balanced and asymmetric left/right splits
                match v % 5 {
                    0 => (n / 2, n - n / 2),
                    1 => ((n / 2).saturating_sub(1).max(1), n - (n / 2).saturating_sub(1).max(1)),
                    2 => ((n - n / 2).min(n - 1), n - (n - n / 2).min(n - 1)),
                    3 => (1, n - 1),
                    _ => (n - 1, 1),
                }
            };

            let sub_var_l = v / 2;
            let sub_var_r = (v + 1) / 2;
            let mut out = Vec::with_capacity(n);
            if left_n > 0 {
                out.extend(layout_region_bsp(left_page, &aspects[..left_n], gap, sub_var_l));
            }
            if left_n < n {
                out.extend(layout_region_bsp(right_page, &aspects[left_n..], gap, sub_var_r));
            }
            out
        }
    }
}

trait RectExt {
    fn shrink_frac(&self, frac: f64) -> Rect;
    fn intersect_or(&self, fallback: Rect) -> Rect;
}

impl RectExt for Rect {
    fn shrink_frac(&self, frac: f64) -> Rect {
        let dx = self.w * frac;
        let dy = self.h * frac;
        Rect::new(self.x + dx, self.y + dy, (self.w - 2.0 * dx).max(0.2), (self.h - 2.0 * dy).max(0.2))
    }

    fn intersect_or(&self, fallback: Rect) -> Rect {
        if self.w > 0.2 && self.h > 0.2 { *self } else { fallback }
    }
}

/// Centers a box of aspect ratio `ar` (`w / h`) inside `bounds` while using at least 78% of the bounds'
/// dominant axis so the layout looks like an editorial SmartAlbums frame.
fn fit_aspect_in_box(bounds: Rect, ar: f64) -> Rect {
    let ar = ar.clamp(0.4, 3.0);
    let box_ar = bounds.w / bounds.h.max(1e-6);
    let (w, h) = if ar >= box_ar {
        let w = bounds.w;
        let h = (w / ar).clamp(bounds.h * 0.45, bounds.h);
        (w, h)
    } else {
        let h = bounds.h;
        let w = (h * ar).clamp(bounds.w * 0.45, bounds.w);
        (w, h)
    };
    Rect::new(
        bounds.x + (bounds.w - w) * 0.5,
        bounds.y + (bounds.h - h) * 0.5,
        w.max(0.15),
        h.max(0.15),
    )
}

/// Binary Space Partitioning (BSP) + Justified Mosaic layout inside a single page region `bounds`.
/// Produces clean geometric Magnetic Frames (Marcos Magnéticos) that fill `bounds` harmoniously without skinny strips.
fn layout_region_bsp(bounds: Rect, aspects: &[f64], gap: f64, variation: usize) -> Vec<Rect> {
    layout_region_bsp_depth(bounds, aspects, gap, variation, 0)
}

fn layout_region_bsp_depth(bounds: Rect, aspects: &[f64], gap: f64, variation: usize, depth: usize) -> Vec<Rect> {
    let n = aspects.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        if depth > 0 {
            return vec![bounds];
        }
        return match variation % 3 {
            0 => vec![bounds],
            1 => vec![fit_aspect_in_box(bounds, aspects[0])],
            _ => vec![bounds.shrink_frac(0.05)],
        };
    }
    if n == 2 {
        let a0 = aspects[0].clamp(0.45, 2.5);
        let a1 = aspects[1].clamp(0.45, 2.5);
        // When inside a recursive subdivision, split along the longer axis of `bounds` so frames stay well-proportioned
        let side_by_side = if depth > 0 {
            bounds.w >= bounds.h * 1.05
        } else {
            let prefer_cols = (a0 + a1) < (bounds.w / bounds.h.max(1e-6)) * 1.45;
            if variation % 2 == 0 { prefer_cols } else { !prefer_cols }
        };
        if side_by_side {
            let avail_w = (bounds.w - gap).max(0.2);
            let ratio = match variation % 4 {
                0 => 0.5,
                1 => (a0 / (a0 + a1)).clamp(0.38, 0.62),
                2 => 0.58,
                _ => 0.42,
            };
            let w0 = (avail_w * ratio).max(0.15);
            let w1 = (avail_w - w0).max(0.15);
            return vec![
                Rect::new(bounds.x, bounds.y, w0, bounds.h),
                Rect::new(bounds.x + w0 + gap, bounds.y, w1, bounds.h),
            ];
        } else {
            let avail_h = (bounds.h - gap).max(0.2);
            let inv0 = 1.0 / a0;
            let inv1 = 1.0 / a1;
            let ratio = match variation % 4 {
                0 => 0.5,
                1 => (inv0 / (inv0 + inv1)).clamp(0.38, 0.62),
                2 => 0.58,
                _ => 0.42,
            };
            let h0 = (avail_h * ratio).max(0.15);
            let h1 = (avail_h - h0).max(0.15);
            return vec![
                Rect::new(bounds.x, bounds.y, bounds.w, h0),
                Rect::new(bounds.x, bounds.y + h0 + gap, bounds.w, h1),
            ];
        }
    }

    if n == 3 {
        let avail_w = (bounds.w - gap).max(0.2);
        let avail_h = (bounds.h - gap).max(0.2);
        return match variation % 4 {
            // Var 0: 1 Hero Left (54%) + 2 Stacked Right (46%)
            0 => {
                let w0 = (avail_w * 0.54).max(0.15);
                let w1 = (avail_w - w0).max(0.15);
                let h_top = (avail_h * 0.5).max(0.15);
                let h_bot = (avail_h - h_top).max(0.15);
                vec![
                    Rect::new(bounds.x, bounds.y, w0, bounds.h),
                    Rect::new(bounds.x + w0 + gap, bounds.y, w1, h_top),
                    Rect::new(bounds.x + w0 + gap, bounds.y + h_top + gap, w1, h_bot),
                ]
            }
            // Var 1: 1 Hero Top (54%) + 2 Side-by-Side Bottom (46%)
            1 => {
                let h0 = (avail_h * 0.54).max(0.15);
                let h1 = (avail_h - h0).max(0.15);
                let w_l = (avail_w * 0.5).max(0.15);
                let w_r = (avail_w - w_l).max(0.15);
                vec![
                    Rect::new(bounds.x, bounds.y, bounds.w, h0),
                    Rect::new(bounds.x, bounds.y + h0 + gap, w_l, h1),
                    Rect::new(bounds.x + w_l + gap, bounds.y + h0 + gap, w_r, h1),
                ]
            }
            // Var 2: 2 Stacked Left (46%) + 1 Hero Right (54%)
            2 => {
                let w0 = (avail_w * 0.46).max(0.15);
                let w1 = (avail_w - w0).max(0.15);
                let h_top = (avail_h * 0.5).max(0.15);
                let h_bot = (avail_h - h_top).max(0.15);
                vec![
                    Rect::new(bounds.x, bounds.y, w0, h_top),
                    Rect::new(bounds.x, bounds.y + h_top + gap, w0, h_bot),
                    Rect::new(bounds.x + w0 + gap, bounds.y, w1, bounds.h),
                ]
            }
            // Var 3: 2 Side-by-Side Top (46%) + 1 Hero Bottom (54%)
            _ => {
                let h0 = (avail_h * 0.46).max(0.15);
                let h1 = (avail_h - h0).max(0.15);
                let w_l = (avail_w * 0.5).max(0.15);
                let w_r = (avail_w - w_l).max(0.15);
                vec![
                    Rect::new(bounds.x, bounds.y, w_l, h0),
                    Rect::new(bounds.x + w_l + gap, bounds.y, w_r, h0),
                    Rect::new(bounds.x, bounds.y + h0 + gap, bounds.w, h1),
                ]
            }
        };
    }

    if n == 4 {
        let avail_w = (bounds.w - gap).max(0.2);
        let avail_h = (bounds.h - gap).max(0.2);
        return match variation % 4 {
            // Var 0: Clean 2×2 Grid
            0 => {
                let w0 = (avail_w * 0.5).max(0.15);
                let w1 = (avail_w - w0).max(0.15);
                let h0 = (avail_h * 0.5).max(0.15);
                let h1 = (avail_h - h0).max(0.15);
                vec![
                    Rect::new(bounds.x, bounds.y, w0, h0),
                    Rect::new(bounds.x + w0 + gap, bounds.y, w1, h0),
                    Rect::new(bounds.x, bounds.y + h0 + gap, w0, h1),
                    Rect::new(bounds.x + w0 + gap, bounds.y + h0 + gap, w1, h1),
                ]
            }
            // Var 1: Asymmetric Mosaic (Top row 58%/42%, Bottom row 42%/58%)
            1 => {
                let w_top_l = (avail_w * 0.58).max(0.15);
                let w_top_r = (avail_w - w_top_l).max(0.15);
                let w_bot_l = (avail_w * 0.42).max(0.15);
                let w_bot_r = (avail_w - w_bot_l).max(0.15);
                let h0 = (avail_h * 0.52).max(0.15);
                let h1 = (avail_h - h0).max(0.15);
                vec![
                    Rect::new(bounds.x, bounds.y, w_top_l, h0),
                    Rect::new(bounds.x + w_top_l + gap, bounds.y, w_top_r, h0),
                    Rect::new(bounds.x, bounds.y + h0 + gap, w_bot_l, h1),
                    Rect::new(bounds.x + w_bot_l + gap, bounds.y + h0 + gap, w_bot_r, h1),
                ]
            }
            // Var 2: 1 Hero Left (52%) + 3 Stacked Right (48%)
            2 => {
                let w0 = (avail_w * 0.52).max(0.15);
                let w1 = (avail_w - w0).max(0.15);
                let avail_h3 = (bounds.h - 2.0 * gap).max(0.3);
                let h_each = (avail_h3 / 3.0).max(0.1);
                let h_last = (avail_h3 - 2.0 * h_each).max(0.1);
                vec![
                    Rect::new(bounds.x, bounds.y, w0, bounds.h),
                    Rect::new(bounds.x + w0 + gap, bounds.y, w1, h_each),
                    Rect::new(bounds.x + w0 + gap, bounds.y + h_each + gap, w1, h_each),
                    Rect::new(bounds.x + w0 + gap, bounds.y + 2.0 * (h_each + gap), w1, h_last),
                ]
            }
            // Var 3: 1 Hero Top (52%) + 3 Columns Bottom (48%)
            _ => {
                let h0 = (avail_h * 0.52).max(0.15);
                let h1 = (avail_h - h0).max(0.15);
                let avail_w3 = (bounds.w - 2.0 * gap).max(0.3);
                let w_each = (avail_w3 / 3.0).max(0.1);
                let w_last = (avail_w3 - 2.0 * w_each).max(0.1);
                vec![
                    Rect::new(bounds.x, bounds.y, bounds.w, h0),
                    Rect::new(bounds.x, bounds.y + h0 + gap, w_each, h1),
                    Rect::new(bounds.x + w_each + gap, bounds.y + h0 + gap, w_each, h1),
                    Rect::new(bounds.x + 2.0 * (w_each + gap), bounds.y + h0 + gap, w_last, h1),
                ]
            }
        };
    }

    // For n >= 5: alternate split axis strictly by depth & bounds aspect so we always get a 2D mosaic
    let split_k = (n / 2).max(1);
    let split_horizontally = if depth > 0 {
        bounds.w > bounds.h * 1.15
    } else {
        variation % 2 == 0 && bounds.w >= bounds.h * 0.9
    };

    let (r0, r1) = if split_horizontally {
        let avail_w = (bounds.w - gap).max(0.2);
        let weight = ((split_k as f64) / (n as f64)).clamp(0.35, 0.65);
        let w0 = (avail_w * weight).max(0.15);
        let w1 = (avail_w - w0).max(0.15);
        (
            Rect::new(bounds.x, bounds.y, w0, bounds.h),
            Rect::new(bounds.x + w0 + gap, bounds.y, w1, bounds.h),
        )
    } else {
        let avail_h = (bounds.h - gap).max(0.2);
        let weight = ((split_k as f64) / (n as f64)).clamp(0.35, 0.65);
        let h0 = (avail_h * weight).max(0.15);
        let h1 = (avail_h - h0).max(0.15);
        (
            Rect::new(bounds.x, bounds.y, bounds.w, h0),
            Rect::new(bounds.x, bounds.y + h0 + gap, bounds.w, h1),
        )
    };

    let mut out = layout_region_bsp_depth(r0, &aspects[..split_k], gap, variation / 2, depth + 1);
    out.extend(layout_region_bsp_depth(r1, &aspects[split_k..], gap, (variation + 1) / 2, depth + 1));
    out
}

// ---- Pre-Press Preventive Warnings (Faces / Text in Gutter or Safe Zone) --------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum WarningKind {
    FaceInGutter,
    FaceOutsideSafeZone,
    TextInGutter,
    TextOutsideSafeZone,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PrepressWarning {
    pub cell_id: u32,
    pub kind: WarningKind,
    /// Danger rectangle in page inches so the canvas can highlight the exact zone.
    pub rect_in: Rect,
    pub message: String,
}

/// Checks every cell on `page` (zero-based) for faces or captions that cross the Center Gutter / Crease
/// or fall outside the Safe Zone (`safe_zone_mm`).
///
/// `face_lookup(photo_id)` returns normalized `[x, y, w, h]` face boxes in `[0, 1]` on the photo.
pub fn inspect_spread_warnings(
    settings: &PrintSettings,
    page: u32,
    mut face_lookup: impl FnMut(u64) -> Vec<[f64; 4]>,
) -> Vec<PrepressWarning> {
    let Ok((pw, ph)) = settings.layout.page_size() else {
        return Vec::new();
    };
    let safe_in = (settings.guides.safe_zone_mm / 25.4).clamp(0.0, pw.min(ph) * 0.25);
    let gutter_in = (settings.guides.gutter_mm / 25.4).clamp(0.05, pw * 0.20);
    let spread_mode = settings.book.spread_mode;
    let mid_x = pw * 0.5;
    let gutter_half = match settings.book.binding_mode {
        BindingMode::LayflatSpreads => (2.5_f64 / 25.4).max(0.08), // 2.5mm crease tolerance on Layflat
        BindingMode::SequentialPdf | BindingMode::SaddleStitch => (gutter_in * 0.5).max(0.18),
    };
    let gutter_rect = Rect::new(mid_x - gutter_half, 0.0, gutter_half * 2.0, ph);
    let safe_rect = Rect::new(safe_in, safe_in, (pw - 2.0 * safe_in).max(0.1), (ph - 2.0 * safe_in).max(0.1));

    let mut warnings = Vec::new();
    for cell in settings.layout.cells_on_page(page) {
        let cr = cell.rect;
        // 1. Check cell caption / text box if present
        if let Some(cap) = &cell.caption
            && !cap.trim().is_empty()
        {
            let text_rect = Rect::new(cr.x, (cr.bottom() - 0.35).max(cr.y), cr.w, 0.35_f64.min(cr.h));
            if spread_mode && text_rect.overlaps(&gutter_rect) {
                warnings.push(PrepressWarning {
                    cell_id: cell.id,
                    kind: WarningKind::TextInGutter,
                    rect_in: text_rect,
                    message: format!("Texto en celda #{} cruza el pliegue central / lomo", cell.id),
                });
            }
            if !safe_rect.contains(&text_rect) {
                warnings.push(PrepressWarning {
                    cell_id: cell.id,
                    kind: WarningKind::TextOutsideSafeZone,
                    rect_in: text_rect,
                    message: format!("Texto en celda #{} sale de la Zona Segura ({:.0} mm)", cell.id, settings.guides.safe_zone_mm),
                });
            }
        }

        // 2. Check detected faces (or central head zone when a cell straddles the spine or trim edge)
        let faces = cell.photo.map(&mut face_lookup).unwrap_or_default();
        for [fx, fy, fw, fh] in faces {
            // Map normalized face box through cell zoom/pan into page inches
            let z = cell.zoom.clamp(1.0, 5.0);
            let cx = 0.5 - cell.pan_x * 0.5 * (1.0 - 1.0 / z);
            let cy = 0.5 - cell.pan_y * 0.5 * (1.0 - 1.0 / z);
            let u0 = ((fx - cx) * z + 0.5).clamp(0.0, 1.0);
            let v0 = ((fy - cy) * z + 0.5).clamp(0.0, 1.0);
            let u1 = ((fx + fw - cx) * z + 0.5).clamp(0.0, 1.0);
            let v1 = ((fy + fh - cy) * z + 0.5).clamp(0.0, 1.0);
            if u1 <= u0 + 0.01 || v1 <= v0 + 0.01 {
                continue;
            }
            let face_in = Rect::new(
                cr.x + u0 * cr.w,
                cr.y + v0 * cr.h,
                (u1 - u0) * cr.w,
                (v1 - v0) * cr.h,
            );
            if spread_mode && face_in.overlaps(&gutter_rect) {
                warnings.push(PrepressWarning {
                    cell_id: cell.id,
                    kind: WarningKind::FaceInGutter,
                    rect_in: face_in,
                    message: format!("Rostro en celda #{} cruza la línea central del pliegue/lomo", cell.id),
                });
            } else if !safe_rect.contains(&face_in) {
                warnings.push(PrepressWarning {
                    cell_id: cell.id,
                    kind: WarningKind::FaceOutsideSafeZone,
                    rect_in: face_in,
                    message: format!("Rostro en celda #{} fuera de la Zona Segura ({:.0} mm del corte)", cell.id, settings.guides.safe_zone_mm),
                });
            }
        }

        // Also warn if a non-panoramic photo cell straddles the center gutter in traditional bound book mode
        if spread_mode
            && settings.book.binding_mode != BindingMode::LayflatSpreads
            && cr.x < mid_x - 0.05
            && cr.right() > mid_x + 0.05
            && !warnings.iter().any(|w| w.cell_id == cell.id && w.kind == WarningKind::FaceInGutter)
        {
            let cross = Rect::new(mid_x - gutter_half, cr.y, gutter_half * 2.0, cr.h);
            warnings.push(PrepressWarning {
                cell_id: cell.id,
                kind: WarningKind::FaceInGutter,
                rect_in: cross,
                message: format!(
                    "Celda #{} cruza la pérdida de papel en el lomo ({:.0} mm)",
                    cell.id, settings.guides.gutter_mm
                ),
            });
        }
    }
    warnings
}

// ---- Three Print-Shop / Photo-Lab Export Modes (300 DPI) -----------------------------------

/// One Saddle-Stitch imposition sheet side (`Frente / Tiro` or `Vuelta / Retiro`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SaddleSheetSide {
    /// 1-based physical sheet number (`1..=N/4`).
    pub sheet_number: usize,
    /// `true` = Frente (Tiro / Front), `false` = Vuelta (Retiro / Back).
    pub is_front: bool,
    /// 1-based page number placed on the Left half (`1..=N`). Pages `> original_pages` are blank guard pages.
    pub left_page: usize,
    /// 1-based page number placed on the Right half (`1..=N`). Pages `> original_pages` are blank guard pages.
    pub right_page: usize,
}

/// Computes the exact Saddle-Stitch (Cuadernillo Compaginado) imposition pairing using the `(N + 1)` formula:
/// - Ensures total pages `N` is a multiple of 4 (padding with blank guard pages if needed).
/// - For sheet `k = 1 .. N/4`:
///   * Hoja `k` Frente (Tiro): `[Página N - 2(k - 1) (Izquierda) | Página 2k - 1 (Derecha)]`
///   * Hoja `k` Vuelta (Retiro): `[Página 2k (Izquierda) | Página N - (2k - 1) (Derecha)]`
pub fn saddle_stitch_imposition(single_page_count: usize) -> (usize, Vec<SaddleSheetSide>) {
    let m = single_page_count.max(1);
    let total_n = m.div_ceil(4) * 4;
    let sheets = total_n / 4;
    let mut out = Vec::with_capacity(sheets * 2);
    for k in 1..=sheets {
        // Hoja k Frente (Tiro): [Página N - 2(k-1) | Página 2k - 1]
        let front_left = total_n - 2 * (k - 1);
        let front_right = 2 * k - 1;
        out.push(SaddleSheetSide {
            sheet_number: k,
            is_front: true,
            left_page: front_left,
            right_page: front_right,
        });
        // Hoja k Vuelta (Retiro): [Página 2k | Página N - (2k - 1)]
        let back_left = 2 * k;
        let back_right = total_n - (2 * k - 1);
        out.push(SaddleSheetSide {
            sheet_number: k,
            is_front: false,
            left_page: back_left,
            right_page: back_right,
        });
    }
    (total_n, out)
}

/// Splits a list of rendered spreads (each `PageRaster` being `[Left Page | Right Page]`) into individual
/// single-page rasters (`Página 1, 2, 3...`), or returns them as-is if `spread_mode` is false.
///
/// When `page_one_right` is true (standard commercial book convention: Page 1 starts as an odd right-hand page),
/// if the first spread has content on both halves, each spread yields `[Left, Right]` sequentially,
/// or if requested adds outer bleed (`bleed_px`) and spine gutter shift (`gutter_px`) with optional
/// guillotine corner crop marks (`crop_marks`).
pub fn split_spreads_to_single_pages(
    spreads: &[PageRaster],
    spread_mode: bool,
    background_srgb: [u8; 3],
    bleed_mm: f64,
    gutter_mm: f64,
    crop_marks: bool,
) -> Result<Vec<PageRaster>, String> {
    let Some(first) = spreads.first() else {
        return Err("no spreads to split".into());
    };
    let ppi = first.ppi;
    let bg = rec2020_from_srgb8(background_srgb);
    let ink = rec2020_from_srgb8(super::compose::mark_ink(background_srgb));

    let mut raw_pages: Vec<Image16> = Vec::new();
    for sp in spreads {
        let w = sp.width as usize;
        let h = sp.height as usize;
        let src = Image16 { w, h, rgb: sp.rgb.clone() };
        if spread_mode && w >= 4 {
            let half_w = w / 2;
            let mut left = Image16::filled(half_w, h, bg)?;
            left.blit(&src, 0, 0, 0, 0, half_w, h);
            let mut right = Image16::filled(half_w, h, bg)?;
            right.blit(&src, half_w, 0, 0, 0, half_w, h);
            raw_pages.push(left);
            raw_pages.push(right);
        } else {
            raw_pages.push(src);
        }
    }

    let bleed_px = ((bleed_mm.clamp(0.0, 15.0) / 25.4) * f64::from(ppi)).round() as usize;
    let gutter_px = ((gutter_mm.clamp(0.0, 25.0) / 25.4) * f64::from(ppi) * 0.5).round() as usize;
    let mark_pad_px = if crop_marks {
        ((6.0 / 25.4) * f64::from(ppi)).round() as usize
    } else {
        0
    };
    let pad = bleed_px + mark_pad_px;

    let mut out = Vec::with_capacity(raw_pages.len());
    for (idx, single) in raw_pages.into_iter().enumerate() {
        let page_num = idx + 1; // 1-based: odd = right-hand (recto), even = left-hand (verso)
        let is_odd_right = page_num % 2 == 1;
        let (sw, sh) = (single.w, single.h);
        let out_w = sw + 2 * pad;
        let out_h = sh + 2 * pad;
        let mut canvas = Image16::filled(out_w, out_h, bg)?;

        // Shift content slightly away from the binding spine (left edge on odd right-hand pages, right edge on even left-hand pages)
        let shift_x = if gutter_px > 0 && gutter_px < sw / 6 {
            if is_odd_right { gutter_px as isize } else { -(gutter_px as isize) }
        } else {
            0
        };
        let dx = (pad as isize + shift_x).clamp(0, out_w as isize - 1) as usize;
        let dy = pad;
        canvas.blit(&single, 0, 0, dx, dy, sw, sh);

        if crop_marks && pad >= 6 {
            draw_guillotine_crop_marks(
                &mut canvas,
                PxRect { x: pad, y: pad, w: sw, h: sh },
                ppi,
                ink,
            );
        }

        out.push(PageRaster {
            width: out_w as u32,
            height: out_h as u32,
            ppi,
            rgb: canvas.rgb,
        });
    }
    Ok(out)
}

/// Composes double-sided Saddle-Stitch (Cuadernillo Compaginado) imposition sheets from single pages
/// using [`saddle_stitch_imposition`] (formula `N + 1`, padded to a multiple of 4 with blank guard pages).
pub fn compose_saddle_stitch_sheets(
    spreads: &[PageRaster],
    spread_mode: bool,
    background_srgb: [u8; 3],
    bleed_mm: f64,
    crop_marks: bool,
) -> Result<(Vec<PageRaster>, Vec<SaddleSheetSide>, usize), String> {
    let Some(first) = spreads.first() else {
        return Err("no spreads to impose".into());
    };
    let ppi = first.ppi;
    let bg = rec2020_from_srgb8(background_srgb);
    let ink = rec2020_from_srgb8(super::compose::mark_ink(background_srgb));

    // Extract clean single pages (without per-page outer crop padding yet)
    let mut singles: Vec<Image16> = Vec::new();
    for sp in spreads {
        let w = sp.width as usize;
        let h = sp.height as usize;
        let src = Image16 { w, h, rgb: sp.rgb.clone() };
        if spread_mode && w >= 4 {
            let half_w = w / 2;
            let mut left = Image16::filled(half_w, h, bg)?;
            left.blit(&src, 0, 0, 0, 0, half_w, h);
            let mut right = Image16::filled(half_w, h, bg)?;
            right.blit(&src, half_w, 0, 0, 0, half_w, h);
            singles.push(left);
            singles.push(right);
        } else {
            singles.push(src);
        }
    }

    let orig_count = singles.len();
    let (pw, ph) = (singles[0].w, singles[0].h);
    let (total_n, plan) = saddle_stitch_imposition(orig_count);

    // Pad with blank guard pages (`página de guarda en blanco`) up to `total_n` (multiple of 4)
    while singles.len() < total_n {
        singles.push(Image16::filled(pw, ph, bg)?);
    }

    let bleed_px = ((bleed_mm.clamp(0.0, 15.0) / 25.4) * f64::from(ppi)).round() as usize;
    let mark_pad_px = if crop_marks {
        ((6.0 / 25.4) * f64::from(ppi)).round() as usize
    } else {
        0
    };
    let pad = bleed_px + mark_pad_px;
    let sheet_w = pw * 2 + 2 * pad;
    let sheet_h = ph + 2 * pad;

    let mut sheets = Vec::with_capacity(plan.len());
    for side in &plan {
        let mut canvas = Image16::filled(sheet_w, sheet_h, bg)?;
        let left_img = &singles[side.left_page - 1];
        let right_img = &singles[side.right_page - 1];
        canvas.blit(left_img, 0, 0, pad, pad, pw, ph);
        canvas.blit(right_img, 0, 0, pad + pw, pad, pw, ph);

        if crop_marks && pad >= 6 {
            let trim_rect = PxRect {
                x: pad,
                y: pad,
                w: pw * 2,
                h: ph,
            };
            draw_guillotine_crop_marks(&mut canvas, trim_rect, ppi, ink);
            // Also draw center fold tick marks at top and bottom of the spine (`x = pad + pw`)
            let fold_x = pad + pw;
            let tick_len = (pad * 2 / 3).max(4);
            canvas.fill_rect(fold_x, 0, 1, tick_len.min(pad.saturating_sub(2)), ink);
            canvas.fill_rect(
                fold_x,
                sheet_h.saturating_sub(tick_len.min(pad.saturating_sub(2))),
                1,
                tick_len.min(pad.saturating_sub(2)),
                ink,
            );
        }

        sheets.push(PageRaster {
            width: sheet_w as u32,
            height: sheet_h as u32,
            ppi,
            rgb: canvas.rgb,
        });
    }

    Ok((sheets, plan, orig_count))
}

/// Draws guillotine corner crop marks around `trim` on `canvas`.
pub fn draw_guillotine_crop_marks(canvas: &mut Image16, trim: PxRect, ppi: f32, ink: [u16; 3]) {
    let gap = ((1.5 / 25.4) * f64::from(ppi)).round() as usize;
    let len = ((4.5 / 25.4) * f64::from(ppi)).round() as usize;
    let weight = ((0.5 / 72.0) * f64::from(ppi)).round().max(1.0) as usize;
    let (x0, y0, x1, y1) = (trim.x as i64, trim.y as i64, (trim.x + trim.w) as i64, (trim.y + trim.h) as i64);
    let (l, g, t) = (len as i64, gap as i64, weight as i64);
    let segments = [
        (x0 - g - l, y0, l, t),
        (x1 + g, y0, l, t),
        (x0 - g - l, y1 - t, l, t),
        (x1 + g, y1 - t, l, t),
        (x0, y0 - g - l, t, l),
        (x0, y1 + g, t, l),
        (x1 - t, y0 - g - l, t, l),
        (x1 - t, y1 + g, t, l),
    ];
    for (sx, sy, sw, sh) in segments {
        let rx0 = sx.max(0) as usize;
        let ry0 = sy.max(0) as usize;
        let rx1 = (sx + sw).clamp(0, canvas.w as i64) as usize;
        let ry1 = (sy + sh).clamp(0, canvas.h as i64) as usize;
        if rx1 > rx0 && ry1 > ry0 {
            canvas.fill_rect(rx0, ry0, rx1 - rx0, ry1 - ry0, ink);
        }
    }
}

/// Encodes a `PageRaster` as a 300 DPI sRGB JPEG with an explicit JFIF 300 DPI density header.
pub fn encode_page_jpeg_srgb(page: &PageRaster, quality: u8) -> Result<Vec<u8>, String> {
    let rgba = super::output::page_to_srgb8(page)?;
    let srgb_icc = lightcraft_codecs::icc::write_named(lightcraft_codecs::NamedSpace::Srgb);
    let dpi = page.ppi.round().clamp(72.0, 1200.0) as u16;
    let img = lightcraft_codecs::EncodeImage::rgba8(&rgba);
    let meta = lightcraft_codecs::EncodeMeta {
        icc: Some(&srgb_icc),
        ppi: Some(dpi),
        ..Default::default()
    };
    let mut jpg = lightcraft_codecs::encode_jpeg(
        &img,
        quality.clamp(70, 100),
        lightcraft_codecs::ChromaSubsampling::S444,
        &meta,
    )
    .map_err(|e| e.to_string())?;
    inject_jfif_dpi(&mut jpg, dpi);
    Ok(jpg)
}

/// Sets the JFIF APP0 density unit to `1` (dots per inch) and `Xdensity = Ydensity = dpi`
/// so photo-lab RIPs (Noritsu, Fuji Frontier, Indigo) read 300 DPI directly from the JPEG header.
fn inject_jfif_dpi(jpeg: &mut [u8], dpi: u16) {
    if jpeg.len() >= 18 && jpeg[0..2] == [0xFF, 0xD8] && jpeg[2..4] == [0xFF, 0xE0] && &jpeg[6..11] == b"JFIF\0" {
        jpeg[13] = 1; // 1 = pixels per inch (DPI)
        let [hi, lo] = dpi.to_be_bytes();
        jpeg[14] = hi;
        jpeg[15] = lo;
        jpeg[16] = hi;
        jpeg[17] = lo;
    }
}

// ---- Portable Print-Shop Project Container (`.mmbook`) -----------------------------------------

/// Magic 8-byte header for MM Studio Portable Photobook Project files (`.mmbook`).
pub const MMBOOK_MAGIC: &[u8; 8] = b"MMBK\x01\x00\x00\x00";

/// Maximum `.mmbook` project archive size (1.5 GiB).
const MAX_MMBOOK_BYTES: usize = 1536 * 1024 * 1024;

/// Metadata for a single photo embedded inside a `.mmbook` portable project file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MmBookPhotoEntry {
    pub original_id: u64,
    pub file_name: String,
    pub width: u32,
    pub height: u32,
    /// Normalized face bounding boxes `[x, y, w, h]` preserved so pre-press warnings work at the print shop.
    #[serde(default)]
    pub faces: Vec<[f64; 4]>,
    pub offset: u64,
    pub length: u64,
}

/// JSON manifest stored at the start of a `.mmbook` portable photobook project file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MmBookManifest {
    pub version: u32,
    pub title: String,
    pub settings: PrintSettings,
    pub photos: Vec<MmBookPhotoEntry>,
}

/// Packs `title`, `settings`, and already-rendered high-resolution photo buffers into a single
/// self-contained `.mmbook` binary project file (`MMBK\x01\x00\x00\x00` + `manifest_len: u64 LE` + `manifest_json` + concatenated photo blobs).
pub fn pack_mmbook_bytes(
    title: &str,
    settings: &PrintSettings,
    mut photos: Vec<(MmBookPhotoEntry, Vec<u8>)>,
) -> Result<Vec<u8>, String> {
    let mut payload_len: usize = 0;
    for (entry, bytes) in &mut photos {
        entry.offset = payload_len as u64;
        entry.length = bytes.len() as u64;
        payload_len = payload_len.checked_add(bytes.len()).ok_or("el proyecto .mmbook es demasiado grande")?;
    }
    let entries: Vec<MmBookPhotoEntry> = photos.iter().map(|(e, _)| e.clone()).collect();
    let manifest = MmBookManifest {
        version: 1,
        title: if title.trim().is_empty() { "Photobook MM Studio".to_string() } else { title.trim().to_string() },
        settings: settings.clone(),
        photos: entries,
    };
    let manifest_bytes = serde_json::to_vec(&manifest).map_err(|e| e.to_string())?;
    let total_size = 16usize
        .checked_add(manifest_bytes.len())
        .and_then(|n| n.checked_add(payload_len))
        .ok_or("el proyecto .mmbook excede el tamaño permitido")?;
    if total_size > MAX_MMBOOK_BYTES {
        return Err("el proyecto .mmbook excede el límite de 1.5 GB".into());
    }
    let mut out = Vec::with_capacity(total_size);
    out.extend_from_slice(MMBOOK_MAGIC);
    out.extend_from_slice(&(manifest_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&manifest_bytes);
    for (_, bytes) in photos {
        out.extend_from_slice(&bytes);
    }
    Ok(out)
}

/// Unpacks a `.mmbook` binary project file into its `MmBookManifest` and embedded photo buffers.
pub fn unpack_mmbook_bytes(data: &[u8]) -> Result<(MmBookManifest, Vec<(MmBookPhotoEntry, Vec<u8>)>), String> {
    if data.len() < 16 || &data[0..8] != MMBOOK_MAGIC {
        return Err("el archivo no es un proyecto válido de MM Studio (.mmbook)".into());
    }
    let mut len_bytes = [0u8; 8];
    len_bytes.copy_from_slice(&data[8..16]);
    let manifest_len = u64::from_le_bytes(len_bytes) as usize;
    let payload_start = 16usize.checked_add(manifest_len).ok_or("cabecera .mmbook corrupta")?;
    if payload_start > data.len() {
        return Err("el proyecto .mmbook está truncado o incompleto".into());
    }
    let mut manifest: MmBookManifest =
        serde_json::from_slice(&data[16..payload_start]).map_err(|e| format!("manifiesto .mmbook inválido: {e}"))?;
    manifest.settings.validate()?;
    let payload = &data[payload_start..];
    let mut photos = Vec::with_capacity(manifest.photos.len());
    for entry in &manifest.photos {
        let start = entry.offset as usize;
        let end = start.checked_add(entry.length as usize).ok_or("índice de foto inválido en .mmbook")?;
        let slice = payload.get(start..end).ok_or("datos de foto incompletos en .mmbook")?;
        photos.push((entry.clone(), slice.to_vec()));
    }
    Ok((manifest, photos))
}

/// A background-ready `.mmbook` project packaging job.
pub struct PreparedMmBook {
    pub title: String,
    pub settings: PrintSettings,
    pub items: Vec<(MmBookPhotoEntry, crate::media::RenderJob)>,
}

impl PreparedMmBook {
    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Renders every referenced photo with its current Develop adjustments into a 300 DPI sRGB 4:4:4 JPEG
    /// and packs the `.mmbook` binary container. Can run on a background worker thread.
    pub fn run(self, progress: &mut dyn FnMut(usize, &str) -> bool) -> Result<Vec<u8>, String> {
        let srgb_icc = lightcraft_codecs::icc::write_named(lightcraft_codecs::NamedSpace::Srgb);
        let dpi = self.settings.job.effective_ppi().round().clamp(72.0, 1200.0) as u16;
        let mut packed_photos = Vec::with_capacity(self.items.len());
        for (idx, (mut entry, job)) in self.items.into_iter().enumerate() {
            if !progress(idx, &entry.file_name) {
                return Err("Empaquetado .mmbook cancelado".into());
            }
            let rendered = job.run().rendered.map_err(|e| format!("{}: {e}", entry.file_name))?;
            entry.width = rendered.image.width as u32;
            entry.height = rendered.image.height as u32;
            let img = lightcraft_codecs::EncodeImage::rgba8(&rendered.image);
            let meta = lightcraft_codecs::EncodeMeta {
                icc: Some(&srgb_icc),
                ppi: Some(dpi),
                ..Default::default()
            };
            let mut jpg = lightcraft_codecs::encode_jpeg(
                &img,
                97,
                lightcraft_codecs::ChromaSubsampling::S444,
                &meta,
            )
            .map_err(|e| format!("{}: {e}", entry.file_name))?;
            inject_jfif_dpi(&mut jpg, dpi);
            packed_photos.push((entry, jpg));
        }
        if !progress(packed_photos.len(), "") {
            return Err("Empaquetado .mmbook cancelado".into());
        }
        pack_mmbook_bytes(&self.title, &self.settings, packed_photos)
    }
}

/// Collects all unique photos used in the Photobook (`settings.layout.cells` + `fallback_ids`) and prepares
/// background `RenderJob`s at 300 DPI print resolution so the `.mmbook` archive can be built without blocking the UI.
pub fn prepare_mmbook(
    session: &mut crate::Session,
    title: &str,
    settings: &PrintSettings,
    fallback_ids: &[lightcraft_catalog::PhotoId],
) -> Result<PreparedMmBook, String> {
    let mut settings = settings.clone();
    settings.validate()?;
    let ppi = f64::from(settings.job.effective_ppi());
    let (pw_in, ph_in) = settings.layout.page_size()?;
    let max_dim = ((pw_in.max(ph_in) * ppi).round() as usize).clamp(1600, 4800);

    // Collect unique photo IDs referenced in cells, or fallback selection
    let mut unique_ids = Vec::new();
    for c in &settings.layout.cells {
        if let Some(pid) = c.photo {
            let id = lightcraft_catalog::PhotoId(pid);
            if session.catalog.photo(id).is_some() && !unique_ids.contains(&id) {
                unique_ids.push(id);
            }
        }
    }
    for &id in fallback_ids {
        if session.catalog.photo(id).is_some() && !unique_ids.contains(&id) {
            unique_ids.push(id);
        }
    }
    if unique_ids.is_empty() {
        for p in session.catalog.photos().take(24) {
            unique_ids.push(p.id);
        }
    }
    if unique_ids.is_empty() {
        return Err("No hay fotos en el fotolibro para empaquetar en el proyecto .mmbook".into());
    }

    let mut items = Vec::with_capacity(unique_ids.len());
    for id in unique_ids {
        let Some(photo) = session.catalog.photo(id).cloned() else { continue };
        let faces: Vec<[f64; 4]> = photo
            .meta
            .regions
            .iter()
            .filter(|r| r.kind == lightcraft_meta::RegionKind::Face)
            .map(|r| {
                [
                    f64::from(r.rect.x0),
                    f64::from(r.rect.y0),
                    f64::from((r.rect.x1 - r.rect.x0).max(0.01)),
                    f64::from((r.rect.y1 - r.rect.y0).max(0.01)),
                ]
            })
            .collect();
        let stem = std::path::Path::new(&photo.file_name)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("Foto");
        let file_name = format!("{stem}_id{}.jpg", id.0);
        let job = session.export_job(
            id,
            max_dim,
            max_dim,
            lightcraft_pipeline::OutputSpace::Srgb,
            lightcraft_pipeline::OutputDepth::U8,
        )?;
        let entry = MmBookPhotoEntry {
            original_id: id.0,
            file_name,
            width: photo.width,
            height: photo.height,
            faces,
            offset: 0,
            length: 0,
        };
        items.push((entry, job));
    }

    Ok(PreparedMmBook {
        title: title.to_string(),
        settings,
        items,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saddle_stitch_imposition_matches_formula_n_plus_1() {
        // 6 pages -> padded to 8 pages (2 sheets, 4 sides)
        let (total_n, sides) = saddle_stitch_imposition(6);
        assert_eq!(total_n, 8);
        assert_eq!(sides.len(), 4);
        // Sheet 1 Front (Tiro): [Page 8 | Page 1]
        assert_eq!((sides[0].left_page, sides[0].right_page, sides[0].is_front), (8, 1, true));
        assert_eq!(sides[0].left_page + sides[0].right_page, total_n + 1);
        // Sheet 1 Back (Retiro): [Page 2 | Page 7]
        assert_eq!((sides[1].left_page, sides[1].right_page, sides[1].is_front), (2, 7, false));
        assert_eq!(sides[1].left_page + sides[1].right_page, total_n + 1);
        // Sheet 2 Front (Tiro): [Page 6 | Page 3]
        assert_eq!((sides[2].left_page, sides[2].right_page, sides[2].is_front), (6, 3, true));
        assert_eq!(sides[2].left_page + sides[2].right_page, total_n + 1);
        // Sheet 2 Back (Retiro): [Page 4 | Page 5]
        assert_eq!((sides[3].left_page, sides[3].right_page, sides[3].is_front), (4, 5, false));
        assert_eq!(sides[3].left_page + sides[3].right_page, total_n + 1);
    }

    #[test]
    fn smart_layout_spread_produces_clean_non_overlapping_cells_across_variations() {
        let aspects = [1.5, 0.67, 1.5, 1.0, 0.75, 1.33];
        for count in 1..=6 {
            for var in 0..8 {
                let rects = smart_layout_spread(
                    20.0,
                    10.0,
                    true,
                    12.0 / 25.4,
                    12.0 / 25.4,
                    4.0 / 25.4,
                    BindingMode::LayflatSpreads,
                    &aspects[..count],
                    var,
                );
                assert_eq!(rects.len(), count);
                for (i, a) in rects.iter().enumerate() {
                    assert!(a.w > 0.1 && a.h > 0.1 && a.x >= -1e-6 && a.y >= -1e-6 && a.right() <= 20.0 + 1e-6 && a.bottom() <= 10.0 + 1e-6);
                    for b in rects.iter().skip(i + 1) {
                        assert!(!a.overlaps(b), "count={count} var={var}: {a:?} overlaps {b:?}");
                    }
                }
            }
        }
    }
}
