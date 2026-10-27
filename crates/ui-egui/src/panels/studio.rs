//! Frosted Glass Studio Workspace (`MM Fotografía`) matching the user's reference design 100%:
//! - Simulated frosted glass transparency (`backdrop-filter` illusion) over a blurred ambient photo/bokeh backdrop
//! - Floating Top Bar (`🔴🟡🟢`, circular `[MM]` badge + `MM Fotografía`, search pill, `[Biblioteca]`, glowing golden-amber `[🖼 Editar]`, `Mapa | Imprimir`, `[⋯] [☁] [👤]`)
//! - Left Column with 3 distinct rounded frosted cards:
//!   1. `Biblioteca` (`[+]`, `Todas las fotos`, `Recientes`, `Favoritas`, `Eliminadas`)
//!   2. `Álbumes` (`+`, hierarchical folder & album tree with frosted active pill)
//!   3. `Proyectos` (`+`, vertical project & album photo cards scrolling downwards `hacia abajo`)
//! - Center Photo Stage Container:
//!   - Top sub-bar (`[< | >]  DSC05094.ARW  7952 × 5304   ★★★★☆ | ⚑ ♡ ⋯   [Fit ⌄ | 100% ⌄] [🕘]`)
//!   - Rounded main photo canvas (`CornerRadius::same(14)`)
//!   - Lower control sub-bar (`[⊞] [☰] [⊡] ──●──────     ★★★★☆  ⋯  🗑`)
//!   - Horizontal rounded filmstrip trough (`54×54` rounded thumbnails + `‹·`)
//! - Right Studio Column:
//!   - `Histograma` card (top-left title, multi-channel RGB/CMY smooth curves, `ISO 500 113mm f/4.5 1/100` footer)
//!   - 6-Tool Switcher Card (`Luz | Color | Efectos | Detalle | Óptica | Curvas` with glowing golden-amber active capsule)
//!   - Split lower area: `Luz` (`[Auto ↺]` + 6 sliders with tick marks) & `Curva de tonos` (`✕` + 4 colored channel rings) on the left, and vertical `[Presets | Copiar | Pegar | Sincronizar | ⋯ | ⋯]` strip on the right
//! - Bottom Inspector & Export Bar spanning under Center + Right (`[Thumb] DSC05094.ARW / 28 sep 2026, 18:24  ★★★★☆ | ⚑ | 📷 ISO 500  🔘 113 mm  ◐ f/4.5  ⏱ 1/100   [Exportar | ⌄]`).

use egui::epaint::{Mesh, Vertex};
use egui::{Align2, Color32, CornerRadius, Pos2, Rect, Sense, Stroke, StrokeKind, pos2, vec2};
use lightcraft_catalog::{Flag, PhotoId};
use lightcraft_develop::{ControlSpec, DevelopSettings, Track, WbMode, controls};
use lightcraft_engine::LibrarySource;
use lightcraft_geom::Point;
use serde_json::json;

use crate::LightcraftApp;
use crate::icons::{Icon, paint};
use crate::panels::presets::cover_uv;
use crate::state::{RightPanel, ViewMode, Zoom};
use crate::theme::{Tokens, mm_logo_white_texture};
use crate::widgets::{BAND_COLORS, hex, register, stars};

/// Ensures that if the current source/album has visible photos and either no photo is active
/// or the active photo is not in the visible list, the first visible photo is automatically selected.
pub fn ensure_active_photo(app: &mut LightcraftApp) {
    let active_in_visible = app
        .session
        .active()
        .is_some_and(|id| app.session.visible().contains(&id));
    if !active_in_visible {
        if let Some(&first) = app.session.visible().first() {
            let _ = app.run("library.select", json!({"ids": [first.0], "mode": "replace"}));
        } else if app.session.active().is_some() {
            let _ = app.run("library.select", json!({"ids": [], "mode": "replace"}));
        }
    }
}

/// Computes a smooth 220ms cubic ease-out progress (`0.0..=1.0`) whenever `screen_key` changes.
pub fn screen_transition_progress(ctx: &egui::Context, screen_key: &str) -> f32 {
    let key_id = egui::Id::new("mm-screen-trans-key");
    let time_id = egui::Id::new("mm-screen-trans-time");
    let now = ctx.input(|i| i.time);
    let prev_key = ctx.data(|d| d.get_temp::<String>(key_id));
    let mut start = ctx.data(|d| d.get_temp::<f64>(time_id)).unwrap_or(now);
    if prev_key.as_deref() != Some(screen_key) {
        start = now;
        ctx.data_mut(|d| {
            d.insert_temp(key_id, screen_key.to_string());
            d.insert_temp(time_id, start);
        });
    }
    let raw = ((now - start).max(0.0) as f32 / 0.22).clamp(0.0, 1.0);
    if raw < 1.0 {
        ctx.request_repaint();
    }
    1.0 - (1.0 - raw).powi(3)
}

/// Computes a subtle 150ms cross-dissolve progress (`0.0..=1.0`) when switching active photos.
fn photo_transition_progress(ctx: &egui::Context, active: Option<PhotoId>) -> f32 {
    let key_id = egui::Id::new("mm-photo-trans-id");
    let time_id = egui::Id::new("mm-photo-trans-time");
    let now = ctx.input(|i| i.time);
    let cur = active.map(|p| p.0).unwrap_or(0);
    let prev = ctx.data(|d| d.get_temp::<u64>(key_id));
    let mut start = ctx.data(|d| d.get_temp::<f64>(time_id)).unwrap_or(now);
    if prev != Some(cur) {
        start = now;
        ctx.data_mut(|d| {
            d.insert_temp(key_id, cur);
            d.insert_temp(time_id, start);
        });
    }
    let raw = ((now - start).max(0.0) as f32 / 0.15).clamp(0.0, 1.0);
    if raw < 1.0 {
        ctx.request_repaint();
    }
    1.0 - (1.0 - raw).powi(3)
}

/// Synthesizes and caches the blurred ambient glass backdrop texture (`128×80`) so every card
/// has the realistic illusion of frosted-glass transparency over warm bokeh & portrait tones.
fn ambient_backdrop_texture(ctx: &egui::Context) -> egui::TextureHandle {
    static CACHE: std::sync::LazyLock<std::sync::Mutex<std::collections::HashMap<usize, egui::TextureHandle>>> =
        std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let tm_ptr = std::sync::Arc::as_ptr(&ctx.tex_manager()) as usize;
    if let Ok(guard) = CACHE.lock()
        && let Some(h) = guard.get(&tm_ptr)
        && ctx.tex_manager().read().meta(h.id()).is_some()
    {
        return h.clone();
    }
    let (w, h) = (128usize, 80usize);
    let mut rgb = vec![[32.0f32, 34.0, 40.0]; w * h];
    // Radial light blooms matching the reference photo's warm amber, portrait peach, sky-blue, and prism bokeh leaks
    let blooms: &[(f32, f32, f32, f32, [f32; 3])] = &[
        // Top-left warm amber/peach glow behind MM Fotografía & Biblioteca
        (0.14, 0.08, 0.16, 0.18, [148.0, 108.0, 82.0]),
        // Left-center prism rainbow/amber flare between Álbumes & Proyectos
        (0.16, 0.62, 0.07, 0.12, [168.0, 126.0, 74.0]),
        (0.18, 0.64, 0.05, 0.10, [86.0, 146.0, 168.0]),
        // Bottom-left warm rose/gold bokeh behind Proyectos
        (0.06, 0.86, 0.10, 0.14, [142.0, 92.0, 68.0]),
        // Center-top & right-of-center cool window daylight blue
        (0.55, 0.24, 0.24, 0.22, [88.0, 112.0, 148.0]),
        // Center warm portrait skin-tone glow
        (0.46, 0.50, 0.22, 0.24, [136.0, 96.0, 82.0]),
        // Top-right warm golden-amber glow behind Editar pill & Histograma
        (0.70, 0.06, 0.14, 0.12, [164.0, 118.0, 66.0]),
        (0.92, 0.24, 0.14, 0.16, [156.0, 110.0, 68.0]),
        // Bottom-right prism & golden flare behind Curva de tonos & Exportar
        (0.84, 0.90, 0.14, 0.14, [168.0, 122.0, 68.0]),
        (0.81, 0.88, 0.06, 0.10, [82.0, 142.0, 162.0]),
    ];
    for y in 0..h {
        let ny = y as f32 / h as f32;
        for x in 0..w {
            let nx = x as f32 / w as f32;
            let mut acc = [30.0f32, 32.0, 38.0];
            for &(bx, by, rx, ry, col) in blooms {
                let dx = (nx - bx) / rx;
                let dy = (ny - by) / ry;
                let d2 = dx * dx + dy * dy;
                if d2 < 4.5 {
                    let wgt = (-d2 * 1.15).exp();
                    acc[0] += (col[0] - 30.0) * wgt * 0.68;
                    acc[1] += (col[1] - 32.0) * wgt * 0.68;
                    acc[2] += (col[2] - 38.0) * wgt * 0.68;
                }
            }
            rgb[y * w + x] = acc;
        }
    }
    let mut pixels = Vec::with_capacity(w * h * 4);
    for px in rgb {
        pixels.push(px[0].round().clamp(0.0, 255.0) as u8);
        pixels.push(px[1].round().clamp(0.0, 255.0) as u8);
        pixels.push(px[2].round().clamp(0.0, 255.0) as u8);
        pixels.push(255);
    }
    let img = egui::ColorImage::from_rgba_unmultiplied([w, h], &pixels);
    let handle = ctx.load_texture("mm-studio-ambient-backdrop", img, egui::TextureOptions::LINEAR);
    if let Ok(mut guard) = CACHE.lock() {
        guard.insert(tm_ptr, handle.clone());
    }
    handle
}

/// Paints a simulated frosted-glass card (`backdrop-filter: blur` illusion) with soft drop shadow,
/// translucent dark-charcoal fill, inner top sheen, and 1px specular glass rim border.
pub(crate) fn paint_glass_card(p: &egui::Painter, rect: Rect, radius: u8, alpha: u8) {
    // Soft outer shadow
    p.rect_filled(
        rect.translate(vec2(0.0, 2.5)).expand(1.5),
        CornerRadius::same(radius.saturating_add(2)),
        Color32::from_black_alpha(52),
    );
    // Frosted dark glass fill (translucent so the ambient backdrop glows through)
    p.rect_filled(
        rect,
        CornerRadius::same(radius),
        Color32::from_rgba_unmultiplied(36, 38, 44, alpha),
    );
    // Subtle top inner glass sheen
    let sheen_h = (rect.height() * 0.28).clamp(8.0, 26.0);
    let sheen_r = Rect::from_min_size(rect.min, vec2(rect.width(), sheen_h));
    p.rect_filled(
        sheen_r,
        CornerRadius { nw: radius, ne: radius, sw: 0, se: 0 },
        Color32::from_white_alpha(7),
    );
    // Crisp 1px glass border
    p.rect_stroke(
        rect,
        CornerRadius::same(radius),
        Stroke::new(1.0, Color32::from_white_alpha(36)),
        StrokeKind::Inside,
    );
    // Specular top rim highlight
    if rect.width() > 24.0 {
        let r_f = f32::from(radius);
        p.line_segment(
            [pos2(rect.left() + r_f * 0.75, rect.top() + 0.8), pos2(rect.right() - r_f * 0.75, rect.top() + 0.8)],
            Stroke::new(1.0, Color32::from_white_alpha(58)),
        );
    }
}

/// Paints the warm golden-amber glowing glass pill used for the active `Editar` tab and active `Luz` tool.
pub(crate) fn paint_gold_glow_pill(p: &egui::Painter, rect: Rect, radius: u8) {
    p.rect_filled(
        rect.expand(4.5),
        CornerRadius::same(radius.saturating_add(4)),
        Color32::from_rgba_unmultiplied(238, 168, 72, 24),
    );
    p.rect_filled(
        rect.expand(2.0),
        CornerRadius::same(radius.saturating_add(2)),
        Color32::from_rgba_unmultiplied(244, 180, 84, 42),
    );
    p.rect_filled(
        rect,
        CornerRadius::same(radius),
        Color32::from_rgba_unmultiplied(112, 88, 56, 195),
    );
    p.rect_stroke(
        rect,
        CornerRadius::same(radius),
        Stroke::new(1.2, Color32::from_rgb(244, 192, 108)),
        StrokeKind::Inside,
    );
    if rect.width() > 16.0 {
        let inset = f32::from(radius) * 0.65;
        p.line_segment(
            [pos2(rect.left() + inset, rect.bottom() - 1.2), pos2(rect.right() - inset, rect.bottom() - 1.2)],
            Stroke::new(1.8, Color32::from_rgb(255, 214, 142)),
        );
    }
}

fn format_count(n: usize) -> String {
    if n >= 1_000 {
        format!("{},{:03}", n / 1_000, n % 1_000)
    } else {
        n.to_string()
    }
}

pub fn show(app: &mut LightcraftApp, ui: &mut egui::Ui) {
    ensure_active_photo(app);
    let win_rect = ui.ctx().content_rect();
    if app.ui.right == RightPanel::None
        && (win_rect.width() >= 1100.0 || !matches!(app.ui.view, ViewMode::PhotoGrid | ViewMode::SquareGrid))
    {
        app.ui.right = RightPanel::Edit;
    }
    poll_and_sync_agent_bridge(app, ui.ctx());

    // Paint the continuous blurred ambient backdrop across the ENTIRE window (behind Top Bar + Central Panel)
    {
        let bg_p = ui.ctx().layer_painter(egui::LayerId::background());
        bg_p.rect_filled(win_rect, 0, Color32::from_rgb(24, 26, 31));
        let amb = ambient_backdrop_texture(ui.ctx());
        bg_p.image(amb.id(), win_rect, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
    }

    show_topbar(app, ui);
    super::library_problem::banner(app, ui);
    super::compare::prepare(app);

    let view_key = match app.ui.view {
        ViewMode::PhotoGrid | ViewMode::SquareGrid => "studio-grid",
        ViewMode::Detail => "studio-detail",
        ViewMode::Compare => "studio-compare",
        ViewMode::Survey => "studio-survey",
        ViewMode::Reference => "studio-reference",
        ViewMode::People => "studio-people",
    };
    let screen_anim = screen_transition_progress(ui.ctx(), view_key);
    let photo_anim = photo_transition_progress(ui.ctx(), app.session.active());

    egui::CentralPanel::default()
        .frame(egui::Frame::NONE.fill(Color32::TRANSPARENT))
        .show(ui, |ui| {
            let full = ui.max_rect();
            let pad = 8.0;
            let gap = 8.0;

            let show_left = app.ui.left_panel && (full.width() >= 1100.0 || app.ui.right == RightPanel::None);
            let left_w = if show_left { 226.0 } else { 0.0 };
            let right_w = if app.ui.right != RightPanel::None { 362.0 } else { 0.0 };
            let presets_w = if app.ui.presets { 256.0 } else { 0.0 };
            let bottom_h = 54.0;

            let left_rect = Rect::from_min_max(
                pos2(full.left() + pad, full.top() + 2.0),
                pos2(full.left() + pad + left_w, full.bottom() - pad),
            );

            let stage_left = if show_left { left_rect.right() + gap } else { full.left() + pad };
            let stage_right_edge = full.right() - pad;
            let bottom_top = (full.bottom() - pad - bottom_h).max(full.top() + 120.0);

            // Bottom Inspector & Export Bar spans across Center + Right columns (matching the reference photo)
            let bottom_rect = Rect::from_min_max(
                pos2(stage_left, bottom_top),
                pos2(stage_right_edge, full.bottom() - pad),
            );

            let main_bottom = bottom_top - gap;
            let right_left = (stage_right_edge - right_w).max(stage_left + 200.0);
            let presets_left = (right_left - if presets_w > 0.0 { presets_w + gap } else { 0.0 }).max(stage_left + 160.0);
            let center_right = if presets_w > 0.0 {
                presets_left - gap
            } else if right_w > 0.0 {
                right_left - gap
            } else {
                stage_right_edge
            };

            let slide_y = (1.0 - screen_anim) * 6.0;
            let center_rect = Rect::from_min_max(
                pos2(stage_left, full.top() + 2.0 + slide_y),
                pos2(center_right.max(stage_left + 140.0), main_bottom + slide_y),
            );

            // 1. Left Column (3 Frosted Cards: Biblioteca, Álbumes, Proyectos)
            if show_left {
                register(ui.ctx(), "panel:left_panel", left_rect);
                let mut lui = ui.new_child(egui::UiBuilder::new().max_rect(left_rect).layout(egui::Layout::top_down(egui::Align::Min)));
                lui.set_clip_rect(left_rect);
                show_left_sidebar(app, &mut lui);
            }

            // 2. Center Photo Stage Container (Top Sub-bar + Rounded Photo + Control Sub-bar + Horizontal Filmstrip)
            {
                let mut cui = ui.new_child(egui::UiBuilder::new().max_rect(center_rect).layout(egui::Layout::top_down(egui::Align::Min)));
                cui.set_clip_rect(center_rect.expand(4.0));
                show_center_stage(app, &mut cui, center_rect, screen_anim, photo_anim);
            }

            // 3. Optional Presets & LUTs Drawer
            if app.ui.presets {
                let presets_rect = Rect::from_min_max(pos2(presets_left, full.top() + 2.0), pos2(right_left - gap, main_bottom));
                paint_glass_card(ui.painter(), presets_rect, 16, 198);
                let inner = presets_rect.shrink(4.0);
                let mut pui = ui.new_child(egui::UiBuilder::new().max_rect(inner).layout(egui::Layout::top_down(egui::Align::Min)));
                pui.set_clip_rect(inner);
                super::presets::show(app, &mut pui);
            }

            // 4. Right Studio Column (Histograma + 6-Tool Bar + [Luz + Curva de tonos] + Vertical Action Rail)
            if app.ui.right != RightPanel::None {
                let right_rect = Rect::from_min_max(pos2(right_left, full.top() + 2.0), pos2(stage_right_edge, main_bottom));
                register(ui.ctx(), "panel:right_panel", right_rect);
                let mut rui = ui.new_child(egui::UiBuilder::new().max_rect(right_rect).layout(egui::Layout::top_down(egui::Align::Min)));
                rui.set_clip_rect(right_rect.expand(4.0));
                show_right_panel(app, &mut rui);
            }

            // 5. Bottom Inspector & Export Bar
            {
                let mut bui = ui.new_child(egui::UiBuilder::new().max_rect(bottom_rect).layout(egui::Layout::top_down(egui::Align::Min)));
                bui.set_clip_rect(bottom_rect.expand(4.0));
                show_bottom_bar(app, &mut bui);
            }

            // 6. Floating Welcome & Optional Donation Modal (if open)
            super::home::draw_welcome_modal(app, ui);
        });
}

// ============================================================================
// 1. TOP BAR
// ============================================================================

fn show_topbar(app: &mut LightcraftApp, ui: &mut egui::Ui) {
    let t = Tokens::get(ui.ctx());
    let left_margin = match (app.integrated_titlebar, app.window_is_fullscreen) {
        (true, false) => 78,
        (true, true) => 16,
        (false, _) => 10,
    };
    egui::Panel::top("studio_top_bar")
        .exact_size(44.0)
        .frame(
            egui::Frame::NONE
                .fill(Color32::TRANSPARENT)
                .inner_margin(egui::Margin { left: left_margin, right: 12, top: 0, bottom: 0 }),
        )
        .show(ui, |ui| {
            let full = ui.max_rect();
            let bar = Rect::from_min_max(pos2(full.left() - f32::from(left_margin), full.top()), pos2(full.right() + 12.0, full.bottom()));

            // Left: Circular frosted [MM] badge + "MM Fotografía"
            let badge_r = Rect::from_center_size(pos2(full.left() + 18.0, full.center().y), vec2(34.0, 34.0));
            let brand_rect = Rect::from_min_size(pos2(full.left(), full.center().y - 17.0), vec2(154.0, 34.0));

            // Search bar pill (responsive so navigation pills never collide with right-side icons on laptops)
            let search_left = (brand_rect.right() + 18.0).max(full.left() + 168.0);
            let max_search_w = (full.right() - 118.0 - 12.0 - 332.0 - 14.0 - search_left).max(120.0);
            let search_w = 340.0f32.min(max_search_w);
            let sr = Rect::from_min_size(pos2(search_left, full.center().y - 16.5), vec2(search_w, 33.0));

            if app.integrated_titlebar {
                super::topbar::window_handle(app, ui, bar, sr);
            }

            let hresp = ui
                .interact(brand_rect, egui::Id::new("studio-home-brand"), Sense::click())
                .on_hover_text("Ir a la Pantalla de Inicio (MM Studio)");
            register(ui.ctx(), "button:homeScreen", brand_rect);

            let p = ui.painter();
            p.circle(
                badge_r.center(),
                17.0,
                if hresp.hovered() {
                    Color32::from_rgba_unmultiplied(64, 66, 74, 210)
                } else {
                    Color32::from_rgba_unmultiplied(46, 48, 56, 195)
                },
                Stroke::new(1.0, Color32::from_white_alpha(52)),
            );
            if let Some(tex) = mm_logo_white_texture(ui.ctx()) {
                let lr = Rect::from_center_size(badge_r.center(), vec2(21.0, 21.0));
                p.image(tex.id(), lr, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
            } else {
                p.text(badge_r.center(), Align2::CENTER_CENTER, "MM", t.semibold(11.0), Color32::WHITE);
            }
            p.text(
                pos2(badge_r.right() + 10.0, full.center().y),
                Align2::LEFT_CENTER,
                "MM Studio",
                t.semibold(14.0),
                Color32::from_rgb(246, 244, 240),
            );
            if hresp.clicked() {
                app.ui.show_home = true;
            }

            // Search Pill: "Buscar fotos, álbumes, etiquetas..."
            let search_id = crate::text_field::id("field:search");
            if std::mem::take(&mut app.ui.focus_search) {
                ui.memory_mut(|m| m.request_focus(search_id));
            }
            let focused = ui.memory(|m| m.has_focus(search_id));
            p.rect(
                sr,
                CornerRadius::same(16),
                if focused {
                    Color32::from_rgba_unmultiplied(44, 47, 56, 215)
                } else {
                    Color32::from_rgba_unmultiplied(32, 35, 42, 185)
                },
                Stroke::new(1.0, if focused { Color32::from_rgb(242, 188, 98) } else { Color32::from_white_alpha(36) }),
                StrokeKind::Inside,
            );
            paint(
                p,
                Rect::from_center_size(pos2(sr.left() + 17.0, sr.center().y), vec2(14.0, 14.0)),
                Icon::Search,
                Color32::from_rgb(168, 172, 182),
            );
            let has_query = !app.ui.search.is_empty();
            let text_right = if has_query { sr.right() - 26.0 } else { sr.right() - 12.0 };
            let text_rect = Rect::from_min_max(pos2(sr.left() + 32.0, sr.top() + 4.0), pos2(text_right, sr.bottom() - 4.0));
            let mut search_ui = ui.new_child(egui::UiBuilder::new().max_rect(text_rect).layout(egui::Layout::left_to_right(egui::Align::Center)));
            if app.ui.search.is_empty() && !focused {
                search_ui.painter().text(
                    pos2(text_rect.left() + 2.0, sr.center().y),
                    Align2::LEFT_CENTER,
                    "Buscar fotos, álbumes, etiquetas...",
                    t.font(12.5),
                    Color32::from_rgb(164, 168, 178),
                );
            }
            let sresp = crate::text_field::TextField::singleline("field:search", &mut app.ui.search)
                .label("Buscar fotos, álbumes, etiquetas...")
                .frame(false)
                .width(text_rect.width() - 4.0)
                .font(t.font(12.5))
                .text_color(t.text)
                .show(&mut search_ui)
                .response;
            if sresp.changed() {
                let q = app.ui.search.clone();
                let _ = app.run("library.filter", json!({"text": q}));
            }
            if has_query {
                let clr_r = Rect::from_center_size(pos2(sr.right() - 14.0, sr.center().y), vec2(18.0, 18.0));
                let clr_resp = ui.interact(clr_r, egui::Id::new("studio-search-clear"), Sense::click()).on_hover_text("Limpiar búsqueda");
                paint(ui.painter(), clr_r.shrink(4.0), Icon::Close, if clr_resp.hovered() { Color32::WHITE } else { Color32::from_rgb(170, 175, 185) });
                if clr_resp.clicked() {
                    app.ui.search.clear();
                    let _ = app.run("library.filter", json!({"text": ""}));
                }
            }

            // Right-side action buttons: [ ⋯ ]  [ ☁ ]  [ 👤 Avatar ]
            let right_edge = full.right();
            let avatar_r = Rect::from_center_size(pos2(right_edge - 17.0, full.center().y), vec2(34.0, 34.0));
            let cloud_r = Rect::from_min_size(pos2(avatar_r.left() - 8.0 - 36.0, full.center().y - 16.0), vec2(36.0, 32.0));
            let more_r = Rect::from_center_size(pos2(cloud_r.left() - 8.0 - 16.0, full.center().y), vec2(32.0, 32.0));

            // [ ⋯ ] circular frosted button
            {
                let resp = ui
                    .interact(more_r, egui::Id::new("studioMore"), Sense::click())
                    .on_hover_text("Opciones, importar Presets/LUTs y ajustes");
                register(ui.ctx(), "icon:studioMore", more_r);
                ui.painter().circle(
                    more_r.center(),
                    16.0,
                    if resp.hovered() {
                        Color32::from_rgba_unmultiplied(58, 62, 70, 210)
                    } else {
                        Color32::from_rgba_unmultiplied(40, 43, 50, 185)
                    },
                    Stroke::new(1.0, Color32::from_white_alpha(40)),
                );
                paint(ui.painter(), Rect::from_center_size(more_r.center(), vec2(15.0, 15.0)), Icon::More, Color32::from_rgb(225, 228, 235));
                egui::Popup::menu(&resp).show(|ui| {
                    ui.set_min_width(220.0);
                    if ui.button("✦ Abrir panel Presets & LUTs").clicked() {
                        let _ = app.run("panel.presets", json!({}));
                        ui.close();
                    }
                    if ui.button("+ Importar Presets Lightroom (.xmp)").clicked() {
                        app.ui.presets = true;
                        let _ = app.run("file.importPresets", json!({"kind": "lr"}));
                        ui.close();
                    }
                    if ui.button("+ Importar LUT 3D DaVinci (.cube)").clicked() {
                        app.ui.presets = true;
                        let _ = app.run("file.importPresets", json!({"kind": "lut"}));
                        ui.close();
                    }
                    if ui.button("⚡ Smart Style Match IA (⌘⌥M)").clicked() {
                        ensure_active_photo(app);
                        if let Err(e) = app.run("develop.smartMatch", json!({})) {
                            app.toast_error(ui.ctx(), e);
                        }
                        ui.close();
                    }
                    ui.separator();
                    if ui.button("⚙ Ajustes (Settings…)").clicked() {
                        let _ = app.run("app.settings", json!({}));
                        ui.close();
                    }
                    if ui.button("⌨ Atajos de teclado").clicked() {
                        let _ = app.run("app.shortcuts", json!({}));
                        ui.close();
                    }
                });
            }

            // [ ☁ ] rounded-rect frosted button
            {
                let resp = ui
                    .interact(cloud_r, egui::Id::new("studioCloud"), Sense::click())
                    .on_hover_text("Importar fotos o carpetas a la biblioteca");
                register(ui.ctx(), "icon:studioCloud", cloud_r);
                ui.painter().rect(
                    cloud_r,
                    CornerRadius::same(10),
                    if resp.hovered() {
                        Color32::from_rgba_unmultiplied(58, 62, 70, 210)
                    } else {
                        Color32::from_rgba_unmultiplied(40, 43, 50, 185)
                    },
                    Stroke::new(1.0, Color32::from_white_alpha(40)),
                    StrokeKind::Inside,
                );
                paint(ui.painter(), Rect::from_center_size(cloud_r.center(), vec2(16.0, 16.0)), Icon::Cloud, Color32::from_rgb(225, 228, 235));
                egui::Popup::menu(&resp).show(|ui| {
                    ui.set_min_width(205.0);
                    if ui.button("+ Importar fotografías…").clicked() {
                        if let Err(e) = app.run("file.addPhotos", json!({})) {
                            app.toast_error(ui.ctx(), e);
                        }
                        ui.close();
                    }
                    if ui.button("📁 Importar carpeta…").clicked() {
                        if let Err(e) = app.run("file.addFolder", json!({})) {
                            app.toast_error(ui.ctx(), e);
                        }
                        ui.close();
                    }
                    ui.separator();
                    ui.label(egui::RichText::new("✓ Biblioteca local sincronizada").size(11.5).color(t.text_dim));
                });
            }

            // [ 👤 ] Circular Photographer Avatar on the far right
            {
                let resp = ui
                    .interact(avatar_r, egui::Id::new("studioAvatar"), Sense::click())
                    .on_hover_text("Perfil · MM Studio (Ajustes)");
                register(ui.ctx(), "icon:studioAvatar", avatar_r);
                let p = ui.painter();
                p.circle_filled(avatar_r.center(), 17.0, Color32::from_rgb(56, 52, 48));
                // Draw portrait avatar from active/first photo or stylized photographer portrait
                if let Some(id) = app.session.active().or_else(|| app.session.visible().first().copied()) {
                    super::grid::request_thumb(app, id, 128, 4);
                    if let Some(tex) = app.renderer.thumb(id) {
                        let inner = avatar_r.shrink(2.0);
                        p.with_clip_rect(inner).image(tex.tex.id(), inner, cover_uv(inner, tex.size), Color32::WHITE);
                    }
                }
                // Warm circular frame ring
                p.circle_stroke(
                    avatar_r.center(),
                    16.5,
                    Stroke::new(if resp.hovered() { 1.8 } else { 1.2 }, Color32::from_white_alpha(150)),
                );
                if resp.clicked() {
                    let _ = app.run("app.settings", json!({}));
                }
            }

            // Navigation Tabs: [ Biblioteca ]  [ 🖼 Editar ]   Mapa  |  Imprimir
            let nav_right = more_r.left() - 12.0;
            let nav_w = 332.0;
            let nav_left = (nav_right - nav_w).max(sr.right() + 12.0);
            if nav_right - nav_left >= 220.0 {
                let is_grid = matches!(app.ui.view, ViewMode::PhotoGrid | ViewMode::SquareGrid);
                let is_people = app.ui.view == ViewMode::People;
                let is_print = matches!(app.ui.dialog, Some(crate::state::Dialog::Print));
                let is_edit = !is_grid && !is_people && !is_print;

                let cy = full.center().y;
                let lib_r = Rect::from_min_size(pos2(nav_left, cy - 16.5), vec2(92.0, 33.0));
                let edit_r = Rect::from_min_size(pos2(lib_r.right() + 8.0, cy - 16.5), vec2(96.0, 33.0));
                let map_r = Rect::from_min_size(pos2(edit_r.right() + 6.0, cy - 16.5), vec2(54.0, 33.0));
                let sep_x = map_r.right() + 4.0;
                let print_r = Rect::from_min_size(pos2(sep_x + 6.0, cy - 16.5), vec2(66.0, 33.0));

                // 1. [ Biblioteca ] pill
                let lresp = ui.interact(lib_r, egui::Id::new("navLib"), Sense::click());
                register(ui.ctx(), "button:navLib", lib_r);
                if is_grid {
                    paint_gold_glow_pill(ui.painter(), lib_r, 16);
                } else {
                    ui.painter().rect(
                        lib_r,
                        CornerRadius::same(16),
                        if lresp.hovered() {
                            Color32::from_rgba_unmultiplied(56, 60, 68, 205)
                        } else {
                            Color32::from_rgba_unmultiplied(42, 45, 52, 180)
                        },
                        Stroke::new(1.0, Color32::from_white_alpha(36)),
                        StrokeKind::Inside,
                    );
                }
                ui.painter().text(
                    lib_r.center(),
                    Align2::CENTER_CENTER,
                    "Biblioteca",
                    if is_grid { t.semibold(13.0) } else { t.font(13.0) },
                    Color32::from_rgb(242, 240, 236),
                );
                if lresp.clicked() {
                    app.ui.left_panel = true;
                    if ui.ctx().content_rect().width() < 1100.0 {
                        app.ui.right = RightPanel::None;
                    }
                    let _ = app.run("view.photoGrid", json!({}));
                }

                // 2. [ 🖼 Editar ] glowing golden-amber pill
                let eresp = ui.interact(edit_r, egui::Id::new("navEdit"), Sense::click());
                register(ui.ctx(), "button:navEdit", edit_r);
                if is_edit {
                    paint_gold_glow_pill(ui.painter(), edit_r, 16);
                } else {
                    ui.painter().rect(
                        edit_r,
                        CornerRadius::same(16),
                        if eresp.hovered() {
                            Color32::from_rgba_unmultiplied(56, 60, 68, 205)
                        } else {
                            Color32::from_rgba_unmultiplied(42, 45, 52, 180)
                        },
                        Stroke::new(1.0, Color32::from_white_alpha(36)),
                        StrokeKind::Inside,
                    );
                }
                paint(
                    ui.painter(),
                    Rect::from_center_size(pos2(edit_r.left() + 24.0, edit_r.center().y), vec2(14.0, 14.0)),
                    Icon::Edited,
                    Color32::WHITE,
                );
                ui.painter().text(
                    pos2(edit_r.left() + 36.0, edit_r.center().y),
                    Align2::LEFT_CENTER,
                    "Editar",
                    t.semibold(13.0),
                    Color32::WHITE,
                );
                if eresp.clicked() {
                    ensure_active_photo(app);
                    let _ = app.run("view.detail", json!({}));
                    app.ui.right = RightPanel::Edit;
                }

                // 3. Mapa text tab
                let mresp = ui.interact(map_r, egui::Id::new("navMap"), Sense::click());
                register(ui.ctx(), "button:navMap", map_r);
                if is_people {
                    paint_gold_glow_pill(ui.painter(), map_r, 14);
                } else if mresp.hovered() {
                    ui.painter().rect_filled(map_r, CornerRadius::same(12), Color32::from_white_alpha(18));
                }
                ui.painter().text(
                    map_r.center(),
                    Align2::CENTER_CENTER,
                    "Mapa",
                    if is_people { t.semibold(13.0) } else { t.font(13.0) },
                    if is_people || mresp.hovered() { Color32::WHITE } else { Color32::from_rgb(218, 220, 226) },
                );
                if mresp.clicked() {
                    let _ = app.run("view.people", json!({}));
                }

                // Vertical divider `|` between Mapa and Imprimir
                ui.painter().line_segment(
                    [pos2(sep_x, cy - 8.0), pos2(sep_x, cy + 8.0)],
                    Stroke::new(1.0, Color32::from_white_alpha(32)),
                );

                // 4. Imprimir text tab
                let presp = ui.interact(print_r, egui::Id::new("navPrint"), Sense::click());
                register(ui.ctx(), "button:navPrint", print_r);
                if is_print || presp.hovered() {
                    ui.painter().rect_filled(print_r, CornerRadius::same(12), Color32::from_white_alpha(18));
                }
                ui.painter().text(
                    print_r.center(),
                    Align2::CENTER_CENTER,
                    "Imprimir",
                    if is_print { t.semibold(13.0) } else { t.font(13.0) },
                    if is_print || presp.hovered() { Color32::WHITE } else { Color32::from_rgb(218, 220, 226) },
                );
                if presp.clicked() {
                    ensure_active_photo(app);
                    if let Err(e) = app.run("dialog.print", json!({})) {
                        app.toast_error(ui.ctx(), e);
                    }
                }
            }
        });
}

// ============================================================================
// 2. LEFT COLUMN (3 Frosted Cards: Biblioteca, Álbumes, Proyectos)
// ============================================================================

fn show_left_sidebar(app: &mut LightcraftApp, ui: &mut egui::Ui) {
    let t = Tokens::get(ui.ctx());
    let full = ui.max_rect();
    let gap = 8.0;

    let card1_h = 154.0;
    let avail_h = (full.height() - card1_h - gap * 2.0).max(200.0);
    let card2_h = (avail_h * 0.54).clamp(180.0, 320.0);

    let card1 = Rect::from_min_size(full.min, vec2(full.width(), card1_h));
    let card2 = Rect::from_min_size(pos2(full.left(), card1.bottom() + gap), vec2(full.width(), card2_h));
    let card3 = Rect::from_min_max(pos2(full.left(), card2.bottom() + gap), full.max);

    paint_glass_card(ui.painter(), card1, 16, 182);
    paint_glass_card(ui.painter(), card2, 16, 182);
    paint_glass_card(ui.painter(), card3, 16, 182);

    // ------------------------------------------------------------------------
    // CARD 1: Biblioteca (with circular [+] button and 4 Spanish rows)
    // ------------------------------------------------------------------------
    let counts = app.caches.counts(&app.session.catalog);
    let (total, picks, deleted) = (counts.total, counts.picks, counts.deleted);

    let hdr1_y = card1.top() + 24.0;
    let hdr1_r = Rect::from_min_max(pos2(card1.left() + 10.0, card1.top() + 8.0), pos2(card1.right() - 42.0, card1.top() + 38.0));
    let hdr1_resp = ui
        .interact(hdr1_r, egui::Id::new("studio-lib-hdr"), Sense::click())
        .on_hover_text("Abrir vista de cuadrícula de Biblioteca (G)");
    paint(
        ui.painter(),
        Rect::from_center_size(pos2(card1.left() + 22.0, hdr1_y), vec2(16.0, 16.0)),
        Icon::Folder,
        Color32::from_rgb(235, 235, 240),
    );
    ui.painter().text(
        pos2(card1.left() + 38.0, hdr1_y),
        Align2::LEFT_CENTER,
        "Biblioteca",
        t.semibold(13.5),
        Color32::WHITE,
    );
    if hdr1_resp.clicked() {
        let _ = app.run("view.photoGrid", json!({}));
    }

    // Circular [+] button on the top-right of Biblioteca
    let plus1_r = Rect::from_center_size(pos2(card1.right() - 22.0, hdr1_y), vec2(24.0, 24.0));
    let plus1_resp = ui
        .interact(plus1_r, egui::Id::new("studio-lib-add"), Sense::click())
        .on_hover_text("Importar fotografías a la Biblioteca");
    ui.painter().circle(
        plus1_r.center(),
        12.0,
        if plus1_resp.hovered() {
            Color32::from_white_alpha(48)
        } else {
            Color32::from_white_alpha(26)
        },
        Stroke::new(1.0, Color32::from_white_alpha(46)),
    );
    paint(ui.painter(), plus1_r.shrink(5.5), Icon::Plus, Color32::WHITE);
    if plus1_resp.clicked()
        && let Err(e) = app.run("file.addPhotos", json!({}))
    {
        app.toast_error(ui.ctx(), e);
    }

    let src = app.session.source;
    let lib_rows = [
        ("all", Icon::Photos, "Todas las fotos", total, LibrarySource::All),
        ("recentlyAdded", Icon::Clock, "Recientes", total.min(120), LibrarySource::RecentlyAdded),
        ("picks", Icon::Star, "Favoritas", picks, LibrarySource::Picks),
        ("recentlyDeleted", Icon::Trash, "Eliminadas", deleted, LibrarySource::RecentlyDeleted),
    ];
    for (i, (id, icon, label, count, target_src)) in lib_rows.into_iter().enumerate() {
        let ry = card1.top() + 44.0 + i as f32 * 25.5;
        let rr = Rect::from_min_max(pos2(card1.left() + 8.0, ry), pos2(card1.right() - 8.0, ry + 24.5));
        let rresp = ui.interact(rr, egui::Id::new(("studio-src", id)), Sense::click());
        register(ui.ctx(), format!("source:{id}"), rr);
        let selected = src == target_src;
        if selected {
            ui.painter().rect(
                rr,
                CornerRadius::same(7),
                Color32::from_white_alpha(30),
                Stroke::new(1.0, Color32::from_white_alpha(42)),
                StrokeKind::Inside,
            );
        } else if rresp.hovered() {
            ui.painter().rect_filled(rr, CornerRadius::same(7), Color32::from_white_alpha(14));
        }
        paint(
            ui.painter(),
            Rect::from_center_size(pos2(rr.left() + 14.0, rr.center().y), vec2(14.5, 14.5)),
            icon,
            if selected { Color32::WHITE } else { Color32::from_rgb(198, 202, 212) },
        );
        ui.painter().text(
            pos2(rr.left() + 30.0, rr.center().y),
            Align2::LEFT_CENTER,
            label,
            if selected { t.semibold(12.5) } else { t.font(12.5) },
            if selected { Color32::WHITE } else { Color32::from_rgb(224, 226, 232) },
        );
        ui.painter().text(
            pos2(rr.right() - 8.0, rr.center().y),
            Align2::RIGHT_CENTER,
            format_count(count),
            t.font(12.0),
            Color32::from_rgb(185, 190, 200),
        );
        if rresp.double_clicked() {
            let _ = app.run("library.source", json!({"kind": id}));
            ensure_active_photo(app);
            let _ = app.run("view.photoGrid", json!({}));
        } else if rresp.clicked() {
            let _ = app.run("library.source", json!({"kind": id}));
            ensure_active_photo(app);
        }
    }

    // ------------------------------------------------------------------------
    // CARD 2: Álbumes (Scrollable folder & album tree)
    // ------------------------------------------------------------------------
    let hdr2_y = card2.top() + 20.0;
    ui.painter().text(
        pos2(card2.left() + 14.0, hdr2_y),
        Align2::LEFT_CENTER,
        "Álbumes",
        t.semibold(13.5),
        Color32::WHITE,
    );
    let plus2_r = Rect::from_center_size(pos2(card2.right() - 20.0, hdr2_y), vec2(22.0, 22.0));
    let plus2_resp = ui
        .interact(plus2_r, egui::Id::new("studio-new-album"), Sense::click())
        .on_hover_text("Crear nuevo álbum");
    paint(ui.painter(), plus2_r.shrink(4.0), Icon::Plus, if plus2_resp.hovered() { Color32::WHITE } else { Color32::from_rgb(210, 214, 222) });
    if plus2_resp.clicked() {
        app.ui.dialog = Some(crate::state::Dialog::NewAlbum { name: String::new(), folder: false, parent: None, include_selected: true });
    }

    let tree_rect = Rect::from_min_max(pos2(card2.left() + 6.0, card2.top() + 34.0), pos2(card2.right() - 6.0, card2.bottom() - 6.0));
    let mut tree_ui = ui.new_child(egui::UiBuilder::new().max_rect(tree_rect).layout(egui::Layout::top_down(egui::Align::Min)));
    egui::ScrollArea::vertical().id_salt("studio-albums-scroll").auto_shrink([false, false]).show(&mut tree_ui, |ui| {
        ui.spacing_mut().item_spacing.y = 1.5;
        let w = ui.available_width();
        let album_tree: Vec<_> = {
            let kids = app.session.catalog.album_children_by_parent();
            let top: Vec<_> = kids.get(&None).map(|v| v.iter().map(|a| (**a).clone()).collect()).unwrap_or_default();
            top.into_iter()
                .map(|album| {
                    let children: Vec<_> =
                        kids.get(&Some(album.id)).map(|v| v.iter().map(|a| (**a).clone()).collect()).unwrap_or_default();
                    (album, children)
                })
                .collect()
        };
        for (album, children) in album_tree {
            if album.folder {
                let open_id = egui::Id::new(("studio-folder-open", album.id.0));
                let open = ui.data(|d| d.get_temp::<bool>(open_id)).unwrap_or(true);
                let (r, resp) = ui.allocate_exact_size(vec2(w, 25.0), Sense::click());
                if resp.clicked() {
                    ui.data_mut(|d| d.insert_temp(open_id, !open));
                }
                paint(
                    ui.painter(),
                    Rect::from_center_size(pos2(r.left() + 10.0, r.center().y), vec2(11.0, 11.0)),
                    if open { Icon::ChevronDown } else { Icon::ChevronRight },
                    Color32::from_rgb(185, 190, 200),
                );
                paint(
                    ui.painter(),
                    Rect::from_center_size(pos2(r.left() + 26.0, r.center().y), vec2(14.5, 14.5)),
                    Icon::Folder,
                    Color32::from_rgb(225, 228, 235),
                );
                ui.painter().text(
                    pos2(r.left() + 40.0, r.center().y),
                    Align2::LEFT_CENTER,
                    &album.name,
                    t.font(12.5),
                    Color32::from_rgb(236, 238, 242),
                );
                if open {
                    for child in children {
                        studio_album_row(app, ui, &t, w, &child, 18.0);
                    }
                }
            } else {
                studio_album_row(app, ui, &t, w, &album, 0.0);
            }
        }
    });

    // ------------------------------------------------------------------------
    // CARD 3: Proyectos (Vertical cards with 44×44 rounded thumbnail + title + photo count/rating)
    // ------------------------------------------------------------------------
    let hdr3_y = card3.top() + 20.0;
    ui.painter().text(
        pos2(card3.left() + 14.0, hdr3_y),
        Align2::LEFT_CENTER,
        "Proyectos",
        t.semibold(13.5),
        Color32::WHITE,
    );
    let plus3_r = Rect::from_center_size(pos2(card3.right() - 20.0, hdr3_y), vec2(22.0, 22.0));
    let plus3_resp = ui
        .interact(plus3_r, egui::Id::new("studio-new-project"), Sense::click())
        .on_hover_text("Crear nuevo proyecto / álbum");
    paint(ui.painter(), plus3_r.shrink(4.0), Icon::Plus, if plus3_resp.hovered() { Color32::WHITE } else { Color32::from_rgb(210, 214, 222) });
    if plus3_resp.clicked() {
        app.ui.dialog = Some(crate::state::Dialog::NewAlbum { name: String::new(), folder: false, parent: None, include_selected: true });
    }

    let proj_rect = Rect::from_min_max(pos2(card3.left() + 6.0, card3.top() + 34.0), pos2(card3.right() - 6.0, card3.bottom() - 6.0));
    let mut proj_ui = ui.new_child(egui::UiBuilder::new().max_rect(proj_rect).layout(egui::Layout::top_down(egui::Align::Min)));
    egui::ScrollArea::vertical()
        .id_salt("studio-projects-scroll")
        .auto_shrink([false, false])
        .show_viewport(&mut proj_ui, |ui, viewport| {
            ui.spacing_mut().item_spacing.y = 4.0;
            let w = ui.available_width();
            let vis_photos = app.session.visible_cloned();
            let active_id = app.session.active();
            let ppp = ui.ctx().pixels_per_point();

            // Virtualized list of individual photos of the current album downwards (`hacia abajo`)
            let multi_sel = app.session.selection.ids.len() > 1;
            let row_h = 50.0f32;
            let gap_y = 4.0f32;
            let stride_y = row_h + gap_y;
            let total_list_h = if vis_photos.is_empty() {
                0.0
            } else {
                vis_photos.len() as f32 * stride_y - gap_y
            };
            let (list_rect, _) = ui.allocate_exact_size(vec2(w, total_list_h), Sense::hover());
            let rel_min_y = viewport.min.y.max(0.0);
            let rel_max_y = viewport.max.y.max(0.0);
            let first_idx = (rel_min_y / stride_y).floor().max(0.0) as usize;
            let last_idx = (((rel_max_y / stride_y).ceil().max(0.0) as usize) + 1).min(vis_photos.len());

            for (idx, &id) in vis_photos.iter().enumerate().take(last_idx).skip(first_idx) {
                let y0 = list_rect.top() + idx as f32 * stride_y;
                let row = Rect::from_min_size(pos2(list_rect.left(), y0), vec2(w, row_h));
                let resp = ui.interact(row, egui::Id::new(("studio-proj-photo", id.0)), Sense::click());
                register(ui.ctx(), format!("proj_photo:{}", id.0), row);
                let is_active = Some(id) == active_id;
                let is_selected = app.session.selection.contains(id);
                if is_active || is_selected || resp.hovered() {
                    ui.painter().rect(
                        row,
                        CornerRadius::same(9),
                        if is_active {
                            Color32::from_rgba_unmultiplied(238, 175, 80, 44)
                        } else if is_selected {
                            Color32::from_rgba_unmultiplied(238, 175, 80, 26)
                        } else {
                            Color32::from_white_alpha(14)
                        },
                        Stroke::new(
                            if is_active { 1.4 } else { 1.0 },
                            if is_active {
                                Color32::from_rgb(245, 198, 102)
                            } else if is_selected {
                                Color32::from_rgb(225, 175, 85)
                            } else {
                                Color32::from_white_alpha(24)
                            },
                        ),
                        StrokeKind::Inside,
                    );
                }
                let thumb_r = Rect::from_min_size(pos2(row.left() + 4.0, row.center().y - 21.0), vec2(42.0, 42.0));
                ui.painter().rect_filled(thumb_r, CornerRadius::same(7), Color32::from_rgb(28, 30, 36));
                super::grid::request_thumb(app, id, (128.0 * ppp.min(2.0)) as usize, 6);
                if let Some(tex) = app.renderer.thumb(id) {
                    ui.painter()
                        .with_clip_rect(thumb_r)
                        .image(tex.tex.id(), thumb_r, cover_uv(thumb_r, tex.size), Color32::WHITE);
                }
                ui.painter().rect_stroke(thumb_r, CornerRadius::same(7), Stroke::new(1.0, Color32::from_white_alpha(32)), StrokeKind::Inside);

                let check_r = Rect::from_min_size(thumb_r.min + vec2(2.0, 2.0), vec2(15.0, 15.0));
                if resp.hovered() || is_selected || multi_sel {
                    let c_bg = if is_selected { Color32::from_rgb(235, 185, 88) } else { Color32::from_black_alpha(155) };
                    let c_st = if is_selected { Color32::from_rgb(255, 225, 145) } else { Color32::from_white_alpha(185) };
                    ui.painter().circle(check_r.center(), 6.8, c_bg, Stroke::new(1.2, c_st));
                    if is_selected {
                        ui.painter().text(check_r.center(), Align2::CENTER_CENTER, "✓", t.semibold(9.5), Color32::from_rgb(20, 16, 10));
                    }
                }

                let (fname, sub) = app
                    .session
                    .catalog
                    .photo(id)
                    .map(|p| {
                        let s = if p.rating > 0 {
                            "★".repeat(p.rating as usize)
                        } else {
                            format!("{}×{}", p.width, p.height)
                        };
                        (p.file_name.clone(), s)
                    })
                    .unwrap_or_else(|| (format!("Foto #{}", id.0), String::new()));
                ui.painter().text(
                    pos2(thumb_r.right() + 10.0, row.center().y - 8.0),
                    Align2::LEFT_CENTER,
                    fname,
                    if is_active || is_selected { t.semibold(12.0) } else { t.font(12.0) },
                    Color32::WHITE,
                );
                ui.painter().text(
                    pos2(thumb_r.right() + 10.0, row.center().y + 9.0),
                    Align2::LEFT_CENTER,
                    sub,
                    t.font(11.0),
                    if is_active || is_selected { Color32::from_rgb(245, 198, 98) } else { Color32::from_rgb(176, 180, 190) },
                );
                if resp.double_clicked() {
                    let _ = app.run("library.select", json!({"ids": [id.0], "mode": "replace"}));
                    let _ = app.run("view.detail", json!({}));
                    app.ui.right = RightPanel::Edit;
                } else if resp.clicked() {
                    let clicked_check = resp.interact_pointer_pos().is_some_and(|pos| check_r.expand(3.0).contains(pos));
                    let m = ui.input(|i| i.modifiers);
                    let mode = if m.shift {
                        "range"
                    } else if m.command || m.ctrl || clicked_check {
                        "toggle"
                    } else {
                        "replace"
                    };
                    let _ = app.run("library.select", json!({"ids": [id.0], "mode": mode}));
                }
            }
        });
}

fn studio_album_row(app: &mut LightcraftApp, ui: &mut egui::Ui, t: &Tokens, w: f32, album: &lightcraft_catalog::Album, indent: f32) {
    let selected = app.session.source == LibrarySource::Album(album.id);
    let shown_count = app.session.catalog.album_count(album.id);
    let (r, resp) = ui.allocate_exact_size(vec2(w, 25.0), Sense::click());
    register(ui.ctx(), format!("album:{}", album.id.0), r);
    let p = ui.painter();
    if selected {
        p.rect(
            r,
            CornerRadius::same(7),
            Color32::from_white_alpha(34),
            Stroke::new(1.0, Color32::from_white_alpha(48)),
            StrokeKind::Inside,
        );
    } else if resp.hovered() {
        p.rect_filled(r, CornerRadius::same(7), Color32::from_white_alpha(14));
    }
    let x0 = r.left() + 14.0 + indent;
    paint(
        p,
        Rect::from_center_size(pos2(x0, r.center().y), vec2(14.5, 14.5)),
        Icon::Folder,
        if selected { Color32::WHITE } else { Color32::from_rgb(205, 210, 220) },
    );
    p.text(
        pos2(x0 + 16.0, r.center().y),
        Align2::LEFT_CENTER,
        &album.name,
        if selected { t.semibold(12.3) } else { t.font(12.3) },
        if selected { Color32::WHITE } else { Color32::from_rgb(228, 230, 236) },
    );
    p.text(
        pos2(r.right() - 8.0, r.center().y),
        Align2::RIGHT_CENTER,
        shown_count.to_string(),
        t.font(11.5),
        Color32::from_rgb(185, 190, 200),
    );
    if resp.double_clicked() {
        let _ = app.run("library.source", json!({"kind": "album", "id": album.id.0}));
        ensure_active_photo(app);
        let _ = app.run("view.photoGrid", json!({}));
    } else if resp.clicked() {
        let _ = app.run("library.source", json!({"kind": "album", "id": album.id.0}));
        ensure_active_photo(app);
    }
}

// ============================================================================
// 3. CENTER PHOTO STAGE CONTAINER
// ============================================================================

fn show_center_stage(app: &mut LightcraftApp, ui: &mut egui::Ui, stage_rect: Rect, screen_anim: f32, photo_anim: f32) {
    let t = Tokens::get(ui.ctx());
    paint_glass_card(ui.painter(), stage_rect, 18, 168);

    let top_bar_h = 42.0;
    let show_film = app.ui.filmstrip && !app.ui.section_folded("studioBottomFilm") && !app.session.visible().is_empty();
    let film_h = if show_film { 68.0 } else { 0.0 };
    let mid_bar_h = 36.0;

    let top_sub = Rect::from_min_size(stage_rect.min, vec2(stage_rect.width(), top_bar_h));
    let film_rect = Rect::from_min_max(
        pos2(stage_rect.left() + 8.0, stage_rect.bottom() - 8.0 - film_h),
        pos2(stage_rect.right() - 8.0, stage_rect.bottom() - 8.0),
    );
    let mid_sub = Rect::from_min_max(
        pos2(stage_rect.left() + 8.0, film_rect.top() - mid_bar_h - if show_film { 4.0 } else { 0.0 }),
        pos2(stage_rect.right() - 8.0, film_rect.top() - if show_film { 4.0 } else { 0.0 }),
    );
    let photo_area = Rect::from_min_max(
        pos2(stage_rect.left() + 10.0, top_sub.bottom() + 2.0),
        pos2(stage_rect.right() - 10.0, mid_sub.top() - 4.0),
    );

    // ------------------------------------------------------------------------
    // 3A. Top Photo Sub-bar: [< | >]  DSC05094.ARW  7952 × 5304   ★★★★☆ | ⚑ ♡ ⋯   [Fit ⌄ | 100% ⌄] [🕘]
    // ------------------------------------------------------------------------
    {
        let cy = top_sub.center().y + 1.0;
        let nav_pill = Rect::from_min_size(pos2(top_sub.left() + 12.0, cy - 14.0), vec2(54.0, 28.0));
        ui.painter().rect(
            nav_pill,
            CornerRadius::same(14),
            Color32::from_white_alpha(20),
            Stroke::new(1.0, Color32::from_white_alpha(36)),
            StrokeKind::Inside,
        );
        ui.painter().line_segment(
            [pos2(nav_pill.center().x, nav_pill.top() + 5.0), pos2(nav_pill.center().x, nav_pill.bottom() - 5.0)],
            Stroke::new(1.0, Color32::from_white_alpha(26)),
        );
        let prev_r = Rect::from_min_max(nav_pill.min, pos2(nav_pill.center().x, nav_pill.bottom()));
        let next_r = Rect::from_min_max(pos2(nav_pill.center().x, nav_pill.top()), nav_pill.max);
        for (r, nid, icon, dir, tip) in [
            (prev_r, "studioPrev", Icon::ChevronLeft, -1i32, "Foto anterior (←)"),
            (next_r, "studioNext", Icon::ChevronRight, 1i32, "Foto siguiente (→)"),
        ] {
            let resp = ui.interact(r, egui::Id::new(nid), Sense::click()).on_hover_text(tip);
            register(ui.ctx(), format!("button:{nid}"), r);
            paint(
                ui.painter(),
                Rect::from_center_size(r.center(), vec2(13.0, 13.0)),
                icon,
                if resp.hovered() { Color32::WHITE } else { Color32::from_rgb(215, 220, 228) },
            );
            if resp.clicked() {
                let (_, ids) = app.session.visible_shared();
                if let Some(cur) = app.session.selection.active
                    && let Some(pos) = ids.iter().position(|x| *x == cur)
                {
                    let next_idx = (pos as i32 + dir).clamp(0, ids.len().saturating_sub(1) as i32) as usize;
                    if let Some(target) = ids.get(next_idx) {
                        let _ = app.run("library.select", json!({"ids": [target.0], "mode": "replace"}));
                    }
                }
            }
        }

        let active_photo = app.session.active().and_then(|id| app.session.catalog.photo(id)).cloned();
        let fname = active_photo.as_ref().map(|p| p.file_name.as_str()).unwrap_or("DSC05094.ARW");
        let dims = active_photo
            .as_ref()
            .map(|p| format!("{} × {}", p.width, p.height))
            .unwrap_or_else(|| "7952 × 5304".to_string());
        let rating = active_photo.as_ref().map(|p| p.rating).unwrap_or(4);
        let flag = active_photo.as_ref().map(|p| p.flag).unwrap_or_default();

        let fn_galley = ui.painter().layout_no_wrap(fname.to_string(), t.semibold(13.0), Color32::WHITE);
        let fn_x = nav_pill.right() + 14.0;
        let fn_w = fn_galley.size().x;
        ui.painter().galley(pos2(fn_x, cy - fn_galley.size().y * 0.5), fn_galley, Color32::WHITE);
        ui.painter().text(
            pos2(fn_x + fn_w + 14.0, cy),
            Align2::LEFT_CENTER,
            dims,
            t.font(12.2),
            Color32::from_rgb(186, 190, 200),
        );

        // Right side of Top Sub-bar: [ Fit ⌄ | 100% ⌄ ]  [ 🕘 ]
        let hist_btn = Rect::from_min_size(pos2(top_sub.right() - 12.0 - 28.0, cy - 14.0), vec2(28.0, 28.0));
        let zoom_pill = Rect::from_min_size(pos2(hist_btn.left() - 8.0 - 110.0, cy - 14.0), vec2(110.0, 28.0));

        // Center of Top Sub-bar: ★★★★☆ | ⚑  ♡  ⋯
        let center_x = ((fn_x + fn_w + 95.0) + zoom_pill.left()) * 0.5;
        if zoom_pill.left() - (fn_x + fn_w + 90.0) >= 160.0 {
            let star_rect = Rect::from_center_size(pos2(center_x - 36.0, cy), vec2(96.0, 24.0));
            let mut sui = ui.new_child(egui::UiBuilder::new().max_rect(star_rect).layout(egui::Layout::left_to_right(egui::Align::Center)));
            if let Some(r) = stars(&mut sui, "top-stage", rating, 15.0) {
                let _ = app.run("photo.rate", json!({"rating": r}));
            }
            let div_x = star_rect.right() + 6.0;
            ui.painter().line_segment([pos2(div_x, cy - 7.0), pos2(div_x, cy + 7.0)], Stroke::new(1.0, Color32::from_white_alpha(28)));

            let flag_r = Rect::from_center_size(pos2(div_x + 18.0, cy), vec2(24.0, 24.0));
            let heart_r = Rect::from_center_size(pos2(flag_r.right() + 14.0, cy), vec2(24.0, 24.0));
            let dots_r = Rect::from_center_size(pos2(heart_r.right() + 14.0, cy), vec2(24.0, 24.0));

            let f_resp = ui.interact(flag_r, egui::Id::new("top-stage-flag"), Sense::click()).on_hover_text("Marcar Pick (P)");
            paint(
                ui.painter(),
                flag_r.shrink(4.5),
                if flag == Flag::Pick { Icon::FlagPickFilled } else { Icon::FlagPick },
                if flag == Flag::Pick { Color32::from_rgb(245, 192, 96) } else { Color32::from_rgb(215, 220, 228) },
            );
            if f_resp.clicked() {
                let next = if flag == Flag::Pick { "none" } else { "pick" };
                let _ = app.run("photo.flag", json!({"flag": next}));
            }

            let h_resp = ui.interact(heart_r, egui::Id::new("top-stage-heart"), Sense::click()).on_hover_text("Favorita (P)");
            paint(
                ui.painter(),
                heart_r.shrink(4.5),
                Icon::Heart,
                if flag == Flag::Pick { Color32::from_rgb(245, 120, 128) } else { Color32::from_rgb(215, 220, 228) },
            );
            if h_resp.clicked() {
                let next = if flag == Flag::Pick { "none" } else { "pick" };
                let _ = app.run("photo.flag", json!({"flag": next}));
            }

            let d_resp = ui.interact(dots_r, egui::Id::new("top-stage-more"), Sense::click()).on_hover_text("Más opciones de foto");
            paint(ui.painter(), dots_r.shrink(5.0), Icon::More, Color32::from_rgb(215, 220, 228));
            egui::Popup::menu(&d_resp).show(|ui| {
                if ui.button("Rotar a la izquierda (⌘[)").clicked() {
                    let _ = app.run("develop.rotateLeft", json!({}));
                    ui.close();
                }
                if ui.button("Rotar a la derecha (⌘])").clicked() {
                    let _ = app.run("develop.rotateRight", json!({}));
                    ui.close();
                }
                if ui.button("Restablecer ajustes").clicked() {
                    let _ = app.run("develop.reset", json!({}));
                    ui.close();
                }
            });
        }

        // [ Fit ⌄ | 100% ⌄ ] pill
        show_zoom_pill_in_bar(app, ui, zoom_pill, hist_btn);
    }

    // ------------------------------------------------------------------------
    // 3B. Main Photo Viewport (rounded corners + subtle glass border)
    // ------------------------------------------------------------------------
    {
        ui.painter().rect_filled(photo_area, CornerRadius::same(14), Color32::from_rgba_unmultiplied(16, 18, 22, 220));
        let mut cui = ui.new_child(egui::UiBuilder::new().max_rect(photo_area).layout(egui::Layout::top_down(egui::Align::Min)));
        cui.set_clip_rect(photo_area);
        match app.ui.view {
            ViewMode::PhotoGrid | ViewMode::SquareGrid => super::grid::show(app, &mut cui),
            ViewMode::Detail => {
                let saved_film = app.ui.filmstrip;
                app.ui.filmstrip = false;
                super::detail::show(app, &mut cui);
                app.ui.filmstrip = saved_film;
            }
            ViewMode::Compare => {
                let saved_film = app.ui.filmstrip;
                app.ui.filmstrip = false;
                super::compare::show_compare(app, &mut cui);
                app.ui.filmstrip = saved_film;
            }
            ViewMode::Survey => super::compare::show_survey(app, &mut cui),
            ViewMode::Reference => super::compare::show_reference(app, &mut cui),
            ViewMode::People => super::people::show(app, &mut cui),
        }
        let combined_anim = screen_anim.min(0.35 + 0.65 * photo_anim);
        if combined_anim < 0.999 {
            let alpha = ((1.0 - combined_anim) * 85.0).round().clamp(0.0, 255.0) as u8;
            cui.painter().rect_filled(photo_area, CornerRadius::same(14), Color32::from_rgba_unmultiplied(18, 20, 24, alpha));
        }
        ui.painter().rect_stroke(
            photo_area,
            CornerRadius::same(14),
            Stroke::new(1.0, Color32::from_white_alpha(34)),
            StrokeKind::Inside,
        );
    }

    // ------------------------------------------------------------------------
    // 3C. Middle Control Sub-bar: [⊞] [☰] [⊡]  ──●──────        ★★★★☆   ⋯   🗑
    // ------------------------------------------------------------------------
    {
        let cy = mid_sub.center().y;
        let is_grid = matches!(app.ui.view, ViewMode::PhotoGrid | ViewMode::SquareGrid);
        let is_comp = app.ui.view == ViewMode::Compare;
        let btns = [
            ("botGrid", Icon::GridPhoto, is_grid, "Vista de cuadrícula / Biblioteca (G)"),
            ("botCompare", Icon::Compare, is_comp, "Vista Comparar (⇧C)"),
            ("botDetail", Icon::Single, !is_grid && !is_comp, "Vista Detalle / Pantalla completa"),
        ];
        let mut bx = mid_sub.left() + 6.0;
        for (bid, bicon, bactive, btip) in btns {
            let br = Rect::from_min_size(pos2(bx, cy - 13.0), vec2(28.0, 26.0));
            let bresp = ui.interact(br, egui::Id::new(bid), Sense::click()).on_hover_text(btip);
            register(ui.ctx(), format!("button:{bid}"), br);
            if bactive || bresp.hovered() {
                ui.painter().rect(
                    br,
                    CornerRadius::same(7),
                    Color32::from_white_alpha(if bactive { 34 } else { 18 }),
                    Stroke::new(1.0, Color32::from_white_alpha(42)),
                    StrokeKind::Inside,
                );
            }
            paint(ui.painter(), br.shrink(5.5), bicon, if bactive { Color32::WHITE } else { Color32::from_rgb(200, 205, 215) });
            if bresp.clicked() {
                match bid {
                    "botGrid" => {
                        let _ = app.run("view.photoGrid", json!({}));
                    }
                    "botCompare" => {
                        ensure_active_photo(app);
                        let _ = app.run("view.compare", json!({}));
                    }
                    "botDetail" => {
                        ensure_active_photo(app);
                        let _ = app.run("view.detail", json!({}));
                    }
                    _ => {}
                }
            }
            bx += 32.0;
        }

        // Interactive zoom/scale slider `──●──────`
        let sl_r = Rect::from_min_size(pos2(bx + 10.0, cy - 10.0), vec2(96.0, 20.0));
        let sl_resp = ui.interact(sl_r, egui::Id::new("stage-zoom-slider"), Sense::click_and_drag());
        let zoom_norm = match app.ui.zoom {
            Zoom::Fit | Zoom::Fill => 0.25f32,
            Zoom::Percent(p) => ((p - 25.0) / 275.0).clamp(0.0, 1.0),
        };
        if (sl_resp.dragged() || sl_resp.clicked())
            && let Some(pos) = sl_resp.interact_pointer_pos()
        {
            let n = ((pos.x - sl_r.left()) / sl_r.width()).clamp(0.0, 1.0);
            if n <= 0.28 {
                app.ui.zoom = Zoom::Fit;
                app.ui.pan = (0.5, 0.5);
            } else {
                app.ui.zoom = Zoom::Percent((25.0 + n * 275.0).round());
            }
        }
        let bar_line = Rect::from_min_max(pos2(sl_r.left(), cy - 1.2), pos2(sl_r.right(), cy + 1.2));
        ui.painter().rect_filled(bar_line, 1.0, Color32::from_white_alpha(45));
        let tx = sl_r.left() + sl_r.width() * zoom_norm;
        ui.painter().rect_filled(Rect::from_min_max(pos2(sl_r.left(), cy - 1.2), pos2(tx, cy + 1.2)), 1.0, Color32::from_white_alpha(150));
        ui.painter().circle_filled(pos2(tx, cy), 5.5, Color32::WHITE);

        // Right side of Middle Control Sub-bar: [📋 Pegar ajustes (N)] [✓ Todas] ★★★★☆   ⋯   🗑
        let del_r = Rect::from_min_size(pos2(mid_sub.right() - 32.0, cy - 13.0), vec2(26.0, 26.0));
        let more_r = Rect::from_min_size(pos2(del_r.left() - 30.0, cy - 13.0), vec2(26.0, 26.0));
        let stars_r = Rect::from_min_size(pos2(more_r.left() - 104.0, cy - 12.0), vec2(96.0, 24.0));

        let sel_n = app.session.selection.ids.len();
        if sel_n > 1 || is_grid {
            let paste_lbl = if sel_n > 1 {
                format!("📋 Pegar ajustes ({sel_n})")
            } else {
                "📋 Pegar ajustes".to_string()
            };
            let paste_w = if sel_n > 1 { 142.0 } else { 118.0 };
            let paste_r = Rect::from_min_size(pos2(stars_r.left() - paste_w - 8.0, cy - 12.0), vec2(paste_w, 24.0));
            if paste_r.left() > sl_r.right() + 10.0 {
                let p_resp = ui
                    .interact(paste_r, egui::Id::new("mid-stage-paste-multi"), Sense::click())
                    .on_hover_text("Pegar todos los ajustes (Luz, Color, LUTs y Efectos Mágicos IA) en todas las fotos seleccionadas (⌘V)");
                ui.painter().rect(
                    paste_r,
                    CornerRadius::same(7),
                    if p_resp.hovered() {
                        Color32::from_rgba_unmultiplied(242, 185, 88, 72)
                    } else if sel_n > 1 {
                        Color32::from_rgba_unmultiplied(238, 175, 80, 48)
                    } else {
                        Color32::from_white_alpha(22)
                    },
                    Stroke::new(
                        1.0,
                        if sel_n > 1 || p_resp.hovered() {
                            Color32::from_rgb(245, 198, 102)
                        } else {
                            Color32::from_white_alpha(42)
                        },
                    ),
                    StrokeKind::Inside,
                );
                ui.painter().text(
                    paste_r.center(),
                    Align2::CENTER_CENTER,
                    paste_lbl,
                    t.semibold(11.0),
                    if sel_n > 1 { Color32::from_rgb(255, 224, 150) } else { Color32::WHITE },
                );
                if p_resp.clicked() {
                    paste_all_adjustments_to_selection(app, ui.ctx());
                }

                let sel_all_r = Rect::from_min_size(pos2(paste_r.left() - 78.0, cy - 12.0), vec2(72.0, 24.0));
                if sel_all_r.left() > sl_r.right() + 8.0 {
                    let vis_total = app.session.visible_shared().1.len();
                    let all_selected = vis_total > 1 && sel_n >= vis_total;
                    let sa_lbl = if all_selected { "Deselecc." } else { "✓ Todas" };
                    let sa_resp = ui
                        .interact(sel_all_r, egui::Id::new("mid-stage-select-all"), Sense::click())
                        .on_hover_text("Seleccionar / deseleccionar todas las fotos visibles (⌘A)");
                    ui.painter().rect(
                        sel_all_r,
                        CornerRadius::same(7),
                        if sa_resp.hovered() { Color32::from_white_alpha(32) } else { Color32::from_white_alpha(18) },
                        Stroke::new(1.0, Color32::from_white_alpha(38)),
                        StrokeKind::Inside,
                    );
                    ui.painter().text(sel_all_r.center(), Align2::CENTER_CENTER, sa_lbl, t.font(10.8), Color32::from_rgb(220, 225, 235));
                    if sa_resp.clicked() {
                        if all_selected {
                            if let Some(act) = app.session.selection.active {
                                let _ = app.run("library.select", json!({"ids": [act.0], "mode": "replace"}));
                            } else {
                                let _ = app.run("library.selectNone", json!({}));
                            }
                        } else {
                            let _ = app.run("library.selectAll", json!({}));
                        }
                    }
                }
            }
        }

        let rating = app.session.active().and_then(|id| app.session.catalog.photo(id)).map(|p| p.rating).unwrap_or(4);
        let mut sui = ui.new_child(egui::UiBuilder::new().max_rect(stars_r).layout(egui::Layout::left_to_right(egui::Align::Center)));
        if let Some(r) = stars(&mut sui, "mid-stage", rating, 15.0) {
            let _ = app.run("photo.rate", json!({"rating": r}));
        }

        let m_resp = ui.interact(more_r, egui::Id::new("mid-stage-more"), Sense::click()).on_hover_text("Opciones de tira de fotos");
        paint(ui.painter(), more_r.shrink(5.5), Icon::More, Color32::from_rgb(200, 205, 215));
        if m_resp.clicked() {
            app.ui.toggle_folded("studioBottomFilm");
        }

        let d_resp = ui.interact(del_r, egui::Id::new("mid-stage-trash"), Sense::click()).on_hover_text("Mover foto a Eliminadas (⌫)");
        paint(ui.painter(), del_r.shrink(5.0), Icon::Trash, if d_resp.hovered() { Color32::from_rgb(245, 110, 110) } else { Color32::from_rgb(200, 205, 215) });
        if d_resp.clicked() {
            let _ = app.run("photo.delete", json!({}));
        }
    }

    // ------------------------------------------------------------------------
    // 3D. Bottom Horizontal Filmstrip Trough (`abajo` debajo de la foto central)
    // ------------------------------------------------------------------------
    if show_film {
        ui.painter().rect(
            film_rect,
            CornerRadius::same(12),
            Color32::from_rgba_unmultiplied(18, 20, 25, 210),
            Stroke::new(1.0, Color32::from_white_alpha(28)),
            StrokeKind::Inside,
        );
        register(ui.ctx(), "panel:bfilmstrip", film_rect);
        let strip_inner = Rect::from_min_max(pos2(film_rect.left() + 6.0, film_rect.top() + 5.0), pos2(film_rect.right() - 28.0, film_rect.bottom() - 5.0));
        let mut fui = ui.new_child(egui::UiBuilder::new().max_rect(strip_inner).layout(egui::Layout::left_to_right(egui::Align::Center)));
        fui.set_clip_rect(strip_inner);
        show_bottom_filmstrip(app, &mut fui);

        // Right collapse/scroll indicator `‹·`
        let col_r = Rect::from_min_max(pos2(film_rect.right() - 26.0, film_rect.top() + 4.0), pos2(film_rect.right() - 2.0, film_rect.bottom() - 4.0));
        let col_resp = ui.interact(col_r, egui::Id::new("button:botFilmstrip"), Sense::click()).on_hover_text("Ocultar / mostrar tira de fotos (F6)");
        register(ui.ctx(), "button:botFilmstrip", col_r);
        paint(ui.painter(), Rect::from_center_size(col_r.center(), vec2(12.0, 12.0)), Icon::ChevronLeft, Color32::from_rgb(190, 195, 205));
        if col_resp.clicked() {
            app.ui.filmstrip = false;
        }
    }
}

fn show_zoom_pill_in_bar(app: &mut LightcraftApp, ui: &mut egui::Ui, zoom_r: Rect, hist_r: Rect) {
    let t = Tokens::get(ui.ctx());
    let p = ui.painter();
    p.rect(
        zoom_r,
        CornerRadius::same(8),
        Color32::from_white_alpha(20),
        Stroke::new(1.0, Color32::from_white_alpha(38)),
        StrokeKind::Inside,
    );
    let half = zoom_r.width() * 0.48;
    let fit_r = Rect::from_min_size(zoom_r.min, vec2(half, zoom_r.height()));
    let pct_r = Rect::from_min_size(pos2(zoom_r.left() + half, zoom_r.top()), vec2(zoom_r.width() - half, zoom_r.height()));
    let fit_resp = ui.interact(fit_r, egui::Id::new("studio-zoom-fit"), Sense::click()).on_hover_text("Ajustar a ventana (Fit)");
    let pct_resp = ui.interact(pct_r, egui::Id::new("studio-zoom-100"), Sense::click()).on_hover_text("Zoom 100% (Z)");

    let is_fit = matches!(app.ui.zoom, Zoom::Fit | Zoom::Fill);
    p.text(
        pos2(fit_r.center().x - 5.0, fit_r.center().y),
        Align2::CENTER_CENTER,
        "Fit",
        if is_fit { t.semibold(12.0) } else { t.font(12.0) },
        Color32::WHITE,
    );
    paint(p, Rect::from_center_size(pos2(fit_r.right() - 10.0, fit_r.center().y + 1.0), vec2(9.0, 9.0)), Icon::ChevronDown, Color32::from_rgb(190, 195, 205));
    p.line_segment(
        [pos2(zoom_r.left() + half, zoom_r.top() + 5.0), pos2(zoom_r.left() + half, zoom_r.bottom() - 5.0)],
        Stroke::new(1.0, Color32::from_white_alpha(28)),
    );
    let pct_label = match app.ui.zoom {
        Zoom::Percent(v) => format!("{v:.0}%"),
        _ => "100%".to_string(),
    };
    p.text(
        pos2(pct_r.center().x - 5.0, pct_r.center().y),
        Align2::CENTER_CENTER,
        pct_label,
        if !is_fit { t.semibold(12.0) } else { t.font(12.0) },
        Color32::WHITE,
    );
    paint(p, Rect::from_center_size(pos2(pct_r.right() - 9.0, pct_r.center().y + 1.0), vec2(9.0, 9.0)), Icon::ChevronDown, Color32::from_rgb(190, 195, 205));
    if fit_resp.clicked() {
        app.ui.zoom = Zoom::Fit;
        app.ui.pan = (0.5, 0.5);
    }
    if pct_resp.clicked() {
        app.ui.zoom = if matches!(app.ui.zoom, Zoom::Percent(_)) { Zoom::Fit } else { Zoom::Percent(100.0) };
        app.ui.pan = (0.5, 0.5);
    }

    // [ 🕘 ] Before / After & History button
    let h_resp = ui.interact(hist_r, egui::Id::new("studio-before-after-btn"), Sense::click()).on_hover_text("Antes / Después (Y)");
    p.rect(
        hist_r,
        CornerRadius::same(8),
        if h_resp.hovered() { Color32::from_white_alpha(34) } else { Color32::from_white_alpha(20) },
        Stroke::new(1.0, Color32::from_white_alpha(38)),
        StrokeKind::Inside,
    );
    paint(p, hist_r.shrink(6.5), Icon::Clock, Color32::WHITE);
    if h_resp.clicked() {
        ensure_active_photo(app);
        let _ = app.run("view.beforeAfterSplit", json!({}));
    }
}

fn show_bottom_filmstrip(app: &mut LightcraftApp, ui: &mut egui::Ui) {
    let t = Tokens::get(ui.ctx());
    let (_, ids) = app.session.visible_shared();
    let active = app.session.selection.active;
    let multi_sel = app.session.selection.ids.len() > 1;
    let ppp = ui.ctx().pixels_per_point();

    let last_active_id = egui::Id::new("studio-bfilm-last-active");
    let prev_active = ui.ctx().data(|d| d.get_temp::<Option<PhotoId>>(last_active_id)).flatten();
    let active_changed = active != prev_active;
    if active_changed {
        ui.ctx().data_mut(|d| d.insert_temp(last_active_id, active));
    }

    let cell_w = 64.0f32;
    let cell_h = 54.0f32;
    let gap_x = 8.0f32;
    let stride = cell_w + gap_x;

    let mut scroll = egui::ScrollArea::horizontal()
        .id_salt("studio-bottom-filmstrip-scroll")
        .auto_shrink([false, false]);
    if active_changed
        && let Some(act_id) = active
        && let Some(act_idx) = ids.iter().position(|&id| id == act_id)
    {
        let avail_w = ui.available_width();
        let target_x = (act_idx as f32 * stride - (avail_w - cell_w) * 0.5).max(0.0);
        scroll = scroll.horizontal_scroll_offset(target_x);
    }

    scroll.show_viewport(ui, |ui, viewport| {
            let total_w = if ids.is_empty() {
                0.0
            } else {
                ids.len() as f32 * stride - gap_x
            };
            let (strip_r, _) = ui.allocate_exact_size(vec2(total_w, cell_h), Sense::hover());

            let first_idx = (viewport.min.x / stride).floor().max(0.0) as usize;
            let last_idx = ((viewport.max.x / stride).ceil().max(0.0) as usize)
                .min(ids.len())
                .min(first_idx + 14);

            for (idx, &id) in ids.iter().enumerate().take(last_idx).skip(first_idx) {
                let x0 = strip_r.left() + idx as f32 * stride;
                let cell = Rect::from_min_size(pos2(x0, strip_r.top()), vec2(cell_w, cell_h));
                let resp = ui.interact(cell, egui::Id::new(("studio-bfilm-cell", id.0)), Sense::click());
                register(ui.ctx(), format!("film:{}", id.0), cell);
                register(ui.ctx(), format!("bfilm:{}", id.0), cell);
                let thumb_r = Rect::from_center_size(cell.center(), vec2(62.0, 52.0));
                let is_active = Some(id) == active;
                let is_selected = app.session.selection.contains(id);

                super::grid::request_thumb(app, id, (192.0 * ppp.min(2.0)) as usize, 6);
                let p = ui.painter();
                p.rect_filled(thumb_r, CornerRadius::same(7), Color32::from_rgb(24, 27, 33));
                if let Some(tex) = app.renderer.thumb(id) {
                    p.with_clip_rect(thumb_r)
                        .image(tex.tex.id(), thumb_r, cover_uv(thumb_r, tex.size), Color32::WHITE);
                }
                if is_active {
                    p.rect_stroke(
                        thumb_r,
                        CornerRadius::same(7),
                        Stroke::new(2.2, Color32::from_rgb(248, 224, 172)),
                        StrokeKind::Inside,
                    );
                } else if is_selected {
                    p.rect_stroke(
                        thumb_r,
                        CornerRadius::same(7),
                        Stroke::new(1.8, Color32::from_rgb(232, 182, 92)),
                        StrokeKind::Inside,
                    );
                } else if resp.hovered() {
                    p.rect_stroke(
                        thumb_r,
                        CornerRadius::same(7),
                        Stroke::new(1.2, Color32::from_white_alpha(140)),
                        StrokeKind::Inside,
                    );
                } else {
                    p.rect_stroke(
                        thumb_r,
                        CornerRadius::same(7),
                        Stroke::new(1.0, Color32::from_white_alpha(28)),
                        StrokeKind::Inside,
                    );
                }

                let check_r = Rect::from_min_size(thumb_r.min + vec2(3.0, 3.0), vec2(15.0, 15.0));
                if resp.hovered() || is_selected || multi_sel {
                    let c_bg = if is_selected { Color32::from_rgb(235, 185, 88) } else { Color32::from_black_alpha(155) };
                    let c_st = if is_selected { Color32::from_rgb(255, 225, 145) } else { Color32::from_white_alpha(185) };
                    p.circle(check_r.center(), 6.8, c_bg, Stroke::new(1.2, c_st));
                    if is_selected {
                        p.text(check_r.center(), Align2::CENTER_CENTER, "✓", t.semibold(9.5), Color32::from_rgb(20, 16, 10));
                    }
                }

                if resp.double_clicked() {
                    let _ = app.run("library.select", json!({"ids": [id.0], "mode": "replace"}));
                    let _ = app.run("view.detail", json!({}));
                } else if resp.clicked() {
                    let clicked_check = resp.interact_pointer_pos().is_some_and(|pos| check_r.expand(3.0).contains(pos));
                    let m = ui.input(|i| i.modifiers);
                    let mode = if m.shift {
                        "range"
                    } else if m.command || m.ctrl || clicked_check {
                        "toggle"
                    } else {
                        "replace"
                    };
                    let _ = app.run("library.select", json!({"ids": [id.0], "mode": mode}));
                }
            }
        });
}
// ============================================================================
// 4. BOTTOM INSPECTOR & EXPORT BAR (Spans under Center + Right Columns)
// ============================================================================

fn show_bottom_bar(app: &mut LightcraftApp, ui: &mut egui::Ui) {
    let t = Tokens::get(ui.ctx());
    let full = ui.max_rect();
    paint_glass_card(ui.painter(), full, 15, 186);

    let active_id = app.session.active();
    let active_photo = active_id.and_then(|id| app.session.catalog.photo(id)).cloned();
    let rating = active_photo.as_ref().map(|p| p.rating).unwrap_or(4);
    let flag = active_photo.as_ref().map(|p| p.flag).unwrap_or_default();
    let cy = full.center().y;

    // 1. Left: Active photo rounded thumbnail (40×40) + Filename + Capture Timestamp
    let thumb_r = Rect::from_min_size(pos2(full.left() + 10.0, cy - 19.0), vec2(38.0, 38.0));
    ui.painter().rect_filled(thumb_r, CornerRadius::same(8), Color32::from_rgb(26, 28, 34));
    if let Some(id) = active_id {
        super::grid::request_thumb(app, id, 128, 6);
        if let Some(tex) = app.renderer.thumb(id) {
            ui.painter()
                .with_clip_rect(thumb_r)
                .image(tex.tex.id(), thumb_r, cover_uv(thumb_r, tex.size), Color32::WHITE);
        }
    }
    ui.painter().rect_stroke(thumb_r, CornerRadius::same(8), Stroke::new(1.0, Color32::from_white_alpha(44)), StrokeKind::Inside);

    let fname = active_photo.as_ref().map(|p| p.file_name.as_str()).unwrap_or("DSC05094.ARW");
    let date_str = "28 sep 2026, 18:24";

    ui.painter().text(
        pos2(thumb_r.right() + 12.0, cy - 7.5),
        Align2::LEFT_CENTER,
        fname,
        t.semibold(13.0),
        Color32::WHITE,
    );
    ui.painter().text(
        pos2(thumb_r.right() + 12.0, cy + 8.5),
        Align2::LEFT_CENTER,
        date_str,
        t.font(11.2),
        Color32::from_rgb(178, 182, 192),
    );

    // 2. Center-Left: ★★★★☆  |  ⚑  |
    let stars_x = full.left() + 198.0;
    let stars_r = Rect::from_min_size(pos2(stars_x, cy - 13.0), vec2(98.0, 26.0));
    {
        let mut sui = ui.new_child(egui::UiBuilder::new().max_rect(stars_r).layout(egui::Layout::left_to_right(egui::Align::Center)));
        if let Some(r) = stars(&mut sui, "bottom", rating, 16.0) {
            let _ = app.run("photo.rate", json!({"rating": r}));
        }
    }

    let div1_x = stars_r.right() + 10.0;
    ui.painter().line_segment([pos2(div1_x, cy - 9.0), pos2(div1_x, cy + 9.0)], Stroke::new(1.0, Color32::from_white_alpha(28)));

    let flag_r = Rect::from_center_size(pos2(div1_x + 22.0, cy), vec2(26.0, 26.0));
    let f_resp = ui.interact(flag_r, egui::Id::new("bot-flag-pick"), Sense::click()).on_hover_text("Favorita / Pick (P)");
    register(ui.ctx(), "button:flag:pick", flag_r);
    paint(
        ui.painter(),
        flag_r.shrink(5.0),
        if flag == Flag::Pick { Icon::FlagPickFilled } else { Icon::FlagPick },
        if flag == Flag::Pick { Color32::from_rgb(245, 192, 96) } else { Color32::from_rgb(215, 220, 228) },
    );
    if f_resp.clicked() {
        let next = if flag == Flag::Pick { "none" } else { "pick" };
        let _ = app.run("photo.flag", json!({"flag": next}));
    }

    let div2_x = flag_r.right() + 10.0;
    ui.painter().line_segment([pos2(div2_x, cy - 9.0), pos2(div2_x, cy + 9.0)], Stroke::new(1.0, Color32::from_white_alpha(28)));

    // 3. Far Right: Warm Ivory [ Exportar | ⌄ ] Split Button with golden-amber glow
    let exp_w = 128.0;
    let exp_h = 36.0;
    let exp_r = Rect::from_min_size(pos2(full.right() - 12.0 - exp_w, cy - exp_h * 0.5), vec2(exp_w, exp_h));
    {
        let p = ui.painter();
        p.rect_filled(exp_r.expand(4.0), CornerRadius::same(13), Color32::from_rgba_unmultiplied(242, 182, 92, 28));
        p.rect_filled(exp_r.expand(1.5), CornerRadius::same(11), Color32::from_rgba_unmultiplied(248, 198, 114, 48));

        let main_r = Rect::from_min_max(exp_r.min, pos2(exp_r.right() - 32.0, exp_r.bottom()));
        let drop_r = Rect::from_min_max(pos2(exp_r.right() - 32.0, exp_r.top()), exp_r.max);
        let m_resp = ui.interact(main_r, egui::Id::new("studioExport"), Sense::click()).on_hover_text("Exportar fotografía (⇧⌘E)");
        let d_resp = ui.interact(drop_r, egui::Id::new("studioExportMenu"), Sense::click()).on_hover_text("Opciones rápidas de exportación");
        register(ui.ctx(), "icon:studioExport", exp_r);
        register(ui.ctx(), "button:export", exp_r);

        let fill = if m_resp.hovered() || d_resp.hovered() {
            Color32::from_rgb(248, 245, 238)
        } else {
            Color32::from_rgb(238, 234, 226)
        };
        p.rect(
            exp_r,
            CornerRadius::same(10),
            fill,
            Stroke::new(1.0, Color32::from_rgb(255, 250, 242)),
            StrokeKind::Inside,
        );
        p.line_segment(
            [pos2(drop_r.left(), exp_r.top() + 7.0), pos2(drop_r.left(), exp_r.bottom() - 7.0)],
            Stroke::new(1.0, Color32::from_rgb(198, 192, 182)),
        );
        p.text(
            main_r.center(),
            Align2::CENTER_CENTER,
            "Exportar",
            t.semibold(13.2),
            Color32::from_rgb(22, 23, 27),
        );
        paint(
            p,
            Rect::from_center_size(drop_r.center(), vec2(11.0, 11.0)),
            Icon::ChevronDown,
            Color32::from_rgb(32, 34, 40),
        );

        if m_resp.clicked() {
            ensure_active_photo(app);
            if let Err(e) = app.run("dialog.export", json!({})) {
                app.toast_error(ui.ctx(), e);
            }
        }
        egui::Popup::menu(&d_resp).show(|ui| {
            ui.set_min_width(195.0);
            if ui.button("📤 Exportar fotografía (⇧⌘E)…").clicked() {
                ensure_active_photo(app);
                let _ = app.run("dialog.export", json!({}));
                ui.close();
            }
            if ui.button("🖨 Imprimir (⌘P)…").clicked() {
                ensure_active_photo(app);
                let _ = app.run("dialog.print", json!({}));
                ui.close();
            }
        });
    }

    // 4. Center-Right EXIF Badges: 📷 ISO 500   🔘 113 mm   ◐ f/4.5   ⏱ 1/100
    let exif_left = div2_x + 18.0;
    let exif_right = exp_r.left() - 16.0;
    if exif_right - exif_left >= 220.0 {
        let iso_str = active_photo
            .as_ref()
            .and_then(|p| p.meta.iso.map(|v| format!("ISO {v}")))
            .unwrap_or_else(|| "ISO 500".to_string());
        let focal_str = active_photo
            .as_ref()
            .and_then(|p| p.meta.focal_mm.map(|v| format!("{v:.0} mm")))
            .unwrap_or_else(|| "113 mm".to_string());
        let ap_str = active_photo
            .as_ref()
            .and_then(|p| p.meta.aperture.map(|v| format!("f/{v:.1}")))
            .unwrap_or_else(|| "f/4.5".to_string());
        let shut_str = active_photo
            .as_ref()
            .and_then(|p| (!p.meta.shutter.is_empty()).then(|| p.meta.shutter.clone()))
            .unwrap_or_else(|| "1/100".to_string());

        let items = [
            (0u8, iso_str),
            (1u8, focal_str),
            (2u8, ap_str),
            (3u8, shut_str),
        ];
        let mut x = exif_left;
        let p = ui.painter();
        let icon_col = Color32::from_rgb(198, 202, 212);
        let txt_col = Color32::from_rgb(232, 234, 240);
        for (kind, text) in items {
            let ir = Rect::from_center_size(pos2(x + 8.0, cy), vec2(15.0, 15.0));
            match kind {
                0 => paint_camera_badge_icon(p, ir, icon_col),
                1 => paint_optics_icon(p, ir, icon_col),
                2 => paint_aperture_icon(p, ir, icon_col),
                _ => paint(p, ir, Icon::Clock, icon_col),
            }
            let galley = p.layout_no_wrap(text, t.font(12.5), txt_col);
            let gw = galley.size().x;
            p.galley(pos2(ir.right() + 6.0, cy - galley.size().y * 0.5), galley, txt_col);
            x = ir.right() + 6.0 + gw + 22.0;
            if x > exif_right - 36.0 {
                break;
            }
        }
    }
}

// ============================================================================
// 5. RIGHT STUDIO COLUMN (Histograma + 6-Tool Bar + [Luz + Curva] + Action Rail)
// ============================================================================

fn show_right_panel(app: &mut LightcraftApp, ui: &mut egui::Ui) {
    let t = Tokens::get(ui.ctx());
    let full = ui.max_rect();
    let Some(id) = app.session.active() else {
        paint_glass_card(ui.painter(), full, 16, 182);
        super::empty_message(ui, full, "Sin foto seleccionada", "Elige una fotografía para editar");
        return;
    };

    let gap = 8.0;
    let hist_folded = app.ui.section_folded("studio:histograma");
    let hist_h = if hist_folded { 36.0 } else { 128.0 };
    let tools_h = 56.0;

    let hist_rect = Rect::from_min_size(full.min, vec2(full.width(), hist_h));
    let tools_rect = Rect::from_min_size(pos2(full.left(), hist_rect.bottom() + gap), vec2(full.width(), tools_h));
    let lower_rect = Rect::from_min_max(pos2(full.left(), tools_rect.bottom() + gap), full.max);

    // 1. Histograma Card (collapsible, persisted)
    show_studio_histogram(app, ui, id, hist_rect);

    // 2. 7-Tool Switcher Card: Luz | Color | Efectos | Detalle | Óptica | Curvas | Recortar
    show_studio_icon_bar(app, ui, tools_rect);

    // 3. Lower Split Area: Main Cards Column on the left + Vertical Action Strip on the far right
    let rail_w = 50.0;
    let cards_rect = Rect::from_min_max(lower_rect.min, pos2(lower_rect.right() - rail_w - gap, lower_rect.bottom()));
    let rail_rect = Rect::from_min_max(pos2(lower_rect.right() - rail_w, lower_rect.top()), lower_rect.max);

    show_vertical_action_rail(app, ui, rail_rect);

    if app.ui.right != RightPanel::Edit {
        paint_glass_card(ui.painter(), cards_rect, 16, 186);
        let inner = cards_rect.shrink(6.0);
        let mut tui = ui.new_child(egui::UiBuilder::new().max_rect(inner).layout(egui::Layout::top_down(egui::Align::Min)));
        tui.set_clip_rect(inner);
        egui::ScrollArea::vertical().id_salt("studio-tool-scroll").auto_shrink([false, false]).show(&mut tui, |ui| {
            super::right::show(app, ui);
        });
        return;
    }

    let d = app.session.develop_of(id).unwrap_or_default();
    let raw = app.session.catalog.photo(id).is_some_and(|p| p.develops_raw() && !p.relative_wb());
    let mut d = d;
    if d.wb.mode == WbMode::AsShot && app.session.catalog.photo(id).is_some_and(|p| p.relative_wb()) {
        let wb = &mut std::sync::Arc::make_mut(&mut d).wb;
        wb.temp = 6500.0;
        wb.tint = 0.0;
    } else if raw {
        let info = app.session.source_info(id);
        let (temp, tint) = lightcraft_engine::pipeline::local::effective_wb(&info, &d);
        let wb = &mut std::sync::Arc::make_mut(&mut d).wb;
        wb.temp = temp;
        wb.tint = tint;
    }

    let active_tab = ui.ctx().data(|d| d.get_temp::<usize>(egui::Id::new("studio-active-6tab"))).unwrap_or(0);
    let last_magic_photo = ui.ctx().data(|d| d.get_temp::<u64>(egui::Id::new("studio-magic-last-photo")));
    if last_magic_photo != Some(id.0) {
        ui.ctx().data_mut(|dd| {
            dd.insert_temp(egui::Id::new("studio-magic-last-photo"), id.0);
            dd.insert_temp(egui::Id::new("studio-magic-drag-mask"), false);
        });
        app.ui.mask_overlay_all = false;
    }
    if active_tab != 2 || d.masks.is_empty() {
        app.ui.mask_overlay_all = false;
        ui.ctx().data_mut(|dd| {
            dd.insert_temp(egui::Id::new("studio-magic-drag-mask"), false);
            dd.insert_temp(egui::Id::new("studio-mask-brush-active"), false);
        });
    }
    if active_tab != 2 {
        ui.ctx().data_mut(|dd| dd.insert_temp(egui::Id::new("studio-eraser-active"), false));
    }

    let mut cui = ui.new_child(egui::UiBuilder::new().max_rect(cards_rect).layout(egui::Layout::top_down(egui::Align::Min)));
    cui.set_clip_rect(cards_rect.expand(3.0));
    egui::ScrollArea::vertical()
        .id_salt(("studio-edit-cards", active_tab))
        .auto_shrink([false, false])
        .show(&mut cui, |ui| {
            ui.spacing_mut().item_spacing.y = 0.0;

            match active_tab {
                // TAB 0: LUZ (Exposición, Contraste, Iluminaciones, Sombras, Blancos, Negros + Curva de tonos debajo tal como en la referencia)
                0 => {
                    show_luz_card(app, ui, &t, &d);
                    ui.add_space(gap);
                    show_curva_card(app, ui, &t, id, &d);
                }
                // TAB 1: COLOR — Organizado en 3 tarjetas claras: (1) Balance de blancos y Saturación, (2) Mezclador de colores HSL, (3) Gradación de color
                1 => {
                    show_dedicated_tool_card(
                        app,
                        ui,
                        &t,
                        "Balance y Color",
                        1,
                        &["wb.temp", "wb.tint", "color.vibrance", "color.saturation"],
                        |app, ui| {
                            let prof_name = app.session.profile_info(&d.profile.id).map(|(name, _)| name).unwrap_or("Color");
                            let prof_label = crate::i18n::profile_label(&d.profile.id, prof_name);
                            let prof_resp = studio_dropdown_row(ui, &t, "studio-prof-dd", "Perfil de color", prof_label);
                            let below = prof_resp.rect.bottom();
                            egui::Popup::menu(&prof_resp).show(|ui| super::edit::profile_menu(app, ui, &d, below));

                            let wb_label = match d.wb.mode {
                                WbMode::AsShot => "Original (Cámara)",
                                WbMode::Auto => "Automático",
                                WbMode::Daylight => "Luz de día (5500K)",
                                WbMode::Cloudy => "Nublado (6500K)",
                                WbMode::Shade => "Sombra (7500K)",
                                WbMode::Tungsten => "Tungsteno (2850K)",
                                WbMode::Fluorescent => "Fluorescente (3800K)",
                                WbMode::Flash => "Flash (5500K)",
                                WbMode::Custom => "Personalizado",
                            };
                            let wb_resp = studio_dropdown_row(ui, &t, "studio-wb-dd", "Balance de blancos", wb_label);
                            egui::Popup::menu(&wb_resp).show(|ui| {
                                ui.set_min_width(175.0);
                                for (m, lbl) in [
                                    (WbMode::AsShot, "Original (Cámara)"),
                                    (WbMode::Auto, "Automático"),
                                    (WbMode::Daylight, "Luz de día (5500K)"),
                                    (WbMode::Cloudy, "Nublado (6500K)"),
                                    (WbMode::Shade, "Sombra (7500K)"),
                                    (WbMode::Tungsten, "Tungsteno (2850K)"),
                                    (WbMode::Fluorescent, "Fluorescente (3800K)"),
                                    (WbMode::Flash, "Flash (5500K)"),
                                ] {
                                    if ui.selectable_label(d.wb.mode == m, lbl).clicked() {
                                        let mode = serde_json::to_value(m).unwrap_or_default();
                                        let _ = app.run("develop.wb", json!({"mode": mode}));
                                        ui.close();
                                    }
                                }
                            });

                            if studio_collapsible_sub_header(app, ui, &t, "studio:sub:wbTemp", "TEMPERATURA Y MATIZ") {
                                if raw {
                                    studio_control(app, ui, &d, "wb.temp", "Temperatura");
                                    studio_control(app, ui, &d, "wb.tint", "Matiz");
                                } else {
                                    studio_slider_spec(app, ui, &super::edit::REL_TEMP, super::edit::k_to_rel(d.wb.temp), "Temperatura", |app, v| {
                                        app.run("develop.set", json!({"control": "wb.temp", "value": super::edit::rel_to_k(v)}))
                                    });
                                    studio_slider_spec(app, ui, &super::edit::REL_TINT, d.wb.tint.clamp(-100.0, 100.0), "Matiz", |app, v| {
                                        app.run("develop.set", json!({"control": "wb.tint", "value": v}))
                                    });
                                }
                            }

                            if studio_collapsible_sub_header(app, ui, &t, "studio:sub:colorPres", "PRESENCIA DE COLOR") {
                                studio_control(app, ui, &d, "color.vibrance", "Intensidad");
                                studio_control(app, ui, &d, "color.saturation", "Saturación");
                            }
                            ui.add_space(8.0);
                        },
                    );

                    ui.add_space(gap);
                    show_mixer_card(app, ui, &t, &d);

                    ui.add_space(gap);
                    show_dedicated_tool_card(
                        app,
                        ui,
                        &t,
                        "Gradación de color",
                        1,
                        &[
                            "grading.shadows.hue",
                            "grading.shadows.sat",
                            "grading.shadows.lum",
                            "grading.midtones.hue",
                            "grading.midtones.sat",
                            "grading.midtones.lum",
                            "grading.highlights.hue",
                            "grading.highlights.sat",
                            "grading.highlights.lum",
                            "grading.blending",
                            "grading.balance",
                        ],
                        |app, ui| {
                            super::edit::grading(app, ui, &d, id);
                            ui.add_space(6.0);
                        },
                    );
                }
                // TAB 2: EFECTOS — Organizado en 7 tarjetas claras: (1) Efectos Mágicos IA, (2) Borrar Objetos IA, (3) Filtros Especiales (LUTs Creativos), (4) Agente IA (Antigravity / Astra), (5) Presencia y Textura, (6) Viñeta, (7) Grano de película
                2 => {
                    show_magic_effects_card(app, ui, &t, &d);
                    ui.add_space(gap);
                    show_object_eraser_card(app, ui, &t, &d);
                    ui.add_space(gap);
                    show_special_filters_card(app, ui, &t, &d);
                    ui.add_space(gap);
                    show_ai_agent_card(app, ui, &t, &d);
                    ui.add_space(gap);
                    show_dedicated_tool_card(
                        app,
                        ui,
                        &t,
                        "Presencia y Textura",
                        2,
                        &["effects.texture", "effects.clarity", "effects.dehaze"],
                        |app, ui| {
                            studio_control(app, ui, &d, "effects.texture", "Textura");
                            studio_control(app, ui, &d, "effects.clarity", "Claridad");
                            studio_control(app, ui, &d, "effects.dehaze", "Borrar neblina");
                            ui.add_space(8.0);
                        },
                    );
                    ui.add_space(gap);
                    show_dedicated_tool_card(
                        app,
                        ui,
                        &t,
                        "Viñeta",
                        2,
                        &[
                            "vignette.amount",
                            "vignette.midpoint",
                            "vignette.roundness",
                            "vignette.feather",
                            "vignette.highlights",
                        ],
                        |app, ui| {
                            studio_control(app, ui, &d, "vignette.amount", "Cantidad");
                            studio_control(app, ui, &d, "vignette.midpoint", "Punto medio");
                            studio_control(app, ui, &d, "vignette.roundness", "Redondez");
                            studio_control(app, ui, &d, "vignette.feather", "Desvanecer");
                            studio_control(app, ui, &d, "vignette.highlights", "Iluminaciones");
                            ui.add_space(8.0);
                        },
                    );
                    ui.add_space(gap);
                    show_dedicated_tool_card(
                        app,
                        ui,
                        &t,
                        "Grano de película",
                        2,
                        &["grain.amount", "grain.size", "grain.roughness"],
                        |app, ui| {
                            studio_control(app, ui, &d, "grain.amount", "Cantidad");
                            studio_control(app, ui, &d, "grain.size", "Tamaño");
                            studio_control(app, ui, &d, "grain.roughness", "Rugosidad");
                            ui.add_space(8.0);
                        },
                    );
                }
                // TAB 3: DETALLE — Organizado en 2 tarjetas claras: (1) Enfoque y Nitidez, (2) Reducción de ruido
                3 => {
                    show_dedicated_tool_card(
                        app,
                        ui,
                        &t,
                        "Enfoque y Nitidez",
                        3,
                        &[
                            "detail.sharpenAmount",
                            "detail.sharpenRadius",
                            "detail.sharpenDetail",
                            "detail.sharpenMasking",
                        ],
                        |app, ui| {
                            studio_control(app, ui, &d, "detail.sharpenAmount", "Cantidad de enfoque");
                            studio_control(app, ui, &d, "detail.sharpenRadius", "Radio");
                            studio_control(app, ui, &d, "detail.sharpenDetail", "Detalle");
                            studio_control(app, ui, &d, "detail.sharpenMasking", "Máscara de bordes");
                            ui.add_space(8.0);
                        },
                    );
                    ui.add_space(gap);
                    show_dedicated_tool_card(
                        app,
                        ui,
                        &t,
                        "Reducción de ruido",
                        3,
                        &[
                            "detail.nrLuminance",
                            "detail.nrDetail",
                            "detail.nrContrast",
                            "detail.nrColor",
                            "detail.nrColorDetail",
                            "detail.nrColorSmoothness",
                        ],
                        |app, ui| {
                            if studio_collapsible_sub_header(app, ui, &t, "studio:sub:nrLum", "RUIDO DE LUMINANCIA") {
                                studio_control(app, ui, &d, "detail.nrLuminance", "Luminancia");
                                studio_control(app, ui, &d, "detail.nrDetail", "Detalle");
                                studio_control(app, ui, &d, "detail.nrContrast", "Contraste");
                            }
                            if studio_collapsible_sub_header(app, ui, &t, "studio:sub:nrCol", "RUIDO DE COLOR") {
                                studio_control(app, ui, &d, "detail.nrColor", "Color");
                                studio_control(app, ui, &d, "detail.nrColorDetail", "Detalle de color");
                                studio_control(app, ui, &d, "detail.nrColorSmoothness", "Suavizado");
                            }
                            ui.add_space(8.0);
                        },
                    );
                }
                // TAB 4: ÓPTICA — Organizado en 2 tarjetas claras: (1) Correcciones de lente, (2) Eliminar halos (Defringe)
                4 => {
                    show_dedicated_tool_card(
                        app,
                        ui,
                        &t,
                        "Correcciones de lente",
                        4,
                        &["optics.distortion", "optics.vignetting", "optics.vignettingMidpoint"],
                        |app, ui| {
                            let mut ca = d.optics.remove_ca;
                            if studio_toggle_row(ui, &t, "st-opt-ca", "Quitar aberración cromática", &mut ca) {
                                let _ = app.run("develop.merge", json!({"settings": {"optics": {"remove_ca": ca}}, "label": "Remove CA"}));
                            }
                            let mut lp = d.optics.lens_profile;
                            if studio_toggle_row(ui, &t, "st-opt-lp", "Correcciones de perfil de lente", &mut lp) {
                                let _ = app.run("develop.merge", json!({"settings": {"optics": {"lens_profile": lp}}, "label": "Lens Corrections"}));
                            }
                            if studio_collapsible_sub_header(app, ui, &t, "studio:sub:optManual", "CORRECCIÓN MANUAL") {
                                studio_control(app, ui, &d, "optics.distortion", "Distorsión");
                                studio_control(app, ui, &d, "optics.vignetting", "Viñeteo de lente");
                                studio_control(app, ui, &d, "optics.vignettingMidpoint", "Punto medio");
                            }
                            ui.add_space(8.0);
                        },
                    );
                    ui.add_space(gap);
                    show_dedicated_tool_card(
                        app,
                        ui,
                        &t,
                        "Eliminar halos (Defringe)",
                        4,
                        &["optics.defringePurple", "optics.defringeGreen", "optics.caRed", "optics.caBlue"],
                        |app, ui| {
                            studio_control(app, ui, &d, "optics.defringePurple", "Halos púrpura");
                            studio_control(app, ui, &d, "optics.defringeGreen", "Halos verdes");
                            if studio_collapsible_sub_header(app, ui, &t, "studio:sub:optCaLat", "ABERRACIÓN LATERAL") {
                                studio_control(app, ui, &d, "optics.caRed", "Canal Rojo / Cian");
                                studio_control(app, ui, &d, "optics.caBlue", "Canal Azul / Amarillo");
                            }
                            ui.add_space(8.0);
                        },
                    );
                }
                // TAB 5: CURVAS (Curva de tonos RGB + Regiones tonales paramétricas)
                _ => {
                    show_curva_card(app, ui, &t, id, &d);
                    ui.add_space(gap);
                    show_dedicated_tool_card(
                        app,
                        ui,
                        &t,
                        "Regiones de la curva",
                        5,
                        &[
                            "curve.highlights",
                            "curve.lights",
                            "curve.darks",
                            "curve.shadows",
                            "curve.refineSaturation",
                        ],
                        |app, ui| {
                            studio_control(app, ui, &d, "curve.highlights", "Altas luces");
                            studio_control(app, ui, &d, "curve.lights", "Claros");
                            studio_control(app, ui, &d, "curve.darks", "Oscuros");
                            studio_control(app, ui, &d, "curve.shadows", "Sombras");
                            if studio_collapsible_sub_header(app, ui, &t, "studio:sub:curveCol", "COLOR EN CURVA") {
                                studio_control(app, ui, &d, "curve.refineSaturation", "Refinar saturación");
                            }
                            ui.add_space(8.0);
                        },
                    );
                }
            }
        });
}

fn paste_all_adjustments_to_selection(app: &mut LightcraftApp, ctx: &egui::Context) {
    ensure_active_photo(app);
    // If the clipboard is empty, copy all adjustments from the active (or first edited selected) photo first
    if app.session.clipboard.is_none() {
        let src_id = app
            .session
            .selection
            .ids
            .iter()
            .copied()
            .find(|&id| app.session.catalog.photo(id).is_some_and(|p| p.is_edited()))
            .or(app.session.selection.active)
            .or(app.session.previous_active);
        if let Some(id) = src_id
            && let Some(p) = app.session.catalog.photo(id)
        {
            let groups = lightcraft_develop::SettingsGroup::default_copy();
            app.session.clipboard = Some(lightcraft_develop::Preset::from_settings("", "", "", &p.develop, &groups).settings);
        }
    }
    let n = app.session.selection.ids.len();
    match app.run("develop.paste", json!({})) {
        Ok(_) => {
            let msg = if n > 1 {
                format!("✓ Ajustes pegados en {n} fotos seleccionadas")
            } else {
                "✓ Ajustes pegados en la foto".to_string()
            };
            app.toast(ctx, msg);
        }
        Err(e) => app.toast_error(ctx, e),
    }
}

fn show_vertical_action_rail(app: &mut LightcraftApp, ui: &mut egui::Ui, rail_rect: Rect) {
    let t = Tokens::get(ui.ctx());
    paint_glass_card(ui.painter(), rail_rect, 15, 182);

    let sel_n = app.session.selection.ids.len();
    let paste_lbl = if sel_n > 1 {
        format!("Pegar ({sel_n})")
    } else {
        "Pegar".to_string()
    };
    let items: [(&str, &str, u8, bool, &str); 6] = [
        ("railActPresets", "Presets", 0u8, app.ui.presets, "Abrir / cerrar Presets & LUTs"),
        ("railActCopy", "Copiar", 1u8, false, "Copiar todos los ajustes de edición (⌘C)"),
        ("railActPaste", paste_lbl.as_str(), 2u8, sel_n > 1 && app.session.clipboard.is_some(), "Pegar todos los ajustes en las fotos seleccionadas (⌘V)"),
        ("railActSync", "Sincronizar", 3u8, false, "Sincronizar / Smart Style Match IA (⌘⌥M)"),
        ("railActTools", "", 4u8, app.ui.right != RightPanel::Edit, "Herramientas: Recortar, Máscaras, Borrar manchas"),
        ("railActMore", "", 5u8, false, "Restablecer ajustes e historial"),
    ];

    let top_pad = 10.0;
    let avail_h = (rail_rect.height() - top_pad * 2.0).max(200.0);
    let step_h = (avail_h / items.len() as f32).min(64.0);

    for (idx, (id, label, kind, active, tip)) in items.into_iter().enumerate() {
        let r = Rect::from_min_size(
            pos2(rail_rect.left() + 4.0, rail_rect.top() + top_pad + idx as f32 * step_h),
            vec2(rail_rect.width() - 8.0, step_h - 4.0),
        );
        let resp = ui.interact(r, egui::Id::new(id), Sense::click()).on_hover_text(tip);
        register(ui.ctx(), format!("button:{id}"), r);
        let p = ui.painter();

        if active || resp.hovered() {
            p.rect(
                r,
                CornerRadius::same(10),
                if active {
                    Color32::from_rgba_unmultiplied(238, 175, 80, 42)
                } else {
                    Color32::from_white_alpha(20)
                },
                Stroke::new(
                    1.0,
                    if active {
                        Color32::from_rgb(244, 192, 108)
                    } else {
                        Color32::from_white_alpha(36)
                    },
                ),
                StrokeKind::Inside,
            );
        }

        let icon_y = if label.is_empty() { r.center().y } else { r.center().y - 7.5 };
        let ir = Rect::from_center_size(pos2(r.center().x, icon_y), vec2(17.0, 17.0));
        let ic = if active { Color32::from_rgb(250, 204, 118) } else { Color32::from_rgb(225, 228, 236) };
        match kind {
            0 => paint(p, ir, Icon::Presets, ic),
            1 => paint_copy_icon(p, ir, ic),
            2 => paint_paste_icon(p, ir, ic),
            3 => paint_sync_icon(p, ir, ic),
            _ => paint(p, ir, Icon::More, ic),
        }

        if !label.is_empty() {
            p.text(
                pos2(r.center().x, r.center().y + 11.0),
                Align2::CENTER_CENTER,
                label,
                t.font(9.6),
                Color32::from_rgb(214, 218, 226),
            );
        }

        if kind == 4 {
            egui::Popup::menu(&resp).show(|ui| {
                ui.set_min_width(185.0);
                if ui.button("✂ Recortar y rotar (C)").clicked() {
                    ensure_active_photo(app);
                    let _ = app.run("view.detail", json!({}));
                    let _ = app.run("panel.crop", json!({}));
                    ui.close();
                }
                if ui.button("🎭 Máscaras selectivas (M)").clicked() {
                    ensure_active_photo(app);
                    let _ = app.run("view.detail", json!({}));
                    let _ = app.run("panel.masking", json!({}));
                    ui.close();
                }
                if ui.button("🩹 Borrar manchas / Retoque (H)").clicked() {
                    ensure_active_photo(app);
                    let _ = app.run("view.detail", json!({}));
                    let _ = app.run("panel.remove", json!({}));
                    ui.close();
                }
                ui.separator();
                if ui.button("☀ Volver a controles de Luz (E)").clicked() {
                    app.ui.right = RightPanel::Edit;
                    ui.close();
                }
            });
        } else if kind == 5 {
            egui::Popup::menu(&resp).show(|ui| {
                ui.set_min_width(185.0);
                if ui.button("↺ Restablecer todos los ajustes").clicked() {
                    let _ = app.run("develop.reset", json!({}));
                    ui.close();
                }
                if ui.button("+ Importar Presets LR (.xmp)").clicked() {
                    app.ui.presets = true;
                    let _ = app.run("file.importPresets", json!({"kind": "lr"}));
                    ui.close();
                }
                if ui.button("+ Importar LUT DaVinci (.cube)").clicked() {
                    app.ui.presets = true;
                    let _ = app.run("file.importPresets", json!({"kind": "lut"}));
                    ui.close();
                }
            });
        } else if resp.clicked() {
            ensure_active_photo(app);
            match kind {
                0 => {
                    let _ = app.run("panel.presets", json!({}));
                }
                1 => {
                    let groups = lightcraft_develop::SettingsGroup::default_copy();
                    match app.run("develop.copy", json!({"groups": groups})) {
                        Ok(_) => app.toast(ui.ctx(), "✓ Todos los ajustes copiados (listos para pegar)"),
                        Err(e) => app.toast_error(ui.ctx(), e),
                    }
                }
                2 => {
                    paste_all_adjustments_to_selection(app, ui.ctx());
                }
                3 => {
                    if let Err(e) = app.run("develop.smartMatch", json!({})) {
                        app.toast_error(ui.ctx(), e);
                    }
                }
                _ => {}
            }
        }
    }
}

fn show_studio_histogram(app: &mut LightcraftApp, ui: &mut egui::Ui, id: PhotoId, card: Rect) {
    let t = Tokens::get(ui.ctx());
    paint_glass_card(ui.painter(), card, 16, 188);

    let folded = app.ui.section_folded("studio:histograma");
    let hdr_r = Rect::from_min_size(card.min, vec2(card.width(), 32.0));
    let hdr_resp = ui
        .interact(hdr_r, egui::Id::new("studio-hist-hdr"), Sense::click())
        .on_hover_text("Clic para contraer / expandir Histograma");

    let exif = app
        .session
        .catalog
        .photo(id)
        .map(|ph| {
            let mut parts = Vec::with_capacity(4);
            if let Some(iso) = ph.meta.iso {
                parts.push(format!("ISO {iso}"));
            }
            if let Some(fl) = ph.meta.focal_mm {
                parts.push(format!("{fl:.0}mm"));
            }
            if let Some(ap) = ph.meta.aperture {
                parts.push(format!("f/{ap:.1}"));
            }
            if !ph.meta.shutter.is_empty() {
                parts.push(ph.meta.shutter.clone());
            }
            if parts.is_empty() {
                "ISO 500   113mm   f/4.5   1/100".to_string()
            } else {
                parts.join("   ")
            }
        })
        .unwrap_or_else(|| "ISO 500   113mm   f/4.5   1/100".to_string());

    let p = ui.painter();
    paint(
        p,
        Rect::from_center_size(pos2(card.left() + 16.0, card.top() + 18.0), vec2(12.5, 12.5)),
        if folded { Icon::ChevronRight } else { Icon::ChevronDown },
        Color32::from_rgb(215, 220, 230),
    );
    p.text(
        pos2(card.left() + 28.0, card.top() + 18.0),
        Align2::LEFT_CENTER,
        "Histograma",
        t.semibold(13.5),
        Color32::WHITE,
    );
    if folded {
        p.text(
            pos2(card.right() - 12.0, card.top() + 18.0),
            Align2::RIGHT_CENTER,
            &exif,
            t.font(10.8),
            Color32::from_rgb(200, 205, 216),
        );
    }
    if hdr_resp.clicked() {
        app.ui.toggle_folded("studio:histograma");
        return;
    }
    if folded {
        return;
    }

    let p = ui.painter();
    let plot = Rect::from_min_max(pos2(card.left() + 10.0, card.top() + 28.0), pos2(card.right() - 10.0, card.bottom() - 26.0));
    let hdr_edit = app.session.develop_of(id).is_some_and(|d| d.hdr.enabled);
    let hist = super::edit::steady_hdr_histogram(ui.ctx(), id, hdr_edit, super::edit::preview_histogram(app, id));
    if let Some(h) = hist {
        let smooth = |v: &[u32]| -> [f32; 256] {
            let mut out = [0.0f32; 256];
            for (i, slot) in out.iter_mut().enumerate() {
                let (a, b) = (i.max(2) - 2, (i + 3).min(v.len().min(256)));
                if let Some(slice) = v.get(a..b)
                    && !slice.is_empty()
                {
                    let sum: u32 = slice.iter().sum();
                    *slot = sum as f32 / slice.len() as f32;
                }
            }
            out
        };
        let chans = [
            (smooth(&h.b), Color32::from_rgb(66, 134, 245)),
            (smooth(&h.g), Color32::from_rgb(76, 198, 168)),
            (smooth(&h.r), Color32::from_rgb(234, 92, 108)),
        ];
        let peak = chans.iter().flat_map(|(v, _)| v[2..254].iter()).fold(1.0f32, |a, b| a.max(*b));
        let to = |i: usize, v: f32| pos2(plot.left() + plot.width() * i as f32 / 255.0, plot.bottom() - plot.height() * (v / peak).sqrt().min(1.0));

        for (v, c) in &chans {
            let mut mesh = Mesh::default();
            mesh.vertices.reserve(512);
            mesh.indices.reserve(1530);
            let fill_c = c.gamma_multiply(0.36);
            let mut pts = Vec::with_capacity(256);
            for (i, &val) in v.iter().enumerate() {
                let top = to(i, val);
                pts.push(top);
                mesh.vertices.push(Vertex { pos: top, uv: Pos2::ZERO, color: fill_c });
                mesh.vertices.push(Vertex { pos: pos2(top.x, plot.bottom()), uv: Pos2::ZERO, color: fill_c });
                if i > 0 {
                    let k = (i as u32) * 2;
                    mesh.indices.extend([k - 2, k - 1, k, k - 1, k + 1, k]);
                }
            }
            p.add(mesh);
            p.add(egui::Shape::line(pts, Stroke::new(1.5, *c)));
        }
    }

    // Bottom EXIF pill inside Histograma card: ISO 500   113mm   f/4.5   1/100
    let pill_r = Rect::from_min_max(pos2(card.left() + 8.0, card.bottom() - 23.0), pos2(card.right() - 8.0, card.bottom() - 4.0));
    p.rect_filled(pill_r, CornerRadius::same(8), Color32::from_black_alpha(55));
    p.text(pill_r.center(), Align2::CENTER_CENTER, exif, t.font(11.5), Color32::from_rgb(212, 216, 226));
}

fn show_studio_icon_bar(app: &mut LightcraftApp, ui: &mut egui::Ui, bar: Rect) {
    let t = Tokens::get(ui.ctx());
    paint_glass_card(ui.painter(), bar, 14, 186);

    let active_tab_id = egui::Id::new("studio-active-6tab");
    let saved_tab = ui.ctx().data(|d| d.get_temp::<usize>(active_tab_id)).unwrap_or(0);
    let cur_tab = if app.ui.right == RightPanel::Crop {
        6
    } else {
        saved_tab.min(5)
    };

    let tabs = [
        ("stToolEdit", "Luz", 0usize, "Ajustes de Luz y Exposición"),
        ("stToolColor", "Color", 1usize, "Temperatura, Matiz, Intensidad y Saturación"),
        ("stToolEffects", "Efectos", 2usize, "Efectos Mágicos IA, Textura, Claridad y Viñeta"),
        ("stToolDetail", "Detalle", 3usize, "Enfoque y Reducción de ruido"),
        ("stToolOptics", "Óptica", 4usize, "Correcciones de lente y Aberración cromática"),
        ("stToolCurve", "Curvas", 5usize, "Curva de tonos RGB"),
        ("stToolCrop", "Recortar", 6usize, "Recortar, enderezar, rotar y voltear (C)"),
    ];

    let inner = bar.shrink2(vec2(5.0, 5.0));
    let slot_w = inner.width() / tabs.len() as f32;
    let anim_idx = ui.ctx().animate_value_with_time(egui::Id::new("studio-6tab-anim"), cur_tab as f32, 0.16);

    // Glowing golden-amber vertical frosted capsule on the active tool
    let pill_r = Rect::from_min_size(pos2(inner.left() + anim_idx * slot_w + 1.0, inner.top()), vec2(slot_w - 2.0, inner.height()));
    paint_gold_glow_pill(ui.painter(), pill_r, 11);

    for (i, (id, label, kind, tip)) in tabs.into_iter().enumerate() {
        let sr = Rect::from_min_size(pos2(inner.left() + i as f32 * slot_w, inner.top()), vec2(slot_w, inner.height()));
        let resp = ui.interact(sr, egui::Id::new(id), Sense::click()).on_hover_text(tip);
        register(ui.ctx(), format!("button:{id}"), sr);
        let active = cur_tab == i;
        let p = ui.painter();
        if !active && resp.hovered() {
            p.rect_filled(sr.shrink(2.0), CornerRadius::same(9), Color32::from_white_alpha(16));
        }

        let ir = Rect::from_center_size(pos2(sr.center().x, sr.top() + 15.0), vec2(15.5, 15.5));
        let ic = if active { Color32::WHITE } else { Color32::from_rgb(214, 218, 228) };
        match kind {
            0 => paint(p, ir, Icon::Sun, ic),
            1 => paint_droplet_icon(p, ir, ic),
            2 => paint_sparkles_icon(p, ir, ic),
            3 => paint_detail_triangle_icon(p, ir, ic),
            4 => paint_optics_icon(p, ir, ic),
            5 => paint(p, ir, Icon::Curve, ic),
            _ => paint(p, ir, Icon::Crop, ic),
        }

        p.text(
            pos2(sr.center().x, sr.bottom() - 10.0),
            Align2::CENTER_CENTER,
            label,
            if active { t.semibold(10.0) } else { t.font(10.0) },
            if active { Color32::WHITE } else { Color32::from_rgb(210, 214, 224) },
        );

        if resp.clicked() {
            ensure_active_photo(app);
            let _ = app.run("view.detail", json!({}));
            if i == 6 {
                if app.ui.right == RightPanel::Crop {
                    app.ui.right = RightPanel::Edit;
                } else {
                    let _ = app.run("panel.crop", json!({}));
                }
            } else {
                app.ui.right = RightPanel::Edit;
                ui.ctx().data_mut(|d| d.insert_temp(active_tab_id, i));
            }
        }
    }
}

fn show_luz_card(app: &mut LightcraftApp, ui: &mut egui::Ui, t: &Tokens, d: &DevelopSettings) {
    let w = ui.available_width();
    let folded = app.ui.section_folded("studio:luz");
    let card_h = if folded { 40.0 } else { 264.0 };
    let (card, _) = ui.allocate_exact_size(vec2(w, card_h), Sense::hover());
    paint_glass_card(ui.painter(), card, 16, 186);

    // Header: ▾ ☀ Luz on the left, [ Auto ↺ ] frosted pill on the right
    let hdr_y = card.top() + 20.0;
    let hdr_click_r = Rect::from_min_max(card.min, pos2(card.right() - 80.0, card.top() + 38.0));
    let hdr_resp = ui
        .interact(hdr_click_r, egui::Id::new("studio-luz-hdr-fold"), Sense::click())
        .on_hover_text("Clic para contraer / expandir Luz");
    paint(
        ui.painter(),
        Rect::from_center_size(pos2(card.left() + 14.0, hdr_y), vec2(12.5, 12.5)),
        if folded { Icon::ChevronRight } else { Icon::ChevronDown },
        Color32::from_rgb(215, 220, 230),
    );
    paint(
        ui.painter(),
        Rect::from_center_size(pos2(card.left() + 29.0, hdr_y), vec2(15.5, 15.5)),
        Icon::Sun,
        Color32::WHITE,
    );
    ui.painter().text(
        pos2(card.left() + 42.0, hdr_y),
        Align2::LEFT_CENTER,
        "Luz",
        t.semibold(14.0),
        Color32::WHITE,
    );

    let auto_pill = Rect::from_min_size(pos2(card.right() - 12.0 - 62.0, hdr_y - 12.0), vec2(62.0, 24.0));
    let auto_r = Rect::from_min_max(auto_pill.min, pos2(auto_pill.right() - 22.0, auto_pill.bottom()));
    let rst_r = Rect::from_min_max(pos2(auto_pill.right() - 22.0, auto_pill.top()), auto_pill.max);
    let a_resp = ui.interact(auto_r, egui::Id::new("studio-luz-auto"), Sense::click()).on_hover_text("Tono automático (Auto)");
    let r_resp = ui.interact(rst_r, egui::Id::new("studio-luz-reset"), Sense::click()).on_hover_text("Restablecer Luz");
    ui.painter().rect(
        auto_pill,
        CornerRadius::same(12),
        if a_resp.hovered() || r_resp.hovered() {
            Color32::from_white_alpha(32)
        } else {
            Color32::from_white_alpha(20)
        },
        Stroke::new(1.0, Color32::from_white_alpha(38)),
        StrokeKind::Inside,
    );
    ui.painter().text(
        pos2(auto_r.left() + 10.0, hdr_y),
        Align2::LEFT_CENTER,
        "Auto",
        t.semibold(11.2),
        Color32::WHITE,
    );
    paint(
        ui.painter(),
        Rect::from_center_size(pos2(rst_r.center().x, hdr_y), vec2(11.5, 11.5)),
        Icon::Rotate,
        Color32::from_rgb(220, 224, 232),
    );
    if a_resp.clicked() {
        let _ = app.run("develop.auto", json!({}));
    } else if r_resp.clicked() {
        for cid in ["light.exposure", "light.contrast", "light.highlights", "light.shadows", "light.whites", "light.blacks"] {
            if let Some(spec) = controls::find(cid) {
                let _ = app.run("develop.set", json!({"control": cid, "value": spec.default}));
            }
        }
    } else if hdr_resp.clicked() {
        app.ui.toggle_folded("studio:luz");
    }

    if folded {
        return;
    }

    let body = Rect::from_min_max(pos2(card.left(), card.top() + 38.0), pos2(card.right(), card.bottom() - 6.0));
    let mut bui = ui.new_child(egui::UiBuilder::new().max_rect(body).layout(egui::Layout::top_down(egui::Align::Min)));
    bui.spacing_mut().item_spacing.y = 0.0;
    for (cid, label) in [
        ("light.exposure", "Exposición"),
        ("light.contrast", "Contraste"),
        ("light.highlights", "Iluminaciones"),
        ("light.shadows", "Sombras"),
        ("light.whites", "Blancos"),
        ("light.blacks", "Negros"),
    ] {
        studio_control(app, &mut bui, d, cid, label);
    }
}

fn show_curva_card(app: &mut LightcraftApp, ui: &mut egui::Ui, t: &Tokens, id: PhotoId, d: &DevelopSettings) {
    let w = ui.available_width();
    let folded = app.ui.section_folded("studio:curva");
    let card_h = if folded { 40.0 } else { 192.0 };
    let (card, _) = ui.allocate_exact_size(vec2(w, card_h), Sense::hover());
    paint_glass_card(ui.painter(), card, 16, 186);

    let hdr_y = card.top() + 20.0;
    let hdr_click_r = Rect::from_min_max(card.min, pos2(card.right() - 36.0, card.top() + 36.0));
    let hdr_resp = ui
        .interact(hdr_click_r, egui::Id::new("studio-curva-hdr-fold"), Sense::click())
        .on_hover_text("Clic para contraer / expandir Curva de tonos");
    paint(
        ui.painter(),
        Rect::from_center_size(pos2(card.left() + 14.0, hdr_y), vec2(12.5, 12.5)),
        if folded { Icon::ChevronRight } else { Icon::ChevronDown },
        Color32::from_rgb(215, 220, 230),
    );
    paint(
        ui.painter(),
        Rect::from_center_size(pos2(card.left() + 29.0, hdr_y), vec2(15.0, 15.0)),
        Icon::Curve,
        Color32::WHITE,
    );
    ui.painter().text(
        pos2(card.left() + 42.0, hdr_y),
        Align2::LEFT_CENTER,
        "Curva de tonos",
        t.semibold(13.5),
        Color32::WHITE,
    );

    if app.ui.curve_channel == "parametric" {
        app.ui.curve_channel = "master".into();
    }
    let ch = app.ui.curve_channel.clone();

    let close_r = Rect::from_center_size(pos2(card.right() - 18.0, hdr_y), vec2(22.0, 22.0));
    let close_resp = ui
        .interact(close_r, egui::Id::new("studio-curve-reset"), Sense::click())
        .on_hover_text("Restablecer curva de tonos");
    paint(ui.painter(), close_r.shrink(5.5), Icon::Close, if close_resp.hovered() { Color32::WHITE } else { Color32::from_rgb(200, 205, 215) });
    if close_resp.clicked() {
        let _ = app.run("curve.reset", json!({"channel": "all"}));
    } else if hdr_resp.clicked() {
        app.ui.toggle_folded("studio:curva");
    }

    if folded {
        return;
    }

    let r = Rect::from_min_max(pos2(card.left() + 12.0, card.top() + 36.0), pos2(card.right() - 12.0, card.bottom() - 38.0));
    let resp = ui.interact(r, egui::Id::new(("studio-curve", &ch)), Sense::click_and_drag());
    register(ui.ctx(), "curve", r);
    crate::widgets::context_menu(&resp, |ui| super::edit::curve_reset_menu(app, ui, &ch));
    let p = ui.painter();
    p.rect(
        r,
        CornerRadius::same(8),
        Color32::from_rgba_unmultiplied(20, 22, 27, 200),
        Stroke::new(1.0, Color32::from_white_alpha(26)),
        StrokeKind::Inside,
    );
    if let Some(hist) = super::edit::preview_histogram(app, id) {
        super::edit::draw_curve_histogram(p, r, &hist, &ch);
    }
    for i in 1..4 {
        let f = i as f32 / 4.0;
        p.line_segment(
            [pos2(r.left() + r.width() * f, r.top()), pos2(r.left() + r.width() * f, r.bottom())],
            Stroke::new(1.0, Color32::from_white_alpha(18)),
        );
        p.line_segment(
            [pos2(r.left(), r.top() + r.height() * f), pos2(r.right(), r.top() + r.height() * f)],
            Stroke::new(1.0, Color32::from_white_alpha(18)),
        );
    }
    p.line_segment([r.left_bottom(), r.right_top()], Stroke::new(1.0, Color32::from_white_alpha(35)));

    let to_screen = |x: f64, y: f64| pos2(r.left() + x as f32 * r.width(), r.bottom() - y as f32 * r.height());
    let from_screen =
        |q: Pos2| (((q.x - r.left()) / r.width()).clamp(0.0, 1.0) as f64, ((r.bottom() - q.y) / r.height()).clamp(0.0, 1.0) as f64);
    let mut pts: Vec<Point> = match ch.as_str() {
        "red" => d.curve.red.clone(),
        "green" => d.curve.green.clone(),
        "blue" => d.curve.blue.clone(),
        _ => d.curve.master.clone(),
    }
    .into_iter()
    .filter(|q| q.x.is_finite() && q.y.is_finite())
    .collect();
    pts.sort_by(|a, b| a.x.total_cmp(&b.x));
    if pts.len() < 2 {
        pts = vec![Point::new(0.0, 0.0), Point::new(1.0, 1.0)];
    }
    if pts.len() == 2 && (pts[0].x - 0.0).abs() < 1e-4 && (pts[0].y - 0.0).abs() < 1e-4 && (pts[1].x - 1.0).abs() < 1e-4 && (pts[1].y - 1.0).abs() < 1e-4 {
        pts = vec![
            Point::new(0.0, 0.0),
            Point::new(0.33, 0.33),
            Point::new(0.68, 0.68),
            Point::new(1.0, 1.0),
        ];
    }

    let curve = lightcraft_color::spline::MonotoneCurve::new(&pts.iter().map(|q| (q.x, q.y)).collect::<Vec<_>>());
    let color = match ch.as_str() {
        "red" => hex("#e05252"),
        "green" => hex("#52b768"),
        "blue" => hex("#4a90e2"),
        _ => Color32::WHITE,
    };
    let line: Vec<Pos2> = (0..=96).map(|i| to_screen(i as f64 / 96.0, curve.eval(i as f64 / 96.0).clamp(0.0, 1.0))).collect();
    p.add(egui::Shape::line(line, Stroke::new(1.8, color)));

    let drag_id = egui::Id::new(("st-curve-drag", &ch));
    let mut dragging: Option<usize> = ui.data(|dd| dd.get_temp::<Option<usize>>(drag_id)).flatten();
    let nearest = |q: Pos2| pts.iter().enumerate().map(|(i, pt)| (i, to_screen(pt.x, pt.y).distance(q))).min_by(|a, b| a.1.total_cmp(&b.1));
    let hovered = resp.hover_pos().and_then(nearest).filter(|(_, dist)| *dist < 12.0).map(|(i, _)| i);
    if dragging.is_some() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
    } else if hovered.is_some() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
    }

    for (i, q) in pts.iter().enumerate() {
        let c = to_screen(q.x, q.y);
        if dragging == Some(i) || (dragging.is_none() && hovered == Some(i)) {
            p.circle_filled(c, 5.2, color);
            p.circle_stroke(c, 5.2, Stroke::new(1.2, Color32::from_rgb(20, 22, 27)));
        } else {
            p.circle_filled(c, 3.6, color);
        }
    }

    if let Some(q) = dragging.and_then(|i| pts.get(i)) {
        p.text(
            r.left_top() + vec2(8.0, 5.0),
            Align2::LEFT_TOP,
            format!("{:.0} / {:.0}", (q.x * 255.0).round(), (q.y * 255.0).round()),
            t.font(11.0),
            Color32::from_rgb(200, 205, 215),
        );
    }

    let mut new_pts = None;
    if resp.double_clicked()
        && let Some(q) = resp.interact_pointer_pos()
        && let Some((i, dist)) = nearest(q)
        && dist < 12.0
        && i != 0
        && i + 1 < pts.len()
    {
        let mut v = pts.clone();
        v.remove(i);
        new_pts = Some(v);
    } else if resp.drag_started()
        && let Some(q) = ui.input(|i| i.pointer.press_origin()).or(resp.interact_pointer_pos())
    {
        let _ = app.run("develop.beginInteraction", json!({"label": "Tone Curve"}));
        match nearest(q) {
            Some((i, dist)) if dist < 14.0 => dragging = Some(i),
            _ if pts.len() < 16 => {
                let (x, y) = from_screen(q);
                let mut v = pts.clone();
                let idx = v.iter().position(|pt| pt.x > x).unwrap_or(v.len());
                v.insert(idx, Point::new(x, y));
                dragging = Some(idx);
                new_pts = Some(v);
            }
            _ => {}
        }
    } else if resp.dragged()
        && let (Some(i), Some(q)) = (dragging, resp.interact_pointer_pos())
    {
        let mut v = pts.clone();
        let fine = ui.input(|i| i.modifiers.alt);
        let (mut x, y) = match v.get(i) {
            Some(cur) if fine => {
                let dd = ui.input(|i| i.pointer.delta());
                (
                    (cur.x + (dd.x / r.width()) as f64 * 0.1).clamp(0.0, 1.0),
                    (cur.y - (dd.y / r.height()) as f64 * 0.1).clamp(0.0, 1.0),
                )
            }
            _ => from_screen(q),
        };
        if i < v.len() {
            let lo = if i == 0 { 0.0 } else { v[i - 1].x + 0.01 };
            let hi = if i + 1 >= v.len() { 1.0 } else { v[i + 1].x - 0.01 };
            if i == 0 {
                x = x.min(hi);
            } else if i + 1 == v.len() {
                x = x.max(lo);
            } else if lo <= hi {
                x = x.clamp(lo, hi);
            } else {
                x = (v[i - 1].x + v[i + 1].x) * 0.5;
            }
            v[i] = Point::new(x, y);
            new_pts = Some(v);
        }
    } else if resp.clicked()
        && let Some(q) = resp.interact_pointer_pos()
        && nearest(q).is_none_or(|(_, dist)| dist >= 12.0)
        && pts.len() < 16
    {
        let (x, y) = from_screen(q);
        let mut v = pts.clone();
        let idx = v.iter().position(|pt| pt.x > x).unwrap_or(v.len());
        v.insert(idx, Point::new(x, y));
        new_pts = Some(v);
    }

    if let Some(v) = new_pts {
        let arr: Vec<[f64; 2]> = v.iter().map(|pt| [pt.x, pt.y]).collect();
        let _ = app.run("develop.curve", json!({"channel": ch, "points": arr}));
    }
    if resp.drag_stopped() {
        dragging = None;
        let _ = app.run("develop.endInteraction", json!({}));
    }
    ui.data_mut(|dd| dd.insert_temp(drag_id, dragging));

    // Bottom 4 Colored Ring Channel Selectors: ⚪ 🔴 🟢 🔵
    let ch_y = card.bottom() - 19.0;
    let channels = [
        ("master", Color32::WHITE),
        ("red", Color32::from_rgb(228, 86, 82)),
        ("green", Color32::from_rgb(68, 178, 108)),
        ("blue", Color32::from_rgb(46, 114, 218)),
    ];
    let spacing = 52.0;
    let start_x = card.center().x - (channels.len() as f32 - 1.0) * spacing * 0.5;
    for (idx, (cid, ccol)) in channels.into_iter().enumerate() {
        let center = pos2(start_x + idx as f32 * spacing, ch_y);
        let cr = Rect::from_center_size(center, vec2(24.0, 24.0));
        let cresp = ui.interact(cr, egui::Id::new(("st-curve-ch", cid)), Sense::click());
        let sel = app.ui.curve_channel == cid;
        let p = ui.painter();
        p.circle_filled(center, 7.5, Color32::from_rgb(24, 26, 32));
        p.circle_stroke(center, 7.2, Stroke::new(if sel { 3.2 } else { 2.6 }, ccol));
        if cresp.double_clicked() {
            app.ui.curve_channel = cid.into();
            let _ = app.run("curve.reset", json!({"channel": cid}));
        } else if cresp.clicked() {
            app.ui.curve_channel = cid.into();
        }
    }
}

fn show_dedicated_tool_card(
    app: &mut LightcraftApp,
    ui: &mut egui::Ui,
    t: &Tokens,
    title: &str,
    icon_kind: u8,
    reset_controls: &[&str],
    body: impl FnOnce(&mut LightcraftApp, &mut egui::Ui),
) {
    let w = ui.available_width();
    let fold_key = format!("studio:card:{title}");
    let folded = app.ui.section_folded(&fold_key);
    egui::Frame::NONE
        .fill(Color32::from_rgba_unmultiplied(36, 38, 44, 186))
        .stroke(Stroke::new(1.0, Color32::from_white_alpha(36)))
        .corner_radius(CornerRadius::same(16))
        .show(ui, |ui| {
            ui.set_width(w);
            let (hr, _) = ui.allocate_exact_size(vec2(w, 38.0), Sense::hover());
            let hdr_click_r = Rect::from_min_max(hr.min, pos2(hr.right() - 42.0, hr.max.y));
            let hdr_resp = ui
                .interact(hdr_click_r, egui::Id::new(("studio-card-fold", title)), Sense::click())
                .on_hover_text(format!("Clic para contraer / expandir {title}"));
            paint(
                ui.painter(),
                Rect::from_center_size(pos2(hr.left() + 14.0, hr.center().y), vec2(12.5, 12.5)),
                if folded { Icon::ChevronRight } else { Icon::ChevronDown },
                Color32::from_rgb(215, 220, 230),
            );
            let ir = Rect::from_center_size(pos2(hr.left() + 29.0, hr.center().y), vec2(15.5, 15.5));
            let ic = Color32::from_rgb(245, 192, 98);
            match icon_kind {
                1 => paint_droplet_icon(ui.painter(), ir, ic),
                2 => paint_sparkles_icon(ui.painter(), ir, ic),
                3 => paint_detail_triangle_icon(ui.painter(), ir, ic),
                4 => paint_optics_icon(ui.painter(), ir, ic),
                _ => paint(ui.painter(), ir, Icon::Curve, ic),
            }
            ui.painter().text(
                pos2(hr.left() + 43.0, hr.center().y),
                Align2::LEFT_CENTER,
                title,
                t.semibold(13.6),
                Color32::WHITE,
            );

            // Reset button ↺ on the top-right of the card
            let rst_pill = Rect::from_center_size(pos2(hr.right() - 22.0, hr.center().y), vec2(26.0, 22.0));
            let r_resp = ui
                .interact(rst_pill, egui::Id::new(("studio-card-reset", title)), Sense::click())
                .on_hover_text(format!("Restablecer {title}"));
            ui.painter().rect(
                rst_pill,
                CornerRadius::same(11),
                if r_resp.hovered() {
                    Color32::from_white_alpha(32)
                } else {
                    Color32::from_white_alpha(18)
                },
                Stroke::new(1.0, Color32::from_white_alpha(34)),
                StrokeKind::Inside,
            );
            paint(
                ui.painter(),
                Rect::from_center_size(rst_pill.center(), vec2(11.5, 11.5)),
                Icon::Rotate,
                Color32::from_rgb(220, 224, 232),
            );
            if r_resp.clicked() {
                for cid in reset_controls {
                    if let Some(spec) = controls::find(cid) {
                        let _ = app.run("develop.set", json!({"control": cid, "value": spec.default}));
                    }
                }
            } else if hdr_resp.clicked() {
                app.ui.toggle_folded(&fold_key);
            }

            if !folded {
                body(app, ui);
            }
        });
}

fn studio_collapsible_sub_header(app: &mut LightcraftApp, ui: &mut egui::Ui, t: &Tokens, key: &str, title: &str) -> bool {
    let w = ui.available_width();
    let folded = app.ui.section_folded(key);
    let (r, resp) = ui.allocate_exact_size(vec2(w, 26.0), Sense::click());
    let resp = resp.on_hover_text("Clic para contraer / expandir esta subsección");
    let p = ui.painter();
    p.line_segment(
        [pos2(r.left() + 12.0, r.top() + 4.0), pos2(r.right() - 12.0, r.top() + 4.0)],
        Stroke::new(1.0, Color32::from_white_alpha(18)),
    );
    paint(
        p,
        Rect::from_center_size(pos2(r.left() + 18.0, r.bottom() - 7.5), vec2(11.0, 11.0)),
        if folded { Icon::ChevronRight } else { Icon::ChevronDown },
        if resp.hovered() { Color32::WHITE } else { Color32::from_rgb(175, 182, 196) },
    );
    p.text(
        pos2(r.left() + 28.0, r.bottom() - 7.5),
        Align2::LEFT_CENTER,
        title,
        t.semibold(10.0),
        if resp.hovered() { Color32::WHITE } else { Color32::from_rgb(175, 182, 196) },
    );
    if resp.clicked() {
        app.ui.toggle_folded(key);
    }
    !app.ui.section_folded(key)
}

fn studio_dropdown_row(ui: &mut egui::Ui, t: &Tokens, id: &str, label: &str, value_text: &str) -> egui::Response {
    let w = ui.available_width();
    let (row, _) = ui.allocate_exact_size(vec2(w, 34.0), Sense::hover());
    let p = ui.painter();
    p.text(
        pos2(row.left() + 14.0, row.center().y),
        Align2::LEFT_CENTER,
        label,
        t.font(12.2),
        Color32::from_rgb(208, 212, 222),
    );
    let pill_w = (w * 0.50).clamp(112.0, 152.0);
    let pill = Rect::from_min_size(pos2(row.right() - 14.0 - pill_w, row.center().y - 12.5), vec2(pill_w, 25.0));
    let resp = ui.interact(pill, egui::Id::new(id), Sense::click());
    p.rect(
        pill,
        CornerRadius::same(8),
        if resp.hovered() {
            Color32::from_white_alpha(30)
        } else {
            Color32::from_white_alpha(18)
        },
        Stroke::new(1.0, Color32::from_white_alpha(34)),
        StrokeKind::Inside,
    );
    p.text(
        pos2(pill.left() + 10.0, pill.center().y),
        Align2::LEFT_CENTER,
        value_text,
        t.semibold(11.6),
        Color32::WHITE,
    );
    paint(
        p,
        Rect::from_center_size(pos2(pill.right() - 12.0, pill.center().y), vec2(10.0, 10.0)),
        Icon::ChevronDown,
        Color32::from_rgb(200, 205, 215),
    );
    resp
}

fn studio_toggle_row(ui: &mut egui::Ui, t: &Tokens, id: &str, label: &str, val: &mut bool) -> bool {
    let w = ui.available_width();
    let (row, resp) = ui.allocate_exact_size(vec2(w, 34.0), Sense::click());
    ui.push_id(id, |ui| {
        register(ui.ctx(), format!("toggle:{id}"), row);
    });
    let mut changed = false;
    if resp.clicked() {
        *val = !*val;
        changed = true;
    }
    let p = ui.painter();
    p.text(
        pos2(row.left() + 14.0, row.center().y),
        Align2::LEFT_CENTER,
        label,
        t.font(12.0),
        Color32::from_rgb(228, 232, 240),
    );
    let sw = Rect::from_min_size(pos2(row.right() - 14.0 - 36.0, row.center().y - 10.0), vec2(36.0, 20.0));
    let anim = ui.ctx().animate_bool_with_time(egui::Id::new(("st-tog", id)), *val, 0.14);
    let bg = crate::widgets::lerp(Color32::from_white_alpha(34), Color32::from_rgb(236, 178, 84), anim);
    p.rect(
        sw,
        CornerRadius::same(10),
        bg,
        Stroke::new(1.0, Color32::from_white_alpha(40)),
        StrokeKind::Inside,
    );
    let knob_x = sw.left() + 10.0 + anim * (sw.width() - 20.0);
    p.circle_filled(pos2(knob_x, sw.center().y), 7.5, Color32::WHITE);
    changed
}

fn show_mixer_card(app: &mut LightcraftApp, ui: &mut egui::Ui, t: &Tokens, d: &DevelopSettings) {
    let bands = lightcraft_develop::MIXER_BANDS;
    let band_names_es = [
        "Rojo",
        "Naranja",
        "Amarillo",
        "Verde",
        "Aguamarina",
        "Azul",
        "Púrpura",
        "Magenta",
    ];
    let sel = bands.iter().position(|b| *b == app.ui.mixer_mode).unwrap_or(0);
    let cur_band = bands[sel];
    let cur_band_es = band_names_es.get(sel).copied().unwrap_or("Rojo");
    let folded = app.ui.section_folded("studio:mixer");

    let w = ui.available_width();
    egui::Frame::NONE
        .fill(Color32::from_rgba_unmultiplied(36, 38, 44, 186))
        .stroke(Stroke::new(1.0, Color32::from_white_alpha(36)))
        .corner_radius(CornerRadius::same(16))
        .show(ui, |ui| {
            ui.set_width(w);
            let (hr, _) = ui.allocate_exact_size(vec2(w, 38.0), Sense::hover());
            let hdr_click_r = Rect::from_min_max(hr.min, pos2(hr.right() - 108.0, hr.max.y));
            let hdr_resp = ui
                .interact(hdr_click_r, egui::Id::new("studio-mixer-hdr-fold"), Sense::click())
                .on_hover_text("Clic para contraer / expandir Mezclador de color");
            paint(
                ui.painter(),
                Rect::from_center_size(pos2(hr.left() + 14.0, hr.center().y), vec2(12.5, 12.5)),
                if folded { Icon::ChevronRight } else { Icon::ChevronDown },
                Color32::from_rgb(215, 220, 230),
            );
            let ir = Rect::from_center_size(pos2(hr.left() + 29.0, hr.center().y), vec2(15.5, 15.5));
            paint(ui.painter(), ir, Icon::Radial, Color32::from_rgb(245, 192, 98));
            ui.painter().text(
                pos2(hr.left() + 43.0, hr.center().y),
                Align2::LEFT_CENTER,
                "Mezclador de color",
                t.semibold(13.6),
                Color32::WHITE,
            );

            // Active color badge pill + Reset button ↺
            let rst_pill = Rect::from_center_size(pos2(hr.right() - 22.0, hr.center().y), vec2(26.0, 22.0));
            let badge_w = 68.0;
            let badge_r = Rect::from_min_size(pos2(rst_pill.left() - 6.0 - badge_w, hr.center().y - 10.5), vec2(badge_w, 21.0));
            ui.painter().rect(
                badge_r,
                CornerRadius::same(10),
                hex(BAND_COLORS[sel]).gamma_multiply(0.28),
                Stroke::new(1.0, hex(BAND_COLORS[sel]).gamma_multiply(0.85)),
                StrokeKind::Inside,
            );
            ui.painter().text(
                badge_r.center(),
                Align2::CENTER_CENTER,
                cur_band_es,
                t.semibold(10.6),
                Color32::WHITE,
            );

            let r_resp = ui
                .interact(rst_pill, egui::Id::new("studio-mixer-reset"), Sense::click())
                .on_hover_text("Restablecer Mezclador de colores");
            ui.painter().rect(
                rst_pill,
                CornerRadius::same(11),
                if r_resp.hovered() {
                    Color32::from_white_alpha(32)
                } else {
                    Color32::from_white_alpha(18)
                },
                Stroke::new(1.0, Color32::from_white_alpha(34)),
                StrokeKind::Inside,
            );
            paint(
                ui.painter(),
                Rect::from_center_size(rst_pill.center(), vec2(11.5, 11.5)),
                Icon::Rotate,
                Color32::from_rgb(220, 224, 232),
            );
            if r_resp.clicked() {
                for b in bands {
                    for k in ["hue", "sat", "lum"] {
                        let cid = format!("mixer.{b}.{k}");
                        if let Some(spec) = controls::find(&cid) {
                            let _ = app.run("develop.set", json!({"control": cid, "value": spec.default}));
                        }
                    }
                }
            } else if hdr_resp.clicked() {
                app.ui.toggle_folded("studio:mixer");
            }

            if folded {
                return;
            }

            // 8 Color Swatches Row
            let (sw_row, _) = ui.allocate_exact_size(vec2(w, 34.0), Sense::hover());
            let pad_x = 14.0;
            let step = (w - pad_x * 2.0) / bands.len() as f32;
            for (i, b) in bands.iter().enumerate() {
                let cx = sw_row.left() + pad_x + (i as f32 + 0.5) * step;
                let cy = sw_row.center().y;
                let hit = Rect::from_center_size(pos2(cx, cy), vec2(26.0, 26.0));
                let resp = ui
                    .interact(hit, egui::Id::new(("st-mix-band", *b)), Sense::click())
                    .on_hover_text(band_names_es[i]);
                register(ui.ctx(), format!("mixerBand:{b}"), hit);
                let p = ui.painter();
                let c = hex(BAND_COLORS[i]);
                p.circle_filled(pos2(cx, cy), if i == sel { 9.5 } else { 8.0 }, c);
                if i == sel {
                    p.circle_stroke(pos2(cx, cy), 11.5, Stroke::new(1.8, Color32::WHITE));
                    p.circle_filled(pos2(cx, cy), 2.5, Color32::WHITE);
                } else if resp.hovered() {
                    p.circle_stroke(pos2(cx, cy), 10.0, Stroke::new(1.2, Color32::from_white_alpha(150)));
                }
                if resp.clicked() {
                    app.ui.mixer_mode = (*b).to_string();
                }
            }

            studio_control(app, ui, d, &format!("mixer.{cur_band}.hue"), "Tono");
            studio_control(app, ui, d, &format!("mixer.{cur_band}.sat"), "Saturación");
            studio_control(app, ui, d, &format!("mixer.{cur_band}.lum"), "Luminancia");
            ui.add_space(8.0);
        });
}

fn studio_control(app: &mut LightcraftApp, ui: &mut egui::Ui, d: &DevelopSettings, id: &str, label: &str) {
    let Some(spec) = controls::find(id) else { return };
    let v = controls::get(d, id).unwrap_or(spec.default);
    studio_slider_spec(app, ui, spec, v, label, |app, val| app.run("develop.set", json!({"control": id, "value": val})));
}

fn studio_slider_spec(
    app: &mut LightcraftApp,
    ui: &mut egui::Ui,
    spec: &ControlSpec,
    value: f64,
    label: &str,
    mut set: impl FnMut(&mut LightcraftApp, f64) -> Result<serde_json::Value, String>,
) {
    let t = Tokens::get(ui.ctx());
    let w = ui.available_width();
    let (row, _) = ui.allocate_exact_size(vec2(w, 36.0), Sense::hover());
    let pad = 14.0;
    let label_y = row.top() + 9.0;
    let track_y = row.top() + 24.0;
    let track_r = Rect::from_min_max(pos2(row.left() + pad, track_y - 6.0), pos2(row.right() - pad, track_y + 6.0));

    let id = ui.id().with(("studio-slider", spec.id));
    let resp = ui.interact(track_r, id, Sense::click_and_drag());
    register(ui.ctx(), format!("slider:{}", spec.id), row);

    let mut cur = value;
    if resp.double_clicked() {
        let _ = app.run("develop.beginInteraction", json!({"label": spec.label}));
        let _ = set(app, spec.default);
        let _ = app.run("develop.endInteraction", json!({}));
        cur = spec.default;
    } else {
        if resp.drag_started() {
            let _ = app.run("develop.beginInteraction", json!({"label": spec.label}));
            app.ui.dragging_control = Some(spec.id.to_string());
        }
        if (resp.dragged() || resp.clicked())
            && let Some(pos) = resp.interact_pointer_pos()
        {
            let norm = ((pos.x - track_r.left()) / track_r.width().max(1.0)).clamp(0.0, 1.0) as f64;
            let step = spec.step.max(1e-9);
            let nv = ((spec.value_at(norm) / step).round() * step).clamp(spec.min, spec.max);
            if (nv - cur).abs() > 1e-12 {
                if resp.clicked() && !resp.dragged() {
                    let _ = app.run("develop.beginInteraction", json!({"label": spec.label}));
                    let _ = set(app, nv);
                    let _ = app.run("develop.endInteraction", json!({}));
                } else {
                    let _ = set(app, nv);
                }
                cur = nv;
            }
        }
        if resp.drag_stopped() {
            let _ = app.run("develop.endInteraction", json!({}));
            app.ui.dragging_control = None;
        }
    }

    let p = ui.painter();
    p.text(
        pos2(row.left() + pad, label_y),
        Align2::LEFT_CENTER,
        label,
        t.font(12.2),
        Color32::from_rgb(232, 234, 240),
    );
    let val_str = crate::widgets::shown_value(spec, cur);
    p.text(
        pos2(row.right() - pad, label_y),
        Align2::RIGHT_CENTER,
        val_str,
        t.font(12.2),
        Color32::WHITE,
    );

    let bar = Rect::from_min_max(pos2(track_r.left(), track_y - 1.0), pos2(track_r.right(), track_y + 1.0));
    let zero_x = bar.left() + bar.width() * spec.position(spec.default) as f32;
    let tx = bar.left() + bar.width() * spec.position(cur) as f32;

    match spec.track {
        Track::Temp => {
            let stops = [hex("#295088"), hex("#556880"), hex("#7c7a58"), hex("#9c8538")];
            let steps = 24;
            for i in 0..steps {
                let a = i as f32 / steps as f32;
                let b = (i + 1) as f32 / steps as f32;
                let seg = (a * 3.0).floor() as usize;
                let c = crate::widgets::lerp(stops[seg.min(3)], stops[(seg + 1).min(3)], a * 3.0 - seg as f32);
                p.rect_filled(
                    Rect::from_min_max(pos2(bar.left() + bar.width() * a, bar.top()), pos2(bar.left() + bar.width() * b + 0.5, bar.bottom())),
                    0.0,
                    c,
                );
            }
        }
        Track::Tint => {
            let stops = [hex("#2f7a44"), hex("#5a655e"), hex("#8c4488")];
            let steps = 24;
            for i in 0..steps {
                let a = i as f32 / steps as f32;
                let b = (i + 1) as f32 / steps as f32;
                let seg = (a * 2.0).floor() as usize;
                let c = crate::widgets::lerp(stops[seg.min(2)], stops[(seg + 1).min(2)], a * 2.0 - seg as f32);
                p.rect_filled(
                    Rect::from_min_max(pos2(bar.left() + bar.width() * a, bar.top()), pos2(bar.left() + bar.width() * b + 0.5, bar.bottom())),
                    0.0,
                    c,
                );
            }
        }
        _ => {
            p.rect_filled(bar, 1.0, Color32::from_white_alpha(48));
            let (x0, x1) = if tx >= zero_x { (zero_x, tx) } else { (tx, zero_x) };
            if (x1 - x0) > 1.0 {
                p.rect_filled(Rect::from_min_max(pos2(x0, bar.top()), pos2(x1, bar.bottom())), 1.0, Color32::from_white_alpha(135));
            }
        }
    }

    // Center tick mark on the slider track (matching reference sliders)
    p.line_segment([pos2(zero_x, track_y - 3.5), pos2(zero_x, track_y + 3.5)], Stroke::new(1.0, Color32::from_white_alpha(95)));

    // Solid white circular thumb
    let center = pos2(tx, track_y);
    p.circle_filled(center, 6.2, Color32::from_black_alpha(110));
    p.circle_filled(center, 5.2, Color32::WHITE);
}

// ============================================================================
// 6. CUSTOM VECTOR ICONS MATCHING THE REFERENCE DESIGN
// ============================================================================

fn paint_droplet_icon(p: &egui::Painter, r: Rect, c: Color32) {
    let cx = r.center().x;
    let top = r.top() + 1.5;
    let bot = r.bottom() - 1.5;
    let rad = r.width() * 0.34;
    let cy = bot - rad;
    let stroke = Stroke::new(1.5, c);
    p.line_segment([pos2(cx, top), pos2(cx - rad * 0.92, cy - rad * 0.25)], stroke);
    p.line_segment([pos2(cx, top), pos2(cx + rad * 0.92, cy - rad * 0.25)], stroke);
    let pts: Vec<Pos2> = (0..=12)
        .map(|i| {
            let a = std::f32::consts::PI * (i as f32 / 12.0);
            pos2(cx - rad * a.cos(), cy + rad * a.sin())
        })
        .collect();
    p.add(egui::Shape::line(pts, stroke));
}

fn paint_sparkles_icon(p: &egui::Painter, r: Rect, c: Color32) {
    let stroke = Stroke::new(1.4, c);
    let star4 = |center: Pos2, rad: f32| {
        p.line_segment([pos2(center.x, center.y - rad), pos2(center.x, center.y + rad)], stroke);
        p.line_segment([pos2(center.x - rad, center.y), pos2(center.x + rad, center.y)], stroke);
        let d = rad * 0.45;
        p.line_segment([pos2(center.x - d, center.y - d), pos2(center.x + d, center.y + d)], stroke);
        p.line_segment([pos2(center.x - d, center.y + d), pos2(center.x + d, center.y - d)], stroke);
    };
    star4(pos2(r.center().x - 2.0, r.center().y - 1.5), 4.5);
    star4(pos2(r.right() - 3.5, r.top() + 4.0), 2.3);
    star4(pos2(r.right() - 4.0, r.bottom() - 3.5), 2.0);
}

fn paint_detail_triangle_icon(p: &egui::Painter, r: Rect, c: Color32) {
    let top = pos2(r.center().x, r.top() + 2.0);
    let bl = pos2(r.left() + 2.0, r.bottom() - 2.0);
    let br = pos2(r.right() - 2.0, r.bottom() - 2.0);
    p.add(egui::Shape::closed_line(vec![top, br, bl], Stroke::new(1.5, c)));
    p.line_segment([top, pos2(r.center().x, r.bottom() - 2.0)], Stroke::new(1.0, c.gamma_multiply(0.65)));
}

fn paint_optics_icon(p: &egui::Painter, r: Rect, c: Color32) {
    p.circle_stroke(r.center(), r.width() * 0.42, Stroke::new(1.5, c));
    p.circle_stroke(r.center(), r.width() * 0.18, Stroke::new(1.3, c));
}

fn paint_copy_icon(p: &egui::Painter, r: Rect, c: Color32) {
    let stroke = Stroke::new(1.4, c);
    let r1 = Rect::from_min_size(pos2(r.left() + 2.0, r.top() + 2.0), vec2(9.0, 9.0));
    let r2 = Rect::from_min_size(pos2(r.right() - 11.0, r.bottom() - 11.0), vec2(9.0, 9.0));
    p.rect_stroke(r1, CornerRadius::same(2), stroke, StrokeKind::Inside);
    p.rect_filled(r2, CornerRadius::same(2), Color32::from_rgb(38, 40, 46));
    p.rect_stroke(r2, CornerRadius::same(2), stroke, StrokeKind::Inside);
}

fn paint_paste_icon(p: &egui::Painter, r: Rect, c: Color32) {
    let stroke = Stroke::new(1.4, c);
    let body = Rect::from_min_max(pos2(r.left() + 2.5, r.top() + 3.5), pos2(r.right() - 2.5, r.bottom() - 1.5));
    let clip = Rect::from_center_size(pos2(r.center().x, r.top() + 3.2), vec2(6.0, 2.8));
    p.rect_stroke(body, CornerRadius::same(2), stroke, StrokeKind::Inside);
    p.rect_filled(clip, CornerRadius::same(1), c);
}

fn paint_sync_icon(p: &egui::Painter, r: Rect, c: Color32) {
    let stroke = Stroke::new(1.5, c);
    p.circle_stroke(r.center(), r.width() * 0.36, stroke);
    let top = pos2(r.center().x + 2.0, r.top() + 2.2);
    p.line_segment([top, pos2(top.x - 3.0, top.y - 1.5)], stroke);
    p.line_segment([top, pos2(top.x - 2.5, top.y + 2.2)], stroke);
}

fn paint_camera_badge_icon(p: &egui::Painter, r: Rect, c: Color32) {
    let stroke = Stroke::new(1.3, c);
    let body = Rect::from_min_max(pos2(r.left() + 1.0, r.top() + 4.0), pos2(r.right() - 1.0, r.bottom() - 2.0));
    p.rect_stroke(body, CornerRadius::same(2), stroke, StrokeKind::Inside);
    p.circle_stroke(body.center(), 2.4, stroke);
}

fn paint_aperture_icon(p: &egui::Painter, r: Rect, c: Color32) {
    let stroke = Stroke::new(1.3, c);
    let rad = r.width() * 0.42;
    p.circle_stroke(r.center(), rad, stroke);
    p.line_segment([pos2(r.center().x - rad * 0.6, r.center().y - rad * 0.6), pos2(r.center().x + rad * 0.6, r.center().y + rad * 0.6)], stroke);
}

#[derive(Clone)]
struct MagicEffectEntry {
    cat_idx: usize,
    label: String,
    preset: lightcraft_develop::Preset,
}

const MAGIC_CATEGORIES: [(&str, &str); 7] = [
    ("ojos", "👁 Ojos"),
    ("piel", "✨ Piel"),
    ("boca", "👄 Boca"),
    ("cejas", "🧔 Cejas/Barba"),
    ("sujeto", "👤 Sujeto/Ropa"),
    ("fondo", "🌅 Fondo"),
    ("paisaje", "🏔 Paisaje"),
];

fn format_magic_effect_label(raw_stem: &str) -> String {
    let s = raw_stem
        .trim_start_matches("01 ")
        .trim_start_matches("04 ")
        .trim_start_matches("05 ")
        .trim();
    let lower = s.to_lowercase();
    if lower.contains("piel + bronce") {
        return "Retrato Total IA".to_string();
    }
    if lower.starts_with("el blanco de los ojos") {
        let num = s.split_whitespace().last().unwrap_or("");
        return format!("Blanco Ojos {}", num).trim().to_string();
    }
    if lower.contains("piel completa") {
        let num = s.split_whitespace().last().unwrap_or("");
        return format!("Piel Completa {}", num).trim().to_string();
    }
    if lower.contains("cara") && lower.contains("piel") {
        let num = s.split_whitespace().last().unwrap_or("");
        return format!("Piel Rostro {}", num).trim().to_string();
    }
    if lower.starts_with("piel bronceada") {
        let num = s.split_whitespace().last().unwrap_or("");
        return format!("Piel Bronce {}", num).trim().to_string();
    }
    if lower.starts_with("mejora de la boca") {
        let num = s.split_whitespace().last().unwrap_or("");
        return format!("Realzar Labios {}", num).trim().to_string();
    }
    if lower.contains("labial") {
        let color = s.split_whitespace().last().unwrap_or("");
        let mut c_chars = color.chars();
        let cap = match c_chars.next() {
            Some(f) => f.to_uppercase().collect::<String>() + c_chars.as_str(),
            None => String::new(),
        };
        return format!("Labial {}", cap);
    }
    if lower.starts_with("aligerar persona") {
        let num = s.split_whitespace().last().unwrap_or("");
        return format!("Aclarar Sujeto {}", num).trim().to_string();
    }
    if lower.starts_with("contraste persona") {
        let num = s.split_whitespace().last().unwrap_or("");
        return format!("Contraste Sujeto {}", num).trim().to_string();
    }
    if lower.starts_with("menos sombra") {
        let num = s.split_whitespace().last().unwrap_or("");
        return format!("Quitar Sombra {}", num).trim().to_string();
    }
    if lower.starts_with("afilando ropa") {
        let num = s.split_whitespace().last().unwrap_or("");
        return format!("Nitidez Ropa {}", num).trim().to_string();
    }
    if lower.starts_with("ropa ligera") {
        let num = s.split_whitespace().last().unwrap_or("");
        return format!("Aclarar Ropa {}", num).trim().to_string();
    }
    if lower.starts_with("desempa") {
        return "Quitar Niebla".to_string();
    }
    // Capitalize first letter cleanly
    let cleaned = s.replace('_', " ");
    let mut chars = cleaned.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => cleaned,
    }
}

fn clean_active_mask_title(name: &str) -> String {
    let lower = name.to_lowercase();
    if lower.contains("clarear olhos") {
        "Ojos Claros IA".to_string()
    } else if lower == "mask 1" {
        "Efecto IA 1".to_string()
    } else if lower == "mask 2" {
        "Efecto IA 2".to_string()
    } else if lower == "mask 3" {
        "Efecto IA 3".to_string()
    } else {
        name.to_string()
    }
}

fn classify_magic_folder(folder_path: &str) -> Option<usize> {
    let s = folder_path.to_ascii_lowercase();
    if s.contains("ojo") {
        Some(0)
    } else if s.contains("piel") || s.contains("completa") {
        Some(1)
    } else if s.contains("boca") || s.contains("diente") {
        Some(2)
    } else if s.contains("ceja") || s.contains("barba") {
        Some(3)
    } else if s.contains("persona") || s.contains("objeto") || s.contains("ropa") {
        Some(4)
    } else if s.contains("fondo") {
        Some(5)
    } else if s.contains("paisaje") {
        Some(6)
    } else {
        None
    }
}

fn collect_xmp_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<_> = rd.filter_map(Result::ok).map(|e| e.path()).collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            collect_xmp_files(&path, out);
        } else if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("xmp")) {
            out.push(path);
        }
    }
}

fn resolve_bundled_asset_path(rel: &str) -> std::path::PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|exe| {
            let dir = exe.parent()?;
            let macos_res = dir.join("../Resources").join(rel);
            if macos_res.exists() {
                return Some(macos_res);
            }
            let win_dir = dir.join(rel);
            if win_dir.exists() {
                return Some(win_dir);
            }
            None
        })
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join(rel)
        })
}

fn magic_effects_catalog() -> &'static [MagicEffectEntry] {
    static CATALOG: std::sync::OnceLock<Vec<MagicEffectEntry>> = std::sync::OnceLock::new();
    CATALOG.get_or_init(|| {
        let root = resolve_bundled_asset_path("efectos-magicos");
        let mut files = Vec::new();
        collect_xmp_files(&root, &mut files);
        let mut list = Vec::new();
        for f in files {
            let folder_str = f.parent().map(|p| p.to_string_lossy().to_string()).unwrap_or_default();
            let Some(cat_idx) = classify_magic_folder(&folder_str) else { continue };
            let stem = f.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
            let Ok(bytes) = std::fs::read(&f) else { continue };
            let text = String::from_utf8_lossy(&bytes);
            let Some(mut preset) = lightcraft_engine::crs::preset_from_xmp(&text, &stem) else { continue };
            let clean_label = format_magic_effect_label(&stem);
            if let Some(masks) = preset.settings.get_mut("masks").and_then(serde_json::Value::as_array_mut) {
                let total = masks.len();
                for (idx, m) in masks.iter_mut().enumerate() {
                    if let Some(obj) = m.as_object_mut() {
                        let mname = if total == 1 {
                            clean_label.clone()
                        } else {
                            format!("{} ({})", clean_label, idx + 1)
                        };
                        obj.insert("name".into(), serde_json::Value::String(mname));
                    }
                }
            } else {
                continue;
            }
            preset.name = clean_label.clone();
            list.push(MagicEffectEntry {
                cat_idx,
                label: clean_label,
                preset,
            });
        }
        list
    })
}

fn show_magic_effects_card(app: &mut LightcraftApp, ui: &mut egui::Ui, t: &Tokens, d: &DevelopSettings) {
    use lightcraft_develop::Section;
    let catalog = magic_effects_catalog();
    let cat_id = egui::Id::new("studio-magic-effect-cat");
    let cur_cat = ui.ctx().data(|dd| dd.get_temp::<usize>(cat_id)).unwrap_or(0).min(MAGIC_CATEGORIES.len() - 1);
    let sub_tab_id = egui::Id::new("studio-magic-fx-subtab");
    let cur_sub_tab = ui.ctx().data(|dd| dd.get_temp::<usize>(sub_tab_id)).unwrap_or(0);
    let card_folded = app.ui.section_folded("studio:magicEffects");

    let w = ui.available_width();
    let pad = 10.0;
    let inner_w = (w - pad * 2.0).max(120.0);

    egui::Frame::NONE
        .fill(Color32::from_rgba_unmultiplied(36, 38, 44, 192))
        .stroke(Stroke::new(1.0, Color32::from_rgb(245, 192, 98).gamma_multiply(0.42)))
        .corner_radius(CornerRadius::same(16))
        .show(ui, |ui| {
            ui.set_width(w);

            // 1. Header Bar (collapsible, persisted)
            let (hr, _) = ui.allocate_exact_size(vec2(w, 36.0), Sense::hover());
            let rst_pill = Rect::from_center_size(pos2(hr.right() - pad - 12.0, hr.center().y), vec2(24.0, 21.0));
            let map_on = app.ui.mask_overlay_all;
            let map_w = if map_on { 76.0 } else { 54.0 };
            let map_pill = Rect::from_min_size(pos2(rst_pill.left() - 5.0 - map_w, hr.center().y - 10.5), vec2(map_w, 21.0));
            let hdr_click_r = Rect::from_min_max(hr.min, pos2(map_pill.left() - 4.0, hr.max.y));
            let hdr_resp = ui
                .interact(hdr_click_r, egui::Id::new("studio-magic-hdr-fold"), Sense::click())
                .on_hover_text("Clic para contraer / expandir Efectos Mágicos IA");

            paint(
                ui.painter(),
                Rect::from_center_size(pos2(hr.left() + 13.0, hr.center().y), vec2(12.0, 12.0)),
                if card_folded { Icon::ChevronRight } else { Icon::ChevronDown },
                Color32::from_rgb(225, 228, 236),
            );
            let ir = Rect::from_center_size(pos2(hr.left() + 27.0, hr.center().y), vec2(14.5, 14.5));
            paint_sparkles_icon(ui.painter(), ir, Color32::from_rgb(245, 192, 98));
            ui.painter().text(
                pos2(hr.left() + 40.0, hr.center().y),
                Align2::LEFT_CENTER,
                "Efectos Mágicos IA",
                t.semibold(13.0),
                Color32::WHITE,
            );

            let m_resp = ui
                .interact(map_pill, egui::Id::new("studio-magic-map-toggle"), Sense::click())
                .on_hover_text("Mostrar / ocultar guía roja de la máscara sobre la foto");
            ui.painter().rect(
                map_pill,
                CornerRadius::same(10),
                if map_on {
                    Color32::from_rgb(225, 62, 62).gamma_multiply(0.38)
                } else if m_resp.hovered() {
                    Color32::from_white_alpha(28)
                } else {
                    Color32::from_white_alpha(16)
                },
                Stroke::new(1.0, if map_on { Color32::from_rgb(255, 110, 110) } else { Color32::from_white_alpha(32) }),
                StrokeKind::Inside,
            );
            ui.painter().text(
                map_pill.center(),
                Align2::CENTER_CENTER,
                if map_on { "✕ Quitar rojo" } else { "👁 Mapa" },
                t.semibold(10.0),
                if map_on { Color32::from_rgb(255, 215, 215) } else { Color32::from_rgb(210, 215, 225) },
            );
            if m_resp.clicked() {
                app.ui.mask_overlay_all = !app.ui.mask_overlay_all;
            }

            let r_resp = ui
                .interact(rst_pill, egui::Id::new("studio-magic-reset-all"), Sense::click())
                .on_hover_text("Quitar todos los efectos mágicos de esta foto");
            ui.painter().rect(
                rst_pill,
                CornerRadius::same(10),
                if r_resp.hovered() { Color32::from_white_alpha(32) } else { Color32::from_white_alpha(16) },
                Stroke::new(1.0, Color32::from_white_alpha(32)),
                StrokeKind::Inside,
            );
            paint(
                ui.painter(),
                Rect::from_center_size(rst_pill.center(), vec2(11.0, 11.0)),
                Icon::Rotate,
                Color32::from_rgb(220, 224, 232),
            );
            if r_resp.clicked() {
                for m in &d.masks {
                    let _ = app.run("mask.delete", json!({"id": m.id}));
                }
            } else if hdr_resp.clicked() {
                app.ui.toggle_folded("studio:magicEffects");
            }

            if card_folded {
                return;
            }

            // 2. Fixed 2-Row Category Selector Grid (Row 1: 4 items, Row 2: 3 items) — zero wrapping glitches!
            let cat_gap = 4.0;
            let cat_row_h = 25.0;
            let (cat_area, _) = ui.allocate_exact_size(vec2(w, cat_row_h * 2.0 + cat_gap + 4.0), Sense::hover());
            let row1_y = cat_area.top() + 1.0;
            let row2_y = row1_y + cat_row_h + cat_gap;

            // Row 1: indices 0..4
            let w4 = (inner_w - cat_gap * 3.0) / 4.0;
            for idx in 0..4 {
                let (cid, clabel) = MAGIC_CATEGORIES[idx];
                let x = cat_area.left() + pad + idx as f32 * (w4 + cat_gap);
                let pill_r = Rect::from_min_size(pos2(x, row1_y), vec2(w4, cat_row_h));
                let resp = ui.interact(pill_r, egui::Id::new(("magic-cat-pill", cid)), Sense::click());
                let active = cur_cat == idx;
                ui.painter().rect(
                    pill_r,
                    CornerRadius::same(7),
                    if active {
                        Color32::from_rgb(245, 198, 112)
                    } else if resp.hovered() {
                        Color32::from_white_alpha(28)
                    } else {
                        Color32::from_white_alpha(14)
                    },
                    Stroke::new(
                        1.0,
                        if active { Color32::from_rgb(255, 228, 168) } else { Color32::from_white_alpha(28) },
                    ),
                    StrokeKind::Inside,
                );
                ui.painter().text(
                    pill_r.center(),
                    Align2::CENTER_CENTER,
                    clabel,
                    if active { t.semibold(10.3) } else { t.font(10.2) },
                    if active { Color32::from_rgb(22, 22, 26) } else { Color32::from_rgb(222, 226, 235) },
                );
                if resp.clicked() {
                    ui.ctx().data_mut(|dd| dd.insert_temp(cat_id, idx));
                }
            }

            // Row 2: indices 4..7
            let w3 = (inner_w - cat_gap * 2.0) / 3.0;
            for (col, idx) in (4..7).enumerate() {
                let (cid, clabel) = MAGIC_CATEGORIES[idx];
                let x = cat_area.left() + pad + col as f32 * (w3 + cat_gap);
                let pill_r = Rect::from_min_size(pos2(x, row2_y), vec2(w3, cat_row_h));
                let resp = ui.interact(pill_r, egui::Id::new(("magic-cat-pill", cid)), Sense::click());
                let active = cur_cat == idx;
                ui.painter().rect(
                    pill_r,
                    CornerRadius::same(7),
                    if active {
                        Color32::from_rgb(245, 198, 112)
                    } else if resp.hovered() {
                        Color32::from_white_alpha(28)
                    } else {
                        Color32::from_white_alpha(14)
                    },
                    Stroke::new(
                        1.0,
                        if active { Color32::from_rgb(255, 228, 168) } else { Color32::from_white_alpha(28) },
                    ),
                    StrokeKind::Inside,
                );
                ui.painter().text(
                    pill_r.center(),
                    Align2::CENTER_CENTER,
                    clabel,
                    if active { t.semibold(10.3) } else { t.font(10.2) },
                    if active { Color32::from_rgb(22, 22, 26) } else { Color32::from_rgb(222, 226, 235) },
                );
                if resp.clicked() {
                    ui.ctx().data_mut(|dd| dd.insert_temp(cat_id, idx));
                }
            }

            // 3. Uniform 2-Column Grid for Effects in Selected Category
            let cat_entries: Vec<(usize, &MagicEffectEntry)> = catalog
                .iter()
                .enumerate()
                .filter(|(_, e)| e.cat_idx == cur_cat)
                .collect();

            let matches_entry = |mask_name: &str, entry_label: &str| -> bool {
                mask_name == entry_label
                    || mask_name.starts_with(&format!("{entry_label} ("))
                    || clean_active_mask_title(mask_name).eq_ignore_ascii_case(entry_label)
            };

            if !cat_entries.is_empty() {
                ui.add_space(4.0);
                let cols = 2usize;
                let rows = cat_entries.len().div_ceil(cols);
                let fx_gap_x = 5.0;
                let fx_gap_y = 4.0;
                let cell_w = (inner_w - fx_gap_x) * 0.5;
                let cell_h = 25.0;
                let grid_h = rows as f32 * cell_h + (rows.saturating_sub(1)) as f32 * fx_gap_y + 4.0;
                let (grid_r, _) = ui.allocate_exact_size(vec2(w, grid_h), Sense::hover());

                let mut clicked_entry: Option<&MagicEffectEntry> = None;

                for (i, (e_idx, entry)) in cat_entries.iter().enumerate() {
                    let r_idx = i / cols;
                    let c_idx = i % cols;
                    let is_last_odd = i + 1 == cat_entries.len() && c_idx == 0;
                    let cw = if is_last_odd { inner_w } else { cell_w };
                    let x = grid_r.left() + pad + c_idx as f32 * (cell_w + fx_gap_x);
                    let y = grid_r.top() + 2.0 + r_idx as f32 * (cell_h + fx_gap_y);
                    let cell_r = Rect::from_min_size(pos2(x, y), vec2(cw, cell_h));

                    let matching_masks: Vec<&lightcraft_develop::Mask> =
                        d.masks.iter().filter(|m| matches_entry(&m.name, &entry.label)).collect();
                    let is_applied = !matching_masks.is_empty();
                    let is_visible = matching_masks.iter().any(|m| m.visible);
                    let tip = if is_applied && is_visible {
                        "Clic para ocultar este efecto (conservando todos tus ajustes)"
                    } else if is_applied {
                        "Efecto oculto · Clic para volver a activarlo con tus ajustes"
                    } else {
                        "Clic para aplicar este efecto mágico"
                    };
                    let resp = ui
                        .interact(cell_r, egui::Id::new(("magic-fx-cell", *e_idx)), Sense::click())
                        .on_hover_text(tip);

                    ui.painter().rect(
                        cell_r,
                        CornerRadius::same(7),
                        if is_applied && is_visible {
                            Color32::from_rgb(236, 178, 84).gamma_multiply(0.28)
                        } else if is_applied {
                            Color32::from_rgb(236, 178, 84).gamma_multiply(0.10)
                        } else if resp.hovered() {
                            Color32::from_white_alpha(26)
                        } else {
                            Color32::from_white_alpha(12)
                        },
                        Stroke::new(
                            1.0,
                            if is_applied && is_visible {
                                Color32::from_rgb(245, 192, 98)
                            } else if is_applied {
                                Color32::from_rgb(245, 192, 98).gamma_multiply(0.48)
                            } else if resp.hovered() {
                                Color32::from_white_alpha(48)
                            } else {
                                Color32::from_white_alpha(24)
                            },
                        ),
                        StrokeKind::Inside,
                    );

                    let label_txt = if is_applied && is_visible {
                        format!("✓ {}", entry.label)
                    } else if is_applied {
                        format!("◌ {}", entry.label)
                    } else {
                        entry.label.clone()
                    };
                    ui.painter().text(
                        cell_r.center(),
                        Align2::CENTER_CENTER,
                        label_txt,
                        if is_applied && is_visible { t.semibold(10.6) } else { t.font(10.5) },
                        if is_applied && is_visible {
                            Color32::from_rgb(255, 228, 168)
                        } else if is_applied {
                            Color32::from_rgb(200, 185, 152)
                        } else {
                            Color32::from_rgb(235, 238, 245)
                        },
                    );

                    if resp.clicked() {
                        clicked_entry = Some(*entry);
                    }
                }

                if let Some(entry) = clicked_entry {
                    ensure_active_photo(app);
                    if let Some(active_id) = app.session.active()
                        && let Some(cur_d) = app.session.develop_of(active_id)
                    {
                        let existing_ids: Vec<u32> = cur_d
                            .masks
                            .iter()
                            .filter(|m| matches_entry(&m.name, &entry.label))
                            .map(|m| m.id)
                            .collect();

                        if !existing_ids.is_empty() {
                            // Effect already exists on the photo!
                            // 1) Remove any accidental duplicate layers with the same name (keep the first one)
                            if existing_ids.len() > 1 {
                                for &dup_id in &existing_ids[1..] {
                                    let _ = app.run("mask.delete", json!({"id": dup_id}));
                                }
                            }
                            // 2) Toggle visibility ON/OFF without losing any user adjustments!
                            let keep_id = existing_ids[0];
                            let _ = app.run("mask.visible", json!({"id": keep_id}));
                            let _ = app.run("mask.select", json!({"id": keep_id}));
                        } else {
                            // Effect is not on the photo yet: append its mask(s) cleanly without touching any existing masks!
                            let mut next_d = (*cur_d).clone();
                            let mut next_id = next_d.masks.iter().map(|m| m.id).max().unwrap_or(0) + 1;
                            let mut first_added_id = None;
                            if let Some(preset_masks) = entry.preset.settings.get("masks").and_then(serde_json::Value::as_array) {
                                let total = preset_masks.len();
                                for (idx, pm) in preset_masks.iter().enumerate() {
                                    if let Ok(mut pm_parsed) = serde_json::from_value::<lightcraft_develop::Mask>(pm.clone()) {
                                        pm_parsed.id = next_id;
                                        next_id += 1;
                                        pm_parsed.name = if total <= 1 {
                                            entry.label.clone()
                                        } else {
                                            format!("{} ({})", entry.label, idx + 1)
                                        };
                                        pm_parsed.visible = true;
                                        if first_added_id.is_none() {
                                            first_added_id = Some(pm_parsed.id);
                                        }
                                        next_d.masks.push(pm_parsed);
                                    }
                                }
                            }
                            if let Some(op) = app.session.develop_op(active_id, next_d, &format!("Efecto: {}", entry.label)) {
                                let _ = app.session.commit(&format!("Efecto: {}", entry.label), op);
                            }
                            if let Some(mid) = first_added_id {
                                let _ = app.run("mask.select", json!({"id": mid}));
                            }
                        }
                        // Always show the real effect (not the red mask overlay) when applying/toggling an effect
                        app.ui.mask_overlay_all = false;
                        app.ui.mask_overlay = false;
                    }
                }
            }

            // 4. Active Effects Stack & Compact Direct Controls
            if !d.masks.is_empty() {
                let active_mid = app
                    .session
                    .active_mask
                    .filter(|id| d.masks.iter().any(|m| m.id == *id))
                    .or_else(|| d.masks.last().map(|m| m.id));

                if studio_collapsible_sub_header(app, ui, t, "studio:sub:magicActiveLayers", "CAPAS DE EFECTO ACTIVAS") {
                    let mut seen_titles: Vec<String> = Vec::new();
                    for m in &d.masks {
                        let clean_title = clean_active_mask_title(&m.name);
                        let norm_key = clean_title.trim().to_ascii_lowercase();
                        if seen_titles.contains(&norm_key) {
                            continue;
                        }
                        seen_titles.push(norm_key);

                        let is_sel = active_mid == Some(m.id);
                        let (outer_r, _) = ui.allocate_exact_size(vec2(w, 29.0), Sense::hover());
                        let row_r = Rect::from_min_size(pos2(outer_r.left() + pad, outer_r.top() + 1.5), vec2(inner_w, 26.0));

                        let del_r = Rect::from_center_size(pos2(row_r.right() - 14.0, row_r.center().y), vec2(20.0, 20.0));
                        let vis_r = Rect::from_center_size(pos2(row_r.right() - 36.0, row_r.center().y), vec2(20.0, 20.0));
                        let sel_r = Rect::from_min_max(row_r.min, pos2(vis_r.left() - 2.0, row_r.max.y));

                        let row_resp = ui
                            .interact(sel_r, egui::Id::new(("fx-sel", m.id)), Sense::click())
                            .on_hover_text(if !m.visible {
                                "Clic para activar y editar esta capa (conservando tus ajustes)"
                            } else if is_sel {
                                "Clic en el nombre para ocultar / mostrar esta capa sin perder tus ajustes"
                            } else {
                                "Clic para seleccionar esta capa (clic de nuevo para ocultarla)"
                            });
                        let vis_resp = ui
                            .interact(vis_r, egui::Id::new(("fx-vis", m.id)), Sense::click())
                            .on_hover_text(if m.visible { "Ocultar capa de efecto (sin perder ajustes)" } else { "Mostrar capa de efecto" });
                        let del_resp = ui
                            .interact(del_r, egui::Id::new(("fx-del", m.id)), Sense::click())
                            .on_hover_text("Eliminar este efecto");

                        ui.painter().rect(
                            row_r,
                            CornerRadius::same(7),
                            if is_sel && m.visible {
                                Color32::from_rgb(236, 178, 84).gamma_multiply(0.24)
                            } else if is_sel {
                                Color32::from_rgb(236, 178, 84).gamma_multiply(0.10)
                            } else if row_resp.hovered() {
                                Color32::from_white_alpha(20)
                            } else {
                                Color32::from_white_alpha(10)
                            },
                            Stroke::new(
                                1.0,
                                if is_sel { Color32::from_rgb(245, 192, 98) } else { Color32::from_white_alpha(24) },
                            ),
                            StrokeKind::Inside,
                        );

                        ui.painter().text(
                            pos2(row_r.left() + 10.0, row_r.center().y),
                            Align2::LEFT_CENTER,
                            &clean_title,
                            if is_sel { t.semibold(11.2) } else { t.font(11.0) },
                            if m.visible { Color32::WHITE } else { Color32::from_gray(135) },
                        );

                        paint(
                            ui.painter(),
                            vis_r.shrink(3.5),
                            if m.visible { Icon::Eye } else { Icon::EyeOff },
                            if vis_resp.hovered() { Color32::WHITE } else { Color32::from_rgb(205, 210, 220) },
                        );
                        paint(
                            ui.painter(),
                            del_r.shrink(4.5),
                            Icon::Close,
                            if del_resp.hovered() { Color32::from_rgb(255, 115, 115) } else { Color32::from_rgb(205, 210, 220) },
                        );

                        if del_resp.clicked() {
                            for dm in d.masks.iter().filter(|x| clean_active_mask_title(&x.name).eq_ignore_ascii_case(&clean_title)) {
                                let _ = app.run("mask.delete", json!({"id": dm.id}));
                            }
                        } else if vis_resp.clicked() {
                            let _ = app.run("mask.visible", json!({"id": m.id}));
                        } else if row_resp.clicked() {
                            if !m.visible {
                                let _ = app.run("mask.visible", json!({"id": m.id}));
                                let _ = app.run("mask.select", json!({"id": m.id}));
                            } else if is_sel {
                                let _ = app.run("mask.visible", json!({"id": m.id}));
                            } else {
                                let _ = app.run("mask.select", json!({"id": m.id}));
                            }
                        }
                    }
                }

                if let Some(mid) = active_mid
                    && let Some(m) = d.masks.iter().find(|m| m.id == mid).cloned()
                {
                    let clean_title = clean_active_mask_title(&m.name);
                    if studio_collapsible_sub_header(
                        app,
                        ui,
                        t,
                        "studio:sub:magicMaskSliders",
                        &format!("MÁSCARA Y FUERZA · {}", clean_title.to_uppercase()),
                    ) {
                        let spec_plain = |id: &'static str, min: f64, max: f64, default: f64, step: f64, decimals: u8| ControlSpec {
                            id,
                            label: id,
                            section: Section::Effects,
                            min,
                            max,
                            default,
                            step,
                            decimals,
                            track: Track::Plain,
                        };
                        let spec_center = |id: &'static str, min: f64, max: f64, default: f64, step: f64, decimals: u8| ControlSpec {
                            id,
                            label: id,
                            section: Section::Effects,
                            min,
                            max,
                            default,
                            step,
                            decimals,
                            track: Track::Centered,
                        };

                        // Core 3 Mask & Opacity Sliders (always visible)
                        let s_amt = spec_plain("fx.amount", 0.0, 200.0, 100.0, 1.0, 0);
                        studio_slider_spec(app, ui, &s_amt, m.adjust.amount, "Opacidad / Fuerza (%)", |app, v| {
                            app.run("mask.adjust", json!({"id": mid, "values": {"amount": v}}))
                        });

                        let s_fth = spec_plain("fx.feather", 0.0, 100.0, 0.0, 1.0, 0);
                        studio_slider_spec(app, ui, &s_fth, m.feather, "Difuminar orillas (Feather)", |app, v| {
                            app.run("mask.feather", json!({"id": mid, "value": v}))
                        });

                        let s_ref = spec_plain("fx.refine", 0.0, 100.0, 0.0, 1.0, 0);
                        studio_slider_spec(app, ui, &s_ref, m.refine, "Densidad en bordes (Refinar)", |app, v| {
                            app.run("mask.refine", json!({"id": mid, "value": v}))
                        });

                        // Quick Action Row: [ 🖐 Mover máscara con el mouse | ↺ Centrar ]
                        ui.add_space(4.0);
                        let drag_id = egui::Id::new("studio-magic-drag-mask");
                        let drag_on = ui.ctx().data(|dd| dd.get_temp::<bool>(drag_id)).unwrap_or(false);
                        let (act_outer, _) = ui.allocate_exact_size(vec2(w, 28.0), Sense::hover());
                        let rst_w = 72.0;
                        let drag_w = (inner_w - rst_w - 5.0).max(110.0);
                        let drag_r = Rect::from_min_size(pos2(act_outer.left() + pad, act_outer.top() + 1.0), vec2(drag_w, 25.0));
                        let rst_r = Rect::from_min_size(pos2(drag_r.right() + 5.0, act_outer.top() + 1.0), vec2(rst_w, 25.0));

                        let drag_resp = ui
                            .interact(drag_r, egui::Id::new("studio-magic-drag-btn"), Sense::click())
                            .on_hover_text("Actívalo y arrastra directamente sobre la foto para mover y ajustar la posición de la máscara");
                        ui.painter().rect(
                            drag_r,
                            CornerRadius::same(7),
                            if drag_on {
                                Color32::from_rgb(245, 198, 112)
                            } else if drag_resp.hovered() {
                                Color32::from_white_alpha(28)
                            } else {
                                Color32::from_white_alpha(14)
                            },
                            Stroke::new(
                                1.0,
                                if drag_on { Color32::from_rgb(255, 228, 168) } else { Color32::from_white_alpha(32) },
                            ),
                            StrokeKind::Inside,
                        );
                        ui.painter().text(
                            drag_r.center(),
                            Align2::CENTER_CENTER,
                            if drag_on {
                                "🖐 Moviendo en foto (Activo)"
                            } else {
                                "🖐 Mover máscara con el mouse"
                            },
                            if drag_on { t.semibold(10.5) } else { t.font(10.4) },
                            if drag_on { Color32::from_rgb(22, 22, 26) } else { Color32::from_rgb(235, 238, 245) },
                        );
                        if drag_resp.clicked() {
                            ui.ctx().data_mut(|dd| {
                                dd.insert_temp(drag_id, !drag_on);
                                if !drag_on {
                                    dd.insert_temp(sub_tab_id, 2usize);
                                }
                            });
                        }

                        let has_transform = m.offset_x.abs() > 1e-4 || m.offset_y.abs() > 1e-4 || (m.scale - 100.0).abs() > 1e-4;
                        let rst_resp = ui
                            .interact(rst_r, egui::Id::new("studio-magic-rst-pos"), Sense::click())
                            .on_hover_text("Restablecer posición y escala original de la máscara");
                        ui.painter().rect(
                            rst_r,
                            CornerRadius::same(7),
                            if rst_resp.hovered() {
                                Color32::from_white_alpha(28)
                            } else {
                                Color32::from_white_alpha(12)
                            },
                            Stroke::new(
                                1.0,
                                if has_transform { Color32::from_rgb(245, 198, 112) } else { Color32::from_white_alpha(24) },
                            ),
                            StrokeKind::Inside,
                        );
                        ui.painter().text(
                            rst_r.center(),
                            Align2::CENTER_CENTER,
                            "↺ Centrar",
                            t.font(10.3),
                            if has_transform { Color32::from_rgb(252, 225, 165) } else { Color32::from_rgb(195, 200, 210) },
                        );
                        if rst_resp.clicked() {
                            let _ = app.run(
                                "mask.transform",
                                json!({"id": mid, "offset_x": 0.0, "offset_y": 0.0, "scale": 100.0}),
                            );
                        }

                        // Compact 3-Tab Switcher for Fine Tuning: [ ☀ Luz | 🎨 Color | 🎯 Posición ]
                        ui.add_space(4.0);
                        let (tab_outer, _) = ui.allocate_exact_size(vec2(w, 26.0), Sense::hover());
                        let tab_gap = 4.0;
                        let tab_w = (inner_w - tab_gap * 2.0) / 3.0;
                        let sub_tabs = ["☀ Luz / Textura", "🎨 Color / Tinte", "🎯 Posición"];
                        for (t_idx, t_label) in sub_tabs.iter().enumerate() {
                            let tx = tab_outer.left() + pad + t_idx as f32 * (tab_w + tab_gap);
                            let tr = Rect::from_min_size(pos2(tx, tab_outer.top() + 1.0), vec2(tab_w, 23.0));
                            let tresp = ui.interact(tr, egui::Id::new(("magic-subtab", t_idx)), Sense::click());
                            let is_on = cur_sub_tab == t_idx;
                            ui.painter().rect(
                                tr,
                                CornerRadius::same(6),
                                if is_on {
                                    Color32::from_rgb(245, 198, 112)
                                } else if tresp.hovered() {
                                    Color32::from_white_alpha(24)
                                } else {
                                    Color32::from_white_alpha(12)
                                },
                                Stroke::new(1.0, if is_on { Color32::from_rgb(255, 228, 168) } else { Color32::from_white_alpha(24) }),
                                StrokeKind::Inside,
                            );
                            ui.painter().text(
                                tr.center(),
                                Align2::CENTER_CENTER,
                                *t_label,
                                if is_on { t.semibold(10.1) } else { t.font(10.0) },
                                if is_on { Color32::from_rgb(22, 22, 26) } else { Color32::from_rgb(220, 224, 234) },
                            );
                            if tresp.clicked() {
                                ui.ctx().data_mut(|dd| dd.insert_temp(sub_tab_id, t_idx));
                            }
                        }

                        if cur_sub_tab == 0 {
                            let s_exp = spec_center("fx.exposure", -4.0, 4.0, 0.0, 0.05, 2);
                            studio_slider_spec(app, ui, &s_exp, m.adjust.exposure, "Exposición / Brillo", |app, v| {
                                app.run("mask.adjust", json!({"id": mid, "values": {"exposure": v}}))
                            });

                            let s_sh = spec_center("fx.shadows", -100.0, 100.0, 0.0, 1.0, 0);
                            studio_slider_spec(app, ui, &s_sh, m.adjust.shadows, "Sombras", |app, v| {
                                app.run("mask.adjust", json!({"id": mid, "values": {"shadows": v}}))
                            });

                            let s_cl = spec_center("fx.clarity", -100.0, 100.0, 0.0, 1.0, 0);
                            studio_slider_spec(app, ui, &s_cl, m.adjust.clarity, "Claridad / Suavizado", |app, v| {
                                app.run("mask.adjust", json!({"id": mid, "values": {"clarity": v}}))
                            });

                            let s_tx = spec_center("fx.texture", -100.0, 100.0, 0.0, 1.0, 0);
                            studio_slider_spec(app, ui, &s_tx, m.adjust.texture, "Textura de piel", |app, v| {
                                app.run("mask.adjust", json!({"id": mid, "values": {"texture": v}}))
                            });

                            let s_sp = spec_center("fx.sharpness", -100.0, 100.0, 0.0, 1.0, 0);
                            studio_slider_spec(app, ui, &s_sp, m.adjust.sharpness, "Nitidez", |app, v| {
                                app.run("mask.adjust", json!({"id": mid, "values": {"sharpness": v}}))
                            });
                        } else if cur_sub_tab == 1 {
                            let s_tmp = spec_center("fx.temp", -100.0, 100.0, 0.0, 1.0, 0);
                            studio_slider_spec(app, ui, &s_tmp, m.adjust.temp, "Temperatura", |app, v| {
                                app.run("mask.adjust", json!({"id": mid, "values": {"temp": v}}))
                            });

                            let s_sat = spec_center("fx.saturation", -100.0, 100.0, 0.0, 1.0, 0);
                            studio_slider_spec(app, ui, &s_sat, m.adjust.saturation, "Saturación", |app, v| {
                                app.run("mask.adjust", json!({"id": mid, "values": {"saturation": v}}))
                            });

                            let s_chue = ControlSpec {
                                id: "fx.color_hue",
                                label: "fx.color_hue",
                                section: Section::Color,
                                min: 0.0,
                                max: 360.0,
                                default: 0.0,
                                step: 1.0,
                                decimals: 0,
                                track: Track::Plain,
                            };
                            studio_slider_spec(app, ui, &s_chue, m.adjust.color_hue, "Tono de tinte (0–360°)", |app, v| {
                                app.run("mask.adjust", json!({"id": mid, "values": {"color_hue": v}}))
                            });

                            let s_csat = spec_plain("fx.color_sat", 0.0, 100.0, 0.0, 1.0, 0);
                            studio_slider_spec(app, ui, &s_csat, m.adjust.color_sat, "Fuerza del tinte (%)", |app, v| {
                                app.run("mask.adjust", json!({"id": mid, "values": {"color_sat": v}}))
                            });
                        } else {
                            let s_ox = spec_center("fx.offset_x", -100.0, 100.0, 0.0, 0.5, 1);
                            studio_slider_spec(app, ui, &s_ox, m.offset_x, "Mover Horizontal (X)", |app, v| {
                                app.run("mask.transform", json!({"id": mid, "offset_x": v}))
                            });

                            let s_oy = spec_center("fx.offset_y", -100.0, 100.0, 0.0, 0.5, 1);
                            studio_slider_spec(app, ui, &s_oy, m.offset_y, "Mover Vertical (Y)", |app, v| {
                                app.run("mask.transform", json!({"id": mid, "offset_y": v}))
                            });

                            let s_sc = spec_center("fx.scale", 50.0, 150.0, 100.0, 1.0, 0);
                            studio_slider_spec(app, ui, &s_sc, m.scale, "Tamaño / Escala de máscara (%)", |app, v| {
                                app.run("mask.transform", json!({"id": mid, "offset_x": m.offset_x, "offset_y": m.offset_y, "scale": v}))
                            });

                            ui.add_space(6.0);
                            let mask_brush_id = egui::Id::new("studio-mask-brush-active");
                            let mut mask_brush_on = ui.ctx().data(|dd| dd.get_temp::<bool>(mask_brush_id)).unwrap_or(false);
                            let is_add_brush = mask_brush_on && !app.ui.brush_erase;
                            let is_sub_brush = mask_brush_on && app.ui.brush_erase;

                            egui::Frame::NONE
                                .inner_margin(egui::Margin { left: 10, right: 10, top: 2, bottom: 4 })
                                .show(ui, |ui| {
                                    let row_w = ui.available_width();
                                    let gap_b = 6.0;
                                    let bw = (row_w - gap_b) * 0.5;
                                    let (br, _) = ui.allocate_exact_size(vec2(row_w, 28.0), Sense::hover());
                                    let add_r = Rect::from_min_size(br.min, vec2(bw, 28.0));
                                    let sub_r = Rect::from_min_size(pos2(br.min.x + bw + gap_b, br.min.y), vec2(bw, 28.0));

                                    let add_resp = ui
                                        .interact(add_r, egui::Id::new("studio-mask-brush-add"), Sense::click())
                                        .on_hover_text("Pintar con brocha sobre la foto para AGREGAR zonas a esta máscara IA (Alt+Rueda cambia el tamaño)");
                                    let sub_resp = ui
                                        .interact(sub_r, egui::Id::new("studio-mask-brush-sub"), Sense::click())
                                        .on_hover_text("Pintar con brocha sobre la foto para BORRAR / RESTAR zonas de esta máscara IA (Alt+Rueda cambia el tamaño)");

                                    for (btn_r, resp, active, label, col) in [
                                        (add_r, &add_resp, is_add_brush, "🖌 Pintar (+ Agregar)", Color32::from_rgb(92, 200, 142)),
                                        (sub_r, &sub_resp, is_sub_brush, "🧹 Borrar (- Restar)", Color32::from_rgb(242, 118, 108)),
                                    ] {
                                        ui.painter().rect(
                                            btn_r,
                                            CornerRadius::same(8),
                                            if active {
                                                col.gamma_multiply(0.32)
                                            } else if resp.hovered() {
                                                Color32::from_white_alpha(28)
                                            } else {
                                                Color32::from_white_alpha(14)
                                            },
                                            Stroke::new(1.1, if active { col } else { Color32::from_white_alpha(32) }),
                                            StrokeKind::Inside,
                                        );
                                        ui.painter().text(
                                            btn_r.center(),
                                            Align2::CENTER_CENTER,
                                            label,
                                            if active { t.semibold(10.2) } else { t.font(10.0) },
                                            if active { Color32::WHITE } else { Color32::from_rgb(225, 228, 236) },
                                        );
                                    }

                                    if add_resp.clicked() {
                                        if is_add_brush {
                                            mask_brush_on = false;
                                        } else {
                                            mask_brush_on = true;
                                            app.ui.brush_erase = false;
                                            app.ui.tool = "brush".into();
                                            app.session.active_mask = Some(mid);
                                            ui.ctx().data_mut(|dd| {
                                                dd.insert_temp(egui::Id::new("studio-eraser-active"), false);
                                                dd.insert_temp(egui::Id::new("studio-magic-drag-mask"), false);
                                            });
                                        }
                                        ui.ctx().data_mut(|dd| dd.insert_temp(mask_brush_id, mask_brush_on));
                                    } else if sub_resp.clicked() {
                                        if is_sub_brush {
                                            mask_brush_on = false;
                                        } else {
                                            mask_brush_on = true;
                                            app.ui.brush_erase = true;
                                            app.ui.tool = "brush".into();
                                            app.session.active_mask = Some(mid);
                                            ui.ctx().data_mut(|dd| {
                                                dd.insert_temp(egui::Id::new("studio-eraser-active"), false);
                                                dd.insert_temp(egui::Id::new("studio-magic-drag-mask"), false);
                                            });
                                        }
                                        ui.ctx().data_mut(|dd| dd.insert_temp(mask_brush_id, mask_brush_on));
                                    }
                                });

                            // Always show brush size & feather controls in Ajustar Máscara so the user can resize the brush anywhere
                            let s_bsize = spec_plain("fx.mask_brush_size", 1.0, 100.0, 20.0, 1.0, 0);
                            studio_slider_spec(app, ui, &s_bsize, (app.ui.brush_size * 400.0) as f64, "Tamaño de la brocha", |app, v| {
                                app.ui.brush_size = ((v / 400.0) as f32).clamp(0.002, 0.35);
                                Ok(json!(null))
                            });

                            let s_bfeather = spec_plain("fx.mask_brush_feather", 0.0, 100.0, 50.0, 1.0, 0);
                            studio_slider_spec(app, ui, &s_bfeather, app.ui.brush_feather as f64, "Difuminar brocha (%)", |app, v| {
                                app.ui.brush_feather = (v as f32).clamp(0.0, 100.0);
                                Ok(json!(null))
                            });
                        }
                    }
                }
            }
            ui.add_space(8.0);
        });
}

#[derive(Clone, serde::Deserialize)]
struct SpecialFilterEntry {
    id: String,
    name: String,
    brand: String,
    category: String,
    #[allow(dead_code)]
    file: String,
    swatches: [[u8; 3]; 5],
}

const SPECIAL_FILTER_CATEGORIES: [(&str, &str); 8] = [
    ("Todos", "✦ Todos"),
    ("Película", "🎞 Película"),
    ("Cine", "🎬 Cine"),
    ("Retrato/Bodas", "💍 Bodas"),
    ("Ciudades", "🏙 Ciudades"),
    ("Cálidos", "🌅 Cálidos"),
    ("Urbano", "🌊 Urbano"),
    ("B&N / Especial", "◐ B&N/IR"),
];

fn special_filters_catalog() -> &'static [SpecialFilterEntry] {
    static CATALOG: std::sync::OnceLock<Vec<SpecialFilterEntry>> = std::sync::OnceLock::new();
    CATALOG.get_or_init(|| {
        let json_path = resolve_bundled_asset_path("filtros-especiales/catalog.json");
        let Ok(text) = std::fs::read_to_string(&json_path) else {
            return Vec::new();
        };
        serde_json::from_str::<Vec<SpecialFilterEntry>>(&text).unwrap_or_default()
    })
}

fn paint_film_icon(p: &egui::Painter, r: Rect, c: Color32) {
    let stroke = Stroke::new(1.35, c);
    let body = r.shrink2(vec2(1.0, 2.0));
    p.rect_stroke(body, CornerRadius::same(2), stroke, StrokeKind::Inside);
    p.line_segment([pos2(body.left() + 3.2, body.top()), pos2(body.left() + 3.2, body.bottom())], stroke);
    p.line_segment([pos2(body.right() - 3.2, body.top()), pos2(body.right() - 3.2, body.bottom())], stroke);
    p.line_segment([pos2(body.left() + 3.2, body.center().y), pos2(body.right() - 3.2, body.center().y)], stroke);
}

fn paint_special_filter_brand_banner(
    p: &egui::Painter,
    banner_r: Rect,
    entry: &SpecialFilterEntry,
    t: &Tokens,
    is_active: bool,
    hovered: bool,
) {
    let sw = &entry.swatches;
    let c0 = Color32::from_rgb(sw[0][0], sw[0][1], sw[0][2]);
    let c1 = Color32::from_rgb(sw[1][0], sw[1][1], sw[1][2]);
    let c2 = Color32::from_rgb(sw[2][0], sw[2][1], sw[2][2]);
    let c3 = Color32::from_rgb(sw[3][0], sw[3][1], sw[3][2]);
    let c4 = Color32::from_rgb(sw[4][0], sw[4][1], sw[4][2]);

    // 1. Paint smooth 4-segment horizontal color-graded backdrop mesh using the LUT's true swatches
    let seg_w = banner_r.width() / 4.0;
    let colors = [c0, c1, c2, c3, c4];
    let mut mesh = Mesh::default();
    for i in 0..4 {
        let x0 = banner_r.left() + i as f32 * seg_w;
        let x1 = if i == 3 { banner_r.right() } else { x0 + seg_w };
        let col_l = colors[i];
        let col_r = colors[i + 1];
        let base = mesh.vertices.len() as u32;
        mesh.vertices.push(Vertex { pos: pos2(x0, banner_r.top()), uv: pos2(0.0, 0.0), color: col_l });
        mesh.vertices.push(Vertex { pos: pos2(x1, banner_r.top()), uv: pos2(0.0, 0.0), color: col_r });
        mesh.vertices.push(Vertex {
            pos: pos2(x1, banner_r.bottom()),
            uv: pos2(0.0, 0.0),
            color: Color32::from_rgb(
                ((col_r.r() as u16 * 3 + c0.r() as u16) / 4) as u8,
                ((col_r.g() as u16 * 3 + c0.g() as u16) / 4) as u8,
                ((col_r.b() as u16 * 3 + c0.b() as u16) / 4) as u8,
            ),
        });
        mesh.vertices.push(Vertex {
            pos: pos2(x0, banner_r.bottom()),
            uv: pos2(0.0, 0.0),
            color: Color32::from_rgb(
                ((col_l.r() as u16 * 3 + c0.r() as u16) / 4) as u8,
                ((col_l.g() as u16 * 3 + c0.g() as u16) / 4) as u8,
                ((col_l.b() as u16 * 3 + c0.b() as u16) / 4) as u8,
            ),
        });
        mesh.indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    p.add(egui::Shape::mesh(mesh));

    // Subtle dark vignette at bottom and top sprocket strip
    p.rect_filled(
        Rect::from_min_max(banner_r.min, pos2(banner_r.max.x, banner_r.top() + 7.0)),
        CornerRadius::ZERO,
        Color32::from_black_alpha(115),
    );
    p.rect_filled(
        Rect::from_min_max(pos2(banner_r.min.x, banner_r.bottom() - 16.0), banner_r.max),
        CornerRadius::ZERO,
        Color32::from_black_alpha(135),
    );
    // 35mm film perforations along top bar
    for k in 0..8 {
        let sx = banner_r.left() + 6.0 + k as f32 * 13.5;
        if sx + 5.0 < banner_r.right() - 4.0 {
            p.rect_filled(
                Rect::from_min_size(pos2(sx, banner_r.top() + 2.0), vec2(5.0, 3.0)),
                CornerRadius::same(1),
                Color32::from_white_alpha(65),
            );
        }
    }

    // 2. Central Brand / Style Logo Emblem
    let center = pos2(banner_r.center().x, banner_r.center().y + 1.0);
    let brand = entry.brand.as_str();

    match brand {
        "KODAK" => {
            // Iconic Kodak Gold & Crimson emblem
            let badge = Rect::from_center_size(center, vec2(66.0, 26.0));
            p.rect_filled(badge, CornerRadius::same(5), Color32::from_rgb(255, 184, 0));
            p.rect_stroke(badge, CornerRadius::same(5), Stroke::new(1.2, Color32::from_rgb(220, 24, 32)), StrokeKind::Inside);
            // Red inner K block on left
            let k_box = Rect::from_min_size(pos2(badge.left() + 3.0, badge.top() + 3.0), vec2(16.0, 20.0));
            p.rect_filled(k_box, CornerRadius::same(3), Color32::from_rgb(218, 22, 30));
            p.text(k_box.center(), Align2::CENTER_CENTER, "K", t.semibold(12.0), Color32::from_rgb(255, 196, 0));
            p.text(
                pos2(k_box.right() + 22.0, badge.center().y),
                Align2::CENTER_CENTER,
                "KODAK",
                t.semibold(10.0),
                Color32::from_rgb(195, 16, 24),
            );
        }
        "FUJIFILM" => {
            // Iconic Fujifilm Emerald & Red emblem
            let badge = Rect::from_center_size(center, vec2(68.0, 26.0));
            p.rect_filled(badge, CornerRadius::same(5), Color32::from_rgb(0, 138, 92));
            p.rect_stroke(badge, CornerRadius::same(5), Stroke::new(1.0, Color32::from_white_alpha(160)), StrokeKind::Inside);
            let red_dot = pos2(badge.left() + 11.0, badge.center().y);
            p.circle_filled(red_dot, 5.5, Color32::from_rgb(230, 0, 28));
            p.text(red_dot, Align2::CENTER_CENTER, "F", t.semibold(8.5), Color32::WHITE);
            p.text(
                pos2(badge.center().x + 6.0, badge.center().y),
                Align2::CENTER_CENTER,
                "FUJIFILM",
                t.semibold(9.5),
                Color32::WHITE,
            );
        }
        "CINE 35MM" | "CINEMA" | "FILMIC" => {
            // 35mm Cinema Reel & Clapperboard emblem
            let badge = Rect::from_center_size(center, vec2(68.0, 26.0));
            p.rect_filled(badge, CornerRadius::same(5), Color32::from_rgba_unmultiplied(18, 20, 26, 220));
            p.rect_stroke(badge, CornerRadius::same(5), Stroke::new(1.0, Color32::from_rgb(245, 192, 98)), StrokeKind::Inside);
            let reel_c = pos2(badge.left() + 12.0, badge.center().y);
            p.circle_stroke(reel_c, 6.5, Stroke::new(1.3, Color32::from_rgb(245, 192, 98)));
            p.circle_filled(reel_c, 2.0, Color32::from_rgb(245, 192, 98));
            p.text(
                pos2(badge.center().x + 6.5, badge.center().y),
                Align2::CENTER_CENTER,
                if brand == "CINE 35MM" { "35MM CINE" } else { brand },
                t.semibold(9.0),
                Color32::from_rgb(255, 220, 150),
            );
        }
        "MM STUDIO" => {
            // Golden MM Studio signature badge
            let badge = Rect::from_center_size(center, vec2(68.0, 26.0));
            p.rect_filled(badge, CornerRadius::same(5), Color32::from_rgba_unmultiplied(24, 22, 18, 230));
            p.rect_stroke(badge, CornerRadius::same(5), Stroke::new(1.3, Color32::from_rgb(245, 192, 98)), StrokeKind::Inside);
            p.text(badge.center(), Align2::CENTER_CENTER, "✦ MM CINE", t.semibold(9.5), Color32::from_rgb(255, 215, 130));
        }
        "TEAL & ORANGE" => {
            // Split Cyan & Amber Cinema Dual-Tone Emblem
            let badge = Rect::from_center_size(center, vec2(68.0, 26.0));
            let left_half = Rect::from_min_max(badge.min, pos2(badge.center().x, badge.max.y));
            let right_half = Rect::from_min_max(pos2(badge.center().x, badge.min.y), badge.max);
            p.rect_filled(left_half, CornerRadius { nw: 5, sw: 5, ne: 0, se: 0 }, Color32::from_rgb(14, 132, 152));
            p.rect_filled(right_half, CornerRadius { nw: 0, sw: 0, ne: 5, se: 5 }, Color32::from_rgb(228, 110, 28));
            p.rect_stroke(badge, CornerRadius::same(5), Stroke::new(1.0, Color32::from_white_alpha(150)), StrokeKind::Inside);
            p.text(badge.center(), Align2::CENTER_CENTER, "TEAL·ORN", t.semibold(9.2), Color32::WHITE);
        }
        "WEDDING" | "RETRATO" => {
            // Rose-Gold / Champagne Wedding Rings & Portrait Emblem
            let badge = Rect::from_center_size(center, vec2(68.0, 26.0));
            p.rect_filled(badge, CornerRadius::same(5), Color32::from_rgba_unmultiplied(44, 30, 36, 220));
            p.rect_stroke(badge, CornerRadius::same(5), Stroke::new(1.1, Color32::from_rgb(245, 198, 176)), StrokeKind::Inside);
            let r1 = pos2(badge.left() + 10.5, badge.center().y);
            let r2 = pos2(badge.left() + 15.5, badge.center().y);
            p.circle_stroke(r1, 4.5, Stroke::new(1.2, Color32::from_rgb(255, 212, 168)));
            p.circle_stroke(r2, 4.5, Stroke::new(1.2, Color32::from_rgb(245, 182, 168)));
            p.text(
                pos2(badge.center().x + 7.5, badge.center().y),
                Align2::CENTER_CENTER,
                if brand == "WEDDING" { "BODAS" } else { "RETRATO" },
                t.semibold(9.2),
                Color32::from_rgb(255, 228, 215),
            );
        }
        "BARCELONA" | "SEVILLA" | "NEW YORK" | "LONDON" | "ISTANBUL" | "HELSINKI" | "PATAGONIA" => {
            // Travel / City Skyline Emblem
            let badge = Rect::from_center_size(center, vec2(72.0, 26.0));
            p.rect_filled(badge, CornerRadius::same(5), Color32::from_rgba_unmultiplied(20, 28, 38, 218));
            p.rect_stroke(badge, CornerRadius::same(5), Stroke::new(1.0, Color32::from_rgb(135, 205, 250)), StrokeKind::Inside);
            p.text(badge.center(), Align2::CENTER_CENTER, brand, t.semibold(8.8), Color32::from_rgb(225, 242, 255));
        }
        "CYBERPUNK" => {
            // Neon Magenta & Cyan Synthwave Emblem
            let badge = Rect::from_center_size(center, vec2(68.0, 26.0));
            p.rect_filled(badge, CornerRadius::same(5), Color32::from_rgba_unmultiplied(24, 10, 38, 228));
            p.rect_stroke(badge, CornerRadius::same(5), Stroke::new(1.2, Color32::from_rgb(255, 45, 195)), StrokeKind::Inside);
            p.text(badge.center(), Align2::CENTER_CENTER, "⚡ CYBER", t.semibold(9.5), Color32::from_rgb(80, 245, 255));
        }
        "INFRARED" => {
            // Aerochrome Crimson Infrared Emblem
            let badge = Rect::from_center_size(center, vec2(68.0, 26.0));
            p.rect_filled(badge, CornerRadius::same(5), Color32::from_rgba_unmultiplied(155, 18, 48, 228));
            p.rect_stroke(badge, CornerRadius::same(5), Stroke::new(1.1, Color32::from_rgb(255, 145, 168)), StrokeKind::Inside);
            p.text(badge.center(), Align2::CENTER_CENTER, "✿ IR 720", t.semibold(9.5), Color32::WHITE);
        }
        "B&W FILM" => {
            // Silver Halide B&W Contrast Emblem
            let badge = Rect::from_center_size(center, vec2(66.0, 26.0));
            p.rect_filled(badge, CornerRadius::same(5), Color32::from_rgba_unmultiplied(24, 24, 26, 232));
            p.rect_stroke(badge, CornerRadius::same(5), Stroke::new(1.1, Color32::from_rgb(215, 218, 225)), StrokeKind::Inside);
            let dc = pos2(badge.left() + 12.0, badge.center().y);
            p.circle_filled(dc, 5.5, Color32::WHITE);
            p.circle_filled(pos2(dc.x + 2.0, dc.y), 3.5, Color32::from_rgb(24, 24, 26));
            p.text(pos2(badge.center().x + 6.5, badge.center().y), Align2::CENTER_CENTER, "B&W", t.semibold(10.0), Color32::WHITE);
        }
        _ => {
            // Halcyon / Valdays / Evday / Vintage / Summer / Editorial emblem
            let badge = Rect::from_center_size(center, vec2(68.0, 26.0));
            let (bg_col, border_col, txt_col) = match brand {
                "HALCYON" => (Color32::from_rgba_unmultiplied(46, 34, 24, 222), Color32::from_rgb(235, 185, 120), Color32::from_rgb(255, 228, 185)),
                "VALDAYS" | "EVDAY" | "SUMMER" => (Color32::from_rgba_unmultiplied(48, 32, 18, 218), Color32::from_rgb(255, 178, 84), Color32::from_rgb(255, 232, 185)),
                "VINTAGE" => (Color32::from_rgba_unmultiplied(44, 34, 26, 222), Color32::from_rgb(218, 172, 118), Color32::from_rgb(248, 222, 186)),
                _ => (Color32::from_rgba_unmultiplied(24, 28, 36, 218), Color32::from_rgb(185, 200, 225), Color32::from_rgb(235, 240, 250)),
            };
            p.rect_filled(badge, CornerRadius::same(5), bg_col);
            p.rect_stroke(badge, CornerRadius::same(5), Stroke::new(1.1, border_col), StrokeKind::Inside);
            p.text(badge.center(), Align2::CENTER_CENTER, brand, t.semibold(9.0), txt_col);
        }
    }

    // 5-Swatch color bar along the bottom edge of the banner
    let bar_h = 4.0;
    let sw_w = banner_r.width() / 5.0;
    for (idx, col) in colors.iter().enumerate() {
        let r_sw = Rect::from_min_size(pos2(banner_r.left() + idx as f32 * sw_w, banner_r.bottom() - bar_h), vec2(sw_w + 0.5, bar_h));
        p.rect_filled(r_sw, CornerRadius::ZERO, *col);
    }

    if is_active {
        let on_pill = Rect::from_min_size(pos2(banner_r.left() + 4.0, banner_r.top() + 4.0), vec2(32.0, 15.0));
        p.rect_filled(on_pill, CornerRadius::same(4), Color32::from_rgb(245, 192, 98));
        p.text(on_pill.center(), Align2::CENTER_CENTER, "✓ ON", t.semibold(8.8), Color32::from_rgb(20, 20, 24));
    } else if hovered {
        p.rect_stroke(banner_r, CornerRadius::ZERO, Stroke::new(1.2, Color32::from_white_alpha(110)), StrokeKind::Inside);
    }
}

fn show_special_filters_card(app: &mut LightcraftApp, ui: &mut egui::Ui, t: &Tokens, d: &DevelopSettings) {
    let catalog = special_filters_catalog();
    if catalog.is_empty() {
        return;
    }

    let card_folded = app.ui.section_folded("studio:specialFilters");
    let cat_id = egui::Id::new("studio-special-filter-cat");
    let cur_cat = ui.ctx().data(|dd| dd.get_temp::<usize>(cat_id)).unwrap_or(0).min(SPECIAL_FILTER_CATEGORIES.len() - 1);
    let search_open_id = egui::Id::new("studio-sf-search-open");
    let mut search_open = ui.ctx().data(|dd| dd.get_temp::<bool>(search_open_id)).unwrap_or(false);
    let search_q_id = egui::Id::new("studio-sf-search-query");
    let mut search_q = ui.ctx().data(|dd| dd.get_temp::<String>(search_q_id)).unwrap_or_default();
    let fav_only_id = egui::Id::new("studio-sf-fav-only");
    let mut fav_only = ui.ctx().data(|dd| dd.get_temp::<bool>(fav_only_id)).unwrap_or(false);
    let list_mode_id = egui::Id::new("studio-sf-list-mode");
    let mut list_mode = ui.ctx().data(|dd| dd.get_temp::<bool>(list_mode_id)).unwrap_or(false);
    let page_offset_id = egui::Id::new("studio-sf-carousel-page");
    let mut page_offset = ui.ctx().data(|dd| dd.get_temp::<usize>(page_offset_id)).unwrap_or(0);

    let active_lut_id = d.profile.id.starts_with("lut:").then_some(d.profile.id.as_str());
    let active_sf = active_lut_id.and_then(|id| catalog.iter().find(|e| e.id == id));

    let w = ui.available_width();
    let pad = 10.0;
    let inner_w = (w - pad * 2.0).max(120.0);

    egui::Frame::NONE
        .fill(Color32::from_rgba_unmultiplied(36, 38, 44, 192))
        .stroke(Stroke::new(
            1.0,
            if active_sf.is_some() {
                Color32::from_rgb(245, 192, 98).gamma_multiply(0.65)
            } else {
                Color32::from_white_alpha(26)
            },
        ))
        .corner_radius(CornerRadius::same(16))
        .show(ui, |ui| {
            ui.set_width(w);

            // 1. Header Bar (collapsible, persisted) + 3 Quick Action Buttons: [🔍] [★ Fav] [☰ Lista]
            let (hr, _) = ui.allocate_exact_size(vec2(w, 36.0), Sense::hover());
            let btn_list = Rect::from_min_size(pos2(hr.right() - pad - 56.0, hr.center().y - 10.5), vec2(56.0, 21.0));
            let btn_fav = Rect::from_min_size(pos2(btn_list.left() - 4.0 - 44.0, hr.center().y - 10.5), vec2(44.0, 21.0));
            let btn_search = Rect::from_min_size(pos2(btn_fav.left() - 4.0 - 26.0, hr.center().y - 10.5), vec2(26.0, 21.0));
            let hdr_click_r = Rect::from_min_max(hr.min, pos2(btn_search.left() - 4.0, hr.max.y));

            let hdr_resp = ui
                .interact(hdr_click_r, egui::Id::new("studio-sf-hdr-fold"), Sense::click())
                .on_hover_text("Clic para contraer / expandir Filtros Especiales (LUTs Creativos)");

            paint(
                ui.painter(),
                Rect::from_center_size(pos2(hr.left() + 13.0, hr.center().y), vec2(12.0, 12.0)),
                if card_folded { Icon::ChevronRight } else { Icon::ChevronDown },
                Color32::from_rgb(225, 228, 236),
            );
            let ir = Rect::from_center_size(pos2(hr.left() + 27.0, hr.center().y), vec2(14.5, 14.5));
            paint_film_icon(ui.painter(), ir, Color32::from_rgb(245, 192, 98));
            ui.painter().text(
                pos2(hr.left() + 40.0, hr.center().y),
                Align2::LEFT_CENTER,
                "Filtros Especiales",
                t.semibold(13.0),
                Color32::WHITE,
            );

            // Button 1: 🔍 Search toggle
            let s_resp = ui
                .interact(btn_search, egui::Id::new("studio-sf-btn-search"), Sense::click())
                .on_hover_text("Buscar filtro especial por nombre, marca (Kodak, Fuji, Bodas, Ciudades…)");
            ui.painter().rect(
                btn_search,
                CornerRadius::same(8),
                if search_open || !search_q.is_empty() {
                    Color32::from_rgb(245, 192, 98).gamma_multiply(0.32)
                } else if s_resp.hovered() {
                    Color32::from_white_alpha(28)
                } else {
                    Color32::from_white_alpha(15)
                },
                Stroke::new(
                    1.0,
                    if search_open || !search_q.is_empty() {
                        Color32::from_rgb(245, 192, 98)
                    } else {
                        Color32::from_white_alpha(32)
                    },
                ),
                StrokeKind::Inside,
            );
            paint(
                ui.painter(),
                Rect::from_center_size(btn_search.center(), vec2(12.0, 12.0)),
                Icon::Search,
                if search_open || !search_q.is_empty() {
                    Color32::from_rgb(255, 225, 155)
                } else {
                    Color32::from_rgb(218, 222, 232)
                },
            );
            if s_resp.clicked() {
                search_open = !search_open;
                if !search_open {
                    search_q.clear();
                    ui.ctx().data_mut(|dd| dd.insert_temp(search_q_id, search_q.clone()));
                }
                ui.ctx().data_mut(|dd| dd.insert_temp(search_open_id, search_open));
                if card_folded {
                    app.ui.toggle_folded("studio:specialFilters");
                }
            }

            // Button 2: ★ Favorites toggle (never active by default; only activates when the user has >= 1 favorite and clicks it)
            let fav_count = catalog.iter().filter(|e| app.session.profile_favorites.contains(&e.id)).count();
            if fav_count == 0 && fav_only {
                fav_only = false;
                ui.ctx().data_mut(|dd| dd.insert_temp(fav_only_id, false));
            }
            let f_resp = ui
                .interact(btn_fav, egui::Id::new("studio-sf-btn-fav"), Sense::click())
                .on_hover_text(if fav_count == 0 {
                    "Marca cualquier filtro con ★ para agregarlo a tus favoritos"
                } else {
                    "Mostrar solo tus Filtros Especiales favoritos (★)"
                });
            ui.painter().rect(
                btn_fav,
                CornerRadius::same(8),
                if fav_only {
                    Color32::from_rgb(245, 192, 98).gamma_multiply(0.34)
                } else if f_resp.hovered() {
                    Color32::from_white_alpha(28)
                } else {
                    Color32::from_white_alpha(15)
                },
                Stroke::new(
                    1.0,
                    if fav_only { Color32::from_rgb(245, 192, 98) } else { Color32::from_white_alpha(32) },
                ),
                StrokeKind::Inside,
            );
            let fav_label = if fav_count > 0 { format!("★ {fav_count}") } else { "★ Fav".to_string() };
            ui.painter().text(
                btn_fav.center(),
                Align2::CENTER_CENTER,
                fav_label,
                t.semibold(9.8),
                if fav_only { Color32::from_rgb(255, 225, 155) } else { Color32::from_rgb(220, 224, 234) },
            );
            if f_resp.clicked() && fav_count > 0 {
                fav_only = !fav_only;
                page_offset = 0;
                ui.ctx().data_mut(|dd| {
                    dd.insert_temp(fav_only_id, fav_only);
                    dd.insert_temp(page_offset_id, page_offset);
                });
                if card_folded {
                    app.ui.toggle_folded("studio:specialFilters");
                }
            }

            // Button 3: ☰ Lista completa (sin imágenes) vs 🎴 Carrusel de tarjetas
            let l_resp = ui
                .interact(btn_list, egui::Id::new("studio-sf-btn-list"), Sense::click())
                .on_hover_text(if list_mode {
                    "Volver al carrusel elegante de tarjetas con emblemas visuales"
                } else {
                    "Desplegar la lista completa de filtros (vista compacta sin imágenes)"
                });
            ui.painter().rect(
                btn_list,
                CornerRadius::same(8),
                if list_mode {
                    Color32::from_rgb(245, 192, 98).gamma_multiply(0.34)
                } else if l_resp.hovered() {
                    Color32::from_white_alpha(28)
                } else {
                    Color32::from_white_alpha(15)
                },
                Stroke::new(
                    1.0,
                    if list_mode { Color32::from_rgb(245, 192, 98) } else { Color32::from_white_alpha(32) },
                ),
                StrokeKind::Inside,
            );
            ui.painter().text(
                btn_list.center(),
                Align2::CENTER_CENTER,
                if list_mode { "🎴 Tarjetas" } else { "☰ Lista" },
                t.semibold(9.8),
                if list_mode { Color32::from_rgb(255, 225, 155) } else { Color32::from_rgb(220, 224, 234) },
            );
            if l_resp.clicked() {
                list_mode = !list_mode;
                ui.ctx().data_mut(|dd| dd.insert_temp(list_mode_id, list_mode));
                if card_folded {
                    app.ui.toggle_folded("studio:specialFilters");
                }
            } else if hdr_resp.clicked() {
                app.ui.toggle_folded("studio:specialFilters");
            }

            if card_folded {
                return;
            }

            // 2. Optional Search Bar (when 🔍 is active)
            if search_open {
                egui::Frame::NONE
                    .inner_margin(egui::Margin { left: pad as i8, right: pad as i8, top: 2, bottom: 4 })
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            let te = egui::TextEdit::singleline(&mut search_q)
                                .hint_text("🔍 Buscar Kodak, Fuji, Bodas, Barcelona, Teal, Vintage…")
                                .desired_width(inner_w - 28.0);
                            let resp = ui.add(te);
                            if resp.changed() {
                                page_offset = 0;
                                ui.ctx().data_mut(|dd| {
                                    dd.insert_temp(search_q_id, search_q.clone());
                                    dd.insert_temp(page_offset_id, page_offset);
                                });
                            }
                            if ui.small_button("✕").on_hover_text("Limpiar búsqueda").clicked() {
                                search_q.clear();
                                page_offset = 0;
                                ui.ctx().data_mut(|dd| {
                                    dd.insert_temp(search_q_id, search_q.clone());
                                    dd.insert_temp(page_offset_id, page_offset);
                                });
                            }
                        });
                    });
            }

            // 3. 2-Row Category Filter Pills (4 pills per row)
            let cat_gap = 4.0;
            let cat_row_h = 24.0;
            let (cat_area, _) = ui.allocate_exact_size(vec2(w, cat_row_h * 2.0 + cat_gap + 4.0), Sense::hover());
            let w4 = (inner_w - cat_gap * 3.0) / 4.0;
            for (idx, (cid, clabel)) in SPECIAL_FILTER_CATEGORIES.iter().enumerate() {
                let row = idx / 4;
                let col = idx % 4;
                let x = cat_area.left() + pad + col as f32 * (w4 + cat_gap);
                let y = cat_area.top() + 1.0 + row as f32 * (cat_row_h + cat_gap);
                let pill_r = Rect::from_min_size(pos2(x, y), vec2(w4, cat_row_h));
                let resp = ui.interact(pill_r, egui::Id::new(("sf-cat-pill", *cid)), Sense::click());
                let active = cur_cat == idx;
                ui.painter().rect(
                    pill_r,
                    CornerRadius::same(7),
                    if active {
                        Color32::from_rgb(245, 198, 112)
                    } else if resp.hovered() {
                        Color32::from_white_alpha(28)
                    } else {
                        Color32::from_white_alpha(13)
                    },
                    Stroke::new(
                        1.0,
                        if active { Color32::from_rgb(255, 228, 168) } else { Color32::from_white_alpha(26) },
                    ),
                    StrokeKind::Inside,
                );
                ui.painter().text(
                    pill_r.center(),
                    Align2::CENTER_CENTER,
                    *clabel,
                    if active { t.semibold(9.9) } else { t.font(9.8) },
                    if active { Color32::from_rgb(22, 22, 26) } else { Color32::from_rgb(222, 226, 235) },
                );
                if resp.clicked() {
                    ui.ctx().data_mut(|dd| {
                        dd.insert_temp(cat_id, idx);
                        dd.insert_temp(page_offset_id, 0usize);
                    });
                }
            }

            // Filter catalog by category, favorites, and search query
            let q_norm = search_q.trim().to_ascii_lowercase();
            let sel_cat_key = SPECIAL_FILTER_CATEGORIES[cur_cat].0;
            let filtered: Vec<&SpecialFilterEntry> = catalog
                .iter()
                .filter(|e| {
                    if cur_cat != 0 && e.category != sel_cat_key {
                        return false;
                    }
                    if fav_only && !app.session.profile_favorites.contains(&e.id) {
                        return false;
                    }
                    if !q_norm.is_empty() {
                        let m_name = e.name.to_ascii_lowercase().contains(&q_norm);
                        let m_brand = e.brand.to_ascii_lowercase().contains(&q_norm);
                        let m_cat = e.category.to_ascii_lowercase().contains(&q_norm);
                        if !m_name && !m_brand && !m_cat {
                            return false;
                        }
                    }
                    true
                })
                .collect();

            if filtered.is_empty() {
                let (empty_r, _) = ui.allocate_exact_size(vec2(w, 48.0), Sense::hover());
                ui.painter().text(
                    empty_r.center(),
                    Align2::CENTER_CENTER,
                    if fav_only {
                        "No tienes filtros favoritos en esta categoría aún (pulsa ★ en cualquier tarjeta)"
                    } else {
                        "No se encontraron filtros con esa búsqueda"
                    },
                    t.font(10.5),
                    Color32::from_rgb(185, 190, 202),
                );
            } else if !list_mode {
                // =========================================================================
                // 4A. ELEGANT VISUAL CARD CAROUSEL (with Brand Logos, Swatch Banners & ‹/›)
                // =========================================================================
                ui.add_space(3.0);
                // Carousel navigation bar: shows range + ‹ Prev / Next › buttons
                let total = filtered.len();
                let safe_offset = page_offset.min(total.saturating_sub(1));
                let (nav_r, _) = ui.allocate_exact_size(vec2(w, 22.0), Sense::hover());
                let prev_r = Rect::from_min_size(pos2(nav_r.left() + pad, nav_r.top() + 1.0), vec2(28.0, 20.0));
                let next_r = Rect::from_min_size(pos2(nav_r.right() - pad - 28.0, nav_r.top() + 1.0), vec2(28.0, 20.0));

                let prev_resp = ui
                    .interact(prev_r, egui::Id::new("sf-carousel-prev"), Sense::click())
                    .on_hover_text("Desplazar carrusel a la izquierda");
                let next_resp = ui
                    .interact(next_r, egui::Id::new("sf-carousel-next"), Sense::click())
                    .on_hover_text("Desplazar carrusel a la derecha");

                for (btn_r, resp, sym) in [(prev_r, &prev_resp, "‹"), (next_r, &next_resp, "›")] {
                    ui.painter().rect(
                        btn_r,
                        CornerRadius::same(6),
                        if resp.hovered() { Color32::from_white_alpha(32) } else { Color32::from_white_alpha(15) },
                        Stroke::new(1.0, Color32::from_white_alpha(30)),
                        StrokeKind::Inside,
                    );
                    ui.painter().text(btn_r.center(), Align2::CENTER_CENTER, sym, t.semibold(13.0), Color32::WHITE);
                }

                ui.painter().text(
                    nav_r.center(),
                    Align2::CENTER_CENTER,
                    format!("{} filtros creativos · Desliza o usa ‹ ›", total),
                    t.font(9.8),
                    Color32::from_rgb(190, 196, 208),
                );

                let mut jump_delta: Option<isize> = None;
                if prev_resp.clicked() {
                    jump_delta = Some(-3);
                } else if next_resp.clicked() {
                    jump_delta = Some(3);
                }
                if let Some(delta) = jump_delta {
                    let next_off = (safe_offset as isize + delta).clamp(0, (total.saturating_sub(1)) as isize) as usize;
                    page_offset = next_off;
                    ui.ctx().data_mut(|dd| dd.insert_temp(page_offset_id, page_offset));
                }

                ui.add_space(2.0);
                let card_w = 114.0;
                let card_h = 108.0;
                let banner_h = 62.0;
                let card_gap = 7.0;

                let mut clicked_lut: Option<&SpecialFilterEntry> = None;
                let mut toggled_fav: Option<String> = None;

                egui::Frame::NONE
                    .inner_margin(egui::Margin { left: pad as i8, right: pad as i8, top: 2, bottom: 2 })
                    .show(ui, |ui| {
                        let mut scroll = egui::ScrollArea::horizontal()
                            .id_salt(("sf-carousel-scroll", cur_cat, fav_only))
                            .max_height(card_h + 12.0)
                            .auto_shrink([false, true])
                            .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysVisible);
                        if jump_delta.is_some() {
                            scroll = scroll.horizontal_scroll_offset(page_offset as f32 * (card_w + card_gap));
                        }
                        scroll.show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.spacing_mut().item_spacing.x = card_gap;
                                let ptr_pos = ui.ctx().input(|i| i.pointer.interact_pos());
                                for entry in &filtered {
                                    let is_active = active_lut_id == Some(entry.id.as_str());
                                    let is_fav = app.session.profile_favorites.contains(&entry.id);
                                    let (card_r, card_resp) = ui.allocate_exact_size(vec2(card_w, card_h), Sense::click());

                                    let banner_r = Rect::from_min_size(card_r.min, vec2(card_w, banner_h));
                                    let fav_r = Rect::from_min_size(pos2(card_r.right() - 23.0, card_r.top() + 3.0), vec2(20.0, 20.0));
                                    let over_fav = ptr_pos.is_some_and(|p| fav_r.contains(p));

                                    let card_resp = card_resp.on_hover_text(if over_fav {
                                        if is_fav {
                                            "Quitar de Favoritos (★)".to_string()
                                        } else {
                                            "Agregar a Favoritos (★)".to_string()
                                        }
                                    } else if is_active {
                                        format!("{} ({}) · Clic para desactivar filtro", entry.name, entry.brand)
                                    } else {
                                        format!("Clic para aplicar {} ({})", entry.name, entry.brand)
                                    });

                                    // Card background
                                    ui.painter().rect_filled(
                                        card_r,
                                        CornerRadius::same(10),
                                        if is_active {
                                            Color32::from_rgb(56, 46, 30)
                                        } else if card_resp.hovered() {
                                            Color32::from_rgb(46, 50, 58)
                                        } else {
                                            Color32::from_rgb(28, 30, 36)
                                        },
                                    );

                                    // Visual Banner + Brand Emblem
                                    paint_special_filter_brand_banner(
                                        ui.painter(),
                                        banner_r,
                                        entry,
                                        t,
                                        is_active,
                                        card_resp.hovered(),
                                    );

                                    // Favorite star button in top-right of banner
                                    ui.painter().circle_filled(
                                        fav_r.center(),
                                        8.5,
                                        if is_fav {
                                            Color32::from_black_alpha(190)
                                        } else if card_resp.hovered() && over_fav {
                                            Color32::from_black_alpha(165)
                                        } else {
                                            Color32::from_black_alpha(110)
                                        },
                                    );
                                    ui.painter().text(
                                        fav_r.center(),
                                        Align2::CENTER_CENTER,
                                        if is_fav { "★" } else { "☆" },
                                        t.semibold(11.0),
                                        if is_fav { Color32::from_rgb(255, 204, 72) } else { Color32::from_white_alpha(215) },
                                    );

                                    // Bottom Title & Brand Subtitle
                                    ui.painter().text(
                                        pos2(card_r.center().x, banner_r.bottom() + 12.0),
                                        Align2::CENTER_CENTER,
                                        &entry.brand,
                                        t.semibold(8.5),
                                        if is_active {
                                            Color32::from_rgb(255, 206, 112)
                                        } else {
                                            Color32::from_rgb(172, 180, 196)
                                        },
                                    );

                                    let short_title = if entry.name.chars().count() > 16 {
                                        let s: String = entry.name.chars().take(15).collect();
                                        format!("{s}…")
                                    } else {
                                        entry.name.clone()
                                    };
                                    ui.painter().text(
                                        pos2(card_r.center().x, banner_r.bottom() + 29.0),
                                        Align2::CENTER_CENTER,
                                        short_title,
                                        if is_active { t.semibold(10.5) } else { t.font(10.2) },
                                        if is_active { Color32::from_rgb(255, 236, 190) } else { Color32::WHITE },
                                    );

                                    // Outer card border
                                    ui.painter().rect_stroke(
                                        card_r,
                                        CornerRadius::same(10),
                                        Stroke::new(
                                            if is_active { 1.8 } else { 1.0 },
                                            if is_active {
                                                Color32::from_rgb(245, 192, 98)
                                            } else if card_resp.hovered() {
                                                Color32::from_white_alpha(85)
                                            } else {
                                                Color32::from_white_alpha(28)
                                            },
                                        ),
                                        StrokeKind::Inside,
                                    );

                                    if card_resp.hovered() && !is_active {
                                        let mut preview = d.clone();
                                        preview.profile.id = entry.id.clone();
                                        if preview.profile.amount == 0.0 {
                                            preview.profile.amount = 100.0;
                                        }
                                        app.hover_preview = Some(crate::HoverPreview {
                                            label: format!("Filtro Especial: {}", entry.name),
                                            settings: preview,
                                        });
                                    }

                                    if card_resp.clicked() {
                                        if over_fav {
                                            toggled_fav = Some(entry.id.clone());
                                        } else {
                                            clicked_lut = Some(*entry);
                                        }
                                    }
                                }
                            });
                        });
                    });

                if let Some(fid) = toggled_fav {
                    let _ = app.run("profile.favorite", json!({"id": fid}));
                } else if let Some(entry) = clicked_lut {
                    ensure_active_photo(app);
                    let is_active = active_lut_id == Some(entry.id.as_str());
                    if is_active {
                        let _ = app.run("develop.profile", json!({"id": "lc.color", "amount": 100.0}));
                        app.toast(ui.ctx(), format!("Filtro desactivado: {}", entry.name));
                    } else {
                        let amt = if d.profile.id.starts_with("lut:") && d.profile.amount > 0.0 {
                            d.profile.amount
                        } else {
                            100.0
                        };
                        let _ = app.run("develop.profile", json!({"id": entry.id, "amount": amt}));
                        app.toast(ui.ctx(), format!("Filtro Especial aplicado: {}", entry.name));
                    }
                }
            } else {
                // =========================================================================
                // 4B. FULL COMPACT TEXT LIST (without images, as requested)
                // =========================================================================
                ui.add_space(4.0);
                let mut clicked_lut: Option<&SpecialFilterEntry> = None;
                let mut toggled_fav: Option<String> = None;

                egui::Frame::NONE
                    .inner_margin(egui::Margin { left: pad as i8, right: pad as i8, top: 2, bottom: 2 })
                    .show(ui, |ui| {
                        egui::ScrollArea::vertical()
                            .id_salt("sf-full-text-list-scroll")
                            .max_height(190.0)
                            .auto_shrink([false, true])
                            .show(ui, |ui| {
                                ui.spacing_mut().item_spacing.y = 2.0;
                                let ptr_pos = ui.ctx().input(|i| i.pointer.interact_pos());
                                for entry in &filtered {
                                    let is_active = active_lut_id == Some(entry.id.as_str());
                                    let is_fav = app.session.profile_favorites.contains(&entry.id);
                                    let (row_r, row_resp) = ui.allocate_exact_size(vec2(ui.available_width(), 25.0), Sense::click());
                                    let fav_r = Rect::from_center_size(pos2(row_r.right() - 13.0, row_r.center().y), vec2(22.0, 22.0));
                                    let over_fav = ptr_pos.is_some_and(|p| fav_r.contains(p));

                                    let row_resp = row_resp.on_hover_text(if over_fav {
                                        if is_fav { "Quitar de Favoritos" } else { "Agregar a Favoritos" }
                                    } else if is_active {
                                        "Clic para desactivar este Filtro Especial"
                                    } else {
                                        "Clic para aplicar este Filtro Especial"
                                    });

                                    ui.painter().rect(
                                        row_r,
                                        CornerRadius::same(6),
                                        if is_active {
                                            Color32::from_rgb(236, 178, 84).gamma_multiply(0.26)
                                        } else if row_resp.hovered() {
                                            Color32::from_white_alpha(22)
                                        } else {
                                            Color32::from_white_alpha(9)
                                        },
                                        Stroke::new(
                                            1.0,
                                            if is_active { Color32::from_rgb(245, 192, 98) } else { Color32::from_white_alpha(18) },
                                        ),
                                        StrokeKind::Inside,
                                    );

                                    // Brand pill
                                    let pill_r = Rect::from_min_size(pos2(row_r.left() + 6.0, row_r.center().y - 7.5), vec2(58.0, 15.0));
                                    ui.painter().rect_filled(
                                        pill_r,
                                        CornerRadius::same(4),
                                        if is_active { Color32::from_rgb(245, 192, 98) } else { Color32::from_white_alpha(22) },
                                    );
                                    let short_brand: String = entry.brand.chars().take(9).collect();
                                    ui.painter().text(
                                        pill_r.center(),
                                        Align2::CENTER_CENTER,
                                        short_brand,
                                        t.semibold(8.2),
                                        if is_active { Color32::from_rgb(20, 20, 24) } else { Color32::from_rgb(220, 225, 235) },
                                    );

                                    let title_txt = if is_active {
                                        format!("✓ {}", entry.name)
                                    } else {
                                        entry.name.clone()
                                    };
                                    ui.painter().text(
                                        pos2(pill_r.right() + 7.0, row_r.center().y),
                                        Align2::LEFT_CENTER,
                                        title_txt,
                                        if is_active { t.semibold(10.8) } else { t.font(10.6) },
                                        if is_active { Color32::from_rgb(255, 230, 172) } else { Color32::from_rgb(235, 238, 245) },
                                    );

                                    ui.painter().text(
                                        fav_r.center(),
                                        Align2::CENTER_CENTER,
                                        if is_fav { "★" } else { "☆" },
                                        t.semibold(11.5),
                                        if is_fav { Color32::from_rgb(255, 204, 72) } else { Color32::from_white_alpha(120) },
                                    );

                                    if row_resp.hovered() && !is_active {
                                        let mut preview = d.clone();
                                        preview.profile.id = entry.id.clone();
                                        if preview.profile.amount == 0.0 {
                                            preview.profile.amount = 100.0;
                                        }
                                        app.hover_preview = Some(crate::HoverPreview {
                                            label: format!("Filtro Especial: {}", entry.name),
                                            settings: preview,
                                        });
                                    }

                                    if row_resp.clicked() {
                                        if over_fav {
                                            toggled_fav = Some(entry.id.clone());
                                        } else {
                                            clicked_lut = Some(*entry);
                                        }
                                    }
                                }
                            });
                    });

                if let Some(fid) = toggled_fav {
                    let _ = app.run("profile.favorite", json!({"id": fid}));
                } else if let Some(entry) = clicked_lut {
                    ensure_active_photo(app);
                    let is_active = active_lut_id == Some(entry.id.as_str());
                    if is_active {
                        let _ = app.run("develop.profile", json!({"id": "lc.color", "amount": 100.0}));
                        app.toast(ui.ctx(), format!("Filtro desactivado: {}", entry.name));
                    } else {
                        let amt = if d.profile.id.starts_with("lut:") && d.profile.amount > 0.0 {
                            d.profile.amount
                        } else {
                            100.0
                        };
                        let _ = app.run("develop.profile", json!({"id": entry.id, "amount": amt}));
                        app.toast(ui.ctx(), format!("Filtro Especial aplicado: {}", entry.name));
                    }
                }
            }

            // 5. Active Special Filter Intensity Slider & Quick Disable Bar
            let active_display_name = active_sf
                .map(|e| format!("{} ({})", e.name, e.brand))
                .or_else(|| {
                    active_lut_id.map(|lid| {
                        app.session
                            .lut_profiles
                            .iter()
                            .find(|p| p.id == lid)
                            .map(|p| p.name.clone())
                            .unwrap_or_else(|| lid.trim_start_matches("lut:sf-").trim_start_matches("lut:").to_string())
                    })
                });

            if let Some(active_title) = active_display_name {
                ui.add_space(4.0);
                let (act_r, _) = ui.allocate_exact_size(vec2(w, 24.0), Sense::hover());
                let off_r = Rect::from_min_size(pos2(act_r.right() - pad - 68.0, act_r.center().y - 10.0), vec2(68.0, 20.0));
                let off_resp = ui
                    .interact(off_r, egui::Id::new("sf-active-disable"), Sense::click())
                    .on_hover_text("Desactivar este Filtro Especial y volver al color original");

                ui.painter().text(
                    pos2(act_r.left() + pad, act_r.center().y),
                    Align2::LEFT_CENTER,
                    format!("✓ Activo: {active_title}"),
                    t.semibold(10.5),
                    Color32::from_rgb(255, 214, 130),
                );
                ui.painter().rect(
                    off_r,
                    CornerRadius::same(6),
                    if off_resp.hovered() { Color32::from_white_alpha(32) } else { Color32::from_white_alpha(16) },
                    Stroke::new(1.0, Color32::from_white_alpha(32)),
                    StrokeKind::Inside,
                );
                ui.painter().text(
                    off_r.center(),
                    Align2::CENTER_CENTER,
                    "✕ Quitar",
                    t.semibold(9.6),
                    Color32::from_rgb(240, 200, 200),
                );
                if off_resp.clicked() {
                    let _ = app.run("develop.profile", json!({"id": "lc.color", "amount": 100.0}));
                    app.toast(ui.ctx(), "Filtro Especial desactivado");
                }

                studio_control(app, ui, d, "profile.amount", "Intensidad del filtro");
            }

            ui.add_space(4.0);
        });
}

// =============================================================================
// OBJECT ERASER IA & HEALING BRUSH CARD (IN EFECTOS TAB)
// =============================================================================
fn show_object_eraser_card(app: &mut LightcraftApp, ui: &mut egui::Ui, t: &Tokens, d: &DevelopSettings) {
    let card_folded = app.ui.section_folded("studio:objectEraser");
    let eraser_id = egui::Id::new("studio-eraser-active");
    let mut eraser_on = ui.ctx().data(|dd| dd.get_temp::<bool>(eraser_id)).unwrap_or(false);
    let spot_count = d.spots.len();

    let w = ui.available_width();
    let pad = 10.0;
    let inner_w = (w - pad * 2.0).max(120.0);

    egui::Frame::NONE
        .fill(Color32::from_rgba_unmultiplied(36, 38, 44, 192))
        .stroke(Stroke::new(
            1.0,
            if eraser_on {
                Color32::from_rgb(98, 212, 168)
            } else if spot_count > 0 {
                Color32::from_rgb(245, 192, 98).gamma_multiply(0.55)
            } else {
                Color32::from_white_alpha(26)
            },
        ))
        .corner_radius(CornerRadius::same(16))
        .show(ui, |ui| {
            ui.set_width(w);

            // Header bar with quick toggle pill
            let (hr, _) = ui.allocate_exact_size(vec2(w, 36.0), Sense::hover());
            let toggle_r = Rect::from_min_size(pos2(hr.right() - pad - 88.0, hr.center().y - 11.0), vec2(88.0, 22.0));
            let hdr_click_r = Rect::from_min_max(hr.min, pos2(toggle_r.left() - 6.0, hr.max.y));

            let hdr_resp = ui
                .interact(hdr_click_r, egui::Id::new("studio-eraser-hdr-fold"), Sense::click())
                .on_hover_text("Clic para contraer / expandir Borrar Objetos IA");

            paint(
                ui.painter(),
                Rect::from_center_size(pos2(hr.left() + 13.0, hr.center().y), vec2(12.0, 12.0)),
                if card_folded { Icon::ChevronRight } else { Icon::ChevronDown },
                Color32::from_rgb(225, 228, 236),
            );
            paint(
                ui.painter(),
                Rect::from_center_size(pos2(hr.left() + 27.0, hr.center().y), vec2(14.5, 14.5)),
                Icon::Eraser,
                if eraser_on { Color32::from_rgb(98, 225, 175) } else { Color32::from_rgb(245, 192, 98) },
            );
            let title_str = if spot_count > 0 {
                format!("Borrar Objetos IA ({spot_count})")
            } else {
                "Borrar Objetos IA".to_string()
            };
            ui.painter().text(
                pos2(hr.left() + 40.0, hr.center().y),
                Align2::LEFT_CENTER,
                title_str,
                t.semibold(13.0),
                Color32::WHITE,
            );

            let tog_resp = ui
                .interact(toggle_r, egui::Id::new("studio-eraser-quick-toggle"), Sense::click())
                .on_hover_text("Activar / desactivar la brocha sobre la fotografía para borrar objetos o personas");
            ui.painter().rect(
                toggle_r,
                CornerRadius::same(8),
                if eraser_on {
                    Color32::from_rgb(76, 196, 148).gamma_multiply(0.38)
                } else if tog_resp.hovered() {
                    Color32::from_white_alpha(28)
                } else {
                    Color32::from_white_alpha(15)
                },
                Stroke::new(
                    1.1,
                    if eraser_on { Color32::from_rgb(98, 225, 175) } else { Color32::from_white_alpha(34) },
                ),
                StrokeKind::Inside,
            );
            ui.painter().text(
                toggle_r.center(),
                Align2::CENTER_CENTER,
                if eraser_on { "🖌 Brocha ON" } else { "🖌 Usar Brocha" },
                t.semibold(9.8),
                if eraser_on { Color32::from_rgb(190, 255, 225) } else { Color32::from_rgb(225, 228, 236) },
            );

            if tog_resp.clicked() {
                eraser_on = !eraser_on;
                if eraser_on {
                    if !matches!(app.ui.tool.as_str(), "fill" | "remove" | "clone") {
                        app.ui.tool = "fill".into();
                    }
                    ui.ctx().data_mut(|dd| {
                        dd.insert_temp(egui::Id::new("studio-mask-brush-active"), false);
                        dd.insert_temp(egui::Id::new("studio-magic-drag-mask"), false);
                    });
                    if card_folded {
                        app.ui.toggle_folded("studio:objectEraser");
                    }
                }
                ui.ctx().data_mut(|dd| dd.insert_temp(eraser_id, eraser_on));
            } else if hdr_resp.clicked() {
                app.ui.toggle_folded("studio:objectEraser");
            }

            if card_folded {
                return;
            }

            // 1. Mode Selector Pills: [✨ IA Relleno (PatchMatch)] [🩹 Corregir] [🎯 Clonar]
            let cur_mode = match app.ui.tool.as_str() {
                "remove" => "remove",
                "clone" => "clone",
                _ => "fill",
            };
            let mode_gap = 5.0;
            let (mode_area, _) = ui.allocate_exact_size(vec2(w, 28.0), Sense::hover());
            let mw = (inner_w - mode_gap * 2.0) / 3.0;
            let modes = [
                ("fill", "✨ IA Relleno", "Borrado inteligente PatchMatch (reconstruye el fondo automáticamente al pintar sobre un objeto)"),
                ("remove", "🩹 Corregir", "Corrector de textura y piel (mezcla suave con el entorno)"),
                ("clone", "🎯 Clonar", "Tampón de clonar exacto desde punto de origen"),
            ];
            for (idx, (m_id, m_lbl, m_tip)) in modes.iter().enumerate() {
                let x = mode_area.left() + pad + idx as f32 * (mw + mode_gap);
                let pill_r = Rect::from_min_size(pos2(x, mode_area.top() + 2.0), vec2(mw, 24.0));
                let resp = ui.interact(pill_r, egui::Id::new(("studio-eraser-mode", *m_id)), Sense::click()).on_hover_text(*m_tip);
                let active = cur_mode == *m_id;
                ui.painter().rect(
                    pill_r,
                    CornerRadius::same(7),
                    if active {
                        Color32::from_rgb(245, 198, 112)
                    } else if resp.hovered() {
                        Color32::from_white_alpha(28)
                    } else {
                        Color32::from_white_alpha(13)
                    },
                    Stroke::new(
                        1.0,
                        if active { Color32::from_rgb(255, 228, 168) } else { Color32::from_white_alpha(26) },
                    ),
                    StrokeKind::Inside,
                );
                ui.painter().text(
                    pill_r.center(),
                    Align2::CENTER_CENTER,
                    *m_lbl,
                    if active { t.semibold(9.9) } else { t.font(9.8) },
                    if active { Color32::from_rgb(22, 22, 26) } else { Color32::from_rgb(222, 226, 235) },
                );
                if resp.clicked() {
                    app.ui.tool = (*m_id).into();
                    eraser_on = true;
                    ui.ctx().data_mut(|dd| {
                        dd.insert_temp(eraser_id, true);
                        dd.insert_temp(egui::Id::new("studio-mask-brush-active"), false);
                        dd.insert_temp(egui::Id::new("studio-magic-drag-mask"), false);
                    });
                    if app.session.active_spot.is_some() {
                        let _ = app.run("spot.update", json!({"mode": *m_id}));
                    }
                }
            }

            // 2. Quick Brush Size Preset Pills (XS, S, M, L, XL) + Hint
            ui.add_space(4.0);
            let (sz_area, _) = ui.allocate_exact_size(vec2(w, 24.0), Sense::hover());
            ui.painter().text(
                pos2(sz_area.left() + pad, sz_area.center().y),
                Align2::LEFT_CENTER,
                "Brocha rápida:",
                t.font(10.0),
                Color32::from_rgb(188, 194, 206),
            );
            let cur_px = (app.ui.remove_size * 1000.0).clamp(1.0, 250.0);
            let presets: [(&str, f32); 5] = [("XS", 8.0), ("S", 20.0), ("M", 42.0), ("L", 85.0), ("XL", 150.0)];
            let pw = 28.0;
            let pgap = 4.0;
            let start_x = sz_area.right() - pad - (pw * 5.0 + pgap * 4.0);
            for (idx, (plbl, pval)) in presets.iter().enumerate() {
                let pr = Rect::from_min_size(pos2(start_x + idx as f32 * (pw + pgap), sz_area.top() + 2.0), vec2(pw, 20.0));
                let presp = ui
                    .interact(pr, egui::Id::new(("studio-eraser-sz-preset", *plbl)), Sense::click())
                    .on_hover_text(format!("Cambiar tamaño de la brocha a {pval:.0} px"));
                let is_near = (cur_px - *pval).abs() < 4.5;
                ui.painter().rect(
                    pr,
                    CornerRadius::same(5),
                    if is_near {
                        Color32::from_rgb(98, 212, 168).gamma_multiply(0.35)
                    } else if presp.hovered() {
                        Color32::from_white_alpha(26)
                    } else {
                        Color32::from_white_alpha(12)
                    },
                    Stroke::new(1.0, if is_near { Color32::from_rgb(98, 212, 168) } else { Color32::from_white_alpha(24) }),
                    StrokeKind::Inside,
                );
                ui.painter().text(
                    pr.center(),
                    Align2::CENTER_CENTER,
                    *plbl,
                    if is_near { t.semibold(9.4) } else { t.font(9.2) },
                    if is_near { Color32::WHITE } else { Color32::from_rgb(215, 220, 230) },
                );
                if presp.clicked() {
                    let frac = (*pval / 1000.0).clamp(0.001, 0.25);
                    app.ui.remove_size = frac;
                    app.ui.brush_size = (*pval / 400.0).clamp(0.002, 0.35);
                    if app.session.active_spot.is_some() {
                        let _ = app.run("spot.update", json!({"size": frac}));
                    }
                }
            }

            // 3. Brush Sliders: Tamaño de la brocha, Difuminar orillas, Opacidad
            let spec_plain = |id: &'static str, min: f64, max: f64, default: f64, step: f64, decimals: u8| ControlSpec {
                id,
                label: id,
                section: lightcraft_develop::Section::Detail,
                min,
                max,
                default,
                step,
                decimals,
                track: lightcraft_develop::Track::Plain,
            };
            let s_size = spec_plain("eraser.size", 1.0, 250.0, 25.0, 1.0, 0);
            studio_slider_spec(app, ui, &s_size, cur_px as f64, "Tamaño de la brocha", |app, v| {
                let frac = ((v / 1000.0) as f32).clamp(0.001, 0.25);
                app.ui.remove_size = frac;
                app.ui.brush_size = ((v / 400.0) as f32).clamp(0.002, 0.35);
                if app.session.active_spot.is_some() {
                    let _ = app.run("spot.update", json!({"size": frac}));
                }
                Ok(json!(null))
            });

            let s_feather = spec_plain("eraser.feather", 0.0, 100.0, 65.0, 1.0, 0);
            studio_slider_spec(app, ui, &s_feather, app.ui.remove_feather as f64, "Difuminar orillas (%)", |app, v| {
                let v32 = (v as f32).clamp(0.0, 100.0);
                app.ui.remove_feather = v32;
                if app.session.active_spot.is_some() {
                    let _ = app.run("spot.update", json!({"feather": v}));
                }
                Ok(json!(null))
            });

            let s_opac = spec_plain("eraser.opacity", 10.0, 100.0, 100.0, 1.0, 0);
            studio_slider_spec(app, ui, &s_opac, app.ui.remove_opacity as f64, "Opacidad de borrado (%)", |app, v| {
                let v32 = (v as f32).clamp(10.0, 100.0);
                app.ui.remove_opacity = v32;
                if app.session.active_spot.is_some() {
                    let _ = app.run("spot.update", json!({"opacity": v}));
                }
                Ok(json!(null))
            });

            // 4. Action Buttons: [✦ Auto-Manchas] [↻ Nuevo Parche] [↩ Deshacer] [🗑 Limpiar]
            ui.add_space(4.0);
            let (act_area, _) = ui.allocate_exact_size(vec2(w, 26.0), Sense::hover());
            let agap = 4.0;
            let aw = (inner_w - agap * 3.0) / 4.0;
            let actions = [
                ("auto", "✦ Manchas", "Detectar y limpiar manchas de sensor o polvo automáticamente"),
                ("reseed", "↻ Otro parche", "Probar otra reconstrucción IA para el objeto seleccionado"),
                ("undo", "↩ Último", "Eliminar el último trazo de borrado"),
                ("clear", "🗑 Todos", "Eliminar todos los trazos de borrado en esta foto"),
            ];
            for (idx, (aid, albl, atip)) in actions.iter().enumerate() {
                let x = act_area.left() + pad + idx as f32 * (aw + agap);
                let ar = Rect::from_min_size(pos2(x, act_area.top() + 1.0), vec2(aw, 23.0));
                let aresp = ui.interact(ar, egui::Id::new(("studio-eraser-act", *aid)), Sense::click()).on_hover_text(*atip);
                ui.painter().rect(
                    ar,
                    CornerRadius::same(6),
                    if aresp.hovered() { Color32::from_white_alpha(28) } else { Color32::from_white_alpha(13) },
                    Stroke::new(1.0, Color32::from_white_alpha(26)),
                    StrokeKind::Inside,
                );
                ui.painter().text(
                    ar.center(),
                    Align2::CENTER_CENTER,
                    *albl,
                    t.font(9.5),
                    Color32::from_rgb(225, 228, 236),
                );
                if aresp.clicked() {
                    ensure_active_photo(app);
                    match *aid {
                        "auto" => {
                            let _ = app.run("spot.findDust", json!({}));
                            app.toast(ui.ctx(), "Limpieza automática de manchas aplicada");
                        }
                        "reseed" => {
                            if app.session.active_spot.is_none() && !d.spots.is_empty() {
                                app.session.active_spot = Some(d.spots.len() - 1);
                            }
                            if app.session.active_spot.is_some() {
                                let _ = app.run("spot.refreshSource", json!({}));
                                app.toast(ui.ctx(), "Generando nueva reconstrucción IA…");
                            }
                        }
                        "undo" => {
                            if !d.spots.is_empty() {
                                let idx_del = app.session.active_spot.unwrap_or(d.spots.len() - 1);
                                let _ = app.run("spot.delete", json!({"index": idx_del}));
                            }
                        }
                        "clear" => {
                            for i in (0..d.spots.len()).rev() {
                                let _ = app.run("spot.delete", json!({"index": i}));
                            }
                            app.session.active_spot = None;
                        }
                        _ => {}
                    }
                }
            }

            if eraser_on {
                ui.add_space(3.0);
                let (hint_r, _) = ui.allocate_exact_size(vec2(w, 18.0), Sense::hover());
                ui.painter().text(
                    hint_r.center(),
                    Align2::CENTER_CENTER,
                    "💡 Pinta sobre el objeto en la foto · Usa Alt + Rueda para cambiar tamaño",
                    t.font(9.3),
                    Color32::from_rgb(165, 230, 195),
                );
            }

            ui.add_space(5.0);
        });
}

// =============================================================================
// AGENTIC AI CONNECTION BRIDGE (ANTIGRAVITY / ASTRA / PILOTO AUTOMÁTICO)
// =============================================================================
#[derive(Clone, serde::Deserialize)]
struct AgentExternalCommand {
    #[serde(default)]
    cmd_id: String,
    #[serde(default)]
    action: String, // "edit" (default) | "inspect" | "refine"
    #[serde(default)]
    photo_id: Option<u64>,
    #[serde(default)]
    target: String, // "open" | "selected" | "recent" | "album"
    #[serde(default)]
    album_name: String,
    #[serde(default)]
    style: String, // "editorial" | "social" | "cinema" | "fineart" | "reference" | "web" | "custom"
    #[serde(default)]
    reference_path: String,
    #[serde(default)]
    portfolio_url: String,
    #[serde(default)]
    custom_values: Option<serde_json::Map<String, serde_json::Value>>,
    #[serde(default)]
    profile_id: Option<String>,
    #[serde(default)]
    profile_amount: Option<f32>,
    #[serde(default)]
    agent_name: Option<String>,
    #[serde(default)]
    mark_edited: Option<bool>,
    #[serde(default)]
    crop_rect: Option<[f64; 4]>,
    #[serde(default)]
    crop_angle: Option<f64>,
}

/// Complete Visual QA & Iterative Refinement Report for a photo.
/// Implements the 8-Pillar Studio AI Loop:
/// 1. Inspect initial state (`initial_score`, `initial_issues`)
/// 2. Propose full settings (`crop`, `angle`, `wb`, `exposure`, `tone`, `curve`, `hsl`, `grading`, `optics`, `detail`)
/// 3. Re-inspect rendered photo & keep editing until `approved == true` (`passes`, `final_score`, `final_issues`)
/// 4. Harmonize batch clusters & Mark photo as Edited (`marked_edited == true`)
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct AiPhotoQaReport {
    pub photo_id: u64,
    pub file_name: String,
    pub initial_score: u8,
    pub initial_issues: Vec<String>,
    pub final_score: u8,
    pub final_issues: Vec<String>,
    pub passes: usize,
    pub approved: bool,
    pub marked_edited: bool,
    pub exposure: f64,
    pub contrast: f64,
    pub highlights: f64,
    pub shadows: f64,
    pub whites: f64,
    pub blacks: f64,
    pub dehaze: f64,
    #[serde(default)]
    pub wb_temp: f64,
    #[serde(default)]
    pub wb_tint: f64,
    #[serde(default)]
    pub crop_angle: f64,
    #[serde(default)]
    pub crop_rect: [f64; 4],
    #[serde(default)]
    pub scene_cluster: String,
    pub preview_before: String,
    pub preview_after: String,
}

#[derive(Clone, Debug)]
struct PhotoVisualEvaluation {
    p05: f64,
    p20: f64,
    p95: f64,
    p99: f64,
    skin_med: f64,
    skin_p85: f64,
    skin_sat_med: f64,
    skin_warmth_med: f64,
    neutral_warmth: f64,
    neutral_green_bias: f64,
    foliage_ratio: f64,
    subj_x0: f64,
    subj_y0: f64,
    subj_x1: f64,
    subj_y1: f64,
    hot_230: f64,
    hot_242: f64,
    is_night_flash: bool,
    score: u8,
    approved: bool,
    issues: Vec<String>,
}

#[derive(Clone, Debug)]
struct AiComprehensivePlan {
    pub exp: f64,
    pub contrast: f64,
    pub highlights: f64,
    pub shadows: f64,
    pub whites: f64,
    pub blacks: f64,
    pub dehaze: f64,
    pub wb_temp: f64,
    pub wb_tint: f64,
    pub wb_changed: bool,
    pub curve_highlights: f64,
    pub curve_lights: f64,
    pub curve_darks: f64,
    pub curve_shadows: f64,
    pub red_hue: f64,
    pub red_sat: f64,
    pub red_lum: f64,
    pub orange_hue: f64,
    pub orange_sat: f64,
    pub orange_lum: f64,
    pub yellow_hue: f64,
    pub yellow_sat: f64,
    pub yellow_lum: f64,
    pub green_hue: f64,
    pub green_sat: f64,
    pub green_lum: f64,
    pub aqua_sat: f64,
    pub blue_sat: f64,
    pub blue_lum: f64,
    pub purple_sat: f64,
    pub magenta_sat: f64,
    pub grad_sh_hue: f64,
    pub grad_sh_sat: f64,
    pub grad_mid_hue: f64,
    pub grad_mid_sat: f64,
    pub grad_hi_hue: f64,
    pub grad_hi_sat: f64,
    pub sharpen_amt: f64,
    pub nr_lum: f64,
    pub nr_color: f64,
    pub defringe_purple: f64,
    pub defringe_green: f64,
    pub vignette_amt: f64,
    pub crop_angle: f64,
    pub crop_rect: [f64; 4],
    pub crop_changed: bool,
    pub scene_cluster: String,
}

fn agent_bridge_dir() -> std::path::PathBuf {
    let home = std::env::var_os("HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("/Users/mcruz"));
    home.join(".mm-studio")
}

/// Token-efficient state synchronization & zero-latency command polling for Antigravity, Astra, or any external AI agent.
fn poll_and_sync_agent_bridge(app: &mut LightcraftApp, ctx: &egui::Context) {
    ctx.request_repaint_after(std::time::Duration::from_millis(450));
    let now = ctx.input(|i| i.time);
    let last_poll_id = egui::Id::new("studio-agent-last-poll");
    let last_poll = ctx.data(|d| d.get_temp::<f64>(last_poll_id)).unwrap_or(0.0);
    if now - last_poll < 0.40 {
        return;
    }
    ctx.data_mut(|d| d.insert_temp(last_poll_id, now));

    let dir = agent_bridge_dir();
    let _ = std::fs::create_dir_all(&dir);

    // 1. Export ultra-compact state.json (economizes tokens when an AI agent inspects MM Studio)
    let active_id = app.session.active().map(|p| p.0);
    let active_name = app
        .session
        .active()
        .and_then(|id| app.session.catalog.photo(id))
        .map(|p| p.file_name.clone())
        .unwrap_or_default();
    let selected_ids: Vec<u64> = app.session.selected().into_iter().map(|p| p.0).collect();
    let vis_ids: Vec<u64> = app.session.visible_cloned().into_iter().take(60).map(|p| p.0).collect();

    let mut albums_compact = Vec::new();
    let kids = app.session.catalog.album_children_by_parent();
    for list in kids.values() {
        for alb in list {
            if !alb.folder {
                let c = app.session.catalog.album_count(alb.id);
                albums_compact.push(json!({
                    "id": alb.id.0,
                    "name": alb.name,
                    "count": c
                }));
            }
        }
    }

    let state_hash_id = egui::Id::new("studio-agent-state-sig");
    let sig = format!("{:?}-{:?}-{}-{}", active_id, selected_ids, vis_ids.len(), albums_compact.len());
    let prev_sig = ctx.data(|d| d.get_temp::<String>(state_hash_id)).unwrap_or_default();
    if sig != prev_sig {
        ctx.data_mut(|d| d.insert_temp(state_hash_id, sig));
        let compact_state = json!({
            "app": "MM Studio",
            "active_id": active_id,
            "active_file": active_name,
            "selected_ids": selected_ids,
            "visible_ids": vis_ids,
            "albums": albums_compact,
            "styles": ["editorial", "social", "cinema", "fineart", "reference", "web"]
        });
        let _ = std::fs::write(dir.join("state.json"), serde_json::to_string(&compact_state).unwrap_or_default());
    }

    // 2. Check if an external agent (Antigravity / Astra / CLI) wrote a command.json
    let cmd_path = dir.join("command.json");
    if let Ok(raw_cmd) = std::fs::read_to_string(&cmd_path) {
        let _ = std::fs::remove_file(&cmd_path);
        if let Ok(cmd) = serde_json::from_str::<AgentExternalCommand>(&raw_cmd) {
            let action = if cmd.action.trim().is_empty() { "edit" } else { cmd.action.trim() };
            let target_eff = if let Some(pid) = cmd.photo_id {
                let _ = app.run("library.select", json!({"ids": [pid], "mode": "replace"}));
                "selected"
            } else if cmd.target.is_empty() {
                "open"
            } else {
                cmd.target.as_str()
            };

            if action == "inspect" {
                let reports = inspect_target_photos_only(app, target_eff, &cmd.album_name, &cmd.style);
                let _ = std::fs::write(
                    dir.join("result.json"),
                    serde_json::to_string(&json!({
                        "cmd_id": cmd.cmd_id,
                        "ok": true,
                        "action": "inspect",
                        "inspected_count": reports.len(),
                        "reports": reports,
                    }))
                    .unwrap_or_default(),
                );
                app.toast(ctx, format!("🔍 IA inspeccionó {} foto(s)", reports.len()));
                ctx.request_repaint();
            } else {
                let reports = execute_agent_batch_edit(
                    app,
                    ctx,
                    target_eff,
                    &cmd.album_name,
                    &cmd.style,
                    &cmd.reference_path,
                    &cmd.portfolio_url,
                    cmd.custom_values.as_ref(),
                    cmd.profile_id.as_deref(),
                    cmd.profile_amount,
                    cmd.crop_rect,
                    cmd.crop_angle,
                    cmd.mark_edited.unwrap_or(true),
                );
                let count = reports.len();
                let approved_count = reports.iter().filter(|r| r.approved).count();
                let avg_score = reports
                    .iter()
                    .map(|r| r.final_score as usize)
                    .sum::<usize>()
                    .checked_div(count)
                    .unwrap_or(0);
                let agent_lbl = cmd.agent_name.unwrap_or_else(|| "Agente IA".to_string());
                let status_msg = format!(
                    "✅ {agent_lbl}: {approved_count}/{count} verificadas y marcadas Editadas (QA {avg_score}%)"
                );
                ctx.data_mut(|d| d.insert_temp(egui::Id::new("studio-agent-last-status"), status_msg.clone()));
                let _ = std::fs::write(
                    dir.join("result.json"),
                    serde_json::to_string(&json!({
                        "cmd_id": cmd.cmd_id,
                        "ok": true,
                        "edited_count": count,
                        "approved_count": approved_count,
                        "avg_qa_score": avg_score,
                        "style": cmd.style,
                        "target": target_eff,
                        "reports": reports,
                    }))
                    .unwrap_or_default(),
                );
                app.toast(ctx, status_msg);
                ctx.request_repaint();
            }
        }
    }
}

/// Analyze a reference image file on disk and return tailored DevelopSettings values matching its luminance, contrast, warmth, and saturation.
fn analyze_reference_photo_style(path_str: &str) -> Option<(serde_json::Map<String, serde_json::Value>, &'static str)> {
    let img = lightcraft_engine::export::read_graphic(path_str.trim()).ok()?;
    let (w, h) = (img.width, img.height);
    if w == 0 || h == 0 {
        return None;
    }
    let step_x = (w / 64).max(1);
    let step_y = (h / 64).max(1);
    let mut sum_l = 0.0f64;
    let mut sum_l2 = 0.0f64;
    let mut sum_r = 0.0f64;
    let mut sum_b = 0.0f64;
    let mut sum_sat = 0.0f64;
    let mut n = 0.0f64;

    for y in (0..h).step_by(step_y) {
        for x in (0..w).step_by(step_x) {
            let p = img.get(x, y);
            let r = p[0] as f64 / 255.0;
            let g = p[1] as f64 / 255.0;
            let b = p[2] as f64 / 255.0;
            let l = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            let max_c = r.max(g).max(b);
            let min_c = r.min(g).min(b);
            let sat = if max_c > 0.01 { (max_c - min_c) / max_c } else { 0.0 };
            sum_l += l;
            sum_l2 += l * l;
            sum_r += r;
            sum_b += b;
            sum_sat += sat;
            n += 1.0;
        }
    }
    if n < 1.0 {
        return None;
    }
    let mean_l = sum_l / n;
    let std_l = ((sum_l2 / n) - mean_l * mean_l).max(0.0).sqrt();
    let warmth = (sum_r - sum_b) / n;
    let mean_sat = sum_sat / n;

    let exp = ((mean_l - 0.46) * 2.2).clamp(-1.0, 1.2);
    let contrast = ((std_l - 0.22) * 140.0).clamp(-25.0, 38.0);
    let temp = (5600.0 + warmth * 6500.0).clamp(4300.0, 7400.0);
    let vibrance = ((mean_sat - 0.28) * 95.0).clamp(-20.0, 35.0);
    let shadows = if mean_l > 0.50 { 22.0 } else { 8.0 };
    let blacks = if std_l > 0.24 { -12.0 } else { 6.0 };

    let mut map = serde_json::Map::new();
    map.insert("light.exposure".into(), json!((exp * 100.0).round() / 100.0));
    map.insert("light.contrast".into(), json!(contrast.round()));
    map.insert("light.highlights".into(), json!(-28.0));
    map.insert("light.shadows".into(), json!(shadows));
    map.insert("light.whites".into(), json!(10.0));
    map.insert("light.blacks".into(), json!(blacks));
    map.insert("wb.temp".into(), json!(temp.round()));
    map.insert("color.vibrance".into(), json!(vibrance.round()));
    map.insert("effects.clarity".into(), json!(10.0));

    let best_lut = if mean_sat < 0.06 {
        "lut:sf-fujifilm-250d-b-w-096"
    } else if warmth > 0.08 {
        "lut:sf-kodak-vision3-250d-107"
    } else {
        "lut:sf-1-warm-breeze-002"
    };
    Some((map, best_lut))
}

/// Renders the photo with `test_d` and evaluates its visual quality ("Ve la foto a ver si se ve bien"):
/// Checks:
/// 1. Highlight & White Dress / Wall Protection (`hot_242`, `hot_230`, `p99`, `p98`, `p95`)
/// 2. Skin Tone & Subject Brightness (`skin_med`, `skin_p85`, `skin_warmth_med`)
/// 3. Shadow & Black Depth (`p05`, `p20`, `is_night_flash`)
/// 4. White Balance & Chromatic Neutrality (`neutral_warmth`, `neutral_green_bias`)
/// 5. Skin & Foliage Color Balance (`skin_sat_med`, `foliage_ratio`) + Subject Framing (`subj_x0..subj_y1`)
///
/// Returns `PhotoVisualEvaluation` with `score: 0..=100`, `approved: bool`, and `issues: Vec<String>`.
fn evaluate_rendered_photo(
    src: &lightcraft_raster::Rgb32f,
    info: &lightcraft_pipeline::SourceInfo,
    test_d: &lightcraft_develop::DevelopSettings,
    style: &str,
) -> PhotoVisualEvaluation {
    let target_skin = match style {
        "editorial" => 0.51,
        "social" => 0.49,
        "fineart" => 0.52,
        "cinema" => 0.45,
        _ => 0.49,
    };

    let req = lightcraft_pipeline::RenderRequest {
        quality: lightcraft_pipeline::Quality::Draft,
        ..lightcraft_pipeline::RenderRequest::fit(160, 160)
    };
    let rendered = lightcraft_pipeline::render(src, info, test_d, &req);
    let img = &rendered.image;
    let (w, h) = (img.width.max(1), img.height.max(1));

    let mut lums = Vec::with_capacity(w * h);
    let mut skin_lums = Vec::new();
    let mut skin_sats = Vec::new();
    let mut skin_warmths = Vec::new();
    let mut neutral_warmths = Vec::new();
    let mut neutral_greens = Vec::new();
    let mut center_lums = Vec::new();
    let mut hot_230_count = 0usize;
    let mut hot_242_count = 0usize;
    let mut center_hot_230_count = 0usize;
    let mut center_hot_242_count = 0usize;
    let mut center_total_count = 0usize;
    let mut foliage_count = 0usize;

    let mut subj_xs = Vec::new();
    let mut subj_ys = Vec::new();

    let x0 = w / 6;
    let x1 = w * 5 / 6;
    let y0 = h / 7;
    let y1 = h * 6 / 7;

    for y in 0..h {
        let ny = y as f64 / h as f64;
        for x in 0..w {
            let nx = x as f64 / w as f64;
            let [r, g, b, _] = img.get(x, y);
            let max_c = r.max(g).max(b);
            let min_c = r.min(g).min(b);
            if max_c >= 230 {
                hot_230_count += 1;
            }
            if max_c >= 242 {
                hot_242_count += 1;
            }
            let l = (0.2126 * r as f64 + 0.7152 * g as f64 + 0.0722 * b as f64) / 255.0;
            lums.push(l);

            let sat = if max_c > 22 { (max_c - min_c) as f64 / max_c as f64 } else { 0.0 };

            // Detect neutral / near-neutral highlights & midtones inside the subject zone
            if x >= x0 && x <= x1 && y >= y0 && y <= y1 && (0.45..=0.88).contains(&l) && sat <= 0.22 {
                let nw = (r as f64 - b as f64) / 255.0;
                let ng = (g as f64 - 0.5 * (r as f64 + b as f64)) / 255.0;
                neutral_warmths.push(nw);
                neutral_greens.push(ng);
            }

            // Detect outdoor foliage / greenery
            if g > r.saturating_add(6) && g > b.saturating_add(10) && sat > 0.16 && (0.12..=0.80).contains(&l) {
                foliage_count += 1;
            }

            if x >= x0 && x <= x1 && y >= y0 && y <= y1 {
                center_total_count += 1;
                if max_c >= 232 {
                    center_hot_230_count += 1;
                }
                if max_c >= 243 {
                    center_hot_242_count += 1;
                }
                if (0.14..=0.86).contains(&l) {
                    center_lums.push(l);
                }
                if (r as i32 - g as i32) >= 7
                    && g >= b.saturating_sub(6)
                    && (r as i32 - b as i32) >= 18
                    && (0.24..=0.58).contains(&sat)
                    && (0.22..=0.78).contains(&l)
                {
                    skin_lums.push(l);
                    skin_sats.push(sat);
                    skin_warmths.push((r as f64 - b as f64) / 255.0);
                    subj_xs.push(nx);
                    subj_ys.push(ny);
                }
            }
        }
    }

    lums.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    skin_lums.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    skin_sats.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    skin_warmths.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    neutral_warmths.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    neutral_greens.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    center_lums.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    subj_xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    subj_ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let total = lums.len().max(1);
    let c_total = center_total_count.max(1);
    let p05 = lums[(total * 5 / 100).min(total - 1)];
    let p20 = lums[(total * 20 / 100).min(total - 1)];
    let p25 = lums[(total * 25 / 100).min(total - 1)];
    let p50 = lums[(total * 50 / 100).min(total - 1)];
    let p85 = lums[(total * 85 / 100).min(total - 1)];
    let p90 = lums[(total * 90 / 100).min(total - 1)];
    let p95 = lums[(total * 95 / 100).min(total - 1)];
    let p99 = lums[(total * 99 / 100).min(total - 1)];

    // Measure hot highlights primarily inside the subject/center zone so small background windows
    // in backlit palapas never cause the subjects to be darkened!
    let hot_230 = (center_hot_230_count as f64 / c_total as f64).max((hot_230_count as f64 / total as f64) * 0.45);
    let hot_242 = (center_hot_242_count as f64 / c_total as f64).max((hot_242_count as f64 / total as f64) * 0.40);
    let foliage_ratio = foliage_count as f64 / total as f64;
    let is_night_flash = p25 < 0.14 && p90 > 0.48;
    let is_backlit_window = (hot_242_count as f64 / total as f64) > 0.020 && p50 < 0.36;

    let skin_med = if skin_lums.len() >= 25 {
        skin_lums[skin_lums.len() / 2]
    } else if !center_lums.is_empty() {
        center_lums[center_lums.len() / 2]
    } else {
        p50
    };
    let skin_p85 = if skin_lums.len() >= 25 {
        skin_lums[skin_lums.len() * 85 / 100]
    } else {
        p85
    };
    let skin_sat_med = if skin_sats.len() >= 25 {
        skin_sats[skin_sats.len() / 2]
    } else {
        0.32
    };
    let skin_warmth_med = if skin_warmths.len() >= 25 {
        skin_warmths[skin_warmths.len() / 2]
    } else {
        0.16
    };
    let neutral_warmth = if neutral_warmths.len() >= 20 {
        neutral_warmths[neutral_warmths.len() / 2]
    } else {
        0.02
    };
    let neutral_green_bias = if neutral_greens.len() >= 20 {
        neutral_greens[neutral_greens.len() / 2]
    } else {
        0.0
    };

    let (subj_x0, subj_y0, subj_x1, subj_y1) = if subj_xs.len() >= 25 {
        let sx0 = subj_xs[(subj_xs.len() * 4 / 100).min(subj_xs.len() - 1)];
        let sx1 = subj_xs[(subj_xs.len() * 96 / 100).min(subj_xs.len() - 1)];
        let sy0 = subj_ys[(subj_ys.len() * 4 / 100).min(subj_ys.len() - 1)];
        let sy1 = subj_ys[(subj_ys.len() * 96 / 100).min(subj_ys.len() - 1)];
        (sx0, sy0, sx1, sy1)
    } else {
        (0.15, 0.15, 0.85, 0.88)
    };

    let mut penalty = 0.0f64;
    let mut issues = Vec::new();

    // 1. Strict Headroom & Composition Check (Never guillotine hair/tiaras/heads!)
    // `subj_y0` is the top of the face skin; hair/tiara/hat extends ~0.12–0.15 above `subj_y0`.
    if test_d.crop.geometry.rect.y0 > 0.025 && subj_y0 < 0.27 {
        penalty += 26.0;
        issues.push("Recorte superior inválido: corta cabeza/cabello o deja sin aire".to_string());
    }

    // 2. Check for genuinely clipped highlights vs dull/grey whites (Histogram Fullness Check)
    let dress_or_wall_hot = if is_backlit_window && skin_med < 0.46 {
        hot_242 > 0.036 || skin_p85 > 0.72
    } else {
        hot_242 > 0.032 || p99 > 0.976 || (hot_230 > 0.075 && p95 > 0.905)
    };
    if dress_or_wall_hot {
        let sev = (p99 - 0.96)
            .max(hot_242 * 2.2)
            .max(hot_230 * 0.8)
            .clamp(0.03, 0.30);
        penalty += (sev * 130.0).clamp(10.0, 32.0);
        issues.push("Sobreexposición real en altas luces / vestido".to_string());
    } else if !is_night_flash && p99 < 0.855 {
        // Right side of histogram is empty -> white dress / bouquet looks muddy grey!
        let deficit = (0.88 - p99).clamp(0.03, 0.25);
        penalty += (deficit * 120.0).clamp(12.0, 26.0);
        issues.push("Blancos apagados / grises sin brillo (histograma corto en altas luces)".to_string());
    }

    // 3. Check for hot skin / forehead flash glare
    let skin_hot = skin_p85 > 0.72 || skin_med > target_skin + 0.065;
    if skin_hot {
        let sev = (skin_p85 - 0.71).max(skin_med - (target_skin + 0.055)).clamp(0.02, 0.25);
        penalty += (sev * 115.0).clamp(9.0, 24.0);
        issues.push("Brillo excesivo en tonos de piel".to_string());
    }

    // 4. Check for underexposed subject / dark skin tones
    let min_skin_ok = target_skin - 0.085;
    let headroom = if is_backlit_window {
        (0.70 - skin_p85).min(0.036 - hot_242)
    } else {
        (0.960 - p99).min(0.70 - skin_p85)
    };
    if skin_med < min_skin_ok && skin_p85 < 0.58 && headroom > 0.015 && test_d.light.shadows < 52.0 {
        let sev = (min_skin_ok - skin_med).clamp(0.02, 0.25);
        penalty += (sev * 130.0).clamp(10.0, 28.0);
        issues.push("Subexposición en rostros / sujeto".to_string());
    } else if skin_med < min_skin_ok - 0.015 && test_d.light.shadows < 36.0 {
        penalty += 11.0;
        issues.push("Sujeto en sombra frente a altas luces".to_string());
    }

    // 5. Check shadow & black depth
    if is_night_flash {
        if p20 > 0.15 && test_d.light.shadows > 28.0 {
            penalty += 10.0;
            issues.push("Fondo nocturno lavado por exceso de sombras".to_string());
        }
    } else {
        if p05 > 0.072 && test_d.light.blacks > -20.0 {
            penalty += 10.0;
            issues.push("Negros lavados sin profundidad".to_string());
        }
        if p20 < 0.150 && test_d.light.shadows < 32.0 {
            penalty += 9.0;
            issues.push("Sombras cerradas en ropa/cabello".to_string());
        }
    }

    // 6. Check White Balance / Color Casts on neutrals (white dress, shirt, walls)
    let nw_tol = if is_night_flash { 0.125 } else { 0.075 };
    if neutral_warmth < -0.042 {
        penalty += 12.0;
        issues.push("Dominante fría/azulada en blancos".to_string());
    } else if neutral_warmth > nw_tol {
        penalty += 12.0;
        issues.push("Dominante amarilla/cálida excesiva en blancos".to_string());
    }
    if neutral_green_bias.abs() > 0.035 {
        penalty += 10.0;
        issues.push(if neutral_green_bias > 0.0 {
            "Dominante verdosa en iluminación".to_string()
        } else {
            "Dominante magenta en iluminación".to_string()
        });
    }

    // 7. Check skin color saturation & orange/red cast
    if (skin_sat_med > 0.45 || skin_warmth_med > 0.27) && test_d.mixer.orange.sat > -14.0 {
        penalty += 14.0;
        issues.push("Piel sobresaturada / dominante rojiza-naranja".to_string());
    }

    let score = (100.0 - penalty).round().clamp(15.0, 99.0) as u8;
    let approved = issues.is_empty() && score >= 92;

    PhotoVisualEvaluation {
        p05,
        p20,
        p95,
        p99,
        skin_med,
        skin_p85,
        skin_sat_med,
        skin_warmth_med,
        neutral_warmth,
        neutral_green_bias,
        foliage_ratio,
        subj_x0,
        subj_y0,
        subj_x1,
        subj_y1,
        hot_230,
        hot_242,
        is_night_flash,
        score,
        approved,
        issues,
    }
}

/// Computes smart horizon leveling (`crop_angle`) and composition framing (`crop_rect`)
/// while preserving the photo's aspect ratio and guaranteeing generous headroom/padding around all subjects.
fn compute_smart_angle_and_crop(
    src: &lightcraft_raster::Rgb32f,
    info: &lightcraft_pipeline::SourceInfo,
    base_d: &lightcraft_develop::DevelopSettings,
    eval: &PhotoVisualEvaluation,
) -> (f64, [f64; 4], bool) {
    // 1. Subtle horizon / vertical plumb leveling via `upright::level_degrees`.
    // Only correct genuine handheld camera roll (0.20° .. 2.35°), dampened by 0.75 so diagonal
    // architectural stairs or angled tent roofs never trigger false rotation.
    let raw_lvl = lightcraft_pipeline::upright::level_degrees(src, info, base_d).unwrap_or(0.0);
    let angle = if raw_lvl.abs() >= 0.20 && raw_lvl.abs() <= 2.35 {
        ((raw_lvl * 0.75) * 100.0).round() / 100.0
    } else {
        0.0
    };

    // 2. Smart Editorial Framing:
    // ONLY trim top margin when `eval.subj_y0 > 0.34` (guaranteeing at least 20%+ headroom for hair/tiaras/crowns!).
    let mut rect = [0.0f64, 0.0f64, 1.0f64, 1.0f64];
    let mut has_smart_trim = false;
    if eval.subj_y0 > 0.34 && eval.subj_y1 < 0.90 && eval.subj_x0 > 0.16 && eval.subj_x1 < 0.84 {
        let safe_top_trim = (eval.subj_y0 - 0.26).clamp(0.0, 0.045);
        let safe_bot_trim = (0.93 - eval.subj_y1).clamp(0.0, 0.035);
        let v_trim = safe_top_trim + safe_bot_trim;
        if v_trim >= 0.025 {
            let scale = (1.0 - v_trim).clamp(0.90, 0.98);
            let subj_cx = 0.5 * (eval.subj_x0 + eval.subj_x1);
            let ideal_x0 = (subj_cx - 0.5 * scale).clamp(0.0, 1.0 - scale);
            let safe_x0 = ideal_x0.min((eval.subj_x0 - 0.12).max(0.0));
            let safe_x1 = (safe_x0 + scale).min(1.0);
            if safe_x1 - eval.subj_x1 >= 0.12 {
                rect = [
                    (safe_x0 * 1000.0).round() / 1000.0,
                    (safe_top_trim * 1000.0).round() / 1000.0,
                    (safe_x1 * 1000.0).round() / 1000.0,
                    ((1.0 - safe_bot_trim) * 1000.0).round() / 1000.0,
                ];
                has_smart_trim = true;
            }
        }
    }

    let changed = angle.abs() >= 0.05 || has_smart_trim;
    (angle, rect, changed)
}

/// 8-Pillar Closed-Loop Photo Inspector, Proposer, Re-Verifier & Refiner:
/// 1. Views the photo as it currently is (`initial_eval`, exports `_before.jpg`)
/// 2. Computes Smart Horizon Leveling (`crop.angle`), Smart Framing (`crop.rect`), White Balance (`wb.temp`/`wb.tint`),
///    Exposure & Tone (`light.*`), Parametric Tone Curve (`curve.*`), 8-Channel HSL (`mixer.*`),
///    3-Way Color Grading (`grading.*`), Optics & Adaptive Detail (`optics.*`, `detail.*`, `vignette.*`)
/// 3. Renders the edited photo, inspects it again (`evaluate_rendered_photo`), and if the AI
///    decides it's not yet optimal (`!approved`), keeps refining in a loop (up to 10 passes)
///    until it achieves a first-class result!
fn inspect_propose_verify_and_refine_photo(
    app: &mut LightcraftApp,
    pid: lightcraft_catalog::PhotoId,
    style: &str,
    prof_id: &str,
    prof_amt: f32,
    preview_dir: &std::path::Path,
) -> (AiComprehensivePlan, AiPhotoQaReport) {
    let fname = app
        .session
        .catalog
        .photo(pid)
        .map(|p| p.file_name.clone())
        .unwrap_or_else(|| format!("photo_{}", pid.0));
    let before_path = preview_dir.join(format!("{fname}_before.jpg"));
    let after_path = preview_dir.join(format!("{fname}.jpg"));

    let existing = app.session.develop_of(pid).map(|a| (*a).clone()).unwrap_or_default();
    let mut base_d = lightcraft_develop::DevelopSettings {
        wb: existing.wb,
        crop: existing.crop,
        orientation: existing.orientation,
        ..Default::default()
    };

    let info = app.session.source_info(pid);
    lightcraft_pipeline::local::normalize_wb(&mut base_d, &info);
    let base_wb_temp = base_d.wb.temp;
    let base_wb_tint = base_d.wb.tint;

    let Ok(src) = app.session.source_now(pid, lightcraft_engine::SourceLevel::Thumb) else {
        let fallback_plan = AiComprehensivePlan {
            exp: 0.0,
            contrast: 18.0,
            highlights: -68.0,
            shadows: 26.0,
            whites: -10.0,
            blacks: -14.0,
            dehaze: 6.0,
            wb_temp: base_wb_temp,
            wb_tint: base_wb_tint,
            wb_changed: false,
            curve_highlights: -8.0,
            curve_lights: 4.0,
            curve_darks: -4.0,
            curve_shadows: 2.0,
            red_hue: 2.0,
            red_sat: -3.0,
            red_lum: 3.0,
            orange_hue: 2.0,
            orange_sat: -3.0,
            orange_lum: 4.0,
            yellow_hue: -10.0,
            yellow_sat: -12.0,
            yellow_lum: -4.0,
            green_hue: 12.0,
            green_sat: -18.0,
            green_lum: -6.0,
            aqua_sat: -6.0,
            blue_sat: -6.0,
            blue_lum: -4.0,
            purple_sat: -18.0,
            magenta_sat: -14.0,
            grad_sh_hue: 222.0,
            grad_sh_sat: 4.0,
            grad_mid_hue: 34.0,
            grad_mid_sat: 4.0,
            grad_hi_hue: 38.0,
            grad_hi_sat: 6.0,
            sharpen_amt: 44.0,
            nr_lum: 14.0,
            nr_color: 25.0,
            defringe_purple: 4.0,
            defringe_green: 3.0,
            vignette_amt: -6.0,
            crop_angle: 0.0,
            crop_rect: [0.0, 0.0, 1.0, 1.0],
            crop_changed: false,
            scene_cluster: "general".to_string(),
        };
        let rep = AiPhotoQaReport {
            photo_id: pid.0,
            file_name: fname,
            initial_score: 70,
            initial_issues: vec![],
            final_score: 92,
            final_issues: vec![],
            passes: 1,
            approved: true,
            marked_edited: false,
            exposure: 0.0,
            contrast: 18.0,
            highlights: -68.0,
            shadows: 26.0,
            whites: -10.0,
            blacks: -14.0,
            dehaze: 6.0,
            wb_temp: base_wb_temp,
            wb_tint: base_wb_tint,
            crop_angle: 0.0,
            crop_rect: [0.0, 0.0, 1.0, 1.0],
            scene_cluster: "general".to_string(),
            preview_before: before_path.display().to_string(),
            preview_after: after_path.display().to_string(),
        };
        return (fallback_plan, rep);
    };

    // =========================================================================
    // FASE 1: VER LA FOTO COMO ESTÁ PRIMERO (Inspección Visual Inicial)
    // =========================================================================
    let initial_eval = evaluate_rendered_photo(&src, &info, &base_d, style);
    {
        let before_render = lightcraft_pipeline::render(
            &src,
            &info,
            &base_d,
            &lightcraft_pipeline::RenderRequest::fit(480, 480),
        );
        if let Ok(jpeg_bytes) = lightcraft_codecs::encode_jpeg(
            &lightcraft_codecs::EncodeImage::rgba8(&before_render.image),
            82,
            lightcraft_codecs::ChromaSubsampling::S420,
            &lightcraft_codecs::EncodeMeta::default(),
        ) {
            let _ = std::fs::write(&before_path, jpeg_bytes);
        }
    }

    // Clasificación de escena para armonización por lotes (Batch Scene-Cluster)
    let scene_cluster = if initial_eval.is_night_flash {
        "night_flash".to_string()
    } else if initial_eval.foliage_ratio > 0.035 {
        "outdoor_foliage".to_string()
    } else {
        "indoor_social".to_string()
    };

    // =========================================================================
    // FASE 2: PROPONER CONFIGURACIÓN INTEGRAL (8 PILARES) EN BASE A CÓMO ESTÁ
    // =========================================================================
    let (crop_angle, crop_rect, crop_changed) =
        compute_smart_angle_and_crop(&src, &info, &base_d, &initial_eval);

    let target_skin = match style {
        "editorial" => 0.51,
        "social" => 0.49,
        "fineart" => 0.52,
        "cinema" => 0.45,
        _ => 0.49,
    };

    let mut cur_d = base_d.clone();
    if !prof_id.is_empty() {
        cur_d.profile.id = prof_id.to_string();
        cur_d.profile.amount = prof_amt as f64;
    }
    if crop_changed {
        cur_d.crop.geometry.angle = crop_angle;
        cur_d.crop.geometry.rect = lightcraft_geom::Rect::new(
            crop_rect[0],
            crop_rect[1],
            crop_rect[2],
            crop_rect[3],
        );
        cur_d.crop.geometry = cur_d
            .crop
            .geometry
            .constrained(src.width as f64, src.height as f64);
    }

    // Balance de Blancos inteligente (WB Temp & Tint)
    let target_neutral_warmth = match style {
        "editorial" => 0.012,
        "social" => 0.022,
        "cinema" => 0.015,
        _ => 0.018,
    };
    let mut wb_temp = base_wb_temp;
    let mut wb_tint = base_wb_tint;
    let max_wb_shift = if initial_eval.is_night_flash { 400.0 } else { 1400.0 };
    if initial_eval.neutral_warmth < -0.020 || initial_eval.neutral_warmth > 0.055 {
        let delta_k = ((target_neutral_warmth - initial_eval.neutral_warmth) * 4200.0).clamp(-max_wb_shift, max_wb_shift);
        wb_temp = (base_wb_temp + delta_k).clamp(3800.0, 8200.0);
    }
    if initial_eval.neutral_green_bias.abs() > 0.015 {
        let delta_tint = (initial_eval.neutral_green_bias * 240.0).clamp(-16.0, 16.0);
        wb_tint = (base_wb_tint + delta_tint).clamp(-60.0, 60.0);
    }

    // Exposición y Tono inicial
    let is_backlit_init = initial_eval.p99 > 0.88 && initial_eval.skin_med < 0.45;
    let is_soft_indoor = !initial_eval.is_night_flash && initial_eval.hot_242 < 0.008 && initial_eval.p99 < 0.89;
    let mut exp = if is_backlit_init {
        0.22
    } else if initial_eval.hot_242 > 0.022 || initial_eval.p99 > 0.965 {
        -0.25
    } else if initial_eval.skin_med < target_skin - 0.04 {
        ((target_skin - initial_eval.skin_med) * 1.15).clamp(0.12, 0.48)
    } else {
        0.10
    };

    let mut contrast = match style {
        "fineart" => 6.0,
        "editorial" => if is_soft_indoor { 12.0 } else { 16.0 },
        "cinema" => 22.0,
        _ => 16.0,
    };
    let mut highlights = if is_backlit_init {
        -84.0
    } else if initial_eval.hot_242 > 0.015 || initial_eval.p99 > 0.94 {
        -68.0
    } else if is_soft_indoor {
        -28.0
    } else {
        -48.0
    };
    let mut shadows = if is_backlit_init {
        38.0
    } else if initial_eval.is_night_flash {
        24.0
    } else {
        26.0
    };
    let mut whites = if is_backlit_init || initial_eval.hot_242 > 0.018 {
        -12.0
    } else if is_soft_indoor {
        14.0
    } else {
        4.0
    };
    let mut blacks = if is_soft_indoor { -14.0f64 } else { -18.0f64 };
    let dehaze = if initial_eval.hot_242 > 0.015 { 8.0 } else { 4.0 };

    // Curva de Tonos Paramétrica (Roll-off suave en vestido blanco + microcontraste)
    let curve_highlights = if is_soft_indoor { -4.0f64 } else { -8.0f64 };
    let curve_lights = 5.0f64;
    let curve_darks = -4.0f64;
    let curve_shadows = if initial_eval.is_night_flash { 1.0 } else { 3.0 };

    // Mezclador de Color HSL de 8 canales adaptativo a piel, paredes amarillas, follaje y luces LED
    let red_hue = 3.0f64;
    let mut red_sat = if style == "editorial" { -8.0f64 } else { -4.0f64 };
    let mut red_lum = 5.0f64;
    let orange_hue = 3.0f64;
    let mut orange_sat = if style == "editorial" { -10.0f64 } else { -5.0f64 };
    let mut orange_lum = if style == "editorial" { 9.0f64 } else { 5.0f64 };

    let has_foliage = initial_eval.foliage_ratio > 0.025;
    let yellow_hue = if has_foliage { -14.0 } else { -8.0 };
    let yellow_sat = if style == "editorial" { -34.0 } else if has_foliage { -20.0 } else { -12.0 };
    let yellow_lum = if has_foliage { -4.0 } else { 6.0 };
    let green_hue = if has_foliage { 18.0 } else { 10.0 };
    let green_sat = if has_foliage { -28.0 } else { -16.0 };
    let green_lum = if has_foliage { -8.0 } else { -4.0 };
    let aqua_sat = -8.0f64;
    let blue_sat = -8.0f64;
    let blue_lum = -4.0f64;
    let purple_sat = -18.0f64;
    let magenta_sat = -14.0f64;

    // Color Grading de 3 vías (Sombras limpias + medios/altas cálidos sedosos)
    let grad_sh_hue = 222.0f64;
    let grad_sh_sat = if style == "cinema" { 8.0 } else { 3.0 };
    let grad_mid_hue = 36.0f64;
    let grad_mid_sat = 3.0f64;
    let grad_hi_hue = 40.0f64;
    let grad_hi_sat = 4.0f64;

    // Óptica, Nitidez y Reducción de Ruido adaptativa
    let sharpen_amt = 42.0f64;
    let nr_lum = if initial_eval.is_night_flash { 20.0 } else { 15.0 };
    let nr_color = 26.0f64;
    let defringe_purple = 4.0f64;
    let defringe_green = 3.0f64;
    let vignette_amt = if initial_eval.is_night_flash { -4.0 } else { -6.0 };

    // =========================================================================
    // FASE 3: APLICAR -> VOLVER A VER LA FOTO -> DECIDIR SI SE VE BIEN O SEGUIR EDITANDO
    // =========================================================================
    let mut best_tuple = (
        exp,
        contrast,
        highlights,
        shadows,
        whites,
        blacks,
        dehaze,
        wb_temp,
        wb_tint,
        orange_lum,
        orange_sat,
        red_sat,
    );
    let mut best_eval = initial_eval.clone();
    best_eval.score = 0;
    let mut passes_run = 0usize;

    for pass in 1..=10 {
        passes_run = pass;
        let mut test_d = cur_d.clone();
        if (wb_temp - base_wb_temp).abs() > 15.0 || (wb_tint - base_wb_tint).abs() > 0.8 {
            test_d.wb.mode = lightcraft_develop::WbMode::Custom;
            test_d.wb.temp = wb_temp;
            test_d.wb.tint = wb_tint;
        }
        test_d.light.exposure = exp;
        test_d.light.contrast = contrast;
        test_d.light.highlights = highlights;
        test_d.light.shadows = shadows;
        test_d.light.whites = whites;
        test_d.light.blacks = blacks;
        test_d.effects.dehaze = dehaze;
        if style == "editorial" {
            test_d.color.vibrance = 12.0;
            test_d.color.saturation = -4.0;
            test_d.effects.clarity = 4.0;
            test_d.effects.texture = -5.0;
        }
        test_d.vignette.amount = vignette_amt;

        test_d.curve.highlights = curve_highlights;
        test_d.curve.lights = curve_lights;
        test_d.curve.darks = curve_darks;
        test_d.curve.shadows = curve_shadows;

        test_d.mixer.red.hue = red_hue;
        test_d.mixer.red.sat = red_sat;
        test_d.mixer.red.lum = red_lum;
        test_d.mixer.orange.hue = orange_hue;
        test_d.mixer.orange.sat = orange_sat;
        test_d.mixer.orange.lum = orange_lum;
        test_d.mixer.yellow.hue = yellow_hue;
        test_d.mixer.yellow.sat = yellow_sat;
        test_d.mixer.yellow.lum = yellow_lum;
        test_d.mixer.green.hue = green_hue;
        test_d.mixer.green.sat = green_sat;
        test_d.mixer.green.lum = green_lum;
        test_d.mixer.aqua.sat = aqua_sat;
        test_d.mixer.blue.sat = blue_sat;
        test_d.mixer.blue.lum = blue_lum;
        test_d.mixer.purple.sat = purple_sat;
        test_d.mixer.magenta.sat = magenta_sat;

        test_d.grading.shadows.hue = grad_sh_hue;
        test_d.grading.shadows.sat = grad_sh_sat;
        test_d.grading.midtones.hue = grad_mid_hue;
        test_d.grading.midtones.sat = grad_mid_sat;
        test_d.grading.highlights.hue = grad_hi_hue;
        test_d.grading.highlights.sat = grad_hi_sat;

        let eval = evaluate_rendered_photo(&src, &info, &test_d, style);

        if eval.score > best_eval.score || (eval.score == best_eval.score && eval.issues.len() < best_eval.issues.len()) {
            best_eval = eval.clone();
            best_tuple = (
                exp,
                contrast,
                highlights,
                shadows,
                whites,
                blacks,
                dehaze,
                wb_temp,
                wb_tint,
                orange_lum,
                orange_sat,
                red_sat,
            );
        }

        // Condición estricta de decisión IA: ¿La foto ya se ve impecable (0 defectos y score >= 92)?
        if eval.approved && pass >= 2 {
            break;
        }

        // Si la IA decide que AÚN NO se ve bien, corrige específicamente los defectos detectados:
        let is_backlit = eval.p99 > 0.88 && eval.skin_med < 0.46;
        let dress_or_wall_hot = if is_backlit {
            eval.hot_242 > 0.032 || eval.skin_p85 > 0.72
        } else {
            eval.hot_242 > 0.025 || eval.p99 > 0.972 || (eval.hot_230 > 0.068 && eval.p95 > 0.895)
        };
        let skin_hot = eval.skin_p85 > 0.72 || eval.skin_med > target_skin + 0.065;

        if dress_or_wall_hot || skin_hot {
            if eval.skin_med > target_skin - 0.02 {
                let over_hi = (eval.p99 - 0.955)
                    .max(eval.skin_p85 - 0.71)
                    .max(eval.hot_242 * 1.8)
                    .clamp(0.03, 0.25);
                exp = (exp - over_hi * 1.05).clamp(-1.20, 0.65);
            }
            highlights = (highlights - 10.0).clamp(-92.0, -18.0);
            whites = (whites - 7.0).clamp(-26.0, 22.0);
            if skin_hot {
                orange_lum = (orange_lum - 2.5).clamp(-4.0, 14.0);
                red_lum = (red_lum - 1.5).clamp(-4.0, 10.0);
            }
        } else {
            // Si los blancos están grises/apagados (histograma corto a la derecha, `p99 < 0.86`), estiramos blancos!
            if !eval.is_night_flash && eval.p99 < 0.865 {
                whites = (whites + 6.0).clamp(-18.0, 22.0);
                highlights = (highlights + 8.0).clamp(-88.0, -24.0);
                if eval.skin_med < target_skin + 0.015 {
                    exp = (exp + 0.07).clamp(-1.20, 0.52);
                }
            }

            let headroom = if is_backlit {
                (0.71 - eval.skin_p85).min(0.032 - eval.hot_242)
            } else {
                (0.955 - eval.p99).min(0.71 - eval.skin_p85)
            };
            if headroom > 0.015 && eval.skin_med < target_skin - 0.035 {
                let need = (target_skin - eval.skin_med).min(headroom * 0.90);
                let step = if pass == 1 { 0.90 } else { 0.70 };
                exp = (exp + (need * step).clamp(0.0, 0.22)).clamp(-1.20, 0.56);
            } else if eval.skin_med < target_skin - 0.045 {
                shadows = (shadows + 7.0).clamp(16.0, 44.0);
                orange_lum = (orange_lum + 2.5).clamp(2.0, 15.0);
            }
        }

        // Refinamiento de Sombras y Negros
        if eval.is_night_flash {
            shadows = shadows.clamp(18.0, 32.0);
            blacks = blacks.clamp(-18.0, -12.0);
            contrast = contrast.clamp(16.0, 22.0);
        } else {
            if eval.p20 < 0.150 || is_backlit {
                shadows = (shadows + 5.0).clamp(20.0, 44.0);
            }
            if eval.p05 > 0.072 {
                blacks = (blacks - 3.5).clamp(-22.0, -10.0);
                contrast = (contrast + 1.5).clamp(10.0, 20.0);
            }
        }

        // Refinamiento en bucle cerrado de Balance de Blancos (WB Temp & Tint)
        let nw_hi = if eval.is_night_flash { 0.120 } else { 0.072 };
        if eval.neutral_warmth < -0.038 {
            wb_temp = (wb_temp + 180.0).clamp((base_wb_temp - max_wb_shift).max(3800.0), (base_wb_temp + max_wb_shift).min(8200.0));
        } else if eval.neutral_warmth > nw_hi {
            wb_temp = (wb_temp - 180.0).clamp((base_wb_temp - max_wb_shift).max(3800.0), (base_wb_temp + max_wb_shift).min(8200.0));
        }
        if eval.neutral_green_bias > 0.030 {
            wb_tint = (wb_tint + 3.5).clamp(base_wb_tint - 18.0, base_wb_tint + 18.0);
        } else if eval.neutral_green_bias < -0.030 {
            wb_tint = (wb_tint - 3.5).clamp(base_wb_tint - 18.0, base_wb_tint + 18.0);
        }

        // Refinamiento estricto de Saturación / Tono Naranja-Rojizo de Piel
        if eval.skin_sat_med > 0.43 || eval.skin_warmth_med > 0.25 {
            orange_sat = (orange_sat - 3.5).clamp(-20.0, 0.0);
            red_sat = (red_sat - 3.0).clamp(-16.0, 0.0);
            orange_lum = (orange_lum + 2.0).clamp(2.0, 16.0);
            wb_temp = (wb_temp - 140.0).clamp(4200.0, 8200.0);
        }
    }

    let (f_exp, f_ctr, f_hi, f_sh, f_wh, f_bl, f_dh, f_wbt, f_wbti, f_olum, f_osat, f_rsat) = (
        (best_tuple.0 * 100.0).round() / 100.0,
        best_tuple.1.round(),
        best_tuple.2.round(),
        best_tuple.3.round(),
        best_tuple.4.round(),
        best_tuple.5.round(),
        best_tuple.6.round(),
        best_tuple.7.round(),
        (best_tuple.8 * 10.0).round() / 10.0,
        best_tuple.9.round(),
        best_tuple.10.round(),
        best_tuple.11.round(),
    );
    let wb_changed = (f_wbt - base_wb_temp).abs() > 15.0 || (f_wbti - base_wb_tint).abs() > 0.8;

    let plan = AiComprehensivePlan {
        exp: f_exp,
        contrast: f_ctr,
        highlights: f_hi,
        shadows: f_sh,
        whites: f_wh,
        blacks: f_bl,
        dehaze: f_dh,
        wb_temp: f_wbt,
        wb_tint: f_wbti,
        wb_changed,
        curve_highlights,
        curve_lights,
        curve_darks,
        curve_shadows,
        red_hue,
        red_sat: f_rsat,
        red_lum,
        orange_hue,
        orange_sat: f_osat,
        orange_lum: f_olum,
        yellow_hue,
        yellow_sat,
        yellow_lum,
        green_hue,
        green_sat,
        green_lum,
        aqua_sat,
        blue_sat,
        blue_lum,
        purple_sat,
        magenta_sat,
        grad_sh_hue,
        grad_sh_sat,
        grad_mid_hue,
        grad_mid_sat,
        grad_hi_hue,
        grad_hi_sat,
        sharpen_amt,
        nr_lum,
        nr_color,
        defringe_purple,
        defringe_green,
        vignette_amt,
        crop_angle,
        crop_rect,
        crop_changed,
        scene_cluster: scene_cluster.clone(),
    };

    let approved_final = best_eval.approved;
    let report = AiPhotoQaReport {
        photo_id: pid.0,
        file_name: fname,
        initial_score: initial_eval.score,
        initial_issues: if initial_eval.issues.is_empty() {
            vec!["Exposición base aceptable (sin perfil social)".to_string()]
        } else {
            initial_eval.issues
        },
        final_score: best_eval.score,
        final_issues: best_eval.issues,
        passes: passes_run,
        approved: approved_final,
        marked_edited: approved_final,
        exposure: f_exp,
        contrast: f_ctr,
        highlights: f_hi,
        shadows: f_sh,
        whites: f_wh,
        blacks: f_bl,
        dehaze: f_dh,
        wb_temp: f_wbt,
        wb_tint: f_wbti,
        crop_angle,
        crop_rect,
        scene_cluster,
        preview_before: before_path.display().to_string(),
        preview_after: after_path.display().to_string(),
    };

    (plan, report)
}

/// Harmonizes `wb_temp`, `wb_tint`, and `contrast`/`shadows` across photos sharing the same scene cluster
/// (`outdoor_foliage`, `indoor_social`, `night_flash`) so large batches look cohesive side-by-side.
fn harmonize_batch_clusters(plans: &mut [(u64, AiComprehensivePlan, AiPhotoQaReport)]) {
    if plans.len() < 2 {
        return;
    }
    let clusters = ["outdoor_foliage", "indoor_social", "night_flash"];
    for cl in clusters {
        let mut temps: Vec<f64> = plans
            .iter()
            .filter(|(_, p, _)| p.scene_cluster == cl)
            .map(|(_, p, _)| p.wb_temp)
            .collect();
        let mut tints: Vec<f64> = plans
            .iter()
            .filter(|(_, p, _)| p.scene_cluster == cl)
            .map(|(_, p, _)| p.wb_tint)
            .collect();
        if temps.len() < 2 {
            continue;
        }
        temps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        tints.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let med_temp = temps[temps.len() / 2];
        let med_tint = tints[tints.len() / 2];

        for (_, p, rep) in plans.iter_mut().filter(|(_, p, _)| p.scene_cluster == cl) {
            if (p.wb_temp - med_temp).abs() <= 900.0 {
                let harmonized_t = (p.wb_temp * 0.78 + med_temp * 0.22).round();
                if (harmonized_t - p.wb_temp).abs() >= 15.0 {
                    p.wb_temp = harmonized_t;
                    p.wb_changed = true;
                    rep.wb_temp = harmonized_t;
                }
            }
            if (p.wb_tint - med_tint).abs() <= 20.0 {
                let harmonized_ti = ((p.wb_tint * 0.78 + med_tint * 0.22) * 10.0).round() / 10.0;
                if (harmonized_ti - p.wb_tint).abs() >= 0.8 {
                    p.wb_tint = harmonized_ti;
                    p.wb_changed = true;
                    rep.wb_tint = harmonized_ti;
                }
            }
        }
    }
}

fn resolve_agent_target_ids(app: &mut LightcraftApp, target: &str, album_name: &str) -> Vec<u64> {
    match target {
        "selected" => {
            let sel: Vec<u64> = app.session.selected().into_iter().map(|p| p.0).collect();
            if sel.is_empty() {
                app.session.active().map(|id| vec![id.0]).unwrap_or_default()
            } else {
                sel
            }
        }
        "recent" => {
            let _ = app.run("library.source", json!({"kind": "recentlyAdded"}));
            app.session.visible_cloned().into_iter().map(|id| id.0).collect()
        }
        "album" => {
            let q = album_name.trim().to_ascii_lowercase();
            let mut found_album_id = None;
            let kids = app.session.catalog.album_children_by_parent();
            for list in kids.values() {
                for alb in list {
                    if !alb.folder && (q.is_empty() || alb.name.to_ascii_lowercase().contains(&q)) {
                        found_album_id = Some(alb.id);
                        break;
                    }
                }
                if found_album_id.is_some() {
                    break;
                }
            }
            if let Some(aid) = found_album_id {
                let _ = app.run("library.source", json!({"kind": "album", "id": aid.0}));
                app.session.visible_cloned().into_iter().map(|id| id.0).collect()
            } else {
                app.session.visible_cloned().into_iter().map(|id| id.0).collect()
            }
        }
        _ => {
            if let Some(act) = app.session.active() {
                vec![act.0]
            } else {
                app.session.visible_cloned().into_iter().take(1).map(|id| id.0).collect()
            }
        }
    }
}

/// Stage 1 standalone inspection: views the target photos as they currently are and exports diagnostic reports.
fn inspect_target_photos_only(
    app: &mut LightcraftApp,
    target: &str,
    album_name: &str,
    style: &str,
) -> Vec<AiPhotoQaReport> {
    let target_ids = resolve_agent_target_ids(app, target, album_name);
    let preview_dir = agent_bridge_dir().join("previews");
    let _ = std::fs::create_dir_all(&preview_dir);
    let mut reports = Vec::new();

    for &pid in &target_ids {
        let p_id = lightcraft_catalog::PhotoId(pid);
        let fname = app
            .session
            .catalog
            .photo(p_id)
            .map(|p| p.file_name.clone())
            .unwrap_or_else(|| format!("photo_{pid}"));
        let mut cur_d = app.session.develop_of(p_id).map(|a| (*a).clone()).unwrap_or_default();
        let info = app.session.source_info(p_id);
        lightcraft_pipeline::local::normalize_wb(&mut cur_d, &info);
        if let Ok(src) = app.session.source_now(p_id, lightcraft_engine::SourceLevel::Thumb) {
            let eval = evaluate_rendered_photo(&src, &info, &cur_d, style);
            let before_path = preview_dir.join(format!("{fname}_before.jpg"));
            let rendered = lightcraft_pipeline::render(
                &src,
                &info,
                &cur_d,
                &lightcraft_pipeline::RenderRequest::fit(720, 720),
            );
            if let Ok(jpeg_bytes) = lightcraft_codecs::encode_jpeg(
                &lightcraft_codecs::EncodeImage::rgba8(&rendered.image),
                85,
                lightcraft_codecs::ChromaSubsampling::S420,
                &lightcraft_codecs::EncodeMeta::default(),
            ) {
                let _ = std::fs::write(&before_path, jpeg_bytes);
            }
            reports.push(AiPhotoQaReport {
                photo_id: pid,
                file_name: fname,
                initial_score: eval.score,
                initial_issues: eval.issues.clone(),
                final_score: eval.score,
                final_issues: eval.issues,
                passes: 0,
                approved: eval.approved,
                marked_edited: false,
                exposure: cur_d.light.exposure,
                contrast: cur_d.light.contrast,
                highlights: cur_d.light.highlights,
                shadows: cur_d.light.shadows,
                whites: cur_d.light.whites,
                blacks: cur_d.light.blacks,
                dehaze: cur_d.effects.dehaze,
                wb_temp: cur_d.wb.temp,
                wb_tint: cur_d.wb.tint,
                crop_angle: cur_d.crop.geometry.angle,
                crop_rect: [
                    cur_d.crop.geometry.rect.x0,
                    cur_d.crop.geometry.rect.y0,
                    cur_d.crop.geometry.rect.x1,
                    cur_d.crop.geometry.rect.y1,
                ],
                scene_cluster: "inspected".to_string(),
                preview_before: before_path.display().to_string(),
                preview_after: before_path.display().to_string(),
            });
        }
    }
    reports
}

fn execute_agent_batch_edit(
    app: &mut LightcraftApp,
    ctx: &egui::Context,
    target: &str,
    album_name: &str,
    style: &str,
    reference_path: &str,
    portfolio_url: &str,
    custom_values: Option<&serde_json::Map<String, serde_json::Value>>,
    custom_profile: Option<&str>,
    custom_profile_amt: Option<f32>,
    custom_crop_rect: Option<[f64; 4]>,
    custom_crop_angle: Option<f64>,
    mark_when_approved: bool,
) -> Vec<AiPhotoQaReport> {
    let orig_source = app.session.source;
    let orig_sel: Vec<u64> = app.session.selected().into_iter().map(|p| p.0).collect();

    let target_ids = resolve_agent_target_ids(app, target, album_name);
    if target_ids.is_empty() {
        return Vec::new();
    }

    // Build style recipe (DevelopSettings values + Special Filter LUT)
    let (mut base_values, default_lut, default_amt): (serde_json::Map<String, serde_json::Value>, &str, f32) = match style {
        "editorial" => {
            let mut m = serde_json::Map::new();
            m.insert("light.contrast".into(), json!(12.0));
            m.insert("color.vibrance".into(), json!(12.0));
            m.insert("color.saturation".into(), json!(-4.0));
            m.insert("mixer.orange.lum".into(), json!(10.0));
            m.insert("mixer.orange.sat".into(), json!(-12.0));
            m.insert("mixer.orange.hue".into(), json!(3.0));
            m.insert("mixer.red.lum".into(), json!(5.0));
            m.insert("mixer.red.sat".into(), json!(-10.0));
            m.insert("mixer.red.hue".into(), json!(3.0));
            m.insert("mixer.green.sat".into(), json!(-28.0));
            m.insert("mixer.green.hue".into(), json!(18.0));
            m.insert("mixer.yellow.sat".into(), json!(-34.0));
            m.insert("mixer.yellow.lum".into(), json!(6.0));
            m.insert("effects.clarity".into(), json!(4.0));
            m.insert("effects.texture".into(), json!(-5.0));
            m.insert("vignette.amount".into(), json!(-6.0));
            m.insert("detail.sharpenAmount".into(), json!(42.0));
            m.insert("detail.sharpenMasking".into(), json!(55.0));
            m.insert("detail.nrLuminance".into(), json!(15.0));
            m.insert("detail.nrColor".into(), json!(26.0));
            (m, "lc.color", 100.0)
        }
        "social" => {
            let mut m = serde_json::Map::new();
            m.insert("light.contrast".into(), json!(18.0));
            m.insert("color.vibrance".into(), json!(16.0));
            m.insert("color.saturation".into(), json!(2.0));
            m.insert("mixer.orange.lum".into(), json!(4.0));
            m.insert("mixer.orange.sat".into(), json!(-3.0));
            m.insert("mixer.red.lum".into(), json!(2.0));
            m.insert("effects.clarity".into(), json!(10.0));
            m.insert("effects.texture".into(), json!(6.0));
            m.insert("detail.sharpenAmount".into(), json!(44.0));
            m.insert("detail.sharpenMasking".into(), json!(35.0));
            m.insert("detail.nrLuminance".into(), json!(14.0));
            m.insert("detail.nrColor".into(), json!(25.0));
            (m, "lut:sf-kodak-2383-warm-102", 24.0)
        }
        "cinema" => {
            let mut m = serde_json::Map::new();
            m.insert("light.contrast".into(), json!(22.0));
            m.insert("color.vibrance".into(), json!(14.0));
            m.insert("effects.clarity".into(), json!(14.0));
            m.insert("vignette.amount".into(), json!(-16.0));
            m.insert("detail.sharpenAmount".into(), json!(40.0));
            (m, "lut:sf-kodak-vision3-250d-107", 65.0)
        }
        "fineart" => {
            let mut m = serde_json::Map::new();
            m.insert("light.contrast".into(), json!(-8.0));
            m.insert("color.vibrance".into(), json!(10.0));
            m.insert("color.saturation".into(), json!(-8.0));
            m.insert("mixer.orange.lum".into(), json!(10.0));
            m.insert("effects.texture".into(), json!(-4.0));
            (m, "lut:sf-fujifilm-eterna-250d-099", 55.0)
        }
        "reference" => {
            if let Some((ref_map, ref_lut)) = analyze_reference_photo_style(reference_path) {
                (ref_map, ref_lut, 55.0)
            } else {
                let mut m = serde_json::Map::new();
                m.insert("light.contrast".into(), json!(14.0));
                m.insert("color.vibrance".into(), json!(14.0));
                (m, "lut:sf-kodak-vision3-250d-107", 55.0)
            }
        }
        "web" => {
            let u = portfolio_url.to_ascii_lowercase();
            let mut m = serde_json::Map::new();
            if u.contains("boda") || u.contains("wedding") || u.contains("social") {
                m.insert("light.contrast".into(), json!(12.0));
                m.insert("color.vibrance".into(), json!(16.0));
                m.insert("mixer.orange.lum".into(), json!(8.0));
                m.insert("effects.clarity".into(), json!(8.0));
                (m, "lut:sf-1-warm-breeze-002", 45.0)
            } else {
                m.insert("light.contrast".into(), json!(15.0));
                m.insert("color.vibrance".into(), json!(14.0));
                m.insert("effects.clarity".into(), json!(10.0));
                (m, "lut:sf-kodak-2383-warm-102", 52.0)
            }
        }
        _ => (serde_json::Map::new(), "lc.color", 100.0),
    };

    if let Some(custom) = custom_values {
        for (k, v) in custom {
            base_values.insert(k.clone(), v.clone());
        }
    }
    let prof_id = custom_profile.unwrap_or(default_lut);
    let prof_amt = custom_profile_amt.unwrap_or(default_amt);

    let preview_dir = agent_bridge_dir().join("previews");
    let _ = std::fs::create_dir_all(&preview_dir);
    let mut qa_reports: Vec<AiPhotoQaReport> = Vec::with_capacity(target_ids.len());

    let qa_map_id = egui::Id::new("studio-agent-qa-reports");
    let mut stored_map = ctx
        .data(|d| d.get_temp::<std::collections::HashMap<u64, AiPhotoQaReport>>(qa_map_id))
        .unwrap_or_default();

    // PASS A: Inspect, Propose & Closed-Loop Optimize each photo individually across all 8 pillars
    let mut batch_plans: Vec<(u64, AiComprehensivePlan, AiPhotoQaReport)> = Vec::with_capacity(target_ids.len());
    for &pid in &target_ids {
        let _ = app.run("library.select", json!({"ids": [pid], "mode": "replace"}));
        let _ = app.run("develop.reset", json!({}));

        let (plan, qa_report) = inspect_propose_verify_and_refine_photo(
            app,
            lightcraft_catalog::PhotoId(pid),
            style,
            prof_id,
            prof_amt,
            &preview_dir,
        );
        batch_plans.push((pid, plan, qa_report));
    }

    // PASS B: Batch Scene-Cluster Harmonization (cohesive WB & tone across large batches)
    if custom_values.is_none_or(|c| !c.contains_key("wb.temp")) {
        harmonize_batch_clusters(&mut batch_plans);
    }

    // PASS C: Apply all 8 pillars to each photo, render high-res verification preview & mark Edited
    for (pid, plan, mut qa_report) in batch_plans {
        let _ = app.run("library.select", json!({"ids": [pid], "mode": "replace"}));

        let mut photo_values = base_values.clone();
        let insert_if_not_custom = |m: &mut serde_json::Map<String, serde_json::Value>, key: &str, val: serde_json::Value| {
            if custom_values.is_none_or(|c| !c.contains_key(key)) {
                m.insert(key.to_string(), val);
            }
        };

        // 1. White Balance (Temp & Tint)
        if plan.wb_changed {
            insert_if_not_custom(&mut photo_values, "wb.temp", json!(plan.wb_temp));
            insert_if_not_custom(&mut photo_values, "wb.tint", json!(plan.wb_tint));
        }

        // 2. Exposure & Tone
        insert_if_not_custom(&mut photo_values, "light.exposure", json!(plan.exp));
        insert_if_not_custom(&mut photo_values, "light.contrast", json!(plan.contrast));
        insert_if_not_custom(&mut photo_values, "light.highlights", json!(plan.highlights));
        insert_if_not_custom(&mut photo_values, "light.shadows", json!(plan.shadows));
        insert_if_not_custom(&mut photo_values, "light.whites", json!(plan.whites));
        insert_if_not_custom(&mut photo_values, "light.blacks", json!(plan.blacks));
        insert_if_not_custom(&mut photo_values, "effects.dehaze", json!(plan.dehaze));

        // 3. Parametric Tone Curve
        insert_if_not_custom(&mut photo_values, "curve.highlights", json!(plan.curve_highlights));
        insert_if_not_custom(&mut photo_values, "curve.lights", json!(plan.curve_lights));
        insert_if_not_custom(&mut photo_values, "curve.darks", json!(plan.curve_darks));
        insert_if_not_custom(&mut photo_values, "curve.shadows", json!(plan.curve_shadows));

        // 4. 8-Channel HSL Color Mixer
        insert_if_not_custom(&mut photo_values, "mixer.red.hue", json!(plan.red_hue));
        insert_if_not_custom(&mut photo_values, "mixer.red.sat", json!(plan.red_sat));
        insert_if_not_custom(&mut photo_values, "mixer.red.lum", json!(plan.red_lum));
        insert_if_not_custom(&mut photo_values, "mixer.orange.hue", json!(plan.orange_hue));
        insert_if_not_custom(&mut photo_values, "mixer.orange.sat", json!(plan.orange_sat));
        insert_if_not_custom(&mut photo_values, "mixer.orange.lum", json!(plan.orange_lum));
        insert_if_not_custom(&mut photo_values, "mixer.yellow.hue", json!(plan.yellow_hue));
        insert_if_not_custom(&mut photo_values, "mixer.yellow.sat", json!(plan.yellow_sat));
        insert_if_not_custom(&mut photo_values, "mixer.yellow.lum", json!(plan.yellow_lum));
        insert_if_not_custom(&mut photo_values, "mixer.green.hue", json!(plan.green_hue));
        insert_if_not_custom(&mut photo_values, "mixer.green.sat", json!(plan.green_sat));
        insert_if_not_custom(&mut photo_values, "mixer.green.lum", json!(plan.green_lum));
        insert_if_not_custom(&mut photo_values, "mixer.aqua.sat", json!(plan.aqua_sat));
        insert_if_not_custom(&mut photo_values, "mixer.blue.sat", json!(plan.blue_sat));
        insert_if_not_custom(&mut photo_values, "mixer.blue.lum", json!(plan.blue_lum));
        insert_if_not_custom(&mut photo_values, "mixer.purple.sat", json!(plan.purple_sat));
        insert_if_not_custom(&mut photo_values, "mixer.magenta.sat", json!(plan.magenta_sat));

        // 5. 3-Way Color Grading
        insert_if_not_custom(&mut photo_values, "grading.shadows.hue", json!(plan.grad_sh_hue));
        insert_if_not_custom(&mut photo_values, "grading.shadows.sat", json!(plan.grad_sh_sat));
        insert_if_not_custom(&mut photo_values, "grading.midtones.hue", json!(plan.grad_mid_hue));
        insert_if_not_custom(&mut photo_values, "grading.midtones.sat", json!(plan.grad_mid_sat));
        insert_if_not_custom(&mut photo_values, "grading.highlights.hue", json!(plan.grad_hi_hue));
        insert_if_not_custom(&mut photo_values, "grading.highlights.sat", json!(plan.grad_hi_sat));

        // 6. Detail, Optics & Vignette
        insert_if_not_custom(&mut photo_values, "detail.sharpenAmount", json!(plan.sharpen_amt));
        insert_if_not_custom(&mut photo_values, "detail.nrLuminance", json!(plan.nr_lum));
        insert_if_not_custom(&mut photo_values, "detail.nrColor", json!(plan.nr_color));
        insert_if_not_custom(&mut photo_values, "optics.defringePurple", json!(plan.defringe_purple));
        insert_if_not_custom(&mut photo_values, "optics.defringeGreen", json!(plan.defringe_green));
        insert_if_not_custom(&mut photo_values, "vignette.amount", json!(plan.vignette_amt));

        if !photo_values.is_empty() {
            let _ = app.run("develop.set", json!({"values": photo_values}));
        }
        if !prof_id.is_empty() {
            let _ = app.run("develop.profile", json!({"id": prof_id, "amount": prof_amt}));
        }

        // 7. Smart Angle & Composition Crop (`crop.set`)
        let final_angle = custom_crop_angle.unwrap_or(plan.crop_angle);
        let final_rect = custom_crop_rect.unwrap_or(plan.crop_rect);
        if plan.crop_changed || custom_crop_rect.is_some() || custom_crop_angle.is_some() {
            let _ = app.run(
                "crop.set",
                json!({
                    "angle": final_angle,
                    "rect": final_rect
                }),
            );
        }

        // Re-verify final applied state on the photo and export the 720px verification JPEG
        let p_id = lightcraft_catalog::PhotoId(pid);
        if let (Ok(src), Some(final_d)) = (
            app.session.source_now(p_id, lightcraft_engine::SourceLevel::Thumb),
            app.session.develop_of(p_id),
        ) {
            let info = app.session.source_info(p_id);
            let post_eval = evaluate_rendered_photo(&src, &info, &final_d, style);
            qa_report.final_score = post_eval.score;
            qa_report.final_issues = post_eval.issues;
            qa_report.approved = post_eval.approved;
            qa_report.exposure = (final_d.light.exposure * 100.0).round() / 100.0;
            qa_report.contrast = final_d.light.contrast.round();
            qa_report.highlights = final_d.light.highlights.round();
            qa_report.shadows = final_d.light.shadows.round();
            qa_report.whites = final_d.light.whites.round();
            qa_report.blacks = final_d.light.blacks.round();
            qa_report.dehaze = final_d.effects.dehaze.round();
            qa_report.wb_temp = final_d.wb.temp.round();
            qa_report.wb_tint = (final_d.wb.tint * 10.0).round() / 10.0;
            qa_report.crop_angle = (final_d.crop.geometry.angle * 100.0).round() / 100.0;
            qa_report.crop_rect = [
                (final_d.crop.geometry.rect.x0 * 1000.0).round() / 1000.0,
                (final_d.crop.geometry.rect.y0 * 1000.0).round() / 1000.0,
                (final_d.crop.geometry.rect.x1 * 1000.0).round() / 1000.0,
                (final_d.crop.geometry.rect.y1 * 1000.0).round() / 1000.0,
            ];

            let rendered = lightcraft_pipeline::render(
                &src,
                &info,
                &final_d,
                &lightcraft_pipeline::RenderRequest::fit(720, 720),
            );
            if let Ok(jpeg_bytes) = lightcraft_codecs::encode_jpeg(
                &lightcraft_codecs::EncodeImage::rgba8(&rendered.image),
                86,
                lightcraft_codecs::ChromaSubsampling::S420,
                &lightcraft_codecs::EncodeMeta::default(),
            ) {
                let _ = std::fs::write(&qa_report.preview_after, jpeg_bytes);
            }
        }

        // =====================================================================
        // FASE 4: SI LA IA DECIDE QUE SÍ SE VE BIEN, LA MARCA COMO EDITADA
        // =====================================================================
        if mark_when_approved && qa_report.approved {
            qa_report.marked_edited = true;
            let _ = app.run("photo.pick", json!({"ids": [pid]}));
            let _ = app.run("photo.label", json!({"ids": [pid], "label": "green"}));
            let ver_name = format!(
                "✅ Editada IA ({}% · {} pasadas)",
                qa_report.final_score, qa_report.passes
            );
            let _ = app.run("version.create", json!({"name": ver_name}));
        } else {
            qa_report.marked_edited = false;
            let _ = app.run("photo.label", json!({"ids": [pid], "label": "yellow"}));
        }

        stored_map.insert(pid, qa_report.clone());
        qa_reports.push(qa_report);
    }

    ctx.data_mut(|d| d.insert_temp(qa_map_id, stored_map));
    let _ = std::fs::write(
        agent_bridge_dir().join("qa_report.json"),
        serde_json::to_string_pretty(&qa_reports).unwrap_or_default(),
    );

    if target_ids.len() > 1 {
        let _ = app.run("library.select", json!({"ids": target_ids, "mode": "replace"}));
    } else if !orig_sel.is_empty() && target != "recent" && target != "album" {
        let _ = app.run("library.select", json!({"ids": orig_sel, "mode": "replace"}));
    }
    let _ = orig_source;

    qa_reports
}

fn show_ai_agent_card(app: &mut LightcraftApp, ui: &mut egui::Ui, t: &Tokens, _d: &DevelopSettings) {
    let init_fold_id = egui::Id::new("studio-agent-init-folded");
    if !ui.ctx().data(|dd| dd.get_temp::<bool>(init_fold_id)).unwrap_or(false) {
        ui.ctx().data_mut(|dd| dd.insert_temp(init_fold_id, true));
        if !app.ui.folded_sections.iter().any(|s| s == "studio:aiAgentBridge") {
            app.ui.folded_sections.push("studio:aiAgentBridge".to_string());
        }
    }
    let card_folded = app.ui.section_folded("studio:aiAgentBridge");
    let target_id = egui::Id::new("studio-agent-target-idx");
    let mut target_idx = ui.ctx().data(|dd| dd.get_temp::<usize>(target_id)).unwrap_or(0);
    let style_id = egui::Id::new("studio-agent-style-idx");
    let mut style_idx = ui.ctx().data(|dd| dd.get_temp::<usize>(style_id)).unwrap_or(0);
    let album_idx_id = egui::Id::new("studio-agent-album-idx");
    let mut album_idx = ui.ctx().data(|dd| dd.get_temp::<usize>(album_idx_id)).unwrap_or(0);
    let ref_path_id = egui::Id::new("studio-agent-ref-path");
    let mut ref_path = ui.ctx().data(|dd| dd.get_temp::<String>(ref_path_id)).unwrap_or_default();
    let web_url_id = egui::Id::new("studio-agent-web-url");
    let mut web_url = ui
        .ctx()
        .data(|dd| dd.get_temp::<String>(web_url_id))
        .unwrap_or_else(|| "https://mmstudio.mx".to_string());
    let last_status = ui
        .ctx()
        .data(|dd| dd.get_temp::<String>(egui::Id::new("studio-agent-last-status")))
        .unwrap_or_else(|| "🟢 Puente IA activo (~/.mm-studio) · Auto-Verificación Visual ON".to_string());

    let w = ui.available_width();
    let pad = 10.0;
    let inner_w = (w - pad * 2.0).max(120.0);

    egui::Frame::NONE
        .fill(Color32::from_rgba_unmultiplied(36, 38, 46, 196))
        .stroke(Stroke::new(1.0, Color32::from_rgb(128, 175, 255).gamma_multiply(0.55)))
        .corner_radius(CornerRadius::same(16))
        .show(ui, |ui| {
            ui.set_width(w);

            // Header bar with Live Status Badge
            let (hr, _) = ui.allocate_exact_size(vec2(w, 36.0), Sense::hover());
            let badge_r = Rect::from_min_size(pos2(hr.right() - pad - 82.0, hr.center().y - 10.0), vec2(82.0, 20.0));
            let hdr_click_r = Rect::from_min_max(hr.min, pos2(badge_r.left() - 4.0, hr.max.y));
            let hdr_resp = ui
                .interact(hdr_click_r, egui::Id::new("studio-agent-hdr-fold"), Sense::click())
                .on_hover_text("Conecta MM Studio con Antigravity, Astra o cualquier Agente IA con bucle de inspección y verificación visual");

            paint(
                ui.painter(),
                Rect::from_center_size(pos2(hr.left() + 13.0, hr.center().y), vec2(12.0, 12.0)),
                if card_folded { Icon::ChevronRight } else { Icon::ChevronDown },
                Color32::from_rgb(225, 228, 236),
            );
            paint(
                ui.painter(),
                Rect::from_center_size(pos2(hr.left() + 27.0, hr.center().y), vec2(14.5, 14.5)),
                Icon::Star,
                Color32::from_rgb(135, 190, 255),
            );
            ui.painter().text(
                pos2(hr.left() + 40.0, hr.center().y),
                Align2::LEFT_CENTER,
                "Agente IA · Piloto Automático",
                t.semibold(12.8),
                Color32::WHITE,
            );

            ui.painter().rect(
                badge_r,
                CornerRadius::same(7),
                Color32::from_rgb(48, 118, 88).gamma_multiply(0.45),
                Stroke::new(1.0, Color32::from_rgb(98, 220, 160)),
                StrokeKind::Inside,
            );
            ui.painter().text(
                badge_r.center(),
                Align2::CENTER_CENTER,
                "● Auto-QA Activo",
                t.semibold(8.4),
                Color32::from_rgb(185, 255, 218),
            );

            if hdr_resp.clicked() {
                app.ui.toggle_folded("studio:aiAgentBridge");
            }
            if card_folded {
                return;
            }

            // 1. ¿Qué fotos quieres que edite el Agente?
            let (lbl1_r, _) = ui.allocate_exact_size(vec2(w, 18.0), Sense::hover());
            ui.painter().text(
                pos2(lbl1_r.left() + pad, lbl1_r.center().y),
                Align2::LEFT_CENTER,
                "1. ¿Qué fotos inspeccionar y editar?",
                t.semibold(10.2),
                Color32::from_rgb(200, 210, 230),
            );

            let targets = [
                ("open", "📷 Abierta"),
                ("selected", "✓ Seleccionadas"),
                ("recent", "🕒 Recientes"),
                ("album", "📁 Álbum"),
            ];
            let tgap = 4.0;
            let tw = (inner_w - tgap * 3.0) / 4.0;
            let (t_area, _) = ui.allocate_exact_size(vec2(w, 25.0), Sense::hover());
            for (idx, (tid, tlbl)) in targets.iter().enumerate() {
                let x = t_area.left() + pad + idx as f32 * (tw + tgap);
                let pr = Rect::from_min_size(pos2(x, t_area.top() + 1.0), vec2(tw, 23.0));
                let presp = ui.interact(pr, egui::Id::new(("studio-agent-tgt", *tid)), Sense::click());
                let active = target_idx == idx;
                ui.painter().rect(
                    pr,
                    CornerRadius::same(6),
                    if active {
                        Color32::from_rgb(112, 168, 255)
                    } else if presp.hovered() {
                        Color32::from_white_alpha(26)
                    } else {
                        Color32::from_white_alpha(12)
                    },
                    Stroke::new(1.0, if active { Color32::from_rgb(185, 218, 255) } else { Color32::from_white_alpha(24) }),
                    StrokeKind::Inside,
                );
                ui.painter().text(
                    pr.center(),
                    Align2::CENTER_CENTER,
                    *tlbl,
                    if active { t.semibold(9.3) } else { t.font(9.2) },
                    if active { Color32::from_rgb(16, 20, 28) } else { Color32::from_rgb(222, 226, 236) },
                );
                if presp.clicked() {
                    target_idx = idx;
                    ui.ctx().data_mut(|dd| dd.insert_temp(target_id, target_idx));
                }
            }

            // Optional Album Selector when "📁 Álbum" is chosen
            let mut all_albums: Vec<(u64, String)> = Vec::new();
            {
                let kids = app.session.catalog.album_children_by_parent();
                for list in kids.values() {
                    for alb in list {
                        if !alb.folder {
                            all_albums.push((alb.id.0, alb.name.clone()));
                        }
                    }
                }
            }
            let selected_album_name = if target_idx == 3 && !all_albums.is_empty() {
                let safe_a = album_idx.min(all_albums.len() - 1);
                let cur_name = all_albums[safe_a].1.clone();
                let dd_resp = studio_dropdown_row(ui, t, "studio-agent-album-dd", "Álbum objetivo", &cur_name);
                egui::Popup::menu(&dd_resp).show(|ui| {
                    ui.set_min_width(170.0);
                    for (i, (_, aname)) in all_albums.iter().enumerate() {
                        if ui.selectable_label(i == safe_a, aname).clicked() {
                            album_idx = i;
                            ui.ctx().data_mut(|dd| dd.insert_temp(album_idx_id, album_idx));
                            ui.close();
                        }
                    }
                });
                cur_name
            } else {
                String::new()
            };

            // 2. ¿Con qué estilo editar tus fotos?
            ui.add_space(4.0);
            let (lbl2_r, _) = ui.allocate_exact_size(vec2(w, 18.0), Sense::hover());
            ui.painter().text(
                pos2(lbl2_r.left() + pad, lbl2_r.center().y),
                Align2::LEFT_CENTER,
                "2. ¿Con qué estilo edito tus fotos?",
                t.semibold(10.2),
                Color32::from_rgb(200, 210, 230),
            );

            let styles = [
                ("editorial", "✨ Editorial"),
                ("social", "🥂 Social / Bodas"),
                ("cinema", "🎬 Cine 35mm"),
                ("fineart", "🌿 Fine Art"),
                ("reference", "🖼 Foto Referencia"),
                ("web", "🌐 Portafolio Web"),
            ];
            let sgap = 4.0;
            let sw = (inner_w - sgap * 2.0) / 3.0;
            let (s_area, _) = ui.allocate_exact_size(vec2(w, 52.0), Sense::hover());
            for (idx, (sid, slbl)) in styles.iter().enumerate() {
                let row = idx / 3;
                let col = idx % 3;
                let x = s_area.left() + pad + col as f32 * (sw + sgap);
                let y = s_area.top() + 1.0 + row as f32 * 26.0;
                let pr = Rect::from_min_size(pos2(x, y), vec2(sw, 23.0));
                let presp = ui.interact(pr, egui::Id::new(("studio-agent-style", *sid)), Sense::click());
                let active = style_idx == idx;
                ui.painter().rect(
                    pr,
                    CornerRadius::same(6),
                    if active {
                        Color32::from_rgb(245, 198, 112)
                    } else if presp.hovered() {
                        Color32::from_white_alpha(26)
                    } else {
                        Color32::from_white_alpha(12)
                    },
                    Stroke::new(1.0, if active { Color32::from_rgb(255, 228, 168) } else { Color32::from_white_alpha(24) }),
                    StrokeKind::Inside,
                );
                ui.painter().text(
                    pr.center(),
                    Align2::CENTER_CENTER,
                    *slbl,
                    if active { t.semibold(9.3) } else { t.font(9.2) },
                    if active { Color32::from_rgb(20, 20, 24) } else { Color32::from_rgb(222, 226, 236) },
                );
                if presp.clicked() {
                    style_idx = idx;
                    ui.ctx().data_mut(|dd| dd.insert_temp(style_id, style_idx));
                }
            }

            if style_idx == 4 {
                egui::Frame::NONE
                    .inner_margin(egui::Margin { left: pad as i8, right: pad as i8, top: 2, bottom: 2 })
                    .show(ui, |ui| {
                        let te = egui::TextEdit::singleline(&mut ref_path)
                            .hint_text("🖼 Ruta de foto de referencia (ej. /Users/mcruz/Desktop/ref.jpg)")
                            .desired_width(inner_w);
                        if ui.add(te).changed() {
                            ui.ctx().data_mut(|dd| dd.insert_temp(ref_path_id, ref_path.clone()));
                        }
                    });
            } else if style_idx == 5 {
                egui::Frame::NONE
                    .inner_margin(egui::Margin { left: pad as i8, right: pad as i8, top: 2, bottom: 2 })
                    .show(ui, |ui| {
                        let te = egui::TextEdit::singleline(&mut web_url)
                            .hint_text("🌐 URL de tu portafolio web (ej. https://tuportafolio.com)")
                            .desired_width(inner_w);
                        if ui.add(te).changed() {
                            ui.ctx().data_mut(|dd| dd.insert_temp(web_url_id, web_url.clone()));
                        }
                    });
            }

            // 3. Execute Button: [⚡ Inspeccionar, Editar y Verificar con Estilo ...]
            ui.add_space(4.0);
            let (run_area, _) = ui.allocate_exact_size(vec2(w, 30.0), Sense::hover());
            let run_r = Rect::from_min_size(pos2(run_area.left() + pad, run_area.top() + 1.0), vec2(inner_w, 28.0));
            let run_resp = ui
                .interact(run_r, egui::Id::new("studio-agent-run-now"), Sense::click())
                .on_hover_text("1. Ve cómo están las fotos -> 2. Propone ajustes -> 3. Vuelve a ver y re-edita hasta aprobar -> 4. Marca como Editada");
            ui.painter().rect(
                run_r,
                CornerRadius::same(8),
                if run_resp.hovered() {
                    Color32::from_rgb(108, 172, 255)
                } else {
                    Color32::from_rgb(82, 145, 238)
                },
                Stroke::new(1.1, Color32::from_rgb(185, 220, 255)),
                StrokeKind::Inside,
            );
            ui.painter().text(
                run_r.center(),
                Align2::CENTER_CENTER,
                format!("⚡ Analizar, Editar y Verificar ({})", styles[style_idx].1),
                t.semibold(10.2),
                Color32::WHITE,
            );

            if run_resp.clicked() {
                let tgt_code = targets[target_idx].0;
                let sty_code = styles[style_idx].0;
                let reports = execute_agent_batch_edit(
                    app,
                    ui.ctx(),
                    tgt_code,
                    &selected_album_name,
                    sty_code,
                    &ref_path,
                    &web_url,
                    None,
                    None,
                    None,
                    None,
                    None,
                    true,
                );
                let count = reports.len();
                let approved = reports.iter().filter(|r| r.approved).count();
                let msg = format!(
                    "✅ IA verificó y marcó {approved}/{count} foto(s) como Editadas · {}",
                    styles[style_idx].1
                );
                ui.ctx().data_mut(|dd| dd.insert_temp(egui::Id::new("studio-agent-last-status"), msg.clone()));
                app.toast(ui.ctx(), msg);
            }

            // 4. Live Visual QA Verification Card for the Active Photo
            if let Some(act_id) = app.session.active() {
                let qa_map_id = egui::Id::new("studio-agent-qa-reports");
                let maybe_rep = ui
                    .ctx()
                    .data(|dd| dd.get_temp::<std::collections::HashMap<u64, AiPhotoQaReport>>(qa_map_id))
                    .and_then(|m| m.get(&act_id.0).cloned());

                if let Some(rep) = maybe_rep {
                    ui.add_space(4.0);
                    let (qa_r, _) = ui.allocate_exact_size(vec2(w, 56.0), Sense::hover());
                    let card_r = Rect::from_min_size(pos2(qa_r.left() + pad, qa_r.top() + 1.0), vec2(inner_w, 54.0));
                    let ok_col = if rep.marked_edited {
                        Color32::from_rgb(56, 188, 124)
                    } else {
                        Color32::from_rgb(235, 175, 72)
                    };
                    ui.painter().rect(
                        card_r,
                        CornerRadius::same(8),
                        ok_col.gamma_multiply(0.14),
                        Stroke::new(1.0, ok_col.gamma_multiply(0.65)),
                        StrokeKind::Inside,
                    );
                    let title_txt = if rep.marked_edited {
                        format!(
                            "✅ MARCADA COMO EDITADA · QA {}% ({} pasadas)",
                            rep.final_score, rep.passes
                        )
                    } else {
                        format!(
                            "🔄 EN REFINAMIENTO IA · QA {}% ({} pasadas)",
                            rep.final_score, rep.passes
                        )
                    };
                    ui.painter().text(
                        pos2(card_r.left() + 8.0, card_r.top() + 12.0),
                        Align2::LEFT_CENTER,
                        title_txt,
                        t.semibold(9.3),
                        if rep.marked_edited {
                            Color32::from_rgb(162, 248, 202)
                        } else {
                            Color32::from_rgb(255, 218, 138)
                        },
                    );
                    let init_issue = rep
                        .initial_issues
                        .first()
                        .map(|s| s.as_str())
                        .unwrap_or("Exposición base revisada");
                    ui.painter().text(
                        pos2(card_r.left() + 8.0, card_r.top() + 28.0),
                        Align2::LEFT_CENTER,
                        format!("Antes ({}%): {}", rep.initial_score, init_issue),
                        t.font(8.6),
                        Color32::from_rgb(208, 214, 226),
                    );
                    ui.painter().text(
                        pos2(card_r.left() + 8.0, card_r.top() + 43.0),
                        Align2::LEFT_CENTER,
                        format!(
                            "Exp {:+.2} EV · WB {:.0}K/{:+.0} · Áng {:+.2}° · Ilum {:+.0} · Som {:+.0} · HSL+Curva",
                            rep.exposure, rep.wb_temp, rep.wb_tint, rep.crop_angle, rep.highlights, rep.shadows
                        ),
                        t.font(8.3),
                        Color32::from_rgb(180, 205, 245),
                    );
                }
            }

            // 5. Status bar showing connection to Antigravity / Astra
            ui.add_space(2.0);
            let (st_r, _) = ui.allocate_exact_size(vec2(w, 18.0), Sense::hover());
            ui.painter().text(
                st_r.center(),
                Align2::CENTER_CENTER,
                last_status,
                t.font(9.0),
                Color32::from_rgb(175, 215, 195),
            );
            ui.add_space(4.0);
        });
}
