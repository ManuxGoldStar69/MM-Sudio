//! Full-screen "Pantalla de Inicio" (Home / Welcome screen) for MM Fotografía · Editor de Fotos.
//! Inspired by the user's reference design: scenic landscape background with dark cinematic
//! vignette, white calligraphic MM monogram, "Edita tu historia." hero actions, 4 frosted-glass
//! workflow cards on the right, and a bottom "Proyectos recientes" glass dock.

use egui::{Align2, Color32, CornerRadius, FontId, Rect, Sense, Stroke, StrokeKind, pos2, vec2};
use serde_json::json;

use crate::LightcraftApp;
use crate::icons::{Icon, paint};
use crate::panels::presets::cover_uv;
use crate::render::Slot;
use crate::state::{RightPanel, ViewMode};
use crate::theme::{Tokens, mm_logo_white_texture};
use crate::widgets::register;

/// Loads and caches the scenic hero landscape (`assets/mm_hero_landscape.jpg`) with a warm
/// golden-hour grade and dark cinematic vignette so text and glass cards read effortlessly.
fn hero_landscape_texture(ctx: &egui::Context) -> Option<(egui::TextureHandle, [usize; 2])> {
    type HeroCache = std::collections::HashMap<usize, (egui::TextureHandle, [usize; 2])>;
    static CACHE: std::sync::LazyLock<std::sync::Mutex<HeroCache>> =
        std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let tm_ptr = std::sync::Arc::as_ptr(&ctx.tex_manager()) as usize;
    if let Ok(guard) = CACHE.lock()
        && let Some((h, sz)) = guard.get(&tm_ptr)
        && ctx.tex_manager().read().meta(h.id()).is_some()
    {
        return Some((h.clone(), *sz));
    }
    let bytes = include_bytes!("../../../../assets/mm_hero_landscape.jpg");
    let decoded = lightcraft_codecs::decode(bytes, Default::default()).ok()?;
    let img = decoded.to_srgb8();
    let (w, h) = (img.width, img.height);
    let mut rgba = img.as_bytes();
    let wf = w.max(1) as f32;
    let hf = h.max(1) as f32;
    for y in 0..h {
        let ny = y as f32 / hf;
        for x in 0..w {
            let nx = x as f32 / wf;
            let idx = (y * w + x) * 4;
            let Some(px) = rgba.get_mut(idx..idx + 4) else { continue };
            let r = px[0] as f32 / 255.0;
            let g = px[1] as f32 / 255.0;
            let b = px[2] as f32 / 255.0;
            // Warm golden-hour grade (enrich amber/gold tones, deepen shadows)
            let gr = (r * 1.08 + 0.03).clamp(0.0, 1.0).powf(1.05);
            let gg = (g * 0.96 + 0.01).clamp(0.0, 1.0).powf(1.08);
            let gb = (b * 0.82).clamp(0.0, 1.0).powf(1.15);
            // Smooth directional vignette: darker on left (for hero headline), right (for cards), and bottom (for dock)
            let left_shade = 0.42 + 0.58 * (nx / 0.52).clamp(0.0, 1.0);
            let bottom_shade = 1.0 - 0.55 * ((ny - 0.58) / 0.42).clamp(0.0, 1.0);
            let top_shade = 0.72 + 0.28 * (ny / 0.22).clamp(0.0, 1.0);
            let vig = (left_shade * bottom_shade * top_shade * 0.82).clamp(0.18, 0.92);
            px[0] = (gr * vig * 255.0).round().clamp(0.0, 255.0) as u8;
            px[1] = (gg * vig * 255.0).round().clamp(0.0, 255.0) as u8;
            px[2] = (gb * vig * 255.0).round().clamp(0.0, 255.0) as u8;
        }
    }
    let color = egui::ColorImage::from_rgba_unmultiplied([w, h], &rgba);
    let handle = ctx.load_texture("mm-hero-landscape", color, egui::TextureOptions::LINEAR);
    let out = (handle, [w, h]);
    if let Ok(mut guard) = CACHE.lock() {
        guard.insert(tm_ptr, out.clone());
    }
    Some(out)
}

/// Renders the full-window Home / Welcome screen (`app.ui.show_home`).
pub fn show(app: &mut LightcraftApp, ui: &mut egui::Ui) {
    let t = Tokens::get(ui.ctx());
    let full = ui.max_rect();

    // Smooth entrance animation (220ms cubic ease-out) when entering Home screen
    let anim = crate::panels::studio::screen_transition_progress(ui.ctx(), "home");
    let slide_y = (1.0 - anim) * 10.0;
    let full = full.translate(vec2(0.0, slide_y));

    // 1. Background: Golden-Hour Mountain & River Landscape + dark cinema base
    ui.painter().rect_filled(full, 0.0, Color32::from_rgb(12, 13, 12));
    let hero_tex = hero_landscape_texture(ui.ctx());
    if let Some((tex, size)) = &hero_tex {
        let uv = cover_uv(full, *size);
        ui.painter().image(tex.id(), full, uv, Color32::WHITE);
    }

    // Allow dragging the window by its top 44pt strip (just like the titlebar)
    if app.integrated_titlebar {
        let drag_bar = Rect::from_min_max(full.min, pos2(full.right() - 140.0, full.top() + 44.0));
        let resp = ui.interact(drag_bar, egui::Id::new("home-titlebar-drag"), Sense::click_and_drag());
        if resp.drag_started_by(egui::PointerButton::Primary) {
            ui.ctx().send_viewport_cmd(egui::ViewportCommand::StartDrag);
        } else if resp.double_clicked_by(egui::PointerButton::Primary) {
            let max = ui.input(|i| i.viewport().maximized).unwrap_or(false);
            ui.ctx().send_viewport_cmd(egui::ViewportCommand::Maximized(!max));
        }
    }

    // 2. Top-left brand lockup: MM Monogram + "MM Studio" + "MM FOTOGRAFÍA · ESTUDIO RAW"
    let pad_x = (full.width() * 0.045).clamp(32.0, 68.0);
    let top_y = full.top() + 36.0;
    let logo_rect = Rect::from_min_size(pos2(full.left() + pad_x, top_y), vec2(48.0, 48.0));
    if let Some(tex) = mm_logo_white_texture(ui.ctx()) {
        let uv = Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0));
        ui.painter().image(tex.id(), logo_rect, uv, Color32::WHITE);
    } else {
        ui.painter().text(logo_rect.center(), Align2::CENTER_CENTER, "MM", t.semibold(20.0), Color32::WHITE);
    }
    ui.painter().text(
        pos2(logo_rect.right() + 12.0, top_y + 16.0),
        Align2::LEFT_CENTER,
        "MM Studio",
        t.semibold(20.5),
        Color32::from_rgb(245, 243, 238),
    );
    ui.painter().text(
        pos2(logo_rect.right() + 13.0, top_y + 35.0),
        Align2::LEFT_CENTER,
        "M M   F O T O G R A F Í A   ·   E S T U D I O   R A W",
        t.font(9.5),
        Color32::from_rgb(218, 196, 162),
    );

    // 3. Top-right circular frosted glass utility buttons (Welcome/Donar, Settings & Enter Studio)
    let enter_rect = Rect::from_center_size(pos2(full.right() - pad_x, top_y + 18.0), vec2(38.0, 38.0));
    let settings_rect = Rect::from_center_size(pos2(enter_rect.left() - 28.0, top_y + 18.0), vec2(38.0, 38.0));
    let welcome_rect = Rect::from_center_size(pos2(settings_rect.left() - 28.0, top_y + 18.0), vec2(38.0, 38.0));

    let w_resp = ui
        .interact(welcome_rect, egui::Id::new("home-welcome-btn"), Sense::click())
        .on_hover_text("Acerca de MM Studio · Apoyar el proyecto");
    register(ui.ctx(), "button:homeWelcome", welcome_rect);
    paint_frosted_circle(ui.painter(), welcome_rect, w_resp.hovered());
    paint(ui.painter(), welcome_rect.shrink(10.0), Icon::Heart, Color32::from_rgb(229, 169, 75));
    if w_resp.clicked() {
        app.ui.show_welcome_modal = true;
    }

    let s_resp = ui
        .interact(settings_rect, egui::Id::new("home-settings-btn"), Sense::click())
        .on_hover_text("Ajustes de la aplicación");
    register(ui.ctx(), "button:homeSettings", settings_rect);
    paint_frosted_circle(ui.painter(), settings_rect, s_resp.hovered());
    paint(ui.painter(), settings_rect.shrink(10.0), Icon::Gear, Color32::from_rgb(240, 238, 232));
    if s_resp.clicked() {
        app.ui.dialog = Some(crate::state::Dialog::Settings { tab: "general".into() });
    }

    let e_resp = ui
        .interact(enter_rect, egui::Id::new("home-enter-btn"), Sense::click())
        .on_hover_text("Ir al Estudio de Edición (Esc)");
    register(ui.ctx(), "button:homeEnterStudio", enter_rect);
    paint_frosted_circle(ui.painter(), enter_rect, e_resp.hovered());
    paint(ui.painter(), enter_rect.shrink(10.0), Icon::Single, Color32::from_rgb(240, 238, 232));
    if e_resp.clicked() || (!app.ui.show_welcome_modal && ui.input(|i| i.key_pressed(egui::Key::Escape))) {
        app.ui.show_home = false;
    }

    // Reserve bottom 182pt for the "Proyectos recientes" frosted glass dock
    let dock_h = 172.0;
    let dock_rect = Rect::from_min_max(
        pos2(full.left() + pad_x * 0.65, full.bottom() - dock_h - 18.0),
        pos2(full.right() - pad_x * 0.65, full.bottom() - 18.0),
    );
    let mid_top = top_y + 72.0;
    let mid_bottom = dock_rect.top() - 20.0;
    let mid_cy = ((mid_top + mid_bottom) * 0.5).max(mid_top + 140.0);

    // 4. Left Hero Column: "Edita tu historia." + Subtitle + Primary & Glass Pill Buttons
    let hero_x = full.left() + pad_x;
    let hero_y = mid_cy - 135.0;
    ui.painter().text(
        pos2(hero_x, hero_y),
        Align2::LEFT_TOP,
        "Edita",
        FontId::new(54.0, egui::FontFamily::Name(crate::theme::FONT_SEMIBOLD.into())),
        Color32::from_rgb(246, 244, 238),
    );
    ui.painter().text(
        pos2(hero_x, hero_y + 56.0),
        Align2::LEFT_TOP,
        "tu historia.",
        FontId::proportional(54.0),
        Color32::from_rgb(238, 226, 206),
    );
    ui.painter().text(
        pos2(hero_x + 2.0, hero_y + 128.0),
        Align2::LEFT_TOP,
        "Un espacio pensado para que tus fotos\nse vean como las recuerdas.",
        t.font(14.5),
        Color32::from_rgb(208, 202, 192),
    );

    // Primary warm cream pill button: "+  Abrir fotos  →"
    let btn_w = 286.0f32.min(full.width() * 0.36);
    let open_btn = Rect::from_min_size(pos2(hero_x, hero_y + 184.0), vec2(btn_w, 46.0));
    let open_resp = ui
        .interact(open_btn, egui::Id::new("home-open-photos"), Sense::click())
        .on_hover_text("Importar o abrir fotografías en tu biblioteca");
    register(ui.ctx(), "button:homeOpenPhotos", open_btn);
    let open_fill = if open_resp.hovered() {
        Color32::from_rgb(252, 250, 245)
    } else {
        Color32::from_rgb(238, 234, 226)
    };
    ui.painter().rect(open_btn, 23.0, open_fill, Stroke::new(1.0, Color32::from_white_alpha(220)), StrokeKind::Inside);
    let dark_ink = Color32::from_rgb(22, 24, 22);
    paint(ui.painter(), Rect::from_center_size(pos2(open_btn.left() + 24.0, open_btn.center().y), vec2(15.0, 15.0)), Icon::Plus, dark_ink);
    ui.painter().text(pos2(open_btn.left() + 44.0, open_btn.center().y), Align2::LEFT_CENTER, "Abrir fotos", t.semibold(13.5), dark_ink);
    paint(ui.painter(), Rect::from_center_size(pos2(open_btn.right() - 24.0, open_btn.center().y), vec2(15.0, 15.0)), Icon::Forward, dark_ink);
    if open_resp.clicked() {
        app.ui.show_home = false;
        let _ = app.run("file.addPhotos", json!({}));
    }

    // Secondary frosted dark-glass pill buttons: "Abrir carpeta" & "Importar"
    let sec_y = open_btn.bottom() + 12.0;
    let sec_w = (btn_w - 10.0) * 0.5;
    let folder_btn = Rect::from_min_size(pos2(hero_x, sec_y), vec2(sec_w, 40.0));
    let import_btn = Rect::from_min_size(pos2(folder_btn.right() + 10.0, sec_y), vec2(sec_w, 40.0));

    let f_resp = ui
        .interact(folder_btn, egui::Id::new("home-open-folder"), Sense::click())
        .on_hover_text("Importar carpeta completa de fotografías");
    register(ui.ctx(), "button:homeOpenFolder", folder_btn);
    paint_frosted_pill(ui.painter(), folder_btn, f_resp.hovered(), 20.0);
    let light_ink = Color32::from_rgb(238, 235, 228);
    paint(ui.painter(), Rect::from_center_size(pos2(folder_btn.left() + 22.0, folder_btn.center().y), vec2(15.0, 15.0)), Icon::Folder, light_ink);
    ui.painter().text(pos2(folder_btn.left() + 38.0, folder_btn.center().y), Align2::LEFT_CENTER, "Abrir carpeta", t.font(12.5), light_ink);
    if f_resp.clicked() {
        app.ui.show_home = false;
        let _ = app.run("file.addFolder", json!({}));
    }

    let i_resp = ui
        .interact(import_btn, egui::Id::new("home-import-presets"), Sense::click())
        .on_hover_text("Importar Presets de Lightroom (.xmp) o LUTs de DaVinci (.cube)");
    register(ui.ctx(), "button:homeImportPresets", import_btn);
    paint_frosted_pill(ui.painter(), import_btn, i_resp.hovered(), 20.0);
    paint(ui.painter(), Rect::from_center_size(pos2(import_btn.left() + 22.0, import_btn.center().y), vec2(15.0, 15.0)), Icon::Presets, light_ink);
    ui.painter().text(pos2(import_btn.left() + 38.0, import_btn.center().y), Align2::LEFT_CENTER, "Importar", t.font(12.5), light_ink);
    if i_resp.clicked() {
        app.ui.show_home = false;
        app.ui.view = ViewMode::Detail;
        app.ui.right = RightPanel::Edit;
        app.ui.presets = true;
        let _ = app.run("file.importPresets", json!({}));
    }

    // 5. Right Column: 4 Frosted Glass Feature Cards with Mini Thumbnails
    let card_w = 370.0f32.min(full.width() * 0.40);
    let card_h = 74.0;
    let card_gap = 10.0;
    let cards_total_h = 4.0 * card_h + 3.0 * card_gap;
    let cards_x = full.right() - pad_x - card_w;
    let cards_start_y = (mid_cy - cards_total_h * 0.5).max(top_y + 58.0);

    let features: [(&str, Icon, &str, &str, [f32; 4]); 4] = [
        (
            "homeCardAdjust",
            Icon::Sun,
            "Ajusta",
            "Luz, color y detalle\ncon controles intuitivos.",
            [0.32, 0.22, 0.68, 0.62],
        ),
        (
            "homeCardStyle",
            Icon::Presets,
            "Crea tu estilo",
            "Presets LR (.xmp) y LUTs (.cube)\ninspirados en mi trabajo.",
            [0.05, 0.35, 0.45, 0.78],
        ),
        (
            "homeCardOrganize",
            Icon::Album,
            "Organiza",
            "Álbumes, etiquetas\ny búsqueda rápida.",
            [0.52, 0.12, 0.95, 0.55],
        ),
        (
            "homeCardPro",
            Icon::Sliders,
            "Resultado profesional",
            "Edición precisa y Match IA\npara cualquier historia.",
            [0.25, 0.55, 0.75, 0.95],
        ),
    ];

    for (idx, (id, icon, title, desc, uv_crop)) in features.iter().enumerate() {
        let cy = cards_start_y + idx as f32 * (card_h + card_gap);
        let card_r = Rect::from_min_size(pos2(cards_x, cy), vec2(card_w, card_h));
        let resp = ui.interact(card_r, egui::Id::new(("home-feature", *id)), Sense::click());
        register(ui.ctx(), format!("card:{id}"), card_r);
        paint_frosted_pill(ui.painter(), card_r, resp.hovered(), 14.0);

        // Left circular icon badge
        let badge_c = pos2(card_r.left() + 28.0, card_r.center().y);
        let badge_r = Rect::from_center_size(badge_c, vec2(32.0, 32.0));
        ui.painter().circle_filled(badge_c, 16.0, Color32::from_rgba_unmultiplied(255, 255, 255, 18));
        paint(ui.painter(), badge_r.shrink(7.0), *icon, Color32::from_rgb(240, 232, 215));

        // Right mini preview thumbnail inside the card
        let thumb_w = 88.0;
        let thumb_h = card_h - 16.0;
        let thumb_r = Rect::from_min_size(pos2(card_r.right() - thumb_w - 8.0, card_r.top() + 8.0), vec2(thumb_w, thumb_h));
        ui.painter().rect_filled(thumb_r, 8.0, Color32::from_rgb(24, 26, 24));
        if let Some((tex, _)) = &hero_tex {
            let uv = Rect::from_min_max(pos2(uv_crop[0], uv_crop[1]), pos2(uv_crop[2], uv_crop[3]));
            ui.painter().image(tex.id(), thumb_r.shrink(1.0), uv, Color32::WHITE);
        }
        ui.painter().rect_stroke(thumb_r, 8.0, Stroke::new(1.0, Color32::from_white_alpha(38)), StrokeKind::Inside);

        // Title and 2-line description
        let text_x = badge_r.right() + 12.0;
        ui.painter().text(
            pos2(text_x, card_r.top() + 16.0),
            Align2::LEFT_TOP,
            *title,
            t.semibold(13.5),
            Color32::from_rgb(246, 244, 238),
        );
        ui.painter().text(
            pos2(text_x, card_r.top() + 34.0),
            Align2::LEFT_TOP,
            *desc,
            t.font(11.0),
            Color32::from_rgb(188, 184, 176),
        );

        if resp.clicked() {
            app.ui.show_home = false;
            match idx {
                0 => {
                    crate::panels::studio::ensure_active_photo(app);
                    let _ = app.run("view.detail", json!({}));
                    app.ui.right = RightPanel::Edit;
                }
                1 => {
                    crate::panels::studio::ensure_active_photo(app);
                    let _ = app.run("view.detail", json!({}));
                    app.ui.right = RightPanel::Edit;
                    app.ui.presets = true;
                }
                2 => {
                    app.ui.left_panel = true;
                    let _ = app.run("view.photoGrid", json!({}));
                }
                _ => {
                    crate::panels::studio::ensure_active_photo(app);
                    let _ = app.run("view.detail", json!({}));
                    app.ui.right = RightPanel::Edit;
                    let _ = app.run("develop.smartMatch", json!({}));
                }
            }
        }
    }

    // 6. Bottom Frosted Glass Dock: "Fotografías de prueba · Inicio"
    paint_frosted_pill(ui.painter(), dock_rect, false, 14.0);
    ui.painter().text(
        pos2(dock_rect.left() + 20.0, dock_rect.top() + 16.0),
        Align2::LEFT_TOP,
        "Fotografías recientes · Inicio",
        t.semibold(13.0),
        Color32::from_rgb(242, 240, 234),
    );

    let visible_photos = app.session.visible().to_vec();
    let default_crops: [[f32; 4]; 6] = [
        [0.05, 0.18, 0.38, 0.72],
        [0.22, 0.25, 0.56, 0.78],
        [0.40, 0.30, 0.75, 0.85],
        [0.15, 0.45, 0.52, 0.92],
        [0.55, 0.18, 0.92, 0.70],
        [0.62, 0.08, 0.98, 0.60],
    ];

    let row_left = dock_rect.left() + 20.0;
    let row_right = dock_rect.right() - 56.0;
    let row_w = (row_right - row_left).max(200.0);
    let n_cards = 6usize;
    let c_gap = 12.0;
    let item_w = ((row_w - c_gap * (n_cards as f32 - 1.0)) / n_cards as f32).max(90.0);
    let img_h = 84.0;
    let item_top = dock_rect.top() + 40.0;

    for i in 0..n_cards {
        let ix = row_left + i as f32 * (item_w + c_gap);
        let item_rect = Rect::from_min_size(pos2(ix, item_top), vec2(item_w, img_h + 36.0));
        let img_rect = Rect::from_min_size(pos2(ix, item_top), vec2(item_w, img_h));
        let photo_id = visible_photos.get(i).copied();
        let crop = default_crops[i];
        let (card_title, card_sub) = photo_id
            .and_then(|pid| app.session.catalog.photo(pid))
            .map(|p| (p.file_name.clone(), format!("Inicio · {}×{}", p.width, p.height)))
            .unwrap_or_else(|| (format!("Muestra {:02}", i + 1), "Álbum Inicio".to_string()));

        let resp = ui
            .interact(item_rect, egui::Id::new(("home-recent-project", i)), Sense::click())
            .on_hover_text("Abrir en el editor");
        register(ui.ctx(), format!("card:homeRecent:{i}"), item_rect);

        ui.painter().rect_filled(img_rect, 8.0, Color32::from_rgb(22, 24, 26));
        let mut drew_photo = false;
        if let Some(pid) = photo_id {
            if let Some(t_entry) = app.renderer.textures.get(&Slot::Thumb(pid)) {
                let uv = cover_uv(img_rect, t_entry.size);
                ui.painter().image(t_entry.tex.id(), img_rect, uv, Color32::WHITE);
                drew_photo = true;
            } else if let Some(job) = app.session.thumb_job(pid, 320) {
                app.renderer.request(Slot::Thumb(pid), job, 10);
            }
        }
        if !drew_photo && let Some((tex, _)) = &hero_tex {
            let uv = Rect::from_min_max(pos2(crop[0], crop[1]), pos2(crop[2], crop[3]));
            ui.painter().image(tex.id(), img_rect, uv, Color32::WHITE);
        }
        let border_col = if resp.hovered() {
            Color32::from_rgb(229, 169, 75)
        } else {
            Color32::from_white_alpha(36)
        };
        ui.painter().rect_stroke(img_rect, 8.0, Stroke::new(if resp.hovered() { 1.5 } else { 1.0 }, border_col), StrokeKind::Inside);

        ui.painter().text(
            pos2(ix + 2.0, img_rect.bottom() + 6.0),
            Align2::LEFT_TOP,
            card_title,
            t.semibold(11.5),
            Color32::from_rgb(238, 236, 230),
        );
        ui.painter().text(
            pos2(ix + 2.0, img_rect.bottom() + 21.0),
            Align2::LEFT_TOP,
            card_sub,
            t.font(10.0),
            Color32::from_rgb(156, 154, 148),
        );

        if resp.clicked() {
            if let Some(pid) = photo_id {
                let _ = app.run("library.select", json!({"ids": [pid.0]}));
            }
            app.ui.show_home = false;
            app.ui.view = ViewMode::Detail;
            app.ui.right = RightPanel::Edit;
        }
    }

    // Right arrow button in the dock to jump to the full Library Grid
    let next_btn = Rect::from_center_size(pos2(dock_rect.right() - 28.0, item_top + img_h * 0.5), vec2(32.0, 32.0));
    let n_resp = ui
        .interact(next_btn, egui::Id::new("home-all-projects"), Sense::click())
        .on_hover_text("Ver toda la biblioteca");
    register(ui.ctx(), "button:homeAllProjects", next_btn);
    paint_frosted_circle(ui.painter(), next_btn, n_resp.hovered());
    paint(ui.painter(), next_btn.shrink(9.0), Icon::ChevronRight, Color32::from_rgb(240, 238, 232));
    if n_resp.clicked() {
        app.ui.show_home = false;
        app.ui.left_panel = true;
        let _ = app.run("view.photoGrid", json!({}));
    }

    if anim < 0.999 {
        let veil_a = ((1.0 - anim) * 115.0).round().clamp(0.0, 255.0) as u8;
        ui.painter().rect_filled(ui.max_rect(), 0.0, Color32::from_black_alpha(veil_a));
    }

    draw_welcome_modal(app, ui);
}

/// Floating closable Welcome & Optional Donation modal window.
pub fn draw_welcome_modal(app: &mut LightcraftApp, ui: &mut egui::Ui) {
    if !app.ui.show_welcome_modal {
        return;
    }

    let t = Tokens::get(ui.ctx());
    let screen = ui.max_rect();

    // Backdrop veil
    ui.painter().rect_filled(screen, 0.0, Color32::from_black_alpha(158));

    let modal_w = 510.0f32.min(screen.width() - 32.0);
    let modal_h = 415.0f32.min(screen.height() - 32.0);
    let card = Rect::from_center_size(screen.center(), vec2(modal_w, modal_h));

    // Close if user presses Escape or clicks outside the card
    let bg_resp = ui.interact(screen, egui::Id::new("welcome-modal-backdrop"), Sense::click());
    // Absorb clicks inside the card so backdrop click only triggers outside
    let _card_absorb = ui.interact(card, egui::Id::new("welcome-modal-card"), Sense::click());

    // Frosted dark glass modal card
    ui.painter().rect(
        card,
        CornerRadius::same(18),
        Color32::from_rgba_unmultiplied(19, 21, 20, 246),
        Stroke::new(1.0, Color32::from_white_alpha(58)),
        StrokeKind::Inside,
    );
    // Warm golden top specular accent line
    ui.painter().line_segment(
        [pos2(card.left() + 28.0, card.top() + 1.0), pos2(card.right() - 28.0, card.top() + 1.0)],
        Stroke::new(1.5, Color32::from_rgba_unmultiplied(229, 169, 75, 160)),
    );

    // Top-right Close (✕) button
    let close_rect = Rect::from_center_size(pos2(card.right() - 24.0, card.top() + 24.0), vec2(28.0, 28.0));
    let close_resp = ui
        .interact(close_rect, egui::Id::new("welcome-modal-close"), Sense::click())
        .on_hover_text("Cerrar ventana");
    register(ui.ctx(), "button:welcomeClose", close_rect);
    paint_frosted_circle(ui.painter(), close_rect, close_resp.hovered());
    paint(ui.painter(), close_rect.shrink(8.0), Icon::Close, Color32::from_rgb(228, 224, 216));

    // Top MM Logo
    let logo_size = vec2(52.0, 52.0);
    let logo_rect = Rect::from_center_size(pos2(card.center().x, card.top() + 44.0), logo_size);
    if let Some(tex) = mm_logo_white_texture(ui.ctx()) {
        let uv = Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0));
        ui.painter().image(tex.id(), logo_rect, uv, Color32::WHITE);
    } else {
        ui.painter().text(logo_rect.center(), Align2::CENTER_CENTER, "MM", t.semibold(22.0), Color32::WHITE);
    }

    // Title & Subtitle
    ui.painter().text(
        pos2(card.center().x, logo_rect.bottom() + 14.0),
        Align2::CENTER_TOP,
        "¡Muchas gracias por instalarme!",
        t.semibold(19.0),
        Color32::from_rgb(246, 244, 238),
    );
    ui.painter().text(
        pos2(card.center().x, logo_rect.bottom() + 39.0),
        Align2::CENTER_TOP,
        "M M   S T U D I O   ·   M A N U E L   C R U Z   M .",
        t.font(9.5),
        Color32::from_rgb(229, 169, 75),
    );

    // Body paragraph
    let body_text = "Este programa está hecho basándose en el repositorio de código abierto\n\
LightCraft (storytold/lightcraft), pero modificado con las herramientas\n\
que yo más priorizo: tiene mejoras en el rendimiento y una galería muy\n\
amplia de LUTs creativos (en el área de Efectos) para disfrutar\n\
y explotar al máximo la creatividad.";
    ui.painter().text(
        pos2(card.center().x, logo_rect.bottom() + 62.0),
        Align2::CENTER_TOP,
        body_text,
        t.font(12.5),
        Color32::from_rgb(212, 208, 200),
    );

    // Optional Donation Box
    let don_box = Rect::from_min_max(
        pos2(card.left() + 26.0, card.bottom() - 166.0),
        pos2(card.right() - 26.0, card.bottom() - 68.0),
    );
    ui.painter().rect(
        don_box,
        CornerRadius::same(12),
        Color32::from_rgba_unmultiplied(30, 32, 30, 220),
        Stroke::new(1.0, Color32::from_rgba_unmultiplied(229, 169, 75, 75)),
        StrokeKind::Inside,
    );

    ui.painter().text(
        pos2(don_box.left() + 16.0, don_box.top() + 13.0),
        Align2::LEFT_TOP,
        "Si gustas donar algo (No es obligatorio), te dejo mi cuenta BBVA:",
        t.font(11.5),
        Color32::from_rgb(218, 212, 200),
    );

    let acct_raw = ["012261", "015344", "234011"].concat();
    let acct_display = ["0122", "6101", "5344", "2340", "11"].join(" ");

    ui.painter().text(
        pos2(don_box.left() + 16.0, don_box.top() + 36.0),
        Align2::LEFT_TOP,
        format!("BBVA:  {acct_display}"),
        t.semibold(14.5),
        Color32::from_rgb(248, 244, 234),
    );
    ui.painter().text(
        pos2(don_box.left() + 16.0, don_box.top() + 60.0),
        Align2::LEFT_TOP,
        "Beneficiario:  Manuel Cruz M.",
        t.font(11.5),
        Color32::from_rgb(186, 182, 172),
    );

    // Copy account button inside donation box
    let now = ui.input(|i| i.time);
    let recently_copied = app
        .ui
        .welcome_account_copied_at
        .is_some_and(|ts| (now - ts).abs() < 2.6);
    if recently_copied {
        ui.ctx().request_repaint_after(std::time::Duration::from_millis(200));
    }

    let copy_btn = Rect::from_min_size(pos2(don_box.right() - 142.0, don_box.top() + 34.0), vec2(126.0, 36.0));
    let copy_resp = ui
        .interact(copy_btn, egui::Id::new("welcome-copy-account"), Sense::click())
        .on_hover_text("Copiar número de cuenta BBVA al portapapeles");
    register(ui.ctx(), "button:welcomeCopyAccount", copy_btn);

    let (copy_fill, copy_border, copy_label, copy_ink) = if recently_copied {
        (
            Color32::from_rgba_unmultiplied(46, 118, 78, 230),
            Color32::from_rgb(110, 215, 150),
            "✓ ¡Copiado!",
            Color32::from_rgb(245, 255, 248),
        )
    } else if copy_resp.hovered() {
        (
            Color32::from_rgb(240, 184, 90),
            Color32::from_white_alpha(200),
            "Copiar cuenta",
            Color32::from_rgb(20, 22, 20),
        )
    } else {
        (
            Color32::from_rgb(229, 169, 75),
            Color32::from_white_alpha(150),
            "Copiar cuenta",
            Color32::from_rgb(20, 22, 20),
        )
    };
    ui.painter().rect(copy_btn, CornerRadius::same(9), copy_fill, Stroke::new(1.0, copy_border), StrokeKind::Inside);
    ui.painter().text(copy_btn.center(), Align2::CENTER_CENTER, copy_label, t.semibold(12.0), copy_ink);
    if copy_resp.clicked() {
        ui.ctx().copy_text(acct_raw);
        app.ui.welcome_account_copied_at = Some(now);
    }

    // Bottom "Comenzar" button
    let start_btn = Rect::from_center_size(pos2(card.center().x, card.bottom() - 34.0), vec2(190.0, 38.0));
    let start_resp = ui
        .interact(start_btn, egui::Id::new("welcome-start-btn"), Sense::click())
        .on_hover_text("Cerrar bienvenida y comenzar a editar");
    register(ui.ctx(), "button:welcomeStart", start_btn);
    paint_frosted_pill(ui.painter(), start_btn, start_resp.hovered(), 19.0);
    ui.painter().text(
        start_btn.center(),
        Align2::CENTER_CENTER,
        "Comenzar a editar",
        t.semibold(12.5),
        Color32::from_rgb(244, 242, 236),
    );

    if close_resp.clicked()
        || start_resp.clicked()
        || ui.input(|i| i.key_pressed(egui::Key::Escape))
        || (bg_resp.clicked() && !card.contains(bg_resp.interact_pointer_pos().unwrap_or(card.center())))
    {
        app.ui.show_welcome_modal = false;
    }
}

fn paint_frosted_pill(painter: &egui::Painter, rect: Rect, hovered: bool, radius: f32) {
    let fill = if hovered {
        Color32::from_rgba_unmultiplied(42, 44, 40, 210)
    } else {
        Color32::from_rgba_unmultiplied(24, 26, 24, 188)
    };
    let border = if hovered {
        Color32::from_white_alpha(90)
    } else {
        Color32::from_white_alpha(42)
    };
    painter.rect(rect, CornerRadius::same(radius.round() as u8), fill, Stroke::new(1.0, border), StrokeKind::Inside);
    if rect.width() > 24.0 {
        painter.line_segment(
            [pos2(rect.left() + radius, rect.top() + 1.0), pos2(rect.right() - radius, rect.top() + 1.0)],
            Stroke::new(1.0, Color32::from_white_alpha(38)),
        );
    }
}

fn paint_frosted_circle(painter: &egui::Painter, rect: Rect, hovered: bool) {
    let c = rect.center();
    let r = rect.width() * 0.5;
    let fill = if hovered {
        Color32::from_rgba_unmultiplied(48, 50, 46, 215)
    } else {
        Color32::from_rgba_unmultiplied(24, 26, 24, 185)
    };
    let border = if hovered {
        Color32::from_white_alpha(95)
    } else {
        Color32::from_white_alpha(45)
    };
    painter.circle(c, r, fill, Stroke::new(1.0, border));
}
