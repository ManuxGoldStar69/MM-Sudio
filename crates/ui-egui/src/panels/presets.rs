//! The Presets column (opens to the left of the Edit panel): grouped presets with hover preview
//! (the loupe shows the photo with the preset while the pointer rests on it; nothing is
//! committed), optional live thumbnails, amount slider, create/favourite, import/export of preset
//! files.

use std::collections::BTreeMap;
use std::sync::Arc;

use egui::{Align2, Color32, Rect, Sense, pos2, vec2};
use lightcraft_develop::{ControlSpec, Preset, Section, Track};
use lightcraft_engine::presets::PresetList;
use serde_json::json;

use crate::button::Button;
use crate::icons::{Icon, paint};
use crate::theme::Tokens;
use crate::widgets::{divider, icon_button, register, slider};
use crate::{HoverPreview, LightcraftApp};

/// Long edge of preset thumbnails (px).
const THUMB_EDGE: usize = 128;

/// The preset amount slider may replace only the undo step it created.
#[derive(Clone)]
struct AppliedPreset {
    id: String,
    amount: f64,
    selected: Vec<lightcraft_catalog::PhotoId>,
    active: Option<lightcraft_catalog::PhotoId>,
    commits: u64,
}

impl AppliedPreset {
    fn new(session: &lightcraft_engine::Session, id: String, amount: f64) -> Self {
        Self { id, amount, selected: session.selection.ids.clone(), active: session.active(), commits: session.commits() }
    }

    fn is_current(&self, session: &lightcraft_engine::Session) -> bool {
        self.selected == session.selection.ids && self.active == session.active() && !self.is_stale(session)
    }

    /// A later edit, undo or redo (on any photo) owns the undo step: the slider may never replace it.
    fn is_stale(&self, session: &lightcraft_engine::Session) -> bool {
        !(self.commits == session.commits()
            && session.interaction.is_none()
            && session.redo.is_empty()
            && session
                .undo
                .last()
                .is_some_and(|entry| session.presets.iter().any(|preset| preset.id == self.id && entry.label == format!("Preset: {}", preset.name))))
    }
}

/// The preset each photo was given, with its settings right after: the panel marks it while the
/// photo still has exactly those settings, so it is still marked on coming back to the photo
/// (issue #2392) and drops once the photo is edited further.
#[derive(Clone, Default)]
struct AppliedByPhoto(std::collections::HashMap<lightcraft_catalog::PhotoId, (String, Arc<lightcraft_develop::DevelopSettings>)>);

impl AppliedByPhoto {
    const ID: &'static str = "preset-applied-by-photo";

    /// Remember `preset` on the photos `preset.apply` just changed (the selection, else the active
    /// photo; an active photo outside the selection wasn't changed and isn't marked).
    fn remember(ctx: &egui::Context, session: &lightcraft_engine::Session, preset: &str) {
        let id = egui::Id::new(Self::ID);
        let mut map: Self = ctx.data(|d| d.get_temp(id)).unwrap_or_default();
        for photo in session.selected() {
            if let Some(d) = session.develop_of(photo) {
                map.0.insert(photo, (preset.to_string(), d));
            }
        }
        ctx.data_mut(|d| d.insert_temp(id, map));
    }

    /// The preset to mark for `photo`: the one it was given, while its settings are unchanged since.
    fn of(ctx: &egui::Context, session: &lightcraft_engine::Session, photo: Option<lightcraft_catalog::PhotoId>) -> Option<String> {
        let photo = photo?;
        let map: Self = ctx.data(|d| d.get_temp(egui::Id::new(Self::ID)))?;
        let (preset, then) = map.0.get(&photo)?;
        let now = session.develop_of(photo)?;
        (Arc::ptr_eq(then, &now) || **then == *now).then(|| preset.clone())
    }
}

pub fn show(app: &mut LightcraftApp, ui: &mut egui::Ui) {
    let t = Tokens::get(ui.ctx());
    egui::Frame::NONE.fill(t.chrome).stroke(egui::Stroke::new(1.0, t.divider)).show(ui, |ui| {
        ui.spacing_mut().item_spacing.y = 0.0;
        let presets_folded = app.ui.section_folded("drawer:presets");
        let luts_folded = app.ui.section_folded("drawer:davinciLuts");
        let (hr, hr_resp) = ui.allocate_exact_size(vec2(ui.available_width(), 46.0), Sense::click());
        crate::theme::paint_glass_card(ui.painter(), hr.shrink2(vec2(8.0, 4.0)), &t, 8.0);
        // Warm bronze editorial accent bar + collapse chevron on the left of the Presets header
        ui.painter().rect_filled(Rect::from_min_size(pos2(hr.left() + 14.0, hr.center().y - 8.0), vec2(3.0, 18.0)), 1.5, t.star);
        paint(
            ui.painter(),
            Rect::from_center_size(pos2(hr.left() + 26.0, hr.center().y), vec2(14.0, 14.0)),
            if presets_folded { Icon::ChevronRight } else { Icon::ChevronDown },
            t.text_label,
        );
        ui.painter().text(
            pos2(hr.left() + 38.0, hr.center().y - 4.0),
            Align2::LEFT_CENTER,
            crate::i18n::tr("Presets"),
            t.semibold(14.5),
            t.text,
        );
        ui.painter().text(
            pos2(hr.left() + 38.0, hr.center().y + 11.0),
            Align2::LEFT_CENTER,
            "LIGHTROOM · .XMP / .LRTEMPLATE",
            t.font(9.5),
            t.text_dim,
        );
        let mut hdr = ui.new_child(
            egui::UiBuilder::new()
                .max_rect(Rect::from_min_max(pos2(hr.right() - 80.0, hr.top()), hr.right_bottom()))
                .layout(egui::Layout::right_to_left(egui::Align::Center)),
        );
        if icon_button(
            &mut hdr,
            "presetCreate",
            Icon::Plus,
            vec2(28.0, 28.0),
            false,
            app.session.active().is_some(),
            crate::i18n::tr("Create Preset…"),
        )
        .clicked()
        {
            app.ui.dialog = Some(crate::state::Dialog::create_preset());
        }
        let more = icon_button(&mut hdr, "presetMore", Icon::More, vec2(28.0, 28.0), false, true, crate::i18n::tr("More preset options"));
        if hr_resp.clicked()
            && ui.ctx().input(|i| i.pointer.interact_pos()).is_some_and(|p| p.x < hr.right() - 82.0)
        {
            app.ui.toggle_folded("drawer:presets");
        }
        egui::Popup::menu(&more).show(|ui| {
            let r = ui.checkbox(&mut app.ui.preset_thumbs, crate::i18n::tr("Show Thumbnails"));
            register(ui.ctx(), "presetMenu:thumbnails", r.rect);
            ui.separator();
            if ui.button(crate::i18n::tr("Import Presets…")).clicked() {
                let _ = app.run("file.importPresets", json!({"kind": "lr"}));
            }
            if ui.button("Importar LUTs DaVinci (.cube)…").clicked() {
                let _ = app.run("file.importPresets", json!({"kind": "lut"}));
            }
            let any_user = app.session.presets.iter().any(|p| !p.builtin);
            if ui.add_enabled(any_user, egui::Button::new(crate::i18n::tr("Export User Presets…"))).clicked() {
                let _ = app.run("file.exportPresets", json!({}));
            }
        });
        let active = app.session.active();
        let current = active.and_then(|id| app.session.develop_of(id)).map(|d| (*d).clone());
        if !presets_folded {
            // Dedicated quick-import bar for Lightroom Presets (.xmp, .lrtemplate, .dng, .zip)
            egui::Frame::NONE.fill(t.inset).inner_margin(egui::Margin { left: 16, right: 16, top: 6, bottom: 6 }).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 6.0;
                    if Button::new("+ Importar LR (.xmp)")
                        .primary()
                        .small()
                        .widget("button:importLrPresets")
                        .show(ui)
                        .on_hover_text("Importar presets de Adobe Lightroom (.xmp, .lrtemplate, .dng, .zip)")
                        .clicked()
                    {
                        let _ = app.run("file.importPresets", json!({"kind": "lr"}));
                    }
                    if Button::new("Guardar")
                        .small()
                        .enabled(app.session.active().is_some())
                        .widget("button:saveCurrentPreset")
                        .show(ui)
                        .on_hover_text("Guardar los ajustes actuales de la foto como un nuevo preset")
                        .clicked()
                    {
                        app.ui.dialog = Some(crate::state::Dialog::create_preset());
                    }
                });
            });
            divider(ui);
            // amount slider (applies to the last applied preset)
            let amt_id = egui::Id::new("preset-amount");
            let last: Option<AppliedPreset> = ui.data(|d| d.get_temp(amt_id));
            if let Some(last) = last.clone().filter(|last| last.is_current(&app.session)) {
                let spec = ControlSpec {
                    id: "presetAmount",
                    label: "Amount",
                    section: Section::Profile,
                    min: 0.0,
                    max: 200.0,
                    default: 100.0,
                    step: 1.0,
                    decimals: 0,
                    track: Track::Plain,
                };
                let out = slider(ui, &spec, last.amount, true, None);
                if let Some(v) = out.value
                    && app.run("edit.undo", json!({})).is_ok()
                {
                    // Only replace the most recent preset; never undo intervening edits.
                    if app.run("preset.apply", json!({"id": &last.id, "amount": v})).is_ok() {
                        AppliedByPhoto::remember(ui.ctx(), &app.session, &last.id);
                        ui.data_mut(|d| d.insert_temp(amt_id, AppliedPreset::new(&app.session, last.id, v)));
                    } else {
                        let _ = app.run("edit.redo", json!({}));
                        ui.data_mut(|d| d.remove::<AppliedPreset>(amt_id));
                    }
                }
                divider(ui);
            } else if last.is_some_and(|last| last.is_stale(&app.session)) {
                // kept across a switch to another photo and back (the slider returns with it); gone once
                // a later edit, undo or redo owns the step
                ui.data_mut(|d| d.remove::<AppliedPreset>(amt_id));
            }
            let marked = AppliedByPhoto::of(ui.ctx(), &app.session, active);
            let thumbs = app.ui.preset_thumbs && active.is_some();
            let row_h = if thumbs { 54.0 } else { 26.0 };
            let groups = grouped(ui.ctx(), &app.session.presets);
            // Reserve space for the separate DaVinci Resolve LUTs (.cube) area
            let luts_reserve = if luts_folded { 52.0 } else { 220.0 };
            let presets_max_h = (ui.available_height() - luts_reserve).max(160.0);
            let scroll = egui::ScrollArea::vertical()
                .id_salt("presets-scroll")
                .max_height(presets_max_h)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing.y = 0.0;
                    let visible = ui.clip_rect().expand(row_h);
                    for grp in groups.iter() {
                        let (g, builtin) = (grp.name.clone(), grp.builtin);
                        let fold_key = format!("presetGroup:{g}");
                        let open = !app.ui.section_folded(&fold_key);
                        let (r, resp) = ui.allocate_exact_size(vec2(ui.available_width(), 32.0), Sense::click());
                        crate::widgets::announce(&resp, g.to_string());
                        register(ui.ctx(), format!("presetGroup:{g}"), r);
                        paint(
                            ui.painter(),
                            Rect::from_center_size(pos2(r.left() + 22.0, r.center().y), vec2(16.0, 16.0)),
                            if open { Icon::ChevronDown } else { Icon::ChevronRight },
                            t.text_label,
                        );
                        ui.painter().text(
                            pos2(r.left() + 36.0, r.center().y),
                            Align2::LEFT_CENTER,
                            crate::i18n::builtin_label(&g, builtin),
                            t.semibold(13.0),
                            t.text_label,
                        );
                        if resp.clicked() {
                            app.ui.toggle_folded(&fold_key);
                        }
                        crate::widgets::context_menu(&resp, |ui| {
                            if ui.button(crate::i18n::tr("Export Group…")).clicked() {
                                let _ = app.run("file.exportPresets", json!({"group": g}));
                            }
                        });
                        if !open {
                            continue;
                        }
                        let (first, end) = rows_in_view(ui.cursor().top(), row_h, grp.items.len(), visible);
                        if first > 0 {
                            ui.allocate_space(vec2(ui.available_width(), first as f32 * row_h));
                        }
                        for &i in grp.items.get(first..end).unwrap_or_default() {
                            let Some(pr) = app.session.presets.get(i).cloned() else { continue };
                            let (pid, name, fav) = (pr.id.clone(), crate::i18n::builtin_label(&pr.name, pr.builtin).to_string(), pr.favorite);
                            let (r, resp) = ui.allocate_exact_size(vec2(ui.available_width(), row_h), Sense::click());
                            crate::widgets::announce(&resp, name.clone());
                            register(ui.ctx(), format!("preset:{pid}"), r);
                            let is_last = marked.as_deref() == Some(pid.as_str());
                            if is_last {
                                register(ui.ctx(), "presetApplied", r);
                                ui.painter().rect_filled(r, 0.0, t.tool_active);
                            } else if resp.hovered() {
                                ui.painter().rect_filled(r, 0.0, t.hover.gamma_multiply(0.7));
                            }
                            let look = if thumbs || resp.hovered() { current.as_ref().map(|d| pr.apply(d, 1.0)) } else { None };
                            let mut text_x = r.left() + 40.0;
                            if thumbs && let (Some(id), Some(s)) = (active, look.as_ref()) {
                                let tr = Rect::from_min_size(pos2(r.left() + 36.0, r.top() + 5.0), vec2(66.0, row_h - 10.0));
                                ui.painter().rect_filled(tr, 2.0, t.canvas);
                                if let Some(job) = app.session.variant_job(id, s, THUMB_EDGE)
                                    && let Some(tex) = app.renderer.variant(job)
                                {
                                    ui.painter().image(tex.tex.id(), tr, cover_uv(tr, tex.size), Color32::WHITE);
                                }
                                text_x = tr.right() + 10.0;
                            }
                            ui.painter().text(pos2(text_x, r.center().y), Align2::LEFT_CENTER, &name, t.font(13.0), t.text_label);
                            if fav {
                                paint(ui.painter(), Rect::from_center_size(pos2(r.right() - 20.0, r.center().y), vec2(16.0, 16.0)), Icon::StarFilled, t.star);
                            }
                            // resting on a preset previews it in the loupe (no history entry)
                            if resp.hovered()
                                && let Some(s) = look
                            {
                                let label = crate::i18n::tr_format!("Preset: {name}", name = name);
                                app.hover_preview = Some(HoverPreview { label: label.clone(), settings: s });
                                if let Some(id) = active {
                                    ui.data_mut(|d| d.insert_temp(egui::Id::new("last-preset-hover"), (id, pr.clone(), label)));
                                }
                            }
                            if resp.clicked() && app.run("preset.apply", json!({"id": pid, "amount": 100})).is_ok() {
                                AppliedByPhoto::remember(ui.ctx(), &app.session, &pid);
                                ui.data_mut(|d| d.insert_temp(amt_id, AppliedPreset::new(&app.session, pid.clone(), 100.0)));
                                ui.data_mut(|d| d.remove::<(lightcraft_catalog::PhotoId, Preset, String)>(egui::Id::new("last-preset-hover")));
                                app.toast(ui.ctx(), crate::i18n::tr_format!("Preset: {name}", name = name));
                            }
                            let (builtin, group) = (pr.builtin, pr.group.clone());
                            crate::widgets::context_menu(&resp, |ui| {
                                if ui.button(crate::i18n::tr(if fav { "Remove from Favorites" } else { "Add to Favorites" })).clicked() {
                                    let _ = app.run("preset.favorite", json!({"id": pid}));
                                }
                                if builtin {
                                    return;
                                }
                                ui.separator();
                                if ui
                                    .add_enabled(app.session.active().is_some(), egui::Button::new(crate::i18n::tr("Update with Current Settings")))
                                    .clicked()
                                {
                                    match app.run("preset.update", json!({"id": pid})) {
                                        Ok(_) => app.toast(ui.ctx(), crate::i18n::tr_format!("Updated “{name}”", name = name)),
                                        Err(e) => app.toast(ui.ctx(), e),
                                    }
                                }
                                if ui.button(crate::i18n::tr("Rename…")).clicked() {
                                    crate::panels::dialogs::prompt(app, "Rename Preset", "Preset name", &name, "preset.rename", json!({"id": pid}), "name");
                                }
                                crate::widgets::menu_button(ui, crate::i18n::tr("Move to Group"), |ui| {
                                    for g in groups.iter().filter(|x| x.user && x.name != group).map(|x| &x.name) {
                                        if ui.button(g).clicked() {
                                            let _ = app.run("preset.move", json!({"id": pid, "group": g}));
                                        }
                                    }
                                    if ui.button(crate::i18n::tr("New Group…")).clicked() {
                                        crate::panels::dialogs::prompt(
                                            app,
                                            "Move Preset to New Group",
                                            "Group name",
                                            "",
                                            "preset.move",
                                            json!({"id": pid}),
                                            "group",
                                        );
                                    }
                                });
                                ui.separator();
                                if ui.button(crate::i18n::tr("Delete Preset")).clicked() {
                                    let _ = app.run("preset.delete", json!({"id": pid}));
                                }
                            });
                        }
                        if end < grp.items.len() {
                            ui.allocate_space(vec2(ui.available_width(), (grp.items.len() - end) as f32 * row_h));
                        }
                    }
                    ui.add_space(16.0);
                });
            let last_hover_id = egui::Id::new("last-preset-hover");
            let pointer_in_scroll = ui.ctx().input(|i| i.pointer.interact_pos().or(i.pointer.hover_pos())).is_some_and(|pos| scroll.inner_rect.contains(pos));
            if pointer_in_scroll {
                if app.hover_preview.is_none()
                    && let (Some(id), Some(curr)) = (active, current.as_ref())
                    && let Some((saved_id, saved_preset, saved_label)) =
                        ui.data(|d| d.get_temp::<(lightcraft_catalog::PhotoId, Preset, String)>(last_hover_id))
                    && saved_id == id
                {
                    app.hover_preview = Some(HoverPreview { label: saved_label, settings: saved_preset.apply(curr, 1.0) });
                }
            } else {
                ui.data_mut(|d| d.remove::<(lightcraft_catalog::PhotoId, Preset, String)>(last_hover_id));
            }
        }

        // ------------------------------------------------------------------
        // Separate dedicated area: DaVinci Resolve 3D LUTs (.cube)
        // ------------------------------------------------------------------
        divider(ui);
        show_davinci_luts_area(app, ui, &t, active, current.as_ref());
    });
}

/// Dedicated bottom section inside the Presets column for DaVinci Resolve `.cube` 3D LUTs.
fn show_davinci_luts_area(
    app: &mut LightcraftApp,
    ui: &mut egui::Ui,
    t: &Tokens,
    active: Option<lightcraft_catalog::PhotoId>,
    current: Option<&lightcraft_develop::DevelopSettings>,
) {
    let folded = app.ui.section_folded("drawer:davinciLuts");
    let (lh, lh_resp) = ui.allocate_exact_size(vec2(ui.available_width(), 42.0), Sense::click());
    crate::theme::paint_glass_card(ui.painter(), lh.shrink2(vec2(8.0, 3.0)), t, 8.0);
    ui.painter().rect_filled(Rect::from_min_size(pos2(lh.left() + 14.0, lh.center().y - 8.0), vec2(3.0, 18.0)), 1.5, t.star);
    paint(
        ui.painter(),
        Rect::from_center_size(pos2(lh.left() + 26.0, lh.center().y), vec2(14.0, 14.0)),
        if folded { Icon::ChevronRight } else { Icon::ChevronDown },
        t.text_label,
    );
    ui.painter().text(
        pos2(lh.left() + 38.0, lh.center().y - 4.0),
        Align2::LEFT_CENTER,
        "LUTs · DaVinci Resolve",
        t.semibold(13.0),
        t.text,
    );
    ui.painter().text(
        pos2(lh.left() + 38.0, lh.center().y + 10.0),
        Align2::LEFT_CENTER,
        "COLOR GRADING 3D (.CUBE)",
        t.font(9.5),
        t.text_dim,
    );
    let mut lhdr = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(Rect::from_min_max(pos2(lh.right() - 115.0, lh.top()), pos2(lh.right() - 10.0, lh.bottom())))
            .layout(egui::Layout::right_to_left(egui::Align::Center)),
    );
    let import_clicked = Button::new("+ .CUBE")
        .primary()
        .small()
        .widget("button:importDavinciLut")
        .show(&mut lhdr)
        .on_hover_text("Importar archivos 3D LUT (.cube o .zip) de DaVinci Resolve")
        .clicked();
    if import_clicked {
        let _ = app.run("file.importPresets", json!({"kind": "lut"}));
    } else if lh_resp.clicked()
        && ui.ctx().input(|i| i.pointer.interact_pos()).is_some_and(|p| p.x < lh.right() - 95.0)
    {
        app.ui.toggle_folded("drawer:davinciLuts");
    }
    if folded {
        return;
    }
    divider(ui);

    // If a .cube LUT is currently active on the photo, show its intensity slider (0–200%) and a clear button
    let active_lut_id = current.and_then(|d| d.profile.id.starts_with("lut:").then(|| d.profile.id.clone()));
    let active_lut_name = active_lut_id.as_deref().map(|lut_id| {
        app.session
            .lut_profiles
            .iter()
            .find(|l| l.id == lut_id)
            .map(|l| l.name.clone())
            .unwrap_or_else(|| lut_id.trim_start_matches("lut:").to_string())
    });
    if let (Some(_lut_id), Some(lut_name), Some(d)) = (active_lut_id.as_deref(), active_lut_name.as_deref(), current) {
        egui::Frame::NONE.inner_margin(egui::Margin { left: 16, right: 16, top: 6, bottom: 2 }).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(format!("LUT activo: {lut_name}")).font(t.semibold(11.5)).color(t.star));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if Button::new("✕ Desactivar")
                        .ghost()
                        .small()
                        .widget("button:clearActiveLut")
                        .show(ui)
                        .on_hover_text("Desactivar LUT .cube y volver al perfil Color estándar")
                        .clicked()
                    {
                        let _ = app.run("develop.profile", json!({"id": "lc.color", "amount": 100.0}));
                        app.toast(ui.ctx(), "LUT DaVinci desactivado");
                    }
                });
            });
        });
        super::edit::control(app, ui, d, "profile.amount", true);
        divider(ui);
    }

    // Deduplicate LUTs by normalized name (preferring the active or favorite entry if duplicated)
    let mut luts = app.session.lut_profiles.clone();
    luts.clear();
    for lut in &app.session.lut_profiles {
        let key = lut.name.trim().to_ascii_lowercase();
        if let Some(existing) = luts.iter_mut().find(|x| x.name.trim().to_ascii_lowercase() == key) {
            if active_lut_id.as_deref() == Some(lut.id.as_str()) {
                *existing = lut.clone();
            }
        } else {
            luts.push(lut.clone());
        }
    }
    egui::ScrollArea::vertical()
        .id_salt("davinci-luts-scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 0.0;
            if luts.is_empty() {
                egui::Frame::NONE.inner_margin(egui::Margin { left: 16, right: 16, top: 14, bottom: 14 }).show(ui, |ui| {
                    let (card, resp) = ui.allocate_exact_size(vec2(ui.available_width(), 64.0), Sense::click());
                    ui.painter().rect(
                        card,
                        6.0,
                        if resp.hovered() { t.hover } else { t.inset },
                        egui::Stroke::new(1.0, t.field_border),
                        egui::StrokeKind::Inside,
                    );
                    ui.painter().text(
                        pos2(card.center().x, card.center().y - 8.0),
                        Align2::CENTER_CENTER,
                        "Sin LUTs .cube importados aún",
                        t.semibold(12.0),
                        t.text_label,
                    );
                    ui.painter().text(
                        pos2(card.center().x, card.center().y + 10.0),
                        Align2::CENTER_CENTER,
                        "Haz clic aquí o en '+ .CUBE' para cargar LUTs de DaVinci",
                        t.font(10.5),
                        t.text_dim,
                    );
                    if resp.clicked() {
                        let _ = app.run("file.importPresets", json!({"kind": "lut"}));
                    }
                });
                return;
            }
            for lut in luts {
                let is_active = active_lut_id.as_deref() == Some(lut.id.as_str())
                    || active_lut_name.as_deref().is_some_and(|n| n.eq_ignore_ascii_case(lut.name.trim()));
                let fav = app.session.profile_favorites.contains(&lut.id);
                let (r, resp) = ui.allocate_exact_size(vec2(ui.available_width(), 28.0), Sense::click());
                let resp = resp.on_hover_text(if is_active {
                    "Clic para desactivar este LUT DaVinci"
                } else {
                    "Clic para aplicar este LUT DaVinci"
                });
                register(ui.ctx(), format!("lut:{}", lut.id), r);
                if is_active {
                    ui.painter().rect_filled(r, 0.0, t.tool_active);
                } else if resp.hovered() {
                    ui.painter().rect_filled(r, 0.0, t.hover.gamma_multiply(0.7));
                }
                // CUBE badge pill
                let badge = Rect::from_min_size(pos2(r.left() + 16.0, r.center().y - 8.0), vec2(36.0, 16.0));
                ui.painter().rect_filled(badge, 3.0, if is_active { t.accent } else { t.button });
                ui.painter().text(
                    badge.center(),
                    Align2::CENTER_CENTER,
                    if is_active { "✓ ON" } else { "CUBE" },
                    t.semibold(9.0),
                    if is_active { Color32::WHITE } else { t.star },
                );
                ui.painter().text(
                    pos2(badge.right() + 8.0, r.center().y),
                    Align2::LEFT_CENTER,
                    &lut.name,
                    if is_active { t.semibold(12.5) } else { t.font(12.5) },
                    if is_active { t.text } else { t.text_label },
                );
                if fav {
                    paint(
                        ui.painter(),
                        Rect::from_center_size(pos2(r.right() - 20.0, r.center().y), vec2(15.0, 15.0)),
                        Icon::StarFilled,
                        t.star,
                    );
                }
                if resp.hovered()
                    && !is_active
                    && let (Some(_), Some(curr)) = (active, current)
                {
                    let mut preview = curr.clone();
                    preview.profile.id = lut.id.clone();
                    if preview.profile.amount == 0.0 {
                        preview.profile.amount = 100.0;
                    }
                    app.hover_preview = Some(HoverPreview { label: format!("LUT DaVinci: {}", lut.name), settings: preview });
                }
                if resp.clicked() {
                    if is_active {
                        if app.run("develop.profile", json!({"id": "lc.color", "amount": 100.0})).is_ok() {
                            app.toast(ui.ctx(), format!("LUT DaVinci desactivado: {}", lut.name));
                        }
                    } else {
                        let amount = current.map(|d| if d.profile.amount > 0.0 { d.profile.amount } else { 100.0 }).unwrap_or(100.0);
                        if app.run("develop.profile", json!({"id": lut.id, "amount": amount})).is_ok() {
                            app.toast(ui.ctx(), format!("LUT DaVinci aplicado: {}", lut.name));
                        }
                    }
                }
                crate::widgets::context_menu(&resp, |ui| {
                    if is_active && ui.button("Desactivar LUT").clicked() {
                        let _ = app.run("develop.profile", json!({"id": "lc.color", "amount": 100.0}));
                    }
                    if ui.button(crate::i18n::tr(if fav { "Remove from Favorites" } else { "Add to Favorites" })).clicked() {
                        let _ = app.run("profile.favorite", json!({"id": lut.id}));
                    }
                    ui.separator();
                    if ui.button("Eliminar LUT (.cube)").clicked() {
                        let _ = app.run("profile.deleteImported", json!({"id": lut.id}));
                    }
                });
            }
            ui.add_space(12.0);
        });
}

/// A preset group as the panel lists it: indices into the session's presets.
struct Group {
    name: String,
    /// Holds a stock preset: a stock group (shown in the UI language), even when user presets
    /// share its name; a group of user presets keeps its name.
    builtin: bool,
    /// Holds a user preset (a Move to Group target).
    user: bool,
    items: Vec<usize>,
}

/// The presets by group (groups sorted by name, presets in list order), rebuilt only when the
/// list changed (its generation).
fn grouped(ctx: &egui::Context, presets: &PresetList) -> Arc<Vec<Group>> {
    let id = egui::Id::new("presets-grouped");
    if let Some((generation, groups)) = ctx.data(|d| d.get_temp::<(u64, Arc<Vec<Group>>)>(id))
        && generation == presets.generation()
    {
        return groups;
    }
    let mut by_name: BTreeMap<&str, Group> = BTreeMap::new();
    for (i, p) in presets.iter().enumerate() {
        let g = by_name.entry(&p.group).or_insert_with(|| Group { name: p.group.clone(), builtin: false, user: false, items: Vec::new() });
        g.builtin |= p.builtin;
        g.user |= !p.builtin;
        g.items.push(i);
    }
    let groups = Arc::new(by_name.into_values().collect::<Vec<_>>());
    ctx.data_mut(|d| d.insert_temp(id, (presets.generation(), groups.clone())));
    groups
}

/// The presets as the rows of a picker (a combo box's list), in list order with a header row
/// wherever the group changes. Only the rows in view are built (10 000+ presets made an open
/// picker take seconds a frame). Returns the id of the preset clicked.
pub fn picker_rows(ui: &mut egui::Ui, salt: &str, presets: &PresetList, selected: Option<&str>) -> Option<String> {
    let id = egui::Id::new("preset-picker-rows");
    let rows = match ui.ctx().data(|d| d.get_temp::<(u64, Arc<Vec<PickRow>>)>(id)) {
        Some((generation, rows)) if generation == presets.generation() => rows,
        _ => {
            let mut rows = Vec::with_capacity(presets.len() + 16);
            let mut group: Option<&str> = None;
            for (i, p) in presets.iter().enumerate() {
                if group != Some(p.group.as_str()) {
                    group = Some(&p.group);
                    rows.push(PickRow::Header(crate::i18n::builtin_label(&p.group, p.builtin).to_string()));
                }
                rows.push(PickRow::Preset(i));
            }
            let rows = Arc::new(rows);
            ui.ctx().data_mut(|d| d.insert_temp(id, (presets.generation(), rows.clone())));
            rows
        }
    };
    let row_h = ui.spacing().interact_size.y;
    let mut out = None;
    egui::ScrollArea::vertical().id_salt(("preset-picker", salt)).max_height(300.0).auto_shrink([false, true]).show_rows(
        ui,
        row_h,
        rows.len(),
        |ui, range| {
            for row in rows.get(range).unwrap_or_default() {
                let size = vec2(ui.available_width(), row_h);
                match row {
                    PickRow::Header(g) => {
                        ui.add_sized(size, egui::Label::new(egui::RichText::new(g).size(10.5).weak()));
                    }
                    PickRow::Preset(i) => {
                        let Some(p) = presets.get(*i) else { continue };
                        let label = crate::i18n::builtin_label(&p.name, p.builtin);
                        if ui.add_sized(size, egui::Button::selectable(selected == Some(p.id.as_str()), label)).clicked() {
                            out = Some(p.id.clone());
                        }
                    }
                }
            }
        },
    );
    out
}

#[derive(Clone)]
enum PickRow {
    /// A group's name (in the UI language for stock groups).
    Header(String),
    /// An index into the session's presets.
    Preset(usize),
}

/// The rows `first..end` of a block of `n` rows `row_h` high starting at `top` that meet
/// `visible`.
fn rows_in_view(top: f32, row_h: f32, n: usize, visible: Rect) -> (usize, usize) {
    if row_h <= 0.0 || !top.is_finite() {
        return (0, n);
    }
    // `as usize` saturates (negative and NaN give 0)
    let first = (((visible.top() - top) / row_h).floor() as usize).min(n);
    let end = (((visible.bottom() - top) / row_h).ceil() as usize).clamp(first, n);
    (first, end)
}

/// The part of a `size` image that covers `r` (centre crop), as texture coordinates.
pub fn cover_uv(r: Rect, size: [usize; 2]) -> Rect {
    let a = size[0] as f32 / size[1].max(1) as f32;
    let ra = r.width() / r.height().max(1.0);
    if a > ra {
        let f = ra / a;
        Rect::from_min_max(pos2((1.0 - f) / 2.0, 0.0), pos2((1.0 + f) / 2.0, 1.0))
    } else {
        let f = a / ra;
        Rect::from_min_max(pos2(0.0, (1.0 - f) / 2.0), pos2(1.0, (1.0 + f) / 2.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn amount_slider_does_not_undo_a_later_edit() {
        let mut session = lightcraft_engine::Session::with_demo();
        let pid = "lc.golden-hour";
        session.execute("preset.apply", &json!({"id": pid})).unwrap();
        let state = AppliedPreset::new(&session, pid.into(), 100.0);
        assert!(state.is_current(&session));

        session.execute("develop.set", &json!({"values": {"light.exposure": 0.75}})).unwrap();
        assert!(!state.is_current(&session), "a newer edit owns the undo step");
    }

    #[test]
    fn amount_slider_does_not_undo_on_another_photo() {
        let mut session = lightcraft_engine::Session::with_demo();
        let pid = "lc.golden-hour";
        session.execute("preset.apply", &json!({"id": pid})).unwrap();
        let state = AppliedPreset::new(&session, pid.into(), 100.0);
        let active = session.active();
        let other = session.visible().iter().copied().find(|id| Some(*id) != active).unwrap();
        session.execute("library.select", &json!({"ids": [other.0]})).unwrap();
        assert!(!state.is_current(&session), "the preset belongs to the earlier selection");
    }
}
