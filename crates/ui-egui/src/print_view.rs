//! The Print dialog (File ▸ Print…): settings on the left, the pages drawn to scale on the right.
//!
//! Every change goes through a `print.*` command ([`lightcraft_engine::cmd`]); this file only draws
//! the session's print ([`lightcraft_engine::Session::print`]) and turns pointer input into
//! commands: dragging a cell is `print.cell.move`, its corner handle `print.cell.resize`, with
//! the guides' snapping done by the command. Widgets register under `print:*`, `button:print*`
//! and `drag:print*` ids for the control channel and the headless tests.
use egui::{Align2, Color32, Pos2, Rect, Sense, Stroke, StrokeKind, Ui, Vec2, pos2, vec2};
use lightcraft_catalog::PhotoId;
use lightcraft_engine::export::ExportOptions;
use lightcraft_engine::print::book::{self, ALBUM_PRESETS, BindingMode, WarningKind};
use lightcraft_engine::print::compose::{self, place};
use lightcraft_engine::print::layout::{self, Snap, Style, Unit};
use lightcraft_engine::print::template::PrintSettings;
use lightcraft_engine::print_color::{ManagedSpace, PrintColorSettings, PrintIntent, PrintMedia, PrintProfile, PrintSharpen};
use serde_json::{Value, json};

use crate::LightcraftApp;
use crate::theme::Tokens;
use crate::widgets::{register, text_button};

/// Width of the settings column.
const SETTINGS_W: f32 = 290.0;
/// Handle for resizing the selected cell, screen points.
const HANDLE: f32 = 12.0;
/// Snap distance while dragging, screen points.
const SNAP_PX: f32 = 8.0;
/// Cell sizes offered as buttons, inches (width, height).
const CELL_SIZES: &[(f64, f64)] = &[(2.0, 2.0), (2.5, 3.5), (3.5, 5.0), (4.0, 5.0), (4.0, 6.0), (5.0, 7.0), (8.0, 10.0), (3.0, 3.0)];

/// A drag of a cell in progress: the cell's rectangle when it began and how far the pointer went.
#[derive(Clone, Debug)]
pub struct Drag {
    cell: u32,
    origin: layout::Rect,
    resize: bool,
    /// Where the pointer was pressed; the distance dragged is measured from it, so the first few
    /// points egui spends deciding between a click and a drag are not lost.
    press: Pos2,
    total: Vec2,
}

/// Active magnetic drag of a photo between frames on the canvas and the "Pila de Fotos".
#[derive(Clone, Debug)]
pub struct PhotoDrag {
    pub photo_id: u64,
    /// `Some(cell_id)` if dragged out of a frame on the spread; `None` if dragged from the Pila de Fotos.
    pub from_cell: Option<u32>,
    pub pos: Pos2,
}

/// What the Print dialog remembers between frames (not saved).
#[derive(Clone, Debug, Default)]
pub struct PrintUi {
    /// The page being shown, zero-based.
    pub page: usize,
    /// The selected cell.
    pub selected: Option<u32>,
    drag: Option<Drag>,
    /// Active pan-drag inside a cell: `(cell_id, start_pan_x, start_pan_y, press_pos)`.
    pan_drag: Option<(u32, f64, f64, Pos2)>,
    /// Cell currently in "Modo Encuadre Interno" (activated by double-clicking a frame or clicking the Encuadrar button).
    pub pan_mode_cell: Option<u32>,
    /// Active magnetic photo drag (from a frame or from the Pila de Fotos).
    pub photo_drag: Option<PhotoDrag>,
    /// Screen rect of the "Pila de Fotos" tray at the bottom of the Photobook view.
    pub pile_rect: Option<Rect>,
    /// Pool of photo IDs participating in this Photobook session (so photos removed from frames stay in the Pila de Fotos).
    pub book_pool: Vec<u64>,
    /// The name typed for Save Template.
    pub template_name: String,
    /// Size of the next cell added, inches.
    pub new_cell: Option<(f64, f64)>,
    /// Photos selected in the bottom filmstrip (1..=8) to drop/auto-layout on the spread.
    pub filmstrip_selection: Vec<u64>,
    /// Draft caption text for the selected cell.
    pub caption_draft: String,
    /// Which cell id `caption_draft` belongs to.
    pub caption_cell: Option<u32>,
}

fn run(app: &mut LightcraftApp, ctx: &egui::Context, id: &str, params: Value) -> Option<Value> {
    match app.run(id, params) {
        Ok(v) => Some(v),
        Err(e) => {
            app.toast(ctx, e);
            None
        }
    }
}

fn patch(app: &mut LightcraftApp, ctx: &egui::Context, params: Value) {
    run(app, ctx, "print.set", params);
}

fn heading(ui: &mut Ui, t: &Tokens, text: &str) {
    ui.add_space(8.0);
    ui.label(egui::RichText::new(crate::i18n::tr(text)).font(t.semibold(12.0)).color(t.text));
    ui.add_space(2.0);
}

fn row<R>(ui: &mut Ui, t: &Tokens, label: &str, add: impl FnOnce(&mut Ui) -> R) -> R {
    ui.horizontal(|ui| {
        ui.allocate_ui_with_layout(vec2(86.0, 22.0), egui::Layout::left_to_right(egui::Align::Center), |ui| {
            ui.label(egui::RichText::new(crate::i18n::tr(label)).color(t.text_label));
        });
        add(ui)
    })
    .inner
}

fn decimals(unit: Unit) -> usize {
    match unit {
        Unit::Inches | Unit::Centimeters => 2,
        Unit::Millimeters => 1,
        Unit::Points => 0,
    }
}

fn fmt_len(inches: f64, unit: Unit) -> String {
    format!("{:.*} {}", decimals(unit), unit.from_inches(inches), unit.label())
}

/// A length field in the ruler's unit; the new value in inches when it changed.
fn length(ui: &mut Ui, id: &str, inches: f64, unit: Unit, min_in: f64, max_in: f64) -> Option<f64> {
    let mut v = unit.from_inches(inches);
    let r = ui.add(
        egui::DragValue::new(&mut v)
            .range(unit.from_inches(min_in)..=unit.from_inches(max_in))
            .speed(unit.per_inch() * 0.01)
            .max_decimals(decimals(unit))
            .suffix(format!(" {}", unit.label())),
    );
    register(ui.ctx(), format!("drag:{id}"), r.rect);
    r.changed().then(|| unit.to_inches(v))
}

fn check(ui: &mut Ui, id: &str, label: &str, value: bool) -> Option<bool> {
    let mut v = value;
    let r = ui.checkbox(&mut v, crate::i18n::tr(label));
    register(ui.ctx(), format!("check:{id}"), r.rect);
    r.changed().then_some(v)
}

fn choice<V: PartialEq + Copy>(ui: &mut Ui, id: &str, options: &[(V, &str)], current: V) -> Option<V> {
    let mut out = None;
    ui.spacing_mut().item_spacing.x = 3.0;
    for (i, (v, l)) in options.iter().enumerate() {
        if text_button(ui, &format!("{id}-{i}"), l, current == *v).clicked() && current != *v {
            out = Some(*v);
        }
    }
    out
}

fn colour(ui: &mut Ui, id: &str, rgb: [u8; 3]) -> Option<[u8; 3]> {
    let mut c = rgb;
    let r = ui.color_edit_button_srgb(&mut c);
    register(ui.ctx(), format!("color:{id}"), r.rect);
    (c != rgb).then_some(c)
}

fn paper_label(name: &str) -> String {
    match name {
        "a3" | "a4" | "a5" => name.to_uppercase(),
        "letter" => "Letter".into(),
        "legal" => "Legal".into(),
        "tabloid" => "Tabloid".into(),
        "custom" => "Personalizado".into(),
        other => other.replace('x', " × ") + " in",
    }
}

/// Activates Photobook mode and immediately auto-paginates all selected photos across the resulting spreads.
fn activate_photobook_mode(app: &mut LightcraftApp, ctx: &egui::Context, pu: &mut PrintUi, st: &PrintSettings, ids: &[PhotoId]) {
    let per_spread = st.book.photos_per_spread.clamp(1, 8);
    pu.book_pool.clear();
    for &id in ids {
        if !pu.book_pool.contains(&id.0) {
            pu.book_pool.push(id.0);
        }
    }
    run(app, ctx, "print.set_format", json!({"format": "12x12", "spreadMode": true, "photosPerSpread": per_spread}));
    run(app, ctx, "print.auto_layout_spread", json!({"autoPaginateAll": true, "photosPerSpread": per_spread}));
    for c in &app.session.print.layout.cells {
        if let Some(pid) = c.photo
            && !pu.book_pool.contains(&pid)
        {
            pu.book_pool.push(pid);
        }
    }
    pu.page = 0;
    pu.selected = None;
}

/// The dialog's body: the settings column and the page preview side by side.
pub fn body(app: &mut LightcraftApp, ui: &mut Ui, t: &Tokens) {
    let ctx = ui.ctx().clone();
    let screen = ctx.content_rect();
    let width = (screen.width() - 80.0).clamp(640.0, 1180.0);
    let height = (screen.height() - 190.0).clamp(280.0, 780.0);
    let mut pu = std::mem::take(&mut app.ui.print);
    let st = app.session.print.clone();
    // (no `id`/`ids` given: the selection, which never fails to parse)
    let ids = app.session.targets(&json!({})).unwrap_or_default();
    // the page shown stays inside the print
    let pages = st.layout.page_count(ids.len());
    pu.page = pu.page.min(pages.saturating_sub(1));
    if pu.selected.is_some_and(|c| st.layout.cell(c).is_none_or(|c| c.page as usize != pu.page)) && pu.drag.is_none() {
        pu.selected = pu.selected.filter(|c| st.layout.cell(*c).is_some());
    }
    if pu.pan_mode_cell.is_some_and(|c| st.layout.cell(c).is_none_or(|c| c.page as usize != pu.page)) {
        pu.pan_mode_cell = None;
    }
    // Si está activo el Modo Encuadre Interno, Escape o Enter sale del modo encuadre sin cerrar la ventana de Imprimir
    if pu.pan_mode_cell.is_some()
        && !ctx.egui_wants_keyboard_input()
        && (ctx.input(|i| i.key_pressed(egui::Key::Escape)) || ctx.input(|i| i.key_pressed(egui::Key::Enter)))
    {
        crate::widgets::take_escape(&ctx);
        ctx.input_mut(|i| {
            i.consume_key(egui::Modifiers::NONE, egui::Key::Escape);
            i.consume_key(egui::Modifiers::NONE, egui::Key::Enter);
        });
        pu.pan_mode_cell = None;
    }
    ui.horizontal_top(|ui| {
        ui.allocate_ui_with_layout(vec2(SETTINGS_W, height), egui::Layout::top_down(egui::Align::Min), |ui| {
            ui.set_min_size(vec2(SETTINGS_W, height));
            egui::ScrollArea::vertical().id_salt("print-settings").max_height(height).auto_shrink([false, false]).show(ui, |ui| {
                ui.set_width(SETTINGS_W - 14.0);
                settings(app, ui, t, &mut pu, &st, ids.len());
            });
        });
        // (a `separator` would take the dialog's whole free height)
        let (line, _) = ui.allocate_exact_size(vec2(1.0, height), Sense::hover());
        ui.painter().line_segment([line.center_top(), line.center_bottom()], Stroke::new(1.0, t.divider));
        let w = (width - SETTINGS_W - 24.0).max(300.0);
        ui.allocate_ui_with_layout(vec2(w, height), egui::Layout::top_down(egui::Align::Min), |ui| {
            ui.set_min_size(vec2(w, height));
            preview(app, ui, t, &mut pu, &st, &ids, pages);
        });
    });
    app.ui.print = pu;
}

fn settings(app: &mut LightcraftApp, ui: &mut Ui, t: &Tokens, pu: &mut PrintUi, st: &PrintSettings, photos: usize) {
    let ctx = ui.ctx().clone();
    let l = &st.layout;
    let unit = st.guides.units;
    ui.spacing_mut().item_spacing.y = 5.0;
    let count_text = if photos == 1 {
        "1 foto seleccionada".to_string()
    } else {
        format!("{photos} fotos seleccionadas")
    };
    ui.label(egui::RichText::new(count_text).color(t.text_dim));

    heading(ui, t, "Plantillas");
    ui.horizontal(|ui| {
        crate::text_field::TextField::singleline("field:printTemplateName", &mut pu.template_name)
            .hint("Nombre de plantilla")
            .width(140.0)
            .show(ui);
        if text_button(ui, "printSaveTemplate", "Guardar", false).clicked() && !pu.template_name.trim().is_empty() {
            run(app, &ctx, "print.template.save", json!({"name": pu.template_name}));
        }
    });
    let names: Vec<String> = app.session.print_templates.iter().map(|t| t.name.clone()).collect();
    for (i, name) in names.iter().enumerate() {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(name).color(t.text_label));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if text_button(ui, &format!("printTemplateDelete-{i}"), "Borrar", false).clicked() {
                    run(app, &ctx, "print.template.delete", json!({"name": name}));
                }
                if text_button(ui, &format!("printTemplateApply-{i}"), "Aplicar", false).clicked() {
                    run(app, &ctx, "print.template.apply", json!({"name": name}));
                    pu.page = 0;
                    pu.selected = None;
                }
            });
        });
    }

    heading(ui, t, "Estilo de Maquetación");
    let styles = [
        (Style::SingleImage, "Individual"),
        (Style::ContactSheet, "Contacto"),
        (Style::PicturePackage, "Paquete"),
        (Style::CustomPackage, "Libre"),
    ];
    ui.horizontal_wrapped(|ui| {
        if let Some(s) = choice(ui, "printStyle", &styles, l.style) {
            patch(app, &ctx, json!({"layout": {"style": s}}));
            pu.page = 0;
        }
    });
    if st.book.enabled {
        photobook_settings(app, ui, t, pu, st);
    }

    heading(ui, t, "Ajustes de Imagen y Marco");
    let im = &st.image;
    if let Some(v) = check(ui, "printZoomFill", "Encajar en Marco (Llenar sin deformar)", im.zoom_to_fill) {
        patch(app, &ctx, json!({"image": {"zoomToFill": v}}));
    }
    if let Some(v) = check(ui, "printRotateFit", "Rotar para encajar", im.rotate_to_fit) {
        patch(app, &ctx, json!({"image": {"rotateToFit": v}}));
    }
    if l.style.is_package() {
        ui.horizontal(|ui| {
            if let Some(v) = check(ui, "printBorder", "Borde de foto", im.border_enabled) {
                patch(app, &ctx, json!({"image": {"borderEnabled": v}}));
            }
            if let Some(w) = length(ui, "printBorderWidth", im.border_width, unit, 0.0, 2.0) {
                patch(app, &ctx, json!({"image": {"borderWidth": w}}));
            }
            if let Some(c) = colour(ui, "printBorderColor", im.border_color) {
                patch(app, &ctx, json!({"image": {"borderColor": c}}));
            }
        });
    }
    ui.horizontal(|ui| {
        if let Some(v) = check(ui, "printStroke", "Trazo interno", im.stroke_enabled) {
            patch(app, &ctx, json!({"image": {"strokeEnabled": v}}));
        }
        let mut pt = im.stroke_width_pt;
        let r = ui.add(egui::DragValue::new(&mut pt).range(0.0..=36.0).speed(0.05).max_decimals(2).suffix(" pt"));
        register(ui.ctx(), "drag:printStrokeWidth", r.rect);
        if r.changed() {
            patch(app, &ctx, json!({"image": {"strokeWidthPt": pt}}));
        }
        if let Some(c) = colour(ui, "printStrokeColor", im.stroke_color) {
            patch(app, &ctx, json!({"image": {"strokeColor": c}}));
        }
    });

    heading(ui, t, "Distribución");
    match l.style {
        Style::SingleImage => {
            ui.label(egui::RichText::new("Una foto llena cada página dentro de los márgenes.").color(t.text_dim));
        }
        Style::ContactSheet => {
            let g = &l.grid;
            row(ui, t, "Cuadrícula", |ui| {
                let (mut c, mut r) = (g.columns, g.rows);
                let a = ui.add(egui::DragValue::new(&mut c).range(1..=layout::MAX_GRID).suffix(" cols"));
                register(ui.ctx(), "drag:printColumns", a.rect);
                let b = ui.add(egui::DragValue::new(&mut r).range(1..=layout::MAX_GRID).suffix(" filas"));
                register(ui.ctx(), "drag:printRows", b.rect);
                if a.changed() || b.changed() {
                    patch(app, &ctx, json!({"layout": {"grid": {"columns": c, "rows": r}}}));
                }
            });
            row(ui, t, "Espaciado", |ui| {
                if let Some(v) = length(ui, "printSpacingX", g.spacing_x, unit, 0.0, 2.0) {
                    patch(app, &ctx, json!({"layout": {"grid": {"spacingX": v}}}));
                }
                if let Some(v) = length(ui, "printSpacingY", g.spacing_y, unit, 0.0, 2.0) {
                    patch(app, &ctx, json!({"layout": {"grid": {"spacingY": v}}}));
                }
            });
            if let Some(v) = check(ui, "printKeepSquare", "Mantener celdas cuadradas", g.keep_square) {
                patch(app, &ctx, json!({"layout": {"grid": {"keepSquare": v}}}));
            }
        }
        Style::PicturePackage | Style::CustomPackage => {
            ui.label(egui::RichText::new("Agregar marco manual").color(t.text_label));
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing = vec2(3.0, 3.0);
                for (i, (w, h)) in CELL_SIZES.iter().enumerate() {
                    if text_button(ui, &format!("printAddCell-{i}"), &format!("{} × {}", trim(*w), trim(*h)), false).clicked() {
                        add_cell(app, &ctx, pu, *w, *h);
                    }
                }
            });
            let (mut w, mut h) = pu.new_cell.unwrap_or((3.0, 2.0));
            row(ui, t, "Tamaño", |ui| {
                if let Some(v) = length(ui, "printNewCellW", w, unit, layout::MIN_CELL_IN, 30.0) {
                    w = v;
                }
                if let Some(v) = length(ui, "printNewCellH", h, unit, layout::MIN_CELL_IN, 30.0) {
                    h = v;
                }
            });
            pu.new_cell = Some((w, h));
            ui.horizontal(|ui| {
                if text_button(ui, "printAddCustom", "Agregar Marco", false).clicked() {
                    add_cell(app, &ctx, pu, w, h);
                }
                if l.style == Style::PicturePackage && text_button(ui, "printAutoLayout", "Auto-Acomodar", false).clicked() {
                    run(app, &ctx, "print.layout.autoLayout", json!({}));
                    pu.page = 0;
                }
            });
            ui.horizontal(|ui| {
                if text_button(ui, "printNewPage", "Nuevo Pliego", false).clicked()
                    && let Some(r) = run(app, &ctx, "print.layout.newPage", json!({}))
                {
                    pu.page = r["page"].as_u64().unwrap_or(0) as usize;
                }
                if text_button(ui, "printClear", "Limpiar Todo", false).clicked() {
                    run(app, &ctx, "print.layout.clear", json!({}));
                    pu.page = 0;
                    pu.selected = None;
                }
            });
            row(ui, t, "Separación", |ui| {
                if let Some(v) = length(ui, "printGap", l.spacing, unit, 0.0, 2.0) {
                    patch(app, &ctx, json!({"layout": {"spacing": v}}));
                }
            });
        }
    }

    heading(ui, t, "Reglas, Cuadrícula y Guías");
    let g = &st.guides;
    if let Some(v) = check(ui, "printRulers", "Mostrar Reglas", g.show_rulers) {
        patch(app, &ctx, json!({"guides": {"showRulers": v}}));
    }
    row(ui, t, "Unidades", |ui| {
        let units = [(Unit::Inches, "in"), (Unit::Centimeters, "cm"), (Unit::Millimeters, "mm"), (Unit::Points, "pt")];
        if let Some(u) = choice(ui, "printUnits", &units, g.units) {
            patch(app, &ctx, json!({"guides": {"units": u}}));
        }
    });
    row(ui, t, "Imán", |ui| {
        if let Some(s) = choice(ui, "printSnap", &[(Snap::Off, "No"), (Snap::Cells, "Marcos"), (Snap::Grid, "Rejilla")], g.snap) {
            patch(app, &ctx, json!({"guides": {"snap": s}}));
        }
    });
    row(ui, t, "Tamaño rejilla", |ui| {
        if let Some(v) = length(ui, "printGridSize", g.grid_size, unit, 0.05, 10.0) {
            patch(app, &ctx, json!({"guides": {"gridSize": v}}));
        }
    });
    for (id, label, on, key) in [
        ("printShowMargins", "Márgenes de página", g.show_margins, "showMargins"),
        ("printShowBleed", "Sangrado de página", g.show_bleed, "showBleed"),
        ("printShowGrid", "Cuadrícula de página", g.show_page_grid, "showPageGrid"),
        ("printShowCells", "Marcos de imagen", g.show_cells, "showCells"),
        ("printShowDims", "Dimensiones", g.show_dimensions, "showDimensions"),
    ] {
        if let Some(v) = check(ui, id, label, on) {
            patch(app, &ctx, json!({"guides": {(key): v}}));
        }
    }

    heading(ui, t, "Configuración de Página");
    row(ui, t, "Papel", |ui| {
        let current = if l.page.paper.eq_ignore_ascii_case("custom") { "custom".to_string() } else { l.page.paper.to_ascii_lowercase() };
        let c = egui::ComboBox::from_id_salt("printPaper").selected_text(paper_label(&current)).show_ui(ui, |ui| {
            for name in layout::paper_names().chain(std::iter::once("custom")) {
                if ui.selectable_label(current == name, paper_label(name)).clicked() {
                    patch(app, &ctx, json!({"layout": {"page": {"paper": name}}}));
                }
            }
        });
        register(ui.ctx(), "combo:printPaper", c.response.rect);
    });
    if l.page.paper.eq_ignore_ascii_case("custom") {
        row(ui, t, "Tamaño", |ui| {
            if let Some(v) = length(ui, "printCustomW", l.page.custom_width, unit, layout::MIN_PAGE_IN, layout::MAX_PAGE_IN) {
                patch(app, &ctx, json!({"layout": {"page": {"customWidth": v}}}));
            }
            if let Some(v) = length(ui, "printCustomH", l.page.custom_height, unit, layout::MIN_PAGE_IN, layout::MAX_PAGE_IN) {
                patch(app, &ctx, json!({"layout": {"page": {"customHeight": v}}}));
            }
        });
    }
    row(ui, t, "Orientación", |ui| {
        let o = [(layout::Orientation::Portrait, "Vertical"), (layout::Orientation::Landscape, "Horizontal")];
        if let Some(v) = choice(ui, "printOrientation", &o, l.page.orientation) {
            patch(app, &ctx, json!({"layout": {"page": {"orientation": v}}}));
        }
    });
    let m = &l.page.margins;
    for (label, id, key, v) in [
        ("Izquierda", "printMarginLeft", "left", m.left),
        ("Derecha", "printMarginRight", "right", m.right),
        ("Arriba", "printMarginTop", "top", m.top),
        ("Abajo", "printMarginBottom", "bottom", m.bottom),
    ] {
        row(ui, t, label, |ui| {
            if let Some(v) = length(ui, id, v, unit, 0.0, layout::MAX_MARGIN_IN) {
                patch(app, &ctx, json!({"layout": {"page": {"margins": {(key): v}}}}));
            }
        });
    }

    heading(ui, t, "Página");
    row(ui, t, "Fondo", |ui| {
        if let Some(c) = colour(ui, "printBackground", st.page.background) {
            patch(app, &ctx, json!({"page": {"background": c}}));
        }
    });
    if compose::has_page_options(l.style) {
        ui.label(egui::RichText::new("Opciones de Página").color(t.text_label));
        let p = &st.page;
        for (id, label, on, key) in [
            ("printPageNumbers", "Números de página", p.page_numbers, "pageNumbers"),
            ("printPageInfo", "Información de página", p.page_info, "pageInfo"),
            ("printCropMarks", "Marcas de corte", p.crop_marks, "cropMarks"),
        ] {
            if let Some(v) = check(ui, id, label, on) {
                patch(app, &ctx, json!({"page": {(key): v}}));
            }
        }
    }

    heading(ui, t, "Trabajo de Impresión");
    row(ui, t, "Resolución", |ui| {
        let mut ppi = st.job.ppi;
        let r = ui.add(egui::DragValue::new(&mut ppi).range(72.0..=720.0).speed(1.0).max_decimals(0).suffix(" ppi"));
        register(ui.ctx(), "drag:printPpi", r.rect);
        if r.changed() {
            patch(app, &ctx, json!({"job": {"ppi": ppi}}));
        }
    });
    print_job_color(app, ui, t, &st.job.color);
}

/// Panel de configuración profesional de Photobook / SmartAlbums y Pre-Prensa a 300 DPI.
fn photobook_settings(app: &mut LightcraftApp, ui: &mut Ui, t: &Tokens, pu: &mut PrintUi, st: &PrintSettings) {
    let ctx = ui.ctx().clone();
    heading(ui, t, "📚 Photobook y Pre-Prensa (SmartAlbums)");
    if let Some(on) = check(ui, "printBookMode", "Modo Photobook Profesional (300 DPI)", st.book.enabled) {
        if on {
            let ids = app.session.targets(&json!({})).unwrap_or_default();
            activate_photobook_mode(app, &ctx, pu, st, &ids);
        } else {
            patch(app, &ctx, json!({"book": {"enabled": false}}));
        }
        return;
    }
    if !st.book.enabled {
        return;
    }

    // 1. Configuración de Formato Físico (300 DPI)
    let preset_matches = |id: &str, cur: &str| {
        cur == id
            || matches!(
                (id, cur),
                ("8x8", "8x16") | ("10x10", "10x20") | ("12x12", "12x24") | ("11x14", "11x28")
            )
    };
    row(ui, t, "Formato 300 DPI", |ui| {
        let cur_paper = st.layout.page.paper.to_ascii_lowercase();
        let cur_label = ALBUM_PRESETS
            .iter()
            .find(|(id, ..)| preset_matches(id, &cur_paper))
            .map(|(_, label, ..)| *label)
            .unwrap_or("Personalizado (Custom)");
        let c = egui::ComboBox::from_id_salt("printAlbumPreset").width(175.0).selected_text(cur_label).show_ui(ui, |ui| {
            for &(id, label, ..) in ALBUM_PRESETS {
                let selected = preset_matches(id, &cur_paper);
                if ui.selectable_label(selected, label).clicked() {
                    if id == "custom" {
                        run(
                            app,
                            &ctx,
                            "print.set_format",
                            json!({
                                "format": "custom",
                                "width": st.layout.page.custom_width,
                                "height": st.layout.page.custom_height,
                                "unit": "in",
                                "spreadMode": st.book.spread_mode
                            }),
                        );
                    } else {
                        run(app, &ctx, "print.set_format", json!({"format": id, "spreadMode": st.book.spread_mode}));
                    }
                    run(
                        app,
                        &ctx,
                        "print.auto_layout_spread",
                        json!({"autoPaginateAll": true, "photosPerSpread": st.book.photos_per_spread}),
                    );
                    pu.page = 0;
                }
            }
        });
        register(ui.ctx(), "combo:printAlbumPreset", c.response.rect);
    });
    if let Some(sm) = check(ui, "printSpreadMode", "Pliegos Dobles (Izquierda + Derecha)", st.book.spread_mode) {
        let cur_paper = st.layout.page.paper.to_ascii_lowercase();
        let fmt = ALBUM_PRESETS
            .iter()
            .find(|(id, ..)| preset_matches(id, &cur_paper))
            .map(|(id, ..)| *id)
            .unwrap_or(cur_paper.as_str());
        run(app, &ctx, "print.set_format", json!({"format": fmt, "spreadMode": sm}));
        run(
            app,
            &ctx,
            "print.auto_layout_spread",
            json!({"autoPaginateAll": true, "photosPerSpread": st.book.photos_per_spread}),
        );
        pu.page = 0;
    }

    // 2. Guías de Pre-Prensa con código de color
    ui.add_space(4.0);
    ui.label(egui::RichText::new("Guías Técnicas de Pre-Prensa (Ocultas al exportar)").font(t.semibold(11.5)).color(t.text));
    // Sangrado / Rebaje (Línea Roja 3..5 mm)
    ui.horizontal(|ui| {
        if let Some(v) = check(ui, "printBookBleed", "🔴 Sangrado (Bleed)", st.guides.show_bleed) {
            patch(app, &ctx, json!({"guides": {"showBleed": v}}));
        }
        let mut mm = st.guides.bleed_mm;
        let r = ui.add(egui::DragValue::new(&mut mm).range(3.0..=5.0).speed(0.1).max_decimals(1).suffix(" mm"));
        register(ui.ctx(), "drag:printBleedMm", r.rect);
        if r.changed() {
            patch(app, &ctx, json!({"guides": {"bleedMm": mm}}));
        }
    });
    // Línea de Corte Final (Trim Line Sólida)
    if let Some(v) = check(ui, "printShowTrim", "⬛ Línea de Corte Final (Guillotina)", st.guides.show_trim) {
        patch(app, &ctx, json!({"guides": {"showTrim": v}}));
    }
    // Zona Segura (Línea Verde Punteada 10..15 mm)
    ui.horizontal(|ui| {
        if let Some(v) = check(ui, "printShowSafeZone", "🟢 Zona Segura", st.guides.show_safe_zone) {
            patch(app, &ctx, json!({"guides": {"showSafeZone": v}}));
        }
        let mut mm = st.guides.safe_zone_mm;
        let r = ui.add(egui::DragValue::new(&mut mm).range(10.0..=15.0).speed(0.2).max_decimals(1).suffix(" mm"));
        register(ui.ctx(), "drag:printSafeZoneMm", r.rect);
        if r.changed() {
            patch(app, &ctx, json!({"guides": {"safeZoneMm": mm}}));
        }
    });
    // Línea Central de Pliegue / Lomo (Center Gutter 10..15 mm)
    ui.horizontal(|ui| {
        if let Some(v) = check(ui, "printShowGutter", "📏 Pliegue Central / Lomo", st.guides.show_gutter) {
            patch(app, &ctx, json!({"guides": {"showGutter": v}}));
        }
        let mut mm = st.guides.gutter_mm;
        let r = ui.add(egui::DragValue::new(&mut mm).range(10.0..=15.0).speed(0.2).max_decimals(1).suffix(" mm"));
        register(ui.ctx(), "drag:printGutterMm", r.rect);
        if r.changed() {
            patch(app, &ctx, json!({"guides": {"gutterMm": mm}}));
        }
    });
    // Portada Dura (Hardcover Wrap 18 mm + Lomo Calculado)
    if let Some(v) = check(ui, "printHardcoverWrap", "📘 Portada Dura (Envuelta 18 mm + Lomo)", st.book.hardcover_wrap) {
        patch(app, &ctx, json!({"book": {"hardcoverWrap": v}}));
    }
    if st.book.hardcover_wrap {
        let spine_mm = st.book.spine_width_mm(st.layout.pages as usize);
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(format!("Envuelta: {:.0} mm · Lomo: {:.1} mm", st.book.wrap_mm, spine_mm)).font(t.font(11.0)).color(t.accent));
            let mut thick = st.book.paper_thickness_mm;
            let r = ui.add(egui::DragValue::new(&mut thick).range(0.10..=1.20).speed(0.02).max_decimals(2).prefix("Hoja ").suffix(" mm"));
            register(ui.ctx(), "drag:printPaperThickness", r.rect);
            if r.changed() {
                patch(app, &ctx, json!({"book": {"paperThicknessMm": thick}}));
            }
        });
    }

    // 3. Motor Auto-Layout estilo Pixellu SmartAlbums
    ui.add_space(4.0);
    ui.label(egui::RichText::new("Auto-Acomodo Inteligente y Marcos Magnéticos").font(t.semibold(11.5)).color(t.text));
    row(ui, t, "Fotos / pliego", |ui| {
        let cur_pps = st.book.photos_per_spread.clamp(1, 8);
        let opts = [(2u8, "2"), (3u8, "3"), (4u8, "4"), (5u8, "5"), (6u8, "6"), (8u8, "8")];
        if let Some(pps) = choice(ui, "printPhotosPerSpread", &opts, cur_pps) {
            run(app, &ctx, "print.auto_layout_spread", json!({"autoPaginateAll": true, "photosPerSpread": pps}));
            pu.page = 0;
        }
    });
    row(ui, t, "Margen interno", |ui| {
        let mut gap = st.book.gap_mm;
        let r = ui.add(egui::DragValue::new(&mut gap).range(0.0..=20.0).speed(0.2).max_decimals(1).suffix(" mm"));
        register(ui.ctx(), "drag:printBookGapMm", r.rect);
        if r.changed() {
            run(app, &ctx, "print.set_format", json!({"gapMm": gap}));
            if !st.layout.cells.is_empty() {
                let cur_var = st.book.variation_for(pu.page);
                run(app, &ctx, "print.cycle_layout", json!({"spread": pu.page, "variation": cur_var}));
            }
        }
    });
    ui.horizontal_wrapped(|ui| {
        if text_button(ui, "printCycleLayout", "🎲 Generar Otra Variante (Espacio / Tab)", true).clicked() {
            run(app, &ctx, "print.cycle_layout", json!({"spread": pu.page}));
        }
        if text_button(ui, "printSmartAutoLayout", "✨ Re-Acomodar Pliego Actual", false).clicked() {
            run(app, &ctx, "print.auto_layout_spread", json!({"spread": pu.page}));
        }
    });
    if text_button(ui, "printAutoPaginateBook", "📚 Re-Acomodar Todas las Fotos Seleccionadas", false).clicked() {
        let ids = app.session.targets(&json!({})).unwrap_or_default();
        activate_photobook_mode(app, &ctx, pu, st, &ids);
    }

    // 4. Tres Modos de Exportación de Imprenta / Laboratorio (a 300 DPI reales)
    ui.add_space(4.0);
    ui.label(egui::RichText::new("Tipo de Encuadernación y Salida (300 DPI)").font(t.semibold(11.5)).color(t.text));
    let cur_mode = st.book.binding_mode;
    let c = egui::ComboBox::from_id_salt("printBindingMode")
        .width(260.0)
        .selected_text(cur_mode.label_es())
        .show_ui(ui, |ui| {
            for mode in [BindingMode::LayflatSpreads, BindingMode::SequentialPdf, BindingMode::SaddleStitch] {
                if ui.selectable_label(cur_mode == mode, mode.label_es()).clicked() {
                    run(app, &ctx, "print.set_format", json!({"bindingMode": mode}));
                }
            }
        });
    register(ui.ctx(), "combo:printBindingMode", c.response.rect);
    let mode_desc: String = match cur_mode {
        BindingMode::LayflatSpreads => "Modo A: Exporta cada pliego 180° completo + sangrado como JPG sRGB a 300 DPI (Pliego_01.jpg, Portada.jpg).".to_string(),
        BindingMode::SequentialPdf => "Modo B: Corta cada pliego por la mitad con sangrado individual, margen de lomo y marcas de guillotina en PDF (Pág 1..N).".to_string(),
        BindingMode::SaddleStitch => {
            let raw_pages = (st.layout.pages as usize) * if st.book.spread_mode { 2 } else { 1 };
            let padded = ((raw_pages.max(1) + 3) / 4) * 4;
            leak_or_static_saddle_desc(raw_pages, padded)
        }
    };
    ui.label(egui::RichText::new(mode_desc).font(t.font(10.5)).color(t.text_dim));

    // 5. Archivo Único de Proyecto para Imprenta (.mmbook)
    ui.add_space(6.0);
    ui.label(egui::RichText::new("Proyecto Portátil para Imprenta (.mmbook)").font(t.semibold(11.5)).color(t.text));
    ui.horizontal_wrapped(|ui| {
        if text_button(ui, "printSaveMmbook", "📦 Guardar Proyecto (.mmbook)…", false).clicked() {
            run(app, &ctx, "app.saveMmbook", json!({}));
        }
        if text_button(ui, "printOpenMmbook", "📂 Abrir Proyecto (.mmbook)…", false).clicked() {
            run(app, &ctx, "app.openMmbook", json!({}));
        }
    });
    ui.label(
        egui::RichText::new(
            "Empaqueta todas las fotos (a 300 DPI con tus ediciones) + maquetación + guías en un solo archivo .mmbook listo para abrir e imprimir en otra computadora con MM Studio.",
        )
        .font(t.font(10.0))
        .color(t.text_dim),
    );
}

fn leak_or_static_saddle_desc(raw_pages: usize, padded: usize) -> String {
    format!("Modo C: Imposición Cuadernillo Múltiplo de 4 ({raw_pages} págs → {padded} págs con guardas, fórmula N+1 Tiro y Retiro).")
}

/// Send one colour setting of the print job (the rest stay as they are).
fn patch_color(app: &mut LightcraftApp, ctx: &egui::Context, key: &str, value: Value) {
    patch(app, ctx, json!({"job": {"color": {(key): value}}}));
}

fn to_value<V: serde::Serialize>(v: V) -> Value {
    serde_json::to_value(v).unwrap_or(Value::Null)
}

/// The Print Job panel's colour management, print adjustment, print sharpening and depth.
fn print_job_color(app: &mut LightcraftApp, ui: &mut Ui, t: &Tokens, color: &PrintColorSettings) {
    let ctx = ui.ctx().clone();
    let managed = |space| PrintProfile::Managed { space };
    let shown = match &color.profile {
        PrintProfile::Managed { space: ManagedSpace::Srgb } => "Gestionado por Impresora (sRGB)".to_string(),
        PrintProfile::Managed { space: ManagedSpace::AdobeRgb } => "Gestionado por Impresora (Adobe RGB)".to_string(),
        PrintProfile::Icc { path } => path.file_stem().map_or_else(|| path.display().to_string(), |n| n.to_string_lossy().into_owned()),
    };
    let mut pick = false;
    row(ui, t, "Perfil", |ui| {
        let c = egui::ComboBox::from_id_salt("print-profile").width(190.0).selected_text(shown.clone()).show_ui(ui, |ui| {
            for (space, label) in [
                (ManagedSpace::Srgb, "Gestionado por Impresora (sRGB)"),
                (ManagedSpace::AdobeRgb, "Gestionado por Impresora (Adobe RGB)"),
            ] {
                let r = ui.selectable_label(color.profile == managed(space), label);
                register(ui.ctx(), format!("item:printProfile-{}", to_value(space).as_str().unwrap_or_default()), r.rect);
                if r.clicked() && color.profile != managed(space) {
                    patch_color(app, &ctx, "profile", to_value(managed(space)));
                }
            }
            if let PrintProfile::Icc { .. } = &color.profile {
                let _ = ui.selectable_label(true, shown.as_str());
            }
            if app.services.pick_display_profile.is_some() {
                let r = ui.selectable_label(false, "Otro perfil ICC…");
                register(ui.ctx(), "item:printProfile-other", r.rect);
                pick |= r.clicked();
            }
        });
        register(ui.ctx(), "combo:printProfile", c.response.rect);
    });
    if pick && let Some(path) = app.services.pick_display_profile.as_mut().and_then(|pick| pick().into_iter().next()) {
        patch_color(app, &ctx, "profile", json!({"kind": "icc", "path": path}));
    }
    row(ui, t, "Intento", |ui| {
        if let Some(v) = choice(ui, "printIntent", &[(PrintIntent::Perceptual, "Perceptual"), (PrintIntent::Relative, "Relativo")], color.intent) {
            patch_color(app, &ctx, "intent", to_value(v));
        }
    });
    ui.add_enabled_ui(color.intent == PrintIntent::Relative, |ui| {
        if let Some(v) = check(ui, "printBpc", "Compensación de punto negro", color.black_point_compensation) {
            patch_color(app, &ctx, "blackPointCompensation", json!(v));
        }
    });
    for (key, id_suffix, label, value) in [
        ("brightness", "Brightness", "Brillo", color.brightness),
        ("contrast", "Contrast", "Contraste", color.contrast),
    ] {
        row(ui, t, label, |ui| {
            let mut v = value;
            let r = ui.add(egui::DragValue::new(&mut v).range(-100.0..=100.0).speed(0.5).max_decimals(0));
            register(ui.ctx(), format!("drag:print{id_suffix}"), r.rect);
            if r.changed() {
                patch_color(app, &ctx, key, json!(v));
            }
        });
    }
    row(ui, t, "Enfoque", |ui| {
        let options = [
            (PrintSharpen::Off, "No"),
            (PrintSharpen::Low, "Bajo"),
            (PrintSharpen::Standard, "Estándar"),
            (PrintSharpen::High, "Alto"),
        ];
        if let Some(v) = choice(ui, "printSharpen", &options, color.sharpen) {
            patch_color(app, &ctx, "sharpen", to_value(v));
        }
    });
    ui.add_enabled_ui(color.sharpen != PrintSharpen::Off, |ui| {
        row(ui, t, "Papel", |ui| {
            if let Some(v) = choice(ui, "printMedia", &[(PrintMedia::Matte, "Mate"), (PrintMedia::Glossy, "Brillante")], color.media) {
                patch_color(app, &ctx, "media", to_value(v));
            }
        });
    });
    if let Some(v) = check(ui, "print16bit", "16 bits por canal", color.sixteen_bit) {
        patch_color(app, &ctx, "sixteenBit", json!(v));
    }
    let bits = if color.sixteen_bit { "16-bit" } else { "8-bit" };
    let summary = format!("PDF · {shown} · {bits} · ediciones actuales");
    ui.label(egui::RichText::new(summary).color(t.text_dim));
}

fn trim(v: f64) -> String {
    let s = format!("{v:.2}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

fn add_cell(app: &mut LightcraftApp, ctx: &egui::Context, pu: &mut PrintUi, w: f64, h: f64) {
    if let Some(r) = run(app, ctx, "print.cell.add", json!({"width": w, "height": h})) {
        pu.selected = r["id"].as_u64().and_then(|v| u32::try_from(v).ok());
        pu.page = r["cell"]["page"].as_u64().unwrap_or(0) as usize;
    }
}

// ---- preview -------------------------------------------------------------------------------

/// Screen geometry of the page on show.
struct Sheet {
    page: Rect,
    /// Screen points per inch.
    scale: f32,
}

impl Sheet {
    fn rect(&self, r: &layout::Rect) -> Rect {
        Rect::from_min_size(self.page.min + vec2(r.x as f32, r.y as f32) * self.scale, vec2(r.w as f32, r.h as f32) * self.scale)
    }
}

fn preview(app: &mut LightcraftApp, ui: &mut Ui, t: &Tokens, pu: &mut PrintUi, st: &PrintSettings, ids: &[PhotoId], pages: usize) {
    let ctx = ui.ctx().clone();

    // Mantener el pool de fotos del Photobook sincronizado con las fotos seleccionadas y las colocadas en celdas
    if st.book.enabled {
        for &id in ids {
            if !pu.book_pool.contains(&id.0) {
                pu.book_pool.push(id.0);
            }
        }
        for c in &st.layout.cells {
            if let Some(pid) = c.photo
                && !pu.book_pool.contains(&pid)
            {
                pu.book_pool.push(pid);
            }
        }
        pu.book_pool.retain(|&pid| app.session.catalog.photo(PhotoId(pid)).is_some());
    }

    // Atajos de teclado Espacio o Tab para ciclar diseños matemáticos estilo Pixellu SmartAlbums
    if (st.book.enabled || st.layout.style == Style::CustomPackage)
        && !ctx.egui_wants_keyboard_input()
        && ctx.input(|i| i.key_pressed(egui::Key::Space) || i.key_pressed(egui::Key::Tab))
    {
        if ctx.input(|i| i.key_pressed(egui::Key::Tab)) {
            ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Tab));
        }
        run(app, &ctx, "print.cycle_layout", json!({"spread": pu.page}));
    }

    // Barra superior rápida para Photobook / SmartAlbums
    if st.book.enabled {
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            if text_button(ui, "printTopBookToggle", "📚 Photobook: ACTIVO", true).clicked() {
                patch(app, &ctx, json!({"book": {"enabled": false}}));
            }
            if text_button(ui, "printTopCycleLayout", "🎲 Otra Variante de Diseño [Espacio / Tab]", false).clicked() {
                run(app, &ctx, "print.cycle_layout", json!({"spread": pu.page}));
            }
            if let Some(sel_id) = pu.selected {
                let in_pan = pu.pan_mode_cell == Some(sel_id);
                let label = if in_pan { "✅ Fijar Encuadre" } else { "✋ Encuadrar Foto (Doble Clic)" };
                if text_button(ui, "printTopPanMode", label, in_pan).clicked() {
                    pu.pan_mode_cell = if in_pan { None } else { Some(sel_id) };
                }
            }
            ui.label(egui::RichText::new("Fotos/pliego:").font(t.font(11.0)).color(t.text_dim));
            let cur_pps = st.book.photos_per_spread.clamp(1, 8);
            for pps in [2u8, 3, 4, 5, 6, 8] {
                if text_button(ui, &format!("printTopPps-{pps}"), &format!("{pps}"), cur_pps == pps).clicked() && cur_pps != pps {
                    run(app, &ctx, "print.auto_layout_spread", json!({"autoPaginateAll": true, "photosPerSpread": pps}));
                    pu.page = 0;
                }
            }
            if text_button(ui, "printTopSaveMmbook", "📦 Guardar .mmbook", false).clicked() {
                run(app, &ctx, "app.saveMmbook", json!({}));
            }
            if text_button(ui, "printTopOpenMmbook", "📂 Abrir .mmbook", false).clicked() {
                run(app, &ctx, "app.openMmbook", json!({}));
            }
        });
    }

    // room for the page buttons, inspector, and optional photobook Pila de Fotos
    let filmstrip_h = if st.book.enabled { 68.0 } else { 0.0 };
    let footer = (if st.layout.style.is_package() { 108.0 } else { 44.0 }) + filmstrip_h;
    let size = vec2(ui.available_width(), (ui.available_height() - footer).max(120.0));
    let (area, _) = ui.allocate_exact_size(size, Sense::hover());
    let painter = ui.painter_at(area);
    painter.rect_filled(area, 4.0, t.canvas);
    let Ok((pw, ph)) = st.layout.page_size() else {
        painter.text(area.center(), Align2::CENTER_CENTER, "Esta configuración de página no es válida", t.font(13.0), t.caution);
        return;
    };
    let g = &st.guides;
    let ruler = if g.show_rulers { 22.0 } else { 0.0 };
    // Leave extra padding when bleed or hardcover wrap is visible so the outer guides fit inside `inner`
    let extra_pad = if st.book.enabled && st.book.hardcover_wrap { 28.0 } else if g.show_bleed { 18.0 } else { 14.0 };
    let inner = Rect::from_min_max(area.min + vec2(ruler, ruler), area.max).shrink(extra_pad);
    let scale = ((inner.width() as f64 / pw).min(inner.height() as f64 / ph) as f32).max(1.0);
    let sheet = Sheet { page: Rect::from_center_size(inner.center(), vec2(pw as f32, ph as f32) * scale), scale };
    register(&ctx, "print:page", sheet.page);

    painter.rect_filled(sheet.page.translate(vec2(2.0, 3.0)), 0.0, Color32::from_black_alpha(70));
    let bg = st.page.background;
    painter.rect_filled(sheet.page, 0.0, Color32::from_rgb(bg[0], bg[1], bg[2]));
    let guide = Stroke::new(1.0, Color32::from_rgb(90, 160, 240));
    if g.show_rulers {
        rulers(&painter, t, &sheet, area, pw, ph, g.units);
    }
    let area_in = st.layout.printable().ok();
    if g.show_bleed && !st.book.enabled {
        let b = (g.bleed_mm / 25.4).clamp(0.05, 0.5);
        dashed(&painter, sheet.page.expand(b as f32 * scale), Stroke::new(1.0, t.caution));
    }
    if g.show_page_grid
        && let Some(a) = area_in
    {
        let faint = Stroke::new(1.0, Color32::from_rgba_unmultiplied(120, 120, 120, 70));
        let step = g.grid_size.max(0.05);
        let mut x = 0.0;
        while a.x + x <= a.right() + 1e-9 && x < 200.0 {
            let sx = sheet.page.left() + (a.x + x) as f32 * scale;
            painter.line_segment([pos2(sx, sheet.page.top()), pos2(sx, sheet.page.bottom())], faint);
            x += step;
        }
        let mut y = 0.0;
        while a.y + y <= a.bottom() + 1e-9 && y < 200.0 {
            let sy = sheet.page.top() + (a.y + y) as f32 * scale;
            painter.line_segment([pos2(sheet.page.left(), sy), pos2(sheet.page.right(), sy)], faint);
            y += step;
        }
    }
    if g.show_margins
        && let Some(a) = area_in
    {
        dashed(&painter, sheet.rect(&layout::Rect::new(a.x, a.y, a.w, a.h)), guide);
    }

    let package = st.layout.style.is_package();
    if package {
        package_cells(app, ui, &painter, t, pu, st, ids, &sheet);
    } else {
        let plan = compose::plan_cells(&st.layout, ids).ok();
        let cells: Vec<(layout::Rect, Option<PhotoId>)> = match plan.as_ref().and_then(|p| p.get(pu.page)) {
            Some(page) => page.iter().map(|(r, id)| (*r, Some(*id))).collect(),
            None => st.layout.grid_rects().unwrap_or_default().into_iter().map(|r| (r, None)).collect(),
        };
        let mut shown = Vec::new();
        for (i, (r, photo)) in cells.iter().enumerate() {
            let rect = sheet.rect(r);
            register(&ctx, format!("print:gridcell:{i}"), rect);
            shown.extend(paint_cell(app, &painter, t, st, rect, *photo, &sheet, r, None));
        }
        page_options(&painter, st, &sheet, &shown, pu.page + 1);
    }

    // Guías Técnicas de Pre-Prensa y Alertas Preventivas (Solo en vista de edición, ocultas al exportar)
    if st.book.enabled {
        draw_prepress_overlays(app, &painter, t, st, &sheet, pu.page);
    }

    // footer: page navigation and the selected cell
    ui.horizontal(|ui| {
        if text_button(ui, "printPagePrev", "◀", false).clicked() && pu.page > 0 {
            pu.page -= 1;
            pu.selected = None;
        }
        let page_title = if st.book.enabled && st.book.spread_mode {
            format!("Pliego {} / {} (Págs {}–{})", pu.page + 1, pages, pu.page * 2 + 1, pu.page * 2 + 2)
        } else {
            format!("Página {} / {}", pu.page + 1, pages)
        };
        let label = ui.label(egui::RichText::new(page_title).color(t.text));
        register(ui.ctx(), "print:pageLabel", label.rect);
        if text_button(ui, "printPageNext", "▶", false).clicked() && pu.page + 1 < pages {
            pu.page += 1;
            pu.selected = None;
        }
        ui.label(egui::RichText::new(fmt_len(pw, g.units) + " × " + &fmt_len(ph, g.units)).color(t.text_dim));
        if package {
            ui.label(egui::RichText::new(format!("{} marcos", st.layout.cells.len())).color(t.text_dim));
            if st.layout.has_overlaps() {
                ui.label(egui::RichText::new("marcos superpuestos").color(t.caution));
            }
        }
        if st.book.enabled {
            let cur_var = st.book.variation_for(pu.page);
            ui.label(
                egui::RichText::new(format!(
                    "· Variante #{} · {} {:.0} DPI",
                    cur_var + 1,
                    st.book.binding_mode.label_es(),
                    st.job.ppi
                ))
                .font(t.font(11.0))
                .color(t.accent),
            );
        } else {
            if text_button(ui, "printFooterBookToggle", "📚 Modo Photobook", true).clicked() {
                activate_photobook_mode(app, &ctx, pu, st, ids);
            }
            if text_button(ui, "printFooterOpenMmbook", "📂 .mmbook", false).clicked() {
                run(app, &ctx, "app.openMmbook", json!({}));
            }
        }
    });
    if package {
        inspector(app, ui, t, pu, st);
    }
    if st.book.enabled {
        draw_photobook_filmstrip(app, ui, t, pu, st, ids, &sheet);
    }
}

/// Dibuja las 4 guías técnicas de Pre-Prensa con código de color y las alertas preventivas sobre el lienzo.
fn draw_prepress_overlays(
    app: &LightcraftApp,
    painter: &egui::Painter,
    t: &Tokens,
    st: &PrintSettings,
    sheet: &Sheet,
    page_idx: usize,
) {
    let g = &st.guides;
    let trim_rect = sheet.page;

    // 0. Modo Portada Dura (Hardcover Wrap de 18 mm + Lomo calculado)
    if st.book.hardcover_wrap {
        let wrap_px = (st.book.wrap_mm / 25.4) as f32 * sheet.scale;
        let spine_mm = st.book.spine_width_mm(st.layout.pages as usize);
        let spine_px = (spine_mm / 25.4) as f32 * sheet.scale;
        let wrap_rect = trim_rect.expand(wrap_px);
        let wrap_color = Color32::from_rgb(245, 175, 50);
        dashed(painter, wrap_rect, Stroke::new(1.5, wrap_color));
        // Lomo central de portada dura
        let spine_rect = Rect::from_center_size(trim_rect.center(), vec2(spine_px.max(4.0), wrap_rect.height()));
        painter.rect_filled(spine_rect, 0.0, Color32::from_rgba_unmultiplied(245, 175, 50, 45));
        painter.rect_stroke(spine_rect, 0.0, Stroke::new(1.2, wrap_color), StrokeKind::Inside);
        let badge = format!("ENVUELTA PORTADA DURA ({:.0} mm) · LOMO {:.1} mm", st.book.wrap_mm, spine_mm);
        let galley = painter.layout_no_wrap(badge, t.font(9.5), Color32::BLACK);
        let pos = pos2(wrap_rect.left() + 4.0, (wrap_rect.top() - galley.size().y - 2.0).max(2.0));
        painter.rect_filled(Rect::from_min_size(pos, galley.size()).expand(2.0), 2.0, wrap_color);
        painter.galley(pos, galley, Color32::BLACK);
    }

    // 1. Línea de Sangrado / Rebaje (Bleed — Línea Roja hacia afuera de 3 mm a 5 mm)
    if g.show_bleed {
        let bleed_mm = g.bleed_mm.clamp(3.0, 5.0);
        let bleed_px = (bleed_mm / 25.4) as f32 * sheet.scale;
        let bleed_rect = trim_rect.expand(bleed_px);
        let red = Color32::from_rgb(235, 52, 52);
        painter.rect_stroke(bleed_rect, 0.0, Stroke::new(1.6, red), StrokeKind::Outside);
        let label = format!("SANGRADO / BLEED ({bleed_mm:.1} mm)");
        let galley = painter.layout_no_wrap(label, t.font(9.0), Color32::WHITE);
        let pos = pos2(bleed_rect.right() - galley.size().x - 4.0, bleed_rect.top() + 2.0);
        painter.rect_filled(Rect::from_min_size(pos, galley.size()).expand(2.0), 2.0, Color32::from_rgba_unmultiplied(210, 40, 40, 220));
        painter.galley(pos, galley, Color32::WHITE);
    }

    // 2. Línea de Corte Final (Trim Line — Línea Negra/Blanca sólida donde pasa la guillotina)
    if g.show_trim {
        painter.rect_stroke(trim_rect, 0.0, Stroke::new(2.0, Color32::WHITE), StrokeKind::Inside);
        painter.rect_stroke(trim_rect, 0.0, Stroke::new(1.0, Color32::BLACK), StrokeKind::Inside);
        // Marcas de guillotina en las esquinas
        let tick = 8.0;
        let tick_stroke = Stroke::new(1.2, Color32::from_rgb(30, 30, 30));
        for (c, dx, dy) in [
            (trim_rect.left_top(), -1.0, -1.0),
            (trim_rect.right_top(), 1.0, -1.0),
            (trim_rect.left_bottom(), -1.0, 1.0),
            (trim_rect.right_bottom(), 1.0, 1.0),
        ] {
            painter.line_segment([c + vec2(dx * 3.0, 0.0), c + vec2(dx * (3.0 + tick), 0.0)], tick_stroke);
            painter.line_segment([c + vec2(0.0, dy * 3.0), c + vec2(0.0, dy * (3.0 + tick))], tick_stroke);
        }
    }

    // 3. Zona Segura (Safe Zone — Línea Verde punteada de 10 mm a 15 mm hacia adentro)
    if g.show_safe_zone {
        let safe_mm = g.safe_zone_mm.clamp(10.0, 15.0);
        let safe_px = (safe_mm / 25.4) as f32 * sheet.scale;
        let green = Color32::from_rgb(42, 212, 108);
        let safe_rect = trim_rect.shrink(safe_px);
        if safe_rect.width() > 4.0 && safe_rect.height() > 4.0 {
            dashed(painter, safe_rect, Stroke::new(1.4, green));
            let label = format!("ZONA SEGURA ({safe_mm:.0} mm)");
            let galley = painter.layout_no_wrap(label, t.font(9.0), Color32::BLACK);
            let pos = pos2(safe_rect.left() + 4.0, safe_rect.top() + 2.0);
            painter.rect_filled(Rect::from_min_size(pos, galley.size()).expand(2.0), 2.0, Color32::from_rgba_unmultiplied(42, 212, 108, 220));
            painter.galley(pos, galley, Color32::BLACK);
        }
    }

    // 4. Línea Central de Pliegue / Lomo (Center Gutter / Crease)
    if g.show_gutter && st.book.spread_mode {
        let cx = trim_rect.center().x;
        match st.book.binding_mode {
            BindingMode::LayflatSpreads => {
                // Modo Layflat (180°): línea sutil de doblez al centro
                let fold_color = Color32::from_rgba_unmultiplied(70, 195, 255, 210);
                painter.extend(egui::Shape::dashed_line(
                    &[pos2(cx, trim_rect.top()), pos2(cx, trim_rect.bottom())],
                    Stroke::new(1.5, fold_color),
                    6.0,
                    4.0,
                ));
                let galley = painter.layout_no_wrap("PLIEGUE 180° LAYFLAT".to_string(), t.font(8.5), Color32::WHITE);
                let pos = pos2(cx - galley.size().x / 2.0, trim_rect.bottom() - galley.size().y - 4.0);
                painter.rect_filled(Rect::from_min_size(pos, galley.size()).expand(2.0), 2.0, Color32::from_rgba_unmultiplied(20, 90, 140, 210));
                painter.galley(pos, galley, Color32::WHITE);
            }
            BindingMode::SequentialPdf | BindingMode::SaddleStitch => {
                // Modo Libro Tradicional (Pegado/Cosido): franja sombreada configurable (10 mm a 15 mm) indicando pérdida en el lomo
                let gutter_mm = g.gutter_mm.clamp(10.0, 15.0);
                let gutter_half_px = (gutter_mm / 25.4) as f32 * sheet.scale;
                let gutter_rect = Rect::from_min_max(
                    pos2((cx - gutter_half_px).max(trim_rect.left()), trim_rect.top()),
                    pos2((cx + gutter_half_px).min(trim_rect.right()), trim_rect.bottom()),
                );
                painter.rect_filled(gutter_rect, 0.0, Color32::from_rgba_unmultiplied(25, 25, 30, 145));
                painter.rect_stroke(gutter_rect, 0.0, Stroke::new(1.2, Color32::from_rgb(255, 165, 50)), StrokeKind::Inside);
                painter.line_segment([pos2(cx, trim_rect.top()), pos2(cx, trim_rect.bottom())], Stroke::new(1.0, Color32::from_rgb(255, 110, 50)));
                let label = format!("LOMO / GUTTER ({gutter_mm:.0} mm)");
                let galley = painter.layout_no_wrap(label, t.font(8.5), Color32::BLACK);
                let pos = pos2(cx - galley.size().x / 2.0, trim_rect.bottom() - galley.size().y - 4.0);
                painter.rect_filled(Rect::from_min_size(pos, galley.size()).expand(2.0), 2.0, Color32::from_rgba_unmultiplied(255, 175, 55, 230));
                painter.galley(pos, galley, Color32::BLACK);
            }
        }
    }

    // 5. Alertas Visuales Preventivas (Rostros o Textos en Pliegue Central o fuera de Zona Segura)
    let warnings = book::inspect_spread_warnings(st, page_idx as u32, |pid| {
        let Some(photo) = app.session.catalog.photo(lightcraft_catalog::PhotoId(pid)) else {
            return Vec::new();
        };
        photo
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
            .collect()
    });
    if !warnings.is_empty() {
        let warn_red = Color32::from_rgb(255, 68, 58);
        for w in &warnings {
            if let Some(cell) = st.layout.cell(w.cell_id) {
                let r = sheet.rect(&cell.rect);
                painter.rect_stroke(r, 2.0, Stroke::new(2.5, warn_red), StrokeKind::Outside);
                let badge = match w.kind {
                    WarningKind::FaceInGutter => "⚠ ROSTRO EN PLIEGUE CENTRAL",
                    WarningKind::FaceOutsideSafeZone => "⚠ ROSTRO FUERA DE ZONA SEGURA",
                    WarningKind::TextInGutter => "⚠ TEXTO EN PLIEGUE CENTRAL",
                    WarningKind::TextOutsideSafeZone => "⚠ TEXTO FUERA DE ZONA SEGURA",
                };
                let galley = painter.layout_no_wrap(badge.to_string(), t.semibold(9.5), Color32::WHITE);
                let pos = pos2(r.left() + 4.0, r.top() + 4.0);
                painter.rect_filled(Rect::from_min_size(pos, galley.size()).expand(3.0), 3.0, Color32::from_rgba_unmultiplied(220, 38, 38, 235));
                painter.galley(pos, galley, Color32::WHITE);
            }
        }
        // Banner preventivo superior
        let msg = &warnings[0].message;
        let extra = if warnings.len() > 1 { format!(" (+{} alerta(s) más)", warnings.len() - 1) } else { String::new() };
        let banner_text = format!("{msg}{extra}");
        let galley = painter.layout_no_wrap(banner_text, t.semibold(11.0), Color32::WHITE);
        let pos = pos2(trim_rect.center().x - galley.size().x / 2.0, trim_rect.top() + 8.0);
        painter.rect_filled(Rect::from_min_size(pos, galley.size()).expand(5.0), 4.0, Color32::from_rgba_unmultiplied(200, 30, 30, 240));
        painter.galley(pos, galley, Color32::WHITE);
    }
}

/// "Pila de Fotos" (Holding Tray) inferior interactiva:
/// - Contiene únicamente las fotos que NO están colocadas actualmente en ningún marco del Photobook.
/// - Al activar el Modo Photobook, todas las fotos seleccionadas quedan acomodadas en los pliegos y la Pila inicia vacía.
/// - El usuario puede arrastrar cualquier foto desde un marco del lienzo hacia la Pila (dejando el marco vacío),
///   o arrastrar una foto desde la Pila hacia cualquier marco del lienzo (encajando por magnetismo y enviando
///   la foto que estaba ahí de regreso a la Pila).
fn draw_photobook_filmstrip(
    app: &mut LightcraftApp,
    ui: &mut Ui,
    t: &Tokens,
    pu: &mut PrintUi,
    st: &PrintSettings,
    _ids: &[PhotoId],
    sheet: &Sheet,
) {
    let ctx = ui.ctx().clone();
    let placed: std::collections::HashSet<u64> = st.layout.cells.iter().filter_map(|c| c.photo).collect();
    let pile_photos: Vec<PhotoId> = pu
        .book_pool
        .iter()
        .copied()
        .filter(|pid| !placed.contains(pid))
        .map(PhotoId)
        .collect();

    ui.add_space(2.0);
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(format!("📥 Pila de Fotos ({} en espera):", pile_photos.len()))
                .font(t.semibold(11.0))
                .color(if pile_photos.is_empty() { t.text_label } else { t.accent }),
        );
        if let Some(cid) = pu.selected
            && st.layout.cell(cid).is_some_and(|c| c.photo.is_some())
            && text_button(ui, "printSendSelectedToPile", "📥 Enviar foto seleccionada a la Pila", false).clicked()
        {
            run(app, &ctx, "print.cell.photo", json!({"id": cid, "photo": null}));
        }
        if pile_photos.is_empty() {
            ui.label(
                egui::RichText::new("Pila vacía (todas tus fotos están acomodadas). Arrastra aquí cualquier foto del lienzo que quieras mover.")
                    .font(t.font(10.5))
                    .color(t.text_dim),
            );
        } else {
            ui.label(
                egui::RichText::new("Arrastra una foto de la Pila hacia un marco del lienzo para encajarla magnéticamente (o haz clic para ponerla en el marco activo).")
                    .font(t.font(10.5))
                    .color(t.text_dim),
            );
        }
    });

    let avail_w = ui.available_width().max(200.0);
    let tray_h = 44.0;
    let (tray_rect, _) = ui.allocate_exact_size(vec2(avail_w, tray_h), Sense::hover());
    pu.pile_rect = Some(tray_rect);
    register(&ctx, "print:photoPile", tray_rect);

    let dragging_from_canvas = pu.photo_drag.as_ref().is_some_and(|d| d.from_cell.is_some());
    let hovered_by_drag = dragging_from_canvas && pu.photo_drag.as_ref().is_some_and(|d| tray_rect.expand(8.0).contains(d.pos));

    let tray_bg = if hovered_by_drag {
        Color32::from_rgba_unmultiplied(40, 120, 190, 65)
    } else {
        Color32::from_rgba_unmultiplied(30, 34, 42, 180)
    };
    ui.painter().rect_filled(tray_rect, 4.0, tray_bg);
    let border_col = if hovered_by_drag {
        Color32::from_rgb(70, 210, 255)
    } else if !pile_photos.is_empty() {
        t.accent
    } else {
        t.divider
    };
    ui.painter().rect_stroke(tray_rect, 4.0, Stroke::new(if hovered_by_drag { 2.0 } else { 1.0 }, border_col), StrokeKind::Inside);

    if pile_photos.is_empty() {
        let msg = if hovered_by_drag {
            "🧲 SUELTA AQUÍ PARA GUARDAR LA FOTO EN LA PILA (EL MARCO QUEDARÁ VACÍO)"
        } else {
            "📥 Zona Magnética de Pila de Fotos — Arrastra aquí una foto desde cualquier marco del pliego para apartarla"
        };
        ui.painter().text(
            tray_rect.center(),
            Align2::CENTER_CENTER,
            msg,
            t.semibold(11.0),
            if hovered_by_drag { Color32::from_rgb(110, 230, 255) } else { t.text_dim },
        );
    } else {
        let mut child_ui = ui.new_child(egui::UiBuilder::new().max_rect(tray_rect.shrink2(vec2(6.0, 3.0))));
        egui::ScrollArea::horizontal().id_salt("print-photo-pile").max_height(tray_h - 6.0).show(&mut child_ui, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 6.0;
                for &pid in &pile_photos {
                    let (rect, resp) = ui.allocate_exact_size(vec2(52.0, 38.0), Sense::click_and_drag());
                    register(&ctx, format!("print:filmstrip:{}", pid.0), rect);
                    if app.renderer.thumb(pid).is_none() {
                        crate::panels::grid::request_thumb(app, pid, 128, 10);
                    }
                    if let Some(tex) = app.renderer.thumb(pid) {
                        ui.painter().image(
                            tex.tex.id(),
                            rect,
                            Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
                            Color32::WHITE,
                        );
                    } else {
                        ui.painter().rect_filled(rect, 3.0, Color32::from_gray(60));
                    }
                    ui.painter().rect_stroke(rect, 3.0, Stroke::new(1.5, t.accent), StrokeKind::Inside);

                    if resp.drag_started() {
                        let pos = resp.interact_pointer_pos().unwrap_or(rect.center());
                        pu.photo_drag = Some(PhotoDrag {
                            photo_id: pid.0,
                            from_cell: None,
                            pos,
                        });
                    }
                    if resp.clicked() {
                        // Si hay un marco seleccionado en el pliego actual (o un marco vacío), encaja la foto ahí
                        let target_cell = pu
                            .selected
                            .filter(|&cid| st.layout.cell(cid).is_some_and(|c| c.page as usize == pu.page))
                            .or_else(|| {
                                st.layout
                                    .cells_on_page(pu.page as u32)
                                    .find(|c| c.photo.is_none())
                                    .map(|c| c.id)
                            })
                            .or_else(|| st.layout.cells_on_page(pu.page as u32).next().map(|c| c.id));
                        if let Some(cid) = target_cell {
                            run(app, &ctx, "print.cell.photo", json!({"id": cid, "photo": pid.0}));
                            pu.selected = Some(cid);
                        }
                    }
                }
            });
        });
    }

    // Resolver el arrastre magnético activo (PhotoDrag) y dibujar vista previa flotante
    finish_magnetic_photo_drag(app, &ctx, t, pu, st, sheet);
}

/// Dibuja la miniatura flotante durante el arrastre magnético y ejecuta el intercambio al soltar el ratón.
fn finish_magnetic_photo_drag(
    app: &mut LightcraftApp,
    ctx: &egui::Context,
    t: &Tokens,
    pu: &mut PrintUi,
    st: &PrintSettings,
    sheet: &Sheet,
) {
    let Some(mut drag) = pu.photo_drag.clone() else { return };
    if let Some(p) = ctx.input(|i| i.pointer.interact_pos().or_else(|| i.pointer.latest_pos())) {
        drag.pos = p;
    }
    let cells_on_spread: Vec<layout::Cell> = st.layout.cells_on_page(pu.page as u32).cloned().collect();
    let hovered_cell = cells_on_spread
        .iter()
        .rev()
        .find(|c| sheet.rect(&c.rect).contains(drag.pos))
        .cloned();

    // Dibujar miniatura flotante siguiendo el cursor en la capa de primer plano
    let layer = egui::LayerId::new(egui::Order::Tooltip, egui::Id::new("print-magnetic-drag-preview"));
    let fg = ctx.layer_painter(layer);
    if let Some(target) = &hovered_cell {
        let trect = sheet.rect(&target.rect);
        let mag_col = Color32::from_rgb(55, 220, 255);
        fg.rect_stroke(trect, 2.0, Stroke::new(3.0, mag_col), StrokeKind::Outside);
        let badge = if Some(target.id) == drag.from_cell {
            "🧲 Marco actual"
        } else if target.photo.is_some() {
            "🧲 Soltar para encajar aquí (la foto actual irá a la Pila)"
        } else {
            "🧲 Soltar para encajar en este Marco Magnético"
        };
        let galley = fg.layout_no_wrap(badge.to_string(), t.semibold(10.5), Color32::BLACK);
        let bpos = pos2(
            (trect.center().x - galley.size().x / 2.0).max(trect.left() + 4.0),
            (trect.top() + 6.0).min(trect.bottom() - galley.size().y - 2.0),
        );
        fg.rect_filled(Rect::from_min_size(bpos, galley.size()).expand(4.0), 4.0, mag_col);
        fg.galley(bpos, galley, Color32::BLACK);
    }

    let thumb_rect = Rect::from_center_size(drag.pos + vec2(18.0, -18.0), vec2(76.0, 56.0));
    fg.rect_filled(thumb_rect.translate(vec2(2.0, 3.0)), 4.0, Color32::from_black_alpha(150));
    if let Some(tex) = app.renderer.thumb(PhotoId(drag.photo_id)) {
        fg.image(
            tex.tex.id(),
            thumb_rect,
            Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
            Color32::WHITE,
        );
    } else {
        fg.rect_filled(thumb_rect, 4.0, Color32::from_gray(80));
    }
    fg.rect_stroke(thumb_rect, 4.0, Stroke::new(2.0, t.accent), StrokeKind::Outside);

    let released = ctx.input(|i| !i.pointer.primary_down());
    if released {
        pu.photo_drag = None;
        if let Some(target) = hovered_cell {
            if Some(target.id) != drag.from_cell {
                if let Some(src_id) = drag.from_cell {
                    run(app, ctx, "print.cell.photo", json!({"id": src_id, "photo": null}));
                }
                run(app, ctx, "print.cell.photo", json!({"id": target.id, "photo": drag.photo_id}));
                pu.selected = Some(target.id);
            }
        } else if let Some(src_id) = drag.from_cell {
            let over_pile = pu.pile_rect.is_some_and(|r| r.expand(16.0).contains(drag.pos));
            let below_sheet = drag.pos.y > sheet.page.bottom() + 12.0;
            if over_pile || below_sheet {
                run(app, ctx, "print.cell.photo", json!({"id": src_id, "photo": null}));
            }
        }
    } else {
        pu.photo_drag = Some(drag);
        ctx.request_repaint();
    }
}

fn press_origin(ui: &Ui, resp: &egui::Response) -> Pos2 {
    ui.input(|i| i.pointer.press_origin()).or_else(|| resp.interact_pointer_pos()).unwrap_or_default()
}

/// Draws and handles every cell of a package page, back to front.
#[allow(clippy::too_many_arguments)]
fn package_cells(
    app: &mut LightcraftApp,
    ui: &mut Ui,
    painter: &egui::Painter,
    t: &Tokens,
    pu: &mut PrintUi,
    st: &PrintSettings,
    ids: &[PhotoId],
    sheet: &Sheet,
) {
    let ctx = ui.ctx().clone();
    let first = ids.first().copied();
    let cells: Vec<layout::Cell> = st.layout.cells_on_page(pu.page as u32).cloned().collect();
    for c in &cells {
        // En Modo Photobook cada celda es un Marco Magnético independiente: si `c.photo` es `None`,
        // se muestra como un marco vacío listo para recibir una foto de la Pila, sin repetir `first`.
        let photo = if st.book.enabled {
            c.photo.map(PhotoId)
        } else if st.layout.style == Style::CustomPackage {
            c.photo.map(PhotoId).or(first)
        } else {
            first
        };
        let rect = sheet.rect(&c.rect);
        let in_pan_mode = pu.pan_mode_cell == Some(c.id);
        paint_cell(app, painter, t, st, rect, photo, sheet, &c.rect, Some(c));
        let resp = ui.interact(rect, egui::Id::new(("print-cell", c.id)), Sense::click_and_drag());
        register(&ctx, format!("print:cell:{}", c.id), rect);
        if resp.double_clicked() && photo.is_some() {
            // Doble clic sobre cualquier marco: activa/desactiva el Modo Encuadre Interno para mover la foto libremente dentro del marco
            pu.selected = Some(c.id);
            pu.photo_drag = None;
            pu.drag = None;
            pu.pan_mode_cell = if in_pan_mode { None } else { Some(c.id) };
        } else if resp.clicked() || resp.drag_started() {
            pu.selected = Some(c.id);
            if pu.pan_mode_cell.is_some() && pu.pan_mode_cell != Some(c.id) {
                pu.pan_mode_cell = None;
            }
        }
        let in_pan_mode = pu.pan_mode_cell == Some(c.id);
        if in_pan_mode && resp.hovered() {
            ctx.set_cursor_icon(if resp.dragged() { egui::CursorIcon::Grabbing } else { egui::CursorIcon::Grab });
        }
        // Zoom individual dentro de cada celda con rueda del ratón (con Alt/Ctrl, en Modo Photobook o en Modo Encuadre)
        if resp.hovered() && (in_pan_mode || st.book.enabled || ui.input(|i| i.modifiers.alt || i.modifiers.command || i.modifiers.ctrl)) {
            let scroll_y = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll_y.abs() > 0.5 {
                let new_zoom = (c.zoom + f64::from(scroll_y) * 0.005).clamp(1.0, 5.0);
                run(app, &ctx, "print.cell.panZoom", json!({"id": c.id, "zoom": new_zoom}));
            }
        }
        let alt_or_right = ui.input(|i| i.modifiers.alt || i.pointer.button_down(egui::PointerButton::Secondary));
        let shift_held = ui.input(|i| i.modifiers.shift);
        if resp.drag_started() {
            if in_pan_mode || alt_or_right {
                pu.pan_drag = Some((c.id, c.pan_x, c.pan_y, press_origin(ui, &resp)));
            } else if st.book.enabled && !shift_held && let Some(pid) = c.photo {
                // En Modo Photobook, arrastrar una foto (sin estar en Modo Encuadre) la lleva a la Pila o a otro marco
                let pos = resp.interact_pointer_pos().unwrap_or(rect.center());
                pu.photo_drag = Some(PhotoDrag {
                    photo_id: pid,
                    from_cell: Some(c.id),
                    pos,
                });
            } else {
                pu.drag = Some(Drag { cell: c.id, origin: c.rect, resize: false, press: press_origin(ui, &resp), total: Vec2::ZERO });
            }
        }
        if resp.dragged() {
            if let Some((cid, ox, oy, press)) = pu.pan_drag.filter(|(cid, ..)| *cid == c.id) {
                let delta = resp.interact_pointer_pos().map_or(Vec2::ZERO, |p| p - press);
                // Calcular el desborde real (overflow) de la foto respecto al marco para que se mueva 1:1 con el cursor
                let (nw, nh) = photo.and_then(|id| app.session.catalog.photo(id)).map_or((3.0, 2.0), |p| {
                    let (w, h) = lightcraft_engine::export::output_size(p, &ExportOptions::default());
                    (w as f64, h as f64)
                });
                let border = if st.layout.style.is_package() && st.image.border_enabled {
                    (st.image.border_width as f32 * sheet.scale).min(rect.width().min(rect.height()).max(2.0) / 2.0 - 1.0).max(0.0)
                } else {
                    0.0
                };
                let avail = rect.shrink(border);
                let (ov_x, ov_y) = place(
                    nw,
                    nh,
                    (f64::from(avail.width()) * 8.0).round() as usize,
                    (f64::from(avail.height()) * 8.0).round() as usize,
                    st.image.zoom_to_fill,
                    st.image.rotate_to_fit,
                )
                .map_or((rect.width() * 0.5, rect.height() * 0.5), |pl| {
                    let base = vec2(pl.w as f32, pl.h as f32) / 8.0;
                    let zoomed = base * (c.zoom.clamp(1.0, 5.0) as f32);
                    ((zoomed.x - avail.width()).max(0.0), (zoomed.y - avail.height()).max(0.0))
                });
                let half_ov_x = (ov_x * 0.5).max(avail.width() * 0.175).max(16.0);
                let half_ov_y = (ov_y * 0.5).max(avail.height() * 0.175).max(16.0);
                let nx = (ox + f64::from(delta.x / half_ov_x)).clamp(-1.0, 1.0);
                let ny = (oy + f64::from(delta.y / half_ov_y)).clamp(-1.0, 1.0);
                run(app, &ctx, "print.cell.panZoom", json!({"id": cid, "panX": nx, "panY": ny}));
            } else if let Some(d) = pu.drag.as_mut().filter(|d| d.cell == c.id && !d.resize) {
                d.total = resp.interact_pointer_pos().map_or(d.total, |p| p - d.press);
                let (x, y) = (d.origin.x + f64::from(d.total.x / sheet.scale), d.origin.y + f64::from(d.total.y / sheet.scale));
                let threshold = f64::from(SNAP_PX / sheet.scale);
                run(app, &ctx, "print.cell.move", json!({"id": c.id, "x": x, "y": y, "snap": true, "threshold": threshold}));
            }
        }
        if resp.drag_stopped() {
            pu.drag = None;
            pu.pan_drag = None;
        }
        let id = c.id;
        let has_photo = c.photo.is_some();
        crate::widgets::context_menu(&resp, |ui| {
            if has_photo {
                let toggle_label = if pu.pan_mode_cell == Some(id) {
                    "✅ Terminar de Encuadrar Foto"
                } else {
                    "✋ Acomodar / Encuadrar Foto dentro del Marco (Doble Clic)"
                };
                if ui.button(toggle_label).clicked() {
                    pu.pan_mode_cell = if pu.pan_mode_cell == Some(id) { None } else { Some(id) };
                    ui.close();
                }
                if ui.button("🎯 Centrar Foto en el Marco").clicked() {
                    run(app, &ctx, "print.cell.panZoom", json!({"id": id, "panX": 0.0, "panY": 0.0}));
                    ui.close();
                }
            }
            if st.book.enabled && has_photo {
                if ui.button("📥 Enviar foto a la Pila de Fotos").clicked() {
                    run(app, &ctx, "print.cell.photo", json!({"id": id, "photo": null}));
                    ui.close();
                }
                ui.separator();
            }
            for (how, label) in [
                ("toFront", "Traer al frente"),
                ("forward", "Enviar adelante"),
                ("backward", "Enviar atrás"),
                ("toBack", "Enviar al fondo"),
            ] {
                if ui.button(label).clicked() {
                    run(app, &ctx, "print.cell.stack", json!({"id": id, "order": how}));
                    ui.close();
                }
            }
            ui.separator();
            if ui.button("Restablecer Encuadre (Pan & Zoom 1.0×)").clicked() {
                run(app, &ctx, "print.cell.panZoom", json!({"id": id, "zoom": 1.0, "panX": 0.0, "panY": 0.0}));
                ui.close();
            }
            if ui.button("Eliminar Marco").clicked() {
                run(app, &ctx, "print.cell.delete", json!({"id": id}));
                ui.close();
            }
        });
    }
    // the selected cell's outline and resize handle go over everything
    let Some(sel) = pu.selected.and_then(|s| cells.iter().find(|c| c.id == s)) else { return };
    let rect = sheet.rect(&sel.rect);
    let in_pan_mode = pu.pan_mode_cell == Some(sel.id);
    if in_pan_mode {
        // Modo Encuadre Interno activo (doble clic): mostrar vista fantasma de toda la foto fuera del marco,
        // cuadrícula de tercios, mira central (+) y borde verde-cian brillante.
        let sel_photo = if st.book.enabled {
            sel.photo.map(PhotoId)
        } else if st.layout.style == Style::CustomPackage {
            sel.photo.map(PhotoId).or(first)
        } else {
            first
        };
        if let Some(pid) = sel_photo {
            let (nw, nh) = app.session.catalog.photo(pid).map_or((3.0, 2.0), |p| {
                let (w, h) = lightcraft_engine::export::output_size(p, &ExportOptions::default());
                (w as f64, h as f64)
            });
            let border = if st.layout.style.is_package() && st.image.border_enabled {
                (st.image.border_width as f32 * sheet.scale).min(rect.width().min(rect.height()).max(2.0) / 2.0 - 1.0).max(0.0)
            } else {
                0.0
            };
            let avail = rect.shrink(border);
            if let Some(p) = place(
                nw,
                nh,
                (f64::from(avail.width()) * 8.0).round() as usize,
                (f64::from(avail.height()) * 8.0).round() as usize,
                st.image.zoom_to_fill,
                st.image.rotate_to_fit,
            ) {
                let zoom = effective_cell_zoom(sel.zoom, sel.pan_x, sel.pan_y, p.w as f32 / 8.0, p.h as f32 / 8.0, avail.width(), avail.height(), st.image.zoom_to_fill);
                let base_size = vec2(p.w as f32, p.h as f32) / 8.0;
                let zoomed_size = base_size * zoom;
                let ov_x = (zoomed_size.x - avail.width()).max(0.0);
                let ov_y = (zoomed_size.y - avail.height()).max(0.0);
                let pan_offset = vec2(ov_x * 0.5 * (sel.pan_x as f32), ov_y * 0.5 * (sel.pan_y as f32));
                let shown = Rect::from_center_size(avail.center() + pan_offset, zoomed_size);
                let page_clip = painter.with_clip_rect(sheet.page.intersect(painter.clip_rect()));
                if let Some(tex) = app.renderer.thumb(pid) {
                    let mut mesh = egui::Mesh::with_texture(tex.tex.id());
                    let uv: [Pos2; 4] = if p.rotated {
                        [pos2(0.0, 1.0), pos2(0.0, 0.0), pos2(1.0, 0.0), pos2(1.0, 1.0)]
                    } else {
                        [pos2(0.0, 0.0), pos2(1.0, 0.0), pos2(1.0, 1.0), pos2(0.0, 1.0)]
                    };
                    let corners = [shown.left_top(), shown.right_top(), shown.right_bottom(), shown.left_bottom()];
                    let ghost_col = Color32::from_rgba_unmultiplied(95, 95, 95, 95);
                    for (pos, uv) in corners.into_iter().zip(uv) {
                        mesh.vertices.push(egui::epaint::Vertex { pos, uv, color: ghost_col });
                    }
                    mesh.indices.extend([0, 1, 2, 0, 2, 3]);
                    page_clip.add(egui::Shape::mesh(mesh));
                }
                dashed(&page_clip, shown, Stroke::new(1.2, Color32::from_rgb(0, 225, 185)));
                // Redibujar la celda nítida al 100% dentro del marco
                paint_cell(app, painter, t, st, rect, sel_photo, sheet, &sel.rect, Some(sel));
            }
        }
        let pan_col = Color32::from_rgb(0, 230, 180);
        let grid_stroke = Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 255, 255, 130));
        for k in 1..=2 {
            let fx = rect.left() + rect.width() * (k as f32 / 3.0);
            let fy = rect.top() + rect.height() * (k as f32 / 3.0);
            painter.line_segment([pos2(fx, rect.top()), pos2(fx, rect.bottom())], grid_stroke);
            painter.line_segment([pos2(rect.left(), fy), pos2(rect.right(), fy)], grid_stroke);
        }
        // Mira central (+) para centrar rostros/novios con precisión
        let cpos = rect.center();
        painter.line_segment([cpos - vec2(8.0, 0.0), cpos + vec2(8.0, 0.0)], Stroke::new(1.6, pan_col));
        painter.line_segment([cpos - vec2(0.0, 8.0), cpos + vec2(0.0, 8.0)], Stroke::new(1.6, pan_col));
        painter.rect_stroke(rect, 0.0, Stroke::new(2.5, pan_col), StrokeKind::Outside);
        let badge = "✋ ENCUADRANDO FOTO — Arrastra para acomodar · Rueda: Zoom · Doble clic para fijar";
        let galley = painter.layout_no_wrap(badge.to_string(), t.semibold(10.0), Color32::BLACK);
        let bpos = pos2(
            (rect.center().x - galley.size().x / 2.0).max(sheet.page.left() + 4.0),
            (rect.top() - galley.size().y - 6.0).max(sheet.page.top() + 4.0),
        );
        painter.rect_filled(Rect::from_min_size(bpos, galley.size()).expand(4.0), 4.0, pan_col);
        painter.galley(bpos, galley, Color32::BLACK);
    } else {
        painter.rect_stroke(rect, 0.0, Stroke::new(2.0, t.accent), StrokeKind::Outside);
    }
    let handle = Rect::from_center_size(rect.right_bottom(), Vec2::splat(HANDLE));
    painter.rect_filled(handle, 2.0, t.accent);
    let resp = ui.interact(handle, egui::Id::new(("print-resize", sel.id)), Sense::drag());
    register(&ctx, format!("print:resize:{}", sel.id), handle);
    if resp.drag_started() {
        pu.drag = Some(Drag { cell: sel.id, origin: sel.rect, resize: true, press: press_origin(ui, &resp), total: Vec2::ZERO });
    }
    if resp.dragged()
        && let Some(d) = pu.drag.as_mut().filter(|d| d.cell == sel.id && d.resize)
    {
        d.total = resp.interact_pointer_pos().map_or(d.total, |p| p - d.press);
        let (w, h) = (d.origin.w + f64::from(d.total.x / sheet.scale), d.origin.h + f64::from(d.total.y / sheet.scale));
        let threshold = f64::from(SNAP_PX / sheet.scale);
        run(
            app,
            &ctx,
            "print.cell.resize",
            json!({"id": sel.id, "width": w.max(layout::MIN_CELL_IN), "height": h.max(layout::MIN_CELL_IN), "snap": true, "threshold": threshold}),
        );
    }
    if resp.drag_stopped() {
        pu.drag = None;
    }
}

/// The selected cell's numbers, its place in the stack and its photo.
fn inspector(app: &mut LightcraftApp, ui: &mut Ui, t: &Tokens, pu: &mut PrintUi, st: &PrintSettings) {
    let ctx = ui.ctx().clone();
    let Some(cell) = pu.selected.and_then(|id| st.layout.cell(id)).cloned() else {
        let hint = if st.book.enabled {
            "Doble clic en un marco para encuadrar su foto · Arrastra para mover a la Pila o cambiar de marco"
        } else {
            "Haz clic en un marco para moverlo, redimensionarlo o acomodarlo"
        };
        ui.label(egui::RichText::new(hint).color(t.text_dim));
        return;
    };
    let (id, r, unit) = (cell.id, cell.rect, st.guides.units);
    if pu.caption_cell != Some(id) {
        pu.caption_cell = Some(id);
        pu.caption_draft = cell.caption.clone().unwrap_or_default();
    }
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        let in_pan = pu.pan_mode_cell == Some(id);
        let pan_btn_label = if in_pan { "✅ Fijar Encuadre" } else { "✋ Encuadrar (Doble Clic)" };
        if text_button(ui, "printCellPanMode", pan_btn_label, in_pan).clicked() {
            pu.pan_mode_cell = if in_pan { None } else { Some(id) };
        }
        if (cell.pan_x.abs() > 0.005 || cell.pan_y.abs() > 0.005 || (cell.zoom - 1.0).abs() > 0.005)
            && text_button(ui, "printCellCenterPhoto", "🎯 Centrar", false).clicked()
        {
            run(app, &ctx, "print.cell.panZoom", json!({"id": id, "panX": 0.0, "panY": 0.0, "zoom": 1.0}));
        }
        if let Some(v) = length(ui, "printCellX", r.x, unit, 0.0, 60.0) {
            run(app, &ctx, "print.cell.move", json!({"id": id, "x": v, "y": r.y}));
        }
        if let Some(v) = length(ui, "printCellY", r.y, unit, 0.0, 60.0) {
            run(app, &ctx, "print.cell.move", json!({"id": id, "x": r.x, "y": v}));
        }
        if let Some(v) = length(ui, "printCellW", r.w, unit, layout::MIN_CELL_IN, 60.0) {
            run(app, &ctx, "print.cell.resize", json!({"id": id, "width": v, "height": r.h}));
        }
        if let Some(v) = length(ui, "printCellH", r.h, unit, layout::MIN_CELL_IN, 60.0) {
            run(app, &ctx, "print.cell.resize", json!({"id": id, "width": r.w, "height": v}));
        }
        for (i, (how, label)) in [("toBack", "Fondo"), ("backward", "Atrás"), ("forward", "Adelante"), ("toFront", "Frente")].into_iter().enumerate()
        {
            if text_button(ui, &format!("printStack-{i}"), label, false).clicked() {
                run(app, &ctx, "print.cell.stack", json!({"id": id, "order": how}));
            }
        }
        if st.layout.style == Style::CustomPackage {
            if text_button(ui, "printCellPhoto", "Usar Foto Activa", false).clicked() {
                let active = app.session.active().map(|p| p.0);
                run(app, &ctx, "print.cell.photo", json!({"id": id, "photo": active}));
            }
            if cell.photo.is_some() && text_button(ui, "printCellPhotoClear", "📥 Enviar a la Pila", false).clicked() {
                run(app, &ctx, "print.cell.photo", json!({"id": id, "photo": null}));
            }
        }
        if text_button(ui, "printCellDelete", "Borrar Marco", false).clicked() {
            run(app, &ctx, "print.cell.delete", json!({"id": id}));
            pu.selected = None;
        }
        // Controles de Pan & Zoom individual y Texto en celda
        let mut z = cell.zoom;
        let rz = ui.add(egui::DragValue::new(&mut z).range(1.0..=5.0).speed(0.03).max_decimals(2).prefix("Zoom ").suffix("×"));
        register(ui.ctx(), "drag:printCellZoom", rz.rect);
        if rz.changed() {
            run(app, &ctx, "print.cell.panZoom", json!({"id": id, "zoom": z}));
        }
        let mut px = cell.pan_x;
        let rpx = ui.add(egui::DragValue::new(&mut px).range(-1.0..=1.0).speed(0.02).max_decimals(2).prefix("Pan X "));
        register(ui.ctx(), "drag:printCellPanX", rpx.rect);
        if rpx.changed() {
            run(app, &ctx, "print.cell.panZoom", json!({"id": id, "panX": px}));
        }
        let mut py = cell.pan_y;
        let rpy = ui.add(egui::DragValue::new(&mut py).range(-1.0..=1.0).speed(0.02).max_decimals(2).prefix("Pan Y "));
        register(ui.ctx(), "drag:printCellPanY", rpy.rect);
        if rpy.changed() {
            run(app, &ctx, "print.cell.panZoom", json!({"id": id, "panY": py}));
        }
        if st.book.enabled {
            let cap_resp = crate::text_field::TextField::singleline("field:printCellCaption", &mut pu.caption_draft)
                .hint("Texto / Pie de foto…")
                .width(120.0)
                .show(ui);
            if cap_resp.response.changed() {
                run(app, &ctx, "print.cell.panZoom", json!({"id": id, "caption": pu.caption_draft}));
            }
        }
    });
}

fn effective_cell_zoom(raw_zoom: f64, pan_x: f64, pan_y: f64, base_w: f32, base_h: f32, avail_w: f32, avail_h: f32, fill: bool) -> f32 {
    let mut z = raw_zoom.clamp(1.0, 5.0) as f32;
    let px = pan_x.clamp(-1.0, 1.0).abs() as f32;
    let py = pan_y.clamp(-1.0, 1.0).abs() as f32;
    if fill {
        if px > 1e-3 && base_w * z < avail_w * (1.0 + px * 0.35) {
            z = z.max((avail_w * (1.0 + px * 0.35)) / base_w.max(1.0));
        }
        if py > 1e-3 && base_h * z < avail_h * (1.0 + py * 0.35) {
            z = z.max((avail_h * (1.0 + py * 0.35)) / base_h.max(1.0));
        }
        z = z.clamp(1.0, 5.0);
    }
    z
}

/// One cell on screen: the photo as it will print (fit or fill, turned, border, stroke, individual pan/zoom and caption).
#[allow(clippy::too_many_arguments)]
fn paint_cell(
    app: &mut LightcraftApp,
    painter: &egui::Painter,
    t: &Tokens,
    st: &PrintSettings,
    cell: Rect,
    photo: Option<PhotoId>,
    sheet: &Sheet,
    inches: &layout::Rect,
    cell_meta: Option<&layout::Cell>,
) -> Option<Rect> {
    let im = &st.image;
    if st.guides.show_cells {
        painter.rect_stroke(cell, 0.0, Stroke::new(1.0, Color32::from_rgba_unmultiplied(120, 150, 190, 150)), StrokeKind::Inside);
    }
    let border = if st.layout.style.is_package() && im.border_enabled {
        (im.border_width as f32 * sheet.scale).min(cell.width().min(cell.height()).max(2.0) / 2.0 - 1.0).max(0.0)
    } else {
        0.0
    };
    let avail = cell.shrink(border);
    if avail.width() < 1.0 || avail.height() < 1.0 {
        return None;
    }
    // the photo's shape: its cropped size, or 3:2 for a placeholder
    let (nw, nh) = photo.and_then(|id| app.session.catalog.photo(id)).map_or((3.0, 2.0), |p| {
        let (w, h) = lightcraft_engine::export::output_size(p, &ExportOptions::default());
        (w as f64, h as f64)
    });
    let scale = 8.0; // sub-point resolution for the rounding in `place`
    let aw = (f64::from(avail.width()) * scale).round() as usize;
    let ah = (f64::from(avail.height()) * scale).round() as usize;
    let p = place(nw, nh, aw, ah, im.zoom_to_fill, im.rotate_to_fit)?;
    let raw_zoom = cell_meta.map_or(1.0, |c| c.zoom);
    let raw_pan_x = cell_meta.map_or(0.0, |c| c.pan_x);
    let raw_pan_y = cell_meta.map_or(0.0, |c| c.pan_y);
    let base_size = vec2(p.w as f32, p.h as f32) / scale as f32;
    let zoom = effective_cell_zoom(raw_zoom, raw_pan_x, raw_pan_y, base_size.x, base_size.y, avail.width(), avail.height(), im.zoom_to_fill);
    let pan_x = raw_pan_x.clamp(-1.0, 1.0) as f32;
    let pan_y = raw_pan_y.clamp(-1.0, 1.0) as f32;
    let zoomed_size = base_size * zoom;
    let overflow_x = (zoomed_size.x - avail.width()).max(0.0);
    let overflow_y = (zoomed_size.y - avail.height()).max(0.0);
    let pan_offset = vec2(overflow_x * 0.5 * pan_x, overflow_y * 0.5 * pan_y);
    let shown = Rect::from_center_size(avail.center() + pan_offset, zoomed_size);
    let visible = shown.intersect(avail);
    if border > 0.0 {
        let frame = visible.expand(border);
        let c = im.border_color;
        painter.rect_filled(frame.intersect(cell), 0.0, Color32::from_rgb(c[0], c[1], c[2]));
    }
    let clip = painter.with_clip_rect(avail.intersect(painter.clip_rect()));
    let tex = photo.and_then(|id| {
        // Use the library's thumbnail when there is one. A thumbnail is kept at one size, so asking
        // for another size here every frame made the preview and the grid behind the dialog
        // re-render each other's thumbnail for as long as the dialog stayed open.
        if app.renderer.thumb(id).is_none() {
            let px = (shown.width().max(shown.height()) * 2.0).clamp(64.0, 1024.0) as usize;
            crate::panels::grid::request_thumb(app, id, px, 10);
        }
        app.renderer.thumb(id).map(|tex| tex.tex.id())
    });
    match tex {
        Some(id) => {
            let mut mesh = egui::Mesh::with_texture(id);
            // an unturned quad maps the texture straight; a turned one (a quarter clockwise) takes
            // the left edge to the top
            let uv: [Pos2; 4] = if p.rotated {
                [pos2(0.0, 1.0), pos2(0.0, 0.0), pos2(1.0, 0.0), pos2(1.0, 1.0)]
            } else {
                [pos2(0.0, 0.0), pos2(1.0, 0.0), pos2(1.0, 1.0), pos2(0.0, 1.0)]
            };
            let corners = [shown.left_top(), shown.right_top(), shown.right_bottom(), shown.left_bottom()];
            for (pos, uv) in corners.into_iter().zip(uv) {
                mesh.vertices.push(egui::epaint::Vertex { pos, uv, color: Color32::WHITE });
            }
            mesh.indices.extend([0, 1, 2, 0, 2, 3]);
            clip.add(egui::Shape::mesh(mesh));
        }
        None => {
            if st.book.enabled && photo.is_none() {
                clip.rect_filled(avail, 2.0, Color32::from_rgb(235, 238, 245));
                dashed(&clip, avail.shrink(2.0), Stroke::new(1.5, Color32::from_rgb(90, 145, 220)));
                clip.text(
                    avail.center(),
                    Align2::CENTER_CENTER,
                    "🧲 Marco Magnético Vacío\nArrastra una foto aquí",
                    t.semibold(11.0),
                    Color32::from_rgb(60, 95, 150),
                );
            } else {
                clip.rect_filled(shown, 0.0, if photo.is_some() { Color32::from_gray(52) } else { Color32::from_gray(200) });
            }
        }
    }
    if im.stroke_enabled && im.stroke_width_pt > 0.0 {
        let w = (im.stroke_width_pt as f32 / 72.0 * sheet.scale).max(1.0).min(visible.width().min(visible.height()) / 2.0);
        let c = im.stroke_color;
        painter.rect_stroke(visible, 0.0, Stroke::new(w, Color32::from_rgb(c[0], c[1], c[2])), StrokeKind::Inside);
    }
    if let Some(cap) = cell_meta.and_then(|c| c.caption.as_deref()).map(str::trim).filter(|s| !s.is_empty()) {
        let galley = painter.layout_no_wrap(cap.to_string(), t.semibold(11.0), Color32::WHITE);
        let at = pos2(cell.center().x - galley.size().x / 2.0, cell.bottom() - galley.size().y - 16.0);
        painter.rect_filled(Rect::from_min_size(at, galley.size()).expand(3.0), 3.0, Color32::from_black_alpha(185));
        painter.galley(at, galley, Color32::WHITE);
    }
    if st.guides.show_dimensions {
        let text = format!("{} × {}", fmt_len(inches.w, st.guides.units), fmt_len(inches.h, st.guides.units));
        let galley = painter.layout_no_wrap(text, t.font(10.0), Color32::WHITE);
        let at = pos2(cell.center().x - galley.size().x / 2.0, cell.bottom() - galley.size().y - 3.0);
        painter.rect_filled(Rect::from_min_size(at, galley.size()).expand(2.0), 2.0, Color32::from_black_alpha(160));
        painter.galley(at, galley, Color32::WHITE);
    }
    Some(visible)
}

/// The Page Options as the print draws them ([`compose`]): crop marks at the corners of each
/// photo, the page number at the bottom right and the print settings at the bottom left.
/// Packages draw none of them, as the print doesn't ([`compose::has_page_options`]).
fn page_options(painter: &egui::Painter, st: &PrintSettings, sheet: &Sheet, photos: &[Rect], number: usize) {
    if !compose::has_page_options(st.layout.style) {
        return;
    }
    let p = &st.page;
    let ink = compose::mark_ink(p.background);
    let ink = Color32::from_rgb(ink[0], ink[1], ink[2]);
    if p.crop_marks {
        let (len, gap) = (compose::CROP_MARK_IN as f32 * sheet.scale, compose::CROP_MARK_GAP_IN as f32 * sheet.scale);
        let stroke = Stroke::new(1.0, ink);
        for r in photos {
            for (corner, dx, dy) in
                [(r.left_top(), -1.0, -1.0), (r.right_top(), 1.0, -1.0), (r.left_bottom(), -1.0, 1.0), (r.right_bottom(), 1.0, 1.0)]
            {
                painter.line_segment([corner + vec2(dx * gap, 0.0), corner + vec2(dx * (gap + len), 0.0)], stroke);
                painter.line_segment([corner + vec2(0.0, dy * gap), corner + vec2(0.0, dy * (gap + len))], stroke);
            }
        }
    }
    let Ok(area) = st.layout.printable() else { return };
    let area = sheet.rect(&area);
    let font = egui::FontId::proportional((compose::PAGE_TEXT_PT as f32 / 72.0 * sheet.scale).max(4.0));
    let y = (area.bottom() + sheet.page.bottom()) / 2.0;
    if p.page_numbers {
        painter.text(pos2(area.right(), y), Align2::RIGHT_CENTER, number.to_string(), font.clone(), ink);
    }
    if p.page_info {
        painter.text(pos2(area.left(), y), Align2::LEFT_CENTER, compose::page_info_text(st), font, ink);
    }
}

fn dashed(painter: &egui::Painter, r: Rect, stroke: Stroke) {
    let pts = [r.left_top(), r.right_top(), r.right_bottom(), r.left_bottom(), r.left_top()];
    painter.extend(egui::Shape::dashed_line(&pts, stroke, 5.0, 4.0));
}

/// Rulers along the top and left of the preview, in `unit`, zero at the page's corner.
fn rulers(painter: &egui::Painter, t: &Tokens, sheet: &Sheet, area: Rect, pw: f64, ph: f64, unit: Unit) {
    let (major, minor) = match unit {
        Unit::Inches => (1.0, 0.25),
        Unit::Centimeters => (1.0, 0.5),
        Unit::Millimeters => (10.0, 5.0),
        Unit::Points => (72.0, 18.0),
    };
    let tick = Stroke::new(1.0, t.text_dim);
    let ink = t.text_dim;
    let top = Rect::from_min_max(area.min, pos2(area.max.x, area.min.y + 20.0));
    let left = Rect::from_min_max(area.min, pos2(area.min.x + 20.0, area.max.y));
    painter.rect_filled(top, 0.0, t.chrome);
    painter.rect_filled(left, 0.0, t.chrome);
    let ticks = |len_in: f64| (0..).map(move |k| f64::from(k) * minor).take_while(move |v| *v <= unit.from_inches(len_in) + 1e-9).take(2000);
    for v in ticks(pw) {
        let x = sheet.page.left() + unit.to_inches(v) as f32 * sheet.scale;
        let is_major = (v / major - (v / major).round()).abs() < 1e-9;
        painter.line_segment([pos2(x, top.bottom()), pos2(x, top.bottom() - if is_major { 10.0 } else { 5.0 })], tick);
        if is_major {
            painter.text(pos2(x + 2.0, top.top() + 1.0), Align2::LEFT_TOP, format!("{}", v.round() as i64), t.font(9.0), ink);
        }
    }
    for v in ticks(ph) {
        let y = sheet.page.top() + unit.to_inches(v) as f32 * sheet.scale;
        let is_major = (v / major - (v / major).round()).abs() < 1e-9;
        painter.line_segment([pos2(left.right(), y), pos2(left.right() - if is_major { 10.0 } else { 5.0 }, y)], tick);
        if is_major {
            painter.text(pos2(left.left() + 1.0, y + 1.0), Align2::LEFT_TOP, format!("{}", v.round() as i64), t.font(9.0), ink);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lengths_read_in_the_ruler_unit() {
        assert_eq!(fmt_len(1.0, Unit::Inches), "1.00 in");
        assert_eq!(fmt_len(1.0, Unit::Millimeters), "25.4 mm");
        assert_eq!(fmt_len(1.0, Unit::Points), "72 pt");
        assert_eq!(trim(4.0), "4");
        assert_eq!(trim(2.5), "2.5");
        assert_eq!(paper_label("a4"), "A4");
        assert_eq!(paper_label("4x6"), "4 × 6 in");
    }
}
