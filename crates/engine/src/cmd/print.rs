//! `print.*`: the Print module as commands.
//!
//! The module's state is [`Session::print`] (layout, image settings, page, guides, job), so the
//! app, CLI, control channel and MCP share one print. Lengths are inches in page space (origin at
//! the top-left of the page as printed); cell ids and photo ids are numbers. Nothing here touches
//! the catalog, so none of it is journaled or undoable; saved templates live in prefs.json.
//!
//! * `print.get`, `print.set {layout?, image?, page?, guides?, job?}`, `print.reset`
//! * `print.cell.add|move|resize|delete|stack|photo|page`, `print.layout.autoLayout|newPage|clear`
//! * `print.exportPdf {path, ids?}`
//! * `print.template.list|save|apply|delete`
use serde_json::{Value, json};

use super::{CommandSpec, always, bad, cmd, f64_req};
use crate::print::layout::{Layout, MAX_PAGES, Rect, Snap, Stack, paper_names, paper_size};
use crate::print::template::{MAX_TEMPLATES, PrintSettings, PrintTemplate, clean_name};
use crate::{Result, Session};

fn rect_json(r: &Rect) -> Value {
    json!({"x": r.x, "y": r.y, "w": r.w, "h": r.h})
}

/// The state, with what a front end needs to draw it (page size, printable area, grid cells).
fn state(s: &Session, photos: usize) -> Value {
    let l = &s.print.layout;
    let mut out = json!({
        "settings": s.print,
        "papers": paper_names().filter_map(|n| paper_size(n).map(|(w, h)| json!({"name": n, "width": w, "height": h}))).collect::<Vec<_>>(),
        "overlaps": l.has_overlaps(),
        "photos": photos,
    });
    if let (Ok((w, h)), Ok(area), Some(o)) = (l.page_size(), l.printable(), out.as_object_mut()) {
        o.insert("pageWidth".into(), json!(w));
        o.insert("pageHeight".into(), json!(h));
        o.insert("printable".into(), rect_json(&area));
        o.insert("pageCount".into(), json!(l.page_count(photos)));
        o.insert("cellsPerPage".into(), json!(l.cells_per_page()));
        if !l.style.is_package() {
            o.insert("gridCells".into(), json!(l.grid_rects().unwrap_or_default().iter().map(rect_json).collect::<Vec<_>>()));
        }
    }
    out
}

fn reply(s: &Session) -> Result<Value> {
    Ok(state(s, s.targets(&Value::Null)?.len()))
}

/// Applies `patch` over `dst`: objects merge key by key, anything else replaces.
fn merge(dst: &mut Value, patch: &Value, depth: usize) {
    match (dst, patch) {
        (Value::Object(d), Value::Object(p)) if depth < 8 => {
            for (k, v) in p {
                match d.get_mut(k) {
                    Some(slot) => merge(slot, v, depth + 1),
                    None => {
                        d.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        (d, p) => *d = p.clone(),
    }
}

fn get(s: &mut Session, p: &Value) -> Result<Value> {
    Ok(state(s, s.targets(p)?.len()))
}

fn set(s: &mut Session, p: &Value) -> Result<Value> {
    const ID: &str = "print.set";
    let patch = p.as_object().ok_or_else(|| bad(ID, "params must be an object"))?;
    if let Some(k) = patch.keys().find(|k| !matches!(k.as_str(), "layout" | "image" | "page" | "guides" | "job" | "book")) {
        return Err(bad(ID, format!("unknown parameter `{k}` (one of layout, image, page, guides, job, book)")));
    }
    if let Some(l) = patch.get("layout").and_then(Value::as_object)
        && let Some(k) = l.keys().find(|k| matches!(k.as_str(), "cells" | "pages" | "nextId"))
    {
        return Err(bad(ID, format!("`layout.{k}` is changed with the print.cell.* and print.layout.* commands")));
    }
    let mut value = serde_json::to_value(&s.print).map_err(|e| bad(ID, e.to_string()))?;
    merge(&mut value, p, 0);
    let mut next: PrintSettings = serde_json::from_value(value).map_err(|e| bad(ID, e.to_string()))?;
    // A new paper, orientation or margin may leave cells outside the page: pull them back.
    if next.layout.page.printable().is_ok() {
        next.layout.fit_cells().map_err(|e| bad(ID, e))?;
    }
    next.validate().map_err(|e| bad(ID, e))?;
    s.print = next;
    reply(s)
}

fn reset(s: &mut Session, _: &Value) -> Result<Value> {
    s.print = PrintSettings::default();
    reply(s)
}

fn cell_id(p: &Value, id: &str) -> Result<u32> {
    p.get("id").and_then(Value::as_u64).and_then(|v| u32::try_from(v).ok()).ok_or_else(|| bad(id, "missing cell `id`"))
}

fn opt_f64(p: &Value, key: &str, id: &str) -> Result<Option<f64>> {
    match p.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v.as_f64().map(Some).ok_or_else(|| bad(id, format!("`{key}` must be a number"))),
    }
}

/// Snaps `r` (a cell being moved or resized) when the call asks for it.
fn snapped(l: &Layout, guides: &crate::print::layout::Guides, id: u32, r: Rect, p: &Value, resize: bool, cmd: &str) -> Result<Rect> {
    let on = match p.get("snap") {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(_) => return Err(bad(cmd, "`snap` must be true or false")),
    };
    if !on || guides.snap == Snap::Off {
        return Ok(r);
    }
    let threshold = opt_f64(p, "threshold", cmd)?.unwrap_or(0.1);
    let f = if resize { Layout::snap_resize } else { Layout::snap_move };
    f(l, id, r, guides.snap, guides.grid_size, threshold).map_err(|e| bad(cmd, e))
}

fn cell_add(s: &mut Session, p: &Value) -> Result<Value> {
    const ID: &str = "print.cell.add";
    let (w, h) = (f64_req(p, "width", ID)?, f64_req(p, "height", ID)?);
    let mut next = s.print.layout.clone();
    let id = next.add_cell(w, h).map_err(|e| bad(ID, e))?;
    if let (Some(x), Some(y)) = (opt_f64(p, "x", ID)?, opt_f64(p, "y", ID)?) {
        if let Some(page) = p.get("page").and_then(Value::as_u64) {
            let page = u32::try_from(page).map_err(|_| bad(ID, "`page` is too large"))?;
            next.set_cell_page(id, page).map_err(|e| bad(ID, e))?;
        }
        next.move_cell(id, x, y).map_err(|e| bad(ID, e))?;
    }
    if let Some(v) = p.get("photo").filter(|v| !v.is_null()) {
        let photo = v.as_u64().ok_or_else(|| bad(ID, "`photo` must be a photo id"))?;
        next.set_cell_photo(id, Some(photo)).map_err(|e| bad(ID, e))?;
    }
    s.print.layout = next;
    let cell = s.print.layout.cell(id).cloned();
    Ok(json!({"id": id, "cell": cell, "pages": s.print.layout.pages}))
}

fn cell_move(s: &mut Session, p: &Value) -> Result<Value> {
    const ID: &str = "print.cell.move";
    let id = cell_id(p, ID)?;
    let (x, y) = (f64_req(p, "x", ID)?, f64_req(p, "y", ID)?);
    let mut next = s.print.layout.clone();
    let size = next.cell(id).map(|c| c.rect).ok_or_else(|| bad(ID, format!("no cell {id}")))?;
    let target = snapped(&next, &s.print.guides, id, Rect::new(x, y, size.w, size.h), p, false, ID)?;
    let r = next.move_cell(id, target.x, target.y).map_err(|e| bad(ID, e))?;
    s.print.layout = next;
    Ok(json!({"id": id, "rect": r}))
}

fn cell_resize(s: &mut Session, p: &Value) -> Result<Value> {
    const ID: &str = "print.cell.resize";
    let id = cell_id(p, ID)?;
    let mut next = s.print.layout.clone();
    let cur = next.cell(id).map(|c| c.rect).ok_or_else(|| bad(ID, format!("no cell {id}")))?;
    let want = Rect::new(
        opt_f64(p, "x", ID)?.unwrap_or(cur.x),
        opt_f64(p, "y", ID)?.unwrap_or(cur.y),
        opt_f64(p, "width", ID)?.unwrap_or(cur.w),
        opt_f64(p, "height", ID)?.unwrap_or(cur.h),
    );
    let target = snapped(&next, &s.print.guides, id, want, p, true, ID)?;
    let r = next.resize_cell(id, target).map_err(|e| bad(ID, e))?;
    s.print.layout = next;
    Ok(json!({"id": id, "rect": r}))
}

fn cell_delete(s: &mut Session, p: &Value) -> Result<Value> {
    const ID: &str = "print.cell.delete";
    let id = cell_id(p, ID)?;
    s.print.layout.delete_cell(id).map_err(|e| bad(ID, e))?;
    Ok(json!({"deleted": id, "cells": s.print.layout.cells.len()}))
}

fn cell_stack(s: &mut Session, p: &Value) -> Result<Value> {
    const ID: &str = "print.cell.stack";
    let id = cell_id(p, ID)?;
    let how = match p.get("order").and_then(Value::as_str) {
        Some("toBack") => Stack::ToBack,
        Some("backward") => Stack::Backward,
        Some("forward") => Stack::Forward,
        Some("toFront") => Stack::ToFront,
        _ => return Err(bad(ID, "`order` must be toBack, backward, forward or toFront")),
    };
    s.print.layout.restack(id, how).map_err(|e| bad(ID, e))?;
    Ok(json!({"order": s.print.layout.cells.iter().map(|c| c.id).collect::<Vec<_>>()}))
}

fn cell_photo(s: &mut Session, p: &Value) -> Result<Value> {
    const ID: &str = "print.cell.photo";
    let id = cell_id(p, ID)?;
    let photo = match p.get("photo") {
        None | Some(Value::Null) => None,
        Some(v) => {
            let n = v.as_u64().ok_or_else(|| bad(ID, "`photo` must be a photo id or null"))?;
            if s.catalog.photo(lightcraft_catalog::PhotoId(n)).is_none() {
                return Err(bad(ID, format!("no photo {n}")));
            }
            Some(n)
        }
    };
    s.print.layout.set_cell_photo(id, photo).map_err(|e| bad(ID, e))?;
    let _ = s.print.layout.set_cell_pan_zoom(id, Some(0.0), Some(0.0), Some(1.0), None);
    Ok(json!({"id": id, "photo": photo}))
}

fn cell_page(s: &mut Session, p: &Value) -> Result<Value> {
    const ID: &str = "print.cell.page";
    let id = cell_id(p, ID)?;
    let page = p.get("page").and_then(Value::as_u64).and_then(|v| u32::try_from(v).ok()).ok_or_else(|| bad(ID, "missing zero-based `page`"))?;
    s.print.layout.set_cell_page(id, page).map_err(|e| bad(ID, e))?;
    Ok(json!({"id": id, "page": page}))
}

fn cell_pan_zoom(s: &mut Session, p: &Value) -> Result<Value> {
    const ID: &str = "print.cell.panZoom";
    let id = cell_id(p, ID)?;
    let pan_x = opt_f64(p, "panX", ID)?;
    let pan_y = opt_f64(p, "panY", ID)?;
    let zoom = opt_f64(p, "zoom", ID)?;
    let caption = match p.get("caption") {
        None => None,
        Some(Value::Null) => Some(None),
        Some(v) => Some(v.as_str().map(str::to_string)),
    };
    s.print
        .layout
        .set_cell_pan_zoom(id, pan_x, pan_y, zoom, caption)
        .map_err(|e| bad(ID, e))?;
    let cell = s.print.layout.cell(id).cloned();
    Ok(json!({"id": id, "cell": cell}))
}

/// Helper to compute aspect ratio (`width / height`) for a photo ID in the catalog.
fn photo_aspect(s: &Session, id: lightcraft_catalog::PhotoId) -> f64 {
    s.catalog.photo(id).map_or(1.5, |p| {
        let (w, h) = crate::export::output_size(p, &crate::export::ExportOptions::default());
        (w.max(1) as f64) / (h.max(1) as f64)
    })
}

/// Helper to look up detected face regions `[x, y, w, h]` in `[0, 1]` for a photo ID.
pub fn photo_faces_norm(s: &Session, photo_id: u64) -> Vec<[f64; 4]> {
    let Some(photo) = s.catalog.photo(lightcraft_catalog::PhotoId(photo_id)) else {
        return Vec::new();
    };
    photo
        .meta
        .regions
        .iter()
        .filter(|r| r.kind == lightcraft_meta::RegionKind::Face)
        .map(|r| [f64::from(r.rect.x0), f64::from(r.rect.y0), f64::from((r.rect.x1 - r.rect.x0).max(0.01)), f64::from((r.rect.y1 - r.rect.y0).max(0.01))])
        .collect()
}

/// `print.set_format`: Configures standard 300 DPI Photobook format, pre-press guides, binding mode, and hardcover wrap.
fn set_format(s: &mut Session, p: &Value) -> Result<Value> {
    const ID: &str = "print.set_format";
    let mut next = s.print.clone();
    next.book.enabled = p.get("enabled").and_then(Value::as_bool).unwrap_or(true);
    let spread_mode = p.get("spreadMode").and_then(Value::as_bool).unwrap_or(next.book.spread_mode);
    next.book.spread_mode = spread_mode;

    if let Some(bm) = p.get("bindingMode").or_else(|| p.get("mode")) {
        next.book.binding_mode = serde_json::from_value(bm.clone()).map_err(|e| bad(ID, format!("invalid bindingMode: {e}")))?;
    }
    if let Some(hw) = p.get("hardcoverWrap").and_then(Value::as_bool) {
        next.book.hardcover_wrap = hw;
    }
    if let Some(w_mm) = opt_f64(p, "wrapMm", ID)? {
        next.book.wrap_mm = w_mm.clamp(5.0, 50.0);
    }
    if let Some(pt_mm) = opt_f64(p, "paperThicknessMm", ID)? {
        next.book.paper_thickness_mm = pt_mm.clamp(0.05, 3.0);
    }
    if let Some(gap_mm) = opt_f64(p, "gapMm", ID)? {
        next.book.gap_mm = gap_mm.clamp(0.0, 40.0);
        next.layout.spacing = (next.book.gap_mm / 25.4).clamp(0.0, 2.0);
    }
    if let Some(bleed_mm) = opt_f64(p, "bleedMm", ID)? {
        next.guides.bleed_mm = bleed_mm.clamp(0.0, 25.0);
    }
    if let Some(safe_mm) = opt_f64(p, "safeZoneMm", ID)? {
        next.guides.safe_zone_mm = safe_mm.clamp(0.0, 50.0);
    }
    if let Some(gutter_mm) = opt_f64(p, "gutterMm", ID)? {
        next.guides.gutter_mm = gutter_mm.clamp(0.0, 50.0);
    }
    if let Some(pps) = p.get("photosPerSpread").and_then(Value::as_u64) {
        next.book.photos_per_spread = (pps as u8).clamp(1, 8);
    }
    if let Some(u) = p.get("unit").or_else(|| p.get("units")) {
        next.guides.units = serde_json::from_value(u.clone()).map_err(|e| bad(ID, format!("invalid unit: {e}")))?;
    }

    let unit_scale = match p.get("unit").or_else(|| p.get("units")).and_then(Value::as_str) {
        Some("cm") => 1.0 / 2.54,
        Some("mm") => 1.0 / 25.4,
        _ => 1.0,
    };
    let custom_w = opt_f64(p, "width", ID)?.map(|v| v * unit_scale);
    let custom_h = opt_f64(p, "height", ID)?.map(|v| v * unit_scale);

    if let Some(preset) = p.get("format").or_else(|| p.get("preset")).or_else(|| p.get("paper")).and_then(Value::as_str) {
        crate::print::book::apply_format_preset(&mut next, preset, spread_mode, custom_w, custom_h).map_err(|e| bad(ID, e))?;
    } else if custom_w.is_some() || custom_h.is_some() {
        crate::print::book::apply_format_preset(&mut next, "custom", spread_mode, custom_w, custom_h).map_err(|e| bad(ID, e))?;
    } else {
        next.job.ppi = 300.0;
        next.guides.show_bleed = true;
        next.guides.show_trim = true;
        next.guides.show_safe_zone = true;
        next.guides.show_gutter = true;
        next.validate().map_err(|e| bad(ID, e))?;
    }

    if let Some(ppi) = opt_f64(p, "ppi", ID)? {
        next.job.ppi = ppi as f32;
        next.validate().map_err(|e| bad(ID, e))?;
    }

    s.print = next;
    reply(s)
}

/// Applies SmartAlbums BSP / Justified Grid layout to one spread (`spread`) for the given `photo_slots` (`1..=8`).
fn apply_smart_spread_slots(
    s: &mut Session,
    spread: u32,
    photo_slots: &[Option<lightcraft_catalog::PhotoId>],
    variation: usize,
) -> Result<Vec<u32>> {
    const ID: &str = "print.auto_layout_spread";
    let (pw, ph) = s.print.layout.page_size().map_err(|e| bad(ID, e))?;
    let safe_in = (s.print.guides.safe_zone_mm / 25.4).clamp(0.0, pw.min(ph) * 0.22);
    let gutter_in = (s.print.guides.gutter_mm / 25.4).clamp(0.0, pw * 0.20);
    let gap_in = (s.print.book.gap_mm / 25.4).clamp(0.0, 2.0);
    let aspects: Vec<f64> = photo_slots
        .iter()
        .map(|slot| slot.map_or(1.4, |id| photo_aspect(s, id)))
        .collect();
    let rects = crate::print::book::smart_layout_spread(
        pw,
        ph,
        s.print.book.spread_mode,
        safe_in,
        gutter_in,
        gap_in,
        s.print.book.binding_mode,
        &aspects,
        variation,
    );
    let placed: Vec<(Rect, Option<u64>)> = rects
        .into_iter()
        .zip(photo_slots.iter())
        .map(|(r, slot)| (r, slot.map(|id| id.0)))
        .collect();
    s.print.book.enabled = true;
    s.print.image.zoom_to_fill = true;
    s.print.book.set_variation(spread as usize, variation);
    s.print.layout.replace_page_cells(spread, placed).map_err(|e| bad(ID, e))
}

/// Applies SmartAlbums BSP / Justified Grid layout to one spread (`spread`) for the given `photo_ids` (`1..=8`).
fn apply_smart_spread(s: &mut Session, spread: u32, photo_ids: &[lightcraft_catalog::PhotoId], variation: usize) -> Result<Vec<u32>> {
    let slots: Vec<Option<lightcraft_catalog::PhotoId>> = photo_ids.iter().copied().map(Some).collect();
    apply_smart_spread_slots(s, spread, &slots, variation)
}

/// `print.auto_layout_spread`: Places 1..8 photos onto a spread using the Pixellu SmartAlbums BSP / Justified Grid
/// algorithm, or auto-paginates all selected/album photos across spreads when `autoPaginateAll: true`.
fn auto_layout_spread(s: &mut Session, p: &Value) -> Result<Value> {
    const ID: &str = "print.auto_layout_spread";
    if let Some(gap_mm) = opt_f64(p, "gapMm", ID)? {
        s.print.book.gap_mm = gap_mm.clamp(0.0, 40.0);
        s.print.layout.spacing = (s.print.book.gap_mm / 25.4).clamp(0.0, 2.0);
    }
    let spread = p
        .get("spread")
        .or_else(|| p.get("page"))
        .and_then(Value::as_u64)
        .and_then(|v| u32::try_from(v).ok())
        .unwrap_or(0);
    let variation = p
        .get("variation")
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .unwrap_or_else(|| s.print.book.variation_for(spread as usize));
    let paginate_all = p.get("autoPaginateAll").and_then(Value::as_bool).unwrap_or(false);

    // Resolve photo IDs: explicit `photos`/`ids`, or existing photos on the spread, or current selection
    let explicit_ids: Option<Vec<lightcraft_catalog::PhotoId>> = p
        .get("photos")
        .or_else(|| p.get("ids"))
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_u64().map(lightcraft_catalog::PhotoId))
                .filter(|id| s.catalog.photo(*id).is_some())
                .collect()
        });

    let photos = match explicit_ids {
        Some(list) if !list.is_empty() => list,
        _ => {
            let on_spread: Vec<lightcraft_catalog::PhotoId> = s
                .print
                .layout
                .cells_on_page(spread)
                .filter_map(|c| c.photo.map(lightcraft_catalog::PhotoId))
                .filter(|id| s.catalog.photo(*id).is_some())
                .collect();
            if !on_spread.is_empty() && !paginate_all {
                on_spread
            } else {
                let targets = s.targets(&Value::Null)?;
                if targets.len() > 1 || (!targets.is_empty() && !paginate_all) {
                    targets
                } else {
                    let vis = s.visible_cloned();
                    if !vis.is_empty() {
                        vis
                    } else if !targets.is_empty() {
                        targets
                    } else {
                        s.catalog.photos().map(|p| p.id).collect()
                    }
                }
            }
        }
    };

    if photos.is_empty() {
        return Err(bad(ID, "no photos available to layout on the spread"));
    }

    if paginate_all {
        let per_spread = p
            .get("photosPerSpread")
            .and_then(Value::as_u64)
            .map(|v| (v as usize).clamp(1, 8))
            .unwrap_or_else(|| (s.print.book.photos_per_spread as usize).clamp(1, 8));
        s.print.book.photos_per_spread = per_spread as u8;
        s.print.layout.clear();
        s.print.layout.style = crate::print::layout::Style::CustomPackage;
        let mut spread_count = 0u32;
        for (idx, chunk) in photos.chunks(per_spread).take(MAX_PAGES as usize).enumerate() {
            let var = (idx * 3) % crate::print::book::variation_count(chunk.len(), s.print.book.spread_mode);
            apply_smart_spread(s, idx as u32, chunk, var)?;
            spread_count = (idx + 1) as u32;
        }
        let warnings = crate::print::book::inspect_spread_warnings(&s.print, 0, |pid| photo_faces_norm(s, pid));
        return Ok(json!({
            "spreads": spread_count,
            "cells": s.print.layout.cells.len(),
            "photosPerSpread": s.print.book.photos_per_spread,
            "spineWidthMm": s.print.book.spine_width_mm(spread_count as usize),
            "warnings": warnings,
        }));
    }

    let max_on_spread = p
        .get("photosPerSpread")
        .and_then(Value::as_u64)
        .map(|v| (v as usize).clamp(1, 8))
        .unwrap_or(8);
    let capped: Vec<_> = photos.into_iter().take(max_on_spread).collect();
    let cell_ids = apply_smart_spread(s, spread, &capped, variation)?;
    let warnings = crate::print::book::inspect_spread_warnings(&s.print, spread, |pid| photo_faces_norm(s, pid));
    Ok(json!({
        "spread": spread,
        "variation": variation,
        "variationCount": crate::print::book::variation_count(capped.len(), s.print.book.spread_mode),
        "cellIds": cell_ids,
        "pages": s.print.layout.pages,
        "warnings": warnings,
    }))
}

/// `print.cycle_layout`: Instantly cycles to the next mathematical layout permutation (`variation + step`)
/// for the frames/photos on `spread` without distorting aspect ratios.
fn cycle_layout(s: &mut Session, p: &Value) -> Result<Value> {
    const ID: &str = "print.cycle_layout";
    let spread = p
        .get("spread")
        .or_else(|| p.get("page"))
        .and_then(Value::as_u64)
        .and_then(|v| u32::try_from(v).ok())
        .unwrap_or(0);
    let step = p.get("step").and_then(Value::as_i64).unwrap_or(1);

    let mut photo_slots: Vec<Option<lightcraft_catalog::PhotoId>> = s
        .print
        .layout
        .cells_on_page(spread)
        .map(|c| {
            c.photo
                .map(lightcraft_catalog::PhotoId)
                .filter(|id| s.catalog.photo(*id).is_some())
        })
        .collect();

    if photo_slots.is_empty() {
        let per_spread = (s.print.book.photos_per_spread as usize).clamp(1, 8);
        let targets = s.targets(&Value::Null)?;
        photo_slots = targets.into_iter().take(per_spread).map(Some).collect();
    }
    if photo_slots.is_empty() {
        return Err(bad(ID, "no photos on spread to cycle layout"));
    }

    let count = crate::print::book::variation_count(photo_slots.len(), s.print.book.spread_mode).max(1);
    let cur = s.print.book.variation_for(spread as usize);
    let next_var = ((cur as isize + step as isize).rem_euclid(count as isize)) as usize;

    let cell_ids = apply_smart_spread_slots(s, spread, &photo_slots, next_var)?;
    let warnings = crate::print::book::inspect_spread_warnings(&s.print, spread, |pid| photo_faces_norm(s, pid));
    Ok(json!({
        "spread": spread,
        "variation": next_var,
        "variationCount": count,
        "cellIds": cell_ids,
        "warnings": warnings,
    }))
}

/// `print.export_book`: Renders and exports the Photobook at 300 DPI in one of the 3 physical production modes:
/// - `layflatSpreads` (Mode A): Sequential 300 DPI sRGB spread JPEGs (`Pliego_01.jpg`, `Pliego_02.jpg`, ..., `Portada.jpg`)
/// - `sequentialPdf` (Mode B): Single-page cut sequential PDF (`Página 1..N`, Page 1 odd right-hand page) with bleed, gutter & crop marks
/// - `saddleStitch` (Mode C): Multiple-of-4 booklet imposition (`Tiro` & `Retiro` using formula `N + 1`)
fn export_book(s: &mut Session, p: &Value) -> Result<Value> {
    const ID: &str = "print.export_book";
    let path = p
        .get("path")
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| bad(ID, "missing `path`"))?;
    s.check_write_target(path).map_err(|e| bad(ID, e))?;

    let mut settings = s.print.clone();
    if let Some(m) = p.get("mode").or_else(|| p.get("bindingMode")) {
        settings.book.binding_mode = serde_json::from_value(m.clone()).map_err(|e| bad(ID, format!("invalid mode: {e}")))?;
    }
    if let Some(ppi) = opt_f64(p, "ppi", ID)? {
        settings.job.ppi = ppi as f32;
    } else if settings.book.enabled && (settings.job.ppi - 240.0).abs() < 1e-3 {
        settings.job.ppi = 300.0;
    }
    let crop_marks = p.get("cropMarks").and_then(Value::as_bool).unwrap_or(settings.page.crop_marks);
    let include_cover = p.get("includeCover").and_then(Value::as_bool).unwrap_or(settings.book.hardcover_wrap);

    let ids = s.targets(p)?;
    crate::print_color::PrintColorJob::new(&settings.job.color).map_err(|e| bad(ID, e))?;
    let title = std::path::Path::new(path).file_stem().and_then(|n| n.to_str()).unwrap_or("Photobook").to_string();
    let work = crate::print::compose::prepare(s, &settings, &ids).map_err(|e| bad(ID, e))?;
    let total = work.len() as u64;
    let row = s.activity.start("export", "Exporting Photobook (300 DPI)", crate::activity::Cancel::No);
    row.progress(0, total);
    let spreads = work
        .run(&mut |done, name| {
            row.progress(done as u64, total);
            row.detail(name);
            true
        })
        .map_err(|e| bad(ID, e))?;

    match settings.book.binding_mode {
        crate::print::book::BindingMode::LayflatSpreads => {
            // Determine output directory for `Pliego_01.jpg`, `Pliego_02.jpg`, ..., `Portada.jpg`
            let target_path = std::path::Path::new(path);
            let out_dir = if target_path.extension().is_some_and(|ext| ext.eq_ignore_ascii_case("pdf") || ext.eq_ignore_ascii_case("jpg") || ext.eq_ignore_ascii_case("jpeg")) {
                target_path.parent().unwrap_or_else(|| std::path::Path::new(".")).to_path_buf()
            } else {
                target_path.to_path_buf()
            };
            std::fs::create_dir_all(&out_dir).map_err(|e| bad(ID, e.to_string()))?;
            let mut written_files = Vec::new();
            for (idx, sp) in spreads.iter().enumerate() {
                let fname = format!("Pliego_{:02}.jpg", idx + 1);
                let fpath = out_dir.join(&fname);
                let fpath_str = fpath.to_string_lossy().to_string();
                s.check_write_target(&fpath_str).map_err(|e| bad(ID, e))?;
                let bytes = crate::print::book::encode_page_jpeg_srgb(sp, 96).map_err(|e| bad(ID, e))?;
                crate::export::write_file(&fpath_str, &bytes).map_err(|e| bad(ID, e))?;
                written_files.push(fpath_str);
            }
            if include_cover && let Some(first_sp) = spreads.first() {
                let cover_path = out_dir.join("Portada.jpg");
                let cover_str = cover_path.to_string_lossy().to_string();
                s.check_write_target(&cover_str).map_err(|e| bad(ID, e))?;
                let bytes = crate::print::book::encode_page_jpeg_srgb(first_sp, 96).map_err(|e| bad(ID, e))?;
                crate::export::write_file(&cover_str, &bytes).map_err(|e| bad(ID, e))?;
                written_files.push(cover_str);
            }
            // If caller specifically requested a `.pdf` path as well, also write the spread PDF there
            if target_path.extension().is_some_and(|ext| ext.eq_ignore_ascii_case("pdf")) {
                let pdf_bytes = crate::print::output::write(&spreads, &settings.job, &title).map_err(|e| bad(ID, e))?;
                crate::export::write_file(path, &pdf_bytes).map_err(|e| bad(ID, e))?;
            }
            Ok(json!({
                "mode": "layflatSpreads",
                "directory": out_dir.to_string_lossy(),
                "files": written_files,
                "spreads": spreads.len(),
                "ppi": settings.job.effective_ppi(),
            }))
        }
        crate::print::book::BindingMode::SequentialPdf => {
            let single_pages = crate::print::book::split_spreads_to_single_pages(
                &spreads,
                settings.book.spread_mode,
                settings.page.background,
                settings.guides.bleed_mm,
                settings.guides.gutter_mm,
                crop_marks,
            )
            .map_err(|e| bad(ID, e))?;
            let pdf_bytes = crate::print::output::write(&single_pages, &settings.job, &title).map_err(|e| bad(ID, e))?;
            s.check_write_target(path).map_err(|e| bad(ID, e))?;
            crate::export::write_file(path, &pdf_bytes).map_err(|e| bad(ID, e))?;
            Ok(json!({
                "mode": "sequentialPdf",
                "path": path,
                "pages": single_pages.len(),
                "cropMarks": crop_marks,
                "ppi": settings.job.effective_ppi(),
                "bytes": pdf_bytes.len(),
            }))
        }
        crate::print::book::BindingMode::SaddleStitch => {
            let (sheets, plan, orig_pages) = crate::print::book::compose_saddle_stitch_sheets(
                &spreads,
                settings.book.spread_mode,
                settings.page.background,
                settings.guides.bleed_mm,
                crop_marks,
            )
            .map_err(|e| bad(ID, e))?;
            let pdf_bytes = crate::print::output::write(&sheets, &settings.job, &title).map_err(|e| bad(ID, e))?;
            s.check_write_target(path).map_err(|e| bad(ID, e))?;
            crate::export::write_file(path, &pdf_bytes).map_err(|e| bad(ID, e))?;
            let padded_total = plan.len() * 2;
            Ok(json!({
                "mode": "saddleStitch",
                "path": path,
                "originalPages": orig_pages,
                "paddedPagesMultipleOf4": padded_total,
                "blankGuardPagesAdded": padded_total.saturating_sub(orig_pages),
                "impositionSheets": sheets.len(),
                "impositionPlan": plan,
                "ppi": settings.job.effective_ppi(),
                "bytes": pdf_bytes.len(),
            }))
        }
    }
}

/// `print.save_project`: Generates a self-contained portable `.mmbook` project file containing all
/// layout settings, pre-press guides, cell coordinates/pan/zoom/captions, and the high-resolution
/// (300 DPI sRGB) rendered photos so a photographer can send a single `.mmbook` file to a print shop.
fn save_project(s: &mut Session, p: &Value) -> Result<Value> {
    const ID: &str = "print.save_project";
    let raw_path = p
        .get("path")
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| bad(ID, "missing `path`"))?;
    let path = if raw_path.to_ascii_lowercase().ends_with(".mmbook") {
        raw_path.to_string()
    } else {
        format!("{raw_path}.mmbook")
    };
    s.check_write_target(&path).map_err(|e| bad(ID, e))?;
    let title = p
        .get("title")
        .and_then(Value::as_str)
        .filter(|t| !t.trim().is_empty())
        .or_else(|| std::path::Path::new(&path).file_stem().and_then(|n| n.to_str()))
        .unwrap_or("Proyecto Fotolibro")
        .to_string();
    let ids = s.targets(p).unwrap_or_default();
    let print_settings = s.print.clone();
    let work = crate::print::book::prepare_mmbook(s, &title, &print_settings, &ids).map_err(|e| bad(ID, e))?;
    let total = work.len() as u64;
    let row = s.activity.start("export", "Empaquetando Proyecto .mmbook", crate::activity::Cancel::No);
    row.progress(0, total);
    let bytes = work
        .run(&mut |done, name| {
            row.progress(done as u64, total);
            row.detail(name);
            true
        })
        .map_err(|e| bad(ID, e))?;
    crate::export::write_file(&path, &bytes).map_err(|e| bad(ID, e))?;
    Ok(json!({
        "format": "mmbook-v1",
        "path": path,
        "title": title,
        "photos": total,
        "embeddedPhotos": total,
        "pages": s.print.layout.pages,
        "bytes": bytes.len(),
    }))
}

/// `print.open_project`: Opens a portable `.mmbook` project file, extracts its embedded 300 DPI photos,
/// registers them in the catalog, remaps every cell's photo ID, and restores the exact Photobook layout
/// and pre-press settings so a print shop can immediately review or print the job.
pub fn open_project(s: &mut Session, p: &Value) -> Result<Value> {
    const ID: &str = "print.open_project";
    let path = p
        .get("path")
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| bad(ID, "missing `path`"))?;
    let data = std::fs::read(path).map_err(|e| bad(ID, format!("{path}: {e}")))?;
    let (mut manifest, photos) = crate::print::book::unpack_mmbook_bytes(&data).map_err(|e| bad(ID, e))?;

    let slug: String = manifest
        .title
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .take(40)
        .collect();
    let extract_dir = std::env::temp_dir()
        .join("mmstudio_mmbook_cache")
        .join(format!("{}_{}", if slug.is_empty() { "book" } else { &slug }, data.len()));
    std::fs::create_dir_all(&extract_dir).map_err(|e| bad(ID, e.to_string()))?;

    let mut written_paths = Vec::with_capacity(photos.len());
    let mut entry_by_path = Vec::with_capacity(photos.len());
    for ( idx, (entry, jpg_bytes)) in photos.into_iter().enumerate() {
        let safe_name = std::path::Path::new(&entry.file_name)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("foto.jpg");
        let fpath = extract_dir.join(format!("{:03}_{}", idx + 1, safe_name));
        std::fs::write(&fpath, &jpg_bytes).map_err(|e| bad(ID, e.to_string()))?;
        let fpath_str = fpath.to_string_lossy().to_string();
        written_paths.push(fpath_str.clone());
        entry_by_path.push((entry, fpath_str));
    }

    if !written_paths.is_empty() {
        let album_title = format!("📦 {}", manifest.title);
        let _ = s.execute(
            "library.import",
            &json!({
                "paths": written_paths,
                "mode": "add",
                "albumName": album_title,
            }),
        )?;
    }

    // Build mapping from original_id -> new PhotoId in catalog
    let mut id_map = std::collections::HashMap::<u64, u64>::new();
    let mut new_photo_ids = Vec::new();
    for (entry, fpath_str) in &entry_by_path {
        let found = s.catalog.photos().find(|ph| match &ph.source {
            lightcraft_catalog::Source::File { path } => path == fpath_str,
            _ => false,
        });
        if let Some(ph) = found {
            let new_id = ph.id;
            id_map.insert(entry.original_id, new_id.0);
            new_photo_ids.push(new_id);
            // Restore face regions if the imported JPEG didn't carry XMP face tags
            if !entry.faces.is_empty() && ph.meta.regions.is_empty() {
                let mut meta = ph.meta.clone();
                for [x, y, w, h] in &entry.faces {
                    meta.regions.push(lightcraft_meta::Region {
                        name: None,
                        description: None,
                        kind: lightcraft_meta::RegionKind::Face,
                        rect: lightcraft_geom::Rect {
                            x0: *x,
                            y0: *y,
                            x1: (*x + *w).clamp(0.0, 1.0),
                            y1: (*y + *h).clamp(0.0, 1.0),
                        },
                    });
                }
                let _ = s.commit("Restore .mmbook Face Regions", lightcraft_catalog::Op::SetMeta { id: new_id, meta: Box::new(meta) });
            }
        }
    }

    for cell in &mut manifest.settings.layout.cells {
        if let Some(old_id) = cell.photo {
            if let Some(&mapped) = id_map.get(&old_id) {
                cell.photo = Some(mapped);
            } else if let Some(first) = new_photo_ids.first() {
                cell.photo = Some(first.0);
            }
        }
    }

    manifest.settings.book.enabled = true;
    manifest.settings.validate().map_err(|e| bad(ID, e))?;
    s.print = manifest.settings;
    if let Some(&first) = new_photo_ids.first() {
        s.selection = crate::Selection {
            active: Some(first),
            ids: new_photo_ids.iter().copied().collect(),
        };
    }

    Ok(json!({
        "format": "mmbook-v1",
        "path": path,
        "title": manifest.title,
        "importedPhotos": new_photo_ids.len(),
        "pages": s.print.layout.pages,
        "spreads": s.print.layout.pages,
        "cells": s.print.layout.cells.len(),
        "bindingMode": s.print.book.binding_mode,
    }))
}

fn auto_layout(s: &mut Session, _: &Value) -> Result<Value> {
    s.print.layout.auto_layout().map_err(|e| bad("print.layout.autoLayout", e))?;
    Ok(json!({"pages": s.print.layout.pages, "cells": s.print.layout.cells}))
}

fn new_page(s: &mut Session, _: &Value) -> Result<Value> {
    let page = s.print.layout.new_page().map_err(|e| bad("print.layout.newPage", e))?;
    Ok(json!({"page": page, "pages": s.print.layout.pages, "max": MAX_PAGES}))
}

fn clear(s: &mut Session, _: &Value) -> Result<Value> {
    s.print.layout.clear();
    Ok(json!({"pages": 1, "cells": 0}))
}

fn export_pdf(s: &mut Session, p: &Value) -> Result<Value> {
    const ID: &str = "print.exportPdf";
    let path = p.get("path").and_then(Value::as_str).filter(|s| !s.trim().is_empty()).ok_or_else(|| bad(ID, "missing path"))?;
    if let Some(k) = p.as_object().and_then(|o| o.keys().find(|k| !matches!(k.as_str(), "path" | "ids" | "id"))) {
        return Err(bad(ID, format!("unknown parameter `{k}` (one of path, ids; change the print with print.set)")));
    }
    if p.get("ids").is_some_and(|v| v.as_array().is_none_or(|a| a.iter().any(|v| v.as_u64().is_none()))) {
        return Err(bad(ID, "ids must be an array of photo ids"));
    }
    if p.get("id").is_some_and(|v| v.as_u64().is_none()) {
        return Err(bad(ID, "id must be a photo id"));
    }
    s.check_write_target(path).map_err(|e| bad(ID, e))?;
    let ids = s.targets(p)?;
    let settings = s.print.clone();
    // a printer profile that can't be used fails the print before anything renders
    crate::print_color::PrintColorJob::new(&settings.job.color).map_err(|e| bad(ID, e))?;
    let title = std::path::Path::new(path).file_stem().and_then(|n| n.to_str()).unwrap_or("Print").to_string();
    let work = crate::print::compose::prepare(s, &settings, &ids).map_err(|e| bad(ID, e))?;
    let total = work.len() as u64;
    let row = s.activity.start("export", "Printing to PDF", crate::activity::Cancel::No);
    row.progress(0, total);
    let pages = work
        .run(&mut |done, name| {
            row.progress(done as u64, total);
            row.detail(name);
            true
        })
        .map_err(|e| bad(ID, e))?;
    let bytes = crate::print::output::write(&pages, &settings.job, &title).map_err(|e| bad(ID, e))?;
    s.check_write_target(path).map_err(|e| bad(ID, e))?;
    crate::export::write_file(path, &bytes).map_err(|e| bad(ID, e))?;
    Ok(json!({"path": path, "pages": pages.len(), "photos": ids.len(), "bytes": bytes.len()}))
}

// ---- templates ------------------------------------------------------------------------------

fn template_list(s: &Session) -> Value {
    json!({"templates": s.print_templates.iter().map(|t| json!({"name": t.name, "style": t.settings.pointer("/layout/style"), "settings": t.settings})).collect::<Vec<_>>()})
}

fn template_name(p: &Value, id: &str) -> Result<String> {
    clean_name(p.get("name").and_then(Value::as_str).ok_or_else(|| bad(id, "missing `name`"))?).map_err(|e| bad(id, e))
}

fn template_save(s: &mut Session, p: &Value) -> Result<Value> {
    const ID: &str = "print.template.save";
    let name = template_name(p, ID)?;
    let t = PrintTemplate::new(&name, &s.print).map_err(|e| bad(ID, e))?;
    match s.print_templates.iter_mut().find(|x| x.name.eq_ignore_ascii_case(&name)) {
        Some(x) => *x = t,
        None => {
            if s.print_templates.len() >= MAX_TEMPLATES {
                return Err(bad(ID, format!("at most {MAX_TEMPLATES} templates; delete one first")));
            }
            s.print_templates.push(t);
        }
    }
    s.save_prefs()?;
    Ok(template_list(s))
}

fn template_apply(s: &mut Session, p: &Value) -> Result<Value> {
    const ID: &str = "print.template.apply";
    let name = template_name(p, ID)?;
    let t = s
        .print_templates
        .iter()
        .find(|x| x.name.eq_ignore_ascii_case(&name))
        .ok_or_else(|| bad(ID, format!("no print template `{name}` (see print.template.list)")))?;
    s.print = t.settings(&s.print.guides).map_err(|e| bad(ID, e))?;
    reply(s)
}

fn template_delete(s: &mut Session, p: &Value) -> Result<Value> {
    const ID: &str = "print.template.delete";
    let name = template_name(p, ID)?;
    let before = s.print_templates.len();
    s.print_templates.retain(|x| !x.name.eq_ignore_ascii_case(&name));
    if s.print_templates.len() == before {
        return Err(bad(ID, format!("no print template `{name}`")));
    }
    s.save_prefs()?;
    Ok(template_list(s))
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        cmd!(query "print.get", "Print Settings", [], None,
            "{ids?} → {settings: {layout, image, page, guides, job}, papers, pageWidth, pageHeight, printable, pageCount, cellsPerPage, gridCells?, overlaps}; lengths are inches",
            always, get),
        cmd!(query "print.set", "Set Print Settings", [], None,
            "{layout?: {style: singleImage|contactSheet|picturePackage|customPackage, page: {paper, customWidth, customHeight, orientation: portrait|landscape, margins: {top,right,bottom,left}}, grid: {columns, rows, spacingX, spacingY, keepSquare}, spacing}, image?: {zoomToFill, rotateToFit, borderEnabled, borderWidth, borderColor: [r,g,b], strokeEnabled, strokeWidthPt, strokeColor}, page?: {background: [r,g,b], pageNumbers, pageInfo, cropMarks} (Page Options: single image and contact sheet only), guides?: {showRulers, units: in|cm|mm|pt, snap: off|cells|grid, gridSize, showMargins, showBleed, showPageGrid, showCells, showDimensions}, job?: {ppi: 72..720, color?: {profile: {kind: managed, space: srgb|adobeRgb} | {kind: icc, path}, intent: perceptual|relative, blackPointCompensation, brightness: -100..100, contrast: -100..100, sharpen: off|low|standard|high, media: matte|glossy, sixteenBit}}} — merges into the current print; cells and pages change with print.cell.* / print.layout.*; → print.get",
            always, set),
        cmd!(query "print.reset", "Reset Print Settings", [], None, "{} → print.get", always, reset),
        cmd!(query "print.cell.add", "Add Print Cell", [], None,
            "{width, height, x?, y?, page?, photo?} inches; picture/custom package → {id, cell, pages}; the cell goes to the first free spot unless x and y are given",
            always, cell_add),
        cmd!(query "print.cell.move", "Move Print Cell", [], None, "{id, x, y, snap?: bool, threshold?: inches} → {id, rect}; snap uses the guides' snap mode", always, cell_move),
        cmd!(query "print.cell.resize", "Resize Print Cell", [], None,
            "{id, width?, height?, x?, y?, snap?: bool, threshold?} → {id, rect}", always, cell_resize),
        cmd!(query "print.cell.delete", "Delete Print Cell", [], None, "{id}", always, cell_delete),
        cmd!(query "print.cell.stack", "Arrange Print Cell", [], None, "{id, order: toBack|backward|forward|toFront} → {order: cell ids back to front}", always, cell_stack),
        cmd!(query "print.cell.photo", "Set Print Cell Photo", [], None, "{id, photo: photo id | null} (custom package: the photo printed in the cell)", always, cell_photo),
        cmd!(query "print.cell.page", "Move Print Cell to Page", [], None, "{id, page: zero-based}", always, cell_page),
        cmd!(query "print.cell.panZoom", "Set Print Cell Pan, Zoom & Caption", [], None,
            "{id, panX?: -1..1, panY?: -1..1, zoom?: 1..5, caption?: string|null} → {id, cell}", always, cell_pan_zoom),
        cmd!(query "print.set_format", "Set Photobook Physical Format & Pre-Press Guides", [], None,
            "{preset?: '8x8'|'10x10'|'12x12'|'11x14'|'8x10'|'a4'|'custom', width?, height?, unit?: 'in'|'cm'|'mm', spreadMode?: bool, bindingMode?: 'layflatSpreads'|'sequentialPdf'|'saddleStitch', bleedMm?: 3..5, safeZoneMm?: 10..15, gutterMm?: 10..15, hardcoverWrap?: bool, wrapMm?: 18, paperThicknessMm?: 0.38, gapMm?: 4, ppi?: 300} → print.get",
            always, set_format),
        cmd!(query "print.auto_layout_spread", "SmartAlbums Auto-Layout Spread (BSP / Justified Grid)", [], None,
            "{spread?: zero-based, photos?: [photo ids 1..8], gapMm?: number, variation?: usize, autoPaginateAll?: bool, photosPerSpread?: 1..8} → {spread, variation, variationCount, cellIds, pages, warnings}",
            always, auto_layout_spread),
        cmd!(query "print.cycle_layout", "Cycle Next SmartAlbums Layout Permutation", [], None,
            "{spread?: zero-based, step?: 1|-1} → {spread, variation, variationCount, cellIds, warnings}",
            always, cycle_layout),
        cmd!(query "print.export_book", "Export Photobook (300 DPI Layflat / Sequential PDF / Saddle-Stitch)", [], None,
            "{path, mode?: 'layflatSpreads'|'sequentialPdf'|'saddleStitch', cropMarks?: bool, includeCover?: bool, ppi?: 300, ids?: [photo ids]} → export summary",
            always, export_book),
        cmd!(query "print.save_project", "Save Portable Photobook Project (.mmbook)", [], None,
            "{path: '.mmbook', title?: string, ids?: [photo ids]} — packages all Photobook layout settings, pre-press guides and 300 DPI rendered photos into a single portable .mmbook file for print shops → {path, title, photos, pages, bytes}",
            always, save_project),
        cmd!("print.open_project", "Open Portable Photobook Project (.mmbook)", [], None,
            "{path: '.mmbook'} — unpacks a portable .mmbook project file, imports its embedded 300 DPI photos, remaps cell IDs and restores the exact Photobook layout ready to print → {path, title, importedPhotos, pages, cells, bindingMode}",
            always, open_project),
        cmd!(query "print.layout.autoLayout", "Auto Layout", [], None, "{} — packs the package's cells onto as few pages as possible → {pages, cells}", always, auto_layout),
        cmd!(query "print.layout.newPage", "New Print Page", [], None, "{} → {page, pages}", always, new_page),
        cmd!(query "print.layout.clear", "Clear Print Layout", [], None, "{} — removes every cell and extra page", always, clear),
        cmd!(query "print.exportPdf", "Print to PDF", [], None,
            "{path, ids?: photo ids (default selection)} → {path, pages, photos, bytes}; prints the current print.get settings; replaces an existing output, never an original",
            always, export_pdf),
        cmd!(query "print.template.list", "Print Templates", [], None, "{} → {templates: [{name, style, settings}]}", always, |s, _| Ok(template_list(s))),
        cmd!(query "print.template.save", "Save Print Template", [], None,
            "{name} — saves the layout, image settings, page and job (not the cells' photos or the guides); an existing name (any case) is replaced → print.template.list",
            always, template_save),
        cmd!(query "print.template.apply", "Apply Print Template", [], None, "{name} → print.get", always, template_apply),
        cmd!(query "print.template.delete", "Delete Print Template", [], None, "{name} → print.template.list", always, template_delete),
    ]
}
