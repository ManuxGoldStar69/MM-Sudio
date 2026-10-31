//! The `print.*` commands: state, cells, templates and the PDF.

use lightcraft_catalog::MemStore;
use serde_json::{Value, json};

use crate::Session;
use crate::library::LibraryStores;

fn run(s: &mut Session, id: &str, p: Value) -> Value {
    s.execute(id, &p).unwrap_or_else(|e| panic!("{id} {p}: {e}"))
}

fn fails(s: &mut Session, id: &str, p: Value) -> String {
    s.execute(id, &p).expect_err(&format!("{id} {p} should fail")).to_string()
}

fn package(s: &mut Session) {
    run(s, "print.set", json!({"layout": {"style": "customPackage", "page": {"paper": "letter"}}}));
}

#[test]
fn get_set_and_reset() {
    let mut s = Session::with_demo();
    let g = run(&mut s, "print.get", json!({}));
    assert_eq!(g["settings"]["layout"]["style"], "singleImage");
    assert_eq!(g["pageWidth"], 8.5);
    assert_eq!(g["printable"]["x"], 0.25);
    assert_eq!(g["settings"]["job"]["ppi"], 240.0);
    assert!(g["papers"].as_array().unwrap().iter().any(|p| p["name"] == "a4"));
    assert_eq!(g["gridCells"].as_array().unwrap().len(), 1);

    let g = run(
        &mut s,
        "print.set",
        json!({"layout": {"style": "contactSheet", "grid": {"columns": 2, "rows": 3}, "page": {"paper": "a4", "orientation": "landscape", "margins": {"left": 1.0}}}, "image": {"zoomToFill": true}, "job": {"ppi": 300.0}}),
    );
    assert_eq!(g["settings"]["layout"]["page"]["margins"]["left"], 1.0);
    assert_eq!(g["settings"]["layout"]["page"]["margins"]["top"], 0.25, "a patch merges");
    assert_eq!(g["gridCells"].as_array().unwrap().len(), 6);
    assert!((g["pageWidth"].as_f64().unwrap() - 11.693).abs() < 1e-3);
    assert_eq!(g["settings"]["image"]["zoomToFill"], true);
    // the page count follows the photos: 6 cells per page
    let g = run(&mut s, "print.get", json!({"ids": vec![1; 7]}));
    assert_eq!(g["pageCount"], 2);

    let g = run(&mut s, "print.reset", json!({}));
    assert_eq!(g["settings"]["job"]["ppi"], 240.0);
}

#[test]
fn bad_settings_are_refused_and_change_nothing() {
    let mut s = Session::with_demo();
    let before = run(&mut s, "print.get", json!({}));
    for p in [
        json!({"typo": 1}),
        json!({"layout": {"typo": 1}}),
        json!({"layout": {"style": "poster"}}),
        json!({"layout": {"cells": []}}),
        json!({"layout": {"pages": 3}}),
        json!({"layout": {"page": {"paper": "poster"}}}),
        json!({"layout": {"page": {"paper": "custom", "customWidth": -1}}}),
        json!({"layout": {"page": {"margins": {"top": 1e9}}}}),
        json!({"layout": {"page": {"margins": {"top": 6.0, "bottom": 5.0}}}}),
        json!({"layout": {"grid": {"columns": 0}}}),
        json!({"layout": {"grid": {"columns": 9999}}}),
        json!({"layout": {"spacing": -1}}),
        json!({"image": {"borderWidth": 99}}),
        json!({"image": {"strokeWidthPt": -2}}),
        json!({"image": {"borderColor": [1, 2]}}),
        json!({"image": {"zoomToFill": "yes"}}),
        json!({"guides": {"gridSize": 0}}),
        json!({"guides": {"units": "furlong"}}),
        json!({"job": {"ppi": 1e30}}),
        json!({"job": {"ppi": 10}}),
        json!({"job": null}),
    ] {
        fails(&mut s, "print.set", p);
    }
    assert_eq!(run(&mut s, "print.get", json!({})), before);
    assert!(s.execute("print.set", &json!([1])).is_err());
}

#[test]
fn cells_are_commands() {
    let mut s = Session::with_demo();
    // grid styles have no cells
    fails(&mut s, "print.cell.add", json!({"width": 2, "height": 2}));
    package(&mut s);
    let a = run(&mut s, "print.cell.add", json!({"width": 3.0, "height": 2.0}))["id"].as_u64().unwrap();
    let b = run(&mut s, "print.cell.add", json!({"width": 3.0, "height": 2.0, "x": 5.0, "y": 5.0}))["id"].as_u64().unwrap();
    let g = run(&mut s, "print.get", json!({}));
    assert_eq!(g["settings"]["layout"]["cells"].as_array().unwrap().len(), 2);
    assert_eq!(g["overlaps"], false);
    // move, clamped inside the margins
    let r = run(&mut s, "print.cell.move", json!({"id": a, "x": 99.0, "y": 0.0}));
    assert_eq!(r["rect"]["x"], 8.25 - 3.0);
    assert_eq!(r["rect"]["y"], 0.25);
    // resize keeps what is not given
    let r = run(&mut s, "print.cell.resize", json!({"id": b, "width": 4.0}));
    assert_eq!((r["rect"]["w"].as_f64().unwrap(), r["rect"]["h"].as_f64().unwrap()), (4.0, 2.0));
    // snapping to the other cell's edge
    let ra = run(&mut s, "print.get", json!({}))["settings"]["layout"]["cells"][0]["rect"].clone();
    let left = ra["x"].as_f64().unwrap();
    let r = run(&mut s, "print.cell.move", json!({"id": b, "x": left - 4.0 - 0.04, "y": 5.0, "snap": true}));
    assert!((r["rect"]["x"].as_f64().unwrap() - (left - 4.0)).abs() < 1e-9, "{r}");
    let r = run(&mut s, "print.cell.move", json!({"id": b, "x": left - 4.0 - 0.04, "y": 5.0}));
    assert!((r["rect"]["x"].as_f64().unwrap() - (left - 4.04)).abs() < 1e-9, "no snap unless asked: {r}");
    run(&mut s, "print.set", json!({"guides": {"snap": "off"}}));
    let r = run(&mut s, "print.cell.move", json!({"id": b, "x": left - 4.0 - 0.04, "y": 5.0, "snap": true}));
    assert!((r["rect"]["x"].as_f64().unwrap() - (left - 4.04)).abs() < 1e-9, "snap off");
    // stacking
    let r = run(&mut s, "print.cell.stack", json!({"id": a, "order": "toFront"}));
    assert_eq!(r["order"], json!([b, a]));
    fails(&mut s, "print.cell.stack", json!({"id": a, "order": "sideways"}));
    // photos and pages
    let photo = s.catalog.photos().next().unwrap().id.0;
    run(&mut s, "print.cell.photo", json!({"id": a, "photo": photo}));
    fails(&mut s, "print.cell.photo", json!({"id": a, "photo": 999_999}));
    run(&mut s, "print.layout.newPage", json!({}));
    run(&mut s, "print.cell.page", json!({"id": b, "page": 1}));
    fails(&mut s, "print.cell.page", json!({"id": b, "page": 7}));
    let g = run(&mut s, "print.get", json!({}));
    assert_eq!(g["pageCount"], 2);
    // delete, clear
    run(&mut s, "print.cell.delete", json!({"id": b}));
    fails(&mut s, "print.cell.delete", json!({"id": b}));
    fails(&mut s, "print.cell.move", json!({"id": 99, "x": 0, "y": 0}));
    fails(&mut s, "print.cell.move", json!({"x": 0, "y": 0}));
    run(&mut s, "print.layout.clear", json!({}));
    let g = run(&mut s, "print.get", json!({}));
    assert_eq!((g["settings"]["layout"]["cells"].as_array().unwrap().len(), g["pageCount"].as_u64()), (0, Some(1)));
}

#[test]
fn hostile_cell_numbers_are_refused() {
    let mut s = Session::with_demo();
    package(&mut s);
    for (w, h) in [(0.0, 2.0), (-1.0, 2.0), (1e12, 1e12), (2.0, 1e-9), (50.0, 50.0)] {
        fails(&mut s, "print.cell.add", json!({"width": w, "height": h}));
    }
    fails(&mut s, "print.cell.add", json!({"width": "wide", "height": 2}));
    fails(&mut s, "print.cell.add", json!({"width": 2}));
    let id = run(&mut s, "print.cell.add", json!({"width": 2, "height": 2}))["id"].as_u64().unwrap();
    fails(&mut s, "print.cell.move", json!({"id": id, "x": "left", "y": 0}));
    fails(&mut s, "print.cell.resize", json!({"id": id, "width": -4}));
    fails(&mut s, "print.cell.resize", json!({"id": id, "width": "x"}));
    fails(&mut s, "print.cell.move", json!({"id": id, "x": 1, "y": 1, "snap": "yes"}));
    fails(&mut s, "print.cell.move", json!({"id": u64::MAX, "x": 1, "y": 1}));
    let r = run(&mut s, "print.cell.move", json!({"id": id, "x": -1e300, "y": 1e300}));
    assert_eq!(r["rect"]["x"], 0.25);
}

#[test]
fn auto_layout_fills_pages_without_overlap() {
    let mut s = Session::with_demo();
    run(&mut s, "print.set", json!({"layout": {"style": "picturePackage"}}));
    for (w, h) in [(4.0, 6.0), (4.0, 6.0), (5.0, 7.0), (2.5, 3.5), (2.5, 3.5), (2.5, 3.5), (2.5, 3.5), (8.0, 10.0)] {
        run(&mut s, "print.cell.add", json!({"width": w, "height": h}));
    }
    let r = run(&mut s, "print.layout.autoLayout", json!({}));
    assert!(r["pages"].as_u64().unwrap() >= 2);
    let g = run(&mut s, "print.get", json!({}));
    assert_eq!(g["overlaps"], false);
    let again = run(&mut s, "print.layout.autoLayout", json!({}));
    assert_eq!(again["cells"], r["cells"], "deterministic");
    // an oversize cell that arrived by a paper change is pulled in, not left outside the page
    run(&mut s, "print.set", json!({"layout": {"page": {"paper": "4x6"}}}));
    let g = run(&mut s, "print.get", json!({}));
    for c in g["settings"]["layout"]["cells"].as_array().unwrap() {
        assert!(c["rect"]["x"].as_f64().unwrap() + c["rect"]["w"].as_f64().unwrap() <= 4.0 - 0.25 + 1e-9);
    }
}

/// The first page's MediaBox width and height in points.
fn media_box(pdf: &[u8]) -> (f64, f64) {
    let text = String::from_utf8_lossy(pdf);
    let at = text.find("/MediaBox [").expect("a MediaBox") + "/MediaBox [".len();
    let nums: Vec<f64> = text[at..].split(']').next().unwrap().split_whitespace().map(|v| v.parse().unwrap()).collect();
    (nums[2], nums[3])
}

fn temp_pdf(name: &str) -> String {
    std::env::temp_dir().join(format!("lightcraft-print-{name}-{}.pdf", std::process::id())).to_string_lossy().to_string()
}

#[test]
fn export_pdf_prints_the_current_settings() {
    let mut s = Session::with_demo();
    let ids: Vec<u64> = s.catalog.photos().take(3).map(|p| p.id.0).collect();
    run(
        &mut s,
        "print.set",
        json!({"layout": {"style": "contactSheet", "grid": {"columns": 1, "rows": 2}, "page": {"paper": "a5"}}, "job": {"ppi": 100.0}}),
    );
    let path = temp_pdf("export");
    let r = run(&mut s, "print.exportPdf", json!({"path": path, "ids": ids}));
    assert_eq!((r["pages"].as_u64(), r["photos"].as_u64()), (Some(2), Some(3)));
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(bytes.len() as u64, r["bytes"].as_u64().unwrap());
    let text = String::from_utf8_lossy(&bytes);
    assert!(text.starts_with("%PDF-") && text.contains("/Count 2"));
    // A5 is 148 x 210 mm = 419.53 x 595.28 pt, rounded to whole pixels at 100 ppi (583 x 827)
    let (w, h) = media_box(&bytes);
    assert!((w - 419.53).abs() < 1.0 && (h - 595.28).abs() < 1.0, "A5, got {w} x {h} pt");
    // the row in the activity stack is gone when the print is done
    assert!(s.activity.list().is_empty());
    fails(&mut s, "print.exportPdf", json!({}));
    fails(&mut s, "print.exportPdf", json!({"path": path, "typo": 1}));
    fails(&mut s, "print.exportPdf", json!({"path": path, "ids": ["x"]}));
    fails(&mut s, "print.exportPdf", json!({"path": path, "ids": [999_999]}));
    // a package with no cells has nothing to print
    run(&mut s, "print.set", json!({"layout": {"style": "picturePackage"}}));
    fails(&mut s, "print.exportPdf", json!({"path": path, "ids": ids}));
}

#[test]
fn export_pdf_with_a_printer_profile() {
    let mut s = Session::with_demo();
    let ids: Vec<u64> = s.catalog.photos().take(1).map(|p| p.id.0).collect();
    let dir = std::env::temp_dir().join(format!("lightcraft-print-icc-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let icc = dir.join("Fine Art Matte.icc");
    std::fs::write(&icc, lightcraft_codecs::icc::write_named(lightcraft_codecs::NamedSpace::AdobeRgb)).unwrap();
    let color = json!({"profile": {"kind": "icc", "path": icc}, "intent": "relative", "sharpen": "high", "media": "matte", "brightness": 10});
    run(
        &mut s,
        "print.set",
        json!({"layout": {"style": "contactSheet", "grid": {"columns": 1, "rows": 1}, "page": {"paper": "4x6"}}, "job": {"ppi": 72.0, "color": color}}),
    );
    assert_eq!(run(&mut s, "print.get", json!({}))["settings"]["job"]["color"]["media"], "matte");
    let path = temp_pdf("icc");
    run(&mut s, "print.exportPdf", json!({"path": path, "ids": ids}));
    let text = String::from_utf8_lossy(&std::fs::read(&path).unwrap()).into_owned();
    assert!(text.contains("/OutputIntents") && text.contains("/ICCBased") && text.contains("/BitsPerComponent 16"));
    // out-of-range adjustments are refused when set; a missing profile fails before rendering
    fails(&mut s, "print.set", json!({"job": {"color": {"contrast": 150}}}));
    run(&mut s, "print.set", json!({"job": {"color": {"profile": {"kind": "icc", "path": dir.join("gone.icc")}}}}));
    let err = fails(&mut s, "print.exportPdf", json!({"path": path, "ids": ids}));
    assert!(err.contains("gone.icc"), "{err}");
    assert!(s.activity.list().is_empty(), "nothing started");
}

#[test]
fn custom_package_prints_each_cells_photo() {
    let mut s = Session::with_demo();
    let ids: Vec<u64> = s.catalog.photos().take(2).map(|p| p.id.0).collect();
    run(&mut s, "print.set", json!({"layout": {"style": "customPackage", "page": {"paper": "4x6"}}, "job": {"ppi": 72.0}}));
    let a = run(&mut s, "print.cell.add", json!({"width": 3.0, "height": 2.0}))["id"].as_u64().unwrap();
    let b = run(&mut s, "print.cell.add", json!({"width": 3.0, "height": 2.0}))["id"].as_u64().unwrap();
    run(&mut s, "print.cell.photo", json!({"id": a, "photo": ids[0]}));
    run(&mut s, "print.cell.photo", json!({"id": b, "photo": ids[1]}));
    let r = run(&mut s, "print.exportPdf", json!({"path": temp_pdf("custom"), "ids": [ids[0]]}));
    assert_eq!(r["pages"], 1);
}

#[test]
fn templates_save_apply_delete_and_persist() {
    let files = MemStore::new();
    let open = |files: &MemStore| {
        let mut s = Session::with_demo();
        let stores = LibraryStores { dir: "print".into(), catalog: Box::new(MemStore::new()), files: Box::new(files.clone()), on_disk: false };
        s.open_library_in(stores, false).unwrap();
        s
    };
    let mut s = open(&files);
    run(
        &mut s,
        "print.set",
        json!({"layout": {"style": "contactSheet", "grid": {"columns": 4, "rows": 5}}, "page": {"background": [10, 20, 30]}, "guides": {"gridSize": 1.0}}),
    );
    let r = run(&mut s, "print.template.save", json!({"name": "Proofs"}));
    assert_eq!(r["templates"][0]["name"], "Proofs");
    assert!(r["templates"][0]["settings"].get("guides").is_none());
    run(&mut s, "print.template.save", json!({"name": "proofs"}));
    assert_eq!(run(&mut s, "print.template.list", json!({}))["templates"].as_array().unwrap().len(), 1, "the same name in another case replaces");

    run(&mut s, "print.reset", json!({}));
    run(&mut s, "print.set", json!({"guides": {"gridSize": 0.5}}));
    let g = run(&mut s, "print.template.apply", json!({"name": "PROOFS"}));
    assert_eq!(g["settings"]["layout"]["grid"]["columns"], 4);
    assert_eq!(g["settings"]["page"]["background"], json!([10, 20, 30]));
    assert_eq!(g["settings"]["guides"]["gridSize"], 0.5, "guides stay as they were");

    // it survives reopening the library
    let mut again = open(&files);
    assert_eq!(run(&mut again, "print.template.list", json!({}))["templates"][0]["name"], "proofs");
    run(&mut again, "print.template.apply", json!({"name": "Proofs"}));

    fails(&mut s, "print.template.save", json!({"name": "  "}));
    fails(&mut s, "print.template.save", json!({}));
    fails(&mut s, "print.template.apply", json!({"name": "Nope"}));
    fails(&mut s, "print.template.delete", json!({"name": "Nope"}));
    run(&mut s, "print.template.delete", json!({"name": "proofs"}));
    assert!(run(&mut s, "print.template.list", json!({}))["templates"].as_array().unwrap().is_empty());
    assert!(run(&mut open(&files), "print.template.list", json!({}))["templates"].as_array().unwrap().is_empty());
}

#[test]
fn a_hostile_template_in_the_preferences_does_not_break_the_library() {
    let files = MemStore::new();
    let hostile = json!({"printTemplates": [
        {"name": "Bad", "settings": {"layout": {"pages": 0}, "extra": true}},
        {"name": "Huge", "settings": {"job": {"ppi": 1e300}}},
        {"name": "Fine", "settings": {"layout": {"style": "contactSheet"}}}
    ], "cacheMb": 123});
    files.set("prefs.json", serde_json::to_vec(&hostile).unwrap());
    let mut s = Session::with_demo();
    let stores = LibraryStores { dir: "print".into(), catalog: Box::new(MemStore::new()), files: Box::new(files.clone()), on_disk: false };
    s.open_library_in(stores, false).unwrap();
    assert!(s.take_library_warnings().is_empty(), "the preferences are not damaged");
    assert_eq!(s.cache_mb, 123);
    assert_eq!(run(&mut s, "print.template.list", json!({}))["templates"].as_array().unwrap().len(), 3);
    fails(&mut s, "print.template.apply", json!({"name": "Bad"}));
    fails(&mut s, "print.template.apply", json!({"name": "Huge"}));
    run(&mut s, "print.template.apply", json!({"name": "Fine"}));
}

#[test]
fn every_print_command_is_registered_with_a_description() {
    for id in [
        "print.get",
        "print.set",
        "print.reset",
        "print.cell.add",
        "print.cell.move",
        "print.cell.resize",
        "print.cell.delete",
        "print.cell.stack",
        "print.cell.photo",
        "print.cell.page",
        "print.layout.autoLayout",
        "print.layout.newPage",
        "print.layout.clear",
        "print.exportPdf",
        "print.template.list",
        "print.template.save",
        "print.template.apply",
        "print.template.delete",
        "print.set_format",
        "print.auto_layout_spread",
        "print.cycle_layout",
        "print.export_book",
        "print.save_project",
        "print.open_project",
        "print.cell.panZoom",
    ] {
        let c = crate::find_command(id).unwrap_or_else(|| panic!("{id} missing"));
        assert!(c.params.len() > 3, "{id}");
    }
}

#[test]
fn smartalbums_photobook_commands_and_all_three_export_modes() {
    let mut s = Session::with_demo();
    let ids: Vec<u64> = s.catalog.photos().take(6).map(|p| p.id.0).collect();

    // 1. Set 12x12" format (open to 12x24" spread), 3.5mm bleed, 12mm safe zone, 12mm gutter, hardcover wrap
    let fmt = run(
        &mut s,
        "print.set_format",
        json!({
            "format": "12x12",
            "spreadMode": true,
            "bleedMm": 3.5,
            "safeZoneMm": 12.0,
            "gutterMm": 12.0,
            "gapMm": 4.0,
            "hardcoverWrap": true,
            "bindingMode": "layflatSpreads",
        }),
    );
    assert_eq!(fmt["settings"]["layout"]["page"]["paper"], "12x24");
    assert_eq!(fmt["settings"]["job"]["ppi"], 300.0);
    assert!(s.print.book.spine_width_mm(s.print.layout.pages as usize) >= 2.0);

    // 2. Auto-layout spread 0 with 4 photos
    let lay = run(
        &mut s,
        "print.auto_layout_spread",
        json!({
            "spread": 0,
            "ids": &ids[0..4],
            "variation": 0,
        }),
    );
    assert_eq!(lay["cellIds"].as_array().unwrap().len(), 4);

    // 3. Cycle layout mathematical permutation
    let cyc = run(&mut s, "print.cycle_layout", json!({"spread": 0}));
    assert_eq!(cyc["variation"], 1);
    assert_eq!(cyc["cellIds"].as_array().unwrap().len(), 4);

    // Use fast 72 PPI for test speed while verifying all 3 export pipelines
    run(&mut s, "print.set", json!({"job": {"ppi": 72.0}}));

    let dir = std::env::temp_dir().join(format!("lc-smartalbums-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // Modo A: Layflat 180° sequential JPG spreads + Portada.jpg
    let mode_a = run(
        &mut s,
        "print.export_book",
        json!({
            "path": dir.join("layflat"),
            "bindingMode": "layflatSpreads",
            "ids": &ids[0..4],
        }),
    );
    assert_eq!(mode_a["mode"], "layflatSpreads");
    assert!(dir.join("layflat").join("Pliego_01.jpg").exists());
    assert!(dir.join("layflat").join("Portada.jpg").exists());

    // Modo B: Commercial Print Sequential PDF (split spreads + gutter + crop marks)
    let pdf_b = dir.join("Secuencial.pdf");
    let mode_b = run(
        &mut s,
        "print.export_book",
        json!({
            "path": &pdf_b,
            "bindingMode": "sequentialPdf",
            "cropMarks": true,
            "ids": &ids[0..4],
        }),
    );
    assert_eq!(mode_b["mode"], "sequentialPdf");
    assert_eq!(mode_b["pages"], 2);
    assert!(pdf_b.exists());

    // Modo C: Saddle-Stitch Booklet Imposition (padded to multiple of 4, N+1 Tiro/Retiro pairing)
    let pdf_c = dir.join("Cuadernillo.pdf");
    let mode_c = run(
        &mut s,
        "print.export_book",
        json!({
            "path": &pdf_c,
            "bindingMode": "saddleStitch",
            "cropMarks": true,
            "ids": &ids[0..4],
        }),
    );
    assert_eq!(mode_c["mode"], "saddleStitch");
    assert_eq!(mode_c["paddedPagesMultipleOf4"], 4);
    assert_eq!(mode_c["impositionSheets"], 2); // 1 sheet front (Tiro: [4|1]) + 1 sheet back (Retiro: [2|3])
    assert!(pdf_c.exists());

    // Portable Print Project (.mmbook) round-trip: save on Photographer session -> open on Print Shop session
    let mmbook_path = dir.join("Boda_Editorial.mmbook");
    let saved = run(
        &mut s,
        "print.save_project",
        json!({
            "path": &mmbook_path,
            "title": "Boda Editorial",
            "ids": &ids[0..4],
        }),
    );
    assert_eq!(saved["format"], "mmbook-v1");
    assert_eq!(saved["embeddedPhotos"], 4);
    assert!(mmbook_path.exists());

    let mut print_shop = Session::new();
    let opened = run(
        &mut print_shop,
        "print.open_project",
        json!({
            "path": &mmbook_path,
        }),
    );
    assert_eq!(opened["format"], "mmbook-v1");
    assert_eq!(opened["title"], "Boda Editorial");
    assert_eq!(opened["importedPhotos"], 4);
    assert!(print_shop.print.book.enabled);
    assert_eq!(print_shop.print.layout.cells.len(), 4);
    assert!(print_shop.print.layout.cells.iter().all(|c| c.photo.is_some_and(|pid| print_shop.catalog.photo(lightcraft_catalog::PhotoId(pid)).is_some())));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn page_options_are_print_settings_and_template_fields() {
    let mut s = Session::with_demo();
    let g = run(&mut s, "print.get", json!({}));
    assert_eq!(g["settings"]["page"], json!({"background": [255, 255, 255], "pageNumbers": false, "pageInfo": false, "cropMarks": false}));
    let g = run(&mut s, "print.set", json!({"page": {"pageNumbers": true, "cropMarks": true}}));
    assert_eq!((g["settings"]["page"]["pageNumbers"].as_bool(), g["settings"]["page"]["pageInfo"].as_bool()), (Some(true), Some(false)));
    assert_eq!(g["settings"]["page"]["background"], json!([255, 255, 255]), "a patch merges");
    fails(&mut s, "print.set", json!({"page": {"pageInfo": "yes"}}));
    fails(&mut s, "print.set", json!({"page": {"pageNumber": true}}));
    // a template keeps them
    run(&mut s, "print.template.save", json!({"name": "Marked"}));
    run(&mut s, "print.reset", json!({}));
    assert!(!s.print.page.crop_marks);
    run(&mut s, "print.template.apply", json!({"name": "Marked"}));
    assert!(s.print.page.page_numbers && s.print.page.crop_marks && !s.print.page.page_info);
    let set = crate::find_command("print.set").unwrap();
    assert!(set.params.contains("pageNumbers") && set.params.contains("pageInfo") && set.params.contains("cropMarks"), "{}", set.params);
}
