//! Studio layout and large-library navigation regressions.
use crate::{
    LightcraftApp, Services,
    headless::Headless,
    state::{RightPanel, ViewMode},
};
use lightcraft_catalog::{Op, Photo, PhotoId, Source};
use serde_json::json;
use std::time::Duration;

fn studio(size: [f32; 2], count: u64) -> Headless {
    let mut session = lightcraft_engine::Session::new();
    for i in 1..=count {
        let photo = Photo::new(
            PhotoId(i),
            Source::File { path: format!("/studio-test/{i}.jpg") },
            &format!("Foto {i}.jpg"),
            "JPEG",
            6000,
            4000,
            "2026-10-22T12:00:00",
        );
        session.catalog.apply(Op::AddPhoto { photo: Box::new(photo) }).unwrap();
    }
    let mut app = LightcraftApp::new(session, Services { png: None, ..Default::default() });
    app.ui.mm_studio = true;
    app.ui.show_home = false;
    app.ui.view = ViewMode::Detail;
    app.ui.right = RightPanel::Edit;
    app.ui.left_panel = true;
    app.ui.filmstrip = true;
    let mut h = Headless::new(app, size, 1.0);
    for _ in 0..5 {
        h.step();
    }
    h
}
fn widget(h: &Headless, id: &str) -> egui::Rect {
    h.app.widgets.iter().find(|(name, _)| name == id).map(|(_, r)| *r).unwrap_or_else(|| panic!("missing {id}"))
}
#[test]
fn studio_virtualizes_a_large_library_and_follows_selection() {
    let mut h = studio([1600.0, 1000.0], 3000);
    let cells = h.app.widgets.iter().filter(|(name, _)| name.starts_with("film:")).count();
    assert!(cells > 0 && cells < 15, "drew {cells} cells for 3000 photos");
    let side = widget(&h, "panel:left_panel");
    let film = widget(&h, "panel:bfilmstrip");
    let right = widget(&h, "panel:right_panel");
    for (id, rect) in &h.app.widgets {
        if id.starts_with("slider:") || id.starts_with("section:") {
            assert!(rect.right() <= right.right(), "{id} overflows the edit panel: {rect:?}");
        }
    }

    assert!(side.right() <= film.left());
    assert!(film.right() < right.left());
    let r = h.request("engine.execute", json!({"command":"library.select","params":{"ids":[3000]}}), Duration::from_secs(20));
    assert_eq!(r["ok"], true, "{r}");
    for _ in 0..4 {
        h.step();
    }
    assert!(film.intersects(widget(&h, "film:3000")), "keyboard selection follows into view");
    assert!(h.app.widgets.iter().filter(|(name, _)| name.starts_with("film:")).count() < 15);
}
#[test]
fn studio_keeps_navigation_and_canvas_clear_on_a_laptop() {
    let h = studio([900.0, 700.0], 20);
    assert!(!h.app.widgets.iter().any(|(name, _)| name == "panel:left_panel"));
    let film = widget(&h, "panel:bfilmstrip");
    let right = widget(&h, "panel:right_panel");
    for (id, rect) in &h.app.widgets {
        if id.starts_with("slider:") || id.starts_with("section:") {
            assert!(rect.right() <= right.right(), "{id} overflows the edit panel: {rect:?}");
        }
    }

    assert!(right.left() - film.left() >= 300.0);
    assert!(film.right() <= right.left());
    let library = widget(&h, "button:navLib");
    let edit = widget(&h, "button:navEdit");
    let print = widget(&h, "button:navPrint");
    assert!(library.right() <= edit.left() && edit.right() <= print.left());
    assert!(print.right() <= widget(&h, "icon:studioMore").left());
}
#[test]
fn studio_empty_source_clears_active_photo() {
    let mut h = studio([1200.0, 800.0], 2);
    assert!(h.app.session.active().is_some());
    let r = h.request("engine.execute", json!({"command":"library.filter","params":{"text":"no-such-photo-xyz"}}), Duration::from_secs(20));
    assert_eq!(r["ok"], true);
    for _ in 0..3 {
        h.step();
    }
    assert!(h.app.session.active().is_none());
}

#[test]
fn studio_light_slider_changes_settings_and_supports_undo() {
    let mut h = studio([1600.0, 1000.0], 2);
    h.app.ui.open_sections = vec!["light".into()];
    h.step();
    let id = h.app.session.active().unwrap();
    let before = h.app.session.develop_of(id).unwrap().light.exposure;
    let r = h.request("ui.clickWidget", json!({"id":"slider:light.exposure","fx":0.75,"fy":0.64}), Duration::from_secs(20));
    assert_eq!(r["ok"], true, "{r}");
    assert!(h.app.session.develop_of(id).unwrap().light.exposure > before);
    h.app.run("edit.undo", json!({})).unwrap();
    assert_eq!(h.app.session.develop_of(id).unwrap().light.exposure, before);
}

#[test]
fn studio_library_exposes_albums_on_a_laptop_and_f6_toggles_the_strip() {
    let mut h = studio([900.0, 700.0], 20);
    h.app.run("view.filmstrip", json!({})).unwrap();
    h.step();
    assert!(!h.app.widgets.iter().any(|(id, _)| id == "panel:bfilmstrip"));
    h.app.run("view.filmstrip", json!({})).unwrap();
    h.step();
    assert!(h.app.widgets.iter().any(|(id, _)| id == "panel:bfilmstrip"));
    let r = h.request("ui.clickWidget", json!({"id":"button:navLib"}), Duration::from_secs(20));
    assert_eq!(r["ok"], true, "{r}");
    h.step();
    assert!(h.app.widgets.iter().any(|(id, _)| id == "panel:left_panel"));
    assert!(!h.app.widgets.iter().any(|(id, _)| id == "panel:right_panel"));
    assert!(h.app.canvas_rect.unwrap().width() > 600.0);
}
