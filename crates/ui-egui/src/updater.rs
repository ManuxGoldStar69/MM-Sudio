//! Auto-updater and on-demand update checker for MM Studio.
//! Checks both the official VPS endpoint (`https://fotos.manuelmendiola.com/api/mm-studio/update`)
//! and GitHub Releases (`https://api.github.com/repos/ManuxGoldStar69/MM-Sudio/releases/latest`),
//! downloads the platform installer (`.exe` on Windows, `.dmg` on macOS) with live progress,
//! and launches it with one click.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UpdateInfo {
    pub version: String,
    pub notes: String,
    pub download_url: String,
    pub asset_name: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum UpdateStatus {
    #[default]
    Idle,
    Checking,
    UpToDate,
    Available(UpdateInfo),
    Downloading {
        info: UpdateInfo,
        downloaded_bytes: u64,
        total_bytes: u64,
    },
    ReadyToInstall {
        version: String,
        installer_path: PathBuf,
    },
    Error(String),
}

struct UpdaterInner {
    status: UpdateStatus,
    cancel: Arc<AtomicBool>,
}

static STATE: LazyLock<Mutex<UpdaterInner>> = LazyLock::new(|| {
    Mutex::new(UpdaterInner {
        status: UpdateStatus::Idle,
        cancel: Arc::new(AtomicBool::new(false)),
    })
});

/// Current version of MM Studio compiled into the binary.
pub const CURRENT_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Returns a snapshot of the current update status.
pub fn status() -> UpdateStatus {
    STATE.lock().map(|g| g.status.clone()).unwrap_or(UpdateStatus::Idle)
}

/// Cancels an active update check or installer download.
pub fn cancel() {
    if let Ok(mut g) = STATE.lock() {
        g.cancel.store(true, Ordering::Relaxed);
        g.status = UpdateStatus::Idle;
    }
}

/// Parses a `"vX.Y.Z"` or `"X.Y.Z"` version string into `(major, minor, patch)`.
pub fn parse_semver(v: &str) -> Option<(u32, u32, u32)> {
    let clean = v.trim().trim_start_matches(['v', 'V']);
    let core = clean.split(['-', '+']).next().unwrap_or(clean);
    let mut parts = core.split('.');
    let major = parts.next()?.parse::<u32>().ok()?;
    let minor = parts.next().unwrap_or("0").parse::<u32>().ok()?;
    let patch = parts.next().unwrap_or("0").parse::<u32>().ok()?;
    Some((major, minor, patch))
}

/// Returns `true` if `candidate` is strictly newer than `current`.
pub fn is_newer_version(current: &str, candidate: &str) -> bool {
    match (parse_semver(current), parse_semver(candidate)) {
        (Some(cur), Some(cand)) => cand > cur,
        _ => false,
    }
}

/// Starts a background check for updates. If `manual` is false (startup check), network errors stay silent.
pub fn check_for_updates(manual: bool) {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = manual;
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let cancel_flag = Arc::new(AtomicBool::new(false));
        if let Ok(mut g) = STATE.lock() {
            if matches!(g.status, UpdateStatus::Checking | UpdateStatus::Downloading { .. }) {
                return;
            }
            g.cancel.store(true, Ordering::Relaxed);
            g.cancel = Arc::clone(&cancel_flag);
            g.status = UpdateStatus::Checking;
        }

        let _ = std::thread::Builder::new()
            .name("mm-studio-update-check".into())
            .spawn(move || {
                let outcome = fetch_latest_update_info(&cancel_flag);
                if cancel_flag.load(Ordering::Relaxed) {
                    return;
                }
                if let Ok(mut g) = STATE.lock() {
                    match outcome {
                        Ok(Some(info)) => {
                            if is_newer_version(CURRENT_VERSION, &info.version) {
                                g.status = UpdateStatus::Available(info);
                            } else {
                                g.status = UpdateStatus::UpToDate;
                            }
                        }
                        Ok(None) => {
                            g.status = UpdateStatus::UpToDate;
                        }
                        Err(err) => {
                            if manual {
                                g.status = UpdateStatus::Error(err);
                            } else {
                                g.status = UpdateStatus::Idle;
                            }
                        }
                    }
                }
            });
    }
}

/// Starts downloading the update installer in the background and launches it when complete.
pub fn start_download(info: UpdateInfo) {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = info;
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let cancel_flag = Arc::new(AtomicBool::new(false));
        if let Ok(mut g) = STATE.lock() {
            if matches!(g.status, UpdateStatus::Downloading { .. }) {
                return;
            }
            g.cancel.store(true, Ordering::Relaxed);
            g.cancel = Arc::clone(&cancel_flag);
            g.status = UpdateStatus::Downloading {
                info: info.clone(),
                downloaded_bytes: 0,
                total_bytes: 0,
            };
        }

        let _ = std::thread::Builder::new()
            .name("mm-studio-update-dl".into())
            .spawn(move || {
                match download_installer_file(&info, &cancel_flag) {
                    Ok(path) => {
                        if cancel_flag.load(Ordering::Relaxed) {
                            return;
                        }
                        if let Ok(mut g) = STATE.lock() {
                            g.status = UpdateStatus::ReadyToInstall {
                                version: info.version.clone(),
                                installer_path: path.clone(),
                            };
                        }
                        launch_installer(&path);
                    }
                    Err(err) => {
                        if cancel_flag.load(Ordering::Relaxed) {
                            return;
                        }
                        if let Ok(mut g) = STATE.lock() {
                            g.status = UpdateStatus::Error(err);
                        }
                    }
                }
            });
    }
}

/// Opens the downloaded `.exe` (Windows) or `.dmg` (macOS) installer using the OS handler.
pub fn launch_installer(path: &Path) {
    #[cfg(target_os = "windows")]
    {
        let _ = std::process::Command::new(path).spawn();
    }
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("open").arg(path).spawn();
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let _ = std::process::Command::new("xdg-open").arg(path).spawn();
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn fetch_json_following_redirects(url_str: &str, cancel: &AtomicBool) -> Result<(u16, serde_json::Value), String> {
    use lightcraft_fetch::http::{Limits, Url, get};
    use std::time::Duration;

    let limits = Limits {
        connect: Duration::from_secs(6),
        stall: Duration::from_secs(8),
        cancel,
    };
    let mut current = Url::parse(url_str).map_err(|e| e.to_string())?;
    for _ in 0..5 {
        let mut resp = get(
            &current,
            &[("Accept", "application/vnd.github+json, application/json".to_string())],
            &limits,
        )
        .map_err(|e| e.to_string())?;

        if matches!(resp.status, 301 | 302 | 303 | 307 | 308) {
            if let Some(loc) = resp.header("location") {
                current = current.join(loc).map_err(|e| e.to_string())?;
                continue;
            }
            return Err("Redirección sin encabezado Location".into());
        }

        if resp.status == 404 {
            return Ok((404, serde_json::Value::Null));
        }
        if !(200..300).contains(&resp.status) {
            return Err(format!("HTTP {}", resp.status));
        }

        let mut buf = Vec::new();
        let mut chunk = [0u8; 8192];
        loop {
            let n = resp.read(&mut chunk, &limits).map_err(|e| e.to_string())?;
            if n == 0 {
                break;
            }
            if let Some(slice) = chunk.get(..n) {
                buf.extend_from_slice(slice);
            }
            if buf.len() > 512 * 1024 {
                return Err("Respuesta demasiado grande".into());
            }
        }
        let val: serde_json::Value = serde_json::from_slice(&buf).map_err(|e| e.to_string())?;
        return Ok((resp.status, val));
    }
    Err("Demasiadas redirecciones".into())
}

#[cfg(not(target_arch = "wasm32"))]
fn fetch_latest_update_info(cancel: &AtomicBool) -> Result<Option<UpdateInfo>, String> {
    // 1. Check VPS endpoint first (allows configuring updates directly from the LensFolio admin panel)
    let vps_endpoints = [
        "https://fotos.manuelmendiola.com/api/mm-studio/update",
        "https://manuelmendiola.com/api/mm-studio/update",
    ];
    for ep in vps_endpoints {
        if let Ok((200, val)) = fetch_json_following_redirects(ep, cancel)
            && let Some(ver) = val.get("version").and_then(|v| v.as_str()).filter(|s| !s.is_empty())
        {
            let notes = val
                .get("notes")
                .and_then(|v| v.as_str())
                .unwrap_or("Mejoras de rendimiento y nuevas herramientas en MM Studio.")
                .to_string();
            let (url_key, default_name) = if cfg!(target_os = "windows") {
                ("windows_url", format!("MM-Studio-Setup-v{}.exe", ver.trim_start_matches('v')))
            } else {
                ("macos_url", format!("MM-Studio-macOS-v{}.dmg", ver.trim_start_matches('v')))
            };
            if let Some(dl_url) = val.get(url_key).and_then(|v| v.as_str()).filter(|s| s.starts_with("https://")) {
                return Ok(Some(UpdateInfo {
                    version: ver.trim_start_matches(['v', 'V']).to_string(),
                    notes,
                    download_url: dl_url.to_string(),
                    asset_name: default_name,
                }));
            }
        }
    }

    // 2. Check GitHub Releases API (`ManuxGoldStar69/MM-Sudio`)
    let gh_url = "https://api.github.com/repos/ManuxGoldStar69/MM-Sudio/releases/latest";
    match fetch_json_following_redirects(gh_url, cancel) {
        Ok((404, _)) => Ok(None),
        Ok((_, val)) => {
            let Some(tag) = val.get("tag_name").and_then(|v| v.as_str()) else {
                return Ok(None);
            };
            let version = tag.trim_start_matches(['v', 'V']).to_string();
            let notes = val
                .get("body")
                .and_then(|v| v.as_str())
                .filter(|s| !s.trim().is_empty())
                .unwrap_or("Nueva actualización oficial de MM Studio.")
                .to_string();

            let wanted_ext = if cfg!(target_os = "windows") { ".exe" } else { ".dmg" };
            let mut chosen_url = String::new();
            let mut chosen_name = String::new();

            if let Some(assets) = val.get("assets").and_then(|v| v.as_array()) {
                for asset in assets {
                    let name = asset.get("name").and_then(|v| v.as_str()).unwrap_or_default();
                    let dl = asset.get("browser_download_url").and_then(|v| v.as_str()).unwrap_or_default();
                    if name.to_ascii_lowercase().ends_with(wanted_ext) && dl.starts_with("https://") {
                        chosen_url = dl.to_string();
                        chosen_name = name.to_string();
                        break;
                    }
                }
            }

            if chosen_url.is_empty() {
                return Ok(None);
            }

            Ok(Some(UpdateInfo {
                version,
                notes,
                download_url: chosen_url,
                asset_name: chosen_name,
            }))
        }
        Err(_) => Err("No se pudo conectar con el servidor de actualizaciones. Verifica tu conexión a internet.".into()),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn download_installer_file(info: &UpdateInfo, cancel: &AtomicBool) -> Result<PathBuf, String> {
    use lightcraft_fetch::http::{Limits, Url, get};
    use std::io::Write;
    use std::time::Duration;

    let limits = Limits {
        connect: Duration::from_secs(15),
        stall: Duration::from_secs(30),
        cancel,
    };

    let mut current = Url::parse(&info.download_url).map_err(|e| e.to_string())?;
    let mut resp_opt = None;
    for _ in 0..8 {
        let r = get(&current, &[], &limits).map_err(|e| e.to_string())?;
        if matches!(r.status, 301 | 302 | 303 | 307 | 308) {
            if let Some(loc) = r.header("location") {
                current = current.join(loc).map_err(|e| e.to_string())?;
                continue;
            }
            return Err("Redirección de descarga inválida".into());
        }
        if !(200..300).contains(&r.status) {
            return Err(format!("Error al descargar instalador (HTTP {})", r.status));
        }
        resp_opt = Some(r);
        break;
    }
    let Some(mut resp) = resp_opt else {
        return Err("Demasiadas redirecciones al descargar".into());
    };

    let total_bytes = resp.content_length().unwrap_or(0);
    if let Ok(mut g) = STATE.lock() {
        g.status = UpdateStatus::Downloading {
            info: info.clone(),
            downloaded_bytes: 0,
            total_bytes,
        };
    }

    let safe_name: String = info
        .asset_name
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_' | ' '))
        .collect();
    let file_name = if safe_name.is_empty() {
        if cfg!(target_os = "windows") {
            "MM-Studio-Setup.exe".to_string()
        } else {
            "MM-Studio-macOS.dmg".to_string()
        }
    } else {
        safe_name
    };

    let dest_dir = user_downloads_dir().unwrap_or_else(std::env::temp_dir);
    let _ = std::fs::create_dir_all(&dest_dir);
    let final_path = dest_dir.join(&file_name);
    let part_path = dest_dir.join(format!("{file_name}.part"));

    let mut file = std::fs::File::create(&part_path).map_err(|e| format!("No se pudo crear archivo temporal: {e}"))?;
    let mut downloaded: u64 = 0;
    let mut buf = [0u8; 64 * 1024];

    loop {
        if cancel.load(Ordering::Relaxed) {
            let _ = std::fs::remove_file(&part_path);
            return Err("Descarga cancelada".into());
        }
        let n = resp.read(&mut buf, &limits).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        if let Some(slice) = buf.get(..n) {
            file.write_all(slice).map_err(|e| e.to_string())?;
        }
        downloaded = downloaded.saturating_add(n as u64);
        if let Ok(mut g) = STATE.lock() {
            g.status = UpdateStatus::Downloading {
                info: info.clone(),
                downloaded_bytes: downloaded,
                total_bytes,
            };
        }
    }
    file.flush().map_err(|e| e.to_string())?;
    drop(file);

    std::fs::rename(&part_path, &final_path).map_err(|e| format!("No se pudo guardar el instalador: {e}"))?;
    Ok(final_path)
}

#[cfg(not(target_arch = "wasm32"))]
fn user_downloads_dir() -> Option<PathBuf> {
    let home = std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" })?;
    let d = PathBuf::from(home).join("Downloads");
    if d.is_dir() { Some(d) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semver_comparison_works() {
        assert_eq!(parse_semver("v0.13.0"), Some((0, 13, 0)));
        assert_eq!(parse_semver("0.14.2-beta"), Some((0, 14, 2)));
        assert!(is_newer_version("0.13.0", "0.13.1"));
        assert!(is_newer_version("0.13.0", "v0.14.0"));
        assert!(is_newer_version("0.13.0", "1.0.0"));
        assert!(!is_newer_version("0.13.0", "0.13.0"));
        assert!(!is_newer_version("0.13.0", "0.12.9"));
    }
}
