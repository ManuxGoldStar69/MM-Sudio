//! SmartSky — Photorealistic 4K AI-Assisted Sky Replacement (100 4K Skies across 7 categories).
//!
//! Categories (100 total):
//! - `Clear` (`1..=2`): 2 Cielos Azules sin Nubes
//! - `Clouds` (`3..=32`): 30 Cielos con Nubes
//! - `Overcast` (`33..=52`): 20 Cielos Muy Nublados
//! - `Rainbow` (`53..=62`): 10 Cielos con Arcoíris
//! - `Night` (`63..=70`): 8 Cielos de Noche con Estrellas
//! - `Psychedelic` (`71..=80`): 10 Cielos Psicodélicos / Aurora
//! - `Sunset` (`81..=100`): 20 Atardeceres de Puesta de Sol
//!
//! Powered by 4K (`3840 × 2160`) Nano Banana photographic sky master plates packed as RGBA8 `u32`
//! texels (`SKY_MAP_W × SKY_MAP_H`) plus a 2D image-adaptive + YuNet face/person-shielded sky matte
//! (`SKY_MATTE_W × SKY_MATTE_H`) and full-resolution per-pixel foliage/branch/hair edge matting in
//! both CPU (`finish.rs`) and GPU (`finish.wgsl`).

use std::sync::{Arc, LazyLock, Mutex};

use lightcraft_develop::SmartSky;
use lightcraft_raster::Rgb32f;

use crate::geometry::Frame;
use crate::nano_banana::{build_sky_packed_map, unpack_rgba8_f32};
pub use crate::sky_model::{SKY_MATTE_H, SKY_MATTE_W};

/// Native 4K UHD dimensions (`3840 × 2160`) for 4K sky textures in both CPU and GPU pipelines.
pub const SKY_4K_W: usize = 3840;
pub const SKY_4K_H: usize = 2160;

/// Resolution of the packed RGBA8 `u32` sky texture uploaded to `aux` (`3840 × 2160` 4K UHD).
pub const SKY_MAP_W: usize = 3840;
pub const SKY_MAP_H: usize = 2160;

/// Category of a 4K SmartSky preset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SkyCategory {
    Clear,
    Clouds,
    Overcast,
    Rainbow,
    Night,
    Psychedelic,
    Sunset,
}

impl SkyCategory {
    pub const ALL: [SkyCategory; 7] = [
        SkyCategory::Clear,
        SkyCategory::Clouds,
        SkyCategory::Overcast,
        SkyCategory::Rainbow,
        SkyCategory::Night,
        SkyCategory::Psychedelic,
        SkyCategory::Sunset,
    ];

    pub fn id(self) -> &'static str {
        match self {
            Self::Clear => "clear",
            Self::Clouds => "clouds",
            Self::Overcast => "overcast",
            Self::Rainbow => "rainbow",
            Self::Night => "night",
            Self::Psychedelic => "psychedelic",
            Self::Sunset => "sunset",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Clear => "☀️ Azul Despejado (2)",
            Self::Clouds => "⛅ Con Nubes (30)",
            Self::Overcast => "🌧️ Muy Nublado (20)",
            Self::Rainbow => "🌈 Con Arcoíris (10)",
            Self::Night => "🌌 Noche Estrellas (8)",
            Self::Psychedelic => "🔮 Psicodélicos (10)",
            Self::Sunset => "🌅 Atardeceres (20)",
        }
    }
}

/// Metadata for a single 4K SmartSky card in the UI carousel.
#[derive(Clone, Debug)]
pub struct SkySpec {
    /// 1-based preset index (`1..=100`).
    pub index: u16,
    pub id: String,
    pub name: String,
    pub subtitle: String,
    pub category: SkyCategory,
    /// Zenith sRGB color `[R, G, B]`.
    pub zenith_rgb: [u8; 3],
    /// Horizon sRGB color `[R, G, B]`.
    pub horizon_rgb: [u8; 3],
    /// Cloud / feature accent sRGB color `[R, G, B]`.
    pub cloud_rgb: [u8; 3],
}

static SKY_CATALOG: LazyLock<Vec<SkySpec>> = LazyLock::new(build_sky_catalog);

/// Returns the full catalog of 100 4K SmartSky presets.
pub fn catalog() -> &'static [SkySpec] {
    &SKY_CATALOG
}

fn build_sky_catalog() -> Vec<SkySpec> {
    let mut out = Vec::with_capacity(100);

    // 1. Cielos Azules sin Nubes (2): Presets 1..=2
    let clear_list: [(&str, &str, [u8; 3], [u8; 3], [u8; 3]); 2] = [
        (
            "Azul Cenital Puro",
            "Cielo despejado 4K de mediodía",
            [28, 92, 196],
            [148, 206, 250],
            [188, 226, 255],
        ),
        (
            "Azul Californiano",
            "Cielo limpio turquesa y brisa solar",
            [36, 114, 212],
            [176, 224, 252],
            [218, 242, 255],
        ),
    ];

    // 2. Cielos con Nubes (30): Presets 3..=32
    let clouds_list: [(&str, &str, [u8; 3], [u8; 3], [u8; 3]); 30] = [
        ("Cúmulos de Verano", "Algodón blanco sobre azul real", [42, 108, 204], [156, 208, 248], [250, 252, 255]),
        ("Cielo Nupcial Suave", "Nubes etéreas ideales para bodas", [58, 124, 212], [178, 218, 250], [255, 253, 250]),
        ("Brisa del Pacífico", "Cúmulos dispersos de buen tiempo", [38, 102, 198], [152, 204, 244], [246, 250, 255]),
        ("Cirros de Altura", "Pinceladas finas de viento alto", [44, 112, 208], [164, 214, 250], [242, 248, 255]),
        ("Tarde de Primavera", "Nubes luminosas con volumen suave", [52, 118, 210], [172, 216, 248], [252, 252, 255]),
        ("Altocúmulos de Plata", "Cielo empedrado con bordes brillantes", [48, 106, 194], [158, 202, 240], [244, 248, 255]),
        ("Valle Soleado", "Horizonte despejado con cúmulos altos", [36, 98, 192], [148, 202, 246], [248, 251, 255]),
        ("Cielo Toscano", "Azul cálido con nubes esculturales", [46, 110, 196], [168, 212, 242], [255, 250, 242]),
        ("Cúmulos Imperiales", "Grandes formaciones blancas en 4K", [34, 94, 188], [144, 198, 242], [252, 254, 255]),
        ("Mañana Alpina", "Aire cristalino con nubes de cumbre", [30, 88, 186], [138, 196, 244], [248, 252, 255]),
        ("Viento Alisio", "Nubes tropicales bajas en desfile", [40, 116, 208], [162, 218, 248], [250, 252, 255]),
        ("Luz de Mediodía", "Contraluz brillante entre cúmulos", [54, 122, 214], [182, 224, 252], [255, 255, 255]),
        ("Pradera Azul", "Horizonte amplio de nubes suaves", [44, 108, 200], [160, 210, 246], [248, 250, 255]),
        ("Plumas del Norte", "Cirros sedosos en diagonal", [50, 114, 204], [168, 214, 248], [244, 248, 255]),
        ("Cielo Mediterráneo", "Azul cobalto y nubes de espuma", [32, 96, 196], [148, 206, 248], [252, 254, 255]),
        ("Cúmulos Dorados", "Nubes de tarde con toque cálido", [56, 116, 198], [186, 214, 238], [255, 246, 232]),
        ("Horizonte Infinito", "Perspectiva profunda de nubes en capas", [42, 104, 196], [156, 206, 244], [248, 252, 255]),
        ("Cielo de Jardín", "Suavidad luminosa para retratos", [62, 128, 214], [184, 222, 250], [254, 254, 255]),
        ("Olas del Cielo", "Ondas nubosas iluminadas por el sol", [46, 112, 202], [162, 212, 246], [248, 250, 255]),
        ("Claridad Boreal", "Azul profundo con nubes nítidas", [28, 86, 182], [136, 192, 240], [246, 250, 255]),
        ("Bruma de Verano", "Cielo cálido con nubes algodonosas", [58, 120, 204], [182, 218, 244], [255, 250, 244]),
        ("Cielo de Viñedo", "Equilibrio clásico de azul y blanco", [44, 108, 198], [158, 208, 244], [250, 252, 255]),
        ("Destello entre Nubes", "Rayos suaves asomando tras cúmulos", [52, 118, 206], [178, 218, 246], [255, 252, 242]),
        ("Costa Esmeralda", "Cielo marino luminoso y fresco", [38, 114, 206], [164, 220, 248], [250, 254, 255]),
        ("Velo de Novia", "Cirros translúcidos delicados", [60, 126, 212], [182, 220, 250], [252, 253, 255]),
        ("Cúmulos del Oeste", "Grandes cielos abiertos de llanura", [36, 100, 194], [150, 204, 244], [248, 251, 255]),
        ("Tarde Serena", "Nubes tranquilas al caer la tarde", [54, 112, 196], [176, 210, 238], [252, 246, 238]),
        ("Cielo de Hacienda", "Nubes coloniales de alto contraste", [34, 92, 186], [146, 198, 240], [252, 254, 255]),
        ("Aire de Montaña", "Cúmulos frescos sobre azul intenso", [30, 90, 188], [142, 198, 244], [250, 252, 255]),
        ("Paraíso de Cúmulos", "Composición majestuosa de cielo 4K", [42, 106, 200], [158, 208, 246], [254, 255, 255]),
    ];

    // 3. Cielos Muy Nublados (20): Presets 33..=52
    let overcast_list: [(&str, &str, [u8; 3], [u8; 3], [u8; 3]); 20] = [
        ("Tormenta Nórdica", "Manto denso gris acero y plata", [68, 78, 94], [138, 148, 164], [188, 196, 210]),
        ("Estratos de Plata", "Cielo nublado elegante para retrato", [92, 102, 118], [164, 174, 188], [214, 220, 230]),
        ("Nimbos Dramáticos", "Nubes oscuras de tormenta inminente", [48, 56, 72], [112, 124, 142], [168, 178, 194]),
        ("Monzón de Terciopelo", "Capas profundas de lluvia pesada", [54, 66, 84], [122, 136, 154], [176, 186, 202]),
        ("Niebla de Tierras Altas", "Cielo cubierto brumoso escocés", [84, 94, 108], [154, 162, 174], [202, 208, 218]),
        ("Gris Editorial", "Fondo nublado neutro de alta moda", [98, 106, 118], [168, 176, 186], [218, 222, 228]),
        ("Tormenta en el Mar", "Cielo turbulento azul pizarra", [52, 68, 88], [118, 136, 158], [172, 188, 206]),
        ("Ruptura de Luz", "Rayo pálido abriéndose entre nimbos", [64, 74, 90], [148, 156, 168], [224, 222, 212]),
        ("Cielo de Invierno", "Cobertura fría y uniforme de enero", [78, 90, 108], [148, 160, 178], [198, 208, 222]),
        ("Ceniza y Plata", "Contraste dramático cinematográfico", [58, 64, 76], [128, 134, 146], [184, 190, 200]),
        ("Frente Frío", "Banda de nubes bajas amenazantes", [62, 74, 92], [132, 144, 162], [182, 192, 208]),
        ("Bruma Londinense", "Cielo perla suave y difuso", [104, 112, 124], [174, 182, 192], [222, 226, 234]),
        ("Trueno Lejano", "Nubes cargadas con matiz violeta", [56, 58, 78], [124, 126, 148], [178, 180, 202]),
        ("Cielo de Islandia", "Textura subártica intensa y fría", [60, 72, 88], [130, 144, 160], [186, 198, 212]),
        ("Lluvia de Otoño", "Nublado cálido con tono peltre", [82, 84, 92], [152, 154, 162], [204, 202, 198]),
        ("Manto de Plomo", "Pesadez atmosférica de gran carácter", [50, 58, 68], [114, 122, 134], [164, 172, 184]),
        ("Viento de Temporal", "Nubes bajas rápidas y desgarradas", [66, 76, 92], [136, 146, 162], [190, 198, 212]),
        ("Calma Pre-Tormenta", "Luz dorada pálida bajo cielo negro", [54, 62, 76], [142, 140, 136], [198, 192, 178]),
        ("Nublado Difuso Softbox", "Iluminación suave perfecta para piel", [108, 116, 128], [178, 186, 196], [228, 232, 238]),
        ("Abismo de Nubes", "Ondulaciones asperitas dramáticas", [46, 54, 68], [108, 118, 136], [162, 174, 192]),
    ];

    // 4. Cielos con Arcoíris (10): Presets 53..=62
    let rainbow_list: [(&str, &str, [u8; 3], [u8; 3], [u8; 3]); 10] = [
        ("Arcoíris Tras la Lluvia", "Arco completo vibrante entre cúmulos", [54, 108, 188], [158, 198, 236], [255, 168, 184]),
        ("Doble Arcoíris Real", "Arco primario y secundario en 4K", [48, 98, 178], [148, 192, 232], [255, 182, 164]),
        ("Arcoíris de Boda Pastel", "Cielo romántico suave con arco irisado", [68, 126, 206], [182, 216, 246], [255, 192, 208]),
        ("Arcoíris en Hora Dorada", "Cielo cálido de tarde con arco luminoso", [72, 104, 168], [218, 186, 158], [255, 196, 138]),
        ("Valle del Arcoíris", "Bruma iluminada y puente de colores", [58, 114, 194], [168, 208, 240], [168, 238, 214]),
        ("Sol y Lluvia", "Cielo dramático claro-oscuro con arcoíris", [52, 78, 124], [154, 178, 208], [255, 176, 152]),
        ("Arcoíris Matutino", "Frescura del alba con arco cristalino", [62, 122, 204], [176, 218, 248], [184, 228, 255]),
        ("Puente Esmeralda", "Cielo limpio con gran bóveda espectral", [44, 106, 196], [154, 204, 244], [152, 242, 198]),
        ("Arcoíris entre Nimbos", "Contraste épico sobre nubes grises", [58, 72, 98], [138, 152, 176], [255, 164, 148]),
        ("Cielo Bendecido", "Rayos solares y arcoíris nupcial", [64, 118, 198], [184, 214, 242], [255, 208, 168]),
    ];

    // 5. Cielos de Noche con Estrellas (8): Presets 63..=70
    let night_list: [(&str, &str, [u8; 3], [u8; 3], [u8; 3]); 8] = [
        ("Vía Láctea Galáctica", "Núcleo galáctico 4K y miles de estrellas", [8, 12, 28], [28, 38, 68], [188, 168, 232]),
        ("Noche Estrellada Pura", "Cielo nocturno profundo de alta montaña", [6, 10, 24], [20, 32, 58], [218, 228, 255]),
        ("Hora Azul Estelar", "Primeras estrellas en crepúsculo índigo", [16, 32, 68], [52, 84, 138], [224, 234, 255]),
        ("Noche Alpina", "Constelaciones nítidas y bruma azul", [10, 16, 36], [32, 48, 82], [204, 222, 255]),
        ("Lluvia de Meteoros", "Campo estelar con destellos fugaces", [8, 14, 30], [26, 42, 74], [238, 244, 255]),
        ("Desierto de Medianoche", "Cielo cálido-nocturno con polvo estelar", [12, 14, 28], [38, 42, 68], [236, 214, 188]),
        ("Noche Polar", "Estrellas diamantinas sobre horizonte frío", [6, 14, 32], [24, 48, 84], [184, 228, 255]),
        ("Terciopelo Astral", "Noche romántica con nebulosa sutil", [10, 12, 30], [34, 38, 72], [214, 196, 248]),
    ];

    // 6. Cielos Psicodélicos / Aurora / Fantasía (10): Presets 71..=80
    let psych_list: [(&str, &str, [u8; 3], [u8; 3], [u8; 3]); 10] = [
        ("Aurora Boreal Esmeralda", "Cortinas verdes y violetas árticas", [12, 28, 48], [38, 168, 138], [168, 96, 244]),
        ("Horizonte Synthwave", "Neón magenta, cian y violeta eléctrico", [28, 12, 64], [244, 62, 168], [78, 228, 255]),
        ("Nebulosa Cósmica", "Fantasía astral púrpura y oro rosa", [24, 14, 52], [148, 64, 184], [255, 168, 204]),
        ("Prisma de Ópalo", "Nubes iridiscentes menta, rosa y lavanda", [68, 92, 188], [248, 168, 214], [142, 244, 228]),
        ("Sueño Ultravioleta", "Cielo surrealista amatista y coral", [38, 18, 84], [188, 78, 212], [255, 138, 124]),
        ("Nubes Bioluminiscentes", "Resplandor eléctrico cian en la noche", [10, 22, 46], [34, 118, 188], [96, 248, 238]),
        ("Fantasía de Dos Soles", "Cielo de otro mundo ámbar y turquesa", [28, 88, 142], [248, 148, 78], [138, 238, 224]),
        ("Oro Líquido y Cian", "Ondas psicodélicas de alto contraste", [18, 58, 96], [238, 178, 68], [88, 224, 236]),
        ("Cortina Boreal Rosa", "Aurora boreal fucsia y verde menta", [18, 20, 48], [218, 78, 164], [118, 242, 188]),
        ("Cielo Ciber-Rosa", "Atardecer futurista holográfico", [42, 24, 88], [255, 96, 158], [148, 218, 255]),
    ];

    // 7. Atardeceres de Puesta de Sol (20): Presets 81..=100
    let sunset_list: [(&str, &str, [u8; 3], [u8; 3], [u8; 3]); 20] = [
        ("Hora Dorada Real", "Explosión cálida de oro y ámbar en 4K", [58, 84, 148], [255, 176, 78], [255, 224, 148]),
        ("Ocaso en Santorini", "Carmesí, oro y azul egeo profundo", [42, 62, 128], [248, 112, 74], [255, 196, 128]),
        ("Atardecer Californiano", "Tonos durazno, coral y oro suave", [68, 92, 156], [255, 154, 108], [255, 212, 168]),
        ("Horizonte Toscano", "Ámbar miel sobre colinas doradas", [64, 88, 142], [248, 162, 68], [255, 218, 142]),
        ("Crepúsculo Rosa y Oro", "Cielo nupcial romántico de ensueño", [78, 88, 152], [252, 148, 138], [255, 208, 176]),
        ("Fuego del Pacífico", "Nubes encendidas en rojo y mandarina", [44, 52, 104], [244, 84, 52], [255, 168, 78]),
        ("Brisa de Maldivas", "Degradado coral, malva y turquesa", [48, 86, 148], [252, 138, 118], [255, 198, 164]),
        ("Sol de Medianoche", "Resplandor bajo infinito de oro viejo", [52, 74, 126], [242, 156, 72], [255, 214, 138]),
        ("Afterglow Magenta", "Última luz violeta y fucsia tras el ocaso", [38, 42, 96], [218, 88, 128], [255, 164, 158]),
        ("Cobre del Desierto", "Cielo limpio cobrizo y bronce cálido", [62, 76, 124], [238, 128, 64], [255, 194, 128]),
        ("Pastel de Venecia", "Acuarela suave de albaricoque y lavanda", [84, 98, 158], [252, 178, 148], [255, 224, 196]),
        ("Puesta de Sol Épica", "Cúmulos dramáticos bañados en oro", [46, 64, 118], [248, 134, 58], [255, 202, 118]),
        ("Cielo de champán", "Atardecer luminoso elegante para bodas", [74, 96, 154], [255, 192, 128], [255, 232, 184]),
        ("Brasas del Ocaso", "Nubes bajas con borde incandescente", [36, 46, 88], [232, 76, 48], [255, 158, 72]),
        ("Alba de Oro Rosa", "Transición delicada de perla a coral", [72, 92, 152], [250, 164, 142], [255, 218, 192]),
        ("Atardecer en Acapulco", "Cielo tropical intenso naranja y púrpura", [48, 54, 112], [252, 118, 56], [255, 188, 92]),
        ("Luz de Catedral", "Rayos crepusculares dorados en el horizonte", [56, 78, 134], [252, 168, 74], [255, 222, 144]),
        ("Terciopelo Carmesí", "Ocaso profundo de cine", [34, 40, 84], [214, 68, 64], [252, 148, 96]),
        ("Suavidad de Verano", "Brillo cálido sereno de fin de tarde", [66, 94, 156], [252, 174, 104], [255, 224, 164]),
        ("Gran Finale Dorado", "El atardecer definitivo de gala en 4K", [48, 68, 128], [255, 148, 64], [255, 214, 128]),
    ];

    let groups: [(SkyCategory, &[(&str, &str, [u8; 3], [u8; 3], [u8; 3])]); 7] = [
        (SkyCategory::Clear, &clear_list),
        (SkyCategory::Clouds, &clouds_list),
        (SkyCategory::Overcast, &overcast_list),
        (SkyCategory::Rainbow, &rainbow_list),
        (SkyCategory::Night, &night_list),
        (SkyCategory::Psychedelic, &psych_list),
        (SkyCategory::Sunset, &sunset_list),
    ];

    let mut idx = 1u16;
    for (cat, list) in groups {
        for (i, (name, sub, zen, hor, cld)) in list.iter().enumerate() {
            out.push(SkySpec {
                index: idx,
                id: format!("sky.{}.{:02}", cat.id(), i + 1),
                name: (*name).to_string(),
                subtitle: (*sub).to_string(),
                category: cat,
                zenith_rgb: *zen,
                horizon_rgb: *hor,
                cloud_rgb: *cld,
            });
            idx += 1;
        }
    }
    out
}

#[inline]
fn sstep(e0: f32, e1: f32, x: f32) -> f32 {
    if e0 >= e1 {
        return if x < e1 { 0.0 } else { 1.0 };
    }
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Generates a full 4K (`3840 × 2160`) RGBA8 buffer (`3840 * 2160 * 4` bytes) for `preset` (`1..=100`).
pub fn render_sky_4k_rgba8(preset: u16) -> Vec<u8> {
    let idx = (preset.saturating_sub(1) as usize).min(catalog().len().saturating_sub(1));
    let spec = &catalog()[idx];
    let packed = build_sky_packed_map(spec, SKY_4K_W, SKY_4K_H, 0.0);
    let mut out = vec![0u8; SKY_4K_W * SKY_4K_H * 4];
    for (i, &w) in packed.iter().enumerate() {
        let u = w.to_bits();
        let o = i * 4;
        out[o] = (u & 0xFF) as u8;
        out[o + 1] = ((u >> 8) & 0xFF) as u8;
        out[o + 2] = ((u >> 16) & 0xFF) as u8;
        out[o + 3] = 255;
    }
    out
}

type SkyCacheEntry = ((u16, i32), Arc<Vec<f32>>);

static SKY_CACHE: LazyLock<Mutex<Vec<SkyCacheEntry>>> = LazyLock::new(|| Mutex::new(Vec::new()));

/// Returns the cached 4K (`3840 × 2160`) packed RGBA8 `f32` sky texture for `preset` (`1..=100`).
pub fn sky_map_cached(preset: u16, defocus: f64) -> Option<Arc<Vec<f32>>> {
    if !(1..=100).contains(&preset) {
        return None;
    }
    let d_key = (defocus.clamp(0.0, 100.0) * 0.25).round() as i32;
    let key = (preset, d_key);
    if let Ok(guard) = SKY_CACHE.lock()
        && let Some((_, map)) = guard.iter().find(|(k, _)| *k == key)
    {
        return Some(map.clone());
    }
    let spec = &catalog()[(preset - 1) as usize];
    let map = Arc::new(build_sky_packed_map(spec, SKY_MAP_W, SKY_MAP_H, defocus as f32));
    if let Ok(mut guard) = SKY_CACHE.lock() {
        if guard.len() >= 4 {
            guard.remove(0);
        }
        guard.push((key, map.clone()));
    }
    Some(map)
}

/// Default horizon-only fallback matte (`640 × 480`) used before `prepare_sky_matte` attaches the photo's neural matte.
fn default_horizon_matte(horizon_y: f32, feather: f32) -> Arc<Vec<f32>> {
    let mut m = vec![0.0f32; SKY_MATTE_W * SKY_MATTE_H];
    for gy in 0..SKY_MATTE_H {
        let ny = (gy as f32 + 0.5) / SKY_MATTE_H as f32;
        let v = 1.0 - sstep(horizon_y - feather * 0.55, horizon_y + feather * 0.55, ny);
        for gx in 0..SKY_MATTE_W {
            m[gy * SKY_MATTE_W + gx] = v;
        }
    }
    Arc::new(m)
}

/// Computes the neural sky matte (`640 × 480`) via `sky_model::compute_neural_sky_matte`.
pub fn compute_photo_sky_matte(
    img: &Rgb32f,
    frame: &Frame,
    ev: f32,
    _horizon_y: f32,
    _feather: f32,
) -> Arc<Vec<f32>> {
    crate::sky_model::compute_neural_sky_matte(img, frame, ev).matte
}

/// Resolved per-render plan for SmartSky replacement, shared between CPU (`finish.rs`) and GPU
/// (`params.rs` / `finish.wgsl`).
#[derive(Clone, Debug)]
pub struct SmartSkyPlan {
    /// Packed RGBA8 `u32` (stored as `f32::from_bits`) 4K sky texture (`SKY_MAP_W × SKY_MAP_H`).
    pub map: Arc<Vec<f32>>,
    /// 2D neural + RGB-guided photo sky matte (`SKY_MATTE_W × SKY_MATTE_H = 640 × 480`).
    pub matte: Arc<Vec<f32>>,
    pub preset: u32,
    pub category_id: u32,
    pub seed: u32,
    pub amount: f32,
    pub horizon_y: f32,
    pub horizon_offset: f32,
    pub detected_horizon_y: f32,
    pub orig_sky_rgb: [f32; 3],
    pub feather: f32,
    pub edge_refine: f32,
    pub relight: f32,
    pub haze: f32,
    pub warmth_shift: f32,
    pub exp_gain: f32,
    pub inv_scale: f32,
    pub shift_x: f32,
    pub shift_y: f32,
    pub flip_x: bool,
    pub ambient_rgb: [f32; 3],
    pub horizon_rgb: [f32; 3],
}

impl SmartSkyPlan {
    pub fn from_settings(sky: &SmartSky) -> Option<Self> {
        if !sky.is_active() {
            return None;
        }
        let preset = sky.preset.round().clamp(1.0, 100.0) as u16;
        let map = sky_map_cached(preset, sky.defocus)?;
        let amount = (sky.amount / 100.0).clamp(0.0, 1.0) as f32;
        if amount <= 0.0 {
            return None;
        }
        let spec = &catalog()[(preset - 1) as usize];
        let warmth_shift = (sky.warmth as f32 / 100.0).clamp(-1.0, 1.0);
        let exp_gain = 2.0f32.powf((sky.brightness as f32 / 100.0).clamp(-1.0, 1.0) * 0.68);

        let hor_srgb = [
            (spec.horizon_rgb[0] as f32 / 255.0 * (1.0 + 0.22 * warmth_shift) * exp_gain).clamp(0.0, 1.0),
            (spec.horizon_rgb[1] as f32 / 255.0 * (1.0 + 0.04 * warmth_shift) * exp_gain).clamp(0.0, 1.0),
            (spec.horizon_rgb[2] as f32 / 255.0 * (1.0 - 0.22 * warmth_shift) * exp_gain).clamp(0.0, 1.0),
        ];
        let zen_srgb = [
            (spec.zenith_rgb[0] as f32 / 255.0 * exp_gain).clamp(0.0, 1.0),
            (spec.zenith_rgb[1] as f32 / 255.0 * exp_gain).clamp(0.0, 1.0),
            (spec.zenith_rgb[2] as f32 / 255.0 * exp_gain).clamp(0.0, 1.0),
        ];
        let ambient_rgb = [
            hor_srgb[0] * 0.62 + zen_srgb[0] * 0.38,
            hor_srgb[1] * 0.62 + zen_srgb[1] * 0.38,
            hor_srgb[2] * 0.62 + zen_srgb[2] * 0.38,
        ];
        let category_id = match spec.category {
            SkyCategory::Clear => 0,
            SkyCategory::Clouds => 1,
            SkyCategory::Overcast => 2,
            SkyCategory::Rainbow => 3,
            SkyCategory::Night => 4,
            SkyCategory::Psychedelic => 5,
            SkyCategory::Sunset => 6,
        };
        let scale = (sky.scale / 100.0).clamp(0.5, 2.5) as f32;
        let horizon_offset = (sky.horizon_pos as f32 / 100.0).clamp(-1.0, 1.0) * 0.36;
        let detected_horizon_y = 0.56f32;
        let horizon_y = (detected_horizon_y + horizon_offset).clamp(0.18, 0.95);
        let feather = (0.04 + (sky.horizon_blend as f32 / 100.0) * 0.36).max(0.02);

        Some(Self {
            map,
            matte: default_horizon_matte(horizon_y, feather),
            preset: preset as u32,
            category_id,
            seed: (preset as u32).wrapping_mul(0x85EB_CA6B),
            amount,
            horizon_y,
            horizon_offset,
            detected_horizon_y,
            orig_sky_rgb: [0.90, 0.93, 0.96],
            feather,
            edge_refine: (sky.edge_refine / 100.0).clamp(0.0, 1.0) as f32,
            relight: (sky.relight / 100.0).clamp(0.0, 1.0) as f32,
            haze: (sky.haze / 100.0).clamp(0.0, 1.0) as f32,
            warmth_shift,
            exp_gain,
            inv_scale: 1.0 / scale,
            shift_x: (sky.pos_x / 100.0).clamp(-1.0, 1.0) as f32 * 0.45,
            shift_y: (sky.pos_y / 100.0).clamp(-1.0, 1.0) as f32 * 0.45,
            flip_x: sky.flip_x >= 0.5,
            ambient_rgb,
            horizon_rgb: hor_srgb,
        })
    }

    /// Attaches the neural (`skyseg.onnx` + `segformer_ade_b0.onnx`) + 3-channel RGB-guided
    /// sky matte (`640 × 480`), original sky reference color, and auto-detected horizon line.
    pub fn attach_photo_matte(&mut self, img: &Rgb32f, frame: &Frame, ev: f32) {
        let res = crate::sky_model::compute_neural_sky_matte(img, frame, ev);
        self.matte = res.matte;
        self.orig_sky_rgb = res.orig_sky_rgb;
        self.detected_horizon_y = res.detected_horizon_y;
        self.horizon_y = (res.detected_horizon_y + self.horizon_offset).clamp(0.18, 0.95);
    }

    #[inline]
    fn sample_matte(&self, nx: f32, ny: f32) -> f32 {
        let mw = SKY_MATTE_W as f32;
        let mh = SKY_MATTE_H as f32;
        let fx = (nx.clamp(0.0, 1.0) * (mw - 1.0)).clamp(0.0, mw - 1.0001);
        let fy = (ny.clamp(0.0, 1.0) * (mh - 1.0)).clamp(0.0, mh - 1.0001);
        let x0 = fx as usize;
        let y0 = fy as usize;
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;
        let i00 = y0 * SKY_MATTE_W + x0;
        let i10 = y0 * SKY_MATTE_W + x0 + 1;
        let i01 = (y0 + 1) * SKY_MATTE_W + x0;
        let i11 = (y0 + 1) * SKY_MATTE_W + x0 + 1;
        let top = self.matte[i00] * (1.0 - tx) + self.matte[i10] * tx;
        let bot = self.matte[i01] * (1.0 - tx) + self.matte[i11] * tx;
        top * (1.0 - ty) + bot * ty
    }

    #[inline]
    fn sample_sky_4k(&self, su: f32, sv: f32) -> [f32; 3] {
        let fw = SKY_MAP_W as f32;
        let fh = SKY_MAP_H as f32;
        let fx = (su.clamp(0.0, 1.0) * (fw - 1.0)).clamp(0.0, fw - 1.0001);
        let fy = (sv.clamp(0.0, 1.0) * (fh - 1.0)).clamp(0.0, fh - 1.0001);
        let x0 = fx as usize;
        let y0 = fy as usize;
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;
        let p00 = unpack_rgba8_f32(self.map[y0 * SKY_MAP_W + x0]);
        let p10 = unpack_rgba8_f32(self.map[y0 * SKY_MAP_W + x0 + 1]);
        let p01 = unpack_rgba8_f32(self.map[(y0 + 1) * SKY_MAP_W + x0]);
        let p11 = unpack_rgba8_f32(self.map[(y0 + 1) * SKY_MAP_W + x0 + 1]);
        let mut rgb = [0.0f32; 3];
        for c in 0..3 {
            let top = p00[c] * (1.0 - tx) + p10[c] * tx;
            let bot = p01[c] * (1.0 - tx) + p11[c] * tx;
            rgb[c] = top * (1.0 - ty) + bot * ty;
        }
        rgb[0] = (rgb[0] * (1.0 + 0.24 * self.warmth_shift) * self.exp_gain).clamp(0.0, 1.0);
        rgb[1] = (rgb[1] * (1.0 + 0.04 * self.warmth_shift) * self.exp_gain).clamp(0.0, 1.0);
        rgb[2] = (rgb[2] * (1.0 - 0.24 * self.warmth_shift) * self.exp_gain).clamp(0.0, 1.0);
        rgb
    }

    /// Evaluates full-resolution optical sky matting, old-sky color decontamination (defringing),
    /// auto-horizon 4K Nano Banana sky sampling, aerial perspective, and skin/white-safe relighting.
    /// Mirrors the `F_SKY` stage in `finish.wgsl`.
    #[inline]
    pub fn apply_srgb(&self, e: [f32; 3], nx: f32, ny: f32, fine_det: f32) -> [f32; 3] {
        let r0 = e[0].clamp(0.0, 1.0);
        let g0 = e[1].clamp(0.0, 1.0);
        let b0 = e[2].clamp(0.0, 1.0);
        let lum = 0.2126 * r0 + 0.7152 * g0 + 0.0722 * b0;
        let mx = r0.max(g0).max(b0);
        let mn = r0.min(g0).min(b0);
        let sat = (mx - mn) / mx.max(1e-4);

        // 1. Sample 640 × 480 RGB-guided neural sky matte
        let mut coarse_matte = self.sample_matte(nx, ny);
        if self.horizon_offset.abs() > 0.005 {
            let user_v = 1.0 - sstep(self.horizon_y - self.feather * 0.55, self.horizon_y + self.feather * 0.55, ny);
            coarse_matte *= user_v;
        }

        // 2. Native-resolution (full-res pixel) optical refinement along foliage, hair & horizon boundaries
        let refined_matte = if coarse_matte <= 0.002 {
            0.0
        } else if coarse_matte >= 0.995 {
            1.0
        } else {
            let os = self.orig_sky_rgb;
            let os_lum = (0.2126 * os[0] + 0.7152 * os[1] + 0.0722 * os[2]).max(0.25);
            let d_orig = ((r0 - os[0]).powi(2) + (g0 - os[1]).powi(2) + (b0 - os[2]).powi(2)).sqrt();

            // Native pixels matching the original sky color (`d_orig` small) inside the transition band
            // are fine sky gaps between pine needles/hair or bright white sky bokeh rims: pull alpha up.
            // Native pixels that are sharply darker than their 1-px neighborhood (`fine_det < 0`) are fine twigs/hair strands.
            let sky_sim = 1.0 - sstep(0.04, 0.26, d_orig);
            let dark_strand = 1.0 - sstep(0.18, 0.48, lum / os_lum);
            let blend_k = 4.0 * coarse_matte * (1.0 - coarse_matte) * (0.55 + 0.45 * self.edge_refine);

            let mut refined = coarse_matte.max(coarse_matte * 0.35 + sky_sim * 0.65 * sstep(0.08, 0.45, coarse_matte));
            refined *= 1.0 - dark_strand * (1.0 - coarse_matte) * 0.45 * self.edge_refine;
            let edge_delta = (fine_det * 0.55).clamp(-0.24, 0.24) * self.edge_refine * blend_k;
            (refined + edge_delta).clamp(0.0, 1.0)
        };
        let sky_alpha = (refined_matte * self.amount).clamp(0.0, 1.0);

        // 3. Sample the 4K SmartSky texture aligned to the photo's auto-detected horizon (`self.horizon_y`)
        let mut sky_rgb = if sky_alpha > 0.001 || coarse_matte > 0.005 {
            let mut su = ((nx - 0.5 - self.shift_x) * self.inv_scale + 0.5).clamp(0.0, 1.0);
            if self.flip_x {
                su = 1.0 - su;
            }
            let horizon_denom = (self.horizon_y + 0.04).max(0.22);
            let sv = (((ny - self.shift_y) / horizon_denom - 0.5) * self.inv_scale + 0.5).clamp(0.0, 0.985);
            self.sample_sky_4k(su, sv)
        } else {
            [0.0; 3]
        };

        // Natural aerial perspective / horizon mist as the sky approaches the horizon line
        if sky_alpha > 0.001 || coarse_matte > 0.005 {
            let t_hor = sstep(self.horizon_y - self.feather * 0.90, self.horizon_y + 0.04, ny);
            let mist_w = (t_hor * (0.18 + 0.42 * self.haze)).clamp(0.0, 0.68);
            for c in 0..3 {
                sky_rgb[c] = sky_rgb[c] * (1.0 - mist_w) + self.horizon_rgb[c] * mist_w;
            }
        }

        // 4. Optical Foreground Color Decontamination (Un-Premultiplying Old Sky Glare / Defringe):
        // Camera boundary pixels satisfy `I = m * S_old + (1 - m) * F_true`.
        // Solving for `F_true = (I - spill * S_old) / (1 - spill)` removes the bright white/cyan
        // halo of the original overcast sky from blurry green treetops, leaves, and hair strands!
        let mut fg = [r0, g0, b0];
        if refined_matte > 0.003 && refined_matte < 0.998 {
            let os = self.orig_sky_rgb;
            let os_lum = (0.2126 * os[0] + 0.7152 * os[1] + 0.0722 * os[2]).max(0.28);
            // If the local sky near the horizon is even brighter than average `os`, adapt `os_eff` upward on neutral sky rims
            let local_boost = if sat < 0.16 && lum > os_lum { lum / os_lum } else { 1.0 };
            let spill = (refined_matte * (0.85 + 0.15 * self.edge_refine)).min(0.94);
            let inv_rem = 1.0 / (1.0 - spill).max(0.06);
            for c in 0..3 {
                let os_c = (os[c] * local_boost).min(0.99);
                let clean_c = ((fg[c] - spill * os_c) * inv_rem).clamp(0.0, fg[c]);
                fg[c] = clean_c;
            }
            // Harmonize any residual wide-aperture bokeh bloom or low-contrast boundary rim toward `sky_rgb`
            let fg_lum = 0.2126 * fg[0] + 0.7152 * fg[1] + 0.0722 * fg[2];
            let bokeh_edge = 4.0 * refined_matte * (1.0 - refined_matte);
            let rim_factor = bokeh_edge.max(sstep(0.15, 0.75, refined_matte) * (1.0 - sat * 0.8));
            let bloom_harmonize = (rim_factor * (0.48 + 0.42 * self.edge_refine) * self.amount).clamp(0.0, 0.85);
            let rel_lum = (fg_lum / os_lum).clamp(0.16, 1.0);
            for c in 0..3 {
                // Preserve natural green leaf chroma while eliminating any white/cyan sky rim above `sky_rgb`
                let target_c = (sky_rgb[c] * rel_lum).max(fg[c].min(sky_rgb[c]));
                fg[c] = fg[c] * (1.0 - bloom_harmonize) + target_c * bloom_harmonize;
            }
        }

        // 5. Skin-safe & white-garment-safe Foreground Relighting
        let fg_weight = (1.0 - sky_alpha) * self.relight;
        if fg_weight > 0.01 {
            let white_protect = sstep(0.52, 0.76, lum) * (1.0 - sstep(0.08, 0.22, sat));
            let skin_protect = if r0 > g0 && g0 > b0 {
                sstep(0.03, 0.12, r0 - b0) * sstep(0.20, 0.40, lum) * (1.0 - sstep(0.78, 0.92, lum))
            } else {
                0.0
            };
            let protect = white_protect.max(skin_protect * 0.85);
            let chroma_str = if self.category_id >= 4 { 0.46 } else { 0.12 };
            let amb_lum = 0.2126 * self.ambient_rgb[0] + 0.7152 * self.ambient_rgb[1] + 0.0722 * self.ambient_rgb[2];
            let lum_scale = if self.category_id == 4 {
                (0.78 + 0.42 * amb_lum).clamp(0.70, 1.02)
            } else {
                (0.90 + 0.22 * amb_lum).clamp(0.86, 1.06)
            };
            let tone_env = 0.30 + 0.70 * (1.0 - (lum - 0.45).abs() * 1.5).clamp(0.0, 1.0);
            let relight_k = fg_weight * tone_env * (1.0 - protect * 0.88) * 0.38;
            for c in 0..3 {
                let amb_c = amb_lum + (self.ambient_rgb[c] - amb_lum) * chroma_str;
                let tinted = fg[c] * lum_scale * (0.74 + 0.52 * amb_c);
                fg[c] = (fg[c] * (1.0 - relight_k) + tinted * relight_k).clamp(0.0, 1.0);
            }
        }

        // 6. Optical composite of decontaminated foreground and 4K sky
        [
            (fg[0] * (1.0 - sky_alpha) + sky_rgb[0] * sky_alpha).clamp(0.0, 1.0),
            (fg[1] * (1.0 - sky_alpha) + sky_rgb[1] * sky_alpha).clamp(0.0, 1.0),
            (fg[2] * (1.0 - sky_alpha) + sky_rgb[2] * sky_alpha).clamp(0.0, 1.0),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_has_100_skies_with_exact_category_counts() {
        let cat = catalog();
        assert_eq!(cat.len(), 100);
        let count = |c: SkyCategory| cat.iter().filter(|s| s.category == c).count();
        assert_eq!(count(SkyCategory::Clear), 2);
        assert_eq!(count(SkyCategory::Clouds), 30);
        assert_eq!(count(SkyCategory::Overcast), 20);
        assert_eq!(count(SkyCategory::Rainbow), 10);
        assert_eq!(count(SkyCategory::Night), 8);
        assert_eq!(count(SkyCategory::Psychedelic), 10);
        assert_eq!(count(SkyCategory::Sunset), 20);
    }

    #[test]
    fn nano_banana_4k_plates_and_sky_matte_work() {
        // 1. Verify 4K SmartSky texture (3840 × 2160)
        let sky_4k = sky_map_cached(3, 0.0).unwrap_or_default();
        assert_eq!(sky_4k.len(), SKY_4K_W * SKY_4K_H);

        // 2. Verify 4K Nature Overlays (Birds 151, Rainbows 161, SunGlow 171, Clouds 181)
        for p in [151u16, 161, 171, 181] {
            let (ovl_4k, w, h, _mode) = crate::overlays::overlay_map_cached(p, 0.0).unwrap_or_else(|| (Arc::new(Vec::new()), 0, 0, 0));
            assert_eq!(w as usize, SKY_4K_W);
            assert_eq!(h as usize, SKY_4K_H);
            assert_eq!(ovl_4k.len(), SKY_4K_W * SKY_4K_H);
        }

        // 3. Verify birds overlay has non-zero alpha bird pixels and transparent background pixels
        let (birds_map, bw, bh, bmode) = crate::overlays::overlay_map_cached(151, 0.0).unwrap_or_else(|| (Arc::new(Vec::new()), 0, 0, 0));
        assert_eq!(bmode, 1);
        let opaque_birds = birds_map.iter().filter(|&&w| crate::nano_banana::unpack_rgba8_f32(w)[3] > 0.5).count();
        let transparent_bg = birds_map.iter().filter(|&&w| crate::nano_banana::unpack_rgba8_f32(w)[3] < 0.01).count();
        assert!(opaque_birds > 5_000, "expected thousands of 4K bird pixels, got {opaque_birds}");
        assert!(transparent_bg > (bw as usize * bh as usize) / 2, "expected clean transparent sky around birds");
    }

    #[test]
    fn visual_test_three_wedding_photos() {
        let scratch = std::path::Path::new("/Users/mcruz/.gemini/antigravity/brain/2a72a23a-97b6-4e22-adce-8754ec739c2e/scratch");
        if !scratch.exists() {
            return;
        }
        let cases: &[(&str, u16)] = &[
            ("P1022243", 82), // Ocaso en Santorini
            ("P1012122", 3),  // Cúmulos de Verano (green trees!)
            ("P1022226", 20), // Cielo de Hacienda
        ];
        for &(stem, preset) in cases {
            let in_path = scratch.join(format!("{stem}_orig.jpg"));
            let Ok(dyn_img) = image::open(&in_path) else {
                continue;
            };
            let rgb8 = dyn_img.to_rgb8();
            let w = rgb8.width() as usize;
            let h = rgb8.height() as usize;

            // Convert sRGB u8 -> linear Rec.2020 Rgb32f
            let to_rec2020 = lightcraft_color::SRGB.to_space(&lightcraft_color::REC2020);
            let mut rec2020 = Rgb32f::new(w, h);
            let mut srgb_buf = vec![[0.0f32; 3]; w * h];
            for y in 0..h {
                for x in 0..w {
                    let p = rgb8.get_pixel(x as u32, y as u32).0;
                    let srgb = [
                        p[0] as f32 / 255.0,
                        p[1] as f32 / 255.0,
                        p[2] as f32 / 255.0,
                    ];
                    srgb_buf[y * w + x] = srgb;
                    let srgb_to_lin = |v: f32| if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) };
                    let lin_srgb = [
                        srgb_to_lin(srgb[0]),
                        srgb_to_lin(srgb[1]),
                        srgb_to_lin(srgb[2]),
                    ];
                    let lin_2020 = to_rec2020.apply_f32(lin_srgb);
                    rec2020.data[y * w + x] = lin_2020;
                }
            }

            let mut sky_cfg = SmartSky::default();
            sky_cfg.preset = preset as f64;
            sky_cfg.amount = 100.0;

            let Some(mut plan) = SmartSkyPlan::from_settings(&sky_cfg) else {
                continue;
            };
            let frame = Frame::new(w, h, &lightcraft_develop::DevelopSettings::default(), false);
            plan.attach_photo_matte(&rec2020, &frame, 0.0);

            // Save matte visualization
            let mut matte_img = image::RgbImage::new(SKY_MATTE_W as u32, SKY_MATTE_H as u32);
            for my in 0..SKY_MATTE_H {
                for mx in 0..SKY_MATTE_W {
                    let v = (plan.matte[my * SKY_MATTE_W + mx].clamp(0.0, 1.0) * 255.0).round() as u8;
                    matte_img.put_pixel(mx as u32, my as u32, image::Rgb([v, v, v]));
                }
            }
            let _ = matte_img.save(scratch.join(format!("{stem}_matte.jpg")));

            // Render final composite
            let mut out_img = image::RgbImage::new(w as u32, h as u32);
            for y in 0..h {
                let ny = (y as f32 + 0.5) / h as f32;
                for x in 0..w {
                    let nx = (x as f32 + 0.5) / w as f32;
                    let c = srgb_buf[y * w + x];
                    let l = srgb_buf[y * w + x.saturating_sub(1)];
                    let r = srgb_buf[y * w + (x + 1).min(w - 1)];
                    let u = srgb_buf[y.saturating_sub(1) * w + x];
                    let d = srgb_buf[(y + 1).min(h - 1) * w + x];
                    let lum_c = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
                    let lum_n = 0.25 * (
                        (0.2126 * l[0] + 0.7152 * l[1] + 0.0722 * l[2]) +
                        (0.2126 * r[0] + 0.7152 * r[1] + 0.0722 * r[2]) +
                        (0.2126 * u[0] + 0.7152 * u[1] + 0.0722 * u[2]) +
                        (0.2126 * d[0] + 0.7152 * d[1] + 0.0722 * d[2])
                    );
                    let fine_det = lum_c - lum_n;
                    let out = plan.apply_srgb(c, nx, ny, fine_det);
                    out_img.put_pixel(
                        x as u32,
                        y as u32,
                        image::Rgb([
                            (out[0] * 255.0).round() as u8,
                            (out[1] * 255.0).round() as u8,
                            (out[2] * 255.0).round() as u8,
                        ]),
                    );
                }
            }
            let _ = out_img.save(scratch.join(format!("{stem}_smartsky.jpg")));
        }
    }
}
