//! High-resolution optical & atmospheric Overlays (5 categories × 30 presets = 150 overlays).
//!
//! Each overlay is synthesized into a cached `OVL_SIDE × OVL_SIDE` linear RGB field and sampled
//! bilinearly in both the CPU per-pixel stage (`finish.rs`) and the GPU compute kernel
//! (`finish.wgsl`) so interactive placement, rotation, scaling, and blending run at 60 FPS with
//! CPU/GPU parity.

use std::sync::{Arc, LazyLock, Mutex};

use lightcraft_develop::CreativeOverlay;

/// Resolution of the pre-baked optical overlay map per axis (`192 × 192 × 3` floats).
pub const OVL_SIDE: usize = 192;

/// Category of an optical overlay in the Effects -> Overlays sub-area.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OverlayCategory {
    LightLeaks,
    Bokeh,
    Sunbeams,
    Prism,
    CinemaWeather,
}

impl OverlayCategory {
    pub const ALL: [OverlayCategory; 5] = [
        OverlayCategory::LightLeaks,
        OverlayCategory::Bokeh,
        OverlayCategory::Sunbeams,
        OverlayCategory::Prism,
        OverlayCategory::CinemaWeather,
    ];

    pub fn id(self) -> &'static str {
        match self {
            Self::LightLeaks => "leaks",
            Self::Bokeh => "bokeh",
            Self::Sunbeams => "sunbeams",
            Self::Prism => "prism",
            Self::CinemaWeather => "cinema",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::LightLeaks => "🔥 Fugas de Luz",
            Self::Bokeh => "✨ Bokeh y Partículas",
            Self::Sunbeams => "☀️ Rayos y Atmósfera",
            Self::Prism => "💎 Prisma y Halación",
            Self::CinemaWeather => "🎬 Anamórfico y Clima",
        }
    }
}

/// Metadata for a single overlay card in the UI carousel.
#[derive(Clone, Debug)]
pub struct OverlaySpec {
    /// 1-based preset index (`1..=150`).
    pub index: u16,
    pub id: String,
    pub name: String,
    pub subtitle: String,
    pub category: OverlayCategory,
    /// Primary and secondary sRGB preview colors `[R, G, B]` for card rendering.
    pub color_a: [u8; 3],
    pub color_b: [u8; 3],
    pub accent: [u8; 3],
}

static CATALOG: LazyLock<Vec<OverlaySpec>> = LazyLock::new(build_catalog);

/// Returns the full catalog of 150 optical overlays (30 per category).
pub fn catalog() -> &'static [OverlaySpec] {
    &CATALOG
}

fn build_catalog() -> Vec<OverlaySpec> {
    let mut out = Vec::with_capacity(150);

    let leaks_names: [(&str, &str, [u8; 3], [u8; 3]); 30] = [
        ("Kodak Gold Burn", "Fuga cálida lateral de película 35mm", [255, 138, 48], [214, 52, 30]),
        ("Sunset Corner Glow", "Resplandor ámbar en esquina superior", [255, 168, 64], [232, 74, 42]),
        ("Portra Edge Bleed", "Velo suave dorado en borde izquierdo", [255, 182, 92], [218, 96, 54]),
        ("Amber Gate Flare", "Destello de compuerta analógica", [255, 148, 42], [196, 48, 28]),
        ("Vintage Rose Leak", "Fuga magenta y coral de carrete clásico", [248, 96, 128], [255, 164, 82]),
        ("Cinema Sprocket Warm", "Doble banda vertical de halación", [255, 124, 52], [188, 38, 48]),
        ("Golden Curtain", "Cortina luminosa diagonal cálida", [255, 194, 88], [224, 112, 38]),
        ("Ruby Film Flash", "Velado rojo rubí en extremo derecho", [238, 58, 62], [255, 142, 68]),
        ("Honey Horizon Leak", "Banda horizontal miel y durazno", [255, 186, 84], [234, 108, 64]),
        ("Super 8 Frame Burn", "Quemado orgánico de película Super 8", [255, 132, 38], [204, 44, 24]),
        ("Peach Blossom Veil", "Bruma pastel durazno y oro rosa", [255, 176, 138], [238, 112, 98]),
        ("Copper Edge Flame", "Lengua de luz cobriza lateral", [244, 118, 48], [182, 52, 28]),
        ("Nostalgic Dual Leak", "Fuga bilateral ámbar y carmesí", [255, 158, 62], [218, 48, 72]),
        ("Tangerine Sun Bleed", "Invasión solar mandarina superior", [255, 144, 36], [242, 82, 34]),
        ("Velvet Magenta Burn", "Velo terciopelo fucsia y dorado", [232, 72, 138], [255, 162, 78]),
        ("Analog Corner Kiss", "Toque sutil de luz en esquina inferior", [255, 174, 82], [212, 84, 46]),
        ("Fuji Warm Pastel", "Fuga suave estilo rollo japonés", [255, 198, 124], [228, 128, 92]),
        ("Crimson Ribbon", "Cinta vertical escarlata suave", [236, 62, 54], [255, 152, 74]),
        ("Solarized Frame Edge", "Borde solarizado de cuarto oscuro", [255, 164, 52], [192, 64, 96]),
        ("Warm Apricot Wash", "Baño luminoso albaricoque lateral", [255, 182, 106], [232, 104, 58]),
        ("champagne Film Mist", "Velo champán de exposición larga", [255, 214, 148], [224, 154, 86]),
        ("scarlet Roll End", "Final de carrete escarlata intenso", [242, 54, 46], [255, 138, 42]),
        ("Diagonal Sunset Sweep", "Barrido diagonal de atardecer", [255, 156, 64], [214, 68, 88]),
        ("Soft Coral Leak", "Fuga coral delicada para retratos", [255, 142, 118], [236, 88, 72]),
        ("Golden Hour Border", "Marco luminoso de hora dorada", [255, 188, 76], [218, 98, 34]),
        ("Retro Lomography Glow", "Doble fuga cruzada estilo Lomo", [255, 134, 58], [186, 56, 124]),
        ("Saffron Light Strip", "Franja vertical azafrán brillante", [255, 172, 44], [226, 78, 32]),
        ("Rose Gold Film Veil", "Velo oro rosa de alta costura", [255, 168, 146], [218, 96, 108]),
        ("Ember Corner Wave", "Onda cálida de brasa en esquina", [248, 108, 42], [194, 38, 28]),
        ("Imperial Gold Leak", "Resplandor imperial dorado envolvente", [255, 202, 96], [232, 124, 42]),
    ];

    let bokeh_names: [(&str, &str, [u8; 3], [u8; 3]); 30] = [
        ("Golden Hour Orbs", "Círculos bokeh dorados desenfocados", [255, 206, 112], [238, 152, 64]),
        ("Champagne Sparkle", "Destellos finos de brindis nupcial", [255, 228, 168], [242, 184, 98]),
        ("Fairy Dust Cascade", "Cascada mágica de micro-partículas", [255, 218, 142], [236, 148, 112]),
        ("Wedding Twinkle Lights", "Luces de recepción en primer plano", [255, 212, 128], [244, 162, 78]),
        ("Creamy 85mm f/1.2", "Esferas grandes ultra suaves de retrato", [255, 198, 124], [222, 142, 88]),
        ("Rose Gold Confetti", "Bokeh circular en tonos oro rosa", [255, 184, 162], [238, 122, 118]),
        ("Candlelight Halo", "Halos cálidos de velas nocturnas", [255, 178, 82], [228, 108, 42]),
        ("Starlight Micro Dust", "Polvo estelar brillante flotante", [255, 236, 192], [212, 178, 124]),
        ("Swirly Petzval Bokeh", "Anillos ópticos periféricos clásicos", [255, 204, 118], [218, 138, 72]),
        ("Diamond Shimmer", "Reflejos cristalinos de alto brillo", [242, 246, 255], [255, 216, 158]),
        ("Warm Amber Fireflies", "Luciérnagas cálidas al atardecer", [255, 188, 68], [234, 118, 38]),
        ("Pastel Dream Orbs", "Esferas suaves crema y durazno", [255, 214, 178], [238, 164, 142]),
        ("Cat-Eye Corner Bokeh", "Discos elípticos de lente gran abertura", [255, 208, 126], [224, 148, 76]),
        ("Floating Gold Leaf", "Destellos dorados suspendidos", [255, 216, 104], [218, 144, 48]),
        ("Silver & Gold Gala", "Mezcla elegante de bokeh plata y oro", [248, 238, 218], [255, 196, 108]),
        ("Romantic String Lights", "Guirnalda luminosa desenfocada", [255, 194, 98], [242, 138, 62]),
        ("Morning Dew Reflex", "Reflejos frescos de rocío matutino", [224, 244, 255], [255, 224, 168]),
        ("Sunset Bokeh Frame", "Borde envolvente de esferas cálidas", [255, 176, 88], [236, 102, 58]),
        ("Crystal Chandelier", "Destellos de candelabro de cristal", [255, 234, 188], [228, 176, 118]),
        ("Soft Ivory Pearls", "Bokeh nacarado marfil para novias", [255, 242, 222], [232, 198, 162]),
        ("Emerald & Gold Garden", "Luces de jardín entre follaje", [238, 218, 128], [148, 204, 132]),
        ("bokeh Crown Top", "Corona superior de luces brillantes", [255, 212, 134], [238, 156, 78]),
        ("Foreground Floral Blur", "Pétalos y luces en primer plano", [255, 192, 168], [244, 148, 118]),
        ("Twilight Blue & Gold", "Contraste bokeh azul hora mágica y oro", [255, 198, 108], [112, 158, 238]),
        ("Sparkler Trail Bokeh", "Chispas cálidas de luces de bengala", [255, 218, 124], [248, 136, 54]),
        ("Velvet Night Orbs", "Esferas luminosas para fotos nocturnas", [255, 186, 94], [214, 118, 68]),
        ("Sunlit Pollen Dust", "Partículas doradas flotando al contraluz", [255, 226, 148], [236, 174, 88]),
        ("Cinema Anamorphic Bokeh", "Discos ovalados verticales de cine", [255, 208, 132], [218, 146, 84]),
        ("Celestial Shimmer", "Polvo luminoso etéreo de fantasía", [255, 232, 184], [238, 178, 138]),
        ("Royal Gold Cascade", "Lluvia densa de bokeh dorado real", [255, 202, 92], [232, 138, 44]),
    ];

    let sunbeam_names: [(&str, &str, [u8; 3], [u8; 3]); 30] = [
        ("Cathedral God Rays", "Rayos volumétricos diagonales", [255, 214, 138], [234, 158, 78]),
        ("Golden Zenith Burst", "Explosión solar desde el borde superior", [255, 208, 116], [244, 144, 58]),
        ("Morning Forest Shafts", "Haces de luz filtrados entre árboles", [255, 222, 152], [218, 168, 92]),
        ("Sunset Corner Sunstar", "Sol radiante en esquina superior izquierda", [255, 188, 88], [238, 114, 48]),
        ("Ethereal Bridal Haze", "Bruma luminosa envolvente para retratos", [255, 234, 204], [236, 192, 148]),
        ("Warm Rim Backlight", "Contraluz dorado suave desde arriba", [255, 204, 122], [232, 142, 68]),
        ("Heavenly Light Fan", "Abanico de rayos celestiales suaves", [255, 228, 172], [228, 174, 108]),
        ("Amber Horizon Glow", "Resplandor solar bajo en el horizonte", [255, 176, 78], [234, 104, 46]),
        ("Window Light Beams", "Haces diagonales de ventana clásica", [255, 226, 168], [214, 164, 104]),
        ("Tropical Sun Flare", "Destello solar cálido e intenso", [255, 196, 94], [242, 128, 52]),
        ("Soft Pastel Mist", "Neblina matutina rosada y dorada", [255, 218, 196], [238, 168, 146]),
        ("Right Corner Sunbeams", "Rayos solares desde esquina derecha", [255, 212, 128], [236, 148, 68]),
        ("Golden Valley Fog", "Niebla baja iluminada por el sol", [255, 216, 148], [222, 158, 88]),
        ("Radiant Crown Light", "Halo solar superior centrado", [255, 224, 156], [238, 166, 84]),
        ("Autumn Sun Shafts", "Rayos cobrizos de tarde otoñal", [255, 182, 92], [226, 112, 48]),
        ("Dreamy Diffusion Veil", "Velo de difusión atmosférica Pro-Mist", [255, 236, 212], [228, 194, 158]),
        ("Dual Sunburst Rays", "Doble abanico de rayos cruzados", [255, 214, 134], [234, 152, 72]),
        ("Peach Sunset Haze", "Atmósfera durazno de puesta de sol", [255, 188, 142], [238, 118, 88]),
        ("Temple Light Column", "Columna vertical de luz sagrada", [255, 228, 164], [224, 168, 96]),
        ("Coastal Golden Mist", "Brisa marina dorada al atardecer", [255, 218, 152], [218, 162, 104]),
        ("Soft Spotlight Halo", "Foco atmosférico suave sobre el sujeto", [255, 232, 188], [224, 178, 122]),
        ("Cinematic Dust Beams", "Haces de proyector con textura suave", [255, 216, 144], [212, 154, 88]),
        ("Rose Dawn Rays", "Primeros rayos del alba en tono rosa", [255, 198, 172], [236, 138, 118]),
        ("High Noon Sunstar", "Estrella solar limpia de mediodía", [255, 238, 196], [232, 188, 124]),
        ("Warm Meadow Glow", "Resplandor de pradera al contraluz", [255, 206, 118], [234, 144, 62]),
        ("Mystic Forest Fog", "Bruma mágica con rayos suaves", [244, 228, 188], [198, 176, 128]),
        ("Imperial Sun Curtain", "Cortina densa de rayos dorados", [255, 208, 112], [238, 142, 54]),
        ("Whispering Light Veil", "Velo sutil de luz diagonal", [255, 234, 198], [228, 184, 134]),
        ("Golden Rim Cascade", "Cascada de luz desde el borde superior", [255, 214, 128], [236, 152, 68]),
        ("Celestial Horizon Burst", "Amanecer radiante de gala", [255, 202, 106], [242, 134, 58]),
    ];

    let prism_names: [(&str, &str, [u8; 3], [u8; 3]); 30] = [
        ("Crystal Spectral Arc", "Arco iris óptico de prisma triangular", [255, 168, 192], [138, 212, 255]),
        ("Cinestill 800T Halation", "Halación roja/ámbar de película de cine", [248, 78, 62], [255, 164, 72]),
        ("Diamond Ring Caustics", "Cáusticas reales de cristal tallado", [255, 228, 196], [168, 218, 255]),
        ("Split-Diopter Refraction", "Refracción lateral de lente dividida", [255, 196, 158], [158, 198, 255]),
        ("Rainbow Corner Flare", "Dispersión cromática en esquina", [255, 148, 168], [128, 224, 212]),
        ("Glass Facet Reflection", "Reflejo especular de prisma en lente", [255, 218, 188], [188, 214, 255]),
        ("Double Exposure Veil", "Velo cromático de doble exposición", [255, 184, 162], [176, 158, 238]),
        ("Chromatic Edge Bloom", "Floración espectral en bordes", [255, 158, 184], [142, 218, 244]),
        ("Golden Crystal Prism", "Prisma cálido con reflejos ámbar y cian", [255, 204, 128], [134, 214, 238]),
        ("Opalescent Halo", "Anillo iridiscente de ópalo y perla", [255, 192, 214], [164, 232, 228]),
        ("Ruby & Cyan Dispersion", "Separación óptica bicromática", [248, 92, 108], [98, 218, 238]),
        ("Dreamy Glass Orb", "Esfera de cristal refractiva en borde", [255, 212, 184], [168, 206, 248]),
        ("Prism Streak Diagonal", "Trazo espectral diagonal limpio", [255, 176, 152], [148, 208, 255]),
        ("Kodachrome Edge Glow", "Halación cálida clásica en altas luces", [255, 138, 68], [232, 68, 58]),
        ("Aurora Borealis Prism", "Ondas espectrales esmeralda y violeta", [138, 238, 196], [196, 148, 255]),
        ("Faceted Kaleidoscope", "Reflejos múltiples de cristal", [255, 198, 168], [168, 218, 248]),
        ("Soft Spectral Feather", "Pluma iridiscente sutil para retratos", [255, 188, 204], [162, 224, 242]),
        ("Warm Halation Ring", "Anillo de halación analógica suave", [255, 128, 64], [228, 62, 48]),
        ("Sapphire & Gold Prism", "Destello real zafiro y oro", [255, 208, 124], [108, 168, 255]),
        ("Iridescent Soap Bubble", "Reflejo nacarado de película delgada", [255, 178, 218], [148, 236, 224]),
        ("Vintage Lens Ghosting", "Fantasmas ópticos de lente sin recubrimiento", [255, 194, 148], [158, 188, 242]),
        ("Crown Glass Spectrum", "Arco espectral doble de alta pureza", [255, 162, 152], [132, 204, 255]),
        ("Rose Quartz Caustic", "Refracción suave en cuarzo rosa", [255, 182, 196], [244, 212, 188]),
        ("Prism Frame Border", "Borde refractivo luminoso en esquinas", [255, 196, 164], [154, 212, 248]),
        ("Neon Film Halation", "Halación intensa estilo cine nocturno", [255, 88, 78], [255, 178, 84]),
        ("Celestial Rainbow Veil", "Velo arcoíris diagonal translúcido", [255, 172, 188], [148, 224, 238]),
        ("Liquid Gold Caustics", "Ondas cáusticas de oro líquido", [255, 212, 118], [238, 152, 68]),
        (" Dichroic Cube Flare", "Reflejo de cubo dicroico magenta/cian", [248, 128, 204], [118, 228, 244]),
        ("Soft Halation Mist", "Difusión cálida de película de medio formato", [255, 156, 98], [234, 88, 68]),
        ("Imperial Diamond Prism", "Destello de diamante con dispersión real", [255, 224, 184], [158, 214, 255]),
    ];

    let cinema_names: [(&str, &str, [u8; 3], [u8; 3]); 30] = [
        ("Panavision Blue Streak", "Destello horizontal azul anamórfico", [88, 178, 255], [188, 226, 255]),
        ("Gold Anamorphic Flare", "Trazo horizontal dorado de lente cine", [255, 196, 92], [238, 134, 48]),
        ("Starburst Filter 4x", "Filtro de cruz estelar de 4 puntas", [255, 232, 188], [238, 182, 118]),
        ("Starburst Filter 8x", "Estrella de 8 puntas para joyas y luces", [255, 238, 204], [218, 188, 255]),
        ("Romantic Raindrops", "Gotas de lluvia suaves al contraluz", [204, 228, 255], [242, 248, 255]),
        ("Winter Snow Flurry", "Copos de nieve desenfocados en vuelo", [238, 246, 255], [198, 218, 244]),
        ("Dual Sci-Fi Anamorphic", "Doble banda horizontal cian y ámbar", [96, 188, 255], [255, 178, 84]),
        ("Warm Cinema Ellipse", "Halo elíptico de lente Cooke Speed Panchro", [255, 198, 128], [224, 138, 78]),
        ("Soft Misty Drizzle", "Bruma húmeda con micro-gotas brillantes", [214, 232, 248], [244, 236, 224]),
        ("Golden Confetti Shower", "Lluvia festiva de destellos dorados", [255, 214, 108], [238, 154, 62]),
        ("Teal & Orange Streak", "Flare anamórfico cinematográfico", [78, 204, 224], [255, 158, 72]),
        ("Snowstorm Bokeh", "Nevada intensa con primer plano cremoso", [244, 248, 255], [208, 224, 244]),
        ("Rose Gold Anamorphic", "Trazo horizontal en tono oro rosa", [255, 178, 168], [238, 118, 128]),
        ("Hollywood Center Streak", "Línea anamórfica suave a media altura", [138, 198, 255], [255, 214, 158]),
        ("Summer Sun Shower", "Lluvia dorada iluminada por el sol", [255, 222, 158], [198, 228, 255]),
        ("8-Point Diamond Glint", "Destellos múltiples de filtro estelar", [255, 242, 214], [196, 224, 255]),
        ("Midnight Blue Flare", "Destello nocturno azul cobalto", [72, 148, 255], [148, 196, 255]),
        ("Amber Headlight Streak", "Trazos cálidos de luces nocturnas", [255, 174, 68], [238, 98, 42]),
        ("Gentle Frost Crystals", "Cristales de escarcha en bordes", [224, 242, 255], [184, 214, 248]),
        ("Champagne Star Glints", "Cruces estelares cálidas de fiesta", [255, 218, 148], [242, 168, 88]),
        ("anamorphic Edge Wing", "Ala luminosa anamórfica lateral", [112, 188, 255], [218, 238, 255]),
        ("Warm Falling Petals", "Brillo suave de pétalos al viento", [255, 198, 184], [244, 148, 138]),
        ("Silver Screen Flare", "Destello clásico monocromo-plateado", [238, 242, 248], [188, 198, 214]),
        ("Golden Horizon Streak", "Trazo anamórfico dorado en tercio superior", [255, 204, 112], [236, 142, 58]),
        ("Evening Rain Reflex", "Reflejos húmedos azules y ámbar", [128, 188, 248], [255, 184, 104]),
        ("Fairy Star Crosses", "Constelación de cruces estelares finas", [255, 234, 192], [238, 188, 138]),
        ("Nordic Snow Mist", "Neblina invernal con copos suaves", [232, 242, 255], [194, 212, 236]),
        ("Sunset Anamorphic Dual", "Doble destello horizontal de atardecer", [255, 168, 84], [238, 92, 78]),
        ("Crystal Rain Veil", "Cortina fina de lluvia cristalina", [212, 234, 255], [238, 246, 255]),
        ("Grand Finale Sparkles", "Destellos estelares y trazos de gala", [255, 216, 128], [168, 212, 255]),
    ];

    let groups: [(OverlayCategory, &[(&str, &str, [u8; 3], [u8; 3]); 30]); 5] = [
        (OverlayCategory::LightLeaks, &leaks_names),
        (OverlayCategory::Bokeh, &bokeh_names),
        (OverlayCategory::Sunbeams, &sunbeam_names),
        (OverlayCategory::Prism, &prism_names),
        (OverlayCategory::CinemaWeather, &cinema_names),
    ];

    let mut idx = 1u16;
    for (cat, list) in groups {
        for (i, (name, sub, ca, cb)) in list.iter().enumerate() {
            let accent = [
                ((ca[0] as u16 + cb[0] as u16) / 2) as u8,
                ((ca[1] as u16 + cb[1] as u16) / 2) as u8,
                ((ca[2] as u16 + cb[2] as u16) / 2) as u8,
            ];
            out.push(OverlaySpec {
                index: idx,
                id: format!("ovl.{}.{:02}", cat.id(), i + 1),
                name: (*name).to_string(),
                subtitle: (*sub).to_string(),
                category: cat,
                color_a: *ca,
                color_b: *cb,
                accent,
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

#[inline]
fn hash_u32(mut x: u32) -> f32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    (x & 0x00ff_ffff) as f32 / 16_777_215.0
}

#[inline]
fn rotate_rgb_warmth(rgb: [f32; 3], warmth: f32) -> [f32; 3] {
    if warmth.abs() < 1e-4 {
        return rgb;
    }
    let w = (warmth / 100.0).clamp(-1.0, 1.0);
    let r = (rgb[0] * (1.0 + 0.35 * w) + rgb[1] * (0.08 * w.max(0.0))).max(0.0);
    let g = (rgb[1] * (1.0 + 0.05 * w)).max(0.0);
    let b = (rgb[2] * (1.0 - 0.35 * w) + rgb[1] * (0.12 * (-w).max(0.0))).max(0.0);
    [r, g, b]
}

type CacheEntry = ((u16, i32, i32), Arc<Vec<f32>>);

static MAP_CACHE: LazyLock<Mutex<Vec<CacheEntry>>> = LazyLock::new(|| Mutex::new(Vec::new()));

/// Returns the cached `OVL_SIDE × OVL_SIDE × 3` linear RGB field for `preset` (`1..=150`),
/// adjusted by `warmth` (`-100..100`) and optical `blur` (`0..100`).
pub fn overlay_map_cached(preset: u16, warmth: f64, blur: f64) -> Option<Arc<Vec<f32>>> {
    if !(1..=150).contains(&preset) {
        return None;
    }
    let w_key = (warmth.clamp(-100.0, 100.0) * 2.0).round() as i32;
    let b_key = (blur.clamp(0.0, 100.0) * 2.0).round() as i32;
    let key = (preset, w_key, b_key);
    if let Ok(guard) = MAP_CACHE.lock()
        && let Some((_, map)) = guard.iter().find(|(k, _)| *k == key)
    {
        return Some(map.clone());
    }
    let map = Arc::new(synthesize_overlay_map(preset, warmth as f32, blur as f32));
    if let Ok(mut guard) = MAP_CACHE.lock() {
        if guard.len() >= 16 {
            guard.remove(0);
        }
        guard.push((key, map.clone()));
    }
    Some(map)
}

fn synthesize_overlay_map(preset: u16, warmth: f32, blur: f32) -> Vec<f32> {
    let spec_idx = (preset.saturating_sub(1) as usize).min(149);
    let spec = &catalog()[spec_idx];
    let sub_idx = ((preset - 1) % 30) as usize;
    let seed = (preset as u32).wrapping_mul(0x9E37_79B9);

    let ca = spec.color_a.map(|v| (v as f32 / 255.0).powf(2.2));
    let cb = spec.color_b.map(|v| (v as f32 / 255.0).powf(2.2));
    let mut buf = vec![0.0f32; OVL_SIDE * OVL_SIDE * 3];

    match spec.category {
        OverlayCategory::LightLeaks => {
            let side_mode = sub_idx % 6;
            let freq1 = 2.0 + (sub_idx as f32 * 0.37) % 3.5;
            let freq2 = 3.2 + (sub_idx as f32 * 0.53) % 4.0;
            let phase = sub_idx as f32 * 0.65;
            for y in 0..OVL_SIDE {
                let v = (y as f32 + 0.5) / OVL_SIDE as f32;
                for x in 0..OVL_SIDE {
                    let u = (x as f32 + 0.5) / OVL_SIDE as f32;
                    let wave = 0.08 * (v * freq1 + phase).sin() + 0.05 * (u * freq2 - phase).cos();
                    let (d1, d2) = match side_mode {
                        0 => ((u + wave).max(0.0), (1.0 - v + wave).max(0.0)), // Left + Bottom
                        1 => ((1.0 - u + wave).max(0.0), (v + wave).max(0.0)), // Right + Top
                        2 => ((v + wave).max(0.0), (u * 0.7 + v * 0.7 + wave).max(0.0)), // Top edge
                        3 => (((u - 0.18 - wave).abs() * 2.2), ((u - 0.82 + wave).abs() * 2.4)), // Sprocket bands
                        4 => ((u * 0.8 + v * 0.6 + wave), ((1.0 - u) * 0.7 + (1.0 - v) * 0.7)), // Corner glow
                        _ => (((u + v) * 0.55 + wave), (u + wave).max(0.0)),
                    };
                    let w1 = (1.0 - sstep(0.0, 0.52, d1)).powf(1.6);
                    let w2 = (1.0 - sstep(0.0, 0.45, d2)).powf(2.0) * 0.75;
                    let idx = (y * OVL_SIDE + x) * 3;
                    for c in 0..3 {
                        buf[idx + c] = ca[c] * w1 + cb[c] * w2;
                    }
                }
            }
        }
        OverlayCategory::Bokeh => {
            let num_orbs = 26 + (sub_idx % 22);
            let mut orbs = Vec::with_capacity(num_orbs);
            for k in 0..num_orbs {
                let s = seed.wrapping_add(k as u32 * 7919);
                let mut ox = hash_u32(s);
                let mut oy = hash_u32(s.wrapping_add(1));
                // Bias orbs towards the frame periphery/edges so center stays natural
                if sub_idx % 3 != 2 && (ox - 0.5).hypot(oy - 0.5) < 0.24 {
                    ox = if ox < 0.5 { ox * 0.55 } else { 1.0 - (1.0 - ox) * 0.55 };
                }
                if sub_idx % 4 == 1 {
                    oy *= 0.65;
                }
                let radius = if k < 6 {
                    0.075 + hash_u32(s.wrapping_add(2)) * 0.085
                } else {
                    0.014 + hash_u32(s.wrapping_add(2)).powi(2) * 0.052
                };
                let mix = hash_u32(s.wrapping_add(3));
                let intensity = 0.28 + 0.72 * hash_u32(s.wrapping_add(4));
                orbs.push((ox, oy, radius, mix, intensity));
            }
            for y in 0..OVL_SIDE {
                let v = (y as f32 + 0.5) / OVL_SIDE as f32;
                for x in 0..OVL_SIDE {
                    let u = (x as f32 + 0.5) / OVL_SIDE as f32;
                    let mut sum = [0.0f32; 3];
                    // Soft ambient corner wash
                    let amb = (1.0 - sstep(0.0, 0.85, (u - 0.1).hypot(v - 0.1))) * 0.14;
                    for c in 0..3 {
                        sum[c] += ca[c] * amb;
                    }
                    for &(ox, oy, r, mix, inten) in &orbs {
                        let dx = u - ox;
                        let dy = v - oy;
                        if dx.abs() > r * 1.25 || dy.abs() > r * 1.25 {
                            continue;
                        }
                        let d = dx.hypot(dy) / r;
                        if d < 1.2 {
                            // Optical aperture disc with soft interior + spherical aberration rim
                            let disc = 1.0 - sstep(0.78, 1.08, d);
                            let rim = sstep(0.55, 0.88, d) * (1.0 - sstep(0.88, 1.12, d));
                            let w = (disc * 0.58 + rim * 0.62) * inten;
                            for c in 0..3 {
                                let col = ca[c] * (1.0 - mix) + cb[c] * mix;
                                sum[c] += col * w;
                            }
                        }
                    }
                    let idx = (y * OVL_SIDE + x) * 3;
                    buf[idx] = sum[0];
                    buf[idx + 1] = sum[1];
                    buf[idx + 2] = sum[2];
                }
            }
        }
        OverlayCategory::Sunbeams => {
            // Source origin along top/left/right edge depending on sub_idx
            let (sx, sy) = match sub_idx % 5 {
                0 => (-0.08 + (sub_idx as f32 * 0.02) % 0.25, -0.08),
                1 => (1.08 - (sub_idx as f32 * 0.02) % 0.25, -0.08),
                2 => (0.50 + ((sub_idx as f32 - 15.0) * 0.015), -0.12),
                3 => (-0.10, 0.18 + (sub_idx as f32 * 0.015) % 0.3),
                _ => (0.22 + (sub_idx as f32 * 0.02) % 0.55, -0.05),
            };
            let ray_freq = 9.0 + (sub_idx % 9) as f32 * 2.2;
            let phase = sub_idx as f32 * 0.85;
            for y in 0..OVL_SIDE {
                let v = (y as f32 + 0.5) / OVL_SIDE as f32;
                for x in 0..OVL_SIDE {
                    let u = (x as f32 + 0.5) / OVL_SIDE as f32;
                    let dx = u - sx;
                    let dy = v - sy;
                    let dist = dx.hypot(dy);
                    let angle = dy.atan2(dx);
                    let r1 = 0.5 + 0.5 * (angle * ray_freq + phase).sin();
                    let r2 = 0.5 + 0.5 * (angle * (ray_freq * 1.65) - phase * 1.3).cos();
                    let rays = (r1.powf(2.6) * 0.65 + (r1 * r2).powf(2.0) * 0.55) * (1.0 - sstep(0.15, 1.35, dist));
                    let core = (1.0 - sstep(0.0, 0.68, dist)).powf(2.2);
                    let idx = (y * OVL_SIDE + x) * 3;
                    for c in 0..3 {
                        buf[idx + c] = ca[c] * (core * 0.75 + rays * 0.65) + cb[c] * (rays * 0.45);
                    }
                }
            }
        }
        OverlayCategory::Prism => {
            let mode = sub_idx % 4;
            let angle = (sub_idx as f32 * 0.42) - 0.8;
            let (ca_cos, ca_sin) = (angle.cos(), angle.sin());
            for y in 0..OVL_SIDE {
                let v = (y as f32 + 0.5) / OVL_SIDE as f32;
                for x in 0..OVL_SIDE {
                    let u = (x as f32 + 0.5) / OVL_SIDE as f32;
                    let idx = (y * OVL_SIDE + x) * 3;
                    if mode == 0 || mode == 2 {
                        // Spectral rainbow dispersion arc / caustic band
                        let cx = if sub_idx % 2 == 0 { -0.15 } else { 1.15 };
                        let cy = 0.5 + ((sub_idx % 5) as f32 - 2.0) * 0.12;
                        let r = (u - cx).hypot(v - cy);
                        let band_center = 0.62 + (sub_idx % 7) as f32 * 0.04;
                        let spread = 0.085;
                        // Chromatic dispersion: R, G, B peak at slightly offset radii
                        let wr = (1.0 - ((r - (band_center - spread * 0.55)) / spread).abs().min(1.0)).powf(2.2);
                        let wg = (1.0 - ((r - band_center) / spread).abs().min(1.0)).powf(2.2);
                        let wb = (1.0 - ((r - (band_center + spread * 0.55)) / spread).abs().min(1.0)).powf(2.2);
                        buf[idx] = wr * 0.85 + ca[0] * wg * 0.25;
                        buf[idx + 1] = wg * 0.78 + cb[1] * 0.20;
                        buf[idx + 2] = wb * 0.90 + cb[2] * wg * 0.25;
                    } else {
                        // Film halation & crystal facet reflection
                        let rx = (u - 0.5) * ca_cos - (v - 0.5) * ca_sin;
                        let ry = (u - 0.5) * ca_sin + (v - 0.5) * ca_cos;
                        let edge_dist = (u.min(1.0 - u)).min(v.min(1.0 - v));
                        let halation = (1.0 - sstep(0.0, 0.26, edge_dist)).powf(2.0);
                        let facet = (1.0 - sstep(0.0, 0.22, (rx - 0.22).abs())).powf(2.4) * (1.0 - sstep(0.0, 0.45, ry.abs()));
                        for c in 0..3 {
                            let disp = if c == 0 { 1.15 } else if c == 2 { 0.85 } else { 1.0 };
                            buf[idx + c] = (ca[c] * halation * 0.72 + cb[c] * facet * 0.80) * disp;
                        }
                    }
                }
            }
        }
        OverlayCategory::CinemaWeather => {
            let kind = sub_idx % 4;
            for y in 0..OVL_SIDE {
                let v = (y as f32 + 0.5) / OVL_SIDE as f32;
                for x in 0..OVL_SIDE {
                    let u = (x as f32 + 0.5) / OVL_SIDE as f32;
                    let idx = (y * OVL_SIDE + x) * 3;
                    match kind {
                        0 | 2 => {
                            // Anamorphic horizontal lens flare streak + elliptical glow
                            let cy = 0.28 + (sub_idx % 7) as f32 * 0.075;
                            let cx = 0.25 + (sub_idx % 5) as f32 * 0.12;
                            let dy = (v - cy).abs();
                            let dx = (u - cx).abs();
                            let streak = (-dy * 38.0).exp() * (-dx * 1.6).exp();
                            let core = (-(dx * dx * 8.0 + dy * dy * 28.0)).exp();
                            let ring = (1.0 - ((dx * 1.4).hypot(dy * 2.8) - 0.28).abs().min(0.12) / 0.12).powf(2.0) * 0.35;
                            for c in 0..3 {
                                buf[idx + c] = ca[c] * (streak * 0.95 + ring) + cb[c] * core * 0.85;
                            }
                        }
                        1 => {
                            // Starburst 4x/8x filter glints
                            let mut sum = [0.0f32; 3];
                            for k in 0..7 {
                                let s = seed.wrapping_add(k as u32 * 4099);
                                let gx = 0.12 + hash_u32(s) * 0.76;
                                let gy = 0.12 + hash_u32(s.wrapping_add(1)) * 0.76;
                                let dx = u - gx;
                                let dy = v - gy;
                                let d = dx.hypot(dy);
                                if d < 0.28 {
                                    let cross1 = (-dx.abs() * 65.0).exp() * (-dy.abs() * 8.5).exp()
                                        + (-dy.abs() * 65.0).exp() * (-dx.abs() * 8.5).exp();
                                    let d1 = (dx + dy) * 0.7071;
                                    let d2 = (dx - dy) * 0.7071;
                                    let cross2 = (-d1.abs() * 75.0).exp() * (-d2.abs() * 11.0).exp()
                                        + (-d2.abs() * 75.0).exp() * (-d1.abs() * 11.0).exp();
                                    let core = (-d * 24.0).exp();
                                    let w = core * 0.8 + cross1 * 0.65 + cross2 * 0.45;
                                    for c in 0..3 {
                                        sum[c] += (ca[c] * 0.6 + cb[c] * 0.4) * w;
                                    }
                                }
                            }
                            buf[idx] = sum[0];
                            buf[idx + 1] = sum[1];
                            buf[idx + 2] = sum[2];
                        }
                        _ => {
                            // Romantic raindrops / winter snow bokeh particles
                            let mut sum = [0.0f32; 3];
                            let is_rain = sub_idx % 8 == 3;
                            for k in 0..42 {
                                let s = seed.wrapping_add(k as u32 * 6151);
                                let px = hash_u32(s);
                                let py = hash_u32(s.wrapping_add(1));
                                let dx = u - px;
                                let dy = v - py;
                                if is_rain {
                                    // Diagonal raindrop streak
                                    let rx = dx * 0.94 + dy * 0.34;
                                    let ry = -dx * 0.34 + dy * 0.94;
                                    if rx.abs() < 0.015 && ry.abs() < 0.065 {
                                        let w = (1.0 - (rx / 0.012).abs().min(1.0)).powf(2.0)
                                            * (1.0 - (ry / 0.060).abs().min(1.0)).powf(1.6)
                                            * 0.55;
                                        for c in 0..3 {
                                            sum[c] += ca[c] * w;
                                        }
                                    }
                                } else {
                                    // Soft snow bokeh flake
                                    let r = 0.010 + hash_u32(s.wrapping_add(2)).powi(2) * 0.032;
                                    let d = dx.hypot(dy) / r;
                                    if d < 1.0 {
                                        let w = (1.0 - d * d).powf(1.8) * 0.65;
                                        for c in 0..3 {
                                            sum[c] += (ca[c] * 0.7 + cb[c] * 0.3) * w;
                                        }
                                    }
                                }
                            }
                            buf[idx] = sum[0];
                            buf[idx + 1] = sum[1];
                            buf[idx + 2] = sum[2];
                        }
                    }
                }
            }
        }
    }

    // Apply optional optical blur
    if blur > 0.5 {
        let radius = ((blur / 100.0) * 14.0).round() as usize;
        if radius > 0 {
            buf = box_blur_rgb(&buf, OVL_SIDE, OVL_SIDE, radius);
        }
    }

    // Apply warmth / hue shift and clamp
    for px in buf.as_chunks_mut::<3>().0 {
        *px = rotate_rgb_warmth(*px, warmth).map(|v| v.clamp(0.0, 2.0));
    }
    buf
}

fn box_blur_rgb(src: &[f32], w: usize, h: usize, radius: usize) -> Vec<f32> {
    let r = radius.min(20).max(1);
    let mut tmp = vec![0.0f32; w * h * 3];
    let mut out = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let x0 = x.saturating_sub(r);
            let x1 = (x + r).min(w - 1);
            let inv = 1.0 / ((x1 - x0 + 1) as f32);
            let mut sum = [0.0f32; 3];
            for xx in x0..=x1 {
                let i = (y * w + xx) * 3;
                sum[0] += src[i];
                sum[1] += src[i + 1];
                sum[2] += src[i + 2];
            }
            let o = (y * w + x) * 3;
            tmp[o] = sum[0] * inv;
            tmp[o + 1] = sum[1] * inv;
            tmp[o + 2] = sum[2] * inv;
        }
    }
    for y in 0..h {
        let y0 = y.saturating_sub(r);
        let y1 = (y + r).min(h - 1);
        let inv = 1.0 / ((y1 - y0 + 1) as f32);
        for x in 0..w {
            let mut sum = [0.0f32; 3];
            for yy in y0..=y1 {
                let i = (yy * w + x) * 3;
                sum[0] += tmp[i];
                sum[1] += tmp[i + 1];
                sum[2] += tmp[i + 2];
            }
            let o = (y * w + x) * 3;
            out[o] = sum[0] * inv;
            out[o + 1] = sum[1] * inv;
            out[o + 2] = sum[2] * inv;
        }
    }
    out
}

/// Resolved per-render parameters for the active creative overlay, shared between CPU (`finish.rs`)
/// and GPU (`params.rs` / `finish.wgsl`).
#[derive(Clone, Debug)]
pub struct OverlayPlan {
    pub map: Arc<Vec<f32>>,
    pub amount: f32,
    pub blend: u32,
    pub cos_a: f32,
    pub sin_a: f32,
    pub inv_scale: f32,
    pub shift_x: f32,
    pub shift_y: f32,
    pub protect_center: f32,
    pub flip_x: bool,
    pub flip_y: bool,
}

impl OverlayPlan {
    pub fn from_settings(ovl: &CreativeOverlay) -> Option<Self> {
        if !ovl.is_active() {
            return None;
        }
        let preset = ovl.preset.round().clamp(1.0, 150.0) as u16;
        let map = overlay_map_cached(preset, ovl.warmth, ovl.blur)?;
        let amount = (ovl.amount / 100.0).clamp(0.0, 1.0) as f32;
        if amount <= 0.0 {
            return None;
        }
        let rad = (ovl.rotation as f32).to_radians();
        let scale = (ovl.scale / 100.0).clamp(0.3, 2.5) as f32;
        Some(Self {
            map,
            amount,
            blend: (ovl.blend.round().clamp(0.0, 3.0)) as u32,
            cos_a: rad.cos(),
            sin_a: rad.sin(),
            inv_scale: 1.0 / scale,
            shift_x: (ovl.pos_x / 100.0).clamp(-1.0, 1.0) as f32 * 0.48,
            shift_y: (ovl.pos_y / 100.0).clamp(-1.0, 1.0) as f32 * 0.48,
            protect_center: (ovl.protect_center / 100.0).clamp(0.0, 1.0) as f32,
            flip_x: ovl.flip_x >= 0.5,
            flip_y: ovl.flip_y >= 0.5,
        })
    }

    /// Evaluates and blends the overlay onto sRGB-encoded pixel `e` (`[0..1; 3]`) at normalized
    /// frame coordinates `(nx, ny)` in `0..1`. Mirrors `apply_overlay` in `finish.wgsl`.
    #[inline]
    pub fn apply_srgb(&self, e: [f32; 3], nx: f32, ny: f32) -> [f32; 3] {
        let dx = nx - 0.5 - self.shift_x;
        let dy = ny - 0.5 - self.shift_y;
        let rx = (dx * self.cos_a + dy * self.sin_a) * self.inv_scale;
        let ry = (-dx * self.sin_a + dy * self.cos_a) * self.inv_scale;
        let mut u = rx + 0.5;
        let mut v = ry + 0.5;
        if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
            return e;
        }
        if self.flip_x {
            u = 1.0 - u;
        }
        if self.flip_y {
            v = 1.0 - v;
        }
        let side = OVL_SIDE as f32;
        let fx = (u * (side - 1.0)).clamp(0.0, side - 1.0001);
        let fy = (v * (side - 1.0)).clamp(0.0, side - 1.0001);
        let x0 = fx as usize;
        let y0 = fy as usize;
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;
        let i00 = (y0 * OVL_SIDE + x0) * 3;
        let i10 = (y0 * OVL_SIDE + x0 + 1) * 3;
        let i01 = ((y0 + 1) * OVL_SIDE + x0) * 3;
        let i11 = ((y0 + 1) * OVL_SIDE + x0 + 1) * 3;
        let mut ov = [0.0f32; 3];
        for c in 0..3 {
            let top = self.map[i00 + c] * (1.0 - tx) + self.map[i10 + c] * tx;
            let bot = self.map[i01 + c] * (1.0 - tx) + self.map[i11 + c] * tx;
            ov[c] = (top * (1.0 - ty) + bot * ty).max(0.0).sqrt();
        }
        // Center / subject protection attenuation
        let center_d = ((nx - 0.5).powi(2) + (ny - 0.5).powi(2)).sqrt() * 2.0;
        let protect = 1.0 - self.protect_center * 0.82 * (1.0 - sstep(0.18, 0.85, center_d));
        let k = (self.amount * protect).clamp(0.0, 1.0);
        if k <= 1e-4 {
            return e;
        }
        let mut out = [0.0f32; 3];
        for c in 0..3 {
            let b = e[c].clamp(0.0, 1.0);
            let s = (ov[c] * k).clamp(0.0, 1.2);
            let blended = match self.blend {
                0 => 1.0 - (1.0 - b) * (1.0 - s.min(1.0)), // Screen (Pantalla)
                1 => {
                    // Soft Light (Luz Suave)
                    let sl = (1.0 - 2.0 * s.min(1.0)) * b * b + 2.0 * s.min(1.0) * b;
                    b + (sl - b) * 0.85 + s * 0.35
                }
                2 => (b + s * 0.90).min(1.0), // Linear Dodge / Add (Añadir)
                _ => {
                    // Overlay / Superponer
                    let ov_b = if b < 0.5 { 2.0 * b * s.min(1.0) } else { 1.0 - 2.0 * (1.0 - b) * (1.0 - s.min(1.0)) };
                    b * (1.0 - k * 0.65) + ov_b * (k * 0.65) + s * 0.25
                }
            };
            out[c] = blended.clamp(0.0, 1.0);
        }
        out
    }
}
