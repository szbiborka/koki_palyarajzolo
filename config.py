# KONFIGURÁCIÓ - Minden útvonal és konstans itt van definiálva.
# Ha szerverre költözik az alkalmazás, CSAK ezt a fájlt kell módosítani.

import os

# --- Adatfájlok útvonalai ---
# Ezek az útvonalak a saját gépen érvényesek. Szerveren felülírhatók
# környezeti változókkal (pl. export ATLAS_PATH=/data/atlas/annotation_25.nrrd)
BASE_DATA_DIR = os.environ.get(
    'PALYAKOVETO_DATA_DIR',
    'C:/Users/szabo.biborka/koki_palyarajzolo/adatfajlok/data_v2/'
)

ATLAS_PATH = os.environ.get(
    'ATLAS_PATH',
    'C:/Users/szabo.biborka/koki_palyarajzolo/adatfajlok/annotation_25.nrrd'
)

DICTIONARY_PATH = os.environ.get(
    'DICTIONARY_PATH',
    'C:/Users/szabo.biborka/koki_palyarajzolo/adatfajlok/query.csv'
)

# --- Soma index fájl ---
# Ez a fájl tárolja el az összes SWC fájl soma-régió megfeleltetését.
# Első futáskor épül fel, utána gyorsan betöltődik.
# Ha új SWC fájlok kerülnek a mappába, a UI-ban lévő "Rebuild index" gombbal frissíthető.
SOMA_INDEX_PATH = os.environ.get(
    'SOMA_INDEX_PATH',
    os.path.join(os.path.dirname(__file__), 'soma_index.csv')
)

# --- Atlas paraméterek ---
# Voxel méret mikrométerben (a 25-ös atlasz 25um felbontású)
VOXEL_SIZE = 25

# --- Alapértelmezett célterületek ---
# Ezek az ID-k az Allen Mouse Brain Atlaszból származnak.
# A felhasználói felületen ezek lesznek előre kiválasztva,
# de a felhasználó bármilyen más régiót is hozzáadhat.
DEFAULT_TARGET_REGIONS = {
    'GPe - Globus Pallidus external': 1022,
    'TRN - Reticular nucleus of thalamus': 262,
}

# --- Virtuális "leszálló agytörzs" célterület (pyramidal tract) ---
# Az Allen ontológiában a "Brain stem" (343) MAGÁBA FOGLALJA a köztiagyat
# (Interbrain), így a THALAMUST is. Ezért ha valaki a 343-as "Brain stem"-et
# célozza, a csak thalamusba vetítő L6 sejtek is átcsúsznak a szűrőn.
#
# Ez a virtuális régió a VALÓDI leszálló agytörzs: Középagy (Midbrain, 313) +
# Utóagy (Hindbrain, 1065 = híd + nyúltvelő), a thalamust KIZÁRVA. Így a
# "vetít-e az agytörzsbe" kérdés tényleg a pyramidal tract sejteket fogja meg.
# Negatív ID, hogy soha ne ütközzön valós Allen régió ID-val.
BRAINSTEM_MOTOR_ID = -1001
BRAINSTEM_MOTOR_NAME = "Brain stem descending — Midbrain+Hindbrain (excl. thalamus)"
BRAINSTEM_MOTOR_ACRONYM = "BS-desc"
BRAINSTEM_MOTOR_COMPONENTS = [313, 1065]  # Midbrain (MB), Hindbrain (HB)

# --- Sejttípus kódok az SWC formátumban ---
# Ez a szabványos SWC specifikáció szerint van definiálva.
SWC_TYPE_SOMA = 1
SWC_TYPE_AXON = 2
SWC_TYPE_AXON_UNDEFINED = 0  # Egyes fájlokban a 0-ás típus is axont jelöl

# --- A VETÍTÉS KRITÉRIUMA (egyben a szűrés alapértelmezése) ---
# Régiónként EGY kritériumkészlet mondja meg, mi számít vetítésnek. Ugyanez hajtja
# a "..._projects" pipát, a szűrést és az összesítő táblákat is, tehát nem lehet
# közöttük ellentmondás.
#
# Alapértelmezés: >=1 végpont ÉS >=1 elágazás = valódi terminális arborizáció.
# Az áthaladó axon (ami csak keresztezi a régiót, de máshol végződik) így nem
# számít vetítésnek. A számok emelésével szigorítható, az elágazás 0-ra
# állításával lazítható "csak végpont" logikára.
DEFAULT_FILTER = {
    'min_endpoints': 1,       # Minimum végpontok száma a célterületen
    'min_branch_points': 1,   # Minimum elágazási pontok száma a célterületen
    'min_axon_length_um': 0,  # Minimum axonhossz mikrométerben (0 = nincs feltétel)
    'min_endpoint_fraction': 0.0,  # Minimum végpont-arány [0..1] (0 = nincs feltétel; L6-szűrő)
}

# --- Vizualizációs beállítások ---
VIZ_REGION_OPACITY = 0.25        # Agyterület felszínek átlátszósága
VIZ_SOMA_RADIUS = 15             # Soma gömb sugara mikrométerben
VIZ_POINT_SIZE = 10              # Vetítési pontok mérete
VIZ_AXON_LINE_WIDTH = 2          # Axon vonalak vastagsága
VIZ_MARCHING_CUBES_STEP = 2      # Felszín-generálás lépésköze (kisebb = szebb, de lassabb)

# --- 3D JELENET TÉMÁK ---
# Két teljes témát kínálunk. A "dark" az alapértelmezett: a vékony axonvonalak
# világos színnel, sötét háttéren sokkal jobban láthatók (ez a fluoreszcens
# mikroszkópia megszokott képi világa is).
#
# MINDKÉT palettát a dataviz validátor ellenőrizte (OKLCH világosság-sáv, króma-
# padló, színtévesztő elkülönülés, kontraszt a háttérhez):
#   dark  (felület #0E1117): minden ellenőrzés PASS, legrosszabb szomszédos
#                            CVD ΔE 8.8 (deutan) / 6.3 (tritan)
#   light (felület #FAF9F6): minden ellenőrzés PASS, legrosszabb szomszédos
#                            CVD ΔE 11.3 (deutan) / 8.4 (tritan)
# A sorrend SZÁMÍT: a validátor a szomszédos párokat méri, és ez a sorrend
# maximalizálja a legrosszabb pár elkülönülését. Ne keverd össze a színeket.
# A régiók nevesítve is megjelennek a jelmagyarázatban, tehát az azonosítás
# soha nem csak a színen múlik.
VIZ_THEMES = {
    'dark': {
        'label': 'Dark (recommended — thin axons stand out)',
        'paper_bg': '#0E1117',        # A teljes ábra háttere
        'scene_bg': '#0E1117',        # A 3D tengelyek háttere
        'grid': '#2A3040',            # Visszafogott rács
        'axis_text': '#9BA4B4',       # Tengelyfeliratok
        'soma': '#FFFFFF',            # A soma fehéren a legjobban látszik
        'axon_default': '#5A6474',    # Nem célterületi axon - jelen van, de háttérbe húzódik
        'brain_outline': '#8B96A8',   # Agy körvonal (nagyon áttetsző)
        'brain_outline_opacity': 0.055,
        'legend_bg': 'rgba(14,17,23,0.85)',
        'legend_border': '#2A3040',
        # A régiófelszínek KONTEXTUST adnak, nem ők a főszereplők: alacsony
        # átlátszatlansággal nem nyomják el a vékony axonvonalakat.
        'region_opacity': 0.13,
        'axon_width': 3,
        'region_palette': [
            '#0284C7',  # kék
            '#EA580C',  # narancs
            '#8B5CF6',  # lila
            '#0D9488',  # türkiz
            '#D97706',  # borostyán
            '#EC4899',  # rózsaszín
            '#16A34A',  # zöld
            '#D946EF',  # fukszia
        ],
    },
    'light': {
        'label': 'Light (matches the app background)',
        'paper_bg': '#FAF9F6',
        'scene_bg': '#FAF9F6',
        'grid': '#DDDDD5',
        'axis_text': '#5E5A4A',
        'soma': '#111111',
        'axon_default': '#B4B0A4',
        'brain_outline': '#6E6A5E',
        'brain_outline_opacity': 0.05,
        'legend_bg': 'rgba(255,255,255,0.85)',
        'legend_border': '#CFD6BC',
        'region_opacity': 0.18,
        'axon_width': 3,
        'region_palette': [
            '#EA580C',  # narancs
            '#1D4ED8',  # kék
            '#A16207',  # barna
            '#0D9488',  # türkiz
            '#BE123C',  # bordó
            '#7C3AED',  # lila
            '#4D7C0F',  # olíva
            '#C026D3',  # fukszia
        ],
    },
}
DEFAULT_VIZ_THEME = 'dark'

# --- Agy körvonal ---
# Térbeli tájékozódáshoz kirajzoljuk a teljes agy felszínét nagyon áttetszően.
# Nem egy konkrét régió ID-ra szűrünk, hanem az ÖSSZES annotált voxelre
# (atlas > 0), így akkor is működik, ha a "root" régió nincs a szótárban.
# A nagyobb lépésköz azért kell, mert ez a legnagyobb felület a jelenetben.
VIZ_BRAIN_OUTLINE_STEP = 6

# --- Visszafelé kompatibilis szín-hozzáférés (a régi kód ezt használja) ---
COLORS = {
    'soma': VIZ_THEMES[DEFAULT_VIZ_THEME]['soma'],
    'axon_default': VIZ_THEMES[DEFAULT_VIZ_THEME]['axon_default'],
    'region_default': '#AAAAAA',
    'region_palette': VIZ_THEMES[DEFAULT_VIZ_THEME]['region_palette'],
}