# =============================================================================
# KONFIGURÁCIÓ - Minden útvonal és konstans itt van definiálva.
# Ha szerverre költözik az alkalmazás, CSAK ezt a fájlt kell módosítani.
# =============================================================================

import os

# --- Adatfájlok útvonalai ---
# Ezek az útvonalak a saját gépen érvényesek. Szerveren felülírhatók
# környezeti változókkal (pl. export ATLAS_PATH=/data/atlas/annotation_25.nrrd)
BASE_DATA_DIR = os.environ.get(
    'PALYAKOVETO_DATA_DIR',
    '/home/bibi/Documents/koki/swc_in_ccf/data_v2/'
)

ATLAS_PATH = os.environ.get(
    'ATLAS_PATH',
    '/home/bibi/Documents/koki/annotation_25.nrrd'
)

DICTIONARY_PATH = os.environ.get(
    'DICTIONARY_PATH',
    '/home/bibi/Documents/koki/query.csv'
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

# --- Sejttípus kódok az SWC formátumban ---
# Ez a szabványos SWC specifikáció szerint van definiálva.
SWC_TYPE_SOMA = 1
SWC_TYPE_AXON = 2
SWC_TYPE_AXON_UNDEFINED = 0  # Egyes fájlokban a 0-ás típus is axont jelöl

# --- Vizualizációs beállítások ---
VIZ_REGION_OPACITY = 0.25        # Agyterület felszínek átlátszósága
VIZ_SOMA_RADIUS = 15             # Soma gömb sugara mikrométerben
VIZ_POINT_SIZE = 10              # Vetítési pontok mérete
VIZ_AXON_LINE_WIDTH = 2         # Axon vonalak vastagsága
VIZ_MARCHING_CUBES_STEP = 2     # Felszín-generálás lépésköze (kisebb = szebb, de lassabb)

# --- Szín paletta ---
# A különböző régiók megjelenítési színei
COLORS = {
    'soma': 'black',
    'axon_default': '#888888',   # Szürke - alapértelmezett axon szín
    'region_default': '#AAAAAA', # Szürke - ha nincs egyedi szín megadva
    'region_palette': [          # Körkörösen hozzárendelt színek a régiókhoz
        '#1f77b4',  # kék
        '#2ca02c',  # zöld
        '#d62728',  # piros
        '#9467bd',  # lila
        '#8c564b',  # barna
        '#e377c2',  # rózsaszín
        '#17becf',  # cián
        '#bcbd22',  # sárga-zöld
    ]
}
