# KONFIGURÁCIÓ - Minden útvonal és konstans itt van definiálva.
# Ha szerverre költözik az alkalmazás, CSAK ezt a fájlt kell módosítani.

import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(PROJECT_DIR, 'adatfajlok')

# --- Adatfájlok útvonalai ---
# Alapértelmezés: a projektmappán belüli 'adatfajlok/'. Szerveren felülírhatók
# környezeti változókkal (pl. export ATLAS_PATH=/data/atlas/annotation_25.nrrd)
BASE_DATA_DIR = os.environ.get('PALYAKOVETO_DATA_DIR', os.path.join(_DATA_DIR, 'data_v2'))
ATLAS_PATH = os.environ.get('ATLAS_PATH', os.path.join(_DATA_DIR, 'annotation_25.nrrd'))
DICTIONARY_PATH = os.environ.get('DICTIONARY_PATH', os.path.join(_DATA_DIR, 'query.csv'))

# --- Adatbázis-metaadat (mouse.digital-brain.cn, kéreg adatkészlet) ---
# Sejtenként: az adatbázis saját soma-régiója, Cre vonal, féltekéje és a
# VETÍTÉSI OSZTÁLY (IT / PT / CT, pl. 'PT-18'). Forrás:
#   https://mouse.digital-brain.cn/projectome/2/srv//info/mouse/cortex/mouse.neuron.info.json
# Ha a fájl hiányzik, az app ezek nélkül is működik.
DATABASE_METADATA_PATH = os.environ.get(
    'PALYAKOVETO_METADATA',
    os.path.join(_DATA_DIR, 'database_metadata', 'cortex_neuron_info.json'))

# --- Kézi ellenőrzés (kuráció) ---
# A kézzel ellenőrzött sejtek ítélete (pl. "eltolódott L6"), hogy ugyanazt a
# sejtet ne kelljen minden elemzésnél újra kiszűrni. Kicsi, verziókövetett fájl.
CURATION_PATH = os.environ.get('PALYAKOVETO_CURATION', os.path.join(PROJECT_DIR, 'curation', 'manual_labels.csv'))
CURATION_LABELS = {
    'shifted_L6': 'Shifted L6 (registration offset → false region hits)',
    'actually_L5': 'Actually L5 (soma on the L5/L6 border)',
    'cortico_cortical': 'Cortico-cortical (IT)',
    'other_problem': 'Other problem',
    'verified': 'Verified OK',
}
# Ezeket a címkéket az app alapértelmezésben KIZÁRJA az elemzésből.
DEFAULT_EXCLUDED_LABELS = ['shifted_L6', 'other_problem']

# --- Soma index fájl ---
# Ez a fájl tárolja el az összes SWC fájl soma-régió megfeleltetését.
# Első futáskor épül fel, utána gyorsan betöltődik.
# Ha új SWC fájlok kerülnek a mappába, a UI-ban lévő "Rebuild index" gombbal frissíthető.
SOMA_INDEX_PATH = os.environ.get(
    'SOMA_INDEX_PATH',
    os.path.join(PROJECT_DIR, 'soma_index.csv')
)

# --- Atlas paraméterek ---
# Voxel méret mikrométerben (a 25-ös atlasz 25um felbontású)
VOXEL_SIZE = 25

# µm -> voxelindex konvenció.
#   'floor' (alapértelmezés, az Allen-eszközök konvenciója): az i. voxel a
#           [25*i, 25*(i+1)) µm tartomány. Az adatbázis saját soma-régióival
#           95,07% az egyezés (18 621 sejt).
#   'round' (a 2026-10-02 előtti viselkedés): 94,04% egyezés. Csak a korábbi
#           (pl. poszter-) táblák pontos reprodukálásához.
# A két konvenció 1650 sejt (8,9%) soma-régióját sorolja be eltérően - ezek
# közül 907-nél a 'floor', 716-nál a 'round' egyezik az adatbázissal.
# Váltás után a soma indexet újra kell építeni (Rebuild Index).
VOXEL_LOOKUP = os.environ.get('PALYAKOVETO_VOXEL_LOOKUP', 'floor')

# --- FÉLTEKE (oldaliság) ---
# Az Allen atlaszban MINDKÉT félteke UGYANAZT a régió-ID-t viseli: nincs külön
# "bal GPe" és "jobb GPe". Ezért önmagában a régió-ID alapján nem lehet
# megkülönböztetni az azonos oldali (ipszilaterális) és az ellenoldali
# (kontralaterális) vetítést.
#
# A középvonal viszont fix koordináta: a 25um-es CCF térfogat alakja
# (AP, DV, ML) = (528, 320, 456), tehát a MEDIO-LATERÁLIS tengely az UTOLSÓ
# (a kódban 'z'). A középvonal ennek a tengelynek a fele. Ebből minden pontról
# eldönthető, hogy a soma oldalán van-e vagy sem.
#
# Biológiailag ez számít: az L5 pyramidal tract sejtek gyakorlatilag
# IPSZILATERÁLISAN vetítenek a GPe/TRN/agytörzs felé, míg a kortiko-kortikális
# axonok átkelnek a középvonalon. Ha mindkét oldalt beleszámoljuk, az csak
# felfelé torzíthatja a GPe/TRN számokat.
MIDLINE_AXIS = 2  # 0=AP, 1=DV, 2=ML (a kódban x, y, z sorrendben)

# Melyik oldali vetítéseket számoljuk. Az alapértelmezés a KORÁBBI viselkedés
# ('both'), hogy a már elküldött eredmények reprodukálhatók maradjanak; az
# oldalsávban átállítható.
# Rövid felirat a rádiógombhoz -> (belső kód, magyarázat)
LATERALITY_MODES = {
    'Both sides':    ('both',   'Counts a projection on either hemisphere (previous behaviour).'),
    'Ipsilateral':   ('ipsi',   'Only the soma’s own side counts. L5 pyramidal-tract cells '
                                'project essentially ipsilaterally, so this is the '
                                'anatomically strict choice.'),
    'Contralateral': ('contra', 'Only the opposite side counts — useful to see how much of a '
                                'target is reached across the midline.'),
}
DEFAULT_LATERALITY = 'both'

# Mennyi kontralaterális AXONHOSSZ (um) kell ahhoz, hogy kijelentsük: az axon
# TÉNYLEG átlépte a középvonalt. Nem nulla a küszöb, mert a középvonal közelében
# futó axon a 25 um-es rács kerekítése miatt néhány mintányit "átlóghat" a
# túloldalra anélkül, hogy valóban átkelne. Két voxelnyi hossz már nem kerekítési
# hiba. Ez CSAK az "átkel-e" jelzőt érinti; a végpontok számlálását nem.
CONTRA_CROSSING_MIN_AXON_UM = 50.0

# A középvonal ±ennyi um-es sávjában lévő pont (és soma) oldala eldönthetetlen:
# sem ipszi-, sem kontralaterálisnak nem számít. A regisztráció ~2 voxeles
# bizonytalansága miatt egy épp átlógó helyi mellékág különben "kontralaterális
# vetítésnek" számítana (a PT sejtek 1,3%-ánál ez volt az egyetlen ok).
MIDLINE_BAND_UM = 50.0

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

# --- Virtuális "thalamus a TRN nélkül" ---
# Az Allen ontológiában a TRN (262) a Thalamus (549) LESZÁRMAZOTTJA. Ha a
# "Thalamus" és a "TRN" egyszerre célterület, minden TRN-végpont thalamikusnak
# is számít: egy NOT-thalamus szabály a TRN-be vetítő sejteket is kizárja, és
# a "TRN only" kategória sosem lehet nem üres.
THALAMUS_NO_TRN_ID = -1002

# Minden virtuális régió: név, rövidítés, a befoglalt és a kivont valós régiók
# (mindegyik a teljes leszármazotti fájával).
VIRTUAL_REGIONS = {
    BRAINSTEM_MOTOR_ID: {'name': BRAINSTEM_MOTOR_NAME, 'acronym': BRAINSTEM_MOTOR_ACRONYM,
                         'include': BRAINSTEM_MOTOR_COMPONENTS, 'exclude': []},
    THALAMUS_NO_TRN_ID: {'name': "Thalamus excluding the reticular nucleus (TRN)", 'acronym': "TH-noRT",
                         'include': [549], 'exclude': [262]},
}

# Régiónként eltérő ALAPÉRTELMEZETT kritérium (a felületen felülírható).
# A leszálló agytörzsnél >= 5 végpont: a valódi PT sejteknek medián 102
# agytörzsi végpontjuk van (98%-uknak >= 5), míg a CT/IT sejtek "agytörzsi
# vetítése" medián 3-4 végpont a thalamussal határos középagyban (MB, APN, MRN) -
# regisztrációs átszivárgás, ugyanaz a jelenség, mint az eltolódott L6 sejteknél.
REGION_DEFAULT_FILTER_OVERRIDES = {
    BRAINSTEM_MOTOR_ID: {'min_endpoints': 5},
}

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
VIZ_MARCHING_CUBES_STEP = 2      # Felszín-generálás lépésköze (kisebb = szebb, de lassabb)
VIZ_DEFAULT_LINE_WIDTH = 4       # Axonvonal vastagsága (px) - a felületen állítható
VIZ_DEFAULT_HEIGHT = 850         # Az ábra magassága (px) - a felületen állítható
# Egy ábrán legfeljebb ennyi axonszakasz: fölötte a fa egyszerűsödik (minden N-edik
# csomópont + MINDEN elágazás és végpont marad, a vonalak folytonosak maradnak).
VIZ_MAX_SEGMENTS = 200_000           # egy sejt nézete
VIZ_MAX_SEGMENTS_COMBINED = 120_000  # a kombinált nézet összesen (áttekintő ábra)

# --- 3D JELENET TÉMÁK ---
# Két teljes téma. A "dark" az alapértelmezett: a vékony axonvonalak világos
# színnel, sötét háttéren sokkal jobban láthatók.
#
# A kategória-paletta a dataviz referencia-palettája (8 szín, rögzített sorrend),
# a mi hátterünkön validálva (2026-10-02, OKLab x100):
#   dark  (#0E1117): egymás melletti párok: CVD ΔE >= 8,4, normál látás >= 19,3,
#                    kontraszt mind >= 3:1 - PASS
#   light (#FAF9F6): CVD ΔE >= 9,1, normál látás >= 19,6 - PASS; három szín 3:1
#                    alatti kontrasztú, ezért a nevük mindig látszik (jelmagyarázat, hover)
# Egy 3D jelenetben BÁRMELY két szín egymás mellé kerülhet ("all pairs"): erre
# csak az első 3 szín felel meg - ezért a sejtek alapértelmezésben a vetítési
# osztály (3 csoport) szerint színeződnek, és 8 csoport fölött a többi a semleges
# "Other" színt kapja. A színt soha nem generáljuk és nem ismételjük ciklikusan.
VIZ_THEMES = {
    'dark': {
        'label': 'Dark (recommended — thin axons stand out)',
        'paper_bg': '#0E1117',        # A teljes ábra háttere
        'scene_bg': '#0E1117',        # A 3D tengelyek háttere
        'grid': '#2C2C2A',            # Visszafogott rács
        'axis_text': '#C3C2B7',       # Tengelyfeliratok
        'soma': '#FFFFFF',            # A soma fehéren a legjobban látszik
        'axon_default': '#A3A29B',    # Nem célterületi axon - jól látható, de színtelen (a szín a célterületeké)
        'neutral': '#898781',         # "Other" csoportok, egyéb régiók
        'soma_region': '#C3C2B7',     # A soma régiója (kontextus)
        'brain_outline': '#898781',   # Agy körvonal (nagyon áttetsző)
        'brain_outline_opacity': 0.05,
        'legend_bg': 'rgba(14,17,23,0.85)',
        'legend_border': '#383835',
        # A régiófelszínek KONTEXTUST adnak, nem ők a főszereplők: alacsony
        # átlátszatlansággal nem nyomják el az axonvonalakat.
        'region_opacity': 0.14,
        # Sok sejt saját színe (OKLCH): két váltakozó világosság, a sötét háttérhez
        'spread_lightness': (0.68, 0.82),
        'spread_chroma': 0.15,
        'region_palette': ['#3987e5', '#d95926', '#199e70', '#c98500',
                           '#d55181', '#008300', '#9085e9', '#e66767'],
    },
    'light': {
        'label': 'Light (matches the app background)',
        'paper_bg': '#FAF9F6',
        'scene_bg': '#FAF9F6',
        'grid': '#E1E0D9',
        'axis_text': '#52514E',
        'soma': '#0B0B0B',
        'axon_default': '#76746C',
        'neutral': '#898781',
        'soma_region': '#52514E',
        'brain_outline': '#52514E',
        'brain_outline_opacity': 0.05,
        'legend_bg': 'rgba(252,252,251,0.9)',
        'legend_border': '#C3C2B7',
        'region_opacity': 0.18,
        'spread_lightness': (0.48, 0.62),
        'spread_chroma': 0.15,
        'region_palette': ['#2a78d6', '#eb6834', '#1baf7a', '#eda100',
                           '#e87ba4', '#008300', '#4a3aa7', '#e34948'],
    },
}
DEFAULT_VIZ_THEME = 'dark'

# --- Agy körvonal ---
# Térbeli tájékozódáshoz kirajzoljuk a teljes agy felszínét nagyon áttetszően.
# Nem egy konkrét régió ID-ra szűrünk, hanem az ÖSSZES annotált voxelre
# (atlas > 0), így akkor is működik, ha a "root" régió nincs a szótárban.
# A nagyobb lépésköz azért kell, mert ez a legnagyobb felület a jelenetben.
VIZ_BRAIN_OUTLINE_STEP = 6