# =============================================================================
# BETÖLTŐ MODUL - Atlas, szótár, SWC fájlok és soma index kezelése.
# A Streamlit @st.cache_data dekorátorral az atlas csak egyszer töltődik be,
# még akkor is, ha a felhasználó újra kattint valamire.
# =============================================================================

import os
import numpy as np
import pandas as pd
import nrrd
import streamlit as st

from config import ATLAS_PATH, DICTIONARY_PATH, VOXEL_SIZE, SOMA_INDEX_PATH


@st.cache_resource(show_spinner="Loading atlas... (this only happens once)")
def load_atlas() -> tuple[np.ndarray, dict]:
    """
    Betölti az Allen Brain Atlas annotációs mátrixát (.nrrd fájl).
    A @st.cache_resource gondoskodik arról, hogy ez csak egyszer fusson le,
    és az eredmény a memóriában maradjon minden felhasználói interakció között.

    Returns:
        atlas_matrix: 3D numpy tömb, ahol minden érték egy régió ID-ja
        header: az nrrd fejléc metaadatai
    """
    if not os.path.isfile(ATLAS_PATH):
        raise FileNotFoundError(
            f"Atlas file not found at: {ATLAS_PATH}\n"
            f"Please check the ATLAS_PATH in config.py"
        )
    atlas_matrix, header = nrrd.read(ATLAS_PATH)
    return atlas_matrix, header


@st.cache_data(show_spinner="Loading region dictionary...")
def load_dictionary() -> pd.DataFrame:
    """
    Betölti az Allen Brain Atlas régió-szótárát (.csv fájl).
    Csak az 'id', 'acronym' és 'safe_name' oszlopokat olvassa be a memória kímélése érdekében.

    Returns:
        DataFrame az összes régió nevével és ID-jával
    """
    if not os.path.isfile(DICTIONARY_PATH):
        raise FileNotFoundError(
            f"Dictionary file not found at: {DICTIONARY_PATH}\n"
            f"Please check the DICTIONARY_PATH in config.py"
        )
    # A 'structure_id_path' oszlop tartalmazza az Allen hierarchiát
    # (pl. /997/8/343/.../), amiből a szülő-régiók (pl. Brain stem, Thalamus)
    # feloldhatók az összes leszármazott magra. Ha egy másik szótárban nincs meg,
    # akkor is működik minden, csak a szülő-régió kibontás marad ki.
    full = pd.read_csv(DICTIONARY_PATH)
    wanted = ['id', 'acronym', 'safe_name', 'structure_id_path', 'parent_structure_id']
    keep = [c for c in wanted if c in full.columns]
    return full[keep].copy()


def build_region_descendants(
    dictionary: pd.DataFrame,
    region_ids: list[int]
) -> dict[int, set[int]]:
    """
    Minden megadott régió ID-hoz visszaadja azoknak az atlasz-ID-knak a halmazát,
    amelyek maga a régió VAGY annak leszármazottai az Allen hierarchiában.

    Ez teszi lehetővé, hogy egy SZÜLŐ régió (pl. Brain stem, id=343) valóban
    "megfogja" az összes alárendelt magot, hiszen az annotációs térfogat csak a
    levél-régiókat címkézi - a szülő ID önmagában 0 voxelt fedne le.

    Ha a szótárban nincs 'structure_id_path' oszlop, akkor mindenki csak
    önmagára oldódik fel (a régi, pontos egyezéses viselkedés).
    """
    if 'structure_id_path' not in dictionary.columns:
        return {int(rid): {int(rid)} for rid in region_ids}

    paths = dictionary['structure_id_path'].fillna('')
    ids = dictionary['id'].astype(int)

    result: dict[int, set[int]] = {}
    for rid in region_ids:
        rid = int(rid)
        # A '/rid/' minta a szeparátorok miatt csak a pontos ID-t fogja meg
        # (a /343/ nem illeszkedik a /3430/-re), és megfogja az összes olyan
        # leszármazottat, amelynek útvonalában szerepel ez az ős.
        mask = paths.str.contains(f'/{rid}/', regex=False)
        descendants = set(int(v) for v in ids[mask])
        descendants.add(rid)
        result[rid] = descendants
    return result


def load_swc(filepath: str) -> pd.DataFrame:
    """
    Egyetlen SWC fájl beolvasása és alap adattisztítás.
    Az SWC formátum oszlopai: id, type, x, y, z, radius, parent_id

    Args:
        filepath: az SWC fájl teljes útvonala

    Returns:
        Tisztított DataFrame az SWC adatokkal

    Raises:
        FileNotFoundError: ha a fájl nem létezik
        ValueError: ha a fájl nem érvényes SWC formátumban van
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"SWC file not found: {filepath}")

    # '#' karakterrel kezdődő sorok kommentek az SWC formátumban
    swc_df = pd.read_csv(
        filepath,
        comment='#',
        sep=r'\s+',
        header=None,
        names=['id', 'type', 'x', 'y', 'z', 'radius', 'pid']
    )

    # Hiányzó értékek kiszűrése
    swc_df = swc_df.dropna(subset=['id', 'x', 'y', 'z', 'pid'])

    # Duplikált ID-k kezelése (MATLAB-os 'last' logika megtartása)
    swc_df = swc_df.drop_duplicates(subset=['id'], keep='last').reset_index(drop=True)

    if swc_df.empty:
        raise ValueError(f"No valid data found in SWC file: {filepath}")

    return swc_df


def get_all_swc_files(base_dir: str) -> dict[str, str]:
    """
    Rekurzívan megkeresi az összes SWC fájlt a megadott mappában.
    A visszaadott szótárban a kulcs a megjelenítési név, az érték a teljes útvonal.

    Args:
        base_dir: az alap mappa útvonala ahol az egér-almappák vannak

    Returns:
        Dict: {'221227/241.swc': '/teljes/ut/221227/241.swc', ...}
    """
    swc_files = {}

    if not os.path.isdir(base_dir):
        return swc_files

    for root, dirs, files in os.walk(base_dir):
        dirs.sort()
        for filename in sorted(files):
            if filename.lower().endswith('.swc'):
                full_path = os.path.join(root, filename)
                relative_path = os.path.relpath(full_path, base_dir)
                swc_files[relative_path] = full_path

    return swc_files


def get_region_name(dictionary: pd.DataFrame, region_id: int) -> str:
    """
    Megkeresi egy régió nevét az ID alapján a szótárban.

    Args:
        dictionary: a betöltött régió szótár DataFrame
        region_id: az Allen Atlas régió ID

    Returns:
        A régió neve, vagy 'Unknown region' ha nem található
    """
    matches = dictionary.loc[dictionary['id'] == region_id, 'safe_name'].tolist()
    return matches[0] if matches else "Unknown region"


def build_region_search_options(dictionary: pd.DataFrame) -> dict[str, int]:
    """
    Létrehozza a keresési szótárat a UI régió-kereső mezőjéhez.
    Formátum: 'Régió neve (RÖVIDÍTÉS)' -> ID

    Args:
        dictionary: a betöltött régió szótár DataFrame

    Returns:
        Dict ami a megjelenítési névből az ID-ra mutat
    """
    options = {}
    for _, row in dictionary.iterrows():
        display_name = f"{row['safe_name']} ({row['acronym']})"
        options[display_name] = int(row['id'])
    return options


# =============================================================================
# SOMA INDEX - Gyors soma-régió megfeleltetés 12000+ fájlhoz
# =============================================================================

def soma_index_exists() -> bool:
    """Visszaadja, hogy létezik-e már a soma index fájl."""
    return os.path.isfile(SOMA_INDEX_PATH)


def load_soma_index() -> pd.DataFrame | None:
    """
    Betölti a soma index CSV-t ha létezik.
    Oszlopok: swc_path (relatív), soma_region_id, soma_region_name

    Returns:
        DataFrame a soma index adataival, vagy None ha nem létezik a fájl
    """
    if not soma_index_exists():
        return None
    return pd.read_csv(SOMA_INDEX_PATH, dtype={'soma_region_id': int})


def build_soma_index(
    base_dir: str,
    atlas_matrix: np.ndarray,
    dictionary: pd.DataFrame,
    progress_callback=None
) -> pd.DataFrame:
    """
    Felépíti a soma index táblázatot az összes SWC fájlhoz.
    Minden SWC-ből csak a soma sort olvassa be (type == 1),
    ezért sokkal gyorsabb mint a teljes analízis.

    A kész index CSV-be mentődik SOMA_INDEX_PATH-ra, hogy
    a következő alkalmazás indításkor azonnal betölthető legyen.

    Args:
        base_dir: az alap mappa ahol az SWC fájlok vannak
        atlas_matrix: az Allen Brain Atlas 3D mátrixa
        dictionary: régió szótár DataFrame
        progress_callback: opcionális függvény(current, total, filename) a haladás jelzéséhez

    Returns:
        DataFrame az elkészült soma indexszel
    """
    max_x, max_y, max_z = atlas_matrix.shape
    all_swc = get_all_swc_files(base_dir)
    total = len(all_swc)

    rows = []
    for i, (rel_path, full_path) in enumerate(all_swc.items()):
        if progress_callback:
            progress_callback(i, total, rel_path)

        try:
            # Csak a soma sort olvassuk be - hatékonyabb mint az egész fájl
            soma_row = _extract_soma_row(full_path)

            if soma_row is not None:
                sx, sy, sz = soma_row
                # Voxel koordináták kiszámítása
                vox_x = int(np.clip(round(sx / VOXEL_SIZE), 0, max_x - 1))
                vox_y = int(np.clip(round(sy / VOXEL_SIZE), 0, max_y - 1))
                vox_z = int(np.clip(round(sz / VOXEL_SIZE), 0, max_z - 1))
                region_id = int(atlas_matrix[vox_x, vox_y, vox_z])
                region_name_matches = dictionary.loc[
                    dictionary['id'] == region_id, 'safe_name'
                ].tolist()
                region_name = region_name_matches[0] if region_name_matches else "Unknown"
            else:
                # Ha nincs type==1 sor a fájlban, ismeretlen soma-t jelzünk
                region_id = -1
                region_name = "No soma"

        except Exception:
            # Sérült vagy érvénytelen SWC fájlok nem állítják meg az indexelést
            region_id = -1
            region_name = "Error reading file"

        rows.append({
            'swc_path': rel_path,
            'soma_region_id': region_id,
            'soma_region_name': region_name,
        })

    index_df = pd.DataFrame(rows)

    # Mentés CSV-be a gyors jövőbeli betöltéshez
    index_df.to_csv(SOMA_INDEX_PATH, index=False)

    return index_df


def _extract_soma_row(filepath: str) -> tuple[float, float, float] | None:
    """
    Hatékonyan kinyeri az SWC fájlból a soma (type==1) koordinátáit.
    Nem olvassa be az egész fájlt - megáll az első type==1 sornál.

    Args:
        filepath: az SWC fájl teljes útvonala

    Returns:
        (x, y, z) koordináta tuple, vagy None ha nincs soma sor
    """
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Komment sorok kihagyása
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 6 and parts[1] == '1':
                # parts: [id, type, x, y, z, radius, pid]
                return float(parts[2]), float(parts[3]), float(parts[4])
    return None


def filter_swc_by_soma_region(
    all_swc: dict[str, str],
    soma_index: pd.DataFrame,
    search_text: str
) -> dict[str, str]:
    """
    Szűri az SWC fájlok listáját soma régió neve alapján.
    A keresés kis-nagybetű érzéketlen, részleges egyezést is elfogad.

    Például: "motor" megtalálja a "Primary motor area Layer 5" régiót is.

    Args:
        all_swc: az összes SWC fájl szótára {rel_path: full_path}
        soma_index: a betöltött soma index DataFrame
        search_text: a keresési szöveg (részleges egyezés)

    Returns:
        Szűrt SWC fájl szótár
    """
    if not search_text.strip():
        return all_swc

    search_lower = search_text.strip().lower()

    # Megkeressük azokat a rel_path értékeket, ahol a soma régió neve illeszkedik
    matching_paths = soma_index.loc[
        soma_index['soma_region_name'].str.lower().str.contains(
            search_lower, na=False, regex=False
        ),
        'swc_path'
    ].tolist()

    matching_set = set(matching_paths)
    return {k: v for k, v in all_swc.items() if k in matching_set}
