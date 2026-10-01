# BETÖLTŐ MODUL - Atlas, szótár, SWC fájlok és soma index kezelése.
# A Streamlit cache dekorátorokkal a lassú lépések (atlas, fájllista, index)
# csak egyszer futnak le, nem minden felhasználói kattintásnál.

import os

import numpy as np
import pandas as pd
import nrrd
import streamlit as st

from config import (
    ATLAS_PATH, DICTIONARY_PATH, VOXEL_SIZE, SOMA_INDEX_PATH,
    BRAINSTEM_MOTOR_ID, BRAINSTEM_MOTOR_NAME, BRAINSTEM_MOTOR_ACRONYM,
    BRAINSTEM_MOTOR_COMPONENTS, SWC_TYPE_SOMA,
)

SWC_COLUMNS = ['id', 'type', 'x', 'y', 'z', 'radius', 'pid']


@st.cache_resource(show_spinner="Loading atlas... (this only happens once)")
def load_atlas() -> np.ndarray:
    """
    Betölti az Allen Brain Atlas annotációs mátrixát (.nrrd fájl).
    3D numpy tömb, ahol minden voxel értéke egy (levél-)régió ID-ja.
    """
    if not os.path.isfile(ATLAS_PATH):
        raise FileNotFoundError(
            f"Atlas file not found at: {ATLAS_PATH}\n"
            f"Please check the ATLAS_PATH in config.py"
        )
    atlas_matrix, _header = nrrd.read(ATLAS_PATH)
    return atlas_matrix


@st.cache_data(show_spinner="Loading region dictionary...")
def load_dictionary() -> pd.DataFrame:
    """
    Betölti az Allen Brain Atlas régió-szótárát (.csv fájl).

    A 'structure_id_path' oszlop tartalmazza az Allen hierarchiát
    (pl. /997/8/343/.../), amiből a szülő-régiók (pl. Brain stem, Thalamus)
    feloldhatók az összes leszármazott magra. Ha egy másik szótárban nincs meg,
    akkor is működik minden, csak a szülő-régió kibontás marad ki.
    """
    if not os.path.isfile(DICTIONARY_PATH):
        raise FileNotFoundError(
            f"Dictionary file not found at: {DICTIONARY_PATH}\n"
            f"Please check the DICTIONARY_PATH in config.py"
        )
    full = pd.read_csv(DICTIONARY_PATH)
    wanted = ['id', 'acronym', 'safe_name', 'structure_id_path']
    return full[[c for c in wanted if c in full.columns]].copy()


def region_name_map(dictionary: pd.DataFrame) -> dict[int, str]:
    """Régió ID -> név, a virtuális "leszálló agytörzzsel" együtt. Ez az egyetlen névforrás."""
    names = dict(zip(dictionary['id'].astype(int), dictionary['safe_name']))
    names[BRAINSTEM_MOTOR_ID] = BRAINSTEM_MOTOR_NAME
    return names


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
    önmagára oldódik fel (pontos egyezéses viselkedés).
    """
    has_hierarchy = 'structure_id_path' in dictionary.columns
    if has_hierarchy:
        paths = dictionary['structure_id_path'].fillna('')
        ids = dictionary['id'].astype(int)

    def _descendants_of(parent_id: int) -> set[int]:
        """Egy valós Allen ID + összes leszármazottja (a structure_id_path alapján)."""
        if not has_hierarchy:
            return {int(parent_id)}
        # A '/parent_id/' minta a szeparátorok miatt csak a pontos ID-t fogja meg
        # (a /343/ nem illeszkedik a /3430/-re), és megfogja az összes olyan
        # leszármazottat, amelynek útvonalában szerepel ez az ős.
        mask = paths.str.contains(f'/{parent_id}/', regex=False)
        found = set(int(v) for v in ids[mask])
        found.add(int(parent_id))
        return found

    result: dict[int, set[int]] = {}
    for rid in region_ids:
        rid = int(rid)
        if rid == BRAINSTEM_MOTOR_ID:
            # Virtuális "leszálló agytörzs": Középagy + Utóagy leszármazottai,
            # a köztiagy/thalamus KIZÁRVA.
            desc: set[int] = set()
            for comp in BRAINSTEM_MOTOR_COMPONENTS:
                desc |= _descendants_of(comp)
            result[rid] = desc
        else:
            result[rid] = _descendants_of(rid)
    return result


def load_swc(filepath: str) -> pd.DataFrame:
    """
    Egyetlen SWC fájl beolvasása és alap adattisztítás.
    Az SWC formátum oszlopai: id, type, x, y, z, radius, parent_id

    Raises:
        FileNotFoundError: ha a fájl nem létezik
        ValueError: ha a fájl nem érvényes SWC formátumban van
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"SWC file not found: {filepath}")

    # '#' karakterrel kezdődő sorok kommentek az SWC formátumban. A usecols
    # kötelező: ha egy fájlban 7-nél több oszlop van, a pandas különben az
    # első oszlopot indexnek venné, és csendben elcsúsznának az oszlopok.
    swc_df = pd.read_csv(
        filepath,
        comment='#',
        sep=r'\s+',
        header=None,
        names=SWC_COLUMNS,
        usecols=range(len(SWC_COLUMNS)),
    )

    swc_df = swc_df.dropna(subset=['id', 'type', 'x', 'y', 'z', 'pid'])

    # Duplikált ID-k kezelése (MATLAB-os 'last' logika megtartása)
    swc_df = swc_df.drop_duplicates(subset=['id'], keep='last').reset_index(drop=True)

    if swc_df.empty:
        raise ValueError(f"No valid data found in SWC file: {filepath}")

    return swc_df


def _normalize_rel_path(path: str) -> str:
    """Operációs rendszertől független relatív útvonal ('/' elválasztóval)."""
    return path.replace('\\', '/')


@st.cache_data(show_spinner="Scanning SWC files...")
def get_all_swc_files(base_dir: str) -> dict[str, str]:
    """
    Rekurzívan megkeresi az összes SWC fájlt a megadott mappában.

    Returns:
        {relatív útvonal '/' elválasztóval: teljes útvonal},
        pl. {'221227/241.swc': '/teljes/ut/221227/241.swc'}
    """
    swc_files = {}
    if not os.path.isdir(base_dir):
        return swc_files

    for root, dirs, files in os.walk(base_dir):
        dirs.sort()
        for filename in sorted(files):
            if filename.lower().endswith('.swc'):
                full_path = os.path.join(root, filename)
                swc_files[_normalize_rel_path(os.path.relpath(full_path, base_dir))] = full_path
    return swc_files


def build_region_search_options(region_names: dict[int, str], dictionary: pd.DataFrame) -> dict[str, int]:
    """
    A UI régió-kereső opciói: 'Régió neve (RÖVIDÍTÉS)' -> ID.
    A virtuális "leszálló agytörzs" legelöl áll, mert a pyramidal tract sejtek
    helyes kiválasztásához ezt kell használni a nyers "Brain stem" helyett.
    """
    options = {f"{region_names[BRAINSTEM_MOTOR_ID]} ({BRAINSTEM_MOTOR_ACRONYM})": BRAINSTEM_MOTOR_ID}
    for rid, acronym in zip(dictionary['id'].astype(int), dictionary['acronym']):
        options[f"{region_names[rid]} ({acronym})"] = rid
    return options


# SOMA INDEX - Gyors soma-régió megfeleltetés 12000+ fájlhoz

def soma_index_exists() -> bool:
    """Visszaadja, hogy létezik-e már a soma index fájl."""
    return os.path.isfile(SOMA_INDEX_PATH)


@st.cache_data(show_spinner=False)
def load_soma_index() -> pd.DataFrame | None:
    """
    Betölti a soma index CSV-t, ha létezik.
    Oszlopok: swc_path (relatív, '/' elválasztóval), soma_region_id, soma_region_name
    """
    if not soma_index_exists():
        return None
    index_df = pd.read_csv(SOMA_INDEX_PATH, dtype={'soma_region_id': int})
    # A Windows alatt épült régi indexek '\'-t tartalmaznak.
    index_df['swc_path'] = index_df['swc_path'].map(_normalize_rel_path)
    return index_df


def build_soma_index(
    base_dir: str,
    atlas_matrix: np.ndarray,
    region_names: dict[int, str],
    progress_callback=None
) -> pd.DataFrame:
    """
    Felépíti a soma index táblázatot az összes SWC fájlhoz, és SOMA_INDEX_PATH-ra menti.
    Minden SWC-ből csak a soma sort olvassa be (type == 1), ezért sokkal
    gyorsabb, mint a teljes analízis.

    Args:
        progress_callback: opcionális függvény(current, total, filename) a haladás jelzéséhez
    """
    shape = atlas_matrix.shape
    all_swc = get_all_swc_files(base_dir)
    total = len(all_swc)

    rows = []
    for i, (rel_path, full_path) in enumerate(all_swc.items()):
        if progress_callback:
            progress_callback(i, total, rel_path)
        try:
            soma_xyz = _extract_soma_row(full_path)
            if soma_xyz is None:
                region_id, region_name = -1, "No soma"
            else:
                voxel = tuple(int(np.clip(round(c / VOXEL_SIZE), 0, n - 1)) for c, n in zip(soma_xyz, shape))
                region_id = int(atlas_matrix[voxel])
                region_name = region_names.get(region_id, "Unknown")
        except Exception:
            # Sérült vagy érvénytelen SWC fájlok nem állítják meg az indexelést
            region_id, region_name = -1, "Error reading file"

        rows.append({'swc_path': rel_path, 'soma_region_id': region_id, 'soma_region_name': region_name})

    index_df = pd.DataFrame(rows)
    index_df.to_csv(SOMA_INDEX_PATH, index=False)
    return index_df


def _extract_soma_row(filepath: str) -> tuple[float, float, float] | None:
    """
    Kinyeri az SWC fájlból az első soma (type==1) sor koordinátáit, a fájl
    többi részének beolvasása nélkül. None, ha nincs soma sor.
    """
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            # A típus lehet '1' vagy '1.0' is - float(...) mindkettőt kezeli.
            try:
                is_soma = len(parts) >= 6 and int(float(parts[1])) == SWC_TYPE_SOMA
            except ValueError:
                is_soma = False
            if is_soma:
                # parts: [id, type, x, y, z, radius, pid]
                return float(parts[2]), float(parts[3]), float(parts[4])
    return None


def filter_swc_by_soma_region(
    all_swc: dict[str, str],
    soma_index: pd.DataFrame,
    search_text: str
) -> dict[str, str]:
    """
    Szűri az SWC fájlok listáját soma régió neve alapján (kis-nagybetű
    érzéketlen, részleges egyezés: "motor" -> "Primary motor area Layer 5").
    """
    search_lower = search_text.strip().lower()
    if not search_lower:
        return all_swc

    matches = soma_index['soma_region_name'].str.lower().str.contains(search_lower, na=False, regex=False)
    matching_set = set(soma_index.loc[matches, 'swc_path'])
    return {k: v for k, v in all_swc.items() if k in matching_set}
