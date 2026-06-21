# =============================================================================
# BETÖLTŐ MODUL - Atlas, szótár és SWC fájlok beolvasása.
# A Streamlit @st.cache_data dekorátorral az atlas csak egyszer töltődik be,
# még akkor is, ha a felhasználó újra kattint valamire.
# =============================================================================

import os
import numpy as np
import pandas as pd
import nrrd
import streamlit as st

from config import ATLAS_PATH, DICTIONARY_PATH, VOXEL_SIZE


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
    return pd.read_csv(DICTIONARY_PATH, usecols=['id', 'acronym', 'safe_name'])


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
    Ez a függvény teszi lehetővé, hogy az egér-mappák automatikusan megjelenjenek a UI-ban.

    Args:
        base_dir: az alap mappa útvonala ahol az egér-almappák vannak

    Returns:
        Dict: {'221227/241.swc': '/teljes/ut/221227/241.swc', ...}
    """
    swc_files = {}

    if not os.path.isdir(base_dir):
        return swc_files

    # Bejárjuk az összes almappát
    for root, dirs, files in os.walk(base_dir):
        # Rendezzük abc sorrendbe az átláthatóság érdekében
        dirs.sort()
        for filename in sorted(files):
            if filename.lower().endswith('.swc'):
                full_path = os.path.join(root, filename)
                # A megjelenítési névben a base_dir-hez képesti relatív útvonalat mutatjuk
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
