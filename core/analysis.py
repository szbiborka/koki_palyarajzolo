# =============================================================================
# ANALÍZIS MODUL - A tudományos számítási logika.
# Ez a modul semmilyen UI elemet nem tartalmaz - csak tiszta Python/NumPy.
# Ez a szándékos: bármikor tesztelhető és bővíthető a Streamlit-től függetlenül.
# =============================================================================

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from config import VOXEL_SIZE, SWC_TYPE_SOMA, SWC_TYPE_AXON, SWC_TYPE_AXON_UNDEFINED


# =============================================================================
# ADATSTRUKTÚRÁK
# =============================================================================

@dataclass
class RegionResult:
    """
    Egyetlen célterület analízisének eredménye.
    A dataclass automatikusan generál __init__, __repr__ stb. metódusokat.
    """
    region_id: int
    region_name: str
    projects_here: bool           # Vetít-e a sejtünk ide (endpoint vagy branch pont alapján)
    endpoint_count: int           # Csak végpontok száma (gyerek nélküli axon csomópontok)
    branch_point_count: int       # Csak elágazási pontok száma (>1 gyerek)
    projection_point_count: int   # Végpontok + elágazási pontok összesen
    axon_length_um: float         # Axonhossz mikrométerben ebben a régióban


@dataclass
class FilterCriteria:
    """
    A felhasználó által megadott szűrési feltételek egy célterületre.
    Ha minden érték 0, nincs szűrés - minden sejt átmegy.

    Fontos: a feltételek ÉS logikával kapcsolódnak -
    minden megadott feltételnek teljesülnie kell egyszerre.
    """
    min_endpoints: int = 0        # Minimum végpontok száma
    min_branch_points: int = 0    # Minimum elágazási pontok száma
    min_axon_length_um: float = 0 # Minimum axonhossz mikrométerben

    def is_active(self) -> bool:
        """Visszaadja, hogy van-e egyáltalán aktív szűrési feltétel."""
        return self.min_endpoints > 0 or self.min_branch_points > 0 or self.min_axon_length_um > 0

    def check(self, region_result: 'RegionResult') -> bool:
        """
        Ellenőrzi, hogy egy régió eredménye teljesíti-e az összes szűrési feltételt.

        Args:
            region_result: a vizsgált régió analízis eredménye

        Returns:
            True ha a sejt teljesíti az összes feltételt, False ha kiesik
        """
        if region_result.endpoint_count < self.min_endpoints:
            return False
        if region_result.branch_point_count < self.min_branch_points:
            return False
        if region_result.axon_length_um < self.min_axon_length_um:
            return False
        return True


@dataclass
class CellAnalysisResult:
    """
    Egy sejt teljes analízisének összesített eredménye.
    Ezt adja vissza a run_analysis() függvény.
    """
    # Soma adatok
    soma_region_id: int
    soma_region_name: str
    soma_coords: tuple[float, float, float]  # (x, y, z) mikrométerben

    # Célterületek eredményei (a felhasználó által megadott listából)
    target_results: list[RegionResult]

    # Egyéb területek ahol szintén van vetítés (nem a céllistából)
    other_projection_regions: list[RegionResult]

    # Teljes axonhossz
    total_axon_length_um: float

    # Szűrési státusz - True ha a sejt átmegy az összes szűrési feltételen
    # None értéket kap, ha nem volt aktív szűrés
    passes_filter: bool | None = None

    # Nyers adatok a vizualizációhoz - itt tároljuk el a feldolgozott koordinátákat
    # hogy a viz modul ne kelljen, hogy újraszámoljon mindent
    coords: dict = field(default_factory=dict)


# =============================================================================
# FŐ ANALÍZIS FÜGGVÉNY
# =============================================================================

def run_analysis(
    swc_df: pd.DataFrame,
    atlas_matrix: np.ndarray,
    dictionary: pd.DataFrame,
    target_region_ids: list[int]
) -> CellAnalysisResult:
    """
    Elvégzi egy sejt teljes analízisét.

    A vetítés meghatározásának logikája (ez a lényeges különbség más szoftverektől):
    - NEM elegendő, hogy az axon áthalad egy területen
    - CSAK akkor számít vetítésnek, ha a területen belül van VÉGPONT vagy ELÁGAZÁS
    - Végpont: 0 gyerek csomópont
    - Elágazás: >1 gyerek csomópont

    Args:
        swc_df: a betöltött és tisztított SWC adatok
        atlas_matrix: az Allen Brain Atlas 3D annotációs mátrixa
        dictionary: régió neveket tartalmazó szótár DataFrame
        target_region_ids: a vizsgálni kívánt célterületek ID listája

    Returns:
        CellAnalysisResult az összes eredménnyel
    """
    # --- 1. Koordináták és típusok kinyerése ---
    max_x, max_y, max_z = atlas_matrix.shape

    id_arr = np.round(swc_df['id'].values).astype(int)
    type_arr = np.round(swc_df['type'].values).astype(int)
    x = swc_df['x'].values
    y = swc_df['y'].values
    z = swc_df['z'].values
    pid_arr = np.round(swc_df['pid'].values).astype(int)

    # Koordináták átváltása voxel indexekké, klippelés a mátrix határain belülre
    vox_x = np.clip(np.round(x / VOXEL_SIZE).astype(int), 0, max_x - 1)
    vox_y = np.clip(np.round(y / VOXEL_SIZE).astype(int), 0, max_y - 1)
    vox_z = np.clip(np.round(z / VOXEL_SIZE).astype(int), 0, max_z - 1)

    # Minden pont melyik régióba esik
    point_regions = atlas_matrix[vox_x, vox_y, vox_z]

    # --- 2. Hálózati struktúra felépítése (szülő-gyerek kapcsolatok) ---
    # ID -> index szótár a gyors kereséshez
    id_to_idx = {val: idx for idx, val in enumerate(id_arr)}

    # Minden pont szülőjének indexe (-1 ha nincs szülő vagy gyökér csomópont)
    parent_row_indices = np.array([id_to_idx.get(p, -1) for p in pid_arr])
    valid_connections = (parent_row_indices != -1) & (pid_arr != -1)

    # Gyerekek száma minden csomópontra
    p_rows = parent_row_indices[valid_connections]
    child_counts = np.bincount(p_rows, minlength=len(id_arr))

    # --- 3. Axon maszk ---
    # A 2-es típus az axon, a 0-ás típus egyes fájlokban szintén axont jelöl
    is_axon = (type_arr == SWC_TYPE_AXON) | (type_arr == SWC_TYPE_AXON_UNDEFINED)

    # --- 4. Vetítési pontok azonosítása (a kulcslépés!) ---
    # Végpontok és elágazások külön-külön tárolva a szűréshez
    ep_idx = np.where((child_counts == 0) & is_axon)[0]
    branch_idx = np.where((child_counts > 1) & is_axon)[0]
    proj_idx = np.union1d(ep_idx, branch_idx)

    ep_regions = point_regions[ep_idx]
    branch_regions = point_regions[branch_idx]
    proj_regions = point_regions[proj_idx]

    # --- 5. Axonhossz számítás (Pitagorasz-tétel vektorizálva) ---
    curr_idx = np.where(valid_connections)[0]
    p_idx = parent_row_indices[curr_idx]
    dx = x[curr_idx] - x[p_idx]
    dy = y[curr_idx] - y[p_idx]
    dz = z[curr_idx] - z[p_idx]
    distances = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    # Axon szegmens maszk (csak az axon típusú csomópontok között mért távolságok)
    axon_mask_curr = is_axon[curr_idx]
    total_axon_length = float(np.sum(distances[axon_mask_curr]))

    # --- 6. Soma megkeresése ---
    soma_idx_arr = np.where(type_arr == SWC_TYPE_SOMA)[0]
    if len(soma_idx_arr) > 0:
        soma_idx = soma_idx_arr[0]
        soma_region_id = int(point_regions[soma_idx])
        soma_name_matches = dictionary.loc[
            dictionary['id'] == soma_region_id, 'safe_name'
        ].tolist()
        soma_name = soma_name_matches[0] if soma_name_matches else "Unknown region"
        soma_coords = (float(x[soma_idx]), float(y[soma_idx]), float(z[soma_idx]))
    else:
        soma_idx = None
        soma_region_id = -1
        soma_name = "No soma found"
        soma_coords = (0.0, 0.0, 0.0)

    # --- 7. Célterületek kiértékelése ---
    target_results = []
    for region_id in target_region_ids:
        # Régió neve a szótárból
        name_matches = dictionary.loc[dictionary['id'] == region_id, 'safe_name'].tolist()
        region_name = name_matches[0] if name_matches else f"Unknown (ID: {region_id})"

        # Végpontok és elágazások száma külön-külön ebben a régióban
        ep_count = int(np.sum(ep_regions == region_id))
        br_count = int(np.sum(branch_regions == region_id))
        proj_count = ep_count + br_count

        # Axonhossz ebben a régióban
        region_axon_mask = axon_mask_curr & (point_regions[curr_idx] == region_id)
        axon_len = float(np.sum(distances[region_axon_mask]))

        target_results.append(RegionResult(
            region_id=region_id,
            region_name=region_name,
            projects_here=(proj_count > 0),
            endpoint_count=ep_count,
            branch_point_count=br_count,
            projection_point_count=proj_count,
            axon_length_um=axon_len
        ))

    # --- 8. Egyéb vetítési területek (amelyek nem szerepelnek a céllistában) ---
    unique_proj_regions = np.unique(proj_regions[proj_regions > 0])
    other_region_ids = unique_proj_regions[
        ~np.isin(unique_proj_regions, target_region_ids) &
        (unique_proj_regions != soma_region_id)
    ]

    other_projection_regions = []
    for region_id in other_region_ids:
        name_matches = dictionary.loc[dictionary['id'] == region_id, 'safe_name'].tolist()
        region_name = name_matches[0] if name_matches else f"Unknown (ID: {region_id})"
        ep_count = int(np.sum(ep_regions == region_id))
        br_count = int(np.sum(branch_regions == region_id))
        proj_count = ep_count + br_count
        region_axon_mask = axon_mask_curr & (point_regions[curr_idx] == region_id)
        axon_len = float(np.sum(distances[region_axon_mask]))

        other_projection_regions.append(RegionResult(
            region_id=int(region_id),
            region_name=region_name,
            projects_here=True,
            endpoint_count=ep_count,
            branch_point_count=br_count,
            projection_point_count=proj_count,
            axon_length_um=axon_len
        ))

    # --- 9. Nyers koordináták csomagolása a vizualizációhoz ---
    # Ezeket a viz modul fogja használni, nem kell újraszámolni
    coords = {
        'x': x, 'y': y, 'z': z,
        'type_arr': type_arr,
        'is_axon': is_axon,
        'point_regions': point_regions,
        'proj_idx': proj_idx,
        'ep_idx': ep_idx,
        'branch_idx': branch_idx,
        'curr_idx': curr_idx,
        'parent_row_indices': parent_row_indices,
        'soma_idx': soma_idx,
        'valid_connections': valid_connections,
    }

    return CellAnalysisResult(
        soma_region_id=soma_region_id,
        soma_region_name=soma_name,
        soma_coords=soma_coords,
        target_results=target_results,
        other_projection_regions=other_projection_regions,
        total_axon_length_um=total_axon_length,
        coords=coords
    )


# =============================================================================
# SZŰRÉS
# =============================================================================

def apply_filter(
    result: CellAnalysisResult,
    criteria_per_region: dict[int, FilterCriteria]
) -> CellAnalysisResult:
    """
    Ellenőrzi, hogy egy sejt analízis eredménye megfelel-e a szűrési feltételeknek.
    A szűrés ÉS logikával működik: MINDEN célterületre teljesülnie kell a feltételnek.

    Módosítja a result.passes_filter mezőt és visszaadja az objektumot.
    Nem változtat semmin más a result-ban - az eredeti adatok megmaradnak.

    Args:
        result: a run_analysis() által visszaadott eredmény
        criteria_per_region: dict, amelynek kulcsa a régió ID, értéke a FilterCriteria

    Returns:
        Az ugyanaz a CellAnalysisResult, passes_filter mezővel kitöltve
    """
    # Ha nincs egyetlen aktív feltétel sem, minden sejt átmegy
    if not any(c.is_active() for c in criteria_per_region.values()):
        result.passes_filter = None  # None = nem volt szűrés
        return result

    # Minden célterületre ellenőrzés
    for tr in result.target_results:
        criteria = criteria_per_region.get(tr.region_id)
        if criteria is None or not criteria.is_active():
            continue  # Erre a régióra nincs feltétel, kihagyjuk

        if not criteria.check(tr):
            result.passes_filter = False
            return result

    result.passes_filter = True
    return result


# =============================================================================
# EXPORTÁLÁS
# =============================================================================

def results_to_dataframe(
    results: list[tuple[str, CellAnalysisResult]],
    target_region_ids: list[int],
    dictionary: pd.DataFrame
) -> pd.DataFrame:
    """
    Több sejt analízis eredményét összesíti egy DataFrame-be exportáláshoz.
    Tartalmazza az endpoint és branch point számokat külön oszlopokban,
    és a passes_filter státuszt is.

    Args:
        results: lista (sejt_név, CellAnalysisResult) párokból
        target_region_ids: a vizsgált célterületek ID-jai
        dictionary: régió szótár

    Returns:
        DataFrame ahol minden sor egy sejt, oszlopok a régiók részletes adataival
    """
    rows = []
    for cell_name, result in results:
        row = {
            'cell': cell_name,
            'soma_region': result.soma_region_name,
            'total_axon_length_um': round(result.total_axon_length_um, 1),
            'passes_filter': result.passes_filter,
        }
        # Minden célterülethez részletes oszlopok
        for tr in result.target_results:
            safe_col = tr.region_name.replace(' ', '_').lower()[:30]
            row[f'{safe_col}_projects'] = tr.projects_here
            row[f'{safe_col}_endpoints'] = tr.endpoint_count
            row[f'{safe_col}_branches'] = tr.branch_point_count
            row[f'{safe_col}_axon_um'] = round(tr.axon_length_um, 1)

        rows.append(row)

    return pd.DataFrame(rows)
