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
    """Egyetlen célterület analízisének eredménye."""
    region_id: int
    region_name: str
    projects_here: bool  # Vetít-e a sejtünk ide (endpoint vagy branch pont alapján)
    endpoint_count: int  # Csak végpontok száma (gyerek nélküli axon csomópontok)
    branch_point_count: int  # Csak elágazási pontok száma (>1 gyerek)
    projection_point_count: int  # Végpontok + elágazási pontok összesen
    axon_length_um: float  # Axonhossz mikrométerben ebben a régióban


@dataclass
class FilterCriteria:
    """
    A felhasználó által megadott szűrési feltételek egy célterületre,
    kiegészítve a halmazműveleti logikával (AND, OR, NOT).
    """
    min_endpoints: int = 0
    min_branch_points: int = 0
    min_axon_length_um: float = 0
    operator: str = 'AND'  # 'AND', 'OR', 'NOT'

    def is_active(self) -> bool:
        """
        A szűrő aktív, ha bármelyik küszöbérték > 0,
        VAGY ha a felhasználó kifejezetten NOT vagy OR logikát állított be.
        """
        return (self.min_endpoints > 0 or
                self.min_branch_points > 0 or
                self.min_axon_length_um > 0 or
                self.operator != 'AND')

    def meets_thresholds(self, region_result: 'RegionResult') -> bool:
        """Kiértékeli, hogy a régió önmagában megüti-e a küszöböt."""
        # Ha a küszöbök 0-k, de a szűrő aktív (pl. NOT vagy OR),
        # akkor simán azt vizsgáljuk, hogy egyáltalán vetít-e ide.
        if self.min_endpoints == 0 and self.min_branch_points == 0 and self.min_axon_length_um == 0:
            return region_result.projects_here

        if region_result.endpoint_count < self.min_endpoints:
            return False
        if region_result.branch_point_count < self.min_branch_points:
            return False
        if region_result.axon_length_um < self.min_axon_length_um:
            return False
        return True


@dataclass
class CellAnalysisResult:
    """Egy sejt teljes analízisének összesített eredménye."""
    soma_region_id: int
    soma_region_name: str
    soma_coords: tuple[float, float, float]
    target_results: list[RegionResult]
    other_projection_regions: list[RegionResult]
    total_axon_length_um: float
    passes_filter: bool | None = None
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
    max_x, max_y, max_z = atlas_matrix.shape

    id_arr = np.round(swc_df['id'].values).astype(int)
    type_arr = np.round(swc_df['type'].values).astype(int)
    x = swc_df['x'].values
    y = swc_df['y'].values
    z = swc_df['z'].values
    pid_arr = np.round(swc_df['pid'].values).astype(int)

    vox_x = np.clip(np.round(x / VOXEL_SIZE).astype(int), 0, max_x - 1)
    vox_y = np.clip(np.round(y / VOXEL_SIZE).astype(int), 0, max_y - 1)
    vox_z = np.clip(np.round(z / VOXEL_SIZE).astype(int), 0, max_z - 1)
    point_regions = atlas_matrix[vox_x, vox_y, vox_z]

    id_to_idx = {val: idx for idx, val in enumerate(id_arr)}
    parent_row_indices = np.array([id_to_idx.get(p, -1) for p in pid_arr])
    valid_connections = (parent_row_indices != -1) & (pid_arr != -1)

    p_rows = parent_row_indices[valid_connections]
    child_counts = np.bincount(p_rows, minlength=len(id_arr))

    is_axon = (type_arr == SWC_TYPE_AXON) | (type_arr == SWC_TYPE_AXON_UNDEFINED)

    ep_idx = np.where((child_counts == 0) & is_axon)[0]
    branch_idx = np.where((child_counts > 1) & is_axon)[0]
    proj_idx = np.union1d(ep_idx, branch_idx)

    ep_regions = point_regions[ep_idx]
    branch_regions = point_regions[branch_idx]
    proj_regions = point_regions[proj_idx]

    curr_idx = np.where(valid_connections)[0]
    p_idx = parent_row_indices[curr_idx]
    distances = np.sqrt((x[curr_idx] - x[p_idx]) ** 2 + (y[curr_idx] - y[p_idx]) ** 2 + (z[curr_idx] - z[p_idx]) ** 2)

    axon_mask_curr = is_axon[curr_idx]
    total_axon_length = float(np.sum(distances[axon_mask_curr]))

    soma_idx_arr = np.where(type_arr == SWC_TYPE_SOMA)[0]
    if len(soma_idx_arr) > 0:
        soma_idx = soma_idx_arr[0]
        soma_region_id = int(point_regions[soma_idx])
        soma_name_matches = dictionary.loc[dictionary['id'] == soma_region_id, 'safe_name'].tolist()
        soma_name = soma_name_matches[0] if soma_name_matches else "Unknown region"
        soma_coords = (float(x[soma_idx]), float(y[soma_idx]), float(z[soma_idx]))
    else:
        soma_idx = None
        soma_region_id = -1
        soma_name = "No soma found"
        soma_coords = (0.0, 0.0, 0.0)

    target_results = []
    for region_id in target_region_ids:
        name_matches = dictionary.loc[dictionary['id'] == region_id, 'safe_name'].tolist()
        region_name = name_matches[0] if name_matches else f"Unknown (ID: {region_id})"

        ep_count = int(np.sum(ep_regions == region_id))
        br_count = int(np.sum(branch_regions == region_id))
        proj_count = ep_count + br_count

        region_axon_mask = axon_mask_curr & (point_regions[curr_idx] == region_id)
        axon_len = float(np.sum(distances[region_axon_mask]))

        target_results.append(RegionResult(
            region_id=region_id, region_name=region_name,
            projects_here=(proj_count > 0),
            endpoint_count=ep_count, branch_point_count=br_count,
            projection_point_count=proj_count, axon_length_um=axon_len
        ))

    unique_proj_regions = np.unique(proj_regions[proj_regions > 0])
    other_region_ids = unique_proj_regions[
        ~np.isin(unique_proj_regions, target_region_ids) & (unique_proj_regions != soma_region_id)]

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
            region_id=int(region_id), region_name=region_name,
            projects_here=True, endpoint_count=ep_count,
            branch_point_count=br_count, projection_point_count=proj_count,
            axon_length_um=axon_len
        ))

    coords = {
        'x': x, 'y': y, 'z': z, 'type_arr': type_arr, 'is_axon': is_axon,
        'point_regions': point_regions, 'proj_idx': proj_idx, 'ep_idx': ep_idx,
        'branch_idx': branch_idx, 'curr_idx': curr_idx, 'parent_row_indices': parent_row_indices,
        'soma_idx': soma_idx, 'valid_connections': valid_connections,
    }

    return CellAnalysisResult(
        soma_region_id=soma_region_id, soma_region_name=soma_name,
        soma_coords=soma_coords, target_results=target_results,
        other_projection_regions=other_projection_regions,
        total_axon_length_um=total_axon_length, coords=coords
    )


# =============================================================================
# SZŰRÉS KOMPLEX LOGIKÁVAL
# =============================================================================

def apply_filter(
        result: CellAnalysisResult,
        criteria_per_region: dict[int, FilterCriteria]
) -> CellAnalysisResult:
    """
    Ellenőrzi, hogy egy sejt analízis eredménye megfelel-e az AND / OR / NOT logikának.
    """
    if not any(c.is_active() for c in criteria_per_region.values()):
        result.passes_filter = None
        return result

    and_passed = True
    or_passed = False
    has_or = False

    for tr in result.target_results:
        crit = criteria_per_region.get(tr.region_id)
        if not crit or not crit.is_active():
            continue

        meets = crit.meets_thresholds(tr)

        if crit.operator == 'AND':
            if not meets:
                and_passed = False
        elif crit.operator == 'NOT':
            if meets:
                and_passed = False  # Elbukik, mert NEM vetíthet ide
        elif crit.operator == 'OR':
            has_or = True
            if meets:
                or_passed = True

    # Végső kiértékelés
    if has_or and not or_passed:
        result.passes_filter = False
    elif not and_passed:
        result.passes_filter = False
    else:
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
    rows = []
    for cell_name, result in results:
        row = {
            'cell': cell_name,
            'soma_region': result.soma_region_name,
            'total_axon_length_um': round(result.total_axon_length_um, 1),
            'passes_filter': result.passes_filter,
        }
        for tr in result.target_results:
            safe_col = tr.region_name.replace(' ', '_').lower()[:30]
            row[f'{safe_col}_projects'] = tr.projects_here
            row[f'{safe_col}_endpoints'] = tr.endpoint_count
            row[f'{safe_col}_branches'] = tr.branch_point_count
            row[f'{safe_col}_axon_um'] = round(tr.axon_length_um, 1)
        rows.append(row)
    return pd.DataFrame(rows)