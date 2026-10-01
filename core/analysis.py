# =============================================================================
# ANALÍZIS MODUL - A tudományos számítási logika.
# =============================================================================

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from config import (
    VOXEL_SIZE, SWC_TYPE_SOMA, SWC_TYPE_AXON, SWC_TYPE_AXON_UNDEFINED,
    DEFAULT_FILTER, MIDLINE_AXIS, DEFAULT_LATERALITY,
    CONTRA_CROSSING_MIN_AXON_UM,
)

MIN_ENDPOINTS_FOR_PROJECTION = DEFAULT_FILTER['min_endpoints']
MIN_BRANCH_POINTS_FOR_PROJECTION = DEFAULT_FILTER['min_branch_points']

def _is_true_projection(endpoint_count: int, branch_point_count: int) -> bool:
    return (endpoint_count >= MIN_ENDPOINTS_FOR_PROJECTION and
            branch_point_count >= MIN_BRANCH_POINTS_FOR_PROJECTION)

@dataclass
class RegionResult:
    region_id: int
    region_name: str
    projects_here: bool
    endpoint_count: int
    branch_point_count: int
    projection_point_count: int
    axon_length_um: float
    endpoint_fraction: float = 0.0
    endpoint_count_ipsi: int = 0
    endpoint_count_contra: int = 0
    branch_point_count_ipsi: int = 0
    branch_point_count_contra: int = 0

@dataclass
class FilterCriteria:
    min_endpoints: int = MIN_ENDPOINTS_FOR_PROJECTION
    min_branch_points: int = MIN_BRANCH_POINTS_FOR_PROJECTION
    min_axon_length_um: float = 0
    min_endpoint_fraction: float = 0.0
    operator: str = 'AND'

    def is_projection(self, endpoint_count: int, branch_point_count: int,
                      axon_length_um: float = 0.0, endpoint_fraction: float = 0.0) -> bool:
        return (endpoint_count >= self.min_endpoints and
                branch_point_count >= self.min_branch_points and
                axon_length_um >= self.min_axon_length_um and
                endpoint_fraction >= self.min_endpoint_fraction)

    def is_active(self) -> bool:
        return self.operator != 'NONE'

    def meets_thresholds(self, region_result: 'RegionResult') -> bool:
        return region_result.projects_here

    def describe(self) -> str:
        parts = []
        if self.min_endpoints > 0: parts.append(f"≥{self.min_endpoints} endpoint")
        if self.min_branch_points > 0: parts.append(f"≥{self.min_branch_points} branch point")
        if self.min_axon_length_um > 0: parts.append(f"≥{self.min_axon_length_um:g} µm axon")
        if self.min_endpoint_fraction > 0: parts.append(f"≥{self.min_endpoint_fraction * 100:g}% endpoint share")
        return " AND ".join(parts) if parts else "any axon presence"

    def slug(self) -> str:
        return f"ep{self.min_endpoints}_br{self.min_branch_points}"

@dataclass
class CellAnalysisResult:
    soma_region_id: int
    soma_region_name: str
    soma_coords: tuple[float, float, float]
    target_results: list[RegionResult]
    other_projection_regions: list[RegionResult]
    total_axon_length_um: float
    passes_filter: bool | None = None
    coords: dict = field(default_factory=dict)
    soma_border_fraction: float = 0.0
    total_endpoint_count: int = 0
    annotated_endpoint_count: int = 0
    laterality: str = 'both'
    soma_side: int = 0
    endpoints_ipsi_total: int = 0
    endpoints_contra_total: int = 0
    axon_length_ipsi_um: float = 0.0
    axon_length_contra_um: float = 0.0
    axon_length_midline_um: float = 0.0

    @property
    def soma_is_border(self) -> bool: return self.soma_border_fraction > 0.0

    @property
    def has_hemisphere(self) -> bool: return self.soma_side != 0

    @property
    def projects_contralaterally(self) -> bool: return self.endpoints_contra_total > 0

    @property
    def crosses_midline(self) -> bool: return self.axon_length_contra_um >= CONTRA_CROSSING_MIN_AXON_UM

    @property
    def contra_endpoint_fraction(self) -> float:
        return self.endpoints_contra_total / self.total_endpoint_count if self.total_endpoint_count > 0 else 0.0

    @property
    def contra_axon_fraction(self) -> float:
        return self.axon_length_contra_um / self.total_axon_length_um if self.total_axon_length_um > 0 else 0.0

    @property
    def laterality_class(self) -> str:
        if not self.has_hemisphere: return 'unknown'
        if self.projects_contralaterally: return 'contra'
        return 'crosses_only' if self.crosses_midline else 'ipsi_only'

def run_analysis(
        swc_df: pd.DataFrame,
        atlas_matrix: np.ndarray,
        dictionary: pd.DataFrame,
        target_region_ids: list[int],
        region_descendants: dict[int, set[int]] | None = None,
        region_names: dict[int, str] | None = None,
        criteria_per_region: dict[int, 'FilterCriteria'] | None = None,
        laterality: str = DEFAULT_LATERALITY
) -> CellAnalysisResult:
    max_x, max_y, max_z = atlas_matrix.shape
    id_arr = np.round(swc_df['id'].values).astype(int)
    type_arr = np.round(swc_df['type'].values).astype(int)
    x, y, z = swc_df['x'].values, swc_df['y'].values, swc_df['z'].values
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

    ep_regions, branch_regions, proj_regions = point_regions[ep_idx], point_regions[branch_idx], point_regions[proj_idx]

    curr_idx = np.where(valid_connections)[0]
    p_idx = parent_row_indices[curr_idx]
    distances = np.sqrt((x[curr_idx] - x[p_idx]) ** 2 + (y[curr_idx] - y[p_idx]) ** 2 + (z[curr_idx] - z[p_idx]) ** 2)

    axon_mask_curr = is_axon[curr_idx]
    total_axon_length = float(np.sum(distances[axon_mask_curr]))

    axon_seg = np.where(axon_mask_curr)[0]
    if len(axon_seg) > 0:
        seg_child, seg_parent, seg_len = curr_idx[axon_seg], p_idx[axon_seg], distances[axon_seg]
        n_samp = np.clip(np.ceil(seg_len / (VOXEL_SIZE * 0.5)).astype(int), 1, 256)
        seg_id = np.repeat(np.arange(len(seg_len)), n_samp)
        starts = np.concatenate(([0], np.cumsum(n_samp)[:-1]))
        k = np.arange(int(n_samp.sum())) - starts[seg_id]
        t = (k + 0.5) / n_samp[seg_id]

        px, py, pz = x[seg_parent], y[seg_parent], z[seg_parent]
        dx, dy, dz = x[seg_child] - px, y[seg_child] - py, z[seg_child] - pz
        s_x, s_y, s_z = px[seg_id] + t * dx[seg_id], py[seg_id] + t * dy[seg_id], pz[seg_id] + t * dz[seg_id]

        s_vx = np.clip(np.round(s_x / VOXEL_SIZE).astype(int), 0, max_x - 1)
        s_vy = np.clip(np.round(s_y / VOXEL_SIZE).astype(int), 0, max_y - 1)
        s_vz = np.clip(np.round(s_z / VOXEL_SIZE).astype(int), 0, max_z - 1)
        samp_region = atlas_matrix[s_vx, s_vy, s_vz].astype(int)
        samp_len = seg_len[seg_id] / n_samp[seg_id]

        ok = samp_region >= 0
        length_by_region = np.bincount(samp_region[ok], weights=samp_len[ok])

        samp_ml = (s_vx, s_vy, s_vz)[MIDLINE_AXIS]
        samp_side = np.sign(samp_ml.astype(float) - (atlas_matrix.shape[MIDLINE_AXIS] / 2.0))
        axon_length_by_side = {
            -1: float(samp_len[samp_side < 0].sum()),
            0: float(samp_len[samp_side == 0].sum()),
            1: float(samp_len[samp_side > 0].sum()),
        }
    else:
        length_by_region = np.zeros(1, dtype=float)
        axon_length_by_side = {-1: 0.0, 0: 0.0, 1: 0.0}

    def _axon_length_in(match_ids: np.ndarray) -> float:
        valid = match_ids[(match_ids >= 0) & (match_ids < len(length_by_region))]
        return float(length_by_region[valid].sum()) if len(valid) else 0.0

    soma_idx_arr = np.where(type_arr == SWC_TYPE_SOMA)[0]
    soma_border_fraction = 0.0
    if len(soma_idx_arr) > 0:
        soma_idx = soma_idx_arr[0]
        soma_region_id = int(point_regions[soma_idx])
        soma_name_matches = dictionary.loc[dictionary['id'] == soma_region_id, 'safe_name'].tolist()
        soma_name = soma_name_matches[0] if soma_name_matches else "Unknown region"
        soma_coords = (float(x[soma_idx]), float(y[soma_idx]), float(z[soma_idx]))

        sx0, sy0, sz0 = int(vox_x[soma_idx]), int(vox_y[soma_idx]), int(vox_z[soma_idx])
        nb = atlas_matrix[max(0, sx0 - 1):sx0 + 2, max(0, sy0 - 1):sy0 + 2, max(0, sz0 - 1):sz0 + 2].ravel()
        if len(nb) > 0:
            soma_border_fraction = float(np.mean(nb != soma_region_id))
    else:
        soma_idx, soma_region_id, soma_name, soma_coords = None, -1, "No soma found", (0.0, 0.0, 0.0)

    ml_vox = (vox_x, vox_y, vox_z)[MIDLINE_AXIS]
    midline = atlas_matrix.shape[MIDLINE_AXIS] / 2.0
    point_side = np.sign(ml_vox.astype(float) - midline).astype(int)
    soma_side = int(point_side[soma_idx]) if soma_idx is not None else 0

    ep_side, branch_side = point_side[ep_idx], point_side[branch_idx]

    if soma_side != 0:
        endpoints_ipsi_total = int((ep_side == soma_side).sum())
        endpoints_contra_total = int((ep_side == -soma_side).sum())
        axon_length_ipsi_um = axon_length_by_side[soma_side]
        axon_length_contra_um = axon_length_by_side[-soma_side]
        axon_length_midline_um = axon_length_by_side[0]
    else:
        endpoints_ipsi_total, endpoints_contra_total = 0, 0
        axon_length_ipsi_um, axon_length_contra_um, axon_length_midline_um = 0.0, 0.0, 0.0

    def _side_mask(sides: np.ndarray) -> np.ndarray:
        if laterality == 'both' or soma_side == 0: return np.ones(len(sides), dtype=bool)
        if laterality == 'ipsi': return sides == soma_side
        return (sides == -soma_side) & (sides != 0)

    ep_keep, branch_keep = _side_mask(ep_side), _side_mask(branch_side)
    total_endpoint_count = int(len(ep_idx))
    region_descendants, region_names, criteria_per_region = region_descendants or {}, region_names or {}, criteria_per_region or {}
    default_criteria = FilterCriteria()

    def _match_ids(region_id: int) -> np.ndarray:
        ids = region_descendants.get(int(region_id))
        return np.fromiter((int(v) for v in ids), dtype=int) if ids else np.array([int(region_id)], dtype=int)

    def _build_region_result(region_id: int) -> RegionResult:
        region_name = region_names.get(region_id) or (dictionary.loc[dictionary['id'] == region_id, 'safe_name'].tolist()[0] if not dictionary.loc[dictionary['id'] == region_id, 'safe_name'].empty else f"Unknown (ID: {region_id})")
        match = _match_ids(region_id)
        ep_in, br_in = np.isin(ep_regions, match), np.isin(branch_regions, match)

        ep_count = int((ep_in & ep_keep).sum())
        br_count = int((br_in & branch_keep).sum())
        proj_count = ep_count + br_count

        ep_ipsi = int((ep_in & (ep_side == soma_side)).sum()) if soma_side != 0 else 0
        ep_contra = int((ep_in & (ep_side == -soma_side)).sum()) if soma_side != 0 else 0
        br_ipsi = int((br_in & (branch_side == soma_side)).sum()) if soma_side != 0 else 0
        br_contra = int((br_in & (branch_side == -soma_side)).sum()) if soma_side != 0 else 0

        axon_len = _axon_length_in(match)
        fraction = (ep_count / total_endpoint_count) if total_endpoint_count > 0 else 0.0

        return RegionResult(
            region_id=int(region_id), region_name=region_name,
            projects_here=criteria_per_region.get(int(region_id), default_criteria).is_projection(ep_count, br_count, axon_len, fraction),
            endpoint_count=ep_count, branch_point_count=br_count, projection_point_count=proj_count, axon_length_um=axon_len,
            endpoint_fraction=fraction, endpoint_count_ipsi=ep_ipsi, endpoint_count_contra=ep_contra,
            branch_point_count_ipsi=br_ipsi, branch_point_count_contra=br_contra,
        )

    target_results = [_build_region_result(region_id) for region_id in target_region_ids]

    covered_ids = set(int(r) for r in target_region_ids)
    for rid in target_region_ids: covered_ids.update(int(v) for v in _match_ids(rid))

    unique_proj_regions = np.unique(proj_regions[proj_regions > 0])
    other_region_ids = [int(rid) for rid in unique_proj_regions if int(rid) not in covered_ids and int(rid) != soma_region_id]
    other_projection_regions = [rr for region_id in other_region_ids if (rr := _build_region_result(region_id)).projects_here]

    coords = {
        'x': x, 'y': y, 'z': z, 'type_arr': type_arr, 'is_axon': is_axon,
        'point_regions': point_regions, 'proj_idx': proj_idx, 'ep_idx': ep_idx,
        'branch_idx': branch_idx, 'curr_idx': curr_idx, 'parent_row_indices': parent_row_indices,
        'soma_idx': soma_idx, 'valid_connections': valid_connections,
    }

    return CellAnalysisResult(
        soma_region_id=soma_region_id, soma_region_name=soma_name, soma_coords=soma_coords,
        target_results=target_results, other_projection_regions=other_projection_regions,
        total_axon_length_um=total_axon_length, coords=coords, soma_border_fraction=soma_border_fraction,
        total_endpoint_count=total_endpoint_count, laterality=laterality,
        annotated_endpoint_count=int(np.sum(ep_regions > 0)), soma_side=soma_side,
        endpoints_ipsi_total=endpoints_ipsi_total, endpoints_contra_total=endpoints_contra_total,
        axon_length_ipsi_um=axon_length_ipsi_um, axon_length_contra_um=axon_length_contra_um,
        axon_length_midline_um=axon_length_midline_um,
    )

def apply_filter(result: CellAnalysisResult, criteria_per_region: dict[int, FilterCriteria]) -> CellAnalysisResult:
    active = {rid: c for rid, c in criteria_per_region.items() if c.is_active()}
    if not active:
        result.passes_filter = None
        return result

    results_by_region = {tr.region_id: tr for tr in result.target_results}
    required_ok, excluded_ok, or_exists, or_ok = True, True, False, False

    for region_id, crit in active.items():
        if (tr := results_by_region.get(region_id)) is None: continue
        meets = crit.meets_thresholds(tr)

        if crit.operator == 'OR':
            or_exists = True; or_ok = or_ok or meets
        elif crit.operator == 'NOT':
            excluded_ok = excluded_ok and not meets
        else:
            required_ok = required_ok and meets

    result.passes_filter = required_ok and excluded_ok and (or_ok or not or_exists)
    return result

def results_to_dataframe(
        results: list[tuple[str, CellAnalysisResult]],
        target_region_ids: list[int],
        dictionary: pd.DataFrame,
        criteria_per_region: dict[int, 'FilterCriteria'] | None = None
) -> pd.DataFrame:
    criteria_per_region = criteria_per_region or {}
    rows = []
    for cell_name, result in results:
        row = {
            'cell': cell_name, 'soma_region': result.soma_region_name,
            'total_axon_length_um': round(result.total_axon_length_um, 1),
            'passes_filter': result.passes_filter, 'run_hemisphere_setting': result.laterality,
            'soma_on_region_border': result.soma_is_border, 'soma_border_fraction': round(result.soma_border_fraction, 2),
            'endpoints_total': result.total_endpoint_count, 'endpoints_in_annotated_regions': result.annotated_endpoint_count,
            'laterality_class': result.laterality_class, 'projects_contralateral': result.projects_contralaterally,
            'crosses_midline': result.crosses_midline, 'endpoints_ipsi_total': result.endpoints_ipsi_total,
            'endpoints_contra_total': result.endpoints_contra_total, 'contra_endpoint_pct': round(result.contra_endpoint_fraction * 100, 2),
            'axon_um_ipsi': round(result.axon_length_ipsi_um, 1), 'axon_um_contra': round(result.axon_length_contra_um, 1),
            'axon_um_midline': round(result.axon_length_midline_um, 1), 'contra_axon_pct': round(result.contra_axon_fraction * 100, 2),
        }
        for tr in result.target_results:
            safe_col = f"{tr.region_name.replace(' ', '_').lower()[:30]}_{tr.region_id}"
            row.update({
                f'{safe_col}_projects': tr.projects_here, f'{safe_col}_endpoints': tr.endpoint_count,
                f'{safe_col}_branches': tr.branch_point_count, f'{safe_col}_axon_um': round(tr.axon_length_um, 1),
                f'{safe_col}_endpoint_pct': round(tr.endpoint_fraction * 100, 2),
                f'{safe_col}_endpoints_ipsi': tr.endpoint_count_ipsi, f'{safe_col}_endpoints_contra': tr.endpoint_count_contra
            })
            if (crit := criteria_per_region.get(tr.region_id)) is not None: row[f'{safe_col}_criterion'] = crit.describe()
        rows.append(row)
    return pd.DataFrame(rows)

def _cell_serial(cell_name: str) -> str: return cell_name[:-4] if cell_name.lower().endswith('.swc') else cell_name
def _region_of(result: CellAnalysisResult, region_id: int) -> RegionResult | None:
    return next((tr for tr in result.target_results if tr.region_id == region_id), None)
def _projects_to(result: CellAnalysisResult, region_id: int) -> bool:
    return bool((tr := _region_of(result, region_id)) and tr.projects_here)
def _only_projects_to(result: CellAnalysisResult, region_id: int, all_region_ids: list[int]) -> bool:
    if not _projects_to(result, region_id): return False
    return not any(_projects_to(result, other) for other in all_region_ids if int(other) != int(region_id))

def build_soma_distribution_summary(results: list[tuple[str, CellAnalysisResult]], filter_was_active: bool) -> pd.DataFrame:
    """Kiszámolja a soma régiónkénti eloszlást és a vetítő sejtek arányát (kiiktatva a UI-ból)."""
    soma_counts = {}
    for cell_name, r in results:
        soma = r.soma_region_name
        if soma not in soma_counts:
            soma_counts[soma] = {'total': 0, 'projecting': 0, 'ids': []}
        soma_counts[soma]['total'] += 1
        is_projecting = bool(r.passes_filter) if filter_was_active else any(tr.projects_here for tr in r.target_results)
        if is_projecting:
            soma_counts[soma]['projecting'] += 1
            soma_counts[soma]['ids'].append(_cell_serial(cell_name))

    df = pd.DataFrame([{
        "Soma Region": soma, "Total Cells": data['total'], "Valid Projections": data['projecting'],
        "Valid Projections %": round(100 * data['projecting'] / data['total'], 1) if data['total'] > 0 else 0.0,
        "Projecting Cell IDs": ", ".join(data['ids']),
    } for soma, data in soma_counts.items()])
    return df.sort_values(by="Total Cells", ascending=False) if not df.empty else df

LATERALITY_CLASS_LABELS = {
    'ipsi_only': 'Ipsilateral only (no midline crossing)', 'crosses_only': 'Crosses midline, but no endpoints there',
    'contra': 'Projects contralaterally (endpoints on the far side)', 'unknown': 'Undetermined (no soma, or soma on the midline)',
}
LATERALITY_CLASS_ORDER = ['ipsi_only', 'crosses_only', 'contra', 'unknown']

def build_laterality_summary(results: list[tuple[str, CellAnalysisResult]]) -> dict:
    counts = {k: [] for k in LATERALITY_CLASS_ORDER}; by_soma, per_cell_rows = {}, []
    for cell_name, r in results:
        cls, serial, soma = r.laterality_class, _cell_serial(cell_name), r.soma_region_name
        counts[cls].append(serial)
        by_soma.setdefault(soma, {k: [] for k in LATERALITY_CLASS_ORDER})[cls].append(serial)
        per_cell_rows.append({
            'Cell': serial, 'Soma region': soma, 'Class': LATERALITY_CLASS_LABELS[cls],
            'Endpoints ipsi': r.endpoints_ipsi_total, 'Endpoints contra': r.endpoints_contra_total,
            'Contra endpoint %': round(r.contra_endpoint_fraction * 100, 2),
            'Axon ipsi (um)': round(r.axon_length_ipsi_um, 1), 'Axon contra (um)': round(r.axon_length_contra_um, 1),
            'Axon midline (um)': round(r.axon_length_midline_um, 1), 'Contra axon %': round(r.contra_axon_fraction * 100, 2),
        })
    n_total, n_decided = len(results), len(results) - len(counts['unknown'])
    overall = pd.DataFrame([{
        'Category': LATERALITY_CLASS_LABELS[cls], 'Cells': len(counts[cls]),
        '% of decided': (round(100 * len(counts[cls]) / n_decided, 1) if n_decided > 0 and cls != 'unknown' else None),
        'Cell IDs': ", ".join(counts[cls]),
    } for cls in LATERALITY_CLASS_ORDER])
    soma_rows = []
    for soma, per_class in by_soma.items():
        decided = sum(len(v) for k, v in per_class.items() if k != 'unknown')
        not_contra = len(per_class['ipsi_only']) + len(per_class['crosses_only'])
        soma_rows.append({
            'Soma region': soma, 'Total cells': sum(len(v) for v in per_class.values()), 'Decided': decided,
            'Ipsilateral only': len(per_class['ipsi_only']), 'Crosses, no endpoints': len(per_class['crosses_only']),
            'Contralateral': len(per_class['contra']), 'Undetermined': len(per_class['unknown']),
            'No contralateral projection': not_contra,
            'No contralateral %': (round(100 * not_contra / decided, 1) if decided > 0 else None),
            'Ipsilateral-only cell IDs': ", ".join(per_class['ipsi_only']),
        })
    by_soma_df = pd.DataFrame(soma_rows).sort_values(by='Total cells', ascending=False) if soma_rows else pd.DataFrame()
    return {'overall': overall, 'by_soma': by_soma_df, 'per_cell': pd.DataFrame(per_cell_rows), 'n_total': n_total, 'n_decided': n_decided, 'counts': {cls: len(ids) for cls, ids in counts.items()}}

def category_slugs(labels: list[str]) -> dict[str, str]:
    slugs, used = {}, set()
    for lab in labels:
        is_only = lab.lower().endswith(' only')
        base = lab[:-5] if is_only else lab
        stem = ''.join(ch if ch.isalnum() else '_' for ch in base.lower())[:24].strip('_')
        stem = f"{stem}_only" if is_only else stem
        candidate, n = stem, 2
        while candidate in used: candidate, n = f"{stem}_{n}", n + 1
        used.add(candidate)
        slugs[lab] = candidate
    return slugs

def build_cortical_summary(
        results: list[tuple[str, CellAnalysisResult]], base_region_id: int | None, numerator_region_ids: list[int],
        region_label_fn, criteria_per_region: dict[int, 'FilterCriteria'] | None = None, laterality: str | None = None,
) -> dict:
    from collections import defaultdict
    criteria_per_region = criteria_per_region or {}
    groups: dict[str, list] = defaultdict(list); skipped_no_soma = 0
    for name, r in results:
        if r.soma_region_id is None or r.soma_region_id <= 0:
            skipped_no_soma += 1; continue
        groups[r.soma_region_name].append((name, r))

    num_labels = [region_label_fn(rid) for rid in numerator_region_ids]
    base_col = f"PT Cells ({region_label_fn(base_region_id) if base_region_id is not None else 'All L5'}=100%)"
    def is_base(r: CellAnalysisResult) -> bool: return True if base_region_id is None else _projects_to(r, base_region_id)
    def meets_all(r: CellAnalysisResult) -> bool: return bool(numerator_region_ids) and all(_projects_to(r, rid) for rid in numerator_region_ids)

    benne_rows, nelkul_rows, axon_rows, cat_all_rows = [], [], [], []
    cat_rows, cat_only_rows = {lab: [] for lab in num_labels}, {lab: [] for lab in num_labels}

    for soma, cells in sorted(groups.items()):
        total, base_cells = len(cells), [(n, r) for (n, r) in cells if is_base(r)]
        nbase = len(base_cells)
        row_b, row_n, row_a = {"Soma Region": soma, base_col: nbase}, {"Soma Region": soma, "Total L5 Cells": total}, {"Soma Region": soma, "PT Cells": nbase}

        for rid, lab in zip(numerator_region_ids, num_labels):
            cb = sum(1 for (_, r) in base_cells if _projects_to(r, rid))
            row_b[f"{lab} n"], row_b[f"{lab} %"] = cb, round(100 * cb / nbase, 1) if nbase else 0.0

            if len(numerator_region_ids) > 1:
                cb_only = sum(1 for (_, r) in base_cells if _only_projects_to(r, rid, numerator_region_ids))
                row_b[f"{lab} only n"], row_b[f"{lab} only %"] = cb_only, round(100 * cb_only / nbase, 1) if nbase else 0.0

            cn = sum(1 for (_, r) in cells if _projects_to(r, rid))
            row_n[f"{lab} n"], row_n[f"{lab} %"] = cn, round(100 * cn / total, 1) if total else 0.0
            if len(numerator_region_ids) > 1:
                cn_only = sum(1 for (_, r) in cells if _only_projects_to(r, rid, numerator_region_ids))
                row_n[f"{lab} only n"], row_n[f"{lab} only %"] = cn_only, round(100 * cn_only / total, 1) if total else 0.0

            lens = [_region_of(r, rid).axon_length_um for (_, r) in base_cells if _projects_to(r, rid)]
            row_a[f"{lab} mean axon µm"] = round(sum(lens) / len(lens), 1) if lens else 0.0

            ids = sorted(_cell_serial(n) for (n, r) in base_cells if _projects_to(r, rid))
            cat_rows[lab].append({"Soma Region": soma, base_col: nbase, f"{lab} Projects": len(ids), f"{lab} % of PT": round(100 * len(ids) / nbase, 1) if nbase else 0.0, "Projecting Cell IDs": ", ".join(ids)})

            if len(numerator_region_ids) > 1:
                ids_only = sorted(_cell_serial(n) for (n, r) in base_cells if _only_projects_to(r, rid, numerator_region_ids))
                cat_only_rows[lab].append({"Soma Region": soma, base_col: nbase, f"{lab} only Projects": len(ids_only), f"{lab} only % of PT": round(100 * len(ids_only) / nbase, 1) if nbase else 0.0, "Projecting Cell IDs": ", ".join(ids_only)})

        cb_all, cn_all = sum(1 for (_, r) in base_cells if meets_all(r)), sum(1 for (_, r) in cells if meets_all(r))
        row_b["All targets n"], row_b["All targets %"] = cb_all, round(100 * cb_all / nbase, 1) if nbase else 0.0
        row_n["All targets n"], row_n["All targets %"] = cn_all, round(100 * cn_all / total, 1) if total else 0.0

        ids_all = sorted(_cell_serial(n) for (n, r) in base_cells if meets_all(r))
        cat_all_rows.append({"Soma Region": soma, base_col: nbase, "All targets Projects": len(ids_all), "All targets % of PT": round(100 * len(ids_all) / nbase, 1) if nbase else 0.0, "Projecting Cell IDs": ", ".join(ids_all)})

        benne_rows.append(row_b); nelkul_rows.append(row_n); axon_rows.append(row_a)

    benne, nelkul, axon = pd.DataFrame(benne_rows).sort_values(base_col, ascending=False), pd.DataFrame(nelkul_rows).sort_values("Total L5 Cells", ascending=False), pd.DataFrame(axon_rows).sort_values("PT Cells", ascending=False)
    categories = {lab: pd.DataFrame(cat_rows[lab]).sort_values(f"{lab} Projects", ascending=False) for lab in num_labels}
    if len(numerator_region_ids) > 1:
        categories.update({f"{lab} only": pd.DataFrame(cat_only_rows[lab]).sort_values(f"{lab} only Projects", ascending=False) for lab in num_labels})
        categories["All targets"] = pd.DataFrame(cat_all_rows).sort_values("All targets Projects", ascending=False)

    involved = ([base_region_id] if base_region_id is not None else []) + list(numerator_region_ids)
    used = [criteria_per_region.get(rid, FilterCriteria()) for rid in involved]
    uniform = all((c.min_endpoints, c.min_branch_points, c.min_axon_length_um, c.min_endpoint_fraction) == (used[0].min_endpoints, used[0].min_branch_points, used[0].min_axon_length_um, used[0].min_endpoint_fraction) for c in used) if used else True

    lat = laterality or (results[0][1].laterality if results else 'both')
    lat_note, lat_slug = {'ipsi': ' · ipsilateral only', 'contra': ' · contralateral only'}.get(lat, ''), {'ipsi': '_ipsi', 'contra': '_contra'}.get(lat, '')

    if uniform and used: criteria_note, slug = used[0].describe() + lat_note, used[0].slug() + lat_slug
    else: criteria_note, slug = " · ".join(f"{region_label_fn(rid)}: {criteria_per_region.get(rid, FilterCriteria()).describe()}" for rid in involved) + lat_note, "mixed" + lat_slug

    return {"benne": benne, "nelkul": nelkul, "axon": axon, "categories": categories, "criteria_note": criteria_note, "slug": slug, "skipped_no_soma": skipped_no_soma}