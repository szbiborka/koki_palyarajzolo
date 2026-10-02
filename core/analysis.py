# =============================================================================
# ANALÍZIS MODUL - A tudományos számítási logika (Streamlit-független).
# =============================================================================

from collections import defaultdict
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from config import (
    VOXEL_SIZE, SWC_TYPE_SOMA, SWC_TYPE_AXON, SWC_TYPE_AXON_UNDEFINED,
    DEFAULT_FILTER, MIDLINE_AXIS, DEFAULT_LATERALITY,
    CONTRA_CROSSING_MIN_AXON_UM, MIDLINE_BAND_UM, VOXEL_LOOKUP,
)

# Egy axonszakasz mintavételezési sűrűsége: fél voxelenként egy minta, legfeljebb
# 256 minta szakaszonként.
_SAMPLES_PER_VOXEL = 2
_MAX_SAMPLES_PER_SEGMENT = 256

_SIDE_NOTES = {'ipsi': 'ipsilateral only', 'contra': 'contralateral only'}


@dataclass
class RegionResult:
    region_id: int
    region_name: str
    projects_here: bool
    endpoint_count: int
    branch_point_count: int
    axon_length_um: float
    endpoint_fraction: float = 0.0
    endpoint_count_ipsi: int = 0
    endpoint_count_contra: int = 0
    branch_point_count_ipsi: int = 0
    branch_point_count_contra: int = 0
    side: str = 'both'  # melyik féltekén számoltuk a fenti értékeket


@dataclass
class FilterCriteria:
    """
    EGY régió vetítési kritériuma - és egyben a szűrési feltétel.

    Ez az egyetlen hely, ahol eldől, mi számít vetítésnek az adott régióban.
    Ugyanez hajtja a "..._projects" pipát, a szűrést (passes_filter) és az
    összesítő táblákat, tehát nem lehet közöttük ellentmondás.

    Alapértelmezés: >=1 végpont ÉS >=1 elágazás (valódi terminális arborizáció);
    az áthaladó axon így nem számít vetítésnek. A min_endpoint_fraction
    méret-független küszöb (a régió végpontjai / a sejt végpontjai): NOT
    operátorral ez a L6-szűrő (pl. "thalamus >= 2.5%" => kizárva).

    Az operator NEM a vetítés definíciója, hanem azt mondja meg, hogyan
    kombináljuk a régiókat egymással:
        'AND'  = ide vetítenie kell
        'NOT'  = ide nem vetíthet
        'OR'   = opcionális (legalább egy OR-régió teljesüljön)
        'NONE' = csak megfigyelés: a számai megjelennek, de nem szűr

    A side régiónként mondja meg, melyik félteke számít ('both' / 'ipsi' /
    'contra'); None = a futás alapértelmezése. Így pl. a GPe ipszilaterálisan,
    a kéreg kontralaterálisan vizsgálható ugyanabban a futásban.
    """
    min_endpoints: int = DEFAULT_FILTER['min_endpoints']
    min_branch_points: int = DEFAULT_FILTER['min_branch_points']
    min_axon_length_um: float = DEFAULT_FILTER['min_axon_length_um']
    min_endpoint_fraction: float = DEFAULT_FILTER['min_endpoint_fraction']
    operator: str = 'AND'
    side: str | None = None

    def is_projection(self, endpoint_count: int, branch_point_count: int,
                      axon_length_um: float, endpoint_fraction: float) -> bool:
        """
        Vetít-e ide a sejt e kritérium szerint (minden feltétel EGYSZERRE).
        Ha minden küszöb 0, a kritérium "bármilyen axon jelenléte": legalább
        valami axonnak lennie kell a régióban, különben minden sejt vetítene mindenhová.
        """
        if not self._has_threshold():
            return endpoint_count + branch_point_count > 0 or axon_length_um > 0
        return (endpoint_count >= self.min_endpoints and
                branch_point_count >= self.min_branch_points and
                axon_length_um >= self.min_axon_length_um and
                endpoint_fraction >= self.min_endpoint_fraction)

    def _has_threshold(self) -> bool:
        return (self.min_endpoints > 0 or self.min_branch_points > 0 or
                self.min_axon_length_um > 0 or self.min_endpoint_fraction > 0)

    def meets_thresholds(self, region_result: RegionResult) -> bool:
        """A régió eredménye teljesíti-e EZT a kritériumot."""
        return self.is_projection(region_result.endpoint_count, region_result.branch_point_count,
                                  region_result.axon_length_um, region_result.endpoint_fraction)

    def is_active(self) -> bool:
        """Részt vesz-e a régió a szűrésben. A 'NONE' (csak megfigyelés) nem."""
        return self.operator != 'NONE'

    def describe(self) -> str:
        """Rövid, emberi olvasásra szánt leírás (exportokhoz, feliratokhoz)."""
        parts = []
        if self.min_endpoints > 0:
            parts.append(f"≥{self.min_endpoints} endpoint")
        if self.min_branch_points > 0:
            parts.append(f"≥{self.min_branch_points} branch point")
        if self.min_axon_length_um > 0:
            parts.append(f"≥{self.min_axon_length_um:g} µm axon")
        if self.min_endpoint_fraction > 0:
            parts.append(f"≥{self.min_endpoint_fraction * 100:g}% endpoint share")
        text = " AND ".join(parts) if parts else "any axon presence"
        if self.side in _SIDE_NOTES:
            text += f" ({_SIDE_NOTES[self.side]})"
        return text

    def slug(self) -> str:
        """Fájlnévbe illeszthető azonosító, pl. 'ep1_br1' vagy 'ep1_br1_len100_frac2.5_ipsi'."""
        slug = f"ep{self.min_endpoints}_br{self.min_branch_points}"
        if self.min_axon_length_um > 0:
            slug += f"_len{self.min_axon_length_um:g}"
        if self.min_endpoint_fraction > 0:
            slug += f"_frac{self.min_endpoint_fraction * 100:g}"
        if self.side in _SIDE_NOTES:
            slug += f"_{self.side}"
        return slug

    def effective(self, default_side: str) -> 'FilterCriteria':
        """Ugyanez a kritérium, a side kitöltve a futás alapértelmezésével."""
        return replace(self, side=self.side or default_side)


@dataclass
class CellAnalysisResult:
    soma_region_id: int
    soma_region_name: str
    soma_coords: tuple[float, float, float]
    target_results: list[RegionResult]
    other_projection_regions: list[RegionResult]
    total_axon_length_um: float
    passes_filter: bool | None = None
    coords: dict = field(default_factory=dict)  # a 3D nézethez szükséges tömbök
    soma_border_fraction: float = 0.0
    total_endpoint_count: int = 0
    annotated_endpoint_count: int = 0
    laterality: str = DEFAULT_LATERALITY
    soma_side: int = 0  # -1 / +1 a középvonalhoz képest, 0 = nincs soma vagy a középvonali sávban ül
    endpoints_ipsi_total: int = 0
    endpoints_contra_total: int = 0
    axon_length_ipsi_um: float = 0.0
    axon_length_contra_um: float = 0.0
    axon_length_midline_um: float = 0.0
    # Sejt-információk a rekonstrukción kívülről (core/cell_info.py tölti ki)
    projection_class: str | None = None      # adatbázis: IT / PT / CT
    projection_subclass: str | None = None   # adatbázis: pl. 'PT-18'
    cre_line: str | None = None
    db_soma_region: str | None = None        # az adatbázis saját soma-régiója
    curation_label: str | None = None        # kézi ítélet (CURATION_LABELS kulcsa)
    summary_soma_region: str | None = None   # réteg-korrigált régió az összesítőkhöz

    @property
    def group_region(self) -> str:
        """Az összesítő táblák soma-csoportja: a réteg-korrigált régió, ha van, különben az atlaszé."""
        return self.summary_soma_region or self.soma_region_name

    @property
    def soma_is_border(self) -> bool:
        """A soma szomszédságában más régió is van -> a soma-régió besorolás bizonytalan."""
        return self.soma_border_fraction > 0.0

    @property
    def has_hemisphere(self) -> bool:
        return self.soma_side != 0

    @property
    def projects_contralaterally(self) -> bool:
        return self.endpoints_contra_total > 0

    @property
    def crosses_midline(self) -> bool:
        """Valóban átkelt-e (a középvonal menti kerekítési maradék nem számít)."""
        return self.axon_length_contra_um >= CONTRA_CROSSING_MIN_AXON_UM

    @property
    def contra_endpoint_fraction(self) -> float:
        if self.total_endpoint_count == 0:
            return 0.0
        return self.endpoints_contra_total / self.total_endpoint_count

    @property
    def contra_axon_fraction(self) -> float:
        if self.total_axon_length_um == 0:
            return 0.0
        return self.axon_length_contra_um / self.total_axon_length_um

    @property
    def laterality_class(self) -> str:
        """
        'ipsi_only' / 'crosses_only' / 'contra' / 'unknown'.
        Az átkelés végpont nélkül külön kategória: egy csonkolt rekonstrukció
        átmehet a túloldalra és ott abbamaradhat - ez se ipsi, se contra bizonyíték.
        Soma nélkül (vagy középvonali sománál) nem tippelünk: 'unknown'.
        """
        if not self.has_hemisphere:
            return 'unknown'
        if self.projects_contralaterally:
            return 'contra'
        return 'crosses_only' if self.crosses_midline else 'ipsi_only'


# =============================================================================
# EGY SEJT ELEMZÉSE
# =============================================================================

def to_voxel(x: np.ndarray, y: np.ndarray, z: np.ndarray, shape: tuple) -> tuple:
    """
    µm koordináták -> (vx, vy, vz) voxelindex-tömbök, az atlasz határaira vágva.
    Alapértelmezés az Allen CCF konvenciója: az i. voxel a [25*i, 25*(i+1)) µm
    tartomány (floor). A régi, kerekítéses viselkedés a VOXEL_LOOKUP = 'round'
    beállítással érhető el (lásd config.py).
    """
    to_index = np.round if VOXEL_LOOKUP == 'round' else np.floor
    return tuple(np.clip(to_index(c / VOXEL_SIZE).astype(int), 0, n - 1) for c, n in zip((x, y, z), shape))


def side_of(ml_um: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Középvonalhoz viszonyított oldal µm-ben: -1 / +1, a középvonal ±MIDLINE_BAND_UM
    sávjában 0 (eldönthetetlen). A középvonal a medio-laterális kiterjedés fele.
    """
    offset = np.asarray(ml_um, dtype=float) - shape[MIDLINE_AXIS] * VOXEL_SIZE / 2.0
    side = np.sign(offset).astype(int)
    side[np.abs(offset) <= MIDLINE_BAND_UM] = 0
    return side


def _sample_axon_segments(x, y, z, child_rows, parent_rows, atlas_matrix):
    """
    Az axonszakaszokat fél voxelenként mintavételezi, hogy a régióhatárt
    átlépő szakasz hossza ARÁNYOSAN oszoljon meg a régiók között.

    Returns:
        (samp_region, samp_side, samp_len): mintánkénti régió-ID, oldal és hossz.
        A samp_len összege pontosan a szakaszok teljes hossza.
    """
    px, py, pz = x[parent_rows], y[parent_rows], z[parent_rows]
    dx, dy, dz = x[child_rows] - px, y[child_rows] - py, z[child_rows] - pz
    seg_len = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    sample_spacing = VOXEL_SIZE / _SAMPLES_PER_VOXEL
    n_samp = np.clip(np.ceil(seg_len / sample_spacing).astype(int), 1, _MAX_SAMPLES_PER_SEGMENT)
    seg_id = np.repeat(np.arange(len(seg_len)), n_samp)
    starts = np.concatenate(([0], np.cumsum(n_samp)[:-1]))
    k = np.arange(int(n_samp.sum())) - starts[seg_id]
    t = (k + 0.5) / n_samp[seg_id]  # minden minta a saját részszakasza közepén

    sample_xyz = (px[seg_id] + t * dx[seg_id], py[seg_id] + t * dy[seg_id], pz[seg_id] + t * dz[seg_id])
    samp_region = atlas_matrix[to_voxel(*sample_xyz, atlas_matrix.shape)]
    samp_side = side_of(sample_xyz[MIDLINE_AXIS], atlas_matrix.shape)
    samp_len = seg_len[seg_id] / n_samp[seg_id]
    return samp_region, samp_side, samp_len


def run_analysis(
        swc_df: pd.DataFrame,
        atlas_matrix: np.ndarray,
        region_names: dict[int, str],
        target_region_ids: list[int],
        region_descendants: dict[int, set[int]] | None = None,
        criteria_per_region: dict[int, FilterCriteria] | None = None,
        laterality: str = DEFAULT_LATERALITY
) -> CellAnalysisResult:
    """
    Egy sejt teljes elemzése: célterületenkénti végpont / elágazás / axonhossz,
    az egyéb vetítési régiók és az egész sejtre vonatkozó oldaliság.

    Végpont = 0 gyerekű axon-csomópont, elágazás = 1-nél több gyerekű.
    Szülő régiók (pl. Brain stem) a region_descendants alapján a levél-ID-kra
    oldódnak fel. Oldaliság ('ipsi'/'contra') esetén a régiós végpont-,
    elágazás- és axonhossz-számok csak a kért oldalt számolják, és a végpont-arány
    nevezője is a kért oldal végpontjainak száma. Az egész sejtre vonatkozó
    ipsi/contra bontás ettől függetlenül mindig kiszámolódik.
    """
    region_descendants = region_descendants or {}
    criteria_per_region = criteria_per_region or {}
    shape = atlas_matrix.shape

    ids = np.round(swc_df['id'].to_numpy()).astype(int)
    types = np.round(swc_df['type'].to_numpy()).astype(int)
    pids = np.round(swc_df['pid'].to_numpy()).astype(int)
    x, y, z = (swc_df[c].to_numpy(dtype=float) for c in ('x', 'y', 'z'))

    # --- Csomópontok helye az atlaszban ---
    voxel = to_voxel(x, y, z, shape)
    point_regions = atlas_matrix[voxel]
    point_side = side_of((x, y, z)[MIDLINE_AXIS], shape)

    # --- Fa-topológia: végpontok és elágazások ---
    id_to_row = {node_id: row for row, node_id in enumerate(ids)}
    parent_rows_all = np.array([id_to_row.get(p, -1) for p in pids], dtype=int)
    child_rows = np.where((parent_rows_all != -1) & (pids != -1))[0]  # sorok, amelyeknek van szülője
    parent_rows = parent_rows_all[child_rows]
    child_counts = np.bincount(parent_rows, minlength=len(ids))

    is_axon = (types == SWC_TYPE_AXON) | (types == SWC_TYPE_AXON_UNDEFINED)
    ep_idx = np.where((child_counts == 0) & is_axon)[0]
    branch_idx = np.where((child_counts > 1) & is_axon)[0]
    proj_idx = np.union1d(ep_idx, branch_idx)
    ep_regions, branch_regions = point_regions[ep_idx], point_regions[branch_idx]
    ep_side, branch_side = point_side[ep_idx], point_side[branch_idx]

    # --- Soma ---
    soma_rows = np.where(types == SWC_TYPE_SOMA)[0]
    if len(soma_rows) > 0:
        soma_idx = int(soma_rows[0])
        soma_region_id = int(point_regions[soma_idx])
        soma_name = region_names.get(soma_region_id, "Unknown region")
        soma_coords = (float(x[soma_idx]), float(y[soma_idx]), float(z[soma_idx]))
        soma_side = int(point_side[soma_idx])
        sx, sy, sz = (int(v[soma_idx]) for v in voxel)
        neighbourhood = atlas_matrix[max(0, sx - 1):sx + 2, max(0, sy - 1):sy + 2, max(0, sz - 1):sz + 2]
        soma_border_fraction = float(np.mean(neighbourhood != soma_region_id))
    else:
        soma_idx, soma_region_id, soma_name, soma_coords = None, -1, "No soma found", (0.0, 0.0, 0.0)
        soma_side, soma_border_fraction = 0, 0.0

    # --- Axonhossz, régiónként és oldalanként ---
    axon_seg = child_rows[is_axon[child_rows]]
    samp_region, samp_side, samp_len = _sample_axon_segments(
        x, y, z, axon_seg, parent_rows_all[axon_seg], atlas_matrix)
    total_axon_length = float(samp_len.sum())

    # --- Egész sejtre vonatkozó oldaliság (célterület nélkül is értelmes) ---
    if soma_side != 0:
        endpoints_ipsi_total = int((ep_side == soma_side).sum())
        endpoints_contra_total = int((ep_side == -soma_side).sum())
        axon_length_ipsi_um = float(samp_len[samp_side == soma_side].sum())
        axon_length_contra_um = float(samp_len[samp_side == -soma_side].sum())
        axon_length_midline_um = float(samp_len[samp_side == 0].sum())
    else:
        endpoints_ipsi_total = endpoints_contra_total = 0
        axon_length_ipsi_um = axon_length_contra_um = axon_length_midline_um = 0.0

    # --- Oldaliság szerinti szűrés a régiós számokhoz (régiónként eltérhet) ---
    def _side_mask(sides: np.ndarray, side: str) -> np.ndarray:
        # Soma nélkül (vagy középvonali sománál) nincs mihez viszonyítani, ezért
        # nem szűrünk oldalra - különben némán nullázódna a sejt.
        if side == 'both' or soma_side == 0:
            return np.ones(len(sides), dtype=bool)
        if side == 'ipsi':
            return sides == soma_side
        return sides == -soma_side

    side_views: dict[str, dict] = {}

    def _side_view(side: str) -> dict:
        """Az adott oldalra szűrt végpontok, elágazások és régiónkénti axonhossz (oldalanként egyszer számolva)."""
        if side not in side_views:
            samp_keep = _side_mask(samp_side, side)
            # Régió-ID-k tömörítése: az Allen ID-k 6*10^8-ig mennek, egy közvetlen
            # bincount sejtenként több GB memóriát foglalna.
            length_ids, inverse = np.unique(samp_region[samp_keep], return_inverse=True)
            ep_keep = _side_mask(ep_side, side)
            side_views[side] = {
                'ep_keep': ep_keep,
                'branch_keep': _side_mask(branch_side, side),
                'endpoint_denominator': int(ep_keep.sum()),
                'length_ids': length_ids,
                'length': np.bincount(inverse, weights=samp_len[samp_keep], minlength=len(length_ids)),
            }
        return side_views[side]

    def _match_ids(region_id: int) -> np.ndarray:
        """A régióhoz tartozó atlasz-ID-k (önmaga + leszármazottai, ha van hierarchia)."""
        ids_ = region_descendants.get(int(region_id))
        return np.fromiter((int(v) for v in ids_), dtype=int) if ids_ else np.array([int(region_id)])

    def _build_region_result(region_id: int) -> RegionResult:
        criteria = criteria_per_region.get(int(region_id), FilterCriteria())
        side = criteria.side or laterality
        view = _side_view(side)
        match = _match_ids(region_id)
        ep_in, br_in = np.isin(ep_regions, match), np.isin(branch_regions, match)
        ep_count = int((ep_in & view['ep_keep']).sum())
        br_count = int((br_in & view['branch_keep']).sum())
        axon_len = float(view['length'][np.isin(view['length_ids'], match)].sum())
        denominator = view['endpoint_denominator']
        fraction = ep_count / denominator if denominator > 0 else 0.0

        def _count_on(in_region: np.ndarray, sides: np.ndarray, side: int) -> int:
            return int((in_region & (sides == side)).sum()) if soma_side != 0 else 0

        return RegionResult(
            region_id=int(region_id),
            region_name=region_names.get(int(region_id), f"Unknown (ID: {region_id})"),
            projects_here=criteria.is_projection(ep_count, br_count, axon_len, fraction),
            endpoint_count=ep_count,
            branch_point_count=br_count,
            axon_length_um=axon_len,
            endpoint_fraction=fraction,
            endpoint_count_ipsi=_count_on(ep_in, ep_side, soma_side),
            endpoint_count_contra=_count_on(ep_in, ep_side, -soma_side),
            branch_point_count_ipsi=_count_on(br_in, branch_side, soma_side),
            branch_point_count_contra=_count_on(br_in, branch_side, -soma_side),
            side=side if soma_side != 0 else 'both',
        )

    target_results = [_build_region_result(rid) for rid in target_region_ids]

    # --- Egyéb vetítési régiók (a célterületeken és a soma régióján kívül) ---
    covered_ids = {int(rid) for rid in target_region_ids}
    for rid in target_region_ids:
        covered_ids.update(int(v) for v in _match_ids(rid))
    proj_regions = point_regions[proj_idx]
    candidate_ids = [int(rid) for rid in np.unique(proj_regions[proj_regions > 0])
                     if int(rid) not in covered_ids and int(rid) != soma_region_id]
    other_projection_regions = [rr for rr in map(_build_region_result, candidate_ids) if rr.projects_here]

    coords = {
        'x': x, 'y': y, 'z': z, 'is_axon': is_axon, 'point_regions': point_regions,
        'proj_idx': proj_idx, 'child_rows': child_rows, 'parent_rows': parent_rows, 'soma_idx': soma_idx,
    }

    return CellAnalysisResult(
        soma_region_id=soma_region_id,
        soma_region_name=soma_name,
        soma_coords=soma_coords,
        target_results=target_results,
        other_projection_regions=other_projection_regions,
        total_axon_length_um=total_axon_length,
        coords=coords,
        soma_border_fraction=soma_border_fraction,
        total_endpoint_count=int(len(ep_idx)),
        annotated_endpoint_count=int(np.sum(ep_regions > 0)),
        laterality=laterality,
        soma_side=soma_side,
        endpoints_ipsi_total=endpoints_ipsi_total,
        endpoints_contra_total=endpoints_contra_total,
        axon_length_ipsi_um=axon_length_ipsi_um,
        axon_length_contra_um=axon_length_contra_um,
        axon_length_midline_um=axon_length_midline_um,
    )


def apply_filter(result: CellAnalysisResult, criteria_per_region: dict[int, FilterCriteria],
                 max_contra_endpoint_pct: float | None = None) -> CellAnalysisResult:
    """
    Eldönti, hogy a sejt átmegy-e a szűrőn (result.passes_filter), tiszta
    halmazműveletekkel:

        passes = (MINDEN 'AND' teljesül)
                 AND (EGYETLEN 'NOT' sem teljesül)
                 AND (ha van 'OR', akkor LEGALÁBB EGY 'OR' teljesül)
                 AND (ha meg van adva: a kontralaterális végpontok aránya <= max_contra_endpoint_pct)

    Sorrendfüggetlen, és egy 'NOT' feltétel hozzáadása a szűrt halmazt csak
    szűkítheti, sosem bővítheti. Ha egyetlen aktív szabály sincs, passes_filter = None.
    A kontralaterális feltétel csak eldönthető oldalú sejtet zár ki (soma nélkül
    nincs mihez viszonyítani).
    """
    active = {rid: c for rid, c in criteria_per_region.items() if c.is_active()}
    if not active and max_contra_endpoint_pct is None:
        result.passes_filter = None
        return result

    contra_ok = (max_contra_endpoint_pct is None or not result.has_hemisphere
                 or result.contra_endpoint_fraction * 100 <= max_contra_endpoint_pct)
    results_by_region = {tr.region_id: tr for tr in result.target_results}
    required_ok, excluded_ok = True, True
    or_exists, or_ok = False, False

    for region_id, crit in active.items():
        tr = results_by_region.get(region_id)
        if tr is None:
            continue
        meets = crit.meets_thresholds(tr)
        if crit.operator == 'OR':
            or_exists = True
            or_ok = or_ok or meets
        elif crit.operator == 'NOT':
            excluded_ok = excluded_ok and not meets
        else:  # 'AND'
            required_ok = required_ok and meets

    result.passes_filter = required_ok and excluded_ok and (or_ok or not or_exists) and contra_ok
    return result


# =============================================================================
# EXPORT ÉS ÖSSZESÍTŐK
# =============================================================================

def _round_pct(fraction: float) -> float:
    return round(fraction * 100, 2)


def results_to_dataframe(
        results: list[tuple[str, CellAnalysisResult]],
        criteria_per_region: dict[int, FilterCriteria] | None = None
) -> pd.DataFrame:
    """Részletes batch-export: soronként egy sejt, régiónként egy oszlopcsoport."""
    criteria_per_region = criteria_per_region or {}
    rows = []
    for cell_name, result in results:
        row = {
            'cell': cell_name,
            'soma_region': result.soma_region_name,
            'summary_soma_region': result.group_region,
            'db_soma_region': result.db_soma_region,
            'projection_class': result.projection_class,
            'projection_subclass': result.projection_subclass,
            'cre_line': result.cre_line,
            'curation_label': result.curation_label,
            'total_axon_length_um': round(result.total_axon_length_um, 1),
            'passes_filter': result.passes_filter,
            'run_hemisphere_setting': result.laterality,
            'soma_on_region_border': result.soma_is_border,
            'soma_border_fraction': round(result.soma_border_fraction, 2),
            'endpoints_total': result.total_endpoint_count,
            'endpoints_in_annotated_regions': result.annotated_endpoint_count,
            'laterality_class': result.laterality_class,
            'projects_contralateral': result.projects_contralaterally,
            'crosses_midline': result.crosses_midline,
            'endpoints_ipsi_total': result.endpoints_ipsi_total,
            'endpoints_contra_total': result.endpoints_contra_total,
            'contra_endpoint_pct': _round_pct(result.contra_endpoint_fraction),
            'axon_um_ipsi': round(result.axon_length_ipsi_um, 1),
            'axon_um_contra': round(result.axon_length_contra_um, 1),
            'axon_um_midline': round(result.axon_length_midline_um, 1),
            'contra_axon_pct': _round_pct(result.contra_axon_fraction),
        }
        for tr in result.target_results:
            # A régió-ID az oszlopnévben: a 30 karakterre vágott nevek ütközhetnek.
            col = f"{tr.region_name.replace(' ', '_').lower()[:30]}_{tr.region_id}"
            row.update({
                f'{col}_projects': tr.projects_here,
                f'{col}_endpoints': tr.endpoint_count,
                f'{col}_branches': tr.branch_point_count,
                f'{col}_axon_um': round(tr.axon_length_um, 1),
                f'{col}_endpoint_pct': _round_pct(tr.endpoint_fraction),
                f'{col}_endpoints_ipsi': tr.endpoint_count_ipsi,
                f'{col}_endpoints_contra': tr.endpoint_count_contra,
                f'{col}_side': tr.side,
            })
            crit = criteria_per_region.get(tr.region_id)
            if crit is not None:
                row[f'{col}_criterion'] = crit.describe()
        rows.append(row)
    return pd.DataFrame(rows)


def _cell_serial(cell_name: str) -> str:
    """'212064/001.swc' -> '212064/001' (a táblákban így jelenik meg a sejt)."""
    return cell_name[:-4] if cell_name.lower().endswith('.swc') else cell_name


def _region_of(result: CellAnalysisResult, region_id: int) -> RegionResult | None:
    return next((tr for tr in result.target_results if tr.region_id == region_id), None)


def _projects_to(result: CellAnalysisResult, region_id: int) -> bool:
    tr = _region_of(result, region_id)
    return tr is not None and tr.projects_here


def _only_projects_to(result: CellAnalysisResult, region_id: int, all_region_ids: list[int]) -> bool:
    """Ide vetít, és a többi felsorolt régió EGYIKÉBE sem (kizárólagos kategória)."""
    if not _projects_to(result, region_id):
        return False
    return not any(_projects_to(result, other) for other in all_region_ids if int(other) != int(region_id))


def _pct(count: int, total: int) -> float:
    return round(100 * count / total, 1) if total else 0.0


def _sorted_df(rows: list[dict], by: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    return df.sort_values(by, ascending=False, kind='stable') if not df.empty else df


def build_soma_distribution_summary(results: list[tuple[str, CellAnalysisResult]],
                                    filter_was_active: bool) -> pd.DataFrame:
    """
    Soma régiónkénti sejtszám és a vetítő sejtek aránya. Aktív szűrőnél a
    "vetítő" = átment a szűrőn, különben = legalább egy célterületbe vetít.
    """
    per_soma: dict[str, dict] = {}
    for cell_name, r in results:
        entry = per_soma.setdefault(r.group_region, {'total': 0, 'ids': []})
        entry['total'] += 1
        if filter_was_active:
            is_projecting = bool(r.passes_filter)
        else:
            is_projecting = any(tr.projects_here for tr in r.target_results)
        if is_projecting:
            entry['ids'].append(_cell_serial(cell_name))

    return _sorted_df([{
        "Soma Region": soma,
        "Total Cells": entry['total'],
        "Valid Projections": len(entry['ids']),
        "Valid Projections %": _pct(len(entry['ids']), entry['total']),
        "Projecting Cell IDs": ", ".join(entry['ids']),
    } for soma, entry in per_soma.items()], "Total Cells")


LATERALITY_CLASS_LABELS = {
    'ipsi_only': 'Ipsilateral only (no midline crossing)',
    'crosses_only': 'Crosses midline, but no endpoints there',
    'contra': 'Projects contralaterally (endpoints on the far side)',
    'unknown': 'Undetermined (no soma, or soma on the midline)',
}
LATERALITY_CLASS_ORDER = ['ipsi_only', 'crosses_only', 'contra', 'unknown']


def build_laterality_summary(results: list[tuple[str, CellAnalysisResult]]) -> dict:
    """
    Féltekei összesítő: összesen, soma régiónként és sejtenként. A százalékok
    nevezője az ELDÖNTHETŐ sejtek száma - a soma nélküli / középvonali sejtek
    saját kategóriát kapnak, és nem torzítják az arányokat.
    """
    ids_by_class = {k: [] for k in LATERALITY_CLASS_ORDER}
    by_soma: dict[str, dict[str, list]] = {}
    per_cell_rows = []

    for cell_name, r in results:
        cls, serial, soma = r.laterality_class, _cell_serial(cell_name), r.group_region
        ids_by_class[cls].append(serial)
        by_soma.setdefault(soma, {k: [] for k in LATERALITY_CLASS_ORDER})[cls].append(serial)
        per_cell_rows.append({
            'Cell': serial,
            'Soma region': soma,
            'Class': LATERALITY_CLASS_LABELS[cls],
            'Endpoints ipsi': r.endpoints_ipsi_total,
            'Endpoints contra': r.endpoints_contra_total,
            'Contra endpoint %': _round_pct(r.contra_endpoint_fraction),
            'Axon ipsi (um)': round(r.axon_length_ipsi_um, 1),
            'Axon contra (um)': round(r.axon_length_contra_um, 1),
            'Axon midline (um)': round(r.axon_length_midline_um, 1),
            'Contra axon %': _round_pct(r.contra_axon_fraction),
        })

    n_total = len(results)
    n_decided = n_total - len(ids_by_class['unknown'])
    overall = pd.DataFrame([{
        'Category': LATERALITY_CLASS_LABELS[cls],
        'Cells': len(ids_by_class[cls]),
        '% of decided': _pct(len(ids_by_class[cls]), n_decided) if n_decided and cls != 'unknown' else None,
        'Cell IDs': ", ".join(ids_by_class[cls]),
    } for cls in LATERALITY_CLASS_ORDER])

    soma_rows = []
    for soma, per_class in by_soma.items():
        decided = sum(len(v) for k, v in per_class.items() if k != 'unknown')
        not_contra = len(per_class['ipsi_only']) + len(per_class['crosses_only'])
        soma_rows.append({
            'Soma region': soma,
            'Total cells': sum(len(v) for v in per_class.values()),
            'Decided': decided,
            'Ipsilateral only': len(per_class['ipsi_only']),
            'Crosses, no endpoints': len(per_class['crosses_only']),
            'Contralateral': len(per_class['contra']),
            'Undetermined': len(per_class['unknown']),
            'No contralateral projection': not_contra,
            'No contralateral %': _pct(not_contra, decided) if decided else None,
            'Ipsilateral-only cell IDs': ", ".join(per_class['ipsi_only']),
        })

    return {
        'overall': overall,
        'by_soma': _sorted_df(soma_rows, 'Total cells'),
        'per_cell': pd.DataFrame(per_cell_rows),
        'n_total': n_total,
        'n_decided': n_decided,
        'counts': {cls: len(ids) for cls, ids in ids_by_class.items()},
    }


def category_slugs(labels: list[str]) -> dict[str, str]:
    """
    Egyedi, fájlnévbe illeszthető azonosító minden kategóriához. Az " only"
    utótag a csonkolás UTÁN kerül vissza, így az inkluzív és a kizárólagos
    tábla sosem kapja ugyanazt a nevet.
    """
    slugs, used = {}, set()
    for label in labels:
        is_only = label.lower().endswith(' only')
        base = label[:-5] if is_only else label
        stem = ''.join(ch if ch.isalnum() else '_' for ch in base.lower())[:24].strip('_')
        if is_only:
            stem += '_only'
        candidate, n = stem, 2
        while candidate in used:
            candidate, n = f"{stem}_{n}", n + 1
        used.add(candidate)
        slugs[label] = candidate
    return slugs


def build_cortical_summary(
        results: list[tuple[str, CellAnalysisResult]],
        base_region_id: int | None,
        numerator_region_ids: list[int],
        region_label_fn,
        criteria_per_region: dict[int, FilterCriteria] | None = None,
        laterality: str | None = None,
) -> dict:
    """
    A kész, soma régiónkénti összesítő táblák egy batch futásból.

    A base régió (tipikusan a leszálló agytörzs) definiálja a PT sejteket = 100%;
    a többi régió a számláló. Szándékosan NEM a szűrési szabályokból (AND/NOT/OR)
    épül, hanem a régiónkénti vetítés-definícióból (projects_here).
      benne      - base = 100%: inkluzív ("GPe n") és, több számlálónál,
                   kizárólagos ("GPe only n") kategóriák + "All targets"
      nelkul     - az összes sejt = 100% (nincs base-követelmény)
      axon       - átlagos axonhossz a célterületben a vetítő PT sejtek között
      categories - kategóriánként a vetítő sejtek sorszámai
    A soma nélküli / annotálatlan régióban ülő somájú sejtek kimaradnak
    (skipped_no_soma_region).
    """
    criteria_per_region = criteria_per_region or {}
    multi = len(numerator_region_ids) > 1

    groups: dict[str, list] = defaultdict(list)
    skipped = 0
    for name, r in results:
        if r.soma_region_id <= 0:
            skipped += 1
            continue
        groups[r.group_region].append((name, r))

    num_labels = [region_label_fn(rid) for rid in numerator_region_ids]
    base_label = region_label_fn(base_region_id) if base_region_id is not None else 'All L5'
    base_col = f"PT Cells ({base_label}=100%)"

    def is_base(r: CellAnalysisResult) -> bool:
        return base_region_id is None or _projects_to(r, base_region_id)

    def projects_to_all(r: CellAnalysisResult) -> bool:
        return bool(numerator_region_ids) and all(_projects_to(r, rid) for rid in numerator_region_ids)

    def category_row(soma: str, n_base: int, label: str, hits: list) -> dict:
        ids = sorted(_cell_serial(n) for n, _ in hits)
        return {
            "Soma Region": soma,
            base_col: n_base,
            f"{label} Projects": len(ids),
            f"{label} % of PT": _pct(len(ids), n_base),
            "Projecting Cell IDs": ", ".join(ids),
        }

    benne_rows, nelkul_rows, axon_rows = [], [], []
    category_rows: dict[str, list] = defaultdict(list)

    for soma, cells in sorted(groups.items()):
        base_cells = [(n, r) for n, r in cells if is_base(r)]
        total, n_base = len(cells), len(base_cells)
        row_b = {"Soma Region": soma, base_col: n_base}
        row_n = {"Soma Region": soma, "Total L5 Cells": total}
        row_a = {"Soma Region": soma, "PT Cells": n_base}

        for rid, lab in zip(numerator_region_ids, num_labels):
            hits_base = [(n, r) for n, r in base_cells if _projects_to(r, rid)]
            n_all = sum(1 for _, r in cells if _projects_to(r, rid))
            row_b[f"{lab} n"], row_b[f"{lab} %"] = len(hits_base), _pct(len(hits_base), n_base)
            if multi:
                only_base = [(n, r) for n, r in base_cells if _only_projects_to(r, rid, numerator_region_ids)]
                row_b[f"{lab} only n"], row_b[f"{lab} only %"] = len(only_base), _pct(len(only_base), n_base)
            row_n[f"{lab} n"], row_n[f"{lab} %"] = n_all, _pct(n_all, total)
            if multi:
                n_only_all = sum(1 for _, r in cells if _only_projects_to(r, rid, numerator_region_ids))
                row_n[f"{lab} only n"], row_n[f"{lab} only %"] = n_only_all, _pct(n_only_all, total)

            lengths = [_region_of(r, rid).axon_length_um for _, r in hits_base]
            row_a[f"{lab} mean axon µm"] = round(sum(lengths) / len(lengths), 1) if lengths else 0.0

            category_rows[lab].append(category_row(soma, n_base, lab, hits_base))
            if multi:
                category_rows[f"{lab} only"].append(category_row(soma, n_base, f"{lab} only", only_base))

        all_base = [(n, r) for n, r in base_cells if projects_to_all(r)]
        n_all_targets = sum(1 for _, r in cells if projects_to_all(r))
        row_b["All targets n"], row_b["All targets %"] = len(all_base), _pct(len(all_base), n_base)
        row_n["All targets n"], row_n["All targets %"] = n_all_targets, _pct(n_all_targets, total)
        category_rows["All targets"].append(category_row(soma, n_base, "All targets", all_base))

        benne_rows.append(row_b)
        nelkul_rows.append(row_n)
        axon_rows.append(row_a)

    categories = {lab: _sorted_df(category_rows[lab], f"{lab} Projects") for lab in num_labels}
    if multi:
        for lab in num_labels:
            categories[f"{lab} only"] = _sorted_df(category_rows[f"{lab} only"], f"{lab} only Projects")
        categories["All targets"] = _sorted_df(category_rows["All targets"], "All targets Projects")

    # A használt kritérium (a régiónként ténylegesen használt féltekével) a
    # fájlnevekbe és a feliratba kerül.
    lat = laterality or (results[0][1].laterality if results else 'both')
    involved = ([base_region_id] if base_region_id is not None else []) + list(numerator_region_ids)
    used = [criteria_per_region.get(rid, FilterCriteria()).effective(lat) for rid in involved]
    if len({c.describe() for c in used}) <= 1:
        crit = used[0] if used else FilterCriteria().effective(lat)
        criteria_note, slug = crit.describe(), crit.slug()
    else:
        criteria_note = " · ".join(f"{region_label_fn(rid)}: {c.describe()}" for rid, c in zip(involved, used))
        sides = {c.side for c in used}
        slug = "mixed" + (f"_{sides.pop()}" if len(sides) == 1 and sides <= set(_SIDE_NOTES) else "")

    return {
        "benne": _sorted_df(benne_rows, base_col),
        "nelkul": _sorted_df(nelkul_rows, "Total L5 Cells"),
        "axon": _sorted_df(axon_rows, "PT Cells"),
        "categories": categories,
        "criteria_note": criteria_note,
        "slug": slug,
        "skipped_no_soma_region": skipped,
    }
