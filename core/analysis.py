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
# A "VALÓDI VETÍTÉS" DEFINÍCIÓJA
# =============================================================================
# Egy sejt akkor vetít VALÓDIAN egy régióba, ha ott terminális arborizációt ad:
# azaz van legalább ennyi végpontja (terminális) ÉS legalább ennyi elágazása.
# A pusztán ÁTHALADÓ axonnak (ami csak keresztezi a régiót, de máshol végződik)
# nincs sem végpontja, sem elágazása a régióban -> így nem számít vetítésnek.
# Régen a kód végpont VAGY elágazás alapján döntött, ezért a főaxon egyetlen
# elágazása (kollaterális leadása) is "vetítésnek" látszott az áthaladt régióban.
MIN_ENDPOINTS_FOR_PROJECTION = 1
MIN_BRANCH_POINTS_FOR_PROJECTION = 1


def _is_true_projection(endpoint_count: int, branch_point_count: int) -> bool:
    """Valódi terminális arborizáció-e: végpont ÉS elágazás is kell hozzá."""
    return (endpoint_count >= MIN_ENDPOINTS_FOR_PROJECTION and
            branch_point_count >= MIN_BRANCH_POINTS_FOR_PROJECTION)


# =============================================================================
# ADATSTRUKTÚRÁK
# =============================================================================

@dataclass
class RegionResult:
    """Egyetlen célterület analízisének eredménye."""
    region_id: int
    region_name: str
    projects_here: bool  # Valódi terminális arborizáció-e (végpont ÉS elágazás alapján)
    endpoint_count: int  # Csak végpontok száma (gyerek nélküli axon csomópontok)
    branch_point_count: int  # Csak elágazási pontok száma (>1 gyerek)
    projection_point_count: int  # Végpontok + elágazási pontok összesen
    axon_length_um: float  # Axonhossz mikrométerben ebben a régióban
    # A régió végpontjainak aránya a sejt ÖSSZES axon-végpontjához képest [0..1].
    # Ez teszi lehetővé a méret-független szűrést, pl. a L6 sejtek kiszűrését,
    # amelyek végpontjaik túlnyomó része a thalamusba esik.
    endpoint_fraction: float = 0.0


@dataclass
class FilterCriteria:
    """
    A felhasználó által megadott szűrési feltételek egy célterületre,
    kiegészítve a halmazműveleti logikával (AND, OR, NOT).
    """
    min_endpoints: int = 0
    min_branch_points: int = 0
    min_axon_length_um: float = 0
    # Méret-független küszöb: a régió végpontjainak minimális aránya a sejt összes
    # végpontjához képest [0..1]. NOT operátorral párosítva ez a L6-szűrő:
    # pl. "thalamus végpont-arány >= 2.5%" => NOT => a L6 sejtek kizárása.
    min_endpoint_fraction: float = 0.0
    operator: str = 'AND'  # 'AND', 'OR', 'NOT'

    def _has_threshold(self) -> bool:
        """Van-e legalább egy tényleges numerikus küszöb beállítva."""
        return (self.min_endpoints > 0 or
                self.min_branch_points > 0 or
                self.min_axon_length_um > 0 or
                self.min_endpoint_fraction > 0)

    def is_active(self) -> bool:
        """
        Aktív-e a feltétel (részt vesz-e a szűrésben).

        JAVÍTVA (pitfall #2): korábban a 'Required (AND)' szabály küszöb nélkül
        INAKTÍV volt, így némán eldobtuk. Emiatt a "vetítsen a GPe-be (AND)"
        elvárás semmit nem csinált, ha nem állítottak be hozzá számot; ráadásul
        egy másik (pl. L6 = NOT) feltétel bekapcsolása hirtelen "aktívvá" tette az
        egész szűrőt, megváltoztatva a számlálás alapját - így fordulhatott elő,
        hogy egy KIZÁRÓ szűrő hatására NŐTT egy régió sejtszáma.

        Most minden EXPLICIT operátor aktív: az AND azt jelenti, "ide vetítenie
        kell" (valódi végpont+elágazás), a NOT azt, "ide nem vetíthet", az OR
        pedig az opcionális uniót. A küszöbök ezt csak tovább szigorítják.
        """
        return True

    def meets_thresholds(self, region_result: 'RegionResult') -> bool:
        """
        Kiértékeli, hogy a régió önmagában megüti-e a küszöböt.

        Fontos: minden megadott küszöbnek EGYSZERRE kell teljesülnie (ÉS-kapcsolat).
        Ha egyetlen numerikus küszöb sincs megadva, akkor pusztán azt vizsgáljuk,
        hogy a sejt valódi terminális arborizációt ad-e ide (projects_here), ami
        önmagában is végpont ÉS elágazás meglétét jelenti.
        """
        if not self._has_threshold():
            return region_result.projects_here

        if region_result.endpoint_count < self.min_endpoints:
            return False
        if region_result.branch_point_count < self.min_branch_points:
            return False
        if region_result.axon_length_um < self.min_axon_length_um:
            return False
        if region_result.endpoint_fraction < self.min_endpoint_fraction:
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
        target_region_ids: list[int],
        region_descendants: dict[int, set[int]] | None = None,
        region_names: dict[int, str] | None = None
) -> CellAnalysisResult:
    """
    region_descendants: opcionális {régió_id -> {atlasz ID-k halmaza}} leképezés
    (lásd loader.build_region_descendants). Ha meg van adva, egy célterület
    minden leszármazott magját is beleszámoljuk - így a SZÜLŐ régiók (Brain stem,
    Thalamus) helyesen fedik le az összes alárendelt magot. Ha None, akkor a régi,
    pontos ID-egyezéses viselkedés marad.

    region_names: opcionális {régió_id -> megjelenítendő név} felülírás. A virtuális
    régióknak (pl. a thalamus nélküli "leszálló agytörzs") nincs soruk a szótárban,
    ezért a nevüket itt adjuk meg.
    """
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

    # A sejt ÖSSZES axon-végpontja - ez a méret-független (%-os) szűrés nevezője.
    total_endpoint_count = int(len(ep_idx))
    region_descendants = region_descendants or {}
    region_names = region_names or {}

    def _match_ids(region_id: int) -> np.ndarray:
        """A régióhoz tartozó atlasz-ID-k (önmaga + leszármazottai, ha van hierarchia)."""
        ids = region_descendants.get(int(region_id))
        if ids:
            return np.fromiter((int(v) for v in ids), dtype=int)
        return np.array([int(region_id)], dtype=int)

    def _build_region_result(region_id: int) -> RegionResult:
        """Egyetlen régió eredményének kiszámítása egységes definícióval.

        Egy helyen dől el, mi számít végpontnak, elágazásnak és VALÓDI
        vetítésnek - így a célterületek, az "egyéb" régiók, a statisztikák és
        a szűrő mind pontosan ugyanazt a logikát látják. A régió a szülő-régió
        esetén az összes leszármazott magot is magába foglalja (_match_ids).
        """
        if region_id in region_names:
            region_name = region_names[region_id]
        else:
            name_matches = dictionary.loc[dictionary['id'] == region_id, 'safe_name'].tolist()
            region_name = name_matches[0] if name_matches else f"Unknown (ID: {region_id})"

        match = _match_ids(region_id)
        ep_count = int(np.isin(ep_regions, match).sum())
        br_count = int(np.isin(branch_regions, match).sum())
        proj_count = ep_count + br_count

        region_axon_mask = axon_mask_curr & np.isin(point_regions[curr_idx], match)
        axon_len = float(np.sum(distances[region_axon_mask]))

        fraction = (ep_count / total_endpoint_count) if total_endpoint_count > 0 else 0.0

        return RegionResult(
            region_id=int(region_id), region_name=region_name,
            # JAVÍTVA: végpont ÉS elágazás is kell, nem "vagy" - így az áthaladó
            # axonok nem számítanak hamis vetítésnek.
            projects_here=_is_true_projection(ep_count, br_count),
            endpoint_count=ep_count, branch_point_count=br_count,
            projection_point_count=proj_count, axon_length_um=axon_len,
            endpoint_fraction=fraction,
        )

    target_results = [_build_region_result(region_id) for region_id in target_region_ids]

    # A célterületek által lefedett összes atlasz-ID (szülő + leszármazottak),
    # hogy egy célrégió alrégiói ne jelenjenek meg tévesen "egyéb" vetítésként.
    covered_ids = set(int(r) for r in target_region_ids)
    for rid in target_region_ids:
        covered_ids.update(int(v) for v in _match_ids(rid))

    # Az "egyéb" vetítéseknél is a valódi-vetítés definíciót használjuk: egy régió
    # csak akkor kerül a listára, ha van ott végpont ÉS elágazás is. Régen elég volt
    # egyetlen áthaladó elágazás, ami rengeteg hamis "egyéb célterületet" adott.
    unique_proj_regions = np.unique(proj_regions[proj_regions > 0])
    other_region_ids = [
        int(rid) for rid in unique_proj_regions
        if int(rid) not in covered_ids and int(rid) != soma_region_id
    ]

    other_projection_regions = [
        rr for region_id in other_region_ids
        if (rr := _build_region_result(region_id)).projects_here
    ]

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
    Eldönti, hogy egy sejt átmegy-e a szűrőn, TISZTA HALMAZMŰVELETEKKEL.

    A feltételeket három, egymástól független csoportba soroljuk, és a végső
    döntés e három csoport metszete:

        passes = (MINDEN 'AND' teljesül)
                 AND (EGYETLEN 'NOT' sem teljesül)
                 AND (ha van 'OR', akkor LEGALÁBB EGY 'OR' teljesül)

    Ez a kiértékelés szándékosan SORRENDFÜGGETLEN: a régiókon való végigiterálás
    sorrendje nem befolyásolja az eredményt, mert csak logikai ÉS/VAGY-ot
    halmozunk. Ebből következik a legfontosabb tulajdonság is, ami a L6-szűrő
    anomáliáját okozta: egy 'NOT' (kizáró) feltétel HOZZÁADÁSA a szűrt halmazt
    csak SZŰKÍTHETI, sosem bővítheti - tehát a L6 sejtek eltávolítása után egyik
    régió sejtszáma sem nőhet.
    """
    active = {rid: c for rid, c in criteria_per_region.items() if c.is_active()}
    if not active:
        result.passes_filter = None
        return result

    results_by_region = {tr.region_id: tr for tr in result.target_results}

    required_ok = True   # minden AND teljesül
    excluded_ok = True   # egyetlen NOT sem teljesül
    or_exists = False
    or_ok = False        # legalább egy OR teljesül

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

    result.passes_filter = required_ok and excluded_ok and (or_ok or not or_exists)
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
            # Végpont-arány %-ban - ez alapján azonosíthatók a L6 sejtek
            # (pl. thalamus-arány > 2.5%).
            row[f'{safe_col}_endpoint_pct'] = round(tr.endpoint_fraction * 100, 2)
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# KÉRGI VETÍTÉSI ÖSSZESÍTŐ (a Nóra által kért végleges táblázatok)
# =============================================================================
# Ez a modul EGYBŐL a helyes összesítőket állítja elő, kézi táblázat-építés nélkül:
#   - "agytörzs = 100%" (PT sejtek a nevezőben)  -> bs_benne
#   - "összes L5 = 100%" (agytörzs-feltétel nélkül) -> bs_nelkul
#   - régiónkénti átlag axonhossz a célterületeken
#   - kategória-táblák a vetítő sejtek sorszámaival
#
# FONTOS: mindenhol a projects_here (végpont ÉS elágazás) definíciót használjuk,
# közvetlenül - NEM a sidebar szűrőt. Így elkerüljük a két korábbi buktatót:
#   (1) a 2.5%-os L6-szűrő, ami a motoros PT sejteket is kidobta, és
#   (2) a rossz nevező (összes L5 helyett PT sejtek).


def _cell_serial(cell_name: str) -> str:
    """A .swc kiterjesztés nélküli sorszám (adatbázis-kereséshez)."""
    return cell_name[:-4] if cell_name.lower().endswith('.swc') else cell_name


def _region_of(result: CellAnalysisResult, region_id: int) -> RegionResult | None:
    for tr in result.target_results:
        if tr.region_id == region_id:
            return tr
    return None


def _projects_to(result: CellAnalysisResult, region_id: int) -> bool:
    tr = _region_of(result, region_id)
    return bool(tr and tr.projects_here)


def build_cortical_summary(
        results: list[tuple[str, CellAnalysisResult]],
        base_region_id: int | None,
        numerator_region_ids: list[int],
        region_label_fn,
) -> dict:
    """
    Kérgi régiónkénti összesítők a Nóra-féle definíciók szerint.

    base_region_id: a "100%" populációt definiáló régió (pl. leszálló agytörzs =
        PT sejtek). Ha None, akkor a nevező az ÖSSZES L5 sejt.
    numerator_region_ids: a célterületek (pl. GPe, TRN), amelyekre a %-ot adjuk.
    region_label_fn: régió_id -> megjelenítendő név.

    Visszatér: {'benne', 'nelkul', 'axon', 'categories'} DataFrame-ekkel.
    """
    from collections import defaultdict

    groups: dict[str, list] = defaultdict(list)
    for name, r in results:
        groups[r.soma_region_name].append((name, r))

    num_labels = [region_label_fn(rid) for rid in numerator_region_ids]
    base_label = region_label_fn(base_region_id) if base_region_id is not None else "All L5"
    base_col = f"PT Cells ({base_label}=100%)"

    def is_base(r: CellAnalysisResult) -> bool:
        return True if base_region_id is None else _projects_to(r, base_region_id)

    def meets_all(r: CellAnalysisResult) -> bool:
        return bool(numerator_region_ids) and all(_projects_to(r, rid) for rid in numerator_region_ids)

    benne_rows, nelkul_rows, axon_rows = [], [], []
    cat_rows: dict[str, list] = {lab: [] for lab in num_labels}
    cat_all_rows: list = []

    for soma, cells in sorted(groups.items()):
        total = len(cells)
        base_cells = [(n, r) for (n, r) in cells if is_base(r)]
        nbase = len(base_cells)

        row_b = {"Soma Region": soma, base_col: nbase}
        row_n = {"Soma Region": soma, "Total L5 Cells": total}
        row_a = {"Soma Region": soma, "PT Cells": nbase}

        for rid, lab in zip(numerator_region_ids, num_labels):
            cb = sum(1 for (_, r) in base_cells if _projects_to(r, rid))
            row_b[f"{lab} n"] = cb
            row_b[f"{lab} %"] = round(100 * cb / nbase, 1) if nbase else 0.0

            cn = sum(1 for (_, r) in cells if _projects_to(r, rid))
            row_n[f"{lab} n"] = cn
            row_n[f"{lab} %"] = round(100 * cn / total, 1) if total else 0.0

            lens = [_region_of(r, rid).axon_length_um for (_, r) in base_cells if _projects_to(r, rid)]
            row_a[f"{lab} mean axon µm"] = round(sum(lens) / len(lens), 1) if lens else 0.0

            ids = sorted(_cell_serial(n) for (n, r) in base_cells if _projects_to(r, rid))
            cat_rows[lab].append({
                "Soma Region": soma, base_col: nbase,
                f"{lab} Projects": len(ids),
                f"{lab} % of PT": round(100 * len(ids) / nbase, 1) if nbase else 0.0,
                "Projecting Cell IDs": ", ".join(ids),
            })

        cb_all = sum(1 for (_, r) in base_cells if meets_all(r))
        row_b["All targets n"] = cb_all
        row_b["All targets %"] = round(100 * cb_all / nbase, 1) if nbase else 0.0
        cn_all = sum(1 for (_, r) in cells if meets_all(r))
        row_n["All targets n"] = cn_all
        row_n["All targets %"] = round(100 * cn_all / total, 1) if total else 0.0

        ids_all = sorted(_cell_serial(n) for (n, r) in base_cells if meets_all(r))
        cat_all_rows.append({
            "Soma Region": soma, base_col: nbase,
            "All targets Projects": len(ids_all),
            "All targets % of PT": round(100 * len(ids_all) / nbase, 1) if nbase else 0.0,
            "Projecting Cell IDs": ", ".join(ids_all),
        })

        benne_rows.append(row_b)
        nelkul_rows.append(row_n)
        axon_rows.append(row_a)

    benne = pd.DataFrame(benne_rows).sort_values(base_col, ascending=False)
    nelkul = pd.DataFrame(nelkul_rows).sort_values("Total L5 Cells", ascending=False)
    axon = pd.DataFrame(axon_rows).sort_values("PT Cells", ascending=False)

    categories = {}
    for lab in num_labels:
        categories[lab] = pd.DataFrame(cat_rows[lab]).sort_values(f"{lab} Projects", ascending=False)
    if len(numerator_region_ids) > 1:
        categories["All targets"] = pd.DataFrame(cat_all_rows).sort_values("All targets Projects", ascending=False)

    return {"benne": benne, "nelkul": nelkul, "axon": axon, "categories": categories}