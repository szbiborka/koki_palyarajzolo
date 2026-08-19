# ANALÍZIS MODUL - A tudományos számítási logika.
# Ez a modul semmilyen UI elemet nem tartalmaz - csak tiszta Python/NumPy.
# Ez a szándékos: bármikor tesztelhető és bővíthető a Streamlit-től függetlenül.

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from config import (
    VOXEL_SIZE, SWC_TYPE_SOMA, SWC_TYPE_AXON, SWC_TYPE_AXON_UNDEFINED,
    DEFAULT_FILTER, MIDLINE_AXIS, DEFAULT_LATERALITY,
)

# A VETÍTÉS DEFINÍCIÓJA - EGYETLEN HELYEN
# Régiónként EGY kritériumkészlet (FilterCriteria) mondja meg, mi számít
# vetítésnek. Ugyanez hajtja a "..._projects" pipát, a szűrést és az összesítő
# táblákat is - nincs külön "globális definíció" és külön "szűrő", amik
# ellentmondhatnának egymásnak.
#
# Alapértelmezés: legalább 1 végpont ÉS legalább 1 elágazás = valódi terminális
# arborizáció. A pusztán ÁTHALADÓ axonnak (ami csak keresztezi a régiót, de
# máshol végződik) nincs végpontja a régióban -> nem számít vetítésnek.
# A küszöbök emelésével szigorítható, az elágazás 0-ra állításával lazítható.
MIN_ENDPOINTS_FOR_PROJECTION = DEFAULT_FILTER['min_endpoints']
MIN_BRANCH_POINTS_FOR_PROJECTION = DEFAULT_FILTER['min_branch_points']


def _is_true_projection(endpoint_count: int, branch_point_count: int) -> bool:
    """Az alapértelmezett kritérium szerinti valódi terminális arborizáció."""
    return (endpoint_count >= MIN_ENDPOINTS_FOR_PROJECTION and
            branch_point_count >= MIN_BRANCH_POINTS_FOR_PROJECTION)


# ADATSTRUKTÚRÁK

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
    # OLDALISÁG: az Allen atlaszban mindkét félteke ugyanazt a régió-ID-t viseli,
    # ezért külön számoljuk a soma oldalán (ipszi) és a túloldalon (kontra) lévő
    # pontokat. Ezek MINDIG a teljes (mindkét oldali) képet mutatják, függetlenül
    # attól, hogy a fenti számok melyik oldalról szólnak - így látható, mekkora a
    # kontralaterális hozzájárulás.
    endpoint_count_ipsi: int = 0
    endpoint_count_contra: int = 0
    branch_point_count_ipsi: int = 0
    branch_point_count_contra: int = 0


@dataclass
class FilterCriteria:
    """
    EGY régió vetítési kritériuma - és egyben a szűrési feltétel.

    Ez az egyetlen hely, ahol eldől, mi számít vetítésnek az adott régióban.
    Ugyanez hajtja a "..._projects" pipát, a szűrést (passes_filter) és az
    összesítő táblákat, tehát nem lehet közöttük ellentmondás.

    Alapértelmezés: >=1 végpont ÉS >=1 elágazás (valódi terminális arborizáció).
    A számok emelésével szigorítható; az elágazás 0-ra állításával lazítható
    "csak végpont" logikára. Az operator (AND/NOT/OR) NEM a vetítés definíciója,
    hanem azt mondja meg, hogyan kombináljuk a régiókat egymással.
    """
    min_endpoints: int = MIN_ENDPOINTS_FOR_PROJECTION
    min_branch_points: int = MIN_BRANCH_POINTS_FOR_PROJECTION
    min_axon_length_um: float = 0
    # Méret-független küszöb: a régió végpontjainak minimális aránya a sejt összes
    # végpontjához képest [0..1]. NOT operátorral párosítva ez a L6-szűrő:
    # pl. "thalamus végpont-arány >= 2.5%" => NOT => a L6 sejtek kizárása.
    min_endpoint_fraction: float = 0.0
    # 'AND'  = ide vetítenie kell
    # 'NOT'  = ide nem vetíthet
    # 'OR'   = opcionális (legalább egy OR-régió teljesüljön)
    # 'NONE' = CSAK MEGFIGYELÉS: a régió számai megjelennek, de a szűrést nem
    #          befolyásolja. Enélkül egy pusztán "megnézni" hozzáadott régió
    #          némán kötelező feltétellé vált volna.
    operator: str = 'AND'

    def is_projection(self, endpoint_count: int, branch_point_count: int,
                      axon_length_um: float = 0.0, endpoint_fraction: float = 0.0) -> bool:
        """Vetít-e ide a sejt E SZERINT a kritérium szerint (minden feltétel EGYSZERRE)."""
        return (endpoint_count >= self.min_endpoints and
                branch_point_count >= self.min_branch_points and
                axon_length_um >= self.min_axon_length_um and
                endpoint_fraction >= self.min_endpoint_fraction)

    def is_active(self) -> bool:
        """
        Részt vesz-e ez a régió a SZŰRÉSBEN.

        Minden explicit szabály (AND / NOT / OR) aktív - korábban a küszöb
        nélküli AND némán kimaradt, ami miatt egy kizáró szűrőtől NŐHETETT egy
        régió sejtszáma. A 'NONE' viszont szándékosan inaktív: így hozzá lehet
        adni egy régiót pusztán megfigyelésre (látszanak a számai, exportba is
        bekerül), anélkül hogy némán kötelező feltétellé válna.
        """
        return self.operator != 'NONE'

    def meets_thresholds(self, region_result: 'RegionResult') -> bool:
        """
        A régió teljesíti-e a kritériumot.

        A projects_here-t MÁR ezzel a kritériummal számoltuk ki a run_analysis-ben,
        ezért itt egyszerűen azt olvassuk vissza. Így a pipa és a szűrő
        definíció szerint ugyanaz - nem tudnak ellentmondani egymásnak.
        """
        return region_result.projects_here

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
        return " AND ".join(parts) if parts else "any axon presence"

    def slug(self) -> str:
        """Fájlnévbe illeszthető rövid azonosító (pl. 'ep1_br1')."""
        return f"ep{self.min_endpoints}_br{self.min_branch_points}"


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
    # A soma körüli 3x3x3 voxel hányad része esik MÁS régióba [0..1]. A 25 um-es
    # voxelrács miatt a határ közeli somák besorolása bizonytalan - ez a mérőszám
    # teszi láthatóvá, mely sejteket érdemes szemre ellenőrizni.
    soma_border_fraction: float = 0.0
    # A sejt axon-végpontjai közül hány esik ANNOTÁLT régióba (region_id > 0).
    # A végpont-arány (endpoint_fraction) nevezője az ÖSSZES végpont, beleértve
    # az atlaszon kívülre esőket is - ez a két szám teszi átláthatóvá a különbséget.
    total_endpoint_count: int = 0
    annotated_endpoint_count: int = 0
    # Melyik oldal(ak) számítottak be a vetítésekbe ('both' / 'ipsi' / 'contra').
    laterality: str = 'both'

    @property
    def soma_is_border(self) -> bool:
        """A soma régióhatáron van-e (a besorolás bizonytalan)."""
        return self.soma_border_fraction > 0.0


# FŐ ANALÍZIS FÜGGVÉNY

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
    """
    laterality: 'both' (alapértelmezés, a korábbi viselkedés), 'ipsi' vagy
    'contra'. Az Allen atlaszban mindkét félteke ugyanazt a régió-ID-t viseli,
    ezért a régió-ID önmagában nem árulja el az oldalt; a középvonalhoz képest
    döntjük el. A RegionResult MINDIG tartalmazza az ipszi/kontra bontást is,
    függetlenül attól, melyik módban futunk.

    criteria_per_region: régiónkénti vetítési kritérium. Ez határozza meg a
    projects_here értéket, így a szűrés és az összesítők is pontosan ezt követik.
    Ha egy régióhoz nincs megadva, az alapértelmezés (>=1 végpont ÉS >=1 elágazás)
    érvényes.

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

    # -------------------------------------------------------------------------
    # AXONHOSSZ RÉGIÓNKÉNT - a régióhatárokon FELDARABOLVA
    # -------------------------------------------------------------------------
    # Régen egy teljes szakasz hossza a GYERMEK csomópont régiójához került, még
    # akkor is, ha a szakasz átlépett egy határt. Egy 125 um-es kéreg->GPe ugrás
    # így teljes egészében a GPe-nek számított, ami felfelé torzította a
    # célterületek hosszát.
    #
    # Most a szakaszokat félvoxelnyi (12.5 um) lépésekre mintavételezzük, és
    # minden mintadarab hosszát abba a régióba könyveljük, amelyikbe a
    # középpontja esik. A rövid szakaszok (a legtöbb SWC szakasz néhány um) 1
    # mintát kapnak, tehát a többletköltség elhanyagolható; a hosszú, határt
    # átlépő szakaszok viszont arányosan oszlanak meg a régiók között.
    axon_seg = np.where(axon_mask_curr)[0]
    if len(axon_seg) > 0:
        seg_child = curr_idx[axon_seg]
        seg_parent = p_idx[axon_seg]
        seg_len = distances[axon_seg]

        # Hány mintára bontsuk az egyes szakaszokat (min. 1, felső korláttal)
        n_samp = np.maximum(1, np.ceil(seg_len / (VOXEL_SIZE * 0.5)).astype(int))
        n_samp = np.minimum(n_samp, 256)

        seg_id = np.repeat(np.arange(len(seg_len)), n_samp)
        starts = np.concatenate(([0], np.cumsum(n_samp)[:-1]))
        k = np.arange(int(n_samp.sum())) - starts[seg_id]
        # A mintadarab KÖZÉPPONTJA a szakasz mentén: (k + 0.5) / n
        t = (k + 0.5) / n_samp[seg_id]

        px, py, pz = x[seg_parent], y[seg_parent], z[seg_parent]
        dx, dy, dz = x[seg_child] - px, y[seg_child] - py, z[seg_child] - pz
        s_x = px[seg_id] + t * dx[seg_id]
        s_y = py[seg_id] + t * dy[seg_id]
        s_z = pz[seg_id] + t * dz[seg_id]

        s_vx = np.clip(np.round(s_x / VOXEL_SIZE).astype(int), 0, max_x - 1)
        s_vy = np.clip(np.round(s_y / VOXEL_SIZE).astype(int), 0, max_y - 1)
        s_vz = np.clip(np.round(s_z / VOXEL_SIZE).astype(int), 0, max_z - 1)
        samp_region = atlas_matrix[s_vx, s_vy, s_vz].astype(int)
        samp_len = seg_len[seg_id] / n_samp[seg_id]

        # Régiónkénti hossz-összeg (a negatív/hibás ID-kat kiszűrjük)
        ok = samp_region >= 0
        length_by_region = np.bincount(samp_region[ok], weights=samp_len[ok])
    else:
        length_by_region = np.zeros(1, dtype=float)

    def _axon_length_in(match_ids: np.ndarray) -> float:
        """A megadott atlasz-ID-khez tartozó, határokon felosztott axonhossz."""
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

        # HATÁRSEJT-JELZŐ: a régióbesorolás 25 um-es voxelrácson történik, ezért a
        # határ közelében lévő somák besorolása bizonytalan (Nóra "S1 határon lévő
        # sejtek" megfigyelése). Megnézzük a soma körüli 3x3x3 voxelt: ha nem
        # mindegyik ugyanabba a régióba esik, a sejt határon van.
        sx0, sy0, sz0 = int(vox_x[soma_idx]), int(vox_y[soma_idx]), int(vox_z[soma_idx])
        nb = atlas_matrix[max(0, sx0 - 1):sx0 + 2,
                          max(0, sy0 - 1):sy0 + 2,
                          max(0, sz0 - 1):sz0 + 2].ravel()
        if len(nb) > 0:
            soma_border_fraction = float(np.mean(nb != soma_region_id))
    else:
        soma_idx = None
        soma_region_id = -1
        soma_name = "No soma found"
        soma_coords = (0.0, 0.0, 0.0)

    # -------------------------------------------------------------------------
    # OLDALISÁG (félteke)
    # -------------------------------------------------------------------------
    # Az atlaszban mindkét félteke UGYANAZT a régió-ID-t viseli, ezért a régió-ID
    # önmagában nem árulja el az oldalt. A középvonal viszont fix koordináta: a
    # medio-laterális tengely (MIDLINE_AXIS) felénél húzódik. Minden pontról így
    # eldönthető, hogy a soma oldalán van-e.
    vox_axes = (vox_x, vox_y, vox_z)
    ml_vox = vox_axes[MIDLINE_AXIS]
    midline = atlas_matrix.shape[MIDLINE_AXIS] / 2.0
    # +1 / -1 a középvonal két oldalán (a pontosan a vonalon lévő 0-t kap)
    point_side = np.sign(ml_vox.astype(float) - midline).astype(int)
    soma_side = int(point_side[soma_idx]) if soma_idx is not None else 0

    ep_side = point_side[ep_idx]
    branch_side = point_side[branch_idx]

    def _side_mask(sides: np.ndarray) -> np.ndarray:
        """Melyik pontok számítanak bele a kért oldaliság szerint."""
        # Ha nincs soma (vagy pont a középvonalon ül), nem tudjuk mihez viszonyítani,
        # ezért ilyenkor nem szűrünk oldalra - különben némán nullázódna a sejt.
        if laterality == 'both' or soma_side == 0:
            return np.ones(len(sides), dtype=bool)
        if laterality == 'ipsi':
            return sides == soma_side
        return (sides == -soma_side) & (sides != 0)

    ep_keep = _side_mask(ep_side)
    branch_keep = _side_mask(branch_side)

    # A sejt ÖSSZES axon-végpontja - ez a méret-független (%-os) szűrés nevezője.
    total_endpoint_count = int(len(ep_idx))
    region_descendants = region_descendants or {}
    region_names = region_names or {}
    criteria_per_region = criteria_per_region or {}
    default_criteria = FilterCriteria()

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
        ep_in = np.isin(ep_regions, match)
        br_in = np.isin(branch_regions, match)

        # A kért oldaliság szerinti (a vetítést eldöntő) számok...
        ep_count = int((ep_in & ep_keep).sum())
        br_count = int((br_in & branch_keep).sum())
        proj_count = ep_count + br_count

        # ...és MINDIG a teljes ipszi/kontra bontás is, hogy látható legyen,
        # mekkora a kontralaterális hozzájárulás.
        if soma_side != 0:
            ep_ipsi = int((ep_in & (ep_side == soma_side)).sum())
            ep_contra = int((ep_in & (ep_side == -soma_side)).sum())
            br_ipsi = int((br_in & (branch_side == soma_side)).sum())
            br_contra = int((br_in & (branch_side == -soma_side)).sum())
        else:
            ep_ipsi = ep_contra = br_ipsi = br_contra = 0

        # Határokon felosztott hossz (lásd length_by_region fent).
        axon_len = _axon_length_in(match)

        fraction = (ep_count / total_endpoint_count) if total_endpoint_count > 0 else 0.0

        return RegionResult(
            region_id=int(region_id), region_name=region_name,
            # A régió SAJÁT kritériuma dönt (alapból: végpont ÉS elágazás) - így az
            # áthaladó axonok nem számítanak hamis vetítésnek, és a pipa pontosan
            # ugyanazt mutatja, amit a szűrő is használ.
            projects_here=criteria_per_region.get(
                int(region_id), default_criteria
            ).is_projection(ep_count, br_count, axon_len, fraction),
            endpoint_count=ep_count, branch_point_count=br_count,
            projection_point_count=proj_count, axon_length_um=axon_len,
            endpoint_fraction=fraction,
            endpoint_count_ipsi=ep_ipsi, endpoint_count_contra=ep_contra,
            branch_point_count_ipsi=br_ipsi, branch_point_count_contra=br_contra,
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
        total_axon_length_um=total_axon_length, coords=coords,
        soma_border_fraction=soma_border_fraction,
        total_endpoint_count=total_endpoint_count,
        laterality=laterality,
        annotated_endpoint_count=int(np.sum(ep_regions > 0)),
    )


# SZŰRÉS KOMPLEX LOGIKÁVAL

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


# EXPORTÁLÁS

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
            'cell': cell_name,
            'soma_region': result.soma_region_name,
            'total_axon_length_um': round(result.total_axon_length_um, 1),
            'passes_filter': result.passes_filter,
            # Határsejt-jelző: a 25 um-es voxelrács miatt bizonytalan besorolás.
            'soma_on_region_border': result.soma_is_border,
            'soma_border_fraction': round(result.soma_border_fraction, 2),
            # A végpont-arány nevezőjének átláthatósága: hány végpont esik
            # egyáltalán annotált agyterületre.
            'endpoints_total': result.total_endpoint_count,
            'endpoints_in_annotated_regions': result.annotated_endpoint_count,
        }
        for tr in result.target_results:
            # A régió ID-t is beletesszük, mert a 30 karakteres csonkolás miatt két
            # hasonló nevű régió oszlopai egymásra íródhattak volna (néma adatvesztés).
            safe_col = f"{tr.region_name.replace(' ', '_').lower()[:30]}_{tr.region_id}"
            row[f'{safe_col}_projects'] = tr.projects_here
            row[f'{safe_col}_endpoints'] = tr.endpoint_count
            row[f'{safe_col}_branches'] = tr.branch_point_count
            row[f'{safe_col}_axon_um'] = round(tr.axon_length_um, 1)
            # Végpont-arány %-ban - ez alapján azonosíthatók a L6 sejtek
            # (pl. thalamus-arány > 2.5%).
            row[f'{safe_col}_endpoint_pct'] = round(tr.endpoint_fraction * 100, 2)
            # Oldaliság-bontás: mindig látszik, mennyi jön a túloldalról.
            row[f'{safe_col}_endpoints_ipsi'] = tr.endpoint_count_ipsi
            row[f'{safe_col}_endpoints_contra'] = tr.endpoint_count_contra
            # Önmagát dokumentáló oszlop: milyen kritériummal készült a döntés.
            crit = criteria_per_region.get(tr.region_id)
            if crit is not None:
                row[f'{safe_col}_criterion'] = crit.describe()
        rows.append(row)
    return pd.DataFrame(rows)


# KÉRGI VETÍTÉSI ÖSSZESÍTŐ (a Nóra által kért végleges táblázatok)
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


def _only_projects_to(result: CellAnalysisResult, region_id: int,
                      all_region_ids: list[int]) -> bool:
    """
    A sejt a célterületek közül KIZÁRÓLAG ebbe vetít.

    Ez felel meg Nóra eredeti három kategóriájának ("GPe + BS, de a TRN-be nem").
    Az inkluzív számok ettől eltérnek: azokban a kettősen vetítők is benne vannak.
    A megkülönböztetés fontos - épp ezt hiányolta a július 2-i levelében
    ("nem vettem figyelembe, hogy a kettőseket nem vetted be az 1x-es vetítésekhez").
    """
    if not _projects_to(result, region_id):
        return False
    return not any(_projects_to(result, other) for other in all_region_ids
                   if int(other) != int(region_id))


def build_cortical_summary(
        results: list[tuple[str, CellAnalysisResult]],
        base_region_id: int | None,
        numerator_region_ids: list[int],
        region_label_fn,
        criteria_per_region: dict[int, 'FilterCriteria'] | None = None,
) -> dict:
    """
    Kérgi régiónkénti összesítők a Nóra-féle definíciók szerint.

    base_region_id: a "100%" populációt definiáló régió (pl. leszálló agytörzs =
        PT sejtek). Ha None, akkor a nevező az ÖSSZES L5 sejt.
    numerator_region_ids: a célterületek (pl. GPe, TRN), amelyekre a %-ot adjuk.
    region_label_fn: régió_id -> megjelenítendő név.

    criteria_per_region: csak dokumentálásra - a projects_here értékeket már a
        run_analysis kiszámolta ezekkel a kritériumokkal. A leírás bekerül a
        visszaadott 'criteria_note' / 'slug' kulcsokba, hogy az export önmagát
        dokumentálja.

    Visszatér: {'benne', 'nelkul', 'axon', 'categories', 'criteria_note', 'slug'}.
    """
    from collections import defaultdict

    criteria_per_region = criteria_per_region or {}

    # A soma nélküli (vagy nem azonosított régiójú) sejteket KIHAGYJUK: nincs
    # kérgi régiójuk, amihez hozzá lehetne rendelni őket, ezért korábban saját
    # "No soma found" sort kaptak az összesítőkben, ami félrevezető volt.
    groups: dict[str, list] = defaultdict(list)
    skipped_no_soma = 0
    for name, r in results:
        if r.soma_region_id is None or r.soma_region_id <= 0:
            skipped_no_soma += 1
            continue
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
    cat_only_rows: dict[str, list] = {lab: [] for lab in num_labels}
    cat_all_rows: list = []

    for soma, cells in sorted(groups.items()):
        total = len(cells)
        base_cells = [(n, r) for (n, r) in cells if is_base(r)]
        nbase = len(base_cells)

        row_b = {"Soma Region": soma, base_col: nbase}
        row_n = {"Soma Region": soma, "Total L5 Cells": total}
        row_a = {"Soma Region": soma, "PT Cells": nbase}

        for rid, lab in zip(numerator_region_ids, num_labels):
            # INKLUZÍV: ide vetít (akár máshova is). Ez felel meg Nóra július 3-i
            # meghatározásának: "az 50 a 100% és a TRN-be vetítők aránya 50%".
            cb = sum(1 for (_, r) in base_cells if _projects_to(r, rid))
            row_b[f"{lab} n"] = cb
            row_b[f"{lab} %"] = round(100 * cb / nbase, 1) if nbase else 0.0

            # EXKLUZÍV: CSAK ide vetít a célterületek közül. Ez felel meg a
            # korábbi három fájlnak ("GPe + BS, de a TRN-be nem"). A kettő nem
            # ugyanaz, ezért mindkettőt kiírjuk - így nem lehet összekeverni.
            if len(numerator_region_ids) > 1:
                cb_only = sum(1 for (_, r) in base_cells if _only_projects_to(r, rid, numerator_region_ids))
                row_b[f"{lab} only n"] = cb_only
                row_b[f"{lab} only %"] = round(100 * cb_only / nbase, 1) if nbase else 0.0

            cn = sum(1 for (_, r) in cells if _projects_to(r, rid))
            row_n[f"{lab} n"] = cn
            row_n[f"{lab} %"] = round(100 * cn / total, 1) if total else 0.0
            if len(numerator_region_ids) > 1:
                cn_only = sum(1 for (_, r) in cells if _only_projects_to(r, rid, numerator_region_ids))
                row_n[f"{lab} only n"] = cn_only
                row_n[f"{lab} only %"] = round(100 * cn_only / total, 1) if total else 0.0

            lens = [_region_of(r, rid).axon_length_um for (_, r) in base_cells if _projects_to(r, rid)]
            row_a[f"{lab} mean axon µm"] = round(sum(lens) / len(lens), 1) if lens else 0.0

            ids = sorted(_cell_serial(n) for (n, r) in base_cells if _projects_to(r, rid))
            cat_rows[lab].append({
                "Soma Region": soma, base_col: nbase,
                f"{lab} Projects": len(ids),
                f"{lab} % of PT": round(100 * len(ids) / nbase, 1) if nbase else 0.0,
                "Projecting Cell IDs": ", ".join(ids),
            })

            if len(numerator_region_ids) > 1:
                ids_only = sorted(_cell_serial(n) for (n, r) in base_cells
                                  if _only_projects_to(r, rid, numerator_region_ids))
                cat_only_rows[lab].append({
                    "Soma Region": soma, base_col: nbase,
                    f"{lab} only Projects": len(ids_only),
                    f"{lab} only % of PT": round(100 * len(ids_only) / nbase, 1) if nbase else 0.0,
                    "Projecting Cell IDs": ", ".join(ids_only),
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
        # Nóra eredeti három kategóriája: kizárólagos bontás + a "mindegyikbe".
        for lab in num_labels:
            categories[f"{lab} only"] = pd.DataFrame(cat_only_rows[lab]).sort_values(
                f"{lab} only Projects", ascending=False)
        categories["All targets"] = pd.DataFrame(cat_all_rows).sort_values("All targets Projects", ascending=False)

    # Az exportok önmagukat dokumentálják: melyik régió milyen kritériummal ment.
    involved = ([base_region_id] if base_region_id is not None else []) + list(numerator_region_ids)
    used = [criteria_per_region.get(rid, FilterCriteria()) for rid in involved]
    uniform = all(
        (c.min_endpoints, c.min_branch_points, c.min_axon_length_um, c.min_endpoint_fraction) ==
        (used[0].min_endpoints, used[0].min_branch_points,
         used[0].min_axon_length_um, used[0].min_endpoint_fraction)
        for c in used
    ) if used else True

    if uniform and used:
        criteria_note = used[0].describe()
        slug = used[0].slug()
    else:
        criteria_note = " · ".join(
            f"{region_label_fn(rid)}: {criteria_per_region.get(rid, FilterCriteria()).describe()}"
            for rid in involved
        )
        slug = "mixed"

    return {"benne": benne, "nelkul": nelkul, "axon": axon,
            "categories": categories, "criteria_note": criteria_note, "slug": slug,
            "skipped_no_soma": skipped_no_soma}