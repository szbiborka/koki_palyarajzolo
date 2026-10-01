# =============================================================================
# REGRESSZIÓS TESZTEK az analízis logikára.
# Külső adat (atlas .nrrd, SWC fájlok) NÉLKÜL futnak: szintetikus mini-atlaszt
# és néhány kézzel megírt SWC sejtet használnak.
#
# Futtatás a projektgyökérből:  python -m pytest tests/
# =============================================================================
import tracemalloc

import numpy as np
import pandas as pd

from config import CONTRA_CROSSING_MIN_AXON_UM, BRAINSTEM_MOTOR_ID
from core.analysis import (
    run_analysis, apply_filter, FilterCriteria, RegionResult, CellAnalysisResult,
    build_laterality_summary, build_cortical_summary, results_to_dataframe,
    LATERALITY_CLASS_LABELS, category_slugs,
)
from core.loader import build_region_descendants, region_name_map

CORTEX, THAL, TRN, GPE = 100, 549, 262, 1022
NAMES = {CORTEX: "Cortex", THAL: "Thalamus", TRN: "TRN", GPE: "GPe"}


def _region_for_x(i: int) -> int:
    if i <= 2:
        return CORTEX
    if i in (3, 4):
        return THAL
    if i == 5:
        return TRN
    return GPE  # i >= 6


def _atlas() -> np.ndarray:
    atlas = np.zeros((10, 4, 4), dtype=int)
    for i in range(10):
        atlas[i, :, :] = _region_for_x(i)
    return atlas


def _swc(rows) -> pd.DataFrame:
    """rows: iterable of (id, type, x_index, parent_id). y,z fixed inside a voxel."""
    data = [[nid, t, xi * 25, 25, 25, 1.0, pid] for (nid, t, xi, pid) in rows]
    return pd.DataFrame(data, columns=["id", "type", "x", "y", "z", "radius", "pid"])


def _xyz_swc(rows) -> pd.DataFrame:
    """rows: iterable of (id, type, x_vox, y_vox, z_vox, parent_id), voxel koordinátákban."""
    data = [[nid, t, xv * 25, yv * 25, zv * 25, 1.0, pid] for (nid, t, xv, yv, zv, pid) in rows]
    return pd.DataFrame(data, columns=["id", "type", "x", "y", "z", "radius", "pid"])


def _region_result(rid, name, projects, axon=100.0):
    return RegionResult(region_id=rid, region_name=name, projects_here=projects,
                        endpoint_count=1 if projects else 0, branch_point_count=1 if projects else 0,
                        axon_length_um=axon if projects else 0.0)


# ---------------------------------------------------------------------------
# PITFALL #1 — az áthaladó axon nem lehet hamis vetítés.
# Az axon a TRN-ben csak elágazik (branch point, végpont NÉLKÜL), majd a GPe-ben
# ad valódi arborizációt. Vetítéshez végpont ÉS elágazás is kell.
# ---------------------------------------------------------------------------
def test_passing_axon_is_not_a_projection():
    cell = _swc([
        (1, 1, 1, -1),  # soma (cortex)
        (2, 2, 2, 1),   # axon (cortex)
        (3, 2, 5, 2),   # axon (TRN)  -> 2 gyerek => elágazás a TRN-ben
        (4, 2, 6, 3),   # axon (GPe)  -> 2 gyerek => elágazás a GPe-ben
        (5, 2, 6, 3),   # axon (GPe)  -> végpont
        (6, 2, 7, 4),   # axon (GPe)  -> végpont
        (7, 2, 7, 4),   # axon (GPe)  -> végpont
    ])
    res = run_analysis(cell, _atlas(), NAMES, [TRN, GPE])
    by = {tr.region_name: tr for tr in res.target_results}

    assert by["TRN"].branch_point_count == 1
    assert by["TRN"].endpoint_count == 0
    assert by["TRN"].projects_here is False   # csak áthalad -> NEM vetítés
    assert by["GPe"].projects_here is True     # valódi arborizáció


# ---------------------------------------------------------------------------
# PITFALL #2 — a végpont-arány (%) helyes, és a L6 (NOT thalamus) szűrő monoton:
# egy KIZÁRÓ szűrő hozzáadása egyetlen régió sejtszámát sem növelheti.
# ---------------------------------------------------------------------------
def _l6_cell():
    # Minden végpont a thalamusban -> tipikus L6.
    return _swc([
        (1, 1, 1, -1),  # soma (cortex)
        (2, 2, 3, 1),   # thalamus
        (3, 2, 4, 2),   # thalamus -> 2 gyerek => elágazás
        (4, 2, 4, 3),   # thalamus végpont
        (5, 2, 4, 3),   # thalamus végpont
    ])


def _l5_cell():
    # Valódi GPe arborizáció, thalamuszban nincs végpont -> L5 PT.
    return _swc([
        (1, 1, 1, -1),  # soma (cortex)
        (2, 2, 4, 1),   # thalamus (áthaladás)
        (3, 2, 6, 2),   # GPe -> 2 gyerek => elágazás
        (4, 2, 7, 3),   # GPe végpont
        (5, 2, 7, 3),   # GPe végpont
    ])


def _mixed_cell():
    # A végpontok 1/3-a a thalamusban (elágazással), 2/3-a a GPe-ben.
    return _swc([
        (1, 1, 1, -1),  # soma (cortex)
        (2, 2, 4, 1),   # thalamus -> 2 gyerek => elágazás
        (3, 2, 4, 2),   # thalamus végpont
        (4, 2, 6, 2),   # GPe -> 2 gyerek => elágazás
        (5, 2, 7, 4),   # GPe végpont
        (6, 2, 7, 4),   # GPe végpont
    ])


def test_endpoint_fraction_identifies_l6():
    res = run_analysis(_l6_cell(), _atlas(), NAMES, [THAL, GPE])
    thal = next(t for t in res.target_results if t.region_name == "Thalamus")
    assert abs(thal.endpoint_fraction - 1.0) < 1e-9   # minden végpont thalamikus

    res5 = run_analysis(_l5_cell(), _atlas(), NAMES, [THAL, GPE])
    thal5 = next(t for t in res5.target_results if t.region_name == "Thalamus")
    assert thal5.endpoint_fraction == 0.0


def test_l6_filter_is_monotonic():
    atlas = _atlas()
    population = [_l5_cell(), _l6_cell(), _mixed_cell()]

    def passing(criteria):
        return [apply_filter(run_analysis(cell, atlas, NAMES, [THAL, GPE], criteria_per_region=criteria),
                             criteria).passes_filter for cell in population]

    base = {GPE: FilterCriteria(operator="AND")}
    with_l6 = {**base, THAL: FilterCriteria(min_endpoint_fraction=0.025, operator="NOT")}

    without, filtered = passing(base), passing(with_l6)
    assert sum(filtered) <= sum(without), "L6 kizárása nem növelheti a sejtszámot"
    assert filtered[0] is True    # L5 bennmarad
    assert filtered[1] is False   # L6 kiesik (nincs GPe vetítése sem)


def test_endpoint_share_threshold_is_really_applied():
    """A %-küszöb TÉNYLEG számít: 33% thalamikus végpont 25%-os küszöbnél kizárt, 50%-osnál nem."""
    atlas = _atlas()

    def passes(threshold):
        crit = {GPE: FilterCriteria(operator="AND"),
                THAL: FilterCriteria(min_endpoint_fraction=threshold, operator="NOT")}
        return apply_filter(run_analysis(_mixed_cell(), atlas, NAMES, [THAL, GPE], criteria_per_region=crit),
                            crit).passes_filter

    assert passes(0.25) is False
    assert passes(0.50) is True


def test_meets_thresholds_uses_its_own_criteria():
    """A szűrő a SAJÁT küszöbeivel dönt, nem egy máshol kiszámolt pipát olvas vissza."""
    tr = RegionResult(THAL, "Thalamus", projects_here=True, endpoint_count=2, branch_point_count=1,
                      axon_length_um=50.0, endpoint_fraction=0.01)
    assert FilterCriteria().meets_thresholds(tr) is True
    assert FilterCriteria(min_endpoint_fraction=0.025).meets_thresholds(tr) is False
    assert FilterCriteria(min_axon_length_um=100.0).meets_thresholds(tr) is False


# ---------------------------------------------------------------------------
# PARENT REGION — a szülő-régió (pl. Brain stem) az összes leszármazott magot
# lefedi. Az annotációs térfogat csak a leveleket címkézi, a szülő ID önmagában
# 0 voxel. A structure_id_path alapján kell feloldani.
# ---------------------------------------------------------------------------
BS_PARENT, BS_LEAF_A, BS_LEAF_B = 343, 7710, 7720


def _atlas_with_bs() -> np.ndarray:
    # x-index 8,9 kapja a brainstem LEVÉL magokat (a szülő 343 sehol nincs)
    atlas = _atlas()
    atlas[8, :, :] = BS_LEAF_A
    atlas[9, :, :] = BS_LEAF_B
    return atlas


def _dictionary_with_hierarchy() -> pd.DataFrame:
    return pd.DataFrame({
        "id": [CORTEX, THAL, TRN, GPE, BS_PARENT, BS_LEAF_A, BS_LEAF_B],
        "safe_name": ["Cortex", "Thalamus", "TRN", "GPe", "Brain stem", "BS leaf A", "BS leaf B"],
        "structure_id_path": [
            "/997/315/",           # cortex
            "/997/549/",           # thalamus
            "/997/549/262/",       # TRN
            "/997/1022/",          # GPe
            "/997/343/",           # Brain stem (parent)
            "/997/343/7710/",      # leaf under brain stem
            "/997/343/7720/",      # leaf under brain stem
        ],
    })


def test_parent_region_matches_descendants():
    atlas = _atlas_with_bs()
    dic = _dictionary_with_hierarchy()
    names = region_name_map(dic)

    desc = build_region_descendants(dic, [BS_PARENT])
    assert BS_LEAF_A in desc[BS_PARENT] and BS_LEAF_B in desc[BS_PARENT]

    # Egy sejt, ami a brainstem LEVÉL magban arborizál (elágazás + végpont).
    cell = _swc([
        (1, 1, 1, -1),   # soma cortex
        (2, 2, 7, 1),    # axon (GPe felé haladva)
        (3, 2, 8, 2),    # BS leaf A -> 2 gyerek => elágazás
        (4, 2, 9, 3),    # BS leaf B -> végpont
        (5, 2, 9, 3),    # BS leaf B -> végpont
    ])

    # Feloldás NÉLKÜL: a 343 nem fog semmit -> nincs BS vetítés.
    assert run_analysis(cell, atlas, names, [BS_PARENT]).target_results[0].projects_here is False

    # Feloldással: a brainstem valódi vetítésként jelenik meg.
    bs = run_analysis(cell, atlas, names, [BS_PARENT], desc).target_results[0]
    assert bs.projects_here is True
    assert bs.endpoint_count == 2 and bs.branch_point_count == 1


# ---------------------------------------------------------------------------
# "LESZÁLLÓ AGYTÖRZS" (Midbrain+Hindbrain) - a thalamust KIZÁRVA, mert az Allen
# ontológiában a "Brain stem" (343) tartalmazza a köztiagyat/thalamust.
# ---------------------------------------------------------------------------
def _brainstem_dictionary() -> pd.DataFrame:
    # Allen-szerű hierarchia: 343 Brain stem > 1129 Interbrain > 549 Thalamus,
    # illetve 343 > 313 Midbrain és 343 > 1065 Hindbrain.
    return pd.DataFrame({
        "id": [100, 549, 5491, 313, 3131, 1065, 10651],
        "safe_name": ["Cortex", "Thalamus", "Thal leaf", "Midbrain", "MB leaf", "Hindbrain", "HB leaf"],
        "structure_id_path": [
            "/997/315/100/",
            "/997/343/1129/549/",
            "/997/343/1129/549/5491/",
            "/997/343/313/",
            "/997/343/313/3131/",
            "/997/343/1065/",
            "/997/343/1065/10651/",
        ],
    })


def test_descending_brainstem_excludes_thalamus():
    dic = _brainstem_dictionary()
    names = region_name_map(dic)

    # A teljes Allen "Brain stem" (343) MAGÁBA foglalja a thalamust (a hiba forrása).
    full = build_region_descendants(dic, [343])
    assert 549 in full[343] and 5491 in full[343]

    # A virtuális "leszálló agytörzs" csak a közép- és utóagyat tartalmazza.
    desc = build_region_descendants(dic, [BRAINSTEM_MOTOR_ID])[BRAINSTEM_MOTOR_ID]
    assert 3131 in desc and 10651 in desc
    assert 549 not in desc and 5491 not in desc

    # Egy csak thalamusba arborizáló (L6-szerű) sejt: a teljes Brain stem
    # vetítésnek látja, a leszálló agytörzs viszont NEM.
    atlas = np.zeros((10, 4, 4), dtype=int)
    atlas[3:5, :, :] = 5491   # thalamus leaf
    atlas[6, :, :] = 3131     # midbrain leaf (a sejt ide nem megy)
    cell = _swc([
        (1, 1, 0, -1),   # soma (annotálatlan régió)
        (2, 2, 3, 1),    # thalamus -> elágazás
        (3, 2, 4, 2),    # thalamus végpont
        (4, 2, 4, 2),    # thalamus végpont
    ])

    r_full = run_analysis(cell, atlas, names, [343], full)
    assert r_full.target_results[0].projects_here is True   # thalamus == "brain stem"

    r_desc = run_analysis(cell, atlas, names, [BRAINSTEM_MOTOR_ID],
                          build_region_descendants(dic, [BRAINSTEM_MOTOR_ID]))
    assert r_desc.target_results[0].projects_here is False  # thalamus kizárva
    assert r_desc.target_results[0].region_name == names[BRAINSTEM_MOTOR_ID]


# ---------------------------------------------------------------------------
# CORTICAL SUMMARY — a végleges bs_benne / bs_nelkul táblák helyes nevezővel.
# ---------------------------------------------------------------------------
def _cell(soma, bs, gpe, trn):
    return CellAnalysisResult(
        soma_region_id=1, soma_region_name=soma, soma_coords=(0, 0, 0),
        target_results=[_region_result(343, 'BS', bs),
                        _region_result(GPE, 'GPe', gpe),
                        _region_result(TRN, 'TRN', trn)],
        other_projection_regions=[], total_axon_length_um=1000.0)


def test_cortical_summary_denominator():
    # M régió: A=BS+GPe+TRN, B=BS+GPe, C=BS, D=GPe(nem PT), E=semmi
    results = [
        ("A.swc", _cell("M", True, True, True)),
        ("B.swc", _cell("M", True, True, False)),
        ("C.swc", _cell("M", True, False, False)),
        ("D.swc", _cell("M", False, True, False)),
        ("E.swc", _cell("M", False, False, False)),
    ]
    label = {343: 'BS', GPE: 'GPe', TRN: 'TRN'}.get
    s = build_cortical_summary(results, base_region_id=343, numerator_region_ids=[GPE, TRN], region_label_fn=label)

    be = s['benne'].iloc[0]
    assert be['PT Cells (BS=100%)'] == 3               # A,B,C project to brain stem
    assert be['GPe n'] == 2 and be['GPe %'] == 66.7    # A,B  -> 2/3
    assert be['TRN n'] == 1 and be['TRN %'] == 33.3    # A    -> 1/3
    assert be['All targets n'] == 1                    # A

    ne = s['nelkul'].iloc[0]
    assert ne['Total L5 Cells'] == 5
    assert ne['GPe n'] == 3 and ne['GPe %'] == 60.0    # A,B,D over all 5
    assert ne['TRN n'] == 1 and ne['TRN %'] == 20.0

    gpe_cat = s['categories']['GPe'].iloc[0]
    assert gpe_cat['GPe Projects'] == 2
    assert gpe_cat['Projecting Cell IDs'] == "A, B"


def test_cortical_summary_slug_records_all_thresholds():
    results = [("A.swc", _cell("M", True, True, True))]
    crit = FilterCriteria(min_axon_length_um=100.0, min_endpoint_fraction=0.025)
    s = build_cortical_summary(results, 343, [GPE], {343: 'BS', GPE: 'GPe'}.get,
                               criteria_per_region={343: crit, GPE: crit}, laterality='ipsi')
    assert s['slug'] == "ep1_br1_len100_frac2.5_ipsi"


def test_summary_excludes_cells_without_soma_region():
    """A soma nélküli sejtek ne kapjanak saját sort az összesítőkben."""
    good = CellAnalysisResult(GPE, 'M2', (0, 0, 0), [RegionResult(GPE, 'GPe', True, 3, 2, 50.0, 0.1)], [], 100.0)
    bad = CellAnalysisResult(-1, 'No soma found', (0, 0, 0),
                             [RegionResult(GPE, 'GPe', True, 3, 2, 50.0, 0.1)], [], 100.0)
    s = build_cortical_summary([("a.swc", good), ("b.swc", bad)], None, [GPE], {GPE: 'GPe'}.get)
    assert list(s['nelkul']['Soma Region']) == ['M2']
    assert s['skipped_no_soma_region'] == 1


def test_exclusive_categories_match_the_original_three_files():
    """Az eredeti bontás: 'GPe + BS, de a TRN-be nem'."""
    pop = ([(f"g{i}.swc", _cell('M2', True, True, False)) for i in range(4)] +
           [(f"t{i}.swc", _cell('M2', True, False, True)) for i in range(3)] +
           [(f"b{i}.swc", _cell('M2', True, True, True)) for i in range(2)] +
           [("n0.swc", _cell('M2', True, False, False))])
    s = build_cortical_summary(pop, 343, [GPE, TRN], {343: 'BS', GPE: 'GPe', TRN: 'TRN'}.get)
    r = s['benne'].iloc[0]

    assert r['GPe n'] == 6 and r['TRN n'] == 5            # inkluzív (a kettősökkel)
    assert r['GPe only n'] == 4 and r['TRN only n'] == 3  # kizárólagos
    assert r['All targets n'] == 2
    # a kizárólagos részek + kettősök + egyik sem = az összes PT sejt
    assert r['GPe only n'] + r['TRN only n'] + r['All targets n'] + 1 == r['PT Cells (BS=100%)']
    assert s['categories']['GPe only'].iloc[0]['Projecting Cell IDs'] == "g0, g1, g2, g3"


# ---------------------------------------------------------------------------
# A VETÍTÉS-DEFINÍCIÓ ÉS A SZŰRŐ MINDIG EGYETÉRT
# ---------------------------------------------------------------------------
def test_projection_definition_is_global_and_consistent():
    """Ha a kritériumot lazítjuk 'csak végpont'-ra, a pipa és a szűrő is követi."""
    atlas = _atlas()
    # Axon, ami a TRN-ben végződik, de ott NEM ágazik el (1 végpont, 0 elágazás)
    cell = _swc([
        (1, 1, 1, -1),   # soma (cortex)
        (2, 2, 2, 1),    # axon (cortex)
        (3, 2, 5, 2),    # axon (TRN) -> 1 gyerek, tehát nem elágazás
        (4, 2, 5, 3),    # axon (TRN) -> végpont
    ])

    strict_crit = {TRN: FilterCriteria(operator='AND')}
    strict = apply_filter(run_analysis(cell, atlas, NAMES, [TRN], criteria_per_region=strict_crit), strict_crit)
    t_strict = strict.target_results[0]
    assert t_strict.endpoint_count == 1 and t_strict.branch_point_count == 0
    assert t_strict.projects_here is False and strict.passes_filter is False

    loose_crit = {TRN: FilterCriteria(min_endpoints=1, min_branch_points=0, operator='AND')}
    loose = apply_filter(run_analysis(cell, atlas, NAMES, [TRN], criteria_per_region=loose_crit), loose_crit)
    assert loose.target_results[0].projects_here is True and loose.passes_filter is True


def test_slug_and_description():
    assert FilterCriteria(1, 0).slug() == "ep1_br0"
    assert FilterCriteria(1, 1, 100.0, 0.025).slug() == "ep1_br1_len100_frac2.5"
    assert "endpoint" in FilterCriteria(1, 0).describe()
    assert FilterCriteria(0, 0).describe() == "any axon presence"


def test_observe_only_region_does_not_filter():
    """Egy 'csak megfigyelés' régió hozzáadása miatt nem eshet ki egyetlen sejt sem."""
    atlas = _atlas()
    # GPe-ben arborizál, a thalamust meg sem érinti
    cell = _swc([(1, 1, 1, -1), (2, 2, 2, 1), (3, 2, 7, 2), (4, 2, 7, 3), (5, 2, 7, 3)])

    def passes(criteria, regions):
        return apply_filter(run_analysis(cell, atlas, NAMES, regions, criteria_per_region=criteria),
                            criteria).passes_filter

    assert passes({GPE: FilterCriteria(operator='AND')}, [GPE]) is True
    observed = {GPE: FilterCriteria(operator='AND'), THAL: FilterCriteria(operator='NONE')}
    assert passes(observed, [GPE, THAL]) is True, "a megfigyelt régió nem szűrhet"
    assert FilterCriteria(operator='NONE').is_active() is False

    # Kontroll: ugyanez 'Required (AND)'-del MÁR kiszűrné
    required = {GPE: FilterCriteria(operator='AND'), THAL: FilterCriteria(operator='AND')}
    assert passes(required, [GPE, THAL]) is False


def test_export_columns_unique_for_similar_region_names():
    """A 30 karakterre csonkolt oszlopnevek nem írhatják felül egymást."""
    n1 = "Brain stem descending Midbrain plus Hindbrain"
    n2 = "Brain stem descending Midbrain minus Thalamus"
    prefix = n1.replace(' ', '_').lower()[:30]
    assert prefix == n2.replace(' ', '_').lower()[:30]  # ütköző prefix

    res = CellAnalysisResult(1, 'M2', (0, 0, 0), [
        RegionResult(11, n1, True, 5, 4, 100.0, 0.1),
        RegionResult(22, n2, False, 0, 0, 0.0, 0.0),
    ], [], 1000.0)
    df = results_to_dataframe([("a.swc", res)])

    assert len([c for c in df.columns if c.endswith('_projects')]) == 2
    assert df.iloc[0][f"{prefix}_11_projects"]
    assert not df.iloc[0][f"{prefix}_22_projects"]


def test_3d_view_understands_parent_regions():
    """A szülő régió felszíne, markerei és az 'Axon-in-region' nézet is működjön."""
    from core.visualization import _region_mask, _expand_ids, _build_axon_trace, _allowed_regions

    atlas = _atlas_with_bs()
    dic = _dictionary_with_hierarchy()
    desc = build_region_descendants(dic, [BS_PARENT])

    # felszín: a szülő önmagában 0 voxel, feloldva viszont van
    assert (atlas == BS_PARENT).sum() == 0
    assert _region_mask(atlas, BS_PARENT, desc).sum() > 0

    cell = _swc([(1, 1, 1, -1), (2, 2, 7, 1), (3, 2, 8, 2), (4, 2, 9, 3), (5, 2, 9, 3)])
    res = run_analysis(cell, atlas, region_name_map(dic), [BS_PARENT], desc)
    co = res.coords
    pr = co['point_regions']

    # vetítési pont markerek: a levél-ID-kra is illeszkedjen
    match = np.fromiter(_expand_ids(BS_PARENT, desc), dtype=int)
    assert len(co['proj_idx'][np.isin(pr[co['proj_idx']], match)]) == 3

    # "Axon-in-region" nézet ne tüntesse el a szülő régió axonjait
    allowed = _allowed_regions([BS_PARENT], desc, res.soma_region_id)
    cmap = {rid: '#1f77b4' for rid in _expand_ids(BS_PARENT, desc)}
    traces = _build_axon_trace(co['x'], co['y'], co['z'], co['child_rows'], co['parent_rows'],
                               co['is_axon'], pr, cmap, 2, allowed)
    assert sum(len(t.x) for t in traces) > 0


# ---------------------------------------------------------------------------
# BETÖLTÉS
# ---------------------------------------------------------------------------
def test_soma_row_parser_accepts_float_type(tmp_path):
    """A '1.0' típusmező is somának számít (különben a sejt kiesne az indexből)."""
    from core.loader import _extract_soma_row

    for type_field in ('1', '1.0'):
        path = tmp_path / f"cell_{type_field}.swc"
        path.write_text(f"1 {type_field} 10 20 30 5 -1\n2 2 11 21 31 1 1\n")
        assert _extract_soma_row(str(path)) == (10.0, 20.0, 30.0), f"típusmező: {type_field}"


def test_load_swc_ignores_extra_columns(tmp_path):
    """Egy 8 oszlopos SWC sor nem csúsztathatja el az oszlopokat."""
    from core.loader import load_swc

    path = tmp_path / "extra.swc"
    path.write_text("# comment\n1 1 10 20 30 5 -1 0\n2 2 11 21 31 1 1 0\n")
    df = load_swc(str(path))
    assert list(df['id']) == [1, 2]
    assert list(df['type']) == [1, 2]
    assert list(df['pid']) == [-1, 1]


def test_swc_paths_use_forward_slashes(tmp_path):
    """A relatív útvonalak OS-függetlenek (a Windows alatt épült index Linuxon is illeszkedjen)."""
    from core.loader import get_all_swc_files

    (tmp_path / "212064").mkdir()
    (tmp_path / "212064" / "001.swc").write_text("1 1 0 0 0 1 -1\n")
    assert list(get_all_swc_files(str(tmp_path)).keys()) == ["212064/001.swc"]


# ---------------------------------------------------------------------------
# PONTOSÍTÁSOK: határon felosztott axonhossz, határsejt-jelző, nevező, memória
# ---------------------------------------------------------------------------
def test_axon_length_is_split_at_region_boundaries():
    """Egy határt átlépő szakasz hossza arányosan oszoljon meg a régiók között."""
    # node2 (cortex, x=50um) -> node3 (GPe, x=175um): egyetlen 125 um-es szakasz,
    # ami áthalad a thalamuson és a TRN-en is.
    cell = _swc([(1, 1, 1, -1), (2, 2, 2, 1), (3, 2, 7, 2), (4, 2, 7, 3), (5, 2, 7, 3)])
    res = run_analysis(cell, _atlas(), NAMES, [CORTEX, THAL, TRN, GPE])
    by = {t.region_name: t.axon_length_um for t in res.target_results}

    assert by["Thalamus"] > 0 and by["TRN"] > 0
    assert by["GPe"] < 125.0
    # a részek összege kiadja a teljes axonhosszt
    assert abs(sum(by.values()) - res.total_axon_length_um) < 1e-6


def test_huge_allen_ids_do_not_blow_up_memory():
    """Az Allen ID-k 6*10^8-ig mennek; a régiónkénti axonhossz nem foglalhat ID-méretű tömböt."""
    huge = 614454277
    atlas = _atlas()
    atlas[6:, :, :] = huge
    cell = _swc([(1, 1, 1, -1), (2, 2, 2, 1), (3, 2, 7, 2), (4, 2, 7, 3), (5, 2, 7, 3)])

    tracemalloc.start()
    res = run_analysis(cell, atlas, {**NAMES, huge: "Huge"}, [huge])
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak < 50e6, f"csúcsmemória: {peak / 1e6:.0f} MB"
    assert res.target_results[0].axon_length_um > 0
    assert res.target_results[0].projects_here is True


def test_soma_border_flag():
    """A régióhatáron ülő somát meg kell jelölni (bizonytalan besorolás)."""
    atlas = np.zeros((20, 20, 20), dtype=int)
    atlas[:10] = CORTEX
    atlas[10:] = GPE

    def cell_at(vox_x):
        return pd.DataFrame(
            [[1, 1, vox_x * 25, 250, 250, 1.0, -1],
             [2, 2, vox_x * 25 + 10, 250, 250, 1.0, 1]],
            columns=["id", "type", "x", "y", "z", "radius", "pid"])

    inside = run_analysis(cell_at(5), atlas, NAMES, [GPE])
    border = run_analysis(cell_at(9), atlas, NAMES, [GPE])
    assert inside.soma_is_border is False and inside.soma_border_fraction == 0.0
    assert border.soma_is_border is True and border.soma_border_fraction > 0.0


def test_endpoint_denominator_is_transparent():
    """Külön látszik az összes és az annotált régióba eső végpontok száma."""
    atlas = np.zeros((20, 20, 20), dtype=int)
    atlas[:5] = CORTEX  # a tér nagy része annotálatlan (0)
    cell = pd.DataFrame([
        [1, 1, 50, 250, 250, 1.0, -1],
        [2, 2, 60, 250, 250, 1.0, 1],
        [3, 2, 400, 250, 250, 1.0, 2],   # annotálatlan területen végződik
        [4, 2, 400, 260, 250, 1.0, 2],   # szintén
    ], columns=["id", "type", "x", "y", "z", "radius", "pid"])
    res = run_analysis(cell, atlas, NAMES, [CORTEX])
    assert res.total_endpoint_count == 2
    assert res.annotated_endpoint_count == 0


# ---------------------------------------------------------------------------
# OLDALISÁG
# ---------------------------------------------------------------------------
def _mirrored_atlas():
    """Mindkét féltekén UGYANAZ a régió-ID (mint az Allen atlaszban). Középvonal: z=20."""
    n = 40
    atlas = np.zeros((n, n, n), dtype=int)
    atlas[5:35, 5:35, :] = CORTEX
    atlas[15:25, 15:25, 4:10] = GPE     # bal oldali GPe
    atlas[15:25, 15:25, 30:36] = GPE    # jobb oldali GPe - azonos ID!
    return atlas


def _cell_with_arbor_at_z(z_end):
    return _xyz_swc([
        (1, 1, 20, 20, 8, -1),                            # soma: BAL oldal
        (2, 2, 20, 20, 8 + (z_end - 8) * 0.5, 1),
        (3, 2, 20, 20, z_end, 2),                         # elágazás
        (4, 2, 21, 20, z_end, 3),                         # végpont
        (5, 2, 19, 20, z_end, 3),                         # végpont
    ])


def _bilateral_cell():
    # Bal oldali soma; 2 végpont a bal GPe-ben, 1 a bal kéregben, 2 a jobb GPe-ben.
    return _xyz_swc([
        (1, 1, 20, 20, 8, -1),   # soma (bal)
        (2, 2, 20, 20, 7, 1),    # bal GPe -> elágazás (3, 4, 5, 9)
        (3, 2, 21, 20, 7, 2),    # bal GPe végpont
        (4, 2, 19, 20, 7, 2),    # bal GPe végpont
        (5, 2, 20, 20, 20, 2),   # középvonal
        (6, 2, 20, 20, 33, 5),   # jobb GPe -> elágazás
        (7, 2, 21, 20, 33, 6),   # jobb GPe végpont
        (8, 2, 19, 20, 33, 6),   # jobb GPe végpont
        (9, 2, 10, 20, 7, 2),    # bal kéreg végpont
    ])


def test_laterality_separates_ipsi_and_contra():
    atlas = _mirrored_atlas()
    names = {CORTEX: "Cortex", GPE: "GPe"}
    ipsi_cell = _cell_with_arbor_at_z(7)     # a soma oldalán
    contra_cell = _cell_with_arbor_at_z(33)  # a túloldalon

    # 'both': a két sejt megkülönböztethetetlen
    assert run_analysis(ipsi_cell, atlas, names, [GPE]).target_results[0].projects_here is True
    assert run_analysis(contra_cell, atlas, names, [GPE]).target_results[0].projects_here is True

    # 'ipsi' már szétválasztja őket
    assert run_analysis(ipsi_cell, atlas, names, [GPE], laterality='ipsi').target_results[0].projects_here is True
    assert run_analysis(contra_cell, atlas, names, [GPE], laterality='ipsi').target_results[0].projects_here is False

    # a bontás akkor is látszik, ha 'both' módban futunk
    t = run_analysis(contra_cell, atlas, names, [GPE]).target_results[0]
    assert t.endpoint_count_ipsi == 0 and t.endpoint_count_contra == 2


def test_ipsi_mode_ignores_contralateral_axon_length():
    """Ipsi módban a túloldali GPe axonhossza sem számít (és nem teljesítheti a hossz-küszöböt)."""
    atlas = _mirrored_atlas()
    names = {CORTEX: "Cortex", GPE: "GPe"}
    # A soma a bal kéregben, a bal GPe-n kívül; az axon a jobb GPe-ben arborizál.
    contra_cell = _xyz_swc([
        (1, 1, 10, 20, 8, -1),
        (2, 2, 20, 20, 20, 1),
        (3, 2, 20, 20, 33, 2),
        (4, 2, 21, 20, 33, 3),
        (5, 2, 19, 20, 33, 3),
    ])
    length_only = {GPE: FilterCriteria(min_endpoints=0, min_branch_points=0, min_axon_length_um=10.0)}

    both = run_analysis(contra_cell, atlas, names, [GPE], criteria_per_region=length_only).target_results[0]
    ipsi = run_analysis(contra_cell, atlas, names, [GPE], criteria_per_region=length_only,
                        laterality='ipsi').target_results[0]
    assert both.axon_length_um > 10.0 and both.projects_here is True
    assert ipsi.axon_length_um == 0.0 and ipsi.projects_here is False


def test_endpoint_fraction_denominator_follows_laterality():
    """A végpont-arány nevezője a kért oldal végpontjainak száma."""
    atlas = _mirrored_atlas()
    names = {CORTEX: "Cortex", GPE: "GPe"}
    cell = _bilateral_cell()

    def gpe(lat):
        return run_analysis(cell, atlas, names, [GPE], laterality=lat).target_results[0]

    assert gpe('both').endpoint_count == 4 and abs(gpe('both').endpoint_fraction - 4 / 5) < 1e-9
    assert gpe('ipsi').endpoint_count == 2 and abs(gpe('ipsi').endpoint_fraction - 2 / 3) < 1e-9
    assert gpe('contra').endpoint_count == 2 and abs(gpe('contra').endpoint_fraction - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# CÉLTERÜLET NÉLKÜLI FUTÁS — a féltekei kérdés az EGÉSZ sejtre vonatkozik,
# ezért célterület kijelölése nélkül is megválaszolható kell legyen.
# ---------------------------------------------------------------------------
def test_whole_cell_laterality_needs_no_target_region():
    atlas = _mirrored_atlas()
    names = {CORTEX: "Cortex", GPE: "GPe"}

    ipsi = run_analysis(_cell_with_arbor_at_z(7), atlas, names, [])
    contra = run_analysis(_cell_with_arbor_at_z(33), atlas, names, [])

    assert ipsi.target_results == [] and contra.target_results == []

    assert ipsi.laterality_class == 'ipsi_only'
    assert ipsi.endpoints_contra_total == 0 and ipsi.endpoints_ipsi_total == 2

    assert contra.laterality_class == 'contra'
    assert contra.endpoints_contra_total == 2 and contra.endpoints_ipsi_total == 0


# ---------------------------------------------------------------------------
# A középvonal ÁTLÉPÉSE és az ellenoldali VÉGZŐDÉS nem ugyanaz: egy csonkolt
# rekonstrukció átmehet a túloldalra és ott abbamaradhat. Külön kategória.
# ---------------------------------------------------------------------------
def test_crossing_without_endpoints_is_a_separate_category():
    atlas = _mirrored_atlas()
    names = {CORTEX: "Cortex", GPE: "GPe"}

    # soma bal oldalt (z=8), az axon átmegy jobbra (z=33), majd VISSZAJÖN és
    # bal oldalt végződik (z=7). Ellenoldali végpont tehát nincs.
    cell = _xyz_swc([(1, 1, 20, 20, 8, -1), (2, 2, 20, 20, 33, 1), (3, 2, 20, 20, 7, 2)])

    r = run_analysis(cell, atlas, names, [])
    assert r.endpoints_contra_total == 0
    assert r.crosses_midline is True
    assert r.laterality_class == 'crosses_only'
    assert r.axon_length_contra_um > CONTRA_CROSSING_MIN_AXON_UM
    # Az ipszi + kontra + középvonali hossz PONTOSAN kiadja a teljes axonhosszt.
    assert abs((r.axon_length_ipsi_um + r.axon_length_contra_um + r.axon_length_midline_um)
               - r.total_axon_length_um) < 1.0


# ---------------------------------------------------------------------------
# Soma nélküli sejtnél nincs mihez viszonyítani: saját kategória, és KIMARAD a
# százalékok nevezőjéből.
# ---------------------------------------------------------------------------
def test_undetermined_cells_are_excluded_from_laterality_percentages():
    atlas = _mirrored_atlas()
    names = {CORTEX: "Cortex", GPE: "GPe"}
    no_soma = _xyz_swc([(1, 2, 20, 20, 8, -1), (2, 2, 20, 20, 7, 1)])

    summary = build_laterality_summary([
        ("a.swc", run_analysis(_cell_with_arbor_at_z(7), atlas, names, [])),
        ("b.swc", run_analysis(_cell_with_arbor_at_z(33), atlas, names, [])),
        ("c.swc", run_analysis(no_soma, atlas, names, [])),
    ])

    assert summary['n_total'] == 3
    assert summary['n_decided'] == 2
    assert summary['counts'] == {'ipsi_only': 1, 'crosses_only': 0, 'contra': 1, 'unknown': 1}

    # A nevező az ELDÖNTHETŐ sejtek száma: 1/2 = 50%, nem 1/3 = 33.3%.
    overall = summary['overall'].set_index('Category')
    assert overall.loc[LATERALITY_CLASS_LABELS['ipsi_only'], '% of decided'] == 50.0
    assert pd.isna(overall.loc[LATERALITY_CLASS_LABELS['unknown'], '% of decided'])

    # A szintetikus soma a GPe voxelblokkban ül - itt a számok a lényeg.
    by_soma = summary['by_soma'].set_index('Soma region')
    assert by_soma.loc["GPe", 'Decided'] == 2
    assert by_soma.loc["GPe", 'No contralateral projection'] == 1
    assert by_soma.loc["GPe", 'No contralateral %'] == 50.0
    assert by_soma.loc["No soma found", 'Undetermined'] == 1
    assert pd.isna(by_soma.loc["No soma found", 'No contralateral %'])


# ---------------------------------------------------------------------------
# A kategória-slugok EGYEDIEK (az inkluzív és az exkluzív tábla nem kaphatja
# ugyanazt a fájlnevet).
# ---------------------------------------------------------------------------
def test_category_slugs_are_unique():
    labels = [
        "Globus pallidus external segment",
        "Globus pallidus external segment only",
        "Reticular nucleus of the thalamus",
        "Reticular nucleus of the thalamus only",
        "All targets",
    ]
    slugs = category_slugs(labels)

    assert len(set(slugs.values())) == len(labels)
    assert slugs["Globus pallidus external segment only"].endswith("_only")
    assert not slugs["Globus pallidus external segment"].endswith("_only")

    # Két KÜLÖNBÖZŐ régió, azonos első 24 karakterrel -> sorszámozás
    tricky = ["Primary somatosensory area barrel field", "Primary somatosensory area mouth"]
    assert len(set(category_slugs(tricky).values())) == 2
