# =============================================================================
# REGRESSZIÓS TESZTEK az analízis logikára.
# Külső adat (atlas .nrrd, SWC fájlok) NÉLKÜL futnak: szintetikus mini-atlaszt
# és néhány kézzel megírt SWC sejtet használnak.
#
# Futtatás:  python -m pytest tests/   VAGY közvetlenül:  python tests/test_analysis.py
# =============================================================================
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.analysis import run_analysis, apply_filter, FilterCriteria

CORTEX, THAL, TRN, GPE = 100, 549, 262, 1022


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


def _dictionary() -> pd.DataFrame:
    return pd.DataFrame({
        "id": [CORTEX, THAL, TRN, GPE],
        "safe_name": ["Cortex", "Thalamus", "TRN", "GPe"],
    })


def _swc(rows) -> pd.DataFrame:
    """rows: iterable of (id, type, x_index, parent_id). y,z fixed inside a voxel."""
    data = [[nid, t, xi * 25, 25, 25, 1.0, pid] for (nid, t, xi, pid) in rows]
    return pd.DataFrame(data, columns=["id", "type", "x", "y", "z", "radius", "pid"])


# ---------------------------------------------------------------------------
# PITFALL #1 — az áthaladó axon nem lehet hamis vetítés.
# Az axon a TRN-ben csak elágazik (branch point, végpont NÉLKÜL), majd a GPe-ben
# ad valódi arborizációt. Régen a TRN "vagy végpont vagy elágazás" alapján
# vetítésnek számított; most végpont ÉS elágazás is kell hozzá.
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
    res = run_analysis(cell, _atlas(), _dictionary(), [TRN, GPE])
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
    # Szinte minden végpont a thalamusban -> tipikus L6.
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


def test_endpoint_fraction_identifies_l6():
    res = run_analysis(_l6_cell(), _atlas(), _dictionary(), [THAL, GPE])
    thal = next(t for t in res.target_results if t.region_name == "Thalamus")
    assert abs(thal.endpoint_fraction - 1.0) < 1e-9   # minden végpont thalamikus

    res5 = run_analysis(_l5_cell(), _atlas(), _dictionary(), [THAL, GPE])
    thal5 = next(t for t in res5.target_results if t.region_name == "Thalamus")
    assert thal5.endpoint_fraction == 0.0


def test_l6_filter_is_monotonic():
    atlas, dic = _atlas(), _dictionary()
    population = [_l5_cell(), _l6_cell()]

    def gpe_pass_count(criteria):
        n = 0
        for cell in population:
            res = apply_filter(run_analysis(cell, atlas, dic, [THAL, GPE]), criteria)
            n += 1 if res.passes_filter else 0
        return n

    base = {GPE: FilterCriteria(operator="AND")}                       # vetítsen a GPe-be
    with_l6 = {**base,
               THAL: FilterCriteria(min_endpoint_fraction=0.025, operator="NOT")}  # + L6 kizárás

    without = gpe_pass_count(base)
    filtered = gpe_pass_count(with_l6)

    assert filtered <= without, "L6 kizárása nem növelheti a sejtszámot"
    # Konkrétan: az L5 bennmarad, az L6 kiesik.
    assert apply_filter(run_analysis(_l5_cell(), atlas, dic, [THAL, GPE]), with_l6).passes_filter is True
    assert apply_filter(run_analysis(_l6_cell(), atlas, dic, [THAL, GPE]), with_l6).passes_filter is False


# ---------------------------------------------------------------------------
# PARENT REGION — a szülő-régió (pl. Brain stem) az összes leszármazott magot
# lefedi. Az annotációs térfogat csak a leveleket címkézi, a szülő ID önmagában
# 0 voxel. A structure_id_path alapján kell feloldani.
# ---------------------------------------------------------------------------
BS_PARENT, BS_LEAF_A, BS_LEAF_B = 343, 7710, 7720


def _atlas_with_bs() -> np.ndarray:
    # x-index 8,9 kapja a brainstem LEVÉL magokat (a szülő 343 sehol nincs)
    atlas = np.zeros((10, 4, 4), dtype=int)
    for i in range(10):
        atlas[i, :, :] = _region_for_x(i)
    atlas[8, :, :] = BS_LEAF_A
    atlas[9, :, :] = BS_LEAF_B
    return atlas


def _dictionary_with_hierarchy() -> pd.DataFrame:
    return pd.DataFrame({
        "id": [CORTEX, THAL, TRN, GPE, BS_PARENT, BS_LEAF_A, BS_LEAF_B],
        "safe_name": ["Cortex", "Thalamus", "TRN", "GPe",
                      "Brain stem", "BS leaf A", "BS leaf B"],
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
    from core.loader import build_region_descendants

    atlas = _atlas_with_bs()
    dic = _dictionary_with_hierarchy()

    # A szülő 343 önmagában 0 voxel; a feloldásnak be kell húznia a leveleket.
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

    # Feloldás NÉLKÜL: a 343 nem fog semmit -> nincs BS vetítés (a régi hiba).
    res_no = run_analysis(cell, atlas, dic, [BS_PARENT])
    assert res_no.target_results[0].projects_here is False

    # Feloldással: a brainstem valódi vetítésként jelenik meg.
    res_yes = run_analysis(cell, atlas, dic, [BS_PARENT], desc)
    bs = res_yes.target_results[0]
    assert bs.projects_here is True
    assert bs.endpoint_count == 2 and bs.branch_point_count == 1


if __name__ == "__main__":
    test_passing_axon_is_not_a_projection()
    test_endpoint_fraction_identifies_l6()
    test_l6_filter_is_monotonic()
    test_parent_region_matches_descendants()
    print("All analysis regression tests passed.")
