# VIZUALIZÁCIÓ MODUL - 3D ábra generálása Plotly-val.
#
# Felépítés:
#   - a régiófelszínek (marching cubes) régiónként EGYSZER számolódnak, utána cache-ből jönnek;
#   - az axon vonalai vektorizáltan, NaN-elválasztással épülnek (nem szakaszonkénti ciklussal);
#   - nagy fáknál egyszerűsítés: minden N-edik csomópont + MINDEN elágazás és végpont
#     marad, mindegyik a legközelebbi megtartott ősével kötve össze - a vonal folytonos;
#   - minden elem jelmagyarázat-csoportban van: egy kattintás a régió felszínét és a
#     pontjait, illetve a sejt axonját és somáját együtt rejti el / mutatja.

import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from skimage.measure import marching_cubes

from config import (
    VOXEL_SIZE, VIZ_MARCHING_CUBES_STEP, VIZ_THEMES, DEFAULT_VIZ_THEME, VIZ_BRAIN_OUTLINE_STEP,
    VOXEL_LOOKUP, MIDLINE_AXIS, VIZ_DEFAULT_LINE_WIDTH, VIZ_DEFAULT_HEIGHT, VIZ_MAX_SEGMENTS,
    VIZ_MAX_SEGMENTS_COMBINED,
)
from core.analysis import CellAnalysisResult, side_of

# A Plotly eszköztár: görgetéses zoom, logó nélkül, és nagy felbontású PNG-mentés
# (a fényképező ikonnal) - dolgozatba, poszterre.
PLOTLY_CONFIG = {
    'displaylogo': False,
    'scrollZoom': True,
    'toImageButtonOptions': {'format': 'png', 'scale': 3, 'filename': 'palyakoveto_3d'},
}
LEGEND_HINT = 'click: show / hide · double-click: only this'

# Sejtek színezése a kombinált nézetben (az első az alapértelmezés)
CELL_COLOR_MODES = {
    'cell': 'Each cell (its own colour)',
    'class': 'Projection class (PT / IT / CT)',
    'soma': 'Soma region (top 7, the rest grey)',
}
_GOLDEN_ANGLE = 137.50776  # fok: az egymást követő árnyalatok a színkör egymástól legtávolabbi pontjaira esnek
_CLASS_ORDER = ['PT', 'IT', 'CT']   # rögzített sorrend: az osztály színe sosem függ a többitől


def get_theme(theme: str | dict | None = None) -> dict:
    """A kért 3D téma beállításai (alapértelmezés: DEFAULT_VIZ_THEME)."""
    if isinstance(theme, dict):
        return theme
    return VIZ_THEMES.get(theme or DEFAULT_VIZ_THEME, VIZ_THEMES[DEFAULT_VIZ_THEME])


def _oklch_to_hex(lightness: float, chroma: float, hue_deg: float) -> str:
    """OKLCH -> sRGB hex; ha a szín kilógna az sRGB-ből, a króma csökken (az árnyalat és a világosság marad)."""
    h = math.radians(hue_deg)
    for c in np.linspace(chroma, 0.0, 16):
        a, b = c * math.cos(h), c * math.sin(h)
        l_ = (lightness + 0.3963377774 * a + 0.2158037573 * b) ** 3
        m_ = (lightness - 0.1055613458 * a - 0.0638541728 * b) ** 3
        s_ = (lightness - 0.0894841775 * a - 1.2914855480 * b) ** 3
        rgb = (4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
               -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
               -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_)
        if all(-1e-6 <= v <= 1 + 1e-6 for v in rgb):
            break
    srgb = [12.92 * v if v <= 0.0031308 else 1.055 * max(v, 0) ** (1 / 2.4) - 0.055 for v in rgb]
    return '#' + ''.join(f'{round(min(max(v, 0), 1) * 255):02x}' for v in srgb)


def _spread_colors(n: int, theme: dict) -> list[str]:
    """
    n különböző árnyalat sok sejthez. Legfeljebb 8 sejtnél a validált palettából;
    fölötte az OKLCH színkörön aranymetszés-szöggel lépkedve, két váltakozó
    világossági szinten. 8-10 szín fölött a szem már nem tudja név szerint
    azonosítani a sejteket a színükről - ezért a név mindig ott van a jelmagyarázatban
    és hoverben -, de az egymásba fonódó fák így szétválnak.
    """
    if n <= len(theme['region_palette']):
        return list(theme['region_palette'][:n])
    low, high = theme['spread_lightness']
    return [_oklch_to_hex(low if i % 2 == 0 else high, theme['spread_chroma'], (25 + i * _GOLDEN_ANGLE) % 360)
            for i in range(n)]


def _slot_color(slot: int, theme: dict) -> str:
    """A paletta adott helyének színe; a 8. hely után a semleges szín (nincs ciklikus ismétlés)."""
    palette = theme['region_palette']
    return palette[slot] if 0 <= slot < len(palette) else theme['neutral']


# =============================================================================
# FELSZÍNEK (cache-elve)
# =============================================================================

def _voxel_to_um(verts: np.ndarray) -> np.ndarray:
    """
    Voxelindex-koordináta -> µm. A CCF-ben az i. voxel a [25*i, 25*(i+1)) µm
    tartomány (lásd analysis.to_voxel), a középpontja tehát (i + 0.5) * 25 µm -
    e nélkül a felszínek fél voxellel elcsúsznának az axonokhoz képest. A régi
    'round' konvencióban a középpont i * 25 µm.
    """
    return (verts + (0.0 if VOXEL_LOOKUP == 'round' else 0.5)) * VOXEL_SIZE


@st.cache_resource(show_spinner="Indexing atlas regions... (this only happens once)")
def _compact_atlas(_atlas_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Az atlasz tömörített címkékkel: (a különböző ID-k rendezve, uint16 indextérfogat).
    Egy régiókészlet maszkja így egy táblázatos kikeresés (~0,1 s), nem egy
    teljes-atlaszos np.isin (~0,5 s régiónként).
    """
    ids = np.unique(_atlas_matrix)
    return ids, np.searchsorted(ids, _atlas_matrix).astype(np.uint16)


def _ids_mask(atlas_matrix: np.ndarray, region_ids) -> np.ndarray:
    """A megadott atlasz-ID-k voxeleinek maszkja (a tömörített atlaszon)."""
    ids, compact = _compact_atlas(atlas_matrix)
    return np.isin(ids, np.asarray(list(region_ids)))[compact]


_LARGE_REGION_VOXELS = 300_000  # ~4,7 mm³ a 25 µm-es atlaszban


@st.cache_resource(show_spinner=False, max_entries=128)
def _mask_geometry(_atlas_matrix: np.ndarray, region_ids: tuple, step: int):
    """
    Felszín (verts µm-ben, faces) a megadott atlasz-ID-k voxeleire, egyszer
    kiszámolva. region_ids == ('all',) = a teljes agy (minden annotált voxel).
    A számolás a régió befoglaló dobozára szűkül, ezért kis régióknál gyors.
    (Az aláhúzásos paramétert a Streamlit nem hash-eli; az appban egy atlasz van.)
    """
    mask = (_atlas_matrix > 0) if region_ids == ('all',) else _ids_mask(_atlas_matrix, region_ids)
    if not mask.any():
        return None
    lo, hi = [], []
    for axis in range(3):
        present = np.flatnonzero(mask.any(axis=tuple(a for a in range(3) if a != axis)))
        lo.append(present[0])
        hi.append(present[-1] + 1)
    sub = np.pad(mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], 1)  # zárt felszín a doboz szélén is
    if region_ids != ('all',) and sub.sum() > _LARGE_REGION_VOXELS:
        step = max(step, 4)  # nagy régió (pl. agytörzs): durvább háló, a forma így is látszik
    verts, faces, _, _ = marching_cubes(sub, level=0.5, step_size=step)
    return np.round(_voxel_to_um(verts + np.array(lo) - 1), 1), faces


def _mesh_trace(geometry, color: str, opacity: float, name: str, legendgroup: str,
                showlegend: bool = True, context: bool = False) -> go.Mesh3d | None:
    if geometry is None:
        return None
    verts, faces = geometry
    lighting = (dict(ambient=0.95, diffuse=0.1, specular=0.0) if context
                else dict(ambient=0.5, diffuse=0.8, specular=0.2, roughness=0.5))
    return go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=color, opacity=opacity, name=name, legendgroup=legendgroup, showlegend=showlegend,
        lighting=lighting, lightposition=dict(x=100, y=200, z=150),
        hoverinfo='skip' if context else 'name',
    )


def _expand_ids(region_id: int, region_descendants: dict[int, set[int]] | None) -> set[int]:
    """A régióhoz tartozó összes atlasz-ID (önmaga + leszármazottai)."""
    ids = (region_descendants or {}).get(int(region_id))
    return set(int(v) for v in ids) if ids else {int(region_id)}


def _region_geometry(atlas_matrix, region_id, region_descendants, step=VIZ_MARCHING_CUBES_STEP):
    return _mask_geometry(atlas_matrix, tuple(sorted(_expand_ids(region_id, region_descendants))), step)


def _region_mask(atlas_matrix: np.ndarray, region_id: int,
                 region_descendants: dict[int, set[int]] | None) -> np.ndarray:
    """Egy régió voxel-maszkja a leszármazottakkal (a szülő ID önmagában 0 voxelt fedne)."""
    return _ids_mask(atlas_matrix, _expand_ids(region_id, region_descendants))


# =============================================================================
# AXONVONALAK (vektorizált)
# =============================================================================

def _drawn_edges(coords: dict, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    A kirajzolandó axonszakaszok (gyerek, ős) indexpárjai.

    stride > 1: csak minden stride-adik csomópont, plusz MINDEN elágazás, végpont és
    gyökér marad; mindegyik a legközelebbi megtartott ősével kötődik össze, így a
    vonal folytonos (a régi "minden 3. szakasz" ritkítás lyukakat hagyott).
    """
    child_rows, parent_rows, is_axon = coords['child_rows'], coords['parent_rows'], coords['is_axon']
    if stride <= 1:
        sel = is_axon[child_rows]
        return child_rows[sel], parent_rows[sel]

    n = len(is_axon)
    parent_of = np.full(n, -1)
    parent_of[child_rows] = parent_rows
    n_children = np.bincount(parent_rows, minlength=n)
    keep = (n_children != 1) | (np.arange(n) % stride == 0) | (parent_of == -1) | ~is_axon
    ancestor = parent_of.copy()
    for _ in range(4 * stride + 64):  # felfelé lépkedés a legközelebbi megtartott ősig
        climb = ancestor >= 0
        climb[climb] = ~keep[ancestor[climb]]
        if not climb.any():
            break
        ancestor[climb] = parent_of[ancestor[climb]]
    child = np.flatnonzero(keep & is_axon & (ancestor >= 0))
    return child, ancestor[child]


def _line_traces(coords: dict, child: np.ndarray, parent: np.ndarray, codes: np.ndarray, colors: dict,
                 width: float, legendgroup: str, name: str, showlegend: bool,
                 opacity: float = 1.0, hover: str | None = None) -> list[go.Scatter3d]:
    """
    Színkódonként egy vonal-réteg; a szakaszokat NaN választja el (egy réteg = egy kattintás).
    hover: ha meg van adva, ez a szöveg jelenik meg, ha az egér a vonal fölé ér (pl. a sejt neve).
    """
    # 0,1 µm-re kerekítve: a Plotly szövegként küldi az ábrát a böngészőnek, és a
    # rövid számok harmadára csökkentik a méretét (a pontosság bőven elég).
    xyz = np.round(np.column_stack([coords['x'], coords['y'], coords['z']]), 1)
    traces = []
    for code in np.unique(codes):
        sel = codes == code
        pts = np.full((int(sel.sum()) * 3, 3), np.nan)
        pts[0::3], pts[1::3] = xyz[child[sel]], xyz[parent[sel]]
        traces.append(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode='lines',
            line=dict(color=colors[int(code)], width=width), opacity=opacity,
            hoverinfo='skip' if hover is None else 'text', hovertext=hover,
            legendgroup=legendgroup, name=name, showlegend=showlegend and not traces,
        ))
    return traces


def _on_side(coords: dict, idx: np.ndarray, side: str, soma_side: int, shape: tuple) -> np.ndarray:
    """Melyik pont esik a régió értékelt oldalára (ipsi / contra); 'both' vagy eldönthetetlen soma: mind."""
    if side == 'both' or soma_side == 0 or len(idx) == 0:
        return np.ones(len(idx), dtype=bool)
    sides = side_of((coords['x'], coords['y'], coords['z'])[MIDLINE_AXIS][idx], shape)
    return sides == (soma_side if side == 'ipsi' else -soma_side)


def _stride_for(n_nodes: int, max_segments: int = VIZ_MAX_SEGMENTS) -> int:
    return max(1, math.ceil(n_nodes / max_segments))


# =============================================================================
# EGY SEJT
# =============================================================================

def build_3d_plot(
        result: CellAnalysisResult, atlas_matrix: np.ndarray, cell_name: str = "",
        show_soma_region: bool = True, show_other_regions: bool = True,
        show_only_target_regions: bool = False,
        region_descendants: dict[int, set[int]] | None = None,
        theme: str | dict | None = None,
        show_brain_outline: bool = True,
        show_projection_points: bool = True,
        view: str = 'free',
        regions_shown: list[int] | None = None,
        line_width: float = VIZ_DEFAULT_LINE_WIDTH,
        height: int = VIZ_DEFAULT_HEIGHT,
) -> go.Figure:
    """
    Egy sejt 3D jelenete. A célterületek a paletta rögzített helyeit kapják a
    célterület-listabeli sorrendjük szerint (a szín a régiót követi, nem azt, hogy
    éppen látszik-e). A régió felszíne, a benne futó axon és a vetítési pontjai
    azonos színűek; a pontok csak a régió értékelt féltekéjén jelennek meg.

    Args:
        regions_shown: a kirajzolt célterületek (None = mind)
    """
    th = get_theme(theme)
    coords = result.coords
    targets = list(enumerate(result.target_results))
    shown_ids = {tr.region_id for tr in result.target_results} if regions_shown is None else set(regions_shown)
    traces: list = []

    if show_brain_outline:
        traces.append(_mesh_trace(_mask_geometry(atlas_matrix, ('all',), VIZ_BRAIN_OUTLINE_STEP),
                                  th['brain_outline'], th['brain_outline_opacity'], 'Brain outline',
                                  'outline', context=True))
    if show_soma_region and result.soma_region_id > 0:
        traces.append(_mesh_trace(_mask_geometry(atlas_matrix, (int(result.soma_region_id),), VIZ_MARCHING_CUBES_STEP),
                                  th['soma_region'], th['region_opacity'] * 0.6,
                                  f'Soma region: {result.soma_region_name}', 'soma_region'))

    region_has_mesh = {}
    for slot, tr in targets:
        if tr.region_id not in shown_ids:
            continue
        mesh = _mesh_trace(_region_geometry(atlas_matrix, tr.region_id, region_descendants),
                           _slot_color(slot, th), th['region_opacity'],
                           f"{'✓' if tr.projects_here else '✗'} {tr.region_name}", f'region{tr.region_id}')
        region_has_mesh[tr.region_id] = mesh is not None
        traces.append(mesh)

    if show_other_regions and result.other_projection_regions:
        # Egyetlen közös felszín (egy PT sejtnek akár 40-50 ilyen régiója is lehet:
        # külön-külön lassú és zsúfolt lenne). A régiók nevei a táblázatban vannak.
        other_ids = tuple(sorted(int(o.region_id) for o in result.other_projection_regions))
        traces.append(_mesh_trace(_mask_geometry(atlas_matrix, other_ids, VIZ_MARCHING_CUBES_STEP),
                                  th['neutral'], th['region_opacity'] * 0.6,
                                  f'Other projection regions ({len(other_ids)})', 'other_regions'))

    # --- Axon: célterületenként színezve, máshol semleges ---
    point_regions = coords['point_regions']
    child, parent = _drawn_edges(coords, _stride_for(int(coords['is_axon'].sum())))
    codes = np.zeros(len(child), dtype=int)  # 0 = semleges
    colors = {0: th['axon_default']}
    allowed = np.zeros(len(child), dtype=bool)
    for slot, tr in targets:
        in_region = np.isin(point_regions[child], np.fromiter(_expand_ids(tr.region_id, region_descendants), int))
        codes[in_region & (codes == 0)] = slot + 1
        colors[slot + 1] = _slot_color(slot, th)
        if tr.region_id in shown_ids:
            allowed |= in_region & _on_side(coords, child, tr.side, result.soma_side, atlas_matrix.shape)
    if show_only_target_regions:  # "Axon-in-region": csak a (látható) célterületekben futó axon
        allowed |= point_regions[child] == result.soma_region_id
        child, parent, codes = child[allowed], parent[allowed], codes[allowed]
    traces.extend(_line_traces(coords, child, parent, codes, colors, line_width, 'axon', 'Axon', True))

    soma_idx = coords['soma_idx']
    if soma_idx is not None:
        traces.append(go.Scatter3d(
            x=[coords['x'][soma_idx]], y=[coords['y'][soma_idx]], z=[coords['z'][soma_idx]], mode='markers',
            marker=dict(size=9, color=th['soma'], symbol='circle', line=dict(color=th['paper_bg'], width=2)),
            name='Soma', legendgroup='axon', showlegend=False,
            hovertext=f'Soma<br>{result.soma_region_name}', hoverinfo='text',
        ))

    # --- Vetítési pontok: régiónként, csak az értékelt féltekén ---
    if show_projection_points:
        proj_idx = coords['proj_idx']
        n_children = np.bincount(coords['parent_rows'], minlength=len(coords['x']))
        for slot, tr in targets:
            if tr.region_id not in shown_ids:
                continue
            match = np.fromiter(_expand_ids(tr.region_id, region_descendants), dtype=int)
            pts = proj_idx[np.isin(point_regions[proj_idx], match)]
            pts = pts[_on_side(coords, pts, tr.side, result.soma_side, atlas_matrix.shape)]
            if len(pts) == 0:
                continue
            is_end = n_children[pts] == 0
            traces.append(go.Scatter3d(
                x=coords['x'][pts], y=coords['y'][pts], z=coords['z'][pts], mode='markers',
                marker=dict(size=np.where(is_end, 6, 4), color=_slot_color(slot, th),
                            symbol=np.where(is_end, 'circle', 'diamond'),
                            line=dict(color=th['paper_bg'], width=1)),
                name=f'{tr.region_name}: endpoints & branch points', legendgroup=f'region{tr.region_id}',
                showlegend=not region_has_mesh.get(tr.region_id, False),
                hovertext=[f"{tr.region_name}<br>{'endpoint' if e else 'branch point'}" for e in is_end],
                hoverinfo='text',
            ))

    fig = go.Figure(data=[t for t in traces if t is not None])
    _apply_scene_layout(fig, th, height=height, view=view,
                        title=f'<b>{cell_name}</b>  |  Soma: {result.soma_region_name}')
    return fig


# =============================================================================
# KAMERA ÉS ELRENDEZÉS
# =============================================================================

# Kameraállások a CCF tengelyeihez: x = anterior->posterior, y = dorsal->ventral,
# z = medio-laterális (bal->jobb). A rögzített nézetek ortografikusak, mint az
# atlaszmetszetek; a 'free' a szokásos forgatható perspektíva.
CAMERA_VIEWS = {
    'free': None,
    'top': dict(eye=dict(x=0, y=-2.2, z=0), up=dict(x=-1, y=0, z=0)),    # felülről, anterior felfelé
    'side': dict(eye=dict(x=0, y=0, z=2.2), up=dict(x=0, y=-1, z=0)),    # oldalról (jobbról), dorsal felfelé
    'front': dict(eye=dict(x=-2.2, y=0, z=0), up=dict(x=0, y=-1, z=0)),  # szemből (anterior felől)
}


def _apply_scene_layout(fig: go.Figure, th: dict, height: int, title: str, view: str = 'free') -> None:
    """Egységes, témafüggő elrendezés a 3D jelenetekhez."""
    axis = dict(backgroundcolor=th['scene_bg'], gridcolor=th['grid'],
                showbackground=True, zeroline=False,
                color=th['axis_text'], title_font=dict(color=th['axis_text']),
                tickfont=dict(color=th['axis_text'], size=10))
    camera = CAMERA_VIEWS.get(view)
    scene_camera = dict(camera, projection=dict(type='orthographic')) if camera else None
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=th['axis_text']), x=0.01),
        scene=dict(
            xaxis=dict(title='AP (µm)', **axis),
            yaxis=dict(title='DV (µm)', **axis),
            zaxis=dict(title='ML (µm)', **axis),
            aspectmode='data',
            camera=scene_camera,
        ),
        legend=dict(bgcolor=th['legend_bg'], bordercolor=th['legend_border'], borderwidth=1,
                    font=dict(size=12, color=th['axis_text']), groupclick='togglegroup',
                    itemsizing='constant',
                    title=dict(text=LEGEND_HINT, font=dict(size=10, color=th['axis_text']))),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor=th['paper_bg'], height=height,
        # Ugyanabban a nézetben a forgatás / zoom megmarad újrarajzoláskor is;
        # nézetváltáskor a kamera az új nézetre áll.
        uirevision=view,
    )


# =============================================================================
# TÖBB SEJT
# =============================================================================

def _n_cells(n: int) -> str:
    return f'{n} cell' if n == 1 else f'{n} cells'


def _cell_groups(results: list[tuple[str, CellAnalysisResult]], color_by: str, theme: dict) -> list[tuple]:
    """
    Sejtenként (csoportkulcs, csoportfelirat, szín).
      'cell'  - minden sejt saját színt kap (_spread_colors);
      'class' - PT / IT / CT rögzített színnel;
      'soma'  - a 7 leggyakoribb soma-régió, a többi a semleges "Other" csoport.
    """
    if color_by == 'cell':
        colors = _spread_colors(len(results), theme)
        return [(name, name, color) for (name, _), color in zip(results, colors)]
    if color_by == 'class':
        keys = [r.projection_class if r.projection_class in _CLASS_ORDER else 'Unknown' for _, r in results]
        slot_of = {c: _CLASS_ORDER.index(c) for c in _CLASS_ORDER if c in keys}
    elif color_by == 'soma':
        keys = [r.group_region for _, r in results]
        counts = {k: keys.count(k) for k in dict.fromkeys(keys)}
        slot_of = {k: i for i, k in enumerate(sorted(counts, key=lambda k: -counts[k])[:7])}
    else:
        raise ValueError(f'unknown colour mode: {color_by}')
    n_per_key = {k: keys.count(k) for k in set(keys)}
    other_n = sum(n for k, n in n_per_key.items() if k not in slot_of)
    out = []
    for key in keys:
        if key in slot_of:
            label = f'{key} ({_n_cells(n_per_key[key])})'
            out.append((key, label, _slot_color(slot_of[key], theme)))
        else:
            out.append(('__other__', f'Other ({_n_cells(other_n)})', theme['neutral']))
    return out


def build_3d_plot_multi(
        results: list[tuple[str, CellAnalysisResult]], atlas_matrix: np.ndarray,
        target_region_ids: list[int], show_target_regions: bool = True,
        show_only_target_regions: bool = False,
        region_descendants: dict[int, set[int]] | None = None,
        theme: str | dict | None = None,
        show_brain_outline: bool = True,
        view: str = 'free',
        color_by: str = 'cell',
        regions_shown: list[int] | None = None,
        line_width: float = VIZ_DEFAULT_LINE_WIDTH - 1,
        height: int = VIZ_DEFAULT_HEIGHT,
        max_segments: int = VIZ_MAX_SEGMENTS_COMBINED,
        line_opacity: float = 1.0,
) -> go.Figure:
    """
    Több sejt együttes nézete. A sejtek csoportonként színeződnek (color_by), és
    egy jelmagyarázat-kattintás az egész csoportot (axon + soma) ki-be kapcsolja.
    Ha az összes axon-csomópont több, mint max_segments, a fák egyformán
    egyszerűsödnek (folytonos vonalakkal).
    """
    th = get_theme(theme)
    traces: list = []
    region_names = {tr.region_id: tr.region_name for tr in results[0][1].target_results} if results else {}
    shown_ids = list(target_region_ids) if regions_shown is None else [r for r in target_region_ids if r in regions_shown]

    if show_brain_outline:
        traces.append(_mesh_trace(_mask_geometry(atlas_matrix, ('all',), VIZ_BRAIN_OUTLINE_STEP),
                                  th['brain_outline'], th['brain_outline_opacity'], 'Brain outline',
                                  'outline', context=True))
    if show_target_regions:
        for slot, region_id in enumerate(target_region_ids):
            if region_id in shown_ids:
                traces.append(_mesh_trace(_region_geometry(atlas_matrix, region_id, region_descendants),
                                          _slot_color(slot, th), th['region_opacity'] * 0.8,
                                          region_names.get(region_id, f'Region {region_id}'), f'region{region_id}'))

    allowed_ids = None
    if show_only_target_regions:
        allowed_ids = set()
        for rid in shown_ids:
            allowed_ids |= _expand_ids(rid, region_descendants)

    stride = _stride_for(sum(int(r.coords['is_axon'].sum()) for _, r in results), max_segments)
    groups = _cell_groups(results, color_by, th)
    legend_done = set()
    for (cell_name, result), (key, label, color) in zip(results, groups):
        coords = result.coords
        child, parent = _drawn_edges(coords, stride)
        if allowed_ids is not None:
            keep = np.isin(coords['point_regions'][child], np.fromiter(allowed_ids | {result.soma_region_id}, int))
            child, parent = child[keep], parent[keep]
        group = f'cells:{key}'
        first = key not in legend_done
        legend_done.add(key)
        traces.extend(_line_traces(coords, child, parent, np.zeros(len(child), dtype=int), {0: color},
                                   line_width, group, label, first, opacity=line_opacity,
                                   hover=f'{cell_name}<br>{result.projection_subclass or ""}'
                                         f'<br>Soma: {result.soma_region_name}'))
        soma_idx = coords['soma_idx']
        if soma_idx is not None:
            traces.append(go.Scatter3d(
                x=[coords['x'][soma_idx]], y=[coords['y'][soma_idx]], z=[coords['z'][soma_idx]], mode='markers',
                marker=dict(size=7, color=color, symbol='circle', line=dict(color=th['paper_bg'], width=2)),
                legendgroup=group, name=label, showlegend=False,
                hovertext=f'{cell_name}<br>{result.projection_subclass or ""}<br>Soma: {result.soma_region_name}',
                hoverinfo='text',
            ))

    fig = go.Figure(data=[t for t in traces if t is not None])
    simplified = f' · simplified 1:{stride}' if stride > 1 else ''
    n_legend = sum(1 for t in fig.data if t.showlegend is not False)
    # Sok jelmagyarázat-elemnél a lista az ábra alá kerül, több oszlopba: jobbra
    # elvenné a szélesség harmadát (az agy széles), alul csak a magasságot növeli.
    legend_rows = math.ceil(n_legend / _LEGEND_COLUMNS) if n_legend > _LEGEND_SIDE_MAX else 0
    _apply_scene_layout(fig, th, height=height + legend_rows * _LEGEND_ROW_PX, view=view,
                        title=f'<b>Combined view</b>  —  {_n_cells(len(results))}{simplified}')
    if legend_rows:
        fig.update_layout(legend=dict(orientation='h', x=0, xanchor='left', y=0, yanchor='top',
                                      entrywidth=1 / _LEGEND_COLUMNS, entrywidthmode='fraction',
                                      title=dict(text='')),
                          margin=dict(b=legend_rows * _LEGEND_ROW_PX + 10))
    return fig


_LEGEND_SIDE_MAX = 15   # ennyi elemig a jelmagyarázat jobbra kerül
_LEGEND_COLUMNS = 5
_LEGEND_ROW_PX = 36  # mért: egy jelmagyarázat-sor ~36 px (12 px-es betű + vonalminta)
