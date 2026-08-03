# VIZUALIZÁCIÓ MODUL - 3D ábra generálása Plotly-val.

import numpy as np
import plotly.graph_objects as go
from skimage.measure import marching_cubes

from config import (
    VOXEL_SIZE, VIZ_REGION_OPACITY, VIZ_MARCHING_CUBES_STEP, COLORS,
    VIZ_THEMES, DEFAULT_VIZ_THEME, VIZ_BRAIN_OUTLINE_STEP
)
from core.analysis import CellAnalysisResult


def get_theme(theme: str | dict | None = None) -> dict:
    """A kért 3D téma beállításai (alapértelmezés: DEFAULT_VIZ_THEME)."""
    if isinstance(theme, dict):
        return theme
    return VIZ_THEMES.get(theme or DEFAULT_VIZ_THEME, VIZ_THEMES[DEFAULT_VIZ_THEME])


def _get_region_color(region_index: int, theme: dict | None = None) -> str:
    palette = (theme or get_theme())['region_palette'] if theme else COLORS['region_palette']
    return palette[region_index % len(palette)]


def _build_brain_outline(atlas_matrix: np.ndarray, theme: dict) -> go.Mesh3d | None:
    """
    A teljes agy külső felszíne, nagyon áttetszően - térbeli tájékozódáshoz.

    Enélkül a sejt a semmiben lebeg: nem látszik, hol van az agyban, melyik
    félteke, mennyire halad előre/hátra. Az ÖSSZES annotált voxelt vesszük
    (atlas > 0), így nem függünk a "root" régió ID-jától. Nagyobb marching-cubes
    lépésköz, mert ez a legnagyobb felület - a körvonalnak nem kell részletesnek
    lennie, csak elhelyeznie a sejtet.
    """
    mask = atlas_matrix > 0
    if not np.any(mask):
        return None
    verts, faces, _, _ = marching_cubes(mask, level=0.5, step_size=VIZ_BRAIN_OUTLINE_STEP)
    verts = verts * VOXEL_SIZE
    return go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=theme['brain_outline'], opacity=theme['brain_outline_opacity'],
        name='Brain outline', showlegend=True,
        # Lapos árnyékolás és kikapcsolt hover: a körvonal legyen jelen, de
        # soha ne vonja el a figyelmet és ne fogja el a kattintásokat.
        lighting=dict(ambient=0.95, diffuse=0.1, specular=0.0),
        hoverinfo='skip',
    )


def _build_mesh_trace(mask: np.ndarray, color: str, opacity: float, name: str,
                      showlegend: bool = True) -> go.Mesh3d | None:
    if not np.any(mask): return None
    verts, faces, _, _ = marching_cubes(mask, level=0.5, step_size=VIZ_MARCHING_CUBES_STEP)
    verts = verts * VOXEL_SIZE
    return go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=color, opacity=opacity, name=name, showlegend=showlegend,
        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.2, roughness=0.5),
        lightposition=dict(x=100, y=200, z=150), hoverinfo='name',
    )


def _build_axon_trace(
        x: np.ndarray, y: np.ndarray, z: np.ndarray,
        curr_idx: np.ndarray, parent_row_indices: np.ndarray,
        is_axon: np.ndarray, point_regions: np.ndarray,
        region_color_map: dict[int, str], line_width: int = 2,
        allowed_regions: set | None = None,
        downsample_factor: int = 1,  # ÚJ PARAMÉTER: Pontok ritkítása
        theme: dict | None = None
) -> list[go.Scatter3d]:
    # A célterületen kívüli axon a téma "halvány" színét kapja: jelen van a
    # kontextus kedvéért, de nem versenyez a színnel jelölt célterületekkel.
    default_axon = (theme or get_theme())['axon_default']
    segments_by_color: dict[str, tuple[list, list, list]] = {}

    for count, i in enumerate(curr_idx):
        if not is_axon[i]: continue

        # RITKÍTÁS LOGIKA: Ha a faktor > 1, csak minden N-edik szakaszt tartjuk meg
        if downsample_factor > 1 and count % downsample_factor != 0:
            continue

        # EXCLUSIVE LOGIKA: Eldobjuk a régiót, ha nincs az engedélyezett listában
        region_int = int(point_regions[i])
        if allowed_regions is not None and region_int not in allowed_regions:
            continue

        color = region_color_map.get(region_int, default_axon)
        p_row = parent_row_indices[i]

        if color not in segments_by_color:
            segments_by_color[color] = ([], [], [])

        xs, ys, zs = segments_by_color[color]

        # A NaN (None) elválasztók miatt a Plotly egyetlen rétegként (trace) kezeli
        # a megszakított vonalakat is, ami lehetővé teszi az egykattintásos ki/be kapcsolást!
        xs.extend([x[i], x[p_row], None])
        ys.extend([y[i], y[p_row], None])
        zs.extend([z[i], z[p_row], None])

    traces = []
    for color, (xs, ys, zs) in segments_by_color.items():
        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs, mode='lines', line=dict(color=color, width=line_width),
            hoverinfo='skip', showlegend=False,
        ))
    return traces


def _region_mask(atlas_matrix: np.ndarray, region_id: int,
                 region_descendants: dict[int, set[int]] | None) -> np.ndarray:
    """
    Egy régió voxel-maszkja, a SZÜLŐ régiókat is beleértve.

    Az annotációs térfogat csak a levél-régiókat címkézi, ezért egy szülő (pl.
    "Brain stem") vagy a virtuális "leszálló agytörzs" ID-ja önmagában 0 voxelt
    fedne - nem rajzolódna ki felület. A leszármazottakra feloldva viszont igen.
    """
    ids = (region_descendants or {}).get(int(region_id))
    if ids:
        return np.isin(atlas_matrix, np.fromiter((int(v) for v in ids), dtype=int))
    return atlas_matrix == region_id


def _expand_ids(region_id: int, region_descendants: dict[int, set[int]] | None) -> set[int]:
    """A régióhoz tartozó összes atlasz-ID (önmaga + leszármazottai)."""
    ids = (region_descendants or {}).get(int(region_id))
    return set(int(v) for v in ids) if ids else {int(region_id)}


def build_3d_plot(
        result: CellAnalysisResult, atlas_matrix: np.ndarray, cell_name: str = "",
        show_soma_region: bool = True, show_other_regions: bool = True,
        show_only_target_regions: bool = False,
        region_descendants: dict[int, set[int]] | None = None,
        theme: str | dict | None = None,
        show_brain_outline: bool = True
) -> go.Figure:
    th = get_theme(theme)
    coords = result.coords
    x, y, z, is_axon, point_regions = coords['x'], coords['y'], coords['z'], coords['is_axon'], coords['point_regions']
    proj_idx, curr_idx, parent_row_indices, soma_idx = coords['proj_idx'], coords['curr_idx'], coords[
        'parent_row_indices'], coords['soma_idx']

    # A színtérkép a TÉNYLEGES atlasz-ID-kra épül: egy szülő régió minden
    # leszármazott magja ugyanazt a színt kapja, különben az ottani axonok
    # szürkék maradnának (a csomópontok levél-ID-t hordoznak, nem a szülőét).
    region_color_map: dict[int, str] = {}
    for i, tr in enumerate(result.target_results):
        color = _get_region_color(i, th)
        for rid in _expand_ids(tr.region_id, region_descendants):
            region_color_map[rid] = color
    traces: list = []

    # Az agy körvonala legelöl, hogy a többi réteg fölé rajzolódjon rá.
    if show_brain_outline:
        if t := _build_brain_outline(atlas_matrix, th):
            traces.append(t)

    if show_soma_region and result.soma_region_id > 0:
        # A soma-régió KONTEXTUS (gyakran nagy kérgi terület), ezért halványabb a
        # célterületeknél - különben elnyomná azokat és az axont is.
        if t := _build_mesh_trace(atlas_matrix == result.soma_region_id, '#c0392b',
                                  th['region_opacity'] * 0.55,
                                  f'Soma: {result.soma_region_name}'):
            traces.append(t)

    for i, tr in enumerate(result.target_results):
        proj_symbol = '✓' if tr.projects_here else '✗'
        if t := _build_mesh_trace(_region_mask(atlas_matrix, tr.region_id, region_descendants),
                                  _get_region_color(i, th), th['region_opacity'],
                                  f'{proj_symbol} {tr.region_name}'):
            traces.append(t)

    if show_other_regions:
        for i, other in enumerate(result.other_projection_regions):
            color = _get_region_color(len(result.target_results) + i, th)
            region_color_map[other.region_id] = color
            if t := _build_mesh_trace(atlas_matrix == other.region_id, color, th['region_opacity'] * 0.6,
                                      f'(other) {other.region_name}'):
                traces.append(t)

    allowed_regions = None
    if show_only_target_regions:
        # Szülő régióknál a leszármazottakat is engedni kell, különben az ott
        # futó axonok teljesen eltűnnének az "Axon-in-region" nézetből.
        allowed_regions = set()
        for tr in result.target_results:
            allowed_regions |= _expand_ids(tr.region_id, region_descendants)
        allowed_regions.add(result.soma_region_id)

    # Szóló sejtnél nincs szükség ritkításra (downsample_factor = 1)
    traces.extend(_build_axon_trace(x, y, z, curr_idx, parent_row_indices, is_axon, point_regions,
                                    region_color_map, th['axon_width'],
                                    allowed_regions, downsample_factor=1, theme=th))

    if soma_idx is not None:
        traces.append(go.Scatter3d(
            x=[x[soma_idx]], y=[y[soma_idx]], z=[z[soma_idx]], mode='markers',
            marker=dict(size=8, color=th['soma'], symbol='circle',
                        line=dict(color=th['paper_bg'], width=1)), name='Soma',
            hovertext=f'Soma<br>{result.soma_region_name}', hoverinfo='text',
        ))

    for i, tr in enumerate(result.target_results):
        match = np.fromiter(_expand_ids(tr.region_id, region_descendants), dtype=int)
        if len(pts := proj_idx[np.isin(point_regions[proj_idx], match)]) > 0:
            traces.append(go.Scatter3d(
                x=x[pts], y=y[pts], z=z[pts], mode='markers',
                marker=dict(size=5, color=_get_region_color(i, th), symbol='diamond',
                            line=dict(color=th['paper_bg'], width=0.5)),
                name=f'Proj. pts: {tr.region_name}', hovertext=[f'{tr.region_name}<br>ep/branch' for _ in pts],
                hoverinfo='text',
            ))

    fig = go.Figure(data=traces)
    _apply_scene_layout(
        fig, th, height=650,
        title=f'<b>{cell_name}</b>  |  Soma: {result.soma_region_name}')
    return fig


def _apply_scene_layout(fig: go.Figure, th: dict, height: int, title: str) -> None:
    """Egységes, témafüggő elrendezés a 3D jelenetekhez."""
    axis = dict(backgroundcolor=th['scene_bg'], gridcolor=th['grid'],
                showbackground=True, zeroline=False,
                color=th['axis_text'], title_font=dict(color=th['axis_text']),
                tickfont=dict(color=th['axis_text'], size=9))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=th['axis_text']), x=0.01),
        scene=dict(
            xaxis=dict(title='X (µm)', **axis),
            yaxis=dict(title='Y (µm)', **axis),
            zaxis=dict(title='Z (µm)', **axis),
            aspectmode='data',
        ),
        legend=dict(bgcolor=th['legend_bg'], bordercolor=th['legend_border'],
                    borderwidth=1, font=dict(size=11, color=th['axis_text'])),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor=th['paper_bg'], height=height,
    )


def build_3d_plot_multi(
        results: list[tuple[str, CellAnalysisResult]], atlas_matrix: np.ndarray,
        target_region_ids: list[int], show_target_regions: bool = True,
        show_only_target_regions: bool = False,
        region_descendants: dict[int, set[int]] | None = None,
        theme: str | dict | None = None,
        show_brain_outline: bool = True
) -> go.Figure:
    th = get_theme(theme)
    palette, traces = th['region_palette'], []
    region_names = {tr.region_id: tr.region_name for tr in results[0][1].target_results} if results else {}

    # Agy körvonal a térbeli tájékozódáshoz (lásd build_3d_plot).
    if show_brain_outline:
        if t := _build_brain_outline(atlas_matrix, th):
            traces.append(t)

    if show_target_regions:
        for i, region_id in enumerate(target_region_ids):
            if t := _build_mesh_trace(_region_mask(atlas_matrix, region_id, region_descendants),
                                      _get_region_color(i, th), th['region_opacity'] * 0.55,
                                      region_names.get(region_id, f'Region {region_id}')):
                traces.append(t)

    for i, (cell_name, result) in enumerate(results):
        cell_color = palette[i % len(palette)]
        coords = result.coords
        uniform_color_map = {int(rid): cell_color for rid in np.unique(coords['point_regions'])}

        allowed_regions = None
        if show_only_target_regions:
            # Szülő régiók leszármazottait is engedni kell (lásd build_3d_plot).
            allowed_regions = set()
            for rid in target_region_ids:
                allowed_regions |= _expand_ids(rid, region_descendants)
            allowed_regions.add(result.soma_region_id)

        # RITKÍTÁS ALKALMAZÁSA: downsample_factor=3 drasztikusan csökkenti a memóriaterhelést
        axon_traces = _build_axon_trace(
            coords['x'], coords['y'], coords['z'], coords['curr_idx'], coords['parent_row_indices'],
            coords['is_axon'], coords['point_regions'], uniform_color_map, 1, allowed_regions,
            downsample_factor=3, theme=th
        )
        for j, tr in enumerate(axon_traces):
            if j == 0: tr.showlegend, tr.name = True, cell_name
            traces.append(tr)

        if coords['soma_idx'] is not None:
            traces.append(go.Scatter3d(
                x=[coords['x'][coords['soma_idx']]], y=[coords['y'][coords['soma_idx']]],
                z=[coords['z'][coords['soma_idx']]],
                mode='markers',
                marker=dict(size=7, color=cell_color, symbol='circle',
                            line=dict(color=th['paper_bg'], width=1)),
                showlegend=False, hovertext=f'{cell_name}<br>Soma: {result.soma_region_name}', hoverinfo='text',
            ))

    fig = go.Figure(data=traces)
    _apply_scene_layout(fig, th, height=700,
                        title=f'<b>Combined view</b>  —  {len(results)} cells')
    return fig


def render_plot_streamlit(fig: go.Figure, key: str) -> None:
    import streamlit as st
    st.plotly_chart(fig, use_container_width=True, key=key)