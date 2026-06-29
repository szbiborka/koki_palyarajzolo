# =============================================================================
# VIZUALIZÁCIÓ MODUL - 3D ábra generálása Plotly-val.
# =============================================================================

import numpy as np
import plotly.graph_objects as go
from skimage.measure import marching_cubes

from config import (
    VOXEL_SIZE, VIZ_REGION_OPACITY, VIZ_MARCHING_CUBES_STEP, COLORS
)
from core.analysis import CellAnalysisResult


def _get_region_color(region_index: int) -> str:
    palette = COLORS['region_palette']
    return palette[region_index % len(palette)]


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
        allowed_regions: set | None = None,  # ÚJ PARAMÉTER AZ EXCLUSIVE NÉZETHEZ
) -> list[go.Scatter3d]:
    segments_by_color: dict[str, tuple[list, list, list]] = {}

    for i in curr_idx:
        if not is_axon[i]: continue

        # EXCLUSIVE LOGIKA: Eldobjuk a régiót, ha nincs az engedélyezett listában
        region_int = int(point_regions[i])
        if allowed_regions is not None and region_int not in allowed_regions:
            continue

        color = region_color_map.get(region_int, COLORS['axon_default'])
        p_row = parent_row_indices[i]

        if color not in segments_by_color:
            segments_by_color[color] = ([], [], [])

        xs, ys, zs = segments_by_color[color]
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


def build_3d_plot(
        result: CellAnalysisResult, atlas_matrix: np.ndarray, cell_name: str = "",
        show_soma_region: bool = True, show_other_regions: bool = True,
        show_only_target_regions: bool = False  # ÚJ KAPCSOLÓ
) -> go.Figure:
    coords = result.coords
    x, y, z, is_axon, point_regions = coords['x'], coords['y'], coords['z'], coords['is_axon'], coords['point_regions']
    proj_idx, curr_idx, parent_row_indices, soma_idx = coords['proj_idx'], coords['curr_idx'], coords[
        'parent_row_indices'], coords['soma_idx']

    region_color_map: dict[int, str] = {tr.region_id: _get_region_color(i) for i, tr in
                                        enumerate(result.target_results)}
    traces: list = []

    if show_soma_region and result.soma_region_id > 0:
        if t := _build_mesh_trace(atlas_matrix == result.soma_region_id, '#c0392b', VIZ_REGION_OPACITY,
                                  f'Soma: {result.soma_region_name}'):
            traces.append(t)

    for tr in result.target_results:
        proj_symbol = '✓' if tr.projects_here else '✗'
        if t := _build_mesh_trace(atlas_matrix == tr.region_id, region_color_map[tr.region_id], VIZ_REGION_OPACITY,
                                  f'{proj_symbol} {tr.region_name}'):
            traces.append(t)

    if show_other_regions:
        for i, other in enumerate(result.other_projection_regions):
            color = _get_region_color(len(result.target_results) + i)
            region_color_map[other.region_id] = color
            if t := _build_mesh_trace(atlas_matrix == other.region_id, color, VIZ_REGION_OPACITY * 0.6,
                                      f'(other) {other.region_name}'):
                traces.append(t)

    # --- EXCLUSIVE NÉZET ÁTADÁSA ---
    allowed_regions = None
    if show_only_target_regions:
        allowed_regions = set(tr.region_id for tr in result.target_results)
        allowed_regions.add(result.soma_region_id)

    traces.extend(_build_axon_trace(x, y, z, curr_idx, parent_row_indices, is_axon, point_regions, region_color_map, 2,
                                    allowed_regions))

    if soma_idx is not None:
        traces.append(go.Scatter3d(
            x=[x[soma_idx]], y=[y[soma_idx]], z=[z[soma_idx]], mode='markers',
            marker=dict(size=8, color='black', symbol='circle'), name='Soma',
            hovertext=f'Soma<br>{result.soma_region_name}', hoverinfo='text',
        ))

    for tr in result.target_results:
        if len(pts := proj_idx[point_regions[proj_idx] == tr.region_id]) > 0:
            traces.append(go.Scatter3d(
                x=x[pts], y=y[pts], z=z[pts], mode='markers',
                marker=dict(size=5, color=region_color_map[tr.region_id], symbol='diamond',
                            line=dict(color='white', width=0.5)),
                name=f'Proj. pts: {tr.region_name}', hovertext=[f'{tr.region_name}<br>ep/branch' for _ in pts],
                hoverinfo='text',
            ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=f'<b>{cell_name}</b>  |  Soma: {result.soma_region_name}', font=dict(size=13, color='#33401F'),
                   x=0.01),
        scene=dict(
            xaxis=dict(title='X (µm)', backgroundcolor='#f8f8f8', gridcolor='#dddddd', showbackground=True),
            yaxis=dict(title='Y (µm)', backgroundcolor='#f8f8f8', gridcolor='#dddddd', showbackground=True),
            zaxis=dict(title='Z (µm)', backgroundcolor='#f8f8f8', gridcolor='#dddddd', showbackground=True),
            aspectmode='data',
        ),
        legend=dict(bgcolor='rgba(255,255,255,0.85)', bordercolor='#CFD6BC', borderwidth=1, font=dict(size=11)),
        margin=dict(l=0, r=0, t=40, b=0), paper_bgcolor='white', height=650,
    )
    return fig


def build_3d_plot_multi(
        results: list[tuple[str, CellAnalysisResult]], atlas_matrix: np.ndarray,
        target_region_ids: list[int], show_target_regions: bool = True,
        show_only_target_regions: bool = False  # ÚJ KAPCSOLÓ
) -> go.Figure:
    palette, traces = COLORS['region_palette'], []
    region_names = {tr.region_id: tr.region_name for tr in results[0][1].target_results} if results else {}

    if show_target_regions:
        for i, region_id in enumerate(target_region_ids):
            if t := _build_mesh_trace(atlas_matrix == region_id, _get_region_color(i), VIZ_REGION_OPACITY * 0.55,
                                      region_names.get(region_id, f'Region {region_id}')):
                traces.append(t)

    for i, (cell_name, result) in enumerate(results):
        cell_color = palette[i % len(palette)]
        coords = result.coords
        uniform_color_map = {int(rid): cell_color for rid in np.unique(coords['point_regions'])}

        # --- EXCLUSIVE NÉZET ÁTADÁSA ---
        allowed_regions = None
        if show_only_target_regions:
            allowed_regions = set(target_region_ids)
            allowed_regions.add(result.soma_region_id)

        axon_traces = _build_axon_trace(
            coords['x'], coords['y'], coords['z'], coords['curr_idx'], coords['parent_row_indices'],
            coords['is_axon'], coords['point_regions'], uniform_color_map, 1, allowed_regions
        )
        for j, tr in enumerate(axon_traces):
            if j == 0: tr.showlegend, tr.name = True, cell_name
            traces.append(tr)

        if coords['soma_idx'] is not None:
            traces.append(go.Scatter3d(
                x=[coords['x'][coords['soma_idx']]], y=[coords['y'][coords['soma_idx']]],
                z=[coords['z'][coords['soma_idx']]],
                mode='markers',
                marker=dict(size=7, color=cell_color, symbol='circle', line=dict(color='white', width=1)),
                showlegend=False, hovertext=f'{cell_name}<br>Soma: {result.soma_region_name}', hoverinfo='text',
            ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=f'<b>Combined view</b>  —  {len(results)} cells', font=dict(size=13, color='#33401F'), x=0.01),
        scene=dict(xaxis=dict(title='X', showbackground=True), yaxis=dict(title='Y', showbackground=True),
                   zaxis=dict(title='Z', showbackground=True), aspectmode='data'),
        legend=dict(bgcolor='rgba(255,255,255,0.85)', bordercolor='#CFD6BC', borderwidth=1),
        margin=dict(l=0, r=0, t=40, b=0), paper_bgcolor='white', height=700,
    )
    return fig


def render_plot_streamlit(fig: go.Figure, key: str) -> None:
    import streamlit as st
    st.plotly_chart(fig, use_container_width=True, key=key)