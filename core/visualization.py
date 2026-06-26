# =============================================================================
# VIZUALIZÁCIÓ MODUL - 3D ábra generálása Plotly-val.
#
# Miért Plotly és nem PyVista/stpyvista?
#   - Plotly WebGL-t használ, natívan fut a böngészőben
#   - st.plotly_chart() azonnal működik mindenkin – nincs Xvfb, nincs VTK.js,
#     nincs stpyvista widget cache bug, nincs Kitware fallback oldal
#   - A go.Mesh3d az agyi struktúrák félig átlátszó felszínéhez
#   - A go.Scatter3d mode='lines' az axon vonalakhoz (None szeparátorral
#     egy trace-ben, ami nagyon hatékony nagy neuronoknál)
#   - A go.Scatter3d mode='markers' a somához és vetítési pontokhoz
#
# Fő függvények:
#   build_3d_plot()       - egyetlen sejt vizualizációja
#   build_3d_plot_multi() - több sejt egyszerre (batch)
#   render_plot_streamlit() - st.plotly_chart() wrapper (app.py hívja)
# =============================================================================

import numpy as np
import plotly.graph_objects as go
from skimage.measure import marching_cubes

from config import (
    VOXEL_SIZE, VIZ_REGION_OPACITY, VIZ_MARCHING_CUBES_STEP, COLORS
)
from core.analysis import CellAnalysisResult


# =============================================================================
# BELSŐ SEGÉDFÜGGVÉNYEK
# =============================================================================

def _get_region_color(region_index: int) -> str:
    """Körkörösen hozzárendel egy hex színt a régió indexe alapján."""
    palette = COLORS['region_palette']
    return palette[region_index % len(palette)]


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """
    Hex színt rgba() stringgé alakít Plotly opacity kezeléséhez.
    Pl. '#1f77b4', 0.25  ->  'rgba(31,119,180,0.25)'
    """
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


def _build_mesh_trace(
    mask: np.ndarray,
    color: str,
    opacity: float,
    name: str,
    showlegend: bool = True,
) -> go.Mesh3d | None:
    """
    Marching cubes-szal épít egy Plotly Mesh3d trace-t egy bináris maszkból.
    Ez az agyi struktúrák félig átlátszó felszínéhez kell.

    Args:
        mask:       3D bináris numpy tömb (True ahol a régió van)
        color:      hex szín string
        opacity:    0.0 - 1.0 közötti átlátszóság
        name:       a legendában megjelenő név
        showlegend: szerepeljen-e a legendában

    Returns:
        go.Mesh3d trace, vagy None ha a maszk üres
    """
    if not np.any(mask):
        return None

    verts, faces, _, _ = marching_cubes(mask, level=0.5, step_size=VIZ_MARCHING_CUBES_STEP)
    verts = verts * VOXEL_SIZE  # pixelkoordináták -> mikrométer

    return go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=color,
        opacity=opacity,
        name=name,
        showlegend=showlegend,
        # flatshading=False -> simított felület (Gouraud shading)
        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.2, roughness=0.5),
        lightposition=dict(x=100, y=200, z=150),
        hoverinfo='name',
    )


def _build_axon_trace(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    curr_idx: np.ndarray,
    parent_row_indices: np.ndarray,
    is_axon: np.ndarray,
    point_regions: np.ndarray,
    region_color_map: dict[int, str],
    line_width: int = 2,
) -> list[go.Scatter3d]:
    """
    Felépíti az axon vonalakat Plotly Scatter3d trace-ként.

    Plotly-ban a vonalakat úgy rajzoljuk hatékonyan, hogy None szeparátorokat
    szúrunk a szegmensek közé egyetlen trace-ben. Ez sokkal gyorsabb mint
    szegmensenként külön trace – egy 10 000 pontos neuronoknál is villámgyors.

    Szín szerint csoportosítunk: minden egyedi szín egy trace lesz,
    hogy a legendában külön szerepeljenek (ha szükséges).

    Args:
        x, y, z:              SWC koordináták (mikrométerben)
        curr_idx:             érvényes csomópontok indexei (parent_row_indices != -1)
        parent_row_indices:   minden pont szülőjének indexe
        is_axon:              boolean maszk – melyik pont axon típusú
        point_regions:        minden pont melyik atlasz régióba esik
        region_color_map:     régió ID -> hex szín szótár
        line_width:           vonal vastagság pixelben

    Returns:
        Lista go.Scatter3d trace objektumokból (egy trace per egyedi szín)
    """
    # Szín szerint gyűjtjük a szegmenseket
    # Minden szegmens: (pont_a_xyz, pont_b_xyz)
    segments_by_color: dict[str, tuple[list, list, list]] = {}
    # Struktúra: {color: (xs, ys, zs)} ahol None szeparátorral vannak elválasztva

    for i in curr_idx:
        if not is_axon[i]:
            continue
        p_row = parent_row_indices[i]
        color = region_color_map.get(int(point_regions[i]), COLORS['axon_default'])

        if color not in segments_by_color:
            segments_by_color[color] = ([], [], [])

        xs, ys, zs = segments_by_color[color]
        # Két pont + None szeparátor = egy szegmens a Plotly vonalban
        xs.extend([x[i], x[p_row], None])
        ys.extend([y[i], y[p_row], None])
        zs.extend([z[i], z[p_row], None])

    traces = []
    for color, (xs, ys, zs) in segments_by_color.items():
        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='lines',
            line=dict(color=color, width=line_width),
            hoverinfo='skip',
            showlegend=False,  # Az axon vonalaknak nincs külön legenda sor
        ))

    return traces


# =============================================================================
# FŐ PLOT ÉPÍTŐK
# =============================================================================

def build_3d_plot(
    result: CellAnalysisResult,
    atlas_matrix: np.ndarray,
    cell_name: str = "",
    show_soma_region: bool = True,
    show_other_regions: bool = True,
) -> go.Figure:
    """
    Egyetlen sejt teljes 3D vizualizációját építi fel Plotly Figure-ként.

    Tartalmazza:
      - Agyi struktúrák félátlátszó felszínei (Mesh3d)
      - Teljes axonfa vonalakként, régiónként színezve (Scatter3d lines)
      - Soma gömbjelölőként (Scatter3d marker)
      - Vetítési pontok (végpontok + elágazások) kiemelve (Scatter3d marker)

    Args:
        result:             run_analysis() eredménye
        atlas_matrix:       Allen Brain Atlas 3D annotációs mátrixa
        cell_name:          megjelenítési név a cím sorában
        show_soma_region:   megjelenítse-e a soma régió felszínét
        show_other_regions: megjelenítse-e az egyéb vetítési területeket

    Returns:
        go.Figure – st.plotly_chart()-tal közvetlenül megjeleníthető
    """
    coords = result.coords
    x, y, z = coords['x'], coords['y'], coords['z']
    is_axon        = coords['is_axon']
    point_regions  = coords['point_regions']
    proj_idx       = coords['proj_idx']
    curr_idx       = coords['curr_idx']
    parent_row_indices = coords['parent_row_indices']
    soma_idx       = coords['soma_idx']

    # Célterületek -> szín hozzárendelés
    region_color_map: dict[int, str] = {}
    for i, tr in enumerate(result.target_results):
        region_color_map[tr.region_id] = _get_region_color(i)

    traces: list = []

    # ------------------------------------------------------------------
    # 1. Agyi struktúrák felszínei (Mesh3d)
    # ------------------------------------------------------------------

    # Soma régiója (piros)
    if show_soma_region and result.soma_region_id > 0:
        t = _build_mesh_trace(
            atlas_matrix == result.soma_region_id,
            color='#c0392b',
            opacity=VIZ_REGION_OPACITY,
            name=f'Soma region: {result.soma_region_name}',
        )
        if t:
            traces.append(t)

    # Célterületek felszínei
    for tr in result.target_results:
        color = region_color_map[tr.region_id]
        proj_symbol = '✓' if tr.projects_here else '✗'
        t = _build_mesh_trace(
            atlas_matrix == tr.region_id,
            color=color,
            opacity=VIZ_REGION_OPACITY,
            name=f'{proj_symbol} {tr.region_name}',
        )
        if t:
            traces.append(t)

    # Egyéb vetítési területek (halványabban)
    if show_other_regions:
        for i, other in enumerate(result.other_projection_regions):
            color = _get_region_color(len(result.target_results) + i)
            region_color_map[other.region_id] = color
            t = _build_mesh_trace(
                atlas_matrix == other.region_id,
                color=color,
                opacity=VIZ_REGION_OPACITY * 0.6,
                name=f'(other) {other.region_name}',
            )
            if t:
                traces.append(t)

    # ------------------------------------------------------------------
    # 2. Axon vonalak
    # ------------------------------------------------------------------
    axon_traces = _build_axon_trace(
        x, y, z, curr_idx, parent_row_indices,
        is_axon, point_regions, region_color_map,
        line_width=2,
    )
    traces.extend(axon_traces)

    # ------------------------------------------------------------------
    # 3. Soma jelölő (fekete gömb)
    # ------------------------------------------------------------------
    if soma_idx is not None:
        traces.append(go.Scatter3d(
            x=[x[soma_idx]], y=[y[soma_idx]], z=[z[soma_idx]],
            mode='markers',
            marker=dict(size=8, color='black', symbol='circle'),
            name='Soma',
            hovertext=f'Soma<br>{result.soma_region_name}',
            hoverinfo='text',
        ))

    # ------------------------------------------------------------------
    # 4. Vetítési pontok (végpontok + elágazások régiónként)
    # ------------------------------------------------------------------
    for tr in result.target_results:
        pts = proj_idx[point_regions[proj_idx] == tr.region_id]
        if len(pts) == 0:
            continue
        color = region_color_map[tr.region_id]
        traces.append(go.Scatter3d(
            x=x[pts], y=y[pts], z=z[pts],
            mode='markers',
            marker=dict(size=5, color=color, symbol='diamond',
                        line=dict(color='white', width=0.5)),
            name=f'Proj. pts: {tr.region_name}',
            hovertext=[f'{tr.region_name}<br>ep/branch' for _ in pts],
            hoverinfo='text',
        ))

    # ------------------------------------------------------------------
    # 5. Figure összerakása
    # ------------------------------------------------------------------
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=f'<b>{cell_name}</b>  |  Soma: {result.soma_region_name}',
            font=dict(size=13, color='#33401F'),
            x=0.01,
        ),
        scene=dict(
            xaxis=dict(title='X (µm)', backgroundcolor='#f8f8f8',
                       gridcolor='#dddddd', showbackground=True),
            yaxis=dict(title='Y (µm)', backgroundcolor='#f8f8f8',
                       gridcolor='#dddddd', showbackground=True),
            zaxis=dict(title='Z (µm)', backgroundcolor='#f8f8f8',
                       gridcolor='#dddddd', showbackground=True),
            aspectmode='data',  # arányos tengelyek – nem nyújtja el az agyat
        ),
        legend=dict(
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor='#CFD6BC',
            borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor='white',
        height=650,
    )

    return fig


def build_3d_plot_multi(
    results: list[tuple[str, CellAnalysisResult]],
    atlas_matrix: np.ndarray,
    target_region_ids: list[int],
    show_target_regions: bool = True,
) -> go.Figure:
    """
    Több sejtet jelenít meg egyszerre – batch összehasonlításhoz.

    Minden sejt saját színt kap (axon + soma), a célterületek
    félig átlátszó felszínként jelennek meg háttérként.

    Args:
        results:             lista (sejt_név, CellAnalysisResult) párokból
        atlas_matrix:        Allen Brain Atlas 3D mátrixa
        target_region_ids:   a vizsgált célterületek ID listája
        show_target_regions: megjelenítse-e a célterületek felszíneit

    Returns:
        go.Figure – st.plotly_chart()-tal közvetlenül megjeleníthető
    """
    palette = COLORS['region_palette']

    # Régió nevek az első eredményből (mindegyik ugyanazokkal futott)
    region_names: dict[int, str] = {}
    if results:
        for tr in results[0][1].target_results:
            region_names[tr.region_id] = tr.region_name

    traces: list = []

    # ------------------------------------------------------------------
    # 1. Célterület felszínek (egyszer épülnek fel)
    # ------------------------------------------------------------------
    if show_target_regions:
        for i, region_id in enumerate(target_region_ids):
            color = _get_region_color(i)
            name = region_names.get(region_id, f'Region {region_id}')
            t = _build_mesh_trace(
                atlas_matrix == region_id,
                color=color,
                opacity=VIZ_REGION_OPACITY * 0.55,
                name=name,
            )
            if t:
                traces.append(t)

    # ------------------------------------------------------------------
    # 2. Minden sejt saját szín
    # ------------------------------------------------------------------
    for i, (cell_name, result) in enumerate(results):
        cell_color = palette[i % len(palette)]
        coords = result.coords
        x, y, z = coords['x'], coords['y'], coords['z']
        is_axon            = coords['is_axon']
        curr_idx           = coords['curr_idx']
        parent_row_indices = coords['parent_row_indices']
        soma_idx           = coords['soma_idx']
        point_regions      = coords['point_regions']

        # Egyforma szín az egész sejthez (nincs régió-alapú színezés batch-ben)
        uniform_color_map = {int(rid): cell_color for rid in np.unique(point_regions)}

        axon_traces = _build_axon_trace(
            x, y, z, curr_idx, parent_row_indices,
            is_axon, point_regions, uniform_color_map,
            line_width=1,
        )
        # A legendában csak az első trace-nek legyen neve (különben minden
        # szín-variáns megjelenne és elárasztaná a legendát)
        for j, tr in enumerate(axon_traces):
            if j == 0:
                tr.showlegend = True
                tr.name = cell_name
            traces.append(tr)

        # Soma jelölő
        if soma_idx is not None:
            traces.append(go.Scatter3d(
                x=[x[soma_idx]], y=[y[soma_idx]], z=[z[soma_idx]],
                mode='markers',
                marker=dict(size=7, color=cell_color, symbol='circle',
                            line=dict(color='white', width=1)),
                showlegend=False,
                hovertext=f'{cell_name}<br>Soma: {result.soma_region_name}',
                hoverinfo='text',
            ))

    # ------------------------------------------------------------------
    # 3. Figure
    # ------------------------------------------------------------------
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=f'<b>Combined view</b>  —  {len(results)} cells',
            font=dict(size=13, color='#33401F'),
            x=0.01,
        ),
        scene=dict(
            xaxis=dict(title='X (µm)', backgroundcolor='#f8f8f8',
                       gridcolor='#dddddd', showbackground=True),
            yaxis=dict(title='Y (µm)', backgroundcolor='#f8f8f8',
                       gridcolor='#dddddd', showbackground=True),
            zaxis=dict(title='Z (µm)', backgroundcolor='#f8f8f8',
                       gridcolor='#dddddd', showbackground=True),
            aspectmode='data',
        ),
        legend=dict(
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor='#CFD6BC',
            borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor='white',
        height=700,
    )

    return fig


# =============================================================================
# STREAMLIT RENDERELŐ
# =============================================================================

def render_plot_streamlit(fig: go.Figure, key: str) -> None:
    """
    Megjeleníti a Plotly Figure-t a Streamlit oldalon.

    use_container_width=True: kitölti a rendelkezésre álló szélességet.
    A key nem kötelező Plotly-nál (nincs widget cache bug mint stpyvista-nál),
    de megtartjuk a konzisztencia kedvéért.

    Args:
        fig: a build_3d_plot() vagy build_3d_plot_multi() által visszaadott Figure
        key: egyedi kulcs (opcionális, de jó szokás)
    """
    import streamlit as st
    st.plotly_chart(fig, use_container_width=True, key=key)


def show_plot_local(fig: go.Figure) -> None:
    """
    Lokálisan nyitja meg az ábrát a böngészőben (fejlesztéshez).
    Szerveren vagy Streamlit-ben NEM hívjuk ezt – ott render_plot_streamlit() kell.
    """
    fig.show()