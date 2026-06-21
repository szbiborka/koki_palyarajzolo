# =============================================================================
# VIZUALIZÁCIÓ MODUL - 3D ábra generálása PyVista-val.
# Jelenleg lokálisan fut (plotter.show()), de a to_trame() metódus
# egyszerűen cserélhető szerver-kompatibilis Trame megjelenítésre.
# =============================================================================

import numpy as np
import pyvista as pv
from skimage.measure import marching_cubes

from config import (
    VOXEL_SIZE, VIZ_REGION_OPACITY, VIZ_SOMA_RADIUS,
    VIZ_POINT_SIZE, VIZ_AXON_LINE_WIDTH, VIZ_MARCHING_CUBES_STEP, COLORS
)
from core.analysis import CellAnalysisResult


def _build_isosurface(mask: np.ndarray) -> pv.PolyData | None:
    """
    Marching cubes algoritmussal egy bináris maszkból 3D felszínt generál.
    Ha a maszk üres, None-t ad vissza.

    Args:
        mask: 3D bináris numpy tömb (True ahol a régió van)

    Returns:
        PyVista PolyData mesh, vagy None ha a maszk üres
    """
    if not np.any(mask):
        return None

    # A marching_cubes pixelkoordinátákban adja vissza a csúcsokat,
    # ezért szorzunk voxel_size-zal hogy mikrométeres koordinátákat kapjunk
    verts, faces, _, _ = marching_cubes(mask, level=0.5, step_size=VIZ_MARCHING_CUBES_STEP)
    verts = verts * VOXEL_SIZE

    # PyVista speciális face formátuma: [vertex_szám, p1, p2, p3, ...]
    padding = np.full((len(faces), 1), 3)
    faces_pv = np.hstack((padding, faces)).flatten()

    return pv.PolyData(verts, faces_pv)


def _get_region_color(region_index: int) -> str:
    """
    Körkörösen hozzárendel egy színt a régió indexe alapján a palettából.

    Args:
        region_index: a régió sorszáma (0-tól indexelt)

    Returns:
        Hex szín string
    """
    palette = COLORS['region_palette']
    return palette[region_index % len(palette)]


def build_3d_plot(
    result: CellAnalysisResult,
    atlas_matrix: np.ndarray,
    cell_name: str = "",
    show_soma_region: bool = True,
    show_other_regions: bool = True,
) -> pv.Plotter:
    """
    Felépíti a teljes 3D vizualizációt egy PyVista Plotter objektumba.
    A Plotter objektum ezután megjelenítható lokálisan (.show())
    vagy a jövőben Trame-en keresztül böngészőben is.

    Args:
        result: a run_analysis() által visszaadott eredmény
        atlas_matrix: az Allen Brain Atlas 3D mátrixa
        cell_name: a sejt neve a cím sorában
        show_soma_region: megjelenítse-e a soma régióját
        show_other_regions: megjelenítse-e az egyéb célterületeket

    Returns:
        Konfigurált pv.Plotter objektum
    """
    coords = result.coords
    x, y, z = coords['x'], coords['y'], coords['z']
    is_axon = coords['is_axon']
    point_regions = coords['point_regions']
    proj_idx = coords['proj_idx']
    curr_idx = coords['curr_idx']
    parent_row_indices = coords['parent_row_indices']
    soma_idx = coords['soma_idx']

    # Célterületek ID-i és hozzájuk rendelt színek szótára
    region_color_map: dict[int, str] = {}
    for i, tr in enumerate(result.target_results):
        region_color_map[tr.region_id] = _get_region_color(i)

    # --- Plotter inicializálása ---
    plotter = pv.Plotter(window_size=[1024, 768], off_screen=False)
    plotter.set_background('white')

    # --- 1. Agyterület felszínek (izoszfelszínek) ---
    # Soma régiója
    if show_soma_region and result.soma_region_id > 0:
        mesh = _build_isosurface(atlas_matrix == result.soma_region_id)
        if mesh:
            plotter.add_mesh(
                mesh, color='red', opacity=VIZ_REGION_OPACITY,
                smooth_shading=True, label=f'Soma region: {result.soma_region_name}'
            )

    # Célterületek felszínei
    for tr in result.target_results:
        color = region_color_map[tr.region_id]
        mesh = _build_isosurface(atlas_matrix == tr.region_id)
        if mesh:
            label = f'{tr.region_name} {"✓ projection" if tr.projects_here else "(no projection)"}'
            plotter.add_mesh(
                mesh, color=color, opacity=VIZ_REGION_OPACITY,
                smooth_shading=True, label=label
            )

    # Egyéb vetítési területek (ha be van kapcsolva)
    if show_other_regions:
        existing_ids = {tr.region_id for tr in result.target_results}
        existing_ids.add(result.soma_region_id)
        for i, other in enumerate(result.other_projection_regions):
            color = _get_region_color(len(result.target_results) + i)
            region_color_map[other.region_id] = color
            mesh = _build_isosurface(atlas_matrix == other.region_id)
            if mesh:
                plotter.add_mesh(
                    mesh, color=color, opacity=VIZ_REGION_OPACITY * 0.7,
                    smooth_shading=True, label=f'{other.region_name} (other target)'
                )

    # --- 2. Axon vonalak ---
    # A vonalakat összegyűjtjük és egyszerre adjuk hozzá a hatékonyság érdekében
    # Szín szerint csoportosítva (egy add_mesh hívás per szín)
    line_segments_by_color: dict[str, list] = {}

    for i in curr_idx:
        if not is_axon[i]:
            continue

        p_row = parent_row_indices[i]
        point_a = np.array([x[i], y[i], z[i]])
        point_b = np.array([x[p_row], y[p_row], z[p_row]])

        # Szín meghatározása: ha a pont egy célterületen van, annak a színe
        region = point_regions[i]
        color = region_color_map.get(int(region), COLORS['axon_default'])

        if color not in line_segments_by_color:
            line_segments_by_color[color] = []
        line_segments_by_color[color].append((point_a, point_b))

    # Hatékony vonalrajzolás: minden szín esetén egy PolyData objektum
    for color, segments in line_segments_by_color.items():
        if not segments:
            continue

        # PyVista vonal formátum: [2, idx_a, idx_b, 2, idx_c, idx_d, ...]
        points_list = []
        lines_list = []
        idx = 0
        for pt_a, pt_b in segments:
            points_list.extend([pt_a, pt_b])
            lines_list.extend([2, idx, idx + 1])
            idx += 2

        poly = pv.PolyData()
        poly.points = np.array(points_list)
        poly.lines = np.array(lines_list)
        plotter.add_mesh(poly, color=color, line_width=VIZ_AXON_LINE_WIDTH)

    # --- 3. Soma gömb ---
    if soma_idx is not None:
        soma_sphere = pv.Sphere(
            radius=VIZ_SOMA_RADIUS,
            center=(x[soma_idx], y[soma_idx], z[soma_idx])
        )
        plotter.add_mesh(soma_sphere, color=COLORS['soma'], label='Soma')

    # --- 4. Vetítési pontok (végpontok és elágazások) ---
    for tr in result.target_results:
        region_pts = proj_idx[point_regions[proj_idx] == tr.region_id]
        if len(region_pts) > 0:
            points = np.column_stack((x[region_pts], y[region_pts], z[region_pts]))
            color = region_color_map[tr.region_id]
            plotter.add_points(
                points, color=color,
                point_size=VIZ_POINT_SIZE,
                render_points_as_spheres=True
            )

    # --- 5. Kamera és tengelyek ---
    plotter.camera.up = (0, -1, 0)
    plotter.show_axes()
    plotter.add_legend(bcolor='white', border=True)

    title = f'Cell: {cell_name}\nSoma: {result.soma_region_name}'
    plotter.add_title(title, font_size=10)

    return plotter


def show_plot_local(plotter: pv.Plotter) -> None:
    """
    Lokális ablakban jeleníti meg az ábrát (fejlesztéshez, PyCharm-ban).
    Szerveren ezt a függvényt NEM hívjuk meg.
    """
    plotter.show()


# =============================================================================
# TRAME INTEGRÁCIÓ - TODO szerver-kompatibilis megjelenítéshez
# =============================================================================
# Amikor a szerverre kerül az alkalmazás, ezt a részt kell implementálni.
# A build_3d_plot() visszaadja a Plotter-t, és a Trame widget
# közvetlenül tudja használni azt.
#
# Példa (a jövőbeli app.py-ban):
#
#   from pyvista.trame.ui import plotter_ui
#   from trame.app import get_server
#
#   server = get_server()
#   plotter = build_3d_plot(result, atlas_matrix, cell_name)
#   with SinglePageLayout(server) as layout:
#       with layout.content:
#           with vuetify.VContainer():
#               view = plotter_ui(plotter)
#   server.start()
# =============================================================================
