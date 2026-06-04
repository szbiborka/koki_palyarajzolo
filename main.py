import os
import numpy as np
import pandas as pd
import nrrd
import pyvista as pv

from skimage.measure import marching_cubes

# ==========================================
# 1. PARAMÉTEREK ÉS ELÉRHETŐSÉGEK
# ==========================================
alap_mappa_utvonal = '/home/bibi/Documents/koki/swc_in_ccf/data_v2/'
szotar_fajl = '/home/bibi/Documents/koki/query.csv'
atlas_fajl = '/home/bibi/Documents/koki/annotation_25.nrrd'

# Célterületek ID-jai (Allen atlaszból kinezve)
target_gpe_id = 1022  # Globus Pallidus external segment
target_rt_id = 262  # Reticular nucleus of the thalamus (TRN)

# Vizsgált sejt megadása
eger_mappa = '221227'
sejt_fajl = '241.swc'
voxel_size = 25
# voxel = 3d pixel

# ==========================================
# 2. ADATOK BETÖLTÉSE
# ==========================================
print('Szótár és Atlasz betöltése...')
# Szótár beolvasása pandas segítségével (csak a megadott oszlopok)
szotar = pd.read_csv(szotar_fajl, usecols=['id', 'acronym', 'safe_name'])

# Atlasz (nrrd) beolvasása
atlas_matrix, header = nrrd.read(atlas_fajl)
max_x, max_y, max_z = atlas_matrix.shape

filename = os.path.join(alap_mappa_utvonal, eger_mappa, sejt_fajl)
if not os.path.isfile(filename):
    raise FileNotFoundError(f'Hiba! Nem található a fájl: {filename}')

print(f'Fájl betöltve: {eger_mappa} / {sejt_fajl}')

# SWC beolvasása (kommentek '#' ignorálása)
# Az swc oszlopai: id, type, x, y, z, radius, parent_id
# SWC beolvasása (kommentek '#' ignorálása)
# Az swc oszlopai: id, type, x, y, z, radius, parent_id
swc_df = pd.read_csv(filename, comment='#', sep=r'\s+', header=None,
                     names=['id', 'type', 'x', 'y', 'z', 'radius', 'pid'])

# ==========================================
# 3. ADATTISZTÍTÁS ÉS FELDOLGOZÁS
# ==========================================
# NaN adatok kiszűrése (ha lenne)
swc_df = swc_df.dropna(subset=['id', 'x', 'y', 'z', 'pid'])

# Ismétlődések kiszedése ('last' logika MATLAB-ból)
swc_df = swc_df.drop_duplicates(subset=['id'], keep='last').reset_index(drop=True)

id_arr = np.round(swc_df['id'].values).astype(int)
type_arr = np.round(swc_df['type'].values).astype(int)
x = swc_df['x'].values
y = swc_df['y'].values
z = swc_df['z'].values
pid_arr = np.round(swc_df['pid'].values).astype(int)

# Térbeli koordináták konvertálása voxel indexekké (Python 0-tól indexel!)
# A klippelés (clip) megakadályozza a mátrix túlindexelését
vox_x = np.clip(np.round(x / voxel_size).astype(int), 0, max_x - 1)
vox_y = np.clip(np.round(y / voxel_size).astype(int), 0, max_y - 1)
vox_z = np.clip(np.round(z / voxel_size).astype(int), 0, max_z - 1)

# Eltárolja, hogy a sejt minden pontja melyik régióba esik (Pythonban ez a sima indexelés)
point_regions = atlas_matrix[vox_x, vox_y, vox_z]

# Soma megkeresése (első type == 1)
soma_idx_arr = np.where(type_arr == 1)[0]
soma_idx = soma_idx_arr[0] if len(soma_idx_arr) > 0 else None

if soma_idx is not None:
    soma_region_id = point_regions[soma_idx]
    # Pandas Series konvertálása Python listává
    soma_name_match = szotar.loc[szotar['id'] == soma_region_id, 'safe_name'].tolist()
    # Lista ellenőrzése (ha van benne elem, az True-ra értékelődik)
    soma_name = soma_name_match[0] if soma_name_match else "Ismeretlen régió"
else:
    soma_region_id = -1
    soma_name = "Nincs soma"

# ==========================================
# 4. HÁLÓZATI (SZÜLŐ-GYERMEK) LOGIKA
# ==========================================
# Létrehozunk egy szótárt az ID -> index gyors kereséshez (ismember helyett)
id_to_idx = {val: idx for idx, val in enumerate(id_arr)}

# Megkeressük minden pont szülőjének az indexét
parent_row_indices = np.array([id_to_idx.get(p, -1) for p in pid_arr])
valid_connections = (parent_row_indices != -1) & (pid_arr != -1)

# Kiszámoljuk a gyerekek számát minden csomópontra (accumarray helyett bincount)
p_rows = parent_row_indices[valid_connections]
child_counts = np.bincount(p_rows, minlength=len(id_arr))

# Axon logika (2-es és 0-ás)
is_axon = (type_arr == 2) | (type_arr == 0)

# Végpontok (0 gyerek) és elágazások (>1 gyerek)
ep_idx = np.where((child_counts == 0) & is_axon)[0]
branch_idx = np.where((child_counts > 1) & is_axon)[0]
proj_idx = np.union1d(ep_idx, branch_idx)
proj_regions = point_regions[proj_idx]

# Axonhossz számítása (Pitagorasz) vektorizáltan
curr_idx = np.where(valid_connections)[0]
p_idx = parent_row_indices[curr_idx]
dx = x[curr_idx] - x[p_idx]
dy = y[curr_idx] - y[p_idx]
dz = z[curr_idx] - z[p_idx]
distances = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

axon_mask_curr = (type_arr[curr_idx] == 2) | (type_arr[curr_idx] == 0)

# ==========================================
# 5. EREDMÉNYEK KIÍRÁSA
# ==========================================
print('\n========================================================')
print(' EREDMÉNYEK:')
print('========================================================')
print('1. SOMA LOKÁCIÓJA:')
print(f'   Régió neve: {soma_name} (ID: {soma_region_id})\n')

# GPe vizsgálata
proj_in_gpe = np.sum(proj_regions == target_gpe_id)
len_gpe = np.sum(distances[axon_mask_curr & (point_regions[curr_idx] == target_gpe_id)])

print('2. VETÍT-E A GPe-BE (Globus Pallidus external segment)?')
if proj_in_gpe > 0:
    print('   IGEN!')
    print(f'   Érvényes vetítési pontok (végpont/elágazás): {proj_in_gpe} db')
    print(f'   Axonhossz a GPe-ben: {len_gpe:.1f} um\n')
else:
    print('   NEM. (Csak áthalad, vagy nincs ott axon)\n')

# RT (TRN) vizsgálata
proj_in_rt = np.sum(proj_regions == target_rt_id)
len_rt = np.sum(distances[axon_mask_curr & (point_regions[curr_idx] == target_rt_id)])

print('3. VETÍT-E AZ RT-BE (Reticular nucleus of the thalamus)?')
if proj_in_rt > 0:
    print('   IGEN!')
    print(f'   Érvényes vetítési pontok (végpont/elágazás): {proj_in_rt} db')
    print(f'   Axonhossz az RT-ben: {len_rt:.1f} um\n')
else:
    print('   NEM. (Csak áthalad, vagy nincs ott axon)\n')

# Egyéb célterületek
unique_proj_regions = np.unique(proj_regions[proj_regions > 0])
other_targets = unique_proj_regions[(unique_proj_regions != target_gpe_id) &
                                    (unique_proj_regions != target_rt_id) &
                                    (unique_proj_regions != soma_region_id)]

print('4. EGYÉB CÉLTERÜLETEK (Ahol van végpont/elágazás):')
if len(other_targets) == 0:
    print('   Nincs más célterület.')
else:
    for r_id in other_targets:
        # Pandas Series konvertálása Python listává
        r_name_match = szotar.loc[szotar['id'] == r_id, 'safe_name'].tolist()
        r_name = r_name_match[0] if r_name_match else "Ismeretlen"
        pts_here = np.sum(proj_regions == r_id)
        print(f'   - {r_name} (Pontok: {pts_here} db)')

print('========================================================')

# ==========================================
# 6. 3D VIZUALIZÁCIÓ (Matplotlib)
# ==========================================
# Cseréld le a fájl elején a matplotlib importokat erre:
# import pyvista as pv
# (A többi matplotlib importot törölheted, ha máshol nem használod)

# ==========================================
# 6. 3D VIZUALIZÁCIÓ (PyVista - GPU gyorsított)
# ==========================================
print('3D ábra generálása PyVista-val...')
#import pyvista as pv

plotter = pv.Plotter(window_size=[1024, 768])
plotter.set_background('white')


def add_isosurface_pv(plotter, mask, color, opacity_val):
    if np.any(mask):
        # step_size itt lehet kisebb (pl. 1 vagy 2), mert a PyVista bírja a terhelést!
        verts, faces, _, _ = marching_cubes(mask, level=0.5, step_size=2)
        verts = verts * voxel_size

        # PyVista speciális face formátuma: [n_points, p1, p2, p3, ...]
        # Készítünk egy (N, 1) alakú oszlopvektort csupa 3-asból
        padding = np.full((len(faces), 1), 3)

        # Vízszintesen összefűzzük a 3-asokat a csúcspont-indexekkel, majd kilapítjuk
        faces_pv = np.hstack((padding, faces)).flatten()

        mesh = pv.PolyData(verts, faces_pv)
        plotter.add_mesh(mesh, color=color, opacity=opacity_val, smooth_shading=True)


# 1. Régiók rajzolása (Gyönyörű, simított felületekkel)
add_isosurface_pv(plotter, (atlas_matrix == soma_region_id), color='red', opacity_val=0.3)
add_isosurface_pv(plotter, (atlas_matrix == target_gpe_id), color='blue', opacity_val=0.3)
add_isosurface_pv(plotter, (atlas_matrix == target_rt_id), color='green', opacity_val=0.3)

# 2. Axon vonalak (Szegmensek) rajzolása
# A PyVista-ban a sok kis vonal hozzáadása helyett hatékonyabb egy nagy vonalhálót építeni,
# de egyszerűség kedvéért egy ciklus is tökéletesen és gyorsan lefut.
for i in curr_idx:
    if is_axon[i]:
        p_row = parent_row_indices[i]

        # Kezdő és végpont koordinátái
        point_a = [x[i], y[i], z[i]]
        point_b = [x[p_row], y[p_row], z[p_row]]

        # Vonal létrehozása
        line = pv.Line(point_a, point_b)

        if point_regions[i] == target_gpe_id:
            plotter.add_mesh(line, color='blue', line_width=3)
        elif point_regions[i] == target_rt_id:
            plotter.add_mesh(line, color='green', line_width=3)

# 3. Pöttyök (Soma és vetítési pontok)
if soma_idx is not None:
    soma_sphere = pv.Sphere(radius=15, center=(x[soma_idx], y[soma_idx], z[soma_idx]))
    plotter.add_mesh(soma_sphere, color='black')

gpe_pts = proj_idx[point_regions[proj_idx] == target_gpe_id]
if len(gpe_pts) > 0:
    points = np.column_stack((x[gpe_pts], y[gpe_pts], z[gpe_pts]))
    plotter.add_points(points, color='blue', point_size=10, render_points_as_spheres=True)

rt_pts = proj_idx[point_regions[proj_idx] == target_rt_id]
if len(rt_pts) > 0:
    points = np.column_stack((x[rt_pts], y[rt_pts], z[rt_pts]))
    plotter.add_points(points, color='green', point_size=10, render_points_as_spheres=True)

# 4. Kamera és megjelenítés beállítása
# Az Y tengely megfordítása a kamerán keresztül
plotter.camera.up = (0, -1, 0)
plotter.show_axes()
plotter.add_title(f'Sejt: {eger_mappa} / {sejt_fajl}\nSoma: {soma_name}', font_size=12)

plotter.show()

print('Kész!')