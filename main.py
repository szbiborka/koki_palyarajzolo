import os
import numpy as np
import pandas as pd
import nrrd
import matplotlib

matplotlib.use('TkAgg')  # Interaktív ablak beállítása
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from skimage.measure import marching_cubes

# --- 1. VÁLTOZÓK ÉS FÁJLOK ---
alap_mappa_utvonal = '/home/bibi/Documents/koki/swc_in_ccf/data_v2/'
szotar_fajl = '/home/bibi/Documents/koki/query.csv'
atlas_fajl = '/home/bibi/Documents/koki/annotation_25.nrrd'

# Célterületek ID-jai
target_gpe_id = 1022
target_rt_id = 262

# Vizsgált sejt megadása
eger_mappa = '221227'
sejt_fajl = '241.swc'
voxel_size = 25

print('Szótár és Atlasz betöltése...')

opts = {'usecols': ['id', 'acronym', 'safe_name']}
szotar = pd.read_csv(szotar_fajl, **opts)

# Python pynrrd modulja alapból ugyanabban a formátumban olvassa be a mátrixot, mint a MATLAB
atlas_matrix, _ = nrrd.read(atlas_fajl)
max_x, max_y, max_z = atlas_matrix.shape

filename = os.path.join(alap_mappa_utvonal, eger_mappa, sejt_fajl)
if not os.path.isfile(filename):
    raise FileNotFoundError(f'Hiba! Nem található a fájl: {filename}')
print(f'Fájl betöltve: {eger_mappa} / {sejt_fajl}')

# --- 2. SWC BEOLVASÁSA ÉS TISZTÍTÁSA ---
swc = pd.read_csv(filename, sep=r'\s+', comment='#', header=None,
                  names=['id', 'type', 'x', 'y', 'z', 'radius', 'pid'])
swc = swc.dropna(subset=['id', 'type', 'x', 'y', 'z', 'pid'])

id_arr = np.round(swc['id'].values).astype(int)
type_arr = np.round(swc['type'].values).astype(int)
x = swc['x'].values
y = swc['y'].values
z = swc['z'].values
pid_arr = np.round(swc['pid'].values).astype(int)

# Ismétlődések kiszedése (a MATLAB 'last' paraméterével megegyezően)
_, unq_idx = np.unique(id_arr[::-1], return_index=True)
unq_idx = len(id_arr) - 1 - unq_idx
id_arr, type_arr = id_arr[unq_idx], type_arr[unq_idx]
x, y, z, pid_arr = x[unq_idx], y[unq_idx], z[unq_idx], pid_arr[unq_idx]

# --- 3. KOORDINÁTÁK ÉS RÉGIÓK ---
vox_x = np.clip(np.round(x / voxel_size).astype(int), 0, max_x - 1)
vox_y = np.clip(np.round(y / voxel_size).astype(int), 0, max_y - 1)
vox_z = np.clip(np.round(z / voxel_size).astype(int), 0, max_z - 1)

# MATLAB sub2ind megfelelője: közvetlen 3D indexelés
point_regions = atlas_matrix[vox_x, vox_y, vox_z]

soma_indices = np.where(type_arr == 1)[0]
soma_idx = soma_indices[0] if len(soma_indices) > 0 else None

soma_region_id = point_regions[soma_idx] if soma_idx is not None else -1
soma_match = szotar[szotar['id'] == soma_region_id]
soma_name = soma_match['safe_name'].values[0] if not soma_match.empty else "Ismeretlen régió"

# --- 4. SZÁMOLÁSOK ÉS EREDMÉNYEK KIÍRÁSA ---
print('\n========================================================')
print(' EREDMÉNYEK:')
print('========================================================')
print('1. SOMA LOKÁCIÓJA:')
print(f'   Régió neve: {soma_name} (ID: {soma_region_id})\n')

id_to_index = {val: idx for idx, val in enumerate(id_arr)}
parent_row_indices = np.array([id_to_index.get(p, -1) for p in pid_arr])
valid_connections = (parent_row_indices != -1) & (pid_arr != -1)

p_rows = parent_row_indices[valid_connections]
child_counts = np.bincount(p_rows, minlength=len(id_arr))

is_axon = (type_arr == 2) | (type_arr == 0)
ep_idx = np.where((child_counts == 0) & is_axon)[0]
branch_idx = np.where((child_counts > 1) & is_axon)[0]
proj_idx = np.union1d(ep_idx, branch_idx)
proj_regions = point_regions[proj_idx]

curr_idx = np.where(valid_connections)[0]
p_idx = parent_row_indices[curr_idx]

dx = x[curr_idx] - x[p_idx]
dy = y[curr_idx] - y[p_idx]
dz = z[curr_idx] - z[p_idx]
d = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

axon_mask = (type_arr[curr_idx] == 2) | (type_arr[curr_idx] == 0)

# GPe
proj_in_gpe = np.sum(proj_regions == target_gpe_id)
len_gpe = np.sum(d[axon_mask & (point_regions[curr_idx] == target_gpe_id)])

print('2. VETÍT-E A GPe-BE (Globus Pallidus external segment)?')
if proj_in_gpe > 0:
    print('   IGEN!')
    print(f'   Érvényes vetítési pontok (végpont/elágazás): {proj_in_gpe} db')
    print(f'   Axonhossz a GPe-ben: {len_gpe:.1f} um\n')
else:
    print('   NEM. (Csak áthalad, vagy nincs ott axon)\n')

# RT
proj_in_rt = np.sum(proj_regions == target_rt_id)
len_rt = np.sum(d[axon_mask & (point_regions[curr_idx] == target_rt_id)])

print('3. VETÍT-E AZ RT-BE (Reticular nucleus of the thalamus)?')
if proj_in_rt > 0:
    print('   IGEN!')
    print(f'   Érvényes vetítési pontok (végpont/elágazás): {proj_in_rt} db')
    print(f'   Axonhossz az RT-ben: {len_rt:.1f} um\n')
else:
    print('   NEM. (Csak áthalad, vagy nincs ott axon)\n')

# Egyéb célterületek
unique_proj_regions = np.unique(proj_regions[proj_regions > 0])
other_targets = [r for r in unique_proj_regions if r not in (target_gpe_id, target_rt_id, soma_region_id)]

print('4. EGYÉB CÉLTERÜLETEK (Ahol van végpont/elágazás):')
if len(other_targets) == 0:
    print('   Nincs más célterület.\n')
else:
    for r_id in other_targets:
        match = szotar[szotar['id'] == r_id]
        r_name = match['safe_name'].values[0] if not match.empty else "Ismeretlen"
        pts_here = np.sum(proj_regions == r_id)
        print(f'   - {r_name} (Pontok: {pts_here} db)')
print('========================================================')

# --- 5. 3D ÁBRA GENERÁLÁSA ---
print('3D ábra generálása...')
fig = plt.figure(figsize=(10, 8), facecolor='w')  # Fehér háttér, ahogy a MATLAB kódban: figure('Color', 'w')
ax = fig.add_subplot(111, projection='3d', facecolor='w')

# Valódi szagitális nézet az X-Y síkon (MATLAB: view(2), YDir reverse)
ax.view_init(elev=90, azim=-90)
ax.invert_yaxis()

ax.set_xlabel('X (\u03bcm) - Orr balra, Farok jobbra')
ax.set_ylabel('Y (\u03bcm) - Hát felül, Has alul')
ax.set_zlabel('Z (\u03bcm) - Bal/Jobb agyfélteke (mélység)')
ax.set_title(f'Sejt: {eger_mappa} / {sejt_fajl}\nSoma: {soma_name}')


# Segédfüggvény patch/isosurface generáláshoz
def plot_region(target_id, color):
    mask = (atlas_matrix == target_id)
    if np.any(mask):
        verts, faces, normals, values = marching_cubes(mask, level=0.5)
        verts = verts * voxel_size  # Nem forgatunk meg semmit, tisztán a beolvasott adatot szorozzuk
        mesh = Poly3DCollection(verts[faces], alpha=0.2)
        mesh.set_facecolor(color)
        mesh.set_edgecolor('none')
        ax.add_collection3d(mesh)


# Régiók rajzolása (Soma piros, GPe kék, RT zöld)
plot_region(soma_region_id, [1.0, 0.2, 0.2])
plot_region(target_gpe_id, [0.2, 0.5, 1.0])
plot_region(target_rt_id, [0.2, 0.8, 0.2])

# Axonok rajzolása (KIZÁRÓLAG a GPe és RT területeken belül, ahogy a MATLAB kód írja)
segments = []
colors = []
for i in range(len(id_arr)):
    if valid_connections[i] and type_arr[i] in (2, 0):
        p_row = parent_row_indices[i]
        reg = point_regions[i]

        if reg == target_gpe_id:
            segments.append([[x[i], y[i], z[i]], [x[p_row], y[p_row], z[p_row]]])
            colors.append([0, 0, 1])  # Kék
        elif reg == target_rt_id:
            segments.append([[x[i], y[i], z[i]], [x[p_row], y[p_row], z[p_row]]])
            colors.append([0, 0.8, 0])  # Zöld

if segments:
    # Sokkal gyorsabb kirajzolás, mint a for ciklus plot3-mal
    lc = Line3DCollection(segments, colors=colors, linewidths=1.5)
    ax.add_collection3d(lc)

# Pöttyök (Soma, és vetítési végpontok)
if soma_idx is not None:
    ax.scatter(x[soma_idx], y[soma_idx], z[soma_idx], s=100, c='k', edgecolors='w', zorder=5)

gpe_pts = proj_idx[point_regions[proj_idx] == target_gpe_id]
if len(gpe_pts) > 0:
    ax.scatter(x[gpe_pts], y[gpe_pts], z[gpe_pts], s=40, c='b', edgecolors='w', zorder=5)

rt_pts = proj_idx[point_regions[proj_idx] == target_rt_id]
if len(rt_pts) > 0:
    ax.scatter(x[rt_pts], y[rt_pts], z[rt_pts], s=40, c='g', edgecolors='w', zorder=5)

# MATLAB "axis equal" megfelelője (méretarányos tér)
try:
    ax.set_box_aspect([1, 1, 1])
except AttributeError:
    pass

# Automatikus dobozméretezés a sejt kiterjedése alapján
if len(x) > 0:
    ax.set_xlim([np.min(x) - 500, np.max(x) + 500])
    ax.set_ylim([np.min(y) - 500, np.max(y) + 500])
    ax.set_zlim([np.min(z) - 500, np.max(z) + 500])

plt.show()
print('Kész!')