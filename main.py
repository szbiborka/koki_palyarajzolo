import os
import numpy as np
import pandas as pd
import nrrd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

# Fájl útvonalak
alap_mappa_utvonal = '/home/bibi/Documents/koki/swc_in_ccf/data_v2/'
szotar_fajl = '/home/bibi/Documents/koki/query.csv'
atlas_fajl = '/home/bibi/Documents/koki/annotation_25.nrrd'

# Célterületek ID-jai
target_gpe_id = 1022  # Globus Pallidus external segment
target_rt_id = 262  # Reticular nucleus of the thalamus (TRN)

# Vizsgált sejt megadása
eger_mappa = '221227'
sejt_fajl = '241.swc'
voxel_size = 25

print('Szótár és Atlasz betöltése...')

# Szótár beolvasása (Pandas)
szotar = pd.read_csv(szotar_fajl, usecols=['id', 'acronym', 'safe_name'])

# Atlasz betöltése (nrrd)
atlas_matrix, header = nrrd.read(atlas_fajl)
max_x, max_y, max_z = atlas_matrix.shape

# Fájl ellenőrzése
filename = os.path.join(alap_mappa_utvonal, eger_mappa, sejt_fajl)
if not os.path.isfile(filename):
    raise FileNotFoundError(f'Hiba! Nem található a fájl: {filename}')
print(f'Fájl betöltve: {eger_mappa} / {sejt_fajl}')

# SWC beolvasása (Pandas segítségével, NaN-ok kihagyásával)
# Oszlopok: 0:id, 1:type, 2:x, 3:y, 4:z, 5:radius, 6:pid
swc = pd.read_csv(filename, sep=r'\s+', comment='#', header=None,
                  names=['id', 'type', 'x', 'y', 'z', 'radius', 'pid'])

swc = swc.dropna(subset=['id', 'type', 'x', 'y', 'z', 'pid'])

# Adatok kinyerése numpy array-be
swc_id = np.round(swc['id'].values).astype(int)
swc_type = np.round(swc['type'].values).astype(int)
x = swc['x'].values
y = swc['y'].values
z = swc['z'].values
pid = np.round(swc['pid'].values).astype(int)

# Ismétlődések kiszedése (megtartva az utolsót, mint a MATLAB 'last' paramétere)
_, unq_idx = np.unique(swc_id[::-1], return_index=True)
unq_idx = len(swc_id) - 1 - unq_idx  # Visszafordítás az eredeti sorrendre
swc_id, swc_type, x, y, z, pid = swc_id[unq_idx], swc_type[unq_idx], x[unq_idx], y[unq_idx], z[unq_idx], pid[unq_idx]

# Koordináták atlasz voxelre alakítása (Python 0-indexelés miatt nincs +1)
vox_x = np.clip(np.round(x / voxel_size).astype(int), 0, max_x - 1)
vox_y = np.clip(np.round(y / voxel_size).astype(int), 0, max_y - 1)
vox_z = np.clip(np.round(z / voxel_size).astype(int), 0, max_z - 1)

# Régiók kinyerése (Pythonban közvetlenül indexelhető a 3D mátrix)
point_regions = atlas_matrix[vox_x, vox_y, vox_z]

# Soma keresése
soma_indices = np.where(swc_type == 1)[0]
soma_idx = soma_indices[0] if len(soma_indices) > 0 else None

soma_name = "Ismeretlen régió"
soma_region_id = -1
if soma_idx is not None:
    soma_region_id = point_regions[soma_idx]
    soma_match = szotar[szotar['id'] == soma_region_id]
    if not soma_match.empty:
        soma_name = soma_match['safe_name'].values[0]

print('\n========================================================')
print(' EREDMÉNYEK:')
print('========================================================')
print('1. SOMA LOKÁCIÓJA:')
print(f'   Régió neve: {soma_name} (ID: {soma_region_id})\n')

# Kapcsolatok megkeresése (Parent mapping)
id_to_index = {val: idx for idx, val in enumerate(swc_id)}
parent_row_indices = np.array([id_to_index.get(p, -1) for p in pid])

valid_connections = (parent_row_indices != -1) & (pid != -1)

# Gyerekek számolása
p_rows = parent_row_indices[valid_connections]
child_counts = np.bincount(p_rows, minlength=len(swc_id))

# Axon logika (2-es vagy 0-ás típus)
is_axon = (swc_type == 2) | (swc_type == 0)

ep_idx = np.where((child_counts == 0) & is_axon)[0]
branch_idx = np.where((child_counts > 1) & is_axon)[0]
proj_idx = np.union1d(ep_idx, branch_idx)
proj_regions = point_regions[proj_idx]

# Axonhossz számítása Pitagorasz-tétellel
curr_idx = np.where(valid_connections)[0]
p_idx = parent_row_indices[curr_idx]

dx = x[curr_idx] - x[p_idx]
dy = y[curr_idx] - y[p_idx]
dz = z[curr_idx] - z[p_idx]
d = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

axon_mask = (swc_type[curr_idx] == 2) | (swc_type[curr_idx] == 0)

# GPe vetítés
proj_in_gpe = np.sum(proj_regions == target_gpe_id)
len_gpe = np.sum(d[axon_mask & (point_regions[curr_idx] == target_gpe_id)])

print('2. VETÍT-E A GPe-BE (Globus Pallidus external segment)?')
if proj_in_gpe > 0:
    print('   IGEN!')
    print(f'   Érvényes vetítési pontok: {proj_in_gpe} db')
    print(f'   Axonhossz a GPe-ben: {len_gpe:.1f} um\n')
else:
    print('   NEM. (Csak áthalad, vagy nincs ott axon)\n')

# RT vetítés
proj_in_rt = np.sum(proj_regions == target_rt_id)
len_rt = np.sum(d[axon_mask & (point_regions[curr_idx] == target_rt_id)])

print('3. VETÍT-E AZ RT-BE (Reticular nucleus of the thalamus)?')
if proj_in_rt > 0:
    print('   IGEN!')
    print(f'   Érvényes vetítési pontok: {proj_in_rt} db')
    print(f'   Axonhossz az RT-ben: {len_rt:.1f} um\n')
else:
    print('   NEM. (Csak áthalad, vagy nincs ott axon)\n')

# Egyéb célterületek
unique_proj_regions = np.unique(proj_regions[proj_regions > 0])
other_targets = [r for r in unique_proj_regions if r not in (target_gpe_id, target_rt_id, soma_region_id)]

print('4. EGYÉB CÉLTERÜLETEK (Ahol van végpont/elágazás):')
if not other_targets:
    print('   Nincs más célterület.')
else:
    for r_id in other_targets:
        match = szotar[szotar['id'] == r_id]
        r_name = match['safe_name'].values[0] if not match.empty else "Ismeretlen"
        pts_here = np.sum(proj_regions == r_id)
        print(f'   - {r_name} (Pontok: {pts_here} db)')
print('========================================================\n')

# --- 3D ÁBRA GENERÁLÁSA ---
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# --- 3D ÁBRA GENERÁLÁSA ---
print('3D ábra generálása...')
# Fekete háttér beállítása, mint a MATLAB-ban
fig = plt.figure(figsize=(12, 8), facecolor='#111111')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('#111111')

ax.set_title(f'Sejt: {eger_mappa} / {sejt_fajl}\nSoma: {soma_name}', color='white')


# Segédfüggvény a régiók 3D-s kirajzolásához
def plot_region(ax, matrix, target_id, color, alpha=0.3):
    mask = (matrix == target_id)
    if np.any(mask):
        verts, faces, normals, values = marching_cubes(mask, level=0.5)
        # JAVÍTÁS: Kivettük a tengelycserét, mert a numpy eleve jól (X, Y, Z) adja át!
        verts = verts * voxel_size
        mesh = Poly3DCollection(verts[faces], alpha=alpha)
        mesh.set_facecolor(color)
        mesh.set_edgecolor('none')
        ax.add_collection3d(mesh)


# Régiók rajzolása
plot_region(ax, atlas_matrix, soma_region_id, [0.7, 0.2, 0.2], alpha=0.4)  # Piros
plot_region(ax, atlas_matrix, target_gpe_id, [0.2, 0.4, 0.8], alpha=0.4)  # Kék
plot_region(ax, atlas_matrix, target_rt_id, [0.2, 0.7, 0.2], alpha=0.4)  # Zöld

# --- ÚJ, GYORSABB VONALRAJZOLÁS (Line3DCollection) ---
segments = []
colors = []
linewidths = []

for i in range(len(swc_id)):
    if valid_connections[i]:
        p_row = parent_row_indices[i]
        # Vonal szakasz hozzáadása (jelenlegi pont -> szülő pont)
        segments.append([[x[i], y[i], z[i]], [x[p_row], y[p_row], z[p_row]]])

        region_id = point_regions[i]
        is_axon = (swc_type[i] in (0, 2))

        # Színek és vastagságok beállítása
        if is_axon and region_id == target_gpe_id:
            colors.append('blue')
            linewidths.append(1.5)
        elif is_axon and region_id == target_rt_id:
            colors.append('lime')
            linewidths.append(1.5)
        else:
            colors.append('lightgrey')  # Itt rajzoljuk a szürke "egyéb" ágakat
            linewidths.append(0.5)

# Vonalak egyidejű, gyors renderelése
lc = Line3DCollection(segments, colors=colors, linewidths=linewidths)
ax.add_collection3d(lc)

# Soma kiemelése
if soma_idx is not None:
    ax.scatter(x[soma_idx], y[soma_idx], z[soma_idx], s=100, c='black', edgecolors='white', zorder=5)

# Vetítési pontok kiemelése
gpe_pts = proj_idx[point_regions[proj_idx] == target_gpe_id]
if len(gpe_pts) > 0:
    ax.scatter(x[gpe_pts], y[gpe_pts], z[gpe_pts], s=30, c='blue', edgecolors='white', zorder=5)

rt_pts = proj_idx[point_regions[proj_idx] == target_rt_id]
if len(rt_pts) > 0:
    ax.scatter(x[rt_pts], y[rt_pts], z[rt_pts], s=30, c='lime', edgecolors='white', zorder=5)

# Tengelyek és nézet beállítása
ax.set_xlabel('X (um) - Orr balra, Farok jobbra', color='white')
ax.set_ylabel('Y (um) - Hát felül, Has alul', color='white')
ax.set_zlabel('Z (um) - Bal/Jobb agyfélteke', color='white')

# Tengelyek színének fehéredítése a sötét háttér miatt
ax.xaxis.label.set_color('white');
ax.tick_params(axis='x', colors='white')
ax.yaxis.label.set_color('white');
ax.tick_params(axis='y', colors='white')
ax.zaxis.label.set_color('white');
ax.tick_params(axis='z', colors='white')

# Limitek automatikus beállítása a sejtre
ax.set_xlim([np.min(x) - 500, np.max(x) + 500])
ax.set_ylim([np.min(y) - 500, np.max(y) + 500])
ax.set_zlim([np.min(z) - 500, np.max(z) + 500])

ax.invert_yaxis()  # Y tengely megfordítása, ahogy a MATLAB kódban volt
ax.view_init(elev=90, azim=-90)  # Felülnézet beállítása a 2D-s "view(2)" hatásért

plt.show()
print('Kész!')