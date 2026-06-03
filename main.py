import os
import numpy as np
import pandas as pd
import nrrd  # pip install pynrrd

# === BEÁLLÍTÁSOK ===
alap_mappa_utvonal = '/home/bibi/Documents/koki/swc_in_ccf/data_v2/'
szotar_fajl = '/home/bibi/Documents/koki/query.csv'
atlas_fajl = '/home/bibi/Documents/koki/annotation_25.nrrd'

# Célterületek ID-jai (Allen atlasból kinézve)
target_gpe_id = 1022  # Globus Pallidus external segment
target_rt_id = 262    # Reticular nucleus of the thalamus (TRN)

# Vizsgált sejt megadása
eger_mappa = '221227'
sejt_fajl = '241.swc'
voxel_size = 25

# === SZÓTÁR ÉS ATLASZ BETÖLTÉSE ===
print('Szótár és Atlasz betöltése...')

szotar = pd.read_csv(szotar_fajl, usecols=['id', 'acronym', 'safe_name'])

atlas_matrix, _ = nrrd.read(atlas_fajl)
max_x, max_y, max_z = atlas_matrix.shape

# === SWC FÁJL BETÖLTÉSE ===
filename = os.path.join(alap_mappa_utvonal, eger_mappa, sejt_fajl)
if not os.path.isfile(filename):
    raise FileNotFoundError(f'Hiba! Nem található a fájl: {filename}')

print(f'Fájl betöltve: {eger_mappa} / {sejt_fajl}')

# Beolvassa az SWC fájlt, kihagyva a #-es komment sorokat
swc = pd.read_csv(filename, sep=r'\s+', comment='#', header=None).values

# SWC mátrix szétszedése adatok szerint
id_   = np.round(swc[:, 0]).astype(float)
type_ = np.round(swc[:, 1]).astype(float)
x     = swc[:, 2].astype(float)
y     = swc[:, 3].astype(float)
z     = swc[:, 4].astype(float)
pid   = np.round(swc[:, 6]).astype(float)

# NaN adatok kiszűrése
valid_rows = (~np.isnan(id_) & ~np.isnan(x) & ~np.isnan(y) &
              ~np.isnan(z) & ~np.isnan(pid))
id_   = id_[valid_rows].astype(int)
type_ = type_[valid_rows].astype(int)
x     = x[valid_rows]
y     = y[valid_rows]
z     = z[valid_rows]
pid   = pid[valid_rows].astype(int)

# Ismétlődések kiszűrése (utolsó előfordulást tartja meg, mint MATLAB 'last')
_, unq_idx = np.unique(id_[::-1], return_index=True)
unq_idx = len(id_) - 1 - unq_idx  # visszafordítás az eredeti indexekre
unq_idx = np.sort(unq_idx)

id_   = id_[unq_idx]
type_ = type_[unq_idx]
x     = x[unq_idx]
y     = y[unq_idx]
z     = z[unq_idx]
pid   = pid[unq_idx]

# Koordináták átkonvertálása az egéragy voxel-koordinátáira
vox_x = np.clip(np.round(x / voxel_size).astype(int), 0, max_x - 1)
vox_y = np.clip(np.round(y / voxel_size).astype(int), 0, max_y - 1)
vox_z = np.clip(np.round(z / voxel_size).astype(int), 0, max_z - 1)

# (+1 MATLAB-ban 1-indexelt volt; Pythonban 0-indexelt, ezért nincs +1)
# A clip már 0-tól max-1-ig határol, ez ekvivalens a MATLAB min(max(...,1),max_x) logikával

# Atlas régió lekérdezése minden pontra
point_regions = atlas_matrix[vox_x, vox_y, vox_z]

# Soma megkeresése (első type==1 pont)
soma_candidates = np.where(type_ == 1)[0]
soma_idx = soma_candidates[0]

soma_region_id = point_regions[soma_idx]
soma_rows = szotar[szotar['id'] == soma_region_id]['safe_name']
soma_name = soma_rows.values[0] if len(soma_rows) > 0 else 'Ismeretlen régió'

print()
print('========================================================')
print(' EREDMÉNYEK:')
print('========================================================')
print('1. SOMA LOKÁCIÓJA:')
print(f'   Régió neve: {soma_name} (ID: {soma_region_id})\n')

# === AXON ELEMZÉS ===
# ID -> sor-index szótár a gyors kereséshez
id_to_row = {v: i for i, v in enumerate(id_)}

# Szülő-gyerek kapcsolatok meghatározása
parent_row_indices = np.array([id_to_row.get(p, -1) for p in pid])
found_parent = parent_row_indices >= 0
valid_connections = found_parent & (pid != -1)

p_rows = parent_row_indices[valid_connections]

# Gyerekszámok accumarray-szerű számítása
child_counts = np.bincount(p_rows, minlength=len(id_))

# Axon maszk (type==2 vagy type==0)
is_axon = (type_ == 2) | (type_ == 0)

# Végpontok és elágazások
ep_idx     = np.where((child_counts == 0) & is_axon)[0]
branch_idx = np.where((child_counts > 1)  & is_axon)[0]
proj_idx   = np.union1d(ep_idx, branch_idx)

proj_regions = point_regions[proj_idx]

# Axonhossz számítása (Pitagorasz)
curr_idx = np.where(valid_connections)[0]
p_idx    = parent_row_indices[valid_connections]

dx = x[curr_idx] - x[p_idx]
dy = y[curr_idx] - y[p_idx]
dz = z[curr_idx] - z[p_idx]
d  = np.sqrt(dx**2 + dy**2 + dz**2)

axon_mask = (type_[curr_idx] == 2) | (type_[curr_idx] == 0)

# === GPe VETÍTÉS ===
proj_in_gpe = np.sum(proj_regions == target_gpe_id)
len_gpe     = np.sum(d[axon_mask & (point_regions[curr_idx] == target_gpe_id)])

print('2. VETÍT-E A GPe-BE (Globus Pallidus external segment)?')
if proj_in_gpe > 0:
    print('   IGEN!')
    print(f'   Érvényes vetítési pontok (végpont/elágazás): {proj_in_gpe} db')
    print(f'   Axonhossz a GPe-ben: {len_gpe:.1f} um\n')
else:
    print('   NEM. (Csak áthalad, vagy nincs ott axon)\n')

# === RT VETÍTÉS ===
proj_in_rt = np.sum(proj_regions == target_rt_id)
len_rt     = np.sum(d[axon_mask & (point_regions[curr_idx] == target_rt_id)])

print('3. VETÍT-E AZ RT-BE (Reticular nucleus of the thalamus)?')
if proj_in_rt > 0:
    print('   IGEN!')
    print(f'   Érvényes vetítési pontok (végpont/elágazás): {proj_in_rt} db')
    print(f'   Axonhossz az RT-ben: {len_rt:.1f} um\n')
else:
    print('   NEM. (Csak áthalad, vagy nincs ott axon)\n')

# === EGYÉB CÉLTERÜLETEK ===
unique_proj_regions = np.unique(proj_regions[proj_regions > 0])
other_targets = unique_proj_regions[
    (unique_proj_regions != target_gpe_id) &
    (unique_proj_regions != target_rt_id) &
    (unique_proj_regions != soma_region_id)
]

print('4. EGYÉB CÉLTERÜLETEK (Ahol van végpont/elágazás):')
if len(other_targets) == 0:
    print('   Nincs más célterület.')
else:
    for r_id in other_targets:
        r_rows = szotar[szotar['id'] == r_id]['safe_name']
        r_name = r_rows.values[0] if len(r_rows) > 0 else 'Ismeretlen'
        pts_here = np.sum(proj_regions == r_id)
        print(f'   - {r_name} (Pontok: {pts_here} db)')