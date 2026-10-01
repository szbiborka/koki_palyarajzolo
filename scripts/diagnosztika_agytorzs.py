# Diagnosztika: MI FEDI LE pontosan a "BS-desc" virtuális agytörzs-régiót?
#
# Futtatás a projekt gyökeréből:
#     python scripts/diagnosztika_agytorzs.py
#
# Azt írja ki, hogy a szótár (query.csv) alapján hány atlasz-ID tartozik a
# leszálló agytörzshöz, és ezek közül melyik hány voxelt címkéz ténylegesen.
# Ha két futás eltérő eredményt ad, akkor a szótár változott meg - itt látszik.

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BRAINSTEM_MOTOR_COMPONENTS, BRAINSTEM_MOTOR_ID, DICTIONARY_PATH, ATLAS_PATH
from core.loader import load_atlas, load_dictionary, build_region_descendants

print(f"szótár : {DICTIONARY_PATH}")
print(f"atlasz : {ATLAS_PATH}")
print()

dictionary = load_dictionary()
print(f"szótár sorok: {len(dictionary)}")
print(f"oszlopok    : {list(dictionary.columns)}")

has_path = 'structure_id_path' in dictionary.columns
print(f"van structure_id_path? {has_path}")
if not has_path:
    print()
    print("!! EZ A BAJ: hierarchia nélkül a szülő-régiók NEM tudnak feloldódni")
    print("   a leszármazottaikra, így a BS-desc gyakorlatilag üres lesz.")
    raise SystemExit

# Hány sor tartozik a Középagy / Utóagy alá?
paths = dictionary['structure_id_path'].fillna('')
for comp in BRAINSTEM_MOTOR_COMPONENTS:
    n = int(paths.str.contains(f'/{comp}/', regex=False).sum())
    print(f"  /{comp}/ alá tartozó szótár-sor: {n}")

desc = build_region_descendants(dictionary, [BRAINSTEM_MOTOR_ID])[BRAINSTEM_MOTOR_ID]
print(f"\nBS-desc összesen {len(desc)} atlasz-ID-t fed le")

# Ezek közül hány címkéz VALÓBAN voxelt? (a szülő ID-k jellemzően nullát)
atlas = load_atlas()
ids, counts = np.unique(atlas, return_counts=True)
voxels = dict(zip(ids.tolist(), counts.tolist()))

labelled = {rid: voxels.get(rid, 0) for rid in desc}
nonzero = {k: v for k, v in labelled.items() if v > 0}
print(f"ebből ténylegesen címkézett: {len(nonzero)} ID, összesen {sum(nonzero.values()):,} voxel")
print()

name_of = dict(zip(dictionary['id'].astype(int), dictionary['safe_name']))
print("A 15 legnagyobb alrégió:")
for rid, v in sorted(nonzero.items(), key=lambda kv: -kv[1])[:15]:
    print(f"  {v:9,d} voxel  id={rid:<7d} {name_of.get(rid, '?')}")

# Ellenőrzés: a thalamusz NEM lehet benne.
THALAMUS = 549
thal_desc = build_region_descendants(dictionary, [THALAMUS])[THALAMUS]
overlap = desc & thal_desc
print()
if overlap:
    print(f"!! HIBA: {len(overlap)} thalamikus ID került a BS-desc-be: {sorted(overlap)[:10]}")
else:
    print("OK: a thalamusz egyetlen ID-ja sincs a BS-desc-ben.")
