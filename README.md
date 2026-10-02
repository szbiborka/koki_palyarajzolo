# Pályakövető — Neuron Projection Analyzer

A browser-based research tool for analyzing and visualizing neuron projections in the Allen Mouse Brain Atlas. Built at the HUN-REN IEM (KOKI Institute).

---

## The core idea

Most publicly available tools — including some well-known online databases — count a neuron as projecting to a brain region simply because its axon passes *through* that region. This leads to a lot of false positives, especially for neurons with long-range axons that cross many structures on their way to their actual targets.

Pályakövető takes a stricter approach: **a neuron is only considered to project to a region if it forms a genuine terminal arborization there — that is, it has both an endpoint *and* a branch point within that region.** Axons that merely pass through (or that only branch there to send a collateral onward) are not counted. This is a more anatomically meaningful definition of a projection, and it is the main reason this tool exists.

On top of that, you can set your own numerical thresholds — requiring a minimum number of endpoints, branch points, a minimum axon length, or a minimum *share of the cell's endpoints* within a region — so you can tune the definition of "projection" to whatever your experiment calls for. The endpoint-share threshold is what lets you separate cortical layers: Layer 6 cells send the overwhelming majority of their endpoints into the thalamus, so an "Excluded (NOT) thalamus, min endpoint share 2.5%" rule removes them.

---

## What it can do

### Single cell analysis
Load any SWC file and get a full breakdown per target region: how many endpoints, how many branch points, total axon length in the region, and a clear yes/no on whether the cell projects there by your chosen criteria. The app also automatically flags any other regions the cell projects to beyond your target list.

### Batch analysis
Select any number of cells at once and run the analysis across all of them. Results appear as a sortable table (one row per cell, one column set per region) that you can download as a CSV for further work in Excel or Python.

Cells can be chosen three ways: all cells matching the sidebar filters, picked one by one, or **pasted as a list** — any separator works (comma, space, new line), and so does the run-together `221044\016.swc221044\019.swc` form copied out of the cell picker. A pasted list is analysed exactly as given; the soma, class and verdict filters do not apply to it.

### One projection criterion per region
Each target region has **one** set of numbers that defines what counts as a projection there:
- minimum number of axon endpoints in the region — default **1**
- minimum number of branch points in the region — default **1**
- minimum total axon length in the region (µm) — default 0 (no constraint)
- minimum share of the cell's endpoints falling in the region (%) — default 0 (no constraint)

All conditions must be met simultaneously. The default (**≥1 endpoint AND ≥1 branch point**) means a genuine terminal arborization, so an axon that merely passes through — or that branches there only to send a collateral onward — is not counted. Raise the numbers to be stricter, or set branch points to 0 for an endpoint-only rule.

Crucially these same numbers drive the `..._projects` check marks, the filter (`passes_filter`) **and** the Cortical Summary, so they can never disagree with each other. The condition rule is separate: it says how regions combine, not what counts as a projection. Four rules are available — **Required (AND)**, **Excluded (NOT)**, **Optional (OR)** and **Observe only**. Use *Observe only* to add a region purely to read its numbers: it is reported and exported, but never removes a cell from the results.

Note that the **Cortical Summary tables deliberately ignore the condition rules** — they are built from which regions each cell projects to, plus the population base you select. An *Excluded (NOT)* rule therefore does not remove cells from those tables (use the brain-stem base to select pyramidal-tract cells instead); the Population Statistics tab does apply the rules. The criteria in force are written into every exported summary file name (e.g. `bs_benne_ep1_br1.csv`) and into a per-region `..._criterion` column in the detailed batch export.

In batch mode, cells that do not meet the criteria are clearly flagged in the results table with a separate color.

### Cortical projection summary (batch mode)
The **Cortical Summary** tab turns a batch run into the finished, ready-to-send tables — no spreadsheet assembly. You pick a **population base region** (typically "Brain stem (descending)", which defines the pyramidal-tract cells), and every remaining target region (GPe, TRN, …) becomes a numerator. It then produces, per cortical soma region:
- **Brain stem = 100%** — the share of PT cells projecting to each target and to all of them together;
- **All L5 = 100%** — the same, without the brain-stem requirement;
- **average axon length** in each target among PT cells;
- **category tables** listing the projecting cells' serial numbers.

These use the strict projection definition (endpoint **and** branch) directly and always divide by the chosen base, so the "which denominator" and Layer-6-over-removal mistakes cannot occur. Every table is downloadable as CSV. Implemented in `build_cortical_summary` (`core/analysis.py`).

### Hemisphere (laterality)
The Allen atlas gives **both hemispheres the same region id**, so "projects to GPe" would otherwise count the opposite side as well. Since L5 pyramidal-tract cells project essentially ipsilaterally, counting both sides can only inflate the GPe/TRN numbers. A **Hemisphere** selector offers *both* (the previous behaviour, and the default so earlier results still reproduce), *ipsilateral only*, or *contralateral only*; the side is decided against the midline of the medio-lateral axis (5,700 µm), relative to the soma; points within ±50 µm of the midline (`MIDLINE_BAND_UM`, registration uncertainty) count for neither side. In the ipsi/contra modes **every** per-region number follows the chosen side — endpoints, branch points and axon length — and the endpoint share is computed against the endpoints *on that side* (so "≥2.5% of endpoints in the thalamus" means 2.5% of the ipsilateral endpoints in ipsilateral mode). The ipsi/contra endpoint split is exported for every region regardless of the mode, so the size of the contralateral contribution is always visible.

The selector is the default for every region; **each region can override it** in its filter panel, so e.g. the GPe can be evaluated ipsilaterally and the cortex contralaterally in the same run. The side used is exported per region (`..._side`) and written into the criterion text and the summary file names.

**Exclude contralaterally projecting cells** is a whole-cell rule on top of the region rules: a cell fails the filter when more than the given share of its endpoints (default 0%) lies on the opposite hemisphere. Cells whose side cannot be decided (no soma, soma on the midline) are kept.

The supervisor's three-criteria query — *projects to the GPe, does not project contralaterally, and x% of its endpoints are in the thalamus* — is set up as: GPe **Required (AND)**; tick *Exclude contralaterally projecting cells*; add **Thalamus excluding the reticular nucleus (TRN)** (not the plain Thalamus — the TRN is part of it) with *Min. endpoint share* = x% and **Required (AND)** (to select the L6-like cells for inspection) or **Excluded (NOT)** (to remove them). Tick *Only cells that pass the filter* on the Cortical Summary tab to get the tables for that population. Note that 62% of the whole-cortex PT cells and 97% of the prefrontal PT cells have at least one endpoint on the opposite hemisphere, so the contralateral rule removes most PT cells.

### Inclusive vs exclusive categories
The summary reports each target twice. **`GPe n` / `TRN n`** are *inclusive* — a cell projecting to both is counted in both, which is what the "brain stem = 100%" percentages describe. **`GPe only n` / `TRN only n`** are *exclusive* — a cell counts only if it projects to that target and to none of the others, which is the mutually exclusive split of the original three category files. Both are shown side by side because mixing them up makes GPe and TRN look inflated; `only` + `only` + `All targets` + non-projectors accounts for every cell in the base population.

### Soma region filtering
With thousands of SWC files, scrolling through a list is not practical. Type part of a region name — "motor", "thalamus", "striatum" — and the file list instantly narrows to only cells whose soma is located in a matching region. This is powered by a one-time index that scans all SWC files on first use and saves the result; subsequent loads are instant.

### Cell types from the database (projection class, Cre line)
The SWC files carry no metadata, but the database publishes it per neuron: its own soma region, the Cre line, and a **projection class** — **IT** (intratelencephalic), **PT** (pyramidal tract, L5, subcerebral) or **CT** (corticothalamic, L6), with sub-classes such as `PT-18`. With `adatfajlok/database_metadata/cortex_neuron_info.json` in place (see *Setup*), the sidebar can filter by class and Cre line, the single-cell views show them, and the batch export gets `projection_class`, `projection_subclass`, `cre_line` and `db_soma_region` columns.

The class is assigned from the axon, not from the soma position, which makes it the tool against the two layer errors of the database:
- **L5 cells registered into L6.** A PT cell whose soma sits on the L5/L6 border can get an "L6a" region, although it descends to the brain stem — something L6 cells do not do. Filter by class **PT** instead of by "layer 5", and/or turn on **Correct soma layer by projection class**: the summaries then count such a cell under the L5 region of the same cortical area (PT in L4/L6a → L5, CT in L5 → L6a; only the adjacent layer is corrected — a PT cell in L2/3 or L6b is flagged, not moved). The correction also applies to the **soma-region search**, so searching "layer 5" finds the PT cells registered into L6a and drops the CT cells registered into L5. The atlas region is still exported as `soma_region`, the corrected one as `summary_soma_region`.
- **Shifted L6 (CT) cells.** A slightly mis-registered L6 CT cell lands its TRN arbor in the GPe and its thalamic arbor in the TRN, producing false GPe/TRN projections. These cells are CT in the database.

The *Population Statistics* tab shows the soma layer × class table of the batch, so the suspects are visible at a glance.

### Manual verdicts (curation)
Cells checked by hand get a **verdict** — *Shifted L6*, *Actually L5*, *Cortico-cortical (IT)*, *Other problem* or *Verified OK* — stored in `curation/manual_labels.csv` (versioned with the code). The sidebar excludes *Shifted L6* and *Other problem* by default, so a cell has to be judged only once. Verdicts are set in the single-cell views (*Manual verdict for this cell*), or imported from a colour-coded Excel sheet:

```bash
python scripts/import_excel_labels.py "poszter_tablak.xlsx" 05_L6a_GPe
```

The import reads the font colour of every cell ID in the *Projecting Cell IDs* column (legend at the bottom of the 05_L6a_GPe sheet; uncoloured = verified). The 198 verdicts of that sheet are already imported.

### Interactive 3D visualization (Plotly, fully browser-native)
For any single cell or a combined batch, open an interactive 3D viewer directly in the browser tab. No installation, no desktop window, no VTK.js issues — the viewer works on any machine that can open the Streamlit app.

The 3D scene shows:
- semi-transparent brain region surface meshes (marching cubes from the Allen Atlas)
- the full axon tree colored by which region each segment falls in
- the soma as a black sphere marker
- projection points (endpoints and branch points) highlighted as diamond markers per region

For batch mode, each cell gets its own color (soma + full axon tree) so individual neurons stay distinguishable in the combined view.

The **Axon-in-region view** toggle hides every axon segment outside the target regions (and the soma region), so only the axon that actually runs through e.g. the GPe is drawn next to the region mesh — useful to see which sub-territory a group of cells innervates.

**Show projection points** hides the markers (circles = endpoints, diamonds = branch points; shown only on the hemisphere each region is evaluated on). **Camera view** switches between the free, rotatable view and fixed orthographic **top (dorsal)**, **side (lateral)** and **front (anterior)** views; the axes are labelled AP / DV / ML. **Axon line width** and **Figure height** are adjustable; the camera icon saves a 3× PNG.

Everything in the scene can be switched on and off: click a legend item to hide / show it (a region's surface and its points, or a cell group's axons and somata, toggle together), double-click to show only that item; above the figure, *Regions in the scene* and — in the combined view — *Cells in the scene* and *Colour cells by* (each cell — the default —, projection class, or soma region) and *Line opacity* select what is drawn and how. Each cell gets its own colour: up to eight from the validated palette, beyond that hues spread around the OKLCH wheel by the golden angle (colours then separate the trees but no longer identify a cell by name — hover an axon or a soma for that). Colours come from a validated 8-colour palette in a fixed order: a colour always means the same region or group, and past eight groups the rest is shown in neutral grey as *Other*. Large trees are simplified for display (every N-th node plus every branch point and endpoint, connected to the nearest kept ancestor), so lines stay continuous. The 3D tabs are only built when opened, and identical figures are reused within a session.

---

## What is coming next

### Population comparison and statistics
Define two groups of cells — for example, M2 cells that project to GPe versus M2 cells that do not — and get an automatic statistical breakdown:

- what percentage of each group projects to every other detected region
- average axon length per region per group
- side-by-side comparison output ready for export

Example output (values are illustrative):

```
M2 cells projecting to GPe:   100% also project to thalamus,  20% to striatum
M2 cells NOT projecting to GPe:  10% project to thalamus,     70% to striatum

Average axon in TRN:
  GPe-projecting:     100 µm
  Non-projecting:      10 µm
```

---

## Project structure

```
palyakoveto/
├── app.py               — Streamlit UI entry point (run this)
├── ui_assets.py         — CSS, SVG icons and small UI helpers
├── config.py            — all paths and constants; only file to edit for deployment
├── soma_index.csv       — auto-generated on first run, do not edit manually
├── requirements.txt     — Python dependencies (requirements-dev.txt adds pytest)
├── core/
│   ├── loader.py        — data loading: atlas, SWC files, dictionary, soma index
│   ├── analysis.py      — science logic: projection detection, filtering, summaries
│   ├── cell_info.py     — database metadata, manual verdicts, cell lists, layer correction
│   └── visualization.py — 3D Plotly figure construction
├── curation/            — manual verdicts per cell (manual_labels.csv)
├── docs/DOKUMENTACIO.md — change log, to-improve status, how the logic works (Hungarian)
├── tests/               — regression tests on synthetic mini-atlases (no data needed)
├── scripts/             — diagnostics and the Excel verdict import
├── dolgozat/            — thesis drafts (LaTeX)
└── adatfajlok/          — local data: atlas, dictionary, SWC files (not in git)
```

The separation is intentional. `app.py` contains only UI code and calls into `core/`. The science logic in `core/analysis.py` does not import Streamlit, so it can be tested, modified, or reused on its own.

---

## Setup

### Requirements
- Python 3.12+
- The Allen Mouse Brain Atlas annotation file (`annotation_25.nrrd`)
- The region dictionary (`query.csv`)
- Your SWC files organized under a base directory

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

By default the data is read from the `adatfajlok/` folder inside the project:

```
adatfajlok/annotation_25.nrrd
adatfajlok/query.csv
adatfajlok/data_v2/<mouse>/<cell>.swc
```

To use other locations, set the `PALYAKOVETO_DATA_DIR`, `ATLAS_PATH` and `DICTIONARY_PATH` environment variables (see below) or edit `config.py`.

Optional, for the cell-type features — the database's neuron metadata (~7.5 MB):

```bash
curl -o adatfajlok/database_metadata/cortex_neuron_info.json "https://mouse.digital-brain.cn/projectome/2/srv//info/mouse/cortex/mouse.neuron.info.json"
```

### Running locally

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. On first use, click **Build soma index** in the sidebar — this takes a few minutes for large datasets but only needs to run once. If you add new SWC files later, use the **Rebuild** button.

### Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/
```

---

## Deploying to the institute server

When server access is available:

1. Copy the project to the server.
2. Set paths via environment variables (no need to edit `config.py` directly):
   ```bash
   export ATLAS_PATH=/data/atlas/annotation_25.nrrd
   export DICTIONARY_PATH=/data/atlas/query.csv
   export PALYAKOVETO_DATA_DIR=/data/swc_files/
   ```
3. Run:
   ```bash
   streamlit run app.py --server.port 8501 --server.address 0.0.0.0
   ```

The 3D visualization now uses Plotly (WebGL, browser-native) and does not require a display or GPU on the server. All features work fully headless.

---

## Adding new features

- **New analysis metric** → add a field to `RegionResult` or `CellAnalysisResult` in `analysis.py`, compute it in `run_analysis()`, display it in `app.py`
- **New filter type** → add a field to `FilterCriteria` and update `is_projection()`, `meets_thresholds()`, `describe()` and `slug()`
- **New UI section** → add to `app.py` only, call existing `core/` functions
- **New atlas or species** → add a config block in `config.py` and a loader branch in `loader.py`

---

## Technical notes

**Why Plotly and not PyVista/stpyvista?**
The original implementation used PyVista with the stpyvista Streamlit component. This caused two problems: (1) `plotter.show()` opens a native desktop window on whichever machine runs the server process, not on the user's browser; (2) stpyvista's VTK.js serializer fails silently on manually constructed `PolyData` objects (which is exactly how axon line geometry is built), showing a blank Kitware fallback page instead. Plotly's `go.Scatter3d` with `None`-separated segments handles the same geometry correctly and renders entirely in the browser with no server-side display requirements.

**Projection detection logic**
A node is an axon endpoint if it has zero children in the SWC parent-child tree. A node is a branch point if it has more than one child. A cell is considered to project to a region only if **both** an endpoint and a branch point fall within that region's voxel boundary in the Allen Atlas (defaults in `DEFAULT_FILTER`, `config.py`). Requiring both is what excludes "passing" axons: a fiber that only branches in a region to send a collateral onward — but terminates elsewhere — has a branch point there but no endpoint, so it is correctly *not* counted as a projection. The per-region endpoint share (`endpoint_fraction`, region endpoints ÷ the cell's total endpoints) supports size-independent thresholds such as the Layer 6 thalamus filter.

**Parent (umbrella) regions**
The Allen annotation volume labels each voxel with a *leaf* structure, not with the broad parent region — so an umbrella region such as "Brain stem" (id 343) or "Thalamus" (id 549) covers **zero** voxels on its own. To make targets like "projects to Brain stem" work, `build_region_descendants` (in `core/loader.py`) expands each selected region to itself plus all of its descendants using the `structure_id_path` column of `query.csv`, and matching is done with `np.isin` against that set. This applies uniformly to projection targets and to the thalamic endpoint-share used by the Layer 6 filter. If the dictionary lacks `structure_id_path`, the code falls back to exact single-id matching.

**Descending brain stem (excludes thalamus)**
A quirk of the Allen ontology is that the umbrella **"Brain stem" (id 343) contains the Interbrain → Thalamus** — so "projects to Brain stem" would also count purely thalamic (Layer 6) axons as brainstem projections. For the pyramidal-tract question this is wrong. The region picker therefore offers a virtual target, **"Brain stem descending — Midbrain+Hindbrain (excl. thalamus)"** (`BRAINSTEM_MOTOR_ID` in `config.py`), which expands to the descendants of Midbrain (313) and Hindbrain (1065) only. Use this instead of the raw "Brain stem" entry when selecting pyramidal-tract cells: Layer 6 cells that project only to the thalamus then fail the brainstem criterion on their own. Its default criterion is **≥5 endpoints** (and ≥1 branch point): genuine PT cells have a median of 102 brain-stem endpoints (98% have ≥5), whereas the CT and IT cells that pass a 1-endpoint rule have a median of 3–4, almost all in midbrain voxels next to the thalamus (unassigned "Midbrain", APN, MRN) — registration leakage of thalamic arbors.

**Thalamus without the TRN**
The reticular nucleus (262) is a descendant of the Thalamus (549) in the Allen ontology, so with both selected every TRN endpoint also counts as thalamic: a NOT-thalamus rule removes TRN projectors and the "TRN only" category can never be filled. The picker offers **"Thalamus excluding the reticular nucleus (TRN)"** (`THALAMUS_NO_TRN_ID`); the app warns whenever two selected regions overlap.

**Voxel lookup**
A point at µm coordinate *c* falls into voxel `floor(c / 25)` — the Allen CCF convention (voxel *i* spans [25·i, 25·(i+1)) µm). Checked against the database's own soma regions: 95.07% agreement with `floor`, 94.04% with `round` (the behaviour before 2026-10-02). The two differ for 8.9% of the somata; the old behaviour can be restored with `VOXEL_LOOKUP = 'round'` (environment variable `PALYAKOVETO_VOXEL_LOOKUP`) to reproduce earlier tables — rebuild the soma index after switching.

**Soma index**
The first run builds a CSV index mapping every SWC file to the atlas region of its soma node. This is done by reading only the `type == 1` row from each file, which is much faster than loading entire SWC files. Subsequent app starts load the index from disk instantly. Paths in the index always use `/`, so an index built on Windows also works on the Linux server.

**Axon length per region**
Each axon segment is sampled every half voxel (12.5 µm), so a segment that crosses a region boundary is split proportionally between the regions. Region ids are compacted before summing — Allen ids go up to ~6·10⁸, and indexing an array by raw id would cost gigabytes per cell.

---

## Data notes

Checked against the database metadata on 2026-10-01.

**Which part of the database we have.** The local `data_v2` holds all **18,621** neurons of the cortex metadata (382 samples), each with metadata:
- 12,264 from *Single-neuron projectome of mouse whole cortex* (2025, samples 212064–233795);
- 6,357 from *Single-neuron projectome of mouse prefrontal cortex (with dendrite)* (2023, samples 17099–201787, all C57BL/6J; [doi:10.12412/BSDC.1690164952.20001](https://doi.org/10.12412/BSDC.1690164952.20001)). Only the CCF-registered copy (`swc_allen_neurite`) is used, filed as `<sample>/<NNN>.swc`. The native-space copy (`swc_raw_neurite`) is kept in `adatfajlok/pfc_raw_native_space/` and must not be analysed against the atlas. Its dendrites are typed 3 and are excluded from all axon measures.

The site as a whole holds 46,175 neurons; the rest belongs to non-cortical datasets (hippocampus, hypothalamus, …).

**Why "layer 5" and "cell type" searches give different cells.** They are independent labels in the database. The layer is where the soma landed after registration to the atlas; the class (IT / PT / CT) comes from the projection pattern, and the Cre line from the mouse. Across all 18,621 cortex neurons:

| Soma layer | CT | IT | PT |
|---|---|---|---|
| 5 | 866 | 3,791 | 3,172 |
| 6a | 1,477 | 505 | 159 |

So 866 CT cells sit in "layer 5", and 159 PT cells in "layer 6a". The Cre lines are not clean either: Rbp4 (an L5 line) contains 704 IT cells. Our own soma assignment agrees with the database's region for 93.8% of the local cells, so the difference is not ours.

**The manual 05_L6a_GPe check vs the database class** (198 cells):

| Manual verdict | Cells | CT | PT | IT |
|---|---|---|---|---|
| Shifted L6 | 123 | 121 | 2 | 0 |
| Actually L5 | 27 | 3 | 24 | 0 |
| Cortico-cortical | 12 | 2 | 1 | 9 |
| Verified (genuine GPe) | 23 | 21 | 2 | 0 |
| Other problem | 13 | 3 | 6 | 4 |

The class reproduces "shifted L6" (98% CT) and "actually L5" (89% PT). It cannot, however, separate shifted L6 cells from the verified genuine L6 → GPe cells: both are CT, both have ~55% of their endpoints in the thalamus, and their GPe points lie equally shallow in the GPe. The GPe points of shifted cells are closer to the TRN/thalamus (median 190 µm vs 430 µm), but the distributions overlap too much for a reliable cut-off. That decision therefore stays manual — but made once, in `curation/manual_labels.csv`.

---

## Atlas

Allen Mouse Brain Atlas, 25 µm resolution (`annotation_25.nrrd`).  
Region dictionary: `query.csv` with fields `id`, `acronym`, `safe_name`.