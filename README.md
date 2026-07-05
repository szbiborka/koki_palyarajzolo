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

### Flexible projection filters
For each target region independently, you can set:
- minimum number of axon endpoints in the region
- minimum number of branch points in the region
- minimum total axon length in the region (µm)

All conditions must be met simultaneously. In batch mode, cells that do not meet the criteria are clearly flagged in the results table with a separate color.

### Soma region filtering
With thousands of SWC files, scrolling through a list is not practical. Type part of a region name — "motor", "thalamus", "striatum" — and the file list instantly narrows to only cells whose soma is located in a matching region. This is powered by a one-time index that scans all SWC files on first use and saves the result; subsequent loads are instant.

### Interactive 3D visualization (Plotly, fully browser-native)
For any single cell or a combined batch, open an interactive 3D viewer directly in the browser tab. No installation, no desktop window, no VTK.js issues — the viewer works on any machine that can open the Streamlit app.

The 3D scene shows:
- semi-transparent brain region surface meshes (marching cubes from the Allen Atlas)
- the full axon tree colored by which region each segment falls in
- the soma as a black sphere marker
- projection points (endpoints and branch points) highlighted as diamond markers per region

For batch mode, each cell gets its own color (soma + full axon tree) so individual neurons stay distinguishable in the combined view.

---

## What is coming next

### Axon-in-region display mode
A visualization toggle that hides everything outside a selected region, showing only the axon segments that actually run through it alongside the region boundary mesh. If you display ten M2 cells this way for the GPe, you immediately see which sub-territory of the GPe those cells tend to innervate. This is the most requested visualization feature from the supervisor's brief.

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

### Set-logic (Boolean) filtering
Currently the filter uses AND logic across all selected regions. The planned upgrade allows full Boolean queries, for example:

> "Show cells that project to GPe OR TRN, but NOT to striatum"

This makes it possible to ask the kinds of population questions described in the original project brief, where intersection, union, and exclusion of projection targets all matter.

---

## Project structure

```
palyakoveto/
├── app.py              — Streamlit UI entry point (run this)
├── config.py           — all paths and constants; only file to edit for deployment
├── soma_index.csv      — auto-generated on first run, do not edit manually
├── requirements.txt    — Python dependencies
├── core/
│   ├── loader.py       — data loading: atlas, SWC files, dictionary, soma index
│   ├── analysis.py     — science logic: projection detection, filtering, axon length
│   └── visualization.py — 3D Plotly figure construction
└── README.md
```

The separation is intentional. `app.py` contains only UI code and calls into `core/`. The science logic in `core/analysis.py` can be tested, modified, or reused independently of Streamlit.

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

Open `config.py` and set the three paths for your machine:

```python
BASE_DATA_DIR   = '/path/to/your/swc_files/'
ATLAS_PATH      = '/path/to/annotation_25.nrrd'
DICTIONARY_PATH = '/path/to/query.csv'
```

### Running locally

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. On first use, click **Build soma index** in the sidebar — this takes a few minutes for large datasets but only needs to run once. If you add new SWC files later, use the **Rebuild** button.

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
- **New filter type** → add a field to `FilterCriteria` and update the `check()` method
- **New UI section** → add to `app.py` only, call existing `core/` functions
- **New atlas or species** → add a config block in `config.py` and a loader branch in `loader.py`

---

## Technical notes

**Why Plotly and not PyVista/stpyvista?**
The original implementation used PyVista with the stpyvista Streamlit component. This caused two problems: (1) `plotter.show()` opens a native desktop window on whichever machine runs the server process, not on the user's browser; (2) stpyvista's VTK.js serializer fails silently on manually constructed `PolyData` objects (which is exactly how axon line geometry is built), showing a blank Kitware fallback page instead. Plotly's `go.Scatter3d` with `None`-separated segments handles the same geometry correctly and renders entirely in the browser with no server-side display requirements.

**Projection detection logic**
A node is an axon endpoint if it has zero children in the SWC parent-child tree. A node is a branch point if it has more than one child. A cell is considered to project to a region only if **both** an endpoint and a branch point fall within that region's voxel boundary in the Allen Atlas (see `MIN_ENDPOINTS_FOR_PROJECTION` / `MIN_BRANCH_POINTS_FOR_PROJECTION` in `core/analysis.py`). Requiring both is what excludes "passing" axons: a fiber that only branches in a region to send a collateral onward — but terminates elsewhere — has a branch point there but no endpoint, so it is correctly *not* counted as a projection. The per-region endpoint share (`endpoint_fraction`, region endpoints ÷ the cell's total endpoints) supports size-independent thresholds such as the Layer 6 thalamus filter.

**Soma index**
The first run builds a CSV index mapping every SWC file to the atlas region of its soma node. This is done by reading only the `type == 1` row from each file, which is much faster than loading entire SWC files. Subsequent app starts load the index from disk instantly.

---

## Atlas

Allen Mouse Brain Atlas, 25 µm resolution (`annotation_25.nrrd`).  
Region dictionary: `query.csv` with fields `id`, `acronym`, `safe_name`.