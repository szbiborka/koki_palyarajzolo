# Pályakövető — Neuron Projection Analyzer

A browser-based research tool for analyzing and visualizing neuron projections in the Allen Mouse Brain Atlas. Built at the Koki Institute.

---

## The core idea

Most publicly available tools (including some well-known online databases) count a neuron as projecting to a brain region simply because its axon passes through that region. This leads to a lot of false positives, especially for neurons with long-range axons that cross many structures on their way to their actual targets.

Pályakövető takes a stricter approach: a neuron is only considered to project to a region if it has an **endpoint** or **branch point** located within that region. Axons that merely pass through are ignored. This is a more anatomically meaningful definition of a projection, and it's the main reason this tool exists.

On top of that, you can set your own numerical thresholds — requiring a minimum number of endpoints, branch points, or a minimum axon length within a region — so you can tune the definition of "projection" to whatever your experiment calls for.

---

## What it can do right now

**Single cell analysis**
Load any SWC file and get a full breakdown per target region: how many endpoints, how many branch points, total axon length in the region, and a clear yes/no on whether the cell projects there by your chosen criteria. The app also automatically flags any other regions the cell projects to beyond your target list.

**Batch analysis**
Select any number of cells at once and run the analysis across all of them. Results appear as a table (one row per cell, one column set per region) that you can download as a CSV for further work in Excel or Python.

**Flexible projection filters**
For each target region independently, you can set:
- minimum number of axon endpoints in the region
- minimum number of branch points in the region
- minimum total axon length in the region (µm)

All conditions must be met simultaneously. In batch mode, cells that don't meet the criteria are clearly flagged in the results table.

**Soma region filtering**
With 12,000+ SWC files, scrolling through a list is not practical. Type part of a region name — "motor", "thalamus", "striatum" — and the file list instantly narrows to only cells whose soma is located in a matching region. This is powered by a one-time index that scans all SWC files on first use and saves the result; subsequent loads are instant.

**3D visualization**
For any single cell, open an interactive 3D viewer showing brain region surfaces as semi-transparent meshes, the full axon tree colored by which region each segment falls in, the soma as a marker, and projection points highlighted. Currently opens as a desktop window (PyVista).

---

## What's coming next

**Axon-in-region display mode** ↳
A visualization toggle that hides everything outside a selected region, showing only the axon segments that actually run through it alongside the region boundary. If you display ten M2 cells this way for the GPe, you immediately see which part of the GPe those cells tend to innervate.

**Population comparison**
Define two groups of cells — for example, M2 cells that project to GPe versus M2 cells that don't — and get a statistical breakdown of where each group projects and in what proportions. Something like: "cells that project to GPe: 100% also project to thalamus, 20% to striatum. Cells that don't: 10% project to thalamus, 70% to striatum." Axon length comparisons between groups will also be included.

**HTML 3D export**
Save any 3D scene as a standalone HTML file that anyone can open in their browser and rotate/zoom without needing to install anything. A practical interim solution before full browser-based rendering is ready.

**In-browser 3D rendering (Trame)**
The desktop window approach works fine locally but won't work on a headless server. The plan is to replace it with PyVista's Trame backend, which streams the 3D view directly into the browser. This is the most technically involved item on the list and is planned after the higher-value science features above.

---

## Project structure

```
palyakoveto/
├── app.py              — Streamlit UI, entry point (run this)
├── config.py           — all paths and constants; the only file to edit for server deployment
├── soma_index.csv      — auto-generated on first run, do not edit manually
├── requirements.txt    — Python dependencies
├── core/
│   ├── loader.py       — data loading: atlas, SWC files, dictionary, soma index
│   ├── analysis.py     — science logic: projection detection, filtering, axon length
│   └── visualization.py — 3D plot construction (PyVista)
└── README.md
```

The separation is intentional. `app.py` contains only UI code and calls into `core/`. This means the science logic in `core/analysis.py` can be tested, modified, or reused independently of Streamlit.

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

When server access is available, the steps are:

1. Copy the project to the server
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

Note: the 3D viewer (`plotter.show()`) requires a display and will not work on a headless server until the Trame backend is implemented. Everything else — analysis, filtering, batch export — works fine without a display.

---

## Adding new features

- **New analysis metric** → add a field to `RegionResult` or `CellAnalysisResult` in `analysis.py`, compute it in `run_analysis()`, display it in `app.py`
- **New filter type** → add a field to `FilterCriteria` and update the `check()` method
- **New UI section** → add to `app.py` only, call existing `core/` functions
- **New atlas or species** → add a config block in `config.py` and a loader branch in `loader.py`

---

## Atlas

Allen Mouse Brain Atlas, 25 µm resolution (`annotation_25.nrrd`).
Region dictionary: `query.csv` with fields `id`, `acronym`, `safe_name`.