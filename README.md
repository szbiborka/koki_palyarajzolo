# 🧠 Pályakövető — Neuron Projection Analyzer

A research tool for analyzing neuron projections in the Allen Mouse Brain Atlas.

**Key difference from other tools:** Projections are detected using endpoints and branch points only —
axons that merely *pass through* a region are NOT counted as projections.

---

## Project Structure

```
palyakoveto/
├── app.py              # Streamlit UI (entry point)
├── config.py           # All paths and constants — edit this for server deployment
├── requirements.txt    # Python dependencies
├── core/
│   ├── loader.py       # Data loading (atlas, SWC, dictionary)
│   ├── analysis.py     # Science logic (projection detection, axon length)
│   └── visualization.py # 3D plot building (PyVista)
└── README.md
```

---

## Setup (Local / PyCharm)

### 1. Install dependencies
In PyCharm terminal (or your venv):
```bash
pip install -r requirements.txt
```

### 2. Configure data paths
Edit `config.py` and set the correct paths for your machine:
```python
BASE_DATA_DIR = '/path/to/your/swc_files/'
ATLAS_PATH    = '/path/to/annotation_25.nrrd'
DICTIONARY_PATH = '/path/to/query.csv'
```

### 3. Run the app
```bash
streamlit run app.py
```
The app will open at `http://localhost:8501` in your browser.

---

## Running on the Institute Server

When you get server access, you only need to:

1. Copy the project files to the server
2. Update paths in `config.py` (or set environment variables):
   ```bash
   export ATLAS_PATH=/data/atlas/annotation_25.nrrd
   export DICTIONARY_PATH=/data/atlas/query.csv
   export PALYAKOVETO_DATA_DIR=/data/swc_files/
   ```
3. Run with:
   ```bash
   streamlit run app.py --server.port 8501 --server.address 0.0.0.0
   ```

**3D viewer note:** The current `plotter.show()` opens a desktop window, which does not work on a
headless server. Server-compatible rendering via PyVista Trame is planned — see the comments in
`core/visualization.py` for the implementation template.

---

## Adding New Features

- **New analysis metric** → add a function to `core/analysis.py` and a new field to `CellAnalysisResult`
- **New UI section** → add to `app.py` only, call existing `core/` functions
- **Change a path or constant** → edit `config.py` only
- **Support a new species/atlas** → add a new atlas config block in `config.py` and a loader branch in `core/loader.py`

---

## Planned Features

- [ ] Trame-based in-browser 3D viewer (server compatible)
- [ ] Statistical summary across all cells (projection matrix heatmap)  
- [ ] Filter cells by soma region
- [ ] Monkey brain atlas support
- [ ] Multi-atlas comparison
