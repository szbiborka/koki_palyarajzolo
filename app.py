# =============================================================================
# APP.PY - A Streamlit alkalmazás belépési pontja.
# Ez a fájl CSAK UI elemet tartalmaz. Semmi tudományos logika nincs itt.
# Futtatás: streamlit run app.py
# =============================================================================

import streamlit as st
import pandas as pd

from config import BASE_DATA_DIR, DEFAULT_TARGET_REGIONS
from core.loader import (
    load_atlas, load_dictionary, load_swc,
    get_all_swc_files, build_region_search_options
)
from core.analysis import run_analysis, results_to_dataframe
from core.visualization import build_3d_plot, show_plot_local

# =============================================================================
# OLDAL KONFIGURÁCIÓ
# =============================================================================
st.set_page_config(
    page_title="Palyakoveto — Neuron Projection Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Egyedi CSS: professzionálisabb, tudományos megjelenés
st.markdown("""
<style>
    /* Fő betűtípus és háttér */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
    }

    /* Oldalsáv fejléc */
    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #1a1a2e;
        margin-bottom: 0.1rem;
    }
    .sidebar-subtitle {
        font-size: 0.78rem;
        color: #666;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    /* Eredmény kártyák */
    .result-card {
        background: #f8f9fb;
        border: 1px solid #e0e4ec;
        border-left: 3px solid #2d6a9f;
        border-radius: 4px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.6rem;
    }
    .result-card.positive {
        border-left-color: #2a7f4f;
    }
    .result-card.negative {
        border-left-color: #b0b0b0;
    }
    .result-card h4 {
        margin: 0 0 0.4rem 0;
        font-size: 0.95rem;
        font-weight: 600;
        color: #1a1a2e;
    }
    .result-card .meta {
        font-size: 0.82rem;
        color: #555;
    }
    .tag-yes {
        display: inline-block;
        background: #e6f4ee;
        color: #1e6b3e;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        padding: 0.15rem 0.5rem;
        border-radius: 3px;
        text-transform: uppercase;
        margin-right: 0.5rem;
    }
    .tag-no {
        display: inline-block;
        background: #f0f0f0;
        color: #888;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        padding: 0.15rem 0.5rem;
        border-radius: 3px;
        text-transform: uppercase;
        margin-right: 0.5rem;
    }

    /* Lap teteje */
    .page-header {
        border-bottom: 2px solid #1a1a2e;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    .page-header h1 {
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        color: #1a1a2e;
        margin: 0;
    }
    .page-header p {
        font-size: 0.85rem;
        color: #555;
        margin: 0.2rem 0 0 0;
    }

    /* Metrika feliratok finomítása */
    [data-testid="metric-container"] {
        background: #f8f9fb;
        border: 1px solid #e0e4ec;
        border-radius: 4px;
        padding: 0.7rem 1rem;
    }

    /* Elválasztó vonal */
    hr {
        border: none;
        border-top: 1px solid #e0e4ec;
        margin: 1.2rem 0;
    }

    /* Táblázat finomítás */
    [data-testid="stDataFrame"] {
        border: 1px solid #e0e4ec;
        border-radius: 4px;
    }

    /* Gombok */
    .stButton > button {
        border-radius: 3px;
        font-weight: 600;
        letter-spacing: 0.03em;
    }

    /* Lábléc elrejtése */
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# GLOBÁLIS ADATOK BETÖLTÉSE (csak egyszer fut le, cache-ből jön vissza ezután)
# =============================================================================
try:
    atlas_matrix, atlas_header = load_atlas()
    dictionary = load_dictionary()
    region_options = build_region_search_options(dictionary)
except FileNotFoundError as e:
    st.error(f"Data file not found. Please check config.py.\n\n{e}")
    st.stop()

# =============================================================================
# OLDALSÁV - Beállítások és fájlválasztás
# =============================================================================
with st.sidebar:
    st.markdown('<div class="sidebar-title">Palyakoveto</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">Neuron Projection Analyzer — KOKI</div>', unsafe_allow_html=True)
    st.divider()

    # --- Célterületek kiválasztása ---
    st.markdown("**Target Brain Regions**")
    st.caption(
        "Projections are detected by endpoints and branch points only. "
        "Axons that merely pass through a region are not counted."
    )

    # Az alapértelmezett célterületek előre be vannak töltve
    # A felhasználó bármilyen régiót kereshet az atlasz szótárból
    selected_region_names = st.multiselect(
        label="Search and select regions",
        options=list(region_options.keys()),
        default=[
            name for name in region_options.keys()
            if region_options[name] in DEFAULT_TARGET_REGIONS.values()
        ],
        help="Type to search by region name or acronym."
    )

    # A kiválasztott nevek alapján összegyűjtjük az ID-kat
    selected_region_ids = [region_options[name] for name in selected_region_names]

    st.divider()

    # --- SWC fájlok kezelése ---
    st.markdown("**Cell Files (SWC)**")

    # Az elérhető SWC fájlok listája az alap mappából
    available_swc = get_all_swc_files(BASE_DATA_DIR)

    if not available_swc:
        st.warning(
            f"No SWC files found in:\n`{BASE_DATA_DIR}`\n\n"
            "Please check BASE_DATA_DIR in config.py"
        )
        selected_swc_paths = []
    else:
        st.caption(f"{len(available_swc)} SWC files available.")

        # Mód váltó: egy sejt vs. batch elemzés
        analysis_mode = st.radio(
            "Analysis mode",
            options=["Single cell", "Batch (multiple cells)"],
            horizontal=True
        )

        if analysis_mode == "Single cell":
            # Egyetlen fájl választó legördülő menü
            selected_name = st.selectbox(
                "Select cell",
                options=list(available_swc.keys()),
                help="Format: mouse_folder/cell.swc"
            )
            selected_swc_paths = [available_swc[selected_name]]

        else:
            # Batch módban: multiselect vagy "select all" gomb
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Select all", use_container_width=True, key="btn_select_all"):
                    st.session_state['batch_selection'] = list(available_swc.keys())
            with col2:
                if st.button("Clear", use_container_width=True, key="btn_clear"):
                    st.session_state['batch_selection'] = []

            batch_default = st.session_state.get('batch_selection', [])
            selected_names = st.multiselect(
                "Select cells for batch analysis",
                options=list(available_swc.keys()),
                default=batch_default,
                help="Select multiple cells. Max recommended: ~50 at once."
            )
            selected_swc_paths = [available_swc[name] for name in selected_names]

    st.divider()

    # --- Vizualizáció beállítások ---
    st.markdown("**Visualization**")
    show_soma_region = st.toggle("Show soma region", value=True)
    show_other_regions = st.toggle("Show other projection regions", value=True)

    st.divider()
    st.caption("Atlas: Allen Mouse Brain Atlas (25 µm)")
    st.caption("Palyakoveto v0.1")

# =============================================================================
# FŐ TARTALOM
# =============================================================================

# Ha nincs semmi kiválasztva, mutassunk egy üdvözlő üzenetet
if not selected_swc_paths or not selected_region_ids:
    st.markdown("""
    <div class="page-header">
        <h1>Palyakoveto</h1>
        <p>Neuron Projection Analyzer &mdash; Koki Institute</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**To get started**, select target brain regions and one or more cell files in the sidebar, then click Run Analysis.")

    # Rövid magyarázó szöveg a módszerről
    with st.expander("About the projection detection method"):
        st.markdown("""
        **What makes Palyakoveto different from other tools?**

        Many tools (including some published databases) mark a neuron as projecting to a region
        simply because the axon passes through that region. This leads to false positives.

        Palyakoveto uses a stricter anatomical criterion: a neuron is counted as projecting to a
        region only if it has an **endpoint** or **branch point** located within that region.

        | Point type | Definition |
        |---|---|
        | **Endpoint** | The axon terminates here — a likely synaptic target |
        | **Branch point** | The axon splits here — suggesting local arborization |
        | Pass-through | Axon crosses the region but does not end or branch — not counted |

        This approach more accurately reflects the anatomical definition of a projection target.
        """)

    st.stop()

# =============================================================================
# ANALÍZIS FUTTATÁSA
# =============================================================================
run_button = st.button(
    f"Run Analysis  —  {len(selected_swc_paths)} cell{'s' if len(selected_swc_paths) > 1 else ''}",
    type="primary",
    use_container_width=True,
    key="btn_run_analysis"
)

if run_button:
    # Eredmények tárolása a session state-ben, hogy a UI frissítésekor megmaradjanak
    st.session_state['results'] = []
    st.session_state['errors'] = []

    progress = st.progress(0, text="Analyzing cells...")

    for i, filepath in enumerate(selected_swc_paths):
        cell_name = list(available_swc.keys())[
            list(available_swc.values()).index(filepath)
        ]
        try:
            swc_df = load_swc(filepath)
            result = run_analysis(swc_df, atlas_matrix, dictionary, selected_region_ids)
            st.session_state['results'].append((cell_name, result))
        except Exception as e:
            st.session_state['errors'].append((cell_name, str(e)))

        progress.progress((i + 1) / len(selected_swc_paths), text=f"Analyzed: {cell_name}")

    progress.empty()

# =============================================================================
# EREDMÉNYEK MEGJELENÍTÉSE
# =============================================================================

if 'errors' in st.session_state and st.session_state['errors']:
    with st.expander(f"{len(st.session_state['errors'])} file(s) could not be loaded"):
        for name, err in st.session_state['errors']:
            st.error(f"**{name}**: {err}")

if 'results' in st.session_state and st.session_state['results']:
    results = st.session_state['results']

    # --- EGYETLEN SEJT NÉZET ---
    if len(results) == 1:
        cell_name, result = results[0]

        st.markdown(f"""
        <div class="page-header">
            <h1>{cell_name}</h1>
            <p>Single cell analysis</p>
        </div>
        """, unsafe_allow_html=True)

        # Felső összefoglaló kártyák
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Soma location", result.soma_region_name)
        with col2:
            proj_count = sum(1 for tr in result.target_results if tr.projects_here)
            st.metric("Confirmed projections", f"{proj_count} / {len(result.target_results)} targets")
        with col3:
            st.metric("Total axon length", f"{result.total_axon_length_um:,.0f} µm")

        st.divider()

        # Célterületek részletes eredménye — egyedi HTML kártyákkal
        st.markdown("**Target Region Results**")
        for tr in result.target_results:
            if tr.projects_here:
                st.markdown(f"""
                <div class="result-card positive">
                    <h4>{tr.region_name} <span style="color:#888;font-weight:400;font-size:0.82rem;">ID {tr.region_id}</span></h4>
                    <span class="tag-yes">Projection confirmed</span>
                    <span class="meta">{tr.projection_point_count} projection points &nbsp;|&nbsp; {tr.axon_length_um:,.1f} µm axon in region</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card negative">
                    <h4>{tr.region_name} <span style="color:#888;font-weight:400;font-size:0.82rem;">ID {tr.region_id}</span></h4>
                    <span class="tag-no">No projection</span>
                    <span class="meta">Axon may pass through but has no endpoints or branch points here.</span>
                </div>
                """, unsafe_allow_html=True)

        # Egyéb vetítési területek
        if result.other_projection_regions:
            st.divider()
            st.markdown("**Other Detected Projection Regions**")
            st.caption("These regions were not in your target list but contain endpoints or branch points.")
            other_df = pd.DataFrame([
                {
                    "Region": r.region_name,
                    "Region ID": r.region_id,
                    "Projection points": r.projection_point_count,
                    "Axon length (µm)": round(r.axon_length_um, 1)
                }
                for r in result.other_projection_regions
            ])
            st.dataframe(other_df, width='stretch', hide_index=True)

        st.divider()

        # 3D vizualizáció
        st.markdown("**3D Visualization**")
        st.caption(
            "Note: the 3D viewer currently opens in a separate desktop window (PyVista). "
            "In-browser rendering via Trame will be available in a future version."
        )
        if st.button("Open 3D Viewer", type="secondary", key="btn_3d_single"):
            with st.spinner("Building 3D plot..."):
                plotter = build_3d_plot(
                    result, atlas_matrix, cell_name,
                    show_soma_region=show_soma_region,
                    show_other_regions=show_other_regions
                )
            show_plot_local(plotter)

    # --- BATCH NÉZET (több sejt) ---
    else:
        st.markdown(f"""
        <div class="page-header">
            <h1>Batch Results</h1>
            <p>{len(results)} cells analyzed</p>
        </div>
        """, unsafe_allow_html=True)

        # Összesített táblázat
        summary_df = results_to_dataframe(results, selected_region_ids, dictionary)
        st.dataframe(summary_df, width='stretch', hide_index=True)

        # Letöltési gomb CSV-ként
        csv_data = summary_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download results as CSV",
            data=csv_data,
            file_name="palyakoveto_batch_results.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.divider()

        # Egyedi sejt részletes nézete
        st.markdown("**Inspect individual cell**")
        inspect_name = st.selectbox(
            "Select a cell to view details",
            options=[name for name, _ in results]
        )
        if inspect_name:
            _, inspect_result = next(r for r in results if r[0] == inspect_name)

            col1, col2, col3 = st.columns(3)
            col1.metric("Soma", inspect_result.soma_region_name)
            proj_count = sum(1 for tr in inspect_result.target_results if tr.projects_here)
            col2.metric("Confirmed projections", f"{proj_count} / {len(inspect_result.target_results)}")
            col3.metric("Total axon length", f"{inspect_result.total_axon_length_um:,.0f} µm")

            st.markdown("<div style='margin-top:0.8rem;'></div>", unsafe_allow_html=True)

            for tr in inspect_result.target_results:
                if tr.projects_here:
                    st.markdown(f"""
                    <div class="result-card positive">
                        <h4>{tr.region_name}</h4>
                        <span class="tag-yes">Projection confirmed</span>
                        <span class="meta">{tr.projection_point_count} projection points &nbsp;|&nbsp; {tr.axon_length_um:.1f} µm</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-card negative">
                        <h4>{tr.region_name}</h4>
                        <span class="tag-no">No projection</span>
                    </div>
                    """, unsafe_allow_html=True)

            if st.button("Open 3D Viewer for this cell", type="secondary", key="btn_3d_batch"):
                with st.spinner("Building 3D plot..."):
                    plotter = build_3d_plot(
                        inspect_result, atlas_matrix, inspect_name,
                        show_soma_region=show_soma_region,
                        show_other_regions=show_other_regions
                    )
                show_plot_local(plotter)
