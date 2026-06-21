# =============================================================================
# APP.PY - A Streamlit alkalmazás belépési pontja.
# Ez a fájl CSAK UI elemet tartalmaz. Semmi tudományos logika nincs itt.
# Futtatás: streamlit run app.py
# =============================================================================

import os
import streamlit as st
import pandas as pd

from config import BASE_DATA_DIR, DEFAULT_TARGET_REGIONS, DEFAULT_FILTER
from core.loader import (
    load_atlas, load_dictionary, load_swc,
    get_all_swc_files, build_region_search_options,
    load_soma_index, build_soma_index, soma_index_exists,
    filter_swc_by_soma_region
)
from core.analysis import (
    run_analysis, apply_filter, results_to_dataframe, FilterCriteria
)
from core.visualization import build_3d_plot, build_3d_plot_multi, show_plot_local

# =============================================================================
# OLDAL KONFIGURÁCIÓ
# =============================================================================
st.set_page_config(
    page_title="Palyakoveto — Neuron Projection Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    :root {
        --cream: #F4F5EE;
        --sage-tint: #E9ECDD;
        --border-sage: #CFD6BC;
        --sage-light: #9DAE89;
        --sage: #5F7350;
        --sage-deep: #33401F;
        --rosewood-tint: #F3E2E0;
        --rosewood-light: #C98F8B;
        --rosewood: #7A3B3B;
        --rosewood-deep: #4A2020;
        --taupe: #8C8775;
        --taupe-tint: #EFEDE6;
        --taupe-deep: #56523F;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
    }

    /* Page + sidebar background: warm cream with a sage-tinted sidebar,
       like a pressed-leaf notebook page. */
    .stApp { background-color: var(--cream); }
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {
        background-color: var(--sage-tint);
    }

    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: var(--sage-deep);
        margin-bottom: 0.1rem;
    }
    .sidebar-subtitle {
        font-size: 0.78rem;
        color: var(--rosewood);
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    .result-card {
        background: var(--cream);
        border: 1px solid var(--border-sage);
        border-left: 3px solid var(--sage);
        border-radius: 4px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.6rem;
    }
    .result-card.positive { border-left-color: var(--sage); }
    .result-card.negative { border-left-color: var(--taupe); }
    .result-card.filtered-out {
        border-left-color: var(--rosewood-deep);
        background: var(--rosewood-tint);
    }
    .result-card h4 {
        margin: 0 0 0.4rem 0;
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--sage-deep);
    }
    .result-card .meta { font-size: 0.82rem; color: var(--taupe-deep); }

    .tag-yes {
        display: inline-block;
        background: var(--sage-tint);
        color: var(--sage-deep);
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        padding: 0.15rem 0.6rem;
        border-radius: 10px;
        text-transform: uppercase;
        margin-right: 0.5rem;
    }
    .tag-no {
        display: inline-block;
        background: var(--taupe-tint);
        color: var(--taupe-deep);
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        padding: 0.15rem 0.6rem;
        border-radius: 10px;
        text-transform: uppercase;
        margin-right: 0.5rem;
    }
    .tag-filtered {
        display: inline-block;
        background: var(--rosewood-tint);
        color: var(--rosewood-deep);
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        padding: 0.15rem 0.6rem;
        border-radius: 10px;
        text-transform: uppercase;
        margin-right: 0.5rem;
    }

    .filter-box {
        background: var(--sage-tint);
        border: 1px solid var(--border-sage);
        border-radius: 4px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
    }
    .filter-box h5 {
        margin: 0 0 0.5rem 0;
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--sage-deep);
    }

    /* Page header: a dashed rule stands in for a myelinated axon segment,
       with the gaps echoing the Nodes of Ranvier. */
    .page-header {
        border-bottom: 3px dashed var(--rosewood-light);
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    .page-header h1 {
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        color: var(--sage-deep);
        margin: 0;
    }
    .page-header p {
        font-size: 0.85rem;
        color: var(--rosewood);
        margin: 0.2rem 0 0 0;
    }

    [data-testid="metric-container"] {
        background: var(--cream);
        border: 1px solid var(--border-sage);
        border-radius: 4px;
        padding: 0.7rem 1rem;
    }

    /* Same dashed-axon motif for every divider in the app. */
    hr { border: none; border-top: 3px dashed var(--sage-light); margin: 1.2rem 0; opacity: 0.9; }

    [data-testid="stDataFrame"] { border: 1px solid var(--border-sage); border-radius: 4px; }

    .stButton > button { border-radius: 14px; font-weight: 600; letter-spacing: 0.03em; }
    [data-testid="baseButton-primary"] {
        background-color: var(--rosewood) !important;
        border-color: var(--rosewood) !important;
        color: var(--cream) !important;
    }
    [data-testid="baseButton-primary"]:hover {
        background-color: var(--rosewood-deep) !important;
        border-color: var(--rosewood-deep) !important;
    }
    [data-testid="baseButton-secondary"] {
        border-color: var(--sage) !important;
        color: var(--sage-deep) !important;
    }
    [data-testid="baseButton-secondary"]:hover {
        background-color: var(--sage-tint) !important;
    }

    /* Selected-region chips in the multiselect, recolored to match. */
    span[data-baseweb="tag"] { background-color: var(--sage) !important; }

    /* A small rosewood dot before every bold section title -
       a quiet nod to a synapse, marking where a new "signal" starts. */
    [data-testid="stMarkdownContainer"] strong::before {
        content: "";
        display: inline-block;
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--rosewood);
        margin-right: 6px;
        vertical-align: middle;
    }

    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# Tiny neuron-and-petal mark used next to the brand name. Six dendrites
# radiating from a soma double as flower petals - the neuroscience/girly
# crossover lives here.
NEURON_MARK = """
<svg width="20" height="20" viewBox="0 0 44 44" xmlns="http://www.w3.org/2000/svg"
     style="vertical-align:-4px;margin-right:6px;">
    <g stroke="#5F7350" stroke-width="1.6" fill="none" stroke-linecap="round">
        <path d="M22 17 Q18 8 12 6"/>
        <path d="M22 17 Q26 7 33 9"/>
        <path d="M27 22 Q36 20 40 14"/>
        <path d="M27 25 Q37 28 41 35"/>
        <path d="M17 27 Q12 35 6 38"/>
        <path d="M17 22 Q7 21 3 16"/>
    </g>
    <g fill="#5F7350">
        <circle cx="12" cy="6" r="1.6"/><circle cx="33" cy="9" r="1.6"/>
        <circle cx="40" cy="14" r="1.6"/><circle cx="41" cy="35" r="1.6"/>
        <circle cx="6" cy="38" r="1.6"/><circle cx="3" cy="16" r="1.6"/>
    </g>
    <circle cx="22" cy="22" r="5" fill="#7A3B3B"/>
</svg>
"""

# =============================================================================
# GLOBÁLIS ADATOK BETÖLTÉSE
# =============================================================================
try:
    atlas_matrix, atlas_header = load_atlas()
    dictionary = load_dictionary()
    region_options = build_region_search_options(dictionary)
except FileNotFoundError as e:
    st.error(f"Data file not found. Please check config.py.\n\n{e}")
    st.stop()

all_swc = get_all_swc_files(BASE_DATA_DIR)

# =============================================================================
# OLDALSÁV
# =============================================================================
with st.sidebar:
    st.markdown(f'<div class="sidebar-title">{NEURON_MARK}Palyakoveto</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">Neuron Projection Analyzer — KOKI</div>', unsafe_allow_html=True)
    st.divider()

    # -------------------------------------------------------------------------
    # SZEKCIÓ 1: Célterületek
    # -------------------------------------------------------------------------
    st.markdown("**Target Brain Regions**")
    st.caption(
        "Projections are detected by endpoints and branch points only. "
        "Axons that merely pass through a region are not counted."
    )

    selected_region_names = st.multiselect(
        label="Search and select regions",
        options=list(region_options.keys()),
        default=[
            name for name in region_options.keys()
            if region_options[name] in DEFAULT_TARGET_REGIONS.values()
        ],
        help="Type to search by region name or acronym.",
        key="region_selector"
    )
    selected_region_ids = [region_options[name] for name in selected_region_names]

    st.divider()

    # -------------------------------------------------------------------------
    # SZEKCIÓ 2: Szűrési feltételek (Phase 1)
    # Minden kiválasztott célterülethez külön szűrők
    # -------------------------------------------------------------------------
    st.markdown("**Projection Filter Criteria**")
    st.caption(
        "Set minimum thresholds per region. All conditions must be met simultaneously. "
        "Leave at 0 to disable filtering."
    )

    # Minden célterülethez egy összecsukható szűrő panel
    # A criteria_per_region szótárba gyűjtjük a feltételeket
    criteria_per_region: dict[int, FilterCriteria] = {}

    if not selected_region_ids:
        st.caption("Select target regions above to set filter criteria.")
    else:
        for region_name_full in selected_region_names:
            region_id = region_options[region_name_full]
            # Rövid megjelenítési nevet kivágunk a zárójelből
            short_name = region_name_full.split('(')[-1].replace(')', '').strip()

            with st.expander(f"Filters for {short_name}", expanded=False):
                min_ep = st.number_input(
                    "Min. endpoints",
                    min_value=0, value=DEFAULT_FILTER['min_endpoints'],
                    step=1,
                    help="Minimum number of axon terminal points in this region.",
                    key=f"filter_ep_{region_id}"
                )
                min_br = st.number_input(
                    "Min. branch points",
                    min_value=0, value=DEFAULT_FILTER['min_branch_points'],
                    step=1,
                    help="Minimum number of axon branching points in this region.",
                    key=f"filter_br_{region_id}"
                )
                min_len = st.number_input(
                    "Min. axon length (µm)",
                    min_value=0.0, value=float(DEFAULT_FILTER['min_axon_length_um']),
                    step=10.0,
                    help="Minimum total axon length within this region.",
                    key=f"filter_len_{region_id}"
                )
                criteria_per_region[region_id] = FilterCriteria(
                    min_endpoints=int(min_ep),
                    min_branch_points=int(min_br),
                    min_axon_length_um=float(min_len)
                )

    st.divider()

    # -------------------------------------------------------------------------
    # SZEKCIÓ 3: Sejtek kiválasztása (Phase 2)
    # Soma régió alapú szűrés + fájlválasztó
    # -------------------------------------------------------------------------
    st.markdown("**Cell Files (SWC)**")

    if not all_swc:
        st.warning(f"No SWC files found in:\n`{BASE_DATA_DIR}`\n\nPlease check BASE_DATA_DIR in config.py")
        selected_swc_paths = []
    else:
        st.caption(f"{len(all_swc)} SWC files available.")

        # --- Soma index kezelése ---
        # Az index nélkül a soma szűrés nem működik, ezért itt kezeljük
        if not soma_index_exists():
            st.warning(
                "Soma region index not built yet. "
                "Build it to enable filtering by soma location."
            )
            if st.button("Build soma index", key="btn_build_index",
                         help="Scans all SWC files to find soma regions. Takes a few minutes."):
                progress_bar = st.progress(0, text="Building soma index...")

                def update_progress(current, total, filename):
                    pct = current / total if total > 0 else 0
                    progress_bar.progress(pct, text=f"Indexing: {filename}")

                with st.spinner("Building soma index..."):
                    build_soma_index(BASE_DATA_DIR, atlas_matrix, dictionary, update_progress)

                progress_bar.empty()
                st.success("Soma index built successfully.")
                st.rerun()

            soma_index = None
        else:
            soma_index = load_soma_index()
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("Rebuild", key="btn_rebuild_index",
                             help="Rebuild the soma index if new SWC files were added."):
                    with st.spinner("Rebuilding soma index..."):
                        build_soma_index(BASE_DATA_DIR, atlas_matrix, dictionary)
                    st.success("Soma index rebuilt.")
                    st.rerun()

        # --- Soma régió szűrő ---
        # Csak akkor jelenik meg ha az index már megvan
        soma_search = ""
        if soma_index is not None:
            with st.expander("Filter by soma region", expanded=False):
                soma_search = st.text_input(
                    "Search soma region",
                    placeholder="e.g. motor, thalamus, striatum...",
                    help="Type part of a region name. Only cells with matching soma regions will be shown.",
                    key="soma_search"
                )
                if soma_search:
                    filtered_swc = filter_swc_by_soma_region(all_swc, soma_index, soma_search)
                    st.caption(f"{len(filtered_swc)} of {len(all_swc)} cells match.")
                else:
                    filtered_swc = all_swc
        else:
            filtered_swc = all_swc

        # --- Analízis mód és fájlválasztás ---
        analysis_mode = st.radio(
            "Analysis mode",
            options=["Single cell", "Batch (multiple cells)"],
            horizontal=True,
            key="analysis_mode"
        )

        if analysis_mode == "Single cell":
            if not filtered_swc:
                st.warning("No cells match the current soma region filter.")
                selected_swc_paths = []
            else:
                selected_name = st.selectbox(
                    "Select cell",
                    options=list(filtered_swc.keys()),
                    help="Format: mouse_folder/cell.swc",
                    key="single_cell_selector"
                )
                selected_swc_paths = [filtered_swc[selected_name]]

                # Ha van soma index, megmutatjuk a kiválasztott sejt soma régióját
                if soma_index is not None:
                    match = soma_index.loc[soma_index['swc_path'] == selected_name, 'soma_region_name']
                    if not match.empty:
                        st.caption(f"Soma region: {match.values[0]}")

        else:
            use_all_matches = False
            if soma_search and len(filtered_swc) < len(all_swc):
                use_all_matches = st.checkbox(
                    f"Use all {len(filtered_swc)} matched cells for batch analysis",
                    value=True,
                    help=(
                        "Skips manual selection below and runs the batch on every cell "
                        "that matched the soma region filter. Uncheck to pick cells by hand instead."
                    ),
                    key="use_all_matches"
                )

            if use_all_matches:
                selected_swc_paths = list(filtered_swc.values())
                st.caption(
                    f"{len(selected_swc_paths)} cells will be analyzed "
                    f"(all matches for \"{soma_search}\")."
                )
            else:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Select all", use_container_width=True, key="btn_select_all"):
                        st.session_state['batch_selection'] = list(filtered_swc.keys())
                with col2:
                    if st.button("Clear", use_container_width=True, key="btn_clear"):
                        st.session_state['batch_selection'] = []

                batch_default = [
                    k for k in st.session_state.get('batch_selection', [])
                    if k in filtered_swc
                ]
                selected_names = st.multiselect(
                    "Select cells for batch analysis",
                    options=list(filtered_swc.keys()),
                    default=batch_default,
                    help="Select multiple cells. Max recommended: ~50 at once.",
                    key="batch_selector"
                )
                selected_swc_paths = [filtered_swc[name] for name in selected_names]

    st.divider()

    # -------------------------------------------------------------------------
    # SZEKCIÓ 4: Vizualizáció beállítások
    # -------------------------------------------------------------------------
    st.markdown("**Visualization**")
    show_soma_region = st.toggle("Show soma region", value=True, key="toggle_soma")
    show_other_regions = st.toggle("Show other projection regions", value=True, key="toggle_other")

    st.divider()
    st.caption("Atlas: Allen Mouse Brain Atlas (25 µm)")
    st.caption("Palyakoveto v0.2")

# =============================================================================
# FŐ TARTALOM
# =============================================================================

if not selected_swc_paths or not selected_region_ids:
    st.markdown(f"""
    <div class="page-header">
        <h1>{NEURON_MARK}Palyakoveto</h1>
        <p>Neuron Projection Analyzer &mdash; Koki Institute</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "**To get started**, select target brain regions and one or more cell files "
        "in the sidebar, then click Run Analysis."
    )

    with st.expander("About the projection detection method"):
        st.markdown("""
        **What makes Palyakoveto different from other tools?**

        Many tools mark a neuron as projecting to a region simply because the axon passes
        through that region. This leads to false positives.

        Palyakoveto uses a stricter anatomical criterion: a neuron is counted as projecting
        to a region only if it has an **endpoint** or **branch point** located within that region.

        | Point type | Definition |
        |---|---|
        | **Endpoint** | The axon terminates here — a likely synaptic target |
        | **Branch point** | The axon splits here — suggesting local arborization |
        | Pass-through | Axon crosses the region but does not end or branch — not counted |

        Additionally, you can set **minimum thresholds** for endpoints, branch points, and axon
        length per region, allowing precise population-level filtering.
        """)
    st.stop()

# =============================================================================
# ANALÍZIS FUTTATÁSA
# =============================================================================

# Ellenőrizzük, hogy van-e aktív szűrési feltétel, hogy a gomb szövege informatív legyen
any_filter_active = any(c.is_active() for c in criteria_per_region.values())
filter_note = "  (filter active)" if any_filter_active else ""

run_button = st.button(
    f"Run Analysis  —  {len(selected_swc_paths)} cell{'s' if len(selected_swc_paths) > 1 else ''}{filter_note}",
    type="primary",
    use_container_width=True,
    key="btn_run_analysis"
)

if run_button:
    st.session_state['results'] = []
    st.session_state['errors'] = []
    st.session_state['criteria_per_region'] = criteria_per_region

    progress = st.progress(0, text="Analyzing cells...")

    for i, filepath in enumerate(selected_swc_paths):
        # Relatív útvonal visszakeresése a megjelenítéshez
        cell_name = next(
            (k for k, v in filtered_swc.items() if v == filepath),
            os.path.basename(filepath)
        )
        try:
            swc_df = load_swc(filepath)
            result = run_analysis(swc_df, atlas_matrix, dictionary, selected_region_ids)
            # Szűrés alkalmazása az analízis után
            result = apply_filter(result, criteria_per_region)
            st.session_state['results'].append((cell_name, result))
        except Exception as e:
            st.session_state['errors'].append((cell_name, str(e)))

        progress.progress(
            (i + 1) / len(selected_swc_paths),
            text=f"Analyzed: {cell_name}"
        )

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
    saved_criteria = st.session_state.get('criteria_per_region', {})
    filter_was_active = any(c.is_active() for c in saved_criteria.values())

    # -------------------------------------------------------------------------
    # EGYETLEN SEJT NÉZET
    # -------------------------------------------------------------------------
    if len(results) == 1:
        cell_name, result = results[0]

        # Szűrési státusz megjelenítése a fejlécben
        filter_status_html = ""
        if result.passes_filter is True:
            filter_status_html = '<span style="color:#2a7f4f;font-size:0.85rem;font-weight:600;">Passes filter</span>'
        elif result.passes_filter is False:
            filter_status_html = '<span style="color:#c0392b;font-size:0.85rem;font-weight:600;">Filtered out</span>'

        st.markdown(f"""
        <div class="page-header">
            <h1>{cell_name}</h1>
            <p>Single cell analysis &nbsp; {filter_status_html}</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Soma location", result.soma_region_name)
        with col2:
            proj_count = sum(1 for tr in result.target_results if tr.projects_here)
            st.metric("Confirmed projections", f"{proj_count} / {len(result.target_results)} targets")
        with col3:
            st.metric("Total axon length", f"{result.total_axon_length_um:,.0f} µm")

        st.divider()

        # Célterületek eredménykártyái - részletes endpoint/branch bontással
        st.markdown("**Target Region Results**")
        for tr in result.target_results:
            cr = saved_criteria.get(tr.region_id, FilterCriteria())
            # A kártya piros ha szűrés volt aktív és a sejt nem felel meg erre a régióra
            fails_this = filter_was_active and not cr.check(tr)

            if fails_this:
                st.markdown(f"""
                <div class="result-card filtered-out">
                    <h4>{tr.region_name} <span style="color:#888;font-weight:400;font-size:0.82rem;">ID {tr.region_id}</span></h4>
                    <span class="tag-filtered">Does not meet criteria</span>
                    <span class="meta">
                        Endpoints: {tr.endpoint_count} &nbsp;|&nbsp;
                        Branch points: {tr.branch_point_count} &nbsp;|&nbsp;
                        Axon: {tr.axon_length_um:,.1f} µm
                    </span>
                </div>
                """, unsafe_allow_html=True)
            elif tr.projects_here:
                st.markdown(f"""
                <div class="result-card positive">
                    <h4>{tr.region_name} <span style="color:#888;font-weight:400;font-size:0.82rem;">ID {tr.region_id}</span></h4>
                    <span class="tag-yes">Projection confirmed</span>
                    <span class="meta">
                        Endpoints: {tr.endpoint_count} &nbsp;|&nbsp;
                        Branch points: {tr.branch_point_count} &nbsp;|&nbsp;
                        Axon: {tr.axon_length_um:,.1f} µm
                    </span>
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
                    "Endpoints": r.endpoint_count,
                    "Branch points": r.branch_point_count,
                    "Axon length (µm)": round(r.axon_length_um, 1)
                }
                for r in result.other_projection_regions
            ])
            st.dataframe(other_df, use_container_width=True, hide_index=True)

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

    # -------------------------------------------------------------------------
    # BATCH NÉZET
    # -------------------------------------------------------------------------
    else:
        # Megszámoljuk a szűrést átmenő és kieső sejteket
        passed = sum(1 for _, r in results if r.passes_filter is True)
        failed = sum(1 for _, r in results if r.passes_filter is False)
        unfiltered = sum(1 for _, r in results if r.passes_filter is None)

        st.markdown(f"""
        <div class="page-header">
            <h1>Batch Results</h1>
            <p>{len(results)} cells analyzed</p>
        </div>
        """, unsafe_allow_html=True)

        # Összesítő metrikák
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total cells", len(results))
        if filter_was_active:
            col2.metric("Pass filter", passed)
            col3.metric("Filtered out", failed)
        else:
            col2.metric("Cells with projections",
                        sum(1 for _, r in results
                            if any(tr.projects_here for tr in r.target_results)))
        col4.metric("Load errors", len(st.session_state.get('errors', [])))

        st.divider()

        # Teljes eredménytáblázat
        summary_df = results_to_dataframe(results, selected_region_ids, dictionary)

        # Ha volt szűrés, a passes_filter oszlop alapján rendezzük
        if filter_was_active:
            summary_df = summary_df.sort_values('passes_filter', ascending=False)

        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        csv_data = summary_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download results as CSV",
            data=csv_data,
            file_name="palyakoveto_batch_results.csv",
            mime="text/csv",
            use_container_width=True,
            key="btn_download_csv"
        )

        st.divider()

        # Összes sejt egyszerre egy 3D ábrán - batch összehasonlításhoz
        st.markdown("**Combined 3D Visualization**")
        st.caption(
            "Plot every analyzed cell together on one 3D scene. Each cell gets its own "
            "color (soma + full axon tree) so you can tell them apart. "
            "Recommended for at most a few dozen cells at once — large batches will be slow."
        )

        max_combined = 60
        if len(results) > max_combined:
            st.warning(
                f"{len(results)} cells in this batch — rendering all of them together "
                f"would be very slow. Only the first {max_combined} will be shown."
            )
        combined_results = results[:max_combined]

        if st.button(
            f"Open combined 3D Viewer — {len(combined_results)} cells",
            type="secondary",
            key="btn_3d_combined"
        ):
            with st.spinner(f"Building combined 3D plot for {len(combined_results)} cells..."):
                plotter = build_3d_plot_multi(
                    combined_results, atlas_matrix, selected_region_ids,
                    show_target_regions=True
                )
            show_plot_local(plotter)

        st.divider()

        # Egyedi sejt részletes nézete a batch-ből
        st.markdown("**Inspect individual cell**")
        inspect_name = st.selectbox(
            "Select a cell to view details",
            options=[name for name, _ in results],
            key="batch_inspect_selector"
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
                cr = saved_criteria.get(tr.region_id, FilterCriteria())
                fails_this = filter_was_active and not cr.check(tr)

                if fails_this:
                    st.markdown(f"""
                    <div class="result-card filtered-out">
                        <h4>{tr.region_name}</h4>
                        <span class="tag-filtered">Does not meet criteria</span>
                        <span class="meta">Endpoints: {tr.endpoint_count} | Branch points: {tr.branch_point_count} | Axon: {tr.axon_length_um:.1f} µm</span>
                    </div>
                    """, unsafe_allow_html=True)
                elif tr.projects_here:
                    st.markdown(f"""
                    <div class="result-card positive">
                        <h4>{tr.region_name}</h4>
                        <span class="tag-yes">Projection confirmed</span>
                        <span class="meta">Endpoints: {tr.endpoint_count} | Branch points: {tr.branch_point_count} | Axon: {tr.axon_length_um:.1f} µm</span>
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