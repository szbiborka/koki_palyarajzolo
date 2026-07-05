# =============================================================================
# APP.PY - Streamlit Application Entry Point
# Contains ONLY UI elements. No scientific logic is executed here.
# Run with: streamlit run app.py
# =============================================================================

import os
import streamlit as st
import pandas as pd

from config import BASE_DATA_DIR, DEFAULT_TARGET_REGIONS, DEFAULT_FILTER
from core.loader import (
    load_atlas, load_dictionary, load_swc,
    get_all_swc_files, build_region_search_options,
    load_soma_index, build_soma_index, soma_index_exists,
    filter_swc_by_soma_region, build_region_descendants
)
from core.analysis import (
    run_analysis, apply_filter, results_to_dataframe, FilterCriteria
)
from core.visualization import (
    build_3d_plot, build_3d_plot_multi, render_plot_streamlit
)

# =============================================================================
# PAGE CONFIGURATION & CSS
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
        --cream: #F9FAED;
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

    /* Background and sidebar */
    .stApp { background-color: var(--cream); }
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {
        background-color: var(--sage-tint);
        border-right: 1px solid var(--border-sage);
    }

    /* Headers */
    .sidebar-title {
        font-size: 1.25rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: var(--sage-deep);
        margin-bottom: 0.2rem;
        display: flex;
        align-items: center;
    }
    .sidebar-subtitle {
        font-size: 0.75rem;
        color: var(--rosewood);
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 1.5rem;
        font-weight: 600;
    }

    /* Floating professional cards */
    .result-card {
        background: #FFFFFF;
        border: none;
        border-left: 4px solid var(--sage);
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.02), 0 1px 3px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.05);
    }
    .result-card.positive { border-left-color: var(--sage); }
    .result-card.negative { border-left-color: var(--taupe); opacity: 0.85; }
    .result-card.filtered-out {
        border-left-color: var(--rosewood-light);
        background: var(--rosewood-tint);
    }

    .result-card h4 {
        margin: 0 0 0.5rem 0;
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--sage-deep);
    }
    .result-card .meta { font-size: 0.85rem; color: var(--taupe-deep); }

    /* Tags */
    .tag-yes, .tag-no, .tag-filtered {
        display: inline-block;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        padding: 0.2rem 0.7rem;
        border-radius: 12px;
        text-transform: uppercase;
        margin-right: 0.6rem;
        margin-bottom: 0.4rem;
    }
    .tag-yes { background: var(--sage-tint); color: var(--sage-deep); }
    .tag-no { background: var(--taupe-tint); color: var(--taupe-deep); }
    .tag-filtered { background: rgba(201, 143, 139, 0.2); color: var(--rosewood-deep); }

    /* Main page header */
    .page-header {
        margin-bottom: 2rem;
    }
    .page-header h1 {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: 0.01em;
        color: var(--sage-deep);
        margin: 0;
    }
    .page-header p {
        font-size: 0.9rem;
        color: var(--rosewood);
        font-weight: 600;
        margin: 0.3rem 0 0 0;
    }

    /* Custom divider (Myelin/Synapse style) */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(to right, transparent, var(--border-sage), transparent);
        position: relative;
        margin: 2.5rem 0;
    }
    hr::after {
        content: "◆";
        color: var(--rosewood-light);
        font-size: 14px;
        position: absolute;
        left: 50%;
        top: -10px;
        transform: translateX(-50%);
    }

    /* Button refinements */
    .stButton > button { 
        border-radius: 6px; 
        font-weight: 600; 
        letter-spacing: 0.03em; 
        transition: all 0.2s;
    }
    [data-testid="baseButton-primary"] {
        background-color: var(--rosewood) !important;
        border-color: var(--rosewood) !important;
        color: var(--cream) !important;
        box-shadow: 0 4px 6px rgba(122, 59, 59, 0.2);
    }
    [data-testid="baseButton-primary"]:hover {
        background-color: var(--rosewood-deep) !important;
        box-shadow: 0 6px 12px rgba(122, 59, 59, 0.3);
    }

    /* Smaller, refined tabs */
    [data-testid="stTabs"] button {
        font-weight: 600;
        color: var(--taupe-deep);
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--sage-deep);
        border-bottom-color: var(--sage);
    }

    footer, #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CUSTOM ICONS (SVG)
# =============================================================================
NEURON_MARK = """
<svg width="24" height="24" viewBox="0 0 44 44" xmlns="http://www.w3.org/2000/svg"
     style="vertical-align:-6px;margin-right:8px;">
    <g stroke="#5F7350" stroke-width="2" fill="none" stroke-linecap="round">
        <path d="M22 17 Q18 8 12 6"/><path d="M22 17 Q26 7 33 9"/>
        <path d="M27 22 Q36 20 40 14"/><path d="M27 25 Q37 28 41 35"/>
        <path d="M17 27 Q12 35 6 38"/><path d="M17 22 Q7 21 3 16"/>
    </g>
    <g fill="#5F7350">
        <circle cx="12" cy="6" r="2"/><circle cx="33" cy="9" r="2"/>
        <circle cx="40" cy="14" r="2"/><circle cx="41" cy="35" r="2"/>
        <circle cx="6" cy="38" r="2"/><circle cx="3" cy="16" r="2"/>
    </g>
    <circle cx="22" cy="22" r="5" fill="#7A3B3B"/>
</svg>
"""

SYNAPSE_MARK = """
<svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" 
     style="vertical-align:-1px;margin-right:6px;">
    <circle cx="12" cy="12" r="7" stroke="#7A3B3B" stroke-width="3"/>
    <circle cx="12" cy="12" r="3" fill="#5F7350"/>
</svg>
"""

# Display label -> internal operator code for the per-region projection rule.
# Kept as an explicit mapping (rather than parsing substrings of the label)
# so the displayed wording can change freely without touching the logic.
RULE_OPERATORS = {
    "Required (AND)": "AND",
    "Excluded (NOT)": "NOT",
    "Optional (OR)": "OR",
}


def section_header(title: str):
    """Generates a custom formatted subheader with the SYNAPSE_MARK icon."""
    st.markdown(f"<h4>{SYNAPSE_MARK}{title}</h4>", unsafe_allow_html=True)


# =============================================================================
# GLOBAL DATA LOADING
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
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown(f'<div class="sidebar-title">{NEURON_MARK} Palyakoveto</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">Neuron Projection Analyzer</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 1. Target Regions
    # -------------------------------------------------------------------------
    st.markdown("**Target Brain Regions**",
                help="Specify the regions where you are looking for projections (innervations). Deep analysis will only be run for the regions in this list.")
    selected_region_names = st.multiselect(
        label="Search and select regions",
        options=list(region_options.keys()),
        default=[name for name in region_options.keys() if region_options[name] in DEFAULT_TARGET_REGIONS.values()],
        help="Start typing the region name or acronym (e.g., M2, GPe).",
        key="region_selector",
        label_visibility="collapsed"
    )
    selected_region_ids = [region_options[name] for name in selected_region_names]

    st.divider()

    # -------------------------------------------------------------------------
    # 2. Filter Criteria & Logic (AND/OR/NOT)
    # -------------------------------------------------------------------------
    st.markdown("**Projection Filter Criteria**",
                help="Define what counts as a valid projection. If a cell's axon passes through the area but does not meet these minimums, it will be marked as 'Filtered out'.")

    criteria_per_region: dict[int, FilterCriteria] = {}

    if not selected_region_ids:
        st.caption("Select target regions above to set filter criteria.")
    else:
        for region_name_full in selected_region_names:
            region_id = region_options[region_name_full]
            short_name = region_name_full.split('(')[-1].replace(')', '').strip()

            with st.expander(f"Filters for {short_name}", expanded=False):
                st.markdown(
                    "<div style='font-size:0.78rem;font-weight:700;letter-spacing:0.04em;"
                    "text-transform:uppercase;color:var(--taupe-deep);margin-bottom:0.3rem;'>"
                    "Condition Rule</div>",
                    unsafe_allow_html=True
                )
                rule_label = st.radio(
                    label="Condition rule",
                    options=list(RULE_OPERATORS.keys()),
                    horizontal=True,
                    help=(
                        "Required: the cell must project here for the thresholds below to be "
                        "evaluated. Excluded: the cell is disqualified if it projects here at "
                        "all. Optional: satisfying this region counts toward an OR-combination "
                        "with other Optional regions, but is not mandatory on its own."
                    ),
                    key=f"filter_rule_{region_id}",
                    label_visibility="collapsed",
                )
                op = RULE_OPERATORS[rule_label]

                min_ep = st.number_input(
                    "Min. endpoints", min_value=0, value=DEFAULT_FILTER['min_endpoints'], step=1,
                    help=f"At least this many axon endpoints must fall within the {short_name} region for the projection to be considered valid.",
                    key=f"filter_ep_{region_id}"
                )
                min_br = st.number_input(
                    "Min. branch points", min_value=0, value=DEFAULT_FILTER['min_branch_points'], step=1,
                    help="At least this many branch points are required within the region.",
                    key=f"filter_br_{region_id}"
                )
                min_len = st.number_input(
                    "Min. axon length (µm)", min_value=0.0, value=float(DEFAULT_FILTER['min_axon_length_um']),
                    step=10.0,
                    help="Minimum total length of axon segments within the target region.",
                    key=f"filter_len_{region_id}"
                )
                min_ep_pct = st.number_input(
                    "Min. endpoint share (%)", min_value=0.0, max_value=100.0,
                    value=float(DEFAULT_FILTER['min_endpoint_fraction'] * 100), step=0.5,
                    help=(
                        "Size-independent threshold: the share of the cell's TOTAL axon "
                        "endpoints that must fall inside this region. Combine with the "
                        "'Excluded (NOT)' rule to remove Layer 6 cells — e.g. set 2.5% on the "
                        "thalamus and mark it NOT to drop cells whose endpoints are mostly thalamic."
                    ),
                    key=f"filter_eppct_{region_id}"
                )

                criteria_per_region[region_id] = FilterCriteria(
                    min_endpoints=int(min_ep),
                    min_branch_points=int(min_br),
                    min_axon_length_um=float(min_len),
                    min_endpoint_fraction=float(min_ep_pct) / 100.0,
                    operator=op  # Átadjuk az operátort a backendnek
                )

    st.divider()

    # -------------------------------------------------------------------------
    # 3. Cell Selection
    # -------------------------------------------------------------------------
    st.markdown("**Cell Files (SWC)**",
                help="Select the neurons to analyze. You can use the soma filter to find targeted populations.")

    if not all_swc:
        st.warning(f"No SWC files found in:\n`{BASE_DATA_DIR}`")
        selected_swc_paths = []
    else:
        if not soma_index_exists():
            st.warning("Soma region index not built yet.")
            if st.button("Build soma index", key="btn_build_index",
                         help="Required on first run. Scans all SWC files to extract soma locations."):
                progress_bar = st.progress(0, text="Building soma index...")


                def update_progress(current, total, filename):
                    progress_bar.progress(current / total if total > 0 else 0, text=f"Indexing: {filename}")


                with st.spinner("Building soma index..."):
                    build_soma_index(BASE_DATA_DIR, atlas_matrix, dictionary, update_progress)
                progress_bar.empty()
                st.rerun()
            soma_index = None
        else:
            soma_index = load_soma_index()
            if st.button("Rebuild Index", key="btn_rebuild_index", use_container_width=True,
                         help="Update this if you copied new SWC files into the data folder."):
                with st.spinner("Rebuilding soma index..."):
                    build_soma_index(BASE_DATA_DIR, atlas_matrix, dictionary)
                st.rerun()

        soma_search = ""
        if soma_index is not None:
            soma_search = st.text_input(
                "Filter by soma region", placeholder="e.g. motor, thalamus...",
                help="Lists only cells whose soma is located in this region. Partial matches are accepted.",
                key="soma_search"
            )
            filtered_swc = filter_swc_by_soma_region(all_swc, soma_index, soma_search) if soma_search else all_swc
        else:
            filtered_swc = all_swc

        analysis_mode = st.radio("Analysis mode", options=["Single cell", "Batch (multiple cells)"], horizontal=True,
                                 key="analysis_mode")

        if analysis_mode == "Single cell":
            selected_name = st.selectbox("Select cell", options=list(filtered_swc.keys()),
                                         help="Select the specific SWC file for analysis.", key="single_cell_selector")
            selected_swc_paths = [filtered_swc[selected_name]] if filtered_swc else []
        else:
            st.markdown(f"**{len(filtered_swc)} cells available for batch analysis.**")

            batch_method = st.radio(
                "Selection method",
                options=["Analyze ALL matched cells", "Select specific cells manually"],
                horizontal=True,
                label_visibility="collapsed"
            )

            if batch_method == "Analyze ALL matched cells":
                selected_swc_paths = list(filtered_swc.values())
                st.info(f"Ready to analyze all **{len(selected_swc_paths)}** cells. Click 'Run Analysis' below.")
            else:
                selected_names = st.multiselect(
                    "Select specific cells",
                    options=list(filtered_swc.keys()),
                    default=[],
                    help="Start typing to search. Select only the cells you need. (A maximum of 50-60 is recommended for joint 3D rendering.)",
                    key="batch_selector"
                )
                selected_swc_paths = [filtered_swc[name] for name in selected_names]

                if not selected_swc_paths:
                    st.warning("Please select at least one cell from the dropdown.")

    st.divider()

    # -------------------------------------------------------------------------
    # 4. Visualization Settings (With Axon-in-Region)
    # -------------------------------------------------------------------------
    st.markdown("**Visualization Settings**",
                help="These toggles only affect the appearance of the 3D Plotly scene, not the numerical analysis.")
    show_soma_region = st.toggle("Show soma region", value=True, key="toggle_soma")
    show_other_regions = st.toggle("Show other projection regions", value=True, key="toggle_other")

    # --- ÚJ KAPCSOLÓ AZ AXON-IN-REGION NÉZETHEZ ---
    show_only_target_regions = st.toggle(
        "Axon-in-region view", value=False,
        help="If toggled, the system hides all axon branches outside the examined regions, clearing up the visual noise.",
        key="toggle_exclusive"
    )

# =============================================================================
# MAIN CONTENT (TABS LAYOUT)
# =============================================================================

if not selected_swc_paths or not selected_region_ids:
    st.markdown(f"""
    <div class="page-header" style="text-align: center; margin-top: 10vh;">
        <div style="display: flex; justify-content: center; margin-bottom: 20px;">{NEURON_MARK}</div>
        <h1>Palyakoveto</h1>
        <p>Neuron Projection Analyzer &mdash; HUN-REN KOKI</p>
    </div>
    """, unsafe_allow_html=True)

    st.info("**Welcome.** Select target brain regions and one or more cell files from the sidebar to begin.")
    st.stop()

# --- Run Button ---
any_filter_active = any(c.is_active() for c in criteria_per_region.values())
filter_note = " (Filters active)" if any_filter_active else ""

_, col_btn, _ = st.columns([1, 2, 1])
with col_btn:
    run_button = st.button(
        f"Run Analysis for {len(selected_swc_paths)} cell{'s' if len(selected_swc_paths) > 1 else ''}{filter_note}",
        type="primary", use_container_width=True
    )

if run_button:
    st.session_state['results'] = []
    st.session_state['errors'] = []
    st.session_state['criteria_per_region'] = criteria_per_region

    # A célterületek szülő->leszármazott feloldása (pl. Brain stem, Thalamus az
    # összes alárendelt magjukra). Egyszer építjük fel, minden sejtre ezt használjuk.
    region_descendants = build_region_descendants(dictionary, selected_region_ids)

    progress = st.progress(0, text="Analyzing cells...")
    for i, filepath in enumerate(selected_swc_paths):
        cell_name = next((k for k, v in filtered_swc.items() if v == filepath), os.path.basename(filepath))
        try:
            swc_df = load_swc(filepath)
            result = run_analysis(swc_df, atlas_matrix, dictionary, selected_region_ids, region_descendants)
            result = apply_filter(result, criteria_per_region)
            if len(st.session_state['results']) >= 60:
                result.coords = {}
            st.session_state['results'].append((cell_name, result))
        except Exception as e:
            st.session_state['errors'].append((cell_name, str(e)))
        progress.progress((i + 1) / len(selected_swc_paths), text=f"Analyzed: {cell_name}")
    progress.empty()

# --- Display Results ---
if 'errors' in st.session_state and st.session_state['errors']:
    with st.expander(f"{len(st.session_state['errors'])} file(s) could not be loaded"):
        for name, err in st.session_state['errors']:
            st.error(f"**{name}**: {err}")

if 'results' in st.session_state and st.session_state['results']:
    results = st.session_state['results']
    saved_criteria = st.session_state.get('criteria_per_region', {})
    filter_was_active = any(c.is_active() for c in saved_criteria.values())

    st.divider()

    # =========================================================================
    # SINGLE CELL VIEW
    # =========================================================================
    if len(results) == 1:
        cell_name, result = results[0]

        tab_data, tab_3d = st.tabs(["Analytics & Data", "Interactive 3D Viewer"])

        with tab_data:
            if result.passes_filter is True:
                filter_status = '<span class="tag-yes" style="margin-left:15px;">Passes filter</span>'
            elif result.passes_filter is False:
                filter_status = '<span class="tag-filtered" style="margin-left:15px;">Filtered out</span>'
            else:
                filter_status = ""
            st.markdown(
                f"<h3>{cell_name}{filter_status}</h3>",
                unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Soma location", result.soma_region_name)
            proj_count = sum(1 for tr in result.target_results if tr.projects_here)
            m2.metric("Confirmed projections", f"{proj_count} / {len(result.target_results)} targets")
            m3.metric("Total axon length", f"{result.total_axon_length_um:,.0f} µm")

            st.markdown("<br>", unsafe_allow_html=True)
            section_header("Target Region Results")

            for tr in result.target_results:
                cr = saved_criteria.get(tr.region_id, FilterCriteria())

                # Vizsgáljuk, hogy ez a régió egyáltalán "aktív" szűrő-e
                is_active_rule = cr.is_active() and filter_was_active
                meets_rule = cr.meets_thresholds(tr)

                # Ha a NOT szűrőt bukja el a sejt, azt külön kiemeljük
                if is_active_rule and cr.operator == 'NOT' and meets_rule:
                    st.markdown(f"""
                    <div class="result-card filtered-out">
                        <h4>{tr.region_name} <span style="color:#888;font-weight:400;font-size:0.82rem;">ID {tr.region_id}</span></h4>
                        <span class="tag-filtered">Violated NOT rule</span>
                        <div class="meta" style="margin-top:6px;">Endpoints: <b>{tr.endpoint_count}</b> &nbsp;|&nbsp; Branch points: <b>{tr.branch_point_count}</b> &nbsp;|&nbsp; Axon: <b>{tr.axon_length_um:,.1f} µm</b></div>
                    </div>
                    """, unsafe_allow_html=True)
                elif is_active_rule and cr.operator == 'AND' and not meets_rule:
                    st.markdown(f"""
                    <div class="result-card filtered-out">
                        <h4>{tr.region_name} <span style="color:#888;font-weight:400;font-size:0.82rem;">ID {tr.region_id}</span></h4>
                        <span class="tag-filtered">Did not meet thresholds</span>
                        <div class="meta" style="margin-top:6px;">Endpoints: <b>{tr.endpoint_count}</b> &nbsp;|&nbsp; Branch points: <b>{tr.branch_point_count}</b> &nbsp;|&nbsp; Axon: <b>{tr.axon_length_um:,.1f} µm</b></div>
                    </div>
                    """, unsafe_allow_html=True)
                elif tr.projects_here:
                    st.markdown(f"""
                    <div class="result-card positive">
                        <h4>{tr.region_name} <span style="color:#888;font-weight:400;font-size:0.82rem;">ID {tr.region_id}</span></h4>
                        <span class="tag-yes">Projection Confirmed</span>
                        <div class="meta" style="margin-top:6px;">Endpoints: <b>{tr.endpoint_count}</b> &nbsp;|&nbsp; Branch points: <b>{tr.branch_point_count}</b> &nbsp;|&nbsp; Axon: <b>{tr.axon_length_um:,.1f} µm</b></div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-card negative">
                        <h4>{tr.region_name} <span style="color:#888;font-weight:400;font-size:0.82rem;">ID {tr.region_id}</span></h4>
                        <span class="tag-no">No Projection</span>
                        <div class="meta" style="margin-top:6px;">Axon may pass through but has no endpoints or branch points here.</div>
                    </div>
                    """, unsafe_allow_html=True)

            if result.other_projection_regions:
                st.markdown("<br>", unsafe_allow_html=True)
                section_header("Other Detected Projections")
                other_df = pd.DataFrame([{"Region": r.region_name, "Endpoints": r.endpoint_count,
                                          "Branch points": r.branch_point_count,
                                          "Length (µm)": round(r.axon_length_um, 1)} for r in
                                         result.other_projection_regions])
                st.dataframe(other_df, use_container_width=True, hide_index=True)

        with tab_3d:
            st.info("**Tip:** Use the left mouse button to rotate, the right button to pan, and the scroll wheel to zoom.")
            with st.spinner("Building interactive 3D plot..."):
                fig = build_3d_plot(
                    result, atlas_matrix, cell_name,
                    show_soma_region=show_soma_region,
                    show_other_regions=show_other_regions,
                    show_only_target_regions=show_only_target_regions  # Bepasszoljuk a UI kapcsolót
                )
            st.plotly_chart(fig, use_container_width=True)

    # =========================================================================
    # BATCH VIEW
    # =========================================================================
    else:
        tab_stats, tab_inspector, tab_3d_multi = st.tabs(
            ["Population Statistics", "Single Cell Inspector", "Combined 3D View"])

        with tab_stats:
            passed = sum(1 for _, r in results if r.passes_filter is True)
            failed = sum(1 for _, r in results if r.passes_filter is False)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total cells analyzed", len(results))
            if filter_was_active:
                c2.metric("Passed filter", passed)
                c3.metric("Filtered out", failed)
            else:
                c2.metric("Projecting cells",
                          sum(1 for _, r in results if any(tr.projects_here for tr in r.target_results)))
            c4.metric("Load errors", len(st.session_state.get('errors', [])))

            st.markdown("<br>", unsafe_allow_html=True)
            section_header("Soma Region Distribution")
            st.caption(
                "Distribution of cell bodies across different regions and the number of successfully projecting cells.")

            soma_counts = {}
            for cell_name, r in results:
                soma = r.soma_region_name
                if soma not in soma_counts:
                    soma_counts[soma] = {'total': 0, 'projecting': 0, 'ids': []}

                soma_counts[soma]['total'] += 1

                if filter_was_active:
                    is_projecting = bool(r.passes_filter)
                else:
                    is_projecting = any(tr.projects_here for tr in r.target_results)

                if is_projecting:
                    soma_counts[soma]['projecting'] += 1
                    # A vetítő sejt sorszáma Nóra formátumában (a .swc kiterjesztés nélkül),
                    # hogy vissza lehessen keresni az adatbázisban.
                    cell_id = cell_name[:-4] if cell_name.lower().endswith('.swc') else cell_name
                    soma_counts[soma]['ids'].append(cell_id)

            soma_df = pd.DataFrame([
                {
                    "Soma Region": soma,
                    "Total Cells": data['total'],
                    "Valid Projections": data['projecting'],
                    # Százalékos arány: a vetítő sejtek hányada a régió összes sejtjéből.
                    "Valid Projections %": round(
                        100 * data['projecting'] / data['total'], 1
                    ) if data['total'] > 0 else 0.0,
                    # A vetítő sejtek sorszámai (mint a korábbi CSV-kben).
                    "Projecting Cell IDs": ", ".join(data['ids']),
                }
                for soma, data in soma_counts.items()
            ]).sort_values(by="Total Cells", ascending=False)

            st.dataframe(
                soma_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Valid Projections %": st.column_config.NumberColumn(
                        "Valid Projections %", format="%.1f%%",
                        help="Valid Projections ÷ Total Cells, region by region."
                    ),
                    "Projecting Cell IDs": st.column_config.TextColumn(
                        "Projecting Cell IDs", width="large",
                        help="Serial numbers of the cells that pass, for lookup in the database."
                    ),
                },
            )

            soma_csv = soma_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Soma Region Summary (CSV)", data=soma_csv,
                file_name="soma_region_summary.csv", mime="text/csv",
                key="download_soma_summary"
            )

            st.markdown("<br>", unsafe_allow_html=True)
            section_header("Detailed Batch Data")

            summary_df = results_to_dataframe(results, selected_region_ids, dictionary)
            if filter_was_active:
                summary_df = summary_df.sort_values('passes_filter', ascending=False)

            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            csv_data = summary_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Dataset (CSV)", data=csv_data, file_name="batch_results.csv", mime="text/csv")

        with tab_inspector:
            st.markdown("Select a single cell from the processed population to view detailed metrics and its 3D scene.")
            inspect_name = st.selectbox("Select cell to inspect", options=[name for name, _ in results],
                                        label_visibility="collapsed")

            if inspect_name:
                _, inspect_result = next(r for r in results if r[0] == inspect_name)

                for tr in inspect_result.target_results:
                    cr = saved_criteria.get(tr.region_id, FilterCriteria())

                    is_active_rule = cr.is_active() and filter_was_active
                    meets_rule = cr.meets_thresholds(tr)

                    if is_active_rule and cr.operator == 'NOT' and meets_rule:
                        st.markdown(
                            f'<div class="result-card filtered-out"><h4>{tr.region_name}</h4><span class="tag-filtered">Violated NOT rule</span></div>',
                            unsafe_allow_html=True)
                    elif is_active_rule and cr.operator == 'AND' and not meets_rule:
                        st.markdown(
                            f'<div class="result-card filtered-out"><h4>{tr.region_name}</h4><span class="tag-filtered">Did not meet thresholds</span></div>',
                            unsafe_allow_html=True)
                    elif tr.projects_here:
                        st.markdown(
                            f'<div class="result-card positive"><h4>{tr.region_name}</h4><span class="tag-yes">Projection confirmed</span></div>',
                            unsafe_allow_html=True)
                    else:
                        st.markdown(
                            f'<div class="result-card negative"><h4>{tr.region_name}</h4><span class="tag-no">No projection</span></div>',
                            unsafe_allow_html=True)

                st.markdown("<br>**3D Inspector**", unsafe_allow_html=True)
                if not inspect_result.coords:
                    st.caption(
                        "3D data not available for this cell. Coordinate data is only kept for "
                        "the first 60 cells to protect memory. Re-run the analysis with a smaller "
                        "selection to view the 3D plot for this cell."
                    )
                else:
                    with st.spinner(f"Building 3D plot for {inspect_name}..."):
                        fig_inspect = build_3d_plot(
                            inspect_result, atlas_matrix, inspect_name,
                            show_soma_region=show_soma_region,
                            show_other_regions=show_other_regions,
                            show_only_target_regions=show_only_target_regions
                        )
                    st.plotly_chart(fig_inspect, use_container_width=True)

        with tab_3d_multi:
            st.caption("Joint rendering of all processed cells. Each cell gets its own colour for easy distinction.")

            # Only cells whose coordinate data was retained can be rendered.
            # Coords are cleared for cells beyond position 60 to protect memory.
            combined_results = [(n, r) for n, r in results if r.coords]
            dropped = len(results) - len(combined_results)

            if dropped > 0:
                st.warning(
                    f"{dropped} cell{'s' if dropped > 1 else ''} beyond the 60-cell 3D limit "
                    f"cannot be rendered. Their numerical results are still available in the "
                    f"Population Statistics tab."
                )

            if not combined_results:
                st.caption("No cells with 3D data available.")
            elif st.button("Generate Combined Scene", type="primary"):
                with st.spinner(f"Rendering {len(combined_results)} cells together..."):
                    fig_multi = build_3d_plot_multi(
                        combined_results, atlas_matrix, selected_region_ids,
                        show_target_regions=True,
                        show_only_target_regions=show_only_target_regions
                    )
                st.plotly_chart(fig_multi, use_container_width=True)