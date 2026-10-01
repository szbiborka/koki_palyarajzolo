# APP.PY - Streamlit Application Entry Point
import concurrent.futures

import pandas as pd
import streamlit as st

from config import (
    BASE_DATA_DIR, DEFAULT_TARGET_REGIONS, DEFAULT_FILTER, BRAINSTEM_MOTOR_ID,
    VIZ_THEMES, DEFAULT_VIZ_THEME, LATERALITY_MODES, DEFAULT_LATERALITY,
    DATABASE_METADATA_PATH, CURATION_PATH, CURATION_LABELS, DEFAULT_EXCLUDED_LABELS,
)
from core.cell_info import (
    load_database_metadata, load_curation, save_curation, set_curation_label,
    parse_cell_list, filter_cells, attach_cell_info, cell_key, soma_layer, layer_mismatch,
)
from core.loader import (
    load_atlas, load_dictionary, load_swc, region_name_map,
    get_all_swc_files, build_region_search_options,
    load_soma_index, build_soma_index, soma_index_exists,
    filter_swc_by_soma_region, build_region_descendants
)
from core.analysis import (
    run_analysis, apply_filter, results_to_dataframe, FilterCriteria, RegionResult,
    build_cortical_summary, build_laterality_summary, LATERALITY_CLASS_LABELS,
    category_slugs, build_soma_distribution_summary
)
from core.visualization import build_3d_plot, build_3d_plot_multi
from ui_assets import setup_css, NEURON_MARK, section_header

st.set_page_config(
    page_title="Palyakoveto — Neuron Projection Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)
setup_css()

RULE_OPERATORS = {
    "Required (AND)": "AND",
    "Excluded (NOT)": "NOT",
    "Optional (OR)": "OR",
    "Observe only": "NONE",
}
# Régiónkénti félteke: None = a futás alapértelmezése (a Hemisphere választó)
REGION_SIDES = {"Run default": None, "Both sides": 'both', "Ipsilateral": 'ipsi', "Contralateral": 'contra'}
CAMERA_LABELS = {"Free (rotate)": 'free', "Top (dorsal)": 'top', "Side (lateral)": 'side', "Front (anterior)": 'front'}
BATCH_METHODS = ["Analyze ALL matched cells", "Select specific cells manually", "Paste a list of cells"]


@st.cache_data(show_spinner="Loading database metadata...")
def _database_metadata():
    return load_database_metadata(DATABASE_METADATA_PATH)


@st.cache_data(show_spinner=False)
def _curation():
    return load_curation(CURATION_PATH)


def _cell_info_block(result) -> None:
    """Az adatbázis és a kézi ellenőrzés információi egy sejtről."""
    c1, c2, c3 = st.columns(3)
    c1.metric("Projection class (database)", result.projection_subclass or "—")
    c2.metric("Cre line", result.cre_line or "—")
    c3.metric("Manual verdict", CURATION_LABELS.get(result.curation_label, "—") if result.curation_label else "—")
    notes = []
    if result.db_soma_region:
        notes.append(f"Database soma region: **{result.db_soma_region}**")
    if layer_mismatch(result.soma_region_name, result.projection_class):
        notes.append(f"⚠️ Soma layer ({soma_layer(result.soma_region_name)}) does not fit the "
                     f"{result.projection_class} class")
    if result.group_region != result.soma_region_name:
        notes.append(f"Counted in summaries as: **{result.group_region}**")
    if notes:
        st.caption(" · ".join(notes))


def _curation_editor(cell_name: str, result) -> None:
    """A sejt kézi ítéletének megadása; a kurációs fájlba íródik, és a következő futásoktól érvényes."""
    with st.expander("Manual verdict for this cell"):
        options = ["(none)"] + list(CURATION_LABELS)
        current = result.curation_label if result.curation_label in CURATION_LABELS else "(none)"
        key = cell_key(cell_name) or cell_name
        label = st.selectbox("Verdict", options, index=options.index(current), key=f"verdict_{key}",
                             format_func=lambda k: CURATION_LABELS.get(k, k))
        note = st.text_input("Note", key=f"verdict_note_{key}")
        if st.button("Save verdict", key=f"verdict_save_{key}"):
            new_label = None if label == "(none)" else label
            save_curation(set_curation_label(_curation(), key, new_label, source='app', note=note), CURATION_PATH)
            _curation.clear()
            result.curation_label = new_label
            st.success(f"Saved to `{CURATION_PATH}`. Exclusions by verdict apply from the next run.")


def _region_card(tr: RegionResult, crit: FilterCriteria) -> None:
    """Egy célterület eredménykártyája: a szűrési szabály és a vetítés állapota + a tényleges számok."""
    meets = crit.meets_thresholds(tr)
    if crit.operator == 'NOT' and meets:
        css, tag_css, tag = 'filtered-out', 'tag-filtered', 'Violated NOT rule'
    elif crit.operator == 'AND' and not meets:
        css, tag_css, tag = 'filtered-out', 'tag-filtered', 'Did not meet thresholds'
    elif tr.projects_here:
        css, tag_css, tag = 'positive', 'tag-yes', 'Projection confirmed'
    else:
        css, tag_css, tag = 'negative', 'tag-no', 'No projection'
    st.markdown(
        f'<div class="result-card {css}"><h4>{tr.region_name}</h4>'
        f'<span class="{tag_css}">{tag}</span>'
        f'<div class="meta" style="margin-top:6px;">'
        f'Endpoints: <b>{tr.endpoint_count}</b> | Branch points: <b>{tr.branch_point_count}</b> | '
        f'Axon: <b>{tr.axon_length_um:,.1f} µm</b> | Endpoint share: <b>{tr.endpoint_fraction * 100:.1f}%</b>'
        f'<br>Criterion: {crit.describe()}</div></div>',
        unsafe_allow_html=True)


def _csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8')


# GLOBAL DATA LOADING
try:
    atlas_matrix = load_atlas()
    dictionary = load_dictionary()
except FileNotFoundError as e:
    st.error(f"Data file not found. Please check config.py.\n\n{e}")
    st.stop()

region_names = region_name_map(dictionary)
names_by_lower = {name.lower(): name for name in region_names.values()}
region_options = build_region_search_options(region_names, dictionary)
all_swc = get_all_swc_files(BASE_DATA_DIR)
metadata = _database_metadata()
curation = _curation()

# SIDEBAR
with st.sidebar:
    st.markdown(f'<div class="sidebar-title">{NEURON_MARK} Palyakoveto</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">Neuron Projection Analyzer</div>', unsafe_allow_html=True)

    st.markdown("**Target Brain Regions** *(optional)*",
                help="Regions where you are looking for projections. Leave empty for hemisphere-only run.")
    selected_region_names = st.multiselect(
        label="Search and select regions",
        options=list(region_options.keys()),
        default=[name for name, rid in region_options.items() if rid in DEFAULT_TARGET_REGIONS.values()],
        key="region_selector", label_visibility="collapsed"
    )
    selected_region_ids = [region_options[name] for name in selected_region_names]
    if not selected_region_ids:
        st.caption("No target region — the run will still produce the **Hemisphere** tab.")

    st.divider()

    st.markdown("**Hemisphere**", help="Ipsilateral vs Contralateral evaluation mode.")
    lat_labels = list(LATERALITY_MODES.keys())
    default_lat_label = next(k for k, (code, _) in LATERALITY_MODES.items() if code == DEFAULT_LATERALITY)
    lat_choice = st.radio(
        label="Hemisphere", options=lat_labels, index=lat_labels.index(default_lat_label),
        horizontal=True, key="laterality_mode", label_visibility="collapsed",
    )
    laterality, lat_help = LATERALITY_MODES[lat_choice]
    st.caption(f"➜ {lat_help} Default for every region; can be overridden per region below.")
    exclude_contra = st.checkbox(
        "Exclude contralaterally projecting cells", key="exclude_contra",
        help="Whole-cell rule: a cell fails the filter if more than the given share of its endpoints "
             "lies on the opposite hemisphere. Cells without a decidable side are kept.")
    max_contra_pct = None
    if exclude_contra:
        max_contra_pct = st.number_input("Max. contralateral endpoints (%)", min_value=0.0, max_value=100.0,
                                         value=0.0, step=0.5, key="max_contra_pct")

    st.divider()

    st.markdown("**Projection Criteria**", help="What counts as a projection in each region.")
    criteria_per_region: dict[int, FilterCriteria] = {}
    if not selected_region_ids:
        st.caption("Select target regions above to set filter criteria.")

    for region_name_full in selected_region_names:
        region_id = region_options[region_name_full]
        short_name = region_name_full.split('(')[-1].replace(')', '').strip()

        with st.expander(f"Filters for {short_name}", expanded=False):
            st.markdown(
                "<div style='font-size:0.78rem;font-weight:700;letter-spacing:0.04em;text-transform:uppercase;"
                "color:var(--taupe-deep);margin-bottom:0.3rem;'>Condition Rule</div>",
                unsafe_allow_html=True)
            rule_label = st.radio(
                label="Condition rule", options=list(RULE_OPERATORS.keys()), horizontal=True,
                key=f"filter_rule_{region_id}", label_visibility="collapsed",
            )
            op = RULE_OPERATORS[rule_label]
            if op == 'NONE':
                st.caption("👁 Observe only — reported, but does not filter any cells.")
            side_label = st.radio("Hemisphere for this region", options=list(REGION_SIDES), horizontal=True,
                                  key=f"filter_side_{region_id}")

            min_ep = st.number_input("Min. endpoints", min_value=0, value=DEFAULT_FILTER['min_endpoints'],
                                     step=1, key=f"filter_ep_{region_id}")
            min_br = st.number_input("Min. branch points", min_value=0, value=DEFAULT_FILTER['min_branch_points'],
                                     step=1, key=f"filter_br_{region_id}")
            min_len = st.number_input("Min. axon length (µm)", min_value=0.0,
                                      value=float(DEFAULT_FILTER['min_axon_length_um']), step=10.0,
                                      key=f"filter_len_{region_id}")
            min_ep_pct = st.number_input("Min. endpoint share (%)", min_value=0.0, max_value=100.0,
                                         value=float(DEFAULT_FILTER['min_endpoint_fraction'] * 100), step=0.5,
                                         key=f"filter_eppct_{region_id}")

            if min_ep_pct > 0 and op == 'NOT' and int(min_br) > 0:
                st.info(f"Tip: this excludes cells that arborise here **and** exceed {min_ep_pct:g}%.", icon="💡")

            criteria_per_region[region_id] = FilterCriteria(
                min_endpoints=int(min_ep), min_branch_points=int(min_br),
                min_axon_length_um=float(min_len), min_endpoint_fraction=float(min_ep_pct) / 100.0,
                operator=op, side=REGION_SIDES[side_label],
            ).effective(laterality)
            st.caption(f"➜ Counts as projecting to {short_name} when: "
                       f"**{criteria_per_region[region_id].describe()}**")

    st.divider()

    st.markdown("**Cell Files (SWC)**", help="Select the neurons to analyze.")
    selected_cells: dict[str, str] = {}  # {megjelenítési név: teljes útvonal}
    correct_layers = False
    if not all_swc:
        st.warning(f"No SWC files found in:\n`{BASE_DATA_DIR}`")
    else:
        soma_index = load_soma_index() if soma_index_exists() else None
        if soma_index is None:
            st.warning("Soma region index not built yet.")
            build_clicked = st.button("Build soma index", key="btn_build_index")
        else:
            build_clicked = st.button("Rebuild Index", key="btn_rebuild_index", width="stretch")
        if build_clicked:
            get_all_swc_files.clear()  # az újonnan bemásolt fájlok is bekerüljenek
            progress_bar = st.progress(0, text="Building soma index...")

            def update_progress(current, total, filename):
                progress_bar.progress(current / total if total > 0 else 0, text=f"Indexing: {filename}")

            build_soma_index(BASE_DATA_DIR, atlas_matrix, region_names, update_progress)
            load_soma_index.clear()
            st.rerun()

        filtered_swc = all_swc
        if soma_index is not None:
            soma_search = st.text_input("Filter by soma region", placeholder="e.g. motor, thalamus...",
                                        key="soma_search")
            filtered_swc = filter_swc_by_soma_region(all_swc, soma_index, soma_search)

        selected_classes, selected_lines = [], []
        if metadata is not None:
            selected_classes = st.multiselect(
                "Projection class (database)", options=sorted(metadata['projection_class'].dropna().unique()),
                key="class_filter",
                help="The database's own cell type, assigned from the projection pattern: IT = intratelencephalic, "
                     "PT = pyramidal tract (L5, subcerebral), CT = corticothalamic (L6). Independent of the "
                     "soma layer — use it to catch L5 PT cells whose soma was registered into L6.")
            selected_lines = st.multiselect("Cre line (database)",
                                            options=sorted(metadata['cre_line'].dropna().unique()), key="line_filter")
        else:
            st.caption(f"No database metadata at `{DATABASE_METADATA_PATH}` — cell-type filters are off.")
        excluded_labels = []
        if not curation.empty:
            excluded_labels = st.multiselect(
                "Exclude manually labelled cells", options=list(CURATION_LABELS),
                default=DEFAULT_EXCLUDED_LABELS, format_func=CURATION_LABELS.get, key="excluded_labels",
                help=f"Verdicts from `{CURATION_PATH}` ({len(curation)} cells).")
        filtered_swc = filter_cells(filtered_swc, metadata, selected_classes, selected_lines,
                                    curation, excluded_labels)
        correct_layers = st.toggle(
            "Correct soma layer by projection class", value=False, key="correct_layers",
            disabled=metadata is None and curation.empty,
            help="Summaries count a PT cell whose soma sits in L4/L6 under the L5 region of the same area "
                 "(and a CT cell in L5 under L6a). A manual 'Actually L5' verdict overrides the class. "
                 "The atlas region is still exported as 'soma_region'.")

        analysis_mode = st.radio("Analysis mode", options=["Single cell", "Batch (multiple cells)"],
                                 horizontal=True, key="analysis_mode")

        if analysis_mode == "Single cell":
            selected_name = st.selectbox("Select cell", options=list(filtered_swc.keys()), key="single_cell_selector")
            if selected_name is not None:
                selected_cells = {selected_name: filtered_swc[selected_name]}
        else:
            st.markdown(f"**{len(filtered_swc)} cells available for batch analysis.**")
            batch_method = st.radio("Selection method", options=BATCH_METHODS, key="batch_method",
                                    label_visibility="collapsed")
            if batch_method == BATCH_METHODS[0]:
                selected_cells = dict(filtered_swc)
                st.info(f"Ready to analyze all **{len(selected_cells)}** cells. Click 'Run Analysis' below.")
            elif batch_method == BATCH_METHODS[1]:
                selected_names = st.multiselect("Select specific cells", options=list(filtered_swc.keys()),
                                                default=[], key="batch_selector")
                selected_cells = {name: filtered_swc[name] for name in selected_names}
                if not selected_cells:
                    st.warning("Please select at least one cell from the dropdown.")
            else:
                pasted = st.text_area("Cells (any separator)", key="pasted_cells", height=120,
                                      placeholder="221044\\016.swc, 221044_019, 233284/066 ...",
                                      help="Exactly these cells are analyzed; the soma, class and verdict "
                                           "filters above do not apply to a pasted list.")
                wanted = parse_cell_list(pasted)
                selected_cells = {name: all_swc[name] for name in wanted if name in all_swc}
                missing = [name for name in wanted if name not in all_swc]
                if wanted:
                    st.caption(f"{len(selected_cells)} of {len(wanted)} cells found.")
                if missing:
                    st.warning(f"Not found in the data folder: {', '.join(missing[:20])}"
                               f"{' …' if len(missing) > 20 else ''}")

    st.divider()

    st.markdown("**Visualization Settings**", help="Toggles affect only the 3D Plotly scene.")
    theme_labels = {v['label']: k for k, v in VIZ_THEMES.items()}
    theme_choice = st.selectbox("Scene theme", options=list(theme_labels.keys()),
                                index=list(theme_labels.values()).index(DEFAULT_VIZ_THEME), key="viz_theme")
    viz_theme = theme_labels[theme_choice]

    show_brain_outline = st.toggle("Show brain outline", value=True, key="toggle_brain_outline")
    show_soma_region = st.toggle("Show soma region", value=True, key="toggle_soma")
    show_other_regions = st.toggle("Show other projection regions", value=True, key="toggle_other")
    show_only_target_regions = st.toggle("Axon-in-region view", value=False, key="toggle_exclusive")
    show_projection_points = st.toggle("Show projection points", value=True, key="toggle_points",
                                       help="The diamond markers on endpoints and branch points.")
    camera_view = CAMERA_LABELS[st.selectbox("Camera view", options=list(CAMERA_LABELS), key="camera_view")]

# MAIN CONTENT
if not selected_cells:
    st.markdown(f"""
    <div class="page-header" style="text-align: center; margin-top: 10vh;">
        <div style="display: flex; justify-content: center; margin-bottom: 20px;">{NEURON_MARK}</div>
        <h1>Palyakoveto</h1>
        <p>Neuron Projection Analyzer &mdash; HUN-REN KOKI</p>
    </div>
    """, unsafe_allow_html=True)
    st.info("**Welcome.** Select one or more cell files from the sidebar to begin.")
    st.stop()

any_filter_active = any(c.is_active() for c in criteria_per_region.values()) or max_contra_pct is not None
n_selected = len(selected_cells)
_, col_btn, _ = st.columns([1, 2, 1])
with col_btn:
    run_button = st.button(
        f"Run Analysis for {n_selected} cell{'s' if n_selected > 1 else ''}"
        f"{' (Filters active)' if any_filter_active else ''}",
        type="primary", width="stretch")

if run_button:
    region_descendants = build_region_descendants(dictionary, selected_region_ids)
    # A futás beállításait elmentjük: a megjelenítés MINDIG ezekből dolgozik, nem
    # az oldalsáv azóta esetleg módosított értékeiből.
    st.session_state['run'] = {
        'region_ids': list(selected_region_ids),
        'criteria': criteria_per_region,
        'descendants': region_descendants,
        'laterality': laterality,
        'max_contra_pct': max_contra_pct,
        'correct_layers': correct_layers,
    }

    def process_single_cell(cell_name: str, filepath: str):
        try:
            result = run_analysis(load_swc(filepath), atlas_matrix, region_names, selected_region_ids,
                                  region_descendants, criteria_per_region, laterality)
            attach_cell_info(result, cell_name, metadata, curation, names_by_lower, correct_layers)
            return cell_name, apply_filter(result, criteria_per_region, max_contra_pct), None
        except Exception as e:
            return cell_name, None, str(e)

    results, errors = [], []
    progress = st.progress(0, text="Analyzing cells...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_single_cell, name, path) for name, path in selected_cells.items()]
        for done_count, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            cell_name, result, error = future.result()
            if error is None:
                results.append((cell_name, result))
            else:
                errors.append((cell_name, error))
            progress.progress(done_count / n_selected, text=f"Analyzed {done_count}/{n_selected}: {cell_name}")
    progress.empty()

    st.session_state['results'] = sorted(results, key=lambda r: r[0])
    st.session_state['errors'] = sorted(errors, key=lambda r: r[0])

# DISPLAY RESULTS
errors = st.session_state.get('errors', [])
if errors:
    with st.expander(f"{len(errors)} file(s) could not be loaded"):
        for name, err in errors:
            st.error(f"**{name}**: {err}")

results = st.session_state.get('results', [])
if results:
    run = st.session_state['run']
    run_region_ids = run['region_ids']
    criteria_used = run['criteria']
    descendants_used = run['descendants']
    filter_was_active = (any(c.is_active() for c in criteria_used.values())
                         or run['max_contra_pct'] is not None)
    plot_options = dict(show_soma_region=show_soma_region, show_other_regions=show_other_regions,
                        show_only_target_regions=show_only_target_regions, region_descendants=descendants_used,
                        theme=viz_theme, show_brain_outline=show_brain_outline,
                        show_projection_points=show_projection_points, view=camera_view)

    if run_region_ids != selected_region_ids:
        st.warning("The target regions changed since the last run — the results below still show the "
                   "previous selection. Click **Run Analysis** to update them.")

    st.divider()

    # SINGLE CELL VIEW
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
            st.markdown(f"<h3>{cell_name}{filter_status}</h3>", unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Soma location", result.soma_region_name)
            n_confirmed = sum(1 for tr in result.target_results if tr.projects_here)
            m2.metric("Confirmed projections", f"{n_confirmed} / {len(result.target_results)} targets")
            m3.metric("Total axon length", f"{result.total_axon_length_um:,.0f} µm")
            _cell_info_block(result)
            _curation_editor(cell_name, result)

            st.markdown("<br>", unsafe_allow_html=True)
            section_header("Hemisphere")
            if result.has_hemisphere:
                h1, h2, h3 = st.columns(3)
                h1.metric("Laterality", LATERALITY_CLASS_LABELS[result.laterality_class])
                h2.metric("Endpoints ipsi / contra", f"{result.endpoints_ipsi_total} / {result.endpoints_contra_total}")
                h3.metric("Axon contra",
                          f"{result.axon_length_contra_um:,.0f} µm ({result.contra_axon_fraction * 100:.1f}%)")
            else:
                st.info("Laterality cannot be determined for this cell (no soma or exactly on midline).")

            st.markdown("<br>", unsafe_allow_html=True)
            if result.target_results:
                section_header("Target Region Results")
            else:
                st.caption("No target region selected — only whole-cell measures are shown.")
            for tr in result.target_results:
                _region_card(tr, criteria_used.get(tr.region_id, FilterCriteria()))

            if result.other_projection_regions:
                st.markdown("<br>", unsafe_allow_html=True)
                section_header("Other Detected Projections")
                other_df = pd.DataFrame([{
                    "Region": r.region_name, "Endpoints": r.endpoint_count,
                    "Branch points": r.branch_point_count, "Length (µm)": round(r.axon_length_um, 1),
                } for r in result.other_projection_regions])
                st.dataframe(other_df, hide_index=True)

        with tab_3d:
            st.info("**Tip:** Use the left mouse button to rotate, the right button to pan, "
                    "and the scroll wheel to zoom.")
            with st.spinner("Building interactive 3D plot..."):
                fig = build_3d_plot(result, atlas_matrix, cell_name, **plot_options)
            st.plotly_chart(fig)

    # BATCH VIEW
    else:
        tab_stats, tab_hemi, tab_summary, tab_inspector, tab_3d_multi = st.tabs(
            ["Population Statistics", "Hemisphere", "Cortical Summary", "Single Cell Inspector", "Combined 3D View"])

        with tab_stats:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total cells analyzed", len(results))
            if filter_was_active:
                c2.metric("Passed filter", sum(1 for _, r in results if r.passes_filter is True))
                c3.metric("Filtered out", sum(1 for _, r in results if r.passes_filter is False))
            else:
                c2.metric("Projecting cells",
                          sum(1 for _, r in results if any(tr.projects_here for tr in r.target_results)))
            c4.metric("Load errors", len(errors))

            st.markdown("<br>", unsafe_allow_html=True)
            section_header("Soma Region Distribution")
            soma_df = build_soma_distribution_summary(results, filter_was_active)
            st.dataframe(soma_df, hide_index=True, column_config={
                "Valid Projections %": st.column_config.NumberColumn("Valid Projections %", format="%.1f%%"),
                "Projecting Cell IDs": st.column_config.TextColumn("Projecting Cell IDs", width="large"),
            })
            if not soma_df.empty:
                st.download_button("Download Soma Region Summary (CSV)", data=_csv(soma_df),
                                   file_name="soma_region_summary.csv", mime="text/csv", key="download_soma_summary")

            if any(r.projection_class for _, r in results):
                with st.expander("Soma layer vs projection class (database)"):
                    st.caption("Rows: the soma's layer in the atlas; columns: the database's projection class. "
                               "PT cells outside L5 and CT cells outside L6 are the layer-assignment suspects.")
                    st.dataframe(pd.crosstab(
                        pd.Series([soma_layer(r.soma_region_name) or "—" for _, r in results], name="Soma layer"),
                        pd.Series([r.projection_class or "—" for _, r in results], name="Class"),
                        margins=True))

            st.markdown("<br>", unsafe_allow_html=True)
            section_header("Detailed Batch Data")
            summary_df = results_to_dataframe(results, criteria_used)
            if filter_was_active:
                summary_df = summary_df.sort_values('passes_filter', ascending=False, kind='stable')
            st.dataframe(summary_df, hide_index=True)
            st.download_button("Download Dataset (CSV)", data=_csv(summary_df),
                               file_name="batch_results.csv", mime="text/csv")

        with tab_hemi:
            section_header("Hemisphere — does the axon cross the midline?")
            lat = build_laterality_summary(results)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Cells analyzed", lat['n_total'])
            m2.metric("Ipsilateral only", lat['counts']['ipsi_only'])
            m3.metric("No contralateral endpoints", lat['counts']['ipsi_only'] + lat['counts']['crosses_only'])
            m4.metric("Projects contralaterally", lat['counts']['contra'])

            if lat['n_decided'] < lat['n_total']:
                st.warning(f"{lat['n_total'] - lat['n_decided']} cell(s) could not be classified "
                           f"(no soma, or soma on midline).")

            st.markdown("<br>", unsafe_allow_html=True)
            section_header("Overall")
            st.dataframe(lat['overall'], hide_index=True, column_config={
                "% of decided": st.column_config.NumberColumn("% of decided", format="%.1f%%"),
                "Cell IDs": st.column_config.TextColumn("Cell IDs", width="large"),
            })

            st.markdown("<br>", unsafe_allow_html=True)
            section_header("By soma region")
            if not lat['by_soma'].empty:
                st.dataframe(lat['by_soma'], hide_index=True, column_config={
                    "No contralateral %": st.column_config.NumberColumn("No contralateral %", format="%.1f%%"),
                    "Ipsilateral-only cell IDs": st.column_config.TextColumn("Ipsilateral-only cell IDs",
                                                                             width="large"),
                })
                st.download_button("Download Hemisphere Summary (CSV)", data=_csv(lat['by_soma']),
                                   file_name="hemisphere_by_soma_region.csv", mime="text/csv",
                                   key="download_hemi_soma")

            st.markdown("<br>", unsafe_allow_html=True)
            section_header("Cell by cell")
            st.dataframe(lat['per_cell'], hide_index=True)
            st.download_button("Download Per-Cell Laterality (CSV)", data=_csv(lat['per_cell']),
                               file_name="hemisphere_per_cell.csv", mime="text/csv", key="download_hemi_cells")

        with tab_summary:
            section_header("Cortical Projection Summary")

            def _summary_label(rid: int) -> str:
                if rid == BRAINSTEM_MOTOR_ID:
                    return "Brain stem (descending)"
                return region_names.get(rid, f"ID {rid}")

            label_to_id = {_summary_label(rid): rid for rid in run_region_ids}
            base_options = ["(All L5 cells — no PT base)"] + list(label_to_id.keys())
            default_idx = next((i + 1 for i, lab in enumerate(label_to_id) if "brain stem" in lab.lower()), 0)

            base_choice = st.selectbox("Population base = 100%", options=base_options, index=default_idx)
            base_id = label_to_id.get(base_choice)  # None az "All L5" opciónál
            numerator_ids = [rid for rid in run_region_ids if rid != base_id]
            only_passing = st.checkbox(
                "Only cells that pass the filter", value=False, disabled=not filter_was_active,
                help="By default the tables use every analysed cell and ignore the AND/NOT/OR rules. "
                     "Tick this to build them from the filtered population only (e.g. after excluding "
                     "contralateral projectors or thalamus-heavy L6 cells).")
            summary_results = [(n, r) for n, r in results if r.passes_filter] if only_passing else results

            if not numerator_ids:
                st.info("Add at least one more target region besides the base to build the summary.")
            else:
                summary = build_cortical_summary(summary_results, base_id, numerator_ids, _summary_label,
                                                 criteria_used, laterality=run['laterality'])
                tag = summary['slug'] + ('_filtered' if only_passing else '')
                if run['correct_layers']:
                    st.caption("Soma regions are layer-corrected by projection class.")
                st.info(f"**Projection criteria used:** {summary['criteria_note']}\n\n"
                        f"Recorded in every downloaded file name (`{tag}`).")
                if summary['skipped_no_soma_region']:
                    st.warning(f"{summary['skipped_no_soma_region']} cell(s) left out: no soma, or the soma lies "
                               f"outside every annotated region.")

                st.markdown("**1. Brain stem = 100% (PT cells)** — *bs_benne*")
                st.dataframe(summary['benne'], hide_index=True)
                st.download_button("⬇ Download bs_benne.csv", _csv(summary['benne']),
                                   file_name=f"bs_benne_{tag}.csv", mime="text/csv", key="dl_benne")

                st.markdown("**2. All L5 = 100% (no brain-stem requirement)** — *bs_nelkul*")
                st.dataframe(summary['nelkul'], hide_index=True)
                st.download_button("⬇ Download bs_nelkul.csv", _csv(summary['nelkul']),
                                   file_name=f"bs_nelkul_{tag}.csv", mime="text/csv", key="dl_nelkul")

                st.markdown("**3. Average axon length in each target (µm), among PT cells**")
                st.dataframe(summary['axon'], hide_index=True)
                st.download_button("⬇ Download axon_length_summary.csv", _csv(summary['axon']),
                                   file_name=f"axon_length_summary_{tag}.csv", mime="text/csv", key="dl_axon")

                st.markdown("**4. Category tables with projecting cell IDs**")
                cat_slugs = category_slugs(list(summary['categories'].keys()))
                for lab, df in summary['categories'].items():
                    safe = cat_slugs[lab]
                    n_cells = int(df[f"{lab} Projects"].sum()) if not df.empty else 0
                    with st.expander(f"{lab} ({n_cells} cells)"):
                        st.dataframe(df, hide_index=True)
                        st.download_button(f"⬇ Download {safe}.csv", _csv(df),
                                           file_name=f"bs_{safe}_{tag}.csv", mime="text/csv", key=f"dl_cat_{safe}")

        with tab_inspector:
            st.markdown("Select a single cell from the processed population to view detailed metrics and its 3D scene.")
            results_by_name = dict(results)
            inspect_name = st.selectbox("Select cell to inspect", options=list(results_by_name.keys()),
                                        label_visibility="collapsed")
            if inspect_name:
                inspect_result = results_by_name[inspect_name]
                _cell_info_block(inspect_result)
                _curation_editor(inspect_name, inspect_result)
                for tr in inspect_result.target_results:
                    _region_card(tr, criteria_used.get(tr.region_id, FilterCriteria()))

                with st.spinner(f"Building 3D plot for {inspect_name}..."):
                    st.plotly_chart(build_3d_plot(inspect_result, atlas_matrix, inspect_name, **plot_options))

        with tab_3d_multi:
            st.caption("Joint rendering of all processed cells. Each cell gets its own colour for easy distinction.")
            show_only_valid = st.toggle("Show only passing cells", value=True)

            def _is_shown(r) -> bool:
                if not show_only_valid:
                    return True
                if filter_was_active:
                    return bool(r.passes_filter)
                return any(tr.projects_here for tr in r.target_results)

            combined_results = [(n, r) for n, r in results if _is_shown(r)]
            if not combined_results:
                st.warning("No cells match the current criteria for 3D rendering.")
            elif st.button(f"Generate Combined Scene ({len(combined_results)} cells)", type="primary"):
                with st.spinner(f"Rendering {len(combined_results)} cells together..."):
                    st.plotly_chart(build_3d_plot_multi(
                        combined_results, atlas_matrix, run_region_ids, show_target_regions=True,
                        show_only_target_regions=show_only_target_regions, region_descendants=descendants_used,
                        theme=viz_theme, show_brain_outline=show_brain_outline, view=camera_view))
