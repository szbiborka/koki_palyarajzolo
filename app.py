# APP.PY - Streamlit Application Entry Point
import os
import streamlit as st
import pandas as pd
import concurrent.futures

from config import (
    BASE_DATA_DIR, DEFAULT_TARGET_REGIONS, DEFAULT_FILTER,
    BRAINSTEM_MOTOR_ID, BRAINSTEM_MOTOR_NAME,
    VIZ_THEMES, DEFAULT_VIZ_THEME,
    LATERALITY_MODES, DEFAULT_LATERALITY,
)
from core.loader import (
    load_atlas, load_dictionary, load_swc,
    get_all_swc_files, build_region_search_options,
    load_soma_index, build_soma_index, soma_index_exists,
    filter_swc_by_soma_region, build_region_descendants
)
from core.analysis import (
    run_analysis, apply_filter, results_to_dataframe, FilterCriteria,
    build_cortical_summary, build_laterality_summary, LATERALITY_CLASS_LABELS,
    category_slugs, build_soma_distribution_summary
)
from core.visualization import (
    build_3d_plot, build_3d_plot_multi, render_plot_streamlit
)

# ÚJ IMPORT A DIZÁJN FÁJLBÓL
from ui_assets import setup_css, NEURON_MARK, SYNAPSE_MARK, section_header

st.set_page_config(
    page_title="Palyakoveto — Neuron Projection Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# DIZÁJN INJEKTÁLÁSA
setup_css()

RULE_OPERATORS = {
    "Required (AND)": "AND",
    "Excluded (NOT)": "NOT",
    "Optional (OR)": "OR",
    "Observe only": "NONE",
}

# GLOBAL DATA LOADING
try:
    atlas_matrix, atlas_header = load_atlas()
    dictionary = load_dictionary()
    region_options = build_region_search_options(dictionary)
except FileNotFoundError as e:
    st.error(f"Data file not found. Please check config.py.\n\n{e}")
    st.stop()

all_swc = get_all_swc_files(BASE_DATA_DIR)

# SIDEBAR
with st.sidebar:
    st.markdown(f'<div class="sidebar-title">{NEURON_MARK} Palyakoveto</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">Neuron Projection Analyzer</div>', unsafe_allow_html=True)

    st.markdown("**Target Brain Regions** *(optional)*",
                help="Regions where you are looking for projections. Leave empty for hemisphere-only run.")
    selected_region_names = st.multiselect(
        label="Search and select regions",
        options=list(region_options.keys()),
        default=[name for name in region_options.keys() if region_options[name] in DEFAULT_TARGET_REGIONS.values()],
        key="region_selector", label_visibility="collapsed"
    )
    selected_region_ids = [region_options[name] for name in selected_region_names]

    if not selected_region_ids:
        st.caption("No target region — the run will still produce the **Hemisphere** tab.")

    st.divider()

    st.markdown("**Hemisphere**", help="Ipsilateral vs Contralateral evaluation mode.")
    _lat_keys = list(LATERALITY_MODES.keys())
    _default_label = next(k for k, (code, _) in LATERALITY_MODES.items() if code == DEFAULT_LATERALITY)
    lat_choice = st.radio(
        label="Hemisphere", options=_lat_keys, index=_lat_keys.index(_default_label),
        horizontal=True, key="laterality_mode", label_visibility="collapsed",
    )
    laterality, _lat_help = LATERALITY_MODES[lat_choice]
    st.caption(f"➜ {_lat_help}")
    if laterality == 'both':
        st.caption("ℹ️ Both sides are counted together (legacy behavior).")

    st.divider()

    st.markdown("**Projection Criteria**", help="What counts as a projection in each region.")
    criteria_per_region: dict[int, FilterCriteria] = {}

    if not selected_region_ids:
        st.caption("Select target regions above to set filter criteria.")
    else:
        for region_name_full in selected_region_names:
            region_id = region_options[region_name_full]
            short_name = region_name_full.split('(')[-1].replace(')', '').strip()

            with st.expander(f"Filters for {short_name}", expanded=False):
                st.markdown(
                    "<div style='font-size:0.78rem;font-weight:700;letter-spacing:0.04em;text-transform:uppercase;color:var(--taupe-deep);margin-bottom:0.3rem;'>Condition Rule</div>",
                    unsafe_allow_html=True)
                rule_label = st.radio(
                    label="Condition rule", options=list(RULE_OPERATORS.keys()), horizontal=True,
                    key=f"filter_rule_{region_id}", label_visibility="collapsed",
                )
                op = RULE_OPERATORS[rule_label]
                if op == 'NONE': st.caption("👁 Observe only — reported, but does not filter any cells.")

                min_ep = st.number_input("Min. endpoints", min_value=0, value=DEFAULT_FILTER['min_endpoints'], step=1,
                                         key=f"filter_ep_{region_id}")
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
                    min_axon_length_um=float(min_len), min_endpoint_fraction=float(min_ep_pct) / 100.0, operator=op
                )
                st.caption(
                    f"➜ Counts as projecting to {short_name} when: **{criteria_per_region[region_id].describe()}**")

    st.divider()

    st.markdown("**Cell Files (SWC)**", help="Select the neurons to analyze.")
    if not all_swc:
        st.warning(f"No SWC files found in:\n`{BASE_DATA_DIR}`")
        selected_swc_paths = []
    else:
        if not soma_index_exists():
            st.warning("Soma region index not built yet.")
            if st.button("Build soma index", key="btn_build_index"):
                progress_bar = st.progress(0, text="Building soma index...")


                def update_progress(current, total, filename): progress_bar.progress(
                    current / total if total > 0 else 0, text=f"Indexing: {filename}")


                with st.spinner("Building soma index..."):
                    build_soma_index(BASE_DATA_DIR, atlas_matrix, dictionary, update_progress)
                progress_bar.empty()
                st.rerun()
            soma_index = None
        else:
            soma_index = load_soma_index()
            if st.button("Rebuild Index", key="btn_rebuild_index", use_container_width=True):
                with st.spinner("Rebuilding soma index..."):
                    build_soma_index(BASE_DATA_DIR, atlas_matrix, dictionary)
                st.rerun()

        soma_search = ""
        if soma_index is not None:
            soma_search = st.text_input("Filter by soma region", placeholder="e.g. motor, thalamus...",
                                        key="soma_search")
            filtered_swc = filter_swc_by_soma_region(all_swc, soma_index, soma_search) if soma_search else all_swc
        else:
            filtered_swc = all_swc

        analysis_mode = st.radio("Analysis mode", options=["Single cell", "Batch (multiple cells)"], horizontal=True,
                                 key="analysis_mode")

        if analysis_mode == "Single cell":
            selected_name = st.selectbox("Select cell", options=list(filtered_swc.keys()), key="single_cell_selector")
            selected_swc_paths = [filtered_swc[selected_name]] if filtered_swc else []
        else:
            st.markdown(f"**{len(filtered_swc)} cells available for batch analysis.**")
            batch_method = st.radio("Selection method",
                                    options=["Analyze ALL matched cells", "Select specific cells manually"],
                                    horizontal=True, label_visibility="collapsed")
            if batch_method == "Analyze ALL matched cells":
                selected_swc_paths = list(filtered_swc.values())
                st.info(f"Ready to analyze all **{len(selected_swc_paths)}** cells. Click 'Run Analysis' below.")
            else:
                selected_names = st.multiselect("Select specific cells", options=list(filtered_swc.keys()), default=[],
                                                key="batch_selector")
                selected_swc_paths = [filtered_swc[name] for name in selected_names]
                if not selected_swc_paths: st.warning("Please select at least one cell from the dropdown.")

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

# MAIN CONTENT
if not selected_swc_paths:
    st.markdown(f"""
    <div class="page-header" style="text-align: center; margin-top: 10vh;">
        <div style="display: flex; justify-content: center; margin-bottom: 20px;">{NEURON_MARK}</div>
        <h1>Palyakoveto</h1>
        <p>Neuron Projection Analyzer &mdash; HUN-REN KOKI</p>
    </div>
    """, unsafe_allow_html=True)
    st.info("**Welcome.** Select one or more cell files from the sidebar to begin.")
    st.stop()

any_filter_active = any(c.is_active() for c in criteria_per_region.values())
filter_note = " (Filters active)" if any_filter_active else ""

_, col_btn, _ = st.columns([1, 2, 1])
with col_btn:
    run_button = st.button(
        f"Run Analysis for {len(selected_swc_paths)} cell{'s' if len(selected_swc_paths) > 1 else ''}{filter_note}",
        type="primary", use_container_width=True)

if run_button:
    st.session_state['results'], st.session_state['errors'] = [], []
    st.session_state['criteria_per_region'] = criteria_per_region

    region_descendants = build_region_descendants(dictionary, selected_region_ids)
    st.session_state['region_descendants'] = region_descendants
    st.session_state['criteria_used'] = criteria_per_region
    st.session_state['laterality_used'] = laterality
    region_names = {BRAINSTEM_MOTOR_ID: BRAINSTEM_MOTOR_NAME}

    progress = st.progress(0, text="Analyzing cells...")


    def process_single_cell(filepath):
        cell_name = next((k for k, v in filtered_swc.items() if v == filepath), os.path.basename(filepath))
        try:
            result = run_analysis(load_swc(filepath), atlas_matrix, dictionary, selected_region_ids, region_descendants,
                                  region_names, criteria_per_region, laterality)
            return (cell_name, apply_filter(result, criteria_per_region), None)
        except Exception as e:
            return (cell_name, None, str(e))


    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(process_single_cell, path): path for path in selected_swc_paths}
        completed_count, total_count = 0, len(selected_swc_paths)
        for future in concurrent.futures.as_completed(futures):
            cell_name, result, error = future.result()
            if error is None:
                st.session_state['results'].append((cell_name, result))
            else:
                st.session_state['errors'].append((cell_name, error))
            completed_count += 1
            progress.progress(completed_count / total_count if total_count > 0 else 0,
                              text=f"Analyzed {completed_count}/{total_count}: {cell_name}")

    st.session_state['results'].sort(key=lambda r: r[0])
    st.session_state['errors'].sort(key=lambda r: r[0])
    progress.empty()

# DISPLAY RESULTS
if 'errors' in st.session_state and st.session_state['errors']:
    with st.expander(f"{len(st.session_state['errors'])} file(s) could not be loaded"):
        for name, err in st.session_state['errors']: st.error(f"**{name}**: {err}")

if 'results' in st.session_state and st.session_state['results']:
    results = st.session_state['results']
    saved_criteria = st.session_state.get('criteria_per_region', {})
    filter_was_active = any(c.is_active() for c in saved_criteria.values())
    criteria_used = st.session_state.get('criteria_used', saved_criteria)
    descendants_used = st.session_state.get('region_descendants', {})
    saved_laterality = st.session_state.get('laterality_used', 'both')

    st.divider()

    # SINGLE CELL VIEW
    if len(results) == 1:
        cell_name, result = results[0]
        tab_data, tab_3d = st.tabs(["Analytics & Data", "Interactive 3D Viewer"])

        with tab_data:
            filter_status = '<span class="tag-yes" style="margin-left:15px;">Passes filter</span>' if result.passes_filter is True else (
                '<span class="tag-filtered" style="margin-left:15px;">Filtered out</span>' if result.passes_filter is False else "")
            st.markdown(f"<h3>{cell_name}{filter_status}</h3>", unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Soma location", result.soma_region_name)
            m2.metric("Confirmed projections",
                      f"{sum(1 for tr in result.target_results if tr.projects_here)} / {len(result.target_results)} targets")
            m3.metric("Total axon length", f"{result.total_axon_length_um:,.0f} µm")

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

            if result.target_results:
                st.markdown("<br>", unsafe_allow_html=True)
                section_header("Target Region Results")
            else:
                st.markdown("<br>", unsafe_allow_html=True)
                st.caption("No target region selected — only whole-cell measures are shown.")

            for tr in result.target_results:
                cr = saved_criteria.get(tr.region_id, FilterCriteria())
                is_active_rule = cr.is_active() and filter_was_active
                meets_rule = cr.meets_thresholds(tr)

                if is_active_rule and cr.operator == 'NOT' and meets_rule:
                    st.markdown(
                        f'<div class="result-card filtered-out"><h4>{tr.region_name}</h4><span class="tag-filtered">Violated NOT rule</span><div class="meta" style="margin-top:6px;">Endpoints: <b>{tr.endpoint_count}</b> | Branch points: <b>{tr.branch_point_count}</b> | Axon: <b>{tr.axon_length_um:,.1f} µm</b></div></div>',
                        unsafe_allow_html=True)
                elif is_active_rule and cr.operator == 'AND' and not meets_rule:
                    st.markdown(
                        f'<div class="result-card filtered-out"><h4>{tr.region_name}</h4><span class="tag-filtered">Did not meet thresholds</span><div class="meta" style="margin-top:6px;">Endpoints: <b>{tr.endpoint_count}</b> | Branch points: <b>{tr.branch_point_count}</b> | Axon: <b>{tr.axon_length_um:,.1f} µm</b></div></div>',
                        unsafe_allow_html=True)
                elif tr.projects_here:
                    st.markdown(
                        f'<div class="result-card positive"><h4>{tr.region_name}</h4><span class="tag-yes">Projection Confirmed</span><div class="meta" style="margin-top:6px;">Endpoints: <b>{tr.endpoint_count}</b> | Branch points: <b>{tr.branch_point_count}</b> | Axon: <b>{tr.axon_length_um:,.1f} µm</b></div></div>',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<div class="result-card negative"><h4>{tr.region_name}</h4><span class="tag-no">No Projection</span><div class="meta" style="margin-top:6px;">Axon may pass through but has no endpoints or branch points here.</div></div>',
                        unsafe_allow_html=True)

            if result.other_projection_regions:
                st.markdown("<br>", unsafe_allow_html=True)
                section_header("Other Detected Projections")
                other_df = pd.DataFrame([{"Region": r.region_name, "Endpoints": r.endpoint_count,
                                          "Branch points": r.branch_point_count,
                                          "Length (µm)": round(r.axon_length_um, 1)} for r in
                                         result.other_projection_regions])
                st.dataframe(other_df, use_container_width=True, hide_index=True)

        with tab_3d:
            st.info(
                "**Tip:** Use the left mouse button to rotate, the right button to pan, and the scroll wheel to zoom.")
            with st.spinner("Building interactive 3D plot..."):
                fig = build_3d_plot(
                    result, atlas_matrix, cell_name, show_soma_region=show_soma_region,
                    show_other_regions=show_other_regions,
                    show_only_target_regions=show_only_target_regions, region_descendants=descendants_used,
                    theme=viz_theme, show_brain_outline=show_brain_outline
                )
            st.plotly_chart(fig, use_container_width=True)

    # BATCH VIEW
    else:
        tab_stats, tab_hemi, tab_summary, tab_inspector, tab_3d_multi = st.tabs(
            ["Population Statistics", "Hemisphere", "Cortical Summary", "Single Cell Inspector", "Combined 3D View"])

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

            # --- ITT TISZTULT KI A KÓD: Külső logika meghívása ---
            soma_df = build_soma_distribution_summary(results, filter_was_active)

            st.dataframe(soma_df, use_container_width=True, hide_index=True,
                         column_config={
                             "Valid Projections %": st.column_config.NumberColumn("Valid Projections %",
                                                                                  format="%.1f%%"),
                             "Projecting Cell IDs": st.column_config.TextColumn("Projecting Cell IDs", width="large"),
                         },
                         )

            if not soma_df.empty:
                soma_csv = soma_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Soma Region Summary (CSV)", data=soma_csv,
                                   file_name="soma_region_summary.csv", mime="text/csv", key="download_soma_summary")

            st.markdown("<br>", unsafe_allow_html=True)
            section_header("Detailed Batch Data")
            summary_df = results_to_dataframe(results, selected_region_ids, dictionary, criteria_used)
            if filter_was_active: summary_df = summary_df.sort_values('passes_filter', ascending=False)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            csv_data = summary_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Dataset (CSV)", data=csv_data, file_name="batch_results.csv", mime="text/csv")

        with tab_hemi:
            section_header("Hemisphere — does the axon cross the midline?")
            lat = build_laterality_summary(results)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Cells analyzed", lat['n_total'])
            m2.metric("Ipsilateral only", lat['counts']['ipsi_only'])
            m3.metric("No contralateral endpoints", lat['counts']['ipsi_only'] + lat['counts']['crosses_only'])
            m4.metric("Projects contralaterally", lat['counts']['contra'])

            if lat['n_decided'] < lat['n_total']:
                st.warning(
                    f"{lat['n_total'] - lat['n_decided']} cell(s) could not be classified (no soma, or soma on midline).")

            st.markdown("<br>", unsafe_allow_html=True)
            section_header("Overall")
            st.dataframe(lat['overall'], use_container_width=True, hide_index=True,
                         column_config={"% of decided": st.column_config.NumberColumn("% of decided", format="%.1f%%"),
                                        "Cell IDs": st.column_config.TextColumn("Cell IDs", width="large")}
                         )

            st.markdown("<br>", unsafe_allow_html=True)
            section_header("By soma region")
            if not lat['by_soma'].empty:
                st.dataframe(lat['by_soma'], use_container_width=True, hide_index=True,
                             column_config={"No contralateral %": st.column_config.NumberColumn("No contralateral %",
                                                                                                format="%.1f%%"),
                                            "Ipsilateral-only cell IDs": st.column_config.TextColumn(
                                                "Ipsilateral-only cell IDs", width="large")}
                             )
                st.download_button("Download Hemisphere Summary (CSV)",
                                   data=lat['by_soma'].to_csv(index=False).encode('utf-8'),
                                   file_name="hemisphere_by_soma_region.csv", mime="text/csv", key="download_hemi_soma")

            st.markdown("<br>", unsafe_allow_html=True)
            section_header("Cell by cell")
            st.dataframe(lat['per_cell'], use_container_width=True, hide_index=True)
            st.download_button("Download Per-Cell Laterality (CSV)",
                               data=lat['per_cell'].to_csv(index=False).encode('utf-8'),
                               file_name="hemisphere_per_cell.csv", mime="text/csv", key="download_hemi_cells")

        with tab_summary:
            section_header("Cortical Projection Summary")


            def _region_label(rid: int) -> str:
                if rid == BRAINSTEM_MOTOR_ID: return "Brain stem (descending)"
                names = dictionary.loc[dictionary['id'] == rid, 'safe_name'].tolist()
                return names[0] if names else f"ID {rid}"


            label_to_id = {_region_label(rid): rid for rid in selected_region_ids}
            base_options = ["(All L5 cells — no PT base)"] + list(label_to_id.keys())
            default_idx = next((i + 1 for i, lab in enumerate(label_to_id.keys()) if "brain stem" in lab.lower()), 0)

            base_choice = st.selectbox("Population base = 100%", options=base_options, index=default_idx)
            base_id = None if base_choice.startswith("(All L5") else label_to_id[base_choice]
            numerator_ids = [rid for rid in selected_region_ids if rid != base_id]

            if not numerator_ids:
                st.info("Add at least one more target region besides the base to build the summary.")
            else:
                summary = build_cortical_summary(results, base_id, numerator_ids, _region_label, criteria_used,
                                                 laterality=saved_laterality)
                tag = summary['slug']
                st.info(
                    f"**Projection criteria used:** {summary['criteria_note']}\nRecorded in every downloaded file name (`{tag}`).")

                st.markdown("**1. Brain stem = 100% (PT cells)** — *bs_benne*")
                st.dataframe(summary['benne'], use_container_width=True, hide_index=True)
                st.download_button("⬇ Download bs_benne.csv", summary['benne'].to_csv(index=False).encode('utf-8'),
                                   file_name=f"bs_benne_{tag}.csv", mime="text/csv", key="dl_benne")

                st.markdown("**2. All L5 = 100% (no brain-stem requirement)** — *bs_nelkul*")
                st.dataframe(summary['nelkul'], use_container_width=True, hide_index=True)
                st.download_button("⬇ Download bs_nelkul.csv", summary['nelkul'].to_csv(index=False).encode('utf-8'),
                                   file_name=f"bs_nelkul_{tag}.csv", mime="text/csv", key="dl_nelkul")

                st.markdown("**3. Average axon length in each target (µm), among PT cells**")
                st.dataframe(summary['axon'], use_container_width=True, hide_index=True)
                st.download_button("⬇ Download axon_length_summary.csv",
                                   summary['axon'].to_csv(index=False).encode('utf-8'),
                                   file_name=f"axon_length_summary_{tag}.csv", mime="text/csv", key="dl_axon")

                st.markdown("**4. Category tables with projecting cell IDs**")
                cat_slugs = category_slugs(list(summary['categories'].keys()))
                for lab, df in summary['categories'].items():
                    safe = cat_slugs[lab]
                    with st.expander(f"{lab} ({int(df.iloc[:, 2].sum())} cells)"):
                        st.dataframe(df, use_container_width=True, hide_index=True)
                        st.download_button(f"⬇ Download {safe}.csv", df.to_csv(index=False).encode('utf-8'),
                                           file_name=f"bs_{safe}_{tag}.csv", mime="text/csv", key=f"dl_cat_{safe}")

        with tab_inspector:
            st.markdown("Select a single cell from the processed population to view detailed metrics and its 3D scene.")
            inspect_name = st.selectbox("Select cell to inspect", options=[name for name, _ in results],
                                        label_visibility="collapsed")

            if inspect_name:
                _, inspect_result = next(r for r in results if r[0] == inspect_name)

                for tr in inspect_result.target_results:
                    cr = saved_criteria.get(tr.region_id, FilterCriteria())
                    meets_rule = cr.meets_thresholds(tr)
                    is_active = cr.is_active() and filter_was_active

                    if is_active and cr.operator == 'NOT' and meets_rule:
                        st.markdown(
                            f'<div class="result-card filtered-out"><h4>{tr.region_name}</h4><span class="tag-filtered">Violated NOT rule</span></div>',
                            unsafe_allow_html=True)
                    elif is_active and cr.operator == 'AND' and not meets_rule:
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

                if inspect_result.coords:
                    with st.spinner(f"Building 3D plot for {inspect_name}..."):
                        st.plotly_chart(build_3d_plot(inspect_result, atlas_matrix, inspect_name, show_soma_region,
                                                      show_other_regions, show_only_target_regions, descendants_used,
                                                      viz_theme, show_brain_outline), use_container_width=True)

        with tab_3d_multi:
            st.caption("Joint rendering of all processed cells. Each cell gets its own colour for easy distinction.")
            show_only_valid = st.toggle("Show only passing cells", value=True)

            combined_results = []
            for n, r in results:
                if not r.coords: continue
                if show_only_valid:
                    if filter_was_active and not r.passes_filter: continue
                    if not filter_was_active and not any(tr.projects_here for tr in r.target_results): continue
                combined_results.append((n, r))

            if not combined_results:
                st.warning("No cells match the current criteria for 3D rendering.")
            elif st.button(f"Generate Combined Scene ({len(combined_results)} cells)", type="primary"):
                with st.spinner(f"Rendering {len(combined_results)} cells together..."):
                    st.plotly_chart(build_3d_plot_multi(combined_results, atlas_matrix, selected_region_ids, True,
                                                        show_only_target_regions, descendants_used, viz_theme,
                                                        show_brain_outline), use_container_width=True)