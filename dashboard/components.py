import streamlit as st
import pandas as pd
from osp import *
from osp.classify import DF_PREDS_METADATA_COLS

def style_prediction_df(df):
    if df.empty:
        return df
    
    # Identify quantitative columns: those starting with 'P' or 'n' or '%' or 'prob_'
    # We want to highlight probability columns
    quant_cols = [c for c in df.columns if "P(" in c or c.startswith('% ') or c.startswith('prob_') or c == 'Accuracy']
    
    styler = df.style
    for col in quant_cols:
        vmax = 1.0 if ("prob_" in col or "Accuracy" in col) else None
        styler = styler.background_gradient(cmap='RdBu', subset=[col], vmin=0, vmax=vmax)
    
    # Other quantitative columns (like 'n'): use column min/max
    n_cols = [c for c in df.columns if c == 'n' or c == 'count']
    for col in n_cols:
        styler = styler.background_gradient(cmap='Blues', subset=[col])

    # Format probabilities
    if quant_cols:
        styler = styler.format({c: "{:.3f}" for c in quant_cols})

    # Color-code discipline column
    if 'discipline' in df.columns:
        def color_discipline(val):
            if val == 'Literature':
                return 'background-color: #b2182b; color: white;'
            elif val == 'Philosophy':
                return 'background-color: #2166ac; color: white;'
            return ''
        styler = styler.applymap(color_discipline, subset=['discipline'])
        
    return styler


def lookup_examples(df_egs):
    if df_egs is None or df_egs.empty:
        return {}
    out = {}
    for _, row in df_egs.iterrows():
        feat = row.get("feat") or row.get("feature")
        if not feat:
            continue
        out.setdefault(feat, []).append(row.to_dict())
    return out


def render_prediction_explorer(df_slices, key_prefix="pred_explorer"):
    """
    Renders the prediction explorer component (two-column view with grouping).
    
    Args:
        df_slices: DataFrame with 'id' and probability columns. 
                   Should already have metadata merged if possible, or we merge it here?
                   Better if it has metadata columns: author, journal, year, title, etc.
        key_prefix: Unique key prefix for widgets to allow multiple instances.
    """
    
    # ensure we have metadata
    required_meta = [c for c in DF_PREDS_METADATA_COLS if c in df_slices.columns]
    
    st.markdown("### Prediction Explorer")
    
    # 2. Side-by-side Tables
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        grouping_cols = st.multiselect(
            "Group results by:",
            options=[c for c in ['author', 'journal', 'title', 'period', 'year', 'discipline'] if c in df_slices.columns],
            default=[c for c in ['period', 'journal'] if c in df_slices.columns],
            key=f"{key_prefix}_grouping"
        )
    
    selected_by = grouping_cols if grouping_cols else (['discipline'] if 'discipline' in df_slices.columns else [])
    
    # --- Aggregation Logic ---
    # We need to emulate get_nice_df_preds2 aggregation
    
    if not selected_by:
        st.warning("Select at least one grouping column.")
        return

    # Filter numeric columns for mean (including boolean 'correct' which averages to accuracy)
    numeric_cols = df_slices.select_dtypes(include=['number', 'bool']).columns.tolist()
    # Exclude runs if present
    if 'run' in numeric_cols: numeric_cols.remove('run')
    
    # Also check if we should aggregate 'target'
    # If target is constant within groups, we can show it.
    # We'll just include 'target' in the aggregation if it's present and not in grouping keys
    # But usually aggregation only handles numerics.
    # We can use 'first' for categorical columns if they are uniform.
    
    # Group by selected columns
    
    # 1. Mean of numeric columns (includes 'correct' -> accuracy)
    df_agg = df_slices.groupby(selected_by)[numeric_cols].mean().reset_index()
    
    # 2. Slice IDs
    df_counts = df_slices.groupby(selected_by)['id'].apply(lambda x: ';'.join(x.astype(str))).reset_index(name='slice_ids')
    
    # 3. Count
    df_n = df_slices.groupby(selected_by).size().reset_index(name='n')
    
    # 4. Target (if present) - take the mode or first
    df_target = None
    if 'target' in df_slices.columns and 'target' not in selected_by:
        # Check if target is unique per group
        # This is expensive. Let's just take the first one and hope it's consistent if grouped appropriately
        # Or better: don't show it if it's mixed.
        # Simple approach: Mode
        df_target = df_slices.groupby(selected_by)['target'].agg(lambda x: x.mode()[0] if not x.mode().empty else '').reset_index()
    
    df_nice = df_agg.merge(df_n, on=selected_by).merge(df_counts, on=selected_by)
    if df_target is not None:
        df_nice = df_nice.merge(df_target, on=selected_by)
    
    # Rename 'correct' to 'Accuracy' if present
    if 'correct' in df_nice.columns:
        df_nice['Accuracy'] = df_nice['correct']
        # Remove 'correct' to avoid confusion or keep it?
        # Let's drop it if we renamed it
        # df_nice = df_nice.drop(columns=['correct'])

    # Search/Filter
    with col_right:
        search_term = st.text_input("Search overview:", placeholder="", key=f"{key_prefix}_search")
    
    if search_term:
        mask = df_nice.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)
        df_filtered = df_nice[mask].copy()
    else:
        df_filtered = df_nice.copy()
        
    # Sort by n descending
    if 'n' in df_filtered.columns:
        df_filtered = df_filtered.sort_values('n', ascending=False)

    # Column config
    col_cfg = {
        "slice_ids": None, # Hidden
    }
    
    # Configure number columns
    for c in df_filtered.columns:
        if c.startswith("prob_") or c.startswith("P(") or c == "Accuracy":
            col_cfg[c] = st.column_config.NumberColumn(
                c,
                format="%.3f",
                width="small"
            )
        if c == 'n':
             col_cfg[c] = st.column_config.NumberColumn(
                "n",
                width="small"
            )

    styled_df_filtered = style_prediction_df(df_filtered)
    
    selection = st.dataframe(
        styled_df_filtered,
        width='stretch',
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config=col_cfg,
        key=f"{key_prefix}_main_table"
    )
    
    # Drill down
    if selection and selection.selection.rows:
        selected_row_idx = selection.selection.rows[0]
        selected_row = df_filtered.iloc[selected_row_idx]
        
        selected_slice_ids_str = selected_row.get('slice_ids', '')
        selected_slice_ids = [s.strip() for s in selected_slice_ids_str.split(';') if s.strip()]
        
        # Filter original slices
        df_text_slices = df_slices[df_slices['id'].isin(selected_slice_ids)].copy()
        
        if df_text_slices.empty:
            st.warning("No slice data found.")
        else:
            st.info(f"Showing {len(df_text_slices)} slices.")
            
            # Prepare display columns
            prob_cols = [c for c in df_text_slices.columns if c.startswith("prob_") or c.startswith("P(")]
            meta_cols_show = ['id'] + [c for c in DF_PREDS_METADATA_COLS if c in df_text_slices.columns and c not in selected_by]
            
            # Add View link
            df_text_slices['View'] = df_text_slices['id'].apply(lambda x: f"/Passages?slice_id={x}")
            
            display_cols = ['View'] + meta_cols_show + prob_cols
            display_cols = [c for c in display_cols if c in df_text_slices.columns]
            
            styled_slices = style_prediction_df(df_text_slices[display_cols])
            
            st.dataframe(
                styled_slices,
                width='stretch',
                hide_index=True,
                column_config={
                    "View": st.column_config.LinkColumn("View", display_text="Open ↗")
                },
                key=f"{key_prefix}_detail_table"
            )
            
    else:
        st.info("Select a group from the left table to view its constituent slices.")


def render_feature_summary(df_feats, name_a, name_b, egs_g1=None, egs_g2=None):
    """Render the two-column feature summary used by FeatureComparison."""
    if df_feats.empty or 'target' not in df_feats.columns:
        st.info("Feature summary data unavailable.")
        return
    
    df_feats['feat_type'] = [str(x).split('_')[0] for x in df_feats.feat]
    raw_lookup: dict[str, dict[str, float]] = {}
    for _, row in df_feats[['feat', 'target', 'raw']].iterrows():
        feat = row.get("feat")
        target = row.get("target")
        if not feat or not target:
            continue
        raw_lookup.setdefault(feat, {})[target] = float(row.get("raw", 0))
    def _format_example(ex):
        slice_id = ex.get("slice_id")
        href = f"/Passages?slice_id={slice_id}"
        meta = get_text_metadata(slice_id)
        title = meta.get("title", "Unknown")
        author = meta.get("author", "Unknown")
        year = meta.get("year", "Unknown")
        journal = meta.get("journal", "Unknown")
        signature = f'—{author}, "{title}", <i>{journal}</i> ({year})'
        html = ex.get("eg_html")
        word_id = ex.get("word_id")
        href_sent = f'/Sentence?slice_id={slice_id}&sent_id={ex.get("sent_id",1)}'
        if word_id is not None:
            href_sent += f'&word_id={word_id}'
        return f'<a href="{href_sent}" target="_blank">{html}</a> <a href="{href}" target="_blank" style="vertical-align: bottom; text-decoration: none; color: inherit; "><br/><small>{signature}</small></a><br/>'

    def _render_metrics(df, target, title, examples_dict, sort_by, rank_col, active_feat_type=None):
        if 'target' not in df.columns:
            st.write("Feature summary missing target information.")
            return
        st.markdown(f"#### {title}")
        df_target = df[df['target'] == target].sort_values(sort_by)
        if active_feat_type:
            df_target = df_target[df_target['feat_type'] == active_feat_type]
        if df_target.empty:
            st.write("No data available.")
            return
        other_name = name_b if target == name_a else name_a
        def _is_more_prevalent(feat_row):
            feat = feat_row.get("feat")
            raw_val = float(feat_row.get("raw", 0))
            other_val = raw_lookup.get(feat, {}).get(other_name, 0.0)
            return raw_val > other_val
        df_target = df_target[df_target.apply(_is_more_prevalent, axis=1)]
        if df_target.empty:
            st.write("No data available.")
            return
        for _, row in df_target.iterrows():
            feat = row["feat"]
            if feat in BAD_SLICE_FEATS:
                continue
            examples = (examples_dict or {}).get(feat, [])
            if isinstance(examples, dict):
                examples = [examples]
            all_html = "\n".join(
                _format_example(ex) for ex in examples if ex.get("eg_html") and ex.get("slice_id")
            )
            st.divider()
            rank_value = row.get(rank_col, "")
            feat_type = feat.split('_')[0]
            feat_name = feat.split('_')[1] if len(feat.split('_')) > 1 else feat
            rank_label = f"{int(rank_value)}. " if pd.notnull(rank_value) and rank_value != "" else ""
            st.markdown(f"##### {rank_label} {FEAT2DESC.get(feat, f'{feat_name} ({feat_type})')}")
            # if all_html:
            wrapped_html = (
                f'<div style="overflow-x:auto; white-space:nowrap; width:100%;">'
                f'{all_html if all_html else "&nbsp;"}'
                f'</div>'
            )
            st.components.v1.html(wrapped_html, height=194, scrolling=True)
            # metrics showing raw/z comparisons
            row_g1_df = df[(df['feat'] == feat) & (df['target'] == name_a)]
            row_g2_df = df[(df['feat'] == feat) & (df['target'] == name_b)]
            row_g1 = row_g1_df.iloc[0] if not row_g1_df.empty else None
            row_g2 = row_g2_df.iloc[0] if not row_g2_df.empty else None
            raw_a = row_g1["raw"] if row_g1 is not None and "raw" in row_g1 else 0
            raw_b = row_g2["raw"] if row_g2 is not None and "raw" in row_g2 else 0
            z_a = row_g1["z"] if row_g1 is not None and "z" in row_g1 else 0
            z_b = row_g2["z"] if row_g2 is not None and "z" in row_g2 else 0
            c1, c1b, c2, c2b, c3 = st.columns([3, 0.5, 3, 0.5, 3], vertical_alignment="top")
            with c1b:
                st.text("–")
            with c2b:
                st.text("=")
            if target == name_b:
                raw_a2 = f"{raw_a:.1f}" if raw_a > 1 else f"{raw_a:.2f}"
                raw_b2 = f"{raw_b:.1f}" if raw_b > 1 else f"{raw_b:.2f}"
                raw_diff2_rev = (
                    f"{raw_b-raw_a:.1f}" if raw_b - raw_a > 1 else f"{raw_b-raw_a:.2f}"
                )
                with c1:
                    st.metric(
                        label=f"{name_b} (G2)",
                        value=f"{raw_b2}",
                        delta=f"{z_b:+.2f}z"
                    )
                with c2:
                    st.metric(
                        label=f"{name_a} (G1)",
                        value=f"{raw_a2}",
                        delta=f"{z_a:+.2f}z"
                    )
                with c3:
                    st.metric(
                        label="Δ (G2-G1)",
                        value=f"{raw_diff2_rev}",
                        delta=f"{z_b-z_a:+.2f}z",
                    )
            else:
                raw_a2 = f"{raw_a:.1f}" if raw_a > 1 else f"{raw_a:.2f}"
                raw_b2 = f"{raw_b:.1f}" if raw_b > 1 else f"{raw_b:.2f}"
                raw_diff2 = (
                    f"{raw_a-raw_b:.1f}" if raw_a - raw_b > 1 else f"{raw_a-raw_b:.2f}"
                )
                with c1:
                    st.metric(
                        label=f"{name_a} (G1)",
                        value=f"{raw_a2}",
                        delta=f"{z_a:+.2f}z"
                    )
                with c2:
                    st.metric(
                        label=f"{name_b} (G2)",
                        value=f"{raw_b2}",
                        delta=f"{z_b:+.2f}z"
                    )
                with c3:
                    st.metric(
                        label="Δ (G1-G2)",
                        value=f"{raw_diff2}",
                        delta=f"{z_a-z_b:+.2f}z",
                    )

    st.markdown(
        """
        <style>
        div[data-testid="column"] div[data-testid="stMetric"] {
            padding: 0.5rem;
            border-radius: 0.75rem;
            border: 1px solid rgba(0,0,0,0.05);
            background: rgba(255,255,255,0.04);
        }
        </style>
        """, unsafe_allow_html=True
    )

    tabs = st.tabs(["All Features", "Clauses", "POS", "Dependency", "Phrases", "TTR"])
    feat_types = ['all', 'sent', 'pos', 'deprel', 'phrase', 'ttr']
    for tab, ftype in zip(tabs, feat_types):
        with tab:
            col1, col2 = st.columns(2)
            with col1:
                _render_metrics(
                    df_feats,
                    name_a,
                    f"{name_a} (G1)",
                    egs_g1,
                    sort_by="feat_rank1",
                    rank_col="feat_rank1",
                    active_feat_type=None if ftype == 'all' else ftype,
                )
            with col2:
                _render_metrics(
                    df_feats,
                    name_b,
                    f"{name_b} (G2)",
                    egs_g2,
                    sort_by="feat_rank2",
                    rank_col="feat_rank2",
                    active_feat_type=None if ftype == 'all' else ftype,
                )

