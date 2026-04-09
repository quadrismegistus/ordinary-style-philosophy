import os

import numpy as np
import pandas as pd
import plotnine as p9

from .data_loaders import get_corpus_metadata


def fisher_test_pos(df_pos, target_col='_target', g1='Philosophy', g2='Literature'):
    """
    Run Fisher's exact test for each POS tag comparing two groups.
    
    For each POS, constructs a 2x2 contingency table:
                    | POS count | Other POS counts |
        Group 1     |    a      |        b         |
        Group 2     |    c      |        d         |
    
    Returns DataFrame with odds ratios and p-values.
    """
    from scipy.stats import fisher_exact
    
    # Split by group
    df_g1 = df_pos[df_pos[target_col] == g1].drop(columns=[target_col])
    df_g2 = df_pos[df_pos[target_col] == g2].drop(columns=[target_col])
    
    # Sum counts across all documents in each group
    g1_totals = df_g1.sum()
    g2_totals = df_g2.sum()
    
    # Total counts per group (across all POS)
    g1_total = g1_totals.sum()
    g2_total = g2_totals.sum()
    
    results = []
    for pos in g1_totals.index:
        # 2x2 contingency table
        a = g1_totals[pos]           # POS count in g1
        b = g1_total - a             # Other POS in g1
        c = g2_totals[pos]           # POS count in g2
        d = g2_total - c             # Other POS in g2
        
        table = [[a, b], [c, d]]
        odds_ratio, p_value = fisher_exact(table)
        
        results.append({
            'feat': pos,
            f'sum1': int(a),
            f'sum2': int(c),
            f'pct1': a / g1_total * 100,
            f'pct2': c / g2_total * 100,
            'odds_ratio': odds_ratio,
            'p_value': p_value,
            'sig': '*' if p_value < 0.05 else ('**' if p_value < 0.01 else ('***' if p_value < 0.001 else ''))
        })
    
    result_df = pd.DataFrame(results).set_index('feat')
    result_df['sig'] = result_df['p_value'].apply(lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else '')))
    return result_df.sort_values('odds_ratio', ascending=False)


def get_mdw_pos(groups, feat_n=None, feat_min_count=None, incl_deprel=True, incl_pos=True, feat_n_egs=None, rename_cols=False, **kwargs):
    from .constants import FEAT_N, FEAT_MIN_COUNT, FEAT2DESC
    from .features import get_pos_counts, get_pos_word_counts, get_pos_word_egs, get_egs
    
    if feat_n is None:
        feat_n = FEAT_N
    if feat_min_count is None:
        feat_min_count = FEAT_MIN_COUNT
    if feat_n_egs is None:
        feat_n_egs = FEAT_N // 2
    
    name1, ids1 = groups[0]
    name2, ids2 = groups[1]
    if isinstance(ids1, str):
        ids1 = get_corpus_metadata().query(ids1).index.tolist()
    if isinstance(ids2, str):
        ids2 = get_corpus_metadata().query(ids2).index.tolist()
    df_pos_grp1 = get_pos_counts(ids1, incl_deprel=incl_deprel, incl_pos=incl_pos)
    df_pos_grp2 = get_pos_counts(ids2, incl_deprel=incl_deprel, incl_pos=incl_pos)

    words_grp1 = get_pos_word_counts(ids1)
    words_grp2 = get_pos_word_counts(ids2)

    egs_grp1 = get_pos_word_egs(ids1)
    egs_grp2 = get_pos_word_egs(ids2)

    df_pos = pd.concat([df_pos_grp1.assign(_target=name1), df_pos_grp2.assign(_target=name2)])
    fisher_results = fisher_test_pos(df_pos, target_col='_target', g1=name1, g2=name2)

    df_means = df_pos.groupby('_target').mean()
    feat2grp2mean = df_means.to_dict()
    df_sums = df_pos.groupby('_target').sum()

    ld = []
    feat1 = None

    colname1 = f'1 ({name1})'
    colname2 = f'2 ({name2})'

    for feat in feat2grp2mean:
        feat_d = {'feat': feat}
        grp2mean = feat2grp2mean[feat]
        
        feat_d[f'fpk1'] = grp2mean[name1]
        feat_d[f'fpk2'] = grp2mean[name2]
        feat_d[f'top1'] = get_egs(words_grp1[feat], n=feat_n, min_count=feat_min_count)
        feat_d[f'top2'] = get_egs(words_grp2[feat], n=feat_n, min_count=feat_min_count)

        feat_d[f'egs1'] = get_egs(words_grp1[feat], n=feat_n_egs, min_count=feat_min_count, word2eg=egs_grp1[feat])
        feat_d[f'egs2'] = get_egs(words_grp2[feat], n=feat_n_egs, min_count=feat_min_count, word2eg=egs_grp2[feat])
        
        ld.append(feat_d)
    odf = pd.DataFrame(ld).dropna().set_index('feat')
    odf['fpk1-fpk2'] = odf[f'fpk1'] - odf[f'fpk2']
    odf['fpk1/fpk2'] = odf[f'fpk1'] / odf[f'fpk2']
    odf = fisher_results.join(odf).sort_values('p_value', ascending=True)
    odf['odds_ratio_log'] = np.log10(odf['odds_ratio'])
    odf['odds_ratio_log_abs'] = np.abs(odf['odds_ratio_log'])
    odf['feat_desc'] = [FEAT2DESC.get(feat, '?') for feat in odf.index]
    odf = odf.reset_index()

    def desc_result(row):
        if row.odds_ratio > 1:
            return f'{row.feat_desc}s are {row.odds_ratio:.1f}x more common in {name1} than {name2}.'
        else:
            return f'{row.feat_desc}s are {1/row.odds_ratio if row.odds_ratio != 0 else 0:.1f}x more common in {name2} than {name1}.'
    
    odf = odf.sort_values('odds_ratio_log_abs', ascending=False)
    odf['mdw_rank'] = [i+1 for i in range(len(odf))]

    odf = odf.sort_values('odds_ratio', ascending=False)
    odf['mdw1_rank'] = [i+1 for i in range(len(odf))]
    odf = odf.sort_values('odds_ratio', ascending=True)
    odf['mdw2_rank'] = [i+1 for i in range(len(odf))]

    odf['result_desc'] = odf.apply(desc_result, axis=1)
    odf = odf[[
        'feat', 
        'feat_desc', 
        'result_desc',
        'fpk1', 'fpk2',
        'odds_ratio',
        'sum1', 'sum2',
        'sig', 'top1', 'top2', 'egs1', 'egs2',
        'mdw_rank',
        'mdw1_rank',
        'mdw2_rank',
    ]]
    odf = odf.rename(columns={
        'sum1': f'{name1} (#)',
        'sum2': f'{name2} (#)',
        'pct1': f'{name1} (%)',
        'pct2': f'{name2} (%)',
        'fpk1': f'{name1} (#/k)',
        'fpk2': f'{name2} (#/k)',
        'odds_ratio': f'{name1} / {name2} (OR)',
        'sig': 'Significance',
        'top1': f'{name1} (top {feat_n})',
        'top2': f'{name2} (top {feat_n})',
        'egs1': f'{name1} (examples)',
        'egs2': f'{name2} (examples)',
        'fpk1-fpk2': f'{name1} - {name2} (#/k)',
    }) if rename_cols else odf
    return odf.set_index(['feat', 'feat_desc', 'result_desc']).sort_values('mdw_rank').dropna()


def printm(o):
    from IPython.display import display, Markdown
    display(Markdown(o))


def get_avgs_df(df, gby=["year", "discipline","comparison"], y="correct"):
    stats_df = (
        df.groupby(gby)[y]
        .agg(
            mean=np.mean,
            stderr=lambda x: x.std() / np.sqrt(len(x)),
            count=len,
        )
    )
    return stats_df.sort_values('mean', ascending=False)

def plot_avgs_df(figdf, **labs):
    fig = (
        p9.ggplot(
            figdf,
            p9.aes(x='year', y='mean', color='discipline'),
        )
        + p9.geom_errorbar(p9.aes(ymin='mean-stderr', ymax='mean+stderr'), width=1, alpha=0.75)
        + p9.geom_point(p9.aes(y='mean',size='count'), shape='x')
        + p9.facet_wrap('comparison')
        + p9.theme_minimal()
        + p9.scale_size_continuous(range=(.5, 5))
        + p9.geom_smooth(method='loess', span=0.5, alpha=0.25, size=1)
        + p9.labs(**labs)
    )
    return fig







def df_to_latex_table(
    df=None,
    save_latex_to: str | None = None,
    table_num: int | None = None,
    caption: str | None = None,
    label: str | None = None,
    position: str = 'H',
    center: bool = True,
    size: str | None = '\\small',
    singlespacing: bool = False,
    resize_to_textwidth: bool = False,
    column_format: str | None = None,
    escape: bool = True,
    index: bool = False,
    float_format: str | None = None,
    na_rep: str = '',
    save_latex_to_suffix: str | None = None,
    return_display: bool = False,
    inner_latex: str | None = None,
    longtable: bool = False,
    header: list[str] | None = None,
    verbose: bool = False,
):
    """Render a DataFrame (or provided LaTeX tabular) into a LaTeX table.

    Parameters
    - df: pandas.DataFrame or None if providing `inner_latex`
    - save_latex_to: where to write the .tex (if None, only return string)
    - table_num: optional table number prefix for caption (e.g., "Table 5: ...")
    - caption, label: LaTeX caption and label
    - position: table float position (e.g., 't', 'H')
    - center: add \\centering
    - size: e.g., '\\small' (set None to skip)
    - singlespacing: add \\singlespacing line
    - resize_to_textwidth: wrap tabular in \\resizebox{\\textwidth}{!}{% ... }
    - column_format, escape, index, float_format, na_rep, header: passed to DataFrame.to_latex
    - save_latex_to_suffix: if provided, insert before .tex (e.g., '.paper_regenerated')
    - return_display: if True and path provided, return IPython Image of compiled PNG
    - inner_latex: full tabular environment string to embed instead of DataFrame.to_latex
    - longtable: pass through to DataFrame.to_latex (no outer table env when True)

    Returns
    - The LaTeX string if not saving to file, otherwise the path or display object when requested.
    """
    # Build tabular (or longtable) body
    if inner_latex is not None:
        tabular_str = inner_latex.strip()
    else:
        if df is None:
            raise ValueError('Either df or inner_latex must be provided')
        # Import pandas lazily
        try:
            import pandas as _pd  # noqa: F401
        except Exception as _e:
            raise RuntimeError('pandas is required for df_to_latex_table when df is provided') from _e

        to_latex_kwargs: dict = {
            'index': index,
            'escape': escape,
            'na_rep': na_rep,
            'longtable': longtable,
            'bold_rows': False,
        }
        if header is not None:
            to_latex_kwargs['header'] = header
        if column_format is not None:
            to_latex_kwargs['column_format'] = column_format
        if float_format is not None:
            # pandas to_latex accepts a function or format string; we pass through strings
            to_latex_kwargs['float_format'] = float_format
        tabular_str = df.to_latex(**to_latex_kwargs).strip()

    # If this is a longtable, do not wrap in a floating table environment
    if longtable:
        latex_str = tabular_str
    else:
        lines: list[str] = []
        lines.append(f'\\begin{{table}}[{position}]')
        if center:
            lines.append('  \\centering')
        if size:
            lines.append(f'  {size}')
        if singlespacing:
            lines.append('  \\singlespacing')
        if resize_to_textwidth:
            lines.append('  \\resizebox{\\textwidth}{!}{%')
        # Ensure the tabular begins on its own line with correct indentation
        lines.append('  ' + tabular_str.replace('\n', '\n  '))
        if resize_to_textwidth:
            lines.append('  }')
        _caption_prefix = f'Table {table_num}: ' if table_num is not None else ''
        if caption is not None:
            lines.append(f'  \\caption{{{_caption_prefix}{caption}}}')
        if label is not None:
            lines.append(f'  \\label{{{label}}}')
        lines.append('\\end{table}')
        latex_str = '\n'.join(lines)

    # Save to file if requested
    if save_latex_to:
        if save_latex_to_suffix:
            if save_latex_to.endswith('.tex'):
                save_path = save_latex_to.replace('.tex', f'.{save_latex_to_suffix}.tex')
            else:
                save_path = f'{save_latex_to}.{save_latex_to_suffix}'
        else:
            save_path = save_latex_to
        if not save_path.endswith('.tex'):
            save_path += '.tex'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            if verbose:
                printm(f'* Writing LaTeX to `{nice_path(save_path)}`')
            f.write(latex_str)

        if return_display:
            try:
                from IPython.display import Image  # type: ignore
                png_path = save_path[:-4] + '.png' if save_path.endswith('.tex') else save_path + '.png'
                try:
                    ok = render_latex_to_png(latex_str, png_path)
                except Exception as e:
                    printm(f'* Warning: Could not render PNG: {e}')
                    pass
                if ok:
                    return Image(png_path)
                else:
                    if verbose:
                        printm(f"* warning: could not render PNG")
            except (NameError, ImportError):
                pass
        return save_path

    return latex_str