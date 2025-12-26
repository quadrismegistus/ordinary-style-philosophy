from . import *
import nltk
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from bs4 import BeautifulSoup
from spacy import displacy
from IPython.display import HTML, display

COLOR_MIN_SCORE = -2
COLOR_MAX_SCORE = 2

def get_sent_obj(txt):
    return get_nlp_doc(txt).sentences[0]


def is_leaf(tree):
    """Check if a tree node is a leaf (preterminal) node."""
    return not isinstance(tree[0], nltk.Tree) if len(tree) > 0 else False


def get_leaf_ids(tree):
    """Get the IDs of the leaf nodes in the tree."""
    return [word.id for word in tree.leaves()]

def iter_preterms(tree):
    """Iterate over the preterminal nodes of the tree."""
    for t in tree.subtrees():
        if is_leaf(t):
            yield t

def get_clause_tokens(tree):
    """
    Advanced version of clause tokenization that assigns clause numbers and types
    to each preterminal in the tree.
    """
    clause_num = 0
    clause_type = None

    def is_it_new_clause(t):
        ttxt = ' '.join(t.leaves())
        if not ttxt: return False
        if not any(c.isalpha() for c in ttxt): return False
        if not t.label() == 'SBAR': return False
        # if not ((not is_in_sbar(t) and t.label() == 'S') or t.label() == 'SBAR'): return False
        return True

    for t in tree.subtrees():
        ttxt = ' '.join(t.leaves())
        # A clause starts if it's an S or SBAR and has alphabetic content
        if ttxt and any(c.isalpha() for c in ttxt) and ((not is_in_sbar(t) and t.label() == 'S') or t.label() == 'SBAR'):
            clause_num += 1
            clause_type = 'DC' if is_in_sbar(t) else 'IC'

        t._clause_num = clause_num
        t._clause_type = clause_type

    o = []
    for preterm_i, preterm in enumerate(iter_preterms(tree)):
        d = {
            'clause_num': getattr(preterm, '_clause_num', 0),
            'clause_type': getattr(preterm, '_clause_type', None),
            'word_num': preterm_i + 1,
            'word': ' '.join(preterm.leaves())
        }
        o.append(d)
    
    if not o:
        return pd.DataFrame()
        
    odf = pd.DataFrame(o)
    # Re-rank to ensure dense numbering
    odf['clause_num'] = odf['clause_num'].rank(method='dense', ascending=True).apply(int)
    return odf

def tokenize_clauses(tree):
    """Alias for get_clause_tokens for compatibility with existing notebooks."""
    return get_clause_tokens(tree)

def get_clauses(sent):
    """
    Get a DataFrame of clauses for a given sentence.
    """
    if isinstance(sent, str):
        sent = get_nlp_doc(sent).sentences[0]
    tree = get_sent_tree(sent)
    odf = get_clause_tokens(tree)
    if odf.empty:
        return odf
        
    # Ensure word count matches, though small mismatches can happen with complex trees
    num_words = len(sent.words)
    num_df = len(odf)
    
    odf['pos'] = [sent.words[i].xpos for i in range(min(num_words, num_df))]
    odf['deprel'] = [sent.words[i].deprel for i in range(min(num_words, num_df))]
    odf['head'] = [sent.words[i].head for i in range(min(num_words, num_df))]
    
    return odf.set_index(['clause_num','clause_type','word_num'])

def get_color(
    feat,
    color_by='weight_z',
    df_feats=None,
    vcenter=0.0,
    vmin=COLOR_MIN_SCORE,
    vmax=COLOR_MAX_SCORE,
    cmap_name='RdBu'
):
    """
    Get a hex color for a feature based on its weight.
    """
    if df_feats is None:
        df_feats = get_current_feat_weights()
    
    val = df_feats.loc[feat, color_by] if feat in df_feats.index else 0
    
    if vmin is None:
        vmin = df_feats[color_by].min()
    if vmax is None:
        vmax = df_feats[color_by].max()

    if vmin < vcenter < vmax:
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(norm(val))
    return mcolors.to_hex(rgba)

def get_displacy_data(sent):
    """
    Convert a Stanza sentence into the manual data format expected by displaCy.
    """
    words = [{"text": word.text, "tag": word.xpos} for word in sent.words]
    arcs = []
    
    for word in sent.words:
        if word.head == 0:
            continue
        
        start = word.head - 1
        end = word.id - 1
        
        if start == end:
            continue
            
        if start < end:
            direction = "left"
        else:
            direction = "right"
            start, end = end, start
            
        arcs.append({
            "start": start,
            "end": end,
            "label": word.deprel,
            "dir": direction
        })
    
    return {"words": words, "arcs": arcs}

def render_sent_displacy(sent, color_by='weight_z', df_feats=None, options=None, jupyter=True, **kwargs):
    """
    Render a sentence using displaCy with elements colored by their feature weights.
    """
    if isinstance(sent, str):
        sent = get_nlp_doc(sent).sentences[0]
        
    data = get_displacy_data(sent)
    
    if options is None:
        options = {'compact': False}
        
    html_output = displacy.render(data, style="dep", manual=True, options=options, jupyter=False)
    soup = BeautifulSoup(html_output, 'html.parser')
    
    if df_feats is None:
        df_feats = get_current_feat_weights()

    # Color tags (POS)
    for tag_elem in soup.find_all('tspan', class_='displacy-tag'):
        feat_name = f'pos_{tag_elem.text}'
        color = get_color(feat_name, color_by=color_by, df_feats=df_feats)
        tag_elem['style'] = f"paint-order: stroke fill; stroke: {color}; stroke-width: 1em; stroke-linecap: round; stroke-linejoin: round; font-size: 1.5em;"
        tag_elem['fill'] = 'black'

    # Color dependency relations (deprel)
    for deprel_elem in soup.find_all('textpath'):
        feat_name = f'deprel_{deprel_elem.text}'
        color = get_color(feat_name, color_by=color_by, df_feats=df_feats)
        deprel_elem['style'] = f"paint-order: stroke fill; stroke: {color}; stroke-width: .3em; font-size: 1.5em;"
        deprel_elem['fill'] = 'black'
    
    for word_elem in soup.find_all('tspan', class_='displacy-word'):
        word_elem['style'] = f"font-size: 1.5em;"

    res_html = str(soup)
    if jupyter:
        display(HTML(res_html))
    else:
        return res_html


def get_all_sent_html(sent, **kwargs):
    psg = get_sent_html(sent, **kwargs)
    diagram = render_sent_displacy(sent, jupyter=False, **kwargs)
    return f'''
    <div style="display: flex; flex-direction: row; gap: 10px;">
    <div style="flex: 1; min-width: 300px;">{psg}</div><div style="flex: 1; min-width: 0; zoom: 0.5;">{diagram}</div>
    </div>
    '''

def get_feat_color_by(color='score_z_diff', min_score=None, max_score=None):
    df_feat_weights = get_current_feat_weights()
    if not color in df_feat_weights.columns:
        return {}
    
    vals = df_feat_weights[color].values
    if min_score is None:
        min_score = min(vals)
    if max_score is None:
        max_score = max(vals)
    color_by = {k: float(max(min_score, min(max_score, v)) )for k, v in zip(df_feat_weights.index, vals)}
    return color_by

def get_sent_html(sent, color='score_z_diff', max_score=None, min_score=None, color_by=None, word_feat_type='deprel', df_feats=None, font_size="1.2em", show_labels=False):
    """
    Returns HTML for a stanza Sentence, with words colored by feature weights.
    """
    if df_feats is None:
        df_feats = get_current_feat_weights()
        
    if color_by is None:
        color_by = get_feat_color_by(color=color, max_score=max_score, min_score=min_score)
        if color_by:
            min_score = min(color_by.values())
            max_score = max(color_by.values())
        else:
            min_score, max_score = 0, 0

    sent_html = []
    import html
    for word in sent.words:
        # Match the feature naming convention used in the codebase (pos_TAG, deprel_REL)
        pos = word.xpos or word.upos
        deprel = word.deprel
        feat_name = f'deprel_{deprel}' if word_feat_type == 'deprel' else f'pos_{pos}'
        
        # Use get_color for consistent coloring with displaCy
        bg_color = get_color(feat_name, color_by=color, df_feats=df_feats, vmin=min_score, vmax=max_score)
        
        # Determine if text should be white or black for readability
        val = df_feats.loc[feat_name, color] if feat_name in df_feats.index else 0
        
        # Normalize val for contrast checking
        if min_score < 0 < max_score:
            intensity = val / max_score if val > 0 else val / abs(min_score)
        elif max_score != min_score:
            intensity = (val - (min_score + max_score) / 2) / ((max_score - min_score) / 2)
        else:
            intensity = 0
            
        text_color = 'white' if abs(intensity) > 0.7 else 'black'
        
        # Escape text for safety and create annotated span
        safe_text = html.escape(word.text)
        # posdeprel = deprel if word_feat_type == 'deprel' else pos
        posdeprel = f'{pos}/{deprel}' if show_labels else ''
        
        label_style = f"opacity: 0.7;" if show_labels else "display: none;"
        
        word_span = (
            f'<span style="background-color: {bg_color}; color: {text_color}; font-size: {font_size}; font-weight: normal; '
            f'display: inline-block; text-align: left; vertical-align: top; line-height: 1.2; '
            f'padding: 2px 4px; border-radius: 4px; margin: 1px;">'
            f'{safe_text}'
            f'<sub style="display: block; font-size: 0.6em; {label_style} line-height: 1; font-weight: normal; padding: 2px; font-family: monospace;">'
            f'{posdeprel}</sub>'
            f'</span>'
        )
        sent_html.append(word_span)
    return "".join(sent_html)

def get_sent_html2(sent, color='score_z_diff', max_score=COLOR_MAX_SCORE, min_score=COLOR_MIN_SCORE, word_feat_type='deprel', df_feats=None, font_size="1.3rem", show_labels=True):
    """
    Returns a list of items for use with streamlit's annotated_text.
    """
    try:
        from annotated_text import annotation
    except ImportError:
        annotation = lambda text, label, background, color, **kwargs: (text, label, background, color)

    if df_feats is None:
        df_feats = get_current_feat_weights()
        
    vals = df_feats[color].values if color in df_feats.columns else [0]
    if min_score is None: min_score = min(vals)
    if max_score is None: max_score = max(vals)

    items = []
    for word in sent.words:
        pos = word.xpos or word.upos
        deprel = word.deprel
        feat_name = f'deprel_{deprel}' if word_feat_type == 'deprel' else f'pos_{pos}'
        
        bg_color = get_color(feat_name, color_by=color, df_feats=df_feats, vmin=min_score, vmax=max_score)
        
        # Contrast logic
        val = df_feats.loc[feat_name, color] if feat_name in df_feats.index else 0
        if min_score < 0 < max_score:
            intensity = val / max_score if val > 0 else val / abs(min_score)
        elif max_score != min_score:
            intensity = (val - (min_score + max_score) / 2) / ((max_score - min_score) / 2)
        else:
            intensity = 0
        text_color = 'white' if abs(intensity) > 0.7 else 'black'
        
        label = (deprel if word_feat_type == 'deprel' else pos) if show_labels else ""
        
        items.append(annotation(
            word.text, 
            label, 
            background=bg_color, 
            color=text_color, 
            font_size=font_size,
            # font_family="Arial"
        ))
        items.append(" ")
        
    return items

def get_doc_annotated_items(doc, **kwargs):
    """
    Returns a flat list of items for the whole document, suitable for annotated_text.
    """
    all_items = []
    for i, sent in enumerate(doc.sentences):
        all_items.extend(get_sent_html2(sent, **kwargs))
        if i < len(doc.sentences) - 1:
            all_items.append("\n\n")
    return all_items

def display_doc_annotated(doc, key_prefix="doc", **kwargs):
    """
    Helper to display a document using st-annotated-text in Streamlit.
    """
    import streamlit as st
    try:
        from annotated_text import annotated_text
    except ImportError:
        st.error("st-annotated-text not installed. Please install with 'pip install st-annotated-text'")
        return

    # Define dialog for displaying sentence diagram
    if hasattr(st, 'dialog'):
        @st.dialog("Sentence Structure", width="large")
        def show_sent_dialog(sent, sent_num):
            st.caption(f"Sentence {sent_num}: {sent.text}")
            # Map 'color' arg to 'color_by' for render_sent_displacy
            render_kwargs = kwargs.copy()
            if 'color' in render_kwargs and 'color_by' not in render_kwargs:
                render_kwargs['color_by'] = render_kwargs['color']
            
            html_content = render_sent_displacy(sent, jupyter=False, **render_kwargs)
            
            # Inject script to scroll to bottom-left
            # displaCy puts words at the bottom. We want to see them immediately.
            html_with_script = f"""
            <div style="zoom: 0.75;">
                {html_content}
            </div>
            <script>
                // Scroll to bottom left on load
                setTimeout(function() {{
                    window.scrollTo(0, document.body.scrollHeight);
                }}, 100);
            </script>
            """
            
            # Use scrolling container for large SVGs
            # Height 600 fits within most laptop screens without scrolling the main page
            st.components.v1.html(html_with_script, height=600, scrolling=True)
    else:
        def show_sent_dialog(sent, sent_num):
            with st.expander(f"Sentence {sent_num} Diagram", expanded=True):
                # Map 'color' arg to 'color_by' for render_sent_displacy
                render_kwargs = kwargs.copy()
                if 'color' in render_kwargs and 'color_by' not in render_kwargs:
                    render_kwargs['color_by'] = render_kwargs['color']
                
                html_content = render_sent_displacy(sent, jupyter=False, **render_kwargs)
                
                html_with_script = f"""
                <div style="zoom: 0.75;">
                    {html_content}
                </div>
                <script>
                    setTimeout(function() {{
                        window.scrollTo(0, document.body.scrollHeight);
                    }}, 100);
                </script>
                """
                st.components.v1.html(html_with_script, height=500, scrolling=True)

    # Sort option for sentences
    sort_order = st.selectbox(
        "Sort sentences by:",
        ["ID (Ascending)", "ID (Descending)", "Length (Ascending)", "Length (Descending)"],
        key=f"{key_prefix}_sort"
    )
    
    # Prepare list of sentences with indices
    sent_list = list(enumerate(doc.sentences))
    
    if "Length" in sort_order:
        sent_list.sort(key=lambda x: len(x[1].words), reverse="Descending" in sort_order)
    elif "Descending" in sort_order:
        sent_list.reverse()
        
    # Header row to mimic table layout
    col1, col2 = st.columns([0.8, 12])
    col1.markdown("**ID**")
    col2.markdown("**Sentence**")
    st.divider()

    for i, sent in sent_list:
        # Create a layout with a small column for the ID/button and large for text
        col1, col2 = st.columns([0.8, 12])
        sent_num = i + 1
        
        with col1:
            # Display ID as a button
            if st.button(f"{sent_num}", key=f"{key_prefix}_sent_btn_{i}", help="Click to view syntactic diagram"):
                show_sent_dialog(sent, sent_num)
        
        with col2:
            # Render annotated HTML directly so it shows in the “table” cell
            sent_html = get_sent_html(
                sent,
                color=kwargs.get("color", "score_z_diff"),
                max_score=kwargs.get("max_score", COLOR_MAX_SCORE),
                min_score=kwargs.get("min_score", COLOR_MIN_SCORE),
                word_feat_type=kwargs.get("word_feat_type", "deprel"),
                df_feats=kwargs.get("df_feats", None),
                font_size="1.1em",
                show_labels=True
            )
            st.markdown(sent_html, unsafe_allow_html=True)



def display_sent_annotated(sent, **kwargs):
    """
    Helper to display a sentence using st-annotated-text in Streamlit.
    """
    try:
        from annotated_text import annotated_text
        items = get_sent_html2(sent, **kwargs)
        annotated_text(*items)
    except ImportError:
        print("st-annotated-text not installed. Please install with 'pip install st-annotated-text'")

def detokenize_sent(sent):
    l = []
    for tok in sent.tokens:
        l.append(tok.text)
        l.append(tok.spaces_after)
    return ''.join(l).strip()

def is_valid_sent_feat(k):
    for bw in ['parens','punct','noun','verb','adjective']:
        if bw in k:
            return False
    return True

def get_sent_feats(sent,per_n_words=None):
    tree = get_sent_tree(sent)
    feats = get_tree_stats(tree)
    feats['sent'] = detokenize_sent(sent)
    feats = {k:v for k,v in feats.items() if k=='sent' or is_valid_sent_feat(k)}
    if per_n_words:
        num_words = get_num_words(tree)
        feats['num_words'] = num_words
        for k,v in feats.items():
            if isinstance(v,(int,float)):
                feats[k] = v / per_n_words
    html = get_sent_html(sent, color='weight_z', min_score=-2, max_score=2,show_labels=True, color_by='weight_z')
    return {'sent_i':sent.index, 'html':html, **feats}

def get_sents_feats(doc,per_n_words=None):
    return pd.DataFrame(get_sent_feats(sent,per_n_words) for sent in doc.sentences)

def get_sents_feats_df(doc,per_n_words=None, color_by='weight_z', html=False, with_weights=False):
    df = get_sents_feats(doc,per_n_words)
    if not html: 
        df = df.drop(columns=['html'])
    else:
        df = df.drop(columns=['sent']).rename(columns={'html':'sent'})
    if with_weights:
        df2 = get_sent_weights(doc, color_by=color_by).drop(columns=['sent'])
        df = df.merge(df2, on='sent_i', how='left')
        
    df['sent_num']=df['sent_i']+1
    df = df.drop(columns=['sent_i'])
    df.columns=[col.replace('_',' ').title() if not '(' in col and col != 'html' else col for col in df.columns]
    return df.set_index(['Sent Num'])

def get_sent_weights(doc, color_by='weight_z'):
    o = []
    # weights = get_current_feat_weights(group_by=('feature',))
    weights_cmp = get_current_feat_weights(group_by=('comparison',))[color_by]

    dfx=get_slice_feats_by_word(doc, color_by)
    dfx_all_s = dfx.groupby(['sent_i']).mean(numeric_only=True).reset_index().set_index('sent_i')[color_by]
    dfx_feat = dfx.groupby(['sent_i','feat_type']).mean(numeric_only=True).reset_index().set_index('sent_i')
    dfx_feat_piv = dfx_feat.reset_index().pivot(index='sent_i', columns='feat_type', values='weight_z')

    for sent_i, sent in enumerate(doc.sentences):
        out_d = {'sent_i':sent_i, 'sent':detokenize_sent(sent)}
        out_d['P(Phil)'] = dfx_all_s.loc[sent_i]
        
        sentrow = dfx_feat_piv.loc[sent_i]
        for col in sentrow.index:
            out_d[f'P(Phil|{col})'] = sentrow.loc[col]

        for cmp,score in weights_cmp.items():
            out_d[f'P(Phil|{cmp.split("-")[0]})'] = score
        o.append(out_d)
    return pd.DataFrame(o)
    
