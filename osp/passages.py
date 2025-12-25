from . import *

def get_sent_html(sent, color='score_z_diff', max_score=None, min_score=None, color_by=None, word_feat_type='deprel'):
    if color_by is None:
        color_by = get_feat_color_by(color=color, max_score=max_score, min_score=min_score)
        min_score = min(color_by.values())
        max_score = max(color_by.values())

    sent_html = []
    for word in sent.words:
        # Match the feature naming convention used in the codebase (pos_TAG, deprel_REL)
        pos = word.xpos or word.upos
        deprel = word.deprel
        
        # Combine scores for the word's features
        score = (
            color_by.get(f'deprel_{deprel}',0) 
            if word_feat_type == 'deprel' 
            else color_by.get(f'pos_{pos}',0)
        )
        
        # Map score to color intensity (clamped to [-1, 1])
        # Blue (1) to Orange (-1)
        intensity = max(min_score, min(max_score, score))
        
        if intensity > 0:
            # Positive score -> Blue
            bg_color = f'rgba(0, 0, 255, {intensity:.2f})'
            text_color = 'white' if intensity > 0.5 else 'black'
        elif intensity < 0:
            # Negative score -> Orange
            bg_color = f'rgba(255, 165, 0, {-intensity:.2f})'
            text_color = 'white' if -intensity > 0.5 else 'black'
        else:
            bg_color = 'transparent'
            text_color = 'black'
        
        # Escape text for safety and create annotated span
        safe_text = html.escape(word.text)
        posdeprel = deprel if word_feat_type == 'deprel' else pos
        word_span = (
            f'<span style="background-color: {bg_color}; color: {text_color}; font-size: 1.2em; font-weight: normal; '
            f'display: inline-block; text-align: left; vertical-align: top; line-height: 1; '
            f'padding: 0 2px;">'
            f'{safe_text}'
            f'<sub style="display: block; font-size: 0.7em; opacity: 0.7; line-height: 1; font-weight: normal; padding: 2px; font-family: monospace;">'
            f'{posdeprel}</sub>'
            f'</span>'
        )
        sent_html.append(word_span)
    return "".join(sent_html)

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

def get_doc_html(doc, color='score_z_diff', max_score=None, min_score=None, word_feat_type='deprel'):
    """
    Returns HTML for a stanza Document.
    """
    output_html = ['<div style="line-height: 2.8; font-family: sans-serif; padding: 10px;">']

    for sent in doc.sentences:
        sent_html = get_sent_html(sent, color=color, max_score=max_score, min_score=min_score, word_feat_type=word_feat_type)
        output_html.append(f'<p>{sent_html}</p>')
    output_html.append('</div>')
    return "".join(output_html)

def get_passage_html(slice_id, color='score_z_diff', max_score=None, min_score=None,word_feat_type='deprel'):
    """
    Displays the passage for a given slice_id in HTML.
    Annotates words with POS and deprel, and colors them by their weight in color_by.
    """
    if slice_id not in STASH_SLICES_NLP:
        print(f"Slice ID {slice_id} not found.")
        return   
    docstr = STASH_SLICES_NLP[slice_id]
    doc = stanza.Document.from_serialized(docstr)
    return get_doc_html(doc, color=color, max_score=max_score, min_score=min_score, word_feat_type=word_feat_type)

def display_passage(slice_id, **kwargs):
    display(HTML(get_passage_html(slice_id, **kwargs)))