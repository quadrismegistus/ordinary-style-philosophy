from . import *

def get_passage_html(slice_id, color='score_z_diff', max_score=None, min_score=None, word_feat_type='deprel'):
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


def get_doc_html(doc, color='score_z_diff', max_score=None, min_score=None, word_feat_type='deprel'):
    """
    Returns HTML for a stanza Document.
    """
    from .sentences import get_sent_html
    output_html = ['<div style="line-height: 2.8; font-family: sans-serif; padding: 10px;">']

    for sent in doc.sentences:
        sent_html = get_sent_html(sent, color=color, max_score=max_score, min_score=min_score, word_feat_type=word_feat_type)
        output_html.append(f'<p>{sent_html}</p>')
    output_html.append('</div>')
    return "".join(output_html)

def get_doc_html2(doc, **kwargs):
    """
    Returns HTML for a stanza Document using the side-by-side view.
    """
    from .sentences import get_all_sent_html
    return "\n".join([get_all_sent_html(sent, **kwargs) for sent in doc.sentences])
