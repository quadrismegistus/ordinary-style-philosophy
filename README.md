# ordinary-style-philosophy

This repo contains the data pipeline and analysis code for the *Ordinary Style Philosophy* project.

At a high level:

1. You start with a JSTOR export (`jsonl.gz`) plus a large JSTOR metadata file you already have.
2. You assemble a local corpus: one `.txt` file per document + a single `data/metadata.csv`.
3. You slice each text into fixed-length chunks (“slices”).
4. You parse slices with Stanza (POS, dependencies, constituency).
5. You extract slice-level features.
6. You train simple classifiers (logistic regression) on those features and explore results in a Streamlit dashboard.

The “library” lives in `osp/`. Most end-to-end runs were originally done via notebooks in `notebooks/`.

---

## Repo layout (the parts that matter for the pipeline)

- `osp/`: core pipeline code (loaders, slicing, NLP, features, models, HTML renderers).
- `notebooks/`: one-off / batch notebooks that assemble corpus data, build stashes, run experiments.
- `data/`: inputs and outputs.
  - `data/raw/`: raw inputs (`*.jsonl.gz`, etc.) and HashStash caches (“stashes”).
  - `data/txt/`: the constructed corpus (one `.txt` file per doc).
  - `data/metadata.csv`: corpus metadata table used everywhere downstream.
- `dashboard/`: Streamlit app that reads the stashes and feature tables.

---

## Data conventions

### IDs

All documents have an `id` that also acts as the path under `data/txt/`:

- Philosophy docs look like `phil/10.2307/40231690`
- Literature/PMLA docs look like `lit/10.2307/12345678`

Text files live at:

`data/txt/<id>.txt`  (so the slashy id becomes folders)

### Core corpus files

- `data/metadata.csv`: metadata indexed by `id` (must include at least `year` and `discipline`).
- `data/txt/.../*.txt`: one file per document, raw-ish full text.

### Stashes (HashStash caches)

`osp/constants.py` defines many `HashStash` stores under `data/raw/stash/`. These are used to cache slow steps:

- `STASH_SLICES`: text → slices
- `STASH_SLICES_NLP`: slice_id → serialized Stanza `Document`
- `STASH_SLICE_FEATS`: slice_id → feature dict
- `STASH_PREDS_FEATS`: cached classifier outputs

If you delete stashes, you’ll need to regenerate downstream artifacts.

---

## Pipeline, step by step

### Step 0 — Put raw inputs in place

The loader code expects these by default (see `osp/constants.py`):

- `data/raw/jstor_data.jsonl.gz` (smaller “data” file; includes full text/pages)
- `data/raw/jstor_metadata_*.jsonl.gz` (large JSTOR metadata dump)
- Optional / legacy: `data/raw/LitStudiesJSTOR.jsonl` (PMLA/LitStudies JSONL)

The exact filenames are hardcoded in `osp/constants.py` (`FN_JSTOR_DATA`, `FN_JSTOR`, `FN_PMLA`).

### Step 1 — Build the corpus folder + metadata table

Goal:

- Write text files into `data/txt/phil/...` and/or `data/txt/lit/...`
- Write `data/metadata.csv`

The canonical implementation is in:

- `notebooks/AssembleCorpus.ipynb`

What it does (roughly):

- Loads philosophy data via `osp.data_loaders.get_jstor_data()` (merges `jstor_data.jsonl.gz` with the large JSTOR metadata file).
- Loads literature/PMLA data via `osp.data_loaders.get_pmla_df()` (optional).
- Normalizes both into a single DataFrame with fields like:
  - `id`, `uuid`, `title`, `author`, `year`, `journal`, `volume`, `issue`, `url`, `publisher`, `discipline`
- Writes:
  - `data/txt/<id>.txt`
  - `data/metadata.csv`

Downstream code assumes `data/metadata.csv` exists and is indexed by `id`.

### Step 2 — Load and lightly clean corpus text

Downstream steps don’t read the `.txt` files verbatim; they call:

- `osp.data_loaders.get_corpus_txt(id)`

This reads `data/txt/<id>.txt`, sentence-tokenizes with NLTK, dehyphenates line-break hyphenation, drops digit-heavy lines, and returns text as one sentence per line.

### Step 3 — Slice each text into fixed-length chunks

Goal: turn each text into many “slices” of a fixed length (default 1000) measured in *recognized words*.

Key functions:

- `osp.slices.get_text_slices(text_id, slice_len=1000)`
- `osp.slices.iter_txt_slices(...)` (internal helper)

Vocabulary filtering:

- recognized words are defined by `osp.data_loaders.get_ok_words()` (derived from a word database file referenced in `osp/constants.py`)

Output:

- slices are cached in `STASH_SLICES` (a HashStash), keyed by `text_id`.
- each slice has a “slice id” like `text_id__NN` (see gotcha below).

The “loop over every document and slice it” example is in:

- `notebooks/CorpusSlices.ipynb`

### Step 4 — Parse slices (Stanza) and stash serialized docs

Goal: run Stanza (`tokenize,mwt,pos,lemma,ner,depparse,constituency`) on each slice and store a serialized `stanza.Document`.

Key functions:

- `osp.nlp_utils.get_nlp()` (builds the Stanza pipeline)
- `osp.nlp_utils.get_nlp_doc(txt, id=...)` (caches per-(id,txt) docs in `NLP_STASH`)

In practice, the large batch parse was done in:

- `notebooks/ParseSlice.ipynb`

That notebook:

- reads slices out of a slice stash (e.g. `osp_slices_1000`)
- writes parsed docs to a “nlp stash” (e.g. `osp_slices_1000_nlp`)
- stores them with:
  - `stash[slice_id] = doc.to_serialized()`

The main codebase assumes parsed slice docs are available in:

- `STASH_SLICES_NLP` (configured in `osp/constants.py`)

### Step 5 — Extract slice-level features

Goal: compute a feature dictionary per parsed slice and store it for modeling and visualization.

Key functions:

- `osp.features.gen_all_slice_feats(...)`
  - iterates `osp.features.get_parsed_slice_ids()` (which are the keys of `STASH_SLICES_NLP`)
  - writes per-slice feature dicts into `STASH_SLICE_FEATS`
- `osp.features.get_all_feats(...)`
  - loads `STASH_SLICE_FEATS` into a DataFrame (rows = slice ids, columns = features)
  - can filter feature types (default: `('pos','deprel','ttr','sent')`)
  - can z-normalize columns for modeling

### Step 6 — Train classifiers and generate predictions

Goal: compare groups (usually Philosophy vs Literature within a period) using logistic regression over slice features.

Comparisons are defined in:

- `osp/constants.py` → `COMPARISONS`

Each entry is a pair of `(name, pandas_query)` strings applied to `data/metadata.csv`.

Key functions:

- `osp.features.get_balanced_cv_data(groups_train, ...)`
  - samples slice ids from the two groups (balanced)
  - returns a feature matrix with `_type` (CV vs Unseen) and `_target`
- `osp.classify.classify_then_predict_group(...)`
- `osp.classify.get_preds_feats(...)`
  - runs all comparisons and stashes results in `STASH_PREDS_FEATS`
- `osp.features.get_current_feat_weights(...)`
  - aggregates feature weights and produces `weight_z` used for coloring passages

### Step 7 — Explore in the Streamlit dashboard

The dashboard reads the corpus metadata, cached parses, features, and model outputs and provides:

- feature tables / comparisons
- predictions browsing
- highlighted passages and sentence visualizations

Entrypoint:

- `dashboard/app.py`

The dashboard imports `osp` and relies on the same `data/` + stashes.

---

## Gotchas / notes

### Slice numbering (`__00` vs `__01`)

There is an indexing mismatch between older notebooks and current slicing code:

- `osp.slices.iter_txt_slices()` currently starts slice numbers at `0` (`__00`, `__01`, ...)
- some older notebook code and helpers were 1-indexed (`__01`, `__02`, ...)

If you have “missing slice” issues (e.g. you can see slices in a stash but the dashboard can’t find them), this is the first thing to check: make sure all downstream steps (parse → feats → preds) are using the same slice-id convention.

### Stash location differences

Some notebooks construct `HashStash('osp_slices_1000')` without the repo’s `data/raw/stash/` pathing, which may place the stash under your global HashStash cache directory (often `~/.cache/hashstash`).

The library code (`osp/constants.py`) creates stashes under `data/raw/stash/` inside this repo.

If you run notebooks and later run the dashboard and “nothing shows up”, it’s often because the notebook wrote to one stash location while the dashboard is reading another.

---

## Quick “mental model” checklist

When something is missing, ask:

1. Does `data/metadata.csv` contain the doc id and a sane `year` / `discipline`?
2. Does `data/txt/<id>.txt` exist?
3. Do slices exist in `STASH_SLICES[text_id]`?
4. Do parsed slice docs exist in `STASH_SLICES_NLP[slice_id]`?
5. Do features exist in `STASH_SLICE_FEATS[slice_id]`?
6. Do predictions exist in `STASH_PREDS_FEATS` (via `osp.classify.get_preds_feats()`)?

If the answer breaks at step N, regenerate from there forward.

