# Ordinary Style Philosophy

Code and data for the paper *Ordinary Style Philosophy*, which uses NLP and logistic regression to analyze stylistic differences between philosophy and literature journal articles from JSTOR (1900–2025).

## Replicating the dataset

The raw text data comes from JSTOR and cannot be redistributed. To replicate, request the corpus through [JSTOR's Text Analysis Support](https://support.jstor.org/hc/en-us/articles/32479181127575-JSTOR-Text-Analysis-Support-Getting-Started) using the item ID lists in `data/`:

| File | Count | Description |
|------|-------|-------------|
| `data/jstor_ids.txt` | 32,782 | Philosophy journal articles |
| `data/jstor_ids_nonphil.txt` | 32,276 | Non-philosophy, non-literature articles (control) |
| `data/jstor_ids_literature.txt` | 22,335 | Literature journal articles (PMLA, ELH, etc.) |

All files contain JSTOR item UUIDs (one per line). Submit them to JSTOR via the [dataset request form](https://www.jstor.org/ta-support/form).

**Note on literature corpus coverage:** The full literature corpus used in the paper contains 25,343 articles, but 3,008 of these could not be resolved to JSTOR UUIDs (they are not present in the JSTOR bibliographic metadata export). Their JSTOR stable IDs are listed in `data/jstor_ids_literature_missing_uuids.txt` for reference. A replication using the 22,335 available articles should produce substantively equivalent results.

### Placing the data files

Once you receive the data exports from JSTOR, place them in `data/raw/` with these names:

| Expected filename | Content | Source request |
|---|---|---|
| `jstor_data.jsonl.gz` | Philosophy articles (full text) | `jstor_ids.txt` |
| `jstor_metadata.jsonl.gz` | JSTOR bibliographic metadata (all of JSTOR) | Available from [JSTOR metadata page](https://jstor.org/ta-support/metadata) |
| `LitStudiesJSTOR.jsonl` | Literature articles (full text) | `jstor_ids_literature.txt` |
| `jstor_data_nonphil.jsonl.gz` | Non-philosophy articles (full text, optional) | `jstor_ids_nonphil.txt` |

If JSTOR delivers files with different names, either rename them or set environment variables:

```bash
export OSP_FN_JSTOR_DATA=/path/to/philosophy_export.jsonl.gz
export OSP_FN_JSTOR_METADATA=/path/to/metadata_2026-01-15.jsonl.gz
export OSP_FN_PMLA=/path/to/literature_export.jsonl
export OSP_FN_JSTOR_DATA_OTHER=/path/to/nonphil_export.jsonl.gz
```

Validate your setup:

```bash
osp check
```

You will also need `data/raw/worddb.byu.txt` (BYU word frequency database) for vocabulary filtering during slicing.

## Setup

```bash
pip install -e ".[dev]"        # install package + test dependencies
pip install -e ".[dashboard]"  # also install streamlit for the dashboard
```

## Running the pipeline

After `pip install -e .`, the `osp` command is available:

```bash
osp pipeline                     # run all steps
osp pipeline assemble            # run one step
osp pipeline parse --limit 100   # parse first 100 texts only
osp pipeline feats --num-proc 8  # parallelize feature extraction
```

### Pipeline steps

| Step | Command | What it does | Output |
|------|---------|-------------|--------|
| 1. **assemble** | `osp pipeline assemble` | Load JSTOR/PMLA exports, write text files + metadata | `data/txt/<id>.txt`, `data/metadata.csv` |
| 2. **slice** | `osp pipeline slice` | Split texts into 1000-recognized-word chunks | Cached in `STASH_SLICES` |
| 3. **parse** | `osp pipeline parse` | Run Stanza NLP (POS, deps, constituency) | Cached in `STASH_SLICES_NLP` |
| 4. **feats** | `osp pipeline feats` | Extract per-slice syntactic features | Cached in `STASH_SLICE_FEATS` |
| 5. **classify** | `osp pipeline classify` | Train logistic regression classifiers by period | Cached in `STASH_PREDS_FEATS` |

Each step depends on the prior. If a stash is deleted, regenerate from that step forward.

All subcommands also work as `python -m osp.pipeline`, `python -m osp.export`, etc.

## Exploring results

### Streamlit dashboard

```bash
osp dashboard
```

Interactive explorer for features, predictions, and annotated passages.

### Exporting derived data

```bash
osp export --output data/release/
```

Exports publication-safe derived data (no copyrighted text): metadata, feature matrices, predictions, and feature weights.

Pre-computed derived data is also available from [GitHub Releases](../../releases) and archived on Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19482402.svg)](https://doi.org/10.5281/zenodo.19482402)

## Tests

```bash
pytest                                # all tests (~7s, requires stanza model)
pytest tests/test_text_processing.py  # fast subset, no stanza needed
pytest --cov=osp                      # with coverage report
```

## Repo layout

```
osp/                  Core library
  constants.py          Paths, stash definitions, feature config
  pipeline.py           CLI for running the data pipeline
  export.py             Export derived data for publication
  check_data.py         Validate raw data files are in place
  data_loaders.py       Load JSTOR/PMLA data, build corpus
  slices.py             Split texts into fixed-length chunks
  nlp_utils.py          Stanza pipeline, clause extraction
  features.py           Extract syntactic features (POS, deprel, TTR, sentence-level)
  classify.py           Logistic regression with cross-validation
  sentences.py          Sentence-level HTML rendering and analysis
  passages.py           Passage-level HTML rendering
  examples.py           Feature example extraction
  statistics.py         Statistical tests and LaTeX table generation
  word_freqs.py         Word frequency tracking

dashboard/            Streamlit app
  app.py                Navigation and page routing
  utils.py              State management, caching helpers
  pages/                Individual page modules

notebooks/            Jupyter notebooks
  AssembleCorpus.ipynb          Pipeline step 1 (worked example)
  CorpusSlices.ipynb            Pipeline step 2
  ParseSlice.ipynb              Pipeline step 3
  GenSliceFeats.ipynb           Pipeline step 4
  ClassifySliceFeats.ipynb      Pipeline step 5
  DescStatsForPaper3.ipynb      Corpus and feature tables for paper
  PredStatsForPaper.ipynb       Classification accuracy tables for paper
  ProbsAnalyze4.ipynb           Prediction figures for paper
  SliceFeatsDiffHist2.ipynb     Feature difference tables for paper
  archive/                      Exploratory notebooks (not needed to reproduce)

data/
  metadata.csv                          Corpus metadata (id, title, author, year, journal, discipline)
  jstor_ids.txt                         Philosophy JSTOR item IDs (UUIDs)
  jstor_ids_nonphil.txt                 Non-philosophy JSTOR item IDs (UUIDs)
  jstor_ids_literature.txt              Literature JSTOR item IDs (UUIDs)
  jstor_ids_literature_missing_uuids.txt  Literature articles without UUIDs (stable IDs)
  raw/                                  Raw inputs and HashStash caches (gitignored)

tests/                Test suite
figures/              Generated figures
```

## Data conventions

- **Document IDs**: `phil/10.2307/40231690` or `lit/461288`. The ID doubles as the path under `data/txt/`.
- **Slice IDs**: `<text_id>__NN` (0-indexed), e.g. `phil/10.2307/40231690__03`.
- **Comparisons**: Defined in `osp/constants.py` as `COMPARISONS` — pairs of `(name, pandas_query)` applied to `data/metadata.csv`, comparing Philosophy vs Literature in 25-year periods.
- **Stashes**: HashStash caches in `data/raw/stash/`. If a notebook writes `HashStash('name')` without the full path, it may go to `~/.cache/hashstash` instead — this is the most common cause of "nothing shows up" issues.

## License

GPL-3.0
