# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Ordinary Style Philosophy** — an NLP research project analyzing stylistic differences between philosophy and literature journal articles (from JSTOR) using syntactic features and logistic regression classifiers. The pipeline processes ~1900–2025 texts in 25-year periods.

## Key Commands

```bash
# Install as editable package
pip install -e ".[dev]"

# Validate data files are in place
osp check

# Run the full pipeline (or specific steps)
osp pipeline
osp pipeline slice parse feats classify
osp pipeline parse --limit 100

# Run tests
pytest                    # all tests (needs stanza model downloaded)
pytest tests/test_text_processing.py  # fast, no stanza

# Export derived data for publication (no copyrighted text)
osp export --output data/release/

# Run the Streamlit dashboard
osp dashboard
```

## Architecture

### Data Pipeline (sequential, each step depends on prior)

Run via `python -m osp.pipeline` or individual steps:

1. **`assemble`** — Load JSTOR/PMLA data → `data/txt/<id>.txt` + `data/metadata.csv`
2. **`slice`** — Split texts into 1000-recognized-word chunks → cached in `STASH_SLICES`
3. **`parse`** — Run Stanza NLP (POS, deps, constituency) → cached in `STASH_SLICES_NLP`
4. **`feats`** — Extract per-slice feature dicts → cached in `STASH_SLICE_FEATS`
5. **`classify`** — Logistic regression comparing Philosophy vs Literature by period → cached in `STASH_PREDS_FEATS`

The Streamlit dashboard (`dashboard/app.py`) reads all caches for interactive exploration.

### Core Library (`osp/`)

- **`constants.py`** — All paths, stash definitions, feature exclusion lists, comparison definitions, and global config. The central configuration file.
- **`cli.py`** — Unified CLI entrypoint (`osp check`, `osp pipeline`, `osp export`, `osp dashboard`).
- **`pipeline.py`** — Data pipeline implementation.
- **`export.py`** — Exports derived data (features, predictions) for publication.
- **`check_data.py`** — Validates raw data files are in place.
- **`data_loaders.py`** — Loads JSTOR/PMLA data, builds corpus. Entry: `get_jstor_data()`, `get_corpus_txt(id)`.
- **`slices.py`** — Splits texts into fixed-length chunks. Entry: `get_text_slices(text_id)`.
- **`nlp_utils.py`** — Stanza pipeline management and clause extraction. Entry: `get_nlp()`, `get_nlp_doc()`.
- **`features.py`** — Extracts syntactic features (POS, deprel, TTR, sentence-level). Entry: `get_all_feats()`, `extract_slice_feats()`.
- **`classify.py`** — Logistic regression with cross-validation. Entry: `classify_data()`, `get_preds_feats()`.
- **`sentences.py`** / **`passages.py`** — HTML rendering of annotated text with feature weights.
- **`__init__.py`** — Re-exports all submodules. Notebooks use `from osp import *`.

### Import Structure

- **`constants.py`** imports only `os`, `sys`, and `hashstash` — no circular dependency.
- Submodules import explicitly from `constants` and standard library at top level.
- Cross-module function imports use lazy imports within functions to avoid circular deps.
- `__init__.py` re-exports everything for notebook convenience (`from osp import *`).

### Key Design Patterns

- **HashStash caching**: All expensive computations are cached in `data/raw/stash/` via `HashStash`. Stash names defined in `constants.py`. Deleting a stash requires regenerating all downstream artifacts.
- **Global mutable state**: `constants.NLP`, `constants.OK_WORDS` are module-level globals set at runtime.
- **Document IDs**: `phil/10.2307/NNNNN` or `lit/10.2307/NNNNN`. Slice IDs append `__NN` (0-indexed).

### Dashboard (`dashboard/`)

Streamlit app with pages in `dashboard/pages/`. Uses `dashboard/utils.py` for state management and `dashboard/components.py` for shared UI components.

## Known Gotchas

- **Stash location mismatch**: Notebooks using `HashStash('name')` without full path write to `~/.cache/hashstash`, while library code writes to `data/raw/stash/`. If the dashboard shows nothing, check this first.
- **Slice numbering**: Current code is 0-indexed (`__00`), some older notebooks were 1-indexed (`__01`). Mismatched indexing causes "missing slice" issues across pipeline stages.
- **Comparisons**: Defined in `constants.py` → `COMPARISONS` as pairs of `(name, pandas_query)` applied to metadata. Training defaults to `COMPARISONS[0]`.
- **LMDB/Python version**: HashStash uses LMDB under the hood. Stashes built with one Python major version (e.g. 3.11) may not be readable from another (e.g. 3.14). If you see `MDB_CORRUPTED` errors, rebuild stashes from scratch with a consistent Python version.
- **Data filenames**: Raw data filenames in `constants.py` can be overridden via environment variables (`OSP_FN_JSTOR_DATA`, `OSP_FN_JSTOR_METADATA`, `OSP_FN_PMLA`, `OSP_FN_JSTOR_DATA_OTHER`) if JSTOR delivers files with different names.

## Acknowledgments

The codebase cleanup for publication — including the `osp` CLI, test suite, import refactoring, data export tooling, CI/CD setup, and documentation — was done in collaboration with Claude Code (claude.ai/code).
