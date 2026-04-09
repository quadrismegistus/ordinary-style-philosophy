"""
CLI entrypoint for running the OSP data pipeline.

Usage:
    python -m osp.pipeline                # run all steps
    python -m osp.pipeline assemble       # run one step
    python -m osp.pipeline parse --limit 100
"""
import argparse
import os
import sys

import pandas as pd
from tqdm import tqdm


def step_assemble(args):
    """Step 1: Load JSTOR + PMLA data, write text files and metadata.csv."""
    from .data_loaders import get_jstor_data, get_pmla_df
    from .constants import PATH_DATA, PATH_METADATA

    path_txt = os.path.join(PATH_DATA, "raw", "txt")

    print("Loading JSTOR data...")
    df_jstor = get_jstor_data().fillna("")

    print("Loading PMLA data...")
    df_pmla = get_pmla_df().fillna("")

    def format_url(url):
        x = url.split(".")
        return ".".join(x[1:])

    def jstor2row(row):
        uuid = row["id"]
        id = "phil/" + row["url"].split("/stable/")[-1]
        return {
            "id": id,
            "uuid": uuid,
            "title": row["title"],
            "author": row["creators_string"],
            "year": int(row["published_date"].split("-")[0]),
            "journal": row["is_part_of"],
            "volume": row["issue_volume"],
            "issue": row["issue_number"],
            "url": format_url(row["url"]),
            "publisher": "; ".join(row["publishers"]),
            "discipline": "Philosophy",
        }

    def pmla2row(row):
        ids = row["identifier"]
        uuid = [d["value"] for d in ids if d["name"] == "local_uuid"]
        uuid = uuid[0] if uuid else None
        id = "lit/" + row["url"].split("/stable/")[-1]
        return {
            "id": id,
            "uuid": uuid,
            "title": row["title"],
            "author": ";".join(row["creator"]),
            "year": int(row["datePublished"].split("-")[0]),
            "journal": row["isPartOf"],
            "volume": row["volumeNumber"],
            "issue": row["issueNumber"],
            "url": format_url(row["url"]),
            "publisher": row["publisher"],
            "discipline": "Literature",
        }

    # Build metadata
    ld = []
    for _, row in df_jstor.iterrows():
        ld.append(jstor2row(row))
    for _, row in df_pmla.iterrows():
        ld.append(pmla2row(row))
    df = pd.DataFrame(ld)
    print(f"  {len(df)} documents total")

    # Write text files
    print("Writing text files...")
    for _, row in tqdm(df_jstor.iterrows(), total=len(df_jstor), desc="JSTOR texts"):
        d = jstor2row(row)
        fn = os.path.join(path_txt, d["id"] + ".txt")
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        with open(fn, "w") as f:
            f.write("\n\n\n".join(row["full_text"]))

    for _, row in tqdm(df_pmla.iterrows(), total=len(df_pmla), desc="PMLA texts"):
        d = pmla2row(row)
        fn = os.path.join(path_txt, d["id"] + ".txt")
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        with open(fn, "w") as f:
            f.write("\n\n\n".join(row["fullText"]))

    # Write metadata
    df.set_index("id").to_csv(PATH_METADATA)
    print(f"  Wrote {PATH_METADATA}")


def step_slice(args):
    """Step 2: Slice each text into fixed-length chunks."""
    from .data_loaders import get_corpus_metadata
    from .slices import get_text_slices

    df_meta = get_corpus_metadata()
    ids = df_meta.index.tolist()
    if args.limit:
        ids = ids[: args.limit]

    print(f"Slicing {len(ids)} texts...")
    for id in tqdm(ids):
        get_text_slices(id)


def step_parse(args):
    """Step 3: Parse slices with Stanza NLP."""
    import stanza
    from .constants import STASH_SLICES, STASH_SLICES_NLP
    from .data_loaders import get_corpus_metadata
    from .slices import get_text_slices

    nlp = None

    def get_nlp():
        nonlocal nlp
        if nlp is None:
            nlp = stanza.Pipeline(
                lang="en",
                processors="tokenize,mwt,pos,lemma,ner,depparse,constituency",
                verbose=False,
            )
        return nlp

    df_meta = get_corpus_metadata()
    ids = df_meta.index.tolist()
    if args.limit:
        ids = ids[: args.limit]

    # Collect all (slice_key, slice_txt) pairs not yet parsed
    to_parse = []
    for text_id in tqdm(ids, desc="Collecting slices"):
        slice_d = get_text_slices(text_id)
        for slice_id, slice_txt in slice_d.items():
            key = f"{text_id}__{int(slice_id):02d}"
            if key not in STASH_SLICES_NLP:
                to_parse.append((key, slice_txt))

    print(f"  {len(to_parse)} slices to parse")
    if not to_parse:
        return

    for key, txt in tqdm(to_parse, desc="Parsing"):
        doc = get_nlp()(txt)
        STASH_SLICES_NLP[key] = doc.to_serialized()


def step_feats(args):
    """Step 4: Extract slice-level features from parsed docs."""
    from .features import gen_all_slice_feats, get_parsed_slice_ids

    num_proc = getattr(args, "num_proc", 1) or 1
    get_parsed_slice_ids(_force=True)
    print("Extracting features...")
    gen_all_slice_feats(num_proc=num_proc)


def step_classify(args):
    """Step 5: Train classifiers and generate predictions."""
    from .constants import COMPARISONS, NORMALIZE_CLASSIFY_DATA
    from .classify import classify_then_predict_comparisons

    num_runs = getattr(args, "num_runs", 10) or 10
    sample_size = getattr(args, "sample_size", 1000) or 1000

    print(f"Running {len(COMPARISONS)} comparisons ({num_runs} runs, {sample_size} samples)...")
    df_preds, df_feats = classify_then_predict_comparisons(
        COMPARISONS,
        also_predict_unseen=True,
        num_runs=num_runs,
        sample_size=sample_size,
        feat_n=25,
        feat_n_egs=10,
        normalize=NORMALIZE_CLASSIFY_DATA,
    )
    print(f"  {len(df_preds)} predictions, {len(df_feats)} feature rows")

    # Print summary accuracy per comparison
    acc = df_preds.groupby("comparison")["correct"].mean()
    for cmp, a in acc.items():
        print(f"  {cmp}: {a * 100:.1f}%")


STEPS = {
    "assemble": step_assemble,
    "slice": step_slice,
    "parse": step_parse,
    "feats": step_feats,
    "classify": step_classify,
}


def main():
    parser = argparse.ArgumentParser(
        description="Run the Ordinary Style Philosophy data pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps (in order):
  assemble   Load JSTOR/PMLA data, write text files + metadata.csv
  slice      Split texts into fixed-length chunks
  parse      Run Stanza NLP on slices
  feats      Extract per-slice features
  classify   Train logistic regression classifiers

Examples:
  python -m osp.pipeline                    # run all steps
  python -m osp.pipeline parse feats        # run specific steps
  python -m osp.pipeline parse --limit 100  # parse first 100 texts only
""",
    )
    parser.add_argument(
        "steps",
        nargs="*",
        choices=list(STEPS.keys()),
        default=list(STEPS.keys()),
        help="Pipeline steps to run (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of texts to process (for testing)",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=1,
        help="Number of parallel processes for feature extraction",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="Number of classification runs (default: 10)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Sample size per group for classification (default: 1000)",
    )

    args = parser.parse_args()

    for step_name in args.steps:
        print(f"\n{'='*60}")
        print(f"Step: {step_name}")
        print(f"{'='*60}")
        STEPS[step_name](args)

    print("\nDone.")


if __name__ == "__main__":
    main()
