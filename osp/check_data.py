"""
Validate that raw data files are in place before running the pipeline.

Usage:
    python -m osp.check_data
"""
import os
import sys


def check():
    from .constants import (
        PATH_DATA, PATH_DATA_RAW, PATH_METADATA,
        FN_JSTOR, FN_JSTOR_DATA, FN_JSTOR_DATA_OTHER, FN_PMLA,
        PATH_WORDDB, PATH_WORD2POS,
    )

    ok = True
    warnings = []

    def check_file(path, label, required=True):
        nonlocal ok
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        size_str = f"{size / 1e6:.1f} MB" if size > 1e6 else f"{size / 1e3:.1f} KB" if size > 1e3 else f"{size} B"

        if exists:
            print(f"  OK   {label}")
            print(f"        -> {path} ({size_str})")
        elif required:
            print(f"  MISS {label}")
            print(f"        -> expected at: {path}")
            ok = False
        else:
            warnings.append(f"  SKIP {label} (optional, not found)")
            print(f"  SKIP {label} (optional)")
            print(f"        -> expected at: {path}")

    print("Checking raw data files...\n")

    check_file(FN_JSTOR_DATA, "Philosophy full-text export (jstor_data.jsonl.gz)")
    check_file(FN_JSTOR, "JSTOR bibliographic metadata (jstor_metadata.jsonl.gz)")
    check_file(FN_PMLA, "Literature full-text export (LitStudiesJSTOR.jsonl)")
    check_file(FN_JSTOR_DATA_OTHER, "Non-philosophy full-text export (jstor_data_nonphil.jsonl.gz)", required=False)
    has_worddb = os.path.exists(PATH_WORDDB)
    has_word2pos = os.path.exists(PATH_WORD2POS)
    if has_worddb:
        check_file(PATH_WORDDB, "Word database (worddb.byu.txt)")
    elif has_word2pos:
        check_file(PATH_WORD2POS, "Word-to-POS mapping (word2pos.json)")
    else:
        check_file(PATH_WORDDB, "Word database (worddb.byu.txt or word2pos.json)")

    print()
    check_file(PATH_METADATA, "Corpus metadata (metadata.csv)", required=False)

    print()
    if ok:
        print("All required files found. Ready to run: python -m osp.pipeline")
    else:
        print("Missing required files. See above for details.")
        print()
        print("JSTOR data files should be placed in:")
        print(f"  {PATH_DATA_RAW}/")
        print()
        print("Expected filenames (override via environment variables):")
        print(f"  jstor_data.jsonl.gz          (or set OSP_FN_JSTOR_DATA)")
        print(f"  jstor_metadata.jsonl.gz      (or set OSP_FN_JSTOR_METADATA)")
        print(f"  LitStudiesJSTOR.jsonl        (or set OSP_FN_PMLA)")
        print(f"  jstor_data_nonphil.jsonl.gz  (or set OSP_FN_JSTOR_DATA_OTHER)")
        print()
        print("Request data from JSTOR using the ID lists in data/:")
        print("  https://support.jstor.org/hc/en-us/articles/32479181127575")

    return ok


def main():
    success = check()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
