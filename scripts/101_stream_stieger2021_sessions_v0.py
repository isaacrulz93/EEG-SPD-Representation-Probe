#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
from src.stieger2021_multiclass_confirmation_v0 import _source_files_from_committed_manifest, load_config
from src.stieger2021_streaming_preprocessing_v0 import process_source_file, validate_streamed_records

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume-safe one-file Stieger2021 session streaming")
    parser.add_argument("--subject", type=int, action="append", help="technical resume selector; does not alter locked cohort")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config, _ = load_config(root)
    output = root / config["project"]["output_dir"]
    cache = root / config["project"]["cache_dir"]
    files = _source_files_from_committed_manifest(output / "source_manifest" / "official_selected_files.json")
    selected = [item for item in files if not args.subject or item.subject in set(args.subject)]
    for position, source in enumerate(selected, 1):
        result = process_source_file(source, config, cache)
        print(json.dumps({"progress": f"{position}/{len(selected)}", **result}, sort_keys=True), flush=True)
    if not args.subject:
        print(json.dumps(validate_streamed_records(files, cache, config), indent=2, sort_keys=True))
