#!/usr/bin/env python3
"""Re-render Conditional Geometry v1 figures with separated top legends.

This is a presentation-only erratum applied after the locked scientific
analysis.  It delegates every validation, source-table construction, verdict,
and report operation to the frozen stage-27 implementation.  The sole change
is a deterministic vertical offset for figure-level legends placed at
``upper center``.  The command refuses to finish unless every figure source
CSV and the Markdown report remain byte-identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reporting_conditional_v1 import (  # noqa: E402
    FIGURE_STEMS,
    create_reporting_outputs,
    load_and_validate_reporting_inputs,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/bnci2014_001_conditional_geometry_v1.yaml"
LEGEND_Y = 1.14


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_hashes(paths: list[Path], root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(paths)
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".json", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    inputs = load_and_validate_reporting_inputs(args.config, root)
    figures_dir = inputs.output_root / "figures"
    report_path = inputs.output_root / str(inputs.config["outputs"]["report_file"])
    protected_paths = [report_path] + [
        figures_dir / f"{stem}.csv" for stem in FIGURE_STEMS
    ]
    before = _stable_hashes(protected_paths, root)

    original_legend = Figure.legend

    def separated_top_legend(self: Figure, *legend_args: Any, **legend_kwargs: Any):
        if (
            legend_kwargs.get("loc") == "upper center"
            and "bbox_to_anchor" not in legend_kwargs
        ):
            legend_kwargs["bbox_to_anchor"] = (0.5, LEGEND_Y)
        return original_legend(self, *legend_args, **legend_kwargs)

    Figure.legend = separated_top_legend
    try:
        artifacts = create_reporting_outputs(inputs)
    finally:
        Figure.legend = original_legend

    after = _stable_hashes(protected_paths, root)
    if before != after:
        changed = sorted(path for path in before if before[path] != after.get(path))
        raise RuntimeError(
            "Presentation erratum changed scientific/report source content: "
            f"{changed}"
        )

    media_paths = sorted(
        path
        for stem in FIGURE_STEMS
        for path in (figures_dir / f"{stem}.png", figures_dir / f"{stem}.pdf")
    )
    decision_path = inputs.output_root / "confirmatory_decision.json"
    provenance = {
        "schema_version": "1.0",
        "purpose": "post_confirmatory_presentation_erratum",
        "scientific_analysis_changed": False,
        "scientific_sources_and_report_byte_identical": True,
        "layout_change": {
            "scope": "figure-level legends with loc=upper center",
            "bbox_to_anchor": [0.5, LEGEND_Y],
            "reason": "prevent suptitle/legend overlap",
        },
        "frozen_analysis_code_commit": str(inputs.unlock["code_commit"]),
        "locked_discovery_head": str(inputs.unlock["locked_head"]),
        "protocol_sha256": inputs.protocol_sha256,
        "config_sha256": inputs.config_sha256,
        "terminal_decision": artifacts.verdicts.terminal_decision,
        "le_robustness_label": artifacts.verdicts.le_robustness_label,
        "renderer": {
            "path": Path(__file__).resolve().relative_to(root).as_posix(),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "protected_content_sha256": after,
        "decision_sha256": _sha256(decision_path),
        "rendered_media_sha256": _stable_hashes(media_paths, root),
    }
    provenance_path = inputs.output_root / "presentation_rendering_provenance.json"
    _atomic_json(provenance_path, provenance)
    print(
        json.dumps(
            {
                "status": "PASS",
                "terminal_decision": artifacts.verdicts.terminal_decision,
                "media_files": len(media_paths),
                "protected_files_unchanged": len(after),
                "provenance": str(provenance_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
