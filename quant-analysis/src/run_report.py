"""Write the final assessment report from existing outputs."""

from __future__ import annotations

import sys
from pathlib import Path

from src.config import parse_path_args, resolve_paths
from src.report import build_report


def run_report(data_dir=None, output_dir=None) -> Path:
    if data_dir is None:
        paths = parse_path_args()
    else:
        paths = resolve_paths(data_dir, output_dir or "outputs")
    report_path = Path(paths.output_dir) / "report.pdf"
    built = build_report(paths.output_dir, report_path)
    print("Wrote", built, flush=True)
    return built


def main(argv: list[str] | None = None) -> int:
    paths = parse_path_args(argv)
    run_report(paths.data_dir, paths.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
