"""Build the deterministic self-contained M4 claim-support annotation UI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kric.evaluation.io import file_sha256  # noqa: E402
from kric.matching.annotation_ui import build_annotation_html  # noqa: E402
from kric.matching.review_context import read_review_csv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    fieldnames, rows = read_review_csv(source)
    if len(rows) != 120:
        raise ValueError(f"M4 annotation UI requires exactly 120 rows, found {len(rows)}")
    html = build_annotation_html(fieldnames, rows, source_sha256=file_sha256(source))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(html, encoding="utf-8", newline="\n")
    temporary.replace(output)
    print(f"Wrote {output} with {len(rows)} embedded review rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
