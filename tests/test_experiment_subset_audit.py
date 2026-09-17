import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.audit_experiment_subset import audit_subset


def _release(root: Path) -> None:
    images = root / "images"
    images.mkdir()
    # Deliberately use non-sorted insertion order; experiment selection sorts IDs.
    annotations = {
        "story-z": {"article": "Z article", "images": {"0": "Z caption"}},
        "story-a": {"article": "A article", "images": {"0": "A caption"}},
        "story-m": {"article": "M article", "images": {"0": "M caption"}},
    }
    splits = {
        "story-z_0.jpg": "val",
        "story-a_0.jpg": "val",
        "story-m_0.jpg": "val",
    }
    (root / "article+caption.json").write_text(json.dumps(annotations), encoding="utf-8")
    (root / "img_splits.json").write_text(json.dumps(splits), encoding="utf-8")
    for sample_id in ("story-z_0", "story-a_0", "story-m_0"):
        Image.new("RGB", (2, 2), "white").save(images / f"{sample_id}.jpg", "JPEG")


def test_subset_audit_uses_experiment_selection_and_print_order(tmp_path: Path) -> None:
    _release(tmp_path)
    report = audit_subset(
        tmp_path, split="dev", max_samples=2, selection="first_by_sample_id"
    )
    assert report["valid"] is True
    assert report["sample_ids"] == ["story-a_0", "story-m_0"]


def test_subset_audit_fails_on_corrupt_selected_image(tmp_path: Path) -> None:
    _release(tmp_path)
    (tmp_path / "images" / "story-a_0.jpg").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="story-a_0: corrupt image"):
        audit_subset(
            tmp_path, split="dev", max_samples=2, selection="first_by_sample_id"
        )
