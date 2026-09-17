import json
import zipfile
from pathlib import Path

import pytest

from scripts.gpu_preflight import collect_gpu_preflight
from scripts.run_goodnews_validation import (
    MODEL_NAME,
    MODEL_REVISION,
    assert_prediction_ids,
    create_output_bundle,
    create_runtime_configs,
    load_validated_ids,
    validate_scientific_controls,
)


class _FakeProperties:
    total_memory = 16_000


class _FakeCuda:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def get_device_properties(index):
        assert index == 0
        return _FakeProperties()

    @staticmethod
    def mem_get_info(index):
        assert index == 0
        return 12_000, 16_000

    @staticmethod
    def get_device_name(index):
        assert index == 0
        return "Tesla T4"


class _FakeTorch:
    __version__ = "test"
    cuda = _FakeCuda()
    version = type("Version", (), {"cuda": "12.1"})()


def test_gpu_preflight_reports_required_fields() -> None:
    report = collect_gpu_preflight(_FakeTorch())
    assert report == {
        "torch_version": "test",
        "cuda_version": "12.1",
        "cuda_available": True,
        "gpu_model": "Tesla T4",
        "total_vram_bytes": 16_000,
        "free_vram_bytes": 12_000,
        "reported_device_total_memory_bytes": 16_000,
    }


def test_runtime_configs_preserve_controlled_comparison(tmp_path: Path) -> None:
    paths = create_runtime_configs(tmp_path / "GoodNews", tmp_path / "outputs")
    report = validate_scientific_controls(paths)
    assert report["valid"] is True
    assert report["model"] == MODEL_NAME
    assert report["revision"] == MODEL_REVISION
    b1_raw = paths["b1"].read_text(encoding="utf-8")
    assert str(tmp_path / "GoodNews") in (tmp_path / "outputs" / "workflow_configs" / "goodnews.runtime.yaml").read_text(encoding="utf-8")
    assert str(paths["b0"]) in b1_raw


def test_prediction_ids_must_match_validated_order(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    rows = [
        {"sample_id": sample_id, "prediction": "p", "reference": "r", "metadata": {}}
        for sample_id in ("a", "b")
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert_prediction_ids(path, ["a", "b"])
    with pytest.raises(RuntimeError, match="differ from validated manifest"):
        assert_prediction_ids(path, ["b", "a"])


def test_validated_id_manifest_has_exactly_fifty_unique_ids() -> None:
    path = Path("configs/dataset/goodnews_validation_50_ids.json")
    values = load_validated_ids(path)
    assert len(values) == len(set(values)) == 50


def test_bundle_contains_only_output_tree(tmp_path: Path) -> None:
    output = tmp_path / "goodnews_validation_50"
    output.mkdir()
    (output / "context_sanity.json").write_text("{}", encoding="utf-8")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "image.jpg").write_bytes(b"not an experiment output")
    bundle = tmp_path / "bundle.zip"
    result = create_output_bundle(output, bundle, overwrite=False)
    assert result["files"] == 1
    with zipfile.ZipFile(bundle) as archive:
        assert archive.namelist() == [
            "goodnews_validation_50/context_sanity.json"
        ]
