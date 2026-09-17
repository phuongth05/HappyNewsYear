import json
import zipfile
from pathlib import Path

import pytest

from scripts.gpu_preflight import GIB, collect_gpu_preflight
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
    total_memory = 16 * GIB


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
        return 13 * GIB, 16 * GIB

    @staticmethod
    def get_device_name(index):
        assert index == 0
        return "Generic CUDA GPU"


class _FakeTorch:
    __version__ = "test"
    cuda = _FakeCuda()
    version = type("Version", (), {"cuda": "12.1"})()


def test_gpu_preflight_reports_required_fields() -> None:
    report = collect_gpu_preflight(
        _FakeTorch(),
        minimum_total_vram_bytes=14 * GIB,
        minimum_free_vram_bytes=12 * GIB,
    )
    assert report == {
        "torch_version": "test",
        "cuda_version": "12.1",
        "cuda_available": True,
        "minimum_total_vram_bytes": 14 * GIB,
        "minimum_free_vram_bytes": 12 * GIB,
        "gpu_model": "Generic CUDA GPU",
        "total_vram_bytes": 16 * GIB,
        "free_vram_bytes": 13 * GIB,
        "total_vram_gib": 16.0,
        "free_vram_gib": 13.0,
        "reported_device_total_memory_bytes": 16 * GIB,
        "sufficient_vram": True,
    }


def test_gpu_preflight_rejects_capacity_without_checking_model_name() -> None:
    report = collect_gpu_preflight(
        _FakeTorch(),
        minimum_total_vram_bytes=20 * GIB,
        minimum_free_vram_bytes=12 * GIB,
    )
    assert report["gpu_model"] == "Generic CUDA GPU"
    assert report["sufficient_vram"] is False


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


def test_all_three_conditions_accept_the_same_ordered_ids(tmp_path: Path) -> None:
    expected = ["a", "b"]
    for condition in ("b0", "b1", "b1_random"):
        directory = tmp_path / condition
        directory.mkdir()
        path = directory / "predictions.jsonl"
        path.write_text(
            "".join(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "prediction": condition,
                        "reference": "reference",
                        "metadata": {},
                    }
                )
                + "\n"
                for sample_id in expected
            ),
            encoding="utf-8",
        )
        assert_prediction_ids(path, expected)


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
    result = create_output_bundle(output, bundle)
    assert result["files"] == 1
    with zipfile.ZipFile(bundle) as archive:
        assert archive.namelist() == [
            "goodnews_validation_50/context_sanity.json"
        ]


def test_core_workflow_has_no_kaggle_path_dependency() -> None:
    for path in (
        Path("scripts/run_goodnews_validation.py"),
        Path("scripts/gpu_preflight.py"),
    ):
        source = path.read_text(encoding="utf-8").lower()
        assert "/kaggle" not in source
        assert "tesla t4" not in source
