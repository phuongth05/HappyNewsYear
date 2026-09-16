from pathlib import Path

import pytest

from kric.captioning.config import B0ExperimentConfig


def _config_text(*, experiment: str = "B0_image_only", do_sample: bool = False) -> str:
    sampling = "true" if do_sample else "false"
    return f"""
experiment: {experiment}
seed: 7
dataset:
  name: goodnews
  config: goodnews.yaml
  split: dev
  max_samples: 2
model:
  type: blip
  name: example/model
  revision: immutable-sha
generation:
  do_sample: {sampling}
output_dir: run
evaluation:
  enabled: false
"""


def test_b0_config_resolves_paths_and_requires_experiment_name(tmp_path: Path) -> None:
    path = tmp_path / "b0.yaml"
    path.write_text(_config_text(), encoding="utf-8")
    config = B0ExperimentConfig.from_file(path)
    assert config.experiment == "B0_image_only"
    assert config.dataset.config_path == tmp_path / "goodnews.yaml"
    assert config.output_dir == tmp_path / "run"

    path.write_text(_config_text(experiment="B1_full_article"), encoding="utf-8")
    with pytest.raises(ValueError, match="B0 runner"):
        B0ExperimentConfig.from_file(path)


def test_b0_rejects_sampling(tmp_path: Path) -> None:
    path = tmp_path / "b0.yaml"
    path.write_text(_config_text(do_sample=True), encoding="utf-8")
    with pytest.raises(ValueError, match="do_sample"):
        B0ExperimentConfig.from_file(path)

