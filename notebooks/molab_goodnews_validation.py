"""marimo orchestration UI for the cloud-agnostic GoodNews validation CLI."""

import marimo


__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import subprocess
    import sys
    from pathlib import Path

    import marimo as mo

    return Path, json, mo, subprocess, sys


@app.cell
def _(mo):
    mo.md(
        """
        # GoodNews B0/B1/B1-random validation

        This notebook only orchestrates repository CLIs. If the repository or
        dependencies are not prepared yet, use a terminal:

        ```bash
        git clone https://github.com/phuongth05/HappyNewsYear.git MERGE
        cd MERGE
        python -m pip install -e ".[captioning,coco-eval,ner]"
        python -m spacy download en_core_web_sm
        ```

        B2 is not implemented or invoked here.
        """
    )
    return


@app.cell
def _(Path, mo):
    repo_path = mo.ui.text(value=str(Path.cwd()), label="Repository root")
    dataset_path = mo.ui.text(
        value=str(Path.cwd() / "GoodNews_validation_50"),
        label="Dataset root",
    )
    output_path = mo.ui.text(
        value=str(Path.cwd() / "runs" / "goodnews_validation_50"),
        label="Output root",
    )
    bundle_path = mo.ui.text(
        value=str(Path.cwd() / "goodnews_validation_50.zip"),
        label="Bundle path",
    )
    mo.vstack([repo_path, dataset_path, output_path, bundle_path])
    return bundle_path, dataset_path, output_path, repo_path


@app.cell
def _(mo):
    environment_check = mo.ui.run_button(label="Check CUDA environment")
    subset_check = mo.ui.run_button(label="Audit exact 50-sample subset")
    run_validation = mo.ui.run_button(label="Run B0/B1/B1-random validation")
    mo.hstack([environment_check, subset_check, run_validation])
    return environment_check, run_validation, subset_check


@app.cell
def _(Path, environment_check, mo, output_path, repo_path, subprocess, sys):
    environment_result = None
    if environment_check.value:
        repo = Path(repo_path.value).expanduser().resolve()
        output = Path(output_path.value).expanduser().resolve()
        command = [
            sys.executable,
            str(repo / "scripts" / "gpu_preflight.py"),
            "--output",
            str(output / "gpu_preflight.manual.json"),
        ]
        completed = subprocess.run(command, cwd=repo, capture_output=True, text=True)
        environment_result = mo.md(
            f"```text\n{completed.stdout}\n{completed.stderr}\n```"
        )
    environment_result
    return


@app.cell
def _(Path, dataset_path, mo, repo_path, subprocess, subset_check, sys):
    subset_result = None
    if subset_check.value:
        repo = Path(repo_path.value).expanduser().resolve()
        command = [
            sys.executable,
            str(repo / "scripts" / "audit_experiment_subset.py"),
            "--dataset-root",
            str(Path(dataset_path.value).expanduser().resolve()),
            "--split",
            "dev",
            "--max-samples",
            "50",
            "--selection",
            "first_by_sample_id",
        ]
        completed = subprocess.run(command, cwd=repo, capture_output=True, text=True)
        subset_result = mo.md(f"```text\n{completed.stdout}\n{completed.stderr}\n```")
    subset_result
    return


@app.cell
def _(
    Path,
    bundle_path,
    dataset_path,
    mo,
    output_path,
    repo_path,
    run_validation,
    subprocess,
    sys,
):
    validation_result = None
    if run_validation.value:
        repo = Path(repo_path.value).expanduser().resolve()
        command = [
            sys.executable,
            str(repo / "scripts" / "run_goodnews_validation.py"),
            "--dataset-root",
            str(Path(dataset_path.value).expanduser().resolve()),
            "--output-root",
            str(Path(output_path.value).expanduser().resolve()),
            "--bundle",
            str(Path(bundle_path.value).expanduser().resolve()),
        ]
        completed = subprocess.run(command, cwd=repo, capture_output=True, text=True)
        validation_result = mo.md(
            f"```text\n{completed.stdout}\n{completed.stderr}\n```"
        )
    validation_result
    return


@app.cell
def _(Path, bundle_path, json, mo, output_path):
    sanity_path = Path(output_path.value).expanduser().resolve() / "context_sanity.json"
    bundle = Path(bundle_path.value).expanduser().resolve()
    if sanity_path.is_file():
        inspection = mo.vstack(
            [
                mo.md("## Saved context sanity analysis"),
                mo.json(json.loads(sanity_path.read_text(encoding="utf-8"))),
                mo.md(f"Bundle: `{bundle}` — exists: `{bundle.is_file()}`"),
            ]
        )
    else:
        inspection = mo.md(
            f"No result yet. Expected context analysis at `{sanity_path}`."
        )
    inspection
    return


if __name__ == "__main__":
    app.run()
