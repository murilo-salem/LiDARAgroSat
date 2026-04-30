from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import subprocess
from typing import Any

from lidaragrosat.benchmarking import require_protocol_id
from lidaragrosat.config import ExperimentConfig
from lidaragrosat.utils.io import read_json, write_json


def _safe_git_hash(cwd: str | Path | None = None) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


@dataclass(slots=True)
class ExperimentManifest:
    experiment_name: str
    family: str
    stage1: str
    stage2: str
    labels: str
    protocol_id: str
    seed: int
    dataset: str
    split: dict[str, Any]
    stage1_config: dict[str, Any]
    refiner: str
    decode_params: dict[str, Any]
    commit_hash: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["protocol_id"] = require_protocol_id(payload["protocol_id"])
        payload["experiment_name"] = build_experiment_name(
            family=self.family,
            stage1=self.stage1,
            stage2=self.stage2,
            labels=self.labels,
            protocol_id=self.protocol_id,
            seed=self.seed,
        )
        return payload


def build_experiment_name(
    family: str,
    stage1: str,
    stage2: str,
    labels: str,
    protocol_id: str,
    seed: int,
) -> str:
    require_protocol_id(protocol_id)
    parts = [
        str(family).strip() or "unknown_family",
        str(stage1).strip() or "none",
        str(stage2).strip() or "none",
        str(labels).strip() or "field_center",
        str(protocol_id).strip(),
        f"seed{int(seed):02d}",
    ]
    return ".".join(parts)


def create_experiment_manifest(
    config: ExperimentConfig,
    split: dict[str, Any],
    decode_params: dict[str, Any] | None = None,
    stage1_config: dict[str, Any] | None = None,
    refiner: str | None = None,
    cwd: str | Path | None = None,
) -> ExperimentManifest:
    protocol_id = require_protocol_id(config.benchmark.protocol_id)
    family = config.benchmark.family or config.model.experiment_name or "experiment"
    stage1 = config.benchmark.stage1 or "none"
    stage2 = config.benchmark.stage2 or refiner or config.model.experiment_name or "none"
    labels = config.benchmark.label_policy or "field_center"
    seed = int(config.training.seed)
    return ExperimentManifest(
        experiment_name=build_experiment_name(
            family=family,
            stage1=stage1,
            stage2=stage2,
            labels=labels,
            protocol_id=protocol_id,
            seed=seed,
        ),
        family=family,
        stage1=stage1,
        stage2=stage2,
        labels=labels,
        protocol_id=protocol_id,
        seed=seed,
        dataset=config.benchmark.dataset_id,
        split=split,
        stage1_config=stage1_config or {},
        refiner=refiner or stage2,
        decode_params=decode_params or {},
        commit_hash=_safe_git_hash(cwd),
    )


def save_experiment_manifest(manifest: ExperimentManifest, path: str | Path) -> None:
    write_json(manifest.to_dict(), path)


def load_experiment_manifest(path: str | Path) -> ExperimentManifest:
    payload = read_json(path)
    payload["protocol_id"] = require_protocol_id(payload.get("protocol_id", ""))
    return ExperimentManifest(**payload)


def aggregate_manifests(run_dirs: list[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        manifest_path = Path(run_dir) / "experiment_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing experiment_manifest.json in {run_dir}")
        rows.append(load_experiment_manifest(manifest_path).to_dict())
    return rows
