from __future__ import annotations

from dataclasses import asdict, dataclass

from lidaragrosat.data.splits import FoldSplit

PROTOCOL_A_ID = "protocol_a_v2_3split"
PROTOCOL_B_ID = "protocol_b_v2_faithful10"
PROTOCOL_NEON_BASIC_ID = "protocol_neon_tree_evaluation_basic_v1"


@dataclass(slots=True)
class BenchmarkProtocol:
    protocol_id: str
    name: str
    description: str
    metrics: tuple[str, ...]
    baselines: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def get_benchmark_protocol(protocol_id: str) -> BenchmarkProtocol:
    protocols = {
        PROTOCOL_A_ID: BenchmarkProtocol(
            protocol_id=PROTOCOL_A_ID,
            name="Comparable 3-split suite",
            description="Three fixed plot-level train/val/test splits for learned methods and official ablations.",
            metrics=("precision", "recall", "f1", "location_mae_m", "height_mae_m", "tp", "fp", "fn"),
            baselines=("baseline_local_maxima", "chm_watershed", "hybrid_proposal_refinement_multimodal_v2"),
        ),
        PROTOCOL_B_ID: BenchmarkProtocol(
            protocol_id=PROTOCOL_B_ID,
            name="Faithful 10-plot benchmark",
            description="Full 10-plot faithful benchmark for non-trained methods or local lightweight training.",
            metrics=("precision", "recall", "f1", "location_mae_m", "height_mae_m", "tp", "fp", "fn"),
            baselines=("baseline_local_maxima", "chm_watershed", "hybrid_proposal_refinement_multimodal_v2"),
        ),
        PROTOCOL_NEON_BASIC_ID: BenchmarkProtocol(
            protocol_id=PROTOCOL_NEON_BASIC_ID,
            name="NEON practical basic split",
            description="Single reproducible train/val/test split for NeonTreeEvaluation imported as LiDARAgroSat scenes.",
            metrics=("precision", "recall", "f1", "location_mae_m", "height_mae_m", "tp", "fp", "fn"),
            baselines=("baseline_local_maxima", "lidar_only", "multimodal_mid_fusion"),
        ),
    }
    try:
        return protocols[protocol_id]
    except KeyError as exc:
        raise ValueError(f"Unknown protocol_id: {protocol_id}") from exc


def require_protocol_id(protocol_id: str) -> str:
    cleaned = str(protocol_id or "").strip()
    if not cleaned:
        raise ValueError("protocol_id is required by benchmark_protocol_v2.")
    get_benchmark_protocol(cleaned)
    return cleaned


def build_protocol_a_splits(plot_ids: list[str]) -> list[FoldSplit]:
    ordered = sorted(plot_ids)
    expected = {f"plot_{index:02d}" for index in range(1, 11)}
    if set(ordered) >= expected:
        return [
            FoldSplit(
                fold_id=0,
                train_plots=["plot_04", "plot_01", "plot_08", "plot_09", "plot_07", "plot_02", "plot_06", "plot_03"],
                val_plots=["plot_10"],
                test_plots=["plot_05"],
            ),
            FoldSplit(
                fold_id=1,
                train_plots=["plot_04", "plot_01", "plot_08", "plot_09", "plot_05", "plot_02", "plot_06", "plot_10"],
                val_plots=["plot_07"],
                test_plots=["plot_03"],
            ),
            FoldSplit(
                fold_id=2,
                train_plots=["plot_04", "plot_01", "plot_08", "plot_07", "plot_05", "plot_02", "plot_06", "plot_03"],
                val_plots=["plot_10"],
                test_plots=["plot_09"],
            ),
        ]
    if len(ordered) < 3:
        raise ValueError("Protocol A requires at least three plots.")
    return [
        FoldSplit(
            fold_id=0,
            train_plots=ordered[:-2],
            val_plots=[ordered[-2]],
            test_plots=[ordered[-1]],
        ),
        FoldSplit(
            fold_id=1,
            train_plots=[plot_id for plot_id in ordered if plot_id not in {ordered[1], ordered[-1]}],
            val_plots=[ordered[1]],
            test_plots=[ordered[-1]],
        ),
        FoldSplit(
            fold_id=2,
            train_plots=[plot_id for plot_id in ordered if plot_id not in {ordered[0], ordered[-2]}],
            val_plots=[ordered[0]],
            test_plots=[ordered[-2]],
        ),
    ]
