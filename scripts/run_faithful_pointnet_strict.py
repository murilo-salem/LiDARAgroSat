from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidaragrosat.paper_faithful import FaithfulMatchingParams, faithful_greedy_match, summarize_faithful_matches  # noqa: E402
from lidaragrosat.paper_faithful.strict_pointnet import (  # noqa: E402
    CombinedPointNetLoss,
    LocalPointNet,
    StrictPointNetConfig,
    StrictProposalDataset,
    build_labeled_proposal_table,
    predict_and_refine,
    run_pointnet_epoch,
    score_pointnet_dataset,
    summarize_proposals_against_reference,
    tune_pointnet_postprocess,
)
from lidaragrosat.utils.io import ensure_dir, write_json, write_text  # noqa: E402


VAL_PLOTS = ["plot_01", "plot_06"]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _load_points_and_reference(cache_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, pd.DataFrame]]:
    points_by_plot: dict[str, np.ndarray] = {}
    reference_by_plot: dict[str, pd.DataFrame] = {}
    for npz_path in sorted(cache_dir.glob("plot_*.npz")):
        plot_id = npz_path.stem
        points_by_plot[plot_id] = np.load(npz_path)["points_xyz"].astype(np.float32)
        reference_by_plot[plot_id] = pd.read_csv(cache_dir / f"{plot_id}_reference.csv")
    return points_by_plot, reference_by_plot


def _build_all_proposals(
    plot_ids: list[str],
    points_by_plot: dict[str, np.ndarray],
    reference_by_plot: dict[str, pd.DataFrame],
    config: StrictPointNetConfig,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows: list[pd.DataFrame] = []
    by_plot: dict[str, pd.DataFrame] = {}
    for plot_id in plot_ids:
        table = build_labeled_proposal_table(
            plot_id=plot_id,
            points_xyz=points_by_plot[plot_id],
            reference=reference_by_plot[plot_id],
            config=config,
        )
        rows.append(table)
        by_plot[plot_id] = table[["x", "y", "height_pred", "confidence"]].copy()
    return pd.concat(rows, ignore_index=True, sort=False), by_plot


def _baseline_comparison(
    plot_ids: list[str],
    destination: Path,
) -> pd.DataFrame:
    sources = {
        "local_maxima_fixed": Path("outputs_faithful_paper_optuna/best_per_plot.csv"),
        "local_maxima_adaptive": Path("outputs_faithful_paper_adaptive_optuna/best_per_plot.csv"),
        "chm_watershed": Path("outputs_faithful_paper_watershed_optuna/best_per_plot.csv"),
        "rgb_lidar_object_fusion": Path("outputs_faithful_paper_object_fusion_optuna/best_per_plot.csv"),
    }
    rows: list[dict] = []
    for method_name, path in sources.items():
        if not path.exists():
            continue
        table = pd.read_csv(path)
        table = table[table["plot_id"].isin(plot_ids)].copy()
        rows.append(
            {
                "method": method_name,
                "plot_scope": ",".join(plot_ids),
                "tp": int(table["tp"].sum()),
                "fp": int(table["fp"].sum()),
                "fn": int(table["fn"].sum()),
                "precision_mean": float(table["precision"].mean()),
                "recall_mean": float(table["recall"].mean()),
                "f1_mean": float(table["f1"].mean()),
                "distance_m_mean": float(table["distance_m"].mean()),
                "height_error_m_mean": float(table["height_error_m"].mean()),
            }
        )
    result = pd.DataFrame.from_records(rows).sort_values("f1_mean", ascending=False).reset_index(drop=True)
    result.to_csv(destination, index=False)
    return result


def _aggregate_metrics(metrics_by_plot: pd.DataFrame, partition_name: str) -> dict[str, float | str | int]:
    table = metrics_by_plot.copy()
    return {
        "partition": partition_name,
        "n_plots": int(len(table)),
        "tp": int(table["tp"].sum()),
        "fp": int(table["fp"].sum()),
        "fn": int(table["fn"].sum()),
        "precision_mean": float(table["precision"].mean()),
        "recall_mean": float(table["recall"].mean()),
        "f1_mean": float(table["f1"].mean()),
        "distance_m_mean": float(table["distance_m"].mean()),
        "height_error_m_mean": float(table["height_error_m"].mean()),
    }


def _pretendido_checklist(config: StrictPointNetConfig) -> pd.DataFrame:
    rows = [
        {
            "requirement": "Nuvem de pontos normalizada",
            "status": "yes",
            "details": "Usa o cache fiel do paper com Z normalizado por solo.",
        },
        {
            "requirement": "Gerador de propostas baseline do paper",
            "status": "yes",
            "details": f"Local maxima 3D com window_size={config.proposal_window_size_m:.1f} m e height_threshold={config.proposal_height_threshold_m:.1f} m.",
        },
        {
            "requirement": "Vizinhança esferica local",
            "status": "yes",
            "details": f"Raio {config.neighborhood_radius_m:.1f} m, {config.n_points} pontos, coordenadas centralizadas na proposta.",
        },
        {
            "requirement": "Rotulagem por proximidade para treinar PointNet",
            "status": "yes",
            "details": f"Positivo quando dist2D <= {config.positive_threshold_m:.1f} m; regressao usa offset XY ate o GT mais proximo.",
        },
        {
            "requirement": "PointNet puro local",
            "status": "yes",
            "details": "Shared MLP 3->64->128->256, max pooling global, cabecas separadas de classificacao e offset XY.",
        },
        {
            "requirement": "Split 8 treino / 2 validacao",
            "status": "yes",
            "details": "Validacao fixa em plot_01 e plot_06; demais parcelas no treino.",
        },
        {
            "requirement": "Loss combinada fiel",
            "status": "yes",
            "details": f"BCEWithLogits pos_weight={config.pos_weight:.1f} + SmoothL1 so nos positivos.",
        },
        {
            "requirement": "Augmentations da nuvem",
            "status": "yes",
            "details": "Rotacao aleatoria em Z, jitter gaussiano leve e subsampling aleatorio em cada leitura.",
        },
        {
            "requirement": "Inferencia com threshold + offset + NMS",
            "status": "yes",
            "details": "Score sigmoide, refinamento XY, filtro por threshold e NMS sobre deteccoes refinadas.",
        },
        {
            "requirement": "NMS 3D do diagrama geral",
            "status": "note",
            "details": "O Pretendido.md conflita: o diagrama cita NMS 3D, mas a Fase 6 fornece nms_2d explicita. A run fiel segue o pseudocodigo da Fase 6.",
        },
        {
            "requirement": "Matching + metricas do paper",
            "status": "yes",
            "details": "Matching guloso fiel com max_dist=5 m e max_height_diff=3 m.",
        },
    ]
    return pd.DataFrame.from_records(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the strict faithful local PointNet pipeline described in Pretendido.md.")
    parser.add_argument("--cache-dir", default="outputs_faithful_paper/cache")
    parser.add_argument("--output-dir", default="outputs_faithful_pointnet_strict")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    config = StrictPointNetConfig(seed=int(args.seed))
    _seed_everything(config.seed)
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"

    cache_dir = Path(args.cache_dir).resolve()
    output_dir = ensure_dir(Path(args.output_dir).resolve())
    predictions_dir = ensure_dir(output_dir / "predictions")

    points_by_plot, reference_by_plot = _load_points_and_reference(cache_dir)
    ordered_plots = sorted(points_by_plot)
    val_plots = [plot_id for plot_id in VAL_PLOTS if plot_id in ordered_plots]
    train_plots = [plot_id for plot_id in ordered_plots if plot_id not in set(val_plots)]

    proposal_table, proposals_by_plot = _build_all_proposals(ordered_plots, points_by_plot, reference_by_plot, config)
    proposal_table.to_csv(output_dir / "proposal_table_all.csv", index=False)

    matching_params = FaithfulMatchingParams(max_dist_m=5.0, max_height_diff_m=3.0)
    proposal_metrics = summarize_proposals_against_reference(proposals_by_plot, reference_by_plot, matching_params)
    proposal_metrics["partition"] = proposal_metrics["plot_id"].map(lambda plot_id: "val" if plot_id in set(val_plots) else "train")
    proposal_metrics.to_csv(output_dir / "proposal_metrics_by_plot.csv", index=False)
    pd.DataFrame.from_records(
        [
            _aggregate_metrics(proposal_metrics[proposal_metrics["partition"] == "train"], "proposal_train"),
            _aggregate_metrics(proposal_metrics[proposal_metrics["partition"] == "val"], "proposal_val"),
            _aggregate_metrics(proposal_metrics, "proposal_all"),
        ]
    ).to_csv(output_dir / "proposal_summary_by_partition.csv", index=False)

    train_table = proposal_table[proposal_table["plot_id"].isin(train_plots)].reset_index(drop=True)
    val_table = proposal_table[proposal_table["plot_id"].isin(val_plots)].reset_index(drop=True)

    train_dataset = StrictProposalDataset(points_by_plot=points_by_plot, proposal_table=train_table, config=config, training=True)
    val_dataset = StrictProposalDataset(points_by_plot=points_by_plot, proposal_table=val_table, config=config, training=False)

    train_loader = DataLoader(train_dataset, batch_size=int(config.batch_size), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=int(config.batch_size), shuffle=False, num_workers=0)

    model = LocalPointNet(dropout=float(config.dropout)).to(device)
    criterion = CombinedPointNetLoss(pos_weight=float(config.pos_weight), alpha=1.0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )

    best_state = None
    best_val_loss = float("inf")
    patience_counter = 0
    history: list[dict] = []
    for epoch in range(int(config.epochs)):
        train_metrics = run_pointnet_epoch(model, train_loader, optimizer, criterion, device=device, train=True)
        val_metrics = run_pointnet_epoch(model, val_loader, optimizer, criterion, device=device, train=False)
        history_row = {
            "epoch": epoch + 1,
            **{f"train_{key}": float(value) for key, value in train_metrics.items()},
            **{f"val_{key}": float(value) for key, value in val_metrics.items()},
        }
        history.append(history_row)
        if float(val_metrics["loss_total"]) < best_val_loss:
            best_val_loss = float(val_metrics["loss_total"])
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= int(config.patience):
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    pd.DataFrame.from_records(history).to_csv(output_dir / "history.csv", index=False)

    all_dataset = StrictProposalDataset(points_by_plot=points_by_plot, proposal_table=proposal_table, config=config, training=False)
    all_scored = score_pointnet_dataset(model, all_dataset, batch_size=int(config.batch_size), device=device)
    all_scored.to_csv(output_dir / "scored_proposals_all.csv", index=False)

    val_scored = all_scored[all_scored["plot_id"].isin(val_plots)].copy()
    best_postprocess, val_metrics_by_plot = tune_pointnet_postprocess(
        scored=val_scored,
        reference_by_plot={plot_id: reference_by_plot[plot_id] for plot_id in val_plots},
        matching_params=matching_params,
        cls_threshold_grid=tuple(float(value) for value in config.cls_threshold_grid),
        nms_radius_grid_m=tuple(float(value) for value in config.nms_radius_grid_m),
    )
    write_json(best_postprocess, output_dir / "best_postprocess.json")
    val_metrics_by_plot["partition"] = "val"
    val_metrics_by_plot.to_csv(output_dir / "validation_metrics_by_plot.csv", index=False)

    prediction_rows: list[dict] = []
    metrics_rows: list[dict] = []
    for plot_id in ordered_plots:
        current_scored = all_scored[all_scored["plot_id"] == plot_id].copy()
        predictions = predict_and_refine(
            scored=current_scored,
            cls_threshold=float(best_postprocess["cls_threshold"]),
            nms_radius=float(best_postprocess["nms_radius_m"]),
        )
        predictions.to_csv(predictions_dir / f"{plot_id}.csv", index=False)
        metrics = summarize_faithful_matches(
            faithful_greedy_match(
                predictions=predictions,
                reference=reference_by_plot[plot_id],
                params=matching_params,
            )
        )
        metrics["plot_id"] = plot_id
        metrics["partition"] = "val" if plot_id in set(val_plots) else "train"
        metrics["n_detections"] = int(len(predictions))
        metrics["n_reference"] = int(len(reference_by_plot[plot_id]))
        metrics_rows.append(metrics)
        prediction_rows.append(
            {
                "plot_id": plot_id,
                "n_scored_proposals": int(len(current_scored)),
                "n_final_detections": int(len(predictions)),
            }
        )

    metrics_by_plot = pd.DataFrame.from_records(metrics_rows).sort_values("plot_id").reset_index(drop=True)
    metrics_by_plot.to_csv(output_dir / "metrics_by_plot.csv", index=False)
    pd.DataFrame.from_records(prediction_rows).sort_values("plot_id").to_csv(output_dir / "prediction_counts.csv", index=False)

    summary_by_partition = pd.DataFrame.from_records(
        [
            _aggregate_metrics(metrics_by_plot[metrics_by_plot["partition"] == "train"], "train"),
            _aggregate_metrics(metrics_by_plot[metrics_by_plot["partition"] == "val"], "val"),
            _aggregate_metrics(metrics_by_plot, "all"),
        ]
    )
    summary_by_partition.to_csv(output_dir / "summary_by_partition.csv", index=False)

    baseline_val = _baseline_comparison(val_plots, output_dir / "baseline_comparison_val.csv")
    baseline_all = _baseline_comparison(ordered_plots, output_dir / "baseline_comparison_all.csv")

    checklist = _pretendido_checklist(config)
    checklist.to_csv(output_dir / "pretendido_checklist.csv", index=False)

    final_val = summary_by_partition[summary_by_partition["partition"] == "val"].iloc[0].to_dict()
    report_lines = [
        "# Resultados Faithful PointNet Strict",
        "",
        "## Configuracao",
        f"- treino: {', '.join(train_plots)}",
        f"- validacao: {', '.join(val_plots)}",
        f"- propostas: local maxima 3D com window_size={config.proposal_window_size_m:.1f} m e height_threshold={config.proposal_height_threshold_m:.1f} m",
        f"- vizinhanca: esfera de raio {config.neighborhood_radius_m:.1f} m com {config.n_points} pontos",
        f"- tuning: cls_threshold em {list(config.cls_threshold_grid)}; nms_radius em {list(config.nms_radius_grid_m)}",
        "",
        "## Melhor pos-processamento na validacao",
        f"- cls_threshold = {best_postprocess['cls_threshold']:.2f}",
        f"- nms_radius_m = {best_postprocess['nms_radius_m']:.2f}",
        "",
        "## Resultado principal na validacao",
        f"- precision_mean = {final_val['precision_mean']:.4f}",
        f"- recall_mean = {final_val['recall_mean']:.4f}",
        f"- f1_mean = {final_val['f1_mean']:.4f}",
        f"- distance_m_mean = {final_val['distance_m_mean']:.4f}",
        f"- height_error_m_mean = {final_val['height_error_m_mean']:.4f}",
        "",
        "## Observacao de fidelidade",
        "- O Pretendido.md conflita entre 'NMS 3D' no diagrama e `nms_2d` na Fase 6. A implementacao e a run seguiram a Fase 6 literalmente.",
    ]
    write_text("\n".join(report_lines) + "\n", output_dir / "REPORT.md")

    write_json(
        {
            "config": asdict(config),
            "device": device,
            "train_plots": train_plots,
            "val_plots": val_plots,
            "best_postprocess": best_postprocess,
            "final_val_summary": final_val,
            "baseline_val_best_f1": float(baseline_val["f1_mean"].max()) if not baseline_val.empty else None,
            "baseline_all_best_f1": float(baseline_all["f1_mean"].max()) if not baseline_all.empty else None,
        },
        output_dir / "run_summary.json",
    )


if __name__ == "__main__":
    main()
