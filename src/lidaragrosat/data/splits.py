from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import cycle

import random


@dataclass(slots=True)
class FoldSplit:
    fold_id: int
    train_plots: list[str]
    val_plots: list[str]
    test_plots: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def build_lopo_splits(plot_ids: list[str], validation_plots_per_fold: int = 1) -> list[FoldSplit]:
    ordered = list(plot_ids)
    if len(ordered) < 3:
        raise ValueError("At least three plots are required for LOPO with internal validation.")

    validation_cycle = cycle(ordered)
    splits: list[FoldSplit] = []
    for fold_id, test_plot in enumerate(ordered):
        candidate_train = [plot for plot in ordered if plot != test_plot]
        val_plots: list[str] = []
        while len(val_plots) < validation_plots_per_fold:
            candidate = next(validation_cycle)
            if candidate != test_plot and candidate not in val_plots:
                val_plots.append(candidate)
        train_plots = [plot for plot in candidate_train if plot not in val_plots]
        splits.append(
            FoldSplit(
                fold_id=fold_id,
                train_plots=train_plots,
                val_plots=val_plots,
                test_plots=[test_plot],
            )
        )
    return splits


def build_basic_split(
    plot_ids: list[str],
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    seed: int = 13,
) -> list[FoldSplit]:
    if len(plot_ids) < 3:
        raise ValueError("At least three plots are required for a basic train/val/test split.")

    ordered = list(plot_ids)
    rng = random.Random(seed)
    rng.shuffle(ordered)

    n_total = len(ordered)
    n_train = max(1, int(round(n_total * train_fraction)))
    n_val = max(1, int(round(n_total * val_fraction)))

    if n_train + n_val >= n_total:
        n_train = max(1, n_total - 2)
        n_val = 1
    n_test = n_total - n_train - n_val
    if n_test < 1:
        n_test = 1
        if n_train > 1:
            n_train -= 1
        else:
            n_val -= 1

    train_plots = ordered[:n_train]
    val_plots = ordered[n_train : n_train + n_val]
    test_plots = ordered[n_train + n_val :]
    return [FoldSplit(fold_id=0, train_plots=train_plots, val_plots=val_plots, test_plots=test_plots)]
