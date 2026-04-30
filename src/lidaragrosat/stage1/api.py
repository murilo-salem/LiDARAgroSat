from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from lidaragrosat.data.dataset import ProcessedPlotBundle


@dataclass(slots=True)
class Stage1SegmentDiagnostics:
    correct_segments: int = 0
    under_segmented: int = 0
    over_segmented: int = 0
    mixed_segment: int = 0
    empty_segment: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "correct_segments": int(self.correct_segments),
            "under_segmented": int(self.under_segmented),
            "over_segmented": int(self.over_segmented),
            "mixed_segment": int(self.mixed_segment),
            "empty_segment": int(self.empty_segment),
        }


@dataclass(slots=True)
class Stage1Output:
    plot_id: str
    candidates: pd.DataFrame
    segment_labels: np.ndarray | None = None
    segment_table: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    diagnostics: Stage1SegmentDiagnostics = field(default_factory=Stage1SegmentDiagnostics)
    metadata: dict = field(default_factory=dict)


class Stage1Generator(Protocol):
    def generate(self, plot_bundle: ProcessedPlotBundle) -> Stage1Output:
        ...
