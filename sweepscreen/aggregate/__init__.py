"""A: temporal aggregation of frame-level detections into per-breast and per-woman decisions."""
from .temporal import BreastResult, WomanResult, aggregate_breast, decide_woman, persistence_scores

__all__ = ["BreastResult", "WomanResult", "aggregate_breast", "decide_woman", "persistence_scores"]
