"""Q: per-frame adequacy (quality) classifier and per-breast coverage rule."""
from .features import FEATURE_NAMES, frame_features
from .gate import FAILURE_MODES, QualityGate, coverage_status, synthesize_failure

__all__ = ["FEATURE_NAMES", "frame_features", "FAILURE_MODES", "QualityGate", "coverage_status", "synthesize_failure"]
