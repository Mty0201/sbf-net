from .semantic_boundary_evaluator import SemanticBoundaryEvaluator
from .semantic_evaluator import SemanticEvaluator
from .axis_side_evaluator import AxisSideEvaluator
from .support_guided_semantic_focus_evaluator import SupportGuidedSemanticFocusEvaluator
from .redesigned_support_focus_evaluator import RedesignedSupportFocusEvaluator


def build_evaluator(cfg: dict | None):
    if cfg is None:
        raise ValueError(
            "evaluator config is required; implicit SemanticBoundaryEvaluator fallback has been removed."
        )
    evaluator_type = cfg.get("type", "SemanticBoundaryEvaluator")
    if evaluator_type == "SemanticBoundaryEvaluator":
        kwargs = {key: value for key, value in cfg.items() if key != "type"}
        return SemanticBoundaryEvaluator(**kwargs)
    if evaluator_type == "SemanticEvaluator":
        return SemanticEvaluator()
    if evaluator_type == "AxisSideEvaluator":
        kwargs = {key: value for key, value in cfg.items() if key != "type"}
        return AxisSideEvaluator(**kwargs)
    if evaluator_type == "SupportGuidedSemanticFocusEvaluator":
        kwargs = {key: value for key, value in cfg.items() if key != "type"}
        return SupportGuidedSemanticFocusEvaluator(**kwargs)
    if evaluator_type == "RedesignedSupportFocusEvaluator":
        kwargs = {key: value for key, value in cfg.items() if key != "type"}
        return RedesignedSupportFocusEvaluator(**kwargs)
    raise ValueError(f"Unsupported evaluator type: {evaluator_type}")


__all__ = [
    "SemanticBoundaryEvaluator",
    "SemanticEvaluator",
    "AxisSideEvaluator",
    "SupportGuidedSemanticFocusEvaluator",
    "RedesignedSupportFocusEvaluator",
    "build_evaluator",
]
