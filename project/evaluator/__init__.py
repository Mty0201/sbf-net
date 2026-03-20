from .semantic_boundary_evaluator import SemanticBoundaryEvaluator
from .semantic_evaluator import SemanticEvaluator


def build_evaluator(cfg: dict | None):
    if cfg is None:
        return SemanticBoundaryEvaluator()
    evaluator_type = cfg.get("type", "SemanticBoundaryEvaluator")
    if evaluator_type == "SemanticBoundaryEvaluator":
        return SemanticBoundaryEvaluator()
    if evaluator_type == "SemanticEvaluator":
        return SemanticEvaluator()
    raise ValueError(f"Unsupported evaluator type: {evaluator_type}")


__all__ = [
    "SemanticBoundaryEvaluator",
    "SemanticEvaluator",
    "build_evaluator",
]
