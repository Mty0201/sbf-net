from .semantic_boundary_evaluator import SemanticBoundaryEvaluator
from .semantic_evaluator import SemanticEvaluator


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
    raise ValueError(f"Unsupported evaluator type: {evaluator_type}")


__all__ = [
    "SemanticBoundaryEvaluator",
    "SemanticEvaluator",
    "build_evaluator",
]
