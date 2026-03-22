from .semantic_boundary_loss import SemanticBoundaryLoss
from .semantic_only_loss import SemanticOnlyLoss


def build_loss(cfg: dict | None):
    if cfg is None:
        return SemanticBoundaryLoss()
    loss_type = cfg.get("type", "SemanticBoundaryLoss")
    if loss_type == "SemanticBoundaryLoss":
        kwargs = {key: value for key, value in cfg.items() if key != "type"}
        return SemanticBoundaryLoss(**kwargs)
    if loss_type == "SemanticOnlyLoss":
        return SemanticOnlyLoss()
    raise ValueError(f"Unsupported loss type: {loss_type}")


__all__ = ["SemanticBoundaryLoss", "SemanticOnlyLoss", "build_loss"]
