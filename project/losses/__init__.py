from .semantic_boundary_loss import SemanticBoundaryLoss
from .semantic_only_loss import SemanticOnlyLoss
from .route_a_loss import RouteASemanticBoundaryLoss
from .axis_side_loss import AxisSideSemanticBoundaryLoss
from .support_shape_loss import SupportShapeLoss


def build_loss(cfg: dict | None):
    if cfg is None:
        raise ValueError(
            "loss config is required; implicit SemanticBoundaryLoss fallback has been removed."
        )
    loss_type = cfg.get("type", "SemanticBoundaryLoss")
    if loss_type == "SemanticBoundaryLoss":
        kwargs = {key: value for key, value in cfg.items() if key != "type"}
        return SemanticBoundaryLoss(**kwargs)
    if loss_type == "SemanticOnlyLoss":
        return SemanticOnlyLoss()
    if loss_type == "RouteASemanticBoundaryLoss":
        kwargs = {key: value for key, value in cfg.items() if key != "type"}
        return RouteASemanticBoundaryLoss(**kwargs)
    if loss_type == "AxisSideSemanticBoundaryLoss":
        kwargs = {key: value for key, value in cfg.items() if key != "type"}
        return AxisSideSemanticBoundaryLoss(**kwargs)
    if loss_type == "SupportShapeLoss":
        kwargs = {key: value for key, value in cfg.items() if key != "type"}
        return SupportShapeLoss(**kwargs)
    raise ValueError(f"Unsupported loss type: {loss_type}")


__all__ = [
    "SemanticBoundaryLoss",
    "SemanticOnlyLoss",
    "RouteASemanticBoundaryLoss",
    "AxisSideSemanticBoundaryLoss",
    "SupportShapeLoss",
    "build_loss",
]
