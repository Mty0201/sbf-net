from .semantic_boundary_loss import SemanticBoundaryLoss
from .semantic_only_loss import SemanticOnlyLoss
from .route_a_loss import RouteASemanticBoundaryLoss
from .axis_side_loss import AxisSideSemanticBoundaryLoss
from .support_shape_loss import SupportShapeLoss
from .support_guided_semantic_focus_loss import SupportGuidedSemanticFocusLoss
from .redesigned_support_focus_loss import RedesignedSupportFocusLoss
from .boundary_proximity_cue_loss import BoundaryProximityCueLoss


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
    if loss_type == "SupportGuidedSemanticFocusLoss":
        kwargs = {key: value for key, value in cfg.items() if key != "type"}
        return SupportGuidedSemanticFocusLoss(**kwargs)
    if loss_type == "RedesignedSupportFocusLoss":
        kwargs = {key: value for key, value in cfg.items() if key != "type"}
        return RedesignedSupportFocusLoss(**kwargs)
    if loss_type == "BoundaryProximityCueLoss":
        kwargs = {key: value for key, value in cfg.items() if key != "type"}
        return BoundaryProximityCueLoss(**kwargs)
    raise ValueError(f"Unsupported loss type: {loss_type}")


__all__ = [
    "SemanticBoundaryLoss",
    "SemanticOnlyLoss",
    "RouteASemanticBoundaryLoss",
    "AxisSideSemanticBoundaryLoss",
    "SupportShapeLoss",
    "SupportGuidedSemanticFocusLoss",
    "RedesignedSupportFocusLoss",
    "BoundaryProximityCueLoss",
    "build_loss",
]
