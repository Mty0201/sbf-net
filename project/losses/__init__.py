from .semantic_boundary_loss import SemanticBoundaryLoss
from .semantic_only_loss import SemanticOnlyLoss
from .route_a_loss import RouteASemanticBoundaryLoss
from .axis_side_loss import AxisSideSemanticBoundaryLoss
from .support_shape_loss import SupportShapeLoss
from .support_guided_semantic_focus_loss import SupportGuidedSemanticFocusLoss
from .redesigned_support_focus_loss import RedesignedSupportFocusLoss
from .boundary_proximity_cue_loss import BoundaryProximityCueLoss
from .serial_derivation_loss import SerialDerivationLoss

from .unweighted_boundary_cue_loss import UnweightedBoundaryCueLoss
from .soft_boundary_loss import SoftBoundaryLoss
from .focal_mse_boundary_loss import FocalMSEBoundaryLoss
from .boundary_upweight_loss import BoundaryUpweightLoss
from .boundary_weighted_semantic_loss import BoundaryWeightedSemanticLoss
from .boundary_binary_loss import BoundaryBinaryLoss
from .dual_supervision_boundary_binary_loss import DualSupervisionBoundaryBinaryLoss
from .pure_bfanet_loss import PureBFANetLoss
from .dual_supervision_pure_bfanet_loss import DualSupervisionPureBFANetLoss
from .support_weighted_bfanet_loss import SupportWeightedBFANetLoss
from .dual_supervision_support_weighted_bfanet_loss import (
    DualSupervisionSupportWeightedBFANetLoss,
)
from .soft_weighted_semantic_loss import SoftWeightedSemanticLoss


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
    if loss_type == "SerialDerivationLoss":
        kwargs = {key: value for key, value in cfg.items() if key != "type"}
        return SerialDerivationLoss(**kwargs)
    if loss_type == "UnweightedBoundaryCueLoss":
        kwargs = {key: value for key, value in cfg.items() if key != "type"}
        return UnweightedBoundaryCueLoss(**kwargs)
    if loss_type == "SoftBoundaryLoss":
        kwargs = {key: value for key, value in cfg.items() if key != "type"}
        return SoftBoundaryLoss(**kwargs)
    if loss_type == "FocalMSEBoundaryLoss":
        kwargs = {key: value for key, value in cfg.items() if key != "type"}
        return FocalMSEBoundaryLoss(**kwargs)
    if loss_type == "BoundaryUpweightLoss":
        kwargs = {key: value for key, value in cfg.items() if key != "type"}
        return BoundaryUpweightLoss(**kwargs)
    if loss_type == "BoundaryWeightedSemanticLoss":
        kwargs = {key: value for key, value in cfg.items() if key != "type"}
        return BoundaryWeightedSemanticLoss(**kwargs)
    if loss_type == "BoundaryBinaryLoss":
        kwargs = {key: value for key, value in cfg.items() if key != "type"}
        return BoundaryBinaryLoss(**kwargs)
    if loss_type == "DualSupervisionBoundaryBinaryLoss":
        kwargs = {key: value for key, value in cfg.items() if key != "type"}
        return DualSupervisionBoundaryBinaryLoss(**kwargs)
    if loss_type == "PureBFANetLoss":
        kwargs = {key: value for key, value in cfg.items() if key != "type"}
        return PureBFANetLoss(**kwargs)
    if loss_type == "DualSupervisionPureBFANetLoss":
        kwargs = {key: value for key, value in cfg.items() if key != "type"}
        return DualSupervisionPureBFANetLoss(**kwargs)
    if loss_type == "SupportWeightedBFANetLoss":
        kwargs = {key: value for key, value in cfg.items() if key != "type"}
        return SupportWeightedBFANetLoss(**kwargs)
    if loss_type == "DualSupervisionSupportWeightedBFANetLoss":
        kwargs = {key: value for key, value in cfg.items() if key != "type"}
        return DualSupervisionSupportWeightedBFANetLoss(**kwargs)
    if loss_type == "SoftWeightedSemanticLoss":
        kwargs = {key: value for key, value in cfg.items() if key != "type"}
        return SoftWeightedSemanticLoss(**kwargs)
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
    "SerialDerivationLoss",

    "UnweightedBoundaryCueLoss",
    "SoftBoundaryLoss",
    "FocalMSEBoundaryLoss",
    "BoundaryUpweightLoss",
    "BoundaryWeightedSemanticLoss",
    "BoundaryBinaryLoss",
    "DualSupervisionBoundaryBinaryLoss",
    "PureBFANetLoss",
    "DualSupervisionPureBFANetLoss",
    "SupportWeightedBFANetLoss",
    "DualSupervisionSupportWeightedBFANetLoss",
    "SoftWeightedSemanticLoss",
    "build_loss",
]
