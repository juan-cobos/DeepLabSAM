#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from deeplabsam.pose.models.backbones import BACKBONES, BaseBackbone
from deeplabsam.pose.models.heads import HEADS, BaseHead
from deeplabsam.pose.models.predictors import PREDICTORS


class PoseModel(nn.Module):
    """A pose estimation model.

    A pose estimation model is composed of a backbone and an arbitrary number of
    heads. Outputs are computed as follows:
    """

    def __init__(
        self,
        cfg: dict,
        backbone: BaseBackbone,
        heads: dict[str, BaseHead],
    ) -> None:
        """
        Args:
            cfg: configuration dictionary for the model.
            backbone: backbone network architecture.
            heads: the heads for the model
        """
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.heads = nn.ModuleDict(heads)
        self.output_features = False

        self._strides = {name: _model_stride(self.backbone.stride, head.stride) for name, head in heads.items()}

    def forward(self, x: torch.Tensor, **backbone_kwargs) -> dict[str, dict[str, torch.Tensor]]:
        """Forward pass of the PoseModel.

        Args:
            x: input images

        Returns:
            Outputs of head groups
        """
        if x.dim() == 3:
            x = x[None, :]
        features = self.backbone(x, **backbone_kwargs)

        outputs = {}
        if self.output_features:
            outputs["backbone"] = dict(features=features)

        for head_name, head in self.heads.items():
            outputs[head_name] = head(features)
        return outputs

    def get_predictions(self, outputs: dict[str, dict[str, torch.Tensor]]) -> dict:
        """Abstract method for the forward pass of the Predictor.

        Args:
            outputs: outputs of the model heads

        Returns:
            A dictionary containing the predictions of each head group
        """
        predictions = {name: head.predictor(self._strides[name], outputs[name]) for name, head in self.heads.items()}
        if self.output_features:
            predictions["backbone"] = outputs["backbone"]

        return predictions

    def get_stride(self, head: str) -> int:
        """
        Args:
            head: The head for which to get the total stride.

        Returns:
            The total stride for the outputs of the head.

        Raises:
            ValueError: If there is no such head.
        """
        return self._strides[head]

    @staticmethod
    def build(cfg: dict, pretrained_backbone: bool = False) -> PoseModel:
        """Build a pose model for inference from a config.

        Constructs the backbone and heads (each with its predictor).
        Training-only pieces (criterion / loss aggregator / target generator)
        are not built — their config keys are ignored by the head constructors.
        Weights are loaded separately by the caller via ``load_state_dict``
        (see ``DLCTorchPose``).

        Args:
            cfg: The configuration of the model to build.
            pretrained_backbone: Whether to load ImageNet-pretrained backbone
                weights from Timm (only useful for transfer learning).

        Returns:
            the built pose model
        """
        cfg["backbone"]["pretrained"] = pretrained_backbone
        backbone = BACKBONES.build(dict(cfg["backbone"]))

        heads = {}
        for name, head_cfg in cfg["heads"].items():
            head_cfg = copy.deepcopy(head_cfg)
            head_cfg["predictor"] = PREDICTORS.build(head_cfg["predictor"])
            heads[name] = HEADS.build(head_cfg)

        return PoseModel(cfg=cfg, backbone=backbone, heads=heads)


def _model_stride(backbone_stride: int | float, head_stride: int | float) -> float:
    """Computes the model stride from a backbone and a head."""
    if head_stride > 0:
        return backbone_stride / head_stride

    return backbone_stride * -head_stride
