"""Pose estimation: the ``PoseHead`` contract plus pluggable backends.

The pipeline depends only on :class:`~deeplabsam.pose.base.PoseHead`; concrete
backends live in :mod:`deeplabsam.pose.backends`. Vendored DeepLabCut model code
sits under the private :mod:`deeplabsam.pose._dlc`, used only by the DLC backend.
"""

from deeplabsam.pose.base import PoseHead

__all__ = ["PoseHead"]
