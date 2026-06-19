"""Concrete pose backends, one module per framework."""

from deeplabsam.pose.backends.dlc import DLCPoseHead

__all__ = ["DLCPoseHead"]
