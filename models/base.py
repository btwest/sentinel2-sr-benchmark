"""Abstract base class for all super-resolution models."""

from abc import ABC, abstractmethod
import numpy as np


class SRModel(ABC):
    """
    Every SR method — classical or learned — implements this interface.
    upscale() takes a float32 numpy array in [0, 1] and returns the same.
    """

    name: str  # set in each subclass

    @abstractmethod
    def upscale(self, lr: np.ndarray) -> np.ndarray:
        """
        Upscale a low-resolution patch.

        Args:
            lr: float32 array of shape (H, W) or (C, H, W), values in [0, 1]

        Returns:
            sr: float32 array of shape (H*scale, W*scale) or (C, H*scale, W*scale)
        """
        ...

    @property
    @abstractmethod
    def scale(self) -> int:
        """Upscale factor (e.g. 2 for 10m -> 5m)."""
        ...
