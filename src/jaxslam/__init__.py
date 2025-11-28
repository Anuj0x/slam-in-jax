"""Modern Simultaneous Localization and Mapping (SLAM) Implementation.

A high-performance, modular SLAM system using modern Python and JAX for
efficient numerical computations and GPU acceleration.
"""

__version__ = "0.1.0"
__author__ = "Modern SLAM Team"

from .core import SLAMSystem, Robot, Landmark, Measurement, Motion
from .algorithms import GraphSLAM, KalmanFilter
from .visualization import SLAMVisualizer
from .config import SLAMConfig

__all__ = [
    "SLAMSystem",
    "Robot",
    "Landmark",
    "Measurement",
    "Motion",
    "GraphSLAM",
    "KalmanFilter",
    "SLAMVisualizer",
    "SLAMConfig",
]
