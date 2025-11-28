"""Configuration management for Modern SLAM."""

from typing import Optional
from pydantic import BaseModel, Field, validator


class SLAMConfig(BaseModel):
    """Configuration for SLAM system parameters."""

    # World parameters
    world_size: float = Field(default=100.0, gt=0, description="Size of the square world")
    num_landmarks: int = Field(default=20, gt=0, description="Number of landmarks in the world")

    # Robot parameters
    measurement_range: float = Field(default=30.0, gt=0, description="Maximum sensing range")
    motion_noise: float = Field(default=1.0, ge=0, description="Noise in robot motion")
    measurement_noise: float = Field(default=1.0, ge=0, description="Noise in measurements")

    # SLAM parameters
    max_iterations: int = Field(default=100, gt=0, description="Maximum SLAM iterations")
    convergence_threshold: float = Field(default=1e-6, gt=0, description="Convergence threshold")
    use_gpu: bool = Field(default=False, description="Enable GPU acceleration")

    # Visualization parameters
    visualization_backend: str = Field(default="plotly", description="Visualization backend")
    show_progress: bool = Field(default=True, description="Show progress bars")

    @validator('visualization_backend')
    def validate_backend(cls, v):
        if v not in ['matplotlib', 'plotly']:
            raise ValueError('Visualization backend must be matplotlib or plotly')
        return v

    class Config:
        """Pydantic config."""
        allow_mutation = False
        validate_assignment = True


# Default configurations for different scenarios
DEFAULT_CONFIG = SLAMConfig()

FAST_CONFIG = SLAMConfig(
    num_landmarks=10,
    max_iterations=50,
    convergence_threshold=1e-4
)

ACCURATE_CONFIG = SLAMConfig(
    num_landmarks=50,
    max_iterations=200,
    convergence_threshold=1e-8,
    motion_noise=0.1,
    measurement_noise=0.1
)
