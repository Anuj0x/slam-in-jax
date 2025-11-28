"""Core data structures and robot simulation for Modern SLAM."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Protocol
import numpy as np
import jax.numpy as jnp
from jax import random as jax_random

from .config import SLAMConfig


@dataclass(frozen=True)
class Position:
    """2D position with x and y coordinates."""
    x: float
    y: float

    def to_array(self) -> jnp.ndarray:
        """Convert to JAX array."""
        return jnp.array([self.x, self.y])

    @classmethod
    def from_array(cls, arr: jnp.ndarray) -> Position:
        """Create from JAX array."""
        return cls(x=float(arr[0]), y=float(arr[1]))

    def distance_to(self, other: Position) -> float:
        """Calculate Euclidean distance to another position."""
        return float(jnp.linalg.norm(self.to_array() - other.to_array()))


@dataclass(frozen=True)
class Landmark:
    """Landmark with ID and position."""
    id: int
    position: Position

    def __repr__(self) -> str:
        return f"Landmark(id={self.id}, pos=({self.position.x:.1f}, {self.position.y:.1f}))"


@dataclass
class Measurement:
    """Sensor measurement of a landmark."""
    landmark_id: int
    dx: float  # measured x-distance
    dy: float  # measured y-distance
    noise_x: float = 0.0
    noise_y: float = 0.0

    @property
    def position_offset(self) -> jnp.ndarray:
        """Get position offset as array."""
        return jnp.array([self.dx, self.dy])

    def __repr__(self) -> str:
        return f"Measurement(id={self.landmark_id}, dx={self.dx:.2f}, dy={self.dy:.2f})"


@dataclass
class Motion:
    """Robot motion command."""
    dx: float
    dy: float
    noise_x: float = 0.0
    noise_y: float = 0.0

    @property
    def displacement(self) -> jnp.ndarray:
        """Get displacement as array."""
        return jnp.array([self.dx, self.dy])

    def __repr__(self) -> str:
        return f"Motion(dx={self.dx:.2f}, dy={self.dy:.2f})"


class NoiseGenerator(Protocol):
    """Protocol for noise generation."""

    def generate_motion_noise(self, config: SLAMConfig) -> Tuple[float, float]:
        """Generate motion noise."""
        ...

    def generate_measurement_noise(self, config: SLAMConfig) -> Tuple[float, float]:
        """Generate measurement noise."""
        ...


class DefaultNoiseGenerator:
    """Default noise generator using uniform distribution."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def generate_motion_noise(self, config: SLAMConfig) -> Tuple[float, float]:
        """Generate motion noise."""
        noise_x = self.rng.uniform(-1.0, 1.0) * config.motion_noise
        noise_y = self.rng.uniform(-1.0, 1.0) * config.motion_noise
        return noise_x, noise_y

    def generate_measurement_noise(self, config: SLAMConfig) -> Tuple[float, float]:
        """Generate measurement noise."""
        noise_x = self.rng.uniform(-1.0, 1.0) * config.measurement_noise
        noise_y = self.rng.uniform(-1.0, 1.0) * config.measurement_noise
        return noise_x, noise_y


@dataclass
class Robot:
    """Modern robot simulation with improved physics and sensing."""

    position: Position
    config: SLAMConfig
    landmarks: List[Landmark] = field(default_factory=list)
    noise_generator: NoiseGenerator = field(default_factory=DefaultNoiseGenerator)

    def __post_init__(self):
        """Initialize landmarks if not provided."""
        if not self.landmarks:
            self.landmarks = self._generate_landmarks()

    def _generate_landmarks(self) -> List[Landmark]:
        """Generate random landmarks in the world."""
        landmarks = []
        rng = random.Random(42)  # Fixed seed for reproducibility

        for i in range(self.config.num_landmarks):
            x = rng.uniform(0, self.config.world_size)
            y = rng.uniform(0, self.config.world_size)
            landmarks.append(Landmark(id=i, position=Position(x, y)))

        return landmarks

    def move(self, motion: Motion) -> bool:
        """Attempt to move the robot with noise."""
        # Add noise to motion
        noise_x, noise_y = self.noise_generator.generate_motion_noise(self.config)

        new_x = self.position.x + motion.dx + noise_x
        new_y = self.position.y + motion.dy + noise_y

        # Check world boundaries
        if (0 <= new_x <= self.config.world_size and
            0 <= new_y <= self.config.world_size):
            self.position = Position(new_x, new_y)
            # Update motion with actual noise applied
            motion.noise_x = noise_x
            motion.noise_y = noise_y
            return True

        return False  # Movement failed - hit boundary

    def sense(self) -> List[Measurement]:
        """Sense landmarks within measurement range."""
        measurements = []

        for landmark in self.landmarks:
            # Calculate true relative position
            dx_true = landmark.position.x - self.position.x
            dy_true = landmark.position.y - self.position.y

            # Check if landmark is within range
            distance = np.sqrt(dx_true**2 + dy_true**2)
            if distance <= self.config.measurement_range:
                # Add measurement noise
                noise_x, noise_y = self.noise_generator.generate_measurement_noise(self.config)

                measurement = Measurement(
                    landmark_id=landmark.id,
                    dx=dx_true + noise_x,
                    dy=dy_true + noise_y,
                    noise_x=noise_x,
                    noise_y=noise_y
                )
                measurements.append(measurement)

        return measurements

    def get_true_position(self) -> Position:
        """Get the robot's true position."""
        return self.position

    def __repr__(self) -> str:
        return f"Robot(pos=({self.position.x:.2f}, {self.position.y:.2f}), landmarks={len(self.landmarks)})"


@dataclass
class SLAMData:
    """Container for SLAM measurements and motions over time."""
    measurements: List[List[Measurement]] = field(default_factory=list)
    motions: List[Motion] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)

    def add_timestep(self, measurements: List[Measurement], motion: Motion, timestamp: float = 0.0):
        """Add a timestep of data."""
        self.measurements.append(measurements)
        self.motions.append(motion)
        self.timestamps.append(timestamp)

    @property
    def num_timesteps(self) -> int:
        """Number of timesteps in the data."""
        return len(self.measurements)


class SLAMSystem:
    """Main SLAM system orchestrator."""

    def __init__(self, config: SLAMConfig):
        self.config = config
        self.robot = Robot(
            position=Position(config.world_size / 2, config.world_size / 2),
            config=config
        )
        self.data = SLAMData()

    def generate_trajectory(self, num_steps: int, step_distance: float = 1.0) -> SLAMData:
        """Generate a trajectory with measurements."""
        import time

        for step in range(num_steps):
            # Generate random motion
            orientation = random.uniform(0, 2 * np.pi)
            dx = np.cos(orientation) * step_distance
            dy = np.sin(orientation) * step_distance

            motion = Motion(dx=dx, dy=dy)

            # Sense before moving
            measurements = self.robot.sense()

            # Attempt to move
            success = self.robot.move(motion)
            if not success:
                # Try a different direction if we hit a boundary
                orientation = random.uniform(0, 2 * np.pi)
                dx = np.cos(orientation) * step_distance
                dy = np.sin(orientation) * step_distance
                motion = Motion(dx=dx, dy=dy)
                self.robot.move(motion)

            # Record data
            self.data.add_timestep(measurements, motion, time.time())

        return self.data

    def run_slam(self) -> dict:
        """Run the complete SLAM algorithm."""
        # This will be implemented with the algorithms module
        raise NotImplementedError("SLAM algorithm not yet implemented")
