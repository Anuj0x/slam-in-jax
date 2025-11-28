"""Tests for core SLAM functionality."""

import pytest
import jax.numpy as jnp
from modern_slam.core import Position, Landmark, Measurement, Motion, Robot, SLAMSystem
from modern_slam.config import SLAMConfig


class TestPosition:
    """Test Position dataclass."""

    def test_creation(self):
        pos = Position(1.0, 2.0)
        assert pos.x == 1.0
        assert pos.y == 2.0

    def test_to_array(self):
        pos = Position(3.0, 4.0)
        arr = pos.to_array()
        assert jnp.allclose(arr, jnp.array([3.0, 4.0]))

    def test_distance_to(self):
        pos1 = Position(0.0, 0.0)
        pos2 = Position(3.0, 4.0)
        assert pos1.distance_to(pos2) == 5.0


class TestLandmark:
    """Test Landmark dataclass."""

    def test_creation(self):
        landmark = Landmark(1, Position(10.0, 20.0))
        assert landmark.id == 1
        assert landmark.position.x == 10.0
        assert landmark.position.y == 20.0


class TestMeasurement:
    """Test Measurement dataclass."""

    def test_creation(self):
        measurement = Measurement(landmark_id=1, dx=2.0, dy=3.0)
        assert measurement.landmark_id == 1
        assert measurement.dx == 2.0
        assert measurement.dy == 3.0

    def test_position_offset(self):
        measurement = Measurement(landmark_id=1, dx=1.0, dy=2.0)
        offset = measurement.position_offset
        assert jnp.allclose(offset, jnp.array([1.0, 2.0]))


class TestMotion:
    """Test Motion dataclass."""

    def test_creation(self):
        motion = Motion(dx=1.0, dy=2.0)
        assert motion.dx == 1.0
        assert motion.dy == 2.0

    def test_displacement(self):
        motion = Motion(dx=3.0, dy=4.0)
        disp = motion.displacement
        assert jnp.allclose(disp, jnp.array([3.0, 4.0]))


class TestRobot:
    """Test Robot class."""

    def test_initialization(self):
        config = SLAMConfig(world_size=50.0, num_landmarks=5)
        robot = Robot(position=Position(25.0, 25.0), config=config)

        assert robot.position.x == 25.0
        assert robot.position.y == 25.0
        assert len(robot.landmarks) == 5

    def test_move_success(self):
        config = SLAMConfig(world_size=50.0)
        robot = Robot(position=Position(25.0, 25.0), config=config)

        motion = Motion(dx=5.0, dy=5.0)
        success = robot.move(motion)

        assert success
        assert robot.position.x == 30.0  # 25 + 5
        assert robot.position.y == 30.0  # 25 + 5

    def test_move_boundary_fail(self):
        config = SLAMConfig(world_size=50.0)
        robot = Robot(position=Position(45.0, 45.0), config=config)

        motion = Motion(dx=10.0, dy=10.0)  # Would go to 55, 55
        success = robot.move(motion)

        assert not success
        assert robot.position.x == 45.0  # Should not change
        assert robot.position.y == 45.0

    def test_sense(self):
        config = SLAMConfig(world_size=50.0, measurement_range=10.0, num_landmarks=3)
        robot = Robot(position=Position(25.0, 25.0), config=config)

        # Manually set landmarks close to robot
        robot.landmarks = [
            Landmark(0, Position(28.0, 26.0)),  # Within range
            Landmark(1, Position(40.0, 40.0)),  # Outside range
            Landmark(2, Position(22.0, 23.0)),  # Within range
        ]

        measurements = robot.sense()

        # Should detect 2 landmarks within range
        assert len(measurements) == 2
        landmark_ids = {m.landmark_id for m in measurements}
        assert landmark_ids == {0, 2}


class TestSLAMSystem:
    """Test SLAM system integration."""

    def test_initialization(self):
        config = SLAMConfig()
        system = SLAMSystem(config)

        assert system.config == config
        assert system.robot is not None
        assert len(system.data.measurements) == 0

    def test_generate_trajectory(self):
        config = SLAMConfig(num_landmarks=5)
        system = SLAMSystem(config)

        data = system.generate_trajectory(num_steps=10)

        assert data.num_timesteps == 10
        assert len(data.measurements) == 10
        assert len(data.motions) == 10

        # Each timestep should have some measurements (may be empty lists)
        assert all(isinstance(m, list) for m in data.measurements)
        assert all(isinstance(m, Motion) for m in data.motions)


if __name__ == "__main__":
    pytest.main([__file__])
