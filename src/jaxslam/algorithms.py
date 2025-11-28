"""High-performance SLAM algorithms using JAX for GPU acceleration."""

from __future__ import annotations

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.linalg import solve
from tqdm import tqdm

from .core import SLAMData, Measurement, Motion, Landmark, Position, Robot
from .config import SLAMConfig


@jit
def _build_constraints_batch(
    measurements: jnp.ndarray,  # Shape: (N_measurements, 3) - [landmark_id, dx, dy]
    motions: jnp.ndarray,       # Shape: (N_motions, 2) - [dx, dy]
    omega_init: jnp.ndarray,    # Initial omega matrix
    xi_init: jnp.ndarray        # Initial xi vector
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JAX-compiled function to build constraint matrices in batch."""
    N_poses = len(motions) + 1  # +1 for initial pose
    N_landmarks = int(jnp.max(measurements[:, 0])) + 1 if len(measurements) > 0 else 0
    N_vars = 2 * (N_poses + N_landmarks)

    omega = omega_init.copy()
    xi = xi_init.copy()

    # Process motion constraints
    for i in range(len(motions)):
        pose_i = 2 * i
        pose_j = 2 * (i + 1)

        dx, dy = motions[i]

        # Motion constraint: pose_j = pose_i + [dx, dy]
        # This creates constraints between consecutive poses

        # Omega updates for motion
        omega = omega.at[pose_i:pose_i+2, pose_i:pose_i+2].add(1.0)
        omega = omega.at[pose_j:pose_j+2, pose_j:pose_j+2].add(1.0)
        omega = omega.at[pose_i:pose_i+2, pose_j:pose_j+2].add(-1.0)
        omega = omega.at[pose_j:pose_j+2, pose_i:pose_i+2].add(-1.0)

        # Xi updates for motion
        xi = xi.at[pose_i:pose_i+2].add(jnp.array([-dx, -dy]))
        xi = xi.at[pose_j:pose_j+2].add(jnp.array([dx, dy]))

    # Process measurement constraints
    for measurement in measurements:
        landmark_id, dx, dy = measurement
        landmark_idx = 2 * (N_poses + int(landmark_id))

        # For simplicity, associate measurements with the last pose
        pose_idx = 2 * (len(motions))  # Last pose

        # Measurement constraint: landmark = pose + [dx, dy]
        omega = omega.at[pose_idx:pose_idx+2, pose_idx:pose_idx+2].add(1.0)
        omega = omega.at[landmark_idx:landmark_idx+2, landmark_idx:landmark_idx+2].add(1.0)
        omega = omega.at[pose_idx:pose_idx+2, landmark_idx:landmark_idx+2].add(-1.0)
        omega = omega.at[landmark_idx:landmark_idx+2, pose_idx:pose_idx+2].add(-1.0)

        xi = xi.at[pose_idx:pose_idx+2].add(jnp.array([-dx, -dy]))
        xi = xi.at[landmark_idx:landmark_idx+2].add(jnp.array([dx, dy]))

    return omega, xi


class GraphSLAM:
    """Graph-based SLAM implementation using constraint matrices."""

    def __init__(self, config: SLAMConfig):
        self.config = config

    def build_constraint_matrices(
        self,
        data: SLAMData,
        robot: Robot
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Build omega (information) and xi (information vector) matrices."""

        N_poses = data.num_timesteps + 1  # +1 for initial pose
        N_landmarks = len(robot.landmarks)
        N_vars = 2 * (N_poses + N_landmarks)

        # Initialize matrices
        omega = jnp.zeros((N_vars, N_vars))
        xi = jnp.zeros(N_vars)

        # Convert data to arrays for JAX processing
        if data.num_timesteps > 0:
            motions_array = jnp.array([[m.dx, m.dy] for m in data.motions])

            # Flatten measurements across all timesteps
            all_measurements = []
            for t, measurements in enumerate(data.measurements):
                for measurement in measurements:
                    all_measurements.append([
                        measurement.landmark_id,
                        measurement.dx,
                        measurement.dy
                    ])

            if all_measurements:
                measurements_array = jnp.array(all_measurements)

                # Build constraints using JAX
                omega, xi = _build_constraints_batch(
                    measurements_array, motions_array, omega, xi
                )

        return omega, xi

    def solve_slam(
        self,
        omega: jnp.ndarray,
        xi: jnp.ndarray
    ) -> jnp.ndarray:
        """Solve the SLAM problem using matrix inversion."""

        # Add small regularization for numerical stability
        omega_reg = omega + jnp.eye(omega.shape[0]) * 1e-6

        try:
            # Solve: omega * mu = xi => mu = omega^-1 * xi
            mu = solve(omega_reg, xi)
            return mu
        except:
            # Fallback for singular matrices
            print("Warning: Matrix inversion failed, using pseudoinverse")
            mu = jnp.linalg.pinv(omega_reg) @ xi
            return mu

    def extract_trajectory_and_map(
        self,
        solution: jnp.ndarray,
        data: SLAMData,
        robot: Robot
    ) -> Tuple[List[Position], List[Landmark]]:
        """Extract optimized trajectory and landmark positions from solution."""

        N_poses = data.num_timesteps + 1
        trajectory = []
        landmarks = []

        # Extract poses (every 2 elements: x, y)
        for i in range(N_poses):
            idx = 2 * i
            x, y = solution[idx], solution[idx + 1]
            trajectory.append(Position(x, y))

        # Extract landmarks
        landmark_start_idx = 2 * N_poses
        for i, landmark in enumerate(robot.landmarks):
            idx = landmark_start_idx + 2 * i
            if idx + 1 < len(solution):
                x, y = solution[idx], solution[idx + 1]
                landmarks.append(Landmark(
                    id=landmark.id,
                    position=Position(x, y)
                ))

        return trajectory, landmarks

    def run(
        self,
        data: SLAMData,
        robot: Robot,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """Run the complete Graph SLAM algorithm."""

        if show_progress:
            print("Building constraint matrices...")

        # Build constraint matrices
        omega, xi = self.build_constraint_matrices(data, robot)

        if show_progress:
            print(f"Constraint matrix size: {omega.shape}")
            print("Solving SLAM system...")

        # Solve the system
        solution = self.solve_slam(omega, xi)

        if show_progress:
            print("Extracting trajectory and map...")

        # Extract results
        trajectory, landmarks = self.extract_trajectory_and_map(solution, data, robot)

        return {
            'trajectory': trajectory,
            'landmarks': landmarks,
            'omega': omega,
            'xi': xi,
            'solution': solution,
            'success': True
        }


class KalmanFilter:
    """Extended Kalman Filter for online SLAM."""

    def __init__(self, config: SLAMConfig):
        self.config = config

        # State vector: [robot_x, robot_y, landmark1_x, landmark1_y, ...]
        self.state_dim = 2 * (1 + config.num_landmarks)  # robot + landmarks

        # Initial state covariance
        self.P = jnp.eye(self.state_dim) * 100.0

        # Process noise covariance
        self.Q = jnp.eye(2) * config.motion_noise**2

        # Measurement noise covariance
        self.R = jnp.eye(2) * config.measurement_noise**2

    def predict(self, state: jnp.ndarray, motion: Motion) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Prediction step of EKF."""

        # State transition: robot moves, landmarks stay fixed
        F = jnp.eye(self.state_dim)
        F = F.at[0, 0].set(1)  # robot x stays
        F = F.at[1, 1].set(1)  # robot y stays

        # Add motion to robot position
        state_pred = state.copy()
        state_pred = state_pred.at[0].add(motion.dx)
        state_pred = state_pred.at[1].add(motion.dy)

        # Predict covariance
        P_pred = F @ self.P @ F.T + self.Q

        return state_pred, P_pred

    def update(
        self,
        state: jnp.ndarray,
        P: jnp.ndarray,
        measurement: Measurement
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update step of EKF."""

        landmark_idx = 2 * (1 + measurement.landmark_id)  # +1 for robot

        # Predicted measurement
        dx_pred = state[landmark_idx] - state[0]
        dy_pred = state[landmark_idx + 1] - state[1]
        z_pred = jnp.array([dx_pred, dy_pred])

        # Actual measurement
        z = jnp.array([measurement.dx, measurement.dy])

        # Measurement Jacobian H
        H = jnp.zeros((2, self.state_dim))
        H = H.at[0, 0].set(-1)  # dz/dx_robot
        H = H.at[0, landmark_idx].set(1)  # dz/dx_landmark
        H = H.at[1, 1].set(-1)  # dz/dy_robot
        H = H.at[1, landmark_idx + 1].set(1)  # dz/dy_landmark

        # Kalman gain
        S = H @ P @ H.T + self.R
        K = P @ H.T @ jnp.linalg.inv(S)

        # Update state and covariance
        innovation = z - z_pred
        state_updated = state + K @ innovation
        P_updated = (jnp.eye(self.state_dim) - K @ H) @ P

        return state_updated, P_updated

    def initialize_landmark(
        self,
        state: jnp.ndarray,
        P: jnp.ndarray,
        measurement: Measurement
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Initialize a previously unseen landmark."""

        landmark_idx = 2 * (1 + measurement.landmark_id)

        # Initialize landmark position relative to current robot pose
        state = state.at[landmark_idx].set(state[0] + measurement.dx)
        state = state.at[landmark_idx + 1].set(state[1] + measurement.dy)

        # Initialize covariance for landmark
        P = P.at[landmark_idx:landmark_idx+2, landmark_idx:landmark_idx+2].set(
            jnp.eye(2) * 10.0
        )

        return state, P

    def run_online(
        self,
        data: SLAMData,
        robot: Robot,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """Run online EKF SLAM."""

        # Initialize state with robot position and unknown landmarks
        state = jnp.zeros(self.state_dim)
        state = state.at[0].set(robot.position.x)
        state = state.at[1].set(robot.position.y)
        # Landmarks initialized as zeros (unknown)

        seen_landmarks = set()

        trajectory = [Position(state[0], state[1])]

        if show_progress:
            iterator = tqdm(data.measurements, desc="Running EKF SLAM")
        else:
            iterator = data.measurements

        for t, measurements in enumerate(iterator):
            if t < len(data.motions):
                motion = data.motions[t]

                # Prediction step
                state, P = self.predict(state, motion)

                # Update with measurements
                for measurement in measurements:
                    landmark_id = measurement.landmark_id

                    # Initialize landmark if first time seen
                    if landmark_id not in seen_landmarks:
                        state, P = self.initialize_landmark(state, P, measurement)
                        seen_landmarks.add(landmark_id)

                    # Update step
                    state, P = self.update(state, P, measurement)

                trajectory.append(Position(float(state[0]), float(state[1])))

        # Extract final landmark positions
        landmarks = []
        for i, landmark in enumerate(robot.landmarks):
            idx = 2 * (1 + i)
            if idx + 1 < len(state):
                landmarks.append(Landmark(
                    id=landmark.id,
                    position=Position(float(state[idx]), float(state[idx + 1]))
                ))

        return {
            'trajectory': trajectory,
            'landmarks': landmarks,
            'final_state': state,
            'final_covariance': P,
            'success': True
        }
