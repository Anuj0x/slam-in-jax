"""Modern visualization tools for SLAM using Plotly and Matplotlib."""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .core import Position, Landmark, Robot, SLAMData
from .config import SLAMConfig


class SLAMVisualizer:
    """Modern visualization system for SLAM results."""

    def __init__(self, config: SLAMConfig):
        self.config = config
        self.backend = config.visualization_backend

    def plot_world_2d(
        self,
        robot: Robot,
        trajectory: Optional[List[Position]] = None,
        estimated_landmarks: Optional[List[Landmark]] = None,
        slam_data: Optional[SLAMData] = None,
        title: str = "SLAM World"
    ) -> go.Figure:
        """Create an interactive 2D plot of the SLAM world."""

        fig = go.Figure()

        # Plot world boundaries
        fig.add_shape(
            type="rect",
            x0=0, y0=0, x1=self.config.world_size, y1=self.config.world_size,
            line=dict(color="black", width=2),
            fillcolor="lightgray",
            opacity=0.1
        )

        # Plot true landmarks
        if robot.landmarks:
            landmark_x = [l.position.x for l in robot.landmarks]
            landmark_y = [l.position.y for l in robot.landmarks]
            fig.add_trace(go.Scatter(
                x=landmark_x, y=landmark_y,
                mode='markers',
                marker=dict(size=12, symbol='x', color='purple'),
                name='True Landmarks',
                hovertemplate='Landmark %{text}<br>x: %{x:.2f}<br>y: %{y:.2f}',
                text=[f'#{l.id}' for l in robot.landmarks]
            ))

        # Plot estimated landmarks
        if estimated_landmarks:
            est_x = [l.position.x for l in estimated_landmarks]
            est_y = [l.position.y for l in estimated_landmarks]
            fig.add_trace(go.Scatter(
                x=est_x, y=est_y,
                mode='markers',
                marker=dict(size=10, symbol='diamond', color='orange'),
                name='Estimated Landmarks',
                hovertemplate='Est Landmark %{text}<br>x: %{x:.2f}<br>y: %{y:.2f}',
                text=[f'#{l.id}' for l in estimated_landmarks]
            ))

        # Plot trajectory
        if trajectory:
            traj_x = [p.x for p in trajectory]
            traj_y = [p.y for p in trajectory]
            fig.add_trace(go.Scatter(
                x=traj_x, y=traj_y,
                mode='lines+markers',
                line=dict(color='blue', width=3),
                marker=dict(size=6, color='blue'),
                name='Estimated Trajectory',
                hovertemplate='Step %{text}<br>x: %{x:.2f}<br>y: %{y:.2f}',
                text=list(range(len(trajectory)))
            ))

        # Plot current robot position
        fig.add_trace(go.Scatter(
            x=[robot.position.x], y=[robot.position.y],
            mode='markers',
            marker=dict(size=15, symbol='circle', color='red'),
            name='Robot',
            hovertemplate='Robot<br>x: %{x:.2f}<br>y: %{y:.2f}'
        ))

        # Plot measurement ranges if slam_data provided
        if slam_data and slam_data.measurements:
            for t, measurements in enumerate(slam_data.measurements[-1:]):  # Last timestep only
                for measurement in measurements:
                    landmark = robot.landmarks[measurement.landmark_id]
                    fig.add_trace(go.Scatter(
                        x=[robot.position.x, landmark.position.x],
                        y=[robot.position.y, landmark.position.y],
                        mode='lines',
                        line=dict(color='green', width=1, dash='dot'),
                        name='Measurements',
                        showlegend=t == 0  # Only show in legend once
                    ))

        # Update layout
        fig.update_layout(
            title=title,
            xaxis=dict(range=[-5, self.config.world_size + 5], autorange=False),
            yaxis=dict(range=[-5, self.config.world_size + 5], autorange=False),
            xaxis_title="X Position",
            yaxis_title="Y Position",
            width=800,
            height=600,
            showlegend=True
        )

        return fig

    def plot_error_analysis(
        self,
        true_trajectory: List[Position],
        estimated_trajectory: List[Position],
        true_landmarks: List[Landmark],
        estimated_landmarks: List[Landmark]
    ) -> go.Figure:
        """Create error analysis plots."""

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Trajectory Position Error",
                "Trajectory Error Over Time",
                "Landmark Position Error",
                "Landmark Error Distribution"
            )
        )

        # Trajectory position errors
        if len(true_trajectory) == len(estimated_trajectory):
            errors = []
            for true_pos, est_pos in zip(true_trajectory, estimated_trajectory):
                error = true_pos.distance_to(est_pos)
                errors.append(error)

            fig.add_trace(
                go.Scatter(
                    x=[p.x for p in true_trajectory],
                    y=[p.y for p in true_trajectory],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=errors,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Error", x=0.45, y=0.8, len=0.4)
                    ),
                    name='Trajectory Errors'
                ),
                row=1, col=1
            )

            # Error over time
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(errors))),
                    y=errors,
                    mode='lines+markers',
                    name='Trajectory Error vs Time'
                ),
                row=1, col=2
            )

        # Landmark position errors
        landmark_errors = []
        landmark_positions = []
        for true_lm, est_lm in zip(true_landmarks, estimated_landmarks):
            error = true_lm.position.distance_to(est_lm.position)
            landmark_errors.append(error)
            landmark_positions.append((true_lm.position.x, true_lm.position.y))

        if landmark_positions:
            lm_x, lm_y = zip(*landmark_positions)
            fig.add_trace(
                go.Scatter(
                    x=lm_x, y=lm_y,
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=landmark_errors,
                        colorscale='Plasma',
                        showscale=True,
                        colorbar=dict(title="Error", x=0.95, y=0.8, len=0.4)
                    ),
                    name='Landmark Errors'
                ),
                row=2, col=1
            )

        # Landmark error distribution
        if landmark_errors:
            fig.add_trace(
                go.Histogram(
                    x=landmark_errors,
                    nbinsx=20,
                    name='Landmark Error Distribution'
                ),
                row=2, col=2
            )

        fig.update_layout(
            height=800,
            width=1000,
            title_text="SLAM Error Analysis",
            showlegend=False
        )

        return fig

    def plot_constraint_matrix(
        self,
        omega: np.ndarray,
        title: str = "Information Matrix (Omega)"
    ) -> go.Figure:
        """Visualize the constraint matrix."""

        # Create heatmap of the matrix
        fig = go.Figure(data=go.Heatmap(
            z=omega,
            colorscale='Viridis',
            hoverongaps=False
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Variable Index",
            yaxis_title="Variable Index",
            width=600,
            height=600
        )

        return fig

    def create_dashboard(
        self,
        slam_results: Dict[str, Any],
        robot: Robot,
        slam_data: SLAMData
    ) -> go.Figure:
        """Create a comprehensive SLAM dashboard."""

        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                "SLAM World View", "Trajectory", "Landmark Errors",
                "Information Matrix", "Error Over Time", "Performance Metrics"
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                [{"type": "heatmap"}, {"type": "scatter"}, {"type": "table"}]
            ]
        )

        # World view
        trajectory = slam_results.get('trajectory', [])
        landmarks = slam_results.get('landmarks', [])

        if trajectory:
            fig.add_trace(
                go.Scatter(
                    x=[p.x for p in trajectory],
                    y=[p.y for p in trajectory],
                    mode='lines+markers',
                    name='Trajectory'
                ),
                row=1, col=1
            )

        # Trajectory plot
        if trajectory:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(trajectory))),
                    y=[p.x for p in trajectory],
                    mode='lines',
                    name='X Position'
                ),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(trajectory))),
                    y=[p.y for p in trajectory],
                    mode='lines',
                    name='Y Position'
                ),
                row=1, col=2
            )

        # Placeholder for other plots
        fig.add_trace(go.Scatter(x=[], y=[]), row=1, col=3)
        fig.add_trace(go.Scatter(x=[], y=[]), row=2, col=1)
        fig.add_trace(go.Scatter(x=[], y=[]), row=2, col=2)

        # Performance metrics table
        metrics = [
            ["Metric", "Value"],
            ["Timesteps", str(len(slam_data.measurements))],
            ["Landmarks", str(len(robot.landmarks))],
            ["Measurements", str(sum(len(m) for m in slam_data.measurements))],
            ["Success", "True" if slam_results.get('success', False) else "False"]
        ]

        fig.add_trace(
            go.Table(
                header=dict(values=["Metric", "Value"]),
                cells=dict(values=list(zip(*metrics[1:])) if len(metrics) > 1 else [[], []])
            ),
            row=2, col=3
        )

        fig.update_layout(
            height=800,
            width=1200,
            title_text="SLAM Analysis Dashboard",
            showlegend=True
        )

        return fig

    def matplotlib_plot_world(
        self,
        robot: Robot,
        trajectory: Optional[List[Position]] = None,
        estimated_landmarks: Optional[List[Landmark]] = None,
        title: str = "SLAM World"
    ) -> plt.Figure:
        """Create a matplotlib version for static plots."""

        fig, ax = plt.subplots(figsize=(10, 8))

        # World boundaries
        ax.add_patch(plt.Rectangle((0, 0), self.config.world_size, self.config.world_size,
                                 fill=False, edgecolor='black', linewidth=2, alpha=0.3))

        # True landmarks
        if robot.landmarks:
            lm_x = [l.position.x for l in robot.landmarks]
            lm_y = [l.position.y for l in robot.landmarks]
            ax.scatter(lm_x, lm_y, c='purple', marker='x', s=100, label='True Landmarks')

        # Estimated landmarks
        if estimated_landmarks:
            est_x = [l.position.x for l in estimated_landmarks]
            est_y = [l.position.y for l in estimated_landmarks]
            ax.scatter(est_x, est_y, c='orange', marker='D', s=80, label='Estimated Landmarks')

        # Trajectory
        if trajectory:
            traj_x = [p.x for p in trajectory]
            traj_y = [p.y for p in trajectory]
            ax.plot(traj_x, traj_y, 'b-', linewidth=2, marker='o', markersize=4,
                   label='Trajectory')

        # Robot position
        ax.scatter(robot.position.x, robot.position.y, c='red', s=200, marker='o',
                  label='Robot', zorder=10)

        ax.set_xlim(-5, self.config.world_size + 5)
        ax.set_ylim(-5, self.config.world_size + 5)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig

    def show(self, fig: go.Figure) -> None:
        """Display the figure based on backend."""
        if self.backend == 'plotly':
            fig.show()
        else:
            # For matplotlib figures
            plt.show()
