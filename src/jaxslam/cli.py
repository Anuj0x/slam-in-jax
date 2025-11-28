"""Command-line interface for Modern SLAM using Typer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from .config import SLAMConfig, DEFAULT_CONFIG, FAST_CONFIG, ACCURATE_CONFIG
from .core import SLAMSystem, Position
from .algorithms import GraphSLAM, KalmanFilter
from .visualization import SLAMVisualizer

app = typer.Typer(
    name="modern-slam",
    help="Modern Simultaneous Localization and Mapping system",
    add_completion=False,
)

console = Console()


@app.command()
def run(
    num_steps: int = typer.Option(50, "--steps", "-n", help="Number of simulation steps"),
    world_size: float = typer.Option(100.0, "--world-size", "-w", help="Size of the world"),
    num_landmarks: int = typer.Option(20, "--landmarks", "-l", help="Number of landmarks"),
    algorithm: str = typer.Option("graph", "--algorithm", "-a",
                                 help="SLAM algorithm: graph or kalman"),
    config_preset: str = typer.Option("default", "--config", "-c",
                                    help="Configuration preset: default, fast, accurate"),
    show_progress: bool = typer.Option(True, "--progress/--no-progress", help="Show progress"),
    visualize: bool = typer.Option(True, "--visualize/--no-visualize", help="Create visualizations"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o",
                                            help="Output directory for results"),
    save_data: bool = typer.Option(False, "--save-data", help="Save simulation data to JSON"),
):
    """Run SLAM simulation and analysis."""

    # Select configuration
    if config_preset == "fast":
        config = FAST_CONFIG
    elif config_preset == "accurate":
        config = ACCURATE_CONFIG
    else:
        config = DEFAULT_CONFIG

    # Override config with CLI options
    config = config.copy()
    config.world_size = world_size
    config.num_landmarks = num_landmarks
    config.show_progress = show_progress

    console.print(f"[bold blue]Modern SLAM Simulation[/bold blue]")
    console.print(f"Algorithm: {algorithm}")
    console.print(f"World size: {world_size}")
    console.print(f"Landmarks: {num_landmarks}")
    console.print(f"Steps: {num_steps}")
    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing SLAM system...", total=None)

        # Create SLAM system
        slam_system = SLAMSystem(config)
        visualizer = SLAMVisualizer(config)

        progress.update(task, description="Generating trajectory...")

        # Generate trajectory data
        data = slam_system.generate_trajectory(num_steps)

        progress.update(task, description="Running SLAM algorithm...")

        # Run SLAM algorithm
        if algorithm == "graph":
            slam_algo = GraphSLAM(config)
            results = slam_algo.run(data, slam_system.robot, show_progress=False)
        elif algorithm == "kalman":
            slam_algo = KalmanFilter(config)
            results = slam_algo.run_online(data, slam_system.robot, show_progress=False)
        else:
            console.print(f"[red]Unknown algorithm: {algorithm}[/red]")
            raise typer.Exit(1)

        progress.update(task, description="Creating visualizations...")

        # Create visualizations if requested
        if visualize:
            # World plot
            world_fig = visualizer.plot_world_2d(
                slam_system.robot,
                results['trajectory'],
                results['landmarks'],
                data,
                title=f"SLAM Results - {algorithm.upper()}"
            )

            # Error analysis
            error_fig = visualizer.plot_error_analysis(
                [Position(slam_system.robot.position.x, slam_system.robot.position.y)] * len(results['trajectory']),
                results['trajectory'],
                slam_system.robot.landmarks,
                results['landmarks']
            )

            progress.update(task, description="Saving results...")

            # Save outputs
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)

                # Save plots as HTML
                world_fig.write_html(output_dir / "world_plot.html")
                error_fig.write_html(output_dir / "error_analysis.html")

                if save_data:
                    # Save data as JSON
                    output_data = {
                        'config': config.dict(),
                        'trajectory': [{'x': p.x, 'y': p.y} for p in results['trajectory']],
                        'landmarks': [{'id': lm.id, 'x': lm.position.x, 'y': lm.position.y}
                                    for lm in results['landmarks']],
                        'true_landmarks': [{'id': lm.id, 'x': lm.position.x, 'y': lm.position.y}
                                         for lm in slam_system.robot.landmarks],
                        'measurements': [[{'id': m.landmark_id, 'dx': m.dx, 'dy': m.dy}
                                        for m in measurements] for measurements in data.measurements],
                        'motions': [{'dx': m.dx, 'dy': m.dy} for m in data.motions]
                    }

                    with open(output_dir / "slam_results.json", 'w') as f:
                        json.dump(output_data, f, indent=2)

                console.print(f"Results saved to: {output_dir}")

        progress.update(task, completed=True)

    # Display results summary
    table = Table(title="SLAM Results Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Algorithm", algorithm.upper())
    table.add_row("Timesteps", str(len(data.measurements)))
    table.add_row("Landmarks", str(len(slam_system.robot.landmarks)))
    table.add_row("Measurements", str(sum(len(m) for m in data.measurements)))
    table.add_row("Success", "✓" if results.get('success', False) else "✗")

    console.print(table)

    if visualize and not output_dir:
        console.print("\n[green]Opening visualizations...[/green]")
        world_fig.show()
        error_fig.show()


@app.command()
def config(
    preset: str = typer.Option("default", "--preset", "-p",
                             help="Configuration preset to display")
):
    """Display or modify SLAM configuration."""

    if preset == "default":
        config = DEFAULT_CONFIG
    elif preset == "fast":
        config = FAST_CONFIG
    elif preset == "accurate":
        config = ACCURATE_CONFIG
    else:
        console.print(f"[red]Unknown preset: {preset}[/red]")
        console.print("Available presets: default, fast, accurate")
        raise typer.Exit(1)

    console.print(f"[bold]Configuration: {preset.upper()}[/bold]")

    table = Table()
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Description", style="yellow")

    for field_name, field in config.__fields__.items():
        value = getattr(config, field_name)
        description = field.field_info.description or ""
        table.add_row(field_name, str(value), description)

    console.print(table)


@app.command()
def benchmark(
    algorithms: list[str] = typer.Option(["graph", "kalman"], "--algorithms", "-a",
                                       help="Algorithms to benchmark"),
    num_steps: int = typer.Option(30, "--steps", "-n", help="Number of simulation steps"),
    num_runs: int = typer.Option(3, "--runs", "-r", help="Number of benchmark runs"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o",
                                             help="Output file for benchmark results"),
):
    """Benchmark different SLAM algorithms."""

    import time
    from statistics import mean, stdev

    console.print(f"[bold blue]Benchmarking SLAM Algorithms[/bold blue]")
    console.print(f"Runs per algorithm: {num_runs}")
    console.print(f"Steps per run: {num_steps}")
    console.print()

    results = {}

    for algorithm in algorithms:
        console.print(f"[cyan]Benchmarking {algorithm.upper()}...[/cyan]")

        run_times = []
        successes = 0

        for run in range(num_runs):
            start_time = time.time()

            try:
                config = FAST_CONFIG.copy()
                config.num_landmarks = 10  # Smaller for faster benchmarking

                slam_system = SLAMSystem(config)
                data = slam_system.generate_trajectory(num_steps)

                if algorithm == "graph":
                    slam_algo = GraphSLAM(config)
                    result = slam_algo.run(data, slam_system.robot, show_progress=False)
                elif algorithm == "kalman":
                    slam_algo = KalmanFilter(config)
                    result = slam_algo.run_online(data, slam_system.robot, show_progress=False)
                else:
                    continue

                if result.get('success', False):
                    successes += 1

                run_time = time.time() - start_time
                run_times.append(run_time)

            except Exception as e:
                console.print(f"[red]Error in run {run + 1}: {e}[/red]")
                run_times.append(float('inf'))

        results[algorithm] = {
            'mean_time': mean(run_times) if run_times else float('inf'),
            'std_time': stdev(run_times) if len(run_times) > 1 else 0,
            'success_rate': successes / num_runs,
            'run_times': run_times
        }

    # Display results
    table = Table(title="Benchmark Results")
    table.add_column("Algorithm", style="cyan")
    table.add_column("Mean Time (s)", style="green")
    table.add_column("Std Dev (s)", style="yellow")
    table.add_column("Success Rate", style="magenta")

    for algorithm, result in results.items():
        mean_time = ".3f" if result['mean_time'] != float('inf') else "∞"
        std_time = ".3f" if result['std_time'] != 0 else "N/A"
        success_rate = ".1%"

        table.add_row(algorithm.upper(), mean_time, std_time, success_rate)

    console.print(table)

    # Save results if requested
    if output_file:
        output_data = {
            'benchmark_config': {
                'algorithms': algorithms,
                'num_steps': num_steps,
                'num_runs': num_runs
            },
            'results': results
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        console.print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    app()
