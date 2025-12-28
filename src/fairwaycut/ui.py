"""UI utilities for FairwayCut using Rich."""

from contextlib import contextmanager
from typing import Optional, Dict, Any, Generator

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TaskID,
)
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich import box

# Global console instance
console = Console()


class RichProgressHandler:
    """
    Handles Rich progress bars for multi-stage processing.
    
    Maps the simple (stage, current, total) callback to structured Rich progress bars.
    """
    
    def __init__(self, console: Console, verbose: bool = False):
        self.console = console
        self.verbose = verbose
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True, # Remove bars when done
        )
        # Track task IDs by stage name
        self.tasks: Dict[str, TaskID] = {}
        self.active_stage = None
        
        # Stage display names
        self.stage_names = {
            "audio_extraction": "🎵 Extracting audio",
            "audio_detection": "🔍 Detecting impacts",
            "pose_estimation": "🦴 Estimating poses",
            "fusion": "🔗 Fusing signals",
            "complete": "✅ Complete",
        }

    def callback(self, stage: str, current: int, total: int):
        """
        Progress callback matching the signature expected by detector.
        
        Args:
            stage: Current processing stage identifier.
            current: Current item index / progress.
            total: Total items / target.
        """
        # Handle stage transitions
        if stage != self.active_stage:
            self.active_stage = stage
            description = self.stage_names.get(stage, stage)
            
            # If we haven't seen this stage, create a task
            if stage not in self.tasks:
                self.tasks[stage] = self.progress.add_task(
                    description, 
                    total=total if total > 0 else None,
                    start=True
                )
            
        # Update progress
        if stage in self.tasks:
            self.progress.update(self.tasks[stage], completed=current, total=total)
            
            # Auto-complete if done
            if current >= total and total > 0:
                self.progress.stop_task(self.tasks[stage])

    @contextmanager
    def live(self) -> Generator["RichProgressHandler", None, None]:
        """Context manager to run the progress display."""
        with self.progress:
            yield self


def print_banner(version: str):
    """Print the FairwayCut banner."""
    grid = Table.grid(expand=True)
    grid.add_column(justify="center", ratio=1)
    grid.add_row(
        Panel(
            f"[bold green]FairwayCut[/bold green] - Golf Swing Auto-Segmentation\nVersion {version}",
            style="green",
            box=box.ROUNDED,
        )
    )
    console.print(grid)


def print_swing_summary(swings: list):
    """Print a nice summary table of detected swings."""
    table = Table(title="📊 Swing Summary", box=box.ROUNDED)
    table.add_column("Swing #", style="cyan", justify="right")
    table.add_column("Impact Time", style="magenta")
    table.add_column("Confidence", style="green")
    
    for swing in swings:
        table.add_row(
            f"#{swing.swing_id}",
            f"{swing.impact_time:.2f}s",
            f"{swing.combined_confidence:.0%}"
        )
    
    console.print(table)
