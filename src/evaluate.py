"""
Evaluation script for aggregating metrics across multiple runs.
Fetches data from WandB and generates comparison figures.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import wandb
import matplotlib.pyplot as plt
import numpy as np


def fetch_run_data(entity: str, project: str, run_id: str) -> Dict:
    """
    Fetch run data from WandB API.
    
    Args:
        entity: WandB entity
        project: WandB project
        run_id: Run ID to fetch
    
    Returns:
        Dictionary with config, summary, and history
    """
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    
    # Get config
    config = dict(run.config)
    
    # Get summary metrics
    summary = dict(run.summary)
    
    # Get history (logged metrics over time)
    history = run.history()
    
    return {
        "config": config,
        "summary": summary,
        "history": history.to_dict("records") if not history.empty else [],
    }


def save_per_run_metrics(results_dir: Path, run_id: str, data: Dict):
    """Save metrics for a single run."""
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract key metrics
    metrics = {
        "run_id": run_id,
        "accuracy": data["summary"].get("accuracy", 0.0),
        "correct_count": data["summary"].get("correct_count", 0),
        "total_count": data["summary"].get("total_count", 0),
    }
    
    # Save metrics
    metrics_file = run_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Saved metrics for {run_id}: {metrics_file}")
    
    # Generate per-run figure: accuracy over time
    if data["history"]:
        history = data["history"]
        problem_ids = [h.get("problem_id", i) for i, h in enumerate(history)]
        accuracies = [h.get("accuracy", 0.0) for h in history]
        
        if accuracies:
            plt.figure(figsize=(10, 6))
            plt.plot(problem_ids, accuracies, marker='o', markersize=2)
            plt.xlabel("Problem ID")
            plt.ylabel("Running Accuracy")
            plt.title(f"Accuracy Over Time: {run_id}")
            plt.grid(True, alpha=0.3)
            
            fig_file = run_dir / "accuracy_over_time.png"
            plt.savefig(fig_file, dpi=100, bbox_inches='tight')
            plt.close()
            print(f"Saved figure: {fig_file}")
    
    return metrics


def create_comparison_figures(results_dir: Path, metrics_by_run: Dict[str, Dict]):
    """Generate comparison figures across runs."""
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    run_ids = list(metrics_by_run.keys())
    accuracies = [metrics_by_run[rid]["accuracy"] for rid in run_ids]
    
    # Bar chart comparing accuracies
    plt.figure(figsize=(10, 6))
    colors = ['#2ecc71' if 'proposed' in rid else '#3498db' for rid in run_ids]
    bars = plt.bar(range(len(run_ids)), accuracies, color=colors)
    plt.xlabel("Run ID")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison Across Methods")
    plt.xticks(range(len(run_ids)), run_ids, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    fig_file = comparison_dir / "accuracy_comparison.png"
    plt.savefig(fig_file, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison figure: {fig_file}")


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate and compare experimental runs")
    parser.add_argument("--results_dir", type=str, required=True, help="Results directory")
    parser.add_argument("--run_ids", type=str, required=True, help="JSON list of run IDs")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    run_ids = json.loads(args.run_ids)
    
    print(f"=== Evaluating {len(run_ids)} runs ===")
    print(f"Run IDs: {run_ids}")
    
    # Get WandB config from environment or first run config
    wandb_entity = os.environ.get("WANDB_ENTITY", "airas")
    wandb_project = os.environ.get("WANDB_PROJECT", "2026-02-12-2")
    
    # Fetch data for each run
    all_metrics = {}
    for run_id in run_ids:
        print(f"\nFetching data for {run_id}...")
        try:
            data = fetch_run_data(wandb_entity, wandb_project, run_id)
            metrics = save_per_run_metrics(results_dir, run_id, data)
            all_metrics[run_id] = metrics
        except Exception as e:
            print(f"Warning: Failed to fetch data for {run_id}: {e}")
            # Try to load from local file if WandB fetch fails
            local_metrics_file = results_dir / run_id / "metrics.json"
            if local_metrics_file.exists():
                with open(local_metrics_file, "r") as f:
                    metrics = json.load(f)
                all_metrics[run_id] = metrics
                print(f"Loaded metrics from local file: {local_metrics_file}")
    
    if not all_metrics:
        print("Error: No metrics available for any run")
        return
    
    # Calculate aggregated metrics
    proposed_runs = {rid: m for rid, m in all_metrics.items() if "proposed" in rid}
    baseline_runs = {rid: m for rid, m in all_metrics.items() if "comparative" in rid}
    
    best_proposed = max(proposed_runs.items(), key=lambda x: x[1]["accuracy"]) if proposed_runs else (None, {"accuracy": 0.0})
    best_baseline = max(baseline_runs.items(), key=lambda x: x[1]["accuracy"]) if baseline_runs else (None, {"accuracy": 0.0})
    
    gap = best_proposed[1]["accuracy"] - best_baseline[1]["accuracy"]
    
    aggregated = {
        "primary_metric": "accuracy",
        "metrics_by_run": all_metrics,
        "best_proposed": {
            "run_id": best_proposed[0],
            "accuracy": best_proposed[1]["accuracy"],
        } if best_proposed[0] else None,
        "best_baseline": {
            "run_id": best_baseline[0],
            "accuracy": best_baseline[1]["accuracy"],
        } if best_baseline[0] else None,
        "gap": gap,
    }
    
    # Save aggregated metrics
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    aggregated_file = comparison_dir / "aggregated_metrics.json"
    with open(aggregated_file, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nSaved aggregated metrics: {aggregated_file}")
    
    # Print summary
    print("\n=== Summary ===")
    if best_proposed[0]:
        print(f"Best proposed: {best_proposed[0]} (accuracy: {best_proposed[1]['accuracy']:.4f})")
    if best_baseline[0]:
        print(f"Best baseline: {best_baseline[0]} (accuracy: {best_baseline[1]['accuracy']:.4f})")
    print(f"Gap: {gap:.4f}")
    
    # Generate comparison figures
    create_comparison_figures(results_dir, all_metrics)
    
    print("\n=== Evaluation complete ===")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
