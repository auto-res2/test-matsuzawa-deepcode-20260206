"""
Evaluation script for C3-AutoCoT experiments.
Retrieves data from WandB and generates comparison analysis.
"""

import os
import json
import argparse
from typing import List, Dict, Any
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, using local files only")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate C3-AutoCoT experiments")
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Results directory path"
    )
    parser.add_argument(
        "--run_ids",
        type=str,
        required=True,
        help="JSON string list of run IDs to evaluate"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=os.getenv("WANDB_ENTITY", ""),
        help="WandB entity"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="c3-autocot-experiment",
        help="WandB project name"
    )
    return parser.parse_args()


def retrieve_wandb_data(entity: str, project: str, run_id: str) -> Dict[str, Any]:
    """
    Retrieve experimental data from WandB API.
    
    Args:
        entity: WandB entity
        project: WandB project name
        run_id: Run ID to retrieve
    
    Returns:
        Dictionary with run data
    """
    if not WANDB_AVAILABLE:
        print(f"WandB not available, skipping run {run_id}")
        return {}
    
    try:
        api = wandb.Api()
        run_path = f"{entity}/{project}/{run_id}"
        print(f"Fetching run: {run_path}")
        
        run = api.run(run_path)
        
        # Get summary metrics
        summary = dict(run.summary)
        
        # Get config
        config = dict(run.config)
        
        # Get history (all logged metrics)
        history = run.history()
        
        return {
            "run_id": run_id,
            "name": run.name,
            "summary": summary,
            "config": config,
            "history": history.to_dict() if history is not None else {},
            "state": run.state,
            "url": run.url
        }
    except Exception as e:
        print(f"Error retrieving run {run_id}: {e}")
        return {"run_id": run_id, "error": str(e)}


def load_local_metrics(results_dir: str, run_id: str) -> Dict[str, Any]:
    """
    Load metrics from local files.
    
    Args:
        results_dir: Results directory
        run_id: Run ID
    
    Returns:
        Dictionary with metrics
    """
    run_dir = os.path.join(results_dir, run_id)
    metrics_path = os.path.join(run_dir, "metrics.json")
    
    if not os.path.exists(metrics_path):
        print(f"Warning: Metrics file not found for {run_id}")
        return {"run_id": run_id}
    
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    
    # Also load detailed results if available
    results_path = os.path.join(run_dir, "results.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
            metrics["detailed_results"] = results
    
    return {
        "run_id": run_id,
        "metrics": metrics
    }


def export_run_metrics(results_dir: str, run_id: str, data: Dict[str, Any]):
    """
    Export run-specific metrics to file.
    
    Args:
        results_dir: Results directory
        run_id: Run ID
        data: Run data
    """
    run_dir = os.path.join(results_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    # Extract key metrics
    metrics = {}
    
    # From WandB summary
    if "summary" in data:
        summary = data["summary"]
        metrics.update({
            "accuracy_mean": summary.get("accuracy_mean"),
            "accuracy_std": summary.get("accuracy_std"),
            "demo_acceptance_rate": summary.get("demo_acceptance_rate"),
            "num_demos_accepted": summary.get("num_demos_accepted"),
            "mean_reliability": summary.get("mean_reliability"),
            "mean_r_sc": summary.get("mean_r_sc"),
            "mean_r_pi": summary.get("mean_r_pi"),
            "mean_r_cc": summary.get("mean_r_cc"),
        })
    
    # From local metrics
    if "metrics" in data:
        metrics.update(data["metrics"])
    
    # Save to file
    output_path = os.path.join(run_dir, "exported_metrics.json")
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Exported metrics for {run_id} to {output_path}")


def aggregate_metrics(results_dir: str, run_ids: List[str]) -> Dict[str, Any]:
    """
    Aggregate metrics across multiple runs.
    
    Args:
        results_dir: Results directory
        run_ids: List of run IDs
    
    Returns:
        Dictionary with aggregated metrics
    """
    all_metrics = {}
    
    for run_id in run_ids:
        # Try to load exported metrics
        metrics_path = os.path.join(results_dir, run_id, "exported_metrics.json")
        if not os.path.exists(metrics_path):
            # Try original metrics.json
            metrics_path = os.path.join(results_dir, run_id, "metrics.json")
        
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
                all_metrics[run_id] = metrics
    
    # Compute comparison table
    comparison = {
        "run_ids": run_ids,
        "methods": {},
        "accuracies": {},
        "demo_metrics": {}
    }
    
    for run_id, metrics in all_metrics.items():
        comparison["methods"][run_id] = "unknown"  # Will be filled from config
        comparison["accuracies"][run_id] = {
            "mean": metrics.get("accuracy_mean", 0.0),
            "std": metrics.get("accuracy_std", 0.0),
            "min": metrics.get("accuracy_min", 0.0),
            "max": metrics.get("accuracy_max", 0.0),
        }
        comparison["demo_metrics"][run_id] = {
            "acceptance_rate": metrics.get("demo_acceptance_rate", 0.0),
            "mean_reliability": metrics.get("mean_reliability", 0.0),
            "mean_r_sc": metrics.get("mean_r_sc", 0.0),
            "mean_r_pi": metrics.get("mean_r_pi", 0.0),
            "mean_r_cc": metrics.get("mean_r_cc", 0.0),
        }
    
    return comparison


def generate_comparison_figures(
    results_dir: str,
    comparison: Dict[str, Any]
):
    """
    Generate comparison figures.
    
    Args:
        results_dir: Results directory
        comparison: Aggregated comparison data
    """
    output_dir = os.path.join(results_dir, "comparison")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    
    # 1. Accuracy comparison bar plot
    fig, ax = plt.subplots()
    
    run_ids = comparison["run_ids"]
    means = [comparison["accuracies"][rid]["mean"] for rid in run_ids]
    stds = [comparison["accuracies"][rid]["std"] for rid in run_ids]
    
    x_pos = np.arange(len(run_ids))
    ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
    ax.set_xlabel("Run ID")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Comparison Across Methods")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(run_ids, rotation=45, ha="right")
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"), dpi=300)
    plt.close()
    
    print(f"Saved figure: {output_dir}/accuracy_comparison.png")
    
    # 2. Reliability components comparison
    fig, ax = plt.subplots()
    
    components = ["mean_r_sc", "mean_r_pi", "mean_r_cc"]
    component_labels = ["Self-Consistency", "Paraphrase Inv.", "Cycle-Consistency"]
    
    bar_width = 0.25
    x_pos = np.arange(len(run_ids))
    
    for i, (comp, label) in enumerate(zip(components, component_labels)):
        values = []
        for rid in run_ids:
            val = comparison["demo_metrics"][rid].get(comp, 0.0)
            values.append(val if val is not None else 0.0)
        
        ax.bar(x_pos + i * bar_width, values, bar_width, label=label, alpha=0.7)
    
    ax.set_xlabel("Run ID")
    ax.set_ylabel("Reliability Component Score")
    ax.set_title("Reliability Components Comparison")
    ax.set_xticks(x_pos + bar_width)
    ax.set_xticklabels(run_ids, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reliability_components.png"), dpi=300)
    plt.close()
    
    print(f"Saved figure: {output_dir}/reliability_components.png")
    
    # 3. Demo acceptance rate comparison
    fig, ax = plt.subplots()
    
    acceptance_rates = [comparison["demo_metrics"][rid]["acceptance_rate"] for rid in run_ids]
    
    ax.bar(x_pos, acceptance_rates, alpha=0.7, color='green')
    ax.set_xlabel("Run ID")
    ax.set_ylabel("Demo Acceptance Rate")
    ax.set_title("Demonstration Acceptance Rate Comparison")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(run_ids, rotation=45, ha="right")
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "acceptance_rate.png"), dpi=300)
    plt.close()
    
    print(f"Saved figure: {output_dir}/acceptance_rate.png")


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Parse run IDs
    run_ids = json.loads(args.run_ids)
    print(f"Evaluating runs: {run_ids}")
    
    # Retrieve data from WandB if available
    if WANDB_AVAILABLE and args.wandb_entity:
        print("\n=== Retrieving data from WandB ===")
        for run_id in run_ids:
            wandb_data = retrieve_wandb_data(
                args.wandb_entity,
                args.wandb_project,
                run_id
            )
            if wandb_data and "error" not in wandb_data:
                export_run_metrics(args.results_dir, run_id, wandb_data)
    
    # Always load local metrics as backup
    print("\n=== Loading local metrics ===")
    for run_id in run_ids:
        local_data = load_local_metrics(args.results_dir, run_id)
        if "metrics" in local_data:
            export_run_metrics(args.results_dir, run_id, local_data)
    
    # Aggregate metrics
    print("\n=== Aggregating metrics ===")
    comparison = aggregate_metrics(args.results_dir, run_ids)
    
    # Save aggregated metrics
    output_dir = os.path.join(args.results_dir, "comparison")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "aggregated_metrics.json"), "w") as f:
        json.dump(comparison, f, indent=2)
    
    print(f"Saved aggregated metrics to {output_dir}/aggregated_metrics.json")
    
    # Generate comparison figures
    print("\n=== Generating comparison figures ===")
    generate_comparison_figures(args.results_dir, comparison)
    
    # Print summary table
    print("\n=== Summary ===")
    print(f"{'Run ID':<30} {'Accuracy':<20} {'Acceptance Rate':<20}")
    print("-" * 70)
    for run_id in run_ids:
        acc = comparison["accuracies"][run_id]
        demo = comparison["demo_metrics"][run_id]
        print(f"{run_id:<30} {acc['mean']:.4f} +/- {acc['std']:.4f}   {demo['acceptance_rate']:.4f}")
    
    print("\n=== Evaluation Complete ===")


if __name__ == "__main__":
    main()
