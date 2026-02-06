"""
Main orchestrator for C3-AutoCoT experiments.
Uses Hydra for configuration management.
"""

import os
import sys
from omegaconf import DictConfig, OmegaConf
import hydra

from .train import train


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Main entry point for experiment execution.
    
    Args:
        cfg: Hydra configuration
    """
    print("=" * 80)
    print("C3-AutoCoT Experiment Runner")
    print("=" * 80)
    
    # Print configuration
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Configure mode-specific settings
    if cfg.mode == "trial":
        print("\n*** TRIAL MODE ***")
        print("Running with reduced dataset sizes and disabled WandB")
        
        # Override settings for trial mode
        cfg.dataset.demo_pool_size = min(cfg.dataset.demo_pool_size, 50)
        cfg.dataset.test_size = min(cfg.dataset.test_size, 20)
        cfg.method_params.max_candidates_per_cluster = min(
            cfg.method_params.max_candidates_per_cluster, 3
        )
        cfg.evaluation.seeds = [cfg.evaluation.seeds[0]]  # Use only first seed
        cfg.wandb.mode = "disabled"
        
        print(f"  Demo pool size: {cfg.dataset.demo_pool_size}")
        print(f"  Test size: {cfg.dataset.test_size}")
        print(f"  Max candidates: {cfg.method_params.max_candidates_per_cluster}")
        print(f"  Seeds: {cfg.evaluation.seeds}")
    elif cfg.mode == "full":
        print("\n*** FULL MODE ***")
        print("Running with full dataset and WandB logging")
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}. Must be 'full' or 'trial'")
    
    # Ensure results directory exists
    os.makedirs(cfg.results_dir, exist_ok=True)
    
    # Run training
    print("\nStarting experiment...")
    try:
        metrics = train(cfg)
        print("\nExperiment completed successfully!")
        return metrics
    except Exception as e:
        print(f"\nExperiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
