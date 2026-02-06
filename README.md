# C3-AutoCoT: Cycle-Consistent & Paraphrase-Invariant Reliability Auto-CoT

Complete implementation of the C3-AutoCoT method for constructing reliable chain-of-thought demonstrations without labels.

## Overview

C3-AutoCoT enhances Auto-CoT by adding three reliability checks for demonstration selection:
1. **Self-Consistency (r_sc)**: Re-derive the problem multiple times and check answer agreement
2. **Paraphrase Invariance (r_pi)**: Generate paraphrases and check answer stability
3. **Cycle-Consistency (r_cc)**: Reconstruct the question from reasoning to verify grounding

## Installation

```bash
uv pip install -e .
```

## Usage

### Training (Full Mode)

Run a complete experiment with full dataset and WandB logging:

```bash
uv run python -u -m src.main run=c3-autocot-proposed results_dir=./results mode=full
```

Run the baseline method:

```bash
uv run python -u -m src.main run=pir-autocot-baseline results_dir=./results mode=full
```

### Training (Trial Mode)

Run a quick trial with reduced dataset sizes and disabled WandB:

```bash
uv run python -u -m src.main run=c3-autocot-proposed results_dir=./results mode=trial
```

### Evaluation

After running experiments, evaluate and compare results:

```bash
uv run python -m src.evaluate \
    --results_dir=./results \
    --run_ids='["c3-autocot-proposed", "pir-autocot-baseline"]'
```

This will:
- Retrieve metrics from WandB (if available)
- Export per-run metrics to `{results_dir}/{run_id}/exported_metrics.json`
- Generate aggregated comparison in `{results_dir}/comparison/aggregated_metrics.json`
- Create visualization figures in `{results_dir}/comparison/`

## Configuration

Experiment configurations are in `config/runs/`:

- `c3-autocot-proposed.yaml`: Proposed C3-AutoCoT method with all three reliability components
- `pir-autocot-baseline.yaml`: Baseline PIR-AutoCoT without cycle-consistency

### Key Parameters

- `method_params.num_clusters`: Number of demonstration clusters (default: 8)
- `method_params.reliability_threshold`: Minimum reliability score to accept demo (default: 0.30)
- `method_params.num_samples`: Samples per reliability check (default: 4)
- `method_params.num_paraphrases`: Number of paraphrases to generate (default: 2)
- `method_params.use_cycle_consistency`: Enable cycle-consistency check (default: true for C3, false for PIR)

## Project Structure

```
.
├── config/
│   ├── config.yaml              # Main Hydra config
│   └── runs/
│       ├── c3-autocot-proposed.yaml
│       └── pir-autocot-baseline.yaml
├── src/
│   ├── __init__.py
│   ├── main.py                  # Hydra orchestrator
│   ├── train.py                 # Training and evaluation logic
│   ├── model.py                 # CoT model wrapper
│   ├── preprocess.py            # Dataset handling
│   └── evaluate.py              # Evaluation and visualization
├── pyproject.toml               # Dependencies
└── README.md
```

## Results

Results are saved in the specified `results_dir`:

```
results/
├── {run_id}/
│   ├── metrics.json             # Aggregated metrics
│   ├── results.json             # Detailed per-example results
│   └── exported_metrics.json    # Metrics exported from WandB
└── comparison/
    ├── aggregated_metrics.json  # Cross-run comparison
    ├── accuracy_comparison.png
    ├── reliability_components.png
    └── acceptance_rate.png
```

## Environment Variables

- `WANDB_API_KEY`: WandB API key (required for WandB logging)
- `WANDB_ENTITY`: WandB entity/username (optional, can be set in config)

## Expected Results

On SVAMP with Qwen3-8B and k=8 demonstrations:

- **PIR-AutoCoT**: 0.27-0.36 accuracy
- **C3-AutoCoT**: 0.30-0.40 accuracy (expected +0.02 to +0.05 improvement)

## Citation

This implements the method described in the research hypothesis for C3-AutoCoT, which adds cycle-consistency grounding to PIR-AutoCoT to detect "plausible but ungrounded" demonstrations.