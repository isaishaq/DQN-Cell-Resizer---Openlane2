# DQN-Based Cell Resizer for OpenLane 2

This directory contains a Deep Q-Learning based cell resizer that can optimize timing and power by intelligently resizing cells.

## Files

- **`dqn_flow.py`** - The main DQN flow definition (extends SequentialFlow)
- **`dqn_resizer_step.py`** - The DQN step implementation (OpenLane Step)
- **`dqn_agent.py`** - The DQN agent and environment implementation
- **`dqn_resizer.tcl`** - TCL script that interfaces with OpenROAD
- **`run_dqn_only.py`** - Standalone script to run DQN on existing runs

## Quick Start

### 1. Complete a Baseline Run

First, run your design through the normal flow up to or past STAPostPNR:

```bash
python -m openlane designs/picorv_test/config.json --tag baseline
```

### 2. Run DQN Optimization

#### Option A: Using the DQN Flow (Recommended)

```bash
python -m openlane designs/picorv_test/config.json \
    --flow designs/picorv_test/scripts/dqn_flow.py \
    --last-run \
    --tag dqn_experiment \
    -p DQN_MAX_ITERATIONS=30 \
    -p DQN_POWER_WEIGHT=0.3
```

#### Option B: Using the Standalone Script

```bash
python designs/picorv_test/scripts/run_dqn_only.py \
    --run-dir designs/picorv_test/runs/baseline \
    --max-iterations 30 \
    --power-weight 0.3
```

### 3. Compare Results

```bash
# View metrics
cat designs/picorv_test/runs/dqn_experiment/final/metrics.json

# Compare with baseline
python -m openlane.common.metrics compare \
    designs/picorv_test/runs/baseline/final/metrics.json \
    designs/picorv_test/runs/dqn_experiment/final/metrics.json
```

## Configuration Variables

### DQN-Specific Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DQN_MODEL_PATH` | str | None | Path to pre-trained model |
| `DQN_MAX_ITERATIONS` | int | 50 | Maximum optimization iterations |
| `DQN_TARGET_SLACK` | float | 0.0 | Target slack in ns |
| `DQN_POWER_WEIGHT` | float | 0.3 | Power optimization weight (0-1) |
| `DQN_TRAINING_MODE` | bool | False | Enable training mode |
| `DQN_EPSILON` | float | 0.1 | Exploration rate (training only) |
| `DQN_LEARNING_RATE` | float | 1e-4 | Learning rate (training only) |
| `DQN_BATCH_SIZE` | int | 32 | Batch size for replay |
| `DQN_MEMORY_SIZE` | int | 10000 | Replay memory size |
| `DQN_SAVE_MODEL` | bool | True | Save model after run |

## Usage Scenarios

### Inference Mode (Using Pre-trained Model)

```bash
# Use a trained model for quick optimization
python -m openlane designs/picorv_test/config.json \
    --flow designs/picorv_test/scripts/dqn_flow.py \
    --last-run \
    --tag dqn_inference \
    -p DQN_MODEL_PATH=models/dqn_resizer_v1.pth \
    -p DQN_TRAINING_MODE=False \
    -p DQN_MAX_ITERATIONS=20
```

### Training Mode (Learn from Current Design)

```bash
# Train the model on your design
python -m openlane designs/picorv_test/config.json \
    --flow designs/picorv_test/scripts/dqn_flow.py \
    --last-run \
    --tag dqn_training \
    -p DQN_TRAINING_MODE=True \
    -p DQN_MAX_ITERATIONS=100 \
    -p DQN_SAVE_MODEL_PATH=models/dqn_trained_on_picorv32a.pth
```

### Aggressive Power Optimization

```bash
# Prioritize power reduction
python -m openlane designs/picorv_test/config.json \
    --flow designs/picorv_test/scripts/dqn_flow.py \
    --last-run \
    --tag dqn_power_opt \
    -p DQN_POWER_WEIGHT=0.7 \
    -p DQN_TARGET_SLACK=0.1
```

### Aggressive Timing Optimization

```bash
# Prioritize timing closure
python -m openlane designs/picorv_test/config.json \
    --flow designs/picorv_test/scripts/dqn_flow.py \
    --last-run \
    --tag dqn_timing_opt \
    -p DQN_POWER_WEIGHT=0.1 \
    -p DQN_TARGET_SLACK=0.0
```

## Integration with Classic Flow

To add DQN resizer to your custom flow:

```python
from openlane.flows import SequentialFlow
from openlane.steps import OpenROAD
from designs.picorv_test.scripts.dqn_resizer_step import DQNResizer

class MyFlow(SequentialFlow):
    Steps = [
        # ... existing steps ...
        OpenROAD.DetailedRouting,
        OpenROAD.RCX,
        OpenROAD.STAPostPNR,
        
        # Add DQN optimization here
        DQNResizer,
        
        # Re-run STA to verify
        OpenROAD.STAPostPNR,
        
        # ... remaining steps ...
    ]
```

## Running Specific Steps

To run from/to specific steps:

```bash
# Run only DQN resizer
python -m openlane designs/picorv_test/config.json \
    --flow designs/picorv_test/scripts/dqn_flow.py \
    --from DQNResizer \
    --to DQNResizer \
    --last-run

# Run DQN and subsequent STA
python -m openlane designs/picorv_test/config.json \
    --flow designs/picorv_test/scripts/dqn_flow.py \
    --from DQNResizer \
    --to OpenROAD.STAPostPNR \
    --last-run
```

## Troubleshooting

### "No state_out.json found"

Make sure your baseline run completed at least up to `OpenROAD.STAPostPNR`. Check:

```bash
ls designs/picorv_test/runs/baseline/*stapostpnr*/
```

### "Module not found: dqn_resizer_step"

Ensure all files are in the same directory:
- `dqn_flow.py`
- `dqn_resizer_step.py`
- `dqn_agent.py`
- `dqn_resizer.tcl`

### "Missing required input: SPEF"

The DQN resizer needs parasitics. Make sure `OpenROAD.RCX` ran successfully in your baseline:

```bash
ls designs/picorv_test/runs/baseline/*rcx*/*.spef
```

## How It Works

1. **State Representation**: Extracts timing (WNS, TNS), power, and per-cell features
2. **Action Space**: Each action resizes a cell to a different drive strength
3. **Reward Function**: Balances timing improvement vs power increase
4. **DQN Agent**: Neural network learns Q-values for state-action pairs
5. **Environment**: Interfaces with OpenROAD ODB database and OpenSTA

## Performance Tips

- Start with **inference mode** using a pre-trained model (faster)
- Use **training mode** for design-specific optimization (slower but better)
- Reduce `DQN_MAX_ITERATIONS` for quick tests (10-20)
- Increase for thorough optimization (100+)
- Adjust `DQN_POWER_WEIGHT` based on your PPA priorities

## Next Steps

1. Train a model on multiple designs to get a good baseline
2. Fine-tune hyperparameters (learning rate, epsilon, batch size)
3. Extend the state features (add congestion, wire length, etc.)
4. Implement multi-objective optimization (Pareto front)
5. Add buffer insertion alongside cell resizing

## References

- OpenLane 2 Documentation: https://openlane2.readthedocs.io/
- OpenROAD ODB Python API: https://openroad.readthedocs.io/
- DQN Paper: Mnih et al., "Playing Atari with Deep Reinforcement Learning"
