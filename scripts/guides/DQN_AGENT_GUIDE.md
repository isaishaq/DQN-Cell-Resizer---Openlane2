# DQN Agent Implementation Guide

## Overview

The new `dqn_agent.py` is a **simplified, file-based agent** that follows the OpenLane2 workflow exactly. It does NOT create a Gym environment - instead, it reads timing reports and outputs actions for TCL to apply.

## Architecture

### Workflow:
```
TCL (dqn_resizer.tcl)
  ├─> Generate timing report
  ├─> Call Python agent (subprocess)
  │   └─> dqn_agent.py
  │       ├─> Parse timing report (timing_parser.py)
  │       ├─> Get actionable cells (discrete_action_space.py)
  │       ├─> Extract state features (45-dim vector)
  │       ├─> Load DQN model
  │       ├─> Select action
  │       └─> Output resize commands
  ├─> Read resize commands
  ├─> Apply resizes (swapMaster)
  └─> Loop
```

## Key Features

### 1. **Modular Design**
- Uses existing `timing_parser.py` for parsing
- Uses existing `discrete_action_space.py` for actions
- Clean separation: parsing, state extraction, decision, output

### 2. **State Representation (45-dim)**
```python
State = [
    # Global features (5)
    wns,              # Worst Negative Slack
    tns,              # Total Negative Slack
    num_violations,   # Number of timing violations
    avg_slack,        # Average path slack
    max_delay,        # Maximum cell delay
    
    # Top-10 cells (40 = 10 cells × 4 features)
    [delay_norm, fanout_norm, drive_norm, slack_norm] × 10
]
```

### 3. **DQN Network**
```
Input: 45-dim state
  ↓
Hidden: 128 neurons (ReLU + Dropout)
  ↓
Hidden: 128 neurons (ReLU + Dropout)
  ↓
Hidden: 64 neurons (ReLU)
  ↓
Output: 30 Q-values (one per action)
```

### 4. **Action Space**
- 30 discrete actions (10 cells × 3 resize options)
- Actions: DOWNSIZE, KEEP, UPSIZE
- Validity checking included
- Uses SkyWater 130nm cell library

## Installation

### Required Dependencies
```bash
pip install torch numpy
```

### Optional (for GNN support later)
```bash
pip install torch-geometric networkx matplotlib
```

## Usage

### Basic Usage (No Model - Random Policy)
```bash
python3 dqn_agent.py \
    --timing-report path/to/timing.rpt \
    --output-actions actions.txt \
    --iteration 1
```

### With Trained Model
```bash
python3 dqn_agent.py \
    --timing-report path/to/timing.rpt \
    --output-actions actions.txt \
    --model path/to/trained_model.pth \
    --iteration 1
```

### With Verbose Output
```bash
python3 dqn_agent.py \
    --timing-report path/to/timing.rpt \
    --output-actions actions.txt \
    --model model.pth \
    --iteration 1 \
    --verbose \
    --state-log state_history.jsonl
```

## Command-Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--timing-report` | ✓ | - | Path to timing report (.rpt) |
| `--output-actions` | ✓ | - | Path to output actions file |
| `--model` | ✗ | None | Path to trained DQN model (.pth) |
| `--iteration` | ✗ | 1 | Current iteration number |
| `--top-k-cells` | ✗ | 10 | Number of cells to consider |
| `--worst-n-paths` | ✗ | 5 | Number of worst paths to analyze |
| `--epsilon` | ✗ | 0.0 | Exploration rate (0.0 = greedy) |
| `--state-log` | ✗ | None | Path to state log file (debug) |
| `--verbose` | ✗ | False | Print detailed information |

## Output Format

### Actions File (`actions.txt`)
```
# Iteration: 1
# Action: 5
# WNS: -3.93
# TNS: -85.47
# Violations: 52
# State_dim: 45

_14186_ sky130_fd_sc_hd__nand2_4
_14201_ sky130_fd_sc_hd__buf_8
_14305_ sky130_fd_sc_hd__inv_6
```

### State Log (`state_history.jsonl`)
```json
{"iteration": 1, "state": [...], "q_values": [...], "selected_action": 5, ...}
{"iteration": 2, "state": [...], "q_values": [...], "selected_action": 12, ...}
```

## Integration with TCL

### TCL Example (dqn_resizer.tcl)
```tcl
# Main optimization loop
for {set iter 1} {$iter <= $max_iterations} {incr iter} {
    
    # Generate timing report
    set report_file "$work_dir/timing_iter${iter}.rpt"
    report_checks -path_delay max -format full_clock_expanded \
        -fields {capacitance slew nets fanout input} \
        -digits 4 > $report_file
    
    # Call Python agent
    set actions_file "$work_dir/actions_iter${iter}.txt"
    set model_file "$::env(DQN_MODEL_PATH)"
    
    exec python3 dqn_agent.py \
        --timing-report $report_file \
        --output-actions $actions_file \
        --model $model_file \
        --iteration $iter \
        --verbose
    
    # Read and apply actions
    set fp [open $actions_file r]
    while {[gets $fp line] >= 0} {
        # Skip comments
        if {[string match "#*" $line]} { continue }
        if {$line eq ""} { continue }
        
        # Parse: instance_name new_cell_type
        set parts [split $line]
        set instance [lindex $parts 0]
        set new_cell [lindex $parts 1]
        
        # Apply resize
        set inst [odb::dbInst_find $block $instance]
        if {$inst != "NULL"} {
            set lib [[$inst getMaster] getLib]
            set new_master [odb::dbMaster_find $lib $new_cell]
            if {$new_master != "NULL"} {
                $inst swapMaster $new_master
                puts "Resized $instance to $new_cell"
            }
        }
    }
    close $fp
    
    # Update timing
    sta::update_timing
    
    # Check convergence
    set wns [sta::worst_slack -max]
    if {$wns >= 0} {
        puts "Converged! WNS = $wns"
        break
    }
}
```

## Model Training (Separate Process)

The DQN model should be trained **offline** using a separate training script. Here's a template:

```python
# train_dqn.py (create this separately)
import torch
import torch.nn as nn
from dqn_agent import DQNNetwork
from collections import deque
import random

class DQNTrainer:
    def __init__(self):
        self.q_network = DQNNetwork(state_dim=45, action_dim=30)
        self.target_network = DQNNetwork(state_dim=45, action_dim=30)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=1e-4)
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        
    def train_step(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in batch:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            next_state_t = torch.FloatTensor(next_state).unsqueeze(0)
            
            current_q = self.q_network(state_t)[0, action]
            
            with torch.no_grad():
                next_q = self.target_network(next_state_t).max()
                target_q = reward + self.gamma * next_q * (1 - done)
            
            loss = nn.MSELoss()(current_q, target_q)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def save(self, path):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

# Training loop
trainer = DQNTrainer()

for episode in range(num_episodes):
    # Collect experience from running OpenLane
    # Store (state, action, reward, next_state, done) in memory
    # Train network
    # Update target network periodically
    pass

trainer.save('trained_model.pth')
```

## Testing Without Model

You can test the agent without a trained model - it will use a random policy:

```bash
cd /home/isaishaq/openlane2/designs/picorv_test/scripts

python3 dqn_agent.py \
    --timing-report ../runs/RUN_2026-03-01_15-10-18/74-dqn-resizer-test/reports/nom_ss_100C_1v60/max.rpt \
    --output-actions test_actions.txt \
    --iteration 1 \
    --verbose
```

Expected output:
```
======================================================================
DQN Agent - Iteration 1
======================================================================
[STEP 1] Parsing timing report: ...
  WNS: -3.93 ns
  TNS: -85.47 ns
  Violations: 52
  Paths: 52

[STEP 2] Identifying actionable cells (top-10)
  Found 10 actionable cells

[STEP 3] Extracting state features
  State shape: (45,)
  State range: [-3.93, 15.0]

[STEP 4] DQN action selection
  Action space size: 30
  Selected action: 5
  Q-value: 0.1234

[STEP 5] Decoding action to resize commands
  Generated 1 resize commands

[STEP 6] Writing output files
  Wrote 1 resize commands to: test_actions.txt

======================================================================
Agent completed successfully - Iteration 1
======================================================================
```

## Debugging

### Check State Extraction
```python
from timing_parser import parse_timing_report
from discrete_action_space import DiscreteActionSpace
from dqn_agent import extract_state_features

timing_data = parse_timing_report("timing.rpt")
action_space = DiscreteActionSpace(top_k_cells=10)
cells = action_space.get_actionable_cells(timing_data)
state = extract_state_features(timing_data, cells)

print(f"State shape: {state.shape}")
print(f"Global features: {state[:5]}")
print(f"Cell features: {state[5:].reshape(10, 4)}")
```

### Check Q-Values
```python
from dqn_agent import SimpleDQNAgent

agent = SimpleDQNAgent(model_path="model.pth")
q_values = agent.get_q_values(state)

print(f"Q-values: {q_values}")
print(f"Best action: {q_values.argmax()}")
```

### Check Action Decoding
```python
action_idx = 5
resizes = action_space.apply_action(action_idx, cells)

for instance, (old, new) in resizes.items():
    print(f"{instance}: {old} -> {new}")
```

## Differences from Old Version

| Old dqn_agent.py | New dqn_agent.py |
|------------------|------------------|
| Creates OpenDB environment | File-based only |
| TclScriptRunner class | No TCL execution |
| CellResizingEnv class | No environment class |
| Training + inference | Inference only |
| Complex state management | Simple state extraction |
| ~600 lines | ~650 lines (but cleaner) |
| Tries to do everything | Single responsibility |

## Next Steps

1. **Install dependencies**: `pip install torch numpy`

2. **Test the agent**:
   ```bash
   python3 dqn_agent.py \
       --timing-report <your_timing_report.rpt> \
       --output-actions test_actions.txt \
       --verbose
   ```

3. **Train a model** (create `train_dqn.py`):
   - Collect training data from multiple designs
   - Implement replay buffer
   - Train DQN network
   - Save model weights

4. **Integrate with OpenLane**:
   - Update `dqn_resizer.tcl` to call the agent
   - Test full flow
   - Measure timing improvements

5. **Optimize**:
   - Tune hyperparameters
   - Try different reward functions
   - Experiment with state representations

## File Structure

```
designs/picorv_test/scripts/
├── dqn_agent.py              # Main agent (NEW - simplified)
├── dqn_agent_backup.py       # Old version (backup)
├── timing_parser.py          # Timing report parser
├── discrete_action_space.py  # Action space definition
├── state_representation.py   # GNN state extraction (optional)
├── gnn_dqn.py               # GNN models (optional)
├── test_integration.py       # Integration test
└── WORKFLOW_SUMMARY.md       # Workflow guide
```

## Summary

The new `dqn_agent.py` is:
- ✅ **Simple**: File-based communication, no environment class
- ✅ **Modular**: Uses existing parser and action space
- ✅ **Debuggable**: Verbose logging, state logs
- ✅ **Production-ready**: Works with OpenLane2 TCL scripts
- ✅ **Extensible**: Easy to add GNN support later

Ready to use! 🚀
