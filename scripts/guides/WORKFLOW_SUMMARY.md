# Recommended Workflow for DQN Cell Sizing in OpenLane2

## Summary

**You DON'T need Gym** - OpenLane2 provides its own framework for custom optimization steps. Your workflow should follow the OpenLane2 Step pattern.

## Recommended Architecture

### **1. Two-Level Iteration Pattern**

```
Outer Loop (TCL - dqn_resizer.tcl):
  └─> Manages OpenROAD database, timing analysis, and design updates
      │
      └─> Inner Loop (Python - dqn_agent.py):
          └─> Receives timing data, makes decisions, returns actions
```

### **2. File-Based Communication**

Instead of Gym's programmatic API, use files:

```
TCL ──creates──> timing_report.rpt
                      ↓
                Python reads & parses
                      ↓
                Python makes decision
                      ↓
TCL ──reads──── actions.txt ←──creates── Python
      ↓
TCL applies resizes
      ↓
Loop continues
```

## Complete Workflow

### Phase 1: OpenLane Invokes Your Step

```python
# Your dqn_resizer_step.py (already created)
@Step.factory.register()
class DQNResizer(OpenROADStep):
    id = "OpenROAD.DQNResizer"
    
    def run(self, state_in: State, **kwargs):
        # OpenLane calls this
        # It will execute your TCL script
        return super().run(state_in, env=env, **kwargs)
```

**What you get:**
- Input: Design loaded in OpenROAD (ODB format)
- Input: Parasitics loaded (SPEF)
- OpenLane handles all file I/O automatically

### Phase 2: TCL Orchestration Loop

```tcl
# Your dqn_resizer.tcl (already exists)

# Main loop in TCL
for {set iter 1} {$iter <= 50} {incr iter} {
    
    # 1. Run OpenROAD timing analysis
    report_checks > timing_iter${iter}.rpt
    
    # 2. Call Python agent (subprocess)
    exec python3 dqn_agent.py \
        --timing-report timing_iter${iter}.rpt \
        --output-actions actions_iter${iter}.txt \
        --iteration $iter
    
    # 3. Read and apply resizes from Python
    apply_cell_resizes actions_iter${iter}.txt
    
    # 4. Check termination
    if {[get_wns] >= 0} { break }
}
```

**What this does:**
- Controls the optimization loop
- Generates timing reports
- Spawns Python process for decisions
- Applies Python's decisions to design
- Repeats until done

### Phase 3: Python Agent Decision-Making

```python
# Your dqn_agent.py (needs implementation)

def main():
    args = parse_args()  # --timing-report, --output-actions, etc.
    
    # 1. Parse timing report
    timing_data = parse_timing_report(args.timing_report)
    wns = timing_data['global_metrics']['wns']
    
    # 2. Identify actionable cells
    action_space = DiscreteActionSpace(top_k_cells=10)
    actionable_cells = action_space.get_actionable_cells(timing_data)
    
    # 3. Create state vector
    state = extract_state(timing_data, actionable_cells)
    
    # 4. DQN selects action
    action_idx = dqn_model.select_action(state)
    
    # 5. Decode to resize commands
    resizes = action_space.apply_action(action_idx, actionable_cells)
    
    # 6. Write actions for TCL
    with open(args.output_actions, 'w') as f:
        for instance, (old, new) in resizes.items():
            f.write(f"{instance} {new}\n")
```

**What this does:**
- Loads timing data
- Uses your DiscreteActionSpace (already implemented!)
- Runs DQN inference
- Outputs resize commands

## Key Differences from Gym

| Gym Environment | OpenLane2 Approach |
|-----------------|-------------------|
| `env.reset()` | TCL loads design from ODB |
| `env.step(action)` | TCL applies action, re-runs STA |
| `env.render()` | TCL `report_checks` |
| Episode loop | TCL for-loop |
| Reward calculation | Can be in Python or TCL |
| State observation | Parse timing report |
| Action application | TCL `swapMaster` |

## Implementation Checklist

### ✅ Already Done
- [x] `dqn_resizer_step.py` - OpenLane Step definition
- [x] `discrete_action_space.py` - Action space with cell library
- [x] `dqn_resizer.tcl` - Existing TCL skeleton

### ⚠️ Needs Completion
- [ ] **`timing_parser.py`** - I created this, test it!
- [ ] **`dqn_agent.py`** - Needs actual DQN model
- [ ] **DQN Model** - Train the neural network

### 📝 Detailed Tasks

#### Task 1: Test Timing Parser
```bash
cd designs/picorv_test/scripts
python3 timing_parser.py \
    ../runs/RUN_2026-03-01_15-10-18/74-dqn-resizer-test/reports/nom_ss_100C_1v60/max.rpt \
    parsed_timing.json

# Verify output
cat parsed_timing.json | jq '.global_metrics'
```

#### Task 2: Integrate Action Space with Agent
```python
# In dqn_agent.py
from discrete_action_space import DiscreteActionSpace
from timing_parser import parse_timing_report

# Use your existing action space
action_space = DiscreteActionSpace(top_k_cells=10)
timing_data = parse_timing_report("timing.rpt")
cells = action_space.get_actionable_cells(timing_data)
```

#### Task 3: Implement DQN Model
```python
import torch
import torch.nn as nn

class DQNNetwork(nn.Module):
    def __init__(self, state_dim=45, action_dim=30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# In dqn_agent.py
model = DQNNetwork(state_dim=45, action_dim=30)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Select action
with torch.no_grad():
    q_values = model(torch.FloatTensor(state))
    action = q_values.argmax().item()
```

#### Task 4: Training Pipeline (Separate Script)
```python
# train_dqn.py (separate from the flow)

for episode in range(num_episodes):
    # Run OpenLane up to DQN step
    # Get initial state
    # Loop: select action, apply, get reward
    # Store in replay buffer
    # Train network
    # Save checkpoint
```

## Testing Strategy

### Test 1: Dry Run (No DQN)
```bash
# Modify dqn_agent.py to output random valid actions
# Run OpenLane flow
# Verify TCL can apply actions
```

### Test 2: With Placeholder Model
```bash
# Use random or heuristic policy
# Verify full loop works
# Check timing improves or at least doesn't break
```

### Test 3: With Trained Model
```bash
# Train on small designs
# Load trained model
# Run production flow
# Measure: WNS improvement, iterations, runtime
```

## Advantages of This Approach

1. **Leverages OpenLane2 Infrastructure**
   - Automatic file handling
   - Integration with flow
   - Standard step pattern

2. **Clean Separation of Concerns**
   - TCL: Design manipulation
   - Python: Decision making
   - Files: Communication

3. **Easy Debugging**
   - Each iteration produces files
   - Can inspect intermediate states
   - Can replay specific iterations

4. **Flexible**
   - Can swap DQN for other algorithms
   - Can modify action space independently
   - Can add more features to state

## Workflow Execution

### During Development
```bash
# Run just your step on existing ODB
cd designs/picorv_test
openroad
> read_db runs/latest/odb/design.odb
> source scripts/dqn_resizer.tcl
```

### In OpenLane Flow
```json
// config.json
{
  "DESIGN_NAME": "picorv32a",
  "CUSTOM_STEPS": {
    "post_cts": ["OpenROAD.DQNResizer"]
  },
  "DQN_MAX_ITERATIONS": 50,
  "DQN_TARGET_SLACK": 0.0
}
```

```bash
python3 -m openlane designs/picorv_test/config.json
```

## Summary: What You Need to Implement

1. **Complete `dqn_agent.py`:**
   - Load trained DQN model
   - Integrate timing parser
   - Integrate action space
   - Select actions
   - Output resize commands

2. **Train DQN Model (Offline):**
   - Collect training data
   - Implement replay buffer
   - Train network
   - Save weights

3. **Test Integration:**
   - Run on test designs
   - Verify timing improvements
   - Measure convergence

**No Gym needed!** OpenLane2 IS your environment.

## Quick Start Command

```bash
cd /home/isaishaq/openlane2/designs/picorv_test

# 1. Test timing parser
python3 scripts/timing_parser.py \
    runs/RUN_2026-03-01_15-10-18/74-dqn-resizer-test/reports/nom_ss_100C_1v60/max.rpt

# 2. Test action space
python3 scripts/discrete_action_space.py

# 3. Run OpenLane with your step
python3 -m openlane config.json

# Watch the logs for DQN iterations
tail -f runs/latest/logs/OpenROAD.DQNResizer.log
```

Good luck! 🚀
