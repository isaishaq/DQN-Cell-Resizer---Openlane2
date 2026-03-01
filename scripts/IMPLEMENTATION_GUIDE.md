# DQN Cell Sizing - Complete Implementation Guide

This guide explains the complete workflow for integrating DRL-based cell sizing in OpenLane2.

## **Architecture Overview**

```
┌─────────────────────────────────────────────────────┐
│           OpenLane2 Flow Execution                   │
│  Synthesis → Placement → CTS → DQNResizer → Route   │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  dqn_resizer_step.py  │ ← OpenLane Step (Python)
         │  (OpenROADStep)       │
         └───────────┬───────────┘
                     │ Invokes
                     ▼
         ┌───────────────────────┐
         │   dqn_resizer.tcl     │ ← TCL Script (OpenROAD)
         └───────────┬───────────┘
                     │ Main Loop
        ┌────────────┴───────────────┐
        │                            │
        ▼                            ▼
┌───────────────┐          ┌─────────────────┐
│  OpenROAD/STA │          │  dqn_agent.py   │
│  Timing       │◄────────►│  (Python DQN)   │
│  Analysis     │          │                 │
└───────────────┘          └─────────────────┘
        │                            │
        │                            │
        ▼                            ▼
┌───────────────┐          ┌─────────────────┐
│ timing_parser │          │ discrete_action │
│ .rpt → JSON   │─────────►│ _space.py       │
└───────────────┘          └─────────────────┘
                                    │
                                    ▼
                          ┌─────────────────┐
                          │  Apply Resizes  │
                          │  (Update ODB)   │
                          └─────────────────┘
```

## **File Structure**

```
designs/picorv_test/scripts/
├── dqn_resizer_step.py           # Main: OpenLane Step definition
├── dqn_resizer.tcl                # TCL: Orchestrator + OpenROAD interface
├── dqn_agent.py                   # Python: DQN model + action selection
├── discrete_action_space.py       # Python: Action space definition
├── timing_parser.py               # Python: Parse OpenSTA reports
├── cell_library.py                # Python: Cell library info
└── models/
    └── dqn_model.pth              # Trained DQN model weights
```

## **Data Flow - Step by Step**

### **1. OpenLane Invokes Your Step**

When OpenLane flow reaches your step:

```python
# In flow configuration (JSON)
{
  "steps": [
    "Classic.Placement",
    "Classic.CTS",
    "OpenROAD.DQNResizer",  # ← Your step
    "Classic.GlobalRouting"
  ]
}
```

### **2. DQNResizer Step Initialization**

```python
# dqn_resizer_step.py
class DQNResizer(OpenROADStep):
    def run(self, state_in: State, **kwargs):
        # Set environment variables for TCL
        env["DQN_MODEL_PATH"] = self.config["DQN_MODEL_PATH"]
        env["DQN_MAX_ITERATIONS"] = str(self.config["DQN_MAX_ITERATIONS"])
        env["DQN_AGENT_SCRIPT"] = "path/to/dqn_agent.py"
        
        # Execute TCL script
        return super().run(state_in, env=env, **kwargs)
```

**What happens:**
- OpenLane loads the design (ODB + SPEF)
- Step sets up environment variables
- Invokes `dqn_resizer.tcl`

### **3. TCL Script Main Loop**

```tcl
# dqn_resizer.tcl

# Load design
read_current_odb
load_rsz_corners

# Main optimization loop
for {set iter 1} {$iter <= $max_iterations} {incr iter} {
    
    # 3a. Run timing analysis
    report_checks -path_delay max \
                  -format full_clock_expanded \
                  > timing_report_iter${iter}.rpt
    
    # 3b. Invoke Python agent
    set action_file "actions_iter${iter}.txt"
    exec python3 $DQN_AGENT_SCRIPT \
        --timing-report timing_report_iter${iter}.rpt \
        --output-action $action_file \
        --iteration $iter
    
    # 3c. Apply resizes from action file
    apply_cell_resizes $action_file
    
    # 3d. Check if timing met
    if {[get_wns] >= $target_slack} {
        break
    }
}

# Save final design
write_db design_final.odb
```

**What happens:**
- TCL runs timing analysis using OpenROAD/OpenSTA
- Generates `.rpt` file
- Calls Python agent script
- Reads resize commands from Python output
- Applies resizes to design database
- Repeats until timing met or max iterations

### **4. Python Agent Execution (Per Iteration)**

```python
# dqn_agent.py (simplified flow)

def run_iteration():
    # 4a. Parse timing report
    timing_data = parse_timing_report("timing_report.rpt")
    
    # 4b. Extract state
    wns = timing_data['wns']
    tns = timing_data['tns']
    critical_cells = get_critical_cells(timing_data)
    state = create_state_vector(wns, tns, critical_cells)
    
    # 4c. Get actionable cells
    action_space = DiscreteActionSpace(top_k_cells=10)
    actionable_cells = action_space.get_actionable_cells(timing_data)
    
    # 4d. DQN selects action
    valid_mask = action_space.get_valid_actions_mask(actionable_cells)
    action_idx = dqn_model.select_action(state, valid_mask)
    
    # 4e. Decode action to resizes
    resize_commands = action_space.apply_action(action_idx, actionable_cells)
    # Returns: {"fanout123": ("buf_4", "buf_8"), ...}
    
    # 4f. Save resize commands for TCL
    with open("actions.txt", "w") as f:
        for instance, (old, new) in resize_commands.items():
            f.write(f"{instance} {new}\n")
    
    return resize_commands
```

**What happens:**
- Parse timing report → structured data
- Identify top-K most critical cells
- Create state vector for DQN
- DQN model predicts best action
- Decode action → cell resize commands
- Write commands to file for TCL to execute

### **5. Action Space Processing**

```python
# discrete_action_space.py

class DiscreteActionSpace:
    def __init__(self, top_k_cells=10):
        self.top_k = top_k_cells
        # Action space: top_k_cells × 3 (downsize/keep/upsize)
        self.n_actions = top_k_cells * 3
    
    def get_actionable_cells(self, timing_data):
        # Extract cells from worst timing paths
        critical_paths = sorted(timing_data['paths'], 
                               key=lambda p: p['slack'])[:10]
        
        # Score cells by criticality
        cells = []
        for path in critical_paths:
            for cell in path['cells']:
                if is_resizable(cell):
                    cells.append(cell)
        
        # Return top-K most critical
        return sorted(cells, key=lambda c: c.criticality)[:self.top_k]
    
    def action_to_cell_resize(self, action_idx, actionable_cells):
        # Decode: action_idx → (cell_index, resize_direction)
        cell_idx = action_idx // 3
        resize_action = action_idx % 3  # 0=down, 1=keep, 2=up
        
        cell = actionable_cells[cell_idx]
        
        # Get new cell size
        if resize_action == 2:  # Upsize
            new_size = get_larger_size(cell.current_size)
        elif resize_action == 0:  # Downsize
            new_size = get_smaller_size(cell.current_size)
        else:  # Keep
            new_size = cell.current_size
        
        return {cell.instance: (cell.cell_type, new_cell_type)}
```

**What happens:**
- Identify top-K cells from critical paths
- Map discrete action index to (cell, resize_direction)
- Validate resize is legal
- Generate new cell type name
- Return resize commands

### **6. TCL Applies Resizes**

```tcl
# Back in dqn_resizer.tcl

proc apply_cell_resizes {resize_file} {
    set fp [open $resize_file r]
    
    while {[gets $fp line] >= 0} {
        # Parse: instance_name new_cell_type
        set parts [split $line]
        set instance [lindex $parts 0]
        set new_cell [lindex $parts 1]
        
        # Get the DB instance
        set db_inst [[ord::get_db_block] findInst $instance]
        set db_master [[ord::get_db] findMaster $new_cell]
        
        # Swap the master (resize the cell)
        $db_inst swapMaster $db_master
        
        puts "[INFO] Resized: $instance → $new_cell"
    }
    
    close $fp
}
```

**What happens:**
- Read resize commands from file
- For each command, update design database
- Swap cell master (resize operation)
- Design is now modified

### **7. Loop Continues**

After resizing:
1. TCL re-runs timing analysis
2. Gets new WNS/TNS
3. If timing met → exit loop
4. If not → continue to next iteration (back to step 3)

### **8. Final Output**

When loop completes:
```tcl
# Save final design
write_db final_design.odb

# Generate final reports
report_checks -path_delay max > final_timing.rpt
report_power > final_power.rpt
```

OpenLane continues with final design to next steps (routing, etc.)

## **Key Design Decisions**

### **Where Does DQN Model Live?**

**Option 1: Separate Python Process (Current)**
- TCL spawns Python subprocess
- Python loads model, makes decision, exits
- Communication via files

**Pros:** Simple, works with any framework
**Cons:** Model loading overhead each iteration

**Option 2: Python Server**
- Python runs as HTTP/gRPC server
- TCL sends requests to server
- Server keeps model loaded

**Pros:** Faster, no reload overhead
**Cons:** More complex setup

**Option 3: OpenROAD Python API**
- Use OpenROAD's Python API directly
- No TCL needed
- Pure Python orchestration

**Pros:** Cleaner, more Pythonic
**Cons:** Different integration pattern

### **Training vs Inference**

**Training Mode:**
```python
# Collect experiences
for episode in range(num_episodes):
    state = env.reset()
    while not done:
        action = agent.select_action(state, epsilon=0.1)  # Explore
        next_state, reward, done = env.step(action)
        
        # Store experience
        replay_buffer.add(state, action, reward, next_state, done)
        
        # Train network
        if len(replay_buffer) > batch_size:
            agent.train(replay_buffer.sample())
        
        state = next_state
```

**Inference Mode:**
```python
# Load pre-trained model
agent.load("trained_model.pth")
agent.eval()

# Pure exploitation
action = agent.select_action(state, epsilon=0.0)  # No explore
```

### **State Representation**

```python
state = [
    # Global metrics (5)
    wns / clock_period,            # Normalized WNS
    tns / 100.0,                   # Normalized TNS  
    num_violations / 100.0,        # Normalized violation count
    current_area / initial_area,   # Area ratio
    iteration / max_iterations,    # Progress
    
    # Per-cell features (10 cells × 4 features = 40)
    # For each of top-10 cells:
    drive_strength / 16.0,         # Normalized drive
    fanout / 20.0,                 # Normalized fanout
    cell_delay / 2.0,              # Normalized delay
    slack_contribution / 5.0,      # Normalized criticality
]
```

**Total state dimension: 45**

### **Reward Function**

```python
def calculate_reward(prev_wns, new_wns, prev_area, new_area):
    # Main reward: slack improvement
    slack_improvement = new_wns - prev_wns
    slack_reward = slack_improvement * 10.0
    
    # Penalty: area increase
    area_ratio = (new_area - prev_area) / prev_area
    area_penalty = -area_ratio * 1.0
    
    # Bonus: fix all violations
    completion_bonus = 100.0 if new_wns >= 0 else 0.0
    
    # Penalty: make things worse
    worse_penalty = -50.0 if new_wns < prev_wns else 0.0
    
    return slack_reward + area_penalty + completion_bonus + worse_penalty
```

## **Execution Example**

### **1. Start OpenLane Flow**

```bash
cd /home/isaishaq/openlane2
python3 -m openlane designs/picorv_test/config.json
```

### **2. Flow Reaches DQN Step**

```
[INFO] Running step: OpenROAD.DQNResizer
[INFO] Loading design from ODB
[INFO] Running dqn_resizer.tcl
```

### **3. TCL Iteration 1**

```
[INFO] DQN Iteration 1/50
[INFO] Running timing analysis...
[INFO] Current WNS: -3.93 ns
[INFO] Invoking DQN agent...
[INFO] python3 dqn_agent.py --timing-report timing_iter1.rpt
```

### **4. Python Agent Iteration 1**

```
[INFO] Parsing timing report...
[INFO] WNS: -3.93 ns, TNS: -85.47 ns
[INFO] Found 10 actionable cells
[INFO] DQN selected action: 7 (Upsize cell fanout1291)
[INFO] Resize: fanout1291 buf_4 → buf_8
[INFO] Commands saved to actions_iter1.txt
```

### **5. TCL Applies Resizes**

```
[INFO] Applying cell resizes...
[INFO] Resized: fanout1291 → sky130_fd_sc_hd__buf_8
[INFO] Applied 1 resize(s)
[INFO] New WNS: -3.12 ns (improvement: 0.81 ns)
```

### **6. TCL Iteration 2**

```
[INFO] DQN Iteration 2/50
[INFO] Current WNS: -3.12 ns
[INFO] Invoking DQN agent...
```

... continues until timing is met or max iterations ...

### **7. Final Output**

```
[SUCCESS] Timing met! WNS = 0.15 ns
[INFO] Total iterations: 23
[INFO] Total resizes: 47
[INFO] Final design saved to: design_dqn_resized.odb
[INFO] DQN optimization complete
```

### **8. OpenLane Continues**

```
[INFO] Continuing to next step: GlobalRouting
```

## **Testing the Implementation**

### **Test 1: Parse Timing Report**

```bash
cd designs/picorv_test/scripts
python3 timing_parser.py ../runs/latest/reports/max.rpt output.json
```

### **Test 2: Action Space**

```python
from discrete_action_space import DiscreteActionSpace
import json

# Load parsed timing data
with open('output.json') as f:
    timing_data = json.load(f)

# Create action space
action_space = DiscreteActionSpace(top_k_cells=10)

# Get actionable cells
cells = action_space.get_actionable_cells(timing_data)
print(f"Actionable cells: {len(cells)}")

# Test action
action = 5
resizes = action_space.action_to_cell_resize(action, cells)
print(f"Action {action} → {resizes}")
```

### **Test 3: Run Single Iteration**

```bash
# Set up environment
export DQN_MODEL_PATH="models/dqn_model.pth"
export DQN_AGENT_SCRIPT="scripts/dqn_agent.py"

# Run agent
python3 dqn_agent.py \
    --odb design.odb \
    --work-dir test_iter \
    --model models/dqn_model.pth \
    --max-iterations 1 \
    --target-slack 0.0 \
    --training 0
```

## **Next Steps**

1. **Implement Actual DQN Model**
   - Replace placeholder in `dqn_agent.py`
   - Use PyTorch/TensorFlow
   - Define network architecture

2. **Training Pipeline**
   - Collect training data from multiple designs
   - Implement replay buffer
   - Train DQN with proper rewards

3. **Integration Testing**
   - Test on small designs first
   - Validate timing improvements
   - Check area/power impact

4. **Production Deployment**
   - Load pre-trained model
   - Set training_mode=False
   - Include in production flow

## **Troubleshooting**

### **Issue: Python agent not found**
```
export DQN_AGENT_SCRIPT="/full/path/to/dqn_agent.py"
```

### **Issue: Import errors**
```bash
cd designs/picorv_test/scripts
export PYTHONPATH=$PWD:$PYTHONPATH
```

### **Issue: Model file not found**
```
mkdir -p models
# Create placeholder or trained model
```

### **Issue: Timing report parsing fails**
Check report format matches parser expectations

### **Issue: Invalid cell resizes**
Verify cell library definitions match your PDK
