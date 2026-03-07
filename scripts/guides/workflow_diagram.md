# DQN Cell Sizing Workflow in OpenLane2

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    OpenLane2 Flow                            │
│  (Synthesis → Placement → CTS → DQN Resizer → Route)        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  DQNResizer Step      │  ← Your custom Step
         │  (dqn_resizer_step.py)│
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │   TCL Script          │
         │ (dqn_resizer.tcl)     │  ← Interfaces with OpenROAD
         └───────────┬───────────┘
                     │
        ┌────────────┴───────────┐
        │                        │
        ▼                        ▼
┌──────────────┐        ┌──────────────┐
│ Timing       │        │ Python DQN   │
│ Analysis     │◄──────►│ Agent        │
│ (OpenSTA)    │        │ (dqn_agent.py)│
└──────────────┘        └──────────────┘
        │                        │
        ▼                        ▼
┌──────────────┐        ┌──────────────┐
│ Timing       │        │ Action       │
│ Parser       │───────►│ Selection    │
│(.rpt→JSON)   │        │(discrete)    │
└──────────────┘        └──────────────┘
                                │
                                ▼
                        ┌──────────────┐
                        │ Apply Resize │
                        │ Commands     │
                        │(update ODB)  │
                        └──────────────┘
                                │
                                ▼
                        ┌──────────────┐
                        │ Re-run       │
                        │ Timing       │
                        │ Analysis     │
                        └──────────────┘
                                │
            ┌───────────────────┴────────────────┐
            │                                    │
            ▼                                    ▼
      Violations                           Timing
      Fixed?                               Met?
            │                                    │
            NO                                  YES
            │                                    │
            └─────► Next Iteration               │
                                                 ▼
                                          Continue Flow
```

## Execution Flow

### Phase 1: Initialization (Once per design)
1. Load design into OpenROAD (ODB + SPEF)
2. Run initial timing analysis
3. Parse timing report
4. Initialize DQN agent (load model or start training)

### Phase 2: Iterative Optimization Loop
```
For iteration in range(MAX_ITERATIONS):
    
    1. Current State Assessment:
       - Run timing analysis (OpenSTA via TCL)
       - Parse timing report → structured JSON
       - Extract WNS, TNS, critical paths
    
    2. Action Selection:
       - Python agent receives timing state
       - Identifies actionable cells (top-K from critical paths)
       - DQN model predicts best action
       - Decode action → resize commands
    
    3. Action Application:
       - Generate TCL commands to resize cells
       - Execute via OpenROAD: dbResize or swap_cell
       - Update design database (ODB)
    
    4. Reward Calculation:
       - Re-run timing analysis
       - Calculate slack improvement
       - Compute area/power penalty
       - Store experience (state, action, reward, next_state)
    
    5. Model Update (Training Mode):
       - Add experience to replay buffer
       - Sample batch and update Q-network
    
    6. Termination Check:
       - If timing met: exit loop
       - If max iterations: exit loop
       - Else: continue to next iteration
```

### Phase 3: Finalization
1. Save final design state (ODB/DEF)
2. Generate final timing reports
3. Save DQN model checkpoint
4. Continue OpenLane flow (routing, etc.)

## Data Flow

```
Timing Report (.rpt)
    │
    ▼
Timing Parser (Python)
    │
    ▼
Structured Data (Dict/JSON)
{
  "paths": [...],
  "cells": [...],
  "metrics": {"wns": -3.93, ...}
}
    │
    ▼
Action Space Module
    │
    ├──► Identify actionable cells (top-K critical)
    ├──► Build action mask (valid resizes)
    └──► Create state vector
    │
    ▼
DQN Agent (PyTorch/TF)
    │
    ├──► Forward pass: Q(state, action)
    ├──► Action selection: argmax or ε-greedy
    └──► Return: action_index
    │
    ▼
Action Decoder
    │
    └──► Convert action → resize commands
         {"fanout123": ("buf_4", "buf_8"), ...}
    │
    ▼
TCL Script Generator
    │
    └──► Generate OpenROAD commands
         "dbResize fanout123 sky130_fd_sc_hd__buf_8"
    │
    ▼
OpenROAD Execution
    │
    └──► Apply to design database
    │
    ▼
Updated Design (ODB) → Loop back to Timing Analysis
```

## Training vs Inference

### Training Mode:
- Enable exploration (ε-greedy)
- Store experiences in replay buffer
- Update neural network weights
- Save checkpoints periodically
- Multiple episodes across different designs

### Inference Mode:
- Load pre-trained model
- Pure exploitation (greedy actions)
- No model updates
- Single-pass optimization
- Use in production flow

## Integration Points with OpenLane2

### 1. Step Registration
```python
@Step.factory.register()
class DQNResizer(OpenROADStep):
    id = "OpenROAD.DQNResizer"
    # Inputs/Outputs/Config
```

### 2. Flow Integration
```json
// In flow configuration
{
  "flow": [
    "Classic.Synthesis",
    "Classic.Floorplan",
    "Classic.Placement",
    "Classic.CTS",
    "OpenROAD.DQNResizer",  ← Your step here
    "Classic.GlobalRouting",
    "Classic.DetailedRouting"
  ]
}
```

### 3. State Management
- Input: ODB (design database) + SPEF (parasitics)
- Output: Modified ODB with resized cells
- OpenLane2 handles file I/O automatically

## Key Design Decisions

### 1. Where to Run Python Code?
**Option A:** Python from TCL (subprocess)
- TCL script calls Python agent
- Returns action via stdout/file
- Simple but slower

**Option B:** HTTP/RPC Server
- Python agent runs as server
- TCL sends requests via HTTP
- Faster for multiple iterations

**Option C:** Hybrid (Recommended)
- TCL does timing analysis only
- Python script orchestrates everything
- Uses OpenROAD Python API directly

### 2. State Persistence
- Save design state between iterations
- Use OpenROAD's write/read_db commands
- Checkpoint every N iterations

### 3. Parallelization
- Can evaluate multiple actions in parallel
- Fork design database
- Useful for tree search methods

## Files You Need to Implement

1. ✅ `discrete_action_space.py` (Done)
2. ⚠️  `dqn_resizer_step.py` (Needs completion)
3. ❌ `dqn_resizer.tcl` (Not yet created)
4. ❌ `dqn_agent.py` (Not yet created)
5. ❌ `timing_parser.py` (Not yet created)
6. ❌ `cell_resizer.py` (Optional helper)
