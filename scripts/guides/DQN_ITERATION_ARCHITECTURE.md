# DQN Cell Resizing - Iteration Architecture

## Problem Statement

How to iterate the DQN process when:
- Python DQN agent makes decisions
- TCL/OpenROAD applies actions and gets new timing
- Cannot directly call OpenROAD commands from Python

## Solution: TCL-Controlled Iteration Loop

The architecture uses **TCL as the main loop controller**, with Python as a stateless subprocess for DQN inference.

```
┌─────────────────────────────────────────────────────────┐
│                    TCL Main Loop                        │
│                  (dqn_resizer.tcl)                      │
└──────────────┬──────────────────────────────────────────┘
               │
               ├─ Iteration 1
               │   ├─1. Generate timing report
               │   │   └─> report_checks > timing.rpt
               │   │
               │   ├─2. Call Python DQN Agent
               │   │   └─> exec python3 dqn_agent.py
               │   │       ├─ Reads: timing.rpt
               │   │       ├─ Infers: Q-values
               │   │       └─ Writes: actions.txt
               │   │
               │   ├─3. Apply resize actions
               │   │   └─> replace_cell inst new_cell
               │   │
               │   ├─4. Update timing
               │   │   └─> estimate_parasitics
               │   │
               │   └─5. Check convergence
               │       ├─ WNS >= 0? → SUCCESS, break
               │       └─ No actions? → converged, break
               │
               ├─ Iteration 2
               │   └─ (repeat steps 1-5)
               │
               └─ Iteration N
                   └─ Max iterations or converged
```

---

## Data Flow

```
TCL (Iteration N)                    Python (Subprocess)
─────────────────                    ───────────────────

1. report_checks          ──>        timing_iter_N.rpt

2. exec python3 dqn_agent.py
                          <──        reads timing.rpt
                                     ├─ parse timing
                                     ├─ extract state
                                     ├─ DQN inference
                                     └─ decode action
                          ──>        actions_iter_N.txt

3. read actions_iter_N.txt
   └─ replace_cell inst new_cell

4. estimate_parasitics

5. check WNS/TNS
   └─ Next iteration or break
```

---

## Key Implementation Details

### 1. TCL Loop Structure

```tcl
set dqn_max_iters 50

for {set iter 1} {$iter <= $dqn_max_iters} {incr iter} {
    # Generate report
    report_checks > timing_iter${iter}.rpt
    
    # Call Python DQN
    exec python3 dqn_agent.py \
        --timing-report timing_iter${iter}.rpt \
        --output-actions actions_iter${iter}.txt \
        --model model.pth \
        --iteration $iter
    
    # Apply actions
    set fp [open actions_iter${iter}.txt r]
    while {[gets $fp line] >= 0} {
        # Parse and apply replace_cell
    }
    close $fp
    
    # Update and check
    estimate_parasitics -global_routing
    set wns [sta::worst_slack -max]
    
    if {$wns >= 0} { break }
}
```

### 2. Python Agent (Stateless)

```python
def main():
    # Parse timing report from file
    timing_data = parse_timing_report(args.timing_report)
    
    # Extract state features
    state = extract_state_features(timing_data)
    
    # Load DQN model and select action
    agent = SimpleDQNAgent(model_path=args.model)
    action = agent.select_action(state)
    
    # Decode action to resize commands
    resizes = decode_action(action, timing_data)
    
    # Write actions to file
    write_actions_file(args.output_actions, resizes)
    
    # Exit (TCL will call again next iteration)
```

### 3. Actions File Format

```
# Iteration: 5
# Action: 23
# WNS: -2.45
# TNS: -67.23

_12345_ sky130_fd_sc_hd__buf_4
_67890_ sky130_fd_sc_hd__and2_2
_11223_ sky130_fd_sc_hd__nand2_4
```

---

## Advantages of This Architecture

### ✅ Pros

1. **Simple IPC**: File-based communication (timing.rpt ↔ actions.txt)
2. **TCL has full control**: Direct access to all OpenROAD commands
3. **Python is stateless**: Easy to debug, no persistent state issues
4. **Crash resilient**: If Python crashes, TCL can retry
5. **Extensible**: Easy to add logging, checkpointing, etc.

### ⚠️ Limitations

1. **No online training**: Python only does inference, training is offline
2. **Subprocess overhead**: ~100ms per call (acceptable for 50 iterations)
3. **Limited RL algorithms**: Hard to implement experience replay

---

## Alternative Architectures (If Needed)

### Option 2: Python Server + TCL Client

For online training or experience replay:

```python
# Python server (persistent)
import socket

server = socket.socket()
server.bind(('localhost', 5555))

while True:
    conn, addr = server.accept()
    timing_report = conn.recv(1024)
    
    # DQN inference
    action = agent.select_action(state)
    
    conn.send(action.encode())
```

```tcl
# TCL client
set socket [socket localhost 5555]
puts $socket $timing_report
set action [gets $socket]
close $socket
```

**Use when**: Need online training, experience replay, or persistent state

### Option 3: OpenROAD Python API

Use OpenROAD's Python bindings directly:

```python
import openroad as ord

# Load design
db = ord.Design()
db.readDb("design.odb")

# DQN loop in Python
for iteration in range(50):
    # Get timing
    sta = db.getSta()
    wns = sta.worstSlack()
    
    # DQN decision
    action = agent.select_action(state)
    
    # Apply resize
    inst = db.findInst(instance_name)
    inst.swapMaster(new_cell)
    
    # Update timing
    sta.updateTiming()
```

**Use when**: Need fine-grained control, but check API availability

---

## For True RL Training (Future Enhancement)

If you want to train DQN online (not just inference), you need:

### 1. Experience Replay Buffer

Store transitions in a persistent database:

```python
# experience_buffer.db (SQLite)
CREATE TABLE transitions (
    iteration INT,
    state BLOB,
    action INT,
    reward FLOAT,
    next_state BLOB,
    done BOOL
);
```

Python agent appends after each iteration:
```python
# After receiving new timing report
reward = calculate_reward(old_wns, new_wns, old_power, new_power)
buffer.append(state, action, reward, next_state, done)

# Periodically train
if iteration % 10 == 0:
    batch = buffer.sample(batch_size=32)
    agent.train_step(batch)
```

### 2. Model Checkpointing

TCL saves model after each episode:
```tcl
# After design episode completes
exec python3 train_dqn.py \
    --experience-buffer transitions.db \
    --save-model model_v${episode}.pth
```

---

## Current Status

✅ **Implemented**: TCL-controlled iteration loop  
✅ **Implemented**: Python DQN inference agent  
✅ **Implemented**: File-based IPC  
✅ **Implemented**: Convergence checking  
⚠️ **TODO**: Online training (optional)  
⚠️ **TODO**: Experience replay (optional)  

---

## Usage

### Run DQN Cell Resizing

```bash
# From OpenROAD
openroad -no_init dqn_resizer.tcl

# Or from Python (launch OpenROAD)
python3 scripts/dqn_resizer_step.py
```

### Set Iteration Limit

```tcl
# In TCL
set ::env(DQN_MAX_ITERATIONS) 100
source dqn_resizer.tcl
```

### Monitor Progress

```bash
# Watch iterations
tail -f reports/iter*/max.rpt

# Check convergence
grep "WNS:" actions/actions_iter*.txt
```

---

## Summary

**Answer to "How do we iterate?"**

→ **TCL controls the loop**, Python is a stateless subprocess called each iteration.

This is the **simplest and most robust** approach for your use case (DQN inference on cell sizing). The architecture already works - you just needed to uncomment the action application code!
