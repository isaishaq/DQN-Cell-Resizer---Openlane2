# OpenROAD Integration Guide for Training

To complete the training system, you need to integrate OpenROAD with `rl_environment.py`.

## Current Status

✅ **Complete:**
- Training infrastructure ([train_dqn.py](train_dqn.py))
- DQN network architecture ([dqn_agent.py](dqn_agent.py))  
- Gym-compatible environment ([rl_environment.py](rl_environment.py))
- Inference pipeline ([dqn_resizer.tcl](dqn_resizer.tcl))

⚠️ **Needs Integration:**
- `CellSizingEnv._run_timing_analysis()` - Currently returns mock data
- `CellSizingEnv._get_design_area()` - Currently returns constant
- `CellSizingEnv._apply_resizes()` - Currently just prints commands

---

## Three Implementation Approaches

### ⭐ Option 1: TCL Helper Script (Recommended)

**Why:** Fastest, most reliable, uses existing OpenROAD infrastructure

**How it works:**
```
Python (rl_environment.py)
    ↓
    Writes action file
    ↓
TCL script (training_helper.tcl)
    Executes OpenROAD commands
    Writes timing report
    ↓
Python reads timing report
    ↓
    Continues episode
```

**Implementation:**

**Step 1:** Create `training_helper.tcl`

```tcl
#!/usr/bin/env tclsh

# training_helper.tcl - OpenROAD helper for training episodes

proc initialize_design {design_path} {
    # Load design from checkpoint or DEF
    read_lef $::env(PDK_ROOT)/sky130A/libs.ref/sky130_fd_sc_hd/techlef
    read_lef $::env(PDK_ROOT)/sky130A/libs.ref/sky130_fd_sc_hd/lef
    read_liberty $::env(PDK_ROOT)/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib
    
    read_def ${design_path}/placement.def
    read_sdc ${design_path}/design.sdc
    
    # Initial timing
    set_propagated_clock [all_clocks]
    estimate_parasitics -placement
    report_checks -path_delay max -format full_clock_expanded \
        > reports/timing_0.rpt
    
    puts "Design initialized"
}

proc apply_resize {instance_name new_cell_name} {
    # Resize one cell instance
    set db_inst [dbget top.insts.name $instance_name -p]
    if {$db_inst != ""} {
        replace_cell $instance_name $new_cell_name
        puts "Resized: $instance_name -> $new_cell_name"
        return 1
    } else {
        puts "Error: Instance $instance_name not found"
        return 0
    }
}

proc update_timing {step_num} {
    # Re-analyze timing after resizes
    estimate_parasitics -placement
    
    report_checks -path_delay max -format full_clock_expanded \
        > reports/timing_${step_num}.rpt
    
    # Also export metrics as JSON for easy parsing
    set wns [sta::worst_slack -max]
    set tns [sta::total_negative_slack -max]
    set violations [llength [sta::find_timing_paths -slack_max 0.0]]
    
    set metrics [dict create \
        wns $wns \
        tns $tns \
        violations $violations \
    ]
    
    set fp [open "reports/metrics_${step_num}.json" w]
    puts $fp [dict_to_json $metrics]
    close $fp
    
    puts "Timing updated: WNS=$wns TNS=$tns"
}

proc dict_to_json {d} {
    set json "{"
    set first 1
    dict for {k v} $d {
        if {!$first} {append json ", "}
        append json "\"$k\": $v"
        set first 0
    }
    append json "}"
    return $json
}

# Main command interface
proc execute_command {cmd args} {
    switch $cmd {
        "init" {
            initialize_design [lindex $args 0]
        }
        "resize" {
            apply_resize [lindex $args 0] [lindex $args 1]
        }
        "update_timing" {
            update_timing [lindex $args 0]
        }
        default {
            puts "Unknown command: $cmd"
        }
    }
}

# Command loop - reads from stdin or file
if {$argc > 0} {
    # Read commands from file
    set cmd_file [lindex $argv 0]
    set fp [open $cmd_file r]
    while {[gets $fp line] >= 0} {
        eval execute_command $line
    }
    close $fp
} else {
    # Interactive mode
    puts "Ready for commands"
    while {[gets stdin line] >= 0} {
        if {$line eq "exit"} break
        eval execute_command $line
    }
}
```

**Step 2:** Update `rl_environment.py` to use TCL helper

```python
# In CellSizingEnv.__init__(), start OpenROAD session
def __init__(self, ...):
    # ... existing code ...
    
    # Start persistent OpenROAD session
    self.openroad_process = None
    self.reports_dir = self.design_dir / "reports"
    self.reports_dir.mkdir(exist_ok=True)

def reset(self) -> np.ndarray:
    """Reset environment."""
    self.current_step = 0
    
    # Start OpenROAD process if not running
    if self.openroad_process is None:
        self._start_openroad_session()
    
    # Initialize design
    self._send_command(f"init {self.design_dir}")
    
    # Parse initial timing
    timing_data = self._parse_timing_report(0)
    
    self.initial_wns = timing_data['global_metrics']['wns']
    self.initial_area = self._get_design_area()
    
    self.actionable_cells = self.action_space_mgr.get_actionable_cells(
        timing_data, worst_n_paths=10
    )
    
    self.current_state = self._extract_state(timing_data)
    return self.current_state

def _start_openroad_session(self):
    """Start persistent OpenROAD session."""
    import subprocess
    
    tcl_script = self.design_dir / "scripts" / "training_helper.tcl"
    
    self.openroad_process = subprocess.Popen(
        ['openroad', '-exit', str(tcl_script)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    print(f"OpenROAD session started (PID: {self.openroad_process.pid})")

def _send_command(self, command: str):
    """Send command to OpenROAD TCL helper."""
    if self.openroad_process:
        self.openroad_process.stdin.write(command + "\n")
        self.openroad_process.stdin.flush()

def _run_timing_analysis(self) -> Dict:
    """Run timing analysis."""
    # Request timing update
    self._send_command(f"update_timing {self.current_step}")
    
    # Wait for report to be written
    time.sleep(0.1)  # Small delay for file write
    
    # Parse timing report
    return self._parse_timing_report(self.current_step)

def _parse_timing_report(self, step: int) -> Dict:
    """Parse timing report from file."""
    from timing_parser import parse_timing_report
    
    timing_file = self.reports_dir / f"timing_{step}.rpt"
    metrics_file = self.reports_dir / f"metrics_{step}.json"
    
    # Parse detailed timing
    timing_data = parse_timing_report(str(timing_file))
    
    # Load metrics JSON (easier to parse)  
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            timing_data['global_metrics'] = {
                'wns': metrics['wns'],
                'tns': metrics['tns'],
                'num_violations': metrics['violations']
            }
    
    return timing_data

def _apply_resizes(self, resize_commands: Dict[str, Tuple[str, str]]):
    """Apply cell resizing."""
    for instance, (old_cell, new_cell) in resize_commands.items():
        self._send_command(f"resize {instance} {new_cell}")
    
    print(f"Applied {len(resize_commands)} resizes")

def close(self):
    """Clean up OpenROAD session."""
    if self.openroad_process:
        self._send_command("exit")
        self.openroad_process.terminate()
        self.openroad_process.wait(timeout=5)
        print("OpenROAD session closed")
```

---

### Option 2: Subprocess Per Action (Simpler, Slower)

**Good for:** Quick prototyping, debugging

**Implementation:**

```python
def _run_timing_analysis(self) -> Dict:
    """Run standalone timing analysis."""
    import subprocess
    from timing_parser import parse_timing_report
    
    tcl_script = f"""
    # Load design
    read_lef ...
    read_def {self.design_dir}/placement.def
    read_sdc {self.design_dir}/design.sdc
    
    # Run timing
    estimate_parasitics -placement
    report_checks -path_delay max > timing_temp.rpt
    
    exit
    """
    
    script_file = self.design_dir / "temp_timing.tcl"
    with open(script_file, 'w') as f:
        f.write(tcl_script)
    
    subprocess.run(['openroad', '-exit', str(script_file)], 
                   capture_output=True)
    
    return parse_timing_report('timing_temp.rpt')

def _apply_resizes(self, resize_commands):
    """Apply resizes via subprocess."""
    tcl_script = "# Load design...\n"
    
    for instance, (old, new) in resize_commands.items():
        tcl_script += f"replace_cell {instance} {new}\n"
    
    tcl_script += "write_def updated.def\nexit\n"
    
    script_file = self.design_dir / "temp_resize.tcl"
    with open(script_file, 'w') as f:
        f.write(tcl_script)
    
    subprocess.run(['openroad', '-exit', str(script_file)])
```

**Pros:** Simple, no state management  
**Cons:** Slow (~500ms per action vs ~50ms with persistent session)

---

### Option 3: Python OpenROAD Bindings

**Good for:** Production, if Python bindings available

**Check if available:**
```bash
python3 -c "import openroad; print('OpenROAD Python bindings available!')"
```

**If available:**
```python
import openroad

class CellSizingEnv:
    def __init__(self, ...):
        # ... existing code ...
        
        # Initialize OpenROAD
        self.db = openroad.Database()
        self.sta = openroad.TimingEngine()
        
    def _run_timing_analysis(self):
        # Direct Python API
        self.sta.update_timing()
        wns = self.sta.worst_negative_slack()
        tns = self.sta.total_negative_slack()
        
        return {
            'global_metrics': {
                'wns': wns,
                'tns': tns,
                'num_violations': self.sta.endpoint_count()
            }
        }
    
    def _apply_resizes(self, resize_commands):
        for instance, (old, new) in resize_commands.items():
            inst = self.db.get_instance(instance)
            inst.swap_master(new)
```

**Note:** OpenROAD Python bindings may not be complete yet. Check documentation.

---

## Testing Your Integration

### Test 1: Single Episode

```python
#!/usr/bin/env python3
from rl_environment import CellSizingEnv
import numpy as np

# Test environment
env = CellSizingEnv(
    design_dir='/path/to/design',
    config_file='/path/to/config.json',
    max_steps=10
)

state = env.reset()
print(f"Initial state shape: {state.shape}")
print(f"Initial WNS: {env.initial_wns}")

for step in range(5):
    action = np.random.randint(0, env.action_space.n)
    next_state, reward, done, info = env.step(action)
    
    print(f"\nStep {step+1}:")
    print(f"  WNS: {info['wns']:.3f}")
    print(f"  Reward: {reward:.2f}")
    print(f"  Done: {done}")
    
    if done:
        break

env.close()
```

### Test 2: Training Loop

```bash
# Quick test with 10 episodes
python3 train_dqn.py \
    --designs test_designs.txt \
    --episodes 10 \
    --output model_test.pth \
    --log-dir logs/test

# Check if training completes without errors
```

---

## Debugging Checklist

If training fails:

- [ ] **OpenROAD accessible:** `which openroad` shows path
- [ ] **Design files exist:** Check DEF, SDC, LEF files
- [ ] **Timing reports generated:** Look in `reports/` directory
- [ ] **timing_parser.py works:** Test with sample report
- [ ] **Permissions:** Script has write access to design directory
- [ ] **PDK configured:** `$PDK_ROOT` environment variable set

---

## Expected Performance

| Metric | Value |
|--------|-------|
| Episode time (Option 1) | ~5-30 seconds (50 steps) |
| Episode time (Option 2) | ~30-120 seconds (50 steps) |
| Training time (1000 episodes) | 2-8 hours (Option 1) |
| Memory usage | ~2-4 GB |

---

## Next Steps

1. **Choose integration approach** (recommend Option 1)
2. **Implement placeholders in rl_environment.py**
3. **Test with single episode** (test script above)
4. **Run 10-episode training test**
5. **Scale to full 1000-episode training**
6. **Deploy trained model** to inference pipeline

---

## Quick Start Commands

```bash
# 1. Create training helper (Option 1)
cd designs/picorv_test/scripts
# ... create training_helper.tcl from template above ...

# 2. Test environment
python3 test_environment.py

# 3. Start training
python3 train_dqn.py \
    --designs ../designs_train.txt \
    --episodes 1000 \
    --output model_trained.pth

# 4. Deploy model
cp model_trained.pth ../runs/RUN_XXX/74-dqn-resizer/model.pth

# 5. Run inference
openroad -exit dqn_resizer.tcl
```

---

## Summary

**Simplest path to working training:**
1. Implement Option 1 (TCL helper script) - 30 minutes
2. Test with single episode - 5 minutes
3. Run 10-episode test - 5 minutes
4. Launch full training - 4-8 hours

**Total time to trained model:** ~5-10 hours (mostly waiting for training)
