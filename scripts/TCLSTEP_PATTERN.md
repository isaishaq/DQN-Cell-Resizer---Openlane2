# DQN Agent - TclStep Pattern Implementation

## Overview
The DQN agent now follows the OpenLane 2 TclStep pattern for executing TCL commands.

## Directory Structure
```
designs/picorv_test/scripts/
├── dqn_agent.py          # Python implementation
├── dqn_resizer.tcl       # Main TCL entry point
└── tcl/                  # TCL script library (like openroad/scripts/)
    ├── get_timing_metrics.tcl
    ├── get_wns.tcl
    ├── get_tns.tcl
    ├── get_power.tcl
    ├── resize_cell.tcl
    └── save_db.tcl
```

## Key Methods

### `_get_script_path(script_name: str) -> str`
Returns the absolute path to a TCL script file.
- **Pattern**: Like `TclStep.get_script_path()`
- **Usage**: `self._get_script_path('get_wns.tcl')`

### `_run_tcl_script(script_name: str, env_vars: Dict) -> CompletedProcess`
Executes a TCL script with OpenROAD.
- **Pattern**: Like `TclStep.run_subprocess()` + `get_command()`
- **Parameters**:
  - `script_name`: Name of script in tcl/ directory
  - `env_vars`: Dict of environment variables for the script
- **Usage**:
  ```python
  result = self._run_tcl_script(
      'get_wns.tcl',
      {'OUTPUT_FILE': output_path}
  )
  ```

### Specialized Query Methods
Each metric has its own dedicated method:
- `_get_wns()` → Uses `get_wns.tcl`
- `_get_tns()` → Uses `get_tns.tcl`
- `_get_power()` → Uses `get_power.tcl`

## Adding New TCL Commands

### 1. Create TCL Script
Create a new file in `tcl/` directory:

**Example: `tcl/get_cell_slack.tcl`**
```tcl
# Get slack for a specific cell
# Environment variables expected:
#   CELL_INSTANCE_NAME - Name of the cell
#   OUTPUT_FILE - Output file path

set inst_name $::env(CELL_INSTANCE_NAME)
set inst [[[ord::get_db] getChip] getBlock]::findInst $inst_name]

# Get worst slack for this cell
set worst_slack 999999.0
foreach pin [$inst getITerms] {
    set slack [sta::pin_slack $pin]
    if { $slack < $worst_slack } {
        set worst_slack $slack
    }
}

set fp [open $::env(OUTPUT_FILE) w]
puts $fp $worst_slack
close $fp
```

### 2. Add Python Method
Add a method to `CellResizingEnv`:

```python
def _get_cell_slack(self, inst_name: str) -> float:
    """Get worst slack through a specific cell"""
    output_file = os.path.join(self.work_dir, "cell_slack.txt")
    
    try:
        result = self._run_tcl_script(
            'get_cell_slack.tcl',
            {
                'CELL_INSTANCE_NAME': inst_name,
                'OUTPUT_FILE': output_file
            }
        )
        
        if result.returncode == 0 and os.path.exists(output_file):
            with open(output_file, 'r') as f:
                return float(f.read().strip())
        else:
            print(f"[ERROR] Failed to get cell slack: {result.stderr}")
            return 0.0
    except Exception as e:
        print(f"[ERROR] Exception getting cell slack: {e}")
        return 0.0
```

## Benefits of This Pattern

1. **Clean Separation**: TCL scripts are separate files, not inline strings
2. **Reusable**: Scripts can be called multiple times without regenerating
3. **Maintainable**: Easy to modify TCL logic without touching Python
4. **Testable**: TCL scripts can be tested independently
5. **Consistent**: Follows the same pattern as OpenLane 2 OpenROAD steps
6. **Debuggable**: Scripts persist on disk for inspection

## Environment Variables

All TCL scripts have access to:
- `$::env(ODB_PATH)` - Path to the current ODB database
- Custom vars passed via `env_vars` parameter

## Example Usage

```python
# Initialize environment
env = CellResizingEnv(
    odb_path="/path/to/design.odb",
    target_slack=0.0,
    power_weight=0.3
)

# Get timing metrics (uses get_timing_metrics.tcl)
wns = env._get_wns()
tns = env._get_tns()
power = env._get_power()

# Save database (uses save_db.tcl)
env.save("/path/to/output.odb")
```

## Comparison to OpenLane 2 Steps

| TclStep Pattern | DQN Agent Equivalent |
|----------------|---------------------|
| `get_script_path()` | `_get_script_path()` |
| `prepare_env()` | Done in `_run_tcl_script()` |
| `get_command()` | Returns `['openroad', '-exit', ...]` |
| `run_subprocess()` | `_run_tcl_script()` |
| Scripts in `scripts/openroad/` | Scripts in `tcl/` |
