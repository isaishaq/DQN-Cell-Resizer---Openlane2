#!/usr/bin/env python3
"""
Quick test script for DQN Resizer
"""
import os
from openlane.config import Config
from openlane.state import State
from dqn_resizer_step import DQNResizer
from openlane.config import Variable


from openlane.config import Config, universal_flow_config_variables
from openlane.steps import OpenROAD


# Load the state from STAPostPNR
# Find your latest run directory

## SET 1
#run_dir = "/home/isaishaq/openlane2/designs/picorv_test/runs/RUN_2026-01-17_18-33-44"
#sta_state_path = f"{run_dir}/53-openroad-stapostpnr/state_out.json"

## SET 2
run_dir = "/home/isaishaq/openlane2/designs/picorv_test/runs/RUN_2026-03-01_15-10-18/"
sta_state_path = f"{run_dir}/51-openroad-stapostpnr/state_out.json"
config_in_path = f"{run_dir}/51-openroad-stapostpnr/config.json"

config, design_dir = Config.load(
    config_in=config_in_path,
    flow_config_vars=(
        universal_flow_config_variables
    ),
    design_dir="/home/isaishaq/openlane2/designs/picorv_test",
    pdk_root="/home/isaishaq/.volare/volare/sky130/versions/0fe599b2afb6708d281543108caf8310912f54af",
)

print(f"Loading state from: {sta_state_path}")
state_in = State.loads(open(sta_state_path, encoding="utf8").read())

# Create output directory
output_dir = f"{run_dir}/74-dqn-resizer-test"
os.makedirs(output_dir, exist_ok=True)

dqn_step = DQNResizer(
    config=config,
    state_in=state_in,
)

print("Running DQNResizer-OpenROAD Step")
state_out = dqn_step.start(step_dir=output_dir)

# print("\n=== Results ===")
# print(f"Views updated: {views_update}")
# print(f"Metrics updated: {metrics_update}")

# Save the new state
#state_out = state_in.copy()
#state_out.update(views_update, metrics_update)
#state_out.save_snapshot(f"{output_dir}/state_out.json")
print(f"\nState saved to: {output_dir}/state_out.json")