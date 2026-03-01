#!/usr/bin/env python3
"""
Quick test script for DQN Resizer
"""
import os
from openlane.config import Config
from openlane.state import State
from dqn_resizer_step import DQNResizer
from openlane.config import Variable

# Load your design config
# config_path = "designs/picorv_test/config.json"
# config = Config.load(config_path)

from openlane.config import Config, universal_flow_config_variables
from openlane.steps import OpenROAD

# additional_config_dict = {
#     "DQN_MODEL_PATH=/home/isaishaq/openlane2/designs/picorv_test/models/dqn_model.pth",
#     "DQN_MAX_ITERATIONS=50",
#     "DQN_TARGET_SLACK=0.0",
#     "DQN_POWER_WEIGHT=0.3",
#     "DQN_TRAINING_MODE=True",
#     "RSZ_DONT_TOUCH_RX=False",
# }

config, design_dir = Config.load(
    config_in="/home/isaishaq/openlane2/designs/picorv_test/runs/RUN_2026-01-17_18-02-01/51-openroad-stapostpnr/config.json",
    flow_config_vars=(
        universal_flow_config_variables
    ),
    design_dir="/home/isaishaq/openlane2/designs/picorv_test",
    #pdk="sky130A",
    #scl="sky130_fd_sc_hd",
    pdk_root="/home/isaishaq/.volare/volare/sky130/versions/0fe599b2afb6708d281543108caf8310912f54af",
)

# Load the state from STAPostPNR
# Find your latest run directory

## SET 1
#run_dir = "/home/isaishaq/openlane2/designs/picorv_test/runs/RUN_2026-01-17_18-33-44"
#sta_state_path = f"{run_dir}/53-openroad-stapostpnr/state_out.json"

## SET 2
run_dir = "/home/isaishaq/openlane2/designs/picorv_test/runs/RUN_2026-03-01_15-10-18/"
sta_state_path = f"{run_dir}/51-openroad-stapostpnr/state_out.json"

print(f"Loading state from: {sta_state_path}")
state_in = State.loads(open(sta_state_path, encoding="utf8").read())

# Create output directory
output_dir = f"{run_dir}/74-dqn-resizer-test"
os.makedirs(output_dir, exist_ok=True)

# Initialize and run DQN step
print("Initializing DQN Resizer...")

dqn_step = DQNResizer(
    config=config,
    state_in=state_in,
)

# dqn_step.config_vars.append(Variable(
#             "DQN_MODEL_PATH",
#             str,
#             "Path to pre-trained DQN model",
#             default="models/dqn_resizer.pth",
#         ),)

print("Running DQN Resizer...")
state_out = dqn_step.start(step_dir=output_dir)

# print("\n=== Results ===")
# print(f"Views updated: {views_update}")
# print(f"Metrics updated: {metrics_update}")

# Save the new state
#state_out = state_in.copy()
#state_out.update(views_update, metrics_update)
#state_out.save_snapshot(f"{output_dir}/state_out.json")
print(f"\nState saved to: {output_dir}/state_out.json")