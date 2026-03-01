"""
Quick Reference: Discrete Action Space for Cell Sizing
=======================================================

This file demonstrates the complete workflow from parsing timing reports
to defining and using the discrete action space for RL-based cell sizing.
"""

# ============================================================================
# PART 1: Understanding Discrete Action Space Design
# ============================================================================

"""
DISCRETE ACTION SPACE APPROACHES:

1. SINGLE-CELL ACTION (Recommended for beginners)
   - Action = (cell_index, resize_operation)
   - Total actions = K × 3, where K = number of actionable cells
   - Example: K=10 → 30 total actions
   
   Action breakdown:
     0-2:   Cell 0 (downsize, keep, upsize)
     3-5:   Cell 1 (downsize, keep, upsize)
     6-8:   Cell 2 (downsize, keep, upsize)
     ...
     27-29: Cell 9 (downsize, keep, upsize)

2. MULTI-CELL ACTION (Advanced - use with caution)
   - Action = combined action for multiple cells
   - Total actions = 3^K (exponential!)
   - Example: K=5 → 243 actions, K=10 → 59,049 actions
   
   Only use multi-cell if:
   - You have hierarchical RL or factored action spaces
   - You're using continuous action space with discretization
   - Your K is very small (≤5)
"""


# ============================================================================
# PART 2: Cell Selection Strategy
# ============================================================================

"""
HOW ACTIONABLE CELLS ARE SELECTED:

1. Parse worst N timing paths (e.g., N=5 to 10)
2. For each cell in these paths, calculate criticality score:
   
   criticality = delay × position_weight × slack_weight
   
   where:
   - delay = cell's delay contribution
   - position_weight = (index + 1) / path_length  (later cells weighted more)
   - slack_weight = 1 + |path_slack| if slack < 0

3. Aggregate criticality across all paths (a cell may appear multiple times)
4. Select top-K cells with highest criticality
5. Filter out non-resizable cells (clock buffers, flip-flops)

RESULT: K most critical cells that can be resized to fix timing
"""


# ============================================================================
# PART 3: Action Encoding/Decoding
# ============================================================================

def example_action_encoding():
    """
    Example of how actions are encoded and decoded.
    """
    
    # Given: 10 actionable cells, 3 possible actions per cell
    # Action space size = 10 × 3 = 30
    
    K = 10  # Number of actionable cells
    N_ACTIONS = 3  # DOWNSIZE=0, KEEP=1, UPSIZE=2
    
    # Agent selects action index (e.g., from Q-network output)
    action_index = 17
    
    # Decode to (cell_index, resize_action)
    cell_index = action_index // N_ACTIONS      # 17 // 3 = 5
    resize_action = action_index % N_ACTIONS    # 17 % 3 = 2 (UPSIZE)
    
    print(f"Action {action_index} → Cell {cell_index}, UPSIZE")
    # Output: "Action 17 → Cell 5, UPSIZE"
    
    # This means: "Upsize the 6th most critical cell"
    
    return cell_index, resize_action


# ============================================================================
# PART 4: Complete Workflow Example
# ============================================================================

def complete_workflow_example():
    """
    Step-by-step workflow from timing report to action execution.
    """
    from discrete_action_space import DiscreteActionSpace, CellLibrary
    
    # -----------------------------
    # STEP 1: Parse timing report
    # -----------------------------
    print("=" * 60)
    print("STEP 1: Parse Timing Report")
    print("=" * 60)
    
    # Assume you have parsed the timing report (using your parser)
    timing_data = {
        'paths': [
            {
                'slack': -3.93,
                'startpoint': 'genblk1.pcpi_mul.rs2[17]_sky130_fd_sc_hd__dfxtp_2_Q',
                'endpoint': 'genblk1.pcpi_mul.rd[63]_sky130_fd_sc_hd__dfxtp_2_Q',
                'cells': [
                    {
                        'instance_name': 'fanout1291',
                        'cell_type': 'sky130_fd_sc_hd__buf_4',
                        'drive_strength': 4,
                        'fanout': 11,
                        'delay': 0.39,
                        'slew': 0.31
                    },
                    {
                        'instance_name': 'fanout1290',
                        'cell_type': 'sky130_fd_sc_hd__buf_4',
                        'drive_strength': 4,
                        'fanout': 10,
                        'delay': 0.46,
                        'slew': 0.28
                    },
                    # ... more cells
                ]
            },
            # ... more paths
        ],
        'global_metrics': {
            'wns': -3.93,
            'tns': -85.47,
            'num_violations': 52
        }
    }
    
    print(f"Worst Negative Slack (WNS): {timing_data['global_metrics']['wns']}")
    print(f"Total Negative Slack (TNS): {timing_data['global_metrics']['tns']}")
    print(f"Number of Violations: {timing_data['global_metrics']['num_violations']}")
    
    # -----------------------------
    # STEP 2: Setup action space
    # -----------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Setup Action Space")
    print("=" * 60)
    
    action_space = DiscreteActionSpace(
        mode='single',      # Single-cell action
        top_k_cells=10,     # Consider top 10 critical cells
        library=CellLibrary()
    )
    
    print(f"Action space size: {action_space.n_actions}")
    print(f"Mode: {action_space.mode}")
    print(f"Top-K cells: {action_space.top_k_cells}")
    
    # -----------------------------
    # STEP 3: Identify actionable cells
    # -----------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Identify Actionable Cells")
    print("=" * 60)
    
    actionable_cells = action_space.get_actionable_cells(
        timing_data,
        worst_n_paths=5
    )
    
    print(f"\nFound {len(actionable_cells)} actionable cells:")
    for i, cell in enumerate(actionable_cells[:5]):  # Show first 5
        print(f"  [{i}] {cell.instance_name}")
        print(f"      Type: {cell.cell_type}")
        print(f"      Drive: {cell.current_drive_strength}, Fanout: {cell.fanout}")
        print(f"      Delay: {cell.delay:.3f}, Criticality: {cell.slack_contribution:.3f}")
    
    # -----------------------------
    # STEP 4: Agent selects action
    # -----------------------------
    print("\n" + "=" * 60)
    print("STEP 4: RL Agent Selects Action")
    print("=" * 60)
    
    # Example: Agent's Q-network outputs action probabilities
    # q_values = agent.forward(state)  # Shape: (30,)
    # action = torch.argmax(q_values).item()
    
    # For demonstration, let's manually select an action
    action = 2  # Upsize cell 0 (most critical cell)
    
    print(f"\nAgent selected action: {action}")
    
    # Decode action
    cell_idx = action // 3
    resize_op = action % 3
    resize_names = ['DOWNSIZE', 'KEEP', 'UPSIZE']
    
    print(f"Decoded: Cell[{cell_idx}] = {actionable_cells[cell_idx].instance_name}, "
          f"Operation = {resize_names[resize_op]}")
    
    # -----------------------------
    # STEP 5: Generate resize commands
    # -----------------------------
    print("\n" + "=" * 60)
    print("STEP 5: Generate Resize Commands")
    print("=" * 60)
    
    resize_commands = action_space.apply_action(action, actionable_cells)
    
    print(f"\nResize commands to execute:")
    for instance, (old_cell, new_cell) in resize_commands.items():
        print(f"  {instance}:")
        print(f"    From: {old_cell}")
        print(f"    To:   {new_cell}")
    
    # -----------------------------
    # STEP 6: Execute resizing
    # -----------------------------
    print("\n" + "=" * 60)
    print("STEP 6: Execute Resizing (OpenROAD)")
    print("=" * 60)
    
    # Generate TCL script for OpenROAD
    tcl_script = generate_openroad_script(resize_commands)
    print("\nGenerated TCL script:")
    print("-" * 40)
    print(tcl_script)
    print("-" * 40)
    
    # In practice: execute the script
    # subprocess.run(['openroad', '-exit', 'resize.tcl'])
    
    # -----------------------------
    # STEP 7: Re-run timing analysis
    # -----------------------------
    print("\n" + "=" * 60)
    print("STEP 7: Re-run Timing Analysis")
    print("=" * 60)
    
    print("\nAfter resizing, you would:")
    print("1. Run OpenSTA/OpenROAD timing analysis")
    print("2. Parse new timing report")
    print("3. Calculate reward = new_slack - old_slack - area_penalty")
    print("4. Feed (state, action, reward, next_state) to RL agent")
    print("5. Repeat until timing is fixed or max steps reached")
    
    # Simulated new metrics
    print("\nSimulated results:")
    print(f"  Old WNS: -3.93 ns")
    print(f"  New WNS: -3.15 ns  ← Improved by 0.78 ns!")
    print(f"  Reward: +0.75 (slack improvement - area penalty)")


def generate_openroad_script(resize_commands):
    """Generate OpenROAD TCL script for cell resizing."""
    script = "# Cell Resizing Script\n\n"
    script += "# Load design\n"
    script += "# (Assume design is already loaded)\n\n"
    
    for instance, (old_cell, new_cell) in resize_commands.items():
        script += f"# Resize {instance}\n"
        script += f"swap_cell {instance} {new_cell}\n\n"
    
    script += "# Report timing\n"
    script += "report_checks -path_delay max -format full_clock_expanded\n"
    script += "report_wns\n"
    script += "report_tns\n"
    
    return script


# ============================================================================
# PART 5: Integration with RL Frameworks
# ============================================================================

def integration_examples():
    """
    Show how to integrate with popular RL frameworks.
    """
    
    print("\n" + "=" * 60)
    print("INTEGRATION WITH RL FRAMEWORKS")
    print("=" * 60)
    
    # ----- Stable-Baselines3 -----
    print("\n1. STABLE-BASELINES3 (Easiest):")
    print("-" * 40)
    print("""
from stable_baselines3 import DQN
from rl_environment import CellSizingEnv

# Create environment
env = CellSizingEnv(design_dir='...', config_file='...')

# Create DQN agent
model = DQN(
    'MlpPolicy',
    env,
    learning_rate=1e-4,
    buffer_size=10000,
    learning_starts=1000,
    batch_size=32,
    gamma=0.99,
    target_update_interval=1000,
    verbose=1
)

# Train
model.learn(total_timesteps=100000)

# Save
model.save('dqn_cell_sizing')

# Use trained agent
obs = env.reset()
action, _ = model.predict(obs)
    """)
    
    # ----- PyTorch Custom -----
    print("\n2. PYTORCH (Custom DQN):")
    print("-" * 40)
    print("""
import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, state):
        return self.network(state)

# Training loop
q_net = QNetwork(state_dim=45, action_dim=30)
optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-4)
...
    """)
    
    # ----- Action Masking -----
    print("\n3. ACTION MASKING (Important!):")
    print("-" * 40)
    print("""
# Some actions may be invalid (e.g., can't upsize already at max)
# Use action masking to prevent invalid actions

valid_actions = action_space.get_valid_actions_mask(actionable_cells)

# In Q-network forward pass:
q_values = q_net(state)
q_values[~valid_actions] = -float('inf')  # Mask invalid actions
action = torch.argmax(q_values)
    """)


# ============================================================================
# MAIN: Run all examples
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("DISCRETE ACTION SPACE - QUICK REFERENCE")
    print("=" * 60)
    
    # Show action encoding
    print("\n--- Action Encoding Example ---")
    example_action_encoding()
    
    # Show complete workflow
    print("\n\n")
    complete_workflow_example()
    
    # Show framework integration
    integration_examples()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Key Takeaways:

1. Action Space Structure:
   - Single-cell mode: K × 3 actions (recommended)
   - Each action = (cell_index, resize_operation)
   
2. Cell Selection:
   - Top-K most critical cells from worst timing paths
   - Criticality = delay × position × slack_weight
   
3. Action Execution:
   - Decode action → get resize commands
   - Generate OpenROAD/OpenSTA script
   - Apply resizing and re-analyze timing
   
4. RL Training:
   - State = timing metrics + cell features
   - Action = discrete index (0 to K×3-1)
   - Reward = slack improvement - area penalty
   
5. Files:
   - discrete_action_space.py: Action space definition
   - rl_environment.py: Gym environment
   - quick_reference.py: This file with examples

Next Steps:
1. Parse your timing reports (max.rpt)
2. Test action space on sample data
3. Integrate with your RL framework
4. Train agent on your designs
    """)
