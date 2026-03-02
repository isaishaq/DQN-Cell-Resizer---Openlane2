#!/usr/bin/env python3
"""
DQN Agent for Cell Resizing - OpenLane2 Integration
====================================================

This agent is invoked by TCL (dqn_resizer.tcl) as a subprocess.
It reads timing reports, makes DQN-based decisions, and outputs actions.

Workflow:
    TCL -> timing.rpt -> Python (this file) -> actions.txt -> TCL

Usage:
    python3 dqn_agent.py \\
        --timing-report path/to/timing.rpt \\
        --output-actions path/to/actions.txt \\
        --model path/to/model.pth \\
        --iteration 1

Author: DQN Cell Resizing Team
"""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# PyTorch imports
try:
    import torch
    import torch.nn as nn
except ImportError:
    print("[ERROR] PyTorch not installed. Install with: pip install torch")
    sys.exit(1)

# Import our modules
from timing_parser import parse_timing_report, TimingReportParser
from discrete_action_space import DiscreteActionSpace, CellLibrary, Cell


# ============================================================================
# State Extraction
# ============================================================================

def extract_state_features(
    timing_data: Dict,
    actionable_cells: List[Cell],
    top_k_cells: int = 10
) -> np.ndarray:
    """
    Extract state features for DQN input.
    
    State structure (45-dim vector):
        - Global metrics (5): WNS, TNS, violations, avg_slack, max_delay
        - Top-K cell features (10 cells × 4 features each):
            - Cell delay (normalized)
            - Cell fanout (normalized)
            - Cell drive strength (normalized)
            - Cell slack (normalized)
    
    Args:
        timing_data: Parsed timing report dictionary
        actionable_cells: List of actionable cells from DiscreteActionSpace
        top_k_cells: Number of cells to include in state
        
    Returns:
        State vector (numpy array)
    """
    global_metrics = timing_data['global_metrics']
    paths = timing_data['paths']
    
    # === Global Features (5) ===
    wns = global_metrics['wns']
    tns = global_metrics['tns']
    num_violations = global_metrics['num_violations']
    
    # Calculate average slack and max delay
    all_slacks = [p.get('slack', 0) for p in paths]
    avg_slack = np.mean(all_slacks) if all_slacks else 0
    
    max_delay = 0
    for path in paths:
        for cell in path.get('cells', []):
            max_delay = max(max_delay, cell.get('delay', 0))
    
    global_features = np.array([
        wns,
        tns,
        num_violations,
        avg_slack,
        max_delay
    ])
    
    # === Cell Features (10 cells × 4 features = 40) ===
    cell_features = []
    
    for i in range(top_k_cells):
        if i < len(actionable_cells):
            cell = actionable_cells[i]
            
            # Normalize features
            delay_norm = cell.delay / max(max_delay, 1e-6)
            fanout_norm = cell.fanout / 20.0  # Assume max fanout ~20
            drive_norm = cell.current_drive_strength / 16.0  # Max drive ~16
            slack_norm = cell.slack_contribution / abs(wns) if wns != 0 else 0
            
            cell_features.extend([
                delay_norm,
                fanout_norm,
                drive_norm,
                slack_norm
            ])
        else:
            # Padding for cells that don't exist
            cell_features.extend([0, 0, 0, 0])
    
    cell_features = np.array(cell_features)
    
    # Concatenate global and cell features
    state = np.concatenate([global_features, cell_features])
    
    return state.astype(np.float32)


# ============================================================================
# DQN Network Definition
# ============================================================================

class DQNNetwork(nn.Module):
    """
    Deep Q-Network for cell resizing decisions.
    
    Architecture:
        Input: State vector (45-dim)
        Hidden: 128 -> 128 -> 64
        Output: Q-values for each action (30 actions)
    """
    
    def __init__(self, state_dim: int = 45, action_dim: int = 30, hidden_dim: int = 128):
        super(DQNNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, x):
        """Forward pass through network."""
        return self.network(x)


# ============================================================================
# DQN Agent
# ============================================================================

class SimpleDQNAgent:
    """
    Simplified DQN agent for inference only.
    Training is done separately offline.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        state_dim: int = 45,
        action_dim: int = 30,
        epsilon: float = 0.0
    ):
        """
        Initialize DQN agent.
        
        Args:
            model_path: Path to trained model weights (.pth file)
            state_dim: State vector dimension
            action_dim: Number of discrete actions
            epsilon: Exploration rate (0.0 for pure exploitation)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = epsilon
        
        # Create network
        self.q_network = DQNNetwork(state_dim, action_dim)
        
        # Load trained weights if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            print(f"[AGENT] Loaded model from: {model_path}")
        else:
            print(f"[AGENT] No model loaded - using random policy")
            if model_path:
                print(f"[AGENT] Model file not found: {model_path}")
        
        self.q_network.eval()  # Set to evaluation mode
    
    def select_action(
        self,
        state: np.ndarray,
        valid_actions_mask: Optional[np.ndarray] = None
    ) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: State vector (numpy array)
            valid_actions_mask: Boolean mask of valid actions (optional)
            
        Returns:
            Action index (0 to action_dim-1)
        """
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            # Random action
            if valid_actions_mask is not None:
                valid_indices = np.where(valid_actions_mask)[0]
                return np.random.choice(valid_indices)
            else:
                return np.random.randint(self.action_dim)
        
        # Greedy action selection
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dim
            q_values = self.q_network(state_tensor)
            q_values = q_values.squeeze(0).numpy()  # Remove batch dim
            
            # Mask invalid actions
            if valid_actions_mask is not None:
                q_values[~valid_actions_mask] = -np.inf
            
            action = np.argmax(q_values)
        
        return int(action)
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Get Q-values for all actions.
        
        Args:
            state: State vector
            
        Returns:
            Q-values array (action_dim,)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.squeeze(0).numpy()
    
    def load_model(self, path: str):
        """Load model weights from file."""
        checkpoint = torch.load(path, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'q_network' in checkpoint:
                self.q_network.load_state_dict(checkpoint['q_network'])
            elif 'model_state_dict' in checkpoint:
                self.q_network.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.q_network.load_state_dict(checkpoint)
        else:
            self.q_network.load_state_dict(checkpoint)
        
        print(f"[AGENT] Model loaded successfully")


# ============================================================================
# Output Generation
# ============================================================================

def write_actions_file(
    output_path: str,
    resizes: Dict[str, Tuple[str, str]],
    iteration: int,
    action_idx: int,
    state_features: np.ndarray,
    timing_metrics: Dict
):
    """
    Write actions to file for TCL to read.
    
    Format:
        # Iteration: 1
        # Action: 5
        # WNS: -3.93
        # TNS: -85.47
        instance_name new_cell_type
        instance_name new_cell_type
        ...
    
    Args:
        output_path: Path to output file
        resizes: Dictionary mapping instance -> (old_cell, new_cell)
        iteration: Current iteration number
        action_idx: Selected action index
        state_features: State vector
        timing_metrics: Timing metrics dict
    """
    # Create folder if not exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        # Write metadata as comments
        f.write(f"# Iteration: {iteration}\n")
        f.write(f"# Action: {action_idx}\n")
        f.write(f"# WNS: {timing_metrics.get('wns', 0):.4f}\n")
        f.write(f"# TNS: {timing_metrics.get('tns', 0):.4f}\n")
        f.write(f"# Violations: {timing_metrics.get('num_violations', 0)}\n")
        f.write(f"# State_dim: {len(state_features)}\n")
        f.write("\n")
        
        # Write resize commands
        if resizes:
            for instance_name, (old_cell, new_cell) in resizes.items():
                f.write(f"{instance_name} {new_cell}\n")
        else:
            f.write("# No actions\n")
    
    print(f"[AGENT] Wrote {len(resizes)} resize commands to: {output_path}")


def write_state_log(
    log_path: str,
    iteration: int,
    state: np.ndarray,
    q_values: np.ndarray,
    action: int,
    timing_data: Dict
):
    """
    Write detailed state information for debugging.
    
    Args:
        log_path: Path to log file
        iteration: Current iteration
        state: State vector
        q_values: Q-values for all actions
        action: Selected action
        timing_data: Full timing data dict
    """
    log_entry = {
        'iteration': iteration,
        'state': state.tolist(),
        'q_values': q_values.tolist(),
        'selected_action': action,
        'selected_q_value': float(q_values[action]),
        'global_metrics': timing_data['global_metrics'],
        'num_paths': len(timing_data['paths'])
    }
    
    # Append to log file
    with open(log_path, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main entry point for DQN agent."""
    
    parser = argparse.ArgumentParser(
        description='DQN Agent for Cell Resizing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--timing-report',
        required=True,
        help='Path to timing report file (.rpt)'
    )
    parser.add_argument(
        '--output-actions',
        required=True,
        help='Path to output actions file (.txt)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--model',
        default=None,
        help='Path to trained DQN model (.pth)'
    )
    parser.add_argument(
        '--iteration',
        type=int,
        default=1,
        help='Current iteration number'
    )
    parser.add_argument(
        '--top-k-cells',
        type=int,
        default=10,
        help='Number of top critical cells to consider'
    )
    parser.add_argument(
        '--worst-n-paths',
        type=int,
        default=5,
        help='Number of worst paths to analyze'
    )
    parser.add_argument(
        '--epsilon',
        type=float,
        default=0.0,
        help='Exploration rate (0.0 for pure exploitation)'
    )
    parser.add_argument(
        '--state-log',
        default=None,
        help='Path to state log file (optional, for debugging)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print verbose debugging information'
    )
    
    args = parser.parse_args()
    
    # ========================================================================
    # Step 1: Parse Timing Report
    # ========================================================================
    
    print("="*70)
    print(f"DQN Agent - Iteration {args.iteration}")
    print("="*70)
    
    if not os.path.exists(args.timing_report):
        print(f"[ERROR] Timing report not found: {args.timing_report}")
        sys.exit(1)
    
    print(f"[STEP 1] Parsing timing report: {args.timing_report}")
    
    try:
        timing_data = parse_timing_report(args.timing_report)
        metrics = timing_data['global_metrics']
        
        print(f"  WNS: {metrics['wns']:.4f} ns")
        print(f"  TNS: {metrics['tns']:.4f} ns")
        print(f"  Violations: {metrics['num_violations']}")
        print(f"  Paths: {metrics['num_paths']}")
        
    except Exception as e:
        print(f"[ERROR] Failed to parse timing report: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ========================================================================
    # Step 2: Get Actionable Cells
    # ========================================================================
    
    print(f"\n[STEP 2] Identifying actionable cells (top-{args.top_k_cells})")
    
    try:
        action_space = DiscreteActionSpace(
            mode='single',
            top_k_cells=args.top_k_cells,
            library=CellLibrary()
        )
        
        actionable_cells = action_space.get_actionable_cells(
            timing_data,
            worst_n_paths=args.worst_n_paths
        )
        
        print(f"  Found {len(actionable_cells)} actionable cells")
        
        if args.verbose and actionable_cells:
            print(f"\n  Top 5 critical cells:")
            # print(actionable_cells[0].__dict__)
            for i, cell in enumerate(actionable_cells[:5]):
                print(f"    {i+1}. {cell.instance_name}: {cell.cell_type}")
                print(f"       Drive: {cell.current_drive_strength}, "
                      f"Fanout: {cell.fanout}, "
                      f"Slack: {cell.slack_contribution:.4f}")
        
        if not actionable_cells:
            print(f"[WARNING] No actionable cells found - design may be optimal")
            # Write empty actions file
            write_actions_file(
                args.output_actions,
                {},
                args.iteration,
                -1,
                np.zeros(45),
                metrics
            )
            sys.exit(0)
        
    except Exception as e:
        print(f"[ERROR] Failed to get actionable cells: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ========================================================================
    # Step 3: Extract State Features
    # ========================================================================
    
    print(f"\n[STEP 3] Extracting state features")
    
    try:
        state = extract_state_features(
            timing_data,
            actionable_cells,
            top_k_cells=args.top_k_cells
        )
        
        print(f"  State shape: {state.shape}")
        print(f"  State range: [{state.min():.4f}, {state.max():.4f}]")
        
        if args.verbose:
            print(f"  Global features: {state[:5]}")
        
    except Exception as e:
        print(f"[ERROR] Failed to extract state: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ========================================================================
    # Step 4: Load DQN and Select Action
    # ========================================================================
    
    print(f"\n[STEP 4] DQN action selection")
    
    try:
        # Create agent
        agent = SimpleDQNAgent(
            model_path=args.model,
            state_dim=len(state),
            action_dim=action_space.n_actions,
            epsilon=args.epsilon
        )
        
        # Get Q-values for debugging
        q_values = agent.get_q_values(state)
        
        # Select action
        action_idx = agent.select_action(state)
        
        print(f"  Action space size: {action_space.n_actions}")
        print(f"  Selected action: {action_idx}")
        print(f"  Q-value: {q_values[action_idx]:.4f}")
        print(f"  Exploration rate: {args.epsilon}")
        
        if args.verbose:
            print(f"\n  Top 5 Q-values:")
            top_actions = np.argsort(q_values)[-5:][::-1]
            for rank, act in enumerate(top_actions):
                print(f"    {rank+1}. Action {act}: Q={q_values[act]:.4f}")
        
    except Exception as e:
        print(f"[ERROR] Failed to select action: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ========================================================================
    # Step 5: Decode Action to Resize Commands
    # ========================================================================
    
    print(f"\n[STEP 5] Decoding action to resize commands")
    
    try:
        resizes = action_space.apply_action(action_idx, actionable_cells)
        
        print(f"  Generated {len(resizes)} resize commands")
        
        if args.verbose and resizes:
            print(f"\n  Resize commands:")
            for instance, (old_cell, new_cell) in resizes.items():
                print(f"    {instance}:")
                print(f"      {old_cell} -> {new_cell}")
        
    except Exception as e:
        print(f"[ERROR] Failed to decode action: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ========================================================================
    # Step 6: Write Output Files
    # ========================================================================
    
    print(f"\n[STEP 6] Writing output files")
    
    try:
        # Write actions file for TCL
        write_actions_file(
            args.output_actions,
            resizes,
            args.iteration,
            action_idx,
            state,
            metrics
        )
        
        # Write state log if requested
        if args.state_log:
            write_state_log(
                args.state_log,
                args.iteration,
                state,
                q_values,
                action_idx,
                timing_data
            )
            print(f"  State log: {args.state_log}")
        
    except Exception as e:
        print(f"[ERROR] Failed to write output: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ========================================================================
    # Done
    # ========================================================================
    
    print("\n" + "="*70)
    print(f"Agent completed successfully - Iteration {args.iteration}")
    print("="*70)


if __name__ == '__main__':
    main()
