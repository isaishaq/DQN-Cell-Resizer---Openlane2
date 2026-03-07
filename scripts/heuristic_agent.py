#!/usr/bin/env python3
"""
Heuristic-Based Cell Sizing Agent (No Training Required)
=========================================================

This provides a simple rule-based policy for cell sizing that works
without any pre-trained model. Good starting point before DQN training.

Rules:
1. Identify cells on critical paths with high delay
2. Upsize cells with high fanout
3. Upsize cells with slow transitions
4. Prioritize cells closer to endpoints

Usage:
    python3 heuristic_agent.py \
        --timing-report timing.rpt \
        --output-actions actions.txt \
        --strategy aggressive
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# Import existing modules
from timing_parser import parse_timing_report
from discrete_action_space import DiscreteActionSpace, CellLibrary, Cell


class HeuristicAgent:
    """
    Simple heuristic-based agent for cell sizing.
    No training required - uses design rules.
    """
    
    def __init__(self, strategy: str = 'balanced'):
        """
        Initialize heuristic agent.
        
        Args:
            strategy: 'aggressive', 'balanced', or 'conservative'
        """
        self.strategy = strategy
        self.library = CellLibrary()
    
    def select_action(
        self,
        actionable_cells: List[Cell],
        timing_data: Dict,
        action_space: DiscreteActionSpace
    ) -> int:
        """
        Select action using heuristic rules.
        
        Strategy:
        1. Score each cell based on:
           - Delay contribution (higher = more critical)
           - Fanout (higher = more impact)
           - Current drive strength (lower = more room to upsize)
           - Slack contribution (worse slack = higher priority)
        
        2. Pick top cell and decide sizing:
           - If high delay + high fanout → upsize significantly
           - If high delay + low fanout → upsize moderately
           - If low delay but high fanout → upsize slightly
        
        Args:
            actionable_cells: List of cells to consider
            timing_data: Parsed timing data
            action_space: Action space for encoding
            
        Returns:
            Action index
        """
        if not actionable_cells:
            return 0  # No-op action
        
        # Score all cells
        scores = []
        for cell in actionable_cells:
            score = self._score_cell(cell, timing_data)
            scores.append((score, cell))
        
        # Sort by score (highest first)
        scores.sort(reverse=True, key=lambda x: x[0])
        
        # Get top cell
        top_score, top_cell = scores[0]
        
        # Decide how much to upsize
        upsize_levels = self._decide_upsize(top_cell, timing_data)
        
        # Create action: resize top cell
        action = self._encode_action(
            top_cell,
            upsize_levels,
            actionable_cells,
            action_space
        )
        
        return action
    
    def _score_cell(self, cell: Cell, timing_data: Dict) -> float:
        """
        Score a cell based on how critical it is.
        Higher score = higher priority to resize.
        """
        wns = timing_data['global_metrics']['wns']
        
        # Component scores
        delay_score = cell.delay * 10.0  # Weight delay heavily
        fanout_score = cell.fanout * 2.0  # Fanout matters
        drive_score = (16.0 - cell.current_drive_strength) * 1.0  # Room to upsize
        slack_score = abs(cell.slack_contribution) / abs(wns) if wns != 0 else 0
        
        # Strategy-dependent weighting
        if self.strategy == 'aggressive':
            weights = [1.5, 1.2, 0.8, 1.5]  # Favor delay and slack
        elif self.strategy == 'conservative':
            weights = [0.8, 1.0, 1.2, 1.0]  # More cautious
        else:  # balanced
            weights = [1.0, 1.0, 1.0, 1.0]
        
        score = (
            delay_score * weights[0] +
            fanout_score * weights[1] +
            drive_score * weights[2] +
            slack_score * weights[3]
        )
        
        return score
    
    def _decide_upsize(self, cell: Cell, timing_data: Dict) -> int:
        """
        Decide how many drive strength levels to increase.
        
        Returns:
            1, 2, 3, or 4 (drive strength multiplier)
        """
        wns = timing_data['global_metrics']['wns']
        
        # Check how bad timing is
        timing_pressure = abs(wns) / 10.0  # Normalize by 10ns
        
        # Check cell characteristics
        high_delay = cell.delay > 0.5
        high_fanout = cell.fanout > 5
        low_drive = cell.current_drive_strength <= 2
        
        if self.strategy == 'aggressive':
            if timing_pressure > 0.5 and (high_delay or high_fanout):
                return 4  # 4x drive
            elif timing_pressure > 0.3:
                return 2  # 2x drive
            else:
                return 1
        
        elif self.strategy == 'conservative':
            if high_delay and high_fanout and low_drive:
                return 2
            else:
                return 1
        
        else:  # balanced
            if high_delay and high_fanout:
                return 2
            elif high_delay or high_fanout:
                return 1
            else:
                return 1
        
    def _encode_action(
        self,
        cell: Cell,
        upsize_levels: int,
        actionable_cells: List[Cell],
        action_space: DiscreteActionSpace
    ) -> int:
        """
        Encode cell resize decision as action index.
        
        Since action space encodes (cell_idx, new_drive_strength),
        we need to find the corresponding action index.
        """
        # Find cell index in actionable list
        try:
            cell_idx = actionable_cells.index(cell)
        except ValueError:
            return 0  # Cell not found, return no-op
        
        # Get target drive strength
        current_drive = cell.current_drive_strength
        target_drive = min(current_drive * upsize_levels, 16)  # Cap at 16
        
        # Find corresponding action in action space
        # Action space format: cell_idx * num_sizes + size_idx
        # For simplicity, map to discrete action
        
        # Simple encoding: top-k cells (10) × 3 actions (upsize 1x, 2x, 4x)
        # = 30 actions
        if upsize_levels == 1:
            size_action = 0
        elif upsize_levels == 2:
            size_action = 1
        else:  # 4
            size_action = 2
        
        action_idx = cell_idx * 3 + size_action
        
        # Make sure it's within bounds
        action_idx = min(action_idx, action_space.n_actions - 1)
        
        return action_idx


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description='Heuristic Cell Sizing Agent',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--timing-report',
        required=True,
        help='Path to timing report file'
    )
    parser.add_argument(
        '--output-actions',
        required=True,
        help='Path to output actions file'
    )
    parser.add_argument(
        '--strategy',
        choices=['aggressive', 'balanced', 'conservative'],
        default='balanced',
        help='Sizing strategy'
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
        help='Number of cells to consider'
    )
    parser.add_argument(
        '--worst-n-paths',
        type=int,
        default=5,
        help='Number of worst paths to analyze'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print(f"Heuristic Agent - Iteration {args.iteration}")
    print(f"Strategy: {args.strategy}")
    print("="*70)
    
    # Parse timing report
    if not os.path.exists(args.timing_report):
        print(f"[ERROR] Timing report not found: {args.timing_report}")
        sys.exit(1)
    
    print(f"\n[STEP 1] Parsing timing report...")
    timing_data = parse_timing_report(args.timing_report)
    metrics = timing_data['global_metrics']
    print(f"  WNS: {metrics['wns']:.4f} ns")
    print(f"  TNS: {metrics['tns']:.4f} ns")
    
    # Get actionable cells
    print(f"\n[STEP 2] Identifying actionable cells...")
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
    
    if not actionable_cells:
        print(f"[WARNING] No actionable cells found")
        # Write empty actions
        os.makedirs(os.path.dirname(args.output_actions), exist_ok=True)
        with open(args.output_actions, 'w') as f:
            f.write(f"# Iteration: {args.iteration}\n")
            f.write(f"# No actions\n")
        sys.exit(0)
    
    # Select action using heuristics
    print(f"\n[STEP 3] Selecting action using heuristic rules...")
    agent = HeuristicAgent(strategy=args.strategy)
    action_idx = agent.select_action(actionable_cells, timing_data, action_space)
    print(f"  Selected action: {action_idx}")
    
    # Decode action
    print(f"\n[STEP 4] Decoding action to resize commands...")
    resizes = action_space.apply_action(action_idx, actionable_cells)
    print(f"  Generated {len(resizes)} resize commands")
    
    if args.verbose and resizes:
        for inst, (old, new) in resizes.items():
            print(f"    {inst}: {old} → {new}")
    
    # Write output
    print(f"\n[STEP 5] Writing actions file...")
    os.makedirs(os.path.dirname(args.output_actions), exist_ok=True)
    with open(args.output_actions, 'w') as f:
        f.write(f"# Iteration: {args.iteration}\n")
        f.write(f"# Strategy: {args.strategy}\n")
        f.write(f"# WNS: {metrics['wns']:.4f}\n")
        f.write(f"# TNS: {metrics['tns']:.4f}\n\n")
        
        if resizes:
            for inst, (old, new) in resizes.items():
                f.write(f"{inst} {new}\n")
        else:
            f.write("# No actions\n")
    
    print(f"  Wrote to: {args.output_actions}")
    print("\n" + "="*70)
    print("Heuristic agent completed")
    print("="*70)


if __name__ == '__main__':
    main()
