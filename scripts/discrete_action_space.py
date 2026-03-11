"""
Discrete Action Space for DRL-based Cell Sizing
================================================

This module defines the discrete action space for automatic cell sizing
to fix timing violations using Deep Reinforcement Learning.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import IntEnum


class ResizeAction(IntEnum):
    """Discrete resize actions."""
    DOWNSIZE = 0    # Reduce drive strength by 1 level
    KEEP = 1        # Keep current size
    UPSIZE = 2      # Increase drive strength by 1 level


@dataclass
class Cell:
    """Represents a cell in the timing path."""
    instance_name: str
    cell_type: str
    current_drive_strength: int
    fanout: int
    delay: float
    slack_contribution: float
    is_resizable: bool = True
    
    def __repr__(self):
        return f"Cell({self.instance_name}, {self.cell_type}, drive={self.current_drive_strength})"


class CellLibrary:
    """
    Manages available cell types and their drive strength variants.
    Based on SkyWater 130nm PDK standard cells.
    """
    
    def __init__(self):
        # Define available drive strengths for each cell type
        # Format: cell_base_type -> list of available drive strengths
        self.available_sizes = {
            'buf': [1, 2, 4, 6, 8, 12, 16],
            'clkbuf': [1, 2, 4, 8, 16],
            'inv': [1, 2, 4, 6, 8],
            'and2': [0, 1, 2, 4],
            'and3': [1, 2, 4],
            'and4': [1, 2, 4],
            'nand2': [1, 2, 4, 8],
            'nand3': [1, 2, 4, 8],
            'nand4': [1, 2, 4],
            'nor2': [1, 2, 4, 8],
            'nor3': [1, 2, 4],
            'nor4': [1, 2, 4],
            'or2': [0, 1, 2, 4],
            'or3': [1, 2, 4],
            'or4': [1, 2, 4],
            'xor2': [1, 2, 4],
            'xnor2': [1, 2, 4],
            'mux2': [1, 2, 4, 8],
            'mux4': [1, 2, 4],
            'a21o': [1, 2, 4],
            'a21oi': [1, 2, 4],
            'a211o': [1, 2, 4],
            'a211oi': [1, 2, 4],
            'a221o': [1, 2, 4],
            'a22o': [1, 2, 4],
            'o21a': [1, 2, 4],
            'o21ai': [1, 2, 4],
            'o211a': [1, 2, 4],
            'o211ai': [1, 2, 4],
            'dfxtp': [1, 2, 4],  # Flip-flops
        }
        
        # Cells that typically shouldn't be resized
        self.non_resizable_types = {
            'clkbuf',  # Clock buffers - usually managed separately
            'dfxtp',   # Flip-flops - usually kept at specific size
        }
    
    def get_available_sizes(self, cell_base_type: str) -> List[int]:
        """Get available drive strengths for a cell type."""
        return self.available_sizes.get(cell_base_type, [1, 2, 4])
    
    def can_resize(self, cell_base_type: str, current_size: int, action: ResizeAction) -> bool:
        """Check if a resize action is valid."""
        available = self.get_available_sizes(cell_base_type)
        
        if action == ResizeAction.KEEP:
            return True
        
        if current_size not in available:
            return False
        
        current_idx = available.index(current_size)
        
        if action == ResizeAction.UPSIZE:
            return current_idx < len(available) - 1
        else:  # DOWNSIZE
            return current_idx > 0
    
    def get_new_size(self, cell_base_type: str, current_size: int, action: ResizeAction) -> int:
        """Get the new drive strength after applying action."""
        if action == ResizeAction.KEEP:
            return current_size
        
        available = self.get_available_sizes(cell_base_type)
        
        if current_size not in available:
            return current_size
        
        current_idx = available.index(current_size)
        
        if action == ResizeAction.UPSIZE and current_idx < len(available) - 1:
            return available[current_idx + 1]
        elif action == ResizeAction.DOWNSIZE and current_idx > 0:
            return available[current_idx - 1]
        else:
            return current_size  # Can't resize further
    
    def is_resizable(self, cell_base_type: str) -> bool:
        """Check if a cell type should be resized."""
        return cell_base_type not in self.non_resizable_types


class DiscreteActionSpace:
    """
    Defines the discrete action space for cell sizing.
    
    Two approaches are supported:
    1. Single-cell action: Select one cell and one action (cell_idx, action)
    2. Multi-cell action: Apply action to multiple critical cells simultaneously
    """
    
    def __init__(self, 
                 mode: str = 'single',  # 'single' or 'multi'
                 top_k_cells: int = 10,  # Number of most critical cells to consider
                 library: Optional[CellLibrary] = None):
        """
        Initialize action space.
        
        Args:
            mode: 'single' for one cell at a time, 'multi' for multiple cells
            top_k_cells: Number of most critical cells to make actionable
            library: Cell library with available sizes
        """
        self.mode = mode
        self.top_k_cells = top_k_cells
        self.library = library or CellLibrary()
        
        # Action dimensions
        if mode == 'single':
            # Action = (cell_index, resize_action)
            # Total actions = top_k_cells * 3 (downsize/keep/upsize)
            self.n_actions = top_k_cells * len(ResizeAction)
        else:
            # Multi-cell: each of top_k cells can take 3 actions
            # Total actions = 3^top_k (combinatorial explosion - use with care!)
            self.n_actions = len(ResizeAction) ** top_k_cells
    
    def get_actionable_cells(self, parsed_timing_data: Dict, 
                            worst_n_paths: int = 5) -> List[Cell]:
        """
        Identify the most critical cells that should be actionable.
        
        Strategy: Select cells from the worst timing paths that contribute
        most to timing violations.
        
        Args:
            parsed_timing_data: Parsed timing report data
            worst_n_paths: Number of worst paths to consider
            
        Returns:
            List of actionable cells, sorted by criticality
        """
        paths = parsed_timing_data['paths']
        
        # Sort paths by slack (most negative first)
        sorted_paths = sorted(paths, key=lambda p: p.get('slack', 0))
        critical_paths = sorted_paths[:worst_n_paths]
        
        # Collect cells with their slack contribution
        cell_criticality = {}  # instance_name -> (cell, criticality_score)
        
        for path in critical_paths:
            path_slack = path.get('slack', 0)
            cells = path.get('cells', [])
            
            for i, cell_data in enumerate(cells):
                instance = cell_data.get('instance_name', '')
                
                # Skip clock buffers and flip-flops
                cell_type_full = cell_data.get('cell_type', '')
                cell_base_type = self._extract_base_cell_type(cell_type_full)
                
                if not self.library.is_resizable(cell_base_type):
                    continue
                
                # Calculate criticality score
                # Higher weight for cells later in the path (closer to endpoint)
                position_weight = (i + 1) / len(cells)
                delay_contribution = cell_data.get('delay', 0)
                slack_weight = abs(path_slack) if path_slack < 0 else 0
                
                criticality = (delay_contribution * position_weight * 
                              (1 + slack_weight))
                
                if instance not in cell_criticality:
                    cell = Cell(
                        instance_name=instance,
                        cell_type=cell_type_full,
                        current_drive_strength=cell_data.get('drive_strength', 1),
                        fanout=cell_data.get('fanout', 0),
                        delay=delay_contribution,
                        slack_contribution=criticality,
                        is_resizable=True
                    )
                    cell_criticality[instance] = (cell, criticality)
                else:
                    # Cell appears in multiple paths - increase criticality
                    existing_cell, existing_crit = cell_criticality[instance]
                    cell_criticality[instance] = (existing_cell, 
                                                  existing_crit + criticality)
        
        # Sort by criticality and return top-k
        sorted_cells = sorted(cell_criticality.values(), 
                            key=lambda x: x[1], reverse=True)
        
        actionable_cells = [cell for cell, _ in sorted_cells[:self.top_k_cells]]
        
        return actionable_cells
    
    def _extract_base_cell_type(self, full_cell_type: str) -> str:
        """Extract base cell type from full name."""
        # e.g., sky130_fd_sc_hd__buf_4 -> buf
        import re
        match = re.search(r'__([a-z0-9]+)_\d+', full_cell_type)
        if match:
            return match.group(1)
        # Handle cells without drive strength
        match = re.search(r'__([a-z0-9]+)', full_cell_type)
        return match.group(1) if match else full_cell_type
    
    def action_to_cell_resize(self, action: int, 
                              actionable_cells: List[Cell]) -> List[Tuple[Cell, ResizeAction]]:
        """
        Convert discrete action index to cell resize operations.
        
        Args:
            action: Discrete action index
            actionable_cells: List of cells that can be resized
            
        Returns:
            List of (cell, resize_action) tuples
        """
        if self.mode == 'single':
            # Decode single-cell action
            cell_idx = action // len(ResizeAction)
            resize_action = ResizeAction(action % len(ResizeAction))
            
            if cell_idx < len(actionable_cells):
                return [(actionable_cells[cell_idx], resize_action)]
            else:
                return []  # Invalid action
        
        else:  # multi mode
            # Decode multi-cell action
            # Each cell gets an action from the combined action index
            resizes = []
            remaining = action
            
            for cell in actionable_cells[:self.top_k_cells]:
                cell_action = ResizeAction(remaining % len(ResizeAction))
                resizes.append((cell, cell_action))
                remaining //= len(ResizeAction)
            
            return resizes
    
    def apply_action(self, action: int, 
                     actionable_cells: List[Cell]) -> Dict[str, Tuple[str, str]]:
        """
        Apply action and return resize commands.
        
        Args:
            action: Discrete action index
            actionable_cells: List of actionable cells
            
        Returns:
            Dictionary mapping instance_name to (old_cell_type, new_cell_type)
        """
        resizes = self.action_to_cell_resize(action, actionable_cells)
        resize_commands = {}
        
        for cell, resize_action in resizes:
            cell_base_type = self._extract_base_cell_type(cell.cell_type)
            old_size = cell.current_drive_strength
            
            # Check if resize is valid
            if not self.library.can_resize(cell_base_type, old_size, resize_action):
                print(f"Invalid resize action for cell {cell.instance_name}: ")
                print(f"  Cell type: {cell.cell_type}, current size: {old_size}, "
                      f"requested action: {resize_action.name}")
                # Get next actionable cells instead

                continue  # Skip invalid resize
            
            new_size = self.library.get_new_size(cell_base_type, old_size, resize_action)
            
            if new_size != old_size:
                # Generate new cell type name
                old_cell_name = cell.cell_type
                new_cell_name = old_cell_name.replace(f'_{old_size}', f'_{new_size}')
                
                resize_commands[cell.instance_name] = (old_cell_name, new_cell_name)
        
        return resize_commands
    
    def get_valid_actions_mask(self, actionable_cells: List[Cell]) -> np.ndarray:
        """
        Get a mask indicating which actions are valid.
        Useful for masking invalid actions during RL training.
        
        Returns:
            Boolean array of shape (n_actions,) where True means valid action
        """
        mask = np.zeros(self.n_actions, dtype=bool)
        
        if self.mode == 'single':
            for action_idx in range(self.n_actions):
                cell_idx = action_idx // len(ResizeAction)
                resize_action = ResizeAction(action_idx % len(ResizeAction))
                
                if cell_idx < len(actionable_cells):
                    cell = actionable_cells[cell_idx]
                    cell_base_type = self._extract_base_cell_type(cell.cell_type)
                    
                    if self.library.can_resize(cell_base_type, 
                                               cell.current_drive_strength, 
                                               resize_action):
                        mask[action_idx] = True
        
        else:  # multi mode - more complex, simplified here
            mask[:] = True  # Assume all combinations valid for now
        
        return mask


# ============================================================================
# Example Usage and Testing
# ============================================================================

def example_usage():
    """Demonstrate how to use the discrete action space."""
    
    # 1. Parse timing report (assuming you have parsed data)
    parsed_timing_data = {
        'paths': [
            {
                'slack': -3.93,
                'cells': [
                    {
                        'instance_name': 'fanout1291',
                        'cell_type': 'sky130_fd_sc_hd__buf_4',
                        'drive_strength': 4,
                        'fanout': 11,
                        'delay': 0.39
                    },
                    {
                        'instance_name': 'fanout1290',
                        'cell_type': 'sky130_fd_sc_hd__buf_4',
                        'drive_strength': 4,
                        'fanout': 10,
                        'delay': 0.46
                    },
                    {
                        'instance_name': 'gate123',
                        'cell_type': 'sky130_fd_sc_hd__nand2_2',
                        'drive_strength': 2,
                        'fanout': 3,
                        'delay': 0.41
                    },
                ]
            },
            {
                'slack': -3.55,
                'cells': [
                    {
                        'instance_name': 'fanout1291',  # Same cell in another path
                        'cell_type': 'sky130_fd_sc_hd__buf_4',
                        'drive_strength': 4,
                        'fanout': 11,
                        'delay': 0.39
                    },
                ]
            }
        ]
    }
    
    # 2. Create action space
    action_space = DiscreteActionSpace(
        mode='single',
        top_k_cells=10,
        library=CellLibrary()
    )
    
    print(f"Action space size: {action_space.n_actions}")
    print(f"(10 cells × 3 actions = {10 * 3} total actions)\n")
    
    # 3. Get actionable cells from timing data
    actionable_cells = action_space.get_actionable_cells(
        parsed_timing_data,
        worst_n_paths=5
    )
    
    print(f"Found {len(actionable_cells)} actionable cells:")
    for i, cell in enumerate(actionable_cells):
        print(f"  {i}: {cell}")
    print()
    
    # 4. Sample an action (e.g., from RL agent)
    action = 2  # Upsize the first critical cell
    
    # 5. Decode action
    resizes = action_space.action_to_cell_resize(action, actionable_cells)
    print(f"Action {action} decodes to:")
    for cell, resize_action in resizes:
        print(f"  {cell.instance_name}: {resize_action.name} "
              f"(current drive: {cell.current_drive_strength})")
    print()
    
    # 6. Apply action and get resize commands
    resize_commands = action_space.apply_action(action, actionable_cells)
    print("Resize commands to apply:")
    for instance, (old_cell, new_cell) in resize_commands.items():
        print(f"  {instance}: {old_cell} -> {new_cell}")
    print()
    
    # 7. Get valid actions mask (for masked RL)
    valid_mask = action_space.get_valid_actions_mask(actionable_cells)
    print(f"Valid actions: {valid_mask.sum()} out of {len(valid_mask)}")
    print(f"First 15 actions valid? {valid_mask[:15]}")


if __name__ == '__main__':
    example_usage()
