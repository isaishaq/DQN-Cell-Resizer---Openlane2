"""
RL Environment for Cell Sizing
===============================

Gym-compatible environment for training RL agents to fix timing violations
through cell sizing.
"""

import gym
from gym import spaces
import numpy as np
import subprocess
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from discrete_action_space import DiscreteActionSpace, CellLibrary, ResizeAction


class CellSizingEnv(gym.Env):
    """
    OpenAI Gym environment for cell sizing optimization.
    
    State: Current timing metrics and cell properties
    Action: Discrete cell resize operations
    Reward: Improvement in timing slack, penalized by area/power increase
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self,
                 design_dir: str,
                 config_file: str,
                 max_steps: int = 50,
                 top_k_cells: int = 10,
                 reward_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the environment.
        
        Args:
            design_dir: Path to design directory
            config_file: OpenLane config file
            max_steps: Maximum optimization steps per episode
            top_k_cells: Number of cells to make actionable
            reward_weights: Weights for reward components {slack, area, power}
        """
        super().__init__()
        
        self.design_dir = Path(design_dir)
        self.config_file = config_file
        self.max_steps = max_steps
        self.top_k_cells = top_k_cells
        
        # Reward weights
        self.reward_weights = reward_weights or {
            'slack': 1.0,      # Slack improvement reward
            'area': -0.01,     # Area increase penalty
            'power': -0.005    # Power increase penalty
        }
        
        # Initialize action space
        self.cell_library = CellLibrary()
        self.action_space_mgr = DiscreteActionSpace(
            mode='single',
            top_k_cells=top_k_cells,
            library=self.cell_library
        )
        
        # Define Gym spaces
        self.action_space = spaces.Discrete(self.action_space_mgr.n_actions)
        
        # State space: [global_metrics + cell_features]
        # Global: WNS, TNS, num_violations, area, power
        # Per-cell (top_k): drive_strength, fanout, delay, slack_contrib
        state_dim = 5 + (top_k_cells * 4)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )
        
        # Episode tracking
        self.current_step = 0
        self.initial_wns = None
        self.initial_area = None
        self.current_state = None
        self.actionable_cells = []
        self.action_history = []
        
    def reset(self) -> np.ndarray:
        """
        Reset environment to initial design state.
        
        Returns:
            Initial observation (state vector)
        """
        self.current_step = 0
        self.action_history = []
        
        # Run initial timing analysis
        timing_data = self._run_timing_analysis()
        
        # Store initial metrics
        self.initial_wns = timing_data['global_metrics']['wns']
        self.initial_area = self._get_design_area()
        
        # Get actionable cells
        self.actionable_cells = self.action_space_mgr.get_actionable_cells(
            timing_data, worst_n_paths=10
        )
        
        # Extract state
        self.current_state = self._extract_state(timing_data)
        
        return self.current_state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Discrete action index
            
        Returns:
            observation: New state
            reward: Reward for this step
            done: Whether episode is finished
            info: Additional information
        """
        self.current_step += 1
        
        # Store previous metrics for reward calculation
        prev_timing = self._run_timing_analysis()
        prev_wns = prev_timing['global_metrics']['wns']
        prev_area = self._get_design_area()
        
        # Apply action (resize cells)
        resize_commands = self.action_space_mgr.apply_action(
            action, self.actionable_cells
        )
        
        # Execute resizing in design
        if resize_commands:
            self._apply_resizes(resize_commands)
            self.action_history.append({
                'step': self.current_step,
                'action': action,
                'resizes': resize_commands
            })
        
        # Run timing analysis on modified design
        new_timing = self._run_timing_analysis()
        new_wns = new_timing['global_metrics']['wns']
        new_area = self._get_design_area()
        
        # Update actionable cells with new timing data
        self.actionable_cells = self.action_space_mgr.get_actionable_cells(
            new_timing, worst_n_paths=10
        )
        
        # Extract new state
        new_state = self._extract_state(new_timing)
        self.current_state = new_state
        
        # Calculate reward
        reward = self._calculate_reward(
            prev_wns, new_wns,
            prev_area, new_area
        )
        
        # Check if episode is done
        done = (
            self.current_step >= self.max_steps or
            new_wns >= 0  # All violations fixed!
        )
        
        # Info dictionary
        info = {
            'wns': new_wns,
            'area': new_area,
            'num_violations': new_timing['global_metrics']['num_violations'],
            'resizes_applied': len(resize_commands),
            'slack_improvement': new_wns - prev_wns
        }
        
        return new_state, reward, done, info
    
    def _extract_state(self, timing_data: Dict) -> np.ndarray:
        """
        Extract state vector from timing data.
        
        State representation:
        - Global metrics: [WNS, TNS, num_violations, normalized_area, step/max_steps]
        - Per-cell features: [drive_strength, fanout, delay, slack_contrib] for top-k cells
        """
        global_metrics = timing_data['global_metrics']
        
        # Normalize metrics
        wns_norm = global_metrics['wns'] / 16.0  # Normalize by clock period
        tns_norm = global_metrics['tns'] / 100.0
        violations_norm = global_metrics['num_violations'] / 100.0
        area_norm = self._get_design_area() / (self.initial_area or 1.0)
        step_norm = self.current_step / self.max_steps
        
        state = [wns_norm, tns_norm, violations_norm, area_norm, step_norm]
        
        # Add cell features (pad if fewer than top_k cells)
        for i in range(self.top_k_cells):
            if i < len(self.actionable_cells):
                cell = self.actionable_cells[i]
                state.extend([
                    cell.current_drive_strength / 16.0,  # Normalize
                    cell.fanout / 20.0,
                    cell.delay / 2.0,
                    cell.slack_contribution / 5.0
                ])
            else:
                # Pad with zeros
                state.extend([0.0, 0.0, 0.0, 0.0])
        
        return np.array(state, dtype=np.float32)
    
    def _calculate_reward(self, prev_wns: float, new_wns: float,
                         prev_area: float, new_area: float) -> float:
        """
        Calculate reward for the action taken.
        
        Reward components:
        1. Slack improvement (main objective)
        2. Area penalty (constraint)
        3. Progress bonus (if closer to goal)
        """
        # Slack improvement reward
        slack_improvement = new_wns - prev_wns
        slack_reward = slack_improvement * self.reward_weights['slack']
        
        # Area penalty
        area_increase = (new_area - prev_area) / prev_area if prev_area > 0 else 0
        area_penalty = area_increase * self.reward_weights['area']
        
        # Bonus for fixing all violations
        completion_bonus = 10.0 if new_wns >= 0 else 0.0
        
        # Penalty for making things worse
        degradation_penalty = -5.0 if new_wns < prev_wns else 0.0
        
        total_reward = (slack_reward + area_penalty + 
                       completion_bonus + degradation_penalty)
        
        return total_reward
    
    def _run_timing_analysis(self) -> Dict:
        """
        Run timing analysis and return parsed results.
        
        In practice, this would:
        1. Run OpenSTA or OpenROAD timing analysis
        2. Parse the timing report
        3. Return structured timing data
        """
        # Placeholder - implement actual timing analysis
        # For now, return mock data structure
        
        # In real implementation:
        # subprocess.run(['openroad', '-exit', 'timing_script.tcl'])
        # timing_data = parse_timing_report('reports/max.rpt')
        
        # Mock data for demonstration
        return {
            'paths': [],
            'global_metrics': {
                'wns': -3.93,
                'tns': -85.47,
                'num_violations': 52
            }
        }
    
    def _get_design_area(self) -> float:
        """Get current design area."""
        # Placeholder - get from OpenROAD metrics
        return 10000.0  # Mock area value
    
    def _apply_resizes(self, resize_commands: Dict[str, Tuple[str, str]]):
        """
        Apply cell resizing commands to the design.
        
        This modifies the netlist by replacing cell instances.
        
        Args:
            resize_commands: Dict mapping instance_name to (old_cell, new_cell)
        """
        # In practice, you would:
        # 1. Read the current netlist (DEF or Verilog)
        # 2. Replace cell instances
        # 3. Write modified netlist
        # 4. Re-run placement/routing if needed
        
        print(f"Applying {len(resize_commands)} resize operations:")
        for instance, (old_cell, new_cell) in resize_commands.items():
            print(f"  {instance}: {old_cell} -> {new_cell}")
        
        # Example TCL script for OpenROAD:
        tcl_script = self._generate_resize_script(resize_commands)
        
        # Write script and execute
        script_path = self.design_dir / "resize_cells.tcl"
        with open(script_path, 'w') as f:
            f.write(tcl_script)
        
        # Run OpenROAD with script
        # subprocess.run(['openroad', '-exit', str(script_path)])
    
    def _generate_resize_script(self, resize_commands: Dict) -> str:
        """Generate TCL script for cell resizing."""
        script = "# Auto-generated cell resize script\n\n"
        
        for instance, (old_cell, new_cell) in resize_commands.items():
            # OpenROAD command to resize a cell
            script += f"resize_instance {instance} {new_cell}\n"
        
        script += "\n# Update timing\n"
        script += "report_checks -path_delay max -format full_clock_expanded\n"
        
        return script
    
    def render(self, mode='human'):
        """Render the environment state."""
        if mode == 'human':
            print(f"\n=== Step {self.current_step}/{self.max_steps} ===")
            print(f"Actionable cells: {len(self.actionable_cells)}")
            if self.current_state is not None:
                wns = self.current_state[0] * 16.0  # Denormalize
                print(f"Current WNS: {wns:.3f} ns")
                print(f"Initial WNS: {self.initial_wns:.3f} ns")
                print(f"Improvement: {wns - self.initial_wns:.3f} ns")
    
    def close(self):
        """Clean up resources."""
        pass


# ============================================================================
# Example RL Training Loop
# ============================================================================

def train_dqn_agent():
    """
    Example training loop using DQN.
    
    This is a simplified example. In practice, you'd use a library like
    Stable-Baselines3, Ray RLlib, or custom PyTorch implementation.
    """
    from collections import deque
    import random
    
    # Create environment
    env = CellSizingEnv(
        design_dir='/home/isaishaq/openlane2/designs/picorv_test',
        config_file='config.json',
        max_steps=30,
        top_k_cells=10
    )
    
    # Training hyperparameters
    num_episodes = 1000
    epsilon = 1.0  # Exploration rate
    epsilon_decay = 0.995
    epsilon_min = 0.01
    
    # Episode history
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                # Explore: random action
                action = env.action_space.sample()
            else:
                # Exploit: best action from Q-network
                # action = agent.select_action(state)
                action = 0  # Placeholder
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            # replay_buffer.add(state, action, reward, next_state, done)
            
            # Update Q-network
            # if len(replay_buffer) > batch_size:
            #     agent.train_step(replay_buffer.sample(batch_size))
            
            episode_reward += reward
            state = next_state
            
            # Render (optional)
            if episode % 10 == 0:
                env.render()
        
        # Decay exploration
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        episode_rewards.append(episode_reward)
        
        print(f"Episode {episode}: Reward = {episode_reward:.2f}, "
              f"WNS = {info['wns']:.3f}, Epsilon = {epsilon:.3f}")
        
        # Save checkpoint
        if episode % 100 == 0:
            print(f"Saving checkpoint at episode {episode}")
            # agent.save(f'checkpoints/dqn_{episode}.pt')
    
    env.close()
    
    return episode_rewards


if __name__ == '__main__':
    # Example: just demonstrate the environment
    env = CellSizingEnv(
        design_dir='/home/isaishaq/openlane2/designs/picorv_test',
        config_file='config.json',
        max_steps=10,
        top_k_cells=5
    )
    
    print("Environment created successfully!")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"State dimension: {env.observation_space.shape[0]}")
    
    # Run one episode with random actions
    print("\n=== Running sample episode ===")
    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    
    for step in range(3):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        
        print(f"\nStep {step + 1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.3f}")
        print(f"  WNS: {info['wns']:.3f}")
        print(f"  Done: {done}")
        
        if done:
            break
    
    env.close()
