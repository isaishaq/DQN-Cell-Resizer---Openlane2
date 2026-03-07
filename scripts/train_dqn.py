#!/usr/bin/env python3
"""
DQN Training Script for Cell Sizing
====================================

Train a Deep Q-Network to optimize cell sizing for timing closure.

Training Flow:
1. Load multiple designs
2. For each episode:
   - Initialize design state
   - Collect trajectory using ε-greedy policy
   - Store transitions in replay buffer
   - Train DQN on mini-batches
3. Save trained model

Usage:
    python3 train_dqn.py \\
        --designs designs_train.txt \\
        --episodes 1000 \\
        --output model_trained.pth \\
        --log-dir logs/

Author: DQN Training Team
"""

import os
import sys
import argparse
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Deque
from collections import deque
import time

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Import DQN components
from dqn_agent import DQNNetwork, extract_state_features
from discrete_action_space import DiscreteActionSpace, CellLibrary, Cell
from timing_parser import parse_timing_report


# ============================================================================
# Replay Buffer
# ============================================================================

class ReplayBuffer:
    """
    Experience replay buffer for DQN training.
    Stores transitions (s, a, r, s', done) and samples mini-batches.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer: Deque = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a random mini-batch from the buffer.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        transitions = random.sample(self.buffer, batch_size)
        
        # Unzip transitions
        states, actions, rewards, next_states, dones = zip(*transitions)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        """Return current buffer size."""
        return len(self.buffer)
    
    def save(self, path: str):
        """Save buffer to disk."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(list(self.buffer), f)
        print(f"[BUFFER] Saved {len(self)} transitions to {path}")
    
    def load(self, path: str):
        """Load buffer from disk."""
        import pickle
        with open(path, 'rb') as f:
            transitions = pickle.load(f)
        self.buffer.extend(transitions)
        print(f"[BUFFER] Loaded {len(transitions)} transitions from {path}")


# ============================================================================
# Reward Function
# ============================================================================

def calculate_reward(
    old_wns: float,
    new_wns: float,
    old_tns: float,
    new_tns: float,
    old_area: Optional[float] = None,
    new_area: Optional[float] = None,
    timing_met: bool = False,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Calculate reward for a cell sizing action.
    
    Reward components:
    - WNS improvement: Positive reward for timing improvement
    - TNS improvement: Reward for reducing total violations
    - Area penalty: Small penalty for area increase
    - Timing met bonus: Large bonus for achieving timing closure
    
    Args:
        old_wns: Previous worst negative slack
        new_wns: New worst negative slack
        old_tns: Previous total negative slack
        new_tns: New total negative slack
        old_area: Previous design area (optional)
        new_area: New design area (optional)
        timing_met: Whether timing constraints are now met
        weights: Custom weights for reward components
        
    Returns:
        Scalar reward value
    """
    if weights is None:
        weights = {
            'wns': 10.0,      # WNS improvement weight
            'tns': 1.0,       # TNS improvement weight
            'area': 0.01,     # Area penalty weight
            'closure': 50.0   # Timing closure bonus
        }
    
    # === WNS Improvement ===
    wns_improvement = new_wns - old_wns
    wns_reward = weights['wns'] * wns_improvement
    
    # === TNS Improvement ===
    tns_improvement = new_tns - old_tns
    tns_reward = weights['tns'] * tns_improvement
    
    # === Area Penalty ===
    area_penalty = 0.0
    if old_area is not None and new_area is not None:
        area_increase = (new_area - old_area) / old_area
        area_penalty = -weights['area'] * area_increase
    
    # === Timing Closure Bonus ===
    closure_bonus = 0.0
    if timing_met and old_wns < 0:
        closure_bonus = weights['closure']
    
    # === Combined Reward ===
    total_reward = wns_reward + tns_reward + area_penalty + closure_bonus
    
    # === Penalty for making timing worse ===
    if new_wns < old_wns:
        total_reward *= 0.5  # Halve reward if timing degrades
    
    return total_reward


# ============================================================================
# Design Episode Runner
# ============================================================================

class DesignEpisode:
    """
    Manages a single training episode on a design.
    Interfaces with OpenROAD/TCL to apply actions and get states.
    """
    
    def __init__(
        self,
        design_name: str,
        design_dir: Path,
        run_dir: Path,
        max_steps: int = 50
    ):
        """
        Initialize design episode.
        
        Args:
            design_name: Design identifier
            design_dir: Path to design folder
            run_dir: Path to run directory
            max_steps: Maximum steps per episode
        """
        self.design_name = design_name
        self.design_dir = Path(design_dir)
        self.run_dir = Path(run_dir)
        self.max_steps = max_steps
        
        self.current_step = 0
        self.initial_wns = None
        self.action_history = []
        
        # Create episode directories
        self.episode_dir = self.run_dir / f"episode_{int(time.time())}"
        self.episode_dir.mkdir(parents=True, exist_ok=True)
    
    def reset(self) -> np.ndarray:
        """
        Reset design to initial state and return initial observation.
        
        Returns:
            Initial state vector
        """
        self.current_step = 0
        self.action_history = []
        
        # Run OpenROAD to get initial timing
        self._run_openroad_initial()
        
        # Parse timing report
        timing_data = self._get_timing_report()
        
        # Get actionable cells
        action_space = DiscreteActionSpace(
            mode='single',
            top_k_cells=10,
            library=CellLibrary()
        )
        self.actionable_cells = action_space.get_actionable_cells(
            timing_data,
            worst_n_paths=5
        )
        
        # Extract state
        state = extract_state_features(
            timing_data,
            self.actionable_cells,
            top_k_cells=10
        )
        
        self.initial_wns = timing_data['global_metrics']['wns']
        
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Apply action and return next state, reward, done, info.
        
        Args:
            action: Action index to apply
            
        Returns:
            (next_state, reward, done, info)
        """
        self.current_step += 1
        
        # Get current metrics
        old_timing = self._get_timing_report()
        old_wns = old_timing['global_metrics']['wns']
        old_tns = old_timing['global_metrics']['tns']
        
        # Apply action
        self._apply_action(action)
        
        # Get new metrics
        new_timing = self._get_timing_report()
        new_wns = new_timing['global_metrics']['wns']
        new_tns = new_timing['global_metrics']['tns']
        
        # Calculate reward
        timing_met = new_wns >= 0
        reward = calculate_reward(
            old_wns, new_wns,
            old_tns, new_tns,
            timing_met=timing_met
        )
        
        # Get next state
        next_state = extract_state_features(
            new_timing,
            self.actionable_cells,
            top_k_cells=10
        )
        
        # Check if done
        done = timing_met or self.current_step >= self.max_steps
        
        # Info
        info = {
            'step': self.current_step,
            'wns': new_wns,
            'tns': new_tns,
            'timing_met': timing_met,
            'reward': reward
        }
        
        return next_state, reward, done, info
    
    def _run_openroad_initial(self):
        """Run OpenROAD to initialize design state."""
        # This would call your OpenROAD TCL script
        # For now, we'll use the existing setup
        pass
    
    def _get_timing_report(self) -> Dict:
        """Get current timing report."""
        # Read from the most recent timing report file
        report_path = self.run_dir / "reports" / "timing.rpt"
        if report_path.exists():
            return parse_timing_report(str(report_path))
        else:
            # Generate new report
            self._generate_timing_report()
            return parse_timing_report(str(report_path))
    
    def _apply_action(self, action: int):
        """Apply resize action using OpenROAD."""
        # Decode action to resize command
        action_space = DiscreteActionSpace(
            mode='single',
            top_k_cells=10,
            library=CellLibrary()
        )
        resizes = action_space.apply_action(action, self.actionable_cells)
        
        # Write action file for TCL to execute
        action_file = self.episode_dir / f"action_step{self.current_step}.txt"
        with open(action_file, 'w') as f:
            for inst, (old, new) in resizes.items():
                f.write(f"{inst} {new}\n")
        
        # Execute TCL to apply resize
        # (This would call OpenROAD to apply the changes)
        self.action_history.append((action, resizes))
    
    def _generate_timing_report(self):
        """Generate timing report using OpenROAD."""
        # Call OpenROAD to generate report
        pass


# ============================================================================
# DQN Trainer
# ============================================================================

class DQNTrainer:
    """
    Main DQN training loop.
    Manages Q-network, target network, optimizer, and training process.
    """
    
    def __init__(
        self,
        state_dim: int = 45,
        action_dim: int = 30,
        lr: float = 0.0001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 10,
        batch_size: int = 64,
        buffer_capacity: int = 10000,
        device: str = 'cpu'
    ):
        """
        Initialize DQN trainer.
        
        Args:
            state_dim: State vector dimension
            action_dim: Number of actions
            lr: Learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Exploration decay rate
            target_update_freq: How often to update target network
            batch_size: Mini-batch size for training
            buffer_capacity: Replay buffer capacity
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)
        
        # Networks
        self.q_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        
        # Training stats
        self.episode_count = 0
        self.total_steps = 0
        self.training_losses = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using ε-greedy policy.
        
        Args:
            state: Current state vector
            training: Whether in training mode (affects exploration)
            
        Returns:
            Action index
        """
        if training and random.random() < self.epsilon:
            # Random exploration
            return random.randint(0, self.q_network.network[-1].out_features - 1)
        else:
            # Greedy exploitation
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
    
    def train_step(self) -> float:
        """
        Perform one training step (mini-batch update).
        
        Returns:
            Training loss
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample mini-batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q-values: Q(s, a)
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values: r + γ * max Q(s', a')
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Compute loss
        loss = F.mse_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Copy Q-network weights to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save_checkpoint(self, path: str, episode: int):
        """Save training checkpoint."""
        checkpoint = {
            'episode': episode,
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
            'training_losses': self.training_losses
        }
        torch.save(checkpoint, path)
        print(f"[CHECKPOINT] Saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.total_steps = checkpoint['total_steps']
        self.training_losses = checkpoint.get('training_losses', [])
        self.episode_count = checkpoint['episode']
        print(f"[CHECKPOINT] Loaded from {path}, episode {self.episode_count}")


# ============================================================================
# Training Loop
# ============================================================================

def train_dqn(
    designs: List[str],
    trainer: DQNTrainer,
    num_episodes: int,
    log_dir: Path,
    checkpoint_freq: int = 50,
    eval_freq: int = 10
):
    """
    Main DQN training loop.
    
    Args:
        designs: List of design paths
        trainer: DQN trainer instance
        num_episodes: Number of episodes to train
        log_dir: Directory for logs and checkpoints
        checkpoint_freq: How often to save checkpoints
        eval_freq: How often to evaluate
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Training log
    log_file = log_dir / 'training.log'
    metrics_file = log_dir / 'metrics.jsonl'
    
    print("="*70)
    print("DQN Training Started")
    print("="*70)
    print(f"Designs: {len(designs)}")
    print(f"Episodes: {num_episodes}")
    print(f"Log dir: {log_dir}")
    print()
    
    for episode in range(num_episodes):
        # Select random design
        design_path = random.choice(designs)
        design_name = Path(design_path).stem
        
        print(f"\n{'='*70}")
        print(f"Episode {episode+1}/{num_episodes} - Design: {design_name}")
        print(f"Epsilon: {trainer.epsilon:.4f}")
        print(f"{'='*70}")
        
        # For this simplified version, we'll simulate episodes
        # In practice, you'd use DesignEpisode class
        
        # Simulate episode (placeholder - needs real implementation)
        episode_reward = simulate_episode(trainer, design_path)
        
        # Decay exploration
        trainer.decay_epsilon()
        
        # Update target network
        if (episode + 1) % trainer.target_update_freq == 0:
            trainer.update_target_network()
            print(f"[TARGET] Updated target network")
        
        # Save checkpoint
        if (episode + 1) % checkpoint_freq == 0:
            checkpoint_path = log_dir / f'checkpoint_ep{episode+1}.pth'
            trainer.save_checkpoint(str(checkpoint_path), episode+1)
        
        # Log metrics
        metrics = {
            'episode': episode + 1,
            'design': design_name,
            'reward': episode_reward,
            'epsilon': trainer.epsilon,
            'buffer_size': len(trainer.replay_buffer)
        }
        
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
        
        print(f"\nEpisode {episode+1} complete:")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Buffer size: {len(trainer.replay_buffer)}")
    
    # Save final model
    final_model_path = log_dir / 'model_final.pth'
    torch.save(trainer.q_network.state_dict(), final_model_path)
    print(f"\n[FINAL] Saved final model to {final_model_path}")
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)


def simulate_episode(trainer: DQNTrainer, design_path: str) -> float:
    """
    Simulate one training episode using CellSizingEnv.
    
    Args:
        trainer: DQN trainer
        design_path: Path to design
        
    Returns:
        Total episode reward
    """
    # Import environment (can move to top if preferred)
    from rl_environment import CellSizingEnv
    
    # Initialize environment for this design
    # Note: You may need to adjust design_dir and config_file based on your setup
    design_dir = str(Path(design_path).parent)
    config_file = str(Path(design_path).parent / 'config.json')
    
    try:
        env = CellSizingEnv(
            design_dir=design_dir,
            config_file=config_file,
            max_steps=50,
            top_k_cells=10
        )
        
        # Reset environment
        state = env.reset()
        done = False
        episode_reward = 0
        step = 0
        
        print(f"  Initial WNS: {env.initial_wns:.3f}")
        
        # Episode loop
        while not done and step < 50:
            # Select action using DQN
            action = trainer.select_action(state, training=True)
            
            # Execute action in environment
            next_state, reward, done, info = env.step(action)
            
            # Store transition in replay buffer
            trainer.replay_buffer.push(state, action, reward, next_state, done)
            
            # Train Q-network
            if len(trainer.replay_buffer) >= trainer.batch_size:
                loss = trainer.train_step()
                trainer.training_losses.append(loss)
            
            # Update for next iteration
            state = next_state
            episode_reward += reward
            step += 1
            
            # Log progress
            if step % 10 == 0:
                print(f"  Step {step}: WNS={info['wns']:.3f}, Reward={reward:.2f}")
        
        # Final info
        print(f"  Final WNS: {info.get('wns', 'N/A')}")
        print(f"  Violations: {info.get('num_violations', 'N/A')}")
        print(f"  Timing met: {'✓' if info.get('wns', -999) >= 0 else '✗'}")
        
        env.close()
        
    except Exception as e:
        print(f"  Error in episode: {e}")
        # Return small negative reward on failure
        episode_reward = -10.0
    
    return episode_reward


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main training script."""
    
    parser = argparse.ArgumentParser(
        description='Train DQN for Cell Sizing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--designs',
        required=True,
        help='Text file with list of design paths (one per line)'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=1000,
        help='Number of training episodes'
    )
    parser.add_argument(
        '--log-dir',
        default='logs/dqn_training',
        help='Directory for logs and checkpoints'
    )
    parser.add_argument(
        '--output',
        default='model_trained.pth',
        help='Path to save final trained model'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.0001,
        help='Learning rate'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='Discount factor'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Mini-batch size'
    )
    parser.add_argument(
        '--buffer-size',
        type=int,
        default=10000,
        help='Replay buffer capacity'
    )
    parser.add_argument(
        '--checkpoint-freq',
        type=int,
        default=50,
        help='Save checkpoint every N episodes'
    )
    parser.add_argument(
        '--resume',
        default=None,
        help='Resume from checkpoint'
    )
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cpu',
        help='Device to use for training'
    )
    
    args = parser.parse_args()
    
    # Load design list
    with open(args.designs, 'r') as f:
        designs = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(designs)} designs for training")
    
    # Initialize trainer
    trainer = DQNTrainer(
        state_dim=45,
        action_dim=30,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_size,
        device=args.device
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    train_dqn(
        designs=designs,
        trainer=trainer,
        num_episodes=args.episodes,
        log_dir=Path(args.log_dir),
        checkpoint_freq=args.checkpoint_freq
    )
    
    # Save final model
    torch.save(trainer.q_network.state_dict(), args.output)
    print(f"\n[SUCCESS] Final model saved to: {args.output}")


if __name__ == '__main__':
    main()
