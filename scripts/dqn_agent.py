#!/usr/bin/env python3
"""
DQN-based Cell Resizer for OpenLane 2
"""
import odb
import numpy as np
import torch
import torch.nn as nn
import argparse
from collections import deque
from typing import List, Tuple, Dict
from openroad import Design, Tech

# ==================== DQN Network ====================
class DQNetwork(nn.Module):
    """
    Deep Q-Network for cell resizing decisions
    
    Input: State features (timing, power, cell properties)
    Output: Q-values for each action (resize options)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
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
        return self.network(x)


# ==================== Environment ====================
class CellResizingEnv:
    """
    RL Environment for cell resizing
    
    State: [wns, tns, power, cell_slack, cell_power, cell_size, ...]
    Action: Resize cell to different drive strength
    Reward: Improvement in PPA
    """
    
    def __init__(self, odb_path: str, target_slack: float = 0.0, 
                 power_weight: float = 0.3):
        self.target_slack = target_slack
        self.power_weight = power_weight
        
        # Load ODB database
        self.tech = Tech()
        self.design = Design(self.tech)
        self.design.readDb(odb_path)
        
        self.db = self.tech.getDB()
        self.block = self.db.getChip().getBlock()
        
        # Get STA engine
        self.sta = self.design.getOpenSta()
        
        # Track resizable cells
        self.resizable_cells = self._find_resizable_cells()
        
        # Initial metrics
        self.initial_wns = self._get_wns()
        self.initial_tns = self._get_tns()
        self.initial_power = self._get_power()
        
        print(f"[ENV] Initial WNS: {self.initial_wns:.4f} ns")
        print(f"[ENV] Initial TNS: {self.initial_tns:.4f} ns")
        print(f"[ENV] Initial Power: {self.initial_power:.6f} W")
        print(f"[ENV] Resizable cells: {len(self.resizable_cells)}")
    
    def _find_resizable_cells(self) -> List[Dict]:
        """
        Find all cells that can be resized
        Returns list of: {instance, current_master, alternatives}
        """
        resizable = []
        
        for inst in self.block.getInsts():
            master = inst.getMaster()
            master_name = master.getName()
            
            # Get cell library and find variants
            lib = master.getLib()
            
            # Pattern: sky130_fd_sc_hd__inv_2 -> find inv_1, inv_4, etc.
            alternatives = self._find_cell_variants(master_name, lib)
            
            if len(alternatives) > 1:
                resizable.append({
                    'instance': inst,
                    'current_master': master,
                    'alternatives': alternatives,
                    'current_idx': alternatives.index(master)
                })
        
        return resizable
    
    def _find_cell_variants(self, cell_name: str, lib) -> List:
        """
        Find all drive strength variants of a cell
        e.g., buf_1, buf_2, buf_4, buf_8, buf_16
        """
        import re
        
        # Extract base name (e.g., sky130_fd_sc_hd__inv)
        match = re.match(r'(.+)_(\d+)$', cell_name)
        if not match:
            return [lib.findMaster(cell_name)]
        
        base_name = match.group(1)
        variants = []
        
        # Common drive strengths
        for strength in [1, 2, 4, 6, 8, 12, 16]:
            variant_name = f"{base_name}_{strength}"
            master = lib.findMaster(variant_name)
            if master:
                variants.append(master)
        
        return variants if variants else [lib.findMaster(cell_name)]
    
    def _get_wns(self) -> float:
        """Get Worst Negative Slack"""
        # Use OpenSTA to get timing
        sta = self.sta
        return float(sta.getWorstSlack())
    
    def _get_tns(self) -> float:
        """Get Total Negative Slack"""
        sta = self.sta
        return float(sta.getTotalNegativeSlack())
    
    def _get_power(self) -> float:
        """Get total power consumption"""
        sta = self.sta
        power = sta.getPower()
        return float(power.total()) if power else 0.0
    
    def get_cell_features(self, cell_info: Dict) -> np.ndarray:
        """
        Extract features for a specific cell
        Returns: [slack, power, drive_strength_ratio, fanout, ...]
        """
        inst = cell_info['instance']
        
        # Get cell slack (worst pin slack)
        slack = self._get_cell_slack(inst)
        
        # Get cell power
        power = self._get_cell_power(inst)
        
        # Get drive strength ratio
        current_idx = cell_info['current_idx']
        max_idx = len(cell_info['alternatives']) - 1
        drive_ratio = current_idx / max(max_idx, 1)
        
        # Get fanout
        fanout = self._get_fanout(inst)
        
        # Get cell area
        area = self._get_cell_area(inst)
        
        return np.array([slack, power, drive_ratio, fanout, area])
    
    def get_state(self) -> np.ndarray:
        """
        Get current design state
        Returns: [wns, tns, power, normalized_wns, normalized_tns, ...]
        """
        wns = self._get_wns()
        tns = self._get_tns()
        power = self._get_power()
        
        # Normalize
        norm_wns = wns / abs(self.initial_wns) if self.initial_wns != 0 else 0
        norm_tns = tns / abs(self.initial_tns) if self.initial_tns != 0 else 0
        norm_power = power / self.initial_power if self.initial_power != 0 else 1
        
        # Count violations
        setup_vios = self._count_setup_violations()
        hold_vios = self._count_hold_violations()
        
        return np.array([
            wns, tns, power,
            norm_wns, norm_tns, norm_power,
            setup_vios, hold_vios
        ])
    
    def _get_cell_slack(self, inst) -> float:
        """Get worst slack through this cell"""
        sta = self.sta
        pins = inst.getITerms()
        worst_slack = float('inf')
        
        for pin in pins:
            slack = sta.getPinSlack(pin)
            if slack < worst_slack:
                worst_slack = slack
        
        return float(worst_slack)
    
    def _get_cell_power(self, inst) -> float:
        """Get power consumption of this cell"""
        # Simplified: use cell area as proxy
        master = inst.getMaster()
        return float(master.getArea())
    
    def _get_fanout(self, inst) -> int:
        """Get fanout count"""
        count = 0
        for term in inst.getITerms():
            if term.isOutputSignal():
                net = term.getNet()
                if net:
                    count += len(list(net.getITerms()))
        return count
    
    def _get_cell_area(self, inst) -> float:
        """Get cell area"""
        return float(inst.getMaster().getArea())
    
    def _count_setup_violations(self) -> int:
        """Count setup timing violations"""
        return sum(1 for _ in self.resizable_cells if self._get_cell_slack(_['instance']) < 0)
    
    def _count_hold_violations(self) -> int:
        """Count hold violations (simplified)"""
        return 0  # Would need min path analysis
    
    def step(self, cell_idx: int, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Take action: resize cell_idx to action (alternative index)
        Returns: (next_state, reward, done)
        """
        if cell_idx >= len(self.resizable_cells):
            return self.get_state(), -10, False
        
        cell_info = self.resizable_cells[cell_idx]
        alternatives = cell_info['alternatives']
        
        if action >= len(alternatives):
            return self.get_state(), -5, False
        
        # Perform resize
        inst = cell_info['instance']
        new_master = alternatives[action]
        old_master = inst.getMaster()
        
        if old_master == new_master:
            return self.get_state(), 0, False  # No change
        
        # Swap cell
        inst.swapMaster(new_master)
        cell_info['current_idx'] = action
        
        # Update timing (incremental STA)
        self.sta.updateTiming()
        
        # Calculate reward
        new_wns = self._get_wns()
        new_tns = self._get_tns()
        new_power = self._get_power()
        
        # Reward function: balance timing improvement and power
        timing_reward = (self.initial_wns - new_wns) + (self.initial_tns - new_tns) / 10
        power_penalty = (new_power - self.initial_power) / self.initial_power
        
        reward = timing_reward - self.power_weight * power_penalty
        
        # Check if done (converged)
        done = new_wns >= self.target_slack or reward < -1.0
        
        # Update baseline
        self.initial_wns = new_wns
        self.initial_tns = new_tns
        self.initial_power = new_power
        
        return self.get_state(), reward, done
    
    def save(self, path: str):
        """Save modified database"""
        self.design.writeDb(path)


# ==================== DQN Agent ====================
class DQNAgent:
    """
    DQN Agent for cell resizing
    """
    
    def __init__(self, state_dim, action_dim, learning_rate=1e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.q_network = DQNetwork(state_dim, action_dim)
        self.target_network = DQNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=10000)
        
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95  # Discount factor
    
    def select_action(self, state, training=False):
        """Epsilon-greedy action selection"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=32):
        """Train on batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        
        for idx in batch:
            state, action, reward, next_state, done = self.memory[idx]
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            
            current_q = self.q_network(state_tensor)[0, action]
            
            with torch.no_grad():
                next_q = self.target_network(next_state_tensor).max()
                target_q = reward + (self.gamma * next_q * (1 - done))
            
            loss = nn.MSELoss()(current_q, target_q)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        """Sync target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, path):
        """Save model"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path):
        """Load model"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']


# ==================== Main ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--odb', required=True, help='Input ODB file')
    parser.add_argument('--model', help='Pre-trained model path')
    parser.add_argument('--max-iterations', type=int, default=50)
    parser.add_argument('--target-slack', type=float, default=0.0)
    parser.add_argument('--power-weight', type=float, default=0.3)
    parser.add_argument('--training', type=int, default=0)
    args = parser.parse_args()
    
    # Initialize environment
    env = CellResizingEnv(
        args.odb,
        target_slack=args.target_slack,
        power_weight=args.power_weight
    )
    
    # Initialize agent
    state_dim = 8  # From get_state()
    action_dim = 10  # Max alternatives (adjust based on your PDK)
    agent = DQNAgent(state_dim, action_dim)
    
    # Load pre-trained model if available
    if args.model and os.path.exists(args.model):
        agent.load(args.model)
        print(f"[AGENT] Loaded model from {args.model}")
    
    training_mode = bool(args.training)
    
    # Run optimization loop
    state = env.get_state()
    total_reward = 0
    
    for iteration in range(args.max_iterations):
        # Select cell to resize (prioritize critical cells)
        cell_idx = np.random.randint(len(env.resizable_cells))
        
        # Get action from agent
        cell_features = env.get_cell_features(env.resizable_cells[cell_idx])
        combined_state = np.concatenate([state, cell_features])
        
        action = agent.select_action(combined_state, training=training_mode)
        
        # Take step
        next_state, reward, done = env.step(cell_idx, action)
        total_reward += reward
        
        # Store experience if training
        if training_mode:
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            
            if iteration % 10 == 0:
                agent.update_target_network()
        
        print(f"[ITER {iteration}] WNS: {next_state[0]:.4f} ns, "
              f"TNS: {next_state[1]:.4f} ns, "
              f"Power: {next_state[2]:.6f} W, "
              f"Reward: {reward:.4f}")
        
        state = next_state
        
        if done:
            print(f"[DONE] Converged at iteration {iteration}")
            break
    
    # Save results
    env.save(args.odb)
    
    if training_mode and args.model:
        agent.save(args.model)
        print(f"[AGENT] Saved model to {args.model}")
    
    print(f"[RESULT] Total reward: {total_reward:.4f}")
    print(f"[RESULT] Final WNS: {state[0]:.4f} ns")
    print(f"[RESULT] Final TNS: {state[1]:.4f} ns")
    print(f"[RESULT] Final Power: {state[2]:.6f} W")


if __name__ == '__main__':
    main()