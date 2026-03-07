# Starting Without a Pre-trained Model

You have **4 options** when you don't have a pre-trained DQN model:

---

## Option 1: Heuristic-Based Policy (Recommended First)

Use rule-based sizing without any training.

### Quick Start

```bash
# In dqn_resizer.tcl, change the agent:
exec python3 $::env(HEURISTIC_AGENT_SCRIPT) \
    --timing-report $report_file \
    --output-actions $actions_file \
    --strategy balanced \
    --iteration $iter
```

Or directly:
```bash
python3 scripts/heuristic_agent.py \
    --timing-report reports/timing.rpt \
    --output-actions actions/actions.txt \
    --strategy aggressive
```

### Strategies

- **`aggressive`**: Upsize more aggressively, faster timing improvement
- **`balanced`**: Moderate sizing, good tradeoff
- **`conservative`**: Minimal sizing, lower area cost

### How It Works

```python
# Heuristic rules:
1. Score cells by: delay + fanout + slack contribution
2. Pick highest-scoring cell (most critical)
3. Decide upsize amount based on:
   - High delay + high fanout → 4x drive
   - High delay OR high fanout → 2x drive
   - Otherwise → 1x drive (next size up)
4. Apply resize
```

**Pros:**
- ✅ Works immediately, no training needed
- ✅ Fast and deterministic
- ✅ Often gets 60-80% optimal results
- ✅ Good baseline to compare DQN against

**Cons:**
- ❌ Not adaptive to design characteristics
- ❌ May not find optimal solution
- ❌ Fixed rules don't learn from experience

---

## Option 2: Random Policy (For Testing Only)

Your current DQN agent already does this when model file doesn't exist.

```bash
python3 scripts/dqn_agent.py \
    --timing-report reports/timing.rpt \
    --output-actions actions/actions.txt \
    --model /nonexistent/model.pth \
    --epsilon 1.0  # Pure random
```

**Use this only to test your pipeline, not for actual resizing.**

---

## Option 3: Imitation Learning from OpenROAD

Bootstrap your DQN by learning from OpenROAD's `repair_timing`.

### Step 1: Collect Expert Demonstrations

```python
# collect_expert_data.py
"""
Run OpenROAD repair_timing and record its actions.
Use this as training data for DQN.
"""

import subprocess
import re

def collect_repair_timing_trajectory(design):
    """
    Run repair_timing and extract which cells it resized.
    
    Returns:
        List of (state, action) pairs
    """
    trajectories = []
    
    # Run repair_timing with logging
    result = subprocess.run([
        'openroad', '-exit', 'repair_timing_logged.tcl'
    ], capture_output=True, text=True)
    
    # Parse log to find resizes
    for line in result.stdout.split('\n'):
        if 'Resized' in line:
            # Extract: Resized _12345_ from buf_1 to buf_4
            match = re.search(r'Resized (\S+) from (\S+) to (\S+)', line)
            if match:
                inst, old_cell, new_cell = match.groups()
                
                # Get state before resize
                state = get_state_from_timing_report()
                
                # Encode action
                action = encode_resize_action(inst, old_cell, new_cell)
                
                trajectories.append((state, action))
    
    return trajectories

# Run on multiple designs
expert_data = []
for design in design_list:
    traj = collect_repair_timing_trajectory(design)
    expert_data.extend(traj)

# Save expert data
save_expert_data(expert_data, 'expert_demos.pkl')
```

### Step 2: Train DQN with Behavioral Cloning

```python
# train_from_expert.py
"""
Initialize DQN by imitating repair_timing behavior.
"""

import torch
import torch.nn as nn
import torch.optim as optim

# Load expert demonstrations
expert_data = load_expert_data('expert_demos.pkl')

# Initialize DQN
dqn = DQNNetwork(state_dim=45, action_dim=30)
optimizer = optim.Adam(dqn.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Behavioral cloning: train to predict expert actions
for epoch in range(100):
    for state, expert_action in expert_data:
        # Predict action probabilities
        q_values = dqn(torch.FloatTensor(state))
        
        # Loss: cross-entropy with expert action
        loss = criterion(q_values.unsqueeze(0), 
                        torch.LongTensor([expert_action]))
        
        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Save initialized model
torch.save(dqn.state_dict(), 'model_from_expert.pth')
```

Now you have a model that mimics `repair_timing`!

---

## Option 4: Train from Scratch (Full RL)

Train DQN with reinforcement learning from random initialization.

### Step 1: Create Training Script

```python
# train_dqn.py
"""
Train DQN from scratch using reinforcement learning.
Requires: Multiple designs, compute time (~hours)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from dqn_agent import DQNNetwork

class DQNTrainer:
    def __init__(self):
        self.q_network = DQNNetwork(state_dim=45, action_dim=30)
        self.target_network = DQNNetwork(state_dim=45, action_dim=30)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0001)
        self.replay_buffer = deque(maxlen=10000)
        
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
    
    def train_episode(self, design_path):
        """
        Train on one design episode.
        """
        # Load design in OpenROAD
        design = load_design(design_path)
        
        episode_reward = 0
        state = get_initial_state(design)
        
        for iteration in range(50):
            # Select action (ε-greedy)
            if random.random() < self.epsilon:
                action = random.randint(0, 29)
            else:
                with torch.no_grad():
                    q_values = self.q_network(torch.FloatTensor(state))
                    action = q_values.argmax().item()
            
            # Apply action
            apply_resize(design, action)
            
            # Get reward
            new_wns = get_wns(design)
            old_wns = state[0]  # WNS from state
            reward = calculate_reward(old_wns, new_wns)
            
            # Get next state
            next_state = get_state(design)
            done = (new_wns >= 0) or (iteration >= 49)
            
            # Store transition
            self.replay_buffer.append((state, action, reward, next_state, done))
            
            # Train on batch
            if len(self.replay_buffer) >= 64:
                self.train_step()
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return episode_reward
    
    def train_step(self):
        """
        Update Q-network using experience replay.
        """
        # Sample batch
        batch = random.sample(self.replay_buffer, 64)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Compute Q(s, a)
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute target: r + γ * max Q(s', a')
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Loss
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_network(self):
        """Update target network (every N episodes)."""
        self.target_network.load_state_dict(self.q_network.state_dict())


def calculate_reward(old_wns, new_wns, old_power=None, new_power=None):
    """
    Reward function for DQN training.
    
    Positive reward for timing improvement, penalty for power increase.
    """
    # Timing improvement
    timing_improvement = new_wns - old_wns
    
    # Power penalty (if available)
    power_penalty = 0
    if old_power and new_power:
        power_increase = new_power - old_power
        power_penalty = 0.1 * power_increase
    
    # Combined reward
    reward = 10.0 * timing_improvement - power_penalty
    
    # Bonus for meeting timing
    if new_wns >= 0 and old_wns < 0:
        reward += 50.0
    
    return reward


# Main training loop
def main():
    trainer = DQNTrainer()
    design_list = load_design_list('designs.txt')
    
    for episode in range(1000):
        # Pick random design
        design = random.choice(design_list)
        
        # Train episode
        reward = trainer.train_episode(design)
        
        print(f"Episode {episode}, Reward: {reward:.2f}, ε: {trainer.epsilon:.3f}")
        
        # Update target network every 10 episodes
        if episode % 10 == 0:
            trainer.update_target_network()
        
        # Save checkpoint every 50 episodes
        if episode % 50 == 0:
            torch.save(trainer.q_network.state_dict(), 
                      f'checkpoints/model_ep{episode}.pth')
    
    # Save final model
    torch.save(trainer.q_network.state_dict(), 'model_trained.pth')
    print("Training complete!")


if __name__ == '__main__':
    main()
```

### Step 2: Run Training

```bash
# Collect designs for training
ls designs/*.v > designs_train.txt

# Start training (may take hours)
python3 train_dqn.py

# Monitor progress
tensorboard --logdir=logs/
```

### Step 3: Evaluate

```bash
# Test on validation designs
python3 evaluate_dqn.py \
    --model checkpoints/model_ep500.pth \
    --designs designs_val.txt
```

**Pros:**
- ✅ Fully learned policy, adaptive
- ✅ Can surpass hand-crafted heuristics
- ✅ Learns design-specific patterns

**Cons:**
- ❌ Requires many designs (100+)
- ❌ Computationally expensive (hours-days)
- ❌ Hyperparameter tuning needed
- ❌ May not converge without careful setup

---

## Recommendation: Start with Heuristics

```
Step 1: Use heuristic_agent.py
   ├─ Get baseline results
   ├─ Validate your pipeline
   └─ Compare against repair_timing

Step 2: Collect data while using heuristics
   ├─ Log states, actions, rewards
   └─ Build dataset

Step 3: Train DQN offline
   ├─ Use collected data
   ├─ Or imitate repair_timing
   └─ Get model.pth

Step 4: Deploy trained model
   └─ Use dqn_agent.py with model.pth
```

---

## Quick Comparison

| Method | Setup Time | Performance | Training Needed |
|--------|------------|-------------|-----------------|
| **Heuristic** | 0 min | 60-80% optimal | ❌ No |
| **Random** | 0 min | <10% optimal | ❌ No |
| **Imitation** | ~2 hours | 70-90% optimal | ✅ Yes (fast) |
| **DQN from scratch** | ~8-48 hours | Up to 100% optimal | ✅ Yes (slow) |

---

## Files to Create

1. ✅ `heuristic_agent.py` - Already created
2. ⚠️ `train_dqn.py` - Full training script (above is template)
3. ⚠️ `collect_expert_data.py` - Imitation learning
4. ⚠️ `evaluate_agent.py` - Compare different methods

Want me to create any of these complete scripts?
