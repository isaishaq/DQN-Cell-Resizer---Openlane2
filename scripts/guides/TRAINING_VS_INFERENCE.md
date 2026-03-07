# Training vs Inference in DQN Cell Resizing

## Current Architecture: Inference Only

### What Happens Each Iteration

```
Iteration N:
┌─────────────────────────────────────────┐
│  exec python3 dqn_agent.py             │
│                                         │
│  1. Start new Python process           │
│  2. Load model.pth from disk           │
│     ├─ Q-network weights loaded        │
│     ├─ Weights are FIXED (trained)     │
│     └─ agent.q_network.eval()          │
│                                         │
│  3. Forward pass (inference)           │
│     q_values = network(state)          │
│     action = argmax(q_values)          │
│                                         │
│  4. Write actions.txt                  │
│  5. Exit (discard Python process)      │
└─────────────────────────────────────────┘
         ↓
Iteration N+1:
┌─────────────────────────────────────────┐
│  exec python3 dqn_agent.py             │
│                                         │
│  1. Start new Python process (again)   │
│  2. Load SAME model.pth from disk      │
│     ├─ SAME weights as before          │
│     └─ No updates were saved           │
│                                         │
│  3. Forward pass (inference)           │
│  4. Write actions.txt                  │
│  5. Exit                                │
└─────────────────────────────────────────┘

Q-network weights: UNCHANGED
No backpropagation, no gradient updates
```

### Why Q-Values Don't Reset

The model weights are stored on disk:

```python
# dqn_agent.py, line ~235
def load_model(self, path: str):
    checkpoint = torch.load(path, map_location='cpu')
    self.q_network.load_state_dict(checkpoint)
    # ↑ Loads saved weights from disk
    # Q-values come from these weights
```

Each iteration loads the **same** weights, so Q-values remain consistent.

---

## Training: How It Actually Works

Training happens **offline**, **before** running the resizing flow.

### Offline Training Process

```
Training Phase (separate script):
═════════════════════════════════

Episode 1: Design A
├─ Initialize random Q-network
├─ For iteration 1 to 50:
│  ├─ Get state from timing report
│  ├─ Select action (Q-network)
│  ├─ Apply action (resize cells)
│  ├─ Get reward (timing improvement)
│  ├─ Store transition: (s, a, r, s')
│  └─ Update Q-network weights ← TRAINING
│     └─ Backprop: loss = (Q_pred - Q_target)²
└─ Save model: model_ep1.pth

Episode 2: Design B
├─ Load model_ep1.pth
├─ For iteration 1 to 50:
│  ├─ Get state, action, reward
│  ├─ Update Q-network weights ← TRAINING
│  └─ ...
└─ Save model: model_ep2.pth

Episode 3: Design C
├─ Load model_ep2.pth
├─ ...
└─ Save model: model_ep3.pth

... (100s-1000s of episodes)

Final model: model_trained.pth
═════════════════════════════════

Inference Phase (your current flow):
═════════════════════════════════

Load model_trained.pth (frozen)
For iteration 1 to 50:
├─ Get state
├─ Select action (no training)
└─ Apply action

Model weights: NEVER updated
```

### Training Algorithm (DQN)

```python
# Training loop (offline, not in your current code)
def train_episode(design, model, optimizer):
    replay_buffer = []
    
    for iteration in range(50):
        # 1. Get state from timing report
        state = get_state(design)
        
        # 2. Select action (ε-greedy)
        action = select_action(state, epsilon=0.1)
        
        # 3. Apply action
        apply_resize(design, action)
        
        # 4. Get reward
        new_wns = get_wns(design)
        reward = calculate_reward(old_wns, new_wns)
        
        # 5. Get next state
        next_state = get_state(design)
        
        # 6. Store transition
        replay_buffer.append((state, action, reward, next_state))
        
        # 7. TRAIN (update Q-network)
        if len(replay_buffer) >= batch_size:
            batch = sample(replay_buffer, batch_size)
            
            # Compute target Q-values
            with torch.no_grad():
                next_q = target_network(next_states).max(1)[0]
                target_q = rewards + gamma * next_q
            
            # Compute current Q-values
            current_q = q_network(states).gather(1, actions)
            
            # Compute loss
            loss = F.mse_loss(current_q, target_q)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # THIS is where Q-values update! ↑
    
    # Save updated model
    torch.save(q_network.state_dict(), f'model_ep{episode}.pth')
```

---

## Impact on Your Current Flow

### ✅ What Works

1. **Consistent decisions**: Same state → same action (deterministic with ε=0)
2. **Pre-trained knowledge**: Model uses patterns learned from training
3. **No training overhead**: Fast inference (~150ms per iteration)
4. **Reproducible**: Loading same model gives same results

### ❌ What Doesn't Happen

1. **No learning during sizing**: Model doesn't adapt to current design
2. **No experience replay**: No buffer of past transitions
3. **No weight updates**: Q-network frozen after loading
4. **No reward signal**: Reward is calculated but not used for training

---

## Two-Stage Workflow

### Stage 1: Train DQN (Offline)

```bash
# Train on multiple designs
python3 train_dqn.py \
    --designs designs/list.txt \
    --episodes 1000 \
    --output model_trained.pth
```

This creates a trained model that learns:
- How to identify critical cells
- Which sizing actions improve timing
- Tradeoffs between timing and power

### Stage 2: Use Trained Model (Inference)

```bash
# Use trained model on new design
openroad -exit dqn_resizer.tcl
# Loads model_trained.pth
# Applies learned policy
# No training, just decision-making
```

---

## If You Want Online Training

If you want the model to **learn during the resizing process**, you need:

### Option 1: Persistent Python Server

```python
# dqn_server.py (persistent process)
class OnlineDQNAgent:
    def __init__(self):
        self.q_network = DQNNetwork()
        self.optimizer = Adam(self.q_network.parameters())
        self.replay_buffer = []
    
    def run_server(self):
        while True:
            # Receive state from TCL
            state = receive_state()
            
            # Select action
            action = self.select_action(state)
            
            # Send action to TCL
            send_action(action)
            
            # Receive reward and next_state
            reward, next_state = receive_feedback()
            
            # Store transition
            self.replay_buffer.append((state, action, reward, next_state))
            
            # TRAIN on batch
            if len(self.replay_buffer) >= batch_size:
                self.train_step()
                # ↑ Q-values update here!
            
            # Save model periodically
            if iteration % 10 == 0:
                torch.save(self.q_network.state_dict(), 'model.pth')
```

### Option 2: Shared Model File with Update

```python
# After each iteration in dqn_agent.py
def main():
    # ... inference code ...
    
    # Optional: Update model if training enabled
    if args.enable_training:
        # Load previous state and reward
        prev_state = load_prev_state()
        reward = calculate_reward(prev_wns, current_wns)
        
        # Train step
        loss = train_step(prev_state, action, reward, current_state)
        
        # Save updated model
        torch.save(agent.q_network.state_dict(), args.model)
        
        # Save current state for next iteration
        save_state(current_state)
```

---

## Recommendation

For your use case, **offline training + inference** is the right approach:

### Why Offline Training?

1. **Data efficiency**: Train on 100s of designs offline
2. **Stability**: Converge to good policy before deployment
3. **Speed**: No training overhead during resizing
4. **Reproducibility**: Same model everywhere
5. **Testing**: Validate model quality before use

### Training Pipeline

```bash
# 1. Collect training data
for design in designs/*.v; do
    run_openlane $design --save-states
done

# 2. Train DQN
python3 train_dqn.py \
    --data training_data/ \
    --episodes 1000 \
    --batch-size 64 \
    --output model_trained.pth

# 3. Evaluate model
python3 evaluate_dqn.py \
    --model model_trained.pth \
    --test-designs test_designs/

# 4. Deploy for inference
cp model_trained.pth production/model.pth
```

---

## Summary

| Aspect | Current Flow | Training Flow |
|--------|-------------|---------------|
| **Purpose** | Apply learned policy | Learn policy |
| **Q-values** | Loaded from disk, frozen | Updated via backprop |
| **Python process** | New each iteration | Persistent or saved |
| **Model file** | Read-only | Read + Write |
| **Learning** | ❌ No | ✅ Yes |
| **Speed** | Fast (~150ms) | Slower (~1-5s) |
| **Use case** | Production inference | Offline training |

### Key Takeaway

**Your current architecture is correct for inference**. Each Python subprocess loads the same pre-trained model, so Q-values remain consistent. Training happens separately, offline, on multiple designs to learn a good policy. This is the standard approach for deploying RL models.

If you want to train the model, you need a separate training script that:
1. Runs episodes on multiple designs
2. Collects transitions (s, a, r, s')
3. Updates Q-network weights via backpropagation
4. Saves the trained model for inference

Would you like me to create a training script for offline DQN training?
