# DQN Training Guide for Cell Sizing

## Complete Training Flow Explained

This guide walks you through training a DQN model for cell sizing optimization.

---

## Overview: What is DQN Training?

**DQN (Deep Q-Network) learns by trial and error:**

```
For each design:
  Start with timing violations
  Try different cell resizing actions
  Observe if timing improves or degrades
  Learn which actions work best in which situations
  Update neural network to predict better actions next time

After 1000s of episodes:
  Network learns general patterns across designs
  Can be deployed for inference on new designs
```

---

## Training Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Loop                            │
│                    (1000 episodes)                          │
└────────────┬────────────────────────────────────────────────┘
             │
             ├─ Episode 1: Design A
             │   ├─ Initialize design (timing violations)
             │   ├─ For step 1 to 50:
             │   │   ├─1. Get state: [WNS, TNS, cell features]
             │   │   ├─2. Select action (ε-greedy)
             │   │   │    └─ ε=0.9: Random (explore)
             │   │   │       ε=0.1: DQN prediction (exploit)
             │   │   ├─3. Apply action (resize cell)
             │   │   ├─4. Get reward (timing improvement)
             │   │   ├─5. Store (s, a, r, s') in buffer
             │   │   └─6. Train DQN on mini-batch
             │   │        └─ Update Q-network weights
             │   └─ End episode when timing met or max steps
             │
             ├─ Episode 2: Design B
             │   └─ (repeat process)
             │
             ├─ Episode 3: Design C
             │   └─ (repeat process)
             │
             └─ Episode 1000: Design X
                 └─ ε=0.05: Mostly exploit learned policy

┌─────────────────────────────────────────────────────────────┐
│              Final Trained Model                            │
│              model_trained.pth                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Training Components Explained

### 1. Q-Network (Neural Network)

```python
Input: State (45-dim vector)
  ├─ Global: [WNS, TNS, violations, avg_slack, max_delay]
  └─ Per-cell: [delay, fanout, drive, slack] × 10 cells

Hidden Layers:
  ├─ Linear(45 → 128) + ReLU + Dropout
  ├─ Linear(128 → 128) + ReLU + Dropout
  └─ Linear(128 → 64) + ReLU

Output: Q-values for each action (30-dim)
  └─ Q(s, a) = Expected future reward for action a in state s
```

**What it learns:**
- Which cells to resize in which situations
- How much to upsize based on current timing
- Tradeoffs between timing, power, and area

### 2. Replay Buffer

```python
Stores: (state, action, reward, next_state, done)

Example transitions:
┌────────────────────────────────────────────────────────────┐
│ State             Action   Reward  Next State        Done  │
├────────────────────────────────────────────────────────────┤
│ WNS=-5.3, TNS=-120 →  12  → +1.8  → WNS=-3.5, TNS=-98 │ False│
│ WNS=-3.5, TNS=-98  →  7   → +2.1  → WNS=-1.4, TNS=-45 │ False│
│ WNS=-1.4, TNS=-45  →  23  → +51.4 → WNS=0.2, TNS=0    │ True │
└────────────────────────────────────────────────────────────┘

Capacity: 10,000 transitions
Purpose: Break correlation between consecutive experiences
```

### 3. Reward Function

```python
reward = wns_improvement × 10.0     # Timing improvement
       + tns_improvement × 1.0      # Total slack improvement
       - area_increase × 0.01       # Area penalty
       + closure_bonus (50.0)        # Achieved timing!
```

**Examples:**

| Old WNS | New WNS | Old TNS | New TNS | Timing Met? | Reward |
|---------|---------|---------|---------|-------------|--------|
| -5.3    | -3.5    | -120    | -98     | ❌ No       | +18.0 + 22.0 = +40.0 |
| -3.5    | -1.4    | -98     | -45     | ❌ No       | +21.0 + 53.0 = +74.0 |
| -1.4    | +0.2    | -45     | 0       | ✅ Yes      | +16.0 + 45.0 + 50.0 = +111.0 |
| -2.5    | -3.1    | -67     | -72     | ❌ No       | -6.0 - 5.0 = -11.0 (bad!) |

**Reward shapes learning:**
- Positive rewards reinforce good actions
- Negative rewards discourage bad actions
- Large bonus for achieving timing closure

### 4. Training Algorithm (DQN)

```python
For each training step:
  
  1. Sample mini-batch (64 transitions)
     batch = replay_buffer.sample(64)
  
  2. Compute current Q-values
     Q_current = q_network(states)[actions]
     # What Q-network currently predicts
  
  3. Compute target Q-values
     Q_target = rewards + γ × max(target_network(next_states))
     # What Q-values should be (Bellman equation)
  
  4. Compute loss
     loss = MSE(Q_current, Q_target)
     # How wrong are our predictions?
  
  5. Backpropagate and update weights
     loss.backward()
     optimizer.step()
     # Update network to make better predictions
```

**Key Concepts:**

- **Target Network:** Stable copy of Q-network, updated every 10 episodes
  - Prevents training instability
  - Provides stable targets for learning

- **Epsilon-Greedy:** Balance exploration vs exploitation
  ```
  ε = 1.0 (start)  → 90% random, 10% learned
  ε = 0.5 (middle) → 50% random, 50% learned
  ε = 0.05 (end)   → 5% random, 95% learned
  ```

- **Gamma (γ):** Discount factor = 0.99
  - Values future rewards (long-term thinking)
  - γ=0: Only care about immediate reward
  - γ=1: Future rewards as important as immediate

---

## Step-by-Step Training Process

### Phase 1: Setup (5 minutes)

```bash
cd /home/isaishaq/openlane2/designs/picorv_test

# 1. Create list of training designs
ls /path/to/designs/*.v > designs_train.txt

# Example designs_train.txt:
# /path/to/designs/spm.v
# /path/to/designs/aes.v
# /path/to/designs/xtea.v
# ... (need 10-100 designs for good results)

# 2. Prepare designs (run up to post-CTS)
# Each design should be at the post-CTS stage
# where you want to optimize cell sizing
```

### Phase 2: Start Training

```bash
# Basic training (CPU, ~2-8 hours for 1000 episodes)
python3 scripts/train_dqn.py \
    --designs designs_train.txt \
    --episodes 1000 \
    --output model_trained.pth \
    --log-dir logs/dqn_training

# With custom hyperparameters
python3 scripts/train_dqn.py \
    --designs designs_train.txt \
    --episodes 2000 \
    --output model_trained.pth \
    --log-dir logs/dqn_training \
    --lr 0.0001 \
    --gamma 0.99 \
    --batch-size 64 \
    --buffer-size 10000 \
    --checkpoint-freq 50

# With GPU (faster if available)
python3 scripts/train_dqn.py \
    --designs designs_train.txt \
    --episodes 1000 \
    --output model_trained.pth \
    --device cuda
```

### Phase 3: Monitor Training

```bash
# Watch training progress
tail -f logs/dqn_training/training.log

# View metrics
cat logs/dqn_training/metrics.jsonl | jq '.'

# Plot learning curves (if you have matplotlib)
python3 scripts/plot_training.py \
    --metrics logs/dqn_training/metrics.jsonl \
    --output training_curves.png
```

**What to look for:**
- **Episode reward increasing:** Model is learning
- **Epsilon decreasing:** Less exploration over time
- **Buffer size growing:** Collecting experiences
- **Loss stabilizing:** Training converging

Example output:
```
======================================================================
Episode 1/1000 - Design: spm
Epsilon: 1.0000
======================================================================
  Step 1: WNS=-5.63, Reward=+18.5
  Step 2: WNS=-4.12, Reward=+15.1
  ...
  Step 15: WNS=0.15, Reward=+111.0 (CLOSURE!)

Episode 1 complete:
  Reward: 245.67
  Buffer size: 15

======================================================================
Episode 50/1000 - Design: aes
Epsilon: 0.6050
======================================================================
  ...
[TARGET] Updated target network

======================================================================
Episode 500/1000 - Design: xtea
Epsilon: 0.0932
======================================================================
  ...
[CHECKPOINT] Saved to logs/dqn_training/checkpoint_ep500.pth

======================================================================
Episode 1000/1000 - Design: picorv32a
Epsilon: 0.0500
======================================================================
  ...

[FINAL] Saved final model to logs/dqn_training/model_final.pth

======================================================================
Training Complete!
======================================================================

[SUCCESS] Final model saved to: model_trained.pth
```

### Phase 4: Resume Training (if interrupted)

```bash
# Resume from checkpoint
python3 scripts/train_dqn.py \
    --designs designs_train.txt \
    --episodes 2000 \
    --output model_trained.pth \
    --resume logs/dqn_training/checkpoint_ep500.pth
```

### Phase 5: Deploy Trained Model

```bash
# Copy trained model to run directory
cp model_trained.pth \
   runs/RUN_XXX/74-dqn-resizer-test/model/dqn_model.pth

# Run inference with trained model
openroad -exit scripts/dqn_resizer.tcl

# Will automatically use DQN agent (model exists)
```

---

## Hyperparameter Guide

### Learning Rate (`--lr`)

| Value | Effect | Use When |
|-------|--------|----------|
| 0.001 | Fast learning, may overshoot | Small number of simple designs |
| **0.0001** | **Balanced (default)** | **General use** |
| 0.00001 | Slow but stable | Large, complex designs |

### Discount Factor (`--gamma`)

| Value | Behavior | Use When |
|-------|----------|----------|
| 0.90 | Short-term focus | Quick timing fixes needed |
| **0.99** | **Long-term planning (default)** | **Optimal solutions** |
| 0.999 | Very long-term | Multi-step dependencies |

### Batch Size (`--batch-size`)

| Value | Training | Memory | Use When |
|-------|----------|--------|----------|
| 32 | More updates, noisier | Low | Small buffer |
| **64** | **Balanced (default)** | **Medium** | **General** |
| 128+ | Fewer updates, stable | High | Lots of data |

### Buffer Size (`--buffer-size`)

| Value | Coverage | Use When |
|-------|----------|----------|
| 5,000 | Recent experiences | Few designs |
| **10,000** | **Good diversity (default)** | **10-50 designs** |
| 50,000+ | Extensive history | 100+ designs |

### Number of Episodes (`--episodes`)

| Value | Training Time | Result Quality |
|-------|--------------|-----------------| 
| 100 | ~20 min | Poor (debugging only) |
| 500 | ~2 hrs | Fair (simple patterns) |
| **1000** | **~4-8 hrs** | **Good (recommended)** |
| 2000+ | ~8-16 hrs | Better (diminishing returns) |

---

## Common Issues & Solutions

### Issue 1: Training is slow

**Solution:**
```bash
# Use GPU
--device cuda

# Reduce episodes for testing
--episodes 100

# Increase batch size (if memory allows)
--batch-size 128
```

### Issue 2: Reward not increasing

**Possible causes:**
- Not enough exploration (ε too low)
- Learning rate too high/low
- Reward function not aligned with goal

**Solutions:**
```bash
# Increase initial exploration
# (modify epsilon_start in code)

# Try different learning rate
--lr 0.0005

# Check reward calculation
# (ensure timing improvement gives positive reward)
```

### Issue 3: Out of memory

**Solution:**
```bash
# Reduce buffer size
--buffer-size 5000

# Reduce batch size
--batch-size 32

# Use CPU instead of GPU
--device cpu
```

### Issue 4: Model overfits to training designs

**Solution:**
- Use more diverse training designs (50+)
- Add validation set for evaluation
- Early stopping when validation performance plateaus

---

## Expected Timeline

### Minimal Setup (10-50 designs)
- **Setup:** 30 minutes
- **Training:** 4-8 hours (1000 episodes)
- **Evaluation:** 1 hour
- **Total:** ~6-10 hours

### Full Setup (100+ designs)
- **Setup:** 2 hours
- **Training:** 12-24 hours (2000 episodes)
- **Evaluation:** 2 hours
- **Total:** ~16-28 hours

---

## Validation & Evaluation

After training, evaluate on test designs:

```bash
# Evaluate trained model
python3 scripts/evaluate_dqn.py \
    --model model_trained.pth \
    --designs designs_test.txt \
    --output evaluation_results.json

# Compare with heuristic baseline
python3 scripts/compare_agents.py \
    --dqn-model model_trained.pth \
    --designs designs_test.txt
```

**Metrics to track:**
- Success rate: % designs achieving timing closure
- Average WNS improvement
- Average number of steps to closure
- Area/power overhead

---

## Next Steps After Training

1. ✅ **Deploy model**
   ```bash
   cp model_trained.pth production/model.pth
   ```

2. ✅ **Use in inference**
   ```bash
   openroad -exit scripts/dqn_resizer.tcl
   ```

3. ✅ **Continue training** (online learning)
   - Collect new experiences from production runs
   - Fine-tune model on specific design families

4. ✅ **Hyperparameter tuning**
   - Grid search for optimal parameters
   - Use validation set for selection

---

## Summary

**Training Flow:**
```
Designs → Initialize → Explore → Collect Data → Train Network → Evaluate → Deploy
```

**Key Points:**
- Training learns from trial and error over 1000s of episodes
- Replay buffer stores experiences for stable learning
- Epsilon-greedy balances exploration and exploitation
- Reward function shapes what the agent learns
- Target network prevents training instability
- Final model can be deployed for fast inference

**Time Investment:**
- 6-10 hours for basic training (10-50 designs)
- 16-28 hours for comprehensive training (100+ designs)

**Expected Results:**
- Good model: 70-90% timing closure on test designs
- Expert model: 90-100% timing closure, near-optimal solutions

Ready to start training? Run:
```bash
python3 scripts/train_dqn.py \
    --designs designs_train.txt \
    --episodes 1000 \
    --output model_trained.pth
```
