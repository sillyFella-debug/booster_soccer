# Comprehensive Training Plan for DDPG Soccer Showdown

## 1. Analyze Training Requirements for All Tasks

### Current Status
- Your baseline model was trained **only** for the penalty goalie kick task
- Score: -2.48 (falls over within 2 steps, minimal progress)
- Issue: Model is **not generalized** to other tasks

### Recommendation: YES, Train for All Tasks
The competition likely includes 3 main tasks:
1. **Penalty Goalie Kick** (defensive)
2. **Regular Kick to Target** (offensive)
3. **Other game scenarios** (mixed)

### Implementation Strategy
- **Option A (Recommended)**: Retrain a **single model** that learns to handle all 3 tasks
  - The SAI environment supports multi-task learning via `task_index` in info dict
  - Your preprocessor already extracts `task_onehot` from info
  - Train with data from all 3 tasks mixed in the replay buffer
  - This ensures generalization across tasks

- **Option B**: Train separate task-specific models
  - Less efficient but may give better per-task performance
  - Requires uploading multiple models during submission

### Modifications Needed
- Collect training data for ALL 3 tasks (not just penalty goalie kick)
- Modify `training_loop()` to accept multi-task data
- Ensure `task_onehot` is properly included in state observations

---

## 2. Guide on Starting DDPG Training in Kaggle

### Step-by-Step Instructions for Kaggle Notebook

#### 2.1 Create and Setup Kaggle Notebook
1. Go to [Kaggle.com](https://kaggle.com) and create a new notebook
2. Set notebook settings:
   - **Accelerator**: GPU (P100 or better, required for 3+ hours training)
   - **Persistent Working Directory**: Enable
   - **Internet**: Enable (for wandb logging)

#### 2.2 Install Dependencies
```python
!pip install -q sai-rl wandb torch gymnasium numpy tqdm
!wandb login  # Paste your W&B API key when prompted
```

#### 2.3 Upload Your Files
- Create a Kaggle dataset with your training files
- OR directly load from GitHub:
  ```python
  !git clone https://github.com/ArenaX-Labs/booster_soccer_showdown.git
  %cd booster_soccer_showdown
  ```

#### 2.4 Run Training Script
```python
import sys
sys.path.append('./training_scripts')
from main import *

# Run training - will continue for the specified timesteps
training_loop(env, model, action_function, Preprocessor, timesteps=1000000)
```

#### 2.5 Key Kaggle Considerations
- Kaggle notebooks timeout after **9 hours**, but your 3+ hour requirement fits
- Use persistent working directory to save checkpoints
- W&B will sync training metrics automatically
- Save final model: `torch.save(model.state_dict(), 'final_model.pt')`

---

## 3. Add Validation Loss Reporting to Console and W&B

### Current Implementation Analysis
- Your `training_loop()` in `training.py` currently logs critic/actor loss to console via tqdm
- These are **training** losses only, not validation losses
- Need to add **separate validation loop** that:
  1. Runs on a subset of held-out data
  2. Computes validation losses without updating model weights
  3. Logs to both console and W&B

### Implementation Plan

#### 3.1 Modify `training.py`
Add a validation function:
```python
def validation_pass(model, replay_buffer, batch_size=64, num_batches=10):
    """
    Compute validation loss on a subset of replay buffer without updating weights.
    Returns average validation losses.
    """
    val_critic_losses = []
    val_actor_losses = []
    
    for _ in range(num_batches):
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        # Forward pass ONLY, no gradient updates
        with torch.no_grad():
            critic_loss, actor_loss = model.model_update(
                states, actions, rewards.reshape(-1, 1), 
                next_states, dones.reshape(-1, 1)
            )
        
        val_critic_losses.append(critic_loss)
        val_actor_losses.append(actor_loss)
    
    avg_val_critic = np.mean(val_critic_losses)
    avg_val_actor = np.mean(val_actor_losses)
    
    return avg_val_critic, avg_val_actor
```

#### 3.2 Integrate into Training Loop
Modify `training_loop()` to call validation and log to W&B:
```python
# After line with pbar.set_description, add:
if total_steps % (update_frequency * 500) == 0:  # Validate every ~2000 steps
    val_critic_loss, val_actor_loss = validation_pass(model, replay_buffer)
    
    # Log to console
    print(f"[Step {total_steps}] Val Critic: {val_critic_loss:.4f} | Val Actor: {val_actor_loss:.4f}")
    
    # Log to W&B if available
    try:
        import wandb
        wandb.log({
            'validation/critic_loss': val_critic_loss,
            'validation/actor_loss': val_actor_loss,
            'training/episode_reward': episode_reward,
        }, step=total_steps)
    except:
        pass  # W&B not available, continue anyway
```

---

## 4. Configure for Long Training Runs (Until Double Descent) with Updated Checkpoints

### 4.1 Understanding Double Descent
- **Double Descent Phenomenon**: Validation loss decreases → increases → decreases again with more training
- This indicates your model is escaping overfitting and learning better generalizations
- To "run until double descent hits", you need to:
  1. Train for extended duration (12+ hours if possible, you want minimum 3 hours)
  2. Monitor validation loss continuously
  3. Continue past initial overfitting until performance improves again

### 4.2 Checkpoint Strategy Modifications

#### Current: Checkpoints every 10K iterations
#### New: Checkpoints every 5K iterations + best model tracking

**Modify `main.py` to include checkpoint saving:**

```python
import os
from datetime import datetime

# Add after model creation
save_dir = f"checkpoints_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(save_dir, exist_ok=True)
best_val_loss = float('inf')

# Wrap training loop
training_loop(
    env, model, action_function, Preprocessor, 
    timesteps=3600000,  # ~1M steps per hour, 3+ hours = 3.6M+ steps
    checkpoint_dir=save_dir,
    checkpoint_interval=5000,  # Save every 5K steps instead of 10K
)
```

#### 4.3 Modified `training.py` with Checkpoint Management

Add checkpoint saving function:
```python
def save_checkpoint(model, iteration, save_dir, is_best=False):
    """Save model checkpoint at specified iteration."""
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint_name = f"model_checkpoint_{iteration}.pt"
    if is_best:
        checkpoint_name = "model_best_validation.pt"
    
    save_path = os.path.join(save_dir, checkpoint_name)
    torch.save({
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
    }, save_path)
    
    print(f"✓ Saved checkpoint: {save_path}")
```

Modify `training_loop()` signature:
```python
def training_loop(
    env: gym.Env,
    model,
    action_function: Optional[Callable] = None,
    preprocess_class: Optional[Callable] = None,
    timesteps=1000,
    checkpoint_dir="checkpoints",
    checkpoint_interval=5000,  # NEW: Every 5K instead of 10K
):
    # ... existing code ...
    
    best_val_loss = float('inf')
    
    # Inside training loop, after validation:
    if total_steps % checkpoint_interval == 0:
        # Check if this is best model
        if val_actor_loss < best_val_loss:
            best_val_loss = val_actor_loss
            save_checkpoint(model, total_steps, checkpoint_dir, is_best=True)
        else:
            save_checkpoint(model, total_steps, checkpoint_dir, is_best=False)
```

### 4.4 Kaggle-Specific Long Training Configuration

Create a Kaggle-optimized training cell:
```python
# Set up for long training on Kaggle GPU
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Avoid GPU memory fragmentation

# Training hyperparameters for long runs
training_config = {
    'timesteps': 5000000,        # ~5 hours of continuous training
    'checkpoint_interval': 5000,  # Save every 5K steps
    'validation_interval': 10000, # Validate every 10K steps
    'batch_size': 64,
    'update_frequency': 4,
}

# Start training with persistent checkpoints
training_loop(
    env, model, action_function, Preprocessor,
    timesteps=training_config['timesteps'],
    checkpoint_dir='/kaggle/working/checkpoints',
    checkpoint_interval=training_config['checkpoint_interval'],
)
```

**Kaggle Runtime Tip**: If you hit the 9-hour limit:
1. Your checkpoints will be preserved in persistent working directory
2. Create a new notebook cell that resumes from the latest checkpoint
3. Load and continue: `model.load_state_dict(torch.load('checkpoint_path'))`

---

## 5. Hyperparameter Tuning Suggestions for DDPG

### Current Hyperparameters (from `main.py`)
```python
neurons=[24, 12, 6],           # Network architecture
learning_rate=0.0001,          # Actor/Critic learning rate
gamma=0.99,                    # Discount factor (in DDPG_FF class)
tau=0.001,                     # Soft update rate (target network)
noise_scale=0.1,               # Action exploration noise
batch_size=64,                 # Replay buffer batch size
```

### Hyperparameter Tuning Strategy

#### 5.1 Learning Rate (HIGHEST PRIORITY)
**Current**: 0.0001 (quite conservative)

**Suggested Range for DDPG**:
- Start with: **0.0003** (slightly higher for faster convergence)
- Range: 0.0001 - 0.001
- If training is too unstable → decrease to 0.00005
- If learning is too slow → increase to 0.0005

**Implementation**:
```python
model = DDPG_FF(
    n_features=87,
    action_space=env.action_space,
    neurons=[24, 12, 6],
    activation_function=F.relu,
    learning_rate=0.0003,  # CHANGED from 0.0001
)
```

#### 5.2 Discount Factor (Gamma)
**Current**: 0.99 (Good default)

**Suggested Adjustment**:
- **0.95-0.99**: Better for shorter-horizon tasks (your tasks are relatively short)
- **0.99**: Current choice is good, keep it
- Only change if reward signals are not propagating properly

#### 5.3 Soft Update Rate (Tau)
**Current**: 0.001 (Conservative)

**Suggested Adjustments**:
- **0.001**: Conservative, stable learning (good for soccer)
- **0.005**: More aggressive target network updates
- **0.01**: Faster adaptation but potentially unstable

**For your soccer task**: Keep at **0.001** (prevents sudden policy shifts)

#### 5.4 Network Architecture (Neurons)
**Current**: [24, 12, 6]

**Suggested Improvements**:
```python
# Option 1: Slightly larger network for complex tasks
neurons=[64, 32, 16]  # More capacity to learn task complexity

# Option 2: Deeper network
neurons=[32, 32, 32]  # Better for learning non-linear dynamics

# Option 3: Keep current but add regularization
# Add L2 regularization to prevent overfitting
```

**Recommendation**: Start with **[64, 32, 16]** for better capacity

#### 5.5 Action Noise (Noise Scale)
**Current**: 0.1

**Suggested Adjustments**:
- **Start at**: 0.1 (current)
- **Decay over time**: Reduce noise as training progresses (exploration → exploitation)
- **Decay strategy**: 
  ```python
  noise_scale = 0.1 * max(0.1, 1.0 - (total_steps / timesteps))
  # This decays noise from 0.1 to 0.01 over training duration
  ```

#### 5.6 Batch Size
**Current**: 64

**Suggested Adjustments**:
- **64**: Current, good balance
- **128**: Better stability, less noisy gradients (if GPU memory allows)
- **32**: Faster updates, noisier gradients

**For Kaggle GPU**: Use **128** if running, you have enough VRAM

#### 5.7 Update Frequency
**Current**: 4 (update after every 4 steps)

**Suggested Adjustments**:
- **4**: Current, reasonable
- **2**: More frequent updates, potentially better convergence
- **8**: Less frequent updates, smoother training

### 5.8 Recommended Hyperparameter Tuning Schedule

**Phase 1 - Baseline (Hours 0-1)**:
```python
learning_rate=0.0001
neurons=[24, 12, 6]
tau=0.001
batch_size=64
noise_scale=0.1
```

**Phase 2 - Increase Capacity (Hours 1-2)**:
```python
learning_rate=0.0003      # Slightly higher
neurons=[64, 32, 16]      # Larger network
tau=0.001
batch_size=128            # Larger batch
noise_scale=0.1 (decaying)
```

**Phase 3 - Fine-tune (Hours 2+)**:
```python
# Monitor validation loss, adjust if needed:
# - If underfitting: increase learning_rate to 0.0005
# - If overfitting: decrease learning_rate to 0.0001
# - If unstable: decrease learning_rate or tau
```

---

## 6. Guide on Motivating/Penalizing Agent Actions

### Current Problem Analysis
Your agent is:
- Falling over within 2 steps
- Moving very little (steps are mm-sized)
- **Root cause**: Environment's reward function likely doesn't properly incentivize the desired behavior

### Understanding Reward Functions

In RL, the reward function (provided by SAI environment) directly shapes behavior:
- **Positive rewards**: Encourage actions
- **Negative rewards (penalties)**: Discourage actions
- **No reward**: Agent doesn't learn the difference

### Solution: Reward Shaping

Since SAI environment provides `r` directly, you have two options:

#### Option A: Modify Reward in Training Loop (Recommended for Quick Wins)
Add reward shaping in `training_loop()`:

```python
# Inside training loop, after: new_s, r, terminated, truncated, info = env.step(action)

# REWARD SHAPING MODIFICATIONS
shaped_reward = r  # Start with original reward

# 1. Penalize falling over
if terminated or truncated:  # Episode ended
    shaped_reward = r - 10.0  # Heavy penalty for falling
else:
    shaped_reward = r + 0.1  # Small reward for staying upright

# 2. Penalize taking too many steps (encourage efficient movement)
episode_steps_penalty = -0.01 * episode_steps  # Penalty grows with steps
shaped_reward += episode_steps_penalty

# 3. Reward moving towards the ball (depends on task)
if 'ball_xpos_rel_robot' in info:
    ball_distance = np.linalg.norm(info['ball_xpos_rel_robot'])
    ball_distance_penalty = -0.05 * ball_distance  # Penalty for being far from ball
    shaped_reward += ball_distance_penalty

# 4. Penalize walking away from ball
if hasattr(model, 'last_ball_distance'):
    current_distance = np.linalg.norm(info['ball_xpos_rel_robot'])
    if current_distance > model.last_ball_distance:
        shaped_reward -= 0.1  # Penalize moving away
    model.last_ball_distance = current_distance
else:
    model.last_ball_distance = np.linalg.norm(info['ball_xpos_rel_robot'])

# Use shaped reward instead of original
replay_buffer.add(s, action, shaped_reward, new_s, done)
```

#### Option B: Task-Specific Reward Shaping
Different tasks need different incentives:

```python
# Detect task from info
task_index = info.get('task_index', 0)
shaped_reward = r

if task_index == 0:  # Penalty goalie kick
    # Penalize moving away from goal
    goal_distance = np.linalg.norm(info['goal_team_1_rel_robot'])
    shaped_reward += -0.1 * goal_distance

elif task_index == 1:  # Kick to target
    # Reward moving towards target
    target_distance = np.linalg.norm(info['target_xpos_rel_robot'])
    shaped_reward += -0.05 * target_distance
    
    # Bonus for being close to target
    if target_distance < 0.5:
        shaped_reward += 1.0

elif task_index == 2:  # Other task
    # Custom rewards based on task requirements
    pass
```

#### Option C: Action Space Constraints
Prevent undesired actions by modifying action selection:

```python
def constrained_action_function(policy, info):
    """
    Select action but constrain based on current state.
    """
    action = action_function(policy)  # Base action selection
    
    # Constraint 1: If already took many steps, reduce movement magnitude
    if episode_steps > 10:
        action *= 0.5  # Reduce action magnitude
    
    # Constraint 2: If falling, shut down action
    # (Check from robot state in observations)
    
    # Constraint 3: If moving away from ball, apply brakes
    if info['ball_xpos_rel_robot'][0] > 0:  # Ball is ahead in x-direction
        action[0] *= 0.3  # Reduce forward movement if going away
    
    return action
```

### 6.1 Reward Shaping Best Practices

**DO**:
- ✅ Use **dense rewards** (reward every step) not sparse rewards
- ✅ **Normalize** reward values to reasonable scale (-1 to +10)
- ✅ **Test incrementally** - add one reward term at a time
- ✅ **Weight terms carefully** - sum of terms should make sense
- ✅ **Log all reward components** to W&B to understand what's working

**DON'T**:
- ❌ Make reward magnitudes too large (>100) - causes instability
- ❌ Change reward function mid-training drastically
- ❌ Use only sparse rewards (agent learns very slowly)
- ❌ Forget about unit consistency (meters vs normalized values)

### 6.2 Implementation in Modified Training Loop

Here's the complete modified section:

```python
# Inside training_loop, after env.step():

new_s, r, terminated, truncated, info = env.step(action)
new_s = preprocessor.modify_state(new_s, info).squeeze()

done = terminated or truncated

# === REWARD SHAPING START ===
shaped_reward = r  # Start with base environment reward

# Component 1: Staying upright bonus
if not done:
    shaped_reward += 0.1
else:
    shaped_reward -= 5.0  # Heavy penalty for falling

# Component 2: Step efficiency penalty
shaped_reward -= 0.02 * episode_steps

# Component 3: Ball proximity reward
if 'ball_xpos_rel_robot' in info:
    ball_pos = info['ball_xpos_rel_robot']
    ball_distance = np.linalg.norm(ball_pos)
    
    # Reward proximity
    shaped_reward += max(0, 1.0 - ball_distance) * 0.1
    
    # Penalize moving away
    if hasattr(model, 'prev_ball_dist'):
        if ball_distance > model.prev_ball_dist:
            shaped_reward -= 0.1
    model.prev_ball_dist = ball_distance

# Log components for debugging
reward_components = {
    'base_reward': float(r),
    'shaped_reward': float(shaped_reward),
    'episode_steps': episode_steps,
}
# === REWARD SHAPING END ===

episode_reward += shaped_reward  # Use shaped reward
replay_buffer.add(s, action, shaped_reward, new_s, done)
```

### 6.3 Debugging Reward Shaping

Add console logging:

```python
# Every 100 steps, print reward breakdown
if total_steps % 100 == 0:
    print(f"Step {total_steps}: Base={reward_components['base_reward']:.2f}, "
          f"Shaped={reward_components['shaped_reward']:.2f}, "
          f"Episode Steps={reward_components['episode_steps']}")
```

Log to W&B:

```python
if total_steps % 1000 == 0:
    wandb.log({
        'rewards/base': reward_components['base_reward'],
        'rewards/shaped': reward_components['shaped_reward'],
        'rewards/episode_steps_penalty': -0.02 * episode_steps,
    }, step=total_steps)
```

---

## 7. Summary of Changes Required

### Files to Modify:
1. **`training_scripts/training.py`**
   - Add `validation_pass()` function
   - Add `save_checkpoint()` function  
   - Modify `training_loop()` to accept `checkpoint_dir` and `checkpoint_interval`
   - Add reward shaping logic
   - Add W&B logging for validation loss

2. **`training_scripts/main.py`**
   - Update `learning_rate` from 0.0001 to 0.0003
   - Update `neurons` from [24, 12, 6] to [64, 32, 16]
   - Update `training_loop()` call with checkpoint parameters
   - Increase `timesteps` from 1000 to 5000000

3. **`training_scripts/ddpg.py`** (Optional)
   - Add noise decay mechanism
   - Consider adding L2 regularization

### Files to Create:
1. **Kaggle Training Notebook** - Step-by-step notebook for running on Kaggle GPU

---

## 8. Execution Order

1. ✅ **Review this plan** (you just did!)
2. 📝 Modify `training.py` to add validation and checkpoints
3. 📝 Modify `main.py` with updated hyperparameters
4. 📝 Add reward shaping to `training.py`
5. 🚀 Test locally with small timesteps (1000 steps)
6. 🚀 Upload to Kaggle and run full 3+ hour training
7. 📊 Monitor W&B dashboard for validation loss double descent
8. 💾 Select best checkpoint and convert to submission format

---

**Ready to proceed with implementation?** Let me know and I'll start making the code changes!
