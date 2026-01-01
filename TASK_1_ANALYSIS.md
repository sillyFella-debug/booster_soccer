# TASK 1: Analyze Training Requirements for All Tasks

## Executive Summary
✅ **YES, you MUST train on all 3 tasks**, not just the penalty goalie kick.
**Root Cause of Low Score (-2.48)**: Your model is overtrained on ONE task and fails to generalize.

---

## Current Situation

### What You Have
- **Current Model**: Trained only on `LowerT1GoaliePenaltyKick-v0`
- **Performance**: Score of -2.48 (falls over in 2 steps, mm-sized movements)
- **Problem**: Model is **task-specific**, not task-agnostic

### What Competition Requires
The SAI competition includes **3 distinct tasks**:

| Task # | Environment Name | Role | Objective |
|--------|-----------------|------|-----------|
| 1 | `LowerT1GoaliePenaltyKick-v0` | **Goalie (Defensive)** | Block incoming penalty kicks at the goal |
| 2 | `LowerT1KickToTarget-v0` | **Striker (Offensive)** | Kick ball to designated target location |
| 3 | `LowerT1SoccerGame-v0` | **Player (Mixed)** | Play mixed game scenarios (or similar) |

### How Tasks Are Identified
When you call `env.reset()`, the returned `info` dictionary includes:
```python
info = {
    'task_index': [1, 0, 0],  # One-hot encoded task identifier
    'ball_xpos_rel_robot': [...],
    'goal_team_0_rel_robot': [...],
    # ... other fields
}
```

The preprocessor in `main.py` already extracts this:
```python
def get_task_onehot(self, info):
    if 'task_index' in info:
        return info['task_index']
    return np.array([])
```

**This one-hot encoding is concatenated to observations**, meaning your network already has the capability to learn task-specific behavior!

---

## Why Single-Task Training Fails

### The Core Issue
When you train ONLY on penalty goalie kick:
1. Robot learns: "Move to goal, block kicks"
2. Robot learns to minimize the **specific reward structure** of that task
3. Network weights optimize for that **single task context**

When tested on other tasks:
1. Input distribution changes (different ball trajectories, target locations)
2. One-hot encoding changes (but network doesn't know what to do)
3. Reward structure is different, but policy is already optimized
4. **Result**: Catastrophic failure (falls over, random movements)

### Why This Happens with RL
- RL policies are **not robust to distribution shift**
- Training on single task = learning a **task-specific policy**, not a general policy
- Even though one-hot encoding is in observations, the agent needs multi-task training to learn what to do with it

---

## Solution: Multi-Task Training

### Approach A: Mixed Training Data (RECOMMENDED ✅)

**Concept**: Collect data from all 3 tasks, mix them in the replay buffer during training.

**Pros**:
- ✅ Single model handles all tasks
- ✅ Generalizes across task variations
- ✅ Smaller submission size
- ✅ Task-specific behavior emerges automatically

**Cons**:
- ⚠️ Takes longer to collect data for all tasks
- ⚠️ Model size stays same, might need bigger network

**Implementation**:
1. Collect training episodes from all 3 tasks
2. Pool all data together in replay buffer
3. Train single DDPG/TD3/SAC agent on mixed data
4. Agent learns implicit task-specific policies from one-hot encoding

**Example flow**:
```python
# During data collection
all_data = []
for task in [task1, task2, task3]:
    data = collect_episodes(task)
    all_data.extend(data)  # Mix together!

# During training
for step in range(timesteps):
    # Replay buffer has mixed data from all tasks
    batch = replay_buffer.sample()  # Contains all task types
    # Agent learns to condition behavior on task_index
    agent.train(batch)
```

---

### Approach B: Task-Specific Models

**Concept**: Train 3 separate models, one per task.

**Pros**:
- ✅ Simpler data collection (separate)
- ✅ Potentially higher per-task performance
- ✅ Easier to debug

**Cons**:
- ❌ 3x the submission size (if SAI allows multiple models)
- ❌ Not recommended for a "generalization" competition
- ❌ More training overhead

---

## Recommended Path Forward

### Step 1: Data Collection (What You Need to Do)
Collect training data for **all 3 tasks**. The existing `collect_data.py` script can help:

```bash
# Collect from penalty goalie (you might already have this)
python booster_control/teleoperate.py --env LowerT1GoaliePenaltyKick-v0
# ... teleoperate and save data ...

# Collect from kick to target
python booster_control/teleoperate.py --env LowerT1KickToTarget-v0
# ... teleoperate and save data ...

# Collect from soccer game
python booster_control/teleoperate.py --env LowerT1SoccerGame-v0
# ... teleoperate and save data ...
```

**Dataset storage structure**:
```
booster_dataset/
  soccer/
    booster_lower_t1/
      penalty_goalie/       # Task 1 data
        episode_001.npz
        episode_002.npz
        ...
      kick_to_target/       # Task 2 data
        episode_001.npz
        ...
      soccer_game/          # Task 3 data
        episode_001.npz
        ...
```

### Step 2: Data Merging
Combine all task data into single numpy arrays:

```python
import numpy as np

def merge_datasets(task1_path, task2_path, task3_path, output_path):
    """Merge multi-task data into single training dataset."""
    
    all_obs = []
    all_actions = []
    
    for task_path in [task1_path, task2_path, task3_path]:
        data = np.load(task_path, allow_pickle=True)
        all_obs.append(data['observations'])
        all_actions.append(data['actions'])
    
    merged_obs = np.concatenate(all_obs, axis=0)
    merged_actions = np.concatenate(all_actions, axis=0)
    
    np.savez_compressed(
        output_path,
        observations=merged_obs,
        actions=merged_actions
    )
    
    print(f"✓ Merged dataset: {merged_obs.shape[0]} steps")

# Usage
merge_datasets(
    'task1_data.npz',
    'task2_data.npz', 
    'task3_data.npz',
    'merged_multi_task.npz'
)
```

### Step 3: Training with Multi-Task Data
When you call `training_loop()`, the replay buffer will contain a mix of all tasks:

```python
# In main.py
training_loop(
    env, 
    model, 
    action_function, 
    Preprocessor,
    timesteps=5000000,  # Long training for multi-task
)
```

The agent will see diverse task contexts and learn to handle all of them.

---

## Algorithm Recommendation

You asked about DDPG vs SAC vs TD3. Here's my recommendation:

### For Your Use Case (Soccer with Continuous Control):

| Algorithm | Stability | Sample Efficiency | Exploration | Recommendation |
|-----------|-----------|-------------------|-------------|-----------------|
| **DDPG** | 🟡 Medium | 🟡 Good | 🔴 Limited | Baseline only |
| **TD3** | 🟢 **Excellent** | 🟢 Good | 🟡 Medium | **USE THIS** ✅ |
| **SAC** | 🟢 Excellent | 🟢 Excellent | 🟢 **Best** | Also great |

### My Recommendation: **TD3** (Twin Delayed DDPG)

**Why TD3?**
1. **Solves DDPG overestimation problem** - TD3 uses 2 critics, prevents value overestimation
2. **Better for your penalty kicks** - More stable with sparse reward signals
3. **Proven in robotics** - Used in many robot control tasks
4. **Step efficiency penalties** - Better handles action-based penalties (too many steps)
5. **Easier to implement** than SAC

**Why NOT SAC right now?**
- SAC is excellent but more complex
- Requires tuning temperature parameter
- Slower in pure deterministic tasks
- Good fallback if TD3 doesn't work

**TD3 vs DDPG Key Differences**:
```python
# DDPG has 1 critic network
self.critic = NeuralNetwork(...)

# TD3 has 2 critic networks (ensemble)
self.critic_1 = NeuralNetwork(...)
self.critic_2 = NeuralNetwork(...)

# TD3 uses delayed updates
if step % policy_delay == 0:  # Usually delay=2
    update_actor()
```

---

## Action Items Before Next Task

### What You Need to Do:

1. **Confirm available datasets**: 
   - Do you have collected data for all 3 tasks?
   - If not, you'll need to teleoperate and collect data
   
2. **Merge datasets** (if you have all 3 task datasets):
   - Combine into single merged dataset
   - Verify shapes are compatible

3. **Decide on algorithm**:
   - ✅ **TD3** (my recommendation)
   - Or stick with DDPG
   - Or implement SAC

4. **Get ready for next task**:
   - Task 2 will implement checkpoint saving + validation loss
   - Task 3 will implement reward shaping
   - Task 4 will fine-tune hyperparameters

---

## Summary

| Question | Answer |
|----------|--------|
| **Do I need to train on all 3 tasks?** | ✅ YES - Critical for competition |
| **Can one model handle all tasks?** | ✅ YES - Your network already has task_index input |
| **Best approach?** | Mixed training data from all 3 tasks |
| **Why did my single-task model fail?** | Distribution shift + task-specific policy |
| **What algorithm should I use?** | TD3 (better than DDPG, simpler than SAC) |

---

## Next Steps

Reply with:
1. ✅ Confirmation that you understand the analysis
2. 📊 Do you have collected data for all 3 tasks?
3. 🔧 Should we proceed with TD3 instead of DDPG?
4. Ready to move to **Task 2** (checkpoint saving + validation loss)?

Once confirmed, I'll move to Task 2 and implement the actual code changes.
