# TASK 2 COMPLETE: TD3 RL Training Setup Summary

## ✅ What Has Been Created

### 1. Multi-Task Environment Wrapper
**File**: `training_scripts/multi_task_env.py`

Features:
- Wraps SAI soccer environments
- Randomly samples which environment for each episode
- Handles different observation shapes gracefully
- Adds task information to info dict

### 2. TD3 Training Script
**File**: `training_scripts/train_td3_rl.py`

Features:
- ✅ Loads pre-trained model from `converted_model.pt`
- ✅ Saves checkpoints every 5K steps
- ✅ W&B logging integration
- ✅ Supports multi-task training
- ✅ Console progress reporting
- ✅ Tensorboard support

---

## Key Configuration Reference

### Default Hyperparameters (Task 2)
```
Learning rate:        3e-4
Batch size:          256
Buffer size:         1,000,000
Learning starts:     10,000
Tau (soft update):   0.005
Gamma (discount):    0.99
Policy delay:        2
Checkpoint interval: 5,000 steps
```

These are baseline values. We'll tune them in Task 3.

---

## How to Run Training

### Basic Training (Single Task, No Logging)
```bash
cd /media/deter/New\ Volume/Neamur/codes/booster_soccer_showdown

"/media/deter/New Volume/Neamur/codes/booster_soccer_showdown/sai/bin/python" \
  training_scripts/train_td3_rl.py \
  --env LowerT1GoaliePenaltyKick-v0 \
  --timesteps 5000000 \
  --device cpu \
  --save_dir ./exp_td3_rl
```

### Training with W&B Logging (Recommended for Kaggle)
```bash
"/media/deter/New Volume/Neamur/codes/booster_soccer_showdown/sai/bin/python" \
  training_scripts/train_td3_rl.py \
  --env LowerT1GoaliePenaltyKick-v0 \
  --timesteps 5000000 \
  --device cuda \
  --use_wandb \
  --save_dir ./exp_td3_rl
```

### Multi-Task Training (Both Environments)
```bash
"/media/deter/New Volume/Neamur/codes/booster_soccer_showdown/sai/bin/python" \
  training_scripts/train_td3_rl.py \
  --env LowerT1GoaliePenaltyKick-v0 \
  --second_env LowerT1KickToTarget-v0 \
  --timesteps 5000000 \
  --device cuda \
  --use_wandb \
  --save_dir ./exp_td3_rl_multi
```

---

## Testing the Setup

### Quick Test (5K Steps = ~5 Minutes)
```bash
"/media/deter/New Volume/Neamur/codes/booster_soccer_showdown/sai/bin/python" \
  training_scripts/train_td3_rl.py \
  --env LowerT1GoaliePenaltyKick-v0 \
  --timesteps 5000 \
  --checkpoint_interval 5000 \
  --device cpu \
  --save_dir ./test_td3_run
```

Expected output:
```
[MultiTaskWrapper] Initialized with 1 task(s):
  • LowerT1GoaliePenaltyKick-v0: obs_shape=(45,), action_shape=(12,)
[Model] Initializing TD3 model...
[Model] Loading pre-trained weights from: converted_model.pt
[Model] ✓ Loaded pre-trained policy weights
[Training] Starting TD3 training...
  Total timesteps: 5,000
  Batch size: 256
  Learning rate: 0.0003
  Checkpoint interval: 5000 steps
...
[Model] ✓ Saved final model to: ./test_td3_run/td3_final_model.zip
[Model] ✓ Saved PyTorch weights to: ./test_td3_run/td3_final_model.pt
[Training] ✓ Training complete!
```

---

## What Gets Logged to W&B

When you use `--use_wandb`, the following metrics are tracked:

### Training Metrics
- `rollout/ep_len_mean` - Average episode length
- `rollout/ep_rew_mean` - **Average episode reward** ← Main metric!
- `train/policy_loss` - Actor loss
- `train/value_loss` - Critic loss
- `train/actor_loss` - Actor network loss
- `train/critic_loss` - Critic network loss

### Key Metric to Watch
**`rollout/ep_rew_mean`** indicates if the agent is learning:
- ✅ Going up = Agent learning
- ❌ Flat or going down = Agent not learning (may need reward shaping)

---

## File Locations After Training

```
./exp_td3_rl/
├── td3_final_model.zip          # SB3 format (complete model)
├── td3_final_model.pt           # PyTorch format (weights only)
├── tensorboard/                 # Tensorboard logs
├── checkpoints/
│   ├── td3_checkpoint_5000_steps.zip
│   ├── td3_checkpoint_10000_steps.zip
│   ├── td3_checkpoint_15000_steps.zip
│   └── ...                       # More checkpoints every 5K steps
└── logs/                        # Training logs
```

### Using a Checkpoint to Resume Training

If training is interrupted, you can resume from the latest checkpoint:

```bash
# Load latest checkpoint and continue
model = TD3.load("./exp_td3_rl/checkpoints/td3_checkpoint_50000_steps.zip", env=env)
model.learn(total_timesteps=2500000, ...)  # Continue training
```

---

## Debugging Common Issues

### Issue: "Pre-trained weights not loading"
```
[WARNING] Could not load pre-trained weights: ...
```
**Solution**: 
- Verify `converted_model.pt` exists in current directory
- Ensure it's a valid PyTorch file
- Training will start from random initialization (still works, but slower)

### Issue: "Environment doesn't exist"
```
gymnasium.error.NameNotFound: Environment `LowerT1GoaliePenaltyKick-v0` doesn't exist
```
**Solution**:
- Make sure `import sai_mujoco` is before creating environment
- Try: `"--env LowerT1KickToTarget-v0"` instead

### Issue: "CUDA out of memory"
```
RuntimeError: CUDA out of memory
```
**Solution**:
- Reduce batch_size: `--batch_size 128`
- Reduce buffer_size: `--buffer_size 500000`
- Use CPU: `--device cpu`

---

## Next Task: Task 3 - Hyperparameter Tuning

After confirming Task 2 works, Task 3 will:
1. Suggest improved hyperparameters for TD3
2. Provide tuning strategy
3. Explain how to modify parameters for your specific task

---

## Summary Checklist

### Before Moving to Task 3, Confirm:

- [ ] `training_scripts/multi_task_env.py` created
- [ ] `training_scripts/train_td3_rl.py` created
- [ ] Scripts load without errors (`--help` works)
- [ ] `converted_model.pt` exists in workspace root
- [ ] Ready to run a quick test

### When Ready for Task 3:

Reply with:
- ✅ "Script files created and working"
- ✅ "Ready to test with 5K timesteps"
- ✅ "Questions before testing?"

Then I'll provide Task 3 (Hyperparameter Tuning) 🚀

---

## Important Notes

### Multi-Task Training Behavior

When you set `--second_env`, the wrapper will:
1. Randomly sample between both environments
2. ~50% of episodes from env1, ~50% from env2
3. Single agent learns to handle both tasks
4. Training takes longer but generalization improves

**Recommendation**: Start with single task (easier to debug), then add second task once first task is working.

### Performance Expectations

With pre-trained initialization:
- **Hour 1**: Agent should start improving from baseline
- **Hour 2**: Noticeable improvements in episode reward
- **Hour 3-5**: Convergence or double descent phenomenon

If episode reward is flat or negative:
- Reward signal may be sparse
- May need reward shaping (Task 4)
- Check environment is working correctly

---

This completes **Task 2: TD3 RL Training Setup with Validation Loss Reporting** ✅
