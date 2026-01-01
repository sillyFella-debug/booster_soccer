# 🚀 KAGGLE TRAINING - QUICK START (Copy-Paste Ready)

## The Absolute Fastest Way to Train on Kaggle

This is the **minimalist version** - just copy and paste each cell in order.

---

## PRE-REQUISITE: Have GPU Enabled

1. Create a new Kaggle notebook
2. Click ⚙ **Session** (top right)
3. Set Accelerator: **GPU** (P100)
4. Click **Save**

---

## Cell 1: Setup & Dependencies

```python
# Install everything needed
!pip install -q sai-rl gymnasium numpy torch stable-baselines3 wandb tensorboard

import sys
import os

# Setup paths
sys.path.insert(0, '/kaggle/working')
os.makedirs('/kaggle/working/checkpoints', exist_ok=True)
os.makedirs('/kaggle/working/models', exist_ok=True)

print("✓ Setup complete!")
```

---

## Cell 2: Clone Repository

```python
# Get the code from GitHub
!git clone https://github.com/ArenaX-Labs/booster_soccer_showdown.git

# Add to path
sys.path.insert(0, '/kaggle/input/booster_soccer_showdown/training_scripts')

print("✓ Repository cloned!")
```

---

## Cell 3: Login to W&B (Optional)

```python
import wandb

# Login - it will show a URL to get your API key from wandb.ai
wandb.login()

print("✓ W&B logged in!")
```

---

## Cell 4: Import Training Script

```python
from train_td3_rl import main

print("✓ Training script imported!")
```

---

## Cell 5: Configure & RUN TRAINING ⭐

```python
# Training configuration
class Args:
    # Environment
    env = "LowerT1GoaliePenaltyKick-v0"
    second_env = None
    
    # Model & device
    pretrained_path = "/kaggle/input/booster_soccer_showdown/converted_model.pt"
    device = "cuda"
    
    # Training hyperparameters
    timesteps = 5000000          # 5M steps = ~5 hours
    batch_size = 256
    learning_rate = 3e-4
    buffer_size = 1000000
    learning_starts = 10000
    
    # TD3 parameters
    tau = 0.005
    gamma = 0.99
    policy_delay = 2
    
    # Checkpoints & logging
    checkpoint_interval = 5000   # Every 5K steps
    log_interval = 100
    save_dir = "/kaggle/working/models"
    checkpoint_dir = "/kaggle/working/checkpoints"
    use_wandb = True

# Run training
args = Args()
print("🚀 STARTING TRAINING...")
main(args)
print("✓ TRAINING COMPLETE!")
```

---

## Cell 6: Download Your Model

```python
import shutil

# Copy final model for download
src = "/kaggle/working/models/td3_final_model.pt"
dst = "/kaggle/working/td3_final_model_FINAL.pt"

if os.path.exists(src):
    shutil.copy(src, dst)
    print(f"✓ Model ready to download: {dst}")
else:
    print("⚠ Model not found")

# List checkpoints
print("\nCheckpoints created:")
for f in sorted(os.listdir("/kaggle/working/checkpoints"))[:10]:
    print(f"  • {f}")
```

---

## That's It! 🎉

**6 cells, ~5 hours of training, one fully trained TD3 agent!**

### What each cell does:
1. Install dependencies
2. Clone code from GitHub
3. Setup W&B monitoring (optional)
4. Import training script
5. **RUN TRAINING** ← Main event!
6. Save & download results

### Monitor training:
- Go to [wandb.ai/home](https://wandb.ai/home)
- Watch `rollout/ep_rew_mean` go up!

### After training:
- Download `td3_final_model_FINAL.pt`
- Test locally or submit to competition

---

## Common Issues & Fixes

| Issue | Fix |
|-------|-----|
| "Module not found" | Re-run cell 2 to clone repo |
| "GPU not available" | Enable GPU in Session settings |
| "Out of memory" | Change `batch_size = 128` |
| "Pre-trained not found" | Set `pretrained_path = None` |

---

**Questions? See the full guide: `KAGGLE_TRAINING_COMPLETE_GUIDE.md`**
