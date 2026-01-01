# 🚀 KAGGLE TRAINING SETUP GUIDE - Complete Step-by-Step

## Overview

This guide walks you through **everything from scratch** - from creating a Kaggle account to running your TD3 training on Kaggle GPU.

**Estimated time:** 20 minutes to complete setup, then training runs automatically.

---

## PART 1: Kaggle Account & Setup (5 minutes)

### Step 1.1: Create/Login to Kaggle

1. Go to [kaggle.com](https://www.kaggle.com)
2. **If you don't have an account:**
   - Click "Sign up"
   - Create account with email
   - Verify email
   - **Go to Settings → API → Generate new token** (save this file!)

3. **If you already have an account:**
   - Login normally
   - Go to Settings → API → Download your `kaggle.json` token

### Step 1.2: Enable GPU on Kaggle

1. Open Kaggle Notebook (we'll create one in Step 2)
2. Click **"⚙ Session"** (gear icon, right side)
3. Under "Accelerator" select: **GPU** (P100 or better)
4. Click "Save"

**Why GPU?** 
- CPU training would take ~50 hours
- GPU training takes ~5 hours
- Kaggle GPU is FREE with Kaggle Pro

### Step 1.3: Check GPU Availability

Kaggle gives you **30 hours/week** of free GPU time. This is plenty for our 5-hour training!

---

## PART 2: Create Your First Kaggle Notebook (5 minutes)

### Step 2.1: Create New Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Click **"+ New Notebook"** (blue button, top right)
3. Choose **Python** (not R)
4. Name it: **"TD3-Soccer-Training"** (or whatever you want)
5. Click **"Create"**

### Step 2.2: Setup Notebook

Now you should see a blank notebook with one empty cell.

**First cell - Title and Setup:**

```python
# TD3 Reinforcement Learning Training for Booster Soccer
# This notebook trains a TD3 agent on SAI soccer environments
# GPU: Kaggle P100 (5 hours for 5M steps)

import os
print(f"Working directory: {os.getcwd()}")
print(f"Available GPU: {os.system('nvidia-smi')}")
```

Run this cell to verify GPU is available.

---

## PART 3: Upload Code & Data to Kaggle (10 minutes)

### Option A: Clone from GitHub (RECOMMENDED - Easiest)

**Create a new cell and run:**

```python
# Clone the booster soccer repository
!git clone https://github.com/ArenaX-Labs/booster_soccer_showdown.git
!ls -la booster_soccer_showdown/
```

This will:
- ✅ Clone entire repo
- ✅ Get all scripts automatically
- ✅ Get the structure you need

**Then verify:**

```python
# Check if key files exist
import os

files_to_check = [
    'booster_soccer_showdown/training_scripts/train_td3_rl.py',
    'booster_soccer_showdown/training_scripts/multi_task_env.py',
]

for f in files_to_check:
    exists = os.path.exists(f)
    print(f"{'✓' if exists else '✗'} {f}")
```

### Option B: Upload Your Local Files (If GitHub doesn't work)

If GitHub clone fails:

1. On Kaggle notebook, click **"+ Add Input"** (top left)
2. Click **"Upload files"**
3. Upload these files from your computer:
   - `train_td3_rl.py`
   - `multi_task_env.py`
   - `converted_model.pt` (your pre-trained model!)
4. They'll appear in `/kaggle/input/`

---

## PART 4: Install Dependencies (2 minutes)

**Create new cell:**

```python
# Install required packages
!pip install -q sai-rl gymnasium numpy torch stable-baselines3 wandb tensorboard

print("✓ Dependencies installed successfully!")
```

**Why these packages?**
- `sai-rl` - SAI environments for soccer
- `gymnasium` - Environment API
- `torch` - Deep learning
- `stable-baselines3` - TD3 algorithm
- `wandb` - Experiment tracking
- `tensorboard` - Training monitoring

---

## PART 5: Setup W&B Logging (Optional but Recommended) (2 minutes)

### Step 5.1: Create W&B Account

1. Go to [wandb.ai](https://wandb.ai)
2. Sign up with GitHub/Google/Email
3. Create free account
4. Go to [wandb.ai/authorize](https://wandb.ai/authorize) to get API key

### Step 5.2: Login in Kaggle

**Create new cell:**

```python
# Setup Weights & Biases logging
import wandb

# Login to W&B (it will ask for API key)
wandb.login()

print("✓ Logged into Weights & Biases!")
```

When you run this:
- It will show a URL to get your API key
- Copy the key from W&B website
- Paste it in the Kaggle prompt
- ✅ You're logged in!

**Why W&B?**
- 📊 Track training metrics in real-time
- 📈 See plots of episode rewards
- 💾 Save training history
- 🔗 Share results with others

---

## PART 6: Prepare Training Configuration (2 minutes)

**Create new cell - Setup paths:**

```python
import os
import sys

# Add training scripts to path
sys.path.insert(0, '/kaggle/working')
sys.path.insert(0, '/kaggle/input/booster_soccer_showdown')

# Create output directory for checkpoints
os.makedirs('/kaggle/working/checkpoints', exist_ok=True)
os.makedirs('/kaggle/working/models', exist_ok=True)

print("✓ Directories created!")
print(f"Working directory: /kaggle/working")

# List available files
print("\nAvailable files:")
for root, dirs, files in os.walk('/kaggle/input/booster_soccer_showdown/training_scripts'):
    for file in files:
        if file.endswith('.py'):
            print(f"  • {file}")
```

---

## PART 7: Import Your Code (1 minute)

**Create new cell:**

```python
# Import custom modules
sys.path.insert(0, '/kaggle/input/booster_soccer_showdown/training_scripts')

from train_td3_rl import main
import argparse

print("✓ Training script imported successfully!")
```

---

## PART 8: RUN TRAINING! (The Main Event!)

**This is it! Create the final cell:**

```python
# Configure training arguments
class Args:
    # Environment
    env = "LowerT1GoaliePenaltyKick-v0"
    second_env = None  # Can add second env later
    
    # Model
    pretrained_path = "/kaggle/input/booster_soccer_showdown/converted_model.pt"
    device = "cuda"  # GPU!
    
    # Training
    timesteps = 5000000  # 5M steps = ~5 hours on GPU
    batch_size = 256
    learning_rate = 3e-4
    buffer_size = 1000000
    learning_starts = 10000
    
    # TD3
    tau = 0.005
    gamma = 0.99
    policy_delay = 2
    
    # Checkpoints (every 5K steps!)
    checkpoint_interval = 5000
    log_interval = 100
    
    # Output
    save_dir = "/kaggle/working/models"
    checkpoint_dir = "/kaggle/working/checkpoints"
    
    # Logging
    use_wandb = True  # Enable W&B logging!

# Create args object
args = Args()

print("=" * 60)
print("🚀 STARTING TD3 TRAINING")
print("=" * 60)
print(f"Environment: {args.env}")
print(f"GPU Device: {args.device}")
print(f"Total timesteps: {args.timesteps:,} (≈ 5 hours)")
print(f"Checkpoint interval: {args.checkpoint_interval:,} steps")
print(f"Network: [512, 256, 128, 64] (DEEP & LARGE)")
print(f"W&B Logging: {'Yes ✓' if args.use_wandb else 'No'}")
print("=" * 60)

# Run training
main(args)

print("\n" + "=" * 60)
print("✓ TRAINING COMPLETE!")
print("=" * 60)
```

---

## PART 9: Monitor Training (During Training)

### Watch Progress in W&B

While training runs:

1. Go to [wandb.ai/home](https://wandb.ai/home)
2. Find your run: **"td3_rl_YYYYMMDD_HHMMSS"**
3. Watch these metrics:
   - **`rollout/ep_rew_mean`** - Episode reward (should go UP!)
   - **`train/policy_loss`** - Actor loss (should go DOWN)
   - **`train/value_loss`** - Critic loss (should go DOWN)

### What to Expect:

```
Hour 0-1: Agent explores, episode reward = low (-10 to 0)
Hour 1-2: Improvement, episode reward = -5 to +5
Hour 2-3: Good progress, episode reward = +5 to +15
Hour 3-5: Convergence, plateau or double descent
```

---

## PART 10: After Training Completes

**New cell - Save final model:**

```python
import torch
import shutil

# Copy final model to working directory
final_model_path = "/kaggle/working/models/td3_final_model.pt"

if os.path.exists(final_model_path):
    # Copy to output folder for download
    shutil.copy(final_model_path, "/kaggle/working/td3_final_model_FINAL.pt")
    print(f"✓ Final model saved to: /kaggle/working/td3_final_model_FINAL.pt")
else:
    print("⚠ Model file not found")

# List all checkpoints
print("\nCheckpoints saved:")
checkpoint_dir = "/kaggle/working/checkpoints"
if os.path.exists(checkpoint_dir):
    checkpoints = sorted(os.listdir(checkpoint_dir))
    for cp in checkpoints[:5]:  # Show first 5
        print(f"  • {cp}")
    if len(checkpoints) > 5:
        print(f"  ... and {len(checkpoints) - 5} more")
else:
    print("No checkpoints directory found")
```

---

## PART 11: Download Your Model (After Training)

After training finishes:

1. Click **"📥 Output"** folder (top right)
2. You'll see:
   - `checkpoints/` - All checkpoint files
   - `td3_final_model_FINAL.pt` - Your final trained model
3. Click the download icon (📥) next to files to download

**Then you can:**
- ✅ Test locally on your computer
- ✅ Submit to SAI competition
- ✅ Fine-tune further

---

## Quick Reference: Full Cell Order

Copy/paste these cells in order:

**Cell 1: Title & GPU Check**
```python
# TD3 Soccer Training
import os
print(f"GPU available: {os.system('nvidia-smi') == 0}")
```

**Cell 2: Clone Repository**
```python
!git clone https://github.com/ArenaX-Labs/booster_soccer_showdown.git
```

**Cell 3: Install Dependencies**
```python
!pip install -q sai-rl gymnasium numpy torch stable-baselines3 wandb tensorboard
```

**Cell 4: Setup W&B (Optional)**
```python
import wandb
wandb.login()
```

**Cell 5: Setup Paths**
```python
import os, sys
sys.path.insert(0, '/kaggle/input/booster_soccer_showdown/training_scripts')
os.makedirs('/kaggle/working/checkpoints', exist_ok=True)
os.makedirs('/kaggle/working/models', exist_ok=True)
```

**Cell 6: Import**
```python
from train_td3_rl import main
```

**Cell 7: Configure & RUN**
```python
class Args:
    env = "LowerT1GoaliePenaltyKick-v0"
    second_env = None
    pretrained_path = "/kaggle/input/booster_soccer_showdown/converted_model.pt"
    device = "cuda"
    timesteps = 5000000
    batch_size = 256
    learning_rate = 3e-4
    buffer_size = 1000000
    learning_starts = 10000
    tau = 0.005
    gamma = 0.99
    policy_delay = 2
    checkpoint_interval = 5000
    log_interval = 100
    save_dir = "/kaggle/working/models"
    checkpoint_dir = "/kaggle/working/checkpoints"
    use_wandb = True

args = Args()
main(args)
```

**Cell 8: Download Results**
```python
import shutil
shutil.copy("/kaggle/working/models/td3_final_model.pt", 
            "/kaggle/working/td3_final_model_FINAL.pt")
print("✓ Ready to download!")
```

---

## Troubleshooting

### Issue: "Module not found: sai_mujoco"
**Solution:**
```python
# Try installing sai-rl differently
!pip install sai-rl --upgrade
```

### Issue: "Out of memory" error
**Solution:**
```python
# Reduce batch size in Args:
batch_size = 128  # Instead of 256
# Or reduce buffer_size to 500000
```

### Issue: "GPU not available"
**Solution:**
1. Click **⚙ Session** (top right)
2. Change Accelerator to **GPU**
3. Click Save and restart

### Issue: Pre-trained model not loading
**Solution:**
```python
# Comment out or use default:
pretrained_path = None  # Train from scratch instead
# Then main() will start from random initialization
```

---

## What Happens After 9 Hours?

Kaggle notebooks timeout after **9 hours**, but that's OK because:

1. ✅ Your training saves checkpoints every 5K steps
2. ✅ Final model is saved automatically
3. ✅ W&B has all your metrics backed up
4. ✅ You can resume from checkpoint later

**To resume:**
```python
# Load from checkpoint
model = TD3.load("/kaggle/working/checkpoints/td3_checkpoint_2500000_steps.zip", env=env)
# Continue training
model.learn(total_timesteps=2500000)  # Train more
```

---

## Summary: From Start to Finish

| Step | Time | Action |
|------|------|--------|
| 1 | 2 min | Create Kaggle notebook |
| 2 | 2 min | Enable GPU |
| 3 | 5 min | Clone repository + install deps |
| 4 | 2 min | Setup W&B (optional) |
| 5 | 1 min | Setup paths and imports |
| 6 | 5 hours | **RUN TRAINING** ← Main event! |
| 7 | 2 min | Download results |
| **TOTAL** | **5 hours 20 min** | Ready to compete! |

---

## Next: What After Training?

After 5 hours of training on Kaggle GPU:

1. **Download the model:** `td3_final_model_FINAL.pt`
2. **Review W&B metrics:** Did episode reward improve?
3. **Consider Task 4:** Reward shaping for even better results
4. **Submit to competition!** Or fine-tune further

---

## Important Notes

### Kaggle GPU Time

- **Free Tier:** 30 hours/week of GPU
- **Our training:** 5 hours
- **You get:** 25 hours left for other experiments

### Pre-trained Model

Your `converted_model.pt` will be used as initialization. This means:
- ✅ Training starts from learned features
- ✅ Converges faster
- ✅ Better final performance

### Multi-Task Training (Later)

Once single-task works, you can add second environment:
```python
args.second_env = "LowerT1KickToTarget-v0"
# Agent will randomly sample between both tasks
```

---

**You're ready! Follow the steps above and you'll have a trained TD3 agent in 5 hours! 🚀**
