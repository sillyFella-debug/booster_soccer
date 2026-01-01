# Training Guide for Booster Soccer Showdown (BC Agent)

This guide provides step-by-step instructions to train the Behavioral Cloning (BC) agent using your datasets.

## 1. GPU Setup & JAX CUDA Installation

The training script uses **JAX**, which requires a CUDA-enabled version to run on your GPU.

### Step 1: Verify CUDA is Available
```bash
nvidia-smi
```
Check the top-right corner for `CUDA Version` (e.g., `12.4`).

### Step 2: Install JAX with CUDA Support
Activate your virtual environment (`sai`), then run:

**For CUDA 12.x (recommended):**
```bash
pip install --upgrade "jax[cuda12]"
```

**For CUDA 11.x:**
```bash
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Step 3: Verify GPU is Detected
```bash
python3 -c "import jax; print(jax.devices())"
```
**Expected output:** `[CudaDevice(id=0)]`
**If you see:** `[CpuDevice(id=0)]` — the CUDA version is not installed correctly.

### Parallelization Notes
    *   **Single GPU**: JAX defaults to running heavily parallelized matrix operations on a single GPU. This is "parallel" in terms of tensor computation (vectorization).
    *   **Multiple GPUs**: The current `train.py` script is designed for single-device training by default. JAX will allocate memory on all GPUs, but operations typically run on the default device (GPU 0) unless the code explicitly shards data (e.g., using `jax.pmap` or `jax.sharding`). For this codebase, standard **Single GPU** training is the expected mode.

## 2. Training Commands

Run these commands from the root directory of the repository (`.../booster_soccer_showdown`).

### Option A: Using the `imitation_learning` Dataset
**Dataset Path**: `./booster_dataset/imitation_learning/booster_soccer_showdown.npz`

```bash
python3 imitation_learning/train.py \
    --agents bc \
    --dataset_dir ./booster_dataset/imitation_learning/booster_soccer_showdown.npz \
    --save_dir ./exp_local \
    --env_name Booster-v0
```

### Option B: Using the `data` Folder Dataset
**Dataset Path**: `./data/GoalPenalty_dataset.npz`

```bash
python3 imitation_learning/train.py \
    --agents bc \
    --dataset_dir ./data/GoalPenalty_dataset.npz \
    --save_dir ./exp_local \
    --env_name GoalPenalty-v0
```

*(Note: `env_name` is mainly used for experiment naming/logging, so you can set it to a descriptive name for your run.)*

## 3. Modifying Hyperparameters

The user requested info on how to modify hyperparameters for "good training". These are defined in the specific agent file.

**File to Edit**: `imitation_learning/agents/bc.py`

Look for the `BC_CONFIG_DICT` dictionary at the top of the file:

```python
BC_CONFIG_DICT = {
    "agent_name": 'bc',
    "lr": 1e-4,              # <--- Learning Rate: Try 3e-4 or 1e-3 if convergence is too slow.
    "batch_size": 1024,       # <--- Batch Size: Increase if you have high GPU VRAM (e.g., 2048).
    "actor_hidden_dims": (256, 256, 256), # <--- Network Size: (512, 512) might capture more complex behaviors.
    "discount": 0.99,
    # ...
}
```

### Recommended Tuning (Manual Modifications)
To experiment with performance, you can manually edit `imitation_learning/agents/bc.py`:
1.  **Learning Rate (`lr`)**: If loss doesn't decrease, try lowering it (`1e-5`). If it decreases too slowly, try raising it (`3e-4`).
2.  **Dataset Size**: Ensure your `.npz` files actually contain enough expert headers. The `train.py` script prints the shape of `data["observations"]` at startup; check this to ensure you have thousands of samples.
3.  **Noise**: The `train.py` script has an argument `--add_noise True --noise_scale 0.01` which can help prevent overfitting if your dataset is small.

## 4. Recommended Hyperparameter Changes (Based on GCBC Example)

Looking at the `gcbc.py` example, here are the **recommended modifications** to `imitation_learning/agents/bc.py`:

| Parameter | Current Value | Recommended Value | Why |
|---|---|---|---|
| `lr` | `1e-4` | `3e-4` | Faster convergence. Standard for Adam. |
| `actor_hidden_dims` | `(256, 256, 256)` | `(512, 512, 512)` | More capacity to learn complex behaviors. |
| `const_std` | `True` | `False` | Allows the network to learn action uncertainty. |

### How to Apply
Edit `imitation_learning/agents/bc.py`, find `BC_CONFIG_DICT`, and change the values:

```python
BC_CONFIG_DICT = {
    "agent_name": 'bc',
    "lr": 3e-4,                            # <-- Changed
    "batch_size": 1024,
    "actor_hidden_dims": (512, 512, 512),  # <-- Changed
    "discount": 0.99,
    "clip_threshold": 100.0,
    "const_std": False,                     # <-- Changed
    # ...
}
```

---

## 5. PyTorch Conversion (Post-Training)

The training **must** be done in JAX/Flax. However, after training, you can convert the model to **TorchScript** for deployment using the included script.

### Conversion Command
After training saves a checkpoint (e.g., `exp/booster/.../agent_100000.pkl`):

```bash
python3 imitation_learning/scripts/jax2torch.py \
    --pkl exp/booster/<your_run_group>/<exp_name>/agent_100000.pkl \
    --out ./converted_model.pt
```

### Using the Converted Model in PyTorch
```python
import torch

model = torch.jit.load("./converted_model.pt")
obs = torch.tensor([...], dtype=torch.float32)  # Your observation
mean, std = model(obs)
action = mean  # Use the mean action for deterministic inference
```

---

## 6. Weights & Biases (W&B) Configuration

By default, the script logs to a project named `booster`.

### Authenticating
Before training, log in to your account from the terminal:
```bash
wandb login
```

### Routing to your Account/Team
If you want to ensure it goes to a specific "Entity" (your username or team name), modify `imitation_learning/train.py` around line 34:

```python
# Change this:
setup_wandb(project='booster', group=args.run_group, name=exp_name)

# To this:
setup_wandb(entity='YOUR_USERNAME_OR_TEAM', project='booster', group=args.run_group, name=exp_name)
```

## 7. Validation Loss

**Is it currently logging?**
No. The current script only logs training loss (`training/actor/actor_loss`).

### How to add Validation Loss
To see validation loss in W&B, you need to split your data and log it periodically. Here are the instructions to modify `imitation_learning/train.py`:

#### Step 1: Split the Data
Around line 43, after loading the data, add a split:

```python
    # After these lines:
    data["observations"] = np.array(train_dataset["observations"], dtype=np.float32)
    data["actions"] = np.array(train_dataset["actions"], dtype=np.float32)

    # ADD THIS:
    num_samples = len(data["observations"])
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    split_idx = int(num_samples * 0.9) # 90% train, 10% val
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_data = {k: v[train_indices] for k, v in data.items()}
    val_data = {k: v[val_indices] for k, v in data.items()}
```

#### Step 2: Create a Validation Dataset
Update the dataset creation logic (around line 50) to use `train_data` and create a `val_dataset`:

```python
    # Create training dataset
    train_dataset = Dataset.create(**train_data)
    # Create validation dataset
    val_dataset = Dataset.create(**val_data)
```

#### Step 3: Log Validation Loss in the Loop
Inside the training loop (around line 83), add the validation calculation:

```python
        # Inside the if i % args.log_interval == 0: block
        if i % args.log_interval == 0:
            # ADD THIS: calculate validation loss
            val_batch = val_dataset.sample(agent_config['batch_size'])
            _, val_info = agent.actor_loss(val_batch, agent.network.params)
            
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            # ADD THIS:
            val_metrics = {f'validation/{k}': v for k, v in val_info.items()}
            
            # Combine them
            all_metrics = {**train_metrics, **val_metrics}
            all_metrics['time/epoch_time'] = (time.time() - last_time) / args.log_interval
            # ...
            wandb.log(sanitize_metrics(all_metrics), step=i)
```

## 8. Monitoring in W&B

Once you start training:
1.  Go to [wandb.ai](https://wandb.ai).
2.  Open the `booster` project.
3.  You will see charts for `training/actor/actor_loss`.
4.  If you implemented Step 5 above, you will also see `validation/actor_loss`.
5.  **Tip**: Check the `validation/actor_loss` to ensure it follows the training loss. If training loss goes down but validation loss goes up, your model is **overfitting**.

## 9. Offline Mode
If you don't want to sync to the cloud immediately, run:
```bash
WANDB_MODE=offline python3 imitation_learning/train.py ...
```
You can sync it later using `wandb sync`.
