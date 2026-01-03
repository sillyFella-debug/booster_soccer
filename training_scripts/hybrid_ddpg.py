"""
DDPG agent with architecture matching converted_model.pt (Gaussian actor with layer norms)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianActor(nn.Module):
    """Actor network with layer normalization matching converted_model.pt architecture."""
    
    def __init__(self, input_dim, hidden_dims=[256, 256, 256], output_dim=12):
        super().__init__()
        
        self.hidden_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        # Build hidden layers with layer normalization
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.hidden_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
        
        # Output heads for mean and log_std
        self.mean_net = nn.Linear(prev_dim, output_dim)
        self.log_std_net = nn.Linear(prev_dim, output_dim)
        
        # Initialize log_std to be low (conservative exploration)
        self.log_std_net.bias.data.fill_(-2.0)
    
    def forward(self, x):
        """
        Forward pass returns mean action (deterministic).
        During training, you'd sample from Normal(mean, exp(log_std)).
        """
        for hidden, layer_norm in zip(self.hidden_layers, self.layer_norms):
            x = hidden(x)
            x = layer_norm(x)
            x = F.relu(x)
        
        mean = self.mean_net(x)
        return torch.tanh(mean)  # Squash to [-1, 1]
    
    def get_mean_and_std(self, x):
        """Return both mean and std for sampling during training."""
        for hidden, layer_norm in zip(self.hidden_layers, self.layer_norms):
            x = hidden(x)
            x = layer_norm(x)
            x = F.relu(x)
        
        mean = torch.tanh(self.mean_net(x))
        log_std = self.log_std_net(x)
        std = torch.exp(log_std)
        return mean, std


class SimpleNeuralNetwork(nn.Module):
    """Simple critic network (no layer normalization)."""
    
    def __init__(self, input_dim, hidden_dims=[256, 256], output_dim=1):
        super().__init__()
        
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        self.output_layer = nn.Linear(prev_dim, output_dim)
    
    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.output_layer(x)


class DDPG_Gaussian(nn.Module):
    """DDPG agent with Gaussian actor matching converted_model.pt architecture."""
    
    def __init__(self, input_dim, action_dim, actor_hidden=[256, 256, 256], 
                 critic_hidden=[256, 256], learning_rate=0.0001):
        super().__init__()
        
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = 0.99
        self.tau = 0.001
        
        # Actor network (Gaussian)
        self.actor = GaussianActor(input_dim, actor_hidden, action_dim)
        self.target_actor = GaussianActor(input_dim, actor_hidden, action_dim)
        self.target_actor.load_state_dict(self.actor.state_dict())
        
        # Critic networks (Q-functions)
        critic_input_dim = input_dim + action_dim
        self.critic = SimpleNeuralNetwork(critic_input_dim, critic_hidden, 1)
        self.target_critic = SimpleNeuralNetwork(critic_input_dim, critic_hidden, 1)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
    
    def soft_update_targets(self):
        """Soft update target networks."""
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def forward(self, state):
        """Get deterministic action (for inference)."""
        return self.actor(state.float())
    
    def select_action(self, state):
        """Select action from state (numpy input/output)."""
        state_tensor = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action = self.actor(state_tensor)
        return action.cpu().numpy()
    
    def train_step(self, states, actions, rewards, next_states, dones):
        """Single training step for critic and actor."""
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32).squeeze()
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).squeeze()
        
        # Ensure 1D if needed
        if rewards.dim() == 0:
            rewards = rewards.unsqueeze(0)
        if dones.dim() == 0:
            dones = dones.unsqueeze(0)
        
        # ===== Critic Update =====
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = self.target_critic(torch.cat([next_states, next_actions], dim=1))
            target_q = target_q.squeeze()
            y = rewards + self.gamma * target_q * (1 - dones)
            if y.dim() == 0:
                y = y.unsqueeze(0)
        
        current_q = self.critic(torch.cat([states, actions], dim=1)).squeeze()
        if current_q.dim() == 0:
            current_q = current_q.unsqueeze(0)
        
        critic_loss = F.mse_loss(current_q, y)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # ===== Actor Update =====
        actions_pred = self.actor(states)
        actor_loss = -self.critic(torch.cat([states, actions_pred], dim=1)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Soft update
        self.soft_update_targets()
        
        return critic_loss.item(), actor_loss.item()
    
    def train(self, states, actions, rewards, next_states, dones, epochs=1):
        """Train for multiple epochs (for compatibility)."""
        total_critic_loss = 0.0
        total_actor_loss = 0.0
        
        for _ in range(epochs):
            critic_loss, actor_loss = self.train_step(states, actions, rewards, next_states, dones)
            total_critic_loss += critic_loss
            total_actor_loss += actor_loss
        
        return total_critic_loss / epochs, total_actor_loss / epochs
    
    def load_pretrained_actor(self, checkpoint_path):
        """Smart loader: Handles both standard checkpoints and original BC TorchScript models."""
        print(f"Attempting to load: {checkpoint_path}")
        
        # -----------------------------------------------------------
        # STRATEGY 1: Try loading as a Standard Checkpoint (Your new files)
        # -----------------------------------------------------------
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Check if it's the dictionary format we saved
            if isinstance(checkpoint, dict) and 'actor' in checkpoint:
                print(">> Detected Standard Dictionary Checkpoint.")
                
                # 1. Load Actor
                self.actor.load_state_dict(checkpoint['actor'])
                self.target_actor.load_state_dict(checkpoint['actor'])
                print("   ✓ Actor loaded")

                # 2. Load Critic (CRITICAL for resuming RL!)
                if 'critic' in checkpoint:
                    self.critic.load_state_dict(checkpoint['critic'])
                    self.target_critic.load_state_dict(checkpoint['critic'])
                    print("   ✓ Critic loaded (Resuming training state)")
                
                return True
        except Exception as e:
            # If it fails, it might be the old TorchScript format, so we ignore this error
            pass

        # -----------------------------------------------------------
        # STRATEGY 2: Fallback to Original BC TorchScript (The old converted_model.pt)
        # -----------------------------------------------------------
        try:
            print(">> Attempting to load as TorchScript (Original BC format)...")
            scripted = torch.jit.load(checkpoint_path, map_location='cpu')
            state_dict = scripted.state_dict()
            
            # Map TorchScript keys to our actor keys
            actor_state_dict = {}
            for i in range(len(self.actor.hidden_layers)):
                w = state_dict[f'actor_net.hidden_layers.{i}.weight']
                b = state_dict[f'actor_net.hidden_layers.{i}.bias']
                
                # Remove truncation logic since we fixed dimensions
                actor_state_dict[f'hidden_layers.{i}.weight'] = w
                actor_state_dict[f'hidden_layers.{i}.bias'] = b
                actor_state_dict[f'layer_norms.{i}.weight'] = state_dict[f'actor_net.layer_norms.{i}.weight']
                actor_state_dict[f'layer_norms.{i}.bias'] = state_dict[f'actor_net.layer_norms.{i}.bias']
            
            # Map heads
            actor_state_dict['mean_net.weight'] = state_dict['mean_net.weight']
            actor_state_dict['mean_net.bias'] = state_dict['mean_net.bias']
            actor_state_dict['log_std_net.weight'] = state_dict['log_std_net.weight']
            actor_state_dict['log_std_net.bias'] = state_dict['log_std_net.bias']
            
            self.actor.load_state_dict(actor_state_dict)
            self.target_actor.load_state_dict(actor_state_dict)
            
            print("   ✓ TorchScript BC model loaded")
            return True
        except Exception as e:
            print(f"❌ Failed to load model. Error: {e}")
            return False
