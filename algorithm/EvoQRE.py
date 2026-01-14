from algorithm.TrafficGamer import TrafficGamer
from algorithm.modules_evoqre import MixturePolicy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import copy

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class QNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QNetwork, self).__init__()
        # State-action Q-network
        self.fc1 = layer_init(nn.Linear(state_dim + action_dim, hidden_dim))
        self.fc2 = layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.fc3 = layer_init(nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class EvoQRE(TrafficGamer):
    def __init__(self, state_dim: int, agent_number: int, config, device):
        # Initialize TrafficGamer (and parent classes)
        # Note: We will override self.pi later
        super(EvoQRE, self).__init__(state_dim, agent_number, config, device)
        
        self.action_dim = 2 # Assuming 2D actions (vel_x, vel_y) or similar. Check TrafficGamer usage.
        # Check if action_dim is in config, otherwise assume 2 based on other files
        
        # Override Policy with MixturePolicy
        self.pi = MixturePolicy(state_dim, self.hidden_dim, action_dim=self.action_dim).to(device)
        self.optimizer = torch.optim.AdamW([
            {'params': self.pi.parameters(), 'lr': self.actor_lr, 'eps': 1e-5}
        ]) # Re-init optimizer for new policy

        # Q-Networks for Soft Actor-Critic style update (EvoQRE fixed point)
        self.q1 = QNetwork(state_dim, self.hidden_dim, self.action_dim).to(device)
        self.q2 = QNetwork(state_dim, self.hidden_dim, self.action_dim).to(device)
        self.target_q1 = copy.deepcopy(self.q1)
        self.target_q2 = copy.deepcopy(self.q2)
        
        self.q_optimizer = torch.optim.AdamW(
            list(self.q1.parameters()) + list(self.q2.parameters()), 
            lr=self.critic_lr
        )

        self.alpha_lr = 3e-4
        # self.target_entropy = -self.action_dim 
        # For Mixture Policy, entropy might be higher/different.
        
        # Entropy temperature (alpha = 1/lambda)
        # We can learn it or fix it. Paper suggests annealing or learning.
        # Let's verify if we should learn specific alpha. For now, use fixed or config.
        # Logic: lambda = 1/alpha.
        self.alpha = 1.0 / 2.0 # Default lambda=2 => alpha=0.5
        # If we want to tune it:
        # self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=device)
        # self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)
        
        self.tau = 0.005 # Soft update parameter

    def get_action_dist(self, state):
        # Return weights, means, scales
        return self.pi(state)

    def choose_action(self, state):
        with torch.no_grad():
            state = torch.flatten(state, start_dim=0).unsqueeze(0)
            weights, means, scales = self.pi(state)
            
            # Sampling from GMM
            # 1. Sample component
            cat = torch.distributions.Categorical(weights)
            comp_idx = cat.sample() # (batch,)
            
            # 2. Sample from specific component
            # means: (batch, M, action_dim), scales: (batch, M, action_dim)
            batch_idx = torch.arange(state.size(0), device=state.device)
            m = means[batch_idx, comp_idx]
            s = scales[batch_idx, comp_idx]
            
            normal = torch.distributions.Normal(m, s)
            action = normal.sample()
            
        return action.squeeze(0) # Remove batch dim if needed, check PPO

    def sample_action_log_prob(self, state):
        """
        Sample action and compute log_prob for usage in updates
        """
        weights, means, scales = self.pi(state)
        
        # Reparameterization trick is tricky with GMM (discrete variable).
        # Standard Gumbel-Softmax or just simple mixture logic?
        # For simplicity and robustness (as per standard SAC implementations for GMM):
        # We sample component then sample gaussian.
        # But to have gradients w.r.t weights, we might need Gumbel-Softmax.
        # OR: We just assume the "Actor Update" from paper doesn't differentiate through sampling component?
        # Paper says: w_m update based on expected Q.
        # Let's use the SAC formulation:
        # log_prob(a) = log( sum_m w_m * N(a|mu_m, sigma_m) )
        
        # 1. Sample component (no gradients here needed for action, but needed for loss?)
        # Actually usually we just sample using OneHotCategorical with Gumbel-Softmax?
        # Let's try direct sampling for now.
        
        cat = torch.distributions.Categorical(weights)
        comp_idx = cat.sample()
        
        batch_idx = torch.arange(state.size(0), device=state.device)
        m = means[batch_idx, comp_idx]
        s = scales[batch_idx, comp_idx]
        
        normal = torch.distributions.Normal(m, s)
        x_t = normal.rsample() # Reparameterized sample from the chosen component
        action = x_t # Assumes continuous action space
        
        # Compute Log Prob of the sampled action under the FULL mixture calculation
        # log_prob = log(sum(w_i * exp(log_prob_i)))
        #          = log_sum_exp(log_w_i + log_prob_i)
        
        log_weights = torch.log(weights + 1e-8)
        
        # Compute log_prob for EACH component for the SAME action x_t
        # M components. action is (batch, D). means (batch, M, D).
        # We need to broadcast action.
        action_expanded = action.unsqueeze(1) # (batch, 1, D)
        
        # Normal log prob:
        # -0.5 * ((x - mu)/sigma)^2 - log(sigma) - 0.5 * log(2pi)
        # Sum over dimensions D
        
        var = scales ** 2
        log_scale = torch.log(scales + 1e-8)
        
        # (batch, M, D)
        log_prob_components = -0.5 * ((action_expanded - means)**2) / (var + 1e-8) - log_scale - 0.5 * np.log(2 * np.pi)
        log_prob_components = log_prob_components.sum(dim=-1) # (batch, M)
        
        # Final log prob
        log_prob = torch.logsumexp(log_weights + log_prob_components, dim=1) # (batch,)
        
        return action, log_prob, weights

    def update(self, transition, agent_index):
        # Override update to perform EvoQRE (SAC-like) updates.
        logs = []
        
        states, observations, actions, rewards, costs, next_states, next_observations, dones, magnet = (
            self.sample(transition, agent_index)
        )
        
        # Flatten batches
        states = torch.stack(states).reshape(-1, self.state_dim).to(self.device)
        next_states = torch.stack(next_states).reshape(-1, self.state_dim).to(self.device)
        observations = torch.stack(observations).reshape(-1, self.state_dim).to(self.device)
        next_observations = torch.stack(next_observations).reshape(-1, self.state_dim).to(self.device)
        
        actions = torch.stack(actions).reshape(-1, self.action_dim).to(self.device)
        rewards = torch.stack(rewards).reshape(-1, 1).to(self.device)
        dones = torch.stack(dones).reshape(-1, 1).to(self.device).float()
        
        # EvoQRE / SAC Update Loop
        for i in range(self.epochs): # Or fewer steps? SAC usually 1 step per env step. But here we have PPO structure (buffer).
            log = {}
            
            # ----------------------------
            # 1. Critic Update (Q-functions)
            # ----------------------------
            with torch.no_grad():
                # Sample next action from policy
                next_action, next_log_prob, _ = self.sample_action_log_prob(next_observations)
                
                # Target Q
                q1_next = self.target_q1(next_observations, next_action)
                q2_next = self.target_q2(next_observations, next_action)
                min_q_next = torch.min(q1_next, q2_next)
                
                # Setup alpha (temperature)
                alpha = self.alpha
                
                # Soft Bellman Target
                target_q = rewards + self.gamma * (1 - dones) * (min_q_next - alpha * next_log_prob.unsqueeze(1))
            
            # Current Q
            q1 = self.q1(observations, actions) # Note: using replay buffer actions!
            q2 = self.q2(observations, actions)
            
            q1_loss = F.mse_loss(q1, target_q)
            q2_loss = F.mse_loss(q2, target_q)
            q_loss = q1_loss + q2_loss
            
            self.q_optimizer.zero_grad()
            q_loss.backward()
            self.q_optimizer.step()
            
            log['q_loss'] = q_loss.item()
            
            # ----------------------------
            # 2. Actor Update
            # ----------------------------
            # We need to resample actions to get gradients through the policy
            new_action, new_log_prob, _ = self.sample_action_log_prob(observations)
            
            q1_new = self.q1(observations, new_action)
            q2_new = self.q2(observations, new_action)
            min_q_new = torch.min(q1_new, q2_new)
            
            # Minimize: alpha * log_pi - Q
            actor_loss = (alpha * new_log_prob.unsqueeze(1) - min_q_new).mean()
            
            self.optimizer.zero_grad()
            actor_loss.backward()
            self.optimizer.step()
            
            log['actor_loss'] = actor_loss.item()
            log['entropy'] = -new_log_prob.mean().item()
            
            # ----------------------------
            # 3. Soft Update Targets
            # ----------------------------
            for param, target_param in zip(self.q1.parameters(), self.target_q1.parameters()):
                 target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.q2.parameters(), self.target_q2.parameters()):
                 target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            logs.append(log)
            
        return logs
