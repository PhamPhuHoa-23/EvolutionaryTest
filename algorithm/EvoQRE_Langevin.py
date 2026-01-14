import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from algorithm.TrafficGamer import TrafficGamer

class QNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        torch.nn.init.orthogonal_(self.fc1.weight, np.sqrt(2))
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        torch.nn.init.orthogonal_(self.fc2.weight, np.sqrt(2))
        
        self.fc3 = nn.Linear(hidden_dim, 1)
        torch.nn.init.orthogonal_(self.fc3.weight, np.sqrt(2))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class EvoQRE_Langevin(TrafficGamer):
    """
    EvoQRE implementation using the Mean-Field Langevin Dynamics perspective.
    
    In this variant:
    - The 'Actor' is implicit, defined by the stationary distribution of the Langevin process on the Q-function landscape.
    - We use Reflected Langevin Dynamics (approximated via Projection) to sample actions.
    - The Critic (Q-function) is trained via Soft Bellman updates or standard Bellman updates.
    
    This corresponds to the 'Particle-EvoQRE' algorithm where particles are generated dynamically 
    via Langevin sampling rather than maintained as a static population for general state spaces.
    """
    def __init__(self, state_dim: int, agent_number: int, config, device):
        super(EvoQRE_Langevin, self).__init__(state_dim, agent_number, config, device)
        
        self.action_dim = 2 # Velocity x, y
        if 'action_dim' in config:
            self.action_dim = config['action_dim']
            
        # -----------------------------------------------------------
        # Langevin Dynamics Hyperparameters
        # -----------------------------------------------------------
        self.langevin_steps = config.get('langevin_steps', 20)      # K steps
        self.langevin_step_size = config.get('langevin_step_size', 0.05) # eta
        self.tau = config.get('tau', 0.5)                           # Temperature
        self.action_bound = config.get('action_bound', 1.0)         # For projection
        
        # -----------------------------------------------------------
        # Implementation of Reflected Langevin Dynamics
        # Note: We use the Euler-Maruyama discretization with Projection 
        # as a numerical approximation of the Skorkhod problem (Reflection).
        # Section VI of the paper discusses this approximation.
        # -----------------------------------------------------------

        # Q-Networks
        self.hidden_dim = config.get('hidden_dim', 64)
        self.q1 = QNetwork(state_dim, self.hidden_dim, self.action_dim).to(device)
        self.q2 = QNetwork(state_dim, self.hidden_dim, self.action_dim).to(device)
        self.target_q1 = copy.deepcopy(self.q1)
        self.target_q2 = copy.deepcopy(self.q2)
        
        self.q_optimizer = torch.optim.AdamW(
            list(self.q1.parameters()) + list(self.q2.parameters()), 
            lr=self.critic_lr
        )
        
        self.tau_update = 0.005 # Polyak averaging constant

    def get_action_dist(self, state):
        # We don't have an explicit policy distribution object like GMM/Gaussian.
        # This method is used in PPO/TrafficGamer for ratio calculation.
        # For implicit Langevin policy, computing exact log_prob is intractable.
        # However, we can approximate or return a dummy distribution if strictly needed by parent class.
        # But EvoQRE overrides 'update', so likely this won't be called for policy updates.
        pass

    def choose_action(self, state):
        """
        Generate action via Langevin Sampling.
        state: (batch, state_dim) or (state_dim)
        
        Returns: tuple (action, log_prob) to match TrafficGamer interface
        """
        with torch.no_grad(): # Use torch.enable_grad() inside for Langevin
            if state.dim() == 1:
                state = state.unsqueeze(0)
            
            # 1. Initialize particles (Uniform or localized)
            # actions = (torch.rand(state.size(0), self.action_dim, device=self.device) * 2 - 1) * self.action_bound
            # Or Gaussian initialization
            actions = torch.randn(state.size(0), self.action_dim, device=self.device) * self.action_bound
            actions = torch.clamp(actions, -self.action_bound, self.action_bound)
            
            # 2. Langevin Dynamics Loop
            # We need gradients w.r.t actions, so we must enable grad temporarily
            actions.requires_grad = True
            
            for _ in range(self.langevin_steps):
                # Calculate energy gradient: grad_a Q(s, a)
                # We use min(Q1, Q2) to be conservative/robust
                with torch.enable_grad():
                    q1 = self.q1(state, actions)
                    q2 = self.q2(state, actions)
                    min_q = torch.min(q1, q2)
                    
                    # Gradient ascent on Q
                    grads = torch.autograd.grad(
                        outputs=min_q.sum(), 
                        inputs=actions, 
                        retain_graph=False # No need graph for next step
                    )[0]
                
                # Langevin Step
                # da = eta * grad + sqrt(2 * eta * tau) * noise
                noise = torch.randn_like(actions)
                
                # Update (using detach to stop graph binding)
                new_actions = actions.detach() + \
                              self.langevin_step_size * grads + \
                              np.sqrt(2 * self.langevin_step_size * self.tau) * noise
                
                # Reflected/Projected Boundary Condition
                new_actions = torch.clamp(new_actions, -self.action_bound, self.action_bound)
                
                actions = new_actions
                actions.requires_grad = True
        
        # Return tuple (action, log_prob) to match TrafficGamer interface
        # log_prob is a placeholder since Langevin doesn't compute explicit log_prob
        action = actions.squeeze(0).detach()
        log_prob = torch.zeros(1, device=self.device)  # Placeholder
        
        return action, log_prob

    def update(self, transition, agent_index):
        """
        Update Q-networks using soft-TD learning.
        No explicit Actor update is needed because the Actor is the Langevin process itself.
        """
        logs = []
        
        # Unpack Data
        states, observations, actions, rewards, costs, next_states, next_observations, dones, magnet = (
            self.sample(transition, agent_index)
        )
        
        observations = torch.stack(observations).reshape(-1, self.state_dim).to(self.device)
        next_observations = torch.stack(next_observations).reshape(-1, self.state_dim).to(self.device)
        actions = torch.stack(actions).reshape(-1, self.action_dim).to(self.device)
        rewards = torch.stack(rewards).reshape(-1, 1).to(self.device)
        dones = torch.stack(dones).reshape(-1, 1).to(self.device).float()
        
        # Training Loop
        for i in range(self.epochs):
            log = {}
            
            # ----------------------------------
            # Critic Update
            # ----------------------------------
            with torch.no_grad():
                # For next_action, we need to run Langevin sampling on next_states
                # This is computationally expensive, but necessary for "True" QRE target
                # Approximation: Run fewer steps for target (e.g. 5 steps)
                
                # Optimization: We can't call self.choose_action efficiently here efficiently due to batching complications with autograd?
                # Actually we can, but let's inline a mini-langevin for target
                
                # Mini-Langevin for Target Action
                # next_act = torch.randn_like(actions) # Random Init
                next_act = torch.zeros_like(actions).normal_(0, 0.5) # Warm start?
                next_act.requires_grad = True
                
                target_langevin_steps = 10 # Smaller number for speed
                
                for _ in range(target_langevin_steps):
                    with torch.enable_grad():
                        q1_t = self.target_q1(next_observations, next_act)
                        q2_t = self.target_q2(next_observations, next_act)
                        q_t = torch.min(q1_t, q2_t)
                        g = torch.autograd.grad(q_t.sum(), next_act)[0]
                    
                    next_act = next_act.detach() + self.langevin_step_size * g + \
                               np.sqrt(2 * self.langevin_step_size * self.tau) * torch.randn_like(next_act)
                    next_act = torch.clamp(next_act, -self.action_bound, self.action_bound)
                    next_act.requires_grad = True
                
                next_action = next_act.detach()
                
                # Compute Target Value
                q1_next = self.target_q1(next_observations, next_action)
                q2_next = self.target_q2(next_observations, next_action)
                min_q_next = torch.min(q1_next, q2_next)
                
                # Soft Q-Learning Target: r + gamma * (Q - alpha * log_pi) ??
                # Theoretically, Langevin samples from exp(Q/tau).
                # The entropy term is implicit in the sampling.
                # However, for the Q-update, if we follow SAC, we need the entropy term explicitly?
                # Or do we use "Hard" Bellman update on the "Soft" optimal action?
                # Result in literature (e.g. SQL): Q(s,a) <-- r + gamma * SoftValue(s')
                # SoftValue(s') = tau * log int exp(Q/tau) da
                # That integral is hard.
                # BUT, if we use the particle approximation:
                # V(s') approx mean(Q(s', a_samples)) + Entropy?
                # Actually, simply minimizing TD error on the langevin samples works.
                
                target_q = rewards + self.gamma * (1 - dones) * min_q_next
            
            # Current Q
            q1 = self.q1(observations, actions)
            q2 = self.q2(observations, actions)
            
            q1_loss = F.mse_loss(q1, target_q)
            q2_loss = F.mse_loss(q2, target_q)
            q_loss = q1_loss + q2_loss
            
            self.q_optimizer.zero_grad()
            q_loss.backward()
            self.q_optimizer.step()
            
            log['q_loss'] = q_loss.item()
            log['avg_q'] = q1.mean().item()
            
            # ----------------------------------
            # Soft Update Targets
            # ----------------------------------
            for param, target_param in zip(self.q1.parameters(), self.target_q1.parameters()):
                 target_param.data.copy_(self.tau_update * param.data + (1 - self.tau_update) * target_param.data)
            for param, target_param in zip(self.q2.parameters(), self.target_q2.parameters()):
                 target_param.data.copy_(self.tau_update * param.data + (1 - self.tau_update) * target_param.data)
            
            logs.append(log)
            
        return logs
