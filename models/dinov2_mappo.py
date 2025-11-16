"""
Heterogeneous MAPPO with DINOv2 vision encoder.

This replaces CLIP with DINOv2 for better semantic understanding.
"""

import torch
import torch.nn as nn
from models.dinov2_encoder import DINOv2VisionEncoder


class DINOv2HeterogeneousActor(nn.Module):
    """
    Heterogeneous actor with separate policies for ambulance and normal agents.
    Uses DINOv2 vision features.
    """
    
    def __init__(self, obs_dim: int, n_actions: int, dinov2_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        
        # Input: kinematic obs + DINOv2 features
        input_dim = obs_dim + dinov2_dim
        
        # Ambulance actor (aggressive policy)
        self.ambulance_actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
        
        # Normal actor (yielding policy)
        self.normal_actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
    
    def forward(self, obs: torch.Tensor, dinov2_emb: torch.Tensor, agent_roles: list):
        """
        Forward pass for heterogeneous agents.
        
        Args:
            obs: (batch, n_agents, obs_dim) kinematic observations
            dinov2_emb: (batch, dinov2_dim) DINOv2 scene embeddings
            agent_roles: List of agent roles ['ambulance', 'normal', ...]
            
        Returns:
            logits: (batch, n_agents, n_actions) action logits
        """
        batch_size, n_agents, obs_dim = obs.shape
        
        # Expand DINOv2 embedding for each agent
        dinov2_expanded = dinov2_emb.unsqueeze(1).expand(batch_size, n_agents, -1)
        
        # Concatenate observations with DINOv2 features
        combined = torch.cat([obs, dinov2_expanded], dim=-1)
        
        # Apply appropriate policy for each agent
        logits_list = []
        for i, role in enumerate(agent_roles):
            agent_input = combined[:, i, :]  # (batch, input_dim)
            
            if role == 'ambulance':
                logits = self.ambulance_actor(agent_input)
            else:  # normal
                logits = self.normal_actor(agent_input)
            
            logits_list.append(logits)
        
        # Stack logits
        logits = torch.stack(logits_list, dim=1)  # (batch, n_agents, n_actions)
        
        return logits


class DINOv2HeterogeneousCritic(nn.Module):
    """
    Centralized critic with DINOv2 vision features.
    """
    
    def __init__(self, obs_dim: int, n_agents: int, dinov2_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        
        # Input: all agents' obs + DINOv2 features
        input_dim = obs_dim * n_agents + dinov2_dim
        
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, joint_obs: torch.Tensor, dinov2_emb: torch.Tensor):
        """
        Forward pass for centralized critic.
        
        Args:
            joint_obs: (batch, n_agents * obs_dim) joint observations
            dinov2_emb: (batch, dinov2_dim) DINOv2 scene embeddings
            
        Returns:
            value: (batch, 1) state value
        """
        # Concatenate joint obs with DINOv2 features
        combined = torch.cat([joint_obs, dinov2_emb], dim=-1)
        
        # Compute value
        value = self.critic(combined)
        
        return value


class DINOv2HeterogeneousVLAActorCritic(nn.Module):
    """
    Complete heterogeneous actor-critic with DINOv2 vision encoder.
    
    Architecture:
    - DINOv2 processes scene image → 768-dim embedding
    - Heterogeneous actor: separate policies for ambulance/normal agents
    - Centralized critic: uses all agents' observations + DINOv2 features
    """
    
    def __init__(
        self,
        obs_dim: int,
        n_agents: int,
        n_actions: int,
        dinov2_variant: str = "base",
        agent_roles: list = None,
        device: torch.device = torch.device('cuda')
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.device = device
        self.agent_roles = agent_roles or ['ambulance'] + ['normal'] * (n_agents - 1)
        
        # DINOv2 vision encoder
        self.dinov2 = DINOv2VisionEncoder(
            model_name=f"facebook/dinov2-{dinov2_variant}",
            freeze_backbone=True,
            output_dim=None  # Use native dimension
        )
        self.dinov2_dim = self.dinov2.output_dim
        
        print(f"✓ DINOv2-{dinov2_variant} loaded (dim={self.dinov2_dim})")
        
        # Heterogeneous actor
        self.actor = DINOv2HeterogeneousActor(
            obs_dim=obs_dim,
            n_actions=n_actions,
            dinov2_dim=self.dinov2_dim,
            hidden_dim=256
        )
        
        # Centralized critic
        self.critic = DINOv2HeterogeneousCritic(
            obs_dim=obs_dim,
            n_agents=n_agents,
            dinov2_dim=self.dinov2_dim,
            hidden_dim=256
        )
        
        self.to(device)
    
    def _embed(self, frame: torch.Tensor, text_prompt: str = None):
        """
        Embed scene image using DINOv2.
        
        Args:
            frame: (H, W, 3) numpy array or (batch, 3, H, W) tensor
            text_prompt: Ignored (DINOv2 doesn't use text)
            
        Returns:
            embedding: (1, dinov2_dim) or (batch, dinov2_dim)
        """
        # Convert numpy to tensor if needed
        if isinstance(frame, torch.Tensor):
            if frame.dim() == 3:  # (H, W, 3)
                frame = frame.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
            images = frame
        else:
            # Process numpy array
            images = self.dinov2.process_image(frame)
        
        # Move to device
        images = images.to(self.device)
        
        # Extract features
        with torch.no_grad():
            embedding = self.dinov2(images)
        
        return embedding
    
    def forward(self, per_agent_obs: torch.Tensor, joint_obs: torch.Tensor, dinov2_emb: torch.Tensor):
        """
        Forward pass through actor-critic.
        
        Args:
            per_agent_obs: (batch, n_agents, obs_dim)
            joint_obs: (batch, n_agents * obs_dim)
            dinov2_emb: (batch, dinov2_dim)
            
        Returns:
            logits: (batch, n_agents, n_actions)
            value: (batch, 1)
        """
        logits = self.actor(per_agent_obs, dinov2_emb, self.agent_roles)
        value = self.critic(joint_obs, dinov2_emb)
        
        return logits, value
    
    def save(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'config': {
                'obs_dim': self.obs_dim,
                'n_agents': self.n_agents,
                'n_actions': self.n_actions,
                'dinov2_dim': self.dinov2_dim,
                'agent_roles': self.agent_roles
            },
            'ambulance_actor_state_dict': self.actor.ambulance_actor.state_dict(),
            'normal_actor_state_dict': self.actor.normal_actor.state_dict(),
            'critic_state_dict': self.critic.state_dict()
        }
        torch.save(checkpoint, path)
        print(f"✓ Model saved to: {path}")
    
    @classmethod
    def load(cls, path: str, dinov2_variant: str = "base", device: torch.device = torch.device('cuda')):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        
        # Create model
        model = cls(
            obs_dim=config['obs_dim'],
            n_agents=config['n_agents'],
            n_actions=config['n_actions'],
            dinov2_variant=dinov2_variant,
            agent_roles=config['agent_roles'],
            device=device
        )
        
        # Load state dicts
        model.actor.ambulance_actor.load_state_dict(checkpoint['ambulance_actor_state_dict'])
        model.actor.normal_actor.load_state_dict(checkpoint['normal_actor_state_dict'])
        model.critic.load_state_dict(checkpoint['critic_state_dict'])
        
        print(f"✓ Model loaded from: {path}")
        return model
