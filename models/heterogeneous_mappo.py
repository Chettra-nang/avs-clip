"""
Heterogeneous Actor-Critic architecture for MAPPO with role-specific policies.

Implements role-based policy networks for ambulance priority scenarios:
- Ambulance Actor: Aggressive policy for emergency vehicle (Agent 0)
- Normal Actor: Yielding policy for cooperative vehicles (Agents 1-3)
- Centralized Critic: Shared value function for cooperative credit assignment
"""
import torch
import torch.nn as nn
from typing import List, Optional, Union
import numpy as np
from PIL import Image

from models.clip_encoder import CLIPBackbone


class HeterogeneousActor(nn.Module):
    """
    Heterogeneous actor with role-specific sub-networks.
    
    Architecture:
        - ambulance_actor: MLP for Agent 0 (ambulance)
        - normal_actor: MLP for Agents 1-3 (normal vehicles)
    
    Both sub-networks share the same architecture but have independent weights,
    allowing each role to learn specialized behaviors.
    
    Args:
        obs_dim: Dimension of per-agent observation (64 for heterogeneous mode with role encoding)
        n_actions: Number of discrete actions (5 for DiscreteMetaAction)
        clip_dim: Dimension of CLIP embeddings (512 for ViT-B/32)
    """
    
    def __init__(self, obs_dim: int = 64, n_actions: int = 5, clip_dim: int = 512):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.clip_dim = clip_dim
        
        # Ambulance actor (Agent 0) - learns aggressive priority behavior
        self.ambulance_actor = nn.Sequential(
            nn.Linear(obs_dim + clip_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )
        
        # Normal actor (Agents 1-3) - learns yielding behavior
        self.normal_actor = nn.Sequential(
            nn.Linear(obs_dim + clip_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )
    
    def forward(
        self,
        obs_per_agent: torch.Tensor,
        clip_emb: torch.Tensor,
        agent_roles: List[str]
    ) -> torch.Tensor:
        """
        Forward pass with role-based routing.
        
        Args:
            obs_per_agent: Per-agent observations [n_agents, obs_dim]
            clip_emb: CLIP image embedding [1, clip_dim]
            agent_roles: List of role strings ['ambulance', 'normal', 'normal', 'normal']
            
        Returns:
            logits: Action logits for each agent [n_agents, n_actions]
        """
        n_agents = obs_per_agent.shape[0]
        
        # Repeat CLIP embedding for each agent
        clip_repeated = clip_emb.repeat(n_agents, 1)  # [n_agents, clip_dim]
        
        # Concatenate observations with CLIP embeddings
        actor_input = torch.cat([obs_per_agent, clip_repeated], dim=-1)  # [n_agents, obs_dim+clip_dim]
        
        # Route each agent to its role-specific sub-network
        logits = []
        for i, role in enumerate(agent_roles):
            if role == 'ambulance':
                logits.append(self.ambulance_actor(actor_input[i:i+1]))  # [1, n_actions]
            else:  # role == 'normal'
                logits.append(self.normal_actor(actor_input[i:i+1]))  # [1, n_actions]
        
        return torch.cat(logits, dim=0)  # [n_agents, n_actions]


class HeterogeneousVLAActorCritic(nn.Module):
    """
    Heterogeneous Actor-Critic with CLIP vision-language fusion for multi-agent RL.
    
    Extends VLAActorCritic with role-specific actor networks while maintaining
    a centralized critic for cooperative credit assignment.
    
    Architecture:
    - Heterogeneous Actor: Routes agents to role-specific sub-networks
    - Centralized Critic: Shared value function (unchanged from VLAActorCritic)
    
    Args:
        obs_dim: Dimension of per-agent observation (64 for heterogeneous mode)
        n_agents: Number of controlled agents (4 for ambulance scenario)
        n_actions: Number of discrete actions (5 for DiscreteMetaAction)
        clip_model_path: Path to fine-tuned CLIP model (or HuggingFace model ID)
        agent_roles: List of role strings ['ambulance', 'normal', 'normal', 'normal']
        device: torch device for computation
    """
    
    def __init__(
        self,
        obs_dim: int,
        n_agents: int,
        n_actions: int = 5,
        clip_model_path: str = "openai/clip-vit-base-patch32",
        agent_roles: List[str] = None,
        device: torch.device = torch.device("cpu")
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.device = device
        self.clip_dim = 512  # CLIP ViT-B/32 embedding dimension
        
        # Validate agent_roles
        if agent_roles is None:
            agent_roles = ['ambulance', 'normal', 'normal', 'normal']
        if len(agent_roles) != n_agents:
            raise ValueError(f"agent_roles length ({len(agent_roles)}) must match n_agents ({n_agents})")
        if agent_roles[0] != 'ambulance':
            raise ValueError("Agent 0 must be 'ambulance' in heterogeneous mode")
        self.agent_roles = agent_roles
        
        # Load CLIP encoder and freeze weights
        self.clip = CLIPBackbone(model_id=clip_model_path, trainable=False)
        self.clip.to(device)
        self.clip.eval()
        
        # Heterogeneous Actor: role-specific sub-networks
        self.actor = HeterogeneousActor(obs_dim, n_actions, self.clip_dim)
        self.actor.to(device)
        
        # Centralized Critic: joint kinematics + CLIP embedding → state value
        # (unchanged from VLAActorCritic)
        self.critic = nn.Sequential(
            nn.Linear(n_agents * obs_dim + self.clip_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.critic.to(device)
    
    def _embed(
        self,
        image: Union[np.ndarray, Image.Image],
        text_prompt: Optional[str] = None
    ) -> torch.Tensor:
        """
        Extract CLIP image embedding with optional text prompt.
        
        Args:
            image: RGB image as numpy array [H, W, 3] or PIL Image
            text_prompt: Optional text prompt for text-conditioned embedding
            
        Returns:
            CLIP embedding [1, 512]
        """
        with torch.no_grad():
            # Extract image embedding
            img_emb = self.clip.img_embed(image, self.device)  # [1, 512]
            
            # If text prompt provided, fuse with text embedding
            if text_prompt is not None:
                txt_emb = self.clip.txt_embed([text_prompt], self.device)  # [1, 512]
                # Average fusion (could also use concatenation or attention)
                emb = (img_emb + txt_emb) / 2.0
                # Re-normalize
                emb = emb / emb.norm(dim=-1, keepdim=True)
                return emb
            
            return img_emb
    
    def forward(
        self,
        obs_per_agent: torch.Tensor,
        joint_obs: torch.Tensor,
        clip_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through heterogeneous actor and centralized critic.
        
        Args:
            obs_per_agent: Per-agent observations [n_agents, obs_dim]
            joint_obs: Joint observation (concatenated) [1, n_agents*obs_dim]
            clip_emb: CLIP image embedding [1, 512]
            
        Returns:
            logits: Action logits for each agent [n_agents, n_actions]
            value: Centralized state value [1, 1]
        """
        # Heterogeneous Actor: route agents to role-specific sub-networks
        logits = self.actor(obs_per_agent, clip_emb, self.agent_roles)  # [n_agents, n_actions]
        
        # Centralized Critic: concatenate joint obs with CLIP embedding
        critic_input = torch.cat([joint_obs, clip_emb], dim=-1)  # [1, n_agents*obs_dim+512]
        value = self.critic(critic_input)  # [1, 1]
        
        return logits, value
    
    def save(self, path: str):
        """
        Save heterogeneous model with both sub-networks.
        
        Args:
            path: Path to save model checkpoint
        """
        checkpoint = {
            'config': {
                'obs_dim': self.obs_dim,
                'n_agents': self.n_agents,
                'n_actions': self.n_actions,
                'agent_roles': self.agent_roles,
                'clip_dim': self.clip_dim
            },
            'ambulance_actor_state_dict': self.actor.ambulance_actor.state_dict(),
            'normal_actor_state_dict': self.actor.normal_actor.state_dict(),
            'critic_state_dict': self.critic.state_dict()
        }
        torch.save(checkpoint, path)
    
    @classmethod
    def load(
        cls,
        path: str,
        clip_model_path: str = "openai/clip-vit-base-patch32",
        device: torch.device = torch.device("cpu")
    ):
        """
        Load heterogeneous model from checkpoint.
        
        Args:
            path: Path to model checkpoint
            clip_model_path: Path to CLIP model
            device: torch device for computation
            
        Returns:
            Loaded HeterogeneousVLAActorCritic model
        """
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        
        # Validate obs_dim for heterogeneous mode
        if config['obs_dim'] != 64:
            raise ValueError(
                f"Model obs_dim ({config['obs_dim']}) doesn't match expected 64 for heterogeneous mode. "
                f"Heterogeneous models require obs_dim=64 (8 vehicles × 8 features with role encoding)."
            )
        
        # Create model
        model = cls(
            obs_dim=config['obs_dim'],
            n_agents=config['n_agents'],
            n_actions=config['n_actions'],
            clip_model_path=clip_model_path,
            agent_roles=config['agent_roles'],
            device=device
        )
        
        # Load state dicts
        model.actor.ambulance_actor.load_state_dict(checkpoint['ambulance_actor_state_dict'])
        model.actor.normal_actor.load_state_dict(checkpoint['normal_actor_state_dict'])
        model.critic.load_state_dict(checkpoint['critic_state_dict'])
        
        return model
