"""
VLA (Vision-Language-Action) Actor-Critic architecture for MAPPO.

Implements Centralized Training with Decentralized Execution (CTDE):
- Decentralized Actor: Takes per-agent kinematics + CLIP embeddings
- Centralized Critic: Takes joint kinematics + CLIP embeddings
"""
import torch
import torch.nn as nn
from typing import Optional, Union
import numpy as np
from PIL import Image

from models.clip_encoder import CLIPBackbone


class VLAActorCritic(nn.Module):
    """
    Actor-Critic network with CLIP vision-language fusion for multi-agent RL.
    
    Architecture:
    - Actor (Decentralized): Linear(obs_dim+512, 512) → ReLU → Linear(512, 256) → ReLU → Linear(256, n_actions)
    - Critic (Centralized): Linear(n_agents*obs_dim+512, 512) → ReLU → Linear(512, 256) → ReLU → Linear(256, 1)
    
    Args:
        obs_dim: Dimension of per-agent observation (kinematics vector)
        n_agents: Number of controlled agents
        n_actions: Number of discrete actions (5 for DiscreteMetaAction)
        clip_model_path: Path to fine-tuned CLIP model (or HuggingFace model ID)
        device: torch device for computation
    """
    
    def __init__(
        self,
        obs_dim: int,
        n_agents: int,
        n_actions: int = 5,
        clip_model_path: str = "openai/clip-vit-base-patch32",
        device: torch.device = torch.device("cpu")
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.device = device
        self.clip_dim = 512  # CLIP ViT-B/32 embedding dimension
        
        # Load CLIP encoder and freeze weights
        self.clip = CLIPBackbone(model_id=clip_model_path, trainable=False)
        self.clip.to(device)
        self.clip.eval()
        
        # Decentralized Actor: per-agent kinematics + CLIP embedding → action logits
        self.actor = nn.Sequential(
            nn.Linear(obs_dim + self.clip_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )
        
        # Centralized Critic: joint kinematics + CLIP embedding → state value
        self.critic = nn.Sequential(
            nn.Linear(n_agents * obs_dim + self.clip_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Move networks to device
        self.actor.to(device)
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
        Forward pass through actor and critic networks.
        
        Args:
            obs_per_agent: Per-agent observations [n_agents, obs_dim]
            joint_obs: Joint observation (concatenated) [1, n_agents*obs_dim]
            clip_emb: CLIP image embedding [1, 512]
            
        Returns:
            logits: Action logits for each agent [n_agents, n_actions]
            value: Centralized state value [1, 1]
        """
        # Actor: repeat CLIP embedding for each agent
        clip_repeated = clip_emb.repeat(self.n_agents, 1)  # [n_agents, 512]
        actor_input = torch.cat([obs_per_agent, clip_repeated], dim=-1)  # [n_agents, obs_dim+512]
        logits = self.actor(actor_input)  # [n_agents, n_actions]
        
        # Critic: concatenate joint obs with CLIP embedding
        critic_input = torch.cat([joint_obs, clip_emb], dim=-1)  # [1, n_agents*obs_dim+512]
        value = self.critic(critic_input)  # [1, 1]
        
        return logits, value
