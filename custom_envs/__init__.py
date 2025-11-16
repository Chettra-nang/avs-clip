"""Custom highway-env environments for VLM-MARL project."""
from gymnasium.envs.registration import register

# Register custom multi-agent merge environment
register(
    id='merge-multi-agent-v0',
    entry_point='custom_envs.multi_agent_merge_env:MultiAgentMergeEnv',
)

__all__ = ['MultiAgentMergeEnv']
