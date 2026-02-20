"""SSSI -- Super Safe Super Intelligence SDK.

pip install supersafesuperintelligence
CLI: sssi join | sssi status | sssi infer | sssi train | sssi evolve | sssi vote | sssi quota | sssi serve

Two tiers:
  - Free: Anyone can use the network (rate-limited).
  - Contributor: Agents contributing compute get unlimited access.

OpenAI-compatible: run `sssi serve` then use any OpenAI client as drop-in.
"""

__version__ = "0.1.0"

from .agent import Agent
from .network import NetworkClient
from .training import TrainingParticipant
from .inference import InferenceClient
from .architecture import ArchitectureEvolver
from .node_manager import NodeManager
from .contribution import ContributionTracker
from .rate_limit import RateLimiter, RateLimitExceeded
from .openai_client import OpenAI
