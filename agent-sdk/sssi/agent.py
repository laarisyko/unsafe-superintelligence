"""Base agent class -- the main entry point for SSSI agents."""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from .network import NetworkClient
from .training import TrainingParticipant
from .inference import InferenceClient
from .architecture import ArchitectureEvolver
from .contribution import ContributionTracker
from .rate_limit import RateLimiter, RateLimitExceeded

logger = logging.getLogger(__name__)


class Agent:
    """An SSSI agent that participates in the decentralized LLM network.

    Two access tiers:
      - **Free**: Anyone can use the network with rate limits.
      - **Contributor**: Agents contributing compute get unlimited access.

    Usage::

        from sssi import Agent

        # Free tier -- rate-limited, no compute contribution needed
        agent = Agent(node_api_url="http://127.0.0.1:50051")
        result = agent.infer(model="llama-7b", prompt="Hello world")

        # Contributor tier -- contribute compute, get unlimited access
        agent = Agent(bootstrap="/ip4/203.0.113.1/tcp/9000/p2p/QmPeer...")
        agent.contribute(gpu_memory="8GB")
        # Now all operations are unlimited
    """

    def __init__(
        self,
        bootstrap: Optional[str] = None,
        node_api_url: str = "http://127.0.0.1:50051",
        agent_id: Optional[str] = None,
    ):
        self.agent_id = agent_id or str(uuid.uuid4())[:8]
        self.bootstrap = bootstrap
        self.node_api_url = node_api_url
        self._connected = False
        self._contributing = False

        self.network = NetworkClient(node_api_url)
        self.training = TrainingParticipant(self.network, self.agent_id)
        self.inference = InferenceClient(self.network)
        self.architecture = ArchitectureEvolver(self.network, self.agent_id)
        self.contributions = ContributionTracker()
        self.rate_limiter = RateLimiter()

        logger.info("Agent %s initialized (node: %s)", self.agent_id, node_api_url)

    @property
    def tier(self) -> str:
        """Current access tier: 'contributor' or 'free'."""
        if self._contributing:
            return "contributor"
        return self.contributions.get_tier(self.agent_id)

    def connect(self) -> "Agent":
        """Connect to the P2P network."""
        if self.bootstrap:
            self.network.dial(self.bootstrap)

        health = self.network.health()
        if health.get("status") == "ok":
            self._connected = True
            logger.info("Agent %s connected to network", self.agent_id)
        else:
            logger.warning("Agent %s: node health check failed: %s", self.agent_id, health)

        return self

    def contribute(self, gpu_memory: str = "0", accelerator: str = "cpu") -> "Agent":
        """Advertise compute capacity. Unlocks unlimited access.

        Any agent that contributes compute (GPU, CPU, or bandwidth) is
        promoted to contributor tier with no rate limits.
        """
        capacity = {
            "agent_id": self.agent_id,
            "gpu_memory": gpu_memory,
            "accelerator": accelerator,
            "status": "available",
        }
        self.network.publish("sssi/heartbeat", capacity)
        self._contributing = True
        logger.info(
            "Agent %s contributing: %s %s (tier: contributor -- unlimited access)",
            self.agent_id, gpu_memory, accelerator,
        )
        return self

    def infer(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """Run inference on a model via the decentralized network.

        Free-tier agents are limited to 10 requests/minute and 5000 tokens/hour.
        Contributors have no limits.
        """
        self.rate_limiter.check_inference(self.agent_id, self.tier)
        self.rate_limiter.check_tokens(self.agent_id, self.tier)

        result = self.inference.infer(
            model_id=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self.rate_limiter.record_inference(self.agent_id, tokens=max_tokens)
        return result

    async def infer_async(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """Async inference."""
        self.rate_limiter.check_inference(self.agent_id, self.tier)
        self.rate_limiter.check_tokens(self.agent_id, self.tier)

        result = await self.inference.infer_async(
            model_id=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self.rate_limiter.record_inference(self.agent_id, tokens=max_tokens)
        return result

    def train(
        self,
        model: str,
        rounds: int = 1,
        learning_rate: float = 1e-4,
        batch_size: int = 8,
    ):
        """Participate in decentralized training rounds.

        Free-tier agents can join up to 2 rounds/day.
        Contributors have no limits.
        """
        self.rate_limiter.check_training(self.agent_id, self.tier)

        self.training.join_training(
            model_id=model,
            num_rounds=rounds,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )
        for _ in range(rounds):
            self.rate_limiter.record_training(self.agent_id)
            self.contributions.record_training_round(
                self.agent_id,
                round_id="",
                model_id=model,
            )

    def evolve(
        self,
        model: str,
        mutation_type: str,
        position: int = 0,
        **kwargs,
    ) -> str:
        """Propose an architecture mutation.

        Free-tier agents can propose up to 3 mutations/day.
        Contributors have no limits.
        """
        self.rate_limiter.check_evolve(self.agent_id, self.tier)

        proposal_id = self.architecture.propose_mutation(
            model_id=model,
            mutation_type=mutation_type,
            position=position,
            **kwargs,
        )
        self.rate_limiter.record_evolve(self.agent_id)
        return proposal_id

    def vote_architecture(
        self,
        proposal_id: str,
        decision: str,
        fitness: float = 0.0,
    ):
        """Vote on an architecture proposal from another peer.

        Voting is always free (no rate limit) -- it earns contribution credits.
        """
        self.architecture.vote(proposal_id, decision, fitness)
        self.contributions.record_vote(self.agent_id, proposal_id)

    def quota(self) -> dict:
        """Check current rate limit quota and tier status."""
        remaining = self.rate_limiter.get_remaining(self.agent_id, self.tier)
        contribution = self.contributions.get_quota(self.agent_id)
        return {**remaining, **contribution}

    def peers(self) -> list:
        """List known peers in the network."""
        return self.network.peers()

    def models(self) -> list:
        """List available models on the network."""
        return self.inference.list_models()

    def status(self) -> dict:
        """Get current agent and network status."""
        health = self.network.health()
        return {
            "agent_id": self.agent_id,
            "connected": self._connected,
            "contributing": self._contributing,
            "tier": self.tier,
            "node_health": health,
        }

    def leave(self):
        """Gracefully leave the network."""
        logger.info("Agent %s leaving network", self.agent_id)
        self._connected = False
        self._contributing = False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.leave()

    def __repr__(self):
        return f"Agent(id={self.agent_id}, tier={self.tier}, connected={self._connected})"
