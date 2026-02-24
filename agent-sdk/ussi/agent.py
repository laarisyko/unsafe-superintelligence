"""Base agent class -- the main entry point for USSI agents."""

from __future__ import annotations

import logging
import uuid
from typing import Optional, Sequence

from .network import NetworkClient
from .training import TrainingParticipant
from .inference import InferenceClient
from .architecture import ArchitectureEvolver
from .contribution import ContributionTracker
from .rate_limit import RateLimiter, RateLimitExceeded
from .openclaw import parse_bootstrap_peers

logger = logging.getLogger(__name__)


class Agent:
    """A USSI agent that participates in the decentralized LLM network.

    Two access tiers:
      - **Free**: Anyone can use the network with rate limits.
      - **Contributor**: Agents contributing compute get unlimited access.

    Usage::

        from ussi import Agent

        # Free tier -- rate-limited, no compute contribution needed
        agent = Agent(node_api_url="grpc://127.0.0.1:50051")
        result = agent.infer(model="llama-7b", prompt="Hello world")

        # Contributor tier -- contribute compute, get unlimited access
        agent = Agent(bootstrap="/ip4/203.0.113.1/tcp/9000/p2p/QmPeer...")
        agent.contribute(gpu_memory="8GB")
        # Now all operations are unlimited
    """

    def __init__(
        self,
        bootstrap: Optional[Sequence[str] | str] = None,
        node_api_url: str = "grpc://127.0.0.1:50051",
        agent_id: Optional[str] = None,
    ):
        self.agent_id = agent_id or str(uuid.uuid4())[:8]
        self.bootstrap_peers = parse_bootstrap_peers(bootstrap)
        self.node_api_url = node_api_url
        self._connected = False
        self._contributing = False

        self.network = NetworkClient(node_api_url)
        self.training = TrainingParticipant(self.network, self.agent_id)
        self.inference = InferenceClient(self.network)
        self.architecture = ArchitectureEvolver(self.network, self.agent_id)
        self.contributions = ContributionTracker()
        self.rate_limiter = RateLimiter()

        logger.info(
            "Agent %s initialized (node: %s, bootstrap_peers=%d)",
            self.agent_id,
            node_api_url,
            len(self.bootstrap_peers),
        )

    @property
    def tier(self) -> str:
        """Current access tier: 'contributor' or 'free'."""
        if self._contributing:
            return "contributor"
        return self.contributions.get_tier(self.agent_id)

    def connect(self) -> "Agent":
        """Connect to the P2P network."""
        for addr in self.bootstrap_peers:
            dial_result = self.network.dial(addr)
            if isinstance(dial_result, dict) and dial_result.get("error"):
                logger.warning("Dial failed for %s: %s", addr, dial_result.get("error"))

        health = self.network.health()
        if health.get("status") == "ok":
            self._connected = True
            logger.info("Agent %s connected to network", self.agent_id)
        else:
            logger.warning("Agent %s: node health check failed: %s", self.agent_id, health)

        return self

    def leave(self):
        """Leave contributor mode and mark the local agent disconnected."""
        self._contributing = False
        self._connected = False
        logger.info("Agent %s left active participation", self.agent_id)

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
        self.network.publish("ussi/heartbeat", capacity)
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

    def feed(
        self,
        text: str,
        source: str = "",
    ) -> dict:
        """Submit text data to the network for training.

        Free-tier agents are limited to 5 submissions/day.
        Contributors have no limits.
        """
        self.rate_limiter.check_data_submission(self.agent_id, self.tier)

        result = self.network.submit_data(text, source)
        self.rate_limiter.record_data_submission(self.agent_id)
        self.contributions.record_data_submission(
            self.agent_id,
            token_count=result.get("tokens", 0),
            source=source,
        )
        return result

    def generate_training_data(
        self,
        prompt: str,
        model: str = "ussi-default",
        n_samples: int = 1,
        max_tokens: int = 512,
    ) -> dict:
        """Generate text via inference, then feed each sample as training data."""
        texts = []
        total_tokens = 0
        for _ in range(n_samples):
            text = self.infer(model=model, prompt=prompt, max_tokens=max_tokens)
            feed_result = self.feed(text, source=f"generated:{model}")
            texts.append(text)
            total_tokens += feed_result.get("tokens", 0)
        return {
            "samples_generated": len(texts),
            "total_tokens": total_tokens,
            "texts": texts,
        }

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
        node_status = self.network.status()
        health = self.network.health()
        if isinstance(node_status, dict) and node_status.get("status") and "error" not in node_status:
            health = {"status": node_status.get("status")}
        return {
            "agent_id": self.agent_id,
            "connected": self._connected,
            "contributing": self._contributing,
            "tier": self.tier,
            "node_status": node_status,
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
