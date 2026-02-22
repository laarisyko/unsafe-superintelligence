"""Synthetic data generation using SOTA teacher models.

Uses SOTA models (Claude, GPT-4) to generate high-quality training text
on diverse topics. Each peer generates data locally using whatever API
keys they have, then trains on it and shares gradients.

The network gets smarter even if individual peers have different API access.
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Set

from ..teacher import TeacherConfig, TeacherModel

logger = logging.getLogger(__name__)


# Diverse topic registry for synthetic data generation.
TOPIC_REGISTRY: List[str] = [
    # Science & Technology
    "quantum mechanics and wave-particle duality",
    "machine learning and neural network architectures",
    "climate science and atmospheric chemistry",
    "molecular biology and gene expression",
    "astrophysics and stellar evolution",
    # History & Culture
    "the Renaissance and its impact on European art",
    "ancient Roman engineering and infrastructure",
    "the Silk Road and cross-cultural exchange",
    "the Industrial Revolution and social change",
    "ancient Egyptian mathematics and astronomy",
    # Mathematics & Logic
    "number theory and prime number distribution",
    "graph theory and network analysis",
    "probability theory and Bayesian inference",
    "topology and geometric transformations",
    # Programming & Software
    "distributed systems and consensus algorithms",
    "functional programming and type theory",
    "database design and query optimization",
    "cryptography and secure communication protocols",
    # Creative & Reasoning
    "philosophical arguments about consciousness",
    "economic theories of market equilibrium",
    "narrative structure and storytelling techniques",
    "ethical frameworks for artificial intelligence",
    "cognitive psychology and decision-making biases",
    # Practical Knowledge
    "nutrition science and metabolic pathways",
    "environmental conservation strategies",
    "urban planning and sustainable architecture",
    "music theory and harmonic analysis",
    "linguistics and language evolution",
    # Dialogue & Instruction
    "a tutorial explaining how photosynthesis works",
    "a debate about renewable vs nuclear energy",
]


def _ngram_set(text: str, n: int = 3) -> Set[str]:
    """Extract character n-grams from text."""
    text = text.lower().strip()
    return {text[i:i + n] for i in range(len(text) - n + 1)} if len(text) >= n else {text}


def ngram_overlap(text_a: str, text_b: str, n: int = 3) -> float:
    """Compute n-gram overlap ratio between two texts.

    Returns a float in [0, 1] where 1 means identical n-gram sets.
    """
    set_a = _ngram_set(text_a, n)
    set_b = _ngram_set(text_b, n)
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""

    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    topics: List[str] = field(default_factory=list)  # empty = use TOPIC_REGISTRY
    samples_per_topic: int = 5
    min_length: int = 200
    max_length: int = 1000
    dedup_threshold: float = 0.8  # reject if >80% n-gram overlap with existing


class SyntheticDataGenerator:
    """Generates diverse training text using a SOTA teacher model."""

    def __init__(self, config: SyntheticConfig, teacher: TeacherModel):
        self.config = config
        self.teacher = teacher
        self.topics = config.topics if config.topics else list(TOPIC_REGISTRY)
        self._generated_texts: List[str] = []
        self._topic_idx = 0

    def _next_topic(self) -> str:
        """Cycle through topics."""
        topic = self.topics[self._topic_idx % len(self.topics)]
        self._topic_idx += 1
        return topic

    def _make_prompt(self, topic: str) -> str:
        """Create a generation prompt for a topic."""
        target_words = (self.config.min_length + self.config.max_length) // 2
        return (
            f"Write a {target_words}-word passage about {topic}. "
            f"Be detailed and informative. Write clearly and use specific examples."
        )

    def _is_duplicate(self, text: str) -> bool:
        """Check if text is too similar to already-generated texts."""
        for existing in self._generated_texts:
            if ngram_overlap(text, existing) > self.config.dedup_threshold:
                return True
        return False

    def generate_for_topic(self, topic: str, n: int = 5) -> List[str]:
        """Generate n text samples for a specific topic.

        Args:
            topic: The topic to generate about.
            n: Number of samples to generate.

        Returns:
            List of generated text strings (after dedup filtering).
        """
        results = []
        prompt = self._make_prompt(topic)
        attempts = 0
        max_attempts = n * 3  # Allow retries for dedup.

        while len(results) < n and attempts < max_attempts:
            attempts += 1
            try:
                text = self.teacher.generate(prompt, max_tokens=self.config.max_length * 2)
            except Exception as e:
                logger.warning("Teacher generation failed for topic '%s': %s", topic, e)
                continue

            if not text or len(text.strip()) < self.config.min_length // 2:
                logger.debug("Generated text too short, retrying")
                continue

            if self._is_duplicate(text):
                logger.debug("Generated text too similar to existing, retrying")
                continue

            self._generated_texts.append(text)
            results.append(text)

        logger.info(
            "Generated %d/%d samples for topic '%s' (%d attempts)",
            len(results), n, topic[:50], attempts,
        )
        return results

    def generate_batch(self, n: int = 10) -> List[str]:
        """Generate n text samples across diverse topics.

        Cycles through topics to ensure diversity.

        Args:
            n: Total number of samples to generate.

        Returns:
            List of generated text strings.
        """
        results = []
        remaining = n

        while remaining > 0:
            topic = self._next_topic()
            batch_size = min(remaining, self.config.samples_per_topic)
            texts = self.generate_for_topic(topic, n=batch_size)
            results.extend(texts)
            remaining -= len(texts)

            if not texts:
                # Topic produced nothing, skip ahead.
                remaining -= batch_size

        return results

    def feed_to_pipeline(self, pipeline, n: int = 50):
        """Generate text and feed directly into a TextDataPipeline.

        Args:
            pipeline: A TextDataPipeline instance.
            n: Number of samples to generate.
        """
        texts = self.generate_batch(n=n)
        fed = 0
        for text in texts:
            if text.strip():
                pipeline.load_text(text)
                fed += 1

        logger.info(
            "Fed %d synthetic samples into pipeline (total tokens: %d)",
            fed, pipeline.total_tokens,
        )
        return fed
