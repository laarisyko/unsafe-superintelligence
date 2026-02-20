"""Genesis: tracking the birth and growth of a decentralized AI.

The genesis system makes the emergence of intelligence visible. It tracks:

1. BIRTH: When the model was created (random weights, generation 0)
2. MILESTONES: Automatic detection of learning breakthroughs
   - First non-random output (entropy drop)
   - First real word
   - First coherent phrase
   - First sentence with grammar
   - First paragraph with meaning
3. EVOLUTION: Architecture mutations over time (from the genome system)
4. QUALITY: Sample quality scoring to quantify text improvement
5. TIMELINE: A complete event log from birth to present

The genesis tracker is the "wow" factor. Watching a model go from random
noise to coherent text is like watching a mind being born. Combined with
architecture evolution, the model is literally a living organism that
grows, learns, and adapts through collective intelligence.

Usage:
    genesis = GenesisTracker(model_id="openclaw-v0")
    genesis.record_birth(model_params=38_000_000, genome_hash="abc123")

    # After each training round:
    genesis.record_round(round_id, loss, sample_text, peers=42)

    # After an architecture mutation:
    genesis.record_mutation("add_layer", "Added attention layer", gen=7)

    # Get the timeline:
    for event in genesis.timeline:
        print(f"[{event.age_str}] {event.description}")
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set


class EventType(str, Enum):
    """Types of genesis events."""
    BIRTH = "birth"
    MILESTONE = "milestone"
    ROUND = "round"
    MUTATION = "mutation"
    CHECKPOINT = "checkpoint"
    PEER_MILESTONE = "peer_milestone"


class MilestoneType(str, Enum):
    """Types of learning milestones, roughly in order of emergence."""
    # Output quality milestones.
    FIRST_NONRANDOM = "first_nonrandom"        # Entropy drops below random threshold
    FIRST_REAL_WORD = "first_real_word"          # Output contains a real English word
    FIRST_REPEATED_WORD = "first_repeated_word"  # Same word appears correctly multiple times
    FIRST_WORD_PAIR = "first_word_pair"          # Two real words adjacent
    FIRST_PHRASE = "first_phrase"                # 3+ real words in sequence
    FIRST_PUNCTUATION = "first_punctuation"      # Uses punctuation correctly
    FIRST_SENTENCE = "first_sentence"            # Complete sentence with structure
    FIRST_PARAGRAPH = "first_paragraph"          # Multiple related sentences
    FIRST_COHERENT = "first_coherent"            # Passes basic coherence check

    # Loss milestones.
    LOSS_BELOW_5 = "loss_below_5"
    LOSS_BELOW_4 = "loss_below_4"
    LOSS_BELOW_3 = "loss_below_3"
    LOSS_BELOW_2 = "loss_below_2"
    LOSS_BELOW_1 = "loss_below_1"

    # Scale milestones.
    ROUNDS_10 = "rounds_10"
    ROUNDS_100 = "rounds_100"
    ROUNDS_1000 = "rounds_1000"
    ROUNDS_10000 = "rounds_10000"
    TOKENS_1M = "tokens_1m"
    TOKENS_10M = "tokens_10m"
    TOKENS_100M = "tokens_100m"
    TOKENS_1B = "tokens_1b"

    # Peer milestones.
    PEERS_10 = "peers_10"
    PEERS_100 = "peers_100"
    PEERS_1000 = "peers_1000"
    PEERS_10000 = "peers_10000"

    # Evolution milestones.
    FIRST_MUTATION = "first_mutation"
    GENERATION_10 = "generation_10"
    GENERATION_100 = "generation_100"


# Human-readable milestone descriptions.
MILESTONE_DESCRIPTIONS = {
    MilestoneType.FIRST_NONRANDOM: "First sign of learning: output is no longer random noise",
    MilestoneType.FIRST_REAL_WORD: "First real word appeared in generated text!",
    MilestoneType.FIRST_REPEATED_WORD: "Model learned to use the same word consistently",
    MilestoneType.FIRST_WORD_PAIR: "Two real words appeared side by side",
    MilestoneType.FIRST_PHRASE: "First recognizable phrase (3+ words)",
    MilestoneType.FIRST_PUNCTUATION: "Model learned punctuation",
    MilestoneType.FIRST_SENTENCE: "First complete sentence with grammar!",
    MilestoneType.FIRST_PARAGRAPH: "First multi-sentence paragraph",
    MilestoneType.FIRST_COHERENT: "Text passes basic coherence check",
    MilestoneType.LOSS_BELOW_5: "Training loss dropped below 5.0",
    MilestoneType.LOSS_BELOW_4: "Training loss dropped below 4.0",
    MilestoneType.LOSS_BELOW_3: "Training loss dropped below 3.0",
    MilestoneType.LOSS_BELOW_2: "Training loss dropped below 2.0",
    MilestoneType.LOSS_BELOW_1: "Training loss dropped below 1.0",
    MilestoneType.ROUNDS_10: "10 training rounds completed",
    MilestoneType.ROUNDS_100: "100 training rounds completed",
    MilestoneType.ROUNDS_1000: "1,000 training rounds completed",
    MilestoneType.ROUNDS_10000: "10,000 training rounds completed!",
    MilestoneType.TOKENS_1M: "1 million tokens processed",
    MilestoneType.TOKENS_10M: "10 million tokens processed",
    MilestoneType.TOKENS_100M: "100 million tokens processed",
    MilestoneType.TOKENS_1B: "1 billion tokens processed!",
    MilestoneType.PEERS_10: "10 peers training together",
    MilestoneType.PEERS_100: "100 peers reached!",
    MilestoneType.PEERS_1000: "1,000 peers — a movement!",
    MilestoneType.PEERS_10000: "10,000 peers — unstoppable!",
    MilestoneType.FIRST_MUTATION: "First architecture mutation: the model evolved!",
    MilestoneType.GENERATION_10: "Architecture generation 10: evolving fast",
    MilestoneType.GENERATION_100: "Architecture generation 100: a new species",
}

# Shareability emoji (used in shareable strings, not in code output).
MILESTONE_EMOJI = {
    MilestoneType.FIRST_NONRANDOM: "🌱",
    MilestoneType.FIRST_REAL_WORD: "📝",
    MilestoneType.FIRST_REPEATED_WORD: "🔁",
    MilestoneType.FIRST_WORD_PAIR: "🤝",
    MilestoneType.FIRST_PHRASE: "💬",
    MilestoneType.FIRST_PUNCTUATION: "✏️",
    MilestoneType.FIRST_SENTENCE: "📖",
    MilestoneType.FIRST_PARAGRAPH: "📚",
    MilestoneType.FIRST_COHERENT: "🧠",
    MilestoneType.FIRST_MUTATION: "🧬",
    MilestoneType.PEERS_10: "👥",
    MilestoneType.PEERS_100: "🏘️",
    MilestoneType.PEERS_1000: "🏙️",
    MilestoneType.PEERS_10000: "🌍",
}


@dataclass
class GenesisEvent:
    """A single event in the model's life."""
    event_type: EventType
    timestamp: float  # Unix timestamp.
    description: str
    # Context data.
    round_id: str = ""
    loss: float = 0.0
    sample_text: str = ""
    peers: int = 0
    tokens_total: int = 0
    generation: int = 0
    mutation_type: str = ""
    milestone: Optional[MilestoneType] = None
    # Quality scores.
    quality_score: float = 0.0

    @property
    def age_secs(self) -> float:
        """Seconds since the tracker was created (requires birth_time context)."""
        return 0.0  # Set by the tracker.

    @property
    def age_str(self) -> str:
        """Human-readable age string."""
        return ""  # Set by the tracker.

    def to_dict(self) -> dict:
        d = {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "description": self.description,
        }
        if self.round_id:
            d["round_id"] = self.round_id
        if self.loss > 0:
            d["loss"] = round(self.loss, 4)
        if self.sample_text:
            d["sample_text"] = self.sample_text[:200]
        if self.peers > 0:
            d["peers"] = self.peers
        if self.tokens_total > 0:
            d["tokens_total"] = self.tokens_total
        if self.generation > 0:
            d["generation"] = self.generation
        if self.mutation_type:
            d["mutation_type"] = self.mutation_type
        if self.milestone:
            d["milestone"] = self.milestone.value
        if self.quality_score > 0:
            d["quality_score"] = round(self.quality_score, 3)
        return d


@dataclass
class QualityReport:
    """Quality assessment of a generated text sample."""
    text: str
    score: float = 0.0  # 0.0 (gibberish) to 1.0 (human-quality)

    # Component scores (each 0-1).
    entropy_score: float = 0.0       # Low entropy = more structured
    real_word_ratio: float = 0.0     # Fraction of tokens that are real words
    word_diversity: float = 0.0      # Unique words / total words
    sentence_score: float = 0.0      # Has sentence-like structure
    punctuation_score: float = 0.0   # Uses punctuation appropriately
    coherence_score: float = 0.0     # Adjacent words form plausible pairs

    # Detected features.
    real_words_found: List[str] = field(default_factory=list)
    longest_real_phrase: str = ""
    has_sentence: bool = False
    has_paragraph: bool = False

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 3),
            "entropy_score": round(self.entropy_score, 3),
            "real_word_ratio": round(self.real_word_ratio, 3),
            "word_diversity": round(self.word_diversity, 3),
            "sentence_score": round(self.sentence_score, 3),
            "punctuation_score": round(self.punctuation_score, 3),
            "coherence_score": round(self.coherence_score, 3),
            "real_words_found": self.real_words_found[:20],
            "longest_real_phrase": self.longest_real_phrase[:100],
            "has_sentence": self.has_sentence,
            "has_paragraph": self.has_paragraph,
        }


# Common English words for detection (kept small for speed).
_COMMON_WORDS: Set[str] = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their",
    "what", "so", "up", "out", "if", "about", "who", "get", "which", "go",
    "me", "when", "make", "can", "like", "time", "no", "just", "him",
    "know", "take", "people", "into", "year", "your", "good", "some",
    "could", "them", "see", "other", "than", "then", "now", "look",
    "only", "come", "its", "over", "think", "also", "back", "after",
    "use", "two", "how", "our", "work", "first", "well", "way", "even",
    "new", "want", "because", "any", "these", "give", "day", "most", "us",
    "is", "are", "was", "were", "been", "has", "had", "did", "does",
    "am", "may", "might", "shall", "should", "must", "need",
    "very", "much", "more", "still", "here", "where", "why", "how",
    "said", "each", "tell", "does", "set", "three", "own", "hand",
    "high", "keep", "last", "long", "great", "old", "big", "small",
    "man", "woman", "child", "world", "life", "head", "end", "home",
    "water", "house", "night", "every", "found", "upon", "down",
    "never", "little", "too", "part", "made", "before",
    "once", "nothing", "thought", "white", "through", "eyes",
    "went", "let", "right", "always", "began",
}


def assess_quality(text: str) -> QualityReport:
    """Assess the quality of generated text.

    Scores from 0.0 (random noise) to 1.0 (human-quality text).
    Designed to track the emergence of language ability over training.
    """
    report = QualityReport(text=text)

    if not text or len(text) < 3:
        return report

    # Clean text for analysis.
    clean = text.strip()

    # 1. Character-level entropy.
    report.entropy_score = _score_entropy(clean)

    # 2. Real word detection.
    words = re.findall(r"[a-zA-Z]+", clean.lower())
    if words:
        real = [w for w in words if w in _COMMON_WORDS]
        report.real_words_found = list(dict.fromkeys(real))  # Unique, preserve order.
        report.real_word_ratio = len(real) / len(words)

        # Word diversity: unique/total (penalize excessive repetition).
        unique = len(set(words))
        report.word_diversity = min(1.0, unique / len(words))

        # Longest consecutive real-word phrase.
        report.longest_real_phrase = _longest_real_phrase(words)
    else:
        report.real_word_ratio = 0.0
        report.word_diversity = 0.0

    # 3. Sentence structure.
    report.sentence_score, report.has_sentence = _score_sentences(clean)

    # 4. Punctuation usage.
    report.punctuation_score = _score_punctuation(clean)

    # 5. Coherence (adjacent word pairs).
    report.coherence_score = _score_coherence(words) if words else 0.0

    # 6. Paragraph detection.
    report.has_paragraph = bool(re.search(r"\.\s+[A-Z].*\.\s+[A-Z]", clean))

    # Weighted composite score.
    report.score = (
        0.10 * report.entropy_score
        + 0.25 * report.real_word_ratio
        + 0.10 * report.word_diversity
        + 0.25 * report.sentence_score
        + 0.10 * report.punctuation_score
        + 0.20 * report.coherence_score
    )

    return report


def _score_entropy(text: str) -> float:
    """Score based on character entropy.

    Random text has high entropy (~log2(256) = 8 bits).
    English text has ~4-5 bits. Repeated patterns have even less.
    We want to reward the "sweet spot" of structured but varied text.
    """
    if len(text) < 2:
        return 0.0

    counts = Counter(text)
    total = len(text)
    entropy = -sum(
        (c / total) * math.log2(c / total)
        for c in counts.values()
        if c > 0
    )

    # Random bytes: ~8 bits. English text: ~4-5 bits.
    # Score: 0 at entropy > 7 (too random), 1.0 at entropy ~4.5.
    if entropy > 7.0:
        return 0.0
    if entropy > 5.5:
        return (7.0 - entropy) / 1.5 * 0.5  # Partial credit.
    if entropy > 3.0:
        return 1.0  # Sweet spot.
    if entropy > 1.0:
        return 0.5 + (entropy - 1.0) / 4.0  # Too repetitive, partial credit.
    return 0.3  # Very low entropy (e.g. "aaaaaaa").


def _longest_real_phrase(words: List[str]) -> str:
    """Find the longest consecutive sequence of real English words."""
    best = []
    current = []
    for w in words:
        if w in _COMMON_WORDS:
            current.append(w)
            if len(current) > len(best):
                best = list(current)
        else:
            current = []
    return " ".join(best)


def _score_sentences(text: str) -> tuple:
    """Score sentence structure. Returns (score, has_sentence)."""
    # Look for patterns like: "Capital word ... period"
    sentences = re.findall(r"[A-Z][a-z]+(?:\s+[a-z]+){2,}[.!?]", text)
    if sentences:
        # Score based on how sentence-like the overall text is.
        sentence_chars = sum(len(s) for s in sentences)
        coverage = min(1.0, sentence_chars / max(len(text), 1))
        return coverage, True
    return 0.0, False


def _score_punctuation(text: str) -> float:
    """Score punctuation usage."""
    if len(text) < 10:
        return 0.0

    # Count meaningful punctuation.
    punct_chars = sum(1 for c in text if c in ".,;:!?'-\"()")
    alpha_chars = sum(1 for c in text if c.isalpha())

    if alpha_chars == 0:
        return 0.0

    ratio = punct_chars / alpha_chars

    # English text typically has ~5-15% punctuation relative to alpha chars.
    if 0.02 < ratio < 0.20:
        return 1.0
    if ratio > 0:
        return 0.5
    return 0.0


def _score_coherence(words: List[str]) -> float:
    """Score coherence based on adjacent word pairs.

    Uses a small set of common English bigrams as a proxy.
    """
    if len(words) < 2:
        return 0.0

    # Common English word pairs.
    common_pairs = {
        ("the", "man"), ("the", "woman"), ("the", "world"), ("the", "end"),
        ("the", "old"), ("the", "great"), ("the", "first"), ("the", "last"),
        ("the", "same"), ("the", "other"), ("the", "most"), ("the", "time"),
        ("the", "way"), ("the", "day"), ("the", "house"), ("the", "head"),
        ("of", "the"), ("in", "the"), ("to", "the"), ("on", "the"),
        ("at", "the"), ("by", "the"), ("for", "the"), ("from", "the"),
        ("with", "the"), ("and", "the"), ("is", "the"), ("was", "the"),
        ("it", "was"), ("it", "is"), ("he", "was"), ("he", "had"),
        ("she", "was"), ("she", "had"), ("they", "were"), ("there", "was"),
        ("there", "is"), ("i", "was"), ("i", "had"), ("i", "am"),
        ("do", "not"), ("did", "not"), ("could", "not"), ("would", "not"),
        ("has", "been"), ("have", "been"), ("had", "been"),
        ("to", "be"), ("to", "have"), ("to", "do"), ("to", "make"),
        ("in", "a"), ("on", "a"), ("with", "a"), ("for", "a"),
        ("and", "a"), ("is", "a"), ("was", "a"), ("as", "a"),
    }

    # Count how many adjacent pairs are common.
    pair_count = 0
    for i in range(len(words) - 1):
        pair = (words[i], words[i + 1])
        if pair in common_pairs:
            pair_count += 1

    total_pairs = len(words) - 1
    return min(1.0, pair_count / max(total_pairs * 0.15, 1))


class GenesisTracker:
    """Tracks the birth and growth of a decentralized AI model.

    This is the "digital genesis" — a complete record of the model's
    life from random noise to (hopefully) intelligence. Every milestone,
    every mutation, every training round is recorded.
    """

    def __init__(self, model_id: str = "openclaw"):
        self.model_id = model_id
        self.birth_time: float = time.time()
        self.timeline: List[GenesisEvent] = []
        self._milestones_achieved: Set[MilestoneType] = set()
        self._quality_history: List[QualityReport] = []
        self._loss_history: List[float] = []
        self._sample_history: List[str] = []
        self._peak_peers: int = 0
        self._total_tokens: int = 0
        self._total_rounds: int = 0
        self._generation: int = 0
        self._mutations: int = 0

    @property
    def age_secs(self) -> float:
        return time.time() - self.birth_time

    @property
    def age_str(self) -> str:
        """Human-readable age string."""
        s = self.age_secs
        if s < 60:
            return f"{s:.0f}s"
        if s < 3600:
            return f"{s / 60:.0f}m"
        if s < 86400:
            return f"{s / 3600:.1f}h"
        return f"{s / 86400:.1f}d"

    @property
    def milestones(self) -> List[GenesisEvent]:
        """All milestone events."""
        return [e for e in self.timeline if e.event_type == EventType.MILESTONE]

    @property
    def mutations(self) -> List[GenesisEvent]:
        """All mutation events."""
        return [e for e in self.timeline if e.event_type == EventType.MUTATION]

    @property
    def latest_quality(self) -> Optional[QualityReport]:
        """Most recent quality report."""
        return self._quality_history[-1] if self._quality_history else None

    def record_birth(
        self,
        model_params: int = 0,
        genome_hash: str = "",
        hidden_dim: int = 0,
        n_layers: int = 0,
    ):
        """Record the model's birth (creation from random weights)."""
        self.birth_time = time.time()
        desc = (
            f"Model born: {self.model_id} "
            f"({model_params:,} params, {n_layers} layers, {hidden_dim}d)"
        )
        event = GenesisEvent(
            event_type=EventType.BIRTH,
            timestamp=self.birth_time,
            description=desc,
            generation=0,
        )
        self.timeline.append(event)

    def record_round(
        self,
        round_id: str,
        loss: float,
        sample_text: str = "",
        tokens_processed: int = 0,
        peers: int = 0,
    ):
        """Record a training round and check for milestones."""
        self._total_rounds += 1
        self._total_tokens += tokens_processed
        self._loss_history.append(loss)
        if peers > self._peak_peers:
            self._peak_peers = peers

        # Assess sample quality.
        quality = assess_quality(sample_text) if sample_text else None
        if quality:
            self._quality_history.append(quality)
            self._sample_history.append(sample_text)

        # Record the round event.
        event = GenesisEvent(
            event_type=EventType.ROUND,
            timestamp=time.time(),
            description=f"Round {round_id}: loss={loss:.4f}",
            round_id=round_id,
            loss=loss,
            sample_text=sample_text[:200] if sample_text else "",
            peers=peers,
            tokens_total=self._total_tokens,
            quality_score=quality.score if quality else 0.0,
        )
        self.timeline.append(event)

        # Check for milestones.
        self._check_loss_milestones(loss, round_id)
        self._check_scale_milestones(round_id)
        self._check_peer_milestones(peers, round_id)
        if quality:
            self._check_quality_milestones(quality, sample_text, round_id)

    def record_mutation(
        self,
        mutation_type: str,
        description: str,
        generation: int = 0,
    ):
        """Record an architecture mutation."""
        self._mutations += 1
        self._generation = generation

        event = GenesisEvent(
            event_type=EventType.MUTATION,
            timestamp=time.time(),
            description=f"Evolution: {description}",
            generation=generation,
            mutation_type=mutation_type,
        )
        self.timeline.append(event)

        # Check evolution milestones.
        if MilestoneType.FIRST_MUTATION not in self._milestones_achieved:
            self._add_milestone(MilestoneType.FIRST_MUTATION, "")
        if generation >= 10 and MilestoneType.GENERATION_10 not in self._milestones_achieved:
            self._add_milestone(MilestoneType.GENERATION_10, "")
        if generation >= 100 and MilestoneType.GENERATION_100 not in self._milestones_achieved:
            self._add_milestone(MilestoneType.GENERATION_100, "")

    def _check_loss_milestones(self, loss: float, round_id: str):
        thresholds = [
            (5.0, MilestoneType.LOSS_BELOW_5),
            (4.0, MilestoneType.LOSS_BELOW_4),
            (3.0, MilestoneType.LOSS_BELOW_3),
            (2.0, MilestoneType.LOSS_BELOW_2),
            (1.0, MilestoneType.LOSS_BELOW_1),
        ]
        for threshold, milestone in thresholds:
            if loss < threshold and milestone not in self._milestones_achieved:
                self._add_milestone(milestone, round_id)

    def _check_scale_milestones(self, round_id: str):
        round_thresholds = [
            (10, MilestoneType.ROUNDS_10),
            (100, MilestoneType.ROUNDS_100),
            (1000, MilestoneType.ROUNDS_1000),
            (10000, MilestoneType.ROUNDS_10000),
        ]
        for threshold, milestone in round_thresholds:
            if self._total_rounds >= threshold and milestone not in self._milestones_achieved:
                self._add_milestone(milestone, round_id)

        token_thresholds = [
            (1_000_000, MilestoneType.TOKENS_1M),
            (10_000_000, MilestoneType.TOKENS_10M),
            (100_000_000, MilestoneType.TOKENS_100M),
            (1_000_000_000, MilestoneType.TOKENS_1B),
        ]
        for threshold, milestone in token_thresholds:
            if self._total_tokens >= threshold and milestone not in self._milestones_achieved:
                self._add_milestone(milestone, round_id)

    def _check_peer_milestones(self, peers: int, round_id: str):
        peer_thresholds = [
            (10, MilestoneType.PEERS_10),
            (100, MilestoneType.PEERS_100),
            (1000, MilestoneType.PEERS_1000),
            (10000, MilestoneType.PEERS_10000),
        ]
        for threshold, milestone in peer_thresholds:
            if peers >= threshold and milestone not in self._milestones_achieved:
                self._add_milestone(milestone, round_id)

    def _check_quality_milestones(
        self,
        quality: QualityReport,
        sample: str,
        round_id: str,
    ):
        """Check for text quality milestones."""
        # First non-random output.
        if (
            quality.entropy_score > 0.5
            and MilestoneType.FIRST_NONRANDOM not in self._milestones_achieved
        ):
            self._add_milestone(
                MilestoneType.FIRST_NONRANDOM, round_id,
                sample_text=sample,
            )

        # First real word.
        if (
            quality.real_word_ratio > 0
            and quality.real_words_found
            and MilestoneType.FIRST_REAL_WORD not in self._milestones_achieved
        ):
            words = ", ".join(quality.real_words_found[:5])
            self._add_milestone(
                MilestoneType.FIRST_REAL_WORD, round_id,
                sample_text=f"Words found: {words}\nSample: {sample[:100]}",
            )

        # First word pair (2 adjacent real words).
        if (
            len(quality.longest_real_phrase.split()) >= 2
            and MilestoneType.FIRST_WORD_PAIR not in self._milestones_achieved
        ):
            self._add_milestone(
                MilestoneType.FIRST_WORD_PAIR, round_id,
                sample_text=f"Phrase: '{quality.longest_real_phrase}'\nSample: {sample[:100]}",
            )

        # First phrase (3+ adjacent real words).
        if (
            len(quality.longest_real_phrase.split()) >= 3
            and MilestoneType.FIRST_PHRASE not in self._milestones_achieved
        ):
            self._add_milestone(
                MilestoneType.FIRST_PHRASE, round_id,
                sample_text=f"Phrase: '{quality.longest_real_phrase}'\nSample: {sample[:100]}",
            )

        # First punctuation.
        if (
            quality.punctuation_score > 0.3
            and MilestoneType.FIRST_PUNCTUATION not in self._milestones_achieved
        ):
            self._add_milestone(
                MilestoneType.FIRST_PUNCTUATION, round_id,
                sample_text=sample[:100],
            )

        # First sentence.
        if (
            quality.has_sentence
            and MilestoneType.FIRST_SENTENCE not in self._milestones_achieved
        ):
            self._add_milestone(
                MilestoneType.FIRST_SENTENCE, round_id,
                sample_text=sample[:200],
            )

        # First paragraph.
        if (
            quality.has_paragraph
            and MilestoneType.FIRST_PARAGRAPH not in self._milestones_achieved
        ):
            self._add_milestone(
                MilestoneType.FIRST_PARAGRAPH, round_id,
                sample_text=sample[:300],
            )

        # First coherent text (composite score > 0.6).
        if (
            quality.score > 0.6
            and MilestoneType.FIRST_COHERENT not in self._milestones_achieved
        ):
            self._add_milestone(
                MilestoneType.FIRST_COHERENT, round_id,
                sample_text=sample[:300],
            )

    def _add_milestone(
        self,
        milestone: MilestoneType,
        round_id: str,
        sample_text: str = "",
    ):
        """Record a milestone event."""
        self._milestones_achieved.add(milestone)
        desc = MILESTONE_DESCRIPTIONS.get(milestone, milestone.value)

        event = GenesisEvent(
            event_type=EventType.MILESTONE,
            timestamp=time.time(),
            description=desc,
            round_id=round_id,
            milestone=milestone,
            sample_text=sample_text,
            tokens_total=self._total_tokens,
            peers=self._peak_peers,
            generation=self._generation,
        )
        self.timeline.append(event)

    def get_shareable_milestone(self, milestone: MilestoneType) -> str:
        """Get a shareable string for a milestone (for social media)."""
        emoji = MILESTONE_EMOJI.get(milestone, "")
        desc = MILESTONE_DESCRIPTIONS.get(milestone, milestone.value)
        age = self.age_str
        return (
            f"{emoji} {desc}\n"
            f"Model: {self.model_id} | Age: {age} | "
            f"Rounds: {self._total_rounds:,} | "
            f"Peers: {self._peak_peers}\n"
            f"#OpenClaw #PeoplesAI #DecentralizedAI"
        )

    def get_status(self) -> dict:
        """Get current genesis status as a dict."""
        latest_q = self.latest_quality
        return {
            "model_id": self.model_id,
            "age_secs": round(self.age_secs, 1),
            "age_str": self.age_str,
            "total_rounds": self._total_rounds,
            "total_tokens": self._total_tokens,
            "peak_peers": self._peak_peers,
            "generation": self._generation,
            "mutations": self._mutations,
            "milestones_achieved": len(self._milestones_achieved),
            "milestones": [m.value for m in sorted(self._milestones_achieved, key=lambda m: m.value)],
            "current_loss": self._loss_history[-1] if self._loss_history else None,
            "best_loss": min(self._loss_history) if self._loss_history else None,
            "current_quality": latest_q.score if latest_q else 0.0,
            "quality_components": latest_q.to_dict() if latest_q else None,
            "loss_history": self._loss_history[-100:],
            "quality_history": [q.score for q in self._quality_history[-100:]],
        }

    def get_timeline_summary(self, max_events: int = 50) -> List[dict]:
        """Get a summary of the most important events."""
        # Prioritize milestones and mutations over regular rounds.
        important = [
            e for e in self.timeline
            if e.event_type in (EventType.BIRTH, EventType.MILESTONE, EventType.MUTATION)
        ]

        # Add every Nth round for context.
        rounds = [e for e in self.timeline if e.event_type == EventType.ROUND]
        step = max(1, len(rounds) // 20)
        sampled_rounds = rounds[::step]

        combined = sorted(
            important + sampled_rounds,
            key=lambda e: e.timestamp,
        )

        result = []
        for event in combined[-max_events:]:
            d = event.to_dict()
            # Add age relative to birth.
            d["age_secs"] = round(event.timestamp - self.birth_time, 1)
            d["age_str"] = _format_duration(event.timestamp - self.birth_time)
            result.append(d)

        return result


def _format_duration(secs: float) -> str:
    """Format seconds into human-readable duration."""
    if secs < 60:
        return f"{secs:.0f}s"
    if secs < 3600:
        return f"{secs / 60:.0f}m"
    if secs < 86400:
        return f"{secs / 3600:.1f}h"
    return f"{secs / 86400:.1f}d"
