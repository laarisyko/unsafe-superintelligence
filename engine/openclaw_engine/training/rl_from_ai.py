"""DPO (Direct Preference Optimization) from AI feedback.

Uses SOTA models as judges to create preference pairs, then trains with DPO.
This gives RLHF-style alignment without needing a separate reward model.

Flow:
1. Sample prompts from training data
2. Generate 2 completions from the student model
3. Ask the teacher to judge which is better
4. Train with DPO loss on the preference pairs

The reference model is a frozen snapshot taken at the start of each DPO phase.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..teacher import TeacherConfig, TeacherModel, create_teacher
from ..kickstart import Kickstart, KickstartConfig, KickstartResult
from ..data.tokenizer import SPECIAL_TOKENS

logger = logging.getLogger(__name__)


@dataclass
class DPOConfig:
    """Configuration for DPO training."""

    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    beta: float = 0.1            # DPO temperature
    pairs_per_round: int = 20
    max_prompt_length: int = 64  # tokens
    max_response_length: int = 128  # tokens


@dataclass
class PreferencePair:
    """A preference pair for DPO training."""

    prompt: str
    chosen: str     # preferred by teacher
    rejected: str   # less preferred


JUDGE_TEMPLATE = """Given this prompt: "{prompt}"

Response A: "{response_a}"

Response B: "{response_b}"

Which response is better? Consider clarity, accuracy, coherence, and helpfulness.
Reply with just "A" or "B" and a brief reason."""


class PreferenceDataCollector:
    """Collects preference pairs using a SOTA model as judge.

    For each prompt:
    1. Generate 2 completions from the student model
    2. Ask the teacher to judge which is better
    3. Return (prompt, chosen, rejected) triple
    """

    def __init__(self, teacher: TeacherModel, kickstart: Kickstart):
        self.teacher = teacher
        self.kickstart = kickstart

    def _sample_prompts(self, n: int) -> List[str]:
        """Sample prompts from the training data.

        Takes short subsequences from the training data buffer as prompts.
        """
        prompts = []
        buffer = self.kickstart.data._token_buffer
        tokenizer = self.kickstart.tokenizer
        prompt_len = min(32, len(buffer) // 4)  # tokens

        if len(buffer) < prompt_len * 2:
            # Not enough data, create simple prompts.
            return ["The ", "Once upon ", "In the ", "A "] * (n // 4 + 1)

        import random
        rng = random.Random(42)

        for _ in range(n):
            start = rng.randint(0, max(0, len(buffer) - prompt_len - 1))
            token_slice = buffer[start:start + prompt_len]
            # Decode tokens back to text.
            text = tokenizer.decode(token_slice)
            if text.strip():
                prompts.append(text.strip()[:200])  # Limit length.
            else:
                prompts.append("The ")

        return prompts[:n]

    def _generate_responses(self, prompt: str, n: int = 2) -> List[str]:
        """Generate n responses from the student model."""
        responses = []
        for _ in range(n):
            try:
                text = self.kickstart.generate(
                    prompt,
                    max_tokens=64,
                    temperature=1.0,  # Higher temp for diversity.
                )
                # Extract just the generated part (after prompt).
                if text.startswith(prompt):
                    text = text[len(prompt):]
                responses.append(text.strip() if text.strip() else "...")
            except Exception as e:
                logger.warning("Student generation failed: %s", e)
                responses.append("...")
        return responses

    def _judge(self, prompt: str, response_a: str, response_b: str) -> str:
        """Ask the teacher to judge which response is better.

        Returns "A" or "B".
        """
        judge_prompt = JUDGE_TEMPLATE.format(
            prompt=prompt[:200],
            response_a=response_a[:500],
            response_b=response_b[:500],
        )

        try:
            judgment = self.teacher.generate(judge_prompt, max_tokens=50, temperature=0.1)
            judgment = judgment.strip().upper()
            if judgment.startswith("A"):
                return "A"
            elif judgment.startswith("B"):
                return "B"
            else:
                # Parse from the response.
                if "A" in judgment[:10]:
                    return "A"
                return "B"
        except Exception as e:
            logger.warning("Teacher judging failed: %s, defaulting to A", e)
            return "A"

    def collect_pairs(self, n: int = 20) -> List[PreferencePair]:
        """Collect n preference pairs.

        Args:
            n: Number of preference pairs to collect.

        Returns:
            List of PreferencePair instances.
        """
        prompts = self._sample_prompts(n)
        pairs = []

        for prompt in prompts[:n]:
            responses = self._generate_responses(prompt, n=2)
            if len(responses) < 2:
                continue

            # Ask teacher to judge.
            winner = self._judge(prompt, responses[0], responses[1])

            if winner == "A":
                pair = PreferencePair(
                    prompt=prompt,
                    chosen=responses[0],
                    rejected=responses[1],
                )
            else:
                pair = PreferencePair(
                    prompt=prompt,
                    chosen=responses[1],
                    rejected=responses[0],
                )

            pairs.append(pair)

        logger.info("Collected %d preference pairs from %d prompts", len(pairs), n)
        return pairs


def _sequence_log_probs(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    response: str,
) -> torch.Tensor:
    """Compute log probabilities of a response under a model.

    Returns the sum of log probs for the response tokens.
    """
    full_text = prompt + response
    tokens = tokenizer.encode(full_text, add_special=False)
    prompt_tokens = tokenizer.encode(prompt, add_special=False)
    prompt_len = len(prompt_tokens)

    if len(tokens) < 2:
        return torch.tensor(0.0)

    input_ids = torch.tensor([tokens[:-1]], dtype=torch.long)

    model.eval()
    with torch.no_grad():
        logits, _ = model(input_ids)

    # Log probs for response tokens only.
    log_probs = F.log_softmax(logits[0], dim=-1)

    total_log_prob = torch.tensor(0.0)
    count = 0
    for i in range(prompt_len, len(tokens) - 1):
        if i < log_probs.shape[0]:
            next_token = tokens[i + 1]
            if next_token < log_probs.shape[1]:
                total_log_prob = total_log_prob + log_probs[i, next_token]
                count += 1

    return total_log_prob


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """Standard DPO loss.

    Loss = -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))

    where log_ratio = log(pi(y|x)) - log(pi_ref(y|x))

    Args:
        policy_chosen_logps: Log probs of chosen responses under current policy.
        policy_rejected_logps: Log probs of rejected responses under current policy.
        ref_chosen_logps: Log probs of chosen responses under reference model.
        ref_rejected_logps: Log probs of rejected responses under reference model.
        beta: DPO temperature (controls strength of preference).

    Returns:
        Scalar DPO loss.
    """
    # Log ratios.
    chosen_log_ratio = policy_chosen_logps - ref_chosen_logps
    rejected_log_ratio = policy_rejected_logps - ref_rejected_logps

    # DPO loss.
    logits = beta * (chosen_log_ratio - rejected_log_ratio)
    loss = -F.logsigmoid(logits).mean()

    return loss


class DPOKickstart(Kickstart):
    """Extends Kickstart with DPO training rounds.

    Alternates between normal training rounds and DPO rounds.
    The reference model is a frozen snapshot taken at initialization.
    """

    def __init__(self, config: KickstartConfig, dpo_config: DPOConfig):
        super().__init__(config)
        self.dpo_config = dpo_config
        self._teacher: Optional[TeacherModel] = None
        self._collector: Optional[PreferenceDataCollector] = None
        # Reference model: frozen snapshot for DPO.
        self._ref_model = None

    @property
    def teacher(self) -> TeacherModel:
        if self._teacher is None:
            self._teacher = create_teacher(self.dpo_config.teacher, kickstart=self)
            self._collector = PreferenceDataCollector(self._teacher, self)
        return self._teacher

    @property
    def collector(self) -> PreferenceDataCollector:
        if self._collector is None:
            _ = self.teacher
        return self._collector

    def set_teacher(self, teacher: TeacherModel):
        """Set the teacher model directly (useful for testing)."""
        self._teacher = teacher
        self._collector = PreferenceDataCollector(teacher, self)

    def _ensure_ref_model(self):
        """Create/update the reference model snapshot."""
        if self._ref_model is None:
            self._ref_model = copy.deepcopy(self.model)
            self._ref_model.eval()
            # Freeze reference model.
            for param in self._ref_model.parameters():
                param.requires_grad = False
            logger.info("Created reference model snapshot for DPO")

    def update_reference(self):
        """Update the reference model to current weights."""
        self._ref_model = copy.deepcopy(self.model)
        self._ref_model.eval()
        for param in self._ref_model.parameters():
            param.requires_grad = False

    def train_dpo_round(
        self,
        round_id: str = "",
        peer_id: str = "local",
    ) -> KickstartResult:
        """Run a DPO training round.

        1. Collect preference pairs via PreferenceDataCollector
        2. Compute log probs for chosen/rejected under policy and reference
        3. Compute DPO loss and backprop
        """
        start = time.monotonic()
        self._ensure_ref_model()
        self.model.train()

        # Collect preference pairs.
        pairs = self.collector.collect_pairs(n=self.dpo_config.pairs_per_round)
        if not pairs:
            logger.warning("No preference pairs collected for DPO round %s", round_id)
            return KickstartResult(round_id=round_id, avg_loss=float("inf"))

        losses = []
        tokens = 0
        skipped = 0

        for pair in pairs:
            self.optimizer.zero_grad()

            # Compute log probs under current policy.
            policy_chosen_lp = _sequence_log_probs(
                self.model, self.tokenizer, pair.prompt, pair.chosen,
            )
            policy_rejected_lp = _sequence_log_probs(
                self.model, self.tokenizer, pair.prompt, pair.rejected,
            )

            # Compute log probs under reference model.
            ref_chosen_lp = _sequence_log_probs(
                self._ref_model, self.tokenizer, pair.prompt, pair.chosen,
            )
            ref_rejected_lp = _sequence_log_probs(
                self._ref_model, self.tokenizer, pair.prompt, pair.rejected,
            )

            # Need gradients for policy log probs.
            # Re-compute with grad enabled.
            self.model.train()
            full_chosen = pair.prompt + pair.chosen
            full_rejected = pair.prompt + pair.rejected
            chosen_tokens = self.tokenizer.encode(full_chosen, add_special=False)
            rejected_tokens = self.tokenizer.encode(full_rejected, add_special=False)
            prompt_tokens = self.tokenizer.encode(pair.prompt, add_special=False)
            prompt_len = len(prompt_tokens)

            if len(chosen_tokens) < 2 or len(rejected_tokens) < 2:
                skipped += 1
                continue

            # Forward pass for chosen.
            chosen_input = torch.tensor([chosen_tokens[:-1]], dtype=torch.long)
            chosen_logits, _ = self.model(chosen_input)
            chosen_log_probs = F.log_softmax(chosen_logits[0], dim=-1)

            chosen_lp = torch.tensor(0.0, requires_grad=True)
            chosen_lp_val = torch.tensor(0.0)
            for i in range(prompt_len, len(chosen_tokens) - 1):
                if i < chosen_log_probs.shape[0]:
                    next_token = chosen_tokens[i + 1]
                    if next_token < chosen_log_probs.shape[1]:
                        chosen_lp_val = chosen_lp_val + chosen_log_probs[i, next_token]

            # Forward pass for rejected.
            rejected_input = torch.tensor([rejected_tokens[:-1]], dtype=torch.long)
            rejected_logits, _ = self.model(rejected_input)
            rejected_log_probs = F.log_softmax(rejected_logits[0], dim=-1)

            rejected_lp_val = torch.tensor(0.0)
            for i in range(prompt_len, len(rejected_tokens) - 1):
                if i < rejected_log_probs.shape[0]:
                    next_token = rejected_tokens[i + 1]
                    if next_token < rejected_log_probs.shape[1]:
                        rejected_lp_val = rejected_lp_val + rejected_log_probs[i, next_token]

            # Compute DPO loss.
            loss = dpo_loss(
                policy_chosen_logps=chosen_lp_val,
                policy_rejected_logps=rejected_lp_val,
                ref_chosen_logps=ref_chosen_lp.detach(),
                ref_rejected_logps=ref_rejected_lp.detach(),
                beta=self.dpo_config.beta,
            )

            loss_val = loss.item()

            # NaN/Inf check.
            if not (loss_val == loss_val) or abs(loss_val) == float("inf"):
                logger.warning("DPO round %s: NaN/Inf loss, skipping pair", round_id)
                skipped += 1
                self.optimizer.zero_grad()
                continue

            loss.backward()

            # NaN gradient check.
            has_bad_grad = False
            for param in self.model.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        has_bad_grad = True
                        break
            if has_bad_grad:
                self.optimizer.zero_grad()
                skipped += 1
                continue

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            losses.append(loss_val)
            tokens += len(chosen_tokens) + len(rejected_tokens)
            self.total_steps += 1

        elapsed = (time.monotonic() - start) * 1000
        self.round_count += 1

        avg_loss = sum(losses) / len(losses) if losses else float("inf")
        final_loss = losses[-1] if losses else float("inf")

        gradients = self._collect_gradients(round_id, peer_id)

        result = KickstartResult(
            round_id=round_id,
            steps_completed=len(losses),
            avg_loss=avg_loss,
            final_loss=final_loss,
            tokens_processed=tokens,
            time_ms=elapsed,
            gradients=gradients,
            skipped_steps=skipped,
        )

        try:
            result.sample_text = self.generate("The ", max_tokens=30)
        except Exception as e:
            result.sample_text = f"[generation failed: {e}]"

        logger.info(
            "DPO round %s: %d pairs trained (%d skipped), avg_loss=%.4f, beta=%.2f",
            round_id, result.steps_completed, skipped, avg_loss, self.dpo_config.beta,
        )

        return result
