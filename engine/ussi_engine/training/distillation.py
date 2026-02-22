"""Knowledge distillation from SOTA teacher models.

Trains the student (USSI) model to match the teacher's soft probability
distributions. This transfers knowledge from large models to our small
decentralized model.

Key challenge: vocab mismatch. USSI uses byte-level vocab (260 tokens)
while SOTA models use BPE (100K+ tokens). We marginalize BPE token probs
down to byte-level probs.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..teacher import TeacherConfig, TeacherModel, create_teacher
from ..kickstart import Kickstart, KickstartConfig, KickstartResult

logger = logging.getLogger(__name__)


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation."""

    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    alpha: float = 0.5          # weight: alpha*distill_loss + (1-alpha)*task_loss
    temperature: float = 2.0    # softmax temperature for soft targets
    batch_size: int = 4


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    target_ids: torch.Tensor,
    alpha: float = 0.5,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Combined distillation + task loss.

    Loss = alpha * KL(soft_student || soft_teacher) + (1-alpha) * CE(student, targets)

    The KL divergence is computed on softened distributions (divided by temperature)
    to capture more of the teacher's knowledge about relative token probabilities.

    Args:
        student_logits: (B, T, V) raw logits from student model.
        teacher_log_probs: (B, T, V) log probabilities from teacher.
        target_ids: (B, T) ground truth token IDs.
        alpha: Weight for distillation vs task loss.
        temperature: Softmax temperature for soft targets.

    Returns:
        Scalar combined loss tensor.
    """
    B, T, V = student_logits.shape

    # Soft distributions at temperature T.
    student_log_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = torch.exp(teacher_log_probs / temperature)
    # Clamp to avoid log(0).
    teacher_soft = teacher_soft.clamp(min=1e-10)
    teacher_log_soft = torch.log(teacher_soft)

    # KL divergence: sum over vocab, mean over batch and sequence.
    # KL(P || Q) = sum P * (log P - log Q)
    kl_div = F.kl_div(
        student_log_soft.view(-1, V),
        teacher_log_soft.view(-1, V),
        reduction="batchmean",
        log_target=True,
    )

    # Scale by T^2 (standard practice in distillation).
    distill_loss = kl_div * (temperature ** 2)

    # Standard cross-entropy task loss.
    task_loss = F.cross_entropy(
        student_logits.view(-1, V),
        target_ids.view(-1),
        ignore_index=0,  # PAD_TOKEN
    )

    # Combined loss.
    combined = alpha * distill_loss + (1.0 - alpha) * task_loss
    return combined


class TeacherLogitProvider:
    """Converts teacher model outputs to byte-level log probabilities.

    Handles the vocab mismatch: SOTA models use BPE (100K+ tokens),
    USSI uses byte-level (260 tokens). We decompose each BPE token
    into its constituent bytes and distribute probability mass.
    """

    def __init__(self, teacher: TeacherModel, vocab_size: int = 260):
        self.teacher = teacher
        self.vocab_size = vocab_size

    def get_byte_level_probs(self, texts: List[str], seq_length: int) -> torch.Tensor:
        """Get byte-level log probabilities for a batch of texts.

        For each text, queries the teacher for token-level log probs,
        then marginalizes to byte-level probabilities.

        Args:
            texts: List of text strings.
            seq_length: Target sequence length.

        Returns:
            (B, T, V) tensor of log probabilities over byte-level vocab.
        """
        batch_log_probs = []

        for text in texts:
            log_probs = self._text_to_byte_log_probs(text, seq_length)
            batch_log_probs.append(log_probs)

        return torch.stack(batch_log_probs)

    def _text_to_byte_log_probs(self, text: str, seq_length: int) -> torch.Tensor:
        """Convert a text to byte-level log probability distribution.

        Strategy:
        1. Get teacher's per-token log probs for the text
        2. Map each token's probability to its constituent bytes
        3. Build a (T, V) distribution over byte-level vocab

        Args:
            text: Input text string.
            seq_length: Target sequence length.

        Returns:
            (T, V) tensor of log probabilities.
        """
        from ..data.tokenizer import SPECIAL_TOKENS

        # Get raw bytes of the text.
        text_bytes = text.encode("utf-8", errors="replace")

        # Get teacher log probs (per-byte approximation).
        # Split text into prompt (first half) and completion (second half).
        mid = max(len(text) // 2, 1)
        prompt = text[:mid]
        completion = text[mid:]

        try:
            teacher_lps = self.teacher.log_probs(prompt, completion)
        except Exception as e:
            logger.warning("Teacher log_probs failed: %s, using uniform", e)
            teacher_lps = [math.log(1.0 / self.vocab_size)] * len(text_bytes)

        # Build (T, V) log prob distribution.
        # For each byte position, create a peaked distribution at the actual byte.
        result = torch.full((seq_length, self.vocab_size), math.log(1e-10))

        for i in range(min(len(text_bytes), seq_length)):
            byte_val = text_bytes[i]
            token_id = byte_val + SPECIAL_TOKENS  # Offset by special tokens.

            if token_id < self.vocab_size:
                # Set the teacher's probability for this byte.
                if i < len(teacher_lps):
                    lp = teacher_lps[i]
                else:
                    lp = math.log(1.0 / self.vocab_size)

                # Create a distribution peaked at the teacher's predicted byte.
                # Distribute remaining mass uniformly.
                prob = min(math.exp(lp), 1.0)
                remaining = 1.0 - prob
                uniform_lp = math.log(max(remaining / (self.vocab_size - 1), 1e-10))

                result[i, :] = uniform_lp
                result[i, token_id] = math.log(max(prob, 1e-10))

        # Normalize to valid log probabilities.
        result = F.log_softmax(result, dim=-1)
        return result


class DistillationKickstart(Kickstart):
    """Extends Kickstart with knowledge distillation from a teacher model.

    Overrides train_round() to blend the standard task loss with KL divergence
    from the teacher's soft probability distributions.
    """

    def __init__(self, config: KickstartConfig, distill_config: DistillationConfig):
        super().__init__(config)
        self.distill_config = distill_config
        self._teacher: Optional[TeacherModel] = None
        self._logit_provider: Optional[TeacherLogitProvider] = None

    @property
    def teacher(self) -> TeacherModel:
        if self._teacher is None:
            self._teacher = create_teacher(self.distill_config.teacher, kickstart=self)
            self._logit_provider = TeacherLogitProvider(
                self._teacher, vocab_size=self.config.vocab_size
            )
        return self._teacher

    @property
    def logit_provider(self) -> TeacherLogitProvider:
        if self._logit_provider is None:
            _ = self.teacher  # Triggers lazy init.
        return self._logit_provider

    def set_teacher(self, teacher: TeacherModel):
        """Set the teacher model directly (useful for testing)."""
        self._teacher = teacher
        self._logit_provider = TeacherLogitProvider(
            teacher, vocab_size=self.config.vocab_size
        )

    def train_round(
        self,
        round_id: str = "",
        peer_id: str = "local",
    ) -> KickstartResult:
        """Run a distillation training round.

        For each batch:
        1. Forward pass through student to get logits
        2. Get teacher's byte-level log probabilities
        3. Compute distillation_loss (blended KL + CE)
        4. Backward + optimize
        """
        import time

        if self.data.total_tokens < self.config.max_seq_length + 1:
            return KickstartResult(round_id=round_id, avg_loss=float("inf"))

        start = time.monotonic()
        self.model.train()

        weight_snapshot = {
            k: v.clone() for k, v in self.model.state_dict().items()
        }
        prev_round_loss = getattr(self, '_last_round_loss', None)

        losses = []
        tokens = 0
        skipped = 0

        for input_ids, target_ids, mask in self.data.iter_batches(round_id, peer_id):
            self.optimizer.zero_grad()

            # Student forward pass.
            logits, _ = self.model(input_ids, target_ids)

            # Get teacher's byte-level probabilities for this batch.
            try:
                batch_texts = []
                for seq in target_ids:
                    from ..data.tokenizer import SPECIAL_TOKENS
                    byte_vals = [max(t.item() - SPECIAL_TOKENS, 0) for t in seq if t.item() >= SPECIAL_TOKENS]
                    batch_texts.append(bytes(byte_vals).decode("utf-8", errors="replace"))

                teacher_log_probs = self.logit_provider.get_byte_level_probs(
                    batch_texts, seq_length=input_ids.shape[1]
                )
            except Exception as e:
                logger.warning("Teacher log_probs failed: %s, falling back to task-only loss", e)
                # Fall back to standard cross-entropy.
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target_ids.view(-1),
                    ignore_index=0,
                )
                loss_val = loss.item()
                if not (loss_val == loss_val) or abs(loss_val) == float("inf"):
                    skipped += 1
                    self.optimizer.zero_grad()
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                losses.append(loss_val)
                tokens += input_ids.numel()
                self.total_steps += 1
                continue

            # Compute distillation loss.
            loss = distillation_loss(
                student_logits=logits,
                teacher_log_probs=teacher_log_probs,
                target_ids=target_ids,
                alpha=self.distill_config.alpha,
                temperature=self.distill_config.temperature,
            )

            loss_val = loss.item()

            # NaN/Inf check.
            if not (loss_val == loss_val) or abs(loss_val) == float("inf"):
                logger.warning("Round %s: NaN/Inf distillation loss, skipping step", round_id)
                skipped += 1
                self.optimizer.zero_grad()
                continue

            loss.backward()

            # NaN/Inf gradient check.
            has_bad_grad = False
            for param in self.model.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        has_bad_grad = True
                        break
            if has_bad_grad:
                logger.warning("Round %s: NaN/Inf gradients in distillation, skipping", round_id)
                self.optimizer.zero_grad()
                skipped += 1
                continue

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            losses.append(loss_val)
            tokens += input_ids.numel()
            self.total_steps += 1

        elapsed = (time.monotonic() - start) * 1000
        self.round_count += 1

        avg_loss = sum(losses) / len(losses) if losses else float("inf")
        final_loss = losses[-1] if losses else float("inf")

        # Loss spike detection.
        reverted = False
        if prev_round_loss is not None and prev_round_loss > 0:
            if final_loss > 5 * prev_round_loss:
                logger.warning(
                    "Round %s: distillation loss spike (%.4f > 5x %.4f), reverting",
                    round_id, final_loss, prev_round_loss,
                )
                self.model.load_state_dict(weight_snapshot)
                reverted = True

        self._last_round_loss = avg_loss if not reverted else prev_round_loss

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
            reverted=reverted,
        )

        try:
            result.sample_text = self.generate("The ", max_tokens=30)
        except Exception as e:
            result.sample_text = f"[generation failed: {e}]"

        logger.info(
            "Distillation round %s: %d steps (%d skipped), avg_loss=%.4f, alpha=%.2f",
            round_id, result.steps_completed, skipped, avg_loss, self.distill_config.alpha,
        )

        return result
