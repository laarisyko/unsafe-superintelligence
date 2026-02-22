"""Bridge between the Rust P2P node and the Python ML engine.

In a full implementation, this would use PyO3 (Rust -> Python FFI) so the
Rust node can invoke the Python engine directly. For the initial version,
we use a gRPC / HTTP bridge where the Rust node calls the Python engine
over localhost.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
from typing import Callable, Dict, Optional

import torch

from .model.shard import ModelShard
from .inference.server import InferenceServer, InferenceRequest
from .training.trainer import LocalTrainer, TrainingConfig
from .kickstart import Kickstart, KickstartConfig

logger = logging.getLogger(__name__)


class NodeBridge:
    """Bridges the Rust P2P node (network layer) with the Python ML engine.

    The bridge exposes methods that the node can call to:
    - Run inference on a local model (via Kickstart or InferenceServer)
    - Execute training steps using real data
    - Get/set gradients for all-reduce (full tensor serialization)
    - Compute Merkle roots for weight verification
    - Evaluate architecture proposals and record votes
    """

    def __init__(
        self,
        inference_server: InferenceServer,
        trainer: Optional[LocalTrainer] = None,
        kickstart: Optional[Kickstart] = None,
    ):
        self.inference_server = inference_server
        self.trainer = trainer
        self.kickstart = kickstart
        self._callbacks: Dict[str, Callable] = {}
        self._cached_gradients: Optional[Dict[str, torch.Tensor]] = None
        self._proposal_votes: Dict[str, list] = {}

    def register_callback(self, event: str, callback: Callable):
        """Register a callback for node events (e.g. 'peer_joined', 'round_started')."""
        self._callbacks[event] = callback

    async def handle_node_message(self, msg_type: str, payload: bytes) -> bytes:
        """Handle a message from the Rust node.

        This is the main entry point called by the Rust node (via FFI or HTTP).
        """
        if msg_type == "infer":
            return await self._handle_infer(payload)
        elif msg_type == "train_step":
            return await self._handle_train_step(payload)
        elif msg_type == "get_gradients":
            return self._handle_get_gradients()
        elif msg_type == "set_gradients":
            return self._handle_set_gradients(payload)
        elif msg_type == "merkle_root":
            return self._handle_merkle_root(payload)
        elif msg_type == "health":
            return self._handle_health()
        elif msg_type == "init_model":
            return self._handle_init_model(payload)
        elif msg_type == "load_data":
            return self._handle_load_data(payload)
        elif msg_type == "stats":
            return self._handle_stats()
        elif msg_type == "architecture_proposal":
            return self._handle_architecture_proposal(payload)
        elif msg_type == "architecture_vote":
            return self._handle_architecture_vote(payload)
        elif msg_type == "generate_synthetic":
            return self._handle_generate_synthetic(payload)
        elif msg_type == "set_teacher":
            return self._handle_set_teacher(payload)
        else:
            return json.dumps({"error": f"unknown message type: {msg_type}"}).encode()

    async def _handle_infer(self, payload: bytes) -> bytes:
        data = json.loads(payload)
        prompt = data.get("prompt", "")
        max_tokens = data.get("max_tokens", 256)
        temperature = data.get("temperature", 0.7)

        # Prefer Kickstart for text generation (has tokenizer + generation loop).
        if self.kickstart is not None:
            try:
                text = self.kickstart.generate(
                    prompt, max_tokens=max_tokens, temperature=temperature
                )
                return json.dumps({
                    "request_id": data.get("request_id", ""),
                    "text": text,
                    "latency_ms": 0.0,
                }).encode()
            except Exception as e:
                logger.warning("Kickstart inference failed: %s, falling back to InferenceServer", e)

        # Fallback to InferenceServer for shard-based pipeline inference.
        request = InferenceRequest(
            model_id=data.get("model_id", ""),
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        response = await self.inference_server.infer_async(request)
        return json.dumps({
            "request_id": response.request_id,
            "text": response.text,
            "latency_ms": response.latency_ms,
        }).encode()

    async def _handle_train_step(self, payload: bytes) -> bytes:
        """Execute a real training round using Kickstart with real data."""
        data = json.loads(payload)

        # Use Kickstart for real training if available.
        if self.kickstart is not None:
            round_id = data.get("round_id", "bridge-round")
            peer_id = data.get("peer_id", "bridge-peer")

            result = self.kickstart.train_round(round_id, peer_id)

            # Cache gradients for subsequent get_gradients call.
            self._cached_gradients = result.gradients

            return json.dumps({
                "loss": result.avg_loss,
                "final_loss": result.final_loss,
                "grad_norm": 0.0,
                "step": result.steps_completed,
                "tokens": result.tokens_processed,
                "skipped_steps": result.skipped_steps,
                "reverted": result.reverted,
            }).encode()

        # Fallback to trainer with synthetic data.
        if self.trainer is None:
            return json.dumps({"error": "no trainer or kickstart configured"}).encode()

        input_shape = data.get("input_shape", [1, 128, 512])
        input_tensor = torch.randn(*input_shape)
        metrics = self.trainer.train_step(input_tensor)
        return json.dumps(metrics).encode()

    def _handle_get_gradients(self) -> bytes:
        """Return full gradient tensors serialized as base64-encoded torch tensors."""
        grads = None

        # Prefer cached gradients from Kickstart.
        if self._cached_gradients is not None:
            grads = self._cached_gradients
        elif self.trainer is not None:
            grads = self.trainer.get_gradients()
        else:
            return json.dumps({"error": "no gradients available"}).encode()

        # Serialize gradient tensors using base64-encoded torch buffers.
        grad_data = {}
        total_params = 0
        for name, tensor in grads.items():
            buf = io.BytesIO()
            torch.save(tensor, buf)
            grad_data[name] = {
                "shape": list(tensor.shape),
                "numel": tensor.numel(),
                "data": base64.b64encode(buf.getvalue()).decode("ascii"),
            }
            total_params += tensor.numel()

        # Compute a simple hash for merkle verification.
        import hashlib
        all_bytes = b""
        for name in sorted(grads.keys()):
            all_bytes += grads[name].cpu().numpy().tobytes()
        merkle_root = hashlib.sha256(all_bytes).hexdigest()[:24]

        return json.dumps({
            "gradients": grad_data,
            "param_count": total_params,
            "merkle_root": merkle_root,
        }).encode()

    def _handle_set_gradients(self, payload: bytes) -> bytes:
        """Deserialize and apply aggregated gradient tensors."""
        data = json.loads(payload)

        grad_data = data.get("gradients", {})
        if not grad_data:
            return json.dumps({"error": "no gradients in payload"}).encode()

        # Deserialize gradient tensors.
        gradients = {}
        for name, info in grad_data.items():
            tensor_bytes = base64.b64decode(info["data"])
            buf = io.BytesIO(tensor_bytes)
            tensor = torch.load(buf, weights_only=True)
            gradients[name] = tensor

        # Apply via Kickstart if available.
        if self.kickstart is not None:
            self.kickstart.apply_aggregated_gradients(gradients)
            return json.dumps({
                "status": "ok",
                "applied": len(gradients),
            }).encode()

        # Fallback to trainer.
        if self.trainer is not None:
            self.trainer.set_gradients(gradients)
            self.trainer.apply_gradients()
            return json.dumps({
                "status": "ok",
                "applied": len(gradients),
            }).encode()

        return json.dumps({"error": "no trainer or kickstart configured"}).encode()

    def _handle_merkle_root(self, payload: bytes) -> bytes:
        data = json.loads(payload)
        model_id = data.get("model_id", "")

        shard = None
        if model_id in self.inference_server._models:
            shard = self.inference_server._models[model_id]

        if shard:
            root = shard.merkle_root()
            return json.dumps({"merkle_root": root.hex()}).encode()
        else:
            return json.dumps({"error": "model not found"}).encode()

    def _handle_health(self) -> bytes:
        stats = self.inference_server.stats()
        has_kickstart = self.kickstart is not None
        return json.dumps({
            "status": "ok",
            "kickstart_available": has_kickstart,
            **stats,
        }).encode()

    def _handle_init_model(self, payload: bytes) -> bytes:
        """Initialize a Kickstart model from config."""
        data = json.loads(payload)
        config = KickstartConfig(
            model_id=data.get("model_id", "ussi-v0"),
            hidden_dim=data.get("hidden_dim", 256),
            n_layers=data.get("n_layers", 6),
            n_heads=data.get("n_heads", 4),
            vocab_size=data.get("vocab_size", 260),
            max_seq_length=data.get("max_seq_length", 128),
            learning_rate=data.get("learning_rate", 3e-4),
            batch_size=data.get("batch_size", 4),
            steps_per_round=data.get("steps_per_round", 10),
        )
        self.kickstart = Kickstart(config)
        return json.dumps({
            "status": "ok",
            "model_id": config.model_id,
            "parameters": self.kickstart.model.num_parameters,
        }).encode()

    def _handle_load_data(self, payload: bytes) -> bytes:
        """Load text data into the training pipeline."""
        if self.kickstart is None:
            return json.dumps({"error": "no kickstart model initialized"}).encode()

        data = json.loads(payload)
        text = data.get("text", "")
        file_path = data.get("file_path", "")

        if text:
            self.kickstart.load_text(text)
        elif file_path:
            self.kickstart.load_file(file_path)
        else:
            return json.dumps({"error": "no text or file_path provided"}).encode()

        return json.dumps({
            "status": "ok",
            "total_tokens": self.kickstart.data.total_tokens,
            "total_sequences": self.kickstart.data.total_sequences,
        }).encode()

    def _handle_stats(self) -> bytes:
        """Return model and training statistics."""
        result = {}
        if self.kickstart is not None:
            result.update(self.kickstart.stats())
        result["has_trainer"] = self.trainer is not None
        result["has_kickstart"] = self.kickstart is not None
        result["cached_gradients"] = self._cached_gradients is not None
        return json.dumps(result).encode()

    def _handle_architecture_proposal(self, payload: bytes) -> bytes:
        """Evaluate an architecture proposal and return a vote."""
        from .architecture.evolution import (
            ArchitectureProposal,
            FitnessEvaluator,
            VoteDecision,
        )

        data = json.loads(payload)
        try:
            proposal = ArchitectureProposal.from_dict(data)
        except Exception as e:
            return json.dumps({"error": f"invalid proposal: {e}"}).encode()

        evaluator = FitnessEvaluator()

        # Validate the proposed genome first.
        errors = proposal.new_genome.validate()
        if errors:
            return json.dumps({
                "proposal_id": proposal.proposal_id,
                "decision": VoteDecision.REJECT.value,
                "reason": f"validation errors: {'; '.join(errors)}",
                "measured_fitness": float("-inf"),
            }).encode()

        # Evaluate fitness.
        current_fitness, proposed_fitness = evaluator.evaluate(
            ArchitectureProposal.from_dict(data).new_genome,
            proposal.new_genome,
        )
        decision = evaluator.should_approve(current_fitness, proposed_fitness)

        return json.dumps({
            "proposal_id": proposal.proposal_id,
            "decision": decision.value,
            "measured_fitness": proposed_fitness,
            "current_fitness": current_fitness,
        }).encode()

    def _handle_generate_synthetic(self, payload: bytes) -> bytes:
        """Trigger synthetic data generation via teacher model."""
        if self.kickstart is None:
            return json.dumps({"error": "no kickstart model initialized"}).encode()

        data = json.loads(payload)
        n_samples = data.get("n_samples", 10)
        provider = data.get("provider", "local")
        model = data.get("model", "")

        from .teacher import TeacherConfig
        teacher_config = TeacherConfig(provider=provider, model=model)

        try:
            fed = self.kickstart.generate_synthetic_data(teacher_config, n_samples)
            return json.dumps({
                "status": "ok",
                "samples_generated": fed,
                "total_tokens": self.kickstart.data.total_tokens,
            }).encode()
        except Exception as e:
            return json.dumps({"error": str(e)}).encode()

    def _handle_set_teacher(self, payload: bytes) -> bytes:
        """Configure teacher model for distillation/DPO."""
        data = json.loads(payload)
        provider = data.get("provider", "local")
        model = data.get("model", "")
        api_key = data.get("api_key", "")

        from .teacher import TeacherConfig
        config = TeacherConfig(provider=provider, model=model, api_key=api_key)

        # Store for later use.
        self._teacher_config = config
        return json.dumps({
            "status": "ok",
            "provider": provider,
            "model": model,
        }).encode()

    def _handle_architecture_vote(self, payload: bytes) -> bytes:
        """Record a vote for an architecture proposal."""
        from .architecture.evolution import ProposalVote

        data = json.loads(payload)
        try:
            vote = ProposalVote.from_dict(data)
        except Exception as e:
            return json.dumps({"error": f"invalid vote: {e}"}).encode()

        pid = vote.proposal_id
        self._proposal_votes.setdefault(pid, []).append(vote)

        votes = self._proposal_votes[pid]
        return json.dumps({
            "proposal_id": pid,
            "total_votes": len(votes),
            "status": "recorded",
        }).encode()


class HttpBridgeServer:
    """Simple HTTP server that the Rust node talks to over localhost.

    This is the fallback bridge when PyO3 is not available.
    """

    def __init__(self, bridge: NodeBridge, port: int = 50052):
        self.bridge = bridge
        self.port = port

    async def start(self):
        """Start the HTTP bridge server."""
        server = await asyncio.start_server(
            self._handle_connection, "127.0.0.1", self.port
        )
        logger.info("Bridge server listening on 127.0.0.1:%d", self.port)
        async with server:
            await server.serve_forever()

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ):
        try:
            # Read headers first.
            header_data = b""
            while b"\r\n\r\n" not in header_data:
                chunk = await reader.read(4096)
                if not chunk:
                    break
                header_data += chunk

            header_end = header_data.find(b"\r\n\r\n")
            if header_end < 0:
                writer.close()
                return

            headers_str = header_data[:header_end].decode("utf-8", errors="replace")
            body_start_data = header_data[header_end + 4:]

            # Parse minimal HTTP.
            lines = headers_str.split("\r\n")
            first_line = lines[0] if lines else ""
            parts = first_line.split()
            path = parts[1] if len(parts) >= 2 else "/"

            # Parse Content-Length and read full body.
            content_length = 0
            for line in lines[1:]:
                if line.lower().startswith("content-length:"):
                    try:
                        content_length = int(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass

            # Read remaining body bytes.
            body = body_start_data
            remaining = content_length - len(body)
            while remaining > 0:
                chunk = await reader.read(min(remaining, 65536))
                if not chunk:
                    break
                body += chunk
                remaining -= len(chunk)

            # Route to bridge.
            msg_type = path.strip("/")
            response_body = await self.bridge.handle_node_message(msg_type, body)

            response = (
                f"HTTP/1.1 200 OK\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(response_body)}\r\n"
                f"Connection: close\r\n\r\n"
            ).encode() + response_body

            writer.write(response)
            await writer.drain()
        finally:
            writer.close()


class DirectBridgeHandler:
    """In-process bridge handler for direct Python-to-Python communication.

    Bypasses HTTP/serialization overhead for local simulation and testing.
    Each simulated peer gets its own handler wrapping its own shard + trainer.
    """

    def __init__(
        self,
        shard: ModelShard,
        trainer: Optional[LocalTrainer] = None,
        training_config: Optional[TrainingConfig] = None,
    ):
        self.shard = shard
        self.trainer = trainer or LocalTrainer(
            shard, training_config or TrainingConfig()
        )
        self._last_gradients: Dict[str, torch.Tensor] = {}

    def train_step(self, input_shape: list = None) -> Dict:
        """Execute a training step and cache gradients."""
        import torch.nn as nn

        input_shape = input_shape or [1, 16]
        x = torch.randn(*input_shape, requires_grad=True)

        target = None
        loss_fn = None
        if self.shard.config.is_last:
            target = torch.randn(input_shape[0], input_shape[-1])
            loss_fn = nn.MSELoss()

        metrics = self.trainer.train_step(x, target, loss_fn)
        self._last_gradients = self.trainer.get_gradients()
        return metrics

    def get_gradients(self) -> Dict[str, torch.Tensor]:
        """Return cached gradient tensors from last training step."""
        return self._last_gradients

    def set_gradients(self, gradients: Dict[str, torch.Tensor]):
        """Apply aggregated gradients and update model weights."""
        self.trainer.set_gradients(gradients)
        self.trainer.apply_gradients()

    def merkle_root(self) -> str:
        """Compute the Merkle root of current weights."""
        return self.shard.merkle_root().hex()
