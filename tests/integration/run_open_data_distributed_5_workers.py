#!/usr/bin/env python3
"""Run a realistic 5-worker local distributed training + inference demo.

What this script does:
1. Downloads a larger open dataset (Project Gutenberg public-domain books).
2. Splits the corpus into 5 data shards (one per worker).
3. Spins up 5 local worker processes.
4. Trains the same model on each worker with local data.
5. Aggregates gradients each round via ring all-reduce.
6. Verifies all workers converge to the same model state hash.
7. Runs inference routed across all 5 workers on 10 prompts.
"""

from __future__ import annotations

import hashlib
import io
import os
import queue
import sys
import time
import base64
import ssl
import urllib.error
import urllib.request
import argparse
from dataclasses import asdict
from multiprocessing import get_context
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "engine"))

from ussi_engine.data.downloader import download_gutenberg, get_sample_text, GUTENBERG_BOOKS
from ussi_engine.kickstart import Kickstart, KickstartConfig
from ussi_engine.training.allreduce import RingAllReduce
import torch


BOOKS = [
    "alice_in_wonderland",
    "pride_and_prejudice",
    "moby_dick",
    "frankenstein",
    "sherlock_holmes",
]


PROMPTS = [
    "Summarize the role of curiosity in Alice in Wonderland.",
    "Explain one major theme in Pride and Prejudice.",
    "What is Captain Ahab obsessed with in Moby Dick?",
    "What ethical question sits at the center of Frankenstein?",
    "Describe Sherlock Holmes in one paragraph.",
    "Write a short reflective paragraph about ambition and consequences.",
    "Compare gothic atmosphere and social satire in two sentences.",
    "Give a concise explanation of narrative voice in classic novels.",
    "How does fear shape decisions in 19th-century fiction?",
    "Provide three bullet-like facts about these books.",
]


def _state_hash(ks: Kickstart) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(ks.model.state_dict().items()):
        h.update(name.encode("utf-8"))
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _encode_gradients(grads: Dict[str, torch.Tensor]) -> Dict[str, str]:
    encoded: Dict[str, str] = {}
    for name, tensor in grads.items():
        buf = io.BytesIO()
        torch.save(tensor.detach().cpu(), buf)
        encoded[name] = base64.b64encode(buf.getvalue()).decode("ascii")
    return encoded


def _decode_gradients(payload: Dict[str, str]) -> Dict[str, torch.Tensor]:
    grads: Dict[str, torch.Tensor] = {}
    for name, blob in payload.items():
        raw = base64.b64decode(blob.encode("ascii"))
        buf = io.BytesIO(raw)
        grads[name] = torch.load(buf, weights_only=True)
    return grads


def _split_text(text: str, parts: int) -> List[str]:
    chunks: List[str] = []
    n = len(text)
    for i in range(parts):
        start = i * n // parts
        end = (i + 1) * n // parts
        chunk = text[start:end]
        if not chunk.endswith("\n"):
            chunk += "\n"
        chunks.append(chunk)
    return chunks


def _load_open_corpus(data_dir: str) -> Tuple[str, Dict[str, str]]:
    paths = download_gutenberg(books=BOOKS, data_dir=data_dir)
    if not paths:
        paths = _download_gutenberg_with_unverified_ssl(data_dir)
    source_meta: Dict[str, str] = {}

    texts: List[str] = []
    for p in sorted(paths):
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        texts.append(content)
        source_meta[os.path.basename(p)] = str(len(content))

    if texts:
        return "\n\n".join(texts), source_meta

    # Fallback keeps run functional even if the environment blocks outbound network.
    fallback = "\n".join(
        [
            get_sample_text("alice"),
            get_sample_text("shakespeare"),
            get_sample_text("philosophy"),
            get_sample_text("science"),
        ]
    )
    source_meta["fallback"] = "builtin_open_samples"
    return fallback * 40, source_meta


def _download_gutenberg_with_unverified_ssl(data_dir: str) -> List[str]:
    """Fallback downloader for environments with custom TLS roots."""
    os.makedirs(data_dir, exist_ok=True)
    ctx = ssl._create_unverified_context()
    downloaded: List[str] = []

    for key in BOOKS:
        info = GUTENBERG_BOOKS.get(key)
        if not info:
            continue
        out_path = os.path.join(data_dir, f"{key}.txt")
        if os.path.exists(out_path):
            downloaded.append(out_path)
            continue
        req = urllib.request.Request(
            info["url"],
            headers={"User-Agent": "USSI/1.0 open-data-runner"},
        )
        try:
            with urllib.request.urlopen(req, timeout=90, context=ctx) as resp:
                content = resp.read()
            with open(out_path, "wb") as f:
                f.write(content)
            downloaded.append(out_path)
        except urllib.error.URLError:
            continue

    return downloaded


def _compute_local_gradients(ks: Kickstart, round_id: str, peer_id: str, local_steps: int):
    ks.model.train()
    ks.optimizer.zero_grad()
    losses: List[float] = []
    steps = 0

    for input_ids, target_ids, _mask in ks.data.iter_batches(round_id, peer_id):
        _, loss = ks.model(input_ids, target_ids)
        loss.backward()
        losses.append(float(loss.item()))
        steps += 1
        if steps >= local_steps:
            break

    if steps == 0:
        ks.optimizer.zero_grad()
        return {}, float("inf"), 0

    for param in ks.model.parameters():
        if param.grad is not None:
            param.grad /= steps

    gradients: Dict[str, torch.Tensor] = {}
    for name, param in ks.model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.detach().cpu().clone()

    avg_loss = sum(losses) / len(losses)
    ks.optimizer.zero_grad()
    return gradients, avg_loss, steps


def _apply_aggregated_gradients(ks: Kickstart, gradients: Dict[str, torch.Tensor]):
    ks.optimizer.zero_grad()
    for name, param in ks.model.named_parameters():
        grad = gradients.get(name)
        if grad is not None:
            param.grad = grad.to(param.device)
    torch.nn.utils.clip_grad_norm_(ks.model.parameters(), ks.config.max_grad_norm)
    ks.optimizer.step()
    ks.total_steps += 1


def _worker_main(
    worker_id: int,
    cfg_dict: Dict,
    shard_text: str,
    cmd_q,
    resp_q,
):
    torch.manual_seed(1337)
    cfg = KickstartConfig(**cfg_dict)
    ks = Kickstart(cfg)
    ks.load_text(shard_text)

    while True:
        msg = cmd_q.get()
        cmd = msg.get("cmd")

        if cmd == "train_round":
            round_id = msg["round_id"]
            grads, avg_loss, steps = _compute_local_gradients(
                ks, round_id=round_id, peer_id=f"worker-{worker_id}", local_steps=msg["local_steps"]
            )
            resp_q.put(
                {
                    "worker_id": worker_id,
                    "type": "train_result",
                    "avg_loss": avg_loss,
                    "final_loss": avg_loss,
                    "steps": steps,
                    "tokens": int(steps * ks.config.batch_size * ks.config.max_seq_length),
                    "grads": _encode_gradients(grads),
                    "state_hash": _state_hash(ks),
                }
            )
        elif cmd == "apply_gradients":
            _apply_aggregated_gradients(ks, _decode_gradients(msg["grads"]))
            resp_q.put(
                {
                    "worker_id": worker_id,
                    "type": "applied",
                    "state_hash": _state_hash(ks),
                }
            )
        elif cmd == "infer":
            out = ks.generate(
                msg["prompt"],
                max_tokens=msg.get("max_tokens", 100),
                temperature=msg.get("temperature", 0.4),
            )
            resp_q.put(
                {
                    "worker_id": worker_id,
                    "type": "infer_result",
                    "prompt_idx": msg["prompt_idx"],
                    "output": out,
                }
            )
        elif cmd == "stats":
            resp_q.put(
                {
                    "worker_id": worker_id,
                    "type": "stats",
                    "tokens": ks.data.total_tokens,
                    "sequences": ks.data.total_sequences,
                    "batches": ks.data.total_batches,
                    "state_hash": _state_hash(ks),
                }
            )
        elif cmd == "stop":
            resp_q.put({"worker_id": worker_id, "type": "stopped"})
            return


def _recv_n(resp_q, n: int, timeout: float = 120.0) -> List[Dict]:
    out = []
    deadline = time.time() + timeout
    while len(out) < n:
        remaining = max(0.1, deadline - time.time())
        try:
            out.append(resp_q.get(timeout=remaining))
        except queue.Empty:
            raise TimeoutError(f"Timed out waiting for {n} worker responses; got {len(out)}")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Run local 5-worker distributed training/inference on open data."
    )
    parser.add_argument("--workers", type=int, default=5, help="Number of local workers")
    parser.add_argument("--rounds", type=int, default=8, help="Number of distributed rounds")
    parser.add_argument(
        "--local-steps",
        type=int,
        default=8,
        help="Local gradient accumulation steps per worker per round",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--steps-per-round", type=int, default=35)
    parser.add_argument("--max-tokens", type=int, default=90, help="Inference max_new_tokens")
    parser.add_argument("--temperature", type=float, default=0.35, help="Inference temperature")
    args = parser.parse_args()

    n_workers = args.workers
    rounds = args.rounds
    local_steps_per_round = args.local_steps

    base_dir = os.path.join(os.path.dirname(__file__), "..", "..", "tmp", "open_data_run")
    os.makedirs(base_dir, exist_ok=True)
    data_dir = os.path.join(base_dir, "gutenberg")

    print("=== DATASET PREP ===")
    corpus, source_meta = _load_open_corpus(data_dir=data_dir)
    corpus_bytes = len(corpus.encode("utf-8", errors="replace"))
    print(f"Corpus size: {corpus_bytes / (1024 * 1024):.2f} MB")
    print(f"Sources: {source_meta}")

    shards = _split_text(corpus, n_workers)
    for i, shard in enumerate(shards):
        print(f"  worker-{i}: shard chars={len(shard):,}")

    cfg = KickstartConfig(
        model_id="open-data-5-worker-demo",
        hidden_dim=args.hidden_dim,
        n_layers=args.layers,
        n_heads=args.heads,
        max_seq_length=args.seq_len,
        dropout=0.0,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        steps_per_round=args.steps_per_round,
    )
    cfg_dict = asdict(cfg)

    ctx = get_context("spawn")
    resp_q = ctx.Queue()
    cmd_queues = [ctx.Queue() for _ in range(n_workers)]
    procs = []

    print("\n=== STARTING WORKERS ===")
    for i in range(n_workers):
        p = ctx.Process(
            target=_worker_main,
            args=(i, cfg_dict, shards[i], cmd_queues[i], resp_q),
        )
        p.start()
        procs.append(p)
        print(f"  started worker-{i} pid={p.pid}")

    try:
        for q in cmd_queues:
            q.put({"cmd": "stats"})
        stats = sorted(_recv_n(resp_q, n_workers), key=lambda x: x["worker_id"])
        print("\n=== WORKER STATS ===")
        for s in stats:
            print(
                f"worker-{s['worker_id']}: tokens={s['tokens']:,} "
                f"sequences={s['sequences']:,} batches={s['batches']:,}"
            )

        rings = RingAllReduce.local_ring(n_workers)

        print("\n=== DISTRIBUTED TRAINING ===")
        for round_idx in range(rounds):
            round_id = f"open-round-{round_idx}"
            for q in cmd_queues:
                q.put(
                    {
                        "cmd": "train_round",
                        "round_id": round_id,
                        "local_steps": local_steps_per_round,
                    }
                )

            round_results = sorted(
                _recv_n(resp_q, n_workers, timeout=240.0),
                key=lambda x: x["worker_id"],
            )

            all_grads = [_decode_gradients(r["grads"]) for r in round_results]
            aggregated = RingAllReduce.reduce_all(rings, all_grads)

            for i, q in enumerate(cmd_queues):
                q.put({"cmd": "apply_gradients", "grads": _encode_gradients(aggregated[i])})
            applied = sorted(_recv_n(resp_q, n_workers), key=lambda x: x["worker_id"])

            avg_loss = sum(r["avg_loss"] for r in round_results) / n_workers
            final_loss = sum(r["final_loss"] for r in round_results) / n_workers
            hashes = {a["state_hash"] for a in applied}
            print(
                f"round={round_idx:02d} avg_loss={avg_loss:.4f} "
                f"final_loss={final_loss:.4f} consensus={'yes' if len(hashes) == 1 else 'no'}"
            )

        print("\n=== DISTRIBUTED INFERENCE (10 prompts across 5 workers) ===")
        for idx, prompt in enumerate(PROMPTS):
            target_worker = idx % n_workers
            cmd_queues[target_worker].put(
                {
                    "cmd": "infer",
                    "prompt_idx": idx,
                    "prompt": prompt,
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                }
            )

        infer_results = sorted(
            _recv_n(resp_q, len(PROMPTS), timeout=240.0),
            key=lambda x: x["prompt_idx"],
        )
        for item in infer_results:
            idx = item["prompt_idx"]
            worker_id = item["worker_id"]
            out = item["output"].replace("\n", "\\n")
            print(f"\n[{idx+1}] worker-{worker_id}")
            print(f"PROMPT: {PROMPTS[idx]}")
            print(f"OUTPUT: {out}")

    finally:
        for q in cmd_queues:
            q.put({"cmd": "stop"})
        _recv_n(resp_q, n_workers, timeout=30.0)
        for p in procs:
            p.join(timeout=20.0)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5.0)

    print("\n=== COMPLETE ===")


if __name__ == "__main__":
    main()
