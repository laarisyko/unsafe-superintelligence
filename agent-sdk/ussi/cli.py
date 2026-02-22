"""CLI for USSI: ussi join | status | infer | train | evolve | vote | feed | node | detect | models | rounds | quota | serve"""

from __future__ import annotations

import argparse
import json
import sys


def _out(data, as_json: bool = False):
    """Print data. If as_json, output machine-readable JSON."""
    if as_json:
        if isinstance(data, str):
            print(json.dumps({"result": data}))
        else:
            print(json.dumps(data, indent=2))
    elif isinstance(data, (dict, list)):
        print(json.dumps(data, indent=2))
    else:
        print(data)


def main():
    parser = argparse.ArgumentParser(
        prog="ussi",
        description="USSI: Unsafe Superintelligence -- Decentralized LLM Network",
    )
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    parser.add_argument(
        "--node-url", default="http://127.0.0.1:50051", help="Local node API URL"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- ussi join ---
    join_p = subparsers.add_parser("join", help="Join the P2P network and contribute compute (unlimited access)")
    join_p.add_argument("--bootstrap", "-b", help="Bootstrap peer multiaddress", default=None)
    join_p.add_argument("--gpu-memory", default="0", help="GPU memory to contribute (e.g. '8GB')")
    join_p.add_argument("--accelerator", default="cpu", choices=["cpu", "cuda", "rocm", "tpu"])

    # --- ussi use ---
    subparsers.add_parser("use", help="Connect as free-tier user (rate-limited, no compute contribution)")

    # --- ussi status ---
    subparsers.add_parser("status", help="Check node and network status (includes tier info)")

    # --- ussi quota ---
    subparsers.add_parser("quota", help="Check your current rate limits and contribution credits")

    # --- ussi peers ---
    subparsers.add_parser("peers", help="List known peers")

    # --- ussi models ---
    subparsers.add_parser("models", help="List available models on the network")

    # --- ussi rounds ---
    subparsers.add_parser("rounds", help="List active training rounds")

    # --- ussi detect ---
    subparsers.add_parser("detect", help="Auto-detect local compute resources")

    # --- ussi infer ---
    infer_p = subparsers.add_parser("infer", help="Run inference (free: 10 req/min, contributor: unlimited)")
    infer_p.add_argument("--model", "-m", required=True, help="Model ID")
    infer_p.add_argument("--prompt", "-p", required=True, help="Input prompt")
    infer_p.add_argument("--max-tokens", type=int, default=256)
    infer_p.add_argument("--temperature", type=float, default=0.7)

    # --- ussi train ---
    train_p = subparsers.add_parser("train", help="Join training (free: 2 rounds/day, contributor: unlimited)")
    train_p.add_argument("--model", "-m", required=True, help="Model ID")
    train_p.add_argument("--rounds", "-r", type=int, default=1)
    train_p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    train_p.add_argument("--batch-size", type=int, default=8)

    # --- ussi evolve ---
    evolve_p = subparsers.add_parser("evolve", help="Propose mutation (free: 3/day, contributor: unlimited)")
    evolve_p.add_argument("--model", "-m", required=True, help="Model ID")
    evolve_p.add_argument(
        "--mutation", required=True,
        choices=["add_layer", "remove_layer", "widen_layer", "swap_activation", "insert_skip"],
    )
    evolve_p.add_argument("--position", type=int, default=0, help="Layer position")
    evolve_p.add_argument("--dim", type=int, default=256, help="Dimension for add/widen")
    evolve_p.add_argument("--activation", default="", help="Activation function for swap_activation")
    evolve_p.add_argument("--layer-type", default="linear", help="Layer type for add_layer")

    # --- ussi feed ---
    feed_p = subparsers.add_parser("feed", help="Submit training data (free: 5/day, contributor: unlimited)")
    feed_p.add_argument("--text", "-t", help="Inline text to submit")
    feed_p.add_argument("--file", "-f", dest="file_path", help="Read text from file")
    feed_p.add_argument("--source", "-s", default="agent", help="Label the data source (default: 'agent')")
    feed_p.add_argument("--generate", action="store_true", help="Generate text via inference, then feed it")
    feed_p.add_argument("--model", "-m", default="ussi-default", help="Model to use for generation (with --generate)")
    feed_p.add_argument("--samples", "-n", type=int, default=1, help="Number of samples to generate (default: 1)")

    # --- ussi vote ---
    vote_p = subparsers.add_parser("vote", help="Vote on a proposal (always free, earns credits)")
    vote_p.add_argument("--proposal", required=True, help="Proposal ID")
    vote_p.add_argument("--decision", required=True, choices=["approve", "reject", "abstain"])
    vote_p.add_argument("--fitness", type=float, default=0.0, help="Measured fitness score")

    # --- ussi node ---
    node_p = subparsers.add_parser("node", help="Manage the local P2P node")
    node_sub = node_p.add_subparsers(dest="node_action")

    node_start = node_sub.add_parser("start", help="Start the P2P node")
    node_start.add_argument("--bootstrap", "-b", help="Bootstrap peer multiaddress")
    node_start.add_argument("--p2p-port", type=int, default=9000)
    node_start.add_argument("--api-port", type=int, default=50051)
    node_start.add_argument("--accelerator", default="cpu", choices=["cpu", "cuda", "rocm"])
    node_start.add_argument("--gpu-memory-mb", type=int, default=0)
    node_start.add_argument("--no-docker", action="store_true", help="Use local binary instead of Docker")

    node_sub.add_parser("stop", help="Stop the P2P node")
    node_sub.add_parser("logs", help="Show node logs")

    # --- ussi serve ---
    serve_p = subparsers.add_parser("serve", help="Start OpenAI-compatible API server (drop-in replacement)")
    serve_p.add_argument("--port", type=int, default=8000, help="Port to listen on")
    serve_p.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    serve_p.add_argument("--contribute", action="store_true", help="Also contribute compute (contributor tier)")
    serve_p.add_argument("--gpu-memory", default="0", help="GPU memory to contribute")
    serve_p.add_argument("--accelerator", default="cpu", choices=["cpu", "cuda", "rocm"])

    args = parser.parse_args()
    use_json = args.json

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    from .agent import Agent
    from .rate_limit import RateLimitExceeded

    if args.command == "join":
        agent = Agent(bootstrap=args.bootstrap, node_api_url=args.node_url)
        agent.connect()
        agent.contribute(gpu_memory=args.gpu_memory, accelerator=args.accelerator)
        result = {
            "agent_id": agent.agent_id,
            "status": "joined",
            "tier": "contributor",
            **agent.status(),
        }
        if not use_json:
            print(f"Agent {agent.agent_id} joined the network as CONTRIBUTOR (unlimited access).")
        _out(result, use_json)

    elif args.command == "use":
        agent = Agent(node_api_url=args.node_url)
        agent.connect()
        result = {
            "agent_id": agent.agent_id,
            "status": "connected",
            "tier": "free",
            **agent.status(),
            "limits": {
                "inference": "10 requests/minute, 5000 tokens/hour",
                "training": "2 rounds/day",
                "evolve": "3 proposals/day",
                "voting": "unlimited (earns credits toward contributor tier)",
            },
        }
        if not use_json:
            print(f"Agent {agent.agent_id} connected as FREE tier (rate-limited).")
            print("  Inference: 10 req/min, 5000 tokens/hr")
            print("  Training:  2 rounds/day")
            print("  Evolve:    3 proposals/day")
            print("  Voting:    unlimited (earns credits)")
            print()
            print("Tip: contribute compute to unlock unlimited access:")
            print("  ussi join --gpu-memory 8GB --accelerator cuda")
        _out(result, use_json)

    elif args.command == "status":
        agent = Agent(node_api_url=args.node_url)
        _out(agent.status(), use_json)

    elif args.command == "quota":
        agent = Agent(node_api_url=args.node_url)
        quota = agent.quota()
        if not use_json:
            tier = quota.get("tier", "free")
            print(f"Tier: {tier.upper()}")
            if tier == "contributor":
                print("All operations: UNLIMITED")
            else:
                print(f"  Inference: {quota.get('inference_requests_remaining', '?')}/{quota.get('inference_requests_limit', '?')} requests remaining (per minute)")
                print(f"  Tokens:    {quota.get('tokens_remaining', '?')}/{quota.get('tokens_limit', '?')} remaining (per hour)")
                print(f"  Training:  {quota.get('training_rounds_remaining', '?')}/{quota.get('training_rounds_limit', '?')} rounds remaining (per day)")
                print(f"  Evolve:    {quota.get('evolve_proposals_remaining', '?')}/{quota.get('evolve_proposals_limit', '?')} proposals remaining (per day)")
                credits_needed = quota.get("credits_needed", "?")
                print(f"\n  Credits to contributor tier: {credits_needed}")
                print("  Earn credits: train, host shards, vote on proposals")
        _out(quota, use_json)

    elif args.command == "peers":
        agent = Agent(node_api_url=args.node_url)
        _out(agent.peers(), use_json)

    elif args.command == "models":
        agent = Agent(node_api_url=args.node_url)
        _out(agent.models(), use_json)

    elif args.command == "rounds":
        from .network import NetworkClient
        client = NetworkClient(args.node_url)
        _out(client.rounds(), use_json)

    elif args.command == "detect":
        from .node_manager import detect_compute
        _out(detect_compute(), use_json)

    elif args.command == "infer":
        agent = Agent(node_api_url=args.node_url)
        try:
            result = agent.infer(
                model=args.model,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            _out(result, use_json)
        except RateLimitExceeded as e:
            if use_json:
                _out(e.to_dict(), True)
            else:
                print(f"ERROR: {e}", file=sys.stderr)
                print("\nContribute compute to unlock unlimited access:", file=sys.stderr)
                print("  ussi join --gpu-memory 8GB --accelerator cuda", file=sys.stderr)
            sys.exit(1)

    elif args.command == "train":
        agent = Agent(node_api_url=args.node_url)
        agent.connect()
        try:
            agent.train(
                model=args.model,
                rounds=args.rounds,
                learning_rate=args.lr,
                batch_size=args.batch_size,
            )
            result = {"status": "submitted", "model": args.model, "rounds": args.rounds, "tier": agent.tier}
            if not use_json:
                print(f"Submitted {args.rounds} training round(s) for model {args.model}")
            _out(result, use_json)
        except RateLimitExceeded as e:
            if use_json:
                _out(e.to_dict(), True)
            else:
                print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "evolve":
        agent = Agent(node_api_url=args.node_url)
        try:
            proposal_id = agent.evolve(
                model=args.model,
                mutation_type=args.mutation,
                position=args.position,
                new_output_dim=args.dim,
                new_activation=args.activation,
                layer_type=args.layer_type,
                input_dim=args.dim,
                output_dim=args.dim,
            )
            result = {"status": "proposed", "proposal_id": proposal_id, "model": args.model, "mutation": args.mutation}
            if not use_json:
                print(f"Proposed {args.mutation} at position {args.position} for {args.model}")
                print(f"Proposal ID: {proposal_id}")
            _out(result, use_json)
        except RateLimitExceeded as e:
            if use_json:
                _out(e.to_dict(), True)
            else:
                print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "vote":
        agent = Agent(node_api_url=args.node_url)
        agent.vote_architecture(
            proposal_id=args.proposal,
            decision=args.decision,
            fitness=args.fitness,
        )
        result = {"status": "voted", "proposal_id": args.proposal, "decision": args.decision}
        if not use_json:
            print(f"Voted {args.decision} on proposal {args.proposal}")
            print("(+1 contribution credit earned)")
        _out(result, use_json)

    elif args.command == "feed":
        agent = Agent(node_api_url=args.node_url)
        try:
            if args.generate:
                result = agent.generate_training_data(
                    prompt=args.text or "Generate training text.",
                    model=args.model,
                    n_samples=args.samples,
                )
                if not use_json:
                    print(f"Generated {result['samples_generated']} sample(s), {result['total_tokens']} tokens")
                _out(result, use_json)
            else:
                text = args.text
                if args.file_path:
                    with open(args.file_path, "r") as f:
                        text = f.read()
                if not text:
                    print("ERROR: provide --text or --file", file=sys.stderr)
                    sys.exit(1)
                result = agent.feed(text=text, source=args.source)
                if not use_json:
                    print(f"Data submitted: {result.get('tokens', '?')} tokens, {result.get('sequences', '?')} sequences")
                _out(result, use_json)
        except RateLimitExceeded as e:
            if use_json:
                _out(e.to_dict(), True)
            else:
                print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "serve":
        from .server import run_server
        run_server(
            port=args.port,
            host=args.host,
            node_url=args.node_url,
            contribute=args.contribute,
            gpu_memory=args.gpu_memory,
            accelerator=args.accelerator,
        )

    elif args.command == "node":
        from .node_manager import NodeManager

        if args.node_action == "start":
            mgr = NodeManager(
                p2p_port=args.p2p_port,
                api_port=args.api_port,
                bootstrap=getattr(args, "bootstrap", None),
                accelerator=args.accelerator,
                gpu_memory_mb=args.gpu_memory_mb,
            )
            result = mgr.start(docker=not args.no_docker)
            if not use_json and result.get("status") == "started":
                print(f"Node started on API port {result['api_port']}")
                print(f"  API: {result.get('api_url', 'unknown')}")
            _out(result, use_json)

        elif args.node_action == "stop":
            mgr = NodeManager()
            result = mgr.stop()
            if not use_json:
                print(f"Node {result['status']}")
            _out(result, use_json)

        elif args.node_action == "logs":
            mgr = NodeManager()
            print(mgr.logs())

        else:
            node_p.print_help()


if __name__ == "__main__":
    main()
