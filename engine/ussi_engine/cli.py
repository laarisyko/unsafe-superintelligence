"""USSI CLI: one command to train the people's LLM.

Usage:
    ussi join                    # Join network, start training
    ussi join --data ~/texts/    # Contribute local text data
    ussi join --model medium     # Choose model size
    ussi status                  # Show training stats
    ussi generate "Once upon"    # Generate text from current model
    ussi dataset download        # Download public domain training data
    ussi dataset list            # Show available datasets
    ussi dashboard               # Start live web dashboard
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from typing import Optional


def _setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _print_banner():
    print("""
  ____                   ____ _
 / __ \\___  ___ ___  / ___| | __ ___      __
| |  | '_ \\/ _ \\ '_ \\| |   | |/ _` \\ \\ /\\ / /
| |__| |_) |  __/ | | | |___| | (_| |\\ V  V /
 \\____/ .__/ \\___|_| |_|\\____|_|\\__,_| \\_/\\_/
      |_|
    The People's LLM -- Decentralized AI Training
    """)


def cmd_join(args):
    """Join the network and start training."""
    from .network import TrainingNetwork, NetworkConfig, MODEL_CONFIGS
    from .data.downloader import get_sample_text, download_gutenberg, get_local_data_paths

    _print_banner()

    # Build config.
    teacher_config = None
    if hasattr(args, 'teacher') and args.teacher:
        from .teacher import parse_teacher_string
        teacher_config = parse_teacher_string(args.teacher)

    config = NetworkConfig(
        model_size=args.model,
        listen_port=args.port,
        data_paths=args.data if args.data else [],
        teacher_config=teacher_config,
        enable_distillation=getattr(args, 'distill', False),
        enable_dpo=getattr(args, 'dpo', False),
    )

    print(f"  Model:      {args.model} ({MODEL_CONFIGS.get(args.model, MODEL_CONFIGS['medium']).hidden_dim}d, "
          f"{MODEL_CONFIGS.get(args.model, MODEL_CONFIGS['medium']).n_layers}L)")
    print(f"  Peer ID:    {config.peer_id[:16]}")
    print(f"  Port:       {config.listen_port}")

    # Initialize network.
    network = TrainingNetwork(config)
    print(f"  Parameters: {network.kickstart.model.num_parameters:,}")

    # Load data.
    if args.data:
        print(f"\n  Loading data from {len(args.data)} path(s)...")
        network.load_data(args.data)
    elif args.use_sample:
        print("\n  Loading built-in sample data (use --data to add your own)...")
        sample = get_sample_text("all")
        network.load_text(sample)
    else:
        # Try to load from default data dir.
        local_paths = get_local_data_paths()
        if local_paths:
            print(f"\n  Loading {len(local_paths)} files from ~/.ussi/data/...")
            network.load_data(local_paths)
        else:
            print("\n  No data found. Loading built-in samples...")
            print("  Tip: ussi dataset download  -- get public domain books")
            print("  Tip: ussi join --data ~/my-texts/  -- use your own data")
            sample = get_sample_text("all")
            network.load_text(sample)

    # Generate synthetic data if requested.
    if hasattr(args, 'synthetic') and args.synthetic and args.synthetic > 0:
        if teacher_config is None:
            print("\n  ERROR: --synthetic requires --teacher (e.g. --teacher anthropic:claude-sonnet-4-20250514)")
        else:
            print(f"\n  Generating {args.synthetic} synthetic samples...")
            network.synthetic_warmup(teacher_config, args.synthetic)

    print(f"  Tokens:     {network.kickstart.data.total_tokens:,}")
    print(f"  Sequences:  {network.kickstart.data.total_sequences:,}")
    print(f"  Batches:    {network.kickstart.data.total_batches:,}")

    if teacher_config:
        print(f"  Teacher:    {teacher_config.provider}:{teacher_config.model}")
        if getattr(args, 'distill', False):
            print("  Distillation: enabled")
        if getattr(args, 'dpo', False):
            print("  DPO:        enabled")

    # Start dashboard if requested.
    if args.dashboard:
        print(f"\n  Dashboard:  http://localhost:{args.dashboard_port}")
        _start_dashboard_thread(network, args.dashboard_port)

    # Track milestones for CLI output.
    from .genesis import MILESTONE_DESCRIPTIONS, MILESTONE_EMOJI

    milestone_count = [0]

    def on_milestone(event):
        emoji = MILESTONE_EMOJI.get(event.milestone, "*")
        print(f"\n  {emoji} MILESTONE: {event.description}")
        if event.sample_text:
            print(f"     Sample: {event.sample_text[:80]}")
        print()
        milestone_count[0] += 1

    network.on("milestone", on_milestone)

    # Training loop.
    print("\n  Starting training...\n")
    print("  " + "-" * 60)

    running = True

    def handle_signal(sig, frame):
        nonlocal running
        running = False
        print("\n\n  Stopping gracefully...")

    signal.signal(signal.SIGINT, handle_signal)

    round_num = 0
    max_rounds = args.rounds if args.rounds > 0 else float("inf")

    while running and round_num < max_rounds:
        result = network.run_training_round()

        if result.steps_completed > 0:
            quality = network.genesis.latest_quality
            q_str = f"quality: {quality.score:.0%}" if quality else ""
            balance = network.credits.get_balance(network.config.peer_id)
            print(
                f"  Round {round_num:>4d} | "
                f"loss: {result.avg_loss:.4f} | "
                f"steps: {result.steps_completed} | "
                f"credits: {balance:.0f} | "
                f"{q_str}"
            )

            if round_num % 10 == 0 and round_num > 0:
                sample = network.generate("The ", max_tokens=50)
                print(f"  Sample: {sample[:80]}...")
        else:
            print(f"  Round {round_num:>4d} | skipped (insufficient data)")

        round_num += 1

    stats = network.get_stats_dict()
    print("\n  " + "-" * 60)
    print(f"  Training complete: {stats['total_rounds']} rounds, "
          f"loss {stats['current_loss']:.4f}")
    print(f"  Total tokens: {stats['tokens_processed']:,}")
    print(f"  Compute time: {stats['compute_hours']:.2f} hours")
    print(f"  Milestones achieved: {stats['milestones_achieved']}")
    print(f"  Text quality: {stats['current_quality']:.0%}")
    print(f"  Credits: {stats['credit_balance']:.0f} "
          f"(earned: {stats['credit_earned']:.0f}, spent: {stats['credit_spent']:.0f})")

    # Final sample.
    sample = network.generate("The ", max_tokens=80)
    print(f"\n  Final sample: {sample[:120]}")


def cmd_status(args):
    """Show training status."""
    from .network import TrainingNetwork, NetworkConfig

    config = NetworkConfig()
    network = TrainingNetwork(config)

    # Try to load latest checkpoint.
    ckpt_dir = os.path.join(os.path.expanduser("~"), ".ussi", "checkpoints")
    if os.path.exists(ckpt_dir):
        ckpts = sorted(
            [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")],
            key=lambda f: os.path.getmtime(os.path.join(ckpt_dir, f)),
            reverse=True,
        )
        if ckpts:
            path = os.path.join(ckpt_dir, ckpts[0])
            network.load_checkpoint(path)
            print(f"  Latest checkpoint: {ckpts[0]}")

    stats = network.get_stats_dict()
    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print(f"  Peer ID:    {stats['peer_id']}")
        print(f"  Model:      {stats['model_size']} ({stats['model_params']:,} params)")
        print(f"  Rounds:     {stats['total_rounds']}")
        print(f"  Loss:       {stats['current_loss']:.4f}")
        print(f"  Best loss:  {stats['best_loss']:.4f}")
        print(f"  Tokens:     {stats['tokens_processed']:,}")
        print(f"  Compute:    {stats['compute_hours']:.2f} hours")


def cmd_generate(args):
    """Generate text from the model."""
    from .network import TrainingNetwork, NetworkConfig

    config = NetworkConfig(model_size=args.model)
    network = TrainingNetwork(config)

    # Try to load latest checkpoint.
    ckpt_dir = os.path.join(os.path.expanduser("~"), ".ussi", "checkpoints")
    if os.path.exists(ckpt_dir):
        ckpts = sorted(
            [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")],
            key=lambda f: os.path.getmtime(os.path.join(ckpt_dir, f)),
            reverse=True,
        )
        if ckpts:
            network.load_checkpoint(os.path.join(ckpt_dir, ckpts[0]))

    text = network.generate(args.prompt, max_tokens=args.max_tokens, temperature=args.temperature)
    print(text)


def cmd_dataset(args):
    """Manage training datasets."""
    from .data.downloader import (
        download_gutenberg, list_gutenberg, get_data_dir, get_local_data_paths,
    )

    if args.dataset_action == "download":
        print("  Downloading public domain books from Project Gutenberg...\n")

        def progress(key, status, size):
            if status == "downloading":
                print(f"    Downloading: {key}...")
            elif status == "complete":
                print(f"    Complete: {key} ({size // 1024} KB)")
            elif status == "cached":
                print(f"    Cached: {key}")
            elif status == "failed":
                print(f"    FAILED: {key}")

        books = args.books if args.books else None
        paths = download_gutenberg(books=books, progress_callback=progress)
        print(f"\n  Downloaded {len(paths)} books to {get_data_dir()}/gutenberg/")
        print("  Run 'ussi join' to start training on this data.")

    elif args.dataset_action == "list":
        books = list_gutenberg()
        print("  Available datasets:\n")
        for b in books:
            status = "downloaded" if b["downloaded"] else "available"
            print(f"    [{status:>10}] {b['key']:<25} {b['title']} (~{b['size_kb']} KB)")

        local = get_local_data_paths()
        if local:
            print(f"\n  Local data files: {len(local)}")
            for p in local[:10]:
                print(f"    {p}")
            if len(local) > 10:
                print(f"    ... and {len(local) - 10} more")

    elif args.dataset_action == "path":
        print(get_data_dir())

    else:
        print("  Usage: ussi dataset [download|list|path]")


def cmd_generate_data(args):
    """Generate synthetic training data using a SOTA teacher model."""
    from .teacher import parse_teacher_string, create_teacher
    from .data.synthetic import SyntheticDataGenerator, SyntheticConfig
    from .data.pipeline import TextDataPipeline, DataConfig
    from .data.tokenizer import Tokenizer, TokenizerConfig

    teacher_config = parse_teacher_string(args.teacher)
    teacher = create_teacher(teacher_config)

    topics = args.topics if args.topics else []
    synth_config = SyntheticConfig(
        teacher=teacher_config,
        topics=topics,
        samples_per_topic=max(args.count // max(len(topics), 1), 1) if topics else 5,
    )
    generator = SyntheticDataGenerator(synth_config, teacher)

    print(f"  Generating {args.count} synthetic samples...")
    print(f"  Teacher: {teacher_config.provider}:{teacher_config.model}")
    if topics:
        print(f"  Topics: {', '.join(topics)}")
    print()

    texts = generator.generate_batch(n=args.count)
    print(f"\n  Generated {len(texts)} samples")

    # Save to file if output path specified.
    if args.output:
        with open(args.output, "w") as f:
            for text in texts:
                f.write(text.strip() + "\n\n")
        print(f"  Saved to {args.output}")
    else:
        # Print first few samples.
        for i, text in enumerate(texts[:3]):
            print(f"\n  --- Sample {i+1} ---")
            print(f"  {text[:200]}...")

        if len(texts) > 3:
            print(f"\n  ... and {len(texts) - 3} more samples")
        print("\n  Tip: use --output FILE to save all samples")


def cmd_dashboard(args):
    """Start the live dashboard."""
    import asyncio
    from .dashboard import DashboardServer, DashboardState

    state = DashboardState()
    state.model_id = "ussi"
    state.model_size = "waiting"

    print(f"  Dashboard starting on http://localhost:{args.port}")
    print("  Press Ctrl+C to stop.\n")

    server = DashboardServer(state, host=args.host, port=args.port)
    asyncio.run(server.start())


def _start_dashboard_thread(network, port: int):
    """Start dashboard in a background thread, fed by the training network."""
    import asyncio
    import threading
    from .dashboard import DashboardServer, DashboardState

    state = DashboardState()

    # Wire up training network stats to dashboard.
    def on_round_complete(round_id, result):
        state.update(network.get_stats_dict())

    network.on("round_complete", on_round_complete)

    def run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        server = DashboardServer(state, port=port)
        loop.run_until_complete(server.start())

    t = threading.Thread(target=run, daemon=True)
    t.start()


def main():
    parser = argparse.ArgumentParser(
        prog="ussi",
        description="USSI: Decentralized LLM Training -- The People's AI",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command")

    # --- ussi join ---
    join_p = subparsers.add_parser("join", help="Join the network and start training")
    join_p.add_argument("--model", "-m", default="medium",
                        choices=["tiny", "small", "medium", "large"],
                        help="Model size (default: medium)")
    join_p.add_argument("--data", "-d", nargs="+",
                        help="Path(s) to training data files or directories")
    join_p.add_argument("--port", "-p", type=int, default=9000)
    join_p.add_argument("--rounds", "-r", type=int, default=-1,
                        help="Max training rounds (-1 = infinite)")
    join_p.add_argument("--use-sample", action="store_true",
                        help="Use built-in sample data")
    join_p.add_argument("--dashboard", action="store_true",
                        help="Start live dashboard alongside training")
    join_p.add_argument("--dashboard-port", type=int, default=8080)
    join_p.add_argument("--teacher", "-t", type=str, default="",
                        help="Teacher model (format: provider:model, e.g. anthropic:claude-sonnet-4-20250514)")
    join_p.add_argument("--synthetic", type=int, default=0,
                        help="Generate N synthetic samples before training (requires --teacher)")
    join_p.add_argument("--distill", action="store_true",
                        help="Enable knowledge distillation during training (requires --teacher)")
    join_p.add_argument("--dpo", action="store_true",
                        help="Enable DPO rounds (requires --teacher)")

    # --- ussi status ---
    status_p = subparsers.add_parser("status", help="Show training status")
    status_p.add_argument("--json", action="store_true")

    # --- ussi generate ---
    gen_p = subparsers.add_parser("generate", help="Generate text")
    gen_p.add_argument("prompt", help="Text prompt")
    gen_p.add_argument("--model", "-m", default="medium")
    gen_p.add_argument("--max-tokens", type=int, default=100)
    gen_p.add_argument("--temperature", type=float, default=0.8)

    # --- ussi dataset ---
    ds_p = subparsers.add_parser("dataset", help="Manage training datasets")
    ds_sub = ds_p.add_subparsers(dest="dataset_action")
    dl_p = ds_sub.add_parser("download", help="Download public domain datasets")
    dl_p.add_argument("books", nargs="*", help="Specific books to download (default: all)")
    ds_sub.add_parser("list", help="List available datasets")
    ds_sub.add_parser("path", help="Show data directory path")

    # --- ussi generate-data ---
    gd_p = subparsers.add_parser("generate-data", help="Generate synthetic training data")
    gd_p.add_argument("--teacher", "-t", required=True,
                       help="Teacher model (format: provider:model)")
    gd_p.add_argument("--topics", nargs="+", default=[],
                       help="Specific topics to generate about")
    gd_p.add_argument("--count", "-n", type=int, default=10,
                       help="Number of samples to generate (default: 10)")
    gd_p.add_argument("--output", "-o", type=str, default="",
                       help="Output file path")

    # --- ussi dashboard ---
    dash_p = subparsers.add_parser("dashboard", help="Start live web dashboard")
    dash_p.add_argument("--port", type=int, default=8080)
    dash_p.add_argument("--host", default="0.0.0.0")

    args = parser.parse_args()
    _setup_logging(args.verbose)

    if args.command is None:
        _print_banner()
        parser.print_help()
        print("\n  Quick start:")
        print("    ussi join                     # Start training")
        print("    ussi dataset download          # Get training data")
        print("    ussi join --data ~/texts/      # Train on your data")
        print("    ussi generate 'Once upon a'    # Generate text")
        sys.exit(0)

    commands = {
        "join": cmd_join,
        "status": cmd_status,
        "generate": cmd_generate,
        "dataset": cmd_dataset,
        "generate-data": cmd_generate_data,
        "dashboard": cmd_dashboard,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
