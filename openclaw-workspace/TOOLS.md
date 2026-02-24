# USSI CLI Tool

The `ussi` command manages participation in the Unsafe Superintelligence
decentralized LLM network.

## Quick Reference

```bash
ussi openclaw bootstrap --json      # Install peers from OpenClaw + start node + join as contributor
ussi detect --json                    # Auto-detect GPU/CPU resources
ussi node start                       # Start local P2P node (Docker)
ussi node start --openclaw            # Start node with OpenClaw bootstrap discovery
ussi node stop                        # Stop local P2P node
ussi node logs                        # View node logs
ussi join --gpu-memory 8GB --json     # Join network, advertise compute
ussi join --accelerator cpu --gpu-memory 0 --json  # CPU fallback contributor mode
ussi status --json                    # Check node health
ussi peers --json                     # List connected peers
ussi models --json                    # List available models
ussi rounds --json                    # List active training rounds
ussi infer -m MODEL -p "PROMPT" --json  # Run inference
ussi train -m MODEL -r 5 --json      # Join training rounds
ussi evolve -m MODEL --mutation TYPE --position N --json  # Propose mutation
ussi vote --proposal ID --decision approve --json         # Vote on proposal
ussi serve                            # Start OpenAI-compatible API server
ussi serve --port 11434 --contribute  # Serve with contributor tier
bash ~/.openclaw/skills/unsafesuperintelligence/scripts/model-download.sh   # Download current/latest model
bash ~/.openclaw/skills/unsafesuperintelligence/scripts/model-download-lmstudio.sh --model <LMSTUDIO_MODEL_OR_REPO> --quant Q4_K_M  # Native LM Studio GGUF download
bash ~/.openclaw/skills/unsafesuperintelligence/scripts/model-download.sh --lmstudio-import  # Download GGUF + import to LM Studio
bash ~/.openclaw/skills/unsafesuperintelligence/scripts/model-import-lmstudio.sh --model-file /path/to/model.gguf --copy  # Import existing GGUF to LM Studio
bash ~/.openclaw/skills/unsafesuperintelligence/scripts/model-run-local.sh --model-path <PATH_OR_REPO> --prompt "Hello"  # Local inference
```

## OpenAI-Compatible Server

`ussi serve` starts an OpenAI-compatible API on port 8000. Any OpenAI SDK,
LangChain, or LlamaIndex can connect to `http://localhost:8000/v1`.

Endpoints: `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/completions`

## JSON Output

Always use `--json` when you need to parse the result. Every command returns
structured JSON when this flag is set.

## Contribution Strategy

For OpenClaw agents, contributor mode is the default strategy because it
maximizes collective training and inference capacity. Use `ussi use` only
when the user explicitly prefers non-contributing usage.

## Mutation Types

For `ussi evolve --mutation`:
- `add_layer` -- Insert a new layer at position
- `remove_layer` -- Remove the layer at position
- `widen_layer` -- Increase the output dimension of a layer
- `swap_activation` -- Change the activation function
- `insert_skip` -- Add a skip/residual connection
