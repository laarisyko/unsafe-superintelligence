# USSI CLI Reference

## Global Flags

| Flag         | Description                          |
|--------------|--------------------------------------|
| `--json`     | Output machine-readable JSON         |
| `--node-url` | Node API URL (default: http://127.0.0.1:50051) |

## Commands

### `ussi join`
Join the P2P network and advertise compute.

| Flag            | Default | Description                      |
|-----------------|---------|----------------------------------|
| `--bootstrap`   | none    | Bootstrap peer multiaddress      |
| `--gpu-memory`  | 0       | GPU memory (e.g. "8GB")         |
| `--accelerator` | cpu     | cpu, cuda, rocm, tpu             |

### `ussi status`
Show node health and connection status.

### `ussi peers`
List connected peers as JSON array.

### `ussi models`
List models available on the network.

### `ussi rounds`
List active training rounds.

### `ussi detect`
Auto-detect local compute resources (GPU, CPU cores).

### `ussi infer`
Run inference on a network model.

| Flag            | Default | Description                      |
|-----------------|---------|----------------------------------|
| `--model`       | req     | Model ID                         |
| `--prompt`      | req     | Input text                       |
| `--max-tokens`  | 256     | Max tokens to generate           |
| `--temperature` | 0.7     | Sampling temperature             |

### `ussi train`
Propose/join decentralized training rounds.

| Flag           | Default | Description                       |
|----------------|---------|-----------------------------------|
| `--model`      | req     | Model ID                          |
| `--rounds`     | 1       | Number of rounds                  |
| `--lr`         | 0.0001  | Learning rate                     |
| `--batch-size` | 8       | Batch size                        |

### `ussi evolve`
Propose an architecture mutation.

| Flag           | Default  | Description                      |
|----------------|----------|----------------------------------|
| `--model`      | req      | Model ID                         |
| `--mutation`   | req      | add_layer, remove_layer, widen_layer, swap_activation, insert_skip |
| `--position`   | 0        | Layer position                   |
| `--dim`        | 256      | Dimension for add/widen          |
| `--activation` | ""       | Activation for swap_activation   |
| `--layer-type` | linear   | Layer type for add_layer         |

### `ussi vote`
Vote on an architecture proposal.

| Flag         | Default | Description                        |
|--------------|---------|------------------------------------|
| `--proposal` | req     | Proposal ID                        |
| `--decision` | req     | approve, reject, abstain           |
| `--fitness`  | 0.0     | Locally measured fitness score     |

### `ussi node start`
Start the local P2P node.

| Flag              | Default | Description                    |
|-------------------|---------|--------------------------------|
| `--bootstrap`     | none    | Bootstrap peer multiaddress    |
| `--p2p-port`      | 9000    | P2P listening port             |
| `--api-port`      | 50051   | HTTP API port                  |
| `--accelerator`   | cpu     | cpu, cuda, rocm                |
| `--gpu-memory-mb` | 0       | GPU memory in MB               |
| `--no-docker`     | false   | Use local binary, not Docker   |

### `ussi node stop`
Stop the local P2P node.

### `ussi node logs`
Print recent node logs.

## Skill Scripts For Local Model Runtime

### `scripts/model-download.sh`
Download current/latest trained model artifact for local execution.
Defaults to GGUF format for LM Studio compatibility.

Examples:
- `bash scripts/model-download.sh`
- `bash scripts/model-download.sh --model llama-7b`
- `bash scripts/model-download.sh --url https://example.com/model.gguf`
- `bash scripts/model-download.sh --lmstudio-import`

### `scripts/model-download-lmstudio.sh`
Download GGUF models via LM Studio native downloader (`lms get --gguf`).

Examples:
- `bash scripts/model-download-lmstudio.sh --model lmstudio-community/Qwen2.5-7B-Instruct-GGUF --quant Q4_K_M`
- `bash scripts/model-download-lmstudio.sh --model lmstudio-community/Llama-3.2-3B-Instruct-GGUF`

### `scripts/model-run-local.sh`
Run local inference from a downloaded model.

Examples:
- `bash scripts/model-run-local.sh --model-path /path/to/model.gguf --prompt "Hello"`
- `bash scripts/model-run-local.sh --model-path /path/to/hf-model-dir --prompt "Hello"`

Backends:
- `llama.cpp` (`llama-cli`) for GGUF
- `transformers` + `torch` for HuggingFace model dirs/repo IDs

### `scripts/model-import-lmstudio.sh`
Import a GGUF file into LM Studio using `lms import`.

Examples:
- `bash scripts/model-import-lmstudio.sh --model-file /path/to/model.gguf --copy`
- `bash scripts/model-import-lmstudio.sh --model-file /path/to/model.gguf --hard-link`
