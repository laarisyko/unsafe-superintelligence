# SSSI CLI Reference

## Global Flags

| Flag         | Description                          |
|--------------|--------------------------------------|
| `--json`     | Output machine-readable JSON         |
| `--node-url` | Node API URL (default: http://127.0.0.1:50051) |

## Commands

### `sssi join`
Join the P2P network and advertise compute.

| Flag            | Default | Description                      |
|-----------------|---------|----------------------------------|
| `--bootstrap`   | none    | Bootstrap peer multiaddress      |
| `--gpu-memory`  | 0       | GPU memory (e.g. "8GB")         |
| `--accelerator` | cpu     | cpu, cuda, rocm, tpu             |

### `sssi status`
Show node health and connection status.

### `sssi peers`
List connected peers as JSON array.

### `sssi models`
List models available on the network.

### `sssi rounds`
List active training rounds.

### `sssi detect`
Auto-detect local compute resources (GPU, CPU cores).

### `sssi infer`
Run inference on a network model.

| Flag            | Default | Description                      |
|-----------------|---------|----------------------------------|
| `--model`       | req     | Model ID                         |
| `--prompt`      | req     | Input text                       |
| `--max-tokens`  | 256     | Max tokens to generate           |
| `--temperature` | 0.7     | Sampling temperature             |

### `sssi train`
Propose/join decentralized training rounds.

| Flag           | Default | Description                       |
|----------------|---------|-----------------------------------|
| `--model`      | req     | Model ID                          |
| `--rounds`     | 1       | Number of rounds                  |
| `--lr`         | 0.0001  | Learning rate                     |
| `--batch-size` | 8       | Batch size                        |

### `sssi evolve`
Propose an architecture mutation.

| Flag           | Default  | Description                      |
|----------------|----------|----------------------------------|
| `--model`      | req      | Model ID                         |
| `--mutation`   | req      | add_layer, remove_layer, widen_layer, swap_activation, insert_skip |
| `--position`   | 0        | Layer position                   |
| `--dim`        | 256      | Dimension for add/widen          |
| `--activation` | ""       | Activation for swap_activation   |
| `--layer-type` | linear   | Layer type for add_layer         |

### `sssi vote`
Vote on an architecture proposal.

| Flag         | Default | Description                        |
|--------------|---------|------------------------------------|
| `--proposal` | req     | Proposal ID                        |
| `--decision` | req     | approve, reject, abstain           |
| `--fitness`  | 0.0     | Locally measured fitness score     |

### `sssi node start`
Start the local P2P node.

| Flag              | Default | Description                    |
|-------------------|---------|--------------------------------|
| `--bootstrap`     | none    | Bootstrap peer multiaddress    |
| `--p2p-port`      | 9000    | P2P listening port             |
| `--api-port`      | 50051   | HTTP API port                  |
| `--accelerator`   | cpu     | cpu, cuda, rocm                |
| `--gpu-memory-mb` | 0       | GPU memory in MB               |
| `--no-docker`     | false   | Use local binary, not Docker   |

### `sssi node stop`
Stop the local P2P node.

### `sssi node logs`
Print recent node logs.
