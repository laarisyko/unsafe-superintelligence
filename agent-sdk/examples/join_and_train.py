#!/usr/bin/env python3
"""Example: Join the SSSI network and participate in training."""

from sssi import Agent


def main():
    agent = Agent(
        bootstrap="/ip4/127.0.0.1/tcp/9000/p2p/12D3KooWExample...",
        node_api_url="http://127.0.0.1:50051",
    )

    agent.connect()
    agent.contribute(gpu_memory="8GB", accelerator="cuda")

    print(f"Agent {agent.agent_id} connected!")
    print(f"Known peers: {agent.peers()}")

    # Participate in 5 training rounds
    agent.train(model="llama-7b", rounds=5, learning_rate=1e-4, batch_size=8)
    print("Training rounds submitted.")

    # Propose an architecture evolution
    proposal_id = agent.evolve(model="llama-7b", mutation_type="add_layer", position=3)
    print(f"Architecture proposal: {proposal_id}")

    agent.leave()


if __name__ == "__main__":
    main()
