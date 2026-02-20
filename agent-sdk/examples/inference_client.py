#!/usr/bin/env python3
"""Example: Run inference via the USSI decentralized network."""

from ussi import Agent


def main():
    agent = Agent(node_api_url="http://127.0.0.1:50051")

    result = agent.infer(
        model="llama-7b",
        prompt="Explain decentralized machine learning in simple terms.",
        max_tokens=512,
        temperature=0.7,
    )
    print("Inference result:")
    print(result)


if __name__ == "__main__":
    main()
