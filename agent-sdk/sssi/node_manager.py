"""Node lifecycle manager -- start, stop, and monitor the local P2P node."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)

DOCKER_IMAGE = "ghcr.io/supersafesuperintelligence/node:latest"
CONTAINER_NAME = "sssi-node"


class NodeManager:
    """Manages the local SSSI P2P node (via Docker or direct binary)."""

    def __init__(
        self,
        p2p_port: int = 9000,
        api_port: int = 50051,
        bootstrap: Optional[str] = None,
        accelerator: str = "cpu",
        gpu_memory_mb: int = 0,
    ):
        self.p2p_port = p2p_port
        self.api_port = api_port
        self.bootstrap = bootstrap
        self.accelerator = accelerator
        self.gpu_memory_mb = gpu_memory_mb

    def start(self, docker: bool = True) -> Dict:
        """Start the P2P node.

        Args:
            docker: If True, start via Docker. If False, use local binary.

        Returns:
            Dict with status information.
        """
        if self.is_running():
            return {"status": "already_running", "container": CONTAINER_NAME}

        if docker:
            return self._start_docker()
        return self._start_binary()

    def stop(self) -> Dict:
        """Stop the P2P node."""
        if not self.is_running():
            return {"status": "not_running"}

        try:
            subprocess.run(
                ["docker", "stop", CONTAINER_NAME],
                capture_output=True, timeout=30,
            )
            subprocess.run(
                ["docker", "rm", CONTAINER_NAME],
                capture_output=True, timeout=10,
            )
            return {"status": "stopped"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def is_running(self) -> bool:
        """Check if the node container/process is running."""
        try:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER_NAME],
                capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip() == "true"
        except Exception:
            return False

    def logs(self, tail: int = 50) -> str:
        """Fetch recent node logs."""
        try:
            result = subprocess.run(
                ["docker", "logs", "--tail", str(tail), CONTAINER_NAME],
                capture_output=True, text=True, timeout=10,
            )
            return result.stdout + result.stderr
        except Exception as e:
            return f"[error fetching logs: {e}]"

    def _start_docker(self) -> Dict:
        """Start node via Docker."""
        if not shutil.which("docker"):
            return {"status": "error", "error": "docker not found in PATH"}

        cmd = [
            "docker", "run", "-d",
            "--name", CONTAINER_NAME,
            "-p", f"{self.p2p_port}:9000",
            "-p", f"{self.api_port}:50051",
        ]

        if self.accelerator == "cuda":
            cmd.extend(["--gpus", "all"])

        cmd.extend([
            DOCKER_IMAGE,
            "openclaw-node",
            "--port", "9000",
            "--api-port", "50051",
            "--gpu-memory-mb", str(self.gpu_memory_mb),
            "--accelerator", self.accelerator,
        ])

        if self.bootstrap:
            cmd.extend(["--bootstrap", self.bootstrap])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                # Wait a moment for the node to initialize
                time.sleep(2)
                return {
                    "status": "started",
                    "container": CONTAINER_NAME,
                    "p2p_port": self.p2p_port,
                    "api_port": self.api_port,
                    "api_url": f"http://127.0.0.1:{self.api_port}",
                }
            return {"status": "error", "error": result.stderr.strip()}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _start_binary(self) -> Dict:
        """Start node via local binary."""
        binary = shutil.which("openclaw-node") or shutil.which("sssi-node")
        if not binary:
            return {"status": "error", "error": "No node binary found. Install via Docker or build from source."}

        cmd = [
            binary,
            "--port", str(self.p2p_port),
            "--api-port", str(self.api_port),
            "--gpu-memory-mb", str(self.gpu_memory_mb),
            "--accelerator", self.accelerator,
        ]
        if self.bootstrap:
            cmd.extend(["--bootstrap", self.bootstrap])

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(2)
            if proc.poll() is None:
                return {
                    "status": "started",
                    "pid": proc.pid,
                    "p2p_port": self.p2p_port,
                    "api_port": self.api_port,
                    "api_url": f"http://127.0.0.1:{self.api_port}",
                }
            return {"status": "error", "error": "Process exited immediately"}
        except Exception as e:
            return {"status": "error", "error": str(e)}


def detect_compute() -> Dict:
    """Auto-detect available compute resources."""
    info: Dict = {
        "accelerator": "cpu",
        "gpu_memory_mb": 0,
        "gpu_name": None,
        "cpu_cores": os.cpu_count() or 1,
    }

    # Try nvidia-smi
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                line = result.stdout.strip().split("\n")[0]
                parts = line.split(", ")
                if len(parts) == 2:
                    info["gpu_name"] = parts[0].strip()
                    info["gpu_memory_mb"] = int(float(parts[1].strip()))
                    info["accelerator"] = "cuda"
        except Exception:
            pass

    # Try torch
    if info["accelerator"] == "cpu":
        try:
            import torch
            if torch.cuda.is_available():
                info["accelerator"] = "cuda"
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_memory_mb"] = torch.cuda.get_device_properties(0).total_mem // (1024 * 1024)
        except ImportError:
            pass

    return info
