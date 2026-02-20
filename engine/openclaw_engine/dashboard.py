"""Live dashboard backend: serves network stats via HTTP + WebSocket.

Provides real-time training metrics for the public dashboard:
    - Peer count
    - Loss curve
    - Compute hours
    - Latest generated text samples
    - Model info

The dashboard is intentionally simple (no framework dependencies) so any
peer can run it. The public dashboard at openclaw.org aggregates from
multiple peers.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import struct
import time
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class DashboardState:
    """Shared state for the dashboard, updated by the training loop."""

    def __init__(self):
        self.peer_count: int = 0
        self.total_rounds: int = 0
        self.total_steps: int = 0
        self.current_loss: float = float("inf")
        self.best_loss: float = float("inf")
        self.tokens_processed: int = 0
        self.compute_hours: float = 0.0
        self.model_id: str = ""
        self.model_params: int = 0
        self.model_size: str = ""
        self.latest_sample: str = ""
        self.loss_history: List[float] = []
        self.peer_locations: List[Dict] = []  # Anonymized locations.
        self.start_time: float = time.time()
        # Genesis tracking.
        self.model_age: str = "0s"
        self.milestones: List[str] = []
        self.current_quality: float = 0.0
        self.quality_history: List[float] = []
        self.generation: int = 0
        self.mutations: int = 0
        # Credit tracking.
        self.credit_balance: float = 0.0
        self.credit_earned: float = 0.0
        self.credit_spent: float = 0.0
        self._subscribers: Set[asyncio.Queue] = set()

    def update(self, stats: dict):
        """Update dashboard state from network stats."""
        self.peer_count = stats.get("connected_peers", self.peer_count)
        self.total_rounds = stats.get("total_rounds", self.total_rounds)
        self.total_steps = stats.get("total_steps", self.total_steps)
        self.current_loss = stats.get("current_loss", self.current_loss)
        self.best_loss = stats.get("best_loss", self.best_loss)
        self.tokens_processed = stats.get("tokens_processed", self.tokens_processed)
        self.compute_hours = stats.get("compute_hours", self.compute_hours)
        self.model_id = stats.get("model_id", self.model_id)
        self.model_params = stats.get("model_params", self.model_params)
        self.model_size = stats.get("model_size", self.model_size)
        self.latest_sample = stats.get("latest_sample", self.latest_sample)
        if "loss_history" in stats:
            self.loss_history = stats["loss_history"]
        # Genesis fields.
        self.model_age = stats.get("model_age", self.model_age)
        if "milestones" in stats:
            self.milestones = stats["milestones"]
        self.current_quality = stats.get("current_quality", self.current_quality)
        if "quality_history" in stats:
            self.quality_history = stats["quality_history"]
        self.generation = stats.get("generation", self.generation)
        self.mutations = stats.get("mutations", self.mutations)
        # Credit fields.
        self.credit_balance = stats.get("credit_balance", self.credit_balance)
        self.credit_earned = stats.get("credit_earned", self.credit_earned)
        self.credit_spent = stats.get("credit_spent", self.credit_spent)

        # Notify WebSocket subscribers.
        snapshot = self.snapshot()
        dead = set()
        for q in self._subscribers:
            try:
                q.put_nowait(snapshot)
            except asyncio.QueueFull:
                dead.add(q)
        self._subscribers -= dead

    def subscribe(self) -> asyncio.Queue:
        """Subscribe to live updates. Returns a queue that receives snapshots."""
        q: asyncio.Queue = asyncio.Queue(maxsize=10)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        """Remove a subscription."""
        self._subscribers.discard(q)

    def snapshot(self) -> dict:
        """Get current state as a JSON-serializable dict."""
        return {
            "peer_count": self.peer_count,
            "total_rounds": self.total_rounds,
            "total_steps": self.total_steps,
            "current_loss": self.current_loss,
            "best_loss": self.best_loss,
            "tokens_processed": self.tokens_processed,
            "compute_hours": round(self.compute_hours, 2),
            "model_id": self.model_id,
            "model_params": self.model_params,
            "model_size": self.model_size,
            "latest_sample": self.latest_sample[:500],
            "loss_history": self.loss_history[-200:],
            "uptime_hours": round((time.time() - self.start_time) / 3600, 2),
            "timestamp": time.time(),
            # Genesis data.
            "model_age": self.model_age,
            "milestones": self.milestones,
            "current_quality": self.current_quality,
            "quality_history": self.quality_history[-200:],
            "generation": self.generation,
            "mutations": self.mutations,
            # Credit data.
            "credit_balance": round(self.credit_balance, 1),
            "credit_earned": round(self.credit_earned, 1),
            "credit_spent": round(self.credit_spent, 1),
        }


class DashboardServer:
    """Minimal HTTP + WebSocket server for the live dashboard.

    Serves:
        GET /api/stats    -> JSON snapshot of current stats
        GET /api/ws       -> WebSocket for live updates
        GET /             -> Dashboard HTML page (self-contained)
    """

    def __init__(self, state: DashboardState, host: str = "0.0.0.0", port: int = 8080):
        self.state = state
        self.host = host
        self.port = port

    async def start(self):
        """Start the dashboard server."""
        server = await asyncio.start_server(
            self._handle_connection, self.host, self.port,
        )
        logger.info("Dashboard server: http://%s:%d", self.host, self.port)
        async with server:
            await server.serve_forever()

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
    ):
        try:
            data = await asyncio.wait_for(reader.read(8192), timeout=10)
            request = data.decode("utf-8", errors="replace")
            lines = request.split("\r\n")
            first = lines[0] if lines else ""
            parts = first.split()
            method = parts[0] if parts else "GET"
            path = parts[1] if len(parts) >= 2 else "/"

            if path == "/api/stats":
                await self._send_json(writer, self.state.snapshot())
            elif path == "/api/ws":
                await self._handle_websocket(reader, writer, request)
            elif path == "/":
                await self._send_html(writer, _DASHBOARD_HTML)
            else:
                await self._send_404(writer)
        except Exception as e:
            logger.debug("Dashboard connection error: %s", e)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _send_json(self, writer: asyncio.StreamWriter, data: dict):
        body = json.dumps(data).encode()
        headers = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: application/json\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n\r\n"
        )
        writer.write(headers.encode() + body)
        await writer.drain()

    async def _send_html(self, writer: asyncio.StreamWriter, html: str):
        body = html.encode()
        headers = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/html; charset=utf-8\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n\r\n"
        )
        writer.write(headers.encode() + body)
        await writer.drain()

    async def _send_404(self, writer: asyncio.StreamWriter):
        body = b"Not Found"
        headers = (
            "HTTP/1.1 404 Not Found\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n\r\n"
        )
        writer.write(headers.encode() + body)
        await writer.drain()

    async def _handle_websocket(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        request: str,
    ):
        """Upgrade to WebSocket and stream live updates."""
        # Parse WebSocket key from headers.
        ws_key = ""
        for line in request.split("\r\n"):
            if line.lower().startswith("sec-websocket-key:"):
                ws_key = line.split(":", 1)[1].strip()

        if not ws_key:
            await self._send_json(writer, {"error": "not a websocket request"})
            return

        # WebSocket handshake.
        magic = "258EAFA5-E914-47DA-95CA-5AB5DC11BE85"
        accept = hashlib.sha1((ws_key + magic).encode()).digest()
        import base64
        accept_b64 = base64.b64encode(accept).decode()

        handshake = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept_b64}\r\n\r\n"
        )
        writer.write(handshake.encode())
        await writer.drain()

        # Subscribe to updates.
        queue = self.state.subscribe()

        # Send initial snapshot.
        await self._ws_send(writer, json.dumps(self.state.snapshot()))

        try:
            while True:
                try:
                    snapshot = await asyncio.wait_for(queue.get(), timeout=30)
                    await self._ws_send(writer, json.dumps(snapshot))
                except asyncio.TimeoutError:
                    # Send ping to keep alive.
                    await self._ws_send(writer, json.dumps({"type": "ping"}))
        except (ConnectionError, OSError):
            pass
        finally:
            self.state.unsubscribe(queue)

    async def _ws_send(self, writer: asyncio.StreamWriter, message: str):
        """Send a WebSocket text frame."""
        payload = message.encode()
        length = len(payload)

        if length < 126:
            header = bytes([0x81, length])
        elif length < 65536:
            header = bytes([0x81, 126]) + struct.pack(">H", length)
        else:
            header = bytes([0x81, 127]) + struct.pack(">Q", length)

        writer.write(header + payload)
        await writer.drain()


# Self-contained dashboard HTML (no external dependencies).
_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OpenClaw - People's LLM Training Dashboard</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
       background: #0a0a0f; color: #e0e0e0; min-height: 100vh; }
.header { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
           padding: 24px 32px; border-bottom: 1px solid #2a2a4a; }
.header h1 { font-size: 24px; color: #fff; }
.header .subtitle { color: #8888aa; font-size: 14px; margin-top: 4px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px; padding: 24px 32px; }
.card { background: #12121f; border: 1px solid #2a2a4a; border-radius: 12px;
        padding: 20px; }
.card .label { color: #6666aa; font-size: 12px; text-transform: uppercase;
               letter-spacing: 1px; }
.card .value { font-size: 32px; font-weight: 700; color: #fff; margin-top: 8px; }
.card .unit { font-size: 14px; color: #8888aa; }
.chart-container { padding: 0 32px 24px; }
.chart-card { background: #12121f; border: 1px solid #2a2a4a; border-radius: 12px;
              padding: 20px; }
.chart-card h2 { font-size: 16px; color: #8888aa; margin-bottom: 16px; }
canvas { width: 100%; height: 200px; }
.sample { padding: 0 32px 24px; }
.sample-card { background: #12121f; border: 1px solid #2a2a4a; border-radius: 12px;
               padding: 20px; }
.sample-card h2 { font-size: 16px; color: #8888aa; margin-bottom: 12px; }
.sample-text { font-family: 'Courier New', monospace; font-size: 14px;
               line-height: 1.6; color: #aaaacc; white-space: pre-wrap;
               word-break: break-word; }
.status { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
          background: #44ff44; margin-right: 8px; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }
.credit-card { border-color: #33aa33; }
.credit-card .value { color: #44ff44; }
.credit-detail { color: #448844; font-size: 10px; margin-top: 4px; }
.genesis-section { padding: 0 32px 24px; }
.genesis-card { background: linear-gradient(135deg, #0d0d1a 0%, #12121f 100%);
                border: 1px solid #3333aa; border-radius: 12px; padding: 20px; }
.genesis-card h2 { font-size: 16px; color: #8888cc; margin-bottom: 16px; }
.quality-bar-container { display: flex; align-items: center; gap: 12px; margin-bottom: 16px; }
.quality-label { color: #6666aa; font-size: 12px; text-transform: uppercase; min-width: 80px; }
.quality-bar { flex: 1; height: 12px; background: #1a1a3a; border-radius: 6px; overflow: hidden; }
.quality-fill { height: 100%; background: linear-gradient(90deg, #ff4444, #ffaa00, #44ff44);
                border-radius: 6px; transition: width 0.5s ease; width: 0%; }
.quality-value { color: #aaaacc; font-size: 14px; min-width: 40px; }
.milestones { display: flex; flex-direction: column; gap: 8px; }
.milestone { display: flex; align-items: center; gap: 12px; padding: 8px 12px;
             background: #1a1a2e; border-radius: 8px; border-left: 3px solid #6644ff; }
.milestone .ms-icon { font-size: 20px; }
.milestone .ms-text { flex: 1; color: #ccccee; font-size: 13px; }
.milestone .ms-age { color: #6666aa; font-size: 11px; }
.milestone-empty { color: #444466; font-style: italic; font-size: 13px; }
.milestone.new { animation: milestone-glow 2s ease; }
@keyframes milestone-glow { 0% { background: #2a2a5e; } 100% { background: #1a1a2e; } }
.footer { text-align: center; padding: 32px; color: #444466; font-size: 12px; }
</style>
</head>
<body>
<div class="header">
  <h1><span class="status"></span>OpenClaw - People's LLM</h1>
  <div class="subtitle">Decentralized training in progress. Owned by everyone, controlled by no one.</div>
</div>

<div class="grid">
  <div class="card">
    <div class="label">Active Peers</div>
    <div class="value" id="peers">-</div>
  </div>
  <div class="card">
    <div class="label">Training Rounds</div>
    <div class="value" id="rounds">-</div>
  </div>
  <div class="card">
    <div class="label">Current Loss</div>
    <div class="value" id="loss">-</div>
  </div>
  <div class="card">
    <div class="label">Best Loss</div>
    <div class="value" id="best-loss">-</div>
  </div>
  <div class="card">
    <div class="label">Tokens Processed</div>
    <div class="value" id="tokens">-</div>
    <div class="unit" id="tokens-unit"></div>
  </div>
  <div class="card">
    <div class="label">Compute Hours</div>
    <div class="value" id="compute">-</div>
  </div>
  <div class="card">
    <div class="label">Model Parameters</div>
    <div class="value" id="params">-</div>
    <div class="unit" id="params-unit"></div>
  </div>
  <div class="card">
    <div class="label">Model Age</div>
    <div class="value" id="age">-</div>
  </div>
  <div class="card credit-card">
    <div class="label">Credits</div>
    <div class="value" id="credits">-</div>
    <div class="unit credit-detail">earned: <span id="credit-earned">0</span> | spent: <span id="credit-spent">0</span></div>
  </div>
</div>

<div class="genesis-section">
  <div class="genesis-card">
    <h2>Genesis Timeline — Watching Intelligence Emerge</h2>
    <div class="quality-bar-container">
      <div class="quality-label">Text Quality</div>
      <div class="quality-bar"><div class="quality-fill" id="quality-fill"></div></div>
      <div class="quality-value" id="quality-value">0%</div>
    </div>
    <div id="milestones" class="milestones">
      <div class="milestone-empty">Waiting for first milestone...</div>
    </div>
  </div>
</div>

<div class="chart-container">
  <div class="chart-card">
    <h2>Training Loss Over Time</h2>
    <canvas id="loss-chart"></canvas>
  </div>
</div>

<div class="chart-container">
  <div class="chart-card">
    <h2>Text Quality Score Over Time</h2>
    <canvas id="quality-chart"></canvas>
  </div>
</div>

<div class="sample">
  <div class="sample-card">
    <h2>Latest Generated Sample</h2>
    <div class="sample-text" id="sample">Waiting for first training round...</div>
  </div>
</div>

<div class="footer">
  OpenClaw: BitTorrent for AI training. A million volunteers training one model, owned by everyone, controlled by no one.
</div>

<script>
function fmt(n) {
  if (n >= 1e9) return (n/1e9).toFixed(1) + 'B';
  if (n >= 1e6) return (n/1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n/1e3).toFixed(1) + 'K';
  return n.toString();
}

var msIcons = {
  'first_nonrandom': '🌱', 'first_real_word': '📝', 'first_word_pair': '🤝',
  'first_phrase': '💬', 'first_punctuation': '✏️', 'first_sentence': '📖',
  'first_paragraph': '📚', 'first_coherent': '🧠', 'first_mutation': '🧬',
  'loss_below_5': '📉', 'loss_below_4': '📉', 'loss_below_3': '📉',
  'loss_below_2': '📉', 'loss_below_1': '🏆', 'rounds_10': '🔄',
  'rounds_100': '🔄', 'rounds_1000': '🔄', 'peers_10': '👥',
  'peers_100': '🏘️', 'peers_1000': '🏙️', 'peers_10000': '🌍'
};
var prevMilestones = 0;

function updateUI(data) {
  document.getElementById('peers').textContent = data.peer_count || 0;
  document.getElementById('rounds').textContent = fmt(data.total_rounds || 0);
  document.getElementById('loss').textContent = data.current_loss < 999
    ? data.current_loss.toFixed(3) : '-';
  document.getElementById('best-loss').textContent = data.best_loss < 999
    ? data.best_loss.toFixed(3) : '-';
  document.getElementById('tokens').textContent = fmt(data.tokens_processed || 0);
  document.getElementById('compute').textContent = (data.compute_hours || 0).toFixed(1);
  document.getElementById('params').textContent = fmt(data.model_params || 0);
  document.getElementById('age').textContent = data.model_age || '-';
  document.getElementById('credits').textContent = fmt(data.credit_balance || 0);
  document.getElementById('credit-earned').textContent = fmt(data.credit_earned || 0);
  document.getElementById('credit-spent').textContent = fmt(data.credit_spent || 0);
  if (data.latest_sample) {
    document.getElementById('sample').textContent = data.latest_sample;
  }
  // Quality bar.
  var q = data.current_quality || 0;
  document.getElementById('quality-fill').style.width = (q * 100) + '%';
  document.getElementById('quality-value').textContent = (q * 100).toFixed(0) + '%';
  // Milestones.
  if (data.milestones && data.milestones.length > 0) {
    var el = document.getElementById('milestones');
    var isNew = data.milestones.length > prevMilestones;
    prevMilestones = data.milestones.length;
    el.innerHTML = '';
    data.milestones.slice().reverse().forEach(function(ms, i) {
      var d = document.createElement('div');
      d.className = 'milestone' + (i === 0 && isNew ? ' new' : '');
      var icon = msIcons[ms] || '⭐';
      d.innerHTML = '<span class="ms-icon">' + icon + '</span>' +
        '<span class="ms-text">' + ms.replace(/_/g, ' ') + '</span>';
      el.appendChild(d);
    });
  }
  if (data.loss_history && data.loss_history.length > 0) {
    drawChart(data.loss_history);
  }
  if (data.quality_history && data.quality_history.length > 0) {
    drawQualityChart(data.quality_history);
  }
}

function drawChart(losses) {
  var canvas = document.getElementById('loss-chart');
  var ctx = canvas.getContext('2d');
  canvas.width = canvas.offsetWidth * 2;
  canvas.height = 400;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (losses.length < 2) return;

  var min = Math.min.apply(null, losses);
  var max = Math.max.apply(null, losses);
  var range = max - min || 1;
  var pad = 40;
  var w = canvas.width - pad * 2;
  var h = canvas.height - pad * 2;

  // Grid lines.
  ctx.strokeStyle = '#1a1a3a';
  ctx.lineWidth = 1;
  for (var i = 0; i <= 4; i++) {
    var y = pad + (h * i / 4);
    ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(pad + w, y); ctx.stroke();
    ctx.fillStyle = '#444466';
    ctx.font = '20px sans-serif';
    ctx.fillText((max - range * i / 4).toFixed(2), 0, y + 6);
  }

  // Loss curve.
  ctx.strokeStyle = '#6644ff';
  ctx.lineWidth = 3;
  ctx.beginPath();
  for (var i = 0; i < losses.length; i++) {
    var x = pad + (w * i / (losses.length - 1));
    var y = pad + h - (h * (losses[i] - min) / range);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function drawQualityChart(scores) {
  var canvas = document.getElementById('quality-chart');
  var ctx = canvas.getContext('2d');
  canvas.width = canvas.offsetWidth * 2;
  canvas.height = 400;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (scores.length < 2) return;
  var pad = 40;
  var w = canvas.width - pad * 2;
  var h = canvas.height - pad * 2;
  // Grid lines.
  ctx.strokeStyle = '#1a1a3a'; ctx.lineWidth = 1;
  for (var i = 0; i <= 4; i++) {
    var y = pad + (h * i / 4);
    ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(pad + w, y); ctx.stroke();
    ctx.fillStyle = '#444466'; ctx.font = '20px sans-serif';
    ctx.fillText((1.0 - i / 4).toFixed(2), 0, y + 6);
  }
  // Quality curve (green gradient).
  ctx.strokeStyle = '#44cc44'; ctx.lineWidth = 3;
  ctx.beginPath();
  for (var i = 0; i < scores.length; i++) {
    var x = pad + (w * i / (scores.length - 1));
    var y = pad + h - (h * scores[i]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

// Connect to stats API.
function fetchStats() {
  fetch('/api/stats')
    .then(function(r) { return r.json(); })
    .then(updateUI)
    .catch(function() {});
}

// Try WebSocket first, fallback to polling.
function connectWS() {
  try {
    var ws = new WebSocket('ws://' + location.host + '/api/ws');
    ws.onmessage = function(e) {
      var data = JSON.parse(e.data);
      if (data.type !== 'ping') updateUI(data);
    };
    ws.onclose = function() { setTimeout(connectWS, 5000); };
    ws.onerror = function() { ws.close(); };
  } catch(e) {
    setInterval(fetchStats, 5000);
  }
}

fetchStats();
connectWS();
</script>
</body>
</html>"""
