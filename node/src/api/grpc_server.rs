use anyhow::Result;
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::{Arc, RwLock};
use tokio::sync::mpsc;
use tracing::info;

use crate::bridge::{EngineBridge, InferenceRequest as BridgeInferRequest};
use crate::consensus::shard_map::{SharedShardMap, ShardMap};
use crate::network::SwarmCommand;

/// Shared state for the API server to access scheduler info.
#[derive(Default, Clone)]
pub struct ApiSharedState {
    pub active_rounds: Arc<RwLock<Vec<RoundInfo>>>,
    pub peer_count: Arc<RwLock<usize>>,
    pub engine_connected: Arc<RwLock<bool>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RoundInfo {
    pub round_id: String,
    pub model_id: String,
    pub phase: String,
    pub participants: usize,
    pub started_at_ms: u64,
}

/// Simple JSON-over-HTTP API server for the OpenClaw node.
///
/// In a full implementation this would use tonic gRPC with the protobuf
/// service definitions. For the initial implementation we use a lightweight
/// HTTP/JSON API with tokio.
pub async fn run(
    port: u16,
    command_tx: mpsc::Sender<SwarmCommand>,
    shard_map: SharedShardMap,
    engine: Option<Arc<EngineBridge>>,
    shared_state: ApiSharedState,
) -> Result<()> {
    let addr: SocketAddr = ([0, 0, 0, 0], port).into();
    let listener = tokio::net::TcpListener::bind(addr).await?;
    info!("API server listening on {}", addr);

    loop {
        let (stream, peer_addr) = listener.accept().await?;
        let cmd_tx = command_tx.clone();
        let smap = shard_map.clone();
        let eng = engine.clone();
        let state = shared_state.clone();

        tokio::spawn(async move {
            if let Err(e) = handle_connection(stream, cmd_tx, smap, eng, state).await {
                tracing::debug!("API connection from {} error: {}", peer_addr, e);
            }
        });
    }
}

async fn handle_connection(
    mut stream: tokio::net::TcpStream,
    command_tx: mpsc::Sender<SwarmCommand>,
    shard_map: SharedShardMap,
    engine: Option<Arc<EngineBridge>>,
    shared_state: ApiSharedState,
) -> Result<()> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let mut buf = vec![0u8; 8192];
    let n = stream.read(&mut buf).await?;
    let request = String::from_utf8_lossy(&buf[..n]);

    // Parse the first line to get method + path.
    let first_line = request.lines().next().unwrap_or("");
    let parts: Vec<&str> = first_line.split_whitespace().collect();
    let (method, path) = if parts.len() >= 2 {
        (parts[0], parts[1])
    } else {
        ("GET", "/")
    };

    let (status, body) = match (method, path) {
        ("GET", "/health") => ("200 OK", serde_json::json!({"status": "ok"}).to_string()),

        ("GET", "/peers") => {
            // Report shard map entries as a proxy for known peers.
            let map = shard_map.read().unwrap();
            let entries: Vec<_> = map.entries().values().collect();
            ("200 OK", serde_json::to_string(&entries).unwrap_or_default())
        }

        ("GET", "/shards") => {
            let map = shard_map.read().unwrap();
            ("200 OK", serde_json::to_string(&map).unwrap_or_default())
        }

        ("GET", "/status") => {
            let peer_count = *shared_state.peer_count.read().unwrap();
            let engine_connected = *shared_state.engine_connected.read().unwrap();
            let response = serde_json::json!({
                "version": env!("CARGO_PKG_VERSION"),
                "peer_count": peer_count,
                "engine_connected": engine_connected,
                "status": "running",
            });
            ("200 OK", response.to_string())
        }

        ("GET", "/rounds") => {
            let rounds = shared_state.active_rounds.read().unwrap();
            ("200 OK", serde_json::to_string(&*rounds).unwrap_or_default())
        }

        ("POST", "/publish") => {
            // Expect JSON body: {"topic": "...", "data": "base64..."}
            let body_start = request.find("\r\n\r\n").map(|i| i + 4).unwrap_or(n);
            let body_str = &request[body_start..];
            match serde_json::from_str::<PublishRequest>(body_str) {
                Ok(req) => {
                    let data = base64_decode(&req.data);
                    let _ = command_tx
                        .send(SwarmCommand::Publish {
                            topic: req.topic,
                            data,
                        })
                        .await;
                    ("200 OK", serde_json::json!({"published": true}).to_string())
                }
                Err(e) => (
                    "400 Bad Request",
                    serde_json::json!({"error": e.to_string()}).to_string(),
                ),
            }
        }

        ("POST", "/infer") => {
            let body_start = request.find("\r\n\r\n").map(|i| i + 4).unwrap_or(n);
            let body_str = &request[body_start..];
            match serde_json::from_str::<InferRequest>(body_str) {
                Ok(req) => {
                    // Forward to Python engine via bridge if available.
                    if let Some(ref eng) = engine {
                        let bridge_req = BridgeInferRequest {
                            model_id: req.model_id.clone(),
                            prompt: req.prompt.clone(),
                            max_tokens: 256,
                            temperature: 0.7,
                        };
                        match eng.infer(&bridge_req).await {
                            Ok(resp) => {
                                let response = serde_json::json!({
                                    "request_id": req.request_id,
                                    "model_id": req.model_id,
                                    "text": resp.text,
                                    "latency_ms": resp.latency_ms,
                                    "status": "ok"
                                });
                                ("200 OK", response.to_string())
                            }
                            Err(e) => {
                                let response = serde_json::json!({
                                    "request_id": req.request_id,
                                    "model_id": req.model_id,
                                    "text": "",
                                    "error": format!("engine inference failed: {}", e),
                                    "status": "error"
                                });
                                ("200 OK", response.to_string())
                            }
                        }
                    } else {
                        let response = serde_json::json!({
                            "request_id": req.request_id,
                            "model_id": req.model_id,
                            "text": "",
                            "error": "engine bridge not connected",
                            "status": "pending_engine_integration"
                        });
                        ("200 OK", response.to_string())
                    }
                }
                Err(e) => (
                    "400 Bad Request",
                    serde_json::json!({"error": e.to_string()}).to_string(),
                ),
            }
        }

        _ => ("404 Not Found", serde_json::json!({"error": "not found"}).to_string()),
    };

    let response = format!(
        "HTTP/1.1 {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        status,
        body.len(),
        body
    );
    stream.write_all(response.as_bytes()).await?;
    Ok(())
}

#[derive(Deserialize)]
struct PublishRequest {
    topic: String,
    data: String,
}

#[derive(Deserialize)]
struct InferRequest {
    request_id: String,
    model_id: String,
    prompt: String,
}

fn base64_decode(input: &str) -> Vec<u8> {
    base64::engine::general_purpose::STANDARD
        .decode(input)
        .unwrap_or_else(|_| input.as_bytes().to_vec())
}
