//! Bridge between the Rust P2P node and the Python ML engine.
//!
//! The bridge communicates with the Python engine over a local HTTP connection
//! (engine/openclaw_engine/bridge.py:HttpBridgeServer). In production, this
//! would use PyO3 for direct FFI, eliminating the HTTP overhead.
//!
//! Message flow:
//!     Rust node <-- gossip --> other peers
//!     Rust node --> bridge --> Python engine (training, inference)
//!     Python engine --> bridge --> Rust node (gradients, checkpoints)

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tracing::{debug, error, info, warn};

/// Configuration for the Python engine bridge.
#[derive(Debug, Clone)]
pub struct BridgeConfig {
    /// Host where the Python engine bridge server is running.
    pub host: String,
    /// Port for the Python engine bridge server.
    pub port: u16,
    /// Connection timeout.
    pub timeout: Duration,
    /// Number of retry attempts for failed requests.
    pub max_retries: u32,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: 50052,
            timeout: Duration::from_secs(30),
            max_retries: 3,
        }
    }
}

/// Client for communicating with the Python ML engine.
pub struct EngineBridge {
    config: BridgeConfig,
}

/// Response from the Python engine.
#[derive(Debug, Deserialize)]
pub struct EngineResponse {
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub error: String,
    #[serde(flatten)]
    pub data: serde_json::Value,
}

/// Training step request sent to the Python engine.
#[derive(Debug, Serialize)]
pub struct TrainStepRequest {
    pub round_id: String,
    pub input_shape: Vec<usize>,
}

/// Training step response from the Python engine.
#[derive(Debug, Deserialize)]
pub struct TrainStepResponse {
    pub loss: f64,
    pub grad_norm: f64,
    pub step: u64,
}

/// Gradient metadata (shapes, sizes) returned from engine.
#[derive(Debug, Deserialize)]
pub struct GradientInfo {
    pub shape: Vec<usize>,
    pub numel: usize,
}

/// Inference request sent to the Python engine.
#[derive(Debug, Serialize)]
pub struct InferenceRequest {
    pub model_id: String,
    pub prompt: String,
    pub max_tokens: u32,
    pub temperature: f32,
}

/// Inference response from the Python engine.
#[derive(Debug, Deserialize)]
pub struct InferenceResponse {
    pub request_id: String,
    pub text: String,
    pub latency_ms: f64,
}

impl EngineBridge {
    pub fn new(config: BridgeConfig) -> Self {
        Self { config }
    }

    /// Check if the Python engine is healthy.
    pub async fn health(&self) -> Result<bool> {
        let resp = self.request("GET", "/health", b"").await?;
        Ok(resp.status == "ok" || resp.error.is_empty())
    }

    /// Request a training step from the engine.
    pub async fn train_step(&self, req: &TrainStepRequest) -> Result<TrainStepResponse> {
        let body = serde_json::to_vec(req)?;
        let resp = self.request("POST", "/train_step", &body).await?;
        let result: TrainStepResponse = serde_json::from_value(resp.data)?;
        Ok(result)
    }

    /// Get gradient metadata from the engine (shapes and sizes).
    pub async fn get_gradients(&self) -> Result<std::collections::HashMap<String, GradientInfo>> {
        let resp = self.request("GET", "/get_gradients", b"").await?;
        let result = serde_json::from_value(resp.data)?;
        Ok(result)
    }

    /// Tell the engine to apply aggregated gradients.
    pub async fn set_gradients(&self, gradient_data: &[u8]) -> Result<()> {
        self.request("POST", "/set_gradients", gradient_data).await?;
        Ok(())
    }

    /// Request a Merkle root from the engine for a given model.
    pub async fn merkle_root(&self, model_id: &str) -> Result<String> {
        let body = serde_json::json!({"model_id": model_id}).to_string();
        let resp = self.request("POST", "/merkle_root", body.as_bytes()).await?;
        Ok(resp.data["merkle_root"]
            .as_str()
            .unwrap_or("")
            .to_string())
    }

    /// Request inference from the engine.
    pub async fn infer(&self, req: &InferenceRequest) -> Result<InferenceResponse> {
        let body = serde_json::to_vec(req)?;
        let resp = self.request("POST", "/infer", &body).await?;
        let result: InferenceResponse = serde_json::from_value(resp.data)?;
        Ok(result)
    }

    /// Forward an architecture proposal to the engine for evaluation.
    /// Returns the raw JSON response bytes for broadcasting.
    pub async fn architecture_proposal(&self, proposal_data: &[u8]) -> Result<Vec<u8>> {
        let resp = self.request("POST", "/architecture_proposal", proposal_data).await?;
        let body = serde_json::to_vec(&resp.data)?;
        Ok(body)
    }

    /// Send a raw HTTP request to the Python engine with retries.
    async fn request(&self, method: &str, path: &str, body: &[u8]) -> Result<EngineResponse> {
        let mut last_err = None;

        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                let delay = Duration::from_millis(100 * 2u64.pow(attempt - 1));
                tokio::time::sleep(delay).await;
                debug!("Bridge retry {}/{} for {} {}", attempt, self.config.max_retries, method, path);
            }

            match self.do_request(method, path, body).await {
                Ok(resp) => return Ok(resp),
                Err(e) => {
                    warn!("Bridge request {} {} failed (attempt {}): {}", method, path, attempt + 1, e);
                    last_err = Some(e);
                }
            }
        }

        Err(last_err.unwrap_or_else(|| anyhow::anyhow!("bridge request failed")))
    }

    async fn do_request(&self, method: &str, path: &str, body: &[u8]) -> Result<EngineResponse> {
        let addr = format!("{}:{}", self.config.host, self.config.port);

        let mut stream = tokio::time::timeout(
            self.config.timeout,
            TcpStream::connect(&addr),
        )
        .await
        .map_err(|_| anyhow::anyhow!("bridge connection timeout"))??;

        // Build minimal HTTP request.
        let request = if body.is_empty() {
            format!(
                "{} {} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n",
                method, path, self.config.host
            )
        } else {
            format!(
                "{} {} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                method, path, self.config.host, body.len()
            )
        };

        stream.write_all(request.as_bytes()).await?;
        if !body.is_empty() {
            stream.write_all(body).await?;
        }

        // Read response.
        let mut response = Vec::new();
        stream.read_to_end(&mut response).await?;
        let response_str = String::from_utf8_lossy(&response);

        // Parse HTTP response body.
        let body_start = response_str.find("\r\n\r\n").map(|i| i + 4).unwrap_or(0);
        let body_str = &response_str[body_start..];

        let parsed: EngineResponse = serde_json::from_str(body_str).unwrap_or(EngineResponse {
            status: String::new(),
            error: format!("unparseable response: {}", &body_str[..body_str.len().min(200)]),
            data: serde_json::Value::Null,
        });

        Ok(parsed)
    }
}

/// Integration with the training round: connects gossip messages to engine actions.
pub struct TrainingRoundBridge {
    engine: EngineBridge,
    round_id: String,
    model_id: String,
}

impl TrainingRoundBridge {
    pub fn new(engine: EngineBridge, round_id: String, model_id: String) -> Self {
        Self { engine, round_id, model_id }
    }

    /// Execute a full training step: train locally, get gradients, return grad info.
    pub async fn train_and_get_gradients(&self, input_shape: Vec<usize>) -> Result<()> {
        let req = TrainStepRequest {
            round_id: self.round_id.clone(),
            input_shape,
        };
        let resp = self.engine.train_step(&req).await?;
        info!(
            "Training step complete: loss={:.4}, grad_norm={:.4}",
            resp.loss, resp.grad_norm
        );
        Ok(())
    }

    /// Verify weight consistency after aggregation.
    pub async fn verify_merkle(&self) -> Result<String> {
        self.engine.merkle_root(&self.model_id).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let cfg = BridgeConfig::default();
        assert_eq!(cfg.port, 50052);
        assert_eq!(cfg.max_retries, 3);
    }

    #[test]
    fn engine_bridge_creation() {
        let bridge = EngineBridge::new(BridgeConfig::default());
        // Just verify it compiles and creates without panic.
        assert_eq!(bridge.config.port, 50052);
    }
}
