pub mod work_queue;

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::api::grpc_server::{ApiSharedState, RoundInfo};
use crate::bridge::{EngineBridge, InferenceRequest, TrainStepRequest};
use crate::consensus::shard_map::{SharedReputationTable, SharedShardMap, ShardMap};
use crate::network::gossip;
use crate::network::{SwarmCommand, SwarmEvent2};

/// Training round timeout in milliseconds (5 minutes).
const ROUND_TIMEOUT_MS: u64 = 5 * 60 * 1000;

/// Scheduler state tracking ongoing training rounds and peer health.
struct SchedulerState {
    /// Known live peers and their last heartbeat time.
    live_peers: HashMap<String, u64>,
    /// Active training rounds this node is participating in.
    active_rounds: HashMap<String, RoundState>,
    /// Engine bridge for communicating with the Python ML engine.
    engine: Option<Arc<EngineBridge>>,
    /// Shared state exposed to the API server.
    api_state: ApiSharedState,
}

#[derive(Debug, Clone)]
struct RoundState {
    round_id: String,
    model_id: String,
    participants: Vec<String>,
    phase: RoundPhase,
    /// Peers that have submitted gradients this round.
    gradient_submissions: Vec<String>,
    /// Merkle root after aggregation (if complete).
    merkle_root: Option<String>,
    /// Timestamp when this round started.
    started_at_ms: u64,
}

#[derive(Debug, Clone, PartialEq)]
enum RoundPhase {
    Proposed,
    Joining,
    Computing,
    Aggregating,
    Checkpointing,
    Complete,
    Failed,
}

impl RoundPhase {
    fn as_str(&self) -> &str {
        match self {
            RoundPhase::Proposed => "proposed",
            RoundPhase::Joining => "joining",
            RoundPhase::Computing => "computing",
            RoundPhase::Aggregating => "aggregating",
            RoundPhase::Checkpointing => "checkpointing",
            RoundPhase::Complete => "complete",
            RoundPhase::Failed => "failed",
        }
    }
}

/// Main scheduler loop. Reacts to network events and dispatches work.
pub async fn run(
    mut event_rx: mpsc::Receiver<SwarmEvent2>,
    command_tx: mpsc::Sender<SwarmCommand>,
    shard_map: SharedShardMap,
    reputation: SharedReputationTable,
    engine: Option<Arc<EngineBridge>>,
    api_state: ApiSharedState,
) {
    let engine_available = engine.is_some();
    if engine_available {
        info!("Scheduler: engine bridge available");
    } else {
        info!("Scheduler: engine bridge not available; training will be deferred");
    }

    let mut state = SchedulerState {
        live_peers: HashMap::new(),
        active_rounds: HashMap::new(),
        engine,
        api_state,
    };

    info!("Scheduler started");

    // Create a cleanup interval for round timeouts.
    let mut cleanup_interval = tokio::time::interval(tokio::time::Duration::from_secs(30));

    loop {
        tokio::select! {
            event = event_rx.recv() => {
                match event {
                    Some(evt) => handle_event(&mut state, &command_tx, &shard_map, &reputation, evt).await,
                    None => break,
                }
            }
            _ = cleanup_interval.tick() => {
                cleanup_timed_out_rounds(&mut state, &command_tx).await;
                sync_api_state(&state);
            }
        }
    }

    info!("Scheduler stopped");
}

async fn handle_event(
    state: &mut SchedulerState,
    command_tx: &mpsc::Sender<SwarmCommand>,
    shard_map: &SharedShardMap,
    reputation: &SharedReputationTable,
    event: SwarmEvent2,
) {
    match event {
        SwarmEvent2::PeerDiscovered(peer_id) => {
            let pid = peer_id.to_string();
            let now = now_ms();
            state.live_peers.insert(pid.clone(), now);

            if let Ok(mut rep) = reputation.write() {
                rep.record_heartbeat(&pid, now);
            }

            // Update API shared state.
            *state.api_state.peer_count.write().unwrap() = state.live_peers.len();

            info!("Peer discovered: {} (total: {})", pid, state.live_peers.len());
        }

        SwarmEvent2::PeerExpired(peer_id) => {
            let pid = peer_id.to_string();
            state.live_peers.remove(&pid);

            // Remove peer's shards from the map.
            if let Ok(mut map) = shard_map.write() {
                map.remove_peer(&pid);
            }

            // Update API shared state.
            *state.api_state.peer_count.write().unwrap() = state.live_peers.len();

            info!("Peer expired: {} (total: {})", pid, state.live_peers.len());
        }

        SwarmEvent2::GossipMessage { source, topic, data } => {
            let domain = gossip::topic_domain(&topic);
            debug!(
                "Gossip from {} on {}: {} bytes",
                source,
                domain,
                data.len()
            );

            match domain {
                "heartbeat" => {
                    handle_heartbeat(state, reputation, &source.to_string(), &data);
                }
                "shard_map" => {
                    handle_shard_map_update(shard_map, &data);
                }
                "training" => {
                    handle_training_message(state, command_tx, &data).await;
                }
                "gradient" => {
                    handle_gradient_message(
                        state,
                        command_tx,
                        reputation,
                        &source.to_string(),
                        &data,
                    )
                    .await;
                }
                "checkpoint" => {
                    handle_checkpoint_message(
                        state,
                        reputation,
                        &source.to_string(),
                        &data,
                    );
                }
                "architecture" => {
                    handle_architecture_message(state, command_tx, &source.to_string(), &data).await;
                }
                _ => {
                    warn!("Unknown gossip topic: {}", topic);
                }
            }
        }
    }
}

fn handle_heartbeat(
    state: &mut SchedulerState,
    reputation: &SharedReputationTable,
    peer_id: &str,
    data: &[u8],
) {
    if let Ok(hb) = serde_json::from_slice::<serde_json::Value>(data) {
        let ts = hb
            .get("timestamp_ms")
            .and_then(|v| v.as_u64())
            .unwrap_or_else(now_ms);
        state.live_peers.insert(peer_id.to_string(), ts);

        if let Ok(mut rep) = reputation.write() {
            rep.record_heartbeat(peer_id, ts);
        }
    }
}

fn handle_shard_map_update(shard_map: &SharedShardMap, data: &[u8]) {
    if let Some(remote) = ShardMap::from_bytes(data) {
        if let Ok(mut local) = shard_map.write() {
            local.merge(&remote);
            debug!(
                "Shard map merged, now {} entries",
                local.entries().len()
            );
        }
    }
}

async fn handle_training_message(
    state: &mut SchedulerState,
    command_tx: &mpsc::Sender<SwarmCommand>,
    data: &[u8],
) {
    // Parse training protocol messages.
    if let Ok(msg) = serde_json::from_slice::<serde_json::Value>(data) {
        let msg_type = msg
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        match msg_type {
            "proposal" => {
                let round_id = msg
                    .get("round_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let model_id = msg
                    .get("model_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                info!(
                    "Received training proposal: round={}, model={}",
                    round_id, model_id
                );
                state.active_rounds.insert(
                    round_id.clone(),
                    RoundState {
                        round_id,
                        model_id,
                        participants: vec![],
                        phase: RoundPhase::Proposed,
                        gradient_submissions: vec![],
                        merkle_root: None,
                        started_at_ms: now_ms(),
                    },
                );
            }
            "join" => {
                let round_id = msg
                    .get("round_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let peer_id = msg
                    .get("peer_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if let Some(round) = state.active_rounds.get_mut(round_id) {
                    if round.phase == RoundPhase::Proposed || round.phase == RoundPhase::Joining {
                        round.participants.push(peer_id.clone());
                        round.phase = RoundPhase::Joining;
                        info!(
                            "Peer {} joined round {} ({} participants)",
                            peer_id,
                            round_id,
                            round.participants.len()
                        );
                    } else {
                        warn!(
                            "Peer {} tried to join round {} in phase {:?}",
                            peer_id, round_id, round.phase
                        );
                    }
                }
            }
            "start_compute" => {
                // Registration deadline passed; transition to compute phase.
                let round_id = msg
                    .get("round_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if let Some(round) = state.active_rounds.get_mut(round_id) {
                    round.phase = RoundPhase::Computing;
                    info!(
                        "Round {} starting compute with {} participants",
                        round_id,
                        round.participants.len()
                    );

                    // Tell the Python engine to start training if connected.
                    if let Some(engine) = &state.engine {
                        let req = TrainStepRequest {
                            round_id: round_id.to_string(),
                            input_shape: vec![1, 16],
                        };
                        match engine.train_step(&req).await {
                            Ok(resp) => {
                                info!(
                                    "Engine training step: loss={:.4}, grad_norm={:.4}",
                                    resp.loss, resp.grad_norm
                                );
                            }
                            Err(e) => {
                                warn!("Engine training step failed: {}", e);
                            }
                        }
                    }
                }
            }
            "aggregate_complete" => {
                let round_id = msg
                    .get("round_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let merkle_root = msg
                    .get("merkle_root")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if let Some(round) = state.active_rounds.get_mut(round_id) {
                    round.phase = RoundPhase::Checkpointing;
                    round.merkle_root = Some(merkle_root.clone());
                    info!(
                        "Round {} aggregation complete, merkle_root={}",
                        round_id,
                        &merkle_root[..merkle_root.len().min(16)]
                    );

                    // Announce checkpoint on gossip.
                    let ckpt_msg = serde_json::json!({
                        "type": "checkpoint_available",
                        "round_id": round_id,
                        "model_id": round.model_id,
                        "merkle_root": merkle_root,
                    });
                    let _ = command_tx
                        .send(SwarmCommand::Publish {
                            topic: gossip::TOPIC_CHECKPOINT.to_string(),
                            data: serde_json::to_vec(&ckpt_msg).unwrap_or_default(),
                        })
                        .await;
                }
            }
            "round_complete" => {
                let round_id = msg
                    .get("round_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if let Some(round) = state.active_rounds.get_mut(round_id) {
                    round.phase = RoundPhase::Complete;
                    info!(
                        "Round {} complete ({} participants)",
                        round_id,
                        round.participants.len()
                    );
                }
            }
            "round_failed" => {
                let round_id = msg
                    .get("round_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let reason = msg
                    .get("reason")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                if let Some(round) = state.active_rounds.get_mut(round_id) {
                    round.phase = RoundPhase::Failed;
                    warn!("Round {} FAILED: {}", round_id, reason);
                }
            }
            _ => {
                debug!("Unknown training message type: {}", msg_type);
            }
        }
    }
}

async fn handle_gradient_message(
    state: &mut SchedulerState,
    command_tx: &mpsc::Sender<SwarmCommand>,
    reputation: &SharedReputationTable,
    source: &str,
    data: &[u8],
) {
    if let Ok(msg) = serde_json::from_slice::<serde_json::Value>(data) {
        let round_id = msg
            .get("round_id")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let peer_id = msg
            .get("peer_id")
            .and_then(|v| v.as_str())
            .unwrap_or(source);
        let msg_type = msg
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("submit");

        match msg_type {
            "submit" => {
                if let Some(round) = state.active_rounds.get_mut(round_id) {
                    if round.phase == RoundPhase::Computing
                        || round.phase == RoundPhase::Aggregating
                    {
                        round.gradient_submissions.push(peer_id.to_string());
                        round.phase = RoundPhase::Aggregating;
                        debug!(
                            "Gradient submission from {} for round {} ({}/{})",
                            peer_id,
                            round_id,
                            round.gradient_submissions.len(),
                            round.participants.len()
                        );

                        // Check if all participants have submitted.
                        let quorum = (round.participants.len() as f64 * 0.67) as usize;
                        if round.gradient_submissions.len() >= quorum.max(3) {
                            info!(
                                "Round {} reached quorum ({}/{}), ready to aggregate",
                                round_id,
                                round.gradient_submissions.len(),
                                round.participants.len()
                            );
                            // Announce aggregation ready on training topic.
                            let agg_msg = serde_json::json!({
                                "type": "aggregate_ready",
                                "round_id": round_id,
                                "submissions": round.gradient_submissions.len(),
                                "total": round.participants.len(),
                            });
                            let _ = command_tx
                                .send(SwarmCommand::Publish {
                                    topic: gossip::TOPIC_TRAINING.to_string(),
                                    data: serde_json::to_vec(&agg_msg).unwrap_or_default(),
                                })
                                .await;
                        }
                    }
                }
            }
            _ => {
                debug!("Unknown gradient message type: {}", msg_type);
            }
        }
    }
}

fn handle_checkpoint_message(
    state: &mut SchedulerState,
    reputation: &SharedReputationTable,
    source: &str,
    data: &[u8],
) {
    if let Ok(msg) = serde_json::from_slice::<serde_json::Value>(data) {
        let msg_type = msg
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        match msg_type {
            "checkpoint_available" => {
                let round_id = msg
                    .get("round_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let merkle_root = msg
                    .get("merkle_root")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                info!(
                    "Checkpoint available from {}: round={}, merkle={}",
                    source,
                    round_id,
                    &merkle_root[..merkle_root.len().min(16)]
                );

                // Mark the round as complete and update reputation.
                if let Some(round) = state.active_rounds.get_mut(round_id) {
                    round.phase = RoundPhase::Complete;
                    round.merkle_root = Some(merkle_root.to_string());

                    // Reward all participants.
                    if let Ok(mut rep) = reputation.write() {
                        for pid in &round.participants {
                            rep.record_round_complete(pid);
                        }
                        // Penalize non-submitters.
                        let submitters: std::collections::HashSet<_> =
                            round.gradient_submissions.iter().collect();
                        for pid in &round.participants {
                            if !submitters.contains(pid) {
                                rep.record_round_failed(pid);
                            }
                        }
                    }
                }
            }
            "checkpoint_verified" => {
                let round_id = msg
                    .get("round_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                debug!("Checkpoint verified for round {} by {}", round_id, source);
            }
            _ => {
                debug!("Unknown checkpoint message type: {}", msg_type);
            }
        }
    }
}

async fn handle_architecture_message(
    state: &mut SchedulerState,
    command_tx: &mpsc::Sender<SwarmCommand>,
    source: &str,
    data: &[u8],
) {
    if let Ok(msg) = serde_json::from_slice::<serde_json::Value>(data) {
        let msg_type = msg
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        match msg_type {
            "architecture_proposal" => {
                let proposal_id = msg
                    .get("proposal_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let model_id = msg
                    .get("model_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                info!(
                    "Architecture proposal from {}: proposal={}, model={}",
                    source, proposal_id, model_id
                );

                // Forward to the Python engine for evaluation and voting.
                if let Some(engine) = &state.engine {
                    let payload = serde_json::to_vec(&msg).unwrap_or_default();
                    match engine.architecture_proposal(&payload).await {
                        Ok(vote_response) => {
                            info!(
                                "Engine voted on proposal {}: {}",
                                proposal_id,
                                String::from_utf8_lossy(&vote_response)
                            );
                            // Broadcast the vote response on gossipsub.
                            let _ = command_tx
                                .send(SwarmCommand::Publish {
                                    topic: gossip::TOPIC_ARCHITECTURE.to_string(),
                                    data: vote_response,
                                })
                                .await;
                        }
                        Err(e) => {
                            warn!("Engine architecture proposal evaluation failed: {}", e);
                        }
                    }
                }
            }
            "architecture_vote" => {
                let proposal_id = msg
                    .get("proposal_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let decision = msg
                    .get("decision")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                debug!(
                    "Architecture vote from {}: proposal={}, decision={}",
                    source, proposal_id, decision
                );
            }
            _ => {
                debug!("Unknown architecture message type: {}", msg_type);
            }
        }
    }
}

/// Clean up timed-out training rounds.
async fn cleanup_timed_out_rounds(
    state: &mut SchedulerState,
    command_tx: &mpsc::Sender<SwarmCommand>,
) {
    let now = now_ms();
    let mut timed_out = vec![];

    for (round_id, round) in &state.active_rounds {
        // Only timeout rounds that are in active (non-terminal) phases.
        if round.phase == RoundPhase::Complete || round.phase == RoundPhase::Failed {
            continue;
        }
        if now - round.started_at_ms > ROUND_TIMEOUT_MS {
            timed_out.push(round_id.clone());
        }
    }

    for round_id in timed_out {
        if let Some(round) = state.active_rounds.get_mut(&round_id) {
            warn!(
                "Round {} timed out in phase {:?} after {}s",
                round_id,
                round.phase,
                ROUND_TIMEOUT_MS / 1000,
            );
            round.phase = RoundPhase::Failed;

            // Broadcast failure message.
            let fail_msg = serde_json::json!({
                "type": "round_failed",
                "round_id": round_id,
                "reason": "timeout",
            });
            let _ = command_tx
                .send(SwarmCommand::Publish {
                    topic: gossip::TOPIC_TRAINING.to_string(),
                    data: serde_json::to_vec(&fail_msg).unwrap_or_default(),
                })
                .await;
        }
    }
}

/// Sync scheduler state to the API shared state.
fn sync_api_state(state: &SchedulerState) {
    let rounds: Vec<RoundInfo> = state
        .active_rounds
        .values()
        .filter(|r| r.phase != RoundPhase::Complete && r.phase != RoundPhase::Failed)
        .map(|r| RoundInfo {
            round_id: r.round_id.clone(),
            model_id: r.model_id.clone(),
            phase: r.phase.as_str().to_string(),
            participants: r.participants.len(),
            started_at_ms: r.started_at_ms,
        })
        .collect();

    if let Ok(mut api_rounds) = state.api_state.active_rounds.write() {
        *api_rounds = rounds;
    }

    *state.api_state.peer_count.write().unwrap() = state.live_peers.len();
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}
