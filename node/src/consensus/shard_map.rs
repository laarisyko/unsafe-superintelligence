use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// A single shard assignment: which peer holds which layers of a model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ShardEntry {
    pub model_id: String,
    pub layer_start: u32,
    pub layer_end: u32,
    pub peer_id: String,
    /// Lamport timestamp for last-writer-wins conflict resolution.
    pub version: u64,
}

/// CRDT-based shard map replicated across all peers.
///
/// Uses a Last-Writer-Wins Element Set: each entry is keyed by
/// `(model_id, layer_start, layer_end)` and the highest `version` wins on
/// merge. This is conflict-free and converges without coordination.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShardMap {
    /// Key: (model_id, layer_start, layer_end) -> ShardEntry
    entries: HashMap<(String, u32, u32), ShardEntry>,
    /// Local Lamport clock.
    clock: u64,
}

impl ShardMap {
    pub fn new() -> Self {
        Self::default()
    }

    /// Assign a shard to a peer. Increments the local Lamport clock.
    pub fn assign(&mut self, model_id: &str, layer_start: u32, layer_end: u32, peer_id: &str) {
        self.clock += 1;
        let entry = ShardEntry {
            model_id: model_id.to_string(),
            layer_start,
            layer_end,
            peer_id: peer_id.to_string(),
            version: self.clock,
        };
        self.entries
            .insert((model_id.to_string(), layer_start, layer_end), entry);
    }

    /// Remove all shards for a given peer (e.g. when a peer leaves).
    pub fn remove_peer(&mut self, peer_id: &str) {
        self.entries.retain(|_, v| v.peer_id != peer_id);
    }

    /// Merge a remote shard map into this one. Higher version wins (LWW).
    pub fn merge(&mut self, other: &ShardMap) {
        for (key, remote_entry) in &other.entries {
            let dominated = self
                .entries
                .get(key)
                .map(|local| local.version < remote_entry.version)
                .unwrap_or(true);
            if dominated {
                self.entries.insert(key.clone(), remote_entry.clone());
            }
        }
        self.clock = self.clock.max(other.clock);
    }

    /// Look up which peer holds a specific layer of a model.
    pub fn lookup(&self, model_id: &str, layer: u32) -> Option<&ShardEntry> {
        self.entries.values().find(|e| {
            e.model_id == model_id && layer >= e.layer_start && layer < e.layer_end
        })
    }

    /// Get the full pipeline for a model: ordered list of (peer_id, layer_start, layer_end).
    pub fn pipeline(&self, model_id: &str) -> Vec<ShardEntry> {
        let mut entries: Vec<_> = self
            .entries
            .values()
            .filter(|e| e.model_id == model_id)
            .cloned()
            .collect();
        entries.sort_by_key(|e| e.layer_start);
        entries
    }

    /// Serialize for gossip transmission.
    pub fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).unwrap_or_default()
    }

    /// Deserialize from gossip.
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        serde_json::from_slice(data).ok()
    }

    pub fn entries(&self) -> &HashMap<(String, u32, u32), ShardEntry> {
        &self.entries
    }
}

/// Thread-safe shared shard map.
pub type SharedShardMap = Arc<RwLock<ShardMap>>;

/// Local reputation table maintained per-peer.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReputationTable {
    scores: HashMap<String, PeerReputation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerReputation {
    pub score: f64,
    pub rounds_participated: u32,
    pub rounds_completed: u32,
    pub rounds_failed: u32,
    pub last_seen_ms: u64,
}

impl Default for PeerReputation {
    fn default() -> Self {
        Self {
            score: 0.5,
            rounds_participated: 0,
            rounds_completed: 0,
            rounds_failed: 0,
            last_seen_ms: 0,
        }
    }
}

impl ReputationTable {
    pub fn record_heartbeat(&mut self, peer_id: &str, timestamp_ms: u64) {
        let entry = self
            .scores
            .entry(peer_id.to_string())
            .or_insert_with(PeerReputation::default);
        entry.last_seen_ms = timestamp_ms;
    }

    pub fn record_round_complete(&mut self, peer_id: &str) {
        let entry = self
            .scores
            .entry(peer_id.to_string())
            .or_insert_with(PeerReputation::default);
        entry.rounds_completed += 1;
        entry.rounds_participated += 1;
        // Increase reputation.
        entry.score = (entry.score + 0.1).min(1.0);
    }

    pub fn record_round_failed(&mut self, peer_id: &str) {
        let entry = self
            .scores
            .entry(peer_id.to_string())
            .or_insert_with(PeerReputation::default);
        entry.rounds_failed += 1;
        entry.rounds_participated += 1;
        // Decrease reputation.
        entry.score = (entry.score - 0.2).max(0.0);
    }

    pub fn get_score(&self, peer_id: &str) -> f64 {
        self.scores
            .get(peer_id)
            .map(|r| r.score)
            .unwrap_or(0.5)
    }

    /// Peers sorted by reputation (highest first).
    pub fn ranked_peers(&self) -> Vec<(String, f64)> {
        let mut peers: Vec<_> = self
            .scores
            .iter()
            .map(|(id, r)| (id.clone(), r.score))
            .collect();
        peers.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        peers
    }
}

pub type SharedReputationTable = Arc<RwLock<ReputationTable>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_map_merge_lww() {
        let mut map_a = ShardMap::new();
        let mut map_b = ShardMap::new();

        map_a.assign("model1", 0, 12, "peer_A");
        map_b.assign("model1", 0, 12, "peer_B");
        map_b.assign("model1", 0, 12, "peer_B"); // version 2

        map_a.merge(&map_b);
        let entry = map_a.lookup("model1", 5).unwrap();
        assert_eq!(entry.peer_id, "peer_B");
    }

    #[test]
    fn pipeline_ordering() {
        let mut map = ShardMap::new();
        map.assign("m", 12, 24, "peer_B");
        map.assign("m", 0, 12, "peer_A");
        map.assign("m", 24, 32, "peer_C");

        let pipeline = map.pipeline("m");
        assert_eq!(pipeline.len(), 3);
        assert_eq!(pipeline[0].peer_id, "peer_A");
        assert_eq!(pipeline[1].peer_id, "peer_B");
        assert_eq!(pipeline[2].peer_id, "peer_C");
    }

    #[test]
    fn reputation_scoring() {
        let mut table = ReputationTable::default();
        table.record_round_complete("peer_A");
        table.record_round_complete("peer_A");
        table.record_round_failed("peer_B");

        assert!(table.get_score("peer_A") > table.get_score("peer_B"));
    }
}
