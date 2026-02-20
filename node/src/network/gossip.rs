use anyhow::Result;
use libp2p::gossipsub::{self, IdentTopic, TopicHash};
use tracing::info;

// Well-known gossipsub topic names for the OpenClaw protocol.
// These are GLOBAL topics -- all peers subscribe to them.
pub const TOPIC_HEARTBEAT: &str = "openclaw/heartbeat";
pub const TOPIC_SHARD_MAP: &str = "openclaw/shard-map";
pub const TOPIC_TRAINING: &str = "openclaw/training";
pub const TOPIC_GRADIENT: &str = "openclaw/gradient";
pub const TOPIC_CHECKPOINT: &str = "openclaw/checkpoint";
pub const TOPIC_ARCHITECTURE: &str = "openclaw/architecture";

// Cluster-scoped topic prefixes. Peers only subscribe to topics for
// their own cluster, preventing gossip floods across 1M+ peers.
pub const TOPIC_CLUSTER_GRADIENT_PREFIX: &str = "openclaw/cluster-gradient/";
pub const TOPIC_CLUSTER_HEARTBEAT_PREFIX: &str = "openclaw/cluster-heartbeat/";
pub const TOPIC_CLUSTER_SYNC_PREFIX: &str = "openclaw/cluster-sync/";
pub const TOPIC_LEADER_GRADIENT_PREFIX: &str = "openclaw/leader-gradient/";

/// All global topics the node subscribes to.
pub fn all_topics() -> Vec<IdentTopic> {
    vec![
        IdentTopic::new(TOPIC_HEARTBEAT),
        IdentTopic::new(TOPIC_SHARD_MAP),
        IdentTopic::new(TOPIC_TRAINING),
        IdentTopic::new(TOPIC_GRADIENT),
        IdentTopic::new(TOPIC_CHECKPOINT),
        IdentTopic::new(TOPIC_ARCHITECTURE),
    ]
}

/// Generate cluster-scoped topics for a specific cluster at a given level.
/// Peers subscribe to these instead of the global gradient topic to limit
/// gossip scope. E.g. "openclaw/cluster-gradient/L0-C42" for level 0, cluster 42.
pub fn cluster_topics(level: u32, cluster_id: u32) -> Vec<IdentTopic> {
    let suffix = format!("L{}-C{}", level, cluster_id);
    vec![
        IdentTopic::new(format!("{}{}", TOPIC_CLUSTER_GRADIENT_PREFIX, suffix)),
        IdentTopic::new(format!("{}{}", TOPIC_CLUSTER_HEARTBEAT_PREFIX, suffix)),
        IdentTopic::new(format!("{}{}", TOPIC_CLUSTER_SYNC_PREFIX, suffix)),
    ]
}

/// Generate leader-level gradient topics for inter-cluster aggregation.
/// Only cluster leaders subscribe to these.
pub fn leader_topics(level: u32, super_cluster_id: u32) -> Vec<IdentTopic> {
    let suffix = format!("L{}-SC{}", level, super_cluster_id);
    vec![IdentTopic::new(format!(
        "{}{}",
        TOPIC_LEADER_GRADIENT_PREFIX, suffix
    ))]
}

/// Subscribe to all OpenClaw gossipsub topics. Returns the topic hashes.
pub fn subscribe_all(gossipsub: &mut gossipsub::Behaviour) -> Result<Vec<TopicHash>> {
    let mut hashes = Vec::new();
    for topic in all_topics() {
        gossipsub.subscribe(&topic)?;
        info!("Subscribed to gossipsub topic: {}", topic);
        hashes.push(topic.hash());
    }
    Ok(hashes)
}

/// Subscribe to cluster-specific topics. Called when a peer is assigned to
/// a cluster during a training round.
pub fn subscribe_cluster(
    gossipsub: &mut gossipsub::Behaviour,
    level: u32,
    cluster_id: u32,
) -> Result<Vec<TopicHash>> {
    let mut hashes = Vec::new();
    for topic in cluster_topics(level, cluster_id) {
        gossipsub.subscribe(&topic)?;
        info!("Subscribed to cluster topic: {}", topic);
        hashes.push(topic.hash());
    }
    Ok(hashes)
}

/// Subscribe to leader-level topics. Called when a peer is elected as a
/// cluster leader.
pub fn subscribe_leader(
    gossipsub: &mut gossipsub::Behaviour,
    level: u32,
    super_cluster_id: u32,
) -> Result<Vec<TopicHash>> {
    let mut hashes = Vec::new();
    for topic in leader_topics(level, super_cluster_id) {
        gossipsub.subscribe(&topic)?;
        info!("Subscribed to leader topic: {}", topic);
        hashes.push(topic.hash());
    }
    Ok(hashes)
}

/// Unsubscribe from cluster-specific topics. Called when a training round
/// ends or a peer changes clusters.
pub fn unsubscribe_cluster(
    gossipsub: &mut gossipsub::Behaviour,
    level: u32,
    cluster_id: u32,
) -> Result<()> {
    for topic in cluster_topics(level, cluster_id) {
        gossipsub.unsubscribe(&topic)?;
        info!("Unsubscribed from cluster topic: {}", topic);
    }
    Ok(())
}

/// Determine which protocol domain a topic belongs to.
pub fn topic_domain(topic: &str) -> &'static str {
    if topic.starts_with(TOPIC_CLUSTER_GRADIENT_PREFIX) {
        return "cluster_gradient";
    }
    if topic.starts_with(TOPIC_CLUSTER_HEARTBEAT_PREFIX) {
        return "cluster_heartbeat";
    }
    if topic.starts_with(TOPIC_CLUSTER_SYNC_PREFIX) {
        return "cluster_sync";
    }
    if topic.starts_with(TOPIC_LEADER_GRADIENT_PREFIX) {
        return "leader_gradient";
    }
    match topic {
        TOPIC_HEARTBEAT => "heartbeat",
        TOPIC_SHARD_MAP => "shard_map",
        TOPIC_TRAINING => "training",
        TOPIC_GRADIENT => "gradient",
        TOPIC_CHECKPOINT => "checkpoint",
        TOPIC_ARCHITECTURE => "architecture",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cluster_topic_format() {
        let topics = cluster_topics(0, 42);
        assert_eq!(topics.len(), 3);
    }

    #[test]
    fn leader_topic_format() {
        let topics = leader_topics(1, 7);
        assert_eq!(topics.len(), 1);
    }

    #[test]
    fn topic_domain_cluster() {
        assert_eq!(
            topic_domain("openclaw/cluster-gradient/L0-C42"),
            "cluster_gradient"
        );
        assert_eq!(
            topic_domain("openclaw/leader-gradient/L1-SC7"),
            "leader_gradient"
        );
        assert_eq!(topic_domain("openclaw/heartbeat"), "heartbeat");
    }
}
