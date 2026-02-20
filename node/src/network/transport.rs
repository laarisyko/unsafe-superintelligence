use anyhow::Result;
use libp2p::{
    gossipsub, identify, kad, mdns, noise, tcp, yamux, Swarm, SwarmBuilder,
};
use std::time::Duration;

use super::OpenClawBehaviour;
use crate::config::NodeConfig;

/// Build a fully configured libp2p swarm with all OpenClaw behaviours.
pub async fn build_swarm(config: &NodeConfig) -> Result<Swarm<OpenClawBehaviour>> {
    let swarm = SwarmBuilder::with_new_identity()
        .with_tokio()
        .with_tcp(
            tcp::Config::default().nodelay(true),
            noise::Config::new,
            yamux::Config::default,
        )?
        .with_behaviour(|key| {
            // Gossipsub with message deduplication.
            let gossipsub_config = gossipsub::ConfigBuilder::default()
                .heartbeat_interval(Duration::from_secs(5))
                .validation_mode(gossipsub::ValidationMode::Strict)
                .max_transmit_size(4 * 1024 * 1024) // 4 MB for gradient chunks
                .build()
                .expect("valid gossipsub config");

            let gossipsub = gossipsub::Behaviour::new(
                gossipsub::MessageAuthenticity::Signed(key.clone()),
                gossipsub_config,
            )
            .expect("valid gossipsub behaviour");

            // Kademlia DHT for peer discovery.
            let peer_id = key.public().to_peer_id();
            let mut kad_config = kad::Config::default();
            kad_config.set_query_timeout(Duration::from_secs(30));
            let store = kad::store::MemoryStore::new(peer_id);
            let kademlia = kad::Behaviour::with_config(peer_id, store, kad_config);

            // mDNS for local network discovery.
            let mdns = mdns::tokio::Behaviour::new(
                mdns::Config::default(),
                peer_id,
            )
            .expect("valid mDNS behaviour");

            // Identify protocol for exchanging peer info.
            let identify = identify::Behaviour::new(identify::Config::new(
                "/openclaw/0.1.0".to_string(),
                key.public(),
            ));

            OpenClawBehaviour {
                gossipsub,
                kademlia,
                mdns,
                identify,
            }
        })?
        .with_swarm_config(|cfg| {
            cfg.with_idle_connection_timeout(Duration::from_secs(120))
        })
        .build();

    Ok(swarm)
}
