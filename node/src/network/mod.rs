pub mod discovery;
pub mod gossip;
pub mod transport;

use anyhow::Result;
use libp2p::{
    gossipsub, identify, kad, mdns, noise, swarm::SwarmEvent, tcp, yamux, Multiaddr, PeerId,
    Swarm, SwarmBuilder,
};
use std::collections::HashSet;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::config::NodeConfig;
use gossip::TOPIC_HEARTBEAT;

/// Events emitted by the swarm and consumed by the scheduler.
#[derive(Debug, Clone)]
pub enum SwarmEvent2 {
    PeerDiscovered(PeerId),
    PeerExpired(PeerId),
    GossipMessage {
        source: PeerId,
        topic: String,
        data: Vec<u8>,
    },
}

/// Commands sent to the swarm driver from other components.
#[derive(Debug)]
pub enum SwarmCommand {
    Publish { topic: String, data: Vec<u8> },
    Dial(Multiaddr),
}

/// Combined libp2p behaviour for the OpenClaw node.
#[derive(libp2p::swarm::NetworkBehaviour)]
pub struct OpenClawBehaviour {
    pub gossipsub: gossipsub::Behaviour,
    pub kademlia: kad::Behaviour<kad::store::MemoryStore>,
    pub mdns: mdns::tokio::Behaviour,
    pub identify: identify::Behaviour,
}

/// Drives the libp2p swarm. Bridges between the swarm and the rest of the node
/// via channels.
pub struct SwarmDriver {
    swarm: Swarm<OpenClawBehaviour>,
    event_tx: mpsc::Sender<SwarmEvent2>,
    command_rx: mpsc::Receiver<SwarmCommand>,
    config: NodeConfig,
}

impl SwarmDriver {
    /// Create a new swarm driver. Returns the driver plus channels for events
    /// and commands.
    pub async fn new(
        config: &NodeConfig,
    ) -> Result<(
        Self,
        mpsc::Receiver<SwarmEvent2>,
        mpsc::Sender<SwarmCommand>,
    )> {
        let (event_tx, event_rx) = mpsc::channel(256);
        let (command_tx, command_rx) = mpsc::channel(256);

        let swarm = transport::build_swarm(config).await?;

        let driver = Self {
            swarm,
            event_tx,
            command_rx,
            config: config.clone(),
        };

        Ok((driver, event_rx, command_tx))
    }

    /// Main loop: drive the swarm and process commands.
    pub async fn run(&mut self) -> Result<()> {
        // Listen on all interfaces.
        let listen_addr: Multiaddr =
            format!("/ip4/0.0.0.0/tcp/{}", self.config.listen_port).parse()?;
        self.swarm.listen_on(listen_addr)?;

        // Subscribe to gossipsub topics.
        let topics = gossip::subscribe_all(&mut self.swarm.behaviour_mut().gossipsub)?;
        info!("Subscribed to {} gossipsub topics", topics.len());

        // Dial bootstrap peers.
        for addr_str in &self.config.bootstrap_peers {
            if let Ok(addr) = addr_str.parse::<Multiaddr>() {
                info!("Dialing bootstrap peer: {}", addr);
                let _ = self.swarm.dial(addr);
            } else {
                warn!("Invalid bootstrap address: {}", addr_str);
            }
        }

        // Set up heartbeat timer.
        let mut heartbeat_interval =
            tokio::time::interval(Duration::from_secs(self.config.heartbeat_interval_secs));

        loop {
            tokio::select! {
                // Process swarm events.
                event = self.swarm.select_next_some() => {
                    self.handle_swarm_event(event).await;
                }
                // Process commands from other components.
                Some(cmd) = self.command_rx.recv() => {
                    self.handle_command(cmd);
                }
                // Periodic heartbeat.
                _ = heartbeat_interval.tick() => {
                    self.send_heartbeat();
                }
            }
        }
    }

    async fn handle_swarm_event(&mut self, event: SwarmEvent<OpenClawBehaviourEvent>) {
        use libp2p::swarm::SwarmEvent::*;
        match event {
            NewListenAddr { address, .. } => {
                let local_peer = *self.swarm.local_peer_id();
                info!("Listening on {}/p2p/{}", address, local_peer);
            }
            Behaviour(OpenClawBehaviourEvent::Mdns(mdns::Event::Discovered(peers))) => {
                for (peer_id, addr) in peers {
                    info!("mDNS discovered peer: {} at {}", peer_id, addr);
                    self.swarm
                        .behaviour_mut()
                        .kademlia
                        .add_address(&peer_id, addr);
                    let _ = self.event_tx.send(SwarmEvent2::PeerDiscovered(peer_id)).await;
                }
            }
            Behaviour(OpenClawBehaviourEvent::Mdns(mdns::Event::Expired(peers))) => {
                for (peer_id, _) in peers {
                    debug!("mDNS peer expired: {}", peer_id);
                    let _ = self.event_tx.send(SwarmEvent2::PeerExpired(peer_id)).await;
                }
            }
            Behaviour(OpenClawBehaviourEvent::Gossipsub(gossipsub::Event::Message {
                propagation_source,
                message,
                ..
            })) => {
                let topic = message.topic.to_string();
                debug!(
                    "Gossip message from {} on topic {}",
                    propagation_source, topic
                );
                let _ = self
                    .event_tx
                    .send(SwarmEvent2::GossipMessage {
                        source: propagation_source,
                        topic,
                        data: message.data,
                    })
                    .await;
            }
            Behaviour(OpenClawBehaviourEvent::Kademlia(
                kad::Event::OutboundQueryProgressed { result, .. },
            )) => {
                if let kad::QueryResult::GetClosestPeers(Ok(ok)) = result {
                    for peer in ok.peers {
                        debug!("Kademlia found peer: {:?}", peer);
                    }
                }
            }
            Behaviour(OpenClawBehaviourEvent::Identify(identify::Event::Received {
                peer_id,
                info: identify_info,
                ..
            })) => {
                debug!(
                    "Identified peer {}: protocols={:?}",
                    peer_id, identify_info.protocols
                );
                for addr in identify_info.listen_addrs {
                    self.swarm
                        .behaviour_mut()
                        .kademlia
                        .add_address(&peer_id, addr);
                }
            }
            ConnectionEstablished { peer_id, .. } => {
                info!("Connection established with {}", peer_id);
                let _ = self.event_tx.send(SwarmEvent2::PeerDiscovered(peer_id)).await;
            }
            ConnectionClosed { peer_id, .. } => {
                info!("Connection closed with {}", peer_id);
                let _ = self.event_tx.send(SwarmEvent2::PeerExpired(peer_id)).await;
            }
            _ => {}
        }
    }

    fn handle_command(&mut self, cmd: SwarmCommand) {
        match cmd {
            SwarmCommand::Publish { topic, data } => {
                let topic = gossipsub::IdentTopic::new(topic);
                if let Err(e) = self.swarm.behaviour_mut().gossipsub.publish(topic, data) {
                    warn!("Failed to publish gossip message: {}", e);
                }
            }
            SwarmCommand::Dial(addr) => {
                if let Err(e) = self.swarm.dial(addr.clone()) {
                    warn!("Failed to dial {}: {}", addr, e);
                }
            }
        }
    }

    fn send_heartbeat(&mut self) {
        let local_peer = *self.swarm.local_peer_id();
        let heartbeat = serde_json::json!({
            "peer_id": local_peer.to_string(),
            "timestamp_ms": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            "gpu_memory_mb": self.config.gpu_memory_mb,
            "ram_mb": self.config.ram_mb,
            "cpu_cores": self.config.cpu_cores,
            "accelerator": self.config.accelerator,
        });
        let data = serde_json::to_vec(&heartbeat).unwrap_or_default();
        let topic = gossipsub::IdentTopic::new(TOPIC_HEARTBEAT);
        if let Err(e) = self.swarm.behaviour_mut().gossipsub.publish(topic, data) {
            debug!("Heartbeat publish failed (may have no peers yet): {}", e);
        }
    }
}

/// Collect connected peers from the swarm.
pub fn connected_peers(swarm: &Swarm<OpenClawBehaviour>) -> HashSet<PeerId> {
    swarm.connected_peers().cloned().collect()
}

use futures::StreamExt;
