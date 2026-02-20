use libp2p::{kad, Multiaddr, PeerId};
use tracing::info;

/// Add a known bootstrap peer to the Kademlia DHT.
pub fn add_bootstrap_peer(
    kademlia: &mut kad::Behaviour<kad::store::MemoryStore>,
    peer_id: PeerId,
    addr: Multiaddr,
) {
    kademlia.add_address(&peer_id, addr.clone());
    info!("Added bootstrap peer {} at {}", peer_id, addr);
}

/// Trigger a random walk on the DHT to discover more peers.
pub fn bootstrap_dht(kademlia: &mut kad::Behaviour<kad::store::MemoryStore>) {
    match kademlia.bootstrap() {
        Ok(query_id) => {
            info!("DHT bootstrap initiated (query {:?})", query_id);
        }
        Err(e) => {
            tracing::warn!("DHT bootstrap failed (no known peers?): {:?}", e);
        }
    }
}

/// Parse a multiaddr string that may contain a peer ID suffix.
/// e.g. "/ip4/1.2.3.4/tcp/9000/p2p/12D3Koo..."
pub fn parse_peer_multiaddr(addr_str: &str) -> Option<(PeerId, Multiaddr)> {
    let addr: Multiaddr = addr_str.parse().ok()?;
    let peer_id = addr.iter().find_map(|proto| {
        if let libp2p::multiaddr::Protocol::P2p(peer_id) = proto {
            Some(peer_id)
        } else {
            None
        }
    })?;

    // Strip /p2p/... from the multiaddr for Kademlia.
    let dial_addr: Multiaddr = addr
        .iter()
        .filter(|p| !matches!(p, libp2p::multiaddr::Protocol::P2p(_)))
        .collect();

    Some((peer_id, dial_addr))
}
