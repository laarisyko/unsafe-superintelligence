use sha2::{Digest, Sha256};

/// Verifiable Random Function output used for deterministic, leaderless work
/// assignment. Every peer computes the same output given the same inputs, so no
/// coordinator is needed.
///
/// The VRF is seeded with `round_id || sorted_peer_ids` and produces a
/// deterministic permutation used to assign data shards and ring positions.
#[derive(Debug, Clone)]
pub struct VrfOutput {
    /// The raw 256-bit hash output.
    pub hash: [u8; 32],
}

impl VrfOutput {
    /// Compute a VRF output from a training round ID and the sorted list of
    /// participating peer IDs.
    pub fn compute(round_id: &str, sorted_peer_ids: &[String]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"openclaw-vrf-v1:");
        hasher.update(round_id.as_bytes());
        hasher.update(b":");
        for pid in sorted_peer_ids {
            hasher.update(pid.as_bytes());
            hasher.update(b",");
        }
        let hash: [u8; 32] = hasher.finalize().into();
        Self { hash }
    }

    /// Derive a deterministic permutation of indices [0..n) from the VRF
    /// output. Used to assign peers to data shards or ring positions.
    pub fn permutation(&self, n: usize) -> Vec<usize> {
        if n == 0 {
            return vec![];
        }

        // Fisher-Yates shuffle seeded by successive hashes.
        let mut indices: Vec<usize> = (0..n).collect();
        let mut seed = self.hash;

        for i in (1..n).rev() {
            // Derive next random bytes.
            let mut hasher = Sha256::new();
            hasher.update(seed);
            hasher.update(&(i as u64).to_le_bytes());
            seed = hasher.finalize().into();

            // Extract a usize from the first 8 bytes.
            let rand_val =
                u64::from_le_bytes(seed[..8].try_into().unwrap()) as usize;
            let j = rand_val % (i + 1);
            indices.swap(i, j);
        }

        indices
    }

    /// Assign `n_peers` to `n_shards` data partitions. Returns a mapping from
    /// shard index to the peer index responsible for it.
    pub fn assign_shards(&self, n_peers: usize, n_shards: usize) -> Vec<usize> {
        let perm = self.permutation(n_peers);
        (0..n_shards).map(|s| perm[s % n_peers]).collect()
    }

    /// Compute ring all-reduce topology: returns the ordered ring of peer
    /// indices.
    pub fn ring_order(&self, n_peers: usize) -> Vec<usize> {
        self.permutation(n_peers)
    }

    /// Assign peers to hierarchical clusters for scalable all-reduce.
    ///
    /// Returns a `Vec<Vec<Vec<usize>>>`: indexed by [level][cluster_id][member_position].
    /// At level 0, all peers are assigned to leaf clusters. At higher levels,
    /// only cluster leaders (first member of each lower-level cluster) participate.
    ///
    /// For 1M agents with cluster_size=1000:
    ///   Level 0: 1000 clusters x 1000 peers = 1M peers
    ///   Level 1: 1000 cluster leaders grouped into 1 ring
    ///   Total rounds: ~3998 (vs ~2,000,000 flat)
    pub fn assign_clusters(
        &self,
        n_peers: usize,
        cluster_size: usize,
        depth: usize,
    ) -> Vec<Vec<Vec<usize>>> {
        if n_peers == 0 || cluster_size == 0 || depth == 0 {
            return vec![];
        }

        let perm = self.permutation(n_peers);
        let mut levels: Vec<Vec<Vec<usize>>> = Vec::with_capacity(depth);

        // Level 0: assign permuted peers to leaf clusters.
        let mut leaf_clusters: Vec<Vec<usize>> = Vec::new();
        for chunk in perm.chunks(cluster_size) {
            let mut cluster: Vec<usize> = chunk.to_vec();
            cluster.sort();
            leaf_clusters.push(cluster);
        }
        levels.push(leaf_clusters);

        // Higher levels: group leaders from previous level.
        for _level in 1..depth {
            let prev_clusters = levels.last().unwrap();
            let leaders: Vec<usize> = prev_clusters
                .iter()
                .map(|c| c[0]) // First member is the leader
                .collect();

            let mut higher_clusters: Vec<Vec<usize>> = Vec::new();
            for chunk in leaders.chunks(cluster_size) {
                let mut cluster: Vec<usize> = chunk.to_vec();
                cluster.sort();
                higher_clusters.push(cluster);
            }
            levels.push(higher_clusters);
        }

        levels
    }

    /// Elect cluster leaders at each level. Returns Vec<Vec<usize>> indexed
    /// by [level][cluster_index] = leader_peer_index.
    pub fn cluster_leaders(
        &self,
        n_peers: usize,
        cluster_size: usize,
        depth: usize,
    ) -> Vec<Vec<usize>> {
        let clusters = self.assign_clusters(n_peers, cluster_size, depth);
        clusters
            .iter()
            .map(|level| level.iter().map(|c| c[0]).collect())
            .collect()
    }

    /// Compute the optimal hierarchy depth for a given number of peers.
    pub fn optimal_depth(n_peers: usize, cluster_size: usize) -> usize {
        if n_peers <= cluster_size {
            return 1;
        }
        let mut depth = 1;
        let mut capacity = cluster_size;
        while capacity < n_peers && depth < 7 {
            depth += 1;
            capacity = cluster_size.saturating_pow(depth as u32);
        }
        depth
    }

    /// Compute hierarchical ring orders: at each level, for each cluster,
    /// return the ring order of its members. This is used to set up the
    /// ring all-reduce within each cluster.
    pub fn hierarchical_ring_orders(
        &self,
        n_peers: usize,
        cluster_size: usize,
        depth: usize,
    ) -> Vec<Vec<Vec<usize>>> {
        // Clusters are already sorted, so ring order = member order
        // (the VRF permutation already provides randomness).
        self.assign_clusters(n_peers, cluster_size, depth)
    }

    /// Compute total communication rounds for hierarchical vs flat all-reduce.
    pub fn scaling_stats(n_peers: usize, cluster_size: usize, depth: usize) -> (usize, usize) {
        let flat_rounds = if n_peers > 1 { 2 * (n_peers - 1) } else { 0 };

        let mut hierarchical_rounds = 0;
        let mut peers_at_level = n_peers;
        for _level in 0..depth {
            let n_clusters = (peers_at_level + cluster_size - 1) / cluster_size;
            let actual_size = std::cmp::min(cluster_size, peers_at_level);
            if actual_size > 1 {
                hierarchical_rounds += 2 * (actual_size - 1);
            }
            peers_at_level = n_clusters;
        }

        (flat_rounds, hierarchical_rounds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_output() {
        let peers = vec!["peerA".into(), "peerB".into(), "peerC".into()];
        let a = VrfOutput::compute("round-1", &peers);
        let b = VrfOutput::compute("round-1", &peers);
        assert_eq!(a.hash, b.hash);
    }

    #[test]
    fn different_rounds_differ() {
        let peers = vec!["peerA".into(), "peerB".into()];
        let a = VrfOutput::compute("round-1", &peers);
        let b = VrfOutput::compute("round-2", &peers);
        assert_ne!(a.hash, b.hash);
    }

    #[test]
    fn permutation_is_valid() {
        let peers = vec!["a".into(), "b".into(), "c".into(), "d".into()];
        let vrf = VrfOutput::compute("test", &peers);
        let perm = vrf.permutation(4);
        assert_eq!(perm.len(), 4);
        let mut sorted = perm.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
    }

    #[test]
    fn shard_assignment_covers_all() {
        let peers = vec!["a".into(), "b".into(), "c".into()];
        let vrf = VrfOutput::compute("test", &peers);
        let assignments = vrf.assign_shards(3, 6);
        assert_eq!(assignments.len(), 6);
        // Each peer should appear at least once.
        for p in 0..3 {
            assert!(assignments.contains(&p));
        }
    }

    #[test]
    fn cluster_assignment_covers_all_peers() {
        let peers: Vec<String> = (0..100).map(|i| format!("peer-{}", i)).collect();
        let vrf = VrfOutput::compute("round-1", &peers);
        let clusters = vrf.assign_clusters(100, 10, 2);

        // Level 0: should have 10 clusters of 10.
        assert_eq!(clusters.len(), 2);
        assert_eq!(clusters[0].len(), 10);

        // All peers should appear exactly once at level 0.
        let mut all_peers: Vec<usize> = clusters[0]
            .iter()
            .flat_map(|c| c.iter().cloned())
            .collect();
        all_peers.sort();
        assert_eq!(all_peers, (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn cluster_leaders_are_deterministic() {
        let peers: Vec<String> = (0..50).map(|i| format!("peer-{}", i)).collect();
        let a = VrfOutput::compute("round-x", &peers);
        let b = VrfOutput::compute("round-x", &peers);
        assert_eq!(a.cluster_leaders(50, 10, 2), b.cluster_leaders(50, 10, 2));
    }

    #[test]
    fn hierarchical_reduces_rounds() {
        // 1M peers, cluster_size=1000, depth=2
        let (flat, hier) = VrfOutput::scaling_stats(1_000_000, 1000, 2);
        assert_eq!(flat, 2 * 999_999);
        // Hierarchical: 2*999 + 2*999 = 3996
        assert_eq!(hier, 2 * 999 + 2 * 999);
        assert!(hier < flat / 100, "Hierarchical should be >100x fewer rounds");
    }

    #[test]
    fn optimal_depth_1m_agents() {
        let depth = VrfOutput::optimal_depth(1_000_000, 1000);
        assert_eq!(depth, 2); // 1000^2 = 1M
    }

    #[test]
    fn optimal_depth_small() {
        let depth = VrfOutput::optimal_depth(50, 1000);
        assert_eq!(depth, 1); // Single ring is fine
    }

    #[test]
    fn three_level_hierarchy() {
        // 1M agents with cluster_size=100 needs 3 levels: 100^3 = 1M
        let (flat, hier) = VrfOutput::scaling_stats(1_000_000, 100, 3);
        assert_eq!(flat, 2 * 999_999);
        // 3 * 2 * 99 = 594
        assert_eq!(hier, 3 * 2 * 99);
        assert!(hier < 1000, "3-level hierarchy with K=100 should be < 1000 rounds");
    }
}
