use sha2::{Digest, Sha256};

/// Compute a Merkle root over a list of weight chunks.
///
/// Each leaf is the SHA-256 hash of a chunk. Internal nodes are
/// `H(left || right)`. If the number of leaves is odd, the last leaf is
/// promoted.
///
/// Used after training rounds to verify that all peers converged to the same
/// weight state.
pub fn merkle_root(chunks: &[&[u8]]) -> [u8; 32] {
    if chunks.is_empty() {
        return [0u8; 32];
    }

    // Hash each chunk to produce leaf nodes.
    let mut layer: Vec<[u8; 32]> = chunks.iter().map(|c| hash_leaf(c)).collect();

    // Reduce pairwise until a single root remains.
    while layer.len() > 1 {
        let mut next = Vec::with_capacity((layer.len() + 1) / 2);
        for pair in layer.chunks(2) {
            if pair.len() == 2 {
                next.push(hash_pair(&pair[0], &pair[1]));
            } else {
                next.push(pair[0]); // odd-one-out promoted
            }
        }
        layer = next;
    }

    layer[0]
}

fn hash_leaf(data: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"\x00"); // leaf prefix
    h.update(data);
    h.finalize().into()
}

fn hash_pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"\x01"); // internal prefix
    h.update(left);
    h.update(right);
    h.finalize().into()
}

/// Verify that two peers computed the same weight state.
pub fn roots_match(a: &[u8; 32], b: &[u8; 32]) -> bool {
    a == b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_chunks() {
        let root = merkle_root(&[]);
        assert_eq!(root, [0u8; 32]);
    }

    #[test]
    fn single_chunk() {
        let data = b"hello weights";
        let root = merkle_root(&[data.as_slice()]);
        assert_ne!(root, [0u8; 32]);
    }

    #[test]
    fn deterministic() {
        let chunks: Vec<&[u8]> = vec![b"a", b"b", b"c", b"d"];
        let r1 = merkle_root(&chunks);
        let r2 = merkle_root(&chunks);
        assert_eq!(r1, r2);
    }

    #[test]
    fn order_matters() {
        let r1 = merkle_root(&[b"a", b"b"]);
        let r2 = merkle_root(&[b"b", b"a"]);
        assert_ne!(r1, r2);
    }

    #[test]
    fn odd_number_of_chunks() {
        let root = merkle_root(&[b"a", b"b", b"c"]);
        assert_ne!(root, [0u8; 32]);
    }
}
