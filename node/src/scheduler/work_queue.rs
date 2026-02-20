use std::collections::VecDeque;
use serde::{Deserialize, Serialize};

/// Types of work a node can be assigned.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkItem {
    /// Train on a data partition for a given round.
    Train {
        round_id: String,
        model_id: String,
        data_partition: u32,
        num_steps: u32,
    },
    /// Serve inference for a model.
    Inference {
        model_id: String,
    },
    /// Participate in gradient aggregation for a round.
    Aggregate {
        round_id: String,
        ring_position: u32,
        ring_size: u32,
    },
    /// Sync weights from peers after a round.
    SyncWeights {
        model_id: String,
        round_id: String,
        source_peer: String,
    },
}

/// Simple priority work queue for the local scheduler.
#[derive(Debug, Default)]
pub struct WorkQueue {
    high_priority: VecDeque<WorkItem>,
    normal_priority: VecDeque<WorkItem>,
}

impl WorkQueue {
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a high-priority work item (training rounds, aggregation).
    pub fn push_high(&mut self, item: WorkItem) {
        self.high_priority.push_back(item);
    }

    /// Push a normal-priority work item (inference serving).
    pub fn push_normal(&mut self, item: WorkItem) {
        self.normal_priority.push_back(item);
    }

    /// Pop the next work item, high priority first.
    pub fn pop(&mut self) -> Option<WorkItem> {
        self.high_priority
            .pop_front()
            .or_else(|| self.normal_priority.pop_front())
    }

    /// Check if the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.high_priority.is_empty() && self.normal_priority.is_empty()
    }

    /// Total number of pending items.
    pub fn len(&self) -> usize {
        self.high_priority.len() + self.normal_priority.len()
    }

    /// Remove all work items for a specific round (e.g. on timeout).
    pub fn cancel_round(&mut self, round_id: &str) {
        self.high_priority.retain(|item| match item {
            WorkItem::Train { round_id: rid, .. }
            | WorkItem::Aggregate { round_id: rid, .. }
            | WorkItem::SyncWeights { round_id: rid, .. } => rid != round_id,
            _ => true,
        });
        self.normal_priority.retain(|item| match item {
            WorkItem::Train { round_id: rid, .. }
            | WorkItem::Aggregate { round_id: rid, .. }
            | WorkItem::SyncWeights { round_id: rid, .. } => rid != round_id,
            _ => true,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn priority_ordering() {
        let mut q = WorkQueue::new();
        q.push_normal(WorkItem::Inference {
            model_id: "m1".into(),
        });
        q.push_high(WorkItem::Train {
            round_id: "r1".into(),
            model_id: "m1".into(),
            data_partition: 0,
            num_steps: 100,
        });

        // High priority comes out first.
        match q.pop() {
            Some(WorkItem::Train { .. }) => {}
            other => panic!("Expected Train, got {:?}", other),
        }
        match q.pop() {
            Some(WorkItem::Inference { .. }) => {}
            other => panic!("Expected Inference, got {:?}", other),
        }
    }

    #[test]
    fn cancel_round() {
        let mut q = WorkQueue::new();
        q.push_high(WorkItem::Train {
            round_id: "r1".into(),
            model_id: "m".into(),
            data_partition: 0,
            num_steps: 10,
        });
        q.push_high(WorkItem::Train {
            round_id: "r2".into(),
            model_id: "m".into(),
            data_partition: 0,
            num_steps: 10,
        });
        q.cancel_round("r1");
        assert_eq!(q.len(), 1);
    }
}
