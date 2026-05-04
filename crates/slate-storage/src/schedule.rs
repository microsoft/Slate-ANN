//! The HDD elevator (SCAN) I/O scheduler.
//!
//! On a spinning disk the dominant cost of servicing a batch of vector reads is
//! the number of head repositionings — *seeks* — not the bytes transferred. A
//! 7200rpm seek is ~9 ms; streaming a few-kilobyte vector once the head is in
//! place is tens of microseconds. So the way to cut query latency is to issue
//! the pending reads in **physical order** and **coalesce** the ones that fall
//! close together, converting a scatter of random seeks into a few sequential
//! runs.
//!
//! [`FetchSchedule::plan`] is a pure planner: given a [`BlockLayout`] and the
//! dense indices a query wants to fetch, it returns the seek-ordered traversal
//! plus the resulting seek / run / byte counts. It performs no I/O and contains
//! no `unsafe`; the caller reads the vectors in [`FetchSchedule::order`] and
//! charges the storage counters once for the whole batch. This is the engine
//! realization of the cost model's `coalescing_seeks_lowers_cost_at_equal_bytes`
//! property: the same payload read in fewer seeks costs less.
//!
//! Coalescing is by block. Because [`BlockLayout::vector_offset`] is monotonic
//! in the dense index, sorting the indices ascending *is* sorting by physical
//! offset — a single forward sweep of the elevator. Two reads share a run when
//! they land on the same block or on immediately adjacent blocks (the head
//! streams across a block boundary without re-seeking); a gap of two or more
//! blocks starts a new run, i.e. a new seek.

use crate::layout::BlockLayout;

/// A seek-minimizing plan for a batch of vector fetches.
///
/// Produced by [`FetchSchedule::plan`]. Holds the requested indices in
/// ascending physical-offset order together with the coalesced cost of
/// servicing them: `seeks` (== `runs`) head positionings and `bytes` of
/// payload transferred.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FetchSchedule {
    order: Vec<usize>,
    seeks: u64,
    runs: u64,
    bytes: u64,
}

impl FetchSchedule {
    /// Plan the seek-ordered, block-coalesced reading of `indices` against
    /// `layout`.
    ///
    /// The indices are sorted ascending and de-duplicated (so the same vector
    /// requested twice is fetched once), then walked to count coalesced runs:
    /// a new run begins only where the next index's block is neither equal to
    /// nor one past the previous index's block. `seeks` equals `runs`; `bytes`
    /// is the payload of the unique vectors (`unique_count * vector_bytes`),
    /// not the span of the runs — read-amplification accounting is deferred
    /// until whole-block reads land.
    ///
    /// An empty `indices` yields an empty, zero-cost schedule.
    #[must_use]
    pub fn plan(layout: &BlockLayout, indices: &[usize]) -> Self {
        if indices.is_empty() {
            return Self {
                order: Vec::new(),
                seeks: 0,
                runs: 0,
                bytes: 0,
            };
        }

        // Ascending dense index == ascending physical offset: the elevator's
        // forward sweep. Dedup so a doubly-requested vector is read once.
        let mut order = indices.to_vec();
        order.sort_unstable();
        order.dedup();

        let bytes = order.len() as u64 * layout.vector_bytes() as u64;

        // Count coalesced runs over the swept order. Same block or adjacent
        // block continues the current run; a gap of >= 2 blocks is a new seek.
        let mut runs: u64 = 0;
        let mut prev_block: Option<usize> = None;
        for &i in &order {
            let block = layout.block_of(i);
            let continues = matches!(prev_block, Some(pb) if block == pb || block == pb + 1);
            if !continues {
                runs += 1;
            }
            prev_block = Some(block);
        }

        Self {
            order,
            seeks: runs,
            runs,
            bytes,
        }
    }

    /// The requested indices in ascending physical-offset order (deduplicated).
    /// Read vectors in this order to follow the elevator sweep.
    #[must_use]
    pub fn order(&self) -> &[usize] {
        &self.order
    }

    /// Number of head positionings to service the batch (one per coalesced
    /// run). This is the value that drives the seek term of the cost model.
    #[must_use]
    pub fn seeks(&self) -> u64 {
        self.seeks
    }

    /// Number of coalesced sequential runs (equal to [`Self::seeks`]).
    #[must_use]
    pub fn runs(&self) -> u64 {
        self.runs
    }

    /// Total payload bytes transferred (`unique_count * vector_bytes`).
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use slate_core::Dtype;

    // 4-dim f32 => 16 bytes/vector. block_size 1024 => 64 vectors/block (many
    // per block, for coalescing); block_size 16 => 1 vector/block (for forced
    // scatter).
    fn wide_block() -> BlockLayout {
        BlockLayout::new(Dtype::F32, 4, 1024).unwrap()
    }

    fn one_per_block() -> BlockLayout {
        BlockLayout::new(Dtype::F32, 4, 16).unwrap()
    }

    #[test]
    fn empty_batch_has_no_seeks() {
        let plan = FetchSchedule::plan(&wide_block(), &[]);
        assert!(plan.order().is_empty());
        assert_eq!(plan.seeks(), 0);
        assert_eq!(plan.runs(), 0);
        assert_eq!(plan.bytes(), 0);
    }

    #[test]
    fn single_block_coalesces_to_one_seek() {
        // All four indices live in block 0 (64 vectors/block).
        let layout = wide_block();
        let plan = FetchSchedule::plan(&layout, &[3, 1, 2, 0]);
        assert_eq!(plan.order(), &[0, 1, 2, 3]);
        assert_eq!(plan.seeks(), 1);
        assert_eq!(plan.runs(), 1);
        assert_eq!(plan.bytes(), 4 * layout.vector_bytes() as u64);
    }

    #[test]
    fn scattered_blocks_each_cost_a_seek() {
        // 1 vector/block, so indices 0, 2, 4 are blocks 0, 2, 4 — all
        // non-adjacent => three seeks.
        let plan = FetchSchedule::plan(&one_per_block(), &[4, 0, 2]);
        assert_eq!(plan.order(), &[0, 2, 4]);
        assert_eq!(plan.seeks(), 3);
        assert_eq!(plan.runs(), 3);
    }

    #[test]
    fn adjacent_blocks_form_one_run() {
        // 1 vector/block, indices 5,6,7 => blocks 5,6,7 (consecutive) => one run.
        let plan = FetchSchedule::plan(&one_per_block(), &[7, 5, 6]);
        assert_eq!(plan.order(), &[5, 6, 7]);
        assert_eq!(plan.seeks(), 1);
        assert_eq!(plan.runs(), 1);
    }

    #[test]
    fn mixed_runs_counted() {
        // 1 vector/block. Blocks {0,1, 5, 9,10}: runs are {0,1}, {5}, {9,10}
        // => 3 runs.
        let plan = FetchSchedule::plan(&one_per_block(), &[10, 0, 9, 1, 5]);
        assert_eq!(plan.order(), &[0, 1, 5, 9, 10]);
        assert_eq!(plan.seeks(), 3);
        assert_eq!(plan.runs(), 3);
    }

    #[test]
    fn dedup_repeated_indices() {
        let layout = one_per_block();
        let plan = FetchSchedule::plan(&layout, &[2, 2, 2]);
        assert_eq!(plan.order(), &[2]);
        assert_eq!(plan.seeks(), 1);
        assert_eq!(plan.bytes(), layout.vector_bytes() as u64);
    }

    #[test]
    fn sorting_is_ascending() {
        let plan = FetchSchedule::plan(&one_per_block(), &[9, 1, 7, 3, 5]);
        let mut sorted = plan.order().to_vec();
        sorted.sort_unstable();
        assert_eq!(plan.order(), sorted.as_slice());
    }

    proptest::proptest! {
        #[test]
        fn invariants_hold(
            mut indices in proptest::collection::vec(0usize..256, 0..64),
            // small block sizes => vectors span many blocks
            log_block in 4u32..=10,
        ) {
            let block_size = 1usize << log_block;
            let layout = BlockLayout::new(Dtype::F32, 4, block_size).unwrap();
            let plan = FetchSchedule::plan(&layout, &indices);

            // order is sorted and unique
            let order = plan.order();
            for w in order.windows(2) {
                proptest::prop_assert!(w[0] < w[1]);
            }

            // unique count and byte accounting
            indices.sort_unstable();
            indices.dedup();
            let unique = indices.len();
            proptest::prop_assert_eq!(order.len(), unique);
            proptest::prop_assert_eq!(plan.bytes(), unique as u64 * layout.vector_bytes() as u64);

            // seeks == runs, and bounded by distinct blocks (<= unique vectors)
            proptest::prop_assert_eq!(plan.seeks(), plan.runs());
            let distinct_blocks = {
                let mut blocks: Vec<usize> = order.iter().map(|&i| layout.block_of(i)).collect();
                blocks.dedup();
                blocks.len() as u64
            };
            proptest::prop_assert!(plan.seeks() <= distinct_blocks);
            proptest::prop_assert!(distinct_blocks <= unique as u64);
            if unique == 0 {
                proptest::prop_assert_eq!(plan.seeks(), 0);
            } else {
                proptest::prop_assert!(plan.seeks() >= 1);
            }
        }
    }
}
