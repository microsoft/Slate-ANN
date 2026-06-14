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
/// servicing them: `seeks` (== `runs`) head positionings, `bytes` of useful
/// payload, and `span_bytes` of bytes actually streamed off the platter
/// (payload plus the block padding dragged along inside the coalesced runs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FetchSchedule {
    order: Vec<usize>,
    seeks: u64,
    runs: u64,
    bytes: u64,
    span_bytes: u64,
}

impl FetchSchedule {
    /// Plan the seek-ordered, block-coalesced reading of `indices` against
    /// `layout`.
    ///
    /// The indices are sorted ascending and de-duplicated (so the same vector
    /// requested twice is fetched once), then walked to count coalesced runs:
    /// a new run begins only where the next index's block is neither equal to
    /// nor one past the previous index's block. `seeks` equals `runs`.
    ///
    /// Two byte counts are reported. `bytes` is the useful payload of the unique
    /// vectors (`unique_count * vector_bytes`). `span_bytes` is the bytes
    /// actually streamed off the platter: the sum over coalesced runs of each
    /// run's contiguous byte span (see [`BlockLayout::run_span`]), which includes
    /// the block-tail padding and skipped slots dragged along inside a run.
    /// `span_bytes >= bytes` always; the gap is read amplification, and it is
    /// `span_bytes` that drives the honest transfer term of the cost model.
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
                span_bytes: 0,
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
        // For each closed run, add the contiguous span the executor reads —
        // payload plus the padding swept along inside the run.
        let mut runs: u64 = 0;
        let mut span_bytes: u64 = 0;
        let mut prev_block: Option<usize> = None;
        let mut run_first = order[0];
        let mut run_last = order[0];
        for &i in &order {
            let block = layout.block_of(i);
            let continues = matches!(prev_block, Some(pb) if block == pb || block == pb + 1);
            if continues {
                run_last = i;
            } else {
                if prev_block.is_some() {
                    // Close the previous run.
                    span_bytes += layout.run_span(run_first, run_last).1 as u64;
                }
                runs += 1;
                run_first = i;
                run_last = i;
            }
            prev_block = Some(block);
        }
        // Close the final run.
        span_bytes += layout.run_span(run_first, run_last).1 as u64;

        Self {
            order,
            seeks: runs,
            runs,
            bytes,
            span_bytes,
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

    /// Total payload bytes transferred (`unique_count * vector_bytes`). This is
    /// the useful data; compare against [`Self::span_bytes`] to see read
    /// amplification.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Total bytes physically streamed off the platter: the sum over coalesced
    /// runs of each run's contiguous span, including block-tail padding and
    /// skipped slots dragged along inside a run. Always `>= bytes()`; this is
    /// the value to charge the cost model's transfer term, since it is the data
    /// the disk actually moves.
    #[must_use]
    pub fn span_bytes(&self) -> u64 {
        self.span_bytes
    }

    /// Group [`Self::order`] into coalesced runs, each returned as a
    /// `(start, count)` window into `order`: the run covers
    /// `order[start..start + count]`, a stretch of indices whose blocks are
    /// equal or adjacent and therefore form one contiguous byte span.
    ///
    /// The number of windows equals [`Self::runs`] (and [`Self::seeks`]): each
    /// window is exactly one head positioning. The elevator executor issues one
    /// positioned read per window.
    #[must_use]
    pub fn run_spans(&self, layout: &BlockLayout) -> Vec<(usize, usize)> {
        let mut spans: Vec<(usize, usize)> = Vec::new();
        let mut prev_block: Option<usize> = None;
        for (pos, &i) in self.order.iter().enumerate() {
            let block = layout.block_of(i);
            let continues = matches!(prev_block, Some(pb) if block == pb || block == pb + 1);
            if continues {
                // Extend the current (last) run.
                if let Some(last) = spans.last_mut() {
                    last.1 += 1;
                }
            } else {
                // Start a new run at this position.
                spans.push((pos, 1));
            }
            prev_block = Some(block);
        }
        spans
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

    #[test]
    fn run_spans_match_run_count_and_cover_order() {
        let layout = one_per_block();
        // Blocks {0,1, 5, 9,10}: runs {0,1}, {5}, {9,10}.
        let plan = FetchSchedule::plan(&layout, &[10, 0, 9, 1, 5]);
        let spans = plan.run_spans(&layout);
        assert_eq!(spans.len() as u64, plan.runs());
        assert_eq!(spans, vec![(0, 2), (2, 1), (3, 2)]);
        // Spans partition order positions [0, order.len()) contiguously.
        let mut next = 0;
        let mut covered = 0;
        for (start, count) in spans {
            assert_eq!(start, next);
            next += count;
            covered += count;
        }
        assert_eq!(covered, plan.order().len());
    }

    #[test]
    fn run_span_byte_range_is_contiguous_and_monotonic() {
        let layout = one_per_block();
        // Adjacent blocks 5,6,7 form one run spanning three slots.
        let plan = FetchSchedule::plan(&layout, &[7, 5, 6]);
        let spans = plan.run_spans(&layout);
        assert_eq!(spans, vec![(0, 3)]);
        let (start_pos, count) = spans[0];
        let first = plan.order()[start_pos];
        let last = plan.order()[start_pos + count - 1];
        let (offset, len) = layout.run_span(first, last);
        assert_eq!(offset, layout.vector_offset(5));
        // 3 one-vector blocks => span covers 3 whole blocks worth of bytes.
        assert_eq!(len, layout.vector_offset(7) + layout.vector_bytes() as usize - layout.vector_offset(5));
    }

    #[test]
    fn span_bytes_equal_payload_for_one_packed_run() {
        // 64 vectors/block, all in block 0, no padding between them => the run
        // spans exactly the payload of the four vectors.
        let layout = wide_block();
        let plan = FetchSchedule::plan(&layout, &[0, 1, 2, 3]);
        assert_eq!(plan.runs(), 1);
        assert_eq!(plan.span_bytes(), plan.bytes());
        assert_eq!(plan.span_bytes(), 4 * layout.vector_bytes() as u64);
    }

    #[test]
    fn span_bytes_exceed_payload_when_padding_is_dragged() {
        // 1 vector/block: a run over adjacent blocks 5,6,7 reads three whole
        // blocks but only carries three vectors of payload. With block_size 16
        // == vector_bytes there is no intra-block padding, so the gap comes from
        // the slots that *would* sit between them if the block held more — here
        // it is exactly payload (each block is one slot). Use a layout with
        // spare room per block so padding actually appears.
        let layout = BlockLayout::new(Dtype::F32, 4, 64).unwrap(); // 16B vec, 4/block
        // Indices 0 and 4 live in blocks 0 and 1 (adjacent) => one run. The
        // executor streams from slot 0 of block 0 through slot 0 of block 1,
        // i.e. across block 0's three trailing slots of padding.
        let plan = FetchSchedule::plan(&layout, &[0, 4]);
        assert_eq!(plan.runs(), 1);
        assert!(plan.span_bytes() > plan.bytes());
        // Payload is two vectors; the span reaches from block 0 slot 0 to block
        // 1 slot 0 inclusive.
        assert_eq!(plan.bytes(), 2 * layout.vector_bytes() as u64);
        let (_, len) = layout.run_span(0, 4);
        assert_eq!(plan.span_bytes(), len as u64);
    }

    #[test]
    fn span_bytes_sum_matches_per_run_spans() {
        // Blocks {0,1, 5, 9,10} over 1-vector blocks => runs {0,1}, {5}, {9,10}.
        let layout = one_per_block();
        let plan = FetchSchedule::plan(&layout, &[10, 0, 9, 1, 5]);
        let order = plan.order();
        let mut expected = 0u64;
        for (start, count) in plan.run_spans(&layout) {
            let first = order[start];
            let last = order[start + count - 1];
            expected += layout.run_span(first, last).1 as u64;
        }
        assert_eq!(plan.span_bytes(), expected);
        assert!(plan.span_bytes() >= plan.bytes());
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

            // run_spans partitions the order into exactly `runs` contiguous
            // windows.
            let spans = plan.run_spans(&layout);
            proptest::prop_assert_eq!(spans.len() as u64, plan.runs());
            let mut next = 0usize;
            let mut expected_span_bytes = 0u64;
            for (start, count) in &spans {
                proptest::prop_assert_eq!(*start, next);
                proptest::prop_assert!(*count >= 1);
                let first = order[*start];
                let last = order[*start + *count - 1];
                expected_span_bytes += layout.run_span(first, last).1 as u64;
                next += count;
            }
            proptest::prop_assert_eq!(next, order.len());

            // span_bytes equals the summed run spans and never undercounts the
            // payload it carries.
            proptest::prop_assert_eq!(plan.span_bytes(), expected_span_bytes);
            proptest::prop_assert!(plan.span_bytes() >= plan.bytes());
        }
    }
}
