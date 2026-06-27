//! Incremental update log layered over an immutable [`Bundle`](crate::Bundle).
//!
//! A built bundle is frozen: the vector store and index frame are laid out once
//! to minimise seeks and never rewritten. Real workloads still need to retract
//! and add a few vectors between rebuilds, so an [`UpdateLog`] sits *beside* the
//! frozen data as a small, query-time overlay:
//!
//! * **Soft-delete** — a tombstone set of vector ids. At query time a tombstoned
//!   id is simply dropped from the result list; the frozen index and its
//!   seek-optimised layout are never touched, so a delete costs no extra I/O.
//! * **Buffered-insert** — new vectors held in RAM (and a sidecar file). Each
//!   query brute-scans the small buffer and merges the matches into the top-k.
//!
//! The log persists as `updates.json` inside the bundle directory, next to
//! `manifest.json` / `vectors.svec` / `index.sidx`. It is the shock absorber
//! between full rebuilds, not a replacement for them.

use std::collections::BTreeSet;
use std::path::Path;

use serde::{Deserialize, Serialize};
use slate_core::{Error, Result};

/// File name of the update log inside a bundle directory.
pub const UPDATES_FILE: &str = "updates.json";

/// Soft-deletes and buffered inserts layered over a frozen bundle.
///
/// Ids below the frozen store's length refer to stored vectors; ids at or above
/// it refer to buffered inserts. `next_id` starts at the store length and only
/// ever grows, so a buffered vector keeps a stable id across reopen.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct UpdateLog {
    /// Tombstoned ids (stored or buffered). A `BTreeSet` keeps the JSON stable.
    tombstones: BTreeSet<u64>,
    /// Buffered insert vectors, in id order; `inserts[i]` has the id
    /// `next_id - inserts.len() + i` (== `base_len + i`).
    inserts: Vec<Vec<f32>>,
    /// Next id to hand out; initialised to the frozen store's length.
    next_id: u64,
}

impl UpdateLog {
    /// A fresh, empty log for a store holding `base_len` vectors.
    #[must_use]
    pub fn new(base_len: u64) -> Self {
        Self {
            tombstones: BTreeSet::new(),
            inserts: Vec::new(),
            next_id: base_len,
        }
    }

    /// Soft-delete a vector id (stored or buffered). Idempotent.
    pub fn delete(&mut self, id: u64) {
        self.tombstones.insert(id);
    }

    /// Append a buffered insert vector and return its freshly assigned id.
    pub fn insert(&mut self, vector: Vec<f32>) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.inserts.push(vector);
        id
    }

    /// Whether `id` has been tombstoned.
    #[must_use]
    pub fn is_tombstoned(&self, id: u64) -> bool {
        self.tombstones.contains(&id)
    }

    /// Whether the log carries no edits (fast path for queries).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tombstones.is_empty() && self.inserts.is_empty()
    }

    /// The tombstoned id set.
    #[must_use]
    pub fn tombstones(&self) -> &BTreeSet<u64> {
        &self.tombstones
    }

    /// The buffered insert vectors, in id order.
    #[must_use]
    pub fn inserts(&self) -> &[Vec<f32>] {
        &self.inserts
    }

    /// The id of the first buffered insert (== the frozen store length).
    #[must_use]
    pub fn first_insert_id(&self) -> u64 {
        self.next_id - self.inserts.len() as u64
    }

    /// The next id that would be handed out.
    #[must_use]
    pub fn next_id(&self) -> u64 {
        self.next_id
    }

    /// Persist the log to `dir/updates.json`.
    pub fn save(&self, dir: &Path) -> Result<()> {
        let json = serde_json::to_vec_pretty(self)
            .map_err(|e| Error::corrupt(format!("failed to serialize updates log: {e}")))?;
        std::fs::write(dir.join(UPDATES_FILE), json)?;
        Ok(())
    }

    /// Load the log from `dir/updates.json`, or a fresh empty log keyed to
    /// `base_len` if the file is absent.
    pub fn load(dir: &Path, base_len: u64) -> Result<Self> {
        let path = dir.join(UPDATES_FILE);
        if !path.exists() {
            return Ok(Self::new(base_len));
        }
        let bytes = std::fs::read(path)?;
        let log: UpdateLog = serde_json::from_slice(&bytes)
            .map_err(|e| Error::corrupt(format!("malformed updates log: {e}")))?;
        Ok(log)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_log_starts_at_base_len_and_is_empty() {
        let log = UpdateLog::new(100);
        assert!(log.is_empty());
        assert_eq!(log.next_id(), 100);
        assert_eq!(log.first_insert_id(), 100);
        assert!(log.inserts().is_empty());
        assert!(log.tombstones().is_empty());
    }

    #[test]
    fn delete_marks_tombstoned_and_is_idempotent() {
        let mut log = UpdateLog::new(10);
        log.delete(3);
        log.delete(3);
        log.delete(7);
        assert!(log.is_tombstoned(3));
        assert!(log.is_tombstoned(7));
        assert!(!log.is_tombstoned(4));
        assert_eq!(log.tombstones().len(), 2);
        assert!(!log.is_empty());
    }

    #[test]
    fn insert_assigns_sequential_ids_from_base_len() {
        let mut log = UpdateLog::new(50);
        let a = log.insert(vec![1.0, 2.0]);
        let b = log.insert(vec![3.0, 4.0]);
        let c = log.insert(vec![5.0, 6.0]);
        assert_eq!((a, b, c), (50, 51, 52));
        assert_eq!(log.next_id(), 53);
        assert_eq!(log.first_insert_id(), 50);
        // first_insert_id + i recovers each buffered id.
        for (i, _) in log.inserts().iter().enumerate() {
            let id = log.first_insert_id() + i as u64;
            assert_eq!(id, 50 + i as u64);
        }
    }

    #[test]
    fn round_trips_through_save_load() {
        let dir = tempfile::tempdir().unwrap();
        let mut log = UpdateLog::new(20);
        log.delete(2);
        log.delete(19);
        log.insert(vec![0.5, 0.25, 0.125]);
        log.insert(vec![9.0, 8.0, 7.0]);
        log.save(dir.path()).unwrap();

        let loaded = UpdateLog::load(dir.path(), 20).unwrap();
        assert_eq!(loaded, log);
        assert_eq!(loaded.next_id(), 22);
        assert!(loaded.is_tombstoned(19));
    }

    #[test]
    fn load_absent_file_yields_fresh_log() {
        let dir = tempfile::tempdir().unwrap();
        let log = UpdateLog::load(dir.path(), 42).unwrap();
        assert!(log.is_empty());
        assert_eq!(log.next_id(), 42);
    }
}
