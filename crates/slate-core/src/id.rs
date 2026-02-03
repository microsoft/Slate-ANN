//! Stable vector identifiers.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Stable identifier for a vector within an index.
///
/// A transparent newtype over `u64`. IDs are assigned densely from `0` at build
/// time and remain valid across the soft-delete / buffered-insert update path,
/// so an ID never refers to a different vector once issued.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
#[serde(transparent)]
pub struct VectorId(pub u64);

impl VectorId {
    /// Construct an identifier from its raw value.
    #[inline]
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    /// Return the raw `u64` value.
    #[inline]
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Return the identifier as a `usize` index (for in-RAM array addressing).
    #[inline]
    pub const fn as_index(self) -> usize {
        self.0 as usize
    }
}

impl From<u64> for VectorId {
    #[inline]
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl From<usize> for VectorId {
    #[inline]
    fn from(value: usize) -> Self {
        Self(value as u64)
    }
}

impl From<VectorId> for u64 {
    #[inline]
    fn from(value: VectorId) -> Self {
        value.0
    }
}

impl From<VectorId> for usize {
    #[inline]
    fn from(value: VectorId) -> Self {
        value.0 as usize
    }
}

impl fmt::Display for VectorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips_through_integer_types() {
        let id = VectorId::new(42);
        assert_eq!(id.get(), 42);
        assert_eq!(id.as_index(), 42usize);
        assert_eq!(u64::from(id), 42);
        assert_eq!(usize::from(id), 42usize);
        assert_eq!(VectorId::from(42u64), id);
        assert_eq!(VectorId::from(42usize), id);
    }

    #[test]
    fn orders_and_displays() {
        assert!(VectorId::new(1) < VectorId::new(2));
        assert_eq!(VectorId::new(7).to_string(), "#7");
    }
}
