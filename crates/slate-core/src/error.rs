//! Central error type for the Slate-ANN engine.

use crate::id::VectorId;

/// Result alias used throughout Slate-ANN.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors produced by the Slate-ANN engine.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An underlying I/O operation failed (file open, read, mmap, etc.).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A vector's dimensionality did not match the index.
    #[error("dimension mismatch: index expects {expected}, got {got}")]
    DimensionMismatch {
        /// Dimensionality the index was built with.
        expected: usize,
        /// Dimensionality of the offending vector.
        got: usize,
    },

    /// A configuration value was invalid or self-inconsistent.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    /// A requested vector id is not present in the index.
    #[error("vector {0} not found")]
    NotFound(VectorId),

    /// An on-disk structure failed validation (bad magic, version, or length).
    #[error("corrupt index: {0}")]
    Corrupt(String),

    /// A requested operation or combination of options is not supported.
    #[error("unsupported: {0}")]
    Unsupported(String),
}

impl Error {
    /// Convenience constructor for [`Error::InvalidConfig`].
    pub fn invalid_config(msg: impl Into<String>) -> Self {
        Error::InvalidConfig(msg.into())
    }

    /// Convenience constructor for [`Error::Corrupt`].
    pub fn corrupt(msg: impl Into<String>) -> Self {
        Error::Corrupt(msg.into())
    }

    /// Convenience constructor for [`Error::Unsupported`].
    pub fn unsupported(msg: impl Into<String>) -> Self {
        Error::Unsupported(msg.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_render() {
        let e = Error::DimensionMismatch {
            expected: 768,
            got: 512,
        };
        assert_eq!(
            e.to_string(),
            "dimension mismatch: index expects 768, got 512"
        );

        let e = Error::NotFound(VectorId::new(9));
        assert_eq!(e.to_string(), "vector #9 not found");
    }

    #[test]
    fn io_errors_convert() {
        let io = std::io::Error::new(std::io::ErrorKind::NotFound, "missing");
        let e: Error = io.into();
        assert!(matches!(e, Error::Io(_)));
    }
}
