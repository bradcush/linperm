use core::fmt;

#[derive(Debug)]
pub enum CoreError {
    NotAPermutation { len: usize, index: usize },
    NotPowerOfTwo { len: usize },
    NumVarsMismatch { expected: usize, got: usize },
    OddNumVars { num_vars: usize },
}

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotAPermutation { len, index } => write!(
                f,
                "not a permutation of [0, {len}), failed at index {index}",
            ),
            Self::NotPowerOfTwo { len } => {
                write!(f, "input length {len} is not a power of two")
            }
            Self::NumVarsMismatch { expected, got } => write!(
                f,
                "number of variables {got} does not match expected {expected}",
            ),
            Self::OddNumVars { num_vars } => write!(
                f,
                "number of variables {num_vars} must be even for splitting",
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CoreError {}
