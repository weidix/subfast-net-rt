use std::path::Path;

use crate::error::{Result, RtError};

#[derive(Debug, Clone)]
pub struct WeightStore {
    data: Vec<u8>,
}

impl WeightStore {
    pub fn load(path: &Path) -> Result<Self> {
        let data = std::fs::read(path).map_err(|e| RtError::io(path, e))?;
        Ok(Self { data })
    }

    pub fn from_bytes(data: Vec<u8>) -> Self {
        Self { data }
    }

    pub fn total_bytes(&self) -> usize {
        self.data.len()
    }

    pub fn bytes(&self) -> &[u8] {
        &self.data
    }

    pub fn slice(&self, offset: usize, byte_length: usize) -> Result<&[u8]> {
        let end = offset
            .checked_add(byte_length)
            .ok_or_else(|| RtError::WeightsOutOfBounds {
                tensor: "<unknown>".into(),
                offset: offset as u64,
                length: byte_length as u64,
                file_len: self.data.len() as u64,
            })?;
        self.data
            .get(offset..end)
            .ok_or_else(|| RtError::WeightsOutOfBounds {
                tensor: "<unknown>".into(),
                offset: offset as u64,
                length: byte_length as u64,
                file_len: self.data.len() as u64,
            })
    }
}
