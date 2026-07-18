use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("json: {0}")]
    Json(#[from] serde_json::Error),

    #[error("missing model file: {0}")]
    MissingModel(PathBuf),

    #[error("invalid plda layout: {0}")]
    Plda(String),

    #[error("pipeline: {0}")]
    Pipeline(String),

    #[error("not implemented: {0}")]
    NotImplemented(&'static str),
}

pub type Result<T> = std::result::Result<T, Error>;
