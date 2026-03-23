//! Language processing module.
//!
//! Minimal Italian tokenization, HDC-based vocabulary with distributional
//! learning, sentence composition, and conversational context.

pub mod tokenizer;
pub mod vocabulary;
pub mod composer;
pub mod context;
pub mod grounding;
