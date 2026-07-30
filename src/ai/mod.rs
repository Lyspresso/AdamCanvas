//! Adam's local, CLI-backed AI chat system.
//!
//! Pure parsing, policy, projection, and composition live beside narrow
//! persistence/runtime edges. The app UI consumes value snapshots from here.

pub mod adam_tools;
pub mod context;
pub mod core;
pub mod host;
pub mod local_lm;
pub mod manage_ui;
pub mod memory;
pub mod policy;
pub mod prompt;
pub mod registration;
pub mod rich_text;
pub mod runtime;
pub mod store;
pub mod system;
pub mod task_tools;
pub mod tools;
pub mod ui;
