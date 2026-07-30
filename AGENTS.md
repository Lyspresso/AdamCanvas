# AdamCanvas — multi-harness coordination

Source of truth: https://github.com/Lyspresso/AdamCanvas (private). All work converges there, gated by CI.

## Branch map

| Branch | Checked out at | What it is |
|---|---|---|
| `main` | `~/Developer/AdamCanvas` | The app — Codex's provider-neutral AI-harness line |
| `work/claude` | `~/Developer/AdamCanvas-claude` | Claude Code's workstream (project infrastructure) |
| `work/codex` | *(nowhere)* | Landing branch for Codex's work, snapshotted from its exports |
| `archive/ai-chat-developer-copy` | *(nowhere)* | The abandoned `src/ai/` chat variant (~35k lines), kept for salvage |

## Codex bridge

Codex currently works in a remote workspace detached from this repo. Its finished source appears under `~/Documents/Codex/<date>/.../outputs/` — everything there is **READ-ONLY**. Ingestion: snapshot the newest export as a commit on `work/codex` (staged via a temp index against the export directory, no checkout needed), then merge `work/codex` into `main`.

Target state: attach Codex to the GitHub repo so its tasks arrive as pull requests instead of folder exports. If Codex ever runs *locally*, give it its own worktree (e.g. `~/Developer/AdamCanvas-codex` on `work/codex`) — never a directory another agent or the user already works in.

## Rules

1. Claude edits only `~/Developer/AdamCanvas-claude` on `work/claude`, and never touches the `Documents/` copies.
2. `main` advances only by merge (or by the user directly). CI must be green.
3. Small, compiling commits. No drive-by refactors in files the other harness owns — Codex currently owns `src/ai*.rs`, `src/chat_core.rs`, and its `app.rs`/`domain.rs` surface.
4. New Cargo dependencies: append at the end of `[dependencies]`, don't re-sort.
5. Done = `cargo fmt --check`, `cargo clippy`, `cargo test` all pass locally.

## Build & test

- `cargo check` / `cargo run` / `cargo test` (267 tests as of 2026-07-30)
- `scripts/build_app.sh` → `build/Adam.app`
- CI (`.github/workflows/ci.yml`): fmt + clippy (advisory) + tests on macOS

## Convergence

1. Re-snapshot the newest Codex export onto `work/codex`; merge into `main`.
2. Merge `work/claude` into `main`.
3. Gate: CI green; `cargo build --release` locally if shipping a build.
