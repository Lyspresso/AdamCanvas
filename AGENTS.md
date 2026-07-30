# Multi-agent coordination

This repo (`~/Developer/AdamCanvas`) coordinates parallel work by Claude Code and OpenAI Codex on the Adam canvas app. Codex works **outside** this repo (remote workspace; it never reads this file) — its finished source gets snapshotted in.

## Layout

| Ref | Checked out at | Meaning |
|---|---|---|
| `main` | `~/Developer/AdamCanvas` | Canonical copy: pre-AI-chat baseline + the `src/ai/` chat implementation. Integration happens here. |
| `work/claude` | `~/Developer/AdamCanvas-claude` | Claude Code's task branch, forked from `main`. |
| `work/codex` | *(nowhere)* | Codex's AI-harness work, forked from the pre-AI-chat root commit. |

History shape: root `5d5c708` (pre-AI-chat baseline) has two children — `main` (with the `src/ai/` implementation) and `work/codex` (with Codex's flat `src/ai*.rs` + `chat_core.rs` harness). **The two AI implementations overlap and must be reconciled at convergence.**

## Codex ingestion (read-only!)

Codex's deliverables land in `~/Documents/Codex/<date>/tanbiralam-claude-code-https-github-com/outputs/` (source exports + built .app bundles). The copy in `~/Documents/adam canvas hub/` is a stale intermediate. Never edit anything under `Documents/` — to ingest, stage the **newest** source export onto `work/codex` with the temp-index recipe (see `git log work/codex` commit for provenance) or a temporary worktree.

## Rules

1. Claude edits only `~/Developer/AdamCanvas-claude` on `work/claude`.
2. Agents don't commit to `main` outside convergence (the user may; it's their canonical copy).
3. Small, compiling commits on task branches — that's what makes merges tractable.
4. Shared hot files (`src/app.rs`, `src/lib.rs`, `src/main.rs`, `Cargo.toml`): minimum wiring only, no drive-by refactors. Append new deps at the end of `[dependencies]`.
5. Task done = `cargo fmt` + `cargo clippy` + `cargo test` + `cargo build` pass, everything committed.

## Build

- `cargo check` / `cargo run` / `cargo test`
- `scripts/build_app.sh` → `Adam.app` bundle in `build/`

## Convergence

1. Re-snapshot the newest Codex export onto `work/codex` (if newer than the last snapshot).
2. Decide the AI-implementation reconciliation (which harness is canonical, what gets ported from the other).
3. Merge `work/codex` and `work/claude` into `main`; resolve conflicts.
4. Gate: `cargo fmt` + `clippy` + `test` + full build.
