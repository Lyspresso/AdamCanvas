# Multi-agent coordination

Two AI agents work on this repo in parallel, each in its own checkout on its own branch:

| Directory | Branch | Who works here |
|---|---|---|
| `~/Developer/AdamCanvas` (this one) | `work/agent-a` | The non-Claude agent harness |
| `~/Developer/AdamCanvas-claude` | `work/claude` | Claude Code |

`main` is the integration branch. It is deliberately not checked out in any worktree and only advances when the two branches are merged at the end ("convergence").

## Rules

1. Work only inside your own directory. Never edit files under the other agent's directory.
2. Stay on your branch. Do not `git switch`/`git checkout` to another branch, and never commit to `main`.
3. Commit early and often on your branch — small, compiling commits. This is what makes the final merge tractable.
4. Stick to the modules your assigned task owns. Avoid drive-by edits (refactors, formatting sweeps, renames) in shared files — `src/app.rs`, `src/lib.rs`, `src/main.rs`, `Cargo.toml` — beyond the minimum wiring your task needs. Widespread edits there guarantee merge conflicts.
5. Adding a dependency to `Cargo.toml` is fine; append to the end of the `[dependencies]` block rather than re-sorting it.
6. Before declaring your task done: `cargo fmt`, `cargo clippy`, `cargo test`, make sure `cargo build` succeeds, and commit everything (no dirty working tree).

## Build

- `cargo check` — fast validation
- `cargo run` — run the app (macOS)
- `cargo test` — tests live in `Tests/` and inline
- `scripts/build_app.sh` — produce the `Adam.app` bundle in `build/`

## Convergence

At the end, both branches get merged into `main` (Claude/user handles this): merge `work/agent-a`, then `work/claude`, resolve conflicts, then run fmt + clippy + test + build as the gate.
