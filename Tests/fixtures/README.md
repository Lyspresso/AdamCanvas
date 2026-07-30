# AI stream fixtures

These files are frozen captures from the locally installed CLI versions on
2026-07-29. They are intentionally compact and must not be regenerated or
pretty-printed in place.

- `codex-minimal.jsonl` — `codex-cli 0.144.1`, successful `exec --json` turn.
- `grok-minimal.jsonl` — successful `--output-format streaming-json` turn.
- `claude-auth-failure.jsonl` — Claude Code 2.1.128, real stream shape from an
  unauthenticated headless turn. A successful full-lifecycle capture is still
  required before enabling Claude by default.

Parser tests replay these byte-for-byte at deliberately hostile chunk sizes.
