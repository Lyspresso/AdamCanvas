# Windows as a first-class platform

Status: **active** — decided 2026-08-02 by Lydia. Audit basis: four-way
read-only portability audit of that day's `main` (findings summarized in the
PR descriptions of the W series).

## The two requirements (Lydia's words, our north stars)

1. **Code changes made on Windows transfer to the Mac version immediately
   with basically no work** — and vice versa. One codebase, never a fork.
   Enforced mechanically: CI builds and tests every change on both OSes, so
   a merge is proof of both. `.gitattributes` normalizes line endings so
   editors on either OS produce byte-identical files.
2. **A person using a Mac and a Windows machine has shared saves with shared
   data.** The library format itself must be OS-neutral: nothing under
   Adam's data root may be persisted as an absolute native path, tree
   hashing and asset identity must not depend on the OS path separator, and
   the existing three-way library merge must treat a Windows-written and a
   Mac-written library as the same lineage.

## Phases

### W1 — builds, boots, and tests itself on Windows (this series' first PR)
- wgpu backend `METAL` → `PRIMARY` (Metal on macOS, DX12/Vulkan on Windows).
- `assets.rs` portability: unix-only imports removed; **tree hash and sort
  use a portable path encoding** (components joined with `/`, UTF-8 —
  byte-identical to today's macOS hashes for UTF-8 names, so existing
  libraries keep their identities; requirement 2 depends on this);
  `O_NOFOLLOW` gated to unix with a symlink pre-check on Windows.
- Cross-process file locks (library + AI resume sidecar) moved from unix
  `flock` to std's portable file locks — the two-instance corruption guard
  now exists on Windows too, one code path everywhere.
- `windows_subsystem = "windows"` so no console window opens behind the app.
- CI gains a `windows-latest` job running `cargo test`. This job is the
  Windows test rig until a physical machine exists.
- Unix-only tests gated `#[cfg(unix)]` (asset symlink/metadata tests).

### W2 — shared saves (requirement 2's main work) — *shipped in the W2 PR*
- Persist data-root-relative paths for everything under the data root
  (managed assets, chat sandboxes); resolve against the *current* root on
  load. Absolute user paths (chosen working folders, file tiles pointing at
  unmanaged files) stay absolute but must round-trip unharmed through a
  foreign-OS session (no normalization, no clobbering on save).
- Forward slashes as the stored separator everywhere; parse either.
- Reconcile with the known library.json landmines (no version gate, no
  restore-on-empty) before changing any persisted representation: additive
  fields only, old readers must keep working.
- Case-insensitive identity consideration for NTFS (drive-letter folding in
  artifact identity keys).
- Cross-OS round-trip tests: write on "mac", read on "windows" (path-shape
  simulation in unit tests), plus a real shared-folder scenario when a
  Windows machine is available.

### W3 — the AI harness works on Windows
- CLI discovery: PATHEXT (`.exe`/`.cmd`/`.bat`), Windows install dirs, npm
  shim spawning via `cmd /C`.
- Process-tree termination via Job Objects (kill-on-close) replacing the
  unix process-group kill; a graceful-shutdown phase before hard kill.
- Provider matrix v1: Claude, Codex, Ollama, LM Studio, HTTP. Grok/Kimi
  join only after Windows-native CLIs are confirmed and their contracts are
  re-captured *on Windows* (the fixture process from
  `tests/fixtures/ai/grok/*/manifest.json`).
- Chat-sandbox honesty: the sandbox-write allowance leans on the CLI's own
  OS sandbox; until a provider's Windows sandboxing is verified, Windows
  keeps the stricter permission gating.
- Install buttons: PowerShell vendor installers or hidden on Windows.
- Port the kill/probe test stubs (`#!/bin/sh` scripts) to portable helpers.

### W4 — feels native (taste decisions, each its own call)
- Explorer-pipeline thumbnails (`IShellItemImageFactory`) replacing the
  qlmanage path for documents/video; HEIC decoding via WIC (iPhone photos —
  the biggest functional gap on a photo canvas); `explorer /select` for
  Reveal; app icon + version resources (`winresource`), installer (Inno or
  MSIX), SmartScreen signing when distribution warrants it; Windows OCR
  (`Windows.Media.Ocr`) if wanted; Excel live-mirror via COM **or**
  explicitly keep the portable save-watcher only.

## Working agreement

- Everything lands behind the two-OS CI gate; a red Windows job blocks merge
  the same as a red macOS job.
- No `#[cfg]` sprawl in feature code: platform differences live in
  `platform.rs`-style seams with one portable call site.
- Codex's file-ownership rules from AGENTS.md continue to apply; W-series
  changes inside Codex-owned files stay minimal and pattern-following, the
  same as the contract-row precedent.
