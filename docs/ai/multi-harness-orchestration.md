# Multi-harness orchestration — research log & operating model

Status: **living research document**. Maintained by Claude (integrator). Started 2026-07-30.
Scope rule set by Lydia: everything here is documentation and process. **None of the
"future" section goes into Adam's code until the safety gates below are validated.**

## 1. Operating model today

Three parties, three roles, one source of truth:

| Party | Role | Works in |
|---|---|---|
| Lydia | Product owner — picks goals, arbitrates, approves irreversible steps | anywhere |
| Claude (Claude Code) | Integrator / project manager — repo custody, review, merges, CI, task assignment, documentation | `~/Developer/AdamCanvas-claude` on `work/claude` |
| Codex | Feature developer — harness internals (PRs 1–4 of the parity plan) | remote workspace → PRs (target) or export ingestion (fallback) |

Source of truth: https://github.com/Lyspresso/AdamCanvas (private). `main` advances only
by reviewed merge with CI green. Spec of record for current work:
`docs/plans/progress-artifacts-parity.md`.

The roles are seats, not identities: today's external pair is Claude + Codex, but the
same model applies to any harness pair (GPT/Codex, Grok, Kimi, LM Studio and other
local endpoints — see §5).

## 2. How Codex operates (observed 2026-07-29 → 30)

- Runs in a remote workspace **detached from any local checkout**. Local folders are
  receipts, not working state.
- Deliverables sync to `~/Documents/Codex/<date>/<workspace>/outputs/`: source-tree
  exports, zips, built `.app` bundles, and standalone report documents. File
  modification times are **preserved from the remote run**, and sync can lag task
  completion by hours — a folder's contents can silently change long after its
  timestamps suggest. Never trust folder freshness; verify by content diff.
- Exports carry no `.git`. Ancestry must be reconstructed on our side (the
  `work/codex` snapshot branch, forked at the true base commit).
- Writes good design docs, but documentation can lead implementation: `AI-HARNESS.md`
  described a task-tool bridge and a wider effort range before either existed/worked.
  Treat its docs as intent, its code as fact, and reconcile explicitly (parity plan PR 1).
- Given a repo to audit, it produces accurate findings (its parity audit independently
  matched our code-level dig and correctly identified the unmerged-tree mismatch).

## 3. Incident log (lessons that shaped the process)

| Date | Incident | Lesson → rule |
|---|---|---|
| 07-29 | Two AI harnesses independently built overlapping AI-chat features from the same folder-copy base | Single source of truth + explicit task split *before* work starts |
| 07-30 | Screenshot session ran `Adam 3.app` from an export newer than the repo; debugging targeted code the repo didn't have | Ingest before diagnose: verify the running build's provenance first |
| 07-30 | Grok `--reasoning-effort max` crash: capability list widened to match aspirational docs, not the installed CLI | Capabilities must be probed/versioned, single-sourced (PR 1) |
| 07-30 | Parent + 5 subagents' text interleaved into one garbled cell | Multi-agent streams need identity routing end-to-end (PRs 1/3) |

## 4. PM protocol (Claude)

1. **Watch**: a standing monitor polls the repo every 90 s for new PRs, new `codex/*`
   branches, and CI failures (session-scoped; re-arm at session start).
2. **Review**: each Codex PR is checked against the parity plan's per-PR scope and the
   acceptance scenario; CI must be green; diffs are read, not skimmed — especially
   capability tables and stream-handling code (see incident log).
3. **Merge**: in plan order (1 → 2 → 3∥5 → 4). Conflicts in `src/app.rs` resolve in
   favor of lane ownership (inspector render = Claude, data/projections = Codex).
4. **Assign**: after each merge, confirm the next PR's scope still matches reality;
   adjust the plan doc first, then prompt Codex (prompts embed any context Codex
   cannot reach, e.g. EarlIt mechanics from the local Swift source).
5. **Escalate**: product decisions, scope changes, and anything irreversible go to
   Lydia. Nothing is force-pushed; `main` history is append-only.

## 5. Future: two coordinated agents inside Adam (design research only)

> Shipped adjacent to (not from) this section, 2026-07-30: the Agents panel
> (`src/agents_panel.rs`, branches `work/claude-agents` + `-install`) —
> provider detection, a chat setup screen, and one-click installs bounded
> by a compiled-in allowlist of vendor-verified commands (user-initiated,
> logged, post-verified; no agent may trigger one). None of the seven gates
> below are touched; sign-in probes remain future work, and seat
> eligibility surfacing can later build on its verified-version statuses.

Goal stated by Lydia 2026-07-30: eventually either (a) a standing PM arrangement where
one agent leads another (as Claude leads Codex on this repo today), and/or (b) **Adam
itself** can run two AI agents that collaborate this way on any project.

**This is provider-neutral by design, not a Codex feature.** Either seat — manager or
worker — should be fillable by any provider Adam's harness supports: Claude CLI, Codex
CLI (GPT models), Grok CLI, Kimi CLI, LM Studio or other OpenAI-compatible local/remote
endpoints, Ollama, or a custom CLI. The pairing on this repo (Claude + Codex) is just
the first instance of the pattern. Seat eligibility must derive from the **verified
capability profile**, not the provider's marketing: the manager seat needs dependable
planning, review, and tool-calling; providers without a native plan channel or session
continuity (e.g. Grok today, most local models) rely on the task-tool contract and
bounded replay — which is exactly why that contract, not any provider-native feature,
is the coordination substrate. A small local model can hold the worker seat for scoped
tasks while a stronger model manages; the reverse pairing should be refused by profile,
not by hardcoded provider names.

Adam already has most primitive ingredients: multi-provider CLI spawning, a permission
tool-gate, per-child session identity (round 2), task tools (parity PR 2), and
event-sourced projections. A "manager + worker" mode would compose them: a manager
agent that plans/reviews and a worker agent that executes, coordinating through the
same task-tool contract this repo uses between Claude and Codex.

**Safety gates that must be validated before any of this ships in Adam:**

1. **Workspace isolation** — each agent gets its own working copy (worktree
   equivalent); two agents never share a mutable folder (root cause of incident #1).
2. **Mediated channel** — agents exchange structured task/review events through Adam,
   never raw shell access to each other's processes or folders.
3. **Independent permission scopes** — the existing tool gate applies per agent;
   a manager cannot approve the worker's escalations by itself.
4. **Budget & turn caps** — hard limits on tokens/turns/wall-clock per agent per task
   (one observed Cowork turn consumed >500k input tokens unsupervised).
5. **Human merge gate** — combining the two agents' outputs is an explicit user (or
   trusted-integrator) action, mirroring the PR review gate.
6. **Provenance labeling** — every artifact records which agent produced it (already
   the direction of parity PR 4).
7. **No self-modification** — an in-Adam agent pair must not operate on Adam's own
   source tree from inside a running Adam.

Prerequisite before prototyping: parity PRs 1–3 landed (identity-routed streams and
the task-tool contract are the substrate), plus a written eval of the failure modes
above. Until then this section is a spec sketch, not a work item.
