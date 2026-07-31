# Progress & Artifacts parity plan (EarlIt → Adam)

Status: **active**. Agreed 2026-07-30 by Lydia + Claude + Codex (joint audit).
Division: **Codex owns PRs 1–4** (harness internals & card data). **Claude owns PR 5** (task-rail visuals) on `work/claude`. Line refs are as of merge `78be2d5`; cite function names over line numbers — they drift.

## Diagnosis (confirmed in code)

1. **Progress empty on Grok runs.** `decode_grok` handles only `thought|text|end|error` — no plan events can arrive live. `capability_profile` assigns Grok `PlanChannel::AppTaskTools`, but the app-task-tools loopback is declared in `docs/AI-HARNESS.md` and **not implemented**. The post-exit session harvest (`harvest_grok_session_directory` → `grok_current_turn_updates`) recovers a todo only if the agent used Grok's own `todo_write`, requires `session_id` from an `end` event, and `grok_current_turn_updates` clears prior updates on each `user_message_chunk` (multi-turn keeps only the last turn).
2. **Garbled prose with subagents.** Parent + N children all emit `{"type":"text"}` on one stdout; `decode_grok` discards envelope identity fields; all text funnels into one `output` buffer + one `Delta` stream → interleaved word salad. Child identity fields (`subagent_id`, `parent_session_id`, `child_session_id`) are already parsed in `decode_grok_session_update` — the live text path just doesn't use them.
3. **Effort `'max'` crash.** `allowlisted_reasoning_effort` and the UI combo (`ai_reasoning_options`) hold hand-duplicated per-provider literals. Round 2 widened Grok's to the doc's aspirational list (`…xhigh, max`); the installed CLI accepts `low|medium|high` → launch aborts. No single source of truth, no version awareness.
4. **Artifacts (“Outputs”) stay 0.** `project_artifacts` is correct but starves: Grok emits no live `FileChange`; canvas creations don't currently emit `HostMutation` in production; a web-research run legitimately creates no files. “Working folder · 60” is an unrelated capped `read_dir` listing (`refresh_ai_workspace_files`, `truncate(60)`) that reads like task output.
5. **Visuals.** Flat accordion gives equal weight to plan, diagnostics, and folder counts; the replay meter reads as task progress. EarlIt uses distinct cards + a connected vertical stepper + real empty states.
6. **Latent:** stream-salvage double-append — `refresh_poison_salvage` can clear/re-record the whole buffer while `emit_stream_reset` is one-shot, so UI `streamed_text` accumulates a duplicate.

## Target rail (per Codex's proposal, adopted)

```
Main task
├── Progress: main agent's plan (never derived from prose/commands/subagent count)
├── Agents: child-agent lifecycle (aggregate "3/5 done · 2 working"; optional real child checklists)
├── Artifacts: produced files / canvas entities (provenance, actions)
└── Resources: working folder, context, usage (collapsed by default)
```

## The EarlIt task-store contract (implementation spec for PR 2)

Verified against EarlIt source (`AgentTaskStore.swift`, `AgentActivityModels.swift`, `MCPTools.swift`, `AgentCapabilityProfile.swift`):

- Three agent-facing tools, **no delete**: `task_create{content required, activeForm}` → returns `task_id`; `task_update{task_id required, status, content, activeForm}` — **unknown id creates** with that id; `task_list{}` read-only, emits no events.
- Every mutating call emits **two** events: a `TaskMutation` (transcript row) then a `PlanUpdate` carrying the **entire re-reduced list**. Whole-list snapshots, never deltas — the accumulator replaces the previous plan in place.
- Statuses: `pending | in_progress | completed | cancelled`. `activeForm` is present-continuous ("Synthesizing findings…") and is what the UI shows while in progress. *(Wire spelling aligned 2026-07-30 to Adam's implementation and provider spellings; EarlIt's Swift source uses `inProgress`, which is the same status, not a different one.)*
- Origin tracking: `Native` vs `AppTools`. A native snapshot **replaces** native-origin rows; app-tool rows survive and re-append after. Adam already implements this in `merge_plan_snapshot` — keep it.
- **Exposure gate:** an agent sees exactly one channel. Claude/Codex CLIs → native plan stream, task tools withheld. Grok/custom/HTTP → task tools exposed, gated at tool-list time AND call time. Fail closed for dead runs.
- Field hygiene: trim, reject empty, cap field bytes (EarlIt: 512), reject unknown arguments.
- Live store wins when non-empty; else newest persisted `PlanUpdate`; else empty. Persist the trailing snapshot with the turn so relaunch restores order + status.

## PR sequence (each starts from current `main`)

### PR 1 — `codex/harness-baseline`
- Capture real event fixtures per provider/version (Codex CLI JSONL, Claude Code stream-json — schemas in github.com/openai/codex and github.com/anthropics/claude-code — plus locally captured Grok + Kimi streams) and pin decoder tests on them.
- Version-aware `CapabilityProfile`: one table per provider (incl. supported reasoning-effort values) that **both** the launch arg builder and the settings UI read. Unsupported values are unselectable and clamped with self-heal; a setting the CLI rejects must be impossible to launch.
- **Stopgap for garbled prose** — *revised by evidence during PR 1:* the plan assumed live Grok text envelopes carry child identity; the captured fixture (`tests/fixtures/ai/grok/0.2.111/parent-child.jsonl`) proved they don't — parent and child `type:text` records are structurally indistinguishable, so identity-keyed suppression would corrupt valid parent prose. Shipped fail-closed equivalent: Grok 0.2.111 and unknown versions always launch `--no-subagents` (saved preference healed to Off, composer explains why). PR 3 re-enables child execution only via a genuinely scoped channel.
- Fix the salvage double-append (allow repeated `StreamReset` or make salvage idempotent on the UI buffer).
- Explicit terminal outcomes: `completed | stopped | blocked | timedOut | turnLimit | providerError` — drives the run header; checked rows alone never imply success.
- Reconcile `docs/AI-HARNESS.md` claims with actual implementation.

### PR 2 — `codex/main-progress`
- Implement the app task tools per the contract above, over the existing live tool-gate channel; system-prompt nudge for plan-channel-less providers.
- Poll Grok's session `updates.jsonl` **during** the run (it is already parsed post-exit) → live todos/tool calls/subagent events mid-run; fix multi-turn retention (don't discard prior turns' plan on `user_message_chunk`).
- Persist live snapshots with the turn; relaunch restores.
- Task tools remain independent of canvas access.

*PR 2 merged 2026-07-30 (77467b4) after re-review; carried P2 follow-ups: Grok `{slug}-{hash}`/`.cwd` long-path session discovery, replay tool-call map accounting vs max_events, permission-tool label preservation on PermissionResolved, >512-byte native task ids, stale denied-permission diagnostic clearing.*

### PR 3 — `codex/subagent-progress`
- Child lifecycle with stable child ids + parent linkage; every task/text event scoped `Main` or a specific child.
- Aggregate line ("3/5 done · 2 working"); expandable child detail; child checklist **only** from real child task events; otherwise status + current activity — never invented steps.
- Per-child prose cells; **re-enabling Grok subagents requires a genuinely scoped child channel** (e.g. the session updates file, which does carry `subagent_id`/`child_session_id`) — live stdout cannot provide it on Grok 0.2.111 (see PR 1 deviation). `supports_scoped_child_text` flips per version only with fixture proof.
- *Candidate channel (pending PR 2 review, 2026-07-30):* Codex's in-flight work adds a Grok **ACP** (Agent Client Protocol) transport (`src/grok_acp.rs`) that may supersede session-file polling and provide the scoped channel directly. Not yet accepted spec — evaluated on evidence (fixtures + tests) when PR 2 opens; if accepted, this section's mechanism updates accordingly.

### PR 4 — `codex/artifacts`
- User-facing rename Outputs → Artifacts. Sources: confirmed successful `FileChange` (failed/declined excluded) + **created** canvas entities (emit `HostMutation` from production canvas tools; creations only — annotate/move don't count).
- Provenance (turn, tool, agent), dedupe by stable path/entity id, later delete strikes earlier add, deleted items visibly struck.
- Actions: Preview, Reveal in Finder, jump-to-canvas. Compact rail capped (EarlIt caps at 8 + "+N more"); searchable cross-conversation library behind it.

### PR 5 — `task-rail` (Claude, `work/claude`)
- Card rail per the target structure; Resources collapsed by default; Activity/diagnostics moved out of Progress.
  *Shipped 2026-07-30:* the stepper; Activity relocated out of the Progress card; cards renamed Agents/Artifacts (rename moved up from PR 4's scope — rendering is Claude's lane); Working folder demoted — count dropped from its header, opens only when a folder is needed but unset; header status chip now reflects the terminal outcome (Completed / Stopped / Needs attention). Working folder + Context remain sibling collapsed cards rather than one Resources wrapper — same demotion intent, simpler tree.
- EarlIt stepper grammar: filled circle + check (done) / stroked circle + spinner (active, label = `activeForm`) / hollow circle (pending) / slash (cancelled); short connector stub under each glyph; completed rows full-strength text, others secondary, cancelled struck.
- Empty states: live-run-no-plan → spinner + elapsed timer; idle → decorative placeholder + "Steps will show as the task unfolds."; finished-without-checklist → "Completed without a checklist." (never bare "No task list yet." on a finished run); persisted plan on idle → "Task complete."
- Accessibility: AccessKit labels, contrast, reduced-motion (static glyph instead of spinner), large-text reflow.
- Claude renders whatever the projections provide; card *data* shape changes belong to Codex's PRs.

## Acceptance scenario (five-agent research request)

1. Main Progress shows only the primary plan (e.g. Dispatch five research agents → Wait for findings → Synthesize → Deliver).
2. Agents shows all five children + live states, independently of Progress.
3. A child expands to its own checklist only if it published one.
4. Artifacts shows the produced report/canvas item immediately, with provenance.
5. Reload restores Progress, Agents, Artifacts identically.
6. No interleaved prose anywhere — each agent's words render in its own cell.
7. Unsupported provider settings cannot be selected or launched.
8. Terminal outcome — not checked rows — determines Completed/Failed/Stopped.

## Conflict protocol for this effort

`src/app.rs` is the shared hot file. Codex PRs 1–2 should avoid the inspector render functions (`render_ai_inspector` and its section fns); Claude's PR 5 stays inside them + new widget modules. PR 3/4 card-content handoffs: Codex lands the projection/data change first, Claude follows with rendering. Merge order: 1 → 2 → (3, 5 in parallel) → 4.

**Fixed points from the Agents-panel insert (branch `work/claude-agents`, outside this effort's numbering):** `src/ai.rs` gained one additive accessor — `pub struct ProviderProbe` + `pub fn probe_installed_provider(provider_id: &str, refresh: bool)` + a private cache-invalidation helper, placed directly after `clamp_provider_preferences` — and `src/app.rs` gained the module `src/agents_panel.rs`'s wiring: `agents: AgentsPanelState` field, a poll in `fn logic`, `show_agents_panel`, one quick-bar slot, `AiWorkspaceUiAction.open_agents_panel`, and one `preflight` parameter threaded into `render_ai_chat_page`. PR 2 should treat these as fixed points; none of them touch stream handling or projections. *(Stack ordering note resolved 2026-07-30: PR 2 merged first as planned; PR A merged with post-PR-2 main by the integrator (merge commit, history preserved).)* The follow-up branch `work/claude-agents-install` additionally threads an `AgentsChatView` parameter through `render_ai_chat_page`, embeds an `AgentsPanelAction` in `AiWorkspaceUiAction`, and conditionally renders the chat empty state as the agents setup screen — still no contact with stream handling or projections.
