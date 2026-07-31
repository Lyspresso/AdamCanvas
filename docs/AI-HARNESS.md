# Adam AI harness

Adam’s AI system is a provider-neutral harness around locally installed agent
CLIs and local or remote chat-model endpoints. The full-page chat and the AI
tile are two views of the same persisted conversation.

This document describes the transferable harness behavior. It intentionally
does not cover historical SwiftUI/Xcode issues or unrelated canvas bugs such
as copy/paste and phantom tiles.

## Surfaces and modes

- **Chat** is a conversational surface. It supplies selected context and
  attachments but tells the provider not to modify the workspace.
- **Cowork** is outcome-oriented. It requires a working folder and lets a
  capable CLI agent inspect, edit, and verify work within that folder.
- **Code** is the same durable task surface with coding-specific instructions:
  inspect before editing, keep changes scoped, and run relevant verification.

The first Cowork or Code turn permanently classifies a conversation as a task,
even if its visible mode is later changed. Conversations can be pinned, show
unread completion state, and appear both as tiles and in the AI-chat sidebar.

## Provider contract

Provider support is derived from a capability profile rather than a single
pile of provider conditionals. Optional runtime controls are a second,
version-pinned profile shared by launch shaping and the composer. Adam probes
the exact installed built-in CLI version and exposes or emits a control only
when that version has a captured, tested contract. An unknown, unparseable, or
unlisted version falls back conservatively to provider defaults: no guessed
reasoning flag is sent, and child-agent execution stays off when the stream
cannot safely identify child text. Custom CLI executables are not probed.

Together, the profiles identify:

- transport: CLI process, local chat endpoint, or remote compatible endpoint;
- stream dialect: structured provider events or plain text;
- declared plan source: provider-native plan events or Adam-owned task tools;
  plan-capable runs expose exactly one of those channels, while adapters
  without a callable or trustworthy plan transport expose neither;
- continuity: native provider session or bounded transcript replay;
- system-instruction channel: a native flag/config/API system message or a
  fenced block inside the prompt;
- tools-off behavior; and
- native sandbox support when the provider exposes a safe supported control.

| Provider | Transport | Activity | Continuity | Intended use |
| --- | --- | --- | --- | --- |
| Claude CLI | Local agent CLI | Structured stream | Native session when an ID is returned; replay otherwise | Chat, Cowork, Code |
| Codex CLI | Local agent CLI | JSON lines | Native `exec resume`; replay otherwise | Chat, Cowork, Code |
| Grok CLI | Local agent CLI | Structured JSON | Native resume when an ID is returned; replay otherwise | Chat, Cowork, Code |
| Kimi CLI | Local agent CLI | Structured stream when supported | Bounded replay | Cowork/Code in the CLI’s supported automatic tool posture |
| LM Studio | Local OpenAI-compatible HTTP endpoint | Streaming compatible events | Bounded replay | Private local chat and analysis |
| Ollama | Local model CLI | Plain text | Bounded replay | Private local chat and analysis |
| OpenAI-compatible | HTTP endpoint | Streaming compatible events | Bounded replay | Hosted or self-hosted chat models |
| Custom CLI | Direct executable, no shell | Plain text | Bounded replay | Advanced local integrations with explicit arguments |

Automatic selection resolves the first available supported provider when a
turn starts. A queued turn captures its provider and complete non-secret
provider profile at enqueue time—model, reasoning effort, fallback, turn
limit, and explicit ability choices—so later settings changes cannot silently
switch its provider or retune already submitted work.
Desktop-launch discovery checks Adam’s process `PATH`, `~/.local/bin`,
`~/.codex/bin`, `~/.grok/bin`, `~/.lmstudio/bin`, `/opt/homebrew/bin`, and
`/usr/local/bin`. A CLI installed only through shell startup tooling can be
entered by absolute path as a Custom CLI; Adam deliberately does not execute a
login shell just to discover commands.

## Detection preflight (Agents panel)

The Agents panel (quick-bar `◎`, module `src/agents_panel.rs`) shows every
provider with a live availability status so a missing CLI is visible before
Send instead of after it fails. Statuses:

- **Not detected** — the binary was not found on the discovery paths above.
- **Detected vX.Y.Z** — found, but no captured runtime contract covers this
  version; provider defaults apply and no tuning controls are exposed.
- **Detected vX.Y.Z · verified** — found and the version matches a captured
  contract row in `runtime_tuning_profile`.

Detection reuses the launch path's own resolver and version cache through the
additive accessor `ai::probe_installed_provider` — what the panel shows and
what a turn runs cannot drift. Scans run on a background worker
(`adam-agents-scan`); the panel's Refresh bypasses the version cache so an
upgraded binary re-probes. The chat composer surfaces a pre-Send banner from
the same cached snapshot (LM Studio is exempt while an endpoint is
configured, and Automatic only warns when all four CLI candidates are
missing). The panel is read-only detection plus install guidance: install
commands are compiled-in copy-only strings from `AGENT_PROVIDERS`, never
executed, and never sourced from user-editable data. Sign-in status is a
planned second axis; one-click install is planned behind the same
compiled-in allowlist.

## Models, reasoning, and abilities

Provider controls live beside the composer because they describe the next
turn, not the contents of the inspector. Choices are stored per provider.
Switching from Codex to Claude and back restores each provider’s last model,
effort, and abilities instead of sharing one global model field.

An empty model or effort means **use the provider or model default**. Adam does
not silently translate that into Medium or another guessed default.

The rows below name the captured runtime contracts. Other installed versions
keep the provider-default reasoning setting until a fixture verifies their
accepted values.

| Provider | Model control | Reasoning control | Additional abilities |
| --- | --- | --- | --- |
| Codex 0.144.1 | Known GPT-5.6 Sol, Terra, and Luna choices plus custom model ID | Sol/Terra: Low through Ultra; Luna: Low through Max; other models: Low through XHigh | Explicit web-search enable |
| Claude Code 2.1.128 | Provider default, Opus, Sonnet, Haiku, or custom model ID | Low, Medium, High, XHigh, Max | Web tools on/off and optional fallback model |
| Grok 0.2.111 / 0.2.114 | Provider default, Grok 4.5, or custom model ID | Low, Medium, High | Web search, planning, memory, and a 1–100 turn limit; subagents forced off |
| Kimi 1.49.0 | Provider default or custom model ID | Provider default | Thinking on/off |
| Ollama 0.32.1 | Required local model ID | Low, Medium, High, or thinking on/off | Local model execution |
| LM Studio | Required loaded model ID | Provider default | Local server endpoint and memory-only key |
| OpenAI-compatible | Required endpoint model ID | Provider default | Endpoint, key environment variable, and memory-only key |
| Custom CLI | Optional custom model ID | Provider default; custom reasoning values are unverified | Direct whole-argument template |

The visible choices map to real provider controls:

- The captured Codex runtime receives `--model`, a TOML-escaped
  `model_reasoning_effort`, and `--search` only when explicitly enabled.
- The captured Claude runtime receives `--model`, `--effort`,
  `--fallback-model`, and explicit WebSearch/WebFetch allow or deny filters.
  Adam does not emit a Claude `--max-turns` flag because the supported
  installed CLI does not expose one.
- Captured Grok 0.2.111 and 0.2.114 runtimes receive
  `--reasoning-effort` only for Low, Medium, or High, plus supported
  ability-disable flags, experimental memory enable when requested,
  `--max-turns`, and an exact read-only or workspace sandbox. Grok 0.2.114
  uses its structured Agent Client Protocol transport so Adam can attach its
  authenticated task-tool server and answer typed permission requests. That
  ACP path disables Grok's native planner even when an older saved preference
  enabled it, preserving one task authority for the run. Forced web denial is
  reported as permission-blocked with the exact WebSearch or WebFetch retry,
  and the per-run bridge credential is redacted across structured fields,
  object keys, and arbitrary streamed-text chunk boundaries before activity
  or transcript persistence.
  Earlier captured versions use the structured CLI stream plus a live
  session-file follower. Neither verified channel safely scopes child prose,
  so Adam always sends `--no-subagents`. Unless web is explicitly switched
  off, read-only Chat research grants only `WebSearch` and `WebFetch`; it
  never turns that into blanket filesystem-mutation approval.
- Kimi receives `--thinking` or `--no-thinking`.
- The captured Ollama runtime receives its supported `--think` level or
  boolean.
- Custom CLI may use `{prompt}`, `{model}`, `{reasoning_effort}`, and
  `{workspace}` as whole safe placeholders. The reasoning placeholder
  currently expands to an empty provider-default value because Adam has no
  verified contract for arbitrary executables. Unknown saved feature keys
  never become arbitrary command-line arguments.

When an installed version does not match a captured row, the composer omits
unverified reasoning choices and launch shaping omits the corresponding flag.
Unsupported saved values are cleared back to provider default rather than
being forwarded optimistically.

Generic OpenAI-compatible HTTP bodies intentionally remain limited to
`model`, `messages`, `stream`, and ordinary OpenAI function descriptors when
Adam's task tools are available. Adam does not assume that every compatible
server accepts OpenAI-, Anthropic-, or xAI-specific reasoning extensions.

## Structured activity

Provider output is normalized into one ordered, typed activity vocabulary:

- assistant text and thinking;
- tool call and tool result;
- command lifecycle;
- file changes;
- web search;
- complete plan snapshots;
- task and host mutations;
- host reads;
- permission prompts;
- child-agent lifecycle updates, including parent identity, status, model,
  tool count, detail, and elapsed time;
- explicit terminal outcomes: completed, user-cancelled, permission-blocked,
  timed out, maximum turns reached, or provider error;
- usage;
- turn errors; and
- model/session information.

Lifecycle updates keep stable identities and durations. Consecutive text and
thinking deltas merge. A newer native plan snapshot replaces native rows,
recovers stable IDs by exact content when necessary, and reappends task-tool
rows without disturbing the rest of the trace. Task mutations remain visible
activity provenance during ordinary runs; only cap pressure folds older state
into one equivalent durable snapshot while retaining a bounded newest
mutation tail. Errors, permission prompts, file changes, host mutations,
response text, and the newest task state are retained. If the must-keep set
alone exceeds the nominal cap, Adam keeps authoritative state instead of
deleting the plan.

Accumulator-generated compaction snapshots are explicitly marked as
compaction, not provider-native replacement. A mutation-only compacted turn
therefore cannot clear saved native tasks during resume. Unresolved ID-only
updates stay as mutations until the persisted task list can supply their
identity.

The same activity log drives every projection:

- response transcript;
- progress inspector;
- inline and right-rail subagent views;
- collapsible activity rows;
- output files and artifacts;
- context-used list;
- token/cost usage; and
- session diagnostics.

Raw provider JSON is never used as the persisted UI model.

## Progress is the agent’s task list

Progress is not a feed of log messages. It is the model’s structured task
checklist: the work items the agent creates, starts, completes, cancels, or
replaces while carrying out the request.

From structured provider output, Adam accepts two task-event shapes:

1. a whole-plan snapshot, where the new native list replaces prior native
   rows while task-tool rows retain their own lifecycle; and
2. task mutations, where create appends a task and update patches the matching
   stable task ID, then exact content as a compatibility fallback.

Codex `todo_list` events become whole-plan snapshots. Claude `TodoWrite`
becomes a snapshot, while Claude `TaskCreate` and `TaskUpdate` become
incremental mutations with their status and active-form label retained. The
projection is intentionally provider-neutral after parsing.

For providers without a trustworthy native plan stream, Adam owns a small
conversation-scoped task store. Tool availability remains run-scoped, and a
capable run exposes exactly three model-callable tools:

- `task_create` requires `content`, accepts an optional present-continuous
  `activeForm`, and returns a stable task ID;
- `task_update` requires `task_id`, may patch status, content, or
  `activeForm`, and creates a row with that exact ID when it does not yet
  exist; and
- `task_list` returns the current ordered list without emitting activity.

The task-tool wire statuses are `pending`, `in_progress`, `completed`, and
`cancelled`. Each successful mutation emits a visible `TaskMutation` followed
by a `PlanUpdate` containing the entire reduced list. Fields are trimmed,
empty required fields and unknown arguments are rejected, and user-facing
text is capped at 512 UTF-8 bytes.
Adam task tools can create at most 512 rows; updates to existing rows remain
available at that limit, while attempts to create another row fail without
emitting activity. A larger provider-native or persisted checklist is retained
in full and remains listable, but task-tool mutations fail closed until a
native snapshot reduces it to the supported mutation limit.

Tool exposure is checked both when a provider lists tools and when it calls
one. Dead runs fail closed. Claude and Codex keep their native task channel and
never receive Adam’s task tools. Verified Grok ACP, compatible HTTP function
calling, and the explicit Custom CLI bridge contract receive Adam’s tools and
do not project a second native plan channel. The tools are independent of
canvas access: a model can maintain Progress without gaining permission to
read or mutate canvas data.

HTTP providers receive ordinary function-tool descriptors and may perform up
to 16 bounded continuation rounds. A streamed or non-streamed function call is
assembled before dispatch, and the assistant tool call plus tool result are
returned to the model for its next response. Both current `tool_calls` arrays
and the legacy single `function_call` shape are normalized. The complete
serialized continuation history has a 32 MiB cumulative request budget,
checked before allocation and send, so repeated large tool results cannot
grow an unbounded request. Because generic compatible endpoints have no
dependable capability handshake, a 400, 404, or 422 response to the first
tool-bearing request is retried once without tools; ordinary chat therefore
remains usable on models without function calling. A Custom CLI wrapper
receives the ephemeral `ADAM_TASK_MCP_URL` and
`ADAM_TASK_MCP_AUTHORIZATION` environment variables. The endpoint is a
loopback-only MCP server with a per-run bearer token. Adam never puts that
token in the prompt or saved configuration, and it expires when the run ends.
The bridge implements the MCP `2025-06-18` Streamable HTTP contract only:
initialization negotiates clients onto that version, and every later request
must carry the matching `MCP-Protocol-Version` header.
As with any arbitrary executable, a hostile Custom CLI could still echo its
own environment into provider output, so Custom CLI wrappers remain trusted
local integrations.

Captured legacy Grok streams also follow the matching session
`updates.jsonl` while the process is running. Complete new lines are reduced
into typed task, tool, and session activity without waiting for process exit;
partial trailing lines wait for the next poll. Reads enforce the line limit
before allocation, and an oversized unterminated record disables the follower
without moving its cursor into provider-controlled payload. New user-message
markers no longer discard the prior plan. Resumed decoding is seeded from the
persisted native checklist before the bounded tail scan, so merge-only updates
retain tasks whose original full snapshot predates that scan window.

The inspector applies one strict precedence rule:

1. any explicit live snapshot or effective live task mutation from the active
   turn;
2. otherwise, the newest persisted task state;
3. otherwise, no checklist.

An empty native snapshot deliberately clears native rows, while independently
created task-tool rows survive until updated, cancelled, or replaced by a
newer authoritative whole-list snapshot. A
subjectless update for an unknown task ID is a true no-op: it cannot create an
empty live plan or hide the saved checklist. On native resume, ID-only live
updates fold onto saved task IDs so the real task keeps its name and status.
Rejected provider updates are not committed optimistically.

It never invents checklist rows from “connected,” “searching,” command output,
file edits, elapsed time, or generic status strings. When an active provider
has not published a task list, Progress says so and shows only a spinner,
elapsed time, and the best honest current-work label. Tool calls, commands,
searches, thinking, and file activity remain available under a separate
collapsible Activity disclosure.

Task rows preserve provider order and one of four UI states:

- Pending: open circle;
- In progress: accent dot and the provider’s present-continuous `activeForm`
  when supplied;
- Completed: checked and struck through; or
- Cancelled/deleted: struck through without being counted as completed.

This distinction matters for providers such as Grok or plain local models that
may stream useful work but no structured task list. Adam shows their real
activity without manufacturing Claude-like progress.

The newest full task snapshot is committed with the assistant turn, including
an explicitly empty list. A later turn seeds its live store from that snapshot,
and relaunch restores the same order, labels, origins, and statuses. A turn
that did not publish or mutate a task list does not manufacture an empty
snapshot that would erase earlier progress.

## Subagents are a separate lifecycle projection

Subagents are not checklist rows and are not inferred from prose such as “I’ll
launch five agents.” Adam creates them only from genuine provider child-agent
events. Each stable child ID is projected through pending, working, completed,
failed, cancelled, or permission-blocked states. Parent IDs preserve a nested
tree when a provider launches children from another child.

The chat shows compact live chips beside the relevant activity update, with
working, done, and stopped totals. The right rail expands the same events into
individual rows with status, model, tool-call count, duration, detail, and
parent/child indentation. **View all** opens a wider Active/Done subagent
panel without leaving the conversation. Repeated status updates replace the
earlier state for that child rather than creating duplicate “agent” rows.

New runs on the captured Grok versions are deliberately excluded from this
projection. Their available streams do not provide a proven, end-to-end scoped
child-prose channel, so Adam forces subagents off instead of leaking child
responses into the parent transcript or inventing ownership. Scoped Grok child
lifecycle and prose events are PR3 work. Codex collaboration calls and
subagent-activity items join thread aliases into the same lifecycle. Claude
Agent/legacy Task calls, task lifecycle notifications, and tool-progress
events join tool-use, task, and provider-agent IDs. Unsupported providers
simply show no Subagents section instead of an invented one.

## Inspector, artifacts, files, and context

The right side is a task workspace with six independent, collapsible
sections:

- **Progress** — the authoritative task projection described above;
- **Agents** — real child-agent lifecycle, counts, and parent/child tree;
- **Artifacts** — files and host artifacts actually created by the conversation;
- **Activity** — tools, commands, searches, and other execution diagnostics;
- **Working folder** — an expandable file tree for the scoped folder; and
- **Context** — supplied attachments, observed tool/search/host context,
  session identity, replay pressure, and usage.

Artifacts are provenance-based. Successful file changes deduplicate by path, and
a later delete strikes the earlier output. A host create may introduce an
output; a host update or delete can revise only an artifact that this trace
already created. Updating pre-existing host data does not falsely claim it as
a newly produced deliverable.

The first eight artifacts are shown by default, with an explicit Show all
control. Text and Markdown outputs open in a wider in-app File view with path,
size, selectable content, Reveal, missing-file feedback, and a 256 KiB preview
bound. Unsupported binary files remain revealable without being decoded as
text. Provider-reported paths are canonicalized and must remain inside the
chosen working folder; only files the user explicitly attached may be previewed
from outside that scope.

The working-folder browser sorts directories first, expands lazily to four
visible levels, previews files on selection, and can reveal either files or
folders in the system file browser. Folder changes are locked during an active
turn so the running agent’s scope cannot drift.

Context deduplicates supplied files by path and shows provider-reported tool,
command, web-search, and host-read use counts. Usage comes from the same typed
event stream and keeps input, cached-input, reasoning, output, and cost fields
separate. If older conversation turns are omitted from bounded replay, the
replay meter says exactly how many.

## Prompt continuity

Adam has two continuity paths.

### Safe native resume

Claude, Codex, and Grok can resume a provider-owned session only when every
gate still matches:

- conversation;
- provider;
- executable basename;
- canonical working folder;
- parser dialect;
- native sandbox identity; and
- last committed conversation sequence.

The provider session ID is machine-local state, not portable workspace data.
It is saved atomically with a validated previous generation. A failed,
cancelled, or ID-less turn forgets the record, and a settings or permission
change invalidates it. Adam never falls back to a provider’s vague “continue
most recent” behavior.

If a native resume fails before producing any text, thinking, tool, command,
file, plan, or other substantive activity, Adam forgets the stale ID and makes
one bounded-replay attempt without duplicating the committed user message.
Once any substantive activity has appeared—even before a poisoned stream reset—
Adam will not replay automatically, avoiding duplicated side effects.

### Bounded replay

Providers without safe native resume receive a contiguous suffix of at most 40
persisted messages and 60,000 characters. The newest message is retained even
when it is unusually large. The inspector shows replay pressure and the number
of older turns omitted.

Standing instructions and a character/persona have independent byte budgets.
On replay, system instructions use a provider-native system channel when one
exists; otherwise Adam inserts an explicitly fenced app-owned block. On native
resume, stable context is not duplicated, while changing working context and
the new message are still sent.

Attachments are explicit. Adam extracts at most 64 KiB of text per attachment
and 256 KiB per turn, rejects obvious binary text extraction, and still
provides the file path for a capable workspace agent.

## Queue and lifecycle

- Up to four provider runs may be active globally.
- Only one run may be active per conversation.
- Additional messages enter a durable FIFO queue, capped at 50 per
  conversation.
- A queued turn carries its text, attachments, provider, legacy-readable model,
  and the full provider preference snapshot.
- Successful and failed terminal turns advance the queue.
- Stop parks the queue instead of unexpectedly launching the next item.
- For HTTP providers, Stop reports immediately but retains the global run slot
  until the blocking network worker exits. This prevents duplicate terminal
  events and repeated cancellation from accumulating hidden provider requests.
- Relaunch also parks recovered queues; the user explicitly resumes them.
- If Adam closed during a user turn, recovery adds a visible interrupted-turn
  marker rather than pretending the provider completed.

Every started run commits one explicit terminal outcome and a visible
assistant turn: completed, user-cancelled, permission-blocked, timed out,
maximum turns reached, or provider error. Failures and stops keep partial
output and typed diagnostic activity. Checklist completion is independent of
the run outcome and cannot turn a failed run into a successful one. The
transcript and inspector name the real terminal state instead of collapsing
permission blocks, timeouts, turn limits, provider errors, and user
cancellation into “Stopped: Cancelled.” **Allow web for this run** is offered
only when the blocked permission explicitly identifies WebSearch or WebFetch;
other permission blocks remain blocked or offer a non-privilege-widening
retry. The web action retries the last request with a narrow one-run grant.
Background completions mark the conversation unread.

The transcript follows a compact work chronology: a Working/Worked/Blocked
header with elapsed time, coalesced activity rows, live subagent chips,
provider commentary, the final response, and the genuine checklist step
indicator near the composer. Consecutive reasoning updates collapse to the
latest useful line instead of rendering dozens of identical “Thinking…”
entries.

## Access stances

Adam uses one five-stance vocabulary:

| Stance | Read | Mutate | Destructive |
| --- | --- | --- | --- |
| Sandbox | Allow | Ask | Ask |
| Ask | Allow | Ask | Ask |
| Plan | Allow | Deny | Deny |
| Auto | Allow | Allow | Ask |
| Bypass | Allow | Allow | Allow |

Permanent deletion remains blocked by the existing Adam domain guard. Adam
does not synthesize undocumented or dangerous provider bypass flags.

Native CLI filesystem posture and Adam host-data permission are separate:

- native sandbox controls are fixed at process spawn when safely supported;
- Adam host permission is re-evaluated for every host call, so changing the
  stance mid-run takes effect immediately.

The provider-neutral host gate already supplies canonical call fingerprints,
deduplicated five-minute prompts, Allow Once, memory-only Always for this
conversation, Deny, destructive-Always refusal, stance re-evaluation,
single-flight execution claims, and run teardown. It fails closed on malformed
or mismatched input.

Direct model-callable Adam canvas tools still require the local loopback bridge
that will feed calls into that gate. Until then, the Canvas inspector actions
are visible user-invoked operations and should not be described as autonomous
model tools.

Adam’s task-tool bridge is deliberately separate from that future canvas
bridge. It is a loopback-only, bearer-authenticated, per-run MCP endpoint. Its
tool list and every call are re-authorized against the active run and the
run’s single declared plan channel; possessing a stale endpoint or token
cannot revive a completed run.

## Data boundaries

- The workspace stores conversations, messages, attachments, typed activity,
  queue state, settings, and checkpoints.
- Provider session IDs and future compaction summaries live in versioned
  machine-local sidecars.
- Temporary API keys remain in memory.
- CLI providers reuse their existing local login.
- Commands are spawned directly; Custom CLI input is not evaluated by a shell.
- The working directory is explicit for Cowork and Code.
- Future or malformed AI enum values fail closed without blanking the whole
  library: unknown access becomes Ask, unknown surfaces become Chat, unknown
  roles become Assistant, and unrecognized typed activity entries are skipped.
- Conversation saves use a process-local mutex and a cross-process file lock
  around a revision-aware three-way merge. Distinct conversations and
  non-conflicting edits survive stale writers. Before atomic replacement Adam
  rotates a validated `library.previous.json`; unreadable files are preserved
  as timestamped recovery copies instead of being silently overwritten.

## Deliberate limits

The current harness does not claim:

- a completed model-callable Adam host-tool server;
- scoped child prose for identity-less Grok streams, which is PR3 work;
- native Adam task-tool attachment for Kimi or Ollama CLI builds that expose
  no verified external-tool transport;
- an interactive approval sheet for every provider permission request—safe
  read-only web requests can be granted narrowly, while a request Adam cannot
  safely answer terminates with a typed permission-blocked outcome;
- scheduled/background dispatch;
- automatic long-history summary generation;
- a finished Character editor and memory UI; or
- that chat-only LM Studio, Ollama, or compatible endpoints can autonomously
  edit files.

Those are separable harness extensions. They are not hidden behind UI labels or
approximated with unsafe provider flags.
