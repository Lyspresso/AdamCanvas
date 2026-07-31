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
when that version has a captured, tested contract. A successfully observed but
unlisted Claude, Codex, LM Studio, or Ollama version falls back conservatively
to provider defaults: no guessed reasoning flag is sent, and child-agent
execution stays off when the stream cannot safely identify child text. Grok
and Kimi are stricter because version selects both transport and permission
semantics: only fixture-verified versions launch. A built-in CLI version probe
has a five-second process-execution deadline plus at most two seconds of
bounded output-pipe cleanup. Detection runs off the UI
thread, caches only successful observations, and shares one result among
overlapping callers. Saved controls are preserved, and launch fails visibly
instead of guessing between structured and legacy adapters. A resumed turn
that fails before provider launch retains its native session only for an exact
same-process Retry of that locally unsent message; an app restart, changed
history, or different prompt falls back to bounded replay. Custom CLI
executables are not probed.

Claude, Codex, and Ollama also re-probe on the run worker whenever the user has
saved a version-sensitive reasoning control. The worker rebuilds the launch
arguments from that fresh verified contract. If detection is unavailable or
the installed version is unverified, the turn fails visibly before provider
launch rather than silently dropping the saved control. Runs that request only
provider defaults do not pay this extra probe cost.

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
| Grok Build CLI | Local agent CLI | Structured ACP events | Native ACP session load when an ID is returned; replay otherwise | Chat, Cowork, Code with scoped CLI subagents and workflows |
| Kimi Code CLI 0.31.0 | Local agent CLI | Structured ACP events | Native ACP session load | Chat, Cowork, Code with root plans plus Agent/AgentSwarm delegation |
| xAI Grok Heavy | Remote xAI Responses API | Leader-visible Responses events plus one aggregate group | `previous_response_id` | Multi-agent research with 4 or 16 opaque server agents |
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

The Agent Harness section (sidebar entry under AI chats, module
`src/agents_panel.rs`) shows every provider with a live availability status
so a missing CLI is visible before Send instead of after it fails. Statuses:

- **Not detected** — the binary was not found on the discovery paths above.
- **Detected vX.Y.Z** — found, but no captured runtime contract covers this
  version; provider defaults apply and no tuning controls are exposed. When
  a contract exists for a different version, the hover names it so drift is
  visible instead of silent.
- **Detected vX.Y.Z · tested series (vA.B.C)** — for providers whose transport
  does not change by patch version, found in the same
  major.minor series as a tested contract version with only a newer patch
  (the usual self-update drift). Safe defaults still apply; full "verified"
  returns with the next contract capture. Never granted on downgrades or a
  different series. Grok and Kimi never receive this badge: their exact
  version selects transport and permission behavior.
- **Detected vX.Y.Z · verified** — found and the version matches a captured
  contract row in `runtime_tuning_profile`.

Drift captures under `tests/fixtures/ai/<provider>/<version>/` back the
generic-provider series policy and the exact Grok/Kimi rows. Grok 0.2.111,
0.2.114, and 0.2.117 remain separate fixture-verified contracts even where
their captured grammar overlaps, because the selected transport and
permission semantics differ. Grok 0.2.117 has a separate redacted parent-child ACP
capture: the parent lifecycle notification names the child session before
that session emits its own prose. Lifecycle and prose carry independent
event IDs; the capture also retains Grok's idless, status-only
`model_changed` child update.

Kimi 0.31.0 and xAI Grok Heavy fixtures make a stricter provenance
distinction. Kimi’s `initialize` exchange was captured locally without a
model prompt. Its config, root-plan, `Agent`, and `AgentSwarm` fixtures are
source-derived from the exact
[`@moonshot-ai/kimi-code@0.31.0` tag](https://github.com/MoonshotAI/kimi-code/tree/bc28e9d802fbec29395a7aed85e880679a050145),
because the local account was signed out and no authenticated run was made.
The xAI request, response, and SSE fixtures are likewise marked
official-schema-derived, not live captures. They follow xAI’s
[`grok-4.20-multi-agent` contract](https://docs.x.ai/developers/model-capabilities/text/multi-agent)
and the official [Responses API schema](https://docs.x.ai/developers/rest-api-reference/inference/chat).
Its billing fixture follows xAI’s documented
[`usage.cost_in_usd_ticks` contract](https://docs.x.ai/developers/cost-tracking).
The manifests retain that distinction so a synthetic fixture can never be
mistaken for observed provider behavior.

Detection reuses the launch path's own resolver and version cache through the
additive accessor `ai::probe_installed_provider` — what the panel shows and
what a turn runs cannot drift. Scans run on a background worker
(`adam-agents-scan`); the panel's Refresh bypasses the version cache so an
upgraded binary re-probes. The chat composer surfaces a pre-Send banner from
the same cached snapshot (LM Studio is exempt while an endpoint is
configured, and Automatic only warns when all four CLI candidates are
missing). When every probed CLI is missing, the chat's empty state becomes
a setup screen with the same rows.

Install buttons execute **only** commands compiled into `AGENT_PROVIDERS`
— vendor-official installers verified against vendor domains, never
sourced from user-editable data (providers without a safe non-interactive
installer get a button that opens the official download page instead).
Execution is user-initiated, one at a time, via `/bin/zsh -lc` with
`set -o pipefail`, output captured with both pipes drained on their own
threads, a five-minute deadline with kill, and a post-install
cache-bypassing re-probe: if the installer finishes but the binary still
does not resolve on the discovery paths, the panel says so honestly and
keeps the full command log.

Sign-in is the second, orthogonal status axis: for Claude (`claude auth
status`) and Codex (`codex login status`) the scan runs the vendor's own
status command against the resolved binary — bounded exactly like installs
(own process group, five-second deadline group-kill, bounded drain) — and
caches the result per resolved path so ordinary rescans stay cheap
(Refresh and the post-install rescan re-probe). Classification: signed-out
markers dominate exit codes, and a command that ran but errored without a
marker is "Sign-in unknown", never "Signed out" — the CLI must state the
auth fact itself. Rows show "Signed in" / "Signed out" (with a copyable
vendor sign-in command) or "Sign-in unknown"; providers without a vendor
status command show nothing — their sign-in happens at launch.

## Models, reasoning, and abilities

Provider controls live beside the composer because they describe the next
turn, not the contents of the inspector. Choices are stored per provider.
Switching from Codex to Claude and back restores each provider’s last model,
effort, and abilities instead of sharing one global model field.

An empty model or effort means **use the provider or model default**. Adam does
not silently translate that into Medium or another guessed default.
The dedicated xAI multi-agent adapter is the one explicit exception: its
composer labels the empty choice “Default · 4 agents” and sends Medium, because
the selected effort is the public REST control that determines whether xAI
allocates 4 or 16 agents.

The rows below name the captured runtime contracts. Other installed versions
keep the provider-default reasoning setting until a fixture verifies their
accepted values.

| Provider | Model control | Reasoning control | Additional abilities |
| --- | --- | --- | --- |
| Codex 0.144.1 | Known GPT-5.6 Sol, Terra, and Luna choices plus custom model ID | Sol/Terra: Low through Ultra; Luna: Low through Max; other models: Low through XHigh | Explicit web-search enable |
| Claude Code 2.1.128 | Provider default, Opus, Sonnet, Haiku, or custom model ID | Low, Medium, High, XHigh, Max | Web tools on/off and optional fallback model |
| Grok 0.2.111 / 0.2.114 | Provider default, Grok 4.5, or custom model ID | Low, Medium, High | Web search, planning, memory, and a 1–100 turn limit; subagents forced off |
| Grok Build 0.2.117 | Provider default, Grok 4.5, or custom model ID | Low, Medium, High | The same controls plus scoped ACP subagents and provider workflow grouping |
| Kimi Code 0.31.0 | Optional model ID, accepted only when the live ACP session advertises it | On/off preference resolved to a value advertised by the selected model | ACP mode plus explicit foreground AgentSwarm preference |
| Kimi CLI 1.49.0 (legacy capture) | Provider default or custom model ID | Provider default | Thinking on/off on the legacy stream contract |
| xAI `grok-4.20-multi-agent` | Fixed multi-agent model | Low/Medium: 4 agents; High/XHigh: 16 agents | Optional hosted web search and `previous_response_id` continuity |
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
- Captured Grok 0.2.111, 0.2.114, and 0.2.117 runtimes receive
  `--reasoning-effort` only for Low, Medium, or High, plus supported
  ability-disable flags, experimental memory enable when requested,
  `--max-turns`, and an exact read-only or workspace sandbox. Grok 0.2.114
  and 0.2.117 use the structured Agent Client Protocol transport and typed
  permission requests. A fresh 0.2.114 run attaches Adam's authenticated task
  tools, disables Grok's native planner, and forces subagents off. Every
  0.2.117 run instead attaches no Adam MCP server at all because Grok documents
  that children inherit connected parent MCP servers by default and Adam's
  resume record does not preserve a session's original child capability.
  Subagents Off still sends `--no-subagents`; it does not change this tool
  boundary. A resumed 0.2.114 session is conservatively treated the same way
  so an unrecorded 0.2.117-to-0.2.114 version transition cannot attach task
  tools to an older child-capable session. These no-server runs use the root
  session's native plan as Main Progress; child plans remain child-scoped.
  Their active-run registry is also marked NativeStream, so retained or stale
  task-server credentials list no tools and reject calls.
  Prompt shaping and preparation read the latest successful background
  observation, so a slow `--version` process never blocks the UI thread. The
  worker freshly re-probes at the process boundary. If the binary changes in
  place after preparation, the queued run fails closed and asks for a retry
  instead of reusing the older version's subagent or task-server contract.

  The 0.2.117 fixture proves the parent lifecycle registration and
  independently identified parent/child prose with event IDs. The normalizer
  also scopes thought, tool, plan, and permission events after a child session
  is registered; unknown sessions are never promoted to Main. Scoped child
  prose/thought/tool/plan and spawn/finish activity without event IDs fails
  closed. Provider status-only updates such as `model_changed` and coalesced
  subagent progress may be ID-less; bounded pre-registration activity is never
  silently evicted. An installed-runtime permission test observed that a child
  tool's permission request can use a root-session envelope. Adam therefore
  resolves the unique session that already owns the tool-call ID before
  applying policy or projecting activity; ambiguous cross-session IDs fail
  closed. A known child remains the owner after its lifecycle closes, while an
  unowned request in a root envelope is answered Cancelled without delegation
  or projection. Grok 0.2.111 and 0.2.114 still receive `--no-subagents`, and
  any lifecycle event received while subagents are disabled is a protocol
  error.

  Forced web denial is reported as permission-blocked with the exact WebSearch
  or WebFetch retry. When the root-only task bridge is present, its per-run
  credential is redacted across structured fields, object keys, and arbitrary
  streamed-text chunk boundaries before activity or transcript persistence.
  Unless web is explicitly switched off, read-only Chat research grants only
  `WebSearch` and `WebFetch`; it never turns that into blanket
  filesystem-mutation approval.
  Grok Build workflows remain part of this CLI contract: a provider
  `workflow_run_id` groups the real scoped child sessions that belong to the
  workflow. It does not turn the workflow into “Grok Heavy,” and it does not
  infer an agent count from reasoning effort.
- Verified Kimi 0.31.0 launches as `kimi acp`. Adam reads the session’s
  advertised `model`, `thinking`, and `mode` config options and sets only
  values the live session offered. “Thinking on” is a semantic preference:
  boolean-thinking models receive their advertised On value, while an
  effort-based model receives a supported non-Off choice rather than a
  hard-coded spelling. Its root `TodoList` display becomes the native
  whole-list plan. The AgentSwarm ability asks the root model to use its real
  foreground `AgentSwarm` tool when the work can be decomposed; Adam does not
  treat that preference as proof that a swarm launched. The root tool’s
  `rawInput` is the first genuine delegation signal and its final `rawOutput`
  supplies stable child IDs and outcomes. Adam sends `mcpServers: []`: Kimi
  children inherit session MCP clients while the ACP adapter filters child
  caller identity, so attaching Adam task tools would break the
  native-XOR-tools safety boundary. Kimi’s root ACP permission requests still
  pass through Adam’s normal stance mapping and fail closed when the request
  cannot be answered safely. Kimi 0.31 also bridges `AskUserQuestion` through
  that same ACP method with a distinct `q0_opt_*` / `q0_skip` option contract.
  Until Adam has an interactive question surface, it recognizes only that exact
  shape and selects Kimi’s explicit Skip response in every permission stance;
  it never fabricates the first answer in Bypass or marks the turn permission-
  blocked solely for dismissing the unsupported question. Background `Agent`
  jobs are refused: Adam’s
  per-turn ACP host cannot truthfully keep such a job alive or receive its
  later notification after the root turn exits. Kimi resume gating and
  preparation use the latest successful background observation; the worker
  freshly re-probes at the process boundary. A transient boundary failure is
  surfaced without selecting legacy mode, and an exact same-process Retry may
  reuse the untouched ACP session. Verified incompatibility uses one bounded,
  explicitly session-free replay with full conversation context. The ACP ID is
  never passed into legacy print mode. Only the exact detected legacy 1.49.0
  runtime receives the visible Cowork/Code
  plus Automatic-access warning; unknown and unverified 0.x builds do not.
- xAI Grok Heavy is a dedicated Responses API adapter, not the generic
  OpenAI-compatible path and not a Grok Build mode. It fixes the model to
  `grok-4.20-multi-agent`, sets `reasoning.effort`, and maps Low or Medium to
  4 server agents and High or XHigh to 16. Adam sends the hosted
  `web_search` tool only when explicitly enabled, sends no client function
  tools, and reads its key from the temporary setting or `XAI_API_KEY`.
  Requests set `store: true` so `previous_response_id` can continue follow-up
  turns. That setting stores the messages sent to xAI and Grok Heavy’s
  responses. Adam discloses the server storage beside the composer and in the
  provider configuration; xAI documents 30-day retention by default.
  If xAI nevertheless reports an unrequested or unknown server-side hosted
  call, Adam quarantines that item without executing or projecting it as a
  local tool, preserves the leader response, and attaches one bounded provider
  notice to the completed group. An unsolicited client `function_call` or
  `custom_tool_call` still fails closed because Adam exposed no such executor.
  The API exposes leader tool calls and the leader’s answer, so the only
  honest lifecycle is one aggregate 4- or 16-agent group. Stop wins a
  serialized terminal race, cancels Adam’s HTTP transport, closes the live
  connection, and joins the bounded worker before releasing the run slot.
  xAI documents no synchronous Responses cancellation endpoint, so Adam does
  not claim a server-side cancel: connection teardown is the truthful client
  boundary. Late provider output cannot outlive that joined worker.
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
server accepts OpenAI-, Anthropic-, or xAI-specific reasoning extensions. The
xAI Grok Heavy adapter is a separate, fixed-host Responses transport; its
`reasoning.effort` mapping never leaks into LM Studio or another compatible
endpoint.

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
- provider-declared agent-group lifecycle, including stable group identity,
  group kind, expected count, visibility, aggregate state, and any members
  the provider actually revealed;
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
one. Dead runs fail closed. Claude, Codex, and Kimi 0.31.0 keep their native
root task channel and never receive Adam’s task tools. Fresh verified Grok
0.2.114 ACP, compatible HTTP function calling, and the explicit Custom CLI
bridge contract receive Adam’s tools and do not project a second main plan
channel. Grok 0.2.117 and resumed Grok ACP runs receive no Adam task server
and use their native root plan instead. This is deliberate mutual exclusion:
Grok and Kimi children inherit connected MCP clients without a trustworthy
root/child caller identity at the external bridge. xAI Grok Heavy receives
neither channel: its server-side group publishes no structured main plan, and
Adam does not send task functions into an opaque multi-agent run. Its group
status belongs under Agents, never as invented Progress rows. The task tools
are independent of canvas access: a model can maintain Progress without
gaining permission to read or mutate canvas data.

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

## Subagents and agent groups are a separate lifecycle projection

Agents are not checklist rows and are never inferred from prose such as “I’ll
launch five agents.” Adam creates an individual child row only from a genuine
provider child identity. It creates an agent-group row only from a genuine
provider workflow, delegation tool, or multi-agent request. A group’s
`visibility` says whether its members are provider-visible or aggregate-only;
the requested count is never expanded into synthetic child IDs.

Where a provider streams child lifecycle, each stable child ID is projected
through pending, working, completed, failed, cancelled, or permission-blocked
states. Parent IDs preserve a nested tree. The chat shows compact group/child
chips beside the relevant activity update, and the right rail expands the same
state into group summaries and individual rows. **View all** opens the wider
Active/Done panel. Repeated updates replace the earlier state for the same
group or child rather than creating duplicate “agent” rows.

The four multi-agent contracts remain deliberately distinct:

- **Grok Build scoped subagents:** verified Grok 0.2.117 ACP parent/child
  sessions provide stable child scope. The child session ID is Adam’s
  canonical ID; a distinct provider subagent ID is retained as an alias.
  Child prose is collected into one response cell and never enters the main
  transcript or root completion. Duplicate event IDs and resume replay do not
  create duplicate cells. Earlier captured Grok Build versions remain forced
  off because their streams do not provide end-to-end child scope.
- **Grok Build workflows:** a real provider `workflow_run_id` becomes a group
  around the real Grok Build child sessions that report that workflow. The
  group folds child state without replacing the individual scoped rows. This
  is a CLI workflow, not the xAI Grok Heavy model.
- **Kimi Agent and AgentSwarm:** Kimi 0.31.0’s ACP adapter intentionally
  forwards only main-agent SDK events. A root `Agent` or `AgentSwarm`
  `rawInput` therefore creates an honest **delegated** group with its requested
  work, not fake working children. The terminal `rawOutput` supplies the real
  `agent_id`, outcome, item, resume marker, and final result for each revealed
  member; only then can Adam create/update stable child rows and their final
  prose cells. Kimi does not expose live per-child prose or live child
  lifecycle through this ACP version, so Adam does not claim it does.
  Duplicate stable IDs or ambiguous result markup keep the delegation
  aggregate-only. Background Agent output is never rendered as a permanently
  working child.
- **xAI Grok Heavy:** `grok-4.20-multi-agent` runs either 4 or 16 agents on
  xAI’s servers. xAI returns only leader-visible tool calls and the leader’s
  final answer by default; intermediate child state is hidden. Adam therefore
  shows one aggregate group with the expected count and terminal state, with
  no child tree, no per-child prose, and no made-up completed count. Adam keeps
  a bounded registry of visible leader web searches; after it fills, further
  searches continue at xAI but fine-grained Activity projection degrades
  instead of aborting the root response. Every projected open tool row is
  failed on cancellation, limits, malformed streams, or transport errors.

Codex collaboration calls and subagent-activity items join thread aliases into
the same child lifecycle. Claude Agent/legacy Task calls, task lifecycle
notifications, and tool-progress events join tool-use, task, and
provider-agent IDs. Providers without a genuine child or group signal simply
show no Agents section instead of an invented one.

High-volume ACP activity uses a degradation budget, not a kill switch.
Grok’s budget is scoped per provider session so one noisy child cannot consume
every sibling’s detail allowance. Kimi applies the same principle to its root
stream. When presentation detail reaches the budget, Adam stops forwarding
fine-grained thought and intermediate update chatter, coalesces the latest
tool/progress/plan state, and continues reading the provider. Session identity,
permission decisions, final plans and tools, root completion, real child
spawn/finish state, final child prose, and terminal outcomes remain observable.
If either ACP transport then fails, it first flushes already-accepted partial
root text plus the latest root tool and plan snapshots, returns the original
error, and does not invent a successful terminal. Line, protocol-byte, text,
and identity bounds remain hard failures because those protect memory and
protocol integrity rather than presentation density. Grok's cumulative child
projection registry is different: after 256 tracked children it stops
projecting newly discovered children, keeps the root turn alive, and denies
permissions whose ownership can no longer be proven. Kimi's bounded root-tool
registry remains a hard identity and permission boundary.

## Inspector, artifacts, files, and context

The right side is a task workspace with six independent, collapsible
sections:

- **Progress** — the authoritative task projection described above;
- **Agents** — real child lifecycle plus honest provider groups, including
  aggregate-only groups that intentionally have no child tree;
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
separate. For xAI, the exact integer `usage.cost_in_usd_ticks` value is retained
and converted with 10,000,000,000 ticks per USD for the shared cost field. Adam
does not estimate or silently display zero when xAI omits that field; the
inspector says that cost was not reported. If older conversation turns are
omitted from bounded replay, the replay meter says exactly how many.

## Prompt continuity

Adam has provider-native continuity where the provider supplies a safe token,
and bounded replay everywhere else.

### Safe native resume

Claude, Codex, Grok Build, and Kimi 0.31.0 can resume a provider-owned session
only when every gate still matches. Kimi uses ACP `session/load`; its replayed
history notifications are transport recovery, not new child or group events.
The gates are:

- conversation;
- provider;
- executable basename;
- canonical working folder;
- parser dialect;
- native sandbox or permission-mode identity; and
- last committed conversation sequence.

The provider session ID is machine-local state, not portable workspace data.
It is saved atomically with a validated previous generation. A failed,
cancelled, or ID-less turn normally forgets the record, and a settings or
permission change invalidates it. The narrow exception is a version-check
failure or Stop before provider launch: Adam can bridge the untouched record
only to the exact same-process Retry action for that locally unsent user
message. The bridge binds provider, session ID, user text and attachments, and
the local terminal sequence; it expires after restart, history changes, a
different prompt, or a successful launch. Adam never falls back to a
provider’s vague “continue most recent” behavior.

The existing CLI-native adapters can make one bounded fresh replay when a
resumed process fails before any text, thinking, tool, command, file, plan, or
other substantive activity begins. Grok Heavy is stricter because a generic
network or provider failure is not proof that its paid server-side inference
did no work: xAI fresh replay additionally requires the structured error code
`previous_response_not_found` with `param` exactly `previous_response_id`.
Lookalike free text, a timeout, a transport error, or any other provider error
cannot launch a second xAI request. Adam then forgets the stale ID and replays
without duplicating the committed user message. Once substantive activity has
appeared, no provider replays automatically, avoiding duplicated side effects.

xAI Grok Heavy uses the Responses API’s `previous_response_id` rather than a
CLI session. Adam stores the last committed response ID only after a completed
turn and sends it on the next compatible turn. A changed provider, model,
conversation generation, or failed response clears the chain. Adam never
combines `previous_response_id` with a guessed transcript replay, and it does
not interpret the opaque prior response as recoverable child-agent history.
Unlike CLI-native sessions, a response ID does not carry request-level
instructions forward. Adam therefore resends its complete standing safety and
untrusted-data instructions on every Grok Heavy request, including resumed
turns, without replaying transcript history.
An xAI incomplete response is a provider error unless its exact reason is the
known `max_output_tokens` terminal, which maps to Adam’s turn-limit outcome.

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
header with elapsed time, coalesced activity rows, live agent/group chips,
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

Grok Build and Kimi ACP permission requests are typed provider events. Adam
answers only the root call whose identity it can prove and preserves the
provider’s cancellation as a permission-blocked terminal outcome. Kimi’s ACP
adapter does not expose child caller identity, which is why Adam launches it
with an empty MCP server list instead of delegating task or canvas authority
to unknown children. For Kimi delegation requests, only an explicit
`subagent_type: explore` is treated as read-only. The provider’s default
`coder` profile—and any missing or unfamiliar profile—is mutation-capable, so
Ask and Sandbox do not silently grant it; background Agent jobs are refused
for every Adam stance because the per-turn ACP host cannot own their later
lifecycle. xAI Grok Heavy receives no local filesystem or Adam function tools
at all. Its optional `web_search` runs on xAI’s servers and is present only
when the user enabled that ability for the turn; an Adam access stance does
not pretend to govern hidden server-agent internals.

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
- Provider session IDs, xAI response IDs, and future compaction summaries live
  in versioned machine-local sidecars. Kimi projects model metadata without
  copying its opaque ACP session ID into portable conversation activity.
- Temporary API keys remain in memory and are scoped to the selected provider,
  so an xAI key cannot follow a later provider switch into a compatible HTTP
  endpoint. Grok Heavy may alternatively read `XAI_API_KEY`; the key is never
  stored in the conversation or activity log, and its runtime container’s
  debug representation redacts the entire temporary-key map.
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
- scoped child prose for Grok versions whose streams do not identify the
  emitting child session;
- live Kimi 0.31.0 child prose or child lifecycle—the ACP root tool reveals
  requested work first and stable member results only when it finishes;
- persistent Kimi background Agent jobs—foreground Agent and AgentSwarm
  delegation are the supported truthful lifecycle;
- child identities, child transcripts, or per-child completion counts for
  xAI Grok Heavy, whose default response keeps server-agent state opaque;
- Adam task-tool attachment for Kimi 0.31.0 or xAI Grok Heavy: Kimi uses its
  native root plan with no MCP servers, while Grok Heavy exposes neither a
  safe native plan nor client task functions;
- native Adam task-tool attachment for Ollama CLI builds that expose no
  verified external-tool transport;
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
