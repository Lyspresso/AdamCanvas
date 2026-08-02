# Pathways — plan of record

Status: **active**. Feature specified by Lydia (2026-08-02) against EarlIt's shipped
implementation. **Codex implements; Claude plans, gates and reviews** — same arrangement
as `progress-artifacts-parity.md`.

A pathway is a rail drawn on a page; tiles are cargo riding it on **wall-clock time** — a
schedule, not an animation. Close Adam for three days, reopen, and a tile is where it
would have been, with history saying so. Tiles enter and exit piles as they pass, and
those crossings are real events at exact instants.

## The one idea everything falls out of: two clocks that never touch

- **Render clock is pure.** `position(assignment, pathway, now) -> point` — a function of
  persisted state and time. No writes, no side effects, no store access. The *same*
  function serves rendering, hit-testing, marquee selection and drag snapshots, so they
  cannot disagree.
- **Durable clock ticks only at transitions** — segment arrival, dwell end, exact pile
  boundary crossing. 40 seconds of gliding = **zero writes**.

Everything below exists to protect that invariant.

## What Adam already has (verified, with refs)

Adam is a better fit than expected. Do not rebuild these:

- **Geometry**: `WorldRect { x, y, w, h }` top-left + size, page-scoped (`model.rs:18`).
  `center()` is derived (`model.rs:100`). Camera never rewrites tile coords.
- **Piles are real and page-scoped** (`domain.rs:1950`), with **the exact four membership
  modes the spec names** — `ContainmentMode { CenterInside, MajorityOverlap,
  CompletelyInside, AnyOverlap }` (`domain.rs:644`) and a **pure** predicate
  `contains(pile, tile) -> bool` (`domain.rs:653`). `MajorityOverlap` is already
  `intersection_area > tile_area * 0.5` (`domain.rs:661`) — the function to solve.
- **Durable wall-clock dwell already exists**: `MembershipProgress` accrues across app
  closures via `count_while_closed` / `wall_delta`, evaluated by the pure
  `evaluate_membership_progress` (`domain.rs:1414`). This is Adam's Keep-Tag equivalent —
  Pathways feeds it, it decides meaning.
- **The arrival seam already accepts caller geometry AND a caller instant**:
  `ReconcileRequest { objects: &[CanvasObject], now: UnixMillis, .. }`
  (`automation.rs:28`) → `reconcile_workspace` (`automation.rs:94`). Pathways can
  evaluate membership *at the exact crossing instant* without inventing a new engine.
- **Append-only log precedent**: `PileHistory` with monotonic `sequence`, duplicate-id
  rejection, `DomainActor`, undo-by-appending (`domain.rs:1847-1919`).
- **Persistence**: advisory lock, atomic temp+fsync+rename, three-way merge across
  processes, explicit carry-forward of unknown JSON members, `#[serde(default)]` on
  `DomainState` — a new `pathways` field is decode-compatible for free (`persistence.rs`).
- **Reconcile-loop precedent**: `poll_automation` (1 Hz, pauses during drag via `settled`).

## What Adam lacks

No pathway/route/node/segment concept. No derived position — **position IS the stored
`tile.rect`**, mutated in place. No wall-clock *next-transition* scheduler (only
`request_repaint_after` off the monotonic clock). No analytic solver (the pile engine
**samples** at 1 Hz). No circuit breaker, no hysteresis, no workspace-level event log.
No wake-from-sleep hook **on either platform**. No yarn/connections.

## Four sharp risks — read before writing code

1. **Undo would silently rewind pathway history.** `restore_workspace` (`app.rs:1462`)
   replaces the whole `Workspace` including `domain`. Pathway assignments and the event
   log must be exempted **exactly like the existing precedent** at `app.rs:1462-1472`,
   which already carries forward `deleted_conversations` and host-artifact provenance as
   *"durable audit state, not reversible canvas layout."* Join that list.
2. **Do not feed projected rects to `reconcile_workspace` naively.**
   `sync_pile_geometry` (`automation.rs:121`) writes `pile.rect = object.rect` back from
   the caller's slice. Passing pathway-projected geometry there would **durably move the
   user's piles** — data corruption sitting on the obvious implementation path. Pathways
   must supply moving-tile geometry without letting pile rects be written back.
3. **The spatial index is a cached snapshot** rebuilt only when `spatial_dirty` is set.
   A tile moving on the wall clock with no user event would be marquee-selectable and
   pile-testable at a **stale** position while rendering correctly — breaking the
   "render and input cannot disagree" invariant. The pure position must be injected at
   the one bridge that builds rects for consumers (`canvas_objects_from_workspace`,
   `automation.rs:67`) and the index must not go stale under wall-clock motion.
4. **The commit hook has no actor identity.** Adam's `changed(bool)` (`app.rs:1422`) sets
   flags with no actor and no operation id, so Pathways cannot disown its own commits the
   way EarlIt does (`guard !isCommittingProjection, !actor.hasPrefix("pathway-")`).
   Without a discriminator, every pathway-owned write re-triggers reconcile — an infinite
   wake loop. A discriminator is required before the reconciler is wired to commits.

Also: Adam's save is a **debounced background write**, not a synchronous transaction, so
the spec's save-failure protocol (failure rows written at the *start of the next commit*,
because a row saying "the save failed" cannot survive the transaction that failed) needs
an Adam-shaped answer, not a direct port.

## Algorithm to port faithfully (~1,050 lines of the 5,460 Swift)

From `EarlIt/Sources/EarlIt/PathwayProjection.swift` (898 lines) and the reconcile half of
`PathwayController.swift` (~lines 784-1502). Everything else is Swift/SwiftData plumbing —
do not port it.

- **Position** (`PathwayProjection.swift:379-432`): `safeLength = max(1e-6, |end-start|)`,
  `speed = max(1, speedPointsPerSecond)`, `travelled = elapsed * speed / safeLength`,
  clamped start progress. Non-moving states read the current node; unresolvable geometry
  falls back to `materializedRoutePoint` rather than failing.
- **Liang-Barsky boundary solve** (`:444-498`) for the linear modes; **piecewise-quadratic
  solve for `MajorityOverlap`** (`:572-700`) over breakpoints where the two 1-D overlap
  functions change slope. **It solves; it does not sample.**
- **Half-open containment everywhere** (`:512`): `x >= minX && x < maxX && y >= minY &&
  y < maxY`, so adjacent piles sharing an edge cannot double-count. Port verbatim.
- **Boundary ordering** (`:492-497`): sort by progress; at equal progress an *entering*
  boundary sorts before an *exiting* one, so a grazing pass reads enter-then-exit.
- **`pointJustAfterBoundary`** (`:500-510`): nudge forward by
  `min(1e-6, 1e-4 / segmentLength)` in progress space to land unambiguously on the
  intended side of a half-open edge.
- **The epsilon protocol is load-bearing and non-uniform**: 1e-9 (parallel test, boundary
  dedup quantization), 1e-7 (planner window), 1e-6 (endpoint exclusion, candidate date
  filter, nudge cap), 1e-5 (departure backdates `lastReconciledAt` by 10µs so a crossing
  exactly at a segment start is not filtered away). Document each where it appears.
- **Deterministic transition ordering**: sort by `(date, assignment_id)`; batch everything
  within 0.5 ms. EarlIt compares UUID *strings* — pick one rule, write it down, and make
  the tie-break independent of time representation.
- **Offline lap-skip**: collapse whole identical laps into one event, **always replay the
  final partial lap exactly**, and refuse to skip when an approval gate is in the loop, the
  walk leaves the route, it never returns to its anchor, or the lap is under 50 ms. The
  clamps must be **identical** in the lap-duration and replay paths or phase drifts.

## End-of-route semantics (Lydia, 2026-08-02) — a tile never travels backwards

**A route either leads to its end or it loops. There is no third behaviour, and nothing
ever returns a tile to where it started.**

- **Completed** — the tile comes to rest **at the final node** and stays there. It does
  *not* return to its pre-enrollment position, and that position is not remembered as a
  destination. Enrollment is one-way.
- **Loop** — `repeats` is a **real closing segment** from the last node back to the first.
  A looping tile *travels* that segment at its speed like any other leg, emitting the pile
  crossings along it. It never teleports to the start.
- **Detached** — the tile stays exactly where it was when authority was lost
  (materialise, then detach). No snap-back.
- **Paused / blocked / waiting** — the tile holds its current position.

The only position a pathway may ever write is the one the route implies at that instant.
No code path may restore a tile's pre-enrollment position for any reason.

*(Distinct from risk 2 below, which is about **piles**: a pile's own rect must never be
modified by pathway motion at all.)*

## PR sequence

Each starts from current `main`, opened as its own PR, CI green, reviewed before the next.
**Do not bundle.** (The one lesson this project has re-learned most expensively.)

### P1 — model, persistence, and the pure geometry module
Five nouns as serde types under `DomainState` (`#[serde(default)]`); the append-only
`PathwayEvent` log modelled on `PileHistory`; **the undo exemption from risk 1**; and a
standalone pure-geometry module (position, Liang-Barsky, majority solve, boundary ordering,
nudge) with unit tests ported from EarlIt's 1,671 test lines. **No motion, no UI.**
Reviewable as pure functions with fixtures.

### P2 — authoring
`PathwayEditingService` equivalent: create/append/move/remove nodes, always
**pause-for-graph-edit first** (never edit a live graph); segments rebuilt with speeds
preserved by `(from,to)`; `repeats` as a **real closing segment**, and removing it must
relocate cargo riding that segment back to the surviving node as `waiting` with
`waitUntil = now`, not leave it wedged on a dead segment id.

### P3 — the pure-position bridge and rendering
Inject `position(...)` at the single rect-building bridge so rendering, hit-testing,
marquee and drag snapshots all read it (**risk 3**, including the stale spatial index).
Repaint scheduling for moving cargo only — a dwelling or blocked tile costs nothing.
Position applies under reduce-motion (it is functional position, not decoration).

### P4 — the reconcile engine
The planner (`nextTransition`), the analytic candidate set, deterministic same-instant
batching, the seven-state machine (`beginAtNode` / `depart`), materialisation as the only
durable position write, the 10,000-transition circuit breaker, dangling-reference
flagging, and the commit-hook discriminator (**risk 4**). Pile crossings evaluated **at the
exact crossing instant** through `ReconcileRequest` — **without** the pile-geometry
writeback of **risk 2**.

### P5 — enrollment, offline catch-up, and safety
Dock resolver: flatten all routes to a value type **once**, then pure arithmetic per
pointer tick with hysteresis so the highlight cannot flicker; review-first enrollment
("At This Spot" vs "At the Beginning"); detach-never-delete on manual drag and on
externally-moved tiles (>0.5 pt from the materialised reference); launch-time catch-up and
lap-skip; the Adam-shaped save-failure protocol.

## Decisions taken (Lydia to revisit if wrong)

- **Yarn is dropped** (confirmed by Lydia 2026-08-02). Adam has no connection primitive; the EarlIt dependency is
  one-directional (yarn reads the projection, never the reverse). Tiles glide, nothing
  trails them. No coherence loss.
- **No full `CanvasTileIndex`.** Pathways needs id→rect, tile type, page scope — all
  already on `CanvasObject` (`domain.rs:732`).
- **Wake-from-sleep is out of scope for v1** (missing on both platforms). Launch-time
  catch-up is implemented; a sleeping-then-waking Mac catches up at next launch or next
  reconcile rather than instantly. Revisit if it feels wrong in use.
- **Cross-platform from day one.** Adam now ships on Windows (PRs #25/#26). No
  macOS-only mechanism may be load-bearing; anything platform-specific needs a Windows
  answer or must degrade honestly.

## Acceptance

1. A tile enrolled on a route with a dwell node and a repeat closure, left for a
   multi-hour app closure, resumes at the position and lap the wall clock implies, and the
   history explains the gap in one aggregate catch-up entry plus an exact final lap.
2. Pile entry/exit fire at the **solved** boundary instants (not 1 Hz samples) and feed
   `MembershipProgress` at those instants.
3. Dragging a tile off a route detaches it, preserving history; the row is never deleted.
4. Undo of an unrelated canvas action does not rewind pathway history.
5. Pile rects are never modified by pathway motion.
6. Marquee selection and hit-testing agree with rendering for a moving tile at any
   instant — including while it moves with **no user input at all** (the stale-index case
   in risk 3). Test by advancing the clock without generating events, then marquee-select.
7. A completed route leaves its tile resting at the final node; a looping route carries it
   around the closing segment. **No path returns a tile to its pre-enrollment position.**
8. Editing a live route is impossible without pausing; removing a repeat closure never
   wedges cargo.
9. A route that would generate unbounded transitions trips the breaker, disables itself,
   and says so — visibly, not silently.

## Lane protocol

Codex owns model, persistence, geometry, reconcile engine, authoring services. Claude owns
inspector/rail rendering surfaces and reviews everything. Protected as always:
`AGENTS.md`, `.github/`, `docs/plans/`. Repo is public — no personal paths in fixtures.
