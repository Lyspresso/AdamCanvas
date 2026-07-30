# Adam

Adam is a fast, local-first visual workspace for macOS, built in Rust.
Drop documents, spreadsheets, presentations, PDFs, images, video, audio,
folders, archives, code, and other Finder items onto an adjustable canvas.
Notes and websites can be created directly in the app.

## What works

- Import multiple files or folders, drag files from Finder, or paste files,
  images, text, websites, and copied Adam tiles.
- Move and resize every tile from any edge or corner—even while zoomed far
  out—on a freely resizable canvas, with an optional grid and snap-to-grid.
- Photo tiles adopt the image’s natural proportions and draw edge-to-edge;
  they keep that aspect while resizing unless Shift is held for freeform.
- Select with Command-click or drag a marquee around a group.
- Copy, paste, duplicate, delete, and drag a selection onto another page in
  the sidebar; hovering a page opens it before the drop.
- Double-click files and websites to open them; press Space for a Quick Look
  preview of the selected file.
- Create, rename, duplicate, reorder, resize, fit, and delete pages.
- Create translucent spatial piles that stay behind their tiles. A pile has
  no membership graph: overlap is membership, its name is inherited as a
  tag, and one broad pile can watch many tiles with an optional timed rule.
- Create standalone tag tiles, persistent AI-chat tiles, rich notes, and
  timed or instant auto-tag rules.
- Chat with locally installed Codex, Grok, or Claude Code from a dedicated
  three-pane workspace. Chats stream structured activity, keep a searchable
  output library, queue safely at the four-run limit, and can resume supported
  provider sessions.
- Choose a permission stance per chat, review sensitive actions, stop live
  runs, retry or regenerate turns, and revert completed Adam changes from
  recorded checkpoints. Protected and out-of-scope canvas items remain gated.
- Organize chats into projects and reusable characters, attach scoped
  append-only memory, manage custom CLI agents and skills, and schedule
  one-time or repeating prompts. Scheduled work is reconciled after relaunch.
- Protect important tiles, filter globally by tag, inspect details and tag
  provenance, and restore items from Adam’s Trash.
- Right-click a photo for an editable two-sentence visual description,
  summary, subject keywords, notes, image facts, tags, piles, and provenance.
  On-demand local text recognition can be copied by itself or exported as a
  complete Adam dossier.
- Undo and redo more than 200 workspace operations.
- Restore the workspace automatically after relaunching.
- Choose System, Light, Dark, or one of 16 named color themes from the
  Appearance menu. Every supplied five-color palette has a compact preview,
  readable derived contrast, and persists across relaunches. Dark mode still
  uses a `#2B2B2B` canvas with black top, sidebar, background, and canvas edge.
- Toggle Adam’s default-on Dots field from the same menu. One continuous Flow
  pattern spans the top bar and sidebar only; the canvas and desk keep their
  solid colors. Its tint and background follow the selected theme, including
  white dots on black in Dark mode and black dots on white in Light mode.
  macOS Reduce Motion freezes the field without changing the saved preference.
- Use square outlined toolbar controls and a square white outline for the
  active canvas in both appearances.
- Use Source Sans 3, the maintained open-source continuation of Foundry’s
  Source Sans Pro default, throughout Adam’s proportional UI text.

Useful shortcuts:

| Action | Shortcut |
| --- | --- |
| Import files | Command-O |
| Import a folder | Shift-Command-O |
| Copy / paste | Command-C / Command-V |
| Duplicate | Command-D |
| Undo / redo | Command-Z / Shift-Command-Z |
| Delete selection | Delete |
| Quick Look | Space |
| Fit canvas | F |
| Zoom in / out / reset | Command-+ / Command-- / Command-0 |
| Pan | Trackpad scroll, middle-drag, or Space-drag |
| Preserve aspect for non-photo tiles | Hold Shift |
| Freeform-resize a photo | Hold Shift |

## Performance design

Adam uses one Metal-backed canvas instead of creating a native view for
every tile. It requests the low-power GPU, uses event-driven repainting, culls
offscreen tiles through a compact spatial index, and draws static previews
until an item is opened.

Dots uses one GPU callback, pipeline, uniform, clock, and full-screen
coordinate field. Two hardware scissors form the connected top-and-sidebar
shape without shading the canvas. It idles at 30 Hz only while Adam is visible
and focused, and stops periodic redraws when hidden or unfocused.

Photo previews are generated only for visible tiles on one bounded background
worker. Their resolution adapts to the tile's on-screen Retina size through
stable 256–4096 px tiers, while lower-resolution previews remain visible during
an upgrade. On macOS, Image I/O downsamples system-supported image formats
during decoding, applies their orientation, and converts color to sRGB without
first allocating a full-resolution raster. Versioned tier files avoid decoding
the same photo again after a relaunch. Other files continue to use bounded
Quick Look thumbnails. Decoded textures have a 128 MB budget and are evicted
when unused. Large pasted images are encoded off the UI thread. Workspace saves
are debounced, atomic, and performed off the UI thread. The release profile
uses thin LTO, one code-generation unit, symbol stripping, and abort-on-panic.

Photo text recognition and scene classification are opt-in, run on one bounded
background worker using Apple Vision, and store their result with the photo
revision. They perform no network request and do no analysis while Adam is
idle.

The model and culling test suite includes canvases above the requested
100-item workload. Actual battery draw and frame pacing still need to be
measured with Instruments on the target Mac before making a
hardware-specific guarantee.

## AI chat

Open **AI Chat** from the toolbar, or add an AI chat tile to the current
canvas. Adam detects supported local CLI agents and launches each run directly,
without a shell. Its three-pane chat workspace includes separate Home, Cowork,
and Code conversation pools plus Cast, a character-centered view across all
three. It also includes searchable history, projects, reusable skills and
characters, an outputs library, durable queues, and local-time schedules. Completed
background replies can appear as native macOS notifications and reopen the
correct chat when clicked. Schedules run while Adam is open; overdue one-shots
catch up, while repeating schedules use a 15-minute catch-up window.

Adam projects only the chat’s permitted workspace context into a run. Tool
requests return through a loopback-only authenticated server, are checked
against the active run and permission stance, and are then applied on Adam’s
UI thread. Secrets selected for a custom agent remain process environment
values and are not written into chat history. Each run receives a short-lived
credential and private process boundary; Stop, timeout, quit, and normal
completion revoke it and clean up descendant processes. Adam reports a built-in
agent as connected only after its registration command succeeds and an
authenticated tool-list request reaches the live server.

Chats can keep scoped, append-only memory through a character or project.
Canvas changes are permission-gated, privacy-checked against the current page,
and recorded with a durable rewind checkpoint before the agent is told they
succeeded. The first chat message can generate a local title when a compatible
local LM Studio endpoint is available; long histories can also be compacted
locally, and memory observations can be synthesized into derived recall without
altering their append-only log. No local-model enrichment is required for
chatting.

Chat transcripts and AI sidecars live alongside Adam’s app data in a separate
`ai-chat-history` store. Canvas chat tiles are lightweight links: copying a tile
creates a fresh chat shell, and deleting a chat permanently removes its linked
tiles.

## Current preview boundaries

Adam keeps the canvas light by showing bounded, static previews. PDFs and
office documents use Quick Look thumbnails and open in their native app for
full interaction; websites use a local card and open in the default browser.

Custom CLI agents can participate in text chat, but Adam canvas, task, and
memory tools currently require the built-in Codex, Grok, or Claude Code
integrations. Provider model traffic follows the selected CLI’s own account,
configuration, and privacy terms; Adam itself does not proxy prompts through
an Adam-hosted model service.

## Build the app

Requirements: macOS 13 or newer, Rust 1.92 or newer, and Xcode 26 or newer
to compile the supplied layered `Adam.icon` with Apple’s asset compiler.

```sh
./scripts/build_app.sh
```

The finished app is written to `build/Adam.app` and ad-hoc signed for local
use. A public release should be signed with an Apple Developer ID and
notarized.

For development:

```sh
cargo test --all-targets
cargo run --release
```

Workspace data is stored in
`~/Library/Application Support/Adam/`. On first launch, an existing Mosaic
library is copied into Adam without deleting the original. Imported files
and folders are copied into Adam’s content-addressed local asset store, so
they remain available if the originals move. Pasted images use the same
managed store.
