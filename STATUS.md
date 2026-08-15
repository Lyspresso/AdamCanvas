# Live web tiles — "native chrome on top" (branch `work/claude-web-nativechrome`)

## The bet
Keep the `WKWebView` as the **front-most** native child so its input just
works (no hit-test tricks, no click forwarding into the page). Then rebuild the
specific egui chrome that has to be usable *over* a website — the bottom-center
**quick bar** and the bottom-right **minimap** — as native AppKit overlays that
Adam raises **above** the web containers every frame. The top toolbar stays egui
(the web view is already clipped to the canvas rect, which starts below the
toolbar, so it never reaches it).

This replaces the old workaround family entirely: no more hiding the whole page
when it would cover chrome, and no more CAShapeLayer "notch" cut in the page for
the quick bar.

## Build / test status
- `cargo build --release --bin Adam` — **compiles cleanly** (2 pre-existing
  `dead_code` warnings in `src/ai.rs`, unrelated to this work).
- `cargo test --release --lib` — **passes**. The `webview_policy` tests were
  updated: the three "quick-bar notch" tests were deleted with the notch, and
  `chrome_overlap` / `quick_bar_rect` were removed from the test inputs.
- **Not run:** the GUI. Per the brief, the human verifies visually. I did not
  open the app.

## What works
- **Deletion of the workaround family** as specified: `transient_chrome_overlap`,
  the `LiveWebPlacement.exclude` field + the CAShapeLayer notch mask, the
  `chrome_overlap` policy gate, and the minimap special-casing are all gone.
  **Kept** (as instructed): the `live_web_shown` set and the "normal cursor over
  the page body" fix in `draw_tile`.
- **Native quick bar** (`AdamQuickBarOverlay`, a custom `NSView` created with
  objc2 `define_class!`): drawn entirely with CALayers — a rounded translucent
  background, one rounded `CALayer` per slot, and a `CATextLayer` glyph per
  functional slot (N P W I T · C V D · ×, plus the 3 empty decorative slots for
  width parity). It is positioned to match the egui bar's rect exactly (shared
  slot-size math via `quick_bar_slot_size`) and raised front-most each frame it
  is shown. Clicks are **real**: `mouseDown:` hit-tests the click x to a slot and
  posts it down a `crossbeam` channel; `apply_quick_bar_click` then runs the
  **same** actions the egui bar runs — arm N/P/W/I/T (double-click locks for
  repeated placement), copy, paste, duplicate, clear. The armed tool slot is
  highlighted (accent fill/border) and the × turns danger-colored while a tool is
  armed, mirroring the egui bar.
- **Native minimap**: a plain layer-backed `NSView` panel (canvas fill + border)
  with a single viewport-indicator sublayer, positioned to match
  `draw_minimap`'s rect and raised above the page.
- **No double-drawing**: an overlay is only shown where a *shown* page's clip
  rect actually intersects that chrome's rect (`shown_clips` in
  `sync_live_webs`). Where no page covers the chrome, the overlay is hidden and
  the egui version (drawn into the Metal layer, unoccluded) is the only one
  visible. So you never see two bars.
- **Picture-stable zoom preserved**: untouched. The WKWebView frame still stays
  the tile's natural size and the camera is still the container layer's
  `sublayerTransform` scale. No native pageZoom, no CSS residual.
- **Toolbar**: left egui. The web view's rect is still clipped to the canvas rect
  (`last_canvas_rect.min_y == TOOLBAR_HEIGHT`), so the page never extends over
  the toolbar — verified by reading the geometry path, not changed.

## What is rough / stubbed (be honest)
- **Toasts + the pathway problem banner will now be COVERED by a full-screen
  page.** Removing the `chrome_overlap` gate means the page no longer steps aside
  for them, and they were left as egui (per the brief's "toasts/banners can stay
  egui for now"). They render *under* the web view until a native version is
  built. This is a real regression for those two transient elements, traded for
  never blanking the page.
- **Minimap parity is partial.** The native panel draws the frame + the viewport
  rectangle, but **not the tile dots** the egui minimap draws. It reads as a
  minimap and shows where you are; it does not mirror content. Acceptable per the
  brief ("visual parity is nice-to-have; being above the web view and not
  blanking the page is the requirement").
- **Arming a create-tool over a full-screen page can't complete the placement.**
  The bar arms N/P/W/I/T correctly, but the *placement* click still has to land
  on the canvas — which the page covers, and pointer events there go to the web
  view. So copy/paste/duplicate/clear work immediately over a page; the four
  create-tools arm but you can only drop them once the page shrinks off / steps
  aside. This is the same fundamental "the page owns those pixels" limitation the
  previous design had; the win here is that the bar is now **visible and
  clickable** over the page at all.
- **Quick-bar glyph rendering**: `CATextLayer` `contentsScale` is hard-coded to
  2.0 (retina assumption; a non-retina display over-samples harmlessly rather
  than reading `backingScaleFactor`, which would have pulled the `NSWindow`
  feature). Vertical centering is a simple box-inset approximation. No pressed /
  hover animation on slots, and the native bar does not render the egui bar's
  tooltips or the "∞" locked-tool marker — the armed highlight is the only lock
  feedback.
- **One-frame latency** on clicks: AppKit delivers `mouseDown:` between egui
  frames; the click is drained on the next `sync_live_webs` and a
  `request_repaint` is issued so the armed state shows promptly. Not noticeable in
  practice but it is a frame late by construction.

## Files / functions changed
- **`Cargo.toml`** — added cargo features: `objc2-app-kit` → `NSEvent`,
  `NSGraphics`; `objc2-quartz-core` → `CATextLayer`; `objc2-core-graphics` →
  `CGColor`.
- **`src/webview_policy.rs`** — removed `chrome_overlap` and `quick_bar_rect`
  from `LiveWebInputs`; removed `exclude` from `LiveWebPlacement`; removed the
  `chrome_overlap` gate and the `exclude`/notch computation from `desired_state`;
  deleted the 3 notch tests and pruned `base_inputs` / the gate-row test.
- **`src/webview_host.rs`** — rewritten:
  - New cross-platform plain-data types (no AppKit): `OverlayRect`,
    `OverlayPalette`, `QuickBarLayout`, `MinimapLayout`, `WebChromeInputs`,
    `QuickBarClick`, and the `QUICK_GLYPHS` table.
  - `LiveWebHost`: dropped `last_exclude` + the CAShapeLayer notch mask (and the
    `CAShapeLayer`/`CGMutablePath`/`kCAFillRuleEvenOdd` imports); added a private
    `parent_view()` accessor.
  - New: `AdamQuickBarOverlay` (`define_class!`), `WebChromeOverlays`
    (`sync` / `hide` / `take_clicks` / `Drop`), and the free helpers
    `configure_quick_bar`, `configure_minimap`, `place`, `cg`, `new_panel_view`.
  - Non-macOS `WebChromeOverlays` stub (no-op), exported alongside `LiveWebHost`.
- **`src/app.rs`**:
  - `AdamApp` fields: added `last_minimap: Option<(Rect, Rect)>` and
    `web_chrome: WebChromeOverlays`; kept `last_quick_bar_rect` (repurposed —
    see deviation below); constructor initializes both.
  - Deleted `transient_chrome_overlap`.
  - `sync_live_webs`: dropped the `chrome_overlap`/`quick_bar_rect` inputs;
    collects `shown_clips`; calls the new `sync_web_chrome`.
  - New methods: `sync_web_chrome` (builds `WebChromeInputs`, calls
    `web_chrome.sync`, drains + applies clicks) and `apply_quick_bar_click`.
  - New free fns: `quick_bar_slot_size`, `to_overlay_rect`, `quick_bar_layout`,
    `web_chrome_palette`.
  - `show_canvas_quick_bar`: now uses the shared `quick_bar_slot_size`.
  - `draw_minimap`: returns `Option<(Rect, Rect)>`; the call site stores it in
    `self.last_minimap`.

## AppKit gotchas (the fiddly parts)
- **`define_class!` (first use in this codebase).** `AdamQuickBarOverlay`
  subclasses `NSView`, so it inherits `MainThreadOnly` automatically. Construction
  is `mtm.alloc::<Self>().set_ivars(…)` → `msg_send![super(this), init]` (init
  routes to `initWithFrame:NSZeroRect`; the real frame is set later). The ivars
  hold the `crossbeam` `Sender` plus `Cell<f64>`/`Cell<usize>` slot geometry, and
  are dropped by the macro-generated `dealloc` — no manual `Drop` on the view.
- **Button target/action was deliberately avoided.** No `NSButton`, no
  target/action wiring (the fiddliest objc2 path). Instead one custom view draws
  every slot as CALayers and dispatches clicks by hit-location in `mouseDown:` →
  channel. This is the brief's sanctioned "defensible first cut," but note it is
  still *real* clicks, not a poll. `acceptsFirstMouse:` returns true so the first
  click works even when the window wasn't key.
- **Subview ordering.** `addSubview:positioned:relativeTo:` with
  `NSWindowOrderingMode::Above` and `relativeTo: nil` raises a view front-most
  over **all** siblings (every live web container), re-applied each shown frame so
  a web host created later can't jump back on top. This method is gated behind the
  `NSGraphics` cargo feature (hence the feature add).
- **Retaining the controls.** `WebChromeOverlays` owns
  `Retained<AdamQuickBarOverlay>` + `Retained<NSView>`; its `Drop` calls
  `removeFromSuperview`. The `Sender` lives in the view's ivars; the matching
  `Receiver` lives in `WebChromeOverlays`. Both overlays are lazily created the
  first frame they're needed and then reused (hidden, not torn down, when idle).
- **`kCAAlignmentCenter` is an extern static** → reading it needs `unsafe`.
- **`CATextLayer::setString` is `unsafe`**; the `NSString` is passed as
  `&AnyObject` via multi-step deref coercion.
- **Coordinate flips.** The winit parent is flipped (top-left origin), so overlay
  *frames* are placed with the same top-left math the web container uses. But each
  overlay is itself unflipped (bottom-left), so its internal slot layout and the
  `mouseDown:` hit-test both use bottom-left coords — self-consistent and
  independent of the parent's flip. The minimap viewport indicator is converted
  from screen (top-left) to overlay-local (bottom-left) explicitly.
- **`WebChromeOverlays` is `!Send`** (holds `Retained<NSView>`), which is fine —
  `AdamApp` already is, via `LiveWebHost`.

## Honest comparison notes vs the "hit-test / web-behind" approach
Strengths of this approach: the page keeps 100% of its pixels and its input path
is completely untouched (no event synthesis, no `hitTest:` games on the web view),
and the picture-stable zoom is entirely undisturbed. The quick bar is genuinely
clickable over a full-screen site.

Weaknesses to weigh: it reintroduces native-view plumbing for *every* piece of
chrome that must sit above a page — today just the quick bar + minimap, but
toasts/banners are now occluded and would each need the same treatment. The
native chrome is a second rendering path that must be kept visually in sync with
the egui original by hand (colors are lifted from the theme, but tiles, tooltips,
lock markers, and hover states are not reproduced). If the set of "chrome above
the page" keeps growing, this cost grows with it.
