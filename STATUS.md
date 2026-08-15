# Live web tiles — "two real layers" (transparent punch-through)

Branch: `work/claude-web-twolayer`. macOS-only feature.

## The idea in one paragraph

The WKWebView is composited **below** the egui Metal layer instead of on top of
it. egui paints the whole window opaque (desk + all chrome), then Adam cuts a
transparent hole exactly at each live page's rect so the web view behind shows
through *only* there. Chrome that overlaps the page (quick bar, minimap, toasts,
banners) is painted after the hole and composites back on top, opaque. Input is
solved separately: the page's container view is still a real subview, so it
participates in hit-testing regardless of its compositing order — a custom
`hitTest:` returns the web view for points over the page and declines (nil) for
points over passthrough chrome, letting those fall through to egui.

This replaces the old family of workarounds (whole-page hide on chrome overlap,
the CAShapeLayer "notch" mask for the quick bar, the minimap special-casing),
all of which are deleted.

## Build / test status

- `cargo build --release --bin Adam` — compiles cleanly (see note below; verified
  green on `cargo check` for lib+bin, release bin build run before commit).
- `cargo test --release --lib` — passes. The `webview_policy` suite was trimmed to
  match the smaller input/placement structs; a new `web_hole` suite covers the
  scissor math and validates the punch shader with naga.
- **Not run as a GUI.** Per instructions I did not launch the app. Everything
  below about *visual* behaviour and *live input* is therefore designed-and-
  reasoned, not observed. This is the honest bit: the mechanism compiles and the
  pure logic is tested, but the actual on-screen compositing and hit-testing were
  not verified on a running window. The risk areas are called out explicitly.

## How the three pieces fit

1. **Transparent surface** (`src/main.rs`, `src/app.rs`)
   - `ViewportBuilder::with_transparent(true)`.
   - `impl eframe::App for AdamApp` now returns `clear_color = [0,0,0,0]`.
   - eframe's transparent path makes egui-wgpu pick a `CompositeAlphaMode` with
     transparency (PreMultiplied/PostMultiplied) and wgpu sets the `CAMetalLayer`
     non-opaque. Confirmed by reading egui-wgpu 0.35 `winit.rs` (it explicitly
     branches on `support_transparent_backbuffer`) — **not** confirmed on a live
     window.

2. **The hole punch** (`src/web_hole.rs`, new)
   - A tiny wgpu pipeline: fullscreen triangle, fragment `= (0,0,0,0)`, blend
     `REPLACE`, drawn once per hole under a scissor. REPLACE overwrites the alpha
     the opaque desk laid down, so the surface is see-through exactly in each
     clip rect and nowhere else. Modeled on `dots.rs`.
   - `show_canvas` reserves a paint slot (a `Shape::Noop`) **after the tiles but
     before the carried preview / note draft / minimap / quick bar / all overlay
     Areas**, so the punch cuts desk+tiles and everything drawn afterward
     composites on top of the page. `sync_live_webs` fills that slot once it knows
     the live set, with one callback carrying every visible clip rect (multiple
     pages supported, capped by `MAX_LIVE_WEB_PAGES`). No live page → the slot
     stays `Noop`, window fully opaque as before.

3. **Below-compositing + input** (`src/webview_host.rs`)
   - The container is now a custom `NSView` subclass `AdamWebContainer`
     (objc2 `define_class!`), still added as a subview of the winit content view
     (so it hit-tests) but with `layer.zPosition = -1`. `raw-window-metal` 1.1.0
     attaches the Metal layer as a **sublayer** of the content view's backing
     layer (verified in its source — "Reasoning behind creating a sublayer"), so
     the container and the Metal layer are sibling sublayers and a negative
     zPosition sorts the page behind egui's drawing.
   - `hitTest:` override: if the incoming point (content-view coords) is inside
     any stored passthrough-chrome rect → return `nil` (content view then returns
     itself → egui handles it); otherwise defer to `super`/the WKWebView.
   - `set_passthrough_chrome(&[PointRect])` converts the quick bar + minimap rects
     into the container's superview space (respecting `isFlipped`) and stashes
     them in an ivar for `hitTest:`.
   - The picture-stable zoom is untouched: WKWebView frame = natural size,
     camera = uniform `sublayerTransform` scale on the container layer. zPosition
     is re-asserted on each geometry change in case a layout pass resets it.

## Files / functions changed

- `src/main.rs` — `with_transparent(true)`.
- `src/lib.rs` — `pub mod web_hole;`.
- `src/web_hole.rs` — **new**. `install`, `paint_callback`, `WebHoleResources`
  (pipeline), `rect_to_scissor`, tests.
- `src/webview_policy.rs` — removed `LiveWebInputs::chrome_overlap`,
  `LiveWebInputs::quick_bar_rect`, `LiveWebPlacement::exclude`; removed the notch
  computation in `desired_state`; removed the three notch tests + the
  `chrome_overlap` gate assertion; trimmed `base_inputs`.
- `src/webview_host.rs` — new `AdamWebContainer` subclass with `hitTest:`;
  reparent-below via zPosition; `LiveWebHost::set_passthrough_chrome`; removed the
  `exclude`/`last_exclude` field + the CAShapeLayer/CGMutablePath notch mask and
  their imports.
- `src/app.rs`
  - `impl eframe::App`: added `clear_color`.
  - Deleted `AdamApp::transient_chrome_overlap`.
  - Struct: removed `last_quick_bar_rect`; added `chrome_passthrough_rects: Vec<Rect>`
    and `web_hole_slot: Option<(Painter, ShapeIdx)>` (+ ctor init, + `web_hole::install`).
  - `show_canvas`: reserve the hole slot after the tile loop; rebuild
    `chrome_passthrough_rects` from the quick bar and (new) minimap rect.
  - `draw_minimap`: now returns `Option<Rect>` (its footprint) instead of `()`.
  - `sync_live_webs`: dropped the `chrome_overlap`/`quick_bar_rect` inputs; collect
    each visible clip rect into `holes` and fill the reserved slot; push the
    passthrough chrome to every host.

## macOS gotchas & dead-ends (read before trusting the visuals)

- **The whole approach hinges on the Metal layer being a *sublayer*.** True for
  `raw-window-metal` 1.1.0 (current). If a future bump makes it the view's root
  backing layer, zPosition on a sibling won't put the page behind it and this
  design needs a different reparent (e.g. an intermediate content view with the
  Metal view and the web view as ordered siblings). Flagged as the #1 risk.
- **Transparent window = every non-opaque pixel now blends with the desktop, not
  a black backdrop.** This is *safe here only because* the desk fill + the three
  panels tile the whole window opaquely as a base, and anything semi-transparent
  (shadows, hovers) is drawn *over* that opaque base, so it stays opaque. The two
  places the desktop can leak are (a) the intended holes, and (b) any region
  where nothing opaque is ever painted — I believe there are none, but I could
  not eyeball it. Window shadow / rounded-corner rendering under a non-opaque
  NSWindow may also look subtly different; unverified.
- **Semi-transparent chrome over a hole.** Toasts and the pathway banner are
  *not* in the passthrough set (they're non-interactive) and are assumed opaque.
  If `colors.toast` / `colors.floating` carry alpha, the page would show through
  them where they overlap it, because the hole cleared the desk underneath. Old
  design hid the whole page for exactly this reason; the new one accepts it. If
  it looks wrong, either (i) add those rects to the punch's exclusion, or (ii)
  paint an opaque plate behind them.
- **hitTest coordinate space.** Assumes the point handed to the subview's
  `hitTest:` is in content-view (superview) coords matching egui logical points.
  Handled for both flipped/unflipped via `isFlipped`, but only the flipped case
  (winit's actual view) was reasoned end-to-end.
- **Zoom + native hit-testing (pre-existing, preserved).** The WKWebView's *view*
  frame is the unscaled natural size; the zoom is a layer transform, which view
  hit-testing ignores. So at zoom ≠ 1 only the unscaled sub-rect of the page is
  directly clickable; the cover-overshoot area falls through. This is unchanged
  from the current shipping behaviour, not introduced here.
- **AppKit may re-sync a subview layer's zPosition on layout.** Mitigated by
  re-asserting `zPosition = -1` on every geometry change in `apply()`. If flicker
  or "page pops in front" is seen, this is the first place to look.
- **Escape / focus** handling is unchanged from the previous host.

## If continuing

Highest-value next step is a single manual run to confirm (a) the desktop is
never visible, (b) a page is visible through its hole with the quick bar/minimap
on top and clickable, and (c) the page is interactive. If the page renders on top
of egui instead of behind, the zPosition ordering (or Metal-layer-is-sublayer
assumption) is wrong and is the thing to fix first.
