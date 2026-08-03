use crate::model::{CanvasPage, Tile, TileKind, WorldRect};
use std::collections::HashMap;
use uuid::Uuid;

pub const DEFAULT_CELL_SIZE: f32 = 512.0;
const MAX_ENUMERATED_CELLS: u64 = 4_096;

type Cell = (i32, i32);

/// A compact uniform-grid index over tile vector indices.
///
/// Query results are returned in ascending source-vector order. The index does
/// not invent or store a z-coordinate; callers remain free to interpret their
/// model order in whichever drawing convention they use.
#[derive(Clone, Debug)]
pub struct SpatialIndex {
    cell_size: f32,
    cells: HashMap<Cell, Vec<usize>>,
    rects: Vec<WorldRect>,
    oversized: Vec<usize>,
}

impl SpatialIndex {
    pub fn new(cell_size: f32) -> Self {
        Self {
            cell_size: valid_cell_size(cell_size),
            cells: HashMap::new(),
            rects: Vec::new(),
            oversized: Vec::new(),
        }
    }

    pub fn from_tiles(tiles: &[Tile], cell_size: f32) -> Self {
        let mut index = Self::new(cell_size);
        index.rebuild(tiles);
        index
    }

    pub fn from_page(page: &CanvasPage, cell_size: f32) -> Self {
        Self::from_tiles(&page.tiles, cell_size)
    }

    pub fn cell_size(&self) -> f32 {
        self.cell_size
    }

    pub fn len(&self) -> usize {
        self.rects.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rects.is_empty()
    }

    pub fn clear(&mut self) {
        self.cells.clear();
        self.rects.clear();
        self.oversized.clear();
    }

    pub fn rebuild(&mut self, tiles: &[Tile]) {
        self.rebuild_rects(tiles.iter().map(|tile| tile.rect));
    }

    pub fn rebuild_rects(&mut self, rects: impl IntoIterator<Item = WorldRect>) {
        self.clear();
        for rect in rects {
            self.insert(rect);
        }
    }

    /// Rebuilds only when the source-ordered geometry changed.
    ///
    /// Wall-clock pathway motion has no model mutation with which to mark the
    /// index dirty, so canvas frames use this comparison to advance the index
    /// through the final projected endpoint as well as the animated frames.
    pub fn refresh_rects(&mut self, rects: impl IntoIterator<Item = WorldRect>) -> bool {
        let rects = rects.into_iter().collect::<Vec<_>>();
        if self.rects == rects {
            return false;
        }
        self.rebuild_rects(rects);
        true
    }

    /// Appends a rectangle and returns the source index assigned to it.
    pub fn insert(&mut self, rect: WorldRect) -> usize {
        let index = self.rects.len();
        self.rects.push(rect);
        self.index_rect(index, rect);
        index
    }

    /// Updates an existing rectangle without changing its source index.
    pub fn update(&mut self, index: usize, rect: WorldRect) -> bool {
        let Some(previous) = self.rects.get(index).copied() else {
            return false;
        };
        self.unindex_rect(index, previous);
        self.rects[index] = rect;
        self.index_rect(index, rect);
        true
    }

    /// Returns source-vector indices whose exact rectangles intersect `visible`.
    ///
    /// The grid only generates candidates; the exact intersection check prevents
    /// false positives. Returned indices are sorted to preserve model order.
    pub fn query_visible(&self, visible: WorldRect) -> Vec<usize> {
        if !visible.is_finite() || self.rects.is_empty() {
            return Vec::new();
        }

        let mut candidates = self.oversized.clone();
        match cell_span(visible, self.cell_size) {
            Some(span) if span.cell_count <= MAX_ENUMERATED_CELLS => {
                for x in span.min_x..=span.max_x {
                    for y in span.min_y..=span.max_y {
                        if let Some(indices) = self.cells.get(&(x, y)) {
                            candidates.extend_from_slice(indices);
                        }
                    }
                }
            }
            // An enormous viewport is cheaper and safer to answer by scanning
            // the compact rectangle array than by walking millions of empty cells.
            _ => candidates.extend(0..self.rects.len()),
        }

        candidates.sort_unstable();
        candidates.dedup();
        candidates.retain(|&index| {
            self.rects
                .get(index)
                .is_some_and(|rect| rect.intersects(visible))
        });
        candidates
    }

    /// Maps an exact spatial query back through the page's source order while
    /// excluding pile regions from ordinary canvas selection.
    pub fn query_non_pile_tile_ids(&self, tiles: &[Tile], visible: WorldRect) -> Vec<Uuid> {
        self.query_visible(visible)
            .into_iter()
            .filter_map(|index| {
                tiles
                    .get(index)
                    .filter(|tile| tile.kind() != TileKind::Pile)
                    .map(|tile| tile.id)
            })
            .collect()
    }

    fn index_rect(&mut self, index: usize, rect: WorldRect) {
        let Some(span) = cell_span(rect, self.cell_size) else {
            return;
        };
        if span.cell_count > MAX_ENUMERATED_CELLS {
            self.oversized.push(index);
            return;
        }
        for x in span.min_x..=span.max_x {
            for y in span.min_y..=span.max_y {
                self.cells.entry((x, y)).or_default().push(index);
            }
        }
    }

    fn unindex_rect(&mut self, index: usize, rect: WorldRect) {
        let Some(span) = cell_span(rect, self.cell_size) else {
            return;
        };
        if span.cell_count > MAX_ENUMERATED_CELLS {
            if let Some(position) = self.oversized.iter().position(|item| *item == index) {
                self.oversized.swap_remove(position);
            }
            return;
        }

        for x in span.min_x..=span.max_x {
            for y in span.min_y..=span.max_y {
                let cell = (x, y);
                let remove_cell = if let Some(indices) = self.cells.get_mut(&cell) {
                    if let Some(position) = indices.iter().position(|item| *item == index) {
                        indices.swap_remove(position);
                    }
                    indices.is_empty()
                } else {
                    false
                };
                if remove_cell {
                    self.cells.remove(&cell);
                }
            }
        }
    }
}

impl Default for SpatialIndex {
    fn default() -> Self {
        Self::new(DEFAULT_CELL_SIZE)
    }
}

#[derive(Clone, Copy, Debug)]
struct CellSpan {
    min_x: i32,
    min_y: i32,
    max_x: i32,
    max_y: i32,
    cell_count: u64,
}

fn cell_span(rect: WorldRect, cell_size: f32) -> Option<CellSpan> {
    if !rect.is_finite() {
        return None;
    }
    let min_x = cell_coordinate(rect.min_x(), cell_size);
    let min_y = cell_coordinate(rect.min_y(), cell_size);
    let max_x = cell_coordinate(rect.max_x(), cell_size);
    let max_y = cell_coordinate(rect.max_y(), cell_size);
    let width = i64::from(max_x) - i64::from(min_x) + 1;
    let height = i64::from(max_y) - i64::from(min_y) + 1;
    let cell_count = u64::try_from(width)
        .ok()?
        .saturating_mul(u64::try_from(height).ok()?);
    Some(CellSpan {
        min_x,
        min_y,
        max_x,
        max_y,
        cell_count,
    })
}

fn cell_coordinate(value: f32, cell_size: f32) -> i32 {
    (value / cell_size).floor() as i32
}

fn valid_cell_size(cell_size: f32) -> f32 {
    if cell_size.is_finite() && cell_size > 0.0 {
        cell_size
    } else {
        DEFAULT_CELL_SIZE
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{TileContent, TileKind};
    use uuid::Uuid;

    fn tile(index: usize, rect: WorldRect) -> Tile {
        Tile {
            id: Uuid::from_u128(index as u128 + 1),
            title: format!("Tile {index}"),
            rect,
            content: TileContent::Note {
                text: String::new(),
            },
            canvas_style: crate::model::CanvasTileStyle::Standard,
            intrinsic_image_size: None,
        }
    }

    #[test]
    fn query_is_exact_and_keeps_source_order_across_cells() {
        let tiles = vec![
            tile(0, WorldRect::new(600.0, 0.0, 50.0, 50.0)),
            tile(1, WorldRect::new(-30.0, -30.0, 50.0, 50.0)),
            tile(2, WorldRect::new(90.0, 90.0, 20.0, 20.0)),
            tile(3, WorldRect::new(2_000.0, 2_000.0, 20.0, 20.0)),
        ];
        let index = SpatialIndex::from_tiles(&tiles, 100.0);

        assert_eq!(
            index.query_visible(WorldRect::new(-10.0, -10.0, 130.0, 130.0)),
            vec![1, 2]
        );
        assert_eq!(tiles[1].kind(), TileKind::Note);
    }

    #[test]
    fn updating_a_rect_moves_its_visibility() {
        let mut index = SpatialIndex::new(64.0);
        let item = index.insert(WorldRect::new(0.0, 0.0, 20.0, 20.0));
        assert_eq!(
            index.query_visible(WorldRect::new(-1.0, -1.0, 22.0, 22.0)),
            vec![item]
        );

        assert!(index.update(item, WorldRect::new(500.0, 500.0, 20.0, 20.0)));
        assert!(
            index
                .query_visible(WorldRect::new(-1.0, -1.0, 22.0, 22.0))
                .is_empty()
        );
        assert_eq!(
            index.query_visible(WorldRect::new(490.0, 490.0, 40.0, 40.0)),
            vec![item]
        );
    }

    #[test]
    fn more_than_one_hundred_tiles_match_brute_force_visibility() {
        let tiles: Vec<_> = (0..180)
            .map(|index| {
                let column = index % 18;
                let row = index / 18;
                tile(
                    index,
                    WorldRect::new(
                        column as f32 * 120.0 - 240.0,
                        row as f32 * 95.0 - 190.0,
                        88.0,
                        66.0,
                    ),
                )
            })
            .collect();
        let visible = WorldRect::new(100.0, 100.0, 720.0, 430.0);
        let expected: Vec<_> = tiles
            .iter()
            .enumerate()
            .filter_map(|(index, tile)| tile.rect.intersects(visible).then_some(index))
            .collect();
        let index = SpatialIndex::from_tiles(&tiles, 256.0);

        assert_eq!(index.query_visible(visible), expected);
    }

    #[test]
    fn huge_tiles_and_viewports_use_bounded_fallbacks() {
        let mut index = SpatialIndex::new(1.0);
        index.insert(WorldRect::new(
            -1_000_000.0,
            -1_000_000.0,
            2_000_000.0,
            2_000_000.0,
        ));
        index.insert(WorldRect::new(2_000_000.0, 2_000_000.0, 10.0, 10.0));

        assert_eq!(
            index.query_visible(WorldRect::new(999_999.0, 999_999.0, 2.0, 2.0)),
            vec![0]
        );
        assert_eq!(
            index.query_visible(WorldRect::new(
                -10_000_000.0,
                -10_000_000.0,
                30_000_000.0,
                30_000_000.0,
            )),
            vec![0, 1]
        );
    }

    #[test]
    fn invalid_cell_size_falls_back_to_a_safe_default() {
        assert_eq!(SpatialIndex::new(0.0).cell_size(), DEFAULT_CELL_SIZE);
        assert_eq!(SpatialIndex::new(f32::NAN).cell_size(), DEFAULT_CELL_SIZE);
    }
}
