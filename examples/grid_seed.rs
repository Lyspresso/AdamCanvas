//! Seeds a throwaway Adam library for exercising the grid view.
//!
//! Run with: cargo run --example grid_seed -- <data-dir> <image-dir>

use adam_canvas::{
    model::{Tile, Workspace, WorldRect},
    persistence::{AppPaths, save_workspace_atomic},
};
use std::{env, fs, path::PathBuf};

fn main() -> anyhow::Result<()> {
    let root = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("pass an Adam data directory"))?;
    let images = env::args_os()
        .nth(2)
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("pass a directory of images"))?;

    let mut sources: Vec<PathBuf> = fs::read_dir(&images)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("png"))
        })
        .collect();
    sources.sort();

    let mut workspace = Workspace::new();
    workspace.active_page_mut().name = "Grid view fixture".into();
    workspace.active_page_mut().set_size([3_600.0, 2_400.0]);

    // Deliberately scattered, deliberately mismatched sizes: the grid must
    // impose uniformity that the page itself does not have.
    let mut index = 0;
    for repeat in 0..3 {
        for source in &sources {
            let column = index % 6;
            let row = index / 6;
            let rect = WorldRect::new(
                80.0 + column as f32 * 300.0 + (repeat as f32 * 17.0),
                80.0 + row as f32 * 260.0,
                200.0 + (index % 4) as f32 * 40.0,
                150.0 + (index % 3) as f32 * 50.0,
            );
            workspace
                .active_page_mut()
                .add_tile(Tile::from_file(source.clone(), rect));
            index += 1;
        }
    }

    // A few non-photo tiles so the mixed-content cells are covered too.
    for (offset, (title, body)) in [
        (
            "Reading list",
            "# Reading\n- Contact sheets\n- Crop anchors",
        ),
        ("Scratch", "uncrop = uv lerp + rect lerp"),
    ]
    .into_iter()
    .enumerate()
    {
        let rect = WorldRect::new(120.0 + offset as f32 * 320.0, 1_500.0, 260.0, 180.0);
        workspace
            .active_page_mut()
            .add_tile(Tile::note(title, body, rect));
    }
    workspace.active_page_mut().add_tile(Tile::website(
        "Origin Kit",
        "https://www.originkit.dev/components/draggable-grid",
        WorldRect::new(760.0, 1_500.0, 260.0, 180.0),
    ));

    let paths = AppPaths::at(&root);
    fs::create_dir_all(&paths.root)?;
    save_workspace_atomic(&paths, &workspace)?;
    println!(
        "seeded {} tiles into {}",
        workspace.active_page().tiles.len(),
        paths.library.display()
    );
    Ok(())
}
