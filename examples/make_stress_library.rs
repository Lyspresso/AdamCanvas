use adam_canvas::{
    domain::{PaletteColor, Pile, UnixMillis},
    model::{Tile, Workspace, WorldRect},
    persistence::{AppPaths, save_workspace_atomic},
};
use std::{env, fs, path::PathBuf};

fn main() -> anyhow::Result<()> {
    let root = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("pass an Adam data directory"))?;
    let sources = root.join("stress-sources");
    fs::create_dir_all(&sources)?;

    let mut workspace = Workspace::new();
    workspace.active_page_mut().name = "100-item stress canvas".into();
    workspace.active_page_mut().set_size([4_096.0, 3_200.0]);

    for index in 0..100 {
        let column = index % 10;
        let row = index / 10;
        let rect = WorldRect::new(
            64.0 + column as f32 * 304.0,
            64.0 + row as f32 * 214.0,
            280.0,
            190.0,
        );
        let tile = if index == 1 {
            let pile_id = uuid::Uuid::new_v4();
            let tag_id = workspace.domain.tags.ensure_tag(
                uuid::Uuid::new_v4(),
                "Stress pile",
                PaletteColor::Teal,
                UnixMillis(0),
            )?;
            workspace.domain.piles.insert(
                pile_id,
                Pile::new(
                    pile_id,
                    workspace.active_page,
                    rect,
                    "Stress pile",
                    tag_id,
                    PaletteColor::Teal,
                )?,
            );
            Tile::pile(pile_id, "Stress pile", rect)
        } else {
            match index % 5 {
                0 => Tile::note(
                    format!("Note {}", index + 1),
                    format!(
                        "# Item {}\n- [ ] Review\nLocal-first stress fixture",
                        index + 1
                    ),
                    rect,
                ),
                1 => Tile::website(
                    format!("Website {}", index + 1),
                    format!("https://example.com/item/{}", index + 1),
                    rect,
                ),
                2 => {
                    let path = sources.join(format!("document-{index}.txt"));
                    fs::write(&path, format!("Adam stress document {index}\n").repeat(8))?;
                    Tile::from_file(path, rect)
                }
                3 => {
                    let path = sources.join(format!("sheet-{index}.csv"));
                    fs::write(
                        &path,
                        "month,total,status\nJuly,120,Ready\nAugust,140,Planned\n",
                    )?;
                    Tile::from_file(path, rect)
                }
                _ => {
                    let path = sources.join(format!("image-{index}.png"));
                    let mut rgba = vec![0_u8; 64 * 64 * 4];
                    for pixel in rgba.chunks_exact_mut(4) {
                        pixel.copy_from_slice(&[
                            40 + index as u8,
                            110,
                            210_u8.saturating_sub(index as u8),
                            255,
                        ]);
                    }
                    image::save_buffer(&path, &rgba, 64, 64, image::ColorType::Rgba8)?;
                    Tile::from_file(path, rect)
                }
            }
        };
        workspace.active_page_mut().add_tile(tile);
    }

    let paths = AppPaths::at(root);
    paths.ensure()?;
    save_workspace_atomic(&paths, &workspace)?;
    Ok(())
}
