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

    // A real workbook with live formulas, so the in-app sheet editor has
    // something honest to open. Written next to the images.
    let sheet_path = images.join("budget.xlsx");
    {
        use rust_xlsxwriter::{Color, Format, FormatBorder, Formula, Workbook as XlsxWorkbook};
        let mut xlsx = XlsxWorkbook::new();
        let sheet = xlsx.add_worksheet();
        sheet.set_name("Budget").map_err(anyhow::Error::from)?;
        // Styled like a real budget sheet, so visual fidelity is visible:
        // coloured bold header, currency columns, a boxed bold total.
        let header_format = Format::new()
            .set_bold()
            .set_background_color(Color::RGB(0x4472C4))
            .set_font_color(Color::RGB(0xFFFFFF));
        let money = Format::new().set_num_format("$#,##0.00");
        let total_box = Format::new()
            .set_num_format("$#,##0.00")
            .set_bold()
            .set_border(FormatBorder::Thin);
        sheet.set_column_width(0, 18).map_err(anyhow::Error::from)?;
        for (column, header) in ["Item", "Price", "Count", "Total"].iter().enumerate() {
            sheet
                .write_string_with_format(0, column as u16, *header, &header_format)
                .map_err(anyhow::Error::from)?;
        }
        let rows: [(&str, f64, f64); 4] = [
            ("Espresso beans", 18.5, 2.0),
            ("Milk", 3.2, 6.0),
            ("Filters", 9.0, 1.0),
            ("Mugs", 12.0, 4.0),
        ];
        for (index, (item, price, count)) in rows.iter().enumerate() {
            let row = index as u32 + 1;
            sheet
                .write_string(row, 0, *item)
                .map_err(anyhow::Error::from)?;
            sheet
                .write_number_with_format(row, 1, *price, &money)
                .map_err(anyhow::Error::from)?;
            sheet
                .write_number(row, 2, *count)
                .map_err(anyhow::Error::from)?;
            sheet
                .write_formula_with_format(
                    row,
                    3,
                    Formula::new(format!("=B{r}*C{r}", r = row + 1)),
                    &money,
                )
                .map_err(anyhow::Error::from)?;
        }
        sheet
            .write_string(6, 0, "Grand total")
            .map_err(anyhow::Error::from)?;
        sheet
            .write_formula_with_format(6, 3, Formula::new("=SUM(D2:D5)"), &total_box)
            .map_err(anyhow::Error::from)?;
        xlsx.save(&sheet_path).map_err(anyhow::Error::from)?;
    }
    workspace.active_page_mut().add_tile(Tile::from_file(
        sheet_path,
        WorldRect::new(1_100.0, 1_500.0, 280.0, 200.0),
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
