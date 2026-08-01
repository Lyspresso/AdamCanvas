//! Reads cell borders straight from an xlsx's XML.
//!
//! umya-spreadsheet 3.0.1 resolves fonts, fills and number formats correctly
//! but loses borders in parsing: the border colour is written into a clone
//! that is immediately dropped, and the edge style never survives either. So
//! borders — one of the visible things a spreadsheet's look is made of —
//! come from this small, focused parser instead: `styles.xml` for the border
//! table and the xf→border mapping, the sheet's own XML for each cell's xf.

use quick_xml::Reader;
use quick_xml::events::Event;
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

/// One cell's border edges, RGBA per present edge.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CellBorders {
    pub left: Option<[u8; 4]>,
    pub right: Option<[u8; 4]>,
    pub top: Option<[u8; 4]>,
    pub bottom: Option<[u8; 4]>,
}

impl CellBorders {
    pub fn any(&self) -> bool {
        self.left.is_some() || self.right.is_some() || self.top.is_some() || self.bottom.is_some()
    }
}

/// The colour Excel means by `<color auto="1"/>` — its automatic border
/// black, softened none.
const AUTO_EDGE: [u8; 4] = [66, 66, 66, 255];

/// Border edges per cell of one sheet, bounded to `rows` × `columns`.
/// `sheet_index` is the workbook-order index, matching the reader's.
pub fn load(
    path: &Path,
    sheet_index: usize,
    rows: usize,
    columns: usize,
) -> Option<HashMap<(usize, usize), CellBorders>> {
    let file = std::fs::File::open(path).ok()?;
    let mut archive = zip::ZipArchive::new(file).ok()?;

    let styles_xml = read_entry(&mut archive, "xl/styles.xml")?;
    let (border_table, xf_to_border) = parse_styles(&styles_xml)?;
    if border_table.iter().all(|borders| !borders.any()) {
        return Some(HashMap::new());
    }

    let sheet_path = sheet_file_for_index(&mut archive, sheet_index)?;
    let sheet_xml = read_entry(&mut archive, &sheet_path)?;
    Some(parse_sheet_cells(
        &sheet_xml,
        &border_table,
        &xf_to_border,
        rows,
        columns,
    ))
}

fn read_entry(archive: &mut zip::ZipArchive<std::fs::File>, name: &str) -> Option<String> {
    let mut entry = archive.by_name(name).ok()?;
    // styles.xml and sheet XML for capped sheets are small; a hard limit
    // keeps a hostile file from ballooning memory.
    const LIMIT: u64 = 64 * 1024 * 1024;
    if entry.size() > LIMIT {
        return None;
    }
    let mut content = String::new();
    entry.read_to_string(&mut content).ok()?;
    Some(content)
}

/// The workbook's Nth sheet file, via workbook.xml order and the rels map.
fn sheet_file_for_index(
    archive: &mut zip::ZipArchive<std::fs::File>,
    sheet_index: usize,
) -> Option<String> {
    let workbook = read_entry(archive, "xl/workbook.xml")?;
    let rels = read_entry(archive, "xl/_rels/workbook.xml.rels")?;

    // rels: Id -> Target
    let mut targets = HashMap::new();
    let mut reader = Reader::from_str(&rels);
    loop {
        match reader.read_event() {
            Ok(Event::Empty(e)) | Ok(Event::Start(e))
                if e.local_name().as_ref() == b"Relationship" =>
            {
                let mut id = None;
                let mut target = None;
                for attribute in e.attributes().flatten() {
                    match attribute.key.local_name().as_ref() {
                        b"Id" => {
                            id =
                                Some(String::from_utf8_lossy(attribute.value.as_ref()).into_owned())
                        }
                        b"Target" => {
                            target = Some(
                                String::from_utf8_lossy(attribute.value.as_ref()).into_owned(),
                            );
                        }
                        _ => {}
                    }
                }
                if let (Some(id), Some(target)) = (id, target) {
                    targets.insert(id, target);
                }
            }
            Ok(Event::Eof) => break,
            Err(_) => return None,
            _ => {}
        }
    }

    // workbook.xml: sheets in document order carry r:id.
    let mut reader = Reader::from_str(&workbook);
    let mut index = 0usize;
    loop {
        match reader.read_event() {
            Ok(Event::Empty(e)) | Ok(Event::Start(e)) if e.local_name().as_ref() == b"sheet" => {
                if index == sheet_index {
                    let rid = e.attributes().flatten().find_map(|attribute| {
                        (attribute.key.local_name().as_ref() == b"id")
                            .then(|| {
                                Some(String::from_utf8_lossy(attribute.value.as_ref()).into_owned())
                            })
                            .flatten()
                    })?;
                    let target = targets.get(&rid)?;
                    let target = target.trim_start_matches('/');
                    return Some(if target.starts_with("xl/") {
                        target.to_string()
                    } else {
                        format!("xl/{target}")
                    });
                }
                index += 1;
            }
            Ok(Event::Eof) => break,
            Err(_) => return None,
            _ => {}
        }
    }
    None
}

/// styles.xml → (border table, xf index → border table index).
fn parse_styles(xml: &str) -> Option<(Vec<CellBorders>, Vec<Option<usize>>)> {
    let mut reader = Reader::from_str(xml);
    let mut border_table = Vec::new();
    let mut xf_to_border = Vec::new();

    let mut in_borders = false;
    let mut in_cell_xfs = false;
    let mut current: Option<CellBorders> = None;
    let mut current_edge: Option<u8> = None;
    let mut current_edge_styled = false;

    loop {
        let event = match reader.read_event() {
            Ok(event) => event,
            Err(_) => return None,
        };
        match event {
            Event::Start(e) | Event::Empty(e) => {
                let name = e.local_name();
                match name.as_ref() {
                    b"borders" => in_borders = true,
                    b"cellXfs" => in_cell_xfs = true,
                    b"border" if in_borders => current = Some(CellBorders::default()),
                    b"left" | b"right" | b"top" | b"bottom" if current.is_some() => {
                        let styled = e.attributes().flatten().any(|attribute| {
                            attribute.key.local_name().as_ref() == b"style"
                                && attribute.value.as_ref() != b"none"
                        });
                        current_edge = Some(match name.as_ref() {
                            b"left" => 0,
                            b"right" => 1,
                            b"top" => 2,
                            _ => 3,
                        });
                        current_edge_styled = styled;
                        if styled {
                            // Colourless styled edge: auto colour until a
                            // nested <color> refines it.
                            set_edge(current.as_mut()?, current_edge?, Some(AUTO_EDGE));
                        }
                    }
                    // Non-edge children (diagonal and friends) end any open
                    // edge, or a styled self-closing edge's state would leak
                    // and their colour would land on the wrong side.
                    b"diagonal" | b"vertical" | b"horizontal" => {
                        current_edge = None;
                        current_edge_styled = false;
                    }
                    b"color" if current_edge.is_some() && current_edge_styled => {
                        let rgb = e.attributes().flatten().find_map(|attribute| {
                            (attribute.key.local_name().as_ref() == b"rgb")
                                .then(|| {
                                    Some(
                                        String::from_utf8_lossy(attribute.value.as_ref())
                                            .into_owned(),
                                    )
                                })
                                .flatten()
                        });
                        if let Some(rgb) = rgb
                            && let Some(color) = parse_argb(&rgb)
                        {
                            set_edge(current.as_mut()?, current_edge?, Some(color));
                        }
                    }
                    b"xf" if in_cell_xfs => {
                        let mut border_id = None;
                        let mut apply = None;
                        for attribute in e.attributes().flatten() {
                            match attribute.key.local_name().as_ref() {
                                b"borderId" => {
                                    border_id = std::str::from_utf8(attribute.value.as_ref())
                                        .ok()
                                        .and_then(|v| v.parse::<usize>().ok());
                                }
                                b"applyBorder" => {
                                    apply = Some(
                                        attribute.value.as_ref() == b"1"
                                            || attribute.value.as_ref() == b"true",
                                    );
                                }
                                _ => {}
                            }
                        }
                        // Excel treats a nonzero borderId as meaningful even
                        // without applyBorder; an explicit applyBorder="0"
                        // switches it off.
                        let effective = match (border_id, apply) {
                            (Some(0), _) | (None, _) => None,
                            (Some(id), Some(false)) => {
                                let _ = id;
                                None
                            }
                            (Some(id), _) => Some(id),
                        };
                        xf_to_border.push(effective);
                    }
                    _ => {}
                }
            }
            Event::End(e) => match e.local_name().as_ref() {
                b"borders" => in_borders = false,
                b"cellXfs" => in_cell_xfs = false,
                b"border" => {
                    if let Some(done) = current.take() {
                        border_table.push(done);
                    }
                }
                b"left" | b"right" | b"top" | b"bottom" => {
                    current_edge = None;
                    current_edge_styled = false;
                }
                _ => {}
            },
            Event::Eof => break,
            _ => {}
        }
    }
    Some((border_table, xf_to_border))
}

fn set_edge(borders: &mut CellBorders, edge: u8, color: Option<[u8; 4]>) {
    match edge {
        0 => borders.left = color,
        1 => borders.right = color,
        2 => borders.top = color,
        _ => borders.bottom = color,
    }
}

/// The sheet's cells: `<c r="B3" s="5">` gives coordinate and xf index.
fn parse_sheet_cells(
    xml: &str,
    border_table: &[CellBorders],
    xf_to_border: &[Option<usize>],
    rows: usize,
    columns: usize,
) -> HashMap<(usize, usize), CellBorders> {
    let mut out = HashMap::new();
    let mut reader = Reader::from_str(xml);
    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) | Ok(Event::Empty(e)) if e.local_name().as_ref() == b"c" => {
                let mut reference = None;
                let mut xf = None;
                for attribute in e.attributes().flatten() {
                    match attribute.key.local_name().as_ref() {
                        b"r" => {
                            reference = Some(
                                String::from_utf8_lossy(attribute.value.as_ref()).into_owned(),
                            );
                        }
                        b"s" => {
                            xf = std::str::from_utf8(attribute.value.as_ref())
                                .ok()
                                .and_then(|value| value.parse::<usize>().ok());
                        }
                        _ => {}
                    }
                }
                let (Some(reference), Some(xf)) = (reference, xf) else {
                    continue;
                };
                let Some(border_index) = xf_to_border.get(xf).copied().flatten() else {
                    continue;
                };
                let Some(borders) = border_table.get(border_index) else {
                    continue;
                };
                if !borders.any() {
                    continue;
                }
                let Some((row, column)) = parse_reference(&reference) else {
                    continue;
                };
                if row < rows && column < columns {
                    out.insert((row, column), *borders);
                }
            }
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
    }
    out
}

/// "B3" → (2, 1), zero-based (row, column).
fn parse_reference(reference: &str) -> Option<(usize, usize)> {
    let split = reference.find(|c: char| c.is_ascii_digit())?;
    let (letters, digits) = reference.split_at(split);
    let mut column = 0usize;
    for c in letters.chars() {
        let c = c.to_ascii_uppercase();
        if !c.is_ascii_uppercase() {
            return None;
        }
        column = column * 26 + (c as usize - 'A' as usize + 1);
    }
    let row: usize = digits.parse().ok()?;
    if column == 0 || row == 0 {
        return None;
    }
    Some((row - 1, column - 1))
}

fn parse_argb(argb: &str) -> Option<[u8; 4]> {
    let hex = argb.trim();
    let (alpha, rgb) = match hex.len() {
        8 => (u8::from_str_radix(&hex[0..2], 16).ok()?, &hex[2..]),
        6 => (255, hex),
        _ => return None,
    };
    if alpha == 0 {
        return None;
    }
    Some([
        u8::from_str_radix(&rgb[0..2], 16).ok()?,
        u8::from_str_radix(&rgb[2..4], 16).ok()?,
        u8::from_str_radix(&rgb[4..6], 16).ok()?,
        alpha,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_xlsxwriter::{Color, Format, FormatBorder, Workbook as XlsxWorkbook};

    #[test]
    fn cell_references_parse() {
        assert_eq!(parse_reference("A1"), Some((0, 0)));
        assert_eq!(parse_reference("B3"), Some((2, 1)));
        assert_eq!(parse_reference("AA100"), Some((99, 26)));
        assert_eq!(parse_reference(""), None);
        assert_eq!(parse_reference("123"), None);
    }

    #[test]
    fn borders_read_from_a_real_file_with_auto_and_explicit_colours() {
        let directory = tempfile::tempdir().expect("tempdir");
        let path = directory.path().join("borders.xlsx");
        let mut workbook = XlsxWorkbook::new();
        let sheet = workbook.add_worksheet();
        // Auto-coloured thin box, the Excel default look.
        let auto_box = Format::new().set_border(FormatBorder::Thin);
        // Explicit red bottom edge only.
        let red_bottom = Format::new()
            .set_border_bottom(FormatBorder::Thin)
            .set_border_bottom_color(Color::RGB(0xCC0000));
        sheet
            .write_string_with_format(2, 0, "boxed", &auto_box)
            .unwrap();
        sheet
            .write_string_with_format(4, 1, "underlined", &red_bottom)
            .unwrap();
        sheet.write_string(0, 0, "plain").unwrap();
        workbook.save(&path).expect("save");

        let borders = load(&path, 0, 10, 4).expect("borders parse");
        let boxed = borders.get(&(2, 0)).expect("boxed cell has borders");
        assert!(boxed.left.is_some() && boxed.right.is_some());
        assert!(boxed.top.is_some() && boxed.bottom.is_some());
        assert_eq!(
            boxed.left,
            Some(AUTO_EDGE),
            "auto colour maps to the default edge"
        );

        let underlined = borders.get(&(4, 1)).expect("underlined cell");
        assert_eq!(underlined.bottom, Some([0xCC, 0x00, 0x00, 255]));
        assert!(underlined.top.is_none() && underlined.left.is_none());

        assert!(!borders.contains_key(&(0, 0)), "plain cells carry nothing");
    }

    #[test]
    fn a_diagonal_colour_cannot_leak_onto_a_self_closed_edge() {
        // A styled self-closing edge followed by a coloured diagonal — the
        // diagonal's colour must not overwrite the edge's auto colour.
        let styles = r#"<styleSheet><borders count="1"><border><left/><right/><top style="thin"/><bottom/><diagonal style="thin"><color rgb="FFFF0000"/></diagonal></border></borders><cellXfs count="1"><xf borderId="0" applyBorder="1"/></cellXfs></styleSheet>"#;
        let (table, _) = parse_styles(styles).expect("parse");
        assert_eq!(table.len(), 1);
        assert_eq!(
            table[0].top,
            Some(AUTO_EDGE),
            "the top edge keeps its auto colour, not the diagonal's red"
        );
    }

    #[test]
    fn a_borderless_workbook_yields_an_empty_map_quickly() {
        let directory = tempfile::tempdir().expect("tempdir");
        let path = directory.path().join("plain.xlsx");
        let mut workbook = XlsxWorkbook::new();
        workbook.add_worksheet().write_string(0, 0, "x").unwrap();
        workbook.save(&path).expect("save");
        let borders = load(&path, 0, 5, 5).expect("parse");
        assert!(borders.is_empty());
    }

    #[test]
    fn missing_files_and_wild_indices_are_none_not_panics() {
        assert!(load(Path::new("/nope.xlsx"), 0, 5, 5).is_none());

        // A borderless workbook short-circuits to an empty map before the
        // sheet index matters — any index is a valid "nothing here".
        let directory = tempfile::tempdir().expect("tempdir");
        let plain = directory.path().join("one.xlsx");
        let mut workbook = XlsxWorkbook::new();
        workbook.add_worksheet().write_string(0, 0, "x").unwrap();
        workbook.save(&plain).expect("save");
        assert_eq!(load(&plain, 7, 5, 5), Some(HashMap::new()));

        // With real borders present, a wild sheet index is a real failure.
        let bordered = directory.path().join("two.xlsx");
        let mut workbook = XlsxWorkbook::new();
        let boxed = Format::new().set_border(FormatBorder::Thin);
        workbook
            .add_worksheet()
            .write_string_with_format(0, 0, "x", &boxed)
            .unwrap();
        workbook.save(&bordered).expect("save");
        assert!(load(&bordered, 7, 5, 5).is_none(), "no such sheet index");
        assert!(load(&bordered, 0, 5, 5).is_some());
    }
}
