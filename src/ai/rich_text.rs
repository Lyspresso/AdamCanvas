//! Small, defensive markdown segmenter for assistant prose.
//!
//! Adam renders user messages literally. Assistant text is segmented here so
//! code and tables can receive dedicated egui cards without trusting a full
//! HTML/markdown renderer. Malformed input always remains visible.

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RichBlock {
    pub id: u64,
    pub kind: RichBlockKind,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RichBlockKind {
    Paragraph(String),
    Heading {
        text: String,
        level: u8,
    },
    Code {
        language: Option<String>,
        code: String,
    },
    Table(String),
    Rule,
}

pub fn segment_assistant_markdown(source: &str) -> Vec<RichBlock> {
    if source.is_empty() {
        return Vec::new();
    }

    let normalized = source.replace("\r\n", "\n").replace('\r', "\n");
    let lines: Vec<&str> = normalized.split('\n').collect();
    let mut blocks = Vec::new();
    let mut paragraph = Vec::<&str>::new();
    let mut index = 0usize;

    let flush_paragraph = |paragraph: &mut Vec<&str>, blocks: &mut Vec<RichBlock>| {
        if paragraph.is_empty() {
            return;
        }
        let text = paragraph.join("\n");
        push_block(blocks, RichBlockKind::Paragraph(text));
        paragraph.clear();
    };

    while index < lines.len() {
        let line = lines[index];
        let trimmed = line.trim();

        if let Some(fence) = parse_fence(trimmed) {
            flush_paragraph(&mut paragraph, &mut blocks);
            let mut code_lines = Vec::new();
            index += 1;
            while index < lines.len() && !is_matching_fence(lines[index].trim(), fence.marker) {
                code_lines.push(lines[index]);
                index += 1;
            }
            // An unterminated fence deliberately consumes through EOF.
            if index < lines.len() {
                index += 1;
            }
            push_block(
                &mut blocks,
                RichBlockKind::Code {
                    language: fence.language,
                    code: code_lines.join("\n"),
                },
            );
            continue;
        }

        if let Some((level, heading)) = parse_heading(trimmed) {
            flush_paragraph(&mut paragraph, &mut blocks);
            push_block(
                &mut blocks,
                RichBlockKind::Heading {
                    text: heading.to_owned(),
                    level,
                },
            );
            index += 1;
            continue;
        }

        if is_rule(trimmed) {
            flush_paragraph(&mut paragraph, &mut blocks);
            push_block(&mut blocks, RichBlockKind::Rule);
            index += 1;
            continue;
        }

        if looks_like_table_header(&lines, index) {
            flush_paragraph(&mut paragraph, &mut blocks);
            let start = index;
            index += 2;
            while index < lines.len()
                && lines[index].contains('|')
                && !lines[index].trim().is_empty()
            {
                index += 1;
            }
            push_block(
                &mut blocks,
                RichBlockKind::Table(lines[start..index].join("\n")),
            );
            continue;
        }

        if trimmed.is_empty() {
            flush_paragraph(&mut paragraph, &mut blocks);
        } else {
            paragraph.push(line);
        }
        index += 1;
    }
    flush_paragraph(&mut paragraph, &mut blocks);
    blocks
}

struct Fence {
    marker: char,
    language: Option<String>,
}

fn parse_fence(line: &str) -> Option<Fence> {
    let marker = line.chars().next()?;
    if marker != '`' && marker != '~' {
        return None;
    }
    let marker_count = line.chars().take_while(|value| *value == marker).count();
    if marker_count < 3 {
        return None;
    }
    let language = line[marker_count..].trim();
    Some(Fence {
        marker,
        language: (!language.is_empty()).then(|| language.to_owned()),
    })
}

fn is_matching_fence(line: &str, marker: char) -> bool {
    line.chars().take_while(|value| *value == marker).count() >= 3
        && line
            .chars()
            .all(|value| value == marker || value.is_whitespace())
}

fn parse_heading(line: &str) -> Option<(u8, &str)> {
    let level = line.chars().take_while(|value| *value == '#').count();
    if !(1..=6).contains(&level) {
        return None;
    }
    let remainder = line.get(level..)?;
    remainder
        .strip_prefix(char::is_whitespace)
        .map(|heading| (level as u8, heading.trim()))
}

fn is_rule(line: &str) -> bool {
    let compact: String = line
        .chars()
        .filter(|value| !value.is_whitespace())
        .collect();
    compact.len() >= 3
        && compact
            .chars()
            .next()
            .is_some_and(|first| matches!(first, '-' | '*' | '_'))
        && compact
            .chars()
            .all(|value| value == compact.chars().next().unwrap())
}

fn looks_like_table_header(lines: &[&str], index: usize) -> bool {
    let Some(separator) = lines.get(index + 1).copied() else {
        return false;
    };
    if !lines[index].contains('|') || !separator.contains('|') {
        return false;
    }
    let mut cells = 0usize;
    for cell in separator.trim().trim_matches('|').split('|') {
        let cell = cell.trim().trim_matches(':').trim();
        if cell.len() < 3 || !cell.chars().all(|value| value == '-') {
            return false;
        }
        cells += 1;
    }
    cells > 0
}

fn push_block(blocks: &mut Vec<RichBlock>, kind: RichBlockKind) {
    let discriminant = match &kind {
        RichBlockKind::Paragraph(_) => "paragraph",
        RichBlockKind::Heading { .. } => "heading",
        RichBlockKind::Code { .. } => "code",
        RichBlockKind::Table(_) => "table",
        RichBlockKind::Rule => "rule",
    };
    let content = match &kind {
        RichBlockKind::Paragraph(text) | RichBlockKind::Table(text) => text.as_str(),
        RichBlockKind::Heading { text, .. } => text.as_str(),
        RichBlockKind::Code { code, .. } => code.as_str(),
        RichBlockKind::Rule => "---",
    };
    let id = stable_block_id(discriminant, content, blocks.len());
    blocks.push(RichBlock { id, kind });
}

fn stable_block_id(kind: &str, content: &str, ordinal: usize) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in kind
        .as_bytes()
        .iter()
        .chain([0xff].iter())
        .chain(content.as_bytes())
        .chain([0xfe].iter())
        .chain(ordinal.to_string().as_bytes())
    {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segments_fenced_code_and_unterminated_fence() {
        let parsed = segment_assistant_markdown("Before\n\n```rust\nfn main() {}\n```\n\nAfter");
        assert!(matches!(
            &parsed[1].kind,
            RichBlockKind::Code {
                language: Some(language),
                code
            } if language == "rust" && code == "fn main() {}"
        ));

        let unterminated = segment_assistant_markdown("~~~sh\necho hello");
        assert!(matches!(
            &unterminated[0].kind,
            RichBlockKind::Code { code, .. } if code == "echo hello"
        ));
    }

    #[test]
    fn table_is_preserved_as_monospace_source() {
        let source = "| A | B |\n|---|:---:|\n| 1 | 2 |";
        let parsed = segment_assistant_markdown(source);
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].kind, RichBlockKind::Table(source.into()));
    }

    #[test]
    fn malformed_markdown_falls_back_to_literal_paragraph() {
        let source = "##no-space\n`unterminated inline";
        let parsed = segment_assistant_markdown(source);
        assert_eq!(parsed[0].kind, RichBlockKind::Paragraph(source.to_owned()));
    }

    #[test]
    fn ids_are_content_stable() {
        let first = segment_assistant_markdown("# Hello\n\nWorld");
        let second = segment_assistant_markdown("# Hello\n\nWorld");
        assert_eq!(
            first.iter().map(|block| block.id).collect::<Vec<_>>(),
            second.iter().map(|block| block.id).collect::<Vec<_>>()
        );
    }
}
