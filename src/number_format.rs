//! Renders numbers the way the spreadsheet's own number formats would.
//!
//! Excel cells carry a format string ("$#,##0.00", "0.0%", "m/d/yy"); the
//! displayed text is the value passed through it. Fetching Excel's displayed
//! string per cell costs ~8ms of Apple Events each — hopeless for a whole
//! sheet — while the format string is one cheap fetch per column. So Adam
//! renders the common formats itself, at fixed cost for any sheet size.
//!
//! This is deliberately a subset: the formats people actually use. Anything
//! unrecognised returns `None` and the caller falls back to the plain
//! rendering — wrong-looking is worse than plain-looking.

use crate::spreadsheet::format_number;

/// Applies `format` to a numeric value. `None` means "no opinion — use the
/// default rendering", either because the format is General or because it is
/// beyond the subset.
pub fn format_value(value: f64, format: &str) -> Option<String> {
    if !value.is_finite() {
        return None;
    }
    // Sections are positive;negative;zero;text — the positive section carries
    // the shape. Color/condition prefixes like [Red] are presentation Adam
    // does not do yet.
    let section = format.split(';').next().unwrap_or(format);
    let section = strip_brackets(section);
    let section = section.trim();
    if section.is_empty() || section.eq_ignore_ascii_case("general") {
        return None;
    }
    if section == "@" {
        return Some(format_number(value));
    }
    // Fractions ("# ?/?") and scientific ("0.00E+00") are out of subset.
    if section.contains('?') || section.to_ascii_uppercase().contains('E') {
        return None;
    }
    if looks_like_date(section) {
        return format_date(value, section);
    }

    let percent = section.contains('%');
    let scaled = if percent { value * 100.0 } else { value };

    // Currency and affixes: everything before the first digit placeholder is
    // a prefix, everything after the last is a suffix.
    let first = section.find(|c: char| "#0".contains(c))?;
    let last = section.rfind(|c: char| "#0".contains(c))?;
    let digits = &section[first..=last];
    let prefix = clean_affix(&section[..first]);
    let suffix = clean_affix(&section[last + 1..]);

    let decimals = digits
        .split_once('.')
        .map(|(_, fraction)| fraction.chars().filter(|c| "0#".contains(*c)).count())
        .unwrap_or(0);
    let thousands = digits.contains(',');

    let negative = scaled < 0.0;
    // Excel rounds halves away from zero; Rust's formatter rounds halves to
    // even, which would print 18.5 through "0" as 18.
    let factor = 10f64.powi(decimals as i32);
    let magnitude = (scaled.abs() * factor).round() / factor;
    let mut body = format!("{magnitude:.decimals$}");
    if thousands {
        body = group_thousands(&body);
    }
    let sign = if negative { "-" } else { "" };
    Some(format!("{sign}{prefix}{body}{suffix}"))
}

/// `[$-409]`, `[Red]`, `[$€-407]` bracket blocks: currency blocks contribute
/// their symbol, the rest vanish.
fn strip_brackets(section: &str) -> String {
    let mut out = String::with_capacity(section.len());
    let mut rest = section;
    while let Some(start) = rest.find('[') {
        out.push_str(&rest[..start]);
        let Some(end) = rest[start..].find(']') else {
            out.push_str(&rest[start..]);
            return out;
        };
        let block = &rest[start + 1..start + end];
        if let Some(symbol) = block.strip_prefix('$') {
            // "[$€-407]" — symbol before the locale dash.
            let symbol = symbol.split('-').next().unwrap_or("");
            out.push_str(symbol);
        }
        rest = &rest[start + end + 1..];
    }
    out.push_str(rest);
    out
}

/// Affixes keep currency symbols and literal text, dropping the formatting
/// machinery (quotes, underscores+skip-char, asterisk fills, spaces).
fn clean_affix(raw: &str) -> String {
    let mut out = String::new();
    let mut chars = raw.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' => {
                for inner in chars.by_ref() {
                    if inner == '"' {
                        break;
                    }
                    out.push(inner);
                }
            }
            // "_x" leaves the width of x; "*x" repeats x as fill. Both are
            // spacing tricks, not content.
            '_' | '*' => {
                chars.next();
            }
            '\\' => {
                if let Some(literal) = chars.next() {
                    out.push(literal);
                }
            }
            ' ' | ',' | '#' | '0' | '.' | '%' => {
                if c == '%' {
                    out.push('%');
                }
            }
            other => out.push(other),
        }
    }
    out
}

fn group_thousands(body: &str) -> String {
    let (integer, fraction) = body.split_once('.').unwrap_or((body, ""));
    let mut grouped = String::new();
    let digits: Vec<char> = integer.chars().collect();
    for (index, digit) in digits.iter().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            grouped.push(',');
        }
        grouped.push(*digit);
    }
    if fraction.is_empty() {
        grouped
    } else {
        format!("{grouped}.{fraction}")
    }
}

/// A format is a date format when it uses date tokens and no digit
/// placeholders — "0.00" is not a date, "m/d/yy" is.
fn looks_like_date(section: &str) -> bool {
    if section.contains(['#', '0']) {
        return false;
    }
    let lower = section.to_ascii_lowercase();
    ["y", "d", "h", "s"]
        .iter()
        .any(|token| lower.contains(*token))
        || lower.contains('m')
}

/// Renders an Excel date serial through a date-shaped format. Field order is
/// honoured; exotic tokens fall back rather than mis-render.
fn format_date(serial: f64, section: &str) -> Option<String> {
    if serial < 0.0 {
        return None;
    }
    let (year, month, day) = crate::spreadsheet::civil_from_serial(serial)?;
    let lower = section.to_ascii_lowercase();

    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let mut out = String::new();
    let mut chars = lower.chars().peekable();
    while let Some(c) = chars.next() {
        let mut run = 1;
        while chars.peek() == Some(&c) {
            chars.next();
            run += 1;
        }
        match c {
            'y' => {
                if run >= 4 {
                    out.push_str(&format!("{year:04}"));
                } else {
                    out.push_str(&format!("{:02}", year % 100));
                }
            }
            'm' => match run {
                1 => out.push_str(&month.to_string()),
                2 => out.push_str(&format!("{month:02}")),
                3 => out.push_str(MONTHS.get(month as usize - 1)?),
                _ => return None,
            },
            'd' => match run {
                1 => out.push_str(&day.to_string()),
                2 => out.push_str(&format!("{day:02}")),
                _ => return None,
            },
            '/' | '-' | '.' | ' ' | ',' => out.push(c),
            // Times, weekday names, elapsed hours: out of subset.
            _ => return None,
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn general_and_unknown_formats_have_no_opinion() {
        assert_eq!(format_value(18.5, "General"), None);
        assert_eq!(format_value(18.5, ""), None);
        assert_eq!(format_value(f64::NAN, "0.00"), None);
        // Exotic: fraction format — out of subset, fall back rather than lie.
        assert_eq!(format_value(0.5, "# ?/?"), None);
    }

    #[test]
    fn plain_decimal_formats_render_fixed_places() {
        assert_eq!(format_value(18.5, "0.00").as_deref(), Some("18.50"));
        assert_eq!(format_value(18.5, "0").as_deref(), Some("19"));
        assert_eq!(format_value(-3.456, "0.0").as_deref(), Some("-3.5"));
        assert_eq!(format_value(0.125, "0.000").as_deref(), Some("0.125"));
    }

    #[test]
    fn thousands_and_currency_render_like_excel() {
        assert_eq!(
            format_value(1234567.891, "#,##0.00").as_deref(),
            Some("1,234,567.89")
        );
        assert_eq!(
            format_value(1234567.0, "#,##0").as_deref(),
            Some("1,234,567")
        );
        assert_eq!(format_value(18.5, "$#,##0.00").as_deref(), Some("$18.50"));
        assert_eq!(format_value(-18.5, "$#,##0.00").as_deref(), Some("-$18.50"));
        // The classic accounting-style format reduces to its essentials.
        assert_eq!(
            format_value(1234.5, r#"_("$"* #,##0.00_)"#).as_deref(),
            Some("$1,234.50")
        );
        // Locale currency blocks keep the symbol.
        assert_eq!(
            format_value(9.99, "[$€-407]#,##0.00").as_deref(),
            Some("€9.99")
        );
        assert_eq!(
            format_value(500.0, "0.00\" kg\"").as_deref(),
            Some("500.00 kg")
        );
    }

    #[test]
    fn percent_formats_scale_by_one_hundred() {
        assert_eq!(format_value(0.125, "0.0%").as_deref(), Some("12.5%"));
        assert_eq!(format_value(0.125, "0%").as_deref(), Some("13%"));
        assert_eq!(format_value(1.0, "0.00%").as_deref(), Some("100.00%"));
    }

    #[test]
    fn date_formats_render_from_serials() {
        // 45000 = 2023-03-15, verified against the reader's own tests.
        assert_eq!(format_value(45_000.0, "m/d/yy").as_deref(), Some("3/15/23"));
        assert_eq!(
            format_value(45_000.0, "mm/dd/yyyy").as_deref(),
            Some("03/15/2023")
        );
        assert_eq!(
            format_value(45_000.0, "d-mmm-yy").as_deref(),
            Some("15-Mar-23")
        );
        assert_eq!(
            format_value(45_000.0, "yyyy-mm-dd").as_deref(),
            Some("2023-03-15")
        );
        // Time-bearing formats are out of subset for now — no opinion, and
        // definitely no wrong opinion.
        assert_eq!(format_value(45_000.5, "m/d/yy h:mm"), None);
    }

    #[test]
    fn negative_sections_do_not_confuse_the_positive_shape() {
        assert_eq!(
            format_value(1234.5, "#,##0.00;(#,##0.00)").as_deref(),
            Some("1,234.50")
        );
        assert_eq!(
            format_value(-1234.5, "#,##0.00;(#,##0.00)").as_deref(),
            Some("-1,234.50"),
            "the subset renders negatives with a minus, not parentheses — \
             close and honest beats parsing every section"
        );
    }
}
