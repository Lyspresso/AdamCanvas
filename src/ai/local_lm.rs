//! Optional local-LM seam for titles, replay compaction, and memory synthesis.
//!
//! Adam never blocks sending on this service. The caller runs `complete` on a
//! worker thread and treats every error as feature-unavailable. Only an HTTP
//! loopback endpoint is accepted so these supporting summaries cannot
//! accidentally send workspace history to a remote host.

use std::{
    error::Error,
    fmt,
    io::{Read, Write},
    net::{IpAddr, SocketAddr, TcpStream, ToSocketAddrs},
    time::Duration,
};

use serde_json::{Value as JsonValue, json};
use url::Url;

use super::{
    prompt::{PromptHistoryTurn, PromptTurnRole, stable_digest, truncate_utf8_visible},
    store::StoredTurn,
};

pub const DEFAULT_LOCAL_LM_ENDPOINT: &str = "http://127.0.0.1:1234/v1/chat/completions";
pub const LOCAL_LM_RESPONSE_CAP: usize = 1_048_576;
pub const COMPACTION_SUMMARY_LIMIT: usize = 4_000;
pub const COMPACTION_DELTA_LIMIT: usize = 16_000;
pub const COMPACTION_TURN_LIMIT: usize = 2_000;
pub const MAX_COMPACTION_CALLS: usize = 3;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LocalLmConfig {
    pub endpoint: String,
    pub model: String,
}

impl Default for LocalLmConfig {
    fn default() -> Self {
        Self {
            endpoint: DEFAULT_LOCAL_LM_ENDPOINT.into(),
            model: "local-model".into(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LocalLmError {
    UnsafeEndpoint,
    InvalidEndpoint,
    Unavailable,
    TimedOut,
    InvalidResponse,
    ResponseTooLarge,
}

impl fmt::Display for LocalLmError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::UnsafeEndpoint => "local inference endpoint must be HTTP loopback",
            Self::InvalidEndpoint => "local inference endpoint is invalid",
            Self::Unavailable => "local inference is unavailable",
            Self::TimedOut => "local inference timed out",
            Self::InvalidResponse => "local inference returned an invalid response",
            Self::ResponseTooLarge => "local inference response exceeded its limit",
        })
    }
}

impl Error for LocalLmError {}

#[derive(Clone, Debug)]
pub struct LocalLmClient {
    config: LocalLmConfig,
}

impl LocalLmClient {
    pub fn new(config: LocalLmConfig) -> Result<Self, LocalLmError> {
        parse_endpoint(&config.endpoint)?;
        if config.model.trim().is_empty() {
            return Err(LocalLmError::InvalidEndpoint);
        }
        Ok(Self { config })
    }

    pub fn config(&self) -> &LocalLmConfig {
        &self.config
    }

    pub fn complete(
        &self,
        system: &str,
        user: &str,
        timeout: Duration,
    ) -> Result<String, LocalLmError> {
        if timeout.is_zero() {
            return Err(LocalLmError::TimedOut);
        }
        let endpoint = parse_endpoint(&self.config.endpoint)?;
        let address = resolve_loopback(&endpoint)?;
        let mut stream = TcpStream::connect_timeout(&address, timeout).map_err(map_io_error)?;
        stream
            .set_read_timeout(Some(timeout))
            .map_err(map_io_error)?;
        stream
            .set_write_timeout(Some(timeout))
            .map_err(map_io_error)?;

        let payload = serde_json::to_vec(&json!({
            "model": self.config.model,
            "stream": false,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ]
        }))
        .map_err(|_| LocalLmError::InvalidResponse)?;
        let host = endpoint.host_str().ok_or(LocalLmError::InvalidEndpoint)?;
        let port = endpoint
            .port_or_known_default()
            .ok_or(LocalLmError::InvalidEndpoint)?;
        let path = match endpoint.query() {
            Some(query) => format!("{}?{query}", endpoint.path()),
            None => endpoint.path().to_owned(),
        };
        write!(
            stream,
            "POST {path} HTTP/1.1\r\nHost: {host}:{port}\r\nContent-Type: application/json\r\nAccept: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            payload.len()
        )
        .map_err(map_io_error)?;
        stream.write_all(&payload).map_err(map_io_error)?;
        stream.flush().map_err(map_io_error)?;

        let mut response = Vec::new();
        stream
            .take((LOCAL_LM_RESPONSE_CAP + 1) as u64)
            .read_to_end(&mut response)
            .map_err(map_io_error)?;
        if response.len() > LOCAL_LM_RESPONSE_CAP {
            return Err(LocalLmError::ResponseTooLarge);
        }
        let body = decode_http_response(&response)?;
        let value: JsonValue =
            serde_json::from_slice(&body).map_err(|_| LocalLmError::InvalidResponse)?;
        extract_completion_text(&value)
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(str::to_owned)
            .ok_or(LocalLmError::InvalidResponse)
    }
}

fn parse_endpoint(value: &str) -> Result<Url, LocalLmError> {
    let url = Url::parse(value).map_err(|_| LocalLmError::InvalidEndpoint)?;
    if url.scheme() != "http"
        || !url.username().is_empty()
        || url.password().is_some()
        || url.fragment().is_some()
    {
        return Err(LocalLmError::UnsafeEndpoint);
    }
    let host = url.host_str().ok_or(LocalLmError::InvalidEndpoint)?;
    let ip: IpAddr = host.parse().map_err(|_| LocalLmError::UnsafeEndpoint)?;
    if !ip.is_loopback() {
        return Err(LocalLmError::UnsafeEndpoint);
    }
    Ok(url)
}

fn resolve_loopback(endpoint: &Url) -> Result<SocketAddr, LocalLmError> {
    let host = endpoint.host_str().ok_or(LocalLmError::InvalidEndpoint)?;
    let port = endpoint
        .port_or_known_default()
        .ok_or(LocalLmError::InvalidEndpoint)?;
    (host, port)
        .to_socket_addrs()
        .map_err(|_| LocalLmError::Unavailable)?
        .find(|address| address.ip().is_loopback())
        .ok_or(LocalLmError::UnsafeEndpoint)
}

fn map_io_error(error: std::io::Error) -> LocalLmError {
    if matches!(
        error.kind(),
        std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
    ) {
        LocalLmError::TimedOut
    } else {
        LocalLmError::Unavailable
    }
}

fn decode_http_response(response: &[u8]) -> Result<Vec<u8>, LocalLmError> {
    let header_end = find_bytes(response, b"\r\n\r\n").ok_or(LocalLmError::InvalidResponse)?;
    let head =
        std::str::from_utf8(&response[..header_end]).map_err(|_| LocalLmError::InvalidResponse)?;
    let mut lines = head.split("\r\n");
    let status = lines.next().ok_or(LocalLmError::InvalidResponse)?;
    let status_code = status
        .split_whitespace()
        .nth(1)
        .and_then(|value| value.parse::<u16>().ok())
        .ok_or(LocalLmError::InvalidResponse)?;
    if status_code != 200 {
        return Err(LocalLmError::Unavailable);
    }
    let mut content_length = None;
    let mut chunked = false;
    for line in lines {
        let Some((name, value)) = line.split_once(':') else {
            return Err(LocalLmError::InvalidResponse);
        };
        match name.trim().to_ascii_lowercase().as_str() {
            "content-length" => {
                content_length = value.trim().parse::<usize>().ok();
            }
            "transfer-encoding" if value.trim().eq_ignore_ascii_case("chunked") => {
                chunked = true;
            }
            _ => {}
        }
    }
    let body = &response[header_end + 4..];
    if chunked {
        decode_chunked(body)
    } else if let Some(length) = content_length {
        if body.len() < length {
            Err(LocalLmError::InvalidResponse)
        } else {
            Ok(body[..length].to_vec())
        }
    } else {
        Ok(body.to_vec())
    }
}

fn decode_chunked(mut input: &[u8]) -> Result<Vec<u8>, LocalLmError> {
    let mut output = Vec::new();
    loop {
        let line_end = find_bytes(input, b"\r\n").ok_or(LocalLmError::InvalidResponse)?;
        let size_text = std::str::from_utf8(&input[..line_end])
            .map_err(|_| LocalLmError::InvalidResponse)?
            .split(';')
            .next()
            .unwrap_or_default();
        let size = usize::from_str_radix(size_text.trim(), 16)
            .map_err(|_| LocalLmError::InvalidResponse)?;
        input = &input[line_end + 2..];
        if size == 0 {
            return Ok(output);
        }
        if input.len() < size + 2 || &input[size..size + 2] != b"\r\n" {
            return Err(LocalLmError::InvalidResponse);
        }
        if output.len().saturating_add(size) > LOCAL_LM_RESPONSE_CAP {
            return Err(LocalLmError::ResponseTooLarge);
        }
        output.extend_from_slice(&input[..size]);
        input = &input[size + 2..];
    }
}

fn extract_completion_text(value: &JsonValue) -> Option<&str> {
    let content = value
        .get("choices")?
        .as_array()?
        .first()?
        .get("message")?
        .get("content")?;
    if let Some(text) = content.as_str() {
        return Some(text);
    }
    content
        .as_array()?
        .iter()
        .find_map(|part| part.get("text").and_then(JsonValue::as_str))
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompactionChunk {
    pub start_turn: usize,
    pub end_turn_exclusive: usize,
    pub text: String,
    pub source_characters: usize,
}

/// Plans at most three summary calls for the omitted prefix. Every call covers
/// at least one turn, including an oversized one.
pub fn plan_compaction_chunks(
    history: &[PromptHistoryTurn],
    already_covered: usize,
    omitted_turns: usize,
) -> Vec<CompactionChunk> {
    let end = omitted_turns.min(history.len());
    let mut cursor = already_covered.min(end);
    let mut chunks = Vec::new();
    while cursor < end && chunks.len() < MAX_COMPACTION_CALLS {
        let start = cursor;
        let mut text = String::new();
        let mut source_characters = 0usize;
        while cursor < end {
            let rendered = render_compaction_turn(&history[cursor]);
            if cursor > start
                && text.len().saturating_add(rendered.len()).saturating_add(2)
                    > COMPACTION_DELTA_LIMIT
            {
                break;
            }
            if !text.is_empty() {
                text.push_str("\n\n");
            }
            text.push_str(&rendered);
            source_characters =
                source_characters.saturating_add(history[cursor].text.chars().count());
            cursor += 1;
            if text.len() >= COMPACTION_DELTA_LIMIT {
                break;
            }
        }
        chunks.push(CompactionChunk {
            start_turn: start,
            end_turn_exclusive: cursor,
            text,
            source_characters,
        });
    }
    chunks
}

fn render_compaction_turn(turn: &PromptHistoryTurn) -> String {
    let role = match turn.role {
        PromptTurnRole::User => "User",
        PromptTurnRole::Assistant => "Assistant",
        PromptTurnRole::System => "System",
    };
    format!(
        "{role}:\n{}",
        clip_turn_for_compaction(turn.text.trim(), COMPACTION_TURN_LIMIT)
    )
}

pub fn clip_turn_for_compaction(text: &str, limit: usize) -> String {
    if text.len() <= limit {
        return text.to_owned();
    }
    if limit < 8 {
        return truncate_utf8_visible(text, limit);
    }
    let head_budget = ((limit as f32) * 0.6) as usize;
    let separator = "\n[…]\n";
    let tail_budget = limit
        .saturating_sub(head_budget)
        .saturating_sub(separator.len());
    let head = truncate_without_ellipsis(text, head_budget);
    let mut tail_start = text.len().saturating_sub(tail_budget);
    while tail_start < text.len() && !text.is_char_boundary(tail_start) {
        tail_start += 1;
    }
    format!("{head}{separator}{}", &text[tail_start..])
}

fn truncate_without_ellipsis(value: &str, max_bytes: usize) -> &str {
    let mut end = value.len().min(max_bytes);
    while end > 0 && !value.is_char_boundary(end) {
        end -= 1;
    }
    &value[..end]
}

pub fn sanitize_compaction_summary(raw: &str, source_characters: usize) -> Option<String> {
    let mut text = raw.trim().to_owned();
    if text.starts_with("```") && text.ends_with("```") && text.len() >= 6 {
        text = text
            .trim_start_matches('`')
            .trim_end_matches('`')
            .trim()
            .to_owned();
    }
    for prefix in ["Summary:", "Conversation summary:", "Here is the summary:"] {
        if text
            .get(..prefix.len())
            .is_some_and(|value| value.eq_ignore_ascii_case(prefix))
        {
            text = text[prefix.len()..].trim().to_owned();
            break;
        }
    }
    if text.is_empty() {
        return None;
    }
    let max = COMPACTION_SUMMARY_LIMIT.min(source_characters.max(1));
    Some(truncate_utf8_visible(&text, max))
}

pub fn transcript_prefix_digest(turns: &[StoredTurn], count: usize) -> String {
    let mut canonical = String::new();
    for turn in turns.iter().take(count) {
        canonical.push_str(match turn.role {
            super::store::TurnRole::User => "user",
            super::store::TurnRole::Assistant => "assistant",
            super::store::TurnRole::System => "system",
        });
        canonical.push('\u{1f}');
        canonical.push_str(&turn.text);
        canonical.push('\u{1e}');
    }
    stable_digest(&canonical)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_or_credentialed_endpoints_are_rejected() {
        assert!(matches!(
            LocalLmClient::new(LocalLmConfig {
                endpoint: "https://example.com/v1/chat/completions".into(),
                model: "x".into(),
            }),
            Err(LocalLmError::UnsafeEndpoint)
        ));
        assert!(matches!(
            LocalLmClient::new(LocalLmConfig {
                endpoint: "http://user:pass@127.0.0.1:1234/x".into(),
                model: "x".into(),
            }),
            Err(LocalLmError::UnsafeEndpoint)
        ));
    }

    #[test]
    fn chunked_response_and_content_array_decode() {
        let body = br#"{"choices":[{"message":{"content":[{"type":"text","text":"Hello"}]}}]}"#;
        let wire = format!("{:x}\r\n", body.len())
            .into_bytes()
            .into_iter()
            .chain(body.iter().copied())
            .chain(b"\r\n0\r\n\r\n".iter().copied())
            .collect::<Vec<_>>();
        let decoded = decode_chunked(&wire).unwrap();
        let value: JsonValue = serde_json::from_slice(&decoded).unwrap();
        assert_eq!(extract_completion_text(&value), Some("Hello"));
    }

    #[test]
    fn compaction_advances_on_oversized_turn_and_caps_calls() {
        let history: Vec<_> = (0..10)
            .map(|index| PromptHistoryTurn {
                role: PromptTurnRole::User,
                text: if index == 0 {
                    "x".repeat(COMPACTION_DELTA_LIMIT * 2)
                } else {
                    format!("turn {index}")
                },
                tool_names: Vec::new(),
            })
            .collect();
        let chunks = plan_compaction_chunks(&history, 0, history.len());
        assert!(!chunks.is_empty());
        assert!(chunks[0].end_turn_exclusive >= 1);
        assert!(chunks[0].text.len() <= COMPACTION_DELTA_LIMIT);
        assert!(chunks.len() <= MAX_COMPACTION_CALLS);
    }

    #[test]
    fn sanitizer_never_returns_more_than_source() {
        let summary =
            sanitize_compaction_summary("```text\nSummary: this is unnecessarily long\n```", 12)
                .unwrap();
        assert!(summary.chars().count() <= 12);
        assert!(!summary.starts_with("```"));
    }
}
