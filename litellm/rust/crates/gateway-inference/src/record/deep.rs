//! The deep `x-litellm-*` headers of Python's `get_custom_headers`, worked out from the call's
//! record: the deployment's model and api base, the call's durations, the time spent in
//! callbacks and the provider's own headers, and on an error answer what the proxy's handler adds
//! to it. Python only sends a header whose value is set, so a value that is empty is left out.
//!
//! Python builds these from the response's `_hidden_params`, which a `/v1/messages` answer only
//! carries for a stream, so the api base, the callback duration and the provider's headers of a
//! complete Messages answer are not sent. An OCR answer carries them, but its provider headers
//! are never copied in, so only its error answer sends them.

use std::time::{Duration, Instant};

use litellm_python_compat::{number::round, repr::float_repr};

use super::{Entries, ErrorAnswer};
use crate::CallType;

/// `OPENAI_RESPONSE_HEADERS` in `litellm/types/utils.py`.
const OPENAI_RESPONSE_HEADERS: [&str; 6] = [
    "x-ratelimit-remaining-requests",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-limit-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-reset-requests",
    "x-ratelimit-reset-tokens",
];

/// The OpenAI headers `get_response_headers` keeps unprefixed, in its order.
const OPENAI_ERROR_HEADERS: [&str; 4] = [
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-tokens",
];

/// `_REQUEST_ID_HEADERS` in `get_headers.py`, in the order `get_provider_request_id` reads them.
const REQUEST_ID_HEADERS: [&str; 4] = [
    "request-id",
    "x-request-id",
    "llm_provider-request-id",
    "llm_provider-x-request-id",
];

const PREFIX: &str = "llm_provider-";

/// Headers by name in the order each name was first written, a later write replacing the value
/// in place, as a Python `dict` keeps them.
#[derive(Default)]
struct Headers(Vec<(String, String)>);

impl Headers {
    fn put(&mut self, name: impl Into<String>, value: impl Into<String>) {
        let (name, value) = (name.into(), value.into());
        match self.0.iter_mut().find(|(held, _)| *held == name) {
            Some((_, held)) => *held = value,
            None => self.0.push((name, value)),
        }
    }

    fn get(&self, name: &str) -> Option<&str> {
        self.0
            .iter()
            .find(|(held, _)| held == name)
            .map(|(_, value)| value.as_str())
    }
}

/// The provider's headers as httpx's `Headers.items()` yields them: lowercase names, the values
/// of a repeated name joined with `, ` at its first place.
fn provider(headers: &[(String, String)]) -> Headers {
    let mut joined = Headers::default();
    for (name, value) in headers {
        let name = name.to_ascii_lowercase();
        let value = match joined.get(&name) {
            Some(held) => format!("{held}, {value}"),
            None => value.clone(),
        };
        joined.put(name, value);
    }
    joined
}

/// `get_provider_request_id`: the first request id header that has a value.
fn request_id(headers: &Headers) -> Option<String> {
    REQUEST_ID_HEADERS
        .iter()
        .find_map(|name| headers.get(name).filter(|value| !value.is_empty()))
        .map(str::to_owned)
}

/// `process_response_headers` over a provider's raw headers: the OpenAI rate limit headers as
/// they are, then every header prefixed `llm_provider-` unless it already is, then the provider's
/// request id as `request-id`.
fn served_provider_headers(headers: &[(String, String)]) -> Vec<(String, String)> {
    let raw = provider(headers);
    let mut out = Headers::default();
    for (name, value) in &raw.0 {
        if OPENAI_RESPONSE_HEADERS.contains(&name.as_str()) {
            out.put(name, value);
        }
    }
    for (name, value) in &raw.0 {
        if name.starts_with(PREFIX) {
            out.put(name, value);
        }
    }
    for (name, value) in &raw.0 {
        if !name.starts_with(PREFIX) {
            out.put(format!("{PREFIX}{name}"), value);
        }
    }
    if let Some(id) = request_id(&raw) {
        out.put("request-id", id);
    }
    out.0
}

/// `get_response_headers` over the headers of a provider's error: every header prefixed
/// `llm_provider-` unless its name holds `llm_provider`, then the OpenAI rate limit headers as
/// they are, then the provider's request id as `request-id`.
fn error_provider_headers(headers: &[(String, String)]) -> Vec<(String, String)> {
    let raw = provider(headers);
    let mut out = Headers::default();
    for (name, value) in &raw.0 {
        if name.contains("llm_provider") {
            out.put(name, value);
        } else {
            out.put(format!("{PREFIX}{name}"), value);
        }
    }
    for name in OPENAI_ERROR_HEADERS {
        if let Some(value) = raw.get(name) {
            out.put(name, value.to_owned());
        }
    }
    if let Some(id) = request_id(&raw) {
        out.put("request-id", id);
    }
    out.0
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

/// A duration in milliseconds as Python's `str` of the float.
fn exact_ms(duration: Duration) -> String {
    float_repr(millis(duration))
}

/// Milliseconds as Python's `str(round(ms, 4))`.
fn rounded_ms(ms: f64) -> String {
    float_repr(round(ms, 4))
}

/// The api base without its query, which could carry a secret, as Python's `split("?")[0]`.
fn api_base(base: &str) -> String {
    base.split('?').next().unwrap_or_default().to_owned()
}

/// The deep headers of a served answer, `now` being when its head is written.
///
/// The response duration runs from the record's start to the provider's response head for a
/// stream, as Python stamps it when the provider stream is handed back, and to when the result
/// was ready for a complete answer. The overhead is that duration less the provider's wait, from
/// its request sent to its response head, which is what `litellm_overhead_time_ms` takes off:
/// the time spent before the provider was called and after it answered. It is not sent for an
/// answer no provider was asked for, such as a cache hit.
pub(super) fn served(entries: &Entries, now: Instant) -> Vec<(String, String)> {
    let stream = entries.is_stream_answer();
    let hidden = stream || entries.call_type() == Some(CallType::Aocr);
    let attempt = entries.attempt();
    let deployment = entries.served_deployment();
    let end = attempt
        .and_then(|seen| {
            if stream {
                seen.head().or(seen.ready())
            } else {
                seen.ready().or(seen.head())
            }
        })
        .unwrap_or(now);
    let total = end.saturating_duration_since(entries.started());
    let mut out = Headers::default();
    if let Some(deployment) = deployment {
        out.put("x-litellm-model-name", deployment.model.clone());
    }
    if hidden && let Some(base) = deployment.and_then(|deployment| deployment.api_base.as_deref()) {
        out.put("x-litellm-model-api-base", api_base(base));
    }
    out.put("x-litellm-response-duration-ms", exact_ms(total));
    if let Some(wait) = attempt.and_then(|seen| seen.wait()) {
        out.put(
            "x-litellm-overhead-duration-ms",
            rounded_ms(millis(total) - millis(wait)),
        );
    }
    if hidden {
        out.put(
            "x-litellm-callback-duration-ms",
            rounded_ms(millis(entries.callback_time())),
        );
    }
    let mut out = out.0;
    if stream && let Some(seen) = attempt {
        out.extend(served_provider_headers(seen.provider_headers()));
    }
    present(out)
}

/// The headers the proxy's handler adds to an answer that ended on `error`: a zero response
/// cost, the timeout of the provider attempt that raised it, and on OCR, whose errors go through
/// `handle_llm_api_exception`, the provider's own headers. `/v1/messages` builds its error
/// headers without them.
pub(super) fn ended(call_type: Option<CallType>, error: &ErrorAnswer) -> Vec<(String, String)> {
    if !error.in_handler {
        return Vec::new();
    }
    let mut out = vec![("x-litellm-response-cost".to_owned(), "0".to_owned())];
    if let Some(timeout) = error.timeout {
        out.push((
            "x-litellm-timeout".to_owned(),
            float_repr(timeout.as_secs_f64()),
        ));
    }
    if call_type == Some(CallType::Aocr) {
        out.extend(error_provider_headers(&error.provider_headers));
    }
    present(out)
}

/// The headers Python's `exclude_values` lets through.
fn present(headers: Vec<(String, String)>) -> Vec<(String, String)> {
    headers
        .into_iter()
        .filter(|(_, value)| !matches!(value.as_str(), "" | "None"))
        .collect()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::{
        headers::Routing,
        record::{AttemptSeen, ServedDeployment},
    };

    fn pairs(headers: &[(&str, &str)]) -> Vec<(String, String)> {
        headers
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect()
    }

    #[rstest]
    #[case::an_openai_rate_limit_header_is_kept_and_prefixed(
        &[("x-ratelimit-remaining-tokens", "9"), ("Content-Type", "application/json")],
        &[
            ("x-ratelimit-remaining-tokens", "9"),
            ("llm_provider-x-ratelimit-remaining-tokens", "9"),
            ("llm_provider-content-type", "application/json"),
        ]
    )]
    #[case::an_already_prefixed_header_is_kept(
        &[("llm_provider-a", "1"), ("b", "2")],
        &[("llm_provider-a", "1"), ("llm_provider-b", "2")]
    )]
    #[case::the_request_id_is_sent_bare(
        &[("x-request-id", "x1"), ("request-id", "r1")],
        &[
            ("llm_provider-x-request-id", "x1"),
            ("llm_provider-request-id", "r1"),
            ("request-id", "r1"),
        ]
    )]
    #[case::a_repeated_header_is_joined(
        &[("via", "a"), ("via", "b")],
        &[("llm_provider-via", "a, b")]
    )]
    fn a_served_streams_provider_headers_go_out_as_process_response_headers_makes_them(
        #[case] headers: &[(&str, &str)],
        #[case] expected: &[(&str, &str)],
    ) {
        assert_eq!(served_provider_headers(&pairs(headers)), pairs(expected));
    }

    #[rstest]
    #[case::the_openai_headers_follow_the_prefixed_ones(
        &[("x-ratelimit-limit-tokens", "9"), ("retry-after", "3")],
        &[
            ("llm_provider-x-ratelimit-limit-tokens", "9"),
            ("llm_provider-retry-after", "3"),
            ("x-ratelimit-limit-tokens", "9"),
        ]
    )]
    #[case::a_name_holding_the_prefix_anywhere_is_kept(
        &[("x-llm_provider-a", "1"), ("x-request-id", "x1")],
        &[
            ("x-llm_provider-a", "1"),
            ("llm_provider-x-request-id", "x1"),
            ("request-id", "x1"),
        ]
    )]
    fn an_errors_provider_headers_go_out_as_get_response_headers_makes_them(
        #[case] headers: &[(&str, &str)],
        #[case] expected: &[(&str, &str)],
    ) {
        assert_eq!(error_provider_headers(&pairs(headers)), pairs(expected));
    }

    #[rstest]
    #[case::whole_milliseconds(Duration::from_millis(1500), "1500.0")]
    #[case::a_fraction(Duration::from_micros(1_250_500), "1250.5")]
    fn a_duration_is_pythons_str_of_its_milliseconds(
        #[case] duration: Duration,
        #[case] expected: &str,
    ) {
        assert_eq!(exact_ms(duration), expected);
    }

    #[rstest]
    #[case::kept_to_four_places(12.345_678, "12.3457")]
    #[case::whole(3.0, "3.0")]
    #[case::none_spent(0.0, "0.0")]
    fn a_rounded_duration_keeps_four_places(#[case] ms: f64, #[case] expected: &str) {
        assert_eq!(rounded_ms(ms), expected);
    }

    #[rstest]
    #[case::no_query("https://a.test/v1", "https://a.test/v1")]
    #[case::a_query_with_a_secret("https://a.test/v1?key=s", "https://a.test/v1")]
    fn an_api_base_goes_out_without_its_query(#[case] base: &str, #[case] expected: &str) {
        assert_eq!(api_base(base), expected);
    }

    /// A call of `call_type` that `d1` served, its answer a stream when `stream` says so.
    fn served_call(call_type: CallType, stream: bool) -> Entries {
        let mut entries = Entries::new();
        entries.serve_as(call_type);
        entries.routed(
            Routing::served_by("d1", "public/model", 0),
            Some(ServedDeployment {
                model: "anthropic/claude-x".to_owned(),
                api_base: Some("https://a.test/v1?key=s".to_owned()),
                timeout: Some(Duration::from_secs(5)),
            }),
            AttemptSeen::default(),
        );
        if stream {
            entries.answers_with_stream();
        }
        entries
    }

    #[rstest]
    #[case::a_complete_messages_answer(
        CallType::AnthropicMessages,
        false,
        &["x-litellm-model-name", "x-litellm-response-duration-ms"]
    )]
    #[case::a_messages_stream(
        CallType::AnthropicMessages,
        true,
        &[
            "x-litellm-model-name",
            "x-litellm-model-api-base",
            "x-litellm-response-duration-ms",
            "x-litellm-callback-duration-ms",
        ]
    )]
    #[case::an_ocr_answer(
        CallType::Aocr,
        false,
        &[
            "x-litellm-model-name",
            "x-litellm-model-api-base",
            "x-litellm-response-duration-ms",
            "x-litellm-callback-duration-ms",
        ]
    )]
    fn a_served_answer_sends_what_pythons_hidden_params_carry_for_it(
        #[case] call_type: CallType,
        #[case] stream: bool,
        #[case] expected: &[&str],
    ) {
        let entries = served_call(call_type, stream);
        let sent = served(&entries, Instant::now());
        assert_eq!(
            sent.iter()
                .map(|(name, _)| name.as_str())
                .collect::<Vec<_>>(),
            expected
        );
    }

    #[rstest]
    fn the_model_name_and_api_base_are_the_served_deployments() {
        let entries = served_call(CallType::Aocr, false);
        let sent = served(&entries, Instant::now());
        assert_eq!(
            &sent[..2],
            &pairs(&[
                ("x-litellm-model-name", "anthropic/claude-x"),
                ("x-litellm-model-api-base", "https://a.test/v1"),
            ])[..]
        );
    }

    #[rstest]
    #[case::refused_before_the_handler(
        CallType::Aocr,
        ErrorAnswer { in_handler: false, ..ErrorAnswer::default() },
        &[]
    )]
    #[case::a_refusal_in_the_handler(
        CallType::AnthropicMessages,
        ErrorAnswer { in_handler: true, ..ErrorAnswer::default() },
        &[("x-litellm-response-cost", "0")]
    )]
    #[case::a_messages_provider_error(
        CallType::AnthropicMessages,
        ErrorAnswer {
            in_handler: true,
            timeout: Some(Duration::from_secs(5)),
            provider_headers: vec![("x-request-id".to_owned(), "x1".to_owned())],
        },
        &[("x-litellm-response-cost", "0"), ("x-litellm-timeout", "5.0")]
    )]
    #[case::an_ocr_provider_error(
        CallType::Aocr,
        ErrorAnswer {
            in_handler: true,
            timeout: Some(Duration::from_millis(2500)),
            provider_headers: vec![("x-request-id".to_owned(), "x1".to_owned())],
        },
        &[
            ("x-litellm-response-cost", "0"),
            ("x-litellm-timeout", "2.5"),
            ("llm_provider-x-request-id", "x1"),
            ("request-id", "x1"),
        ]
    )]
    fn an_error_answer_sends_what_the_proxys_handler_adds(
        #[case] call_type: CallType,
        #[case] error: ErrorAnswer,
        #[case] expected: &[(&str, &str)],
    ) {
        assert_eq!(ended(Some(call_type), &error), pairs(expected));
    }
}
