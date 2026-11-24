//! The head of every answer on the inference routes, written in one pass from the call's record.
//! [`render`] runs inside `record::scoped`, so it reads the record of the call it answers, and it
//! closes the record's header bag as it writes, so a header added after the head, a stream's end
//! included, is refused.

use axum::{
    extract::Request,
    http::{HeaderMap, HeaderName, HeaderValue},
    middleware::Next,
    response::Response,
};
use serde_json::Value;

use super::{AttemptSeen, Entries};
use crate::{ExposedHeader, headers::Routing};

/// Writes the head of the answer the route gave from the call's record.
pub(crate) async fn render(request: Request, next: Next) -> Response {
    let mut response = next.run(request).await;
    super::write(|entries| head(entries, response.headers_mut()));
    response
}

/// One header of the record as the answer carries it, `None` when its name or value is no
/// header. A text is sent as written and any other value as its JSON.
fn header(name: &str, value: &Value) -> Option<(HeaderName, HeaderValue)> {
    let value = match value {
        Value::String(text) => HeaderValue::try_from(text.as_str()),
        other => HeaderValue::try_from(other.to_string()),
    };
    Some((HeaderName::try_from(name).ok()?, value.ok()?))
}

/// Writes the head `entries` give an answer over `headers`, the route's own. A served answer
/// carries the served attempt's cache key, then the headers callbacks and Rust code added, each
/// replacing a header of the route's by its name, then what the router did for the call, which
/// replaces a callback's header of the same name. An answer to a call that ended on an error
/// carries only what the router did for the attempt it ended on.
fn head(entries: &mut Entries, headers: &mut HeaderMap) {
    let ended = entries.is_error_answer();
    let cache_key = entries
        .attempt()
        .and_then(AttemptSeen::cache_key)
        .filter(|_| !ended)
        .and_then(|key| HeaderValue::from_str(key).ok());
    let routing = entries.routing().map(Routing::headers);
    if let Some(key) = cache_key {
        headers.insert(HeaderName::from(ExposedHeader::CacheKey), key);
    }
    let added = entries.send_head();
    if !ended {
        for (name, value) in added.iter().filter_map(|(name, value)| header(name, value)) {
            headers.insert(name, value);
        }
    }
    for (name, value) in routing.into_iter().flatten() {
        if let Some(value) = value.and_then(|value| HeaderValue::from_str(&value).ok()) {
            headers.insert(HeaderName::from_static(name), value);
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::record::HeadSent;

    /// A call `d1` served from the cached answer `k1` after one retry, with `added` added to its
    /// answer, and whether it then ended on an error on `d1`.
    fn entries(added: &[(&str, Value)], ended: bool) -> Entries {
        let mut entries = Entries::new();
        entries.routed(
            Routing::served_by("d1", "public/model", 1),
            None,
            AttemptSeen::cached("k1"),
        );
        for (name, value) in added {
            entries
                .add_header((*name).to_owned(), value.clone())
                .unwrap();
        }
        if ended {
            entries.answered_error(Some(Routing::failed_on("d1")));
        }
        entries
    }

    /// The head `entries` give an answer whose route wrote only its content type.
    fn written(entries: &mut Entries) -> Vec<(String, String)> {
        let mut headers = HeaderMap::from_iter([(
            HeaderName::from_static("content-type"),
            HeaderValue::from_static("application/json"),
        )]);
        head(entries, &mut headers);
        headers
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_str().unwrap().to_owned()))
            .collect()
    }

    fn pairs(expected: &[(&str, &str)]) -> Vec<(String, String)> {
        expected
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect()
    }

    #[rstest]
    #[case::a_served_answer(
        &[("x-callback", json!("yes")), ("x-count", json!(3))],
        false,
        &[
            ("content-type", "application/json"),
            ("x-litellm-cache-key", "k1"),
            ("x-callback", "yes"),
            ("x-count", "3"),
            ("x-litellm-model-id", "d1"),
            ("x-litellm-model-group", "public/model"),
            ("x-litellm-attempted-retries", "1"),
            ("x-litellm-attempted-fallbacks", "0"),
            ("x-litellm-max-retries", "1"),
        ]
    )]
    #[case::an_answer_to_a_call_that_ended_on_an_error(
        &[("x-callback", json!("yes"))],
        true,
        &[("content-type", "application/json"), ("x-litellm-model-id", "d1")]
    )]
    fn the_head_is_the_routes_own_then_the_cache_key_then_what_was_added_then_the_routing(
        #[case] added: &[(&str, Value)],
        #[case] ended: bool,
        #[case] expected: &[(&str, &str)],
    ) {
        assert_eq!(written(&mut entries(added, ended)), pairs(expected));
    }

    #[rstest]
    #[case::the_routes_own_header(
        "content-type",
        &[
            ("content-type", "text/plain"),
            ("x-litellm-cache-key", "k1"),
            ("x-litellm-model-id", "d1"),
            ("x-litellm-model-group", "public/model"),
            ("x-litellm-attempted-retries", "1"),
            ("x-litellm-attempted-fallbacks", "0"),
            ("x-litellm-max-retries", "1"),
        ]
    )]
    #[case::the_cache_key(
        "x-litellm-cache-key",
        &[
            ("content-type", "application/json"),
            ("x-litellm-cache-key", "text/plain"),
            ("x-litellm-model-id", "d1"),
            ("x-litellm-model-group", "public/model"),
            ("x-litellm-attempted-retries", "1"),
            ("x-litellm-attempted-fallbacks", "0"),
            ("x-litellm-max-retries", "1"),
        ]
    )]
    #[case::never_a_routing_header(
        "x-litellm-model-id",
        &[
            ("content-type", "application/json"),
            ("x-litellm-cache-key", "k1"),
            ("x-litellm-model-id", "d1"),
            ("x-litellm-model-group", "public/model"),
            ("x-litellm-attempted-retries", "1"),
            ("x-litellm-attempted-fallbacks", "0"),
            ("x-litellm-max-retries", "1"),
        ]
    )]
    fn a_callbacks_header_replaces_in_place(#[case] name: &str, #[case] expected: &[(&str, &str)]) {
        let mut entries = entries(&[(name, json!("text/plain"))], false);
        assert_eq!(written(&mut entries), pairs(expected));
    }

    #[rstest]
    fn a_header_added_after_the_head_is_refused_and_never_sent() {
        let mut entries = entries(&[], false);
        written(&mut entries);
        assert_eq!(
            (
                entries.add_header("x-late".to_owned(), json!("yes")),
                entries.response_headers().contains_key("x-late"),
            ),
            (Err(HeadSent), false)
        );
    }

    #[rstest]
    #[case::text("x-a", json!("one"), Some("one"))]
    #[case::a_number_as_python_formats_it("x-a", json!(7), Some("7"))]
    #[case::a_name_no_header_can_have("x a", json!("one"), None)]
    fn a_header_added_goes_out_as_text(
        #[case] name: &str,
        #[case] value: Value,
        #[case] sent: Option<&str>,
    ) {
        let header = header(name, &value);
        assert_eq!(
            header.as_ref().and_then(|(_, value)| value.to_str().ok()),
            sent
        );
    }
}
