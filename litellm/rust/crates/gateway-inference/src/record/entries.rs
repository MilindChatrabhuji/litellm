//! What the gateway knows of one call so far, as typed entries. How a callback reads and writes
//! them by contract field is the callbacks' own view (`gateway/src/callbacks/fields.rs`).

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value};

use super::{AttemptSeen, ServedDeployment};
use crate::{CallType, ServedCall, headers::Routing};

const MODEL: &str = "model";
const STREAM: &str = "stream";

/// The request body of one call. The route owns it, so it is only held while the pre-call hook
/// runs, and what it named of the call is all that stays once it is handed back. The model and
/// the stream flag are read from whichever is there, so a change to a held body is what every
/// later reader sees.
enum Request {
    Held(Map<String, Value>),
    Named { model: Option<String>, stream: bool },
}

impl Request {
    fn model(&self) -> Option<&str> {
        match self {
            Self::Held(body) => body.get(MODEL).and_then(Value::as_str),
            Self::Named { model, .. } => model.as_deref(),
        }
    }

    fn stream(&self) -> bool {
        match self {
            Self::Held(body) => body.get(STREAM).and_then(Value::as_bool) == Some(true),
            Self::Named { stream, .. } => *stream,
        }
    }

    fn named(&self) -> Self {
        Self::Named {
            model: self.model().map(str::to_owned),
            stream: self.stream(),
        }
    }
}

/// A header added to an answer whose head was already written.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("the answer's head was already written")]
pub struct HeadSent;

/// The headers callbacks and Rust code add to the answer, in the order they were added, until
/// its head is written. After that the bag is closed and an add is refused.
#[derive(Default)]
struct ResponseHeaders {
    added: Map<String, Value>,
    sent: bool,
}

/// What the answer to a call that ended on an error tells of how it ended.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ErrorAnswer {
    /// The error was raised inside the route's handler, past the checks the proxy makes before
    /// it, so its answer carries the proxy's error headers.
    pub in_handler: bool,
    /// The timeout of the provider attempt the call ended on, `None` when no provider attempt
    /// raised the error.
    pub timeout: Option<Duration>,
    /// The provider's response headers on the error the call ended on.
    pub provider_headers: Vec<(String, String)>,
}

/// How a call ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ended {
    Served,
    Failed,
}

/// One call as it stands.
pub struct Entries {
    call_type: Option<CallType>,
    start: f64,
    started: Instant,
    request: Request,
    rate_limit_statuses: Option<Value>,
    response_headers: ResponseHeaders,
    served: Option<ServedCall>,
    ended: Option<Ended>,
    routing: Option<Routing>,
    served_deployment: Option<ServedDeployment>,
    attempt: Option<AttemptSeen>,
    /// The call ended on an error, whose answer names only what [`Entries::answered_error`]
    /// was told.
    error_answer: Option<ErrorAnswer>,
    stream_timeout: bool,
    stream_answer: bool,
    callback_time: Duration,
}

fn unix_now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |elapsed| elapsed.as_secs_f64())
}

impl Entries {
    pub(super) fn new() -> Self {
        Self {
            call_type: None,
            start: unix_now(),
            started: Instant::now(),
            request: Request::Named {
                model: None,
                stream: false,
            },
            rate_limit_statuses: None,
            response_headers: ResponseHeaders::default(),
            served: None,
            ended: None,
            routing: None,
            served_deployment: None,
            attempt: None,
            error_answer: None,
            stream_timeout: false,
            stream_answer: false,
            callback_time: Duration::ZERO,
        }
    }

    /// What the router did for the call once it served it, the deployment that served it and
    /// what the served attempt saw. No attempt that failed or was taken over is written here.
    pub fn routed(
        &mut self,
        routing: Routing,
        deployment: Option<ServedDeployment>,
        attempt: AttemptSeen,
    ) {
        self.routing = Some(routing);
        self.served_deployment = deployment;
        self.attempt = Some(attempt);
    }

    /// The call ended on an error, which is answered with `routing`, what the router did for the
    /// attempt it ended on, in place of what it did for the call before.
    pub fn answered_error(&mut self, routing: Option<Routing>, answer: ErrorAnswer) {
        self.routing = routing;
        self.error_answer = Some(answer);
    }

    /// How the call ended, for a call answered with an error.
    pub fn error_answer(&self) -> Option<&ErrorAnswer> {
        self.error_answer.as_ref()
    }

    /// Whether the call's attempts run under a deployment's stream timeout, by the route's own
    /// rule.
    pub fn times_as_stream(&mut self, streamed: bool) {
        self.stream_timeout = streamed;
    }

    pub fn stream_timeout(&self) -> bool {
        self.stream_timeout
    }

    /// The served answer is a stream, whose head goes out before its body.
    pub fn answers_with_stream(&mut self) {
        self.stream_answer = true;
    }

    pub fn is_stream_answer(&self) -> bool {
        self.stream_answer
    }

    /// When the record was made, as the request came in on its route.
    pub fn started(&self) -> Instant {
        self.started
    }

    /// Adds the time one hook spent in the callbacks.
    pub fn add_callback_time(&mut self, spent: Duration) {
        self.callback_time += spent;
    }

    /// The time the callbacks spent on the call's hooks so far.
    pub fn callback_time(&self) -> Duration {
        self.callback_time
    }

    pub fn routing(&self) -> Option<&Routing> {
        self.routing.as_ref()
    }

    pub fn served_deployment(&self) -> Option<&ServedDeployment> {
        self.served_deployment.as_ref()
    }

    /// What the attempt that served the call saw.
    pub fn attempt(&self) -> Option<&AttemptSeen> {
        self.attempt.as_ref()
    }

    /// The route the call came in on, which the route names once it runs.
    pub fn serve_as(&mut self, call_type: CallType) {
        self.call_type = Some(call_type);
    }

    pub fn call_type(&self) -> Option<CallType> {
        self.call_type
    }

    /// When the call began, in seconds since the Unix epoch.
    pub fn start(&self) -> f64 {
        self.start
    }

    /// Takes the route's `body` for the pre-call hook.
    pub fn hold(&mut self, body: Map<String, Value>) {
        self.request = Request::Held(body);
    }

    /// The body as the pre-call hook left it, back to the route.
    pub fn release(&mut self) -> Map<String, Value> {
        let named = self.request.named();
        match std::mem::replace(&mut self.request, named) {
            Request::Held(body) => body,
            Request::Named { .. } => Map::new(),
        }
    }

    /// The request body while it is held, `None` once the route has it back.
    pub fn body(&self) -> Option<&Map<String, Value>> {
        match &self.request {
            Request::Held(body) => Some(body),
            Request::Named { .. } => None,
        }
    }

    pub fn body_mut(&mut self) -> Option<&mut Map<String, Value>> {
        match &mut self.request {
            Request::Held(body) => Some(body),
            Request::Named { .. } => None,
        }
    }

    pub fn model(&self) -> Option<&str> {
        self.request.model()
    }

    pub fn stream(&self) -> bool {
        self.request.stream()
    }

    pub fn rate_limit_statuses(&self) -> Option<&Value> {
        self.rate_limit_statuses.as_ref()
    }

    pub fn set_rate_limit_statuses(&mut self, statuses: Value) {
        self.rate_limit_statuses = Some(statuses);
    }

    pub fn response_headers(&self) -> &Map<String, Value> {
        &self.response_headers.added
    }

    /// Adds a header to the answer. Refused once the answer's head was written.
    pub fn add_header(&mut self, name: String, value: Value) -> Result<(), HeadSent> {
        if self.response_headers.sent {
            return Err(HeadSent);
        }
        self.response_headers.added.insert(name, value);
        Ok(())
    }

    /// Closes the header bag as the answer's head is written, and hands back what was added.
    /// `record::render` is its one caller.
    pub fn send_head(&mut self) -> &Map<String, Value> {
        self.response_headers.sent = true;
        &self.response_headers.added
    }

    pub fn serve(&mut self, served: ServedCall) {
        self.served = Some(served);
        self.ended = Some(Ended::Served);
    }

    pub fn fail(&mut self) {
        self.ended = Some(Ended::Failed);
    }

    pub fn served(&self) -> Option<&ServedCall> {
        self.served.as_ref()
    }

    pub fn ended(&self) -> Option<Ended> {
        self.ended
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn body(stream: bool) -> Map<String, Value> {
        Map::from_iter([
            ("model".to_owned(), json!("claude")),
            ("stream".to_owned(), json!(stream)),
        ])
    }

    #[rstest]
    #[case::held(false)]
    #[case::released(true)]
    fn the_model_and_the_stream_flag_outlast_the_body(#[case] released: bool) {
        let mut entries = Entries::new();
        entries.hold(body(true));
        entries
            .body_mut()
            .unwrap()
            .insert(MODEL.to_owned(), json!("claude-fallback"));
        if released {
            entries.release();
        }
        assert_eq!(
            (entries.model(), entries.stream(), entries.body().is_some()),
            (Some("claude-fallback"), true, !released)
        );
    }

    #[rstest]
    fn a_header_added_after_the_head_was_written_is_refused() {
        let mut entries = Entries::new();
        entries
            .add_header("x-early".to_owned(), json!("yes"))
            .unwrap();
        entries.send_head();
        assert_eq!(
            (
                entries.add_header("x-late".to_owned(), json!("yes")),
                json!(entries.response_headers())
            ),
            (Err(HeadSent), json!({"x-early": "yes"}))
        );
    }
}
