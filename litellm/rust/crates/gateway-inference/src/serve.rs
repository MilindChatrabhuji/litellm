//! One inference call from its body to its response. A route declares what it is as an
//! [`Endpoint`], and [`run`] serves it with the host's callback hooks around it (`hooks.rs`): the
//! model its body names goes through router-core, the hooks hear the outcome that served it, the
//! route renders it, and a call that ends on an error is answered in the route's error format.

use std::sync::Arc;

use axum::{
    extract::FromRequestParts,
    http::{HeaderMap, request::Parts},
    response::Response,
};
use litellm_gateway_auth::Authenticated;
use litellm_llms_types::formats::chat_completions::ChatCompletionsUsage;
use litellm_router_core::Op;
use serde_json::{Map, Value};

use crate::{
    Ended, Error, Gateway, RequestId,
    error::ErrorFormat,
    headers::Routing,
    hooks::{self, CallType},
    model_restamp::ReportedModel,
    record::{self, AttemptSeen, Entries, ErrorAnswer},
    request,
    routing::{self, Host, RouteAttempt},
    stream_event::Counts,
};

/// One request to an inference route, apart from its body.
pub struct Call<H: Host> {
    pub(crate) gateway: Arc<Gateway<H>>,
    pub(crate) identity: Authenticated<H::Rows>,
    pub(crate) headers: HeaderMap,
    pub(crate) request_id: Option<String>,
}

impl<H: Host> FromRequestParts<Arc<Gateway<H>>> for Call<H> {
    type Rejection = litellm_gateway_auth::Error;

    async fn from_request_parts(
        parts: &mut Parts,
        gateway: &Arc<Gateway<H>>,
    ) -> Result<Self, Self::Rejection> {
        let identity = Authenticated::from_request_parts(parts, gateway).await?;
        let Ok(RequestId(request_id)) = RequestId::from_request_parts(parts, gateway).await;
        Ok(Self {
            gateway: gateway.clone(),
            identity,
            headers: parts.headers.clone(),
            request_id,
        })
    }
}

/// What a route says about itself, next to the attempt it runs for the router.
pub(crate) trait Endpoint: RouteAttempt {
    const CALL_TYPE: CallType;
    const OP: Op;
    /// The path an unknown model is reported under.
    const PATH: &'static str;
    const ERRORS: ErrorFormat;

    /// What a complete outcome counted. A stream's counts are read from its events.
    fn counts(outcome: &Self::Outcome) -> Option<Counts> {
        let _ = outcome;
        None
    }

    /// Those counts as Python's litellm `Usage`, `None` for a route that counts no tokens.
    fn usage(counts: Counts) -> Option<ChatCompletionsUsage> {
        let _ = counts;
        None
    }

    /// Whether a call of this route with `body` runs under a deployment's stream timeout.
    fn streamed(body: &Map<String, Value>) -> bool {
        routing::streamed(body)
    }

    /// What the attempt that ran `outcome` saw, taken out of it.
    fn attempt(outcome: &mut Self::Outcome) -> AttemptSeen;

    /// The outcome that served the call as the response the client reads.
    fn render(outcome: Self::Outcome, reported: ReportedModel<'_>) -> Result<Response, Error>;
}

/// Serves `body` as one call of the route `build` makes of it. A body that could not be read is
/// answered here too, so it is told in the route's error format.
pub(crate) async fn run<H, E, B>(
    call: &Call<H>,
    body: Result<Map<String, Value>, Error>,
    build: B,
) -> Response
where
    H: Host,
    E: Endpoint,
    B: FnOnce(Map<String, Value>) -> Result<E, Error>,
{
    let (gateway, identity) = (&*call.gateway, &call.identity);
    let serve = |hooks: Arc<H::Hooks>, body: Map<String, Value>| async move {
        let model = request::model(&body)?.to_owned();
        let streamed = E::streamed(&body);
        record::write(|entries| entries.times_as_stream(streamed));
        let endpoint = build(body)?;
        let routed = routing::call(gateway, identity, E::OP, E::PATH, &model, &endpoint).await?;
        let routing = Routing::served(&routed);
        let deployment = routing::served_deployment(gateway, &routed.deployment_id, streamed);
        let reported = ReportedModel::of(&model, &routed);
        let model_group = routed.model_group;
        let mut outcome = routed.output.response;
        let seen = E::attempt(&mut outcome);
        let stream = E::held(&mut outcome).is_some();
        record::write(|entries| {
            entries.routed(routing, deployment, seen);
            if stream {
                entries.answers_with_stream();
            }
        });
        let told = hooks::answered::<E, _>(&hooks, &mut outcome, model_group).await?;
        Ok(told.answer(E::render(outcome, reported)?))
    };
    record::write(|entries| entries.serve_as(E::CALL_TYPE));
    let served = match body {
        Ok(body) => hooks::around(gateway, identity, body, serve).await,
        Err(error) => Err(Ended::from(error)),
    };
    served.unwrap_or_else(|mut ended| {
        let routing = ended.routing.take();
        let answer = error_answer(gateway, routing.as_ref(), &ended.error);
        record::write(|entries| entries.answered_error(routing, answer));
        ended.response(E::ERRORS, call.request_id.as_deref())
    })
}

/// How a call that ended on `error` is answered, `routing` naming the attempt it ended on.
fn error_answer<H: Host>(
    gateway: &Gateway<H>,
    routing: Option<&Routing>,
    error: &Error,
) -> ErrorAnswer {
    let streamed = record::read(Entries::stream_timeout).unwrap_or(false);
    let timeout = routing
        .filter(|_| error.raised_by_provider())
        .and_then(|routing| routing::served_deployment(gateway, routing.deployment_id(), streamed))
        .and_then(|deployment| deployment.timeout);
    ErrorAnswer {
        in_handler: error.in_handler(),
        timeout,
        provider_headers: error.provider_headers(),
    }
}
