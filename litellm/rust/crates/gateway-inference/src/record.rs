//! The record of one inference call: what the route, the callbacks and every later reader know of
//! it so far. [`scoped`] makes one for each request on the inference routes and runs the handler,
//! every layer inside it and the response body with it in a task-local, so code reaches the call's
//! record through [`read`] and [`write`] and never takes it as a parameter. A task spawned for the
//! call keeps it through [`spawn_in_call`]. Each attempt runs in its own [`AttemptSeen`] through
//! [`attempt`], which [`Recorder`] writes, and the route hands the served one to the record.
//! [`render`] writes the answer's head from the record in one pass.

mod attempt;
mod entries;
mod render;

use std::{
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use axum::{
    body::{Body, Bytes, HttpBody},
    extract::Request,
    middleware::Next,
    response::Response,
};
use http_body::{Frame, SizeHint};
use parking_lot::Mutex;
use serde_json::Value;
use tokio::task::{JoinHandle, futures::TaskLocalFuture};

pub(crate) use self::render::render;
pub use self::{
    attempt::{AttemptSeen, Recorder, ServedDeployment},
    entries::{Ended, Entries, HeadSent},
};

tokio::task_local! {
    static RECORD: Arc<CallRecord>;
    static ATTEMPT: Arc<Mutex<AttemptSeen>>;
}

/// One call as it stands. Every access is one short lock, never held across an await.
pub struct CallRecord(Mutex<Entries>);

impl CallRecord {
    pub fn new() -> Arc<Self> {
        Arc::new(Self(Mutex::new(Entries::new())))
    }
}

/// Runs `future` as part of the call `record` keeps.
pub fn scope<F: Future>(record: Arc<CallRecord>, future: F) -> TaskLocalFuture<Arc<CallRecord>, F> {
    RECORD.scope(record, future)
}

/// Changes the record of the call this code runs in. `None` outside a call, which is a bug.
pub fn write<R>(change: impl FnOnce(&mut Entries) -> R) -> Option<R> {
    RECORD
        .try_with(|record| change(&mut record.0.lock()))
        .map_err(|error| missed("call record", error))
        .ok()
}

/// Reads the record of the call this code runs in. `None` outside a call, which is a bug.
pub fn read<R>(look: impl FnOnce(&Entries) -> R) -> Option<R> {
    RECORD
        .try_with(|record| look(&record.0.lock()))
        .map_err(|error| missed("call record", error))
        .ok()
}

/// Adds a header to the answer of the call this code runs in. Refused once the answer's head was
/// written, which for a stream is before its first byte.
pub fn add_header(name: impl Into<String>, value: impl Into<Value>) -> Result<(), HeadSent> {
    write(|entries| entries.add_header(name.into(), value.into())).unwrap_or(Ok(()))
}

/// Runs `future` as one attempt of the call, and hands back what it saw next to its output. The
/// call's record stays in scope.
pub async fn attempt<F: Future>(future: F) -> (F::Output, AttemptSeen) {
    let seen = Arc::new(Mutex::new(AttemptSeen::default()));
    let output = ATTEMPT.scope(seen.clone(), future).await;
    let seen = std::mem::take(&mut *seen.lock());
    (output, seen)
}

/// Changes what the attempt this code runs in saw. Outside an attempt, which is a bug, it does
/// nothing.
fn see(change: impl FnOnce(&mut AttemptSeen)) {
    let _ = ATTEMPT
        .try_with(|seen| change(&mut seen.lock()))
        .map_err(|error| missed("attempt", error));
}

/// Spawns `future` as part of the call this code runs in, so it reads and writes the same record.
/// Outside a call it is spawned as it is.
pub fn spawn_in_call<F>(future: F) -> JoinHandle<F::Output>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    match RECORD.try_get() {
        #[expect(
            clippy::disallowed_methods,
            reason = "the one sanctioned spawn, which carries the call's record"
        )]
        Ok(record) => tokio::spawn(RECORD.scope(record, future)),
        #[expect(
            clippy::disallowed_methods,
            reason = "the one sanctioned spawn, outside a call there is no record to carry"
        )]
        Err(_) => tokio::spawn(future),
    }
}

/// A read or write with no `what` in scope: a spawn that skipped [`spawn_in_call`], or code run
/// outside a call or an attempt.
fn missed(what: &str, error: impl std::fmt::Display) {
    if cfg!(debug_assertions) {
        panic!("no {what} in scope: {error}");
    }
    tracing::error!(%error, "no {what} in scope");
}

/// The layer outermost on the inference routes. It makes the request's record and runs the rest
/// of the request with it, the response body included, since hyper polls the body after the
/// handler returned.
pub(crate) async fn scoped(request: Request, next: Next) -> Response {
    let record = CallRecord::new();
    let response = RECORD.scope(record.clone(), next.run(request)).await;
    response.map(|body| {
        Body::new(Scoped {
            record,
            body: Some(body),
        })
    })
}

/// A response body polled, and dropped, as part of its call, so a stream's end and a stream the
/// client dropped still reach the record.
struct Scoped {
    record: Arc<CallRecord>,
    body: Option<Body>,
}

impl HttpBody for Scoped {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, axum::Error>>> {
        let this = &mut *self;
        let Some(body) = this.body.as_mut() else {
            return Poll::Ready(None);
        };
        RECORD.sync_scope(this.record.clone(), || Pin::new(body).poll_frame(context))
    }

    fn is_end_stream(&self) -> bool {
        self.body.as_ref().is_none_or(HttpBody::is_end_stream)
    }

    fn size_hint(&self) -> SizeHint {
        self.body
            .as_ref()
            .map_or_else(|| SizeHint::with_exact(0), HttpBody::size_hint)
    }
}

impl Drop for Scoped {
    fn drop(&mut self) {
        let body = self.body.take();
        RECORD.sync_scope(self.record.clone(), || drop(body));
    }
}

#[cfg(test)]
mod tests {
    use axum::{Router, routing::get};
    use futures_util::StreamExt;
    use rstest::rstest;
    use serde_json::json;
    use tower::ServiceExt;

    use super::*;

    fn added() -> Option<Value> {
        read(|entries| json!(entries.response_headers()))
    }

    fn router() -> Router {
        Router::new()
            .route(
                "/",
                get(|| async {
                    add_header("x-handler", "yes").unwrap();
                    let body = futures_util::stream::once(async {
                        add_header("x-body", "yes").unwrap();
                        Ok::<_, std::convert::Infallible>(Bytes::from(added().unwrap().to_string()))
                    });
                    Body::from_stream(body)
                }),
            )
            .layer(axum::middleware::from_fn(scoped))
    }

    #[rstest]
    #[tokio::test]
    async fn the_handler_and_the_body_write_the_same_record() {
        let response = router()
            .oneshot(Request::get("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&body).unwrap(),
            json!({"x-handler": "yes", "x-body": "yes"})
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_task_spawned_in_a_call_writes_its_record() {
        let record = CallRecord::new();
        scope(record.clone(), async {
            spawn_in_call(async { add_header("x-spawned", "yes").unwrap() })
                .await
                .unwrap();
        })
        .await;
        assert_eq!(
            scope(record, async { added() }).await,
            Some(json!({"x-spawned": "yes"}))
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_body_dropped_before_its_end_is_dropped_in_its_call() {
        struct Dropped;
        impl Drop for Dropped {
            fn drop(&mut self) {
                add_header("x-dropped", "yes").unwrap();
            }
        }
        let record = CallRecord::new();
        let guard = Dropped;
        let stream = futures_util::stream::pending::<Result<Bytes, std::convert::Infallible>>()
            .map(move |frame| {
                let _ = &guard;
                frame
            });
        drop(Scoped {
            record: record.clone(),
            body: Some(Body::from_stream(stream)),
        });
        assert_eq!(
            scope(record, async { added() }).await,
            Some(json!({"x-dropped": "yes"}))
        );
    }
}
