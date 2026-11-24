//! What one attempt saw of its provider, and the deployment that served the call. Each attempt
//! runs in its own [`AttemptSeen`], which [`Recorder`] fills from the core's interceptor hooks,
//! and only the served attempt's reaches the call's record.

use std::time::{Duration, Instant};

use litellm_host::interceptors::{
    ExecutionFacts, Interceptors, RawResponse, RequestContext, ResultSource, RetriedRequest,
    WireRequest,
};

/// What one attempt saw of its provider: where it called, when its request went out and its
/// response head came back, the headers that head carried and where its result came from.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AttemptSeen {
    url: Option<String>,
    sent: Option<Instant>,
    head: Option<Instant>,
    wait: Option<Duration>,
    provider_headers: Vec<(String, String)>,
    source: Option<ResultSource>,
}

impl AttemptSeen {
    /// The wire URL the attempt called.
    pub fn url(&self) -> Option<&str> {
        self.url.as_deref()
    }

    /// When the provider request was sent, a retried one by its resend.
    pub fn sent(&self) -> Option<Instant> {
        self.sent
    }

    /// When the provider's response head arrived, for a stream when it opened.
    pub fn head(&self) -> Option<Instant> {
        self.head
    }

    /// How long the provider took from the request sent to its response head.
    pub fn wait(&self) -> Option<Duration> {
        self.wait
    }

    pub fn provider_headers(&self) -> &[(String, String)] {
        &self.provider_headers
    }

    pub fn source(&self) -> Option<&ResultSource> {
        self.source.as_ref()
    }

    /// The key of the cached answer the attempt was served from.
    pub fn cache_key(&self) -> Option<&str> {
        match self.source.as_ref()? {
            ResultSource::Cache { key } => Some(key),
            ResultSource::Provider | ResultSource::Host => None,
        }
    }

    fn send(&mut self, url: Option<String>) {
        if url.is_some() {
            self.url = url;
        }
        self.sent = Some(Instant::now());
    }

    fn received(&mut self, headers: Option<Vec<(String, String)>>) {
        let head = Instant::now();
        self.head = Some(head);
        self.wait = self.sent.map(|sent| head.saturating_duration_since(sent));
        if let Some(headers) = headers {
            self.provider_headers = headers;
        }
    }

    /// The headers a stream's head carried, which its encoder does not pass on.
    pub(crate) fn stream_head(&mut self, headers: Vec<(String, String)>) {
        self.provider_headers = headers;
    }
}

/// The deployment that served a call, as the router registry and `routing::timeout` name it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServedDeployment {
    /// The provider's model name the deployment calls.
    pub model: String,
    pub api_base: Option<String>,
    pub timeout: Option<Duration>,
}

/// The core's interceptor hooks as they reach the attempt in scope. It holds nothing itself, so
/// a retried or replaced attempt leaves nothing behind.
#[derive(Clone, Copy, Debug, Default)]
pub struct Recorder;

impl<E> Interceptors<E> for Recorder {
    async fn before_provider_request(
        &self,
        wire: WireRequest,
        _: RequestContext,
    ) -> Result<WireRequest, E> {
        super::see(|seen| seen.send(Some(wire.url.clone())));
        Ok(wire)
    }

    async fn provider_request_retried(&self, _: RetriedRequest) -> Result<(), E> {
        super::see(|seen| seen.send(None));
        Ok(())
    }

    async fn after_provider_response(&self, raw: RawResponse) -> Result<(), E> {
        super::see(|seen| seen.received(raw.head.map(|head| head.headers)));
        Ok(())
    }

    async fn result_ready(&self, facts: ExecutionFacts) -> Result<(), E> {
        super::see(|seen| seen.source = Some(facts.source));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use litellm_host::interceptors::{Cost, ProviderIdentity, ResponseHead};
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    type Seen = (Option<String>, bool, Vec<(String, String)>, Option<String>);

    fn wire() -> WireRequest {
        WireRequest {
            url: "https://provider.test/v1/messages".to_owned(),
            headers: Vec::new(),
            body: json!({}),
        }
    }

    fn context() -> RequestContext {
        RequestContext {
            model: "m".to_owned(),
            custom_llm_provider: "anthropic".to_owned(),
            optional_params: json!({}),
            secret_fields: Vec::new(),
            api_key: None,
            host_transport: false,
        }
    }

    fn head() -> RawResponse {
        RawResponse::new(String::new()).with_head(ResponseHead {
            status: 200,
            headers: vec![("request-id".to_owned(), "req_1".to_owned())],
            sent_body: bytes::Bytes::new(),
            streamed: true,
        })
    }

    fn ready(source: ResultSource) -> ExecutionFacts {
        ExecutionFacts {
            provider: ProviderIdentity {
                model: "m".to_owned(),
                provider: "anthropic".to_owned(),
            },
            source,
            cost: Cost::Deferred,
        }
    }

    /// What `Recorder` left in the attempt in scope once the core told it of a request `sends`, a
    /// response head `answers` and a result from `source`.
    async fn seen(sends: bool, answers: bool, source: ResultSource) -> Seen {
        let ((), seen) = super::super::attempt(async {
            let recorder = &Recorder;
            if sends {
                Interceptors::<()>::before_provider_request(recorder, wire(), context())
                    .await
                    .unwrap();
            }
            if answers {
                Interceptors::<()>::after_provider_response(recorder, head())
                    .await
                    .unwrap();
            }
            Interceptors::<()>::result_ready(recorder, ready(source))
                .await
                .unwrap();
        })
        .await;
        (
            seen.url().map(str::to_owned),
            seen.wait().is_some() && seen.head() >= seen.sent(),
            seen.provider_headers().to_vec(),
            seen.cache_key().map(str::to_owned),
        )
    }

    #[rstest]
    #[case::a_provider_answer(
        true,
        true,
        ResultSource::Provider,
        (
            Some("https://provider.test/v1/messages".to_owned()),
            true,
            vec![("request-id".to_owned(), "req_1".to_owned())],
            None,
        )
    )]
    #[case::a_cache_hit(
        false,
        false,
        ResultSource::Cache { key: "k1".to_owned() },
        (None, false, Vec::new(), Some("k1".to_owned()))
    )]
    #[tokio::test]
    async fn the_recorder_writes_what_the_core_tells_it_into_the_attempt_in_scope(
        #[case] sends: bool,
        #[case] answers: bool,
        #[case] source: ResultSource,
        #[case] expected: Seen,
    ) {
        assert_eq!(seen(sends, answers, source).await, expected);
    }
}
