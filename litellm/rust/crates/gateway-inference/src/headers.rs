use axum::http::HeaderName;
use litellm_router_core::RouterResponse;

/// `LITELLM_UI_ALLOW_HEADERS` in `litellm/constants.py`.
#[derive(Clone, Copy, Debug, strum::VariantArray, strum::IntoStaticStr)]
pub enum ExposedHeader {
    #[strum(serialize = "x-litellm-semantic-filter")]
    SemanticFilter,
    #[strum(serialize = "x-litellm-semantic-filter-tools")]
    SemanticFilterTools,
    #[strum(serialize = "x-litellm-adaptive-router-model")]
    AdaptiveRouterModel,
    #[strum(serialize = "x-litellm-applied-guardrails")]
    AppliedGuardrails,
    #[strum(serialize = "x-litellm-guardrail-scan-id")]
    GuardrailScanId,
    #[strum(serialize = "x-litellm-guardrail-scan-metadata")]
    GuardrailScanMetadata,
    #[strum(serialize = "x-litellm-cache-key")]
    CacheKey,
}

impl From<ExposedHeader> for HeaderName {
    fn from(header: ExposedHeader) -> Self {
        HeaderName::from_static(header.into())
    }
}

/// What the router did for a request. The route writes it to the call's record, and
/// `record::render` turns it into the proxy's `x-litellm-*` headers, the same for a complete
/// answer and for a stream, whose head is written after its attempt was served.
#[derive(Clone, Debug)]
pub struct Routing {
    /// The deployment the last attempt ran on.
    deployment_id: String,
    /// `None` for a call that ended on a failed attempt, which the proxy names only the
    /// deployment of.
    served: Option<Served>,
}

#[derive(Clone, Debug)]
struct Served {
    /// The name the call ran under, an alias as the client wrote it (`x-litellm-model-group`).
    model_group: String,
    /// The group that name resolves to, which the router counts usage under.
    counted_group: String,
    attempted_retries: u32,
    attempted_fallbacks: u32,
    max_retries: Option<u32>,
}

impl Routing {
    pub(crate) fn served<R>(routed: &RouterResponse<R>) -> Self {
        Self {
            deployment_id: routed.deployment_id.clone(),
            served: Some(Served {
                model_group: routed.name.clone(),
                counted_group: routed.model_group.clone(),
                attempted_retries: routed.attempted_retries,
                attempted_fallbacks: routed.attempted_fallbacks,
                max_retries: routed.max_retries,
            }),
        }
    }

    /// The model group that served the call, an alias resolved, `None` for a call that ended on
    /// a failed attempt.
    pub(crate) fn counted_group(&self) -> Option<&str> {
        Some(&self.served.as_ref()?.counted_group)
    }

    /// For a call that ended on a failed attempt on `deployment_id`.
    pub(crate) fn failed_on(deployment_id: &str) -> Self {
        Self {
            deployment_id: deployment_id.to_owned(),
            served: None,
        }
    }

    /// The routing headers in the order the proxy sends them, each with its value when the call
    /// has one.
    pub(crate) fn headers(&self) -> [(&'static str, Option<String>); 5] {
        let served = self.served.as_ref();
        [
            ("x-litellm-model-id", Some(self.deployment_id.clone())),
            (
                "x-litellm-model-group",
                served.map(|served| served.model_group.clone()),
            ),
            (
                "x-litellm-attempted-retries",
                served.map(|served| served.attempted_retries.to_string()),
            ),
            (
                "x-litellm-attempted-fallbacks",
                served.map(|served| served.attempted_fallbacks.to_string()),
            ),
            (
                "x-litellm-max-retries",
                served
                    .and_then(|served| served.max_retries)
                    .map(|max| max.to_string()),
            ),
        ]
    }
}

#[cfg(test)]
impl Routing {
    /// A call `deployment_id` served under `model_group` after `retries` retries.
    pub(crate) fn served_by(deployment_id: &str, model_group: &str, retries: u32) -> Self {
        Self {
            deployment_id: deployment_id.to_owned(),
            served: Some(Served {
                model_group: model_group.to_owned(),
                counted_group: model_group.to_owned(),
                attempted_retries: retries,
                attempted_fallbacks: 0,
                max_retries: (retries > 0).then_some(retries),
            }),
        }
    }
}
