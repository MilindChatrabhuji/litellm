//! The proxy's own route allowlist, refused in the native app the way the Python proxy refuses it.

use std::sync::Arc;

use axum::{
    Json,
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use litellm_config::GeneralSettings;
use litellm_gateway_auth_core::ExposedRoutes;
use serde_json::json;

/// `general_settings.allowed_routes`, as `gateway-auth-core` reads it.
#[derive(Debug, Default)]
pub struct Settings {
    pub routes: ExposedRoutes,
}

impl Settings {
    pub fn from_general(general: &GeneralSettings) -> Self {
        Self {
            routes: ExposedRoutes::new(general),
        }
    }
}

/// The route allowlist `pre_db_read_auth_checks` runs before it reads a key, with the rule the
/// Python proxy reads `general_settings.allowed_routes` with. It sits inside CORS, as Python's
/// check sits inside its middleware, so a preflight is never refused. The body is the OpenAI one
/// the proxy's auth error handler writes, on every route.
pub(super) async fn refuse_unlisted(
    State(routes): State<Arc<ExposedRoutes>>,
    request: Request,
    next: Next,
) -> Response {
    match routes.permits(request.uri().path()) {
        Ok(()) => next.run(request).await,
        Err(denial) => (
            StatusCode::FORBIDDEN,
            Json(json!({"error": {
                "message": denial.to_string(),
                "type": "auth_error",
                "param": null,
                "code": "403",
            }})),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use axum::{Router, body::Body, middleware::from_fn_with_state, routing::get};
    use litellm_config::Config;
    use rstest::rstest;
    use tower::ServiceExt;

    use super::*;

    async fn answered(yaml: &str, path: &str) -> (StatusCode, serde_json::Value) {
        let config = Config::from_yaml(yaml).unwrap();
        let routes = Arc::new(Settings::from_general(&config.general_settings).routes);
        let app = Router::new()
            .route("/health", get(|| async { Json(json!("served")) }))
            .route("/v1/models", get(|| async { Json(json!("served")) }))
            .layer(from_fn_with_state(routes, refuse_unlisted));
        let response = app
            .oneshot(Request::get(path).body(Body::empty()).unwrap())
            .await
            .unwrap();
        let status = response.status();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        (status, serde_json::from_slice(&body).unwrap())
    }

    #[rstest]
    #[case::no_allowlist("model_list: []\n", "/v1/models", StatusCode::OK, json!("served"))]
    #[case::a_listed_route(
        "general_settings:\n  allowed_routes: [/health]\n",
        "/health",
        StatusCode::OK,
        json!("served")
    )]
    #[case::an_unlisted_route(
        "general_settings:\n  allowed_routes: [/health]\n",
        "/v1/models",
        StatusCode::FORBIDDEN,
        json!({"error": {
            "message": "Access forbidden: Route /v1/models not allowed",
            "type": "auth_error",
            "param": null,
            "code": "403",
        }})
    )]
    #[tokio::test]
    async fn a_route_the_allowlist_leaves_out_is_refused_before_its_handler(
        #[case] yaml: &str,
        #[case] path: &str,
        #[case] status: StatusCode,
        #[case] body: serde_json::Value,
    ) {
        assert_eq!(answered(yaml, path).await, (status, body));
    }
}
