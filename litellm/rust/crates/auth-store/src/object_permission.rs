//! An object permission as the auth cache holds it, nested in the key that points at it.

use litellm_redis_store::auth_cache::{CachedObjectPermission, ProfileGrants};
use litellm_repositories::ObjectPermission;

use crate::{Error, key::nullable};

/// What `LiteLLM_ObjectPermissionTable.model_dump(mode="json")` holds for `row`, or the column the
/// model would refuse the row for. The repository's projection leaves `apps` at the model's default.
pub(crate) fn cached_object_permission(
    row: ObjectPermission,
) -> Result<CachedObjectPermission, Error> {
    let ObjectPermission {
        object_permission_id,
        mcp_servers,
        mcp_access_groups,
        mcp_tool_permissions,
        vector_stores,
        agents,
        agent_access_groups,
        models,
        blocked_tools,
        mcp_toolsets,
        search_tools,
        mcp_tool_search_enabled,
        skills,
        mcp_auth_profiles,
    } = row;
    Ok(CachedObjectPermission {
        object_permission_id,
        mcp_servers: Some(mcp_servers),
        mcp_access_groups: Some(mcp_access_groups),
        mcp_tool_permissions: nullable("mcp_tool_permissions", mcp_tool_permissions.as_ref())?,
        vector_stores: Some(vector_stores),
        agents: Some(agents),
        agent_access_groups: Some(agent_access_groups),
        models: Some(models),
        mcp_toolsets: Some(mcp_toolsets),
        blocked_tools: Some(blocked_tools),
        search_tools: Some(search_tools),
        mcp_tool_search_enabled,
        skills: Some(skills),
        apps: None,
        mcp_auth_profiles: nullable("mcp_auth_profiles", mcp_auth_profiles.as_ref())
            .unwrap_or(Some(ProfileGrants::Unreadable))
            .unwrap_or_default(),
        unread: Vec::new(),
    })
}
