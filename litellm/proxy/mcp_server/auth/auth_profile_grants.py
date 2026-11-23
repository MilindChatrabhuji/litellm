from types import MappingProxyType
from typing import Final

from litellm._logging import verbose_proxy_logger
from litellm.models.object_permission import LiteLLM_ObjectPermissionTable, StoredProfileGrantError
from litellm.proxy._types import UI_TEAM_ID, UserAPIKeyAuth
from litellm.proxy.agent_endpoints.auth.managed_authorization import managed_agent_policy
from litellm.proxy.auth.auth_checks import get_object_permission
from litellm.proxy.auth.jwt import ClaimReading
from litellm.proxy.db.db_lookup_gate import bounded_db_lookup
from litellm.proxy.mcp_server.auth.user_api_key_auth_mcp import MCPRequestHandler, is_mcp_admitted_user_subject
from litellm.repositories.table_repositories import AgentsRepository
from litellm.types.mcp_server.auth_profile_grants import ProfileGrants, UnreadableGrants


def _unreadable(link: str | None, permission: LiteLLM_ObjectPermissionTable | None) -> bool:
    return link is not None and (permission is None or link != permission.object_permission_id)


async def _agent_permission(caller: UserAPIKeyAuth) -> LiteLLM_ObjectPermissionTable | UnreadableGrants | None:
    if caller.agent_id is None:
        return None
    managed: Final = managed_agent_policy(caller)
    if managed is not None:
        permission: Final = managed.object_permission
        return LiteLLM_ObjectPermissionTable.model_validate(permission) if permission is not None else None
    from litellm.proxy.proxy_server import prisma_client, user_api_key_cache

    if prisma_client is None:
        return UnreadableGrants()
    row: Final = await bounded_db_lookup(
        AgentsRepository(prisma_client, use_writer=True).table.find_unique(where={"agent_id": caller.agent_id}),
        name="agent_link",
    )
    if row is None:
        return UnreadableGrants()
    if row.object_permission_id is None:
        return None
    return await get_object_permission(row.object_permission_id, prisma_client, user_api_key_cache, check_db_only=True)


def _one_team(caller: UserAPIKeyAuth) -> bool:
    if caller.team_id in (None, "", UI_TEAM_ID):
        return False
    if not caller.jwt_claims:
        return True
    from litellm.proxy.proxy_server import litellm_jwtauth

    teams: Final = ClaimReading(litellm_jwtauth).get_all_jwt_team_ids(caller.jwt_claims)
    return len(frozenset(teams)) <= 1


async def _projected_grants(
    permission: LiteLLM_ObjectPermissionTable, *, fresh: bool
) -> ProfileGrants | StoredProfileGrantError:
    if "mcp_auth_profiles" in permission.model_fields_set:
        return permission.mcp_auth_profiles or {}
    from litellm.proxy.proxy_server import prisma_client, user_api_key_cache

    if prisma_client is None:
        return StoredProfileGrantError.UNREADABLE
    projected: Final = await get_object_permission(
        permission.object_permission_id,
        prisma_client,
        user_api_key_cache,
        check_db_only=fresh,
        require_profile_grants=True,
    )
    return (projected.mcp_auth_profiles or {}) if projected is not None else StoredProfileGrantError.UNREADABLE


async def caller_auth_profile_grants(caller: UserAPIKeyAuth | None) -> tuple[ProfileGrants, ...] | UnreadableGrants:
    if caller is None or (
        caller.managed_agent_context is not None and caller.managed_agent_context.mode == "delegated"
    ):
        return ()
    try:
        admitted_subject: Final = is_mcp_admitted_user_subject(caller)
        fresh: Final = bool(caller.requires_fresh_policy)
        key_id: Final = None if admitted_subject or caller.is_session_token else caller.object_permission_id
        key: Final = (
            await MCPRequestHandler.key_object_permission_hydrated(caller.for_key_grant_read()) if key_id else None
        )
        if _unreadable(key_id, key):
            return UnreadableGrants()
        has_team: Final = not admitted_subject and _one_team(caller)
        team: Final = await MCPRequestHandler.permission_team(caller) if has_team else None
        if has_team and (team is None or _unreadable(team.object_permission_id, team.object_permission)):
            return UnreadableGrants()
        agent: Final = await _agent_permission(caller)
        if isinstance(agent, UnreadableGrants):
            return agent
        permissions: Final = (agent, key, team.object_permission if team is not None else None)
        grants: Final = [
            await _projected_grants(permission, fresh=fresh) for permission in permissions if permission is not None
        ]
        if StoredProfileGrantError.UNREADABLE in grants:
            return UnreadableGrants()
        return tuple(
            MappingProxyType(dict(grant)) for grant in grants if not isinstance(grant, StoredProfileGrantError)
        )
    except Exception as exc:  # noqa: BLE001  # store and cache backends can raise different read failures
        verbose_proxy_logger.warning("Could not read MCP auth profile grants (%s)", type(exc).__name__)
        return UnreadableGrants()
