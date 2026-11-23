import asyncio
from collections.abc import Mapping
from contextlib import closing
from types import SimpleNamespace
from typing import Final, Literal
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from mcp.shared.exceptions import MCPError
from mcp.types import INVALID_REQUEST, ListToolsRequest

from litellm.constants import PROXY_DB_LOOKUP_DEADLINE_SECONDS
from litellm.models.object_permission import StoredProfileGrantError
from litellm.proxy import proxy_server
from litellm.proxy._types import (
    UI_TEAM_ID,
    LiteLLM_JWTAuth,
    LiteLLM_ObjectPermissionTable,
    LiteLLM_TeamTableCachedObj,
    UserAPIKeyAuth,
)
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.db.db_lookup_gate import db_lookup_stall_tracker
from litellm.proxy.mcp_server.auth.auth_profile_grants import caller_auth_profile_grants
from litellm.proxy.mcp_server.auth.user_api_key_auth_mcp import MCPRequestHandler
from litellm.proxy.mcp_server.contracts import OperationContext
from litellm.proxy.mcp_server.operations import GatewayOperations
from litellm.redis.stores.auth_cache import AuthObject
from litellm.types.agents import AgentResponse
from litellm.types.mcp_server.auth_profile_grants import UnreadableGrants
from litellm.types.proxy.agent_identity import AgentIdentityBinding, ManagedAgentContext
from tests._support.clock_loops import AutojumpClockLoop
from tests.unit.proxy.rust_auth_stores import (
    FakeKeyStore,
    FakeTeamStore,
    RustAuthStores,
    RustStoreRows,
    install_rust_auth_stores,
)


def _permission(permission_id: str, grants: Mapping[str, str] | None = None) -> LiteLLM_ObjectPermissionTable:
    return LiteLLM_ObjectPermissionTable(object_permission_id=permission_id, mcp_auth_profiles=grants)


def _caller(
    *,
    key: LiteLLM_ObjectPermissionTable | None = None,
    agent: Mapping[str, str] | None = None,
    mode: Literal["autonomous", "delegated"] | None = None,
    team_id: str | None = "payments",
    admitted_subject: bool = False,
) -> UserAPIKeyAuth:
    managed: Final = (
        {}
        if agent is None
        else {
            "managed_agent_policy": AgentResponse(
                agent_id="agents-runner",
                agent_name="agents-runner",
                agent_card_params={},
                object_permission={"object_permission_id": "perm-agent", "mcp_auth_profiles": dict(agent)},
            ),
        }
    )
    context: Final = (
        {}
        if mode is None
        else {
            "managed_agent_context": ManagedAgentContext(
                agent_id="agents-runner", mode=mode, user_id="reader" if mode == "delegated" else None
            )
        }
    )
    auth: Final = UserAPIKeyAuth(
        api_key="hashed-key",
        team_id=team_id,
        agent_id=None if agent is None else "agents-runner",
        object_permission=key,
        object_permission_id=None if key is None else key.object_permission_id,
    )
    return auth.model_copy(update={"mcp_admitted_user_subject": admitted_subject, **managed, **context})


@pytest.fixture
def database(monkeypatch: pytest.MonkeyPatch) -> None:
    permissions: Final = {
        "perm-team": _permission("perm-team", TEAM),
        "perm-bots": _permission("perm-bots", {"tickets": "bots"}),
    }
    install_rust_auth_stores(
        monkeypatch,
        RustAuthStores(
            team=FakeTeamStore(rows={"payments": {"team_id": "payments", "object_permission_id": "perm-team"}}),
            object_permission=RustStoreRows(
                reader={
                    **permissions,
                    "agent-current": _permission("agent-current", {"tickets": "retired"}),
                    "agent-deleted": _permission("agent-deleted", {"tickets": "retired"}),
                    "timeout": _permission("timeout", {"tickets": "retired"}),
                    "disconnected": _permission("disconnected", {"tickets": "retired"}),
                },
                writer={
                    **permissions,
                    "agent-current": _permission("agent-current", {"tickets": "bots"}),
                    "timeout": TimeoutError(),
                    "disconnected": RuntimeError("database disconnected"),
                },
            ),
        ),
    )
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    monkeypatch.setattr(proxy_server, "user_api_key_cache", UserApiKeyCache())


KEY: Final = _permission("perm-key", {"tickets": "bots"})
EMPTY: Final[Mapping[str, str]] = {}
TEAM: Final = {"tickets": "ops"}


@pytest.mark.asyncio
@pytest.mark.usefixtures("database")
@pytest.mark.parametrize(
    ("caller", "expected"),
    [
        pytest.param(_caller(), (TEAM,), id="a-single-team-grants-by-itself"),
        pytest.param(_caller(key=KEY), ({"tickets": "bots"}, TEAM), id="the-key-comes-before-the-team"),
        pytest.param(
            _caller(key=_permission("perm-empty")), (EMPTY, TEAM), id="a-key-without-grants-still-comes-before-the-team"
        ),
        pytest.param(
            _caller(key=KEY, agent={"tickets": "agents"}, mode="autonomous"),
            ({"tickets": "agents"}, {"tickets": "bots"}, TEAM),
            id="R6-an-autonomous-agent-comes-before-the-key",
        ),
        pytest.param(
            _caller(key=KEY, agent={"tickets": "agents"}, mode="delegated"),
            (),
            id="R7-a-delegated-agent-holds-no-carrier",
        ),
        pytest.param(_caller(key=KEY).model_copy(update={"is_session_token": True}), (TEAM,), id="session-team-only"),
        pytest.param(
            _caller(key=KEY, team_id=None), ({"tickets": "bots"},), id="a-key-without-a-team-has-no-team-step"
        ),
        pytest.param(
            _caller(key=KEY, team_id=UI_TEAM_ID), ({"tickets": "bots"},), id="the-dashboard-team-holds-no-carrier"
        ),
        pytest.param(
            _caller(key=KEY, admitted_subject=True),
            (),
            id="R8-a-caller-admitted-through-several-teams-skips-the-key-and-team-steps",
        ),
        pytest.param(None, (), id="no-caller-holds-no-carrier"),
    ],
)
async def test_grants_follow_the_most_specific_carrier_first(
    caller: UserAPIKeyAuth | None, expected: tuple[Mapping[str, str], ...]
) -> None:
    assert await caller_auth_profile_grants(caller) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("caller", "expected"),
    [
        pytest.param(
            UserAPIKeyAuth(api_key="hashed-key", object_permission_id="perm-bots"),
            ({"tickets": "bots"},),
            id="a-key-cached-without-its-row-is-hydrated-by-main",
        ),
        pytest.param(UserAPIKeyAuth(api_key="hashed-key"), (), id="a-key-without-a-row-grants-nothing"),
    ],
)
@pytest.mark.usefixtures("database")
async def test_a_key_cached_without_its_permission_row_gets_what_main_hydrates(
    caller: UserAPIKeyAuth, expected: tuple[Mapping[str, str], ...]
) -> None:
    assert await caller_auth_profile_grants(caller) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "database_connected",
    [
        pytest.param(True, id="R10-a-key-naming-a-row-that-cannot-be-found-has-unreadable-grants"),
        pytest.param(False, id="R10-a-key-naming-a-row-on-a-proxy-without-a-database-has-unreadable-grants"),
    ],
)
async def test_a_key_naming_a_permission_row_that_cannot_be_read_has_unreadable_grants(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest, database_connected: bool
) -> None:
    if database_connected:
        request.getfixturevalue("database")
    else:
        monkeypatch.setattr(proxy_server, "prisma_client", None)

    assert (
        await caller_auth_profile_grants(UserAPIKeyAuth(api_key="hashed-key", object_permission_id="perm-missing"))
        == UnreadableGrants()
    )


@pytest.mark.asyncio
@pytest.mark.usefixtures("database")
@pytest.mark.parametrize(("fresh", "expected"), [(True, (EMPTY, TEAM)), (False, UnreadableGrants())])
async def test_cached_team_with_a_missing_permission_is_unreadable_until_hydrated(
    fresh: bool, expected: tuple[Mapping[str, str], ...] | UnreadableGrants
) -> None:
    await proxy_server.user_api_key_cache.async_set_cache(
        key=AuthObject.team("payments"),
        value=LiteLLM_TeamTableCachedObj(team_id="payments", object_permission_id="perm-team"),
    )
    caller: Final = _caller(agent={}, mode="autonomous").model_copy(update={"requires_fresh_policy": fresh})
    assert await caller_auth_profile_grants(caller) == expected


@pytest.mark.asyncio
@pytest.mark.usefixtures("database")
@pytest.mark.parametrize(
    ("teams", "expected"), [(("payments",), (TEAM,)), (("payments", "payments"), (TEAM,)), (("payments", "design"), ())]
)
async def test_jwt_only_inherits_one_unambiguous_team(
    monkeypatch: pytest.MonkeyPatch, teams: tuple[str, ...], expected: tuple[Mapping[str, str], ...]
) -> None:
    monkeypatch.setattr(proxy_server, "litellm_jwtauth", LiteLLM_JWTAuth(team_ids_jwt_field="groups"))
    caller: Final = _caller().model_copy(update={"jwt_claims": {"groups": list(teams)}})
    assert await caller_auth_profile_grants(caller) == expected


@pytest.mark.asyncio
async def test_unreadable_agent_link_cannot_degrade_to_the_keys_grant(monkeypatch: pytest.MonkeyPatch) -> None:
    prisma: Final = MagicMock(**{"writer_db.litellm_agentstable.find_unique": AsyncMock(side_effect=ConnectionError)})
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", UserApiKeyCache())
    caller: Final = _caller(key=KEY, team_id=None).model_copy(update={"agent_id": "agents-runner"})
    assert await caller_auth_profile_grants(caller) == UnreadableGrants()


@pytest.mark.asyncio
@pytest.mark.usefixtures("database")
async def test_a_scoped_permission_cannot_substitute_for_a_missing_key_permission() -> None:
    caller: Final = _caller(key=KEY).model_copy(update={"object_permission_id": "original"})
    assert await caller_auth_profile_grants(caller) == UnreadableGrants()


@pytest.mark.asyncio
async def test_malformed_managed_agent_permission_has_unreadable_grants() -> None:
    policy: Final = AgentResponse(
        agent_id="agents-runner",
        agent_name="runner",
        agent_card_params={},
        object_permission={"object_permission_id": "perm-agent", "mcp_auth_profiles": ["unreadable"]},
    )
    caller: Final = UserAPIKeyAuth(agent_id=policy.agent_id, object_permission=KEY).model_copy(
        update={"managed_agent_policy": policy}
    )
    assert await caller_auth_profile_grants(caller) == UnreadableGrants()


@pytest.mark.asyncio
async def test_a_validated_malformed_key_row_returns_unreadable_grants() -> None:
    row: Final = LiteLLM_ObjectPermissionTable.model_validate(
        {"object_permission_id": "perm-key", "mcp_auth_profiles": {"tickets": ["Build"]}}
    )
    assert row.mcp_auth_profiles is StoredProfileGrantError.UNREADABLE
    assert await caller_auth_profile_grants(_caller(key=row, team_id=None)) == UnreadableGrants()


@pytest.mark.asyncio
@pytest.mark.parametrize("current_agent", [None, "new-agent"])
async def test_refresh_requires_readmission_when_the_key_changes_agent(
    monkeypatch: pytest.MonkeyPatch, current_agent: str | None
) -> None:
    caller: Final = UserAPIKeyAuth(api_key="fixture", agent_id="original")
    caller.via_virtual_key = True
    rows: Final = FakeKeyStore(
        rows={"fixture": {"token": "fixture", "agent_id": "original"}},
        writer={"fixture": {"token": "fixture", "agent_id": current_agent}},
    )
    install_rust_auth_stores(monkeypatch, RustAuthStores(key=rows))
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    with pytest.raises(HTTPException) as refused:
        await MCPRequestHandler.refresh_catalog_authority(caller)
    assert (refused.value.status_code, refused.value.detail) == (401, "Key agent binding changed. Authenticate again.")


@pytest.mark.asyncio
@pytest.mark.usefixtures("database")
async def test_team_cache_entry_with_an_unprojected_permission_uses_the_saved_grant() -> None:
    await proxy_server.user_api_key_cache.async_set_cache(
        key=AuthObject.team("payments"),
        value=LiteLLM_TeamTableCachedObj(
            team_id="payments",
            object_permission_id="perm-team",
            object_permission=LiteLLM_ObjectPermissionTable(object_permission_id="perm-team"),
        ),
    )
    assert await caller_auth_profile_grants(_caller()) == (TEAM,)


@pytest.mark.asyncio
@pytest.mark.parametrize("grants", [None, {}, {"tickets": "bots"}])
async def test_a_projected_permission_needs_only_one_writer_read(
    monkeypatch: pytest.MonkeyPatch, grants: Mapping[str, str] | None
) -> None:
    rows: Final = RustStoreRows(writer={"linked": _permission("linked", grants)})
    install_rust_auth_stores(monkeypatch, RustAuthStores(object_permission=rows))
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    monkeypatch.setattr(proxy_server, "user_api_key_cache", UserApiKeyCache())
    old: Final = LiteLLM_ObjectPermissionTable(object_permission_id="linked")
    await proxy_server.user_api_key_cache.async_set_cache(key=AuthObject.object_permission("linked"), value=old)
    caller: Final = _caller(key=old, team_id=None)
    assert await caller_auth_profile_grants(caller) == (grants or {},)
    assert await caller_auth_profile_grants(caller) == (grants or {},)
    assert rows.lookups == [("linked", True)]
    fresh: Final = caller.model_copy(update={"requires_fresh_policy": True})
    assert await caller_auth_profile_grants(fresh) == (grants or {},)
    assert rows.lookups == [("linked", True), ("linked", True)]


@pytest.mark.asyncio
@pytest.mark.usefixtures("database")
async def test_agent_grants_follow_the_current_link_across_edits(monkeypatch: pytest.MonkeyPatch) -> None:
    agent: Final = SimpleNamespace(object_permission_id="perm-team")
    unbound: Final = SimpleNamespace(object_permission_id=None)
    reassigned: Final = SimpleNamespace(object_permission_id="perm-bots")
    table: Final = SimpleNamespace(find_unique=AsyncMock(side_effect=[None, unbound, agent, reassigned, unbound]))
    db: Final = SimpleNamespace(litellm_agentstable=table)
    prisma: Final = SimpleNamespace(db=db, writer_db=db)
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)
    caller: Final = UserAPIKeyAuth(agent_id="runner")
    assert await caller_auth_profile_grants(caller) == UnreadableGrants()
    assert await caller_auth_profile_grants(caller) == ()
    assert await caller_auth_profile_grants(caller) == (TEAM,)
    assert await caller_auth_profile_grants(caller) == ({"tickets": "bots"},)
    assert await caller_auth_profile_grants(caller) == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("grants", [None, {}, {"tickets": "Build"}])
async def test_catalog_refresh_and_grant_read_share_one_writer_permission(
    monkeypatch: pytest.MonkeyPatch, grants: Mapping[str, str] | None
) -> None:
    key: Final = FakeKeyStore(writer={"fixture": {"token": "fixture", "object_permission_id": "linked"}})
    rows: Final = RustStoreRows(writer={"linked": _permission("linked", grants)})
    install_rust_auth_stores(monkeypatch, RustAuthStores(key=key, object_permission=rows))
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    monkeypatch.setattr(proxy_server, "user_api_key_cache", UserApiKeyCache())
    caller: Final = UserAPIKeyAuth(api_key="fixture", object_permission_id="linked")
    caller.via_virtual_key = True

    refreshed: Final = await MCPRequestHandler.refresh_catalog_authority(caller)
    assert refreshed is not None
    assert await caller_auth_profile_grants(refreshed) == (grants or {},)
    assert key.lookups == [("fixture", True)]
    assert rows.lookups == [("linked", True)]

    rows.writer["linked"] = _permission("linked", TEAM)
    assert await caller_auth_profile_grants(refreshed) == (TEAM,)
    assert rows.lookups == [("linked", True), ("linked", True)]


@pytest.mark.asyncio
@pytest.mark.parametrize("virtual_key", [False, True])
async def test_catalog_refresh_loads_current_policy_for_an_unmanaged_agent(
    monkeypatch: pytest.MonkeyPatch, virtual_key: bool, identity: AgentIdentityBinding
) -> None:
    ordinary: Final = AgentResponse(agent_id="runner", agent_name="Runner", agent_card_params={})
    managed: Final = ordinary.model_copy(update={"identity_managed": True, "identity": identity})
    lookup: Final = AsyncMock(side_effect=[ordinary, managed, RuntimeError("store unavailable")])
    monkeypatch.setattr(
        proxy_server, "prisma_client", MagicMock(**{"writer_db.litellm_agentstable.find_unique": lookup})
    )
    key: Final = FakeKeyStore(writer={"fixture": {"token": "fixture", "agent_id": "runner"}})
    install_rust_auth_stores(monkeypatch, RustAuthStores(key=key))
    caller: Final = UserAPIKeyAuth(api_key="fixture", agent_id="runner")
    caller.via_virtual_key = virtual_key

    refreshed: Final = await MCPRequestHandler.refresh_catalog_authority(caller)
    assert refreshed is not None and refreshed.managed_agent_policy is None
    with pytest.raises(HTTPException) as identity_required:
        await MCPRequestHandler.refresh_catalog_authority(caller)
    assert (identity_required.value.status_code, identity_required.value.detail) == (
        403,
        "This agent requires its bound identity provider token",
    )
    with pytest.raises(HTTPException) as unavailable:
        await MCPRequestHandler.refresh_catalog_authority(caller)
    assert (unavailable.value.status_code, unavailable.value.detail) == (503, "Agent policy could not be loaded")


def test_first_catalog_page_reports_a_policy_read_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    async def slow_agent_row(**_: object) -> AgentResponse:
        await asyncio.sleep(2 * PROXY_DB_LOOKUP_DEADLINE_SECONDS)
        return AgentResponse(agent_id="runner", agent_name="Runner", agent_card_params={})

    prisma: Final = MagicMock(**{"writer_db.litellm_agentstable.find_unique": slow_agent_row})
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)
    monkeypatch.setattr(proxy_server, "general_settings", {"supported_db_objects": []})
    caller: Final = UserAPIKeyAuth(agent_id="runner")
    try:
        with closing(AutojumpClockLoop()) as loop:
            with pytest.raises(MCPError) as failed_page:
                loop.run_until_complete(
                    GatewayOperations().execute(ListToolsRequest(), OperationContext(_caller=caller))
                )
            assert (failed_page.value.error.code, failed_page.value.error.message) == (
                INVALID_REQUEST,
                "Agent policy could not be loaded",
            )
            with pytest.raises(HTTPException) as unavailable:
                loop.run_until_complete(MCPRequestHandler.refresh_catalog_authority(caller))
            assert unavailable.value.status_code == 503
    finally:
        db_lookup_stall_tracker.clear()


@pytest.mark.asyncio
async def test_catalog_refresh_requires_reauthentication_when_an_agent_becomes_ordinary(
    monkeypatch: pytest.MonkeyPatch, identity: AgentIdentityBinding
) -> None:
    ordinary: Final = AgentResponse(agent_id="runner", agent_name="Runner", agent_card_params={})
    managed: Final = ordinary.model_copy(update={"identity_managed": True, "identity": identity})
    lookup: Final = AsyncMock(side_effect=[managed, ordinary])
    monkeypatch.setattr(
        proxy_server, "prisma_client", MagicMock(**{"writer_db.litellm_agentstable.find_unique": lookup})
    )
    context: Final = ManagedAgentContext(agent_id="runner", binding_revision="current", mode="autonomous")
    caller: Final = UserAPIKeyAuth(agent_id="runner").model_copy(update={"managed_agent_context": context})
    admitted: Final = await MCPRequestHandler.refresh_catalog_authority(caller)
    with pytest.raises(HTTPException) as changed:
        await MCPRequestHandler.refresh_catalog_authority(admitted)
    assert (changed.value.status_code, changed.value.detail) == (401, "Agent policy changed. Authenticate again.")


@pytest.mark.asyncio
@pytest.mark.usefixtures("database")
async def test_legacy_agent_cache_absence_checks_the_current_link(monkeypatch: pytest.MonkeyPatch) -> None:
    await proxy_server.user_api_key_cache.async_set_cache(
        key=AuthObject.agent_object_permission("runner"), value="__agent_no_mcp_permission__"
    )
    prisma: Final = MagicMock(
        **{
            "db.litellm_agentstable.find_unique": AsyncMock(return_value=None),
            "writer_db.litellm_agentstable.find_unique": AsyncMock(
                return_value=SimpleNamespace(object_permission_id="perm-team")
            ),
        }
    )
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)
    assert await caller_auth_profile_grants(UserAPIKeyAuth(agent_id="runner")) == (TEAM,)


@pytest.mark.asyncio
@pytest.mark.usefixtures("database")
async def test_a_writer_timeout_on_an_unprojected_grant_records_the_stall() -> None:
    caller: Final = _caller(key=LiteLLM_ObjectPermissionTable(object_permission_id="timeout"), team_id=None)
    db_lookup_stall_tracker.clear()
    try:
        assert await caller_auth_profile_grants(caller) == UnreadableGrants()
        assert db_lookup_stall_tracker.stalled_within(60)
    finally:
        db_lookup_stall_tracker.clear()


@pytest.mark.asyncio
@pytest.mark.usefixtures("database")
@pytest.mark.parametrize("link", ["timeout", "disconnected"])
async def test_fresh_key_permission_store_failure_returns_unreadable_grants(link: str) -> None:
    caller: Final = UserAPIKeyAuth(
        object_permission_id=link, object_permission=_permission(link, {"tickets": "retired"})
    )
    assert await caller_auth_profile_grants(caller) == ({"tickets": "retired"},)
    fresh: Final = caller.model_copy(update={"requires_fresh_policy": True})
    assert await caller_auth_profile_grants(fresh) == UnreadableGrants()


@pytest.mark.asyncio
@pytest.mark.usefixtures("database")
async def test_key_bound_agent_uses_the_same_freshness_for_its_link_and_grant(monkeypatch: pytest.MonkeyPatch) -> None:
    agent_link: Final = "agent-current"
    stale: Final = AsyncMock(return_value=SimpleNamespace(object_permission_id="perm-team"))
    current: Final = AsyncMock(return_value=SimpleNamespace(object_permission_id=agent_link))
    prisma: Final = MagicMock(
        **{
            "db.litellm_agentstable.find_unique": stale,
            "writer_db.litellm_agentstable.find_unique": current,
        }
    )
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)
    caller: Final = UserAPIKeyAuth(agent_id="runner", team_id="payments")
    assert await caller_auth_profile_grants(caller) == ({"tickets": "bots"}, TEAM)
    fresh: Final = caller.model_copy(update={"requires_fresh_policy": True})
    assert await caller_auth_profile_grants(fresh) == ({"tickets": "bots"}, TEAM)


@pytest.fixture
def identity() -> AgentIdentityBinding:
    return AgentIdentityBinding(
        agent_id="runner",
        provider="microsoft_entra",
        tenant_id="tenant",
        client_id="client",
        issuer="issuer",
        service_principal_id="fixture-principal",
        revision="current",
    )


@pytest.mark.asyncio
async def test_catalog_refresh_replaces_a_projected_managed_agent_policy(
    monkeypatch: pytest.MonkeyPatch, identity: AgentIdentityBinding
) -> None:
    admitted: Final = AgentResponse(
        agent_id="runner",
        agent_name="Runner",
        agent_card_params={},
        identity_managed=True,
        identity=identity,
        object_permission=_permission("original", {"tickets": "Build"}).model_dump(),
    )
    current: Final = admitted.model_copy(update={"object_permission": _permission("new", TEAM).model_dump()})
    prisma: Final = MagicMock(**{"writer_db.litellm_agentstable.find_unique": AsyncMock(return_value=current)})
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)
    context: Final = ManagedAgentContext(agent_id="runner", binding_revision="current", mode="autonomous")
    caller: Final = UserAPIKeyAuth(agent_id="runner").model_copy(
        update={"managed_agent_context": context, "managed_agent_policy": admitted, "requires_fresh_policy": True}
    )
    assert await caller_auth_profile_grants(caller) == ({"tickets": "Build"},)
    refreshed: Final = await MCPRequestHandler.refresh_catalog_authority(caller)
    assert await caller_auth_profile_grants(refreshed) == (TEAM,)


@pytest.mark.asyncio
@pytest.mark.parametrize("caller", [UserAPIKeyAuth(agent_id="runner"), UserAPIKeyAuth(team_id="payments")])
async def test_applicable_carriers_without_a_database_are_unreadable(
    monkeypatch: pytest.MonkeyPatch, caller: UserAPIKeyAuth
) -> None:
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    assert await caller_auth_profile_grants(caller) == UnreadableGrants()


@pytest.mark.asyncio
@pytest.mark.usefixtures("database")
@pytest.mark.parametrize(
    ("link", "expected"), [("agent-current", ({"tickets": "bots"},)), ("agent-deleted", UnreadableGrants())]
)
async def test_managed_agent_with_an_unprojected_permission_reads_its_linked_grants(
    link: str, expected: tuple[Mapping[str, str], ...] | UnreadableGrants
) -> None:
    stale: Final = _permission(link, {"tickets": "retired"})
    await proxy_server.user_api_key_cache.async_set_cache(key=AuthObject.object_permission(link), value=stale)
    policy: Final = AgentResponse(
        agent_id="agent", agent_name="agent", agent_card_params={}, object_permission={"object_permission_id": link}
    )
    caller: Final = UserAPIKeyAuth(agent_id=policy.agent_id).model_copy(
        update={"managed_agent_policy": policy, "requires_fresh_policy": True}
    )
    assert await caller_auth_profile_grants(caller) == expected
