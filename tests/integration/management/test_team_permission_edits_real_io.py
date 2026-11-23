import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest
from fastapi import Request
from litellm_proxy_extras.prisma_toolchain import prisma_migrate_deploy_timeout, run_prisma

from litellm.proxy import proxy_server
from litellm.proxy._types import LitellmUserRoles, NewTeamRequest, ProxyException, UserAPIKeyAuth
from litellm.proxy.management_endpoints.team_endpoints import new_team
from tests.integration._support.database import scratch_database
from tests.integration._support.prisma_clients import connected_proxy_prisma


@pytest.fixture(scope="module")
def team_database() -> Iterator[str]:
    schema: Final = Path(__file__).resolve().parents[3] / "litellm/proxy-extras/litellm_proxy_extras/schema.prisma"
    with scratch_database() as url:
        run_prisma(
            (sys.executable, "-I", "-m", "prisma", "migrate", "deploy", "--schema", str(schema)),
            env={**os.environ, "DATABASE_URL": url},
            timeout=prisma_migrate_deploy_timeout(),
        )
        with psycopg.connect(url) as db:
            db.execute('INSERT INTO "LiteLLM_MCPServerTable" (server_id) VALUES (%s)', ("tickets",))
        yield url


def _one_connection(url: str) -> str:
    parts: Final = urlsplit(url)
    return urlunsplit(parts._replace(query="&".join(filter(None, (parts.query, "connection_limit=1&pool_timeout=2")))))


_ADMIN: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)


@pytest.mark.asyncio
async def test_create_with_one_connection_commits_or_rolls_back_all_related_rows(
    team_database: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    connection: Final = _one_connection(team_database)
    monkeypatch.setenv("DATABASE_URL", connection)
    request: Final = Request({"type": "http", "method": "POST", "path": "/team/new", "headers": []})
    async with connected_proxy_prisma(connection) as client:
        monkeypatch.setattr(proxy_server, "prisma_client", client)
        with psycopg.connect(team_database, autocommit=True) as db:
            db.execute("""ALTER TABLE "LiteLLM_TeamTable" ADD CHECK (team_id <> 'declined-create')""")
            db.execute(
                """ALTER TABLE "LiteLLM_ObjectPermissionTable" ADD CHECK (NOT ('declined-tool' = ANY(agents)))"""
            )
            db.execute(
                """INSERT INTO "LiteLLM_AccessGroupTable" (access_group_id, access_group_name) VALUES ('g', 'g')"""
            )
            db.execute(
                """ALTER TABLE "LiteLLM_AccessGroupTable"
                   ADD CHECK (NOT ('declined-mirror' = ANY(assigned_team_ids)))"""
            )
            args: Final = {
                "access_group_ids": ["g"],
                "team_member_budget": 7,
                "budget_duration": "1d",
                "team_member_rollover_max_budget": 10,
                "model_aliases": {"short": "one"},
                "object_permission": {"mcp_tool_permissions": {"tickets": ["list"]}},
            }

            def counts() -> tuple | None:
                return db.execute(
                    'SELECT (SELECT count(*) FROM "LiteLLM_ModelTable"), (SELECT count(*) FROM "LiteLLM_BudgetTable"), '
                    '(SELECT count(*) FROM "LiteLLM_ObjectPermissionTable"), (SELECT count(*) FROM "LiteLLM_TeamTable")'
                ).fetchone()

            await new_team(NewTeamRequest.model_validate({**args, "team_id": "ordinary"}), request, _ADMIN, None)
            assert db.execute(
                "SELECT b.max_budget, b.rollover_max_budget, b.budget_duration, m.aliases, p.mcp_tool_permissions "
                'FROM "LiteLLM_TeamTable" t '
                "JOIN \"LiteLLM_BudgetTable\" b ON b.budget_id = t.metadata->>'team_member_budget_id' "
                'JOIN "LiteLLM_ModelTable" m ON m.id = t.model_id '
                'JOIN "LiteLLM_ObjectPermissionTable" p USING (object_permission_id) WHERE t.team_id = %s',
                ("ordinary",),
            ).fetchone() == (7, 10, "1d", {"short": "one"}, {"tickets": ["list"]})
            before: Final = counts()
            for overrides, status in (
                ({"team_id": "declined-create"}, "500"),
                ({"team_id": "invalid-cap", "team_member_rollover_max_budget": 6}, "400"),
                ({"team_id": "declined-permission", "object_permission": {"agents": ["declined-tool"]}}, "500"),
                ({"team_id": "declined-mirror"}, "500"),
            ):
                with pytest.raises(ProxyException) as error:
                    await new_team(NewTeamRequest.model_validate({**args, **overrides}), request, _ADMIN, None)
                assert error.value.code == status
                assert counts() == before
            assert db.execute(
                """SELECT assigned_team_ids FROM "LiteLLM_AccessGroupTable" WHERE access_group_id = 'g'"""
            ).fetchone() == (["ordinary"],)
