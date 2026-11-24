SET LOCAL lock_timeout = '10s';

CREATE TABLE IF NOT EXISTS "LiteLLM_SpendLogReplication" (
    "id" INTEGER PRIMARY KEY CHECK ("id" = 1),
    "region" TEXT NOT NULL,
    "enabled" BOOLEAN NOT NULL DEFAULT true
);

CREATE TABLE IF NOT EXISTS "LiteLLM_SpendLogOutbox" (
    "id" BIGSERIAL PRIMARY KEY,
    "row" JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS "LiteLLM_ReplicatedSpendLogs" (
    "request_id" TEXT NOT NULL,
    "call_type" TEXT NOT NULL,
    "api_key" TEXT NOT NULL DEFAULT '',
    "spend" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "total_tokens" INTEGER NOT NULL DEFAULT 0,
    "prompt_tokens" INTEGER NOT NULL DEFAULT 0,
    "completion_tokens" INTEGER NOT NULL DEFAULT 0,
    "startTime" TIMESTAMP(3) NOT NULL,
    "endTime" TIMESTAMP(3) NOT NULL,
    "request_duration_ms" INTEGER,
    "completionStartTime" TIMESTAMP(3),
    "model" TEXT NOT NULL DEFAULT '',
    "model_id" TEXT,
    "model_group" TEXT,
    "custom_llm_provider" TEXT,
    "user" TEXT,
    "end_user" TEXT,
    "team_id" TEXT,
    "organization_id" TEXT,
    "status" TEXT,
    "litellm_call_id" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL,
    "updated_at" TIMESTAMP(3) NOT NULL,
    "source_region" TEXT NOT NULL,
    PRIMARY KEY ("request_id", "startTime", "source_region")
);

CREATE INDEX IF NOT EXISTS "LiteLLM_ReplicatedSpendLogs_startTime_idx"
ON "LiteLLM_ReplicatedSpendLogs" ("startTime");

CREATE TABLE IF NOT EXISTS "LiteLLM_FleetBudgetSpend" (
    "request_id" TEXT NOT NULL,
    "startTime" TIMESTAMP(3) NOT NULL,
    "source_region" TEXT,
    "api_key" TEXT,
    "team_id" TEXT,
    "user" TEXT,
    "end_user" TEXT,
    "spend" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "created_at" TIMESTAMP(3) NOT NULL,
    "updated_at" TIMESTAMP(3) NOT NULL,
    PRIMARY KEY ("request_id", "startTime")
);

CREATE INDEX IF NOT EXISTS "LiteLLM_FleetBudgetSpend_api_key_startTime_idx"
ON "LiteLLM_FleetBudgetSpend" ("api_key", "startTime") WHERE "source_region" IS NOT NULL;
CREATE INDEX IF NOT EXISTS "LiteLLM_FleetBudgetSpend_team_id_startTime_idx"
ON "LiteLLM_FleetBudgetSpend" ("team_id", "startTime") WHERE "source_region" IS NOT NULL;
CREATE INDEX IF NOT EXISTS "LiteLLM_FleetBudgetSpend_user_startTime_idx"
ON "LiteLLM_FleetBudgetSpend" ("user", "startTime") WHERE "source_region" IS NOT NULL;
CREATE INDEX IF NOT EXISTS "LiteLLM_FleetBudgetSpend_end_user_startTime_idx"
ON "LiteLLM_FleetBudgetSpend" ("end_user", "startTime") WHERE "source_region" IS NOT NULL;

CREATE TABLE IF NOT EXISTS "LiteLLM_FleetKeyBudgetReset" (
    "api_key" TEXT PRIMARY KEY,
    "reset_at" TIMESTAMP(3) NOT NULL
);

CREATE OR REPLACE FUNCTION litellm_track_replicated_budget_spend() RETURNS trigger
LANGUAGE plpgsql SET search_path FROM CURRENT AS $$
BEGIN
    -- data-migration-ok: runs for the replica that caused the trigger
    INSERT INTO "LiteLLM_FleetBudgetSpend" AS ledger
        ("request_id", "startTime", "source_region", "api_key", "team_id", "user", "end_user",
         "spend", "created_at", "updated_at")
    SELECT NEW.request_id, NEW."startTime", NEW.source_region, NEW.api_key, NEW.team_id, NEW."user",
           NEW.end_user, NEW.spend, NEW.created_at, NEW.updated_at
    WHERE NOT EXISTS (
        SELECT 1 FROM "LiteLLM_SpendLogs" AS local_log
        WHERE local_log.request_id = NEW.request_id AND local_log."startTime" = NEW."startTime"
    )
    ON CONFLICT ("request_id", "startTime") DO UPDATE
    SET "api_key" = EXCLUDED.api_key, "team_id" = EXCLUDED.team_id, "user" = EXCLUDED."user",
        "end_user" = EXCLUDED.end_user, "spend" = EXCLUDED.spend, "updated_at" = EXCLUDED.updated_at
    WHERE ledger.source_region = EXCLUDED.source_region
        AND ledger.created_at = EXCLUDED.created_at AND ledger.updated_at < EXCLUDED.updated_at;
    RETURN NULL;
END;
$$;

CREATE OR REPLACE VIEW "LiteLLM_SpendLogReporting" AS
SELECT
    local_log."request_id", local_log."call_type", local_log."api_key", local_log."spend",
    local_log."total_tokens", local_log."prompt_tokens", local_log."completion_tokens", local_log."startTime",
    local_log."endTime", local_log."request_duration_ms", local_log."completionStartTime", local_log."model",
    local_log."model_id", local_log."model_group", local_log."custom_llm_provider", local_log."user",
    local_log."team_id", local_log."organization_id", local_log."status", local_log."litellm_call_id",
    local_log."created_at", local_log."updated_at", local_log."api_base", local_log."metadata",
    local_log."cache_hit", local_log."cache_key", local_log."request_tags", local_log."end_user",
    local_log."requester_ip_address", local_log."messages", local_log."response", local_log."session_id",
    local_log."mcp_namespaced_tool_name", local_log."billing_agent_id", local_log."agent_id", local_log."proxy_server_request",
    NULL::text AS source_region
FROM "LiteLLM_SpendLogs" AS local_log
UNION ALL
SELECT
    replica."request_id", replica."call_type", replica."api_key", replica."spend",
    replica."total_tokens", replica."prompt_tokens", replica."completion_tokens", replica."startTime",
    replica."endTime", replica."request_duration_ms", replica."completionStartTime", replica."model",
    replica."model_id", replica."model_group", replica."custom_llm_provider", replica."user",
    replica."team_id", replica."organization_id", replica."status", replica."litellm_call_id",
    replica."created_at", replica."updated_at", NULL::text, NULL::jsonb,
    NULL::text, NULL::text, NULL::jsonb, replica."end_user",
    NULL::text, NULL::jsonb, NULL::jsonb, NULL::text,
    NULL::text, NULL::text, NULL::text, NULL::jsonb,
    replica.source_region
FROM "LiteLLM_ReplicatedSpendLogs" AS replica
WHERE NOT EXISTS (
    SELECT 1 FROM "LiteLLM_SpendLogs" AS local_log WHERE local_log.request_id = replica.request_id
    AND local_log."startTime" = replica."startTime"
);

CREATE OR REPLACE FUNCTION litellm_capture_spend_log() RETURNS trigger
LANGUAGE plpgsql SET search_path FROM CURRENT AS $$
BEGIN
    IF current_setting('litellm.spend_log_replica', true) IS DISTINCT FROM 'on'
        AND EXISTS (SELECT 1 FROM "LiteLLM_SpendLogReplication" WHERE "id" = 1) THEN
        -- data-migration-ok: tracks only rows changed by the triggering statement
        INSERT INTO "LiteLLM_FleetBudgetSpend" AS ledger
            ("request_id", "startTime", "created_at", "updated_at")
        SELECT DISTINCT ON (log.request_id, log."startTime")
            log.request_id, log."startTime", log.created_at, log.updated_at
        FROM new_spend_logs AS log
        ON CONFLICT ("request_id", "startTime") DO UPDATE SET "source_region" = NULL,
            "api_key" = NULL, "team_id" = NULL, "user" = NULL, "end_user" = NULL, "spend" = 0
        WHERE ledger.source_region IS NOT NULL;
    END IF;
    IF current_setting('litellm.spend_log_replica', true) IS DISTINCT FROM 'on'
        AND EXISTS (SELECT 1 FROM "LiteLLM_SpendLogReplication" WHERE "id" = 1 AND "enabled") THEN
        -- data-migration-ok: runs only in the trigger on rows changed by the triggering statement
        INSERT INTO "LiteLLM_SpendLogOutbox" ("row")
        SELECT jsonb_build_object(
            'request_id', log.request_id,
            'call_type', log.call_type,
            'api_key', CASE WHEN log.api_key ~ '^[[:xdigit:]]{64}$' THEN log.api_key ELSE '' END,
            'spend', log.spend,
            'total_tokens', log.total_tokens,
            'prompt_tokens', log.prompt_tokens,
            'completion_tokens', log.completion_tokens,
            'startTime', log."startTime",
            'endTime', log."endTime",
            'request_duration_ms', log.request_duration_ms,
            'completionStartTime', log."completionStartTime",
            'model', log.model,
            'model_id', log.model_id,
            'model_group', log.model_group,
            'custom_llm_provider', log.custom_llm_provider,
            'user', log."user",
            'end_user', log.end_user,
            'team_id', log.team_id,
            'organization_id', log.organization_id,
            'status', log.status,
            'litellm_call_id', log.litellm_call_id,
            'created_at', log.created_at,
            'updated_at', log.updated_at
        ) FROM new_spend_logs AS log;
    END IF;
    RETURN NULL;
END;
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgrelid = '"LiteLLM_ReplicatedSpendLogs"'::regclass AND tgname = 'litellm_track_replicated_budget_spend'
    ) THEN
        CREATE TRIGGER litellm_track_replicated_budget_spend
        AFTER INSERT OR UPDATE ON "LiteLLM_ReplicatedSpendLogs"
        FOR EACH ROW EXECUTE PROCEDURE litellm_track_replicated_budget_spend();
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgrelid = '"LiteLLM_SpendLogs"'::regclass AND tgname = 'litellm_capture_spend_log_insert'
    ) THEN
        CREATE TRIGGER litellm_capture_spend_log_insert
        AFTER INSERT ON "LiteLLM_SpendLogs" REFERENCING NEW TABLE AS new_spend_logs
        FOR EACH STATEMENT EXECUTE PROCEDURE litellm_capture_spend_log();
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgrelid = '"LiteLLM_SpendLogs"'::regclass AND tgname = 'litellm_capture_spend_log_update'
    ) THEN
        CREATE TRIGGER litellm_capture_spend_log_update
        AFTER UPDATE ON "LiteLLM_SpendLogs" REFERENCING NEW TABLE AS new_spend_logs
        FOR EACH STATEMENT EXECUTE PROCEDURE litellm_capture_spend_log();
    END IF;
END;
$$;
