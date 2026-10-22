BEGIN;

CREATE TABLE IF NOT EXISTS zsc_control_events (
    event_id UUID PRIMARY KEY,
    session_id UUID NOT NULL,
    actor_id TEXT NOT NULL CHECK (length(actor_id) BETWEEN 1 AND 128),
    sequence_no BIGINT NOT NULL CHECK (sequence_no > 0),
    event_type TEXT NOT NULL CONSTRAINT zsc_control_events_event_type_check CHECK (
        event_type IN (
            'SESSION_OPENED',
            'CARRIER_PREPARED',
            'RELAY_MEDIA_VERIFIED',
            'RELAY_RTP_STARTED',
            'RELAY_RTP_FINISHED',
            'RECEIVER_RTP_READY',
            'RECEIVER_MEDIA_CAPTURED',
            'RECEIVER_DECODED',
            'RECEIVER_SYNC_VERIFIED',
            'RETRANSMIT_REQUESTED',
            'RETRANSMIT_STARTED',
            'RETRANSMIT_FINISHED',
            'SESSION_COMPLETED',
            'SESSION_ABORTED'
        )
    ),
    correlation_id TEXT NOT NULL CHECK (length(correlation_id) BETWEEN 1 AND 256),
    issued_at TIMESTAMPTZ NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL CHECK (expires_at > issued_at),
    event_nonce TEXT NOT NULL CHECK (length(event_nonce) BETWEEN 16 AND 128),
    body JSONB NOT NULL CHECK (jsonb_typeof(body) = 'object'),
    hmac_sha256 CHAR(64) NOT NULL CHECK (hmac_sha256 ~ '^[0-9a-f]{64}$'),
    received_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    UNIQUE (session_id, actor_id, sequence_no),
    UNIQUE (session_id, event_nonce)
);

-- Idempotent v1 -> v2 event vocabulary migration for databases initialized
-- before RECEIVER_RTP_READY was introduced.  The explicit readiness event
-- removes the RTP sender/receiver race without weakening existing checks.
ALTER TABLE zsc_control_events
    DROP CONSTRAINT IF EXISTS zsc_control_events_event_type_check;
ALTER TABLE zsc_control_events
    ADD CONSTRAINT zsc_control_events_event_type_check CHECK (
        event_type IN (
            'SESSION_OPENED',
            'CARRIER_PREPARED',
            'RELAY_MEDIA_VERIFIED',
            'RELAY_RTP_STARTED',
            'RELAY_RTP_FINISHED',
            'RECEIVER_RTP_READY',
            'RECEIVER_MEDIA_CAPTURED',
            'RECEIVER_DECODED',
            'RECEIVER_SYNC_VERIFIED',
            'RETRANSMIT_REQUESTED',
            'RETRANSMIT_STARTED',
            'RETRANSMIT_FINISHED',
            'SESSION_COMPLETED',
            'SESSION_ABORTED'
        )
    );

CREATE INDEX IF NOT EXISTS zsc_control_events_session_time_idx
    ON zsc_control_events (session_id, received_at);

CREATE INDEX IF NOT EXISTS zsc_control_events_lookup_idx
    ON zsc_control_events (session_id, event_type, correlation_id, received_at);

COMMENT ON TABLE zsc_control_events IS
    'Append-only HMAC-authenticated ZSC-TSD synchronization metadata. Secret payload material is forbidden.';

REVOKE UPDATE, DELETE, TRUNCATE ON zsc_control_events FROM PUBLIC;

COMMIT;
