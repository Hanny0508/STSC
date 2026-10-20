# STSC PostgreSQL/HMAC Control Channel v1

This module is the authenticated synchronization channel for the A/B RTP/Opus
testbed. It transports metadata only. Secret plaintext, AES keys, ciphertext,
RS bytes, hidden frames, stegotext, sender token IDs, and reference transcripts
must never be written to this database.

## Security properties

- Append-only PostgreSQL event log.
- Per-session, per-actor HMAC keys derived from a pre-shared control master key
  with HKDF-SHA256.
- HMAC-SHA256 over a domain-separated canonical UTF-8 JSON envelope.
- Unique event UUID, `(session, actor, sequence)` and `(session, nonce)` prevent
  database replay.
- Issuance and expiry timestamps are authenticated and checked by readers.
- Unknown, expired, future, malformed, unsigned, or modified events fail closed.
- The control master key is independent of the AES-GCM message key.

## Install the lightweight control dependency

On A and B, use a small Python environment:

```powershell
python -m venv E:\ZSC-TSD-VoIP\control-venv
E:\ZSC-TSD-VoIP\control-venv\Scripts\python.exe -m pip install --upgrade pip
E:\ZSC-TSD-VoIP\control-venv\Scripts\python.exe -m pip install -r requirements\control.txt
```

The source tree must also be available on both hosts, or the package can be
installed with `python -m pip install -e .`.

## Secrets and connection variables

Generate the control master key once on A, then transfer it to B through a
separate authenticated administrative path. Do not place it in SQL or Git.

```powershell
python -c "import base64,secrets;print(base64.b64encode(secrets.token_bytes(32)).decode())"
```

Set the same key on A and B for the experiment process only:

```powershell
$env:ZSC_TSD_CONTROL_MASTER_KEY_B64='<BASE64 KEY>'
$env:ZSC_TSD_CONTROL_DSN='host=192.0.2.11 port=55432 dbname=zsc_tsd_control user=zsc_tsd_app connect_timeout=20 sslmode=disable'
```

Do not paste either secret into an experiment report or console transcript.

## Initialize schema once on B

```powershell
python tools\zts_control_channel_cli.py init-db
```

PostgreSQL should listen only on `127.0.0.1:55432` and B's Tailscale address
`192.0.2.11:55432`. The Windows firewall and `pg_hba.conf` should allow
only A (`192.0.2.10/32`) and B itself.

## CPU-only validation

```powershell
python -m unittest tests.test_control_hmac -v
python -m py_compile tools\zts_control_channel_cli.py
```

## Minimal A to B signed-event smoke

Create a session on A:

```powershell
$session = python tools\zts_control_channel_cli.py new-session
```

Publish an authenticated session record on A (replace all hashes with real
64-hex fingerprints):

```powershell
python tools\zts_control_channel_cli.py publish `
  --session-id $session `
  --actor-id zsc-voip-a `
  --sequence-no 1 `
  --event-type SESSION_OPENED `
  --correlation-id session `
  --body '{"protocol":"ZSC-TSD-SECURE-FRAGMENT-v1","zts_manifest_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","dataset_sha256":"f8a3671e6d7275aa256c5676da6e85284e17665eccd333de8c4903d2db3b5704","rtp":{"codec":"opus","bitrate_bps":24000,"frame_duration_ms":20,"payload_type":111}}'
```

Read and verify it on B:

```powershell
python tools\zts_control_channel_cli.py list --session-id $session --event-type SESSION_OPENED
```

The CLI verifies the HMAC again after reading the JSONB record. A modified row
must raise `ControlEventSignatureError`.

## Integration contract

For every carrier:

1. A inserts `CARRIER_PREPARED` with exact shared prompt, prompt hash, source
   WAV hash, shard index, attempt, and RTP profile.
2. B reads the event and verifies HMAC, expiry, sequence, prompt metadata and
   WAV hash.
3. B inserts relay events around the real RTP/Opus FFmpeg process.
4. A starts its receiver and inserts `RECEIVER_RTP_READY`; B must not transmit
   before this authenticated readiness transition.
5. A records RTP, inserts `RECEIVER_MEDIA_CAPTURED`, then runs ASR/candidates/
   ZTS receiver.
6. A inserts `RECEIVER_DECODED` with pass/erasure and report hashes, but no
   recovered secret bytes.
7. The next retry or shard begins only after the signed state transition is
   visible.

`RECEIVER_RTP_READY` was added by the idempotent v2 schema migration. Existing
event rows are retained; only the allowed event vocabulary constraint changes.
