# Reviewer Guide

## 0. One-command path (recommended)

Do not begin by selecting historical files under `tools/`. Use the root wrapper:

```powershell
.\reviewer.ps1 -Action PlanA
.\reviewer.ps1 -Action SetupA -AcceptModelLicenses
.\reviewer.ps1 -Action QuickTestA
```

This creates isolated environments and provisions only model snapshots that
match `configs/reviewer_models_v1.json`. Node B uses `PlanB`, `SetupB`, and
`QuickTestB`. Full commands are documented in
`docs/ONE_CLICK_REPRODUCTION.md`.

## 1. CPU-only integrity tests

```powershell
python -m pip install -e ".[dev,control]"
python -m pytest -q
```

These tests do not download models. They verify the cryptographic envelope,
CRC framing, HMAC event protocol, configuration inheritance, transport labels,
and RTP/Opus command profile.

## 2. Inspect the frozen experiment profile

```powershell
python -m zsc_tsd preflight --profile paper_r1
python -m zsc_tsd show-config local_e2e
python -m zsc_tsd show-config voip
```

## 3. Full model path

The full run needs the model directories already referenced by the frozen
profiles and three isolated environments:

```text
zsc-tsd-zts   ZTS sender/receiver and orchestration
zsc-tsd-tts   TransformerTTS worker
zsc-tsd-asr   Whisper worker
```

Run the frozen secure batch through the stable package entry point:

```powershell
python -m zsc_tsd secure-imdb-batch `
  --bundle-public <public_commitment.json> `
  --bundle-private <sender_private.json>
```

The runner refuses to overwrite prior outputs and validates frozen source
hashes before using private bundle material.

## 4. Two-host scope

Node A is the model/receiver host. Node B is the PostgreSQL and RTP relay host.
The media path is RTP/UDP with Opus over Tailscale. The control path is
PostgreSQL/TCP with SCRAM-SHA-256 plus HKDF-HMAC-SHA-256 signed events.

The artifact does not claim SIP signaling. The database contains only control
metadata and never carries hidden payload bytes.

The executable v2 workflow is:

```text
A: dual-channel-run
   SESSION_OPENED -> CARRIER_PREPARED -> wait for B

B: dual-channel-relay
   verify/download committed carrier -> RELAY_MEDIA_VERIFIED

A: start RTP receiver -> RECEIVER_RTP_READY
B: Opus/RTP transmission -> RELAY_RTP_STARTED/FINISHED
A: capture/decode -> RECEIVER_MEDIA_CAPTURED/RECEIVER_DECODED
A: SESSION_COMPLETED
```

The A-to-B carrier download is explicitly labeled experimental provisioning,
not VoIP traffic. Every event is read back from PostgreSQL and HMAC-verified.
The v2 schema migration must be applied once before running this workflow.
Before each two-host run, use `HostPreflightA` and `HostPreflightB` from
`reviewer.ps1`. These checks validate the exact Tailscale addresses, the
PostgreSQL listener, the FFmpeg libopus encoder and RTP/UDP protocols, and A's
required free ports. They fail with a concrete repair instruction and do not
silently reconfigure the host.

## 5. Result interpretation

The primary correctness conditions are:

```text
Reed-Solomon reconstruction = true
AES-GCM authentication      = true
plaintext exact match       = true
```

Carrier acceptance rate is reported separately and includes failed attempts.
Audio quality and ASR measurements characterize the speech transport layer;
they are not substitutes for text-domain steganographic security evaluation.

A result ending in `MEDIA_PASS` proves only the two-host media and authenticated
control paths. Only `END_TO_END_PASS` asserts secret recovery, and only when
all four cryptographic/recovery conditions in the decoder report are true.

The reviewer-facing full path is `BatchMediaA` on A plus one `BatchRelayB`
process on B. The A-side selection manifest is written before the first network
session and includes every frozen carrier WAV and sender-metadata hash. Thus
network failures and ASR/fragment erasures remain in the reported denominator;
the runner cannot choose carriers after observing their network outcome.
