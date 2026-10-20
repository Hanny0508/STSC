# One-command reviewer reproduction

The reviewer interface is `reviewer.ps1`. Historical scripts remain available
for audit, but they are not required for the normal reproduction path.

## Safety and scope

- Existing Conda environments are detected and left unchanged unless the
  reviewer explicitly uses the Python bootstrap's `--refresh-existing` flag.
- Model repositories are pinned to immutable revisions. Large weight files are
  accepted only after their frozen SHA-256 checks pass.
- The Llama-2 and TransformerTTS licenses must be accepted explicitly.
- Passwords are entered at a prompt. Passwords and master keys are never printed
  or written into generated commands.
- A media result is not reported as end-to-end secret recovery unless the
  decoder report proves RS recovery, AES-GCM authentication, and exact plaintext
  equality.

## Node A: complete setup

From an ordinary PowerShell window in the project root:

```powershell
.\reviewer.ps1 -Action PlanA
.\reviewer.ps1 -Action SetupA -AcceptModelLicenses
.\reviewer.ps1 -Action QuickTestA
```

`SetupA` creates four isolated environments: core/orchestration, ZTS, ASR, and
TransformerTTS. It downloads the pinned Hugging Face snapshots. For the frozen
TransformerTTS step-90000 checkpoint, either place the authors' archive at
`artifacts\tts_models\bdf06b9_ljspeech_step_90000.zip` or set
`ZSC_TSD_TTS_MODEL_URL`; the extracted config and weights must match the lock.

Where access to Hugging Face is slow, an endpoint may be supplied without
weakening verification:

```powershell
.\reviewer.ps1 -Action SetupA -AcceptModelLicenses -HfEndpoint https://<mirror>
```

The default GPU wheel is PyTorch 2.7.1 CUDA 12.8 from the official index. A
verified local or institutional mirror wheel can be selected explicitly with
`-TorchWheelUrl <wheel-url>`; the installed runtime is recorded by self-test.

## Node B: relay setup

```powershell
.\reviewer.ps1 -Action PlanB
.\reviewer.ps1 -Action SetupB
.\reviewer.ps1 -Action QuickTestB
```

PostgreSQL, FFmpeg, and Tailscale are host services and are deliberately not
installed silently. Their addresses and restricted firewall rules must be
checked against the deployment guide before the two-host test. Run the explicit
host check after every reboot; it detects the case where PostgreSQL starts
before Tailscale and therefore binds only to localhost:

```powershell
.\reviewer.ps1 -Action HostPreflightB -FFmpeg <ffmpeg.exe>
```

If it reports that the Tailscale listener is absent, restart the named
PostgreSQL service once and repeat the same preflight. The check itself never
changes services or firewall rules.

## Run a frozen secure batch on A

```powershell
.\reviewer.ps1 -Action SecureBatchA `
  -BundlePublic <public_commitment.json> `
  -BundlePrivate <sender_private.json>
```

## Run the authenticated RTP/Opus experiment

Start A first:

```powershell
.\reviewer.ps1 -Action HostPreflightA -FFmpeg <ffmpeg.exe>
.\reviewer.ps1 -Action MediaA `
  -SourceWav <carrier.wav> `
  -FFmpeg <ffmpeg.exe>
```

A prints a fresh `session_id` and `correlation_id`. Then run B:

```powershell
.\reviewer.ps1 -Action RelayB `
  -SessionId <SESSION_ID_FROM_A> `
  -CorrelationId <CORRELATION_ID_FROM_A> `
  -FFmpeg <ffmpeg.exe>
```

The measured speech path is B-to-A Opus over RTP/UDP. The carrier provisioning
step is separate and is not labeled VoIP; SIP signaling is not implemented.
Conda is invoked with `--no-capture-output`, so the session identifiers and
progress are printed immediately instead of appearing only after completion.

## Full post-VoIP secret recovery batch

The single-carrier command above proves the authenticated media path only. It
cannot prove Reed-Solomon reconstruction or exact secret recovery. For that
claim, A freezes every `carrier.wav` already present in one secure v1.4 batch,
records their hashes before network start, sends every carrier through B, runs
Whisper and the fragment receiver on every post-network WAV, and only then runs
RS and AES-GCM finalization.

Start A first. It immediately prints `batch_id` and `carrier_count`:

```powershell
.\reviewer.ps1 -Action BatchMediaA `
  -SecureBatchRoot <secure_imdb_batch_v1_4_DIRECTORY> `
  -BundlePrivate <secure_bundle_private.json> `
  -FFmpeg <ffmpeg.exe> `
  -ASRPython <zsc-tsd-asr\python.exe> `
  -ZTSPython <zsc-tsd-zts\python.exe> `
  -ASRModel <whisper-small.en_DIRECTORY>
```

Then start one persistent relay command on B; it handles the whole batch:

```powershell
.\reviewer.ps1 -Action BatchRelayB `
  -BatchId <BATCH_ID_FROM_A> `
  -ExpectedSessions <CARRIER_COUNT_FROM_A> `
  -FFmpeg <ffmpeg.exe>
```

Only `ZSC_TSD_VOIP_SECURE_BATCH_END_TO_END_PASS` authorizes the paper claim of
post-VoIP recovery. It requires at least 22 unique shards, successful RS
reconstruction, successful AES-GCM authentication, and exact plaintext match.
