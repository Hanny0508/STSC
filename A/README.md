# STSC Research Artifact

![STSC dual-channel A-B-C workflow](docs/framework.png)

STSC is a generative **text-steganography system with synthesized-speech
transport**. Secret bits affect token selection during text generation; the
resulting text is synthesized, transported as speech, transcribed by ASR, and
decoded at the linguistic layer. No secret bits are inserted directly into WAV
samples or codec parameters.

This repository is the reviewer-facing integration package for the IEEE IoT
Journal revision. It keeps the original experimental scripts for auditability
while exposing a small, stable package and command-line interface.

## What has been validated

The following paths have completed real experiments on the authors' testbed:

1. **Secure IMDB speech-transport batch**
   - AES-256-GCM authenticated encryption;
   - Reed-Solomon erasure recovery;
   - 30 authenticated 40-bit ZTS shard frames;
   - TransformerTTS LJSpeech step-90000 synthesis;
   - Whisper small.en ASR;
   - payload-independent candidate generation and isolated fragment receiver;
   - 24/30 shards recovered from 137 carriers;
   - RS reconstruction, AES-GCM authentication, and exact plaintext match.

2. **Two-host RTP/Opus media path**
   - node B sends Opus in RTP/UDP to node A over Tailscale;
   - node A records 16-kHz PCM WAV;
   - pre-network and post-network Whisper transcripts matched in the smoke run;
   - downstream ZTS fragment decoding completed successfully.

3. **Bidirectional authenticated control path**
   - PostgreSQL 17 on node B, restricted to localhost and the B Tailscale IP;
   - SCRAM-SHA-256 database authentication;
   - canonical JSON events authenticated with per-session/per-actor
     HKDF-HMAC-SHA-256 keys;
   - event expiry, replay protection, and fail-closed tamper checks;
   - A→B `SESSION_OPENED` and B→A signed acknowledgement independently verified.

The validated network experiment uses **RTP/UDP + Opus**, not SIP signaling.
The code and paper must not describe it as a SIP call.

## Prototype scope

This package is a research prototype. It is implemented for, and validated on, a
two-server testbed: one node encrypts the payload and synthesizes the speech
carriers, the second node relays them over a real RTP/UDP--Opus path, and the
first node performs the receiver-side decoding. The deployment is code-level, on
two physical servers; it is not a deployment into a production VoIP service.
Environment-specific details such as absolute paths, device selection and codec
availability may need adaptation by a practitioner. The runnable end-to-end
scripts and the recorded reference outputs are included so that the complete
flow can be reproduced or debugged.

## Synchronization gate

Sender and receiver must hold an identical synchronization profile Pi: the model
checkpoint, its tokenizer, the generation parameters, the implementation
sources, the runtime versions and the cover-corpus fingerprint. The profile is
never transmitted during a session; only hashes travel, and the receiver
re-derives the prompt from its own pinned corpus.

The gate runs inside the main recovery path, before the decoder is invoked, and
its verdict is published on the authenticated control channel and folded into the
end-to-end success criterion, so a session cannot report a recovery while the two
ends disagree.

    python tools/freeze_manifest.py --root . --output artifacts/freezes/my_freeze/zts_manifest_r3.json
    python tools/verify_sync.py --root . --manifest artifacts/freezes/my_freeze/zts_manifest_r3.json

The sensitivity check is reproduced with `tools/verify_sync.py`: perturbing any
single component (the checkpoint, the tokenizer, a generation parameter, an
implementation source, a software version, or the manifest's own stored
fingerprint) is rejected, while the unmodified profile passes.

## Start here

```text
README.md                         Project scope and verified claims
REVIEWER_GUIDE.md                Minimal reproduction commands
configs/                         Frozen experiment profiles
zsc_tsd/                         Stable integration package
experiments/                     Reviewer-facing experiment entry points
tests/                           CPU tests plus optional model tests
tools/                           Frozen and historical experiment utilities
artifacts/                       Generated evidence (not Python source)
docs/CONTROL_CHANNEL_V1.md       PostgreSQL/HMAC protocol
reviewer.ps1                     One-command setup, verification, and runs
```

For a new reviewer machine, begin with:

```powershell
.\reviewer.ps1 -Action PlanA
.\reviewer.ps1 -Action SetupA -AcceptModelLicenses
.\reviewer.ps1 -Action QuickTestA
```

Node B uses `PlanB`, `SetupB`, and `QuickTestB`. See
`docs/ONE_CLICK_REPRODUCTION.md` for the frozen model lock, mirror selection,
secure batch action, and the two one-command physical-node actions.

The many versioned files under `tools/` are intentionally preserved. They are
the development and freeze history behind the paper results. New code should
import `zsc_tsd` or call `python -m zsc_tsd`; it should not import a random
`*_dev.py` file directly.

## Stable package map

```text
zsc_tsd/
  cli.py                 one reviewer-facing CLI
  config.py              YAML profile inheritance and validation
  manifest.py            SHA-256 provenance manifests
  report.py              consistent public/private/receiver run layout
  control/               PostgreSQL + HKDF-HMAC authenticated events
  crypto/                AES-256-GCM, keys, CRC framing
  dual_channel/          A coordinator, B relay, provisioning, evidence
  pipeline/              sender, receiver, local integration facades
  speech/                TransformerTTS, Whisper, normalization, metrics
  transport/             local adapter and RTP/Opus command construction
  zts/                   verified ZTS encoder, decoder, adapter, manifest
```

## Environment design

Incompatible heavy dependencies are intentionally separated:

| Environment | Responsibility |
|---|---|
| `zsc-tsd-zts` | ZTS model, orchestration, candidate receiver |
| `zsc-tsd-tts` | TensorFlow TransformerTTS synthesis |
| `zsc-tsd-asr` | Whisper ASR |
| evaluation environment | NISQA and text-steganalysis experiments |
| B control venv | PostgreSQL/HMAC verification only |

The paper uses **DAQ-TTS** as the name of the combined speech-generation
architecture. The reproduced end-to-end path in this artifact records its
actual backend precisely: public TransformerTTS LJSpeech step-90000. This
avoids presenting the public checkpoint as a newly trained model.

## Not bundled

Model checkpoints and datasets are intentionally not shipped in this repository (code and configs for every component are included):

| Component | Missing asset | How to obtain |
|---|---|---|
| ZTS | `legacy/zero-shot-GLS/tmp_saves/local_models/llama-2-7b-chat-gptq-4bit-32g-actorder/` | `tools/download_models.py` |
| ZTS prompts | `legacy/zero-shot-GLS/zgls_data/imdb.csv` (public IMDb sentence CSV: `sentence_id,plaintext`) | `tools/collect_d3_dataset.py` |
| TransformerTTS (DAQ-TTS front end) | `artifacts/tts_models/bdf06b9_ljspeech_step_90000/{config.yaml,model_weights.hdf5}` | `tools/setup_daqtts_reviewer.py` |
| FACodec (DAQ-TTS back end) | `ns3_facodec_encoder_v2.bin`, `ns3_facodec_decoder_v2.bin` | `tools/setup_daqtts_reviewer.py` (Hugging Face) |
| ASR | `artifacts/asr_models/whisper-small.en/` | `tools/download_models.py` |
| NISQA | `legacy/NISQA/weights/*.tar` | download from the NISQA release |

The reference voice for FACodec (`artifacts/facodec_reference/official_audio_2.wav`) is included. Tests that require absent assets skip automatically, so `python -m pytest -q` passes on a clean checkout.

## Installation

Python 3.10 is the frozen orchestration version.

**PyTorch is mandatory.** Install it before the editable package, choosing
the wheel that matches the local CUDA version:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e ".[dev,control]"
```

> The editable install alone is **not** sufficient to run the test suite.
> Without PyTorch, `python -m pytest` fails during collection with
> `ModuleNotFoundError: No module named 'torch'`, because
> `zsc_tsd/zts/logits_memo.py` imports PyTorch at module scope.
> The heavyweight packages for ZTS, TTS, ASR and FACodec are declared under
> `requirements/` and are installed per environment; see
> `## Environment design` above.

Copy `.env.example` to `.env` only for local use. Do not commit PostgreSQL
passwords, AES keys, control master keys, model weights, or private sender
reports.

## Basic verification

```powershell
python -m zsc_tsd preflight --profile paper_r1
python -m zsc_tsd show-config voip
python -m pytest -q
```

CPU tests cover AES-GCM authentication, CRC framing, configuration resolution,
local-transport labeling, RTP/Opus plans, and HMAC tamper/expiry handling.
Model tests skip when frozen model files or CUDA are unavailable.

## Reproduce the secure IMDB batch

The stable CLI delegates to the frozen v1.4 implementation without changing
receiver rules:

```powershell
python -m zsc_tsd secure-imdb-batch `
  --bundle-public artifacts\new_e2e\secure_batch_v1_protocol_dev\public_commitment.json `
  --bundle-private artifacts\new_e2e\secure_batch_v1_protocol_dev\sender_private.json
```

Expected final evidence is written below a new timestamped directory as
`public\final_result_after_reveal.json`. Existing evidence is never
overwritten.

## Build a two-host RTP/Opus plan

```powershell
python -m zsc_tsd rtp-plan `
  --ffmpeg artifacts\voip_tools\ffmpeg_from_b\bin\ffmpeg.exe `
  --source-wav artifacts\new_e2e\voip\carrier.wav `
  --receiver-ip 192.0.2.10 `
  --receiver-sdp artifacts\new_e2e\voip\receiver.sdp `
  --received-wav artifacts\new_e2e\voip\received.wav
```

This prints separate sender and receiver commands and an SDP description. It
does not pretend to launch both physical hosts from one process.

## Run the authenticated two-host experiment

This workflow uses one command per physical node. Node A prints a new
`session_id` and `correlation_id`; those identifiers are then supplied to the
single relay command on B.

Before the first v2 run, apply the idempotent schema migration on A or B:

```powershell
python tools\zts_control_channel_cli.py init-db `
  --schema zsc_tsd\control\schema.sql
```

Start node A first:

```powershell
python -m zsc_tsd dual-channel-run `
  --source-wav <carrier.wav> `
  --ffmpeg artifacts\voip_tools\ffmpeg_from_b\bin\ffmpeg.exe `
  --a-ip 192.0.2.10 `
  --b-ip 192.0.2.11
```

Then start node B with the identifiers printed by A:

```powershell
python -m zsc_tsd dual-channel-relay `
  --session-id <SESSION_ID_FROM_A> `
  --correlation-id <CORRELATION_ID_FROM_A> `
  --ffmpeg D:\ffmpeg-2026-06-10-git-b29bdd3715-essentials_build\bin\ffmpeg.exe `
  --expected-a-ip 192.0.2.10 `
  --b-ip 192.0.2.11
```

The A-to-B HTTP fetch is a restricted, hash-committed **experiment carrier
provisioning step**. It is not counted or reported as VoIP. The measured media
path is the subsequent B-to-A RTP/UDP Opus transmission. PostgreSQL contains
only signed metadata.

Without `--decoder-command-json`, a successful run ends with
`ZSC_TSD_DUAL_CHANNEL_MEDIA_PASS`. It is intentionally impossible for that
run to claim exact message recovery. `ZSC_TSD_DUAL_CHANNEL_END_TO_END_PASS`
requires a decoder report in which receiver decoding, Reed-Solomon recovery,
AES-GCM authentication, and exact plaintext matching are all true.

## Run the complete FACodec network end-to-end batch

The single-session workflow above verifies media delivery. The complete
FACodec end-to-end path is split into two stages.

**Carrier generation (DAQ-TTS = TransformerTTS + FACodec).** The sender first
generates the speech carriers from the steganographic text through the
DAQ-TTS back end (the `zsc-tsd-tts` and `zsc-tsd-facodec` environments,
prepared by `tools/setup_daqtts_reviewer.py`). Use
`run_local_e2e_stage_demo.bat` or `04_run_full_experiment.ps1` to produce a
batch directory with `public/<carrier>/carrier.wav` and
`sender_private/<carrier>/sender.json`.

**Network end-to-end recovery.** The batch tools then take those frozen
carriers and execute, for every carrier: carrier preparation, a signed
control handshake, B-to-A RTP/UDP Opus transmission, receiver-side ASR
(Whisper), candidate recovery, ZTS decoding, CRC validation, and finally
Reed--Solomon reconstruction plus AES-GCM plaintext verification.

Node A (sender + decoder) starts first:

```powershell
python -m zsc_tsd dual-channel-batch-a `
  --secure-batch-root <batch_root> `
  --bundle-private <bundle_private.json> `
  --ffmpeg <ffmpeg.exe> `
  --asr-python <zsc-tsd-asr-python.exe> `
  --zts-python <zsc-tsd-zts-python.exe> `
  --asr-model <whisper-small.en-directory> `
  --a-ip <A-ip> --b-ip <B-ip> `
  --output-base <run_root_A>
```

A prints `ZSC_TSD_BATCH_A_READY` with a `batch_id`; forward that identifier
to node B:

```powershell
python -m zsc_tsd dual-channel-batch-relay `
  --batch-id <BATCH_ID_FROM_A> `
  --expected-sessions <N> `
  --ffmpeg <ffmpeg.exe> `
  --expected-a-ip <A-ip> --b-ip <B-ip> `
  --output-base <run_root_B>
```

A run reports `ZSC_TSD_VOIP_SECURE_BATCH_END_TO_END_PASS` only if the number
of distinct recovered shards meets the bundle requirement, Reed--Solomon
reconstruction succeeds, AES-GCM authentication succeeds, and the recovered
plaintext equals the expected secret. A run that only completes media
delivery reports `ZSC_TSD_DUAL_CHANNEL_BATCH_RELAY_PASS` on B and a
`ZSC_TSD_VOIP_SECURE_BATCH_END_TO_END_FAIL` / `ZSC_TSD_BATCH_A_READY` chain
on A, never an end-to-end claim.

To run the same path through the single-session command, pass the decoder
through `--decoder-command-json`; the decoder receives `{received_wav}`,
`{output_json}`, and `{run_root}` placeholders and must write a report
containing the receiver decode, Reed--Solomon, AES-GCM, and plaintext-match
booleans.

## Control channel

See `docs/CONTROL_CHANNEL_V1.md`. The database carries authenticated metadata
only. Secret plaintext, AES keys, ciphertext, RS bytes, hidden fragment bytes,
stegotext, token IDs, and reference transcripts are prohibited from control
events.

## Artifact policy

- `zsc_tsd/` is the maintained interface.
- `tools/*_frozen.py` and hash-locked sources are immutable evidence.
- `tools/*_dev.py`, postmortems, and generator scripts remain available for
  audit, but are not reviewer entry points.
- Every new run uses a new output directory and records SHA-256 hashes.
- Local file copying is labeled as local integration, never VoIP.
- RTP/Opus transport is labeled accurately; SIP is not claimed.
- Retries, erasures, candidate-space truncation, and failed carriers remain in
  reports and are not removed from denominators.
