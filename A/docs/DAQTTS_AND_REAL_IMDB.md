# DAQ-TTS and real IMDb experiment path

## Terminology

In this artifact, **DAQ-TTS is a project name for an engineering composition**:

`TransformerTTS source WAV + natural reference WAV -> FACodec V2 voice conversion`

Neither TransformerTTS nor FACodec is claimed as a new algorithm.  The raw
TransformerTTS waveform is retained as private evidence. The source supplies
content/prosody codes; a separately committed clean WAV supplies the speaker
embedding. Source residual acoustic-detail codes are excluded with
`vq2emb(source_ids, use_residual=False)`. Only the converted waveform is named
`carrier.wav` and passed to RTP/UDP with Opus.

FACodec is not a vocoder replacement for TransformerTTS in this implementation.
The V1 self-reconstruction command is retained only as a codec-integrity smoke
test; it is not the formal carrier path and is not expected to change the voice
substantially. FACodec V2 is reference-conditioned voice conversion, not an
automatic enhancement algorithm. Improved naturalness must be established by
listening, ASR/WER, NISQA, or MOS rather than assumed from its presence.

## Dataset provenance

The v1.5 path does not use the historical copied `legacy/.../imdb.csv`.  It
downloads `aclImdb_v1.tar.gz` from the Stanford dataset page, checks the frozen
MD5 `7c2ac02c03563afcf9b574c7e56c153a`, and deterministically extracts one
eligible sentence from each sorted raw review.  `dataset_manifest.json` records
the official URL, archive MD5/SHA-256, source review member and per-review
SHA-256, prompt rules, prompt CSV SHA-256, and counts.

## Reviewer commands on node A

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\reviewer.ps1 `
  -Action SetupDAQTTS -Conda "D:\Anaconda3\Scripts\conda.exe" `
  -HfEndpoint "https://hf-mirror.com"

powershell -NoProfile -ExecutionPolicy Bypass -File .\reviewer.ps1 `
  -Action DAQVoiceSmokeA -Conda "D:\Anaconda3\Scripts\conda.exe" `
  -SourceWav ".\artifacts\new_e2e\voip_rtp_smoke_v1\sender\source_carrier.wav"

powershell -NoProfile -ExecutionPolicy Bypass -File .\reviewer.ps1 `
  -Action SecureDAQBatchA -Conda "D:\Anaconda3\Scripts\conda.exe" `
  -BundlePublic <secure_bundle_public.json> `
  -BundlePrivate <secure_bundle_private.json> `
  -ReferenceWav <fixed-clean-reference.wav>
```

Node B needs no FACodec environment: it relays the already reconstructed WAV
over the existing authenticated control channel and RTP/Opus media channel.

## Required evidence per carrier

- `tts.transformertts_raw.wav` and its SHA-256;
- `tts.transformertts.json`;
- `tts.facodec.json`, including both input commitments, checkpoint hashes,
  tensor shapes, and proof that source residual codes were excluded;
- `tts.json` with `DAQ_TTS_REFERENCE_CONDITIONED_PASS`;
- public `carrier.wav`, whose hash equals the FACodec report output hash;
- subsequent Whisper and fragment-receiver evidence.
