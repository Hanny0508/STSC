param(
    [string]$ProjectRoot = (Split-Path -Parent $MyInvocation.MyCommand.Path),
    [string]$Conda = '',
    [Parameter(Mandatory=$true)][string]$ModelRoot,
    [Parameter(Mandatory=$true)][switch]$AcceptModelLicenses,
    [string]$HfEndpoint = '',
    [string]$TransformerTTSArchive = '',
    [string]$TransformerTTSUrl = ''
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
if (!(Test-Path -LiteralPath (Join-Path $ProjectRoot 'pyproject.toml'))) { throw "Not a ZSC-TSD project root: $ProjectRoot" }
if (!$AcceptModelLicenses) { throw 'Explicit -AcceptModelLicenses is required by upstream model terms' }
if (!$Conda) { if ($env:CONDA_EXE) {$Conda=$env:CONDA_EXE} else {$c=Get-Command conda.exe -ErrorAction SilentlyContinue;if($c){$Conda=$c.Source}} }
if (!$Conda -or !(Test-Path -LiteralPath $Conda)) { throw 'Conda not found; pass -Conda <conda.exe>' }

$ModelRoot = [IO.Path]::GetFullPath($ModelRoot)
New-Item -ItemType Directory -Force -Path $ModelRoot | Out-Null
$env:HF_HOME = Join-Path $ModelRoot 'huggingface_cache'
if ($HfEndpoint) { $env:HF_ENDPOINT = $HfEndpoint.TrimEnd('/') }

function Ensure-ModelJunction([string]$Expected,[string]$Target) {
    New-Item -ItemType Directory -Force -Path $Target | Out-Null
    $parent = Split-Path -Parent $Expected
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    if (Test-Path -LiteralPath $Expected) {
        $item = Get-Item -LiteralPath $Expected -Force
        $children = @(Get-ChildItem -LiteralPath $Expected -Force -ErrorAction SilentlyContinue)
        if ($item.LinkType -eq 'Junction') { Write-Host "= junction already present: $Expected"; return }
        if ($children.Count -gt 0) { Write-Host "= verified model location already contains files; preserving: $Expected"; return }
        Remove-Item -LiteralPath $Expected
    }
    New-Item -ItemType Junction -Path $Expected -Target $Target | Out-Null
    Write-Host "= model junction: $Expected -> $Target"
}

Ensure-ModelJunction (Join-Path $ProjectRoot 'artifacts\asr_models\whisper-small.en') (Join-Path $ModelRoot 'asr\whisper-small.en')
Ensure-ModelJunction (Join-Path $ProjectRoot 'legacy\zero-shot-GLS\tmp_saves\local_models\llama-2-7b-chat-gptq-4bit-32g-actorder') (Join-Path $ModelRoot 'zts\llama-2-7b-chat-gptq-4bit-32g-actorder')
Ensure-ModelJunction (Join-Path $ProjectRoot 'artifacts\tts_models\bdf06b9_ljspeech_step_90000\bdf06b9_ljspeech_step_90000') (Join-Path $ModelRoot 'tts\bdf06b9_ljspeech_step_90000')

$ttsArchiveTarget = Join-Path $ProjectRoot 'artifacts\tts_models\bdf06b9_ljspeech_step_90000.zip'
if ($TransformerTTSArchive) {
    $source = (Resolve-Path -LiteralPath $TransformerTTSArchive).Path
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $ttsArchiveTarget) | Out-Null
    if (!(Test-Path -LiteralPath $ttsArchiveTarget)) { Copy-Item -LiteralPath $source -Destination $ttsArchiveTarget }
}
if ($TransformerTTSUrl) { $env:ZSC_TSD_TTS_MODEL_URL = $TransformerTTSUrl }

$bootstrap = Join-Path $ProjectRoot 'tools\reviewer_bootstrap.py'
$modelArgs = @('run','--no-capture-output','-n','zsc-tsd-core','python',$bootstrap,'models','--accept-model-licenses')
if ($HfEndpoint) { $modelArgs += @('--hf-endpoint',$HfEndpoint) }
& $Conda @modelArgs
if ($LASTEXITCODE -ne 0) { throw 'Frozen ASR/ZTS/TTS model download or verification failed' }

$daqArgs = @('run','--no-capture-output','-n','zsc-tsd-core','python',(Join-Path $ProjectRoot 'tools\setup_daqtts_reviewer.py'),'--conda',$Conda)
if ($HfEndpoint) { $daqArgs += @('--hf-endpoint',$HfEndpoint) }
& $Conda @daqArgs
if ($LASTEXITCODE -ne 0) { throw 'FACodec checkpoint/reference provisioning failed' }

& $Conda run --no-capture-output -n zsc-tsd-core python $bootstrap verify-models
if ($LASTEXITCODE -ne 0) { throw 'Final model verification failed' }

$config = [ordered]@{
    schema='zsc-tsd.local-model-paths.v1';status='ZSC_TSD_ALL_MODELS_PASS';created_at=(Get-Date).ToString('o')
    model_root=$ModelRoot;hf_home=$env:HF_HOME;hf_endpoint=$HfEndpoint
    facodec_reference=(Join-Path $ProjectRoot 'artifacts\facodec_reference\official_audio_2.wav')
}
$configPath = Join-Path $ProjectRoot 'configs\local_model_paths.json'
$config | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $configPath -Encoding UTF8
Write-Host 'ZSC-TSD ALL MODELS DOWNLOAD AND HASH VERIFICATION PASS'
Write-Host "model_root=$ModelRoot"
Write-Host "configuration=$configPath"