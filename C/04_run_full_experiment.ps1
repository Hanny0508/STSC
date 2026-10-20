param(
    [string]$ProjectRoot = (Split-Path -Parent $MyInvocation.MyCommand.Path),
    [string]$Conda = '',
    [string]$BundlePublic = '',
    [string]$BundlePrivate = '',
    [int]$StartRow = 0,
    [int]$MaxAttemptsPerShard = 8,
    [switch]$SkipTests,
    [switch]$DownloadIMDbIfMissing
)

$ErrorActionPreference='Stop'
$ProjectRoot=(Resolve-Path -LiteralPath $ProjectRoot).Path
Set-Location -LiteralPath $ProjectRoot
if(!(Test-Path -LiteralPath 'pyproject.toml')){throw "Not a ZSC-TSD project root: $ProjectRoot"}
if(!$Conda){if($env:CONDA_EXE){$Conda=$env:CONDA_EXE}else{$c=Get-Command conda.exe -ErrorAction SilentlyContinue;if($c){$Conda=$c.Source}}}
if(!$Conda -or !(Test-Path -LiteralPath $Conda)){throw 'Conda not found; pass -Conda <conda.exe>'}
if(!$BundlePublic){$BundlePublic=Join-Path $ProjectRoot 'artifacts\new_e2e\secure_batch_v1_protocol_dev\public_commitment.json'}
if(!$BundlePrivate){$BundlePrivate=Join-Path $ProjectRoot 'artifacts\new_e2e\secure_batch_v1_protocol_dev\sender_private.json'}
foreach($p in @($BundlePublic,$BundlePrivate)){if(!(Test-Path -LiteralPath $p -PathType Leaf)){throw "Required frozen bundle missing: $p"}}

function Get-EnvPython([string]$Name){
    $lines=@(& $Conda run -n $Name python -c 'import sys;print(sys.executable)' 2>&1)
    if($LASTEXITCODE -ne 0){throw "Conda environment missing: $Name"}
    $value=[string]($lines|Where-Object{$_ -match '(?i)python\.exe$'}|Select-Object -Last 1)
    if(!$value){throw "Cannot resolve Python for $Name"};return $value.Trim()
}
$ztsPython=Get-EnvPython 'zsc-tsd-zts';$asrPython=Get-EnvPython 'zsc-tsd-asr';$ttsPython=Get-EnvPython 'zsc-tsd-tts';$facodecPython=Get-EnvPython 'zsc-tsd-facodec'
$oldAsr=$env:ZSC_TSD_ASR_PYTHON;$oldTts=$env:ZSC_TSD_TTS_PYTHON;$oldFacodec=$env:ZSC_TSD_FACODEC_PYTHON;$oldHfHome=$env:HF_HOME;$oldHfEndpoint=$env:HF_ENDPOINT
$env:ZSC_TSD_ASR_PYTHON=$asrPython;$env:ZSC_TSD_TTS_PYTHON=$ttsPython;$env:ZSC_TSD_FACODEC_PYTHON=$facodecPython
$modelConfig=Join-Path $ProjectRoot 'configs\local_model_paths.json'
if(Test-Path -LiteralPath $modelConfig){$mc=Get-Content -LiteralPath $modelConfig -Raw|ConvertFrom-Json;if($mc.hf_home){$env:HF_HOME=[string]$mc.hf_home};if($mc.hf_endpoint){$env:HF_ENDPOINT=[string]$mc.hf_endpoint}}

$logDir=Join-Path $ProjectRoot ('artifacts\experiment_logs\paper_r1_'+(Get-Date -Format 'yyyyMMdd_HHmmss'));New-Item -ItemType Directory -Path $logDir|Out-Null
$transcript=Join-Path $logDir 'console.log';Start-Transcript -LiteralPath $transcript|Out-Null
try{
    & $Conda run --no-capture-output -n zsc-tsd-zts python -m zsc_tsd preflight --profile paper_r1
    if($LASTEXITCODE -ne 0){throw 'Project preflight failed'}
    & $Conda run --no-capture-output -n zsc-tsd-core python tools\reviewer_bootstrap.py verify-models
    if($LASTEXITCODE -ne 0){throw 'Model hash verification failed'}
    if(!$SkipTests){& $Conda run --no-capture-output -n zsc-tsd-core python -m pytest -q tests;if($LASTEXITCODE -ne 0){throw 'Test suite failed'}}

    $dataset=Join-Path $ProjectRoot 'artifacts\datasets\imdb_acl_v1';$csv=Join-Path $dataset 'prompts.csv';$manifest=Join-Path $dataset 'dataset_manifest.json'
    if(!(Test-Path -LiteralPath $csv) -or !(Test-Path -LiteralPath $manifest)){
        if(!$DownloadIMDbIfMissing){throw 'Real ACL IMDb dataset missing. Rerun with -DownloadIMDbIfMissing.'}
        & $Conda run --no-capture-output -n zsc-tsd-core python -m zsc_tsd prepare-acl-imdb --archive (Join-Path $ProjectRoot 'artifacts\datasets\aclImdb_v1.tar.gz') --output-dir $dataset --download
        if($LASTEXITCODE -ne 0){throw 'ACL IMDb download/preparation failed'}
    }
    $reference=Join-Path $ProjectRoot 'artifacts\facodec_reference\official_audio_2.wav'
    if(!(Test-Path -LiteralPath $reference)){throw "FACodec reference missing: $reference"}
    $referenceHash=(Get-FileHash -LiteralPath $reference -Algorithm SHA256).Hash.ToLower()
    if($referenceHash -ne '81faa543ca1ae17bb75acbfd7a883b7498db5fe1d0d779c2e8676727a4159e8a'){throw "FACodec reference hash mismatch: $referenceHash"}

    $started=[DateTime]::UtcNow
    & $Conda run --no-capture-output -n zsc-tsd-zts python -m zsc_tsd secure-imdb-daqtts-batch --bundle-public $BundlePublic --bundle-private $BundlePrivate --imdb-csv $csv --imdb-manifest $manifest --facodec-python $facodecPython --reference-wav $reference --start-row $StartRow --max-attempts-per-shard $MaxAttemptsPerShard
    if($LASTEXITCODE -ne 0){throw 'Full ACL IMDb/DAQ-TTS experiment failed'}

    $final=Get-ChildItem (Join-Path $ProjectRoot 'artifacts\new_e2e') -Recurse -Filter 'final_result_after_reveal.json'|Where-Object{$_.LastWriteTimeUtc -ge $started.AddSeconds(-2)}|Sort-Object LastWriteTimeUtc -Descending|Select-Object -First 1
    if(!$final){throw 'New final result was not found'}
    $j=Get-Content -LiteralPath $final.FullName -Raw|ConvertFrom-Json
    if($j.status -ne 'ZSC_TSD_SECURE_IMDB_BATCH_V1_4_PASS' -or !$j.plaintext_exact_match -or !$j.rs_reconstruction_pass -or !$j.aes_gcm_authentication_pass){throw 'Final cryptographic/recovery assertions failed'}
    if($j.v1_5.prompt_dataset.historical_compatibility_dataset_used -ne $false){throw 'Historical prompt dataset was unexpectedly used'}
    if([string]$j.v1_5.speech_backend.facodec_mode -ne 'V2 two-WAV zero-shot voice conversion'){throw 'FACodec V2 provenance missing'}
    $summary=[ordered]@{status='ZSC_TSD_PAPER_R1_FULL_EXPERIMENT_PASS';created_at=(Get-Date).ToString('o');final_result=$final.FullName;final_result_sha256=(Get-FileHash -LiteralPath $final.FullName -Algorithm SHA256).Hash.ToLower();plaintext_exact_match=$j.plaintext_exact_match;rs_reconstruction_pass=$j.rs_reconstruction_pass;aes_gcm_authentication_pass=$j.aes_gcm_authentication_pass;unique_shards_recovered=$j.unique_shards_recovered;shards_total=$j.shards_total;total_carriers=$j.total_carriers;carrier_acceptance_rate=$j.carrier_acceptance_rate;dataset=$j.v1_5.prompt_dataset.name;facodec_mode=$j.v1_5.speech_backend.facodec_mode;network_scope='Core local recovery run. Two-host RTP/Opus is executed separately on physical A and B.'}
    $summaryPath=Join-Path $logDir 'paper_r1_summary.json';$summary|ConvertTo-Json -Depth 5|Set-Content -LiteralPath $summaryPath -Encoding UTF8
    $summary|ConvertTo-Json -Depth 5;Write-Host "summary=$summaryPath"
}
finally{Stop-Transcript -ErrorAction SilentlyContinue|Out-Null;$env:ZSC_TSD_ASR_PYTHON=$oldAsr;$env:ZSC_TSD_TTS_PYTHON=$oldTts;$env:ZSC_TSD_FACODEC_PYTHON=$oldFacodec;$env:HF_HOME=$oldHfHome;$env:HF_ENDPOINT=$oldHfEndpoint}