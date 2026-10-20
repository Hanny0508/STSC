param(
    [string]$ProjectRoot = (Split-Path -Parent $MyInvocation.MyCommand.Path),
    [string]$Conda = '',
    [switch]$RefreshExisting,
    [string]$TorchIndexUrl = 'https://download.pytorch.org/whl/cu128'
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
if (!(Test-Path -LiteralPath (Join-Path $ProjectRoot 'pyproject.toml'))) {
    throw "Not a ZSC-TSD project root: $ProjectRoot"
}

if (!$Conda) {
    if ($env:CONDA_EXE) { $Conda = $env:CONDA_EXE }
    else {
        $command = Get-Command conda.exe -ErrorAction SilentlyContinue
        if ($command) { $Conda = $command.Source }
    }
}
if (!$Conda -or !(Test-Path -LiteralPath $Conda)) {
    throw 'Conda was not found. Pass -Conda D:\Anaconda3\Scripts\conda.exe'
}

function Invoke-Checked([string[]]$Arguments) {
    Write-Host ('+ ' + $Conda + ' ' + ($Arguments -join ' '))
    & $Conda @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Conda command failed with exit code $LASTEXITCODE" }
}

$bootstrap = Join-Path $ProjectRoot 'tools\reviewer_bootstrap.py'
$setupArgs = @('run','--no-capture-output','-n','base','python',$bootstrap,'setup','--profile','full-a','--conda',$Conda)
if ($RefreshExisting) { $setupArgs += '--refresh-existing' }
Invoke-Checked $setupArgs

$relayArgs = @('run','--no-capture-output','-n','base','python',$bootstrap,'setup','--profile','relay-b','--conda',$Conda)
if ($RefreshExisting) { $relayArgs += '--refresh-existing' }
Invoke-Checked $relayArgs

$envJson = & $Conda env list --json
if ($LASTEXITCODE -ne 0) { throw 'Cannot list Conda environments' }
$environmentNames = @((ConvertFrom-Json ($envJson -join "`n")).envs | ForEach-Object { (Split-Path -Leaf $_).ToLowerInvariant() })
$facodecExists = $environmentNames -contains 'zsc-tsd-facodec'

if (!$facodecExists) {
    Invoke-Checked @('create','-y','-n','zsc-tsd-facodec','python=3.10','pip')
}

if (!$facodecExists -or $RefreshExisting) {
    Invoke-Checked @('run','--no-capture-output','-n','zsc-tsd-facodec','python','-m','pip','install','torch==2.7.1','torchaudio==2.7.1','--index-url',$TorchIndexUrl)
    Invoke-Checked @('run','--no-capture-output','-n','zsc-tsd-facodec','python','-m','pip','install','-r',(Join-Path $ProjectRoot 'requirements\reviewer-facodec.txt'))
    $facodecSource = 'https://github.com/lifeiteng/naturalspeech3_facodec/archive/aac16b7268b1b96e70b9e3526b13e216984e80d0.zip'
    Invoke-Checked @('run','--no-capture-output','-n','zsc-tsd-facodec','python','-m','pip','install',$facodecSource)
    Invoke-Checked @('run','--no-capture-output','-n','zsc-tsd-facodec','python','-m','pip','install','--no-deps','-e',$ProjectRoot)
}
else {
    Write-Host '= preserving existing environment: zsc-tsd-facodec'
}

$required = @(
    'zsc-tsd-core','zsc-tsd-zts','zsc-tsd-asr','zsc-tsd-tts','zsc-tsd-facodec','zsc-tsd-relay'
)
$rows = @(
    foreach ($name in $required) {
        $python = @(& $Conda run -n $name python -c 'import sys;print(sys.executable)' 2>&1 | Where-Object { $_ -match '(?i)python\.exe$' } | Select-Object -Last 1)
        if ($LASTEXITCODE -ne 0 -or !$python) { throw "Environment verification failed: $name" }
        [pscustomobject]@{ Environment=$name; Python=([string]$python).Trim(); Status='PASS' }
    }
)

$espeak = 'C:\Program Files\eSpeak NG\espeak-ng.exe'
if (!(Test-Path -LiteralPath $espeak)) {
    Write-Warning "eSpeak NG is not installed at $espeak. TransformerTTS needs it before the full run."
}

$logDir = Join-Path $ProjectRoot 'artifacts\setup_logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$report = [ordered]@{
    status='ZSC_TSD_ALL_CONDA_ENVS_PASS'
    created_at=(Get-Date).ToString('o')
    project_root=$ProjectRoot
    conda=$Conda
    environments=$rows
    espeak_present=(Test-Path -LiteralPath $espeak)
}
$reportPath = Join-Path $logDir 'conda_environments.json'
$report | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $reportPath -Encoding UTF8
$rows | Format-Table -AutoSize
Write-Host "ZSC-TSD ALL CONDA ENVIRONMENTS PASS"
Write-Host "report=$reportPath"