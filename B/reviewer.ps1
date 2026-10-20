param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('PlanA','PlanB','SetupA','SetupB','VerifyModels','QuickTestA','QuickTestB','HostPreflightA','HostPreflightB','SecureBatchA','MediaA','RelayB','BatchMediaA','BatchRelayB')]
    [string]$Action,
    [string]$Conda = '',
    [switch]$AcceptModelLicenses,
    [string]$HfEndpoint = '',
    [string]$TorchWheelUrl = '',
    [string]$BundlePublic = '',
    [string]$BundlePrivate = '',
    [string]$SourceWav = '',
    [string]$FFmpeg = '',
    [string]$SessionId = '',
    [string]$CorrelationId = '',
    [string]$BatchId = '',
    [int]$ExpectedSessions = 0,
    [string]$SecureBatchRoot = '',
    [string]$ASRPython = '',
    [string]$ZTSPython = '',
    [string]$ASRModel = '',
    [string]$AIp = '100.104.205.23',
    [string]$BIp = '100.104.210.91',
    [int]$PostgresPort = 55432,
    [string]$PostgresUser = 'zsc_tsd_app',
    [string]$PostgresDatabase = 'zsc_tsd_control',
    [string]$PostgresService = 'postgresql-x64-17',
    [string]$Tailscale = 'C:\Program Files\Tailscale\tailscale.exe'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$bootstrap = Join-Path $root 'tools\reviewer_bootstrap.py'
$python = if (Get-Command py.exe -ErrorAction SilentlyContinue) { 'py.exe' } elseif (Get-Command python.exe -ErrorAction SilentlyContinue) { 'python.exe' } else { throw 'Python launcher not found' }
$prefix = @()
if ($python -eq 'py.exe') { $prefix = @('-3') }
$condaArgs = @()
if ($Conda) { $condaArgs = @('--conda', $Conda) }
$condaExe = if ($Conda) { $Conda } elseif ($env:CONDA_EXE) { $env:CONDA_EXE } else { (Get-Command conda.exe -ErrorAction SilentlyContinue).Source }

function Invoke-WithControlEnvironment([string]$Dsn, [string]$KeyPath, [scriptblock]$Body) {
    if (!(Test-Path -LiteralPath $KeyPath)) { throw "Control master key missing: $KeyPath" }
    $secret = Read-Host 'Enter zsc_tsd_app PostgreSQL password' -AsSecureString
    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secret)
    try {
        $env:PGPASSWORD = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
        $env:ZSC_TSD_CONTROL_DSN = $Dsn
        $env:ZSC_TSD_CONTROL_MASTER_KEY_B64 = (Get-Content -LiteralPath $KeyPath -Raw).Trim()
        & $Body
    }
    finally {
        $env:PGPASSWORD = $null
        $env:ZSC_TSD_CONTROL_DSN = $null
        $env:ZSC_TSD_CONTROL_MASTER_KEY_B64 = $null
        if ($pointer -ne [IntPtr]::Zero) { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer) }
    }
}

function Assert-TailscaleAddress([string]$ExpectedAddress) {
    if (!(Test-Path -LiteralPath $Tailscale)) { throw "Tailscale executable missing: $Tailscale" }
    $addresses = @(& $Tailscale ip -4 2>&1)
    if ($LASTEXITCODE -ne 0) { throw "Tailscale ip query failed: $($addresses -join ' ')" }
    if ($addresses -notcontains $ExpectedAddress) {
        throw "Tailscale address mismatch: expected $ExpectedAddress; got $($addresses -join ', ')"
    }
}

function Assert-FFmpegOpusRtp([string]$Path) {
    if (!$Path) { throw 'FFmpeg path is required' }
    if (!(Test-Path -LiteralPath $Path -PathType Leaf)) { throw "FFmpeg missing: $Path" }
    $encoders = (& $Path -hide_banner -encoders 2>&1 | Out-String)
    if ($LASTEXITCODE -ne 0 -or $encoders -notmatch '(?m)^\s*A\S*\s+libopus\s') {
        throw "FFmpeg does not expose the libopus encoder: $Path"
    }
    $protocols = (& $Path -hide_banner -protocols 2>&1 | Out-String)
    if ($LASTEXITCODE -ne 0 -or $protocols -notmatch '(?m)^\s+rtp\s*$' -or $protocols -notmatch '(?m)^\s+udp\s*$') {
        throw "FFmpeg does not expose both RTP and UDP protocols: $Path"
    }
}

function Assert-NodeAHost {
    Assert-TailscaleAddress $AIp
    Assert-FFmpegOpusRtp $FFmpeg
    $database = Test-NetConnection -ComputerName $BIp -Port $PostgresPort -InformationLevel Quiet
    if (!$database) {
        throw "A cannot reach PostgreSQL at ${BIp}:$PostgresPort. On B, run HostPreflightB and restart $PostgresService if its Tailscale listener is absent."
    }
    $tcp = @(Get-NetTCPConnection -LocalPort 5602 -State Listen -ErrorAction SilentlyContinue)
    $udp = @(Get-NetUDPEndpoint -LocalPort 5004 -ErrorAction SilentlyContinue)
    if ($tcp.Count -ne 0) { throw 'A TCP port 5602 is already occupied' }
    if ($udp.Count -ne 0) { throw 'A UDP port 5004 is already occupied' }
    [pscustomobject]@{
        Status='ZSC_TSD_HOST_PREFLIGHT_A_PASS'; TailscaleIP=$AIp;
        PostgreSQL="$BIp`:$PostgresPort"; FFmpeg=(Resolve-Path -LiteralPath $FFmpeg).Path;
        TCP5602Free=$true; UDP5004Free=$true
    } | ConvertTo-Json -Depth 3
}

function Assert-NodeBHost {
    Assert-TailscaleAddress $BIp
    Assert-FFmpegOpusRtp $FFmpeg
    $service = Get-Service -Name $PostgresService -ErrorAction SilentlyContinue
    if (!$service) { throw "PostgreSQL service not found: $PostgresService" }
    if ($service.Status -ne 'Running') { throw "PostgreSQL service is not running: $PostgresService" }
    $listeners = @(Get-NetTCPConnection -LocalPort $PostgresPort -State Listen -ErrorAction SilentlyContinue)
    $localAddresses = @($listeners | ForEach-Object { $_.LocalAddress })
    if ($localAddresses -notcontains '127.0.0.1' -or $localAddresses -notcontains $BIp) {
        throw "PostgreSQL must listen on 127.0.0.1 and $BIp at port $PostgresPort. Current: $($localAddresses -join ', '). Run: Restart-Service -Name '$PostgresService', then repeat HostPreflightB."
    }
    [pscustomobject]@{
        Status='ZSC_TSD_HOST_PREFLIGHT_B_PASS'; TailscaleIP=$BIp;
        PostgreSQLService=$PostgresService; PostgreSQLListeners=$localAddresses;
        PostgreSQLPort=$PostgresPort; FFmpeg=(Resolve-Path -LiteralPath $FFmpeg).Path
    } | ConvertTo-Json -Depth 3
}

switch ($Action) {
    'PlanA' { & $python @prefix $bootstrap plan --profile full-a @condaArgs }
    'PlanB' { & $python @prefix $bootstrap plan --profile relay-b @condaArgs }
    'SetupA' {
        if (-not $AcceptModelLicenses) { throw 'SetupA requires -AcceptModelLicenses' }
        $endpointArgs = @()
        if ($HfEndpoint) { $endpointArgs = @('--hf-endpoint', $HfEndpoint) }
        $torchArgs = @()
        if ($TorchWheelUrl) { $torchArgs = @('--torch-wheel-url', $TorchWheelUrl) }
        & $python @prefix $bootstrap setup --profile full-a --with-models --accept-model-licenses @endpointArgs @torchArgs @condaArgs
    }
    'SetupB' { & $python @prefix $bootstrap setup --profile relay-b @condaArgs }
    'VerifyModels' { & $python @prefix $bootstrap verify-models }
    'QuickTestA' { & $python @prefix $bootstrap quick-test --profile full-a @condaArgs }
    'QuickTestB' { & $python @prefix $bootstrap quick-test --profile relay-b @condaArgs }
    'HostPreflightA' { Assert-NodeAHost }
    'HostPreflightB' { Assert-NodeBHost }
    'SecureBatchA' {
        if (!$condaExe) { throw 'Conda not found; pass -Conda <path>' }
        if (!$BundlePublic -or !$BundlePrivate) { throw 'SecureBatchA requires -BundlePublic and -BundlePrivate' }
        & $condaExe run --no-capture-output -n zsc-tsd-zts python -m zsc_tsd secure-imdb-batch --bundle-public $BundlePublic --bundle-private $BundlePrivate
    }
    'MediaA' {
        if (!$condaExe) { throw 'Conda not found; pass -Conda <path>' }
        if (!$SourceWav -or !$FFmpeg) { throw 'MediaA requires -SourceWav and -FFmpeg' }
        Assert-NodeAHost | Out-Host
        $dsn = "host=$BIp port=$PostgresPort dbname=$PostgresDatabase user=$PostgresUser connect_timeout=20 sslmode=disable"
        $key = Join-Path $root 'artifacts\control_private\control_master_key.b64'
        Invoke-WithControlEnvironment $dsn $key {
            & $condaExe run --no-capture-output -n zsc-tsd-core python -m zsc_tsd dual-channel-run --source-wav $SourceWav --ffmpeg $FFmpeg --a-ip $AIp --b-ip $BIp --provisioning-port 5602 --rtp-port 5004 --wait-timeout-seconds 600
        }
    }
    'RelayB' {
        if (!$condaExe) { throw 'Conda not found; pass -Conda <path>' }
        if (!$SessionId -or !$CorrelationId -or !$FFmpeg) { throw 'RelayB requires -SessionId, -CorrelationId, and -FFmpeg' }
        Assert-NodeBHost | Out-Host
        $dsn = "host=127.0.0.1 port=$PostgresPort dbname=$PostgresDatabase user=$PostgresUser connect_timeout=20 sslmode=disable"
        $key = Join-Path $root 'private\control_master_key.b64'
        Invoke-WithControlEnvironment $dsn $key {
            & $condaExe run --no-capture-output -n zsc-tsd-relay python -m zsc_tsd dual-channel-relay --session-id $SessionId --correlation-id $CorrelationId --ffmpeg $FFmpeg --expected-a-ip $AIp --b-ip $BIp --wait-timeout-seconds 600
        }
    }
    'BatchMediaA' {
        if (!$condaExe) { throw 'Conda not found; pass -Conda <path>' }
        if (!$SecureBatchRoot -or !$BundlePrivate -or !$FFmpeg -or !$ASRPython -or !$ZTSPython -or !$ASRModel) {
            throw 'BatchMediaA requires -SecureBatchRoot, -BundlePrivate, -FFmpeg, -ASRPython, -ZTSPython, and -ASRModel'
        }
        Assert-NodeAHost | Out-Host
        $dsn = "host=$BIp port=$PostgresPort dbname=$PostgresDatabase user=$PostgresUser connect_timeout=20 sslmode=disable"
        $key = Join-Path $root 'artifacts\control_private\control_master_key.b64'
        Invoke-WithControlEnvironment $dsn $key {
            & $condaExe run --no-capture-output -n zsc-tsd-core python -m zsc_tsd dual-channel-batch-a --secure-batch-root $SecureBatchRoot --bundle-private $BundlePrivate --ffmpeg $FFmpeg --asr-python $ASRPython --zts-python $ZTSPython --asr-model $ASRModel --a-ip $AIp --b-ip $BIp --provisioning-port 5602 --rtp-port 5004 --wait-timeout-seconds 900
        }
    }
    'BatchRelayB' {
        if (!$condaExe) { throw 'Conda not found; pass -Conda <path>' }
        if (!$BatchId -or $ExpectedSessions -lt 1 -or !$FFmpeg) {
            throw 'BatchRelayB requires -BatchId, positive -ExpectedSessions, and -FFmpeg'
        }
        Assert-NodeBHost | Out-Host
        $dsn = "host=127.0.0.1 port=$PostgresPort dbname=$PostgresDatabase user=$PostgresUser connect_timeout=20 sslmode=disable"
        $key = Join-Path $root 'private\control_master_key.b64'
        Invoke-WithControlEnvironment $dsn $key {
            & $condaExe run --no-capture-output -n zsc-tsd-relay python -m zsc_tsd dual-channel-batch-relay --batch-id $BatchId --expected-sessions $ExpectedSessions --ffmpeg $FFmpeg --expected-a-ip $AIp --b-ip $BIp --wait-timeout-seconds 900
        }
    }
}
if ($LASTEXITCODE -ne 0) { throw "reviewer action failed with exit code $LASTEXITCODE" }
