param(
    [Parameter(Mandatory=$true)][ValidateSet('A','B')][string]$Role,
    [string]$AIp = '192.0.2.10',
    [string]$BIp = '192.0.2.11',
    [int]$PostgresPort = 55432,
    [string]$PostgresService = 'postgresql-x64-17',
    [string]$PostgresData = 'E:\ZSC-TSD-VoIP\PostgreSQL\17\data',
    [string]$Database = 'zsc_tsd_control',
    [string]$DatabaseRole = 'zsc_tsd_app',
    [string]$Tailscale = 'C:\Program Files\Tailscale\tailscale.exe'
)

$ErrorActionPreference='Stop'
$identity=[Security.Principal.WindowsIdentity]::GetCurrent()
$principal=New-Object Security.Principal.WindowsPrincipal($identity)
if(!$principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)){throw 'Run PowerShell as Administrator'}
if(!(Test-Path -LiteralPath $Tailscale)){throw "Tailscale missing: $Tailscale"}
$expected=if($Role -eq 'A'){$AIp}else{$BIp}
$actual=@(& $Tailscale ip -4 2>&1)
if($LASTEXITCODE -ne 0 -or $actual -notcontains $expected){throw "Tailscale IP mismatch: expected=$expected actual=$($actual -join ',')"}

function Set-ExactFirewallRule([string]$Name,[string]$Protocol,[int]$Port,[string]$Local,[string]$Remote){
    @(Get-NetFirewallRule -DisplayName $Name -ErrorAction SilentlyContinue)|ForEach-Object{Remove-NetFirewallRule -Name $_.Name}
    New-NetFirewallRule -DisplayName $Name -Direction Inbound -Action Allow -Enabled True -Profile Any -Protocol $Protocol -LocalAddress $Local -LocalPort $Port -RemoteAddress $Remote -EdgeTraversalPolicy Block|Out-Null
}

if($Role -eq 'A'){
    Set-ExactFirewallRule 'ZSC-TSD Carrier Provisioning TCP 5602 from B only' 'TCP' 5602 $AIp $BIp
    Set-ExactFirewallRule 'ZSC-TSD RTP Opus UDP 5004 from B only' 'UDP' 5004 $AIp $BIp
    $db=Test-NetConnection -ComputerName $BIp -Port $PostgresPort -InformationLevel Quiet
    $result=[ordered]@{status=if($db){'ZSC_TSD_NETWORK_A_PASS'}else{'ZSC_TSD_NETWORK_A_FIREWALL_PASS_DATABASE_UNREACHABLE'};role='A';tailscale_ip=$AIp;postgres="$BIp`:$PostgresPort";postgres_reachable=$db;tcp_5602='B-only inbound';udp_5004='B-only inbound'}
    $result|ConvertTo-Json -Depth 4
    if(!$db){Write-Warning 'A firewall is configured, but PostgreSQL on B is unreachable. Run this script with -Role B on B.'}
    exit 0
}

$conf=Join-Path $PostgresData 'postgresql.conf';$hba=Join-Path $PostgresData 'pg_hba.conf'
if(!(Test-Path -LiteralPath $conf) -or !(Test-Path -LiteralPath $hba)){throw "PostgreSQL data files missing: $PostgresData"}
$backup=Join-Path (Split-Path -Parent $PostgresData) ('Config-Backup-ZSC-TSD-'+(Get-Date -Format 'yyyyMMdd_HHmmss'))
New-Item -ItemType Directory -Path $backup|Out-Null
Copy-Item -LiteralPath $conf -Destination (Join-Path $backup 'postgresql.conf')
Copy-Item -LiteralPath $hba -Destination (Join-Path $backup 'pg_hba.conf')

$text=[IO.File]::ReadAllText($conf)
function Set-Setting([string]$Text,[string]$Name,[string]$Value){
    $pattern="(?m)^\s*#?\s*"+[regex]::Escape($Name)+"\s*=.*$"
    if([regex]::IsMatch($Text,$pattern)){return [regex]::Replace($Text,$pattern,"$Name = $Value",1)}
    return $Text+[Environment]::NewLine+"$Name = $Value"+[Environment]::NewLine
}
$text=Set-Setting $text 'listen_addresses' "'$BIp,127.0.0.1'"
$text=Set-Setting $text 'port' ([string]$PostgresPort)
$text=Set-Setting $text 'password_encryption' "'scram-sha-256'"
$utf8=New-Object Text.UTF8Encoding($false);[IO.File]::WriteAllText($conf,$text,$utf8)

$old=[IO.File]::ReadAllText($hba)
$old=[regex]::Replace($old,'(?s)# BEGIN ZSC-TSD RESTRICTED CONTROL.*?# END ZSC-TSD RESTRICTED CONTROL\r?\n?','')
$block=@"
# BEGIN ZSC-TSD RESTRICTED CONTROL
host all all 127.0.0.1/32 scram-sha-256
host all all ::1/128 scram-sha-256
host $Database $DatabaseRole $AIp/32 scram-sha-256
host all all $AIp/32 reject
host all all 0.0.0.0/0 reject
host all all ::/0 reject
# END ZSC-TSD RESTRICTED CONTROL
"@
[IO.File]::WriteAllText($hba,$block+[Environment]::NewLine+$old,$utf8)
Set-ExactFirewallRule "ZSC-TSD PostgreSQL $PostgresPort from A Tailscale only" 'TCP' $PostgresPort $BIp $AIp
Restart-Service -Name $PostgresService
Start-Sleep -Seconds 4
$listeners=@(Get-NetTCPConnection -LocalPort $PostgresPort -State Listen -ErrorAction SilentlyContinue)
$addresses=@($listeners|ForEach-Object{$_.LocalAddress})
if($addresses -notcontains '127.0.0.1' -or $addresses -notcontains $BIp){throw "PostgreSQL listener verification failed: $($addresses -join ',')"}
[ordered]@{status='ZSC_TSD_NETWORK_B_PASS';role='B';tailscale_ip=$BIp;postgres_port=$PostgresPort;listeners=$addresses;allowed_remote=$AIp;backup=$backup}|ConvertTo-Json -Depth 4