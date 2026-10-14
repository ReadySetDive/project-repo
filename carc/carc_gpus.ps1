<#
.SYNOPSIS
  CARC (Discovery) GPU snapshot: inventory, free cards, and queue demand per GPU type.

.DESCRIPTION
  Sends a small read-only bash script (sinfo + squeue) to CARC over one SSH session,
  then summarizes locally. Changes nothing on the cluster.
  Needs the `carc` SSH alias (see carc_wiki.md) and the USC VPN when off campus.

.EXAMPLE
  .\carc_gpus.ps1
  .\carc_gpus.ps1 -Raw            # also pending-reason breakdown + raw sinfo/squeue lines
  .\carc_gpus.ps1 -MinCpus 8 -MinRamGB 64   # what my job needs besides the cards
  .\carc_gpus.ps1 -HostAlias me@discovery.usc.edu
#>
param(
    [string]$HostAlias = 'carc',
    # A node only counts toward 'Nodes N+' if it also has at least this much free alongside the cards
    [int]$MinCpus = 4,
    [int]$MinRamGB = 16,
    [switch]$Raw
)

# Remote side: read-only queries, sections marked so login-shell noise is ignored.
# Reason is last because it can contain spaces.
$remote = @'
echo '##SINFO'
sinfo -N -h -p gpu,debug -O "NodeList:30,Partition:15,StateCompact:15,Gres:60,GresUsed:80,CPUsState:25,Memory:15,AllocMem:15,Features:150"
echo '##SQUEUE'
squeue -h -p gpu,debug -O "Partition:15,State:15,NumNodes:8,NumTasks:8,tres-per-node:60,tres-per-task:60,tres-per-job:60,Feature:60,Reason:60"
echo '##END'
'@ -replace "`r", ''

# Windows PowerShell prefixes piped stdin with a UTF-8 BOM (bash then fails on line 1),
# so ship the script base64-encoded and pipe it into bash on the CARC side instead
$b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($remote))
$out = ssh -o BatchMode=yes $HostAlias "echo $b64 | base64 -d | bash -l -s"
if ($LASTEXITCODE -ne 0 -or -not ($out -match '^##END')) {
    Write-Error "SSH/query failed (exit $LASTEXITCODE). VPN up? 'ssh $HostAlias' working?"
    exit 1
}

$section = $null
$sinfo = @(); $squeue = @()
foreach ($line in $out) {
    if ($line -match '^##(\w+)') { $section = $matches[1]; continue }
    if (-not $line.Trim()) { continue }
    if ($section -eq 'SINFO') { $sinfo += $line }
    elseif ($section -eq 'SQUEUE') { $squeue += $line }
}

# Split A100s by memory using node features (a100-80gb / a100-40gb) or job constraints
function Resolve-GpuType([string]$type, [string]$features) {
    if ($type -eq 'a100') {
        if ($features -match 'a100-80gb') { return 'a100-80gb' }
        if ($features -match 'a100-40gb') { return 'a100-40gb' }
        return 'a100 (either)'
    }
    return $type
}

# ---- Nodes: total / used / free per GPU type ----
# States where a node takes no new work: its idle cards are not "free"
$unusable = '^(down|drain|drng|fail|maint|resv|inval|unk|futr|pow|boot|reboot)'
$nodes = @{}
foreach ($l in $sinfo) {
    $f = $l.Trim() -split '\s+'
    if ($f.Count -lt 9) { continue }
    $name, $part, $state, $gres, $used, $cpus, $memMB, $allocMB, $feat = $f[0..8]
    # CPUsState = allocated/idle/other/total
    $idleCpus = 0
    if ($cpus -match '^\d+/(\d+)/') { $idleCpus = [int]$matches[1] }
    $freeMemGB = [math]::Floor(([double]$memMB - [double]$allocMB) / 1024)
    if ($gres -notmatch 'gpu:([a-z0-9]+):(\d+)') { continue }
    $type = Resolve-GpuType $matches[1] $feat
    $total = [int]$matches[2]
    $nUsed = 0
    if ($used -match 'gpu:[a-z0-9]+:(\d+)') { $nUsed = [int]$matches[1] }
    $base = $state.TrimEnd('*', '~', '#', '!', '%', '$', '@', '^', '-', '+')
    if ($nodes.ContainsKey($name)) { $nodes[$name].Partitions += ",$part"; continue }  # node listed once per partition
    $nodes[$name] = [pscustomobject]@{
        Node = $name; Type = $type; Total = $total; Used = $nUsed; State = $base; Partitions = $part
        FreeCpus = $idleCpus; FreeMemGB = $freeMemGB
        Free = $(if ($base -match $unusable) { 0 } else { [math]::Max(0, $total - $nUsed) })
        Offline = $(if ($base -match $unusable) { $total - $nUsed } else { 0 })
    }
}

# ---- Queue: pending/running jobs and GPUs requested per type ----
$jobs = foreach ($l in $squeue) {
    $f = $l.Trim() -split '\s+', 9
    if ($f.Count -lt 8) { continue }
    $part, $state, $nn, $nt, $perNode, $perTask, $perJob, $feat = $f[0..7]
    $reason = $(if ($f.Count -ge 9) { $f[8] } else { '' })
    $gpus = 0; $type = $null
    foreach ($pair in @(@($perJob, 1), @($perNode, [int]$nn), @($perTask, [int]$nt))) {
        if ($pair[0] -match 'gres/gpu(?::([a-z0-9]+))?[:=](\d+)') {
            $type = $(if ($matches[1]) { $matches[1] } else { 'any' })
            $gpus = [int]$matches[2] * [math]::Max(1, $pair[1])
            break
        }
    }
    if (-not $type) { continue }  # CPU-only job in a GPU partition
    [pscustomobject]@{
        Partition = $part; State = $state; Gpus = $gpus; Reason = $reason
        Type = Resolve-GpuType $type $feat
    }
}

# ---- Report ----
# A multi-card job must fit on ONE node, so what matters is how many nodes have N cards free at once.
# Cards/CPUs/RAM on a node must all be free together; the node table shows all three.
$types = @($nodes.Values.Type) + @($jobs | ForEach-Object Type) | Sort-Object -Unique
$table = foreach ($t in $types) {
    $n = @($nodes.Values | Where-Object Type -eq $t)
    $fits = @($n | Where-Object { $_.FreeCpus -ge $MinCpus -and $_.FreeMemGB -ge $MinRamGB })
    # Jobs actually competing for cards now (not held / waiting on dependencies / over user limits)
    $q = @($jobs | Where-Object { $_.Type -eq $t -and $_.State -eq 'PENDING' -and
            $_.Reason -match '^(Priority|Resources|None|ReqNodeNotAvail)' })
    [pscustomobject]@{
        GPU          = $t
        Nodes        = $n.Count
        'Cards/node' = [int]($n | Measure-Object Total -Maximum).Maximum
        Cards        = [int]($n | Measure-Object Total -Sum).Sum
        FreeCards    = [int]($n | Measure-Object Free -Sum).Sum
        'Nodes 1+'   = @($fits | Where-Object Free -ge 1).Count
        'Nodes 2+'   = @($fits | Where-Object Free -ge 2).Count
        'Nodes 3+'   = @($fits | Where-Object Free -ge 3).Count
        Offline      = @($n | Where-Object Offline -gt 0).Count
        QueuedJobs   = $q.Count
        QueuedGPUs   = [int]($q | Measure-Object Gpus -Sum).Sum
    }
}

Write-Host "CARC Discovery GPU snapshot  $(Get-Date -Format 'yyyy-MM-dd HH:mm')" -ForegroundColor Cyan
Write-Host "Per type: 'Nodes N+' = nodes where an N-card job could start right now (N free cards + >= $MinCpus CPUs + >= $MinRamGB GB RAM free)." -ForegroundColor DarkGray
$table | Format-Table -AutoSize

Write-Host "Nodes with free cards (a job also needs its CPUs + RAM free on the same node):" -ForegroundColor Cyan
$free = $nodes.Values | Where-Object Free -ge 1 | Sort-Object Type, @{e = 'Free'; Descending = $true }, Node
if ($free) {
    $free | Select-Object Node, @{n = 'GPU'; e = { $_.Type } },
        @{n = 'Free/Total'; e = { "$($_.Free)/$($_.Total)" } },
        @{n = 'FreeCPUs'; e = { $_.FreeCpus } }, @{n = 'FreeRAM_GB'; e = { $_.FreeMemGB } },
        State, Partitions | Format-Table -AutoSize
} else { Write-Host '  (none)' }

Write-Host ("Total cards: {0}   Free now: {1}" -f [int]($table | Measure-Object Cards -Sum).Sum, [int]($table | Measure-Object FreeCards -Sum).Sum)
foreach ($part in 'gpu', 'debug') {
    $pj = @($jobs | Where-Object Partition -eq $part)
    Write-Host ("{0,-6} queue: {1} pending, {2} running (GPU jobs)" -f $part,
        @($pj | Where-Object State -eq 'PENDING').Count, @($pj | Where-Object State -eq 'RUNNING').Count)
}
Write-Host "Notes: free cards on drained/down/reserved nodes don't count ('Offline' = such nodes)." -ForegroundColor DarkGray
Write-Host "       State 'plnd' = Slurm is holding that node for a queued job; only short jobs can backfill onto it." -ForegroundColor DarkGray
Write-Host "       'a100 (either)' = A100 jobs with no 40/80gb constraint. 'any' = untyped --gpus requests." -ForegroundColor DarkGray
Write-Host "       Queued = pending for Priority/Resources (really competing for cards), not held/dependency/limit jobs." -ForegroundColor DarkGray

if ($Raw) {
    $jobs | Where-Object State -eq 'PENDING' | Group-Object { ($_.Reason -split '[,(]')[0].Trim() } |
        Sort-Object Count -Descending | Select-Object Count, @{n = 'PendingReason'; e = { $_.Name } } | Format-Table -AutoSize
}

if ($Raw) {
    Write-Host '--- raw sinfo ---'; $sinfo
    Write-Host '--- raw squeue ---'; $squeue
}
