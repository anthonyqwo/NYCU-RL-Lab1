$afterProc = Get-Process -Name 'td_afterstate' -ErrorAction SilentlyContinue
$stateProc = Get-Process -Name 'td_state' -ErrorAction SilentlyContinue

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "           NYCU RL Lab 1 - 2048 Training Progress" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# td_afterstate
Write-Host "`n[1] TD-Afterstate" -ForegroundColor Yellow
if ($afterProc) {
    $cpu = [math]::Round($afterProc.CPU, 1)
    $ram = [math]::Round($afterProc.WorkingSet64 / 1MB, 1)
    Write-Host "  Status: RUNNING (PID: $($afterProc.Id), CPU: ${cpu}s, RAM: ${ram} MB)" -ForegroundColor Green
} else {
    Write-Host "  Status: Not running or completed" -ForegroundColor DarkGray
}

$afterLog = "C:\Code\NYCU_RL\lab1\logs\td_afterstate_1200k_resume.log"
if (Test-Path $afterLog) {
    $lines = @(Get-Content $afterLog | Where-Object { $_ -match '^\d+\s+mean' })
    if ($lines.Count -gt 0) {
        $lastLine = $lines[-1]
        $tokens = $lastLine -split '\s+'
        if ($tokens.Count -ge 10) {
            $curEp = [int]$tokens[0]
            $mean = $tokens[3]
            $win = $tokens[6]
            $elapsed = [double]($tokens[9].TrimEnd('s'))
            $totalDone = 850000 + $curEp
            $pct = [math]::Round(($totalDone / 1200000) * 100, 2)
            $curPct = [math]::Round(($curEp / 350000) * 100, 1)
            $eps = if ($elapsed -gt 0) { [math]::Round($curEp / $elapsed, 1) } else { 0 }
            $remainSec = if ($eps -gt 0) { [math]::Round((350000 - $curEp) / $eps) } else { 0 }
            $etaMin = [math]::Round($remainSec / 60, 1)
            $etaHour = [math]::Round($etaMin / 60, 2)
            
            Write-Host "  Current session: $curEp / 350,000 (${curPct}%)"
            Write-Host "  Total progress : $totalDone / 1,200,000 (${pct}%)" -ForegroundColor Green
            Write-Host "  Latest metrics : Mean = $mean | Win2048 = $win"
            Write-Host "  Speed          : $eps ep/s | ETA: ${etaMin} min (~${etaHour} hr)"
        }
    }
}

# td_state
Write-Host "`n[2] TD-State" -ForegroundColor Yellow
if ($stateProc) {
    $cpu = [math]::Round($stateProc.CPU, 1)
    $ram = [math]::Round($stateProc.WorkingSet64 / 1MB, 1)
    Write-Host "  Status: RUNNING (PID: $($stateProc.Id), CPU: ${cpu}s, RAM: ${ram} MB)" -ForegroundColor Green
} else {
    Write-Host "  Status: Not running or completed" -ForegroundColor DarkGray
}

$stateLog = "C:\Code\NYCU_RL\lab1\logs\td_state_1200k_resume.log"
if (Test-Path $stateLog) {
    $lines = @(Get-Content $stateLog | Where-Object { $_ -match '^\d+\s+mean' })
    if ($lines.Count -gt 0) {
        $lastLine = $lines[-1]
        $tokens = $lastLine -split '\s+'
        if ($tokens.Count -ge 10) {
            $curEp = [int]$tokens[0]
            $mean = $tokens[3]
            $win = $tokens[6]
            $elapsed = [double]($tokens[9].TrimEnd('s'))
            $totalDone = 430000 + $curEp
            $pct = [math]::Round(($totalDone / 1200000) * 100, 2)
            $curPct = [math]::Round(($curEp / 770000) * 100, 1)
            $eps = if ($elapsed -gt 0) { [math]::Round($curEp / $elapsed, 1) } else { 0 }
            $remainSec = if ($eps -gt 0) { [math]::Round((770000 - $curEp) / $eps) } else { 0 }
            $etaMin = [math]::Round($remainSec / 60, 1)
            $etaHour = [math]::Round($etaMin / 60, 2)
            
            Write-Host "  Current session: $curEp / 770,000 (${curPct}%)"
            Write-Host "  Total progress : $totalDone / 1,200,000 (${pct}%)" -ForegroundColor Green
            Write-Host "  Latest metrics : Mean = $mean | Win2048 = $win"
            Write-Host "  Speed          : $eps ep/s | ETA: ${etaMin} min (~${etaHour} hr)"
        }
    }
}
Write-Host "============================================================" -ForegroundColor Cyan
