$r = Get-Content logs/run_1200k_20260924_232129.json -Raw | ConvertFrom-Json
$r.status
$r.variants.td_state.status
$r.variants.td_afterstate.status