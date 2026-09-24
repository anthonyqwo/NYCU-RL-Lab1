"""Resume interrupted 1200k training from last checkpoints.

Uses cmd.exe wrappers so output redirection is managed by cmd, not Python.
The cmd.exe processes are started with WindowStyle Hidden via PowerShell.
"""
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    run_id = '20260924_232129'
    resume_ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    tasks = [
        ('td_state',      330_000, 1_100_000),
        ('td_afterstate', 750_000, 1_100_000),
    ]

    pids = {}
    for name, done, target in tasks:
        remaining = target - done
        ckpt  = ROOT / f'models/{name}_1200k_{run_id}.bin'
        exe   = ROOT / f'build/{name}.exe'
        c_log = ROOT / f'logs/{name}_1200k_resume_{resume_ts}.log'
        c_csv = ROOT / f'logs/{name}_1200k_resume_{resume_ts}.csv'

        for p in (ckpt, exe):
            if not p.exists():
                sys.exit(f'Missing: {p}')

        # Write a tiny .bat that cmd.exe will run with its own output redirection
        bat = ROOT / f'tmp/{name}_resume.bat'
        bat.parent.mkdir(exist_ok=True)
        bat.write_text(
            f'@"{exe}" --load "{ckpt}" --episodes {remaining}'
            f' --alpha 0.1 --seed 44 --stats 10000'
            f' --checkpoint-every 10000'
            f' --save "{ckpt}"'
            f' --log "{c_csv}"'
            f' > "{c_log}" 2>&1\n',
            encoding='utf-8',
        )

        # Launch via PowerShell Start-Process → fully independent hidden window
        ps_cmd = (
            f'Start-Process cmd.exe -ArgumentList \'/c "{bat}"\''
            f' -WindowStyle Hidden -WorkingDirectory "{ROOT}"'
        )
        subprocess.run(['powershell', '-Command', ps_cmd], check=True)

        # Find the PID of the exe (most recent by name)
        import time; time.sleep(1)
        r = subprocess.run(
            ['powershell', '-Command',
             f'(Get-Process {name} -ErrorAction SilentlyContinue | Sort-Object StartTime -Descending | Select-Object -First 1).Id'],
            capture_output=True, text=True)
        pid = int(r.stdout.strip()) if r.stdout.strip() else '?'
        pids[name] = pid

        print(f'{name}: PID {pid}, {remaining:,} episodes remaining')
        print(f'  console: logs/{c_log.name}')
        print(f'  csv:     logs/{c_csv.name}')
        print()

    info = {
        'resume_ts': resume_ts,
        'original_run_id': run_id,
        'started': datetime.now().astimezone().isoformat(),
        'seed': 44,
        'pids': pids,
        'tasks': {name: {'checkpoint_episodes': done, 'remaining': target - done}
                  for name, done, target in tasks},
    }
    out = ROOT / f'logs/resume_{resume_ts}.json'
    out.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Resume info: logs/{out.name}')
    print('Processes are fully independent — safe to close this terminal.')


if __name__ == '__main__':
    main()
