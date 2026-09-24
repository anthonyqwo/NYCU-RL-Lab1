"""Continue both 100k models to 1.2M episodes, then evaluate and plot results."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import traceback

ROOT = Path(__file__).resolve().parents[1]
BASE = '20260924_202933'
VARIANTS = ('td_state', 'td_afterstate')
ADDITIONAL = 1100000
TOTAL = 1200000


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def combine_and_measure(variant, run_id):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    # Plotting happens on the main thread after both workers have finished.
    combined = ROOT / f'logs/{variant}_1200k_{run_id}_train.csv'
    parts = [ROOT / f'logs/{variant}_100k_{BASE}_train.csv',
             ROOT / f'logs/{variant}_1200k_{run_id}_continuation.csv']
    xs, means, scores = [], [], []
    elapsed_offset = 0.0
    count = 0
    with combined.open('w', newline='', encoding='utf-8') as out:
        writer = csv.DictWriter(out, fieldnames=['episode','score','max_tile','moves','elapsed_seconds'])
        writer.writeheader()
        for part, expected in zip(parts, (100000, ADDITIONAL)):
            episode_offset = count
            with part.open(newline='') as f:
                for local_n, row in enumerate(csv.DictReader(f), 1):
                    if int(row['episode']) != local_n:
                        raise ValueError(f'Invalid episode sequence: {part}')
                    count += 1
                    row['episode'] = episode_offset + local_n
                    local_elapsed = float(row['elapsed_seconds'])
                    row['elapsed_seconds'] = f'{elapsed_offset + local_elapsed:.6f}'
                    writer.writerow(row)
                    scores.append(int(row['score']))
                    if count % 1000 == 0:
                        xs.append(count); means.append(sum(scores) / len(scores)); scores.clear()
            if local_n != expected:
                raise ValueError(f'Wrong episode count: {part}')
            elapsed_offset += local_elapsed
    assert count == TOTAL
    with (ROOT / f'logs/{variant}_1200k_{run_id}_eval.csv').open(newline='') as f:
        games = list(csv.DictReader(f))
    assert len(games) == 1000
    result = {'total_training_episodes': count, 'training_elapsed_seconds': elapsed_offset,
              'evaluation_games': len(games),
              'mean_score': sum(int(r['score']) for r in games) / len(games),
              'max_score': max(int(r['score']) for r in games),
              'wins_2048': sum(int(r['max_tile']) >= 2048 for r in games)}
    result['win_rate_2048_percent'] = result['wins_2048'] / 10
    plt.plot(xs, means, label=variant.replace('_', '-'), linewidth=1)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    run_id = args.run_id
    if not all(c.isdigit() or c == '_' for c in run_id):
        raise ValueError('run-id must contain only digits and underscores')
    status_file = ROOT / f'logs/run_1200k_{run_id}.json'
    if status_file.exists():
        raise FileExistsError(f'Run already exists: {status_file}')
    lock = threading.Lock()
    status = {'run_id': run_id, 'base_run_id': BASE, 'runner_pid': os.getpid(),
              'started_at': datetime.now().astimezone().isoformat(), 'status': 'running',
              'base_episodes': 100000, 'additional_episodes': ADDITIONAL, 'total_episodes': TOTAL,
              'alpha': 0.1, 'continuation_seed': 43, 'evaluation_seed': 2026,
              'checkpoint_every': 10000,
              'note': 'Weight-only continuation; RNG is reseeded. GUI and report still use the 100k run.',
              'variants': {}}

    def save_status():
        temporary = status_file.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(status_file)

    for variant in VARIANTS:
        model = ROOT / f'models/{variant}_100k_{BASE}.bin'
        executable = ROOT / f'build/{variant}.exe'
        source = ROOT / f'{variant}.cpp'
        for path in (model, executable, source):
            if not path.is_file(): raise FileNotFoundError(path)
        status['variants'][variant] = {'status': 'pending', 'base_model': str(model),
                                      'base_model_sha256': sha256(model),
                                      'executable_sha256': sha256(executable),
                                      'source_sha256': sha256(source)}
    save_status()

    def run_variant(variant):
        item = status['variants'][variant]
        executable = ROOT / f'build/{variant}.exe'
        model = ROOT / f'models/{variant}_1200k_{run_id}.bin'
        csv_path = ROOT / f'logs/{variant}_1200k_{run_id}_continuation.csv'
        console = ROOT / f'logs/{variant}_1200k_{run_id}_console.log'
        train = [str(executable), '--load', item['base_model'], '--episodes', str(ADDITIONAL),
                 '--alpha', '0.1', '--seed', '43', '--stats', '10000',
                 '--checkpoint-every', '10000', '--save', str(model), '--log', str(csv_path)]
        try:
            with console.open('w', encoding='utf-8') as out:
                proc = subprocess.Popen(train, stdout=out, stderr=subprocess.STDOUT, cwd=ROOT,
                                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                with lock:
                    item.update(status='training', pid=proc.pid, command=train,
                                model=str(model), training_csv=str(csv_path), console_log=str(console))
                    save_status()
                code = proc.wait()
                if code: raise RuntimeError(f'Training exited with code {code}; see {console}')
                with lock:
                    item['status'] = 'evaluating'; save_status()
                evaluation = [str(executable), '--eval', '--load', str(model), '--episodes', '1000',
                              '--seed', '2026', '--stats', '1000', '--log',
                              str(ROOT / f'logs/{variant}_1200k_{run_id}_eval.csv')]
                subprocess.run(evaluation, stdout=out, stderr=subprocess.STDOUT, cwd=ROOT, check=True,
                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            with lock:
                item.update(status='complete', model_sha256=sha256(model),
                            completed_at=datetime.now().astimezone().isoformat())
                save_status()
        except Exception as exc:
            with lock:
                item.update(status='failed', error=str(exc)); save_status()
            raise

    errors = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {pool.submit(run_variant, v): v for v in VARIANTS}
        for job in as_completed(jobs):
            try: job.result()
            except Exception as exc:
                errors.append(str(exc)); traceback.print_exc()
    if not errors:
        try:
            import matplotlib
            matplotlib.use('Agg')
            from matplotlib import pyplot as plt
            plt.figure(figsize=(10, 5), layout='constrained')
            results = {v: combine_and_measure(v, run_id) for v in VARIANTS}
            plt.axvline(100000, color='gray', linestyle='--', linewidth=1, label='Resume from 100k')
            plt.xlabel('Cumulative training episodes'); plt.ylabel('Mean score per 1,000 episodes')
            plt.title('2048: 1,200,000 training episodes per variant')
            plt.legend(); plt.grid(alpha=.2)
            plt.savefig(ROOT / f'logs/training_curve_1200k_{run_id}.png', dpi=180)
            plt.close()
            (ROOT / f'logs/results_1200k_{run_id}.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
        except Exception as exc:
            errors.append(str(exc)); traceback.print_exc()
    status['status'] = 'failed' if errors else 'complete'
    status['errors'] = errors
    status['finished_at'] = datetime.now().astimezone().isoformat()
    save_status()
    if errors: raise SystemExit(1)


if __name__ == '__main__':
    main()
