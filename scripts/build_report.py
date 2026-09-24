"""Build the editable XeLaTeX source instead of the old ReportLab layout."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--xelatex', help='Path to xelatex if it is not on PATH')
args = parser.parse_args()
local = Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs/MiKTeX/miktex/bin/x64/xelatex.exe'
compiler = args.xelatex or shutil.which('xelatex') or (str(local) if local.is_file() else None)
if compiler is None:
    raise SystemExit('XeLaTeX not found. Use --xelatex PATH.')
source = ROOT / 'output/latex/RL_Lab1_Report.tex'
build = ROOT / 'tmp/pdfs/xelatex-build'
build.mkdir(parents=True, exist_ok=True)
output = ROOT / 'output/pdf/RL_Lab1_Report.pdf'
output.parent.mkdir(parents=True, exist_ok=True)
for _ in range(2):
    subprocess.run([compiler, '--disable-installer', '-interaction=nonstopmode',
                    '-halt-on-error', f'--output-directory={build.as_posix()}', source.name],
                   cwd=source.parent, check=True)
log = (build / 'RL_Lab1_Report.log').read_text(encoding='utf-8', errors='replace')
for issue in ('Missing character:', 'Overfull \\hbox', 'Overfull \\vbox'):
    if issue in log:
        raise SystemExit(f'Layout verification failed: {issue}; inspect {build}')
shutil.copyfile(build / output.name, output)
print(output)
