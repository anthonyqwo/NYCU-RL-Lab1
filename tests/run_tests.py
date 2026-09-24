"""Compile and test both standalone submissions. Run from any directory."""
import csv
import hashlib
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
BUILD.mkdir(exist_ok=True)
COMPILER = os.environ.get("CXX", "g++")
SUFFIX = ".exe" if os.name == "nt" else ""


def run(args, success=True):
    result = subprocess.run([str(a) for a in args], cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if (result.returncode == 0) != success:
        raise RuntimeError(f"Unexpected exit {result.returncode}: {args}\n{result.stdout}")
    return result.stdout


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def scores(path):
    with path.open(newline="") as stream:
        return [(r["episode"], r["score"], r["max_tile"], r["moves"])
                for r in csv.DictReader(stream)]


for variant in ("td_state", "td_afterstate"):
    executable = BUILD / (variant + SUFFIX)
    run([COMPILER, "-std=c++11", "-O3", "-Wall", "-Wextra", "-Wpedantic",
         ROOT / (variant + ".cpp"), "-o", executable])
    unit = BUILD / ("test_" + variant + SUFFIX)
    flags = ["-DTEST_STATE"] if variant == "td_state" else []
    run([COMPILER, "-std=c++11", "-O2", "-Wall", "-Wextra", "-Wpedantic", *flags,
         ROOT / "tests/td_tests.cpp", "-o", unit])
    print(run([unit]), end="")
    model = BUILD / ("integration_" + variant + ".bin")
    train_log = BUILD / ("integration_" + variant + "_train.csv")
    log_a = BUILD / ("integration_" + variant + "_eval_a.csv")
    log_b = BUILD / ("integration_" + variant + "_eval_b.csv")
    broken = BUILD / ("integration_" + variant + "_broken.bin")
    wrong = BUILD / ("integration_" + variant + "_wrong.bin")
    print(run([executable, "--episodes", 30, "--stats", 17, "--seed", 42,
               "--checkpoint-every", 15, "--save", model, "--log", train_log]), end="")
    assert len(scores(train_log)) == 30
    before = digest(model)
    for path in (log_a, log_b):
        run([executable, "--eval", "--load", model, "--episodes", 20,
             "--seed", 123, "--stats", 13, "--log", path])
    assert scores(log_a) == scores(log_b), "evaluation is not reproducible"
    assert len(scores(log_a)) == 20
    assert digest(model) == before, "evaluation changed the model file"
    with model.open("rb") as stream:
        broken.write_bytes(stream.read(128))
    other = "td_afterstate" if variant == "td_state" else "td_state"
    wrong.write_bytes(f"RL2048-V1-{other}\n".encode())
    for args in (["--eval"], ["--episodes", "0"], ["--alpha", "nan"],
                 ["--seed", "-1"], ["--unknown", "x"],
                 ["--eval", "--load", "build/does-not-exist.bin"],
                 ["--eval", "--load", broken], ["--eval", "--load", wrong]):
        run([executable, *args], success=False)
    print(f"{variant}: CLI, checkpoint, model reload, deterministic evaluation and rejection tests passed.")
    # Remove only these individually named test outputs inside this workspace.
    for path in (model, train_log, log_a, log_b, broken, wrong):
        assert path.resolve().parent == BUILD.resolve()
        path.unlink()

print("All tests passed.")
