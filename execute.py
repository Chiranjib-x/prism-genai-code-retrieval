"""Execution verification: does a candidate program solve the query's examples?

APPS problem statements carry worked examples ("-----Examples-----" then
Input/Output pairs) and the corpus is standalone stdin->stdout Python. So
relevance can be *checked*, not just estimated: run the candidate on each
example input and compare its output.

Sandbox: every run is a fresh `python -I` subprocess (isolated mode: no user
site, no PYTHON* env, cwd not on sys.path) in a throwaway temp dir, with a
hard timeout and a stripped environment.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

EXAMPLES_HEAD = re.compile(r"-{3,}\s*Examples?\s*-{3,}", re.I)
NOTE_HEAD = re.compile(r"-{3,}\s*Notes?\s*-{3,}", re.I)
PAIR = re.compile(r"^\s*Input\s*\n(.*?)\n\s*Output\s*\n(.*?)(?=^\s*Input\s*$|\Z)",
                  re.S | re.M)

# ponytail: timeout + isolated interpreter + temp cwd + stripped env. Windows has
# no rlimit, so no memory cap or network block here -- the Docker image adds
# --network none and --memory. Upgrade path if run bare: Windows job objects.
# Examples are tiny; correct programs finish in ~0.1s. 3s is headroom for load --
# never run execution while an encoder saturates the CPU (spurious timeouts).
TIMEOUT_S = 3.0
_ENV = {"SYSTEMROOT": os.environ.get("SYSTEMROOT", ""), "PYTHONIOENCODING": "utf-8"}
_WORKDIR = Path(tempfile.mkdtemp(prefix="apps_exec_"))


RESULTS_LOG = Path("cache/exec.tsv")   # append-only "<doc>|<examples-hash>\t0|1"; gitignored


def load_results() -> dict[str, bool]:
    """Every check ever run. Append-only, so a killed run loses nothing but its
    last unflushed lines -- rerunning resumes instead of starting over."""
    out: dict[str, bool] = {}
    if RESULTS_LOG.exists():
        for line in RESULTS_LOG.open(encoding="utf-8"):
            key, _, val = line.rstrip("\n").partition("\t")
            if val in ("0", "1"):            # skip a line truncated mid-write
                out[key] = val == "1"
    return out


def parse_examples(query: str) -> list[tuple[str, str]]:
    """Extract (stdin, expected stdout) pairs. [] if the query has none."""
    m = EXAMPLES_HEAD.search(query)
    if not m:
        return []
    body = NOTE_HEAD.split(query[m.end():])[0]
    return [(i.strip() + "\n", o.strip()) for i, o in PAIR.findall(body)]


def same_output(got: str, want: str) -> bool:
    """Token-wise equality, case-insensitive (judges accept YES/Yes/yes);
    numbers compare with 1e-6 tolerance (the usual judge rule)."""
    a, b = got.split(), want.split()
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x.lower() == y.lower():
            continue
        try:
            fx, fy = float(x), float(y)
        except ValueError:
            return False
        if abs(fx - fy) > 1e-6 * max(1.0, abs(fy)):
            return False
    return True


def _code_file(code: str) -> Path:
    path = _WORKDIR / f"{hashlib.sha1(code.encode()).hexdigest()}.py"
    if not path.exists():
        path.write_text(code, encoding="utf-8")
    return path


def run(code: str, stdin: str) -> str | None:
    """Candidate's stdout on this input, or None on timeout."""
    with tempfile.TemporaryDirectory(dir=_WORKDIR) as cwd:
        try:
            p = subprocess.run([sys.executable, "-I", str(_code_file(code))],
                               input=stdin, capture_output=True, text=True,
                               encoding="utf-8", errors="replace",
                               timeout=TIMEOUT_S, cwd=cwd, env=_ENV)
        except subprocess.TimeoutExpired:
            return None
    return p.stdout


def passes(code: str, examples: list[tuple[str, str]]) -> bool:
    """True only if the candidate reproduces every example. Stops at first miss."""
    for stdin, want in examples:
        got = run(code, stdin)
        if got is None or not same_output(got, want):
            return False
    return bool(examples)


if __name__ == "__main__":
    query = """Find the max.

-----Input-----
Two ints.

-----Examples-----
Input
3 5

Output
5

Input
-2 -7

Output
-2

-----Note-----
Input
ignore this
"""
    ex = parse_examples(query)
    assert ex == [("3 5\n", "5"), ("-2 -7\n", "-2")], ex
    assert passes("a,b=map(int,input().split());print(max(a,b))", ex)
    assert not passes("a,b=map(int,input().split());print(min(a,b))", ex)
    assert not passes("while True: pass", ex[:1])                     # timeout
    assert not passes("import sys; sys.exit(1)", ex)                  # no output
    assert same_output("0.3333333", "0.33333333") and not same_output("1 2", "1 2 3")
    assert same_output("Yes\nNO", "yes\nno") and not same_output("yes", "no")
    assert parse_examples("no examples here") == []
    print("execute self-check passed")
