#!/usr/bin/env python3
"""Verify gating artifacts for a given step before it can start.

Usage:
    python scripts/check_gate.py 02   # check if Step 02 can begin
    python scripts/check_gate.py all   # check all steps

Exit code 0 = gate satisfied, 1 = missing prerequisites.
"""

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAVED = ROOT / "model" / "saved"
MANIFEST = SAVED / "MANIFEST.json"


def _manifest_has(filename: str) -> bool:
    if not MANIFEST.exists():
        return False
    manifest = json.loads(MANIFEST.read_text())
    return any(a["filename"] == filename for a in manifest.get("artifacts", []))


def _function_exists(filepath: str, func_name: str) -> bool:
    path = ROOT / filepath
    if not path.exists():
        return False
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == func_name:
                return True
    return False


GATES = {
    "01": [],
    "02": [
        ("function", "model/data/descriptors.py", "build_features"),
    ],
    "03": [
        ("file", "model/saved/nn_pcsaft.pt"),
    ],
    "04": [
        ("file", "model/registry.py"),
    ],
    "05": [
        ("any_file", [
            "model/saved/rf_m.joblib",
            "model/saved/nn_pcsaft.pt",
        ]),
    ],
    "06": [
        ("file", "serving/app.py"),
        ("file", "serving/model_loader.py"),
    ],
    "07": [
        ("dir", "k8s"),
    ],
    "08": [
        ("file", "serving/app.py"),
        ("file", "pipeline/pipeline.py"),
    ],
}


def check_gate(step: str) -> tuple[bool, list[str]]:
    """Return (satisfied, list_of_missing) for a step."""
    checks = GATES.get(step, [])
    missing = []
    for check in checks:
        kind = check[0]
        if kind == "function":
            filepath, func_name = check[1], check[2]
            if not _function_exists(filepath, func_name):
                missing.append(f"function {func_name} in {filepath}")
        elif kind == "file":
            if not (ROOT / check[1]).exists():
                missing.append(f"file {check[1]}")
        elif kind == "dir":
            d = ROOT / check[1]
            if not d.exists() or not any(d.iterdir()):
                missing.append(f"directory {check[1]} (non-empty)")
        elif kind == "any_file":
            if not any((ROOT / f).exists() for f in check[1]):
                missing.append(f"any of {check[1]}")
    return len(missing) == 0, missing


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/check_gate.py <step_number|all>")
        sys.exit(1)

    target = sys.argv[1]

    if target == "all":
        steps = sorted(GATES.keys())
    else:
        steps = [target.zfill(2)]

    all_ok = True
    for step in steps:
        ok, missing = check_gate(step)
        status = "OK" if ok else "BLOCKED"
        print(f"Step {step}: {status}")
        for m in missing:
            print(f"  missing: {m}")
        if not ok:
            all_ok = False

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
