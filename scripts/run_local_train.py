"""CLI: generate synthetic data and train CRNN using configs/local.yaml."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    cmds = [
        [
            sys.executable,
            "scripts/generate_data.py",
            "--config",
            "configs/local.yaml",
            "--large",
            "--num-samples",
            "15000",
        ],
        [sys.executable, "-m", "src.recognition.train", "--config", "configs/local.yaml"],
    ]
    for cmd in cmds:
        print("Running:", " ".join(cmd))
        rc = subprocess.call(cmd, cwd=str(ROOT))
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
