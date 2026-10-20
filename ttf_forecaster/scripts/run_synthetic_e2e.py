"""Run the SYNTHETIC end-to-end pipeline (both worlds) and write the report. Architecture check only."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ttf_forecaster.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["synthetic-e2e", *sys.argv[1:]]))
