"""Real-data ingestion entry point — intentionally not implemented before the data boundary.

Live vendor data needs identifiers and explicit authorisation (docs/live_data_requirements.md §8).
Use scripts/run_synthetic_e2e.py to exercise the pipeline on synthetic data.
"""

import sys

if __name__ == "__main__":
    sys.exit("run_ingestion: real-data runs are blocked at the data boundary; see docs/live_data_requirements.md")
