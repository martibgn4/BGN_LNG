"""Calibration diagnostics - plots of how the data behaved, per tenor.

    python diagnose.py                      # latest calibration
    python diagnose.py --calib 2026-09-24

See src/deliveryoption/diagnostics.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # BGN_LNG

from DeliveryOptionModel.src.deliveryoption.diagnostics import main  # noqa: E402

if __name__ == "__main__":
    main()
