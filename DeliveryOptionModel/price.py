"""Job 2 - pricing off calibration files. See src/deliveryoption/price_job.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # BGN_LNG

from DeliveryOptionModel.src.deliveryoption.price_job import main  # noqa: E402

if __name__ == "__main__":
    main()
