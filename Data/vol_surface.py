
import numpy as np
from scipy.interpolate import RegularGridInterpolator

class VolSurface:
    """Volatility surface with axes: strikes and tenors.
    vol_surface: dict[strike -> dict[tenor -> vol]]
    interpolation: 'linear' supported.
    extrapolation: by clamping to bounds.
    """
    def __init__(self, vol_surface):
        strikes = sorted(vol_surface.keys())
        tenors = sorted(next(iter(vol_surface.values())).keys())
        Z = np.array([[vol_surface[s][t] for t in tenors] for s in strikes], dtype=float)
        self.strikes = np.array(strikes, dtype=float)
        self.tenors = np.array(tenors, dtype=float)
        self._interp = RegularGridInterpolator(
            (self.strikes, self.tenors), Z, method='linear',
            bounds_error=False, fill_value=None
        )

    def _clamp(self, x, grid):
        return np.clip(x, grid[0], grid[-1])

    def vol(self, strike, tenor):
        s = self._clamp(float(strike), self.strikes)
        t = self._clamp(float(tenor), self.tenors)
        val = self._interp([[s, t]])[0]
        return float(val)
