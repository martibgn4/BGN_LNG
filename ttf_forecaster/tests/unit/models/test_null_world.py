"""Sanity check on the NULL synthetic world (prices independent of everything).

A ridge on random features must not beat the zero-return (forward) benchmark by more than noise.
This verifies the code path cannot manufacture skill; it is NOT a result about TTF.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tests.unit.models.conftest import null_world_returns
from ttf_forecaster.models.interface import make_model


def test_ridge_on_random_features_does_not_beat_zero_in_null_world() -> None:
    h = 5
    y = null_world_returns(horizon=h)
    rng = np.random.default_rng(123)
    X = pd.DataFrame(rng.normal(size=(len(y), 20)), index=y.index,
                     columns=[f"noise__f{j}" for j in range(20)])
    n = len(y)
    starts = [int(n * f) for f in (0.5, 0.65, 0.8)]
    ends = [*starts[1:], n]
    se_model, se_zero = [], []
    for s, e in zip(starts, ends, strict=True):
        tr = slice(0, s - h)  # purge h rows whose labels overlap the first test origin
        m = make_model("ridge", horizon=h).fit(X.iloc[tr], y.iloc[tr])
        z = make_model("zero_return", horizon=1).fit(X.iloc[tr], y.iloc[tr])
        yt = y.iloc[s:e]
        se_model.append(((yt - m.predict(X.iloc[s:e])) ** 2).to_numpy())
        se_zero.append(((yt - z.predict(X.iloc[s:e])) ** 2).to_numpy())
    r2 = 1 - np.concatenate(se_model).mean() / np.concatenate(se_zero).mean()
    assert r2 < 0.02, f"ridge 'beats' the forward benchmark in the null world: R2_oos={r2:.4f}"
