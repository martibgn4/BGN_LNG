"""Estimators.

Three layers, deliberately separate:

  ``envelope``     economics. Where the basis CAN sit, from regas cost and the
                   east-west arb. Needs no data fitting and does not change
                   when the sample does.
  ``ecm``          the conditional mean. Where the basis is GOING, given how
                   far it currently sits from the level fundamentals justify.
  ``seasonal_ou``  the distribution. How WIDE the answer is, and the parameter
                   set the desk's option pricer consumes.

Keeping them apart means a disagreement between them is visible. If the ECM's
implied half-life and the OU's disagree, one of them is misspecified, and that
is information rather than a number to average away.
"""

from nwebasis.model import ecm, envelope, seasonal_ou

__all__ = ["ecm", "envelope", "seasonal_ou"]
