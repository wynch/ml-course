"""Module 00a — the perceptron, its mistake bound, and least squares.

Everything here is numpy-level and from scratch: no estimator library is used
for any algorithm the module teaches. scikit-learn appears only in
``tests/`` as an independent cross-check.
"""

from . import data, lstsq, mlp, mp_neuron, rosenblatt

__all__ = ["data", "lstsq", "mlp", "mp_neuron", "rosenblatt"]
