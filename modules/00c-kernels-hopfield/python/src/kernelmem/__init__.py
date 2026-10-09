"""Module 00c — kernels, memory, and the modern bridge.

Four small from-scratch pieces:

- ``svm``      — the hard-margin SVM in its dual form, solved by projected
                 gradient ascent, with linear / polynomial / RBF kernels.
- ``lift``     — the explicit quadratic feature map behind the polynomial
                 kernel, so you can see the 2D → 3D lift instead of trusting it.
- ``hopfield`` — the classical binary Hopfield network (Hebbian storage,
                 asynchronous updates, energy descent) and its modern
                 softmax cousin, which is one attention step.
- ``polyreg``  — minimum-norm polynomial regression, the vehicle for the
                 bias-variance U and the double-descent spike.
"""

from . import hopfield, lift, polyreg, svm

__all__ = ["hopfield", "lift", "polyreg", "svm"]
