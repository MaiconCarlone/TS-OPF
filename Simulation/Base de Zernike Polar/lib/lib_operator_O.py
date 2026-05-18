# =============================================================================
# lib_operator_O.py — Intensity Operator (Eq. 8a)
# =============================================================================
# Implements the first of the five Phase I exacerbation operators (Eq. 8a
# of README.md, v7.0):
#
#     hat{O} |mu_m>_z = ( 1/N' * sum_k mu_1^{(k,m)} ) |mu_m>_z
#
# Reads the Piston coefficient (q=1, Python index 0) along the radial
# axis of every angular sector m, averages it across the N' radial
# bands, and rescales the entire tensor of that ray by the resulting
# scalar. Coefficients q=2,3,4 (Tilt-X, Tilt-Y, Defocus) are passive
# riders: they are scaled together with q=1 but do not influence the
# scalar amplitude. This is the operator for which gross intensity
# anomalies (e.g., hyperintense lesions) produce the strongest signal;
# isointense anomalies must be detected by texture operators
# (D_r, R) or by hemispheric symmetry breaking (S, chi).
#
# Dependencies: numpy, lib_operator_base
# Reference: README.md (v7.0) -- Section 4, Eq. 8a; OpInZernikeBasis.md,
# Section 3 (Operator 1: Intensity)
#
# Usage:
#   from lib_hilbert import HilbertSpace
#   from lib_operator_O import OperatorIntensity
#
#   hs = HilbertSpace(mu_tensor, r_k_array)   # only inner_product_J is used
#   op = OperatorIntensity(mu_tensor, r_k_array, hs)
#   res  = op.apply()        # {'psi': ..., 'scalar_per_ray': ...}
#   diag = op.diagnostics()  # {'correctness_error': ..., 'status': ..., ...}
#
# Author: Carlone M. (maicon.carlone@gmail.com)
# =============================================================================

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import numpy as np

from lib_operator_base import OperatorBase


class OperatorIntensity(OperatorBase):
    """
    Concrete implementation of the Intensity operator hat{O} (Eq. 8a).

    apply() returns:
        - 'psi': ndarray (M, N', Q), the operator action on the full
                 tensor. Literal reproduction of Eq. 8a:
                     psi_O[m] = scalar_O[m] * mu_tensor[m]
        - 'scalar_per_ray': ndarray (M,), the per-ray Piston average:
                     scalar_O[m] = (1/N') * sum_k mu_tensor[m, k, 0]

    diagnostics() returns:
        - 'correctness_error': float, the maximum absolute deviation
          between psi and the literal product scalar_per_ray * mu_tensor.
          A correct implementation yields machine-precision agreement
          (typically 0.0).
        - 'finite': bool, True if neither psi nor scalar_per_ray
          contain NaN or Inf.
        - 'piston_range': float, max(scalar_per_ray) - min(scalar_per_ray).
          A non-zero range is a necessary (but not sufficient) condition
          for the operator to provide angular discrimination.
        - 'status': 'PASS' if correctness_error < 1e-12, finite is True,
          and piston_range > 0. 'FAIL' otherwise.
        - 'tolerance': float, the threshold applied to correctness_error.
    """

    # Numerical tolerance for the formula-match diagnostic. Since the
    # operator action is a literal scalar product, machine precision is
    # expected. The tolerance is set conservatively at 1e-12 to absorb
    # accumulated rounding from the np.mean() reduction.
    CORRECTNESS_TOLERANCE = 1e-12

    # -------------------------------------------------------------------------
    # Apply -- Eq. 8a
    # -------------------------------------------------------------------------
    def _compute_apply(self) -> dict:
        """
        Compute the literal action of Eq. 8a on the cached mu_tensor.

        Strategy: fully vectorized along the angular axis to avoid
        Python-level loops. The Piston coefficient column is
        mu_tensor[:, :, 0]; its average over the radial axis (axis=1)
        yields the (M,)-shaped scalar_per_ray. Broadcasting
        scalar_per_ray[:, None, None] against mu_tensor produces psi.
        """
        # Piston coefficient (q=1 in the README convention, index 0 in Python)
        # along the radial axis of every ray: shape (M, N')
        piston_profile = self.mu_tensor[:, :, 0]

        # (1/N') * sum_k mu_1^{(k,m)} for every m: shape (M,)
        scalar_per_ray = np.mean(piston_profile, axis=1)

        # psi[m] = scalar[m] * mu[m]: broadcast (M,)*(M,N',Q) -> (M,N',Q)
        psi = scalar_per_ray[:, None, None] * self.mu_tensor

        return {
            'psi': psi,
            'scalar_per_ray': scalar_per_ray,
        }

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------
    def _compute_diagnostics(self) -> dict:
        """
        Numerical correctness checks for Eq. 8a.

        - correctness_error: machine-precision agreement between psi and
          scalar_per_ray * mu_tensor.
        - finite: integrity of NaN/Inf in both outputs.
        - piston_range: a degenerate operator (range == 0) provides no
          angular discrimination.
        """
        result = self._apply_cache
        psi = result['psi']
        scalar_per_ray = result['scalar_per_ray']

        correctness_error = self._scalar_modulation_error(psi, scalar_per_ray)
        finite = self._check_finite(psi, scalar_per_ray)
        piston_range = float(np.max(scalar_per_ray) - np.min(scalar_per_ray))

        passed = (
            finite
            and correctness_error < self.CORRECTNESS_TOLERANCE
            and piston_range > 0.0
        )

        return {
            'correctness_error': correctness_error,
            'finite': finite,
            'piston_range': piston_range,
            'status': 'PASS' if passed else 'FAIL',
            'tolerance': self.CORRECTNESS_TOLERANCE,
        }
