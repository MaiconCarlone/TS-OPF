# =============================================================================
# lib_operator_Dr.py — Radial Divergence Operator (Eq. 8d)
# =============================================================================
# Implements the fourth of the five Phase I exacerbation operators
# (Eq. 8d of README.md, v7.0):
#
#     hat{D}_r |mu_m>_z = ( sum_{k=1..N'-1}
#         ln( || mu_{k+1,m} - mu_{k,m} ||_2 + epsilon ) ) |mu_m>_z
#
# Measures local radial texture discontinuity. The L^2 norm of the
# difference between adjacent radial packets captures simultaneous
# changes in all Q Zernike modes (Piston, Tilt-X, Tilt-Y, Defocus);
# the logarithm with regularization epsilon compresses smooth tissue
# variations and amplifies abrupt transitions characteristic of
# anomaly boundaries. This is the operator most sensitive to
# isointense micro-lesions, which are invisible to hat{O} (no DC
# perturbation) but produce a local discontinuity in the higher-order
# Zernike modes.
#
# Dependencies: numpy, lib_operator_base
# Reference: README.md (v7.0) -- Section 4, Eq. 8d, plus Conventions
# table for epsilon ~ 1e-6; OpInZernikeBasis.md, Section 3
# (Operator 4: Radial Divergence)
#
# Usage:
#   from lib_hilbert import HilbertSpace
#   from lib_operator_Dr import OperatorRadialDivergence
#
#   hs = HilbertSpace(mu_tensor, r_k_array)
#   op = OperatorRadialDivergence(mu_tensor, r_k_array, hs)
#   res  = op.apply()        # {'psi', 'scalar_per_ray', 'diff_norm_map'}
#   diag = op.diagnostics()
#
# Author: Carlone M. (maicon.carlone@gmail.com)
# =============================================================================

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import numpy as np

from lib_operator_base import OperatorBase


class OperatorRadialDivergence(OperatorBase):
    """
    Concrete implementation of the Radial Divergence operator hat{D}_r
    (Eq. 8d).

    apply() returns:
        - 'psi': ndarray (M, N', Q),
                     psi_Dr[m] = dr_scalar[m] * mu_tensor[m].
        - 'scalar_per_ray': ndarray (M,),
                     dr_scalar[m] = sum_k ln( ||delta_k||_2 + epsilon )
                  where delta_k = mu_tensor[m, k+1] - mu_tensor[m, k]
                  is the per-band difference vector in R^Q.
        - 'diff_norm_map': ndarray (M, N'-1), the raw Euclidean norm
                  of the inter-band difference vector for every
                  (m, k). Diagnostic auxiliary -- not multiplied by mu.
        - 'log_diff_map': ndarray (M, N'-1), the per-element
                  ln( ||delta_k||_2 + epsilon ). Useful as a display
                  overlay (sugestao F5 of the parecer): the heatmap
                  of log_diff_map exposes where the logarithmic
                  compression vs amplification regime acts.

    diagnostics() returns:
        - 'correctness_error': psi == dr_scalar * mu, machine precision.
        - 'finite': integrity check on all four returned arrays.
        - 'dr_stats': dict {mean, std, min, max, median} of dr_scalar.
        - 'epsilon_floor': float, ln(epsilon). A ray with dr_scalar
          numerically equal to (N'-1) * ln(epsilon) within tolerance
          indicates pure homogeneity along the radius (||delta|| == 0
          everywhere); typically background sectors. The diagnostic
          'pure_homogeneous_rays' counts such rays.
        - 'pure_homogeneous_rays': int, number of rays for which the
          radial divergence saturates the epsilon floor.
        - 'status': PASS / FAIL.
    """

    EPSILON = 1e-6
    CORRECTNESS_TOLERANCE = 1e-12

    # Tolerance for detecting a ray that has saturated the epsilon
    # floor. With N'-1 bands and ||delta|| identically zero, the
    # operator returns (N'-1) * ln(EPSILON). We declare a ray as
    # pure-homogeneous when dr_scalar is within EPSILON_FLOOR_SLACK of
    # this minimum.
    EPSILON_FLOOR_SLACK = 1e-9

    # -------------------------------------------------------------------------
    # Apply -- Eq. 8d
    # -------------------------------------------------------------------------
    def _compute_apply(self) -> dict:
        """
        Compute the literal action of Eq. 8d, vectorized along the
        angular axis.

        Strategy:
            1. delta = mu_tensor[:, 1:, :] - mu_tensor[:, :-1, :]
                shape (M, N'-1, Q): the per-band difference vector.
            2. diff_norm_map = ||delta||_2 along axis=2
                shape (M, N'-1): the L^2 norm in R^Q at each (m, k).
            3. log_diff_map = ln(diff_norm_map + epsilon)
                shape (M, N'-1).
            4. dr_scalar = sum over k of log_diff_map
                shape (M,).
            5. psi[m] = dr_scalar[m] * mu_tensor[m].
        """
        mu = self.mu_tensor
        eps = self.EPSILON

        # Step 1: per-band difference vector in R^Q
        delta = mu[:, 1:, :] - mu[:, :-1, :]

        # Step 2: L^2 norm over the Q-mode axis
        diff_norm_map = np.linalg.norm(delta, axis=2)

        # Step 3: regularized logarithm
        log_diff_map = np.log(diff_norm_map + eps)

        # Step 4: per-ray scalar
        dr_scalar = np.sum(log_diff_map, axis=1)

        # Step 5: psi[m] = dr_scalar[m] * mu[m]
        psi = dr_scalar[:, None, None] * mu

        return {
            'psi': psi,
            'scalar_per_ray': dr_scalar,
            'diff_norm_map': diff_norm_map,
            'log_diff_map': log_diff_map,
        }

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------
    def _compute_diagnostics(self) -> dict:
        """
        Numerical correctness checks for Eq. 8d.
        """
        result = self._apply_cache
        psi = result['psi']
        dr_scalar = result['scalar_per_ray']
        diff_norm_map = result['diff_norm_map']
        log_diff_map = result['log_diff_map']

        correctness_error = self._scalar_modulation_error(psi, dr_scalar)
        finite = self._check_finite(psi, dr_scalar, diff_norm_map, log_diff_map)

        # Saturation against the epsilon floor: a ray of pure
        # homogeneity (||delta|| == 0 everywhere) yields the minimum
        # possible dr_scalar = (N'-1) * ln(epsilon).
        floor = (self.N_prime - 1) * np.log(self.EPSILON)
        pure_homogeneous_rays = int(np.sum(
            np.abs(dr_scalar - floor) < self.EPSILON_FLOOR_SLACK
        ))

        dr_stats = {
            'mean':   float(np.mean(dr_scalar)),
            'std':    float(np.std(dr_scalar)),
            'min':    float(np.min(dr_scalar)),
            'max':    float(np.max(dr_scalar)),
            'median': float(np.median(dr_scalar)),
        }

        passed = (
            finite
            and correctness_error < self.CORRECTNESS_TOLERANCE
            and dr_stats['std'] > 0.0
        )

        return {
            'correctness_error': correctness_error,
            'finite': finite,
            'dr_stats': dr_stats,
            'epsilon_floor': float(floor),
            'pure_homogeneous_rays': pure_homogeneous_rays,
            'status': 'PASS' if passed else 'FAIL',
            'tolerance': self.CORRECTNESS_TOLERANCE,
            'epsilon': self.EPSILON,
        }
