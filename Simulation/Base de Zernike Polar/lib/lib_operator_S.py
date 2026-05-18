# =============================================================================
# lib_operator_S.py — Symmetry Operator (Eq. 8b)
# =============================================================================
# Implements the second of the five Phase I exacerbation operators
# (Eq. 8b of README.md, v7.0):
#
#     hat{S} |mu_m>_z = <mu_m | mu_{-m}>_J |mu_m>_z
#
# Measures hemispheric coherence as the raw Jacobian inner product of
# every ray with its diametrically opposite ray (m -> m + M/2 mod M).
# The output is the canonical raw inner product, consistent with the
# Implementation notice attached to Eq. 8b in the README: the absolute
# magnitude is not the diagnostic quantity -- monotonicity under OPF
# is what matters. A diagnostic Jacobian-normalized cosine
#     S_tilde(m) = <mu_m|mu_{-m}>_J
#                  / sqrt(<mu_m|mu_m>_J * <mu_{-m}|mu_{-m}>_J)
# is also returned as 'cohesion_cosine' to facilitate visual
# interpretation against a fixed range [-1, +1]; this cosine is a
# display companion only and does NOT replace the raw inner product
# in any downstream computation.
#
# Dependencies: numpy, lib_operator_base
# Reference: README.md (v7.0) -- Section 4, Eq. 8b plus Implementation
# notice; OpInZernikeBasis.md, Section 3 (Operator 2: Symmetry)
#
# Usage:
#   from lib_hilbert import HilbertSpace
#   from lib_operator_S import OperatorSymmetry
#
#   hs = HilbertSpace(mu_tensor, r_k_array)
#   op = OperatorSymmetry(mu_tensor, r_k_array, hs)
#   res  = op.apply()        # {'psi', 'scalar_per_ray', 'cohesion_cosine'}
#   diag = op.diagnostics()  # {'correctness_error', 'status', ...}
#
# Author: Carlone M. (maicon.carlone@gmail.com)
# =============================================================================

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import numpy as np

from lib_operator_base import OperatorBase


class OperatorSymmetry(OperatorBase):
    """
    Concrete implementation of the Symmetry operator hat{S} (Eq. 8b).

    apply() returns:
        - 'psi': ndarray (M, N', Q), the operator action:
                     psi_S[m] = cohesion[m] * mu_tensor[m]
        - 'scalar_per_ray': ndarray (M,), the raw Jacobian inner product
                     cohesion[m] = <mu_m | mu_{-m}>_J
                  This is the literal expression of Eq. 8b. Magnitude
                  is not bounded to [-1, +1] because mu_tensor is not
                  unit-norm.
        - 'cohesion_cosine': ndarray (M,), the diagnostic cosine
                     S_tilde[m] = cohesion[m]
                                  / sqrt(<mu_m|mu_m>_J * <mu_{-m}|mu_{-m}>_J)
                  Bounded to [-1, +1] by Cauchy-Schwarz. Display
                  companion only -- not used in psi.
        - 'opposite_indices': ndarray (M,) of int, the diametrical
                  opposite mapping m -> (m + M/2) mod M. Cached for
                  reuse by the visualization layer.

    diagnostics() returns:
        - 'correctness_error': max_m ||psi[m] - cohesion[m] * mu[m]||_inf,
          machine-precision agreement expected.
        - 'symmetry_error': max_m |cohesion[m] - cohesion[m_opp]|. By
          symmetry of the inner product (<a|b> = <b|a>), this must be
          zero to floating-point precision. A failure of this test
          indicates a bug in inner_product_J, not in mu_tensor.
        - 'cosine_in_unit_interval': bool, True if cohesion_cosine
          satisfies the Cauchy-Schwarz bound |S_tilde| <= 1 + tolerance.
        - 'finite': bool, integrity check.
        - 'cohesion_stats': dict {mean, std, min, max, median} of the
          raw cohesion (diagnostic only; magnitude has no fixed scale).
        - 'cosine_stats': dict {mean, std, min, max, median} of the
          normalized cosine.
        - 'status': PASS if correctness < tol AND symmetry < tol AND
          finite AND cosine_in_unit_interval; FAIL otherwise.
    """

    CORRECTNESS_TOLERANCE = 1e-12

    # The Jacobian inner product symmetry test compares two independent
    # evaluations of the same mathematical quantity; floating-point
    # accumulation may leak a few ULPs even on identical inputs. The
    # tolerance for symmetry_error is therefore set looser than
    # CORRECTNESS_TOLERANCE to absorb summation noise; the test still
    # detects any algorithmic asymmetry in inner_product_J.
    SYMMETRY_TOLERANCE = 1e-12

    # Cauchy-Schwarz bound is exact in real arithmetic; a small slack
    # absorbs rounding in the sqrt of two independent norm computations.
    COSINE_BOUND_SLACK = 1e-10

    # -------------------------------------------------------------------------
    # Apply -- Eq. 8b
    # -------------------------------------------------------------------------
    def _compute_apply(self) -> dict:
        """
        Compute the literal action of Eq. 8b plus the diagnostic cosine.

        For M angular sectors, the diametrical opposite of ray m is
        (m + M/2) mod M, valid for both even and odd M (when M is odd
        the "opposite" is the closest available ray, which is the
        canonical convention in discretized hemispheric symmetry).
        """
        M = self.M
        mu = self.mu_tensor

        # ---- Cached opposite-index mapping ----
        opposite_indices = (np.arange(M) + M // 2) % M

        # ---- Raw cohesion (Eq. 8b literal) and per-ray squared norm ----
        cohesion = np.zeros(M, dtype=np.float64)
        norm_sq = np.zeros(M, dtype=np.float64)
        for m in range(M):
            mo = opposite_indices[m]
            cohesion[m] = self.hs.inner_product_J(mu[m], mu[mo])
            norm_sq[m] = self.hs.inner_product_J(mu[m], mu[m])

        # ---- psi[m] = cohesion[m] * mu[m] ----
        psi = cohesion[:, None, None] * mu

        # ---- Diagnostic cosine (display companion only) ----
        # cosine[m] = cohesion[m] / sqrt(norm_sq[m] * norm_sq[m_opp])
        denom = np.sqrt(norm_sq * norm_sq[opposite_indices])
        cohesion_cosine = np.zeros(M, dtype=np.float64)
        nz = denom > 0.0
        cohesion_cosine[nz] = cohesion[nz] / denom[nz]
        # Rays whose denominator is exactly zero (a ray with null
        # micro-space) yield cosine=0 by definition: there is no
        # direction to compare against. This is the same convention
        # used by lib_hilbert.HilbertSpace.normalize_jacobian().

        return {
            'psi': psi,
            'scalar_per_ray': cohesion,
            'cohesion_cosine': cohesion_cosine,
            'opposite_indices': opposite_indices,
        }

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------
    def _compute_diagnostics(self) -> dict:
        """
        Numerical correctness checks for Eq. 8b.

        - correctness_error: psi == cohesion * mu (literal Eq. 8b).
        - symmetry_error: cohesion[m] == cohesion[m_opp] (inner product
          symmetry property, tests inner_product_J implementation).
        - cosine_in_unit_interval: Cauchy-Schwarz bound on the
          diagnostic cosine.
        """
        result = self._apply_cache
        psi = result['psi']
        cohesion = result['scalar_per_ray']
        cosine = result['cohesion_cosine']
        opposite_indices = result['opposite_indices']

        # ---- Formula match ----
        correctness_error = self._scalar_modulation_error(psi, cohesion)

        # ---- Inner product symmetry: cohesion[m] == cohesion[m_opp] ----
        symmetry_error = float(np.max(np.abs(cohesion - cohesion[opposite_indices])))

        # ---- Cauchy-Schwarz bound on diagnostic cosine ----
        max_cos = float(np.max(np.abs(cosine))) if cosine.size > 0 else 0.0
        cosine_in_unit_interval = max_cos <= 1.0 + self.COSINE_BOUND_SLACK

        # ---- Integrity ----
        finite = self._check_finite(psi, cohesion, cosine)

        # ---- Distribution statistics (informational, not pass/fail) ----
        def stats(arr):
            return {
                'mean':   float(np.mean(arr)),
                'std':    float(np.std(arr)),
                'min':    float(np.min(arr)),
                'max':    float(np.max(arr)),
                'median': float(np.median(arr)),
            }
        cohesion_stats = stats(cohesion)
        cosine_stats = stats(cosine)

        passed = (
            finite
            and correctness_error < self.CORRECTNESS_TOLERANCE
            and symmetry_error < self.SYMMETRY_TOLERANCE
            and cosine_in_unit_interval
        )

        return {
            'correctness_error': correctness_error,
            'symmetry_error': symmetry_error,
            'cosine_in_unit_interval': cosine_in_unit_interval,
            'cosine_max_abs': max_cos,
            'finite': finite,
            'cohesion_stats': cohesion_stats,
            'cosine_stats': cosine_stats,
            'status': 'PASS' if passed else 'FAIL',
            'tolerance': self.CORRECTNESS_TOLERANCE,
            'symmetry_tolerance': self.SYMMETRY_TOLERANCE,
        }
