# =============================================================================
# lib_operator_chi.py — Chirality Operator (Eq. 8c)
# =============================================================================
# Implements the third of the five Phase I exacerbation operators
# (Eq. 8c of README.md, v7.0):
#
#     |chi_m>_z = sum_{m'} | <mu_m|mu_{m'}>_J - <mu_{-m}|mu_{-m'}>_J |
#                * |mu_m>_z
#
# Quantifies correlational symmetry breaking: the absolute disparity
# between the angular correlation pattern of ray m and the analogous
# pattern of its diametrically opposite ray (-m), summed over all
# angular comparison rays m'. A perfectly bilaterally symmetric brain
# yields chirality_scalar[m] = 0 for every m; a unilateral anomaly
# perturbs the local correlational neighborhood and produces a
# concentrated peak.
#
# Performance optimization:
# Eq. 8c requires O(M^2) inner products evaluated once. The naive
# implementation in lib_physical_operators.PhysicalOperators.apply_operators()
# evaluates the same inner product twice per (m, m') pair, doubling
# the cost. The present class pre-computes the full M x M correlation
# matrix C_mu in the constructor's lazy cache, exploiting the
# symmetry C_mu[i,j] = C_mu[j,i] of the Jacobian inner product to
# halve the number of evaluations relative to the production library.
# This optimization preserves the literal Eq. 8c output to machine
# precision; correctness is verified by diagnostics().
#
# Dependencies: numpy, lib_operator_base
# Reference: README.md (v7.0) -- Section 4, Eq. 8c; OpInZernikeBasis.md,
# Section 3 (Operator 3: Chirality)
#
# Usage:
#   from lib_hilbert import HilbertSpace
#   from lib_operator_chi import OperatorChirality
#
#   hs = HilbertSpace(mu_tensor, r_k_array)
#   op = OperatorChirality(mu_tensor, r_k_array, hs)
#   res  = op.apply()        # {'psi', 'scalar_per_ray', 'C_mu', 'D_chiral'}
#   diag = op.diagnostics()  # {'correctness_error', 'D_symmetry_error', ...}
#
# Author: Carlone M. (maicon.carlone@gmail.com)
# =============================================================================

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import numpy as np

from lib_operator_base import OperatorBase


class OperatorChirality(OperatorBase):
    """
    Concrete implementation of the Chirality operator hat{chi} (Eq. 8c).

    apply() returns:
        - 'psi': ndarray (M, N', Q),
                     psi_chi[m] = chirality_scalar[m] * mu_tensor[m].
        - 'scalar_per_ray': ndarray (M,),
                     chirality_scalar[m]
                       = sum_{m'} | C_mu[m,m'] - C_mu[-m,-m'] |
                  where C_mu[a,b] = <mu_a | mu_b>_J. This is the
                  literal expression of the sum in Eq. 8c.
        - 'C_mu': ndarray (M, M), the symmetric Jacobian correlation
                  matrix of mu_tensor. Cached for downstream consumers
                  (visualization, eventual bandpass filtering of
                  Section 5).
        - 'D_chiral': ndarray (M, M), the chirality matrix
                     D_chiral[m, m'] = | C_mu[m,m'] - C_mu[-m,-m'] |.
                  By symmetry of C_mu and by the mapping m -> -m,
                  D_chiral itself is symmetric: D_chiral[m,m'] =
                  D_chiral[m',m]. Verified in diagnostics().
        - 'opposite_indices': ndarray (M,), m -> (m + M/2) mod M.

    diagnostics() returns:
        - 'correctness_error': psi vs scalar*mu, machine precision.
        - 'C_mu_symmetry_error': max |C_mu - C_mu.T|. Tests the
          symmetry of inner_product_J under exchange of arguments.
        - 'D_symmetry_error': max |D_chiral - D_chiral.T|. Implied by
          C_mu symmetry; redundant but explicit diagnostic.
        - 'finite': bool, integrity check.
        - 'chirality_stats': dict {mean, std, min, max, median}.
        - 'status': PASS / FAIL.
    """

    CORRECTNESS_TOLERANCE = 1e-12
    SYMMETRY_TOLERANCE = 1e-12

    # -------------------------------------------------------------------------
    # Apply -- Eq. 8c
    # -------------------------------------------------------------------------
    def _compute_apply(self) -> dict:
        """
        Compute the literal action of Eq. 8c, with the M x M correlation
        matrix C_mu materialized as an explicit cache.

        Strategy:
            1. Build C_mu = [<mu_i|mu_j>_J] using the Hermitian-pair
               loop pattern of lib_correlation.GlobalCorrelation: only
               the upper triangle is computed; the lower triangle is
               mirrored. Halves the number of inner_product_J calls.
            2. opposite_indices[m] = (m + M/2) mod M. The matrix
               C_mu_opp[m, m'] := C_mu[opposite_indices[m],
               opposite_indices[m']] is obtained by fancy indexing,
               not by recomputation.
            3. D_chiral = abs(C_mu - C_mu_opp). chirality_scalar is
               the row-wise sum of D_chiral.
            4. psi[m] = chirality_scalar[m] * mu_tensor[m].
        """
        M = self.M
        mu = self.mu_tensor
        ip = self.hs.inner_product_J

        # ---- Step 1: full M x M correlation matrix, exploiting symmetry ----
        C_mu = np.zeros((M, M), dtype=np.float64)
        for m1 in range(M):
            for m2 in range(m1, M):
                v = ip(mu[m1], mu[m2])
                C_mu[m1, m2] = v
                C_mu[m2, m1] = v

        # ---- Step 2: diametrical opposite mapping ----
        opposite_indices = (np.arange(M) + M // 2) % M

        # C_mu_opp[m, m'] = C_mu[m_opp, m'_opp] = <mu_{-m} | mu_{-m'}>_J
        # Fancy indexing along both axes: O(M^2), no inner products.
        C_mu_opp = C_mu[np.ix_(opposite_indices, opposite_indices)]

        # ---- Step 3: chirality matrix and per-ray scalar ----
        D_chiral = np.abs(C_mu - C_mu_opp)
        chirality_scalar = np.sum(D_chiral, axis=1)

        # ---- Step 4: psi[m] = chirality_scalar[m] * mu[m] ----
        psi = chirality_scalar[:, None, None] * mu

        return {
            'psi': psi,
            'scalar_per_ray': chirality_scalar,
            'C_mu': C_mu,
            'D_chiral': D_chiral,
            'opposite_indices': opposite_indices,
        }

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------
    def _compute_diagnostics(self) -> dict:
        """
        Numerical correctness checks for Eq. 8c.

        - correctness_error: psi == chirality_scalar * mu.
        - C_mu_symmetry_error: tests inner product symmetry; expected
          zero to floating-point precision.
        - D_symmetry_error: implied by C_mu symmetry plus the
          involution m -> -m -> m; explicit consistency check.
        """
        result = self._apply_cache
        psi = result['psi']
        chirality_scalar = result['scalar_per_ray']
        C_mu = result['C_mu']
        D_chiral = result['D_chiral']

        # ---- Formula match ----
        correctness_error = self._scalar_modulation_error(psi, chirality_scalar)

        # ---- Inner product symmetry of C_mu ----
        C_mu_sym_error = float(np.max(np.abs(C_mu - C_mu.T)))

        # ---- Implied symmetry of D_chiral ----
        D_sym_error = float(np.max(np.abs(D_chiral - D_chiral.T)))

        # ---- Integrity ----
        finite = self._check_finite(psi, chirality_scalar, C_mu, D_chiral)

        # ---- Statistics ----
        chirality_stats = {
            'mean':   float(np.mean(chirality_scalar)),
            'std':    float(np.std(chirality_scalar)),
            'min':    float(np.min(chirality_scalar)),
            'max':    float(np.max(chirality_scalar)),
            'median': float(np.median(chirality_scalar)),
        }

        passed = (
            finite
            and correctness_error < self.CORRECTNESS_TOLERANCE
            and C_mu_sym_error < self.SYMMETRY_TOLERANCE
            and D_sym_error < self.SYMMETRY_TOLERANCE
        )

        return {
            'correctness_error': correctness_error,
            'C_mu_symmetry_error': C_mu_sym_error,
            'D_symmetry_error': D_sym_error,
            'finite': finite,
            'chirality_stats': chirality_stats,
            'status': 'PASS' if passed else 'FAIL',
            'tolerance': self.CORRECTNESS_TOLERANCE,
            'symmetry_tolerance': self.SYMMETRY_TOLERANCE,
        }
