# =============================================================================
# lib_operator_R.py — Reciprocal Operator (Eq. 8e)
# =============================================================================
# Implements the fifth of the five Phase I exacerbation operators
# (Eq. 8e of README.md, v7.0):
#
#     tilde{mu}_{k,m} = || (1/sqrt(N')) sum_{kappa} exp(-i 2*pi/N'
#                          * (k-1)(kappa-1)) mu_{kappa,m} ||_C
#
# Transforms each angular sector m to the radial frequency domain via
# the discrete Fourier transform along the radial axis, applied
# independently to every Zernike mode q. The complex magnitude
# preserves the real tensorial structure (R^{N' x Q}). Unlike the
# other four operators, the reciprocal operator does NOT produce a
# scalar rescaling: its action replaces the radial axis by a
# frequency axis, returning a tensor of the same shape but in a
# distinct domain. Consequently 'scalar_per_ray' is None for this
# operator; the correctness diagnostic is Parseval's theorem instead.
#
# Dependencies: numpy, scipy.fftpack, lib_operator_base
# Reference: README.md (v7.0) -- Section 4, Eq. 8e; OpInZernikeBasis.md,
# Section 3 (Operator 5: Reciprocal)
#
# Usage:
#   from lib_hilbert import HilbertSpace
#   from lib_operator_R import OperatorReciprocal
#
#   hs = HilbertSpace(mu_tensor, r_k_array)
#   op = OperatorReciprocal(mu_tensor, r_k_array, hs)
#   res  = op.apply()        # {'psi', 'spectral_energy', 'energy_concentration', ...}
#   diag = op.diagnostics()  # {'parseval_error', 'status', 'energy_stats'}
#
# Author: Carlone M. (maicon.carlone@gmail.com)
# =============================================================================

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import numpy as np
import scipy.fftpack

from lib_operator_base import OperatorBase


class OperatorReciprocal(OperatorBase):
    """
    Concrete implementation of the Reciprocal operator hat{R} (Eq. 8e).

    apply() returns:
        - 'psi': ndarray (M, N', Q),
                     psi_R[m, k, q] = | FFT(mu[m, :, q])[k] | / sqrt(N')
                  The Fourier-domain magnitude tensor. Note that the
                  axis of the second dimension represents frequency
                  harmonics (not spatial radii); index k=0 is DC,
                  k=1..N'/2-1 are positive frequencies, the upper half
                  carries the negative-frequency mirror.
        - 'scalar_per_ray': None
                  R does not produce a scalar rescaling. Set explicitly
                  to None to make the contract divergence with O/S/chi/Dr
                  visible to consumers.
        - 'spectral_energy': ndarray (M, Q),
                     spectral_energy[m, q] = sum_k psi_R[m, k, q]^2.
                  Total Fourier energy per ray per Zernike mode.
        - 'energy_concentration': ndarray (M, Q), the fraction of
                  energy contained in the first K_FREQ low-frequency
                  bins, divided by the total energy of the same
                  (m, q) channel:
                     concentration[m, q]
                       = sum_{k<K_FREQ} psi_R[m,k,q]^2
                         / sum_k psi_R[m,k,q]^2.
                  A high concentration means the radial profile of
                  mode q at ray m is dominated by low-frequency
                  (smooth) variation; a low concentration indicates
                  significant high-frequency content.
        - 'zero_energy_rays_per_mode': ndarray (Q,), int. Counts the
                  rays for which the q-th channel has identically zero
                  total energy (background sectors with null micro-
                  space). These rays are excluded from the
                  energy_concentration check in diagnostics().
        - 'K_freq': int, the cutoff index used in the energy
                  concentration computation.

    diagnostics() returns:
        - 'parseval_error': float, the maximum absolute deviation
          across (m, q) between the spatial-domain energy
          sum_k mu[m,k,q]^2 and the frequency-domain energy
          sum_k psi_R[m,k,q]^2. With the explicit 1/sqrt(N')
          normalization applied to the unnormalized scipy.fftpack.fft,
          Parseval is satisfied to within accumulated rounding from the
          summation across N' bands. The tolerance is set at 1e-10
          to absorb this rounding (well above the per-element machine
          epsilon of ~2e-16).
        - 'finite': bool.
        - 'energy_stats': dict, per-mode statistics of
          energy_concentration: {q: {mean, std}}.
        - 'status': PASS / FAIL.
    """

    PARSEVAL_TOLERANCE = 1e-10

    # K_FREQ is the cutoff for the low-frequency energy concentration
    # diagnostic. Set to min(5, N'/10): sufficiently small to capture
    # only the truly low-frequency content for typical N' >> 50, while
    # gracefully degrading for small N'.
    K_FREQ_RULE = lambda self, N: min(5, N // 10) if N >= 10 else max(1, N // 2)

    # -------------------------------------------------------------------------
    # Apply -- Eq. 8e
    # -------------------------------------------------------------------------
    def _compute_apply(self) -> dict:
        """
        Compute the literal action of Eq. 8e.

        Strategy:
            1. For each ray m, apply scipy.fftpack.fft along the radial
               axis (axis=0 in the per-ray slice mu[m] of shape
               (N', Q)). The resulting array is complex, shape (N', Q),
               with the FFT applied independently per Zernike column.
            2. Take the complex modulus and divide by sqrt(N') to
               restore Parseval's theorem (scipy.fftpack.fft is
               unnormalized).
            3. Collect spectral energy per (m, q) and the low-frequency
               concentration ratio.

        Note: the loop over m is preserved (instead of vectorized
        FFT(mu, axis=1)) for explicit symmetry with the production
        library lib_physical_operators.PhysicalOperators, ensuring
        bit-for-bit numerical equivalence on the same input. The
        per-ray FFT is internally vectorized across the Q-axis by
        scipy and is not the bottleneck.
        """
        M = self.M
        N = self.N_prime
        Q = self.Q
        mu = self.mu_tensor
        sqrtN = np.sqrt(N)

        # ---- Step 1+2: FFT and Parseval normalization ----
        psi = np.zeros_like(mu)
        for m in range(M):
            fft_complex = scipy.fftpack.fft(mu[m], axis=0)  # (N, Q), complex
            psi[m] = np.abs(fft_complex) / sqrtN

        # ---- Step 3: spectral energy and low-frequency concentration ----
        # Total energy per (m, q): sum over the frequency axis (k).
        spectral_energy = np.sum(psi ** 2, axis=1)  # (M, Q)

        # Low-frequency cutoff (per current N')
        K_freq = self.K_FREQ_RULE(N)

        # Energy concentration in the first K_freq bins
        low_freq_energy = np.sum(psi[:, :K_freq, :] ** 2, axis=1)  # (M, Q)

        # Safe division: zero-energy rays (background) yield 0 concentration.
        # This avoids the warning and matches the convention of the
        # legacy implementation in Study_Operators.py.
        energy_concentration = np.zeros((M, Q), dtype=np.float64)
        nonzero = spectral_energy > 0.0
        energy_concentration[nonzero] = (
            low_freq_energy[nonzero] / spectral_energy[nonzero]
        )

        zero_energy_rays_per_mode = np.zeros(Q, dtype=np.int64)
        for q in range(Q):
            zero_energy_rays_per_mode[q] = int(np.sum(spectral_energy[:, q] <= 0.0))

        return {
            'psi': psi,
            'scalar_per_ray': None,           # contract divergence: R has no scalar
            'spectral_energy': spectral_energy,
            'energy_concentration': energy_concentration,
            'zero_energy_rays_per_mode': zero_energy_rays_per_mode,
            'K_freq': K_freq,
        }

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------
    def _compute_diagnostics(self) -> dict:
        """
        Numerical correctness checks for Eq. 8e.

        Parseval's theorem under the 1/sqrt(N) normalization:
            sum_k mu[m,k,q]^2 = sum_k psi_R[m,k,q]^2
        across all (m, q). This is the operator's correctness metric
        in the absence of a scalar rescaling.
        """
        result = self._apply_cache
        psi = result['psi']

        mu = self.mu_tensor

        # Per-(m, q) energy in spatial and frequency domains
        energy_spatial = np.sum(mu ** 2, axis=1)  # (M, Q)
        energy_freq = np.sum(psi ** 2, axis=1)    # (M, Q)
        parseval_error = float(np.max(np.abs(energy_spatial - energy_freq)))

        finite = self._check_finite(psi)

        energy_concentration = result['energy_concentration']
        energy_stats = {}
        for q in range(self.Q):
            energy_stats[q] = {
                'mean': float(np.mean(energy_concentration[:, q])),
                'std':  float(np.std(energy_concentration[:, q])),
            }

        passed = finite and parseval_error < self.PARSEVAL_TOLERANCE

        return {
            'parseval_error': parseval_error,
            'correctness_error': parseval_error,  # alias for uniform reporting
            'finite': finite,
            'energy_stats': energy_stats,
            'zero_energy_rays_per_mode': result['zero_energy_rays_per_mode'].tolist(),
            'K_freq': result['K_freq'],
            'status': 'PASS' if passed else 'FAIL',
            'tolerance': self.PARSEVAL_TOLERANCE,
        }
