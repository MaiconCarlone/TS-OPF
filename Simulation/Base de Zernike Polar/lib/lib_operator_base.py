# =============================================================================
# lib_operator_base.py — Abstract Contract for Phase I Exacerbation Operators
# =============================================================================
# Defines the abstract base class OperatorBase, which standardizes the
# interface of the five physical exacerbation operators (Eqs. 8a-8e of
# README.md) acting on the micro-space tensor |mu_m> in R^{N' x Q}.
#
# Architectural rationale:
#   - Single Responsibility: each concrete operator subclass implements
#     exactly one equation (8a, 8b, 8c, 8d, 8e). The base class orchestrates
#     the lifecycle (construct -> apply -> diagnostics) without performing
#     any operator-specific computation.
#   - Constructor cache: all read-only inputs (mu_tensor, r_k_array, the
#     HilbertSpace instance) are stored once at construction; subclasses
#     may cache additional derived quantities (e.g., the M x M correlation
#     matrix C_mu in OperatorChirality) without recomputing them across
#     repeated apply() calls.
#   - Uniform return contract: apply() returns a dict with at minimum the
#     keys 'psi' (the operator action on the full tensor, shape
#     (M, N', Q)) and 'scalar_per_ray' (the M-dimensional scalar that
#     modulates each ray, where applicable). Subclasses extend the
#     dictionary with operator-specific auxiliary tensors (e.g., 'C_mu'
#     and 'D_chiral' for chirality, 'diff_norm_map' for radial divergence,
#     'spectral_energy' for the reciprocal operator).
#   - Diagnostics separation: numerical correctness checks (formula match,
#     symmetry of inner products, Parseval, NaN/Inf integrity) are
#     produced by diagnostics() and are independent of visualization or
#     reporting concerns.
#
# Dependencies: numpy, abc
# Reference: README.md (v7.0) -- Section 4 (Eqs. 8a-8e); Rules.md
#
# Usage:
#   from lib_operator_base import OperatorBase
#
#   class OperatorO(OperatorBase):
#       def apply(self):
#           # implement Eq. 8a
#           ...
#       def diagnostics(self):
#           # implement correctness checks specific to O
#           ...
#
#   op = OperatorO(mu_tensor, r_k_array, hilbert_space)
#   result = op.apply()              # cached on first call
#   diag   = op.diagnostics()        # cached on first call
#
# Author: Carlone M. (maicon.carlone@gmail.com)
# =============================================================================

from abc import ABC, abstractmethod

import numpy as np


class OperatorBase(ABC):
    """
    Abstract base class for the five Phase I exacerbation operators.

    Subclasses must implement:
        - apply(): compute the operator action on the cached mu_tensor and
                   return a dict whose canonical keys are 'psi' and
                   'scalar_per_ray'. Operator-specific auxiliary fields are
                   permitted as additional keys.
        - diagnostics(): compute numerical correctness checks (formula
                         match against the canonical equation, symmetry
                         tests where applicable, integrity of NaN/Inf) and
                         return a dict whose canonical keys are
                         'correctness_error' (float), 'status' ('PASS' or
                         'FAIL'), and operator-specific auxiliary fields.

    Constructor cache:
        - self.mu_tensor:       (M, N', Q) micro-space tensor (read-only)
        - self.r_k_array:       (N',) Jacobian radial weights (read-only)
        - self.hs:              HilbertSpace instance providing
                                inner_product_J under the Jacobian metric
        - self.M, self.N_prime, self.Q: dimensions extracted from mu_tensor
        - self._apply_cache:    apply() result, lazily populated
        - self._diag_cache:     diagnostics() result, lazily populated

    Idempotence guarantee:
        Calling apply() or diagnostics() multiple times returns the same
        cached object. The base class handles caching; subclasses
        implement the protected hooks _compute_apply() and
        _compute_diagnostics().
    """

    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------
    def __init__(self, mu_tensor: np.ndarray, r_k_array: np.ndarray, hilbert_space):
        """
        Cache all read-only inputs and pre-validate dimensional consistency.

        Parameters
        ----------
        mu_tensor : ndarray of shape (M, N', Q)
            Micro-space tensor produced by the orthogonal projection of
            Eq. 7. Must be float-valued; NaN/Inf are detected here and
            raise ValueError.
        r_k_array : ndarray of shape (N',)
            Central radii of the radial bands, used by the Jacobian
            inner product (Eq. 2).
        hilbert_space : HilbertSpace
            Pre-instantiated HilbertSpace object providing
            inner_product_J(V1, V2) under the Jacobian metric. The
            internal V_tensor of hilbert_space is irrelevant for the
            operator pipeline; only the inner product method is consumed.

        Raises
        ------
        ValueError
            If mu_tensor contains NaN or Inf, if its rank is not 3, or
            if the radial dimension of r_k_array does not match the
            second axis of mu_tensor.
        """
        # ---- Read-only inputs ----
        self.mu_tensor = mu_tensor
        self.r_k_array = r_k_array
        self.hs = hilbert_space

        # ---- Dimensional cache ----
        if mu_tensor.ndim != 3:
            raise ValueError(
                f"[OperatorBase] mu_tensor must have rank 3 (M, N', Q); "
                f"got shape {mu_tensor.shape}."
            )
        self.M, self.N_prime, self.Q = mu_tensor.shape

        if r_k_array.shape != (self.N_prime,):
            raise ValueError(
                f"[OperatorBase] r_k_array shape {r_k_array.shape} does not "
                f"match radial dimension N'={self.N_prime} of mu_tensor."
            )

        # ---- Integrity check ----
        if np.any(np.isnan(mu_tensor)) or np.any(np.isinf(mu_tensor)):
            raise ValueError(
                "[OperatorBase] mu_tensor contains NaN or Inf entries; "
                "refusing to construct operator on corrupted input."
            )

        # ---- Lazy result caches ----
        self._apply_cache = None
        self._diag_cache = None

    # -------------------------------------------------------------------------
    # Public API (idempotent, cached)
    # -------------------------------------------------------------------------
    def apply(self) -> dict:
        """
        Return the operator action on the cached mu_tensor.

        On the first call, _compute_apply() is invoked and its result
        is cached. Subsequent calls return the same dict object.

        Returns
        -------
        dict
            Must contain at least:
                - 'psi': ndarray of shape (M, N', Q), the operator action
                  on the full tensor. The literal expression of the
                  canonical equation must be reproduced to machine
                  precision (see diagnostics()).
                - 'scalar_per_ray': ndarray of shape (M,), the per-ray
                  scalar that modulates the state. For operators whose
                  output is not a simple scalar rescaling (e.g.,
                  OperatorReciprocal, which returns a transformed tensor
                  of the same shape), this key may be None or absent and
                  must be documented by the subclass.
            Additional keys are permitted and documented per subclass.
        """
        if self._apply_cache is None:
            self._apply_cache = self._compute_apply()
        return self._apply_cache

    def diagnostics(self) -> dict:
        """
        Return numerical correctness diagnostics for the operator.

        On the first call, _compute_diagnostics() is invoked. apply()
        must have been called at least once, or will be invoked
        internally to produce the result tensor against which the
        diagnostics are computed.

        Returns
        -------
        dict
            Must contain at least:
                - 'correctness_error': float, the maximum absolute
                  deviation between the operator output and the literal
                  expression of the canonical equation. Subclasses
                  define the exact metric (typically
                  max_m ||psi[m] - scalar[m] * mu[m]||_inf or, for the
                  reciprocal operator, the Parseval residual).
                - 'status': str, 'PASS' or 'FAIL', based on whether
                  correctness_error < tolerance and integrity checks
                  succeed.
            Additional keys document subclass-specific tests.
        """
        if self._diag_cache is None:
            if self._apply_cache is None:
                self.apply()
            self._diag_cache = self._compute_diagnostics()
        return self._diag_cache

    # -------------------------------------------------------------------------
    # Subclass hooks (must be implemented)
    # -------------------------------------------------------------------------
    @abstractmethod
    def _compute_apply(self) -> dict:
        """
        Concrete implementation of the canonical equation for this
        operator. Must populate at minimum {'psi', 'scalar_per_ray'}.
        """
        raise NotImplementedError

    @abstractmethod
    def _compute_diagnostics(self) -> dict:
        """
        Concrete numerical correctness checks for this operator.
        Must populate at minimum {'correctness_error', 'status'}.
        """
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Shared helpers (available to all subclasses)
    # -------------------------------------------------------------------------
    def _check_finite(self, *arrays) -> bool:
        """
        Verify that all provided arrays are free of NaN/Inf. Returns
        True if all are finite, False otherwise.
        """
        for arr in arrays:
            if arr is None:
                continue
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                return False
        return True

    def _scalar_modulation_error(self, psi: np.ndarray, scalar_per_ray: np.ndarray) -> float:
        """
        Compute the maximum absolute deviation between the operator
        output psi and the literal scalar-modulation prescription
        psi[m] = scalar_per_ray[m] * mu_tensor[m].

        Used by all operators whose output has the form
            psi_m = scalar(m) * mu_m
        i.e., O, S, chi, Dr. The reciprocal operator R, which produces
        a Fourier-domain tensor and not a scalar rescaling, does not
        consume this helper; it implements its own Parseval-based
        correctness metric.

        Returns
        -------
        float
            max_m ||psi[m] - scalar_per_ray[m] * mu_tensor[m]||_inf
        """
        expected = scalar_per_ray[:, None, None] * self.mu_tensor
        return float(np.max(np.abs(psi - expected)))
