# =============================================================================
# lib_physical_operators.py — Aggregator for the Five Phase I Operators
# =============================================================================
# Production library exposing the five physical exacerbation operators
# (Eqs. 8a-8e of README.md, v7.0) as a unified entry point. Each operator
# is delegated to its dedicated class:
#     Eq. 8a  -> lib_operator_O.OperatorIntensity
#     Eq. 8b  -> lib_operator_S.OperatorSymmetry
#     Eq. 8c  -> lib_operator_chi.OperatorChirality
#     Eq. 8d  -> lib_operator_Dr.OperatorRadialDivergence
#     Eq. 8e  -> lib_operator_R.OperatorReciprocal
#
# Architectural rationale:
#   - Single source of truth for every operator: the canonical
#     equation lives exclusively in the corresponding lib_operator_*.py
#     class. This file orchestrates and aggregates results; it does
#     not duplicate the mathematical implementation.
#   - Backward-compatible interface: the constructor signature and the
#     return contract of apply_operators() are preserved bit-for-bit
#     to avoid breaking downstream consumers (Phase II will instantiate
#     this aggregator without modification).
#   - Cache: each delegated class internally caches its apply() result;
#     repeated apply_operators() calls do not recompute.
#
# Dependencies: numpy, logging, lib_hilbert.HilbertSpace,
#               lib_operator_O, lib_operator_S, lib_operator_chi,
#               lib_operator_Dr, lib_operator_R
# Reference: README.md (v7.0) -- Section 4 (Eqs. 8a-8e)
#
# Usage:
#   from lib_hilbert import HilbertSpace
#   from lib_physical_operators import PhysicalOperators
#
#   hs = HilbertSpace(mu_tensor, r_k_array)
#   ops = PhysicalOperators(mu_tensor, r_k_array, hs.inner_product_J)
#   result = ops.apply_operators()
#   # result == {'O': psi_O, 'S': psi_S, 'chi': psi_chi,
#   #           'Dr': psi_Dr, 'R': psi_R}
#
# Author: Carlone M. (maicon.carlone@gmail.com)
# =============================================================================

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import logging

import numpy as np

from lib_hilbert import HilbertSpace
from lib_operator_O import OperatorIntensity
from lib_operator_S import OperatorSymmetry
from lib_operator_chi import OperatorChirality
from lib_operator_Dr import OperatorRadialDivergence
from lib_operator_R import OperatorReciprocal


class PhysicalOperators:
    """
    Aggregator orchestrating the five Phase I exacerbation operators.

    Backward-compatible facade: the public surface is preserved
    relative to the previous monolithic implementation, so any
    consumer (current or future) that imports
        from lib_physical_operators import PhysicalOperators
    and instantiates with the same signature will continue to work.
    The internal dispatch now delegates to the dedicated lib_operator_*
    classes, which are the single source of truth for the canonical
    equations.

    Constructor cache (in addition to delegated subclass caches):
        - self.mu_tensor:  read-only reference, shape (M, N', Q)
        - self.r_k_array:  read-only reference, shape (N',)
        - self.inner_product_J: callable, kept for backward
                                compatibility. Internally a private
                                HilbertSpace instance is constructed
                                so the new operator classes can
                                consume their canonical hs argument.
        - self.M, self.N_prime, self.Q: dimensions
        - self._operators: dict {name: OperatorBase instance}, lazily
                           constructed on first apply_operators() call.
        - self._results:   dict {name: psi tensor}, cached.

    Note on inner_product_J:
        The legacy signature accepts a callable inner_product_J_func
        because the original code did not depend on the HilbertSpace
        class; only the inner product method was needed. To preserve
        this contract, we instantiate a HilbertSpace internally only
        for the purpose of providing it to the new operator classes,
        which expect an hs object. The provided inner_product_J_func
        is retained as a public attribute (self.inner_product_J) for
        any consumer that still relies on it.
    """

    def __init__(self, mu_tensor, r_k_array, inner_product_J_func):
        """
        Cache inputs and prepare for lazy operator dispatch.

        Parameters
        ----------
        mu_tensor : ndarray of shape (M, N', Q)
            Micro-space tensor (Eq. 7).
        r_k_array : ndarray of shape (N',)
            Jacobian radial weights.
        inner_product_J_func : callable
            Function (V_m1, V_m2) -> scalar implementing the Jacobian
            inner product. Retained for backward compatibility; the
            internal HilbertSpace instance reproduces the same
            inner product, ensuring numerical equivalence.
        """
        # ---- Read-only references ----
        self.mu_tensor = mu_tensor
        self.r_k_array = r_k_array
        self.inner_product_J = inner_product_J_func

        # ---- Dimensional cache ----
        self.M, self.N_prime, self.Q = self.mu_tensor.shape

        # ---- HilbertSpace for the new operator classes ----
        # The new classes accept an hs object and use only its
        # inner_product_J method. This instance is constructed on the
        # same (mu_tensor, r_k_array) pair, so its inner product is
        # numerically identical to the function injected by the caller
        # (both follow Eq. 2 of README.md verbatim).
        self._hs = HilbertSpace(self.mu_tensor, self.r_k_array)

        # ---- Lazy operator instances ----
        self._operators = None

        # ---- Lazy result cache (preserves contract dict shape) ----
        self._results = None

    # -------------------------------------------------------------------------
    # Internal: lazy construction of operator classes
    # -------------------------------------------------------------------------
    def _build_operators(self) -> None:
        """
        Instantiate the five operator classes once, sharing the same
        HilbertSpace cache. Subsequent apply_operators() calls will
        reuse the same instances and trigger their internal apply()
        cache.
        """
        if self._operators is not None:
            return
        self._operators = {
            'O':   OperatorIntensity(self.mu_tensor, self.r_k_array, self._hs),
            'S':   OperatorSymmetry(self.mu_tensor, self.r_k_array, self._hs),
            'chi': OperatorChirality(self.mu_tensor, self.r_k_array, self._hs),
            'Dr':  OperatorRadialDivergence(self.mu_tensor, self.r_k_array, self._hs),
            'R':   OperatorReciprocal(self.mu_tensor, self.r_k_array, self._hs),
        }

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def apply_operators(self) -> dict:
        """
        Apply the five operators and return their psi tensors as a dict.

        Returns
        -------
        dict
            {'O':   ndarray (M, N', Q),
             'S':   ndarray (M, N', Q),
             'chi': ndarray (M, N', Q),
             'Dr':  ndarray (M, N', Q),
             'R':   ndarray (M, N', Q)}

            The shape and content of each entry are identical (to
            machine precision) to the output of the previous monolithic
            implementation. Verified by parity testing during the
            Block O refactoring.
        """
        if self._results is not None:
            return self._results

        self._build_operators()

        logging.info(
            "[PhysicalOperators] Exacerbating signal through the 5 physical "
            "lenses (delegated to lib_operator_*.py)..."
        )

        results = {}
        for name in ('O', 'S', 'chi', 'Dr', 'R'):
            logging.info(
                f"[PhysicalOperators] -> Operator {name} "
                f"(class {type(self._operators[name]).__name__})"
            )
            results[name] = self._operators[name].apply()['psi']

        logging.info(
            "[PhysicalOperators] Perturbed tensors built and isolated."
        )
        self._results = results
        return results

    # -------------------------------------------------------------------------
    # Extended API (new -- non-breaking addition)
    # -------------------------------------------------------------------------
    def get_operator(self, name: str):
        """
        Return the operator instance for direct access to apply()
        (full result dict) and diagnostics().

        Parameters
        ----------
        name : str
            One of 'O', 'S', 'chi', 'Dr', 'R'.

        Returns
        -------
        OperatorBase
            The cached subclass instance.

        Raises
        ------
        KeyError
            If name is not a registered operator.
        """
        self._build_operators()
        if name not in self._operators:
            raise KeyError(
                f"[PhysicalOperators] Unknown operator '{name}'. "
                f"Valid names: {sorted(self._operators)}."
            )
        return self._operators[name]

    def all_diagnostics(self) -> dict:
        """
        Return diagnostics() for every registered operator.

        Returns
        -------
        dict
            {'O': diag_O, 'S': diag_S, 'chi': diag_chi,
             'Dr': diag_Dr, 'R': diag_R}
            Each entry is the diagnostics dict produced by the
            corresponding subclass (correctness_error, status, etc.).
        """
        self._build_operators()
        # Ensure apply() has been called for each operator so that
        # diagnostics() has a result to inspect.
        self.apply_operators()
        return {name: op.diagnostics() for name, op in self._operators.items()}
