# =============================================================================
# Study_Operators.py — Phase I Exacerbation Operator Validation Harness
# =============================================================================
# Validation orchestrator for the five physical exacerbation operators
# (Eqs. 8a-8e of README.md, v7.0). This module is purely a *visual and
# diagnostic harness*: all numerical work is delegated to the canonical
# operator classes in lib_operator_*.py, which constitute the single
# source of truth for the equations.
#
# Architecture:
#   - Loads the cached pipeline state from Transposition_State.npz
#     (produced by Study_Transposicao.py).
#   - Instantiates the five operator classes (OperatorIntensity,
#     OperatorSymmetry, OperatorChirality, OperatorRadialDivergence,
#     OperatorReciprocal) on the cached mu_tensor.
#   - For each operator, calls apply() and diagnostics(), then renders
#     a 1x3 PDF figure with operator-specific panels and logs the
#     diagnostic outcome.
#
# Reference direction: m=45, theta=45 deg, consistent with
# Study_Transposicao.py (selected_angle_deg=45.0).
#
# Reviewer feedback incorporated (parecer.md, ack iteration):
#   - Operator S panel: reports both raw cohesion (Eq. 8b) and the
#     diagnostic Jacobian-normalized cosine.
#   - chi heatmap: P99 color saturation + diagonal-mean annotation.
#   - Dr: logarithmic overlay heatmap added.
#   - R: diagnostic Tilt-Y panel (mu_2 vs mu_3 radial profile and
#         spectrum) added to investigate the anomalously low Tilt-Y
#         spectral energy reported in earlier validation runs.
#   - Figures use gridspec from the start, no set_visible(False).
#   - Angular axes in degrees (chi, Dr); frequency axis in cycles/pixel
#     (R).
#   - LaTeX-rendered statistical annotations.
#   - Optional population-data hook for future cross-cohort overlays.
#
# Anatomical reference panel: omitted per acknowledged reviewer
# decision -- the anatomical context is provided once by
# Estudo_Transposicao_Espacial.pdf, not duplicated per operator.
#
# Dependencies: numpy, matplotlib, lib_hilbert,
#               lib_operator_O, lib_operator_S, lib_operator_chi,
#               lib_operator_Dr, lib_operator_R
# Reference: README.md (v7.0) -- Section 4 (Eqs. 8a-8e); IMP.md
#
# Usage:
#   python Study_Operators.py --operator all
#   python Study_Operators.py --operator O
#
# Author: Carlone M. (maicon.carlone@gmail.com)
# =============================================================================

import argparse
import logging
import os
import sys

# ---------------------------------------------------------------------------
# Path resolution: this script lives in study/; libs live in ../lib/
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))   # .../study
_BASE = os.path.dirname(_HERE)                        # .../Base de Zernike Polar
_LIB  = os.path.join(_BASE, 'lib')
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

from lib_hilbert import HilbertSpace
from lib_operator_O import OperatorIntensity
from lib_operator_S import OperatorSymmetry
from lib_operator_chi import OperatorChirality
from lib_operator_Dr import OperatorRadialDivergence
from lib_operator_R import OperatorReciprocal


# =============================================================================
# Visual constants
# =============================================================================
MODE_COLORS = ['#1a1a1a', '#2ca02c', '#ff7f0e', '#9467bd']
MODE_LABELS = [
    r'$Z_0^0$: Piston ($q=1$)',
    r'$Z_1^1$: Tilt-X ($q=2$)',
    r'$Z_1^{-1}$: Tilt-Y ($q=3$)',
    r'$Z_2^0$: Defocus ($q=4$)',
]
BG_COLOR    = '#f8f9fa'
GRID_COLOR  = '#dee2e6'
FONT_TITLE  = {'fontweight': 'bold', 'fontsize': 11}
STATS_BBOX  = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.92,
                   edgecolor=GRID_COLOR)


def _apply_axis_style(ax):
    """Consistent Cartesian axes background and grid."""
    ax.set_facecolor(BG_COLOR)
    ax.grid(True, alpha=0.3, color=GRID_COLOR)
    for spine in ax.spines.values():
        spine.set_color('#cccccc')


def _add_stats_box(ax, lines):
    """
    Place a monospace LaTeX-formatted stats box in the upper-right.

    Parameters
    ----------
    ax : matplotlib axes
    lines : list of str, each rendered LaTeX-friendly via mathtext.
    """
    text = "\n".join(lines)
    ax.text(0.97, 0.97, text, transform=ax.transAxes, va='top', ha='right',
            bbox=STATS_BBOX, fontsize=8, family='monospace')


def _stats_lines(label_value_pairs):
    r"""
    Build LaTeX-formatted stats lines from a list of (latex_label,
    value, fmt) tuples. fmt is a Python format string that does not
    include braces (e.g., '.4f' or '.3e').
    """
    out = []
    for label, value, fmt in label_value_pairs:
        out.append(f"{label} = {format(value, fmt)}")
    return out


# =============================================================================
# OperatorStudy -- harness orchestrator
# =============================================================================
class OperatorStudy:
    """
    Harness coordinating loading of the cached state, instantiation of
    the five operator classes, generation of the diagnostic PDFs, and
    structured logging.

    Cache (constructor):
        self.base_dir, self.npz_path, self.log_file: I/O paths.
        self.mu_tensor, self.r_k_array, self.theta_m_array:
            tensors loaded from Transposition_State.npz.
        self.M, self.N_prime, self.Q: dimensions.
        self.delta_theta, self.r_min, self.R_max: geometric scalars.
        self.delta_r: scalar spacing of the radial bands (used for the
                      cycles-per-pixel axis of operator R).
        self.selected_m: reference angular index (theta = 45 deg).
        self.hs: HilbertSpace instance (shared by all operator classes).
        self._operators: dict {name: OperatorBase}, lazily built.

    Hooks:
        population_data (optional kwarg of run_pipeline): a dict
            {operator_name: {'samples': (N_pop,)}} carrying the
            empirical distribution of the per-ray scalar across a
            healthy population. When provided, overlays are activated
            on the histogram panels. Defaults to None (no overlay).

    Usage:
        study = OperatorStudy()
        study.run_pipeline(operators='all')
    """

    def __init__(self):
        self.base_dir = _BASE
        self.npz_path = os.path.join(_BASE, "Transposition_State.npz")
        self.log_file = os.path.join(_BASE, 'log', "Log_Operators.txt")
        self._setup_logger()

        # Lazy state -- populated by _load_state().
        self.mu_tensor = None
        self.r_k_array = None
        self.theta_m_array = None
        self.M = self.N_prime = self.Q = 0
        self.delta_theta = self.delta_r = self.r_min = self.R_max = 0.0
        self.selected_m = 0
        self.hs = None

        # Lazy operator instances -- built once, reused by all panels.
        self._operators = None

        # Optional population overlay (set in run_pipeline kwargs).
        self.population_data = None

    # -------------------------------------------------------------------------
    # Logger
    # -------------------------------------------------------------------------
    def _setup_logger(self):
        """Dual logging to file (Log_Operators.txt) and console."""
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh = logging.FileHandler(self.log_file, mode='w')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # -------------------------------------------------------------------------
    # State loading and validation
    # -------------------------------------------------------------------------
    def _load_state(self):
        """Load Transposition_State.npz; populate dimensional cache."""
        if not os.path.exists(self.npz_path):
            logging.error(
                f"[Operators] MISSING: {self.npz_path} not found. "
                "Run Study_Transposicao.py first."
            )
            sys.exit(1)

        logging.info(f"[Operators] Loading state from: {self.npz_path}")
        data = np.load(self.npz_path, allow_pickle=True)

        self.mu_tensor     = data['mu_tensor']
        self.r_k_array     = data['r_k_array']
        self.theta_m_array = data['theta_m_array']

        self.M       = int(data['n_angular_sectors'])
        self.N_prime = int(data['n_radial_bands'])
        self.Q       = int(data['n_zernike_modes'])

        self.delta_theta = float(data['delta_theta'])
        self.delta_r     = float(data['delta_r'])
        self.r_min       = float(data['r_min'])
        self.R_max       = float(data['R_max'])

        # Reference angular index closest to theta = 45 deg.
        target_rad = np.radians(45.0) % (2.0 * np.pi)
        self.selected_m = int(np.argmin(np.abs(self.theta_m_array - target_rad)))
        logging.info(
            f"[Operators] Dimensions: M={self.M}, N'={self.N_prime}, Q={self.Q}. "
            f"Reference sector: m={self.selected_m} (theta=45.0 deg)."
        )

    def _validate_input(self):
        """Verify integrity of mu_tensor and instantiate HilbertSpace."""
        logging.info("[Operators] Validating input tensor integrity...")
        if np.any(np.isnan(self.mu_tensor)) or np.any(np.isinf(self.mu_tensor)):
            logging.error("[Operators] NaN/Inf detected in mu_tensor. Aborting.")
            sys.exit(1)
        expected = (self.M, self.N_prime, self.Q)
        if self.mu_tensor.shape != expected:
            logging.error(
                f"[Operators] mu_tensor shape: {self.mu_tensor.shape}, "
                f"expected {expected}. Aborting."
            )
            sys.exit(1)
        self.hs = HilbertSpace(self.mu_tensor, self.r_k_array)
        logging.info("[Operators] Integrity confirmed. HilbertSpace cached.")

    # -------------------------------------------------------------------------
    # Lazy operator factory (single source of truth: lib_operator_*.py)
    # -------------------------------------------------------------------------
    def _build_operators(self):
        """Instantiate the five operator classes once, sharing the same hs."""
        if self._operators is not None:
            return
        self._operators = {
            'O':   OperatorIntensity        (self.mu_tensor, self.r_k_array, self.hs),
            'S':   OperatorSymmetry         (self.mu_tensor, self.r_k_array, self.hs),
            'chi': OperatorChirality        (self.mu_tensor, self.r_k_array, self.hs),
            'Dr':  OperatorRadialDivergence (self.mu_tensor, self.r_k_array, self.hs),
            'R':   OperatorReciprocal       (self.mu_tensor, self.r_k_array, self.hs),
        }

    # -------------------------------------------------------------------------
    # Logger helper for diagnostic dictionaries
    # -------------------------------------------------------------------------
    def _log_diagnostics(self, name: str, diag: dict):
        """Emit a structured INFO log line for every diagnostic key."""
        logging.info("=" * 60)
        logging.info(f"DIAGNOSTICS: Operator {name}")
        logging.info("=" * 60)
        for k, v in diag.items():
            if isinstance(v, dict):
                summary = ", ".join(
                    f"{kk}={vv:.4e}" if isinstance(vv, float) else f"{kk}={vv}"
                    for kk, vv in v.items()
                )
                logging.info(f"  {k}: {summary}")
            elif isinstance(v, float):
                logging.info(f"  {k}: {v:.4e}")
            else:
                logging.info(f"  {k}: {v}")

    # =====================================================================
    # Operator O -- Intensity (Eq. 8a)
    # =====================================================================
    def _render_operator_O(self) -> dict:
        """
        Render the diagnostic figure for Operator O (Eq. 8a). Three
        panels (1x3 GridSpec):
            [0] Zernike packet profile of the reference ray (before
                solid / after dashed), illustrating the uniform scalar
                rescaling.
            [1] Polar plot of |scalar_O| over angular sectors.
            [2] Histogram of scalar_O with mean line and stats box.
        """
        op = self._operators['O']
        result = op.apply()
        diag = op.diagnostics()
        self._log_diagnostics('O', diag)

        scalar_O = result['scalar_per_ray']
        psi_O    = result['psi']
        sel      = self.selected_m
        ss       = diag['scalar_stats'] if 'scalar_stats' in diag else None
        # OperatorIntensity does not include scalar_stats; recompute here.
        ss = {
            'mean':   float(np.mean(scalar_O)),
            'std':    float(np.std(scalar_O)),
            'min':    float(np.min(scalar_O)),
            'max':    float(np.max(scalar_O)),
            'median': float(np.median(scalar_O)),
        }

        fig = plt.figure(figsize=(18, 5.5))
        gs = GridSpec(1, 3, figure=fig, wspace=0.32)
        fig.suptitle(
            r"Operator $\hat{O}$: Intensity (Eq. 8a)  |  "
            rf"$m={sel},\ \theta=45^\circ$",
            fontsize=13, fontweight='bold', y=0.99
        )

        # Panel 1: Zernike packet profile (before / after)
        ax = fig.add_subplot(gs[0])
        for q in range(self.Q):
            ax.plot(self.r_k_array, self.mu_tensor[sel, :, q],
                    color=MODE_COLORS[q], lw=1.5, label=MODE_LABELS[q])
            ax.plot(self.r_k_array, psi_O[sel, :, q],
                    color=MODE_COLORS[q], lw=1.0, ls='--', alpha=0.5)
        ax.set_title("Zernike Packet (solid: before / dashed: after)", **FONT_TITLE)
        ax.set_xlabel(r"Radial position $r_k$ (px)")
        ax.set_ylabel("Coefficient amplitude")
        ax.legend(fontsize=7, ncol=2, loc='best')
        _apply_axis_style(ax)

        # Panel 2: Polar |scalar_O|(theta)
        ax_polar = fig.add_subplot(gs[1], projection='polar')
        ax_polar.plot(self.theta_m_array, np.abs(scalar_O),
                      lw=1.2, color='#d62728')
        ax_polar.fill_between(self.theta_m_array, 0, np.abs(scalar_O),
                              alpha=0.25, color='#d62728')
        ax_polar.set_title(
            r"$\left|\langle\mu_1\rangle(\theta)\right|$ per angular sector",
            pad=15, **FONT_TITLE
        )
        ax_polar.set_theta_zero_location('E')

        # Panel 3: Histogram of scalar_O
        ax = fig.add_subplot(gs[2])
        ax.hist(scalar_O, bins=50, color='#d62728', alpha=0.7,
                edgecolor='white', linewidth=0.4)
        ax.axvline(ss['mean'], color='black', lw=2, ls='--', alpha=0.7)
        # Population overlay hook
        if self.population_data and 'O' in self.population_data:
            pop = self.population_data['O'].get('samples')
            if pop is not None:
                ax.hist(pop, bins=50, color='#1f77b4', alpha=0.25,
                        edgecolor='none', label='Healthy population')
                ax.legend(fontsize=8, loc='upper left')
        _add_stats_box(ax, _stats_lines([
            (r"$\mu$",      ss['mean'],   '.3e'),
            (r"$\sigma$",   ss['std'],    '.3e'),
            ("min",         ss['min'],    '.3e'),
            ("max",         ss['max'],    '.3e'),
            ("median",      ss['median'], '.3e'),
        ]))
        ax.set_title("Mean Piston Residual Distribution", **FONT_TITLE)
        ax.set_xlabel(r"$\frac{1}{N'}\sum_k \mu_1^{(k,m)}$")
        ax.set_ylabel("Count")
        _apply_axis_style(ax)

        out_pdf = os.path.join(self.base_dir, 'figures', "Estudo_Operador_O.pdf")
        fig.savefig(out_pdf, dpi=300, format='pdf', bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Figure exported: {out_pdf}")
        return diag

    # =====================================================================
    # Operator S -- Symmetry (Eq. 8b)
    # =====================================================================
    def _render_operator_S(self) -> dict:
        """
        Render the diagnostic figure for Operator S (Eq. 8b). Three
        panels (1x3 GridSpec):
            [0] Zernike profiles of the reference ray m and its
                opposite -m, plus the cohesion scalar of the pair.
            [1] Polar plot reporting BOTH the raw Jacobian inner
                product (Eq. 8b literal) AND the diagnostic
                Jacobian-normalized cosine, on a shared angular axis
                with twin radial scales.
            [2] Histograms of the cohesion (raw) and the cosine
                (display companion), stacked.
        """
        op = self._operators['S']
        result = op.apply()
        diag = op.diagnostics()
        self._log_diagnostics('S', diag)

        cohesion = result['scalar_per_ray']
        cosine   = result['cohesion_cosine']
        opp      = result['opposite_indices']
        sel      = self.selected_m
        m_opp    = int(opp[sel])
        cs       = diag['cohesion_stats']
        css      = diag['cosine_stats']

        fig = plt.figure(figsize=(18, 5.5))
        gs = GridSpec(1, 3, figure=fig, wspace=0.32)
        fig.suptitle(
            r"Operator $\hat{S}$: Symmetry (Eq. 8b)  |  "
            rf"$m={sel}\ \mathrm{{vs}}\ -m={m_opp}$",
            fontsize=13, fontweight='bold', y=0.99
        )

        # Panel 1: Zernike profiles m vs -m
        ax = fig.add_subplot(gs[0])
        for q in range(self.Q):
            ax.plot(self.r_k_array, self.mu_tensor[sel, :, q],
                    color=MODE_COLORS[q], lw=1.5, label=f"{MODE_LABELS[q]} (m)")
            ax.plot(self.r_k_array, self.mu_tensor[m_opp, :, q],
                    color=MODE_COLORS[q], lw=1.0, ls='--', alpha=0.5)
        _add_stats_box(ax, [
            rf"raw cohesion $= {cohesion[sel]:.4f}$",
            rf"cosine $\widetilde{{S}} = {cosine[sel]:.4f}$",
        ])
        ax.set_title("Zernike: m (solid) vs -m (dashed)", **FONT_TITLE)
        ax.set_xlabel(r"Radial position $r_k$ (px)")
        ax.set_ylabel("Coefficient amplitude")
        ax.legend(fontsize=6, ncol=2, loc='best')
        _apply_axis_style(ax)

        # Panel 2: Polar plot of raw cohesion AND cosine
        # The two quantities live on different scales: cohesion is
        # unbounded, cosine in [-1, +1]. We use a single polar axes
        # with the raw cohesion as the primary curve (filled), and
        # overlay the cosine as a dashed line scaled by the maximum
        # absolute cohesion to share the radial axis. The legend
        # documents the scaling.
        ax_polar = fig.add_subplot(gs[1], projection='polar')
        max_abs_coh = max(float(np.max(np.abs(cohesion))), 1e-30)
        cosine_scaled = cosine * max_abs_coh
        ax_polar.fill_between(self.theta_m_array, 0, cohesion,
                              alpha=0.35, color='#2ca02c',
                              label=r'raw $\langle\mu_m|\mu_{-m}\rangle_J$ (Eq. 8b)')
        ax_polar.plot(self.theta_m_array, cohesion, lw=1.2, color='darkgreen')
        ax_polar.plot(self.theta_m_array, cosine_scaled, lw=1.0, ls='--',
                      color='#1f77b4',
                      label=rf'cosine $\widetilde{{S}}\,\times\,{max_abs_coh:.2f}$ (display)')
        ax_polar.plot(self.theta_m_array[sel], cohesion[sel], 'o',
                      color='red', markersize=6, label='reference m')
        ax_polar.plot(self.theta_m_array[m_opp], cohesion[m_opp], 'o',
                      color='blue', markersize=6, label='opposite -m')
        ax_polar.set_title(
            r"Hemispheric coherence per angular sector",
            pad=15, **FONT_TITLE
        )
        ax_polar.legend(fontsize=6, loc='upper right',
                        bbox_to_anchor=(1.30, 1.10))

        # Panel 3: Stacked histograms (raw cohesion top, cosine bottom)
        gs_inner = gs[2].subgridspec(2, 1, hspace=0.55)
        ax_top = fig.add_subplot(gs_inner[0])
        ax_top.hist(cohesion, bins=50, color='darkgreen', alpha=0.7,
                    edgecolor='white', linewidth=0.4)
        ax_top.axvline(cs['mean'], color='black', lw=2, ls='--', alpha=0.7)
        if self.population_data and 'S' in self.population_data:
            pop = self.population_data['S'].get('cohesion_samples')
            if pop is not None:
                ax_top.hist(pop, bins=50, color='#1f77b4', alpha=0.25,
                            edgecolor='none', label='Healthy population')
                ax_top.legend(fontsize=7, loc='upper left')
        _add_stats_box(ax_top, _stats_lines([
            (r"$\mu$",    cs['mean'], '.4f'),
            (r"$\sigma$", cs['std'],  '.4f'),
            ("min",       cs['min'],  '.4f'),
            ("max",       cs['max'],  '.4f'),
        ]))
        ax_top.set_title(r"Raw cohesion $\langle\mu_m|\mu_{-m}\rangle_J$",
                         fontsize=10, fontweight='bold')
        ax_top.set_xlabel("value")
        ax_top.set_ylabel("count")
        _apply_axis_style(ax_top)

        ax_bot = fig.add_subplot(gs_inner[1])
        ax_bot.hist(cosine, bins=50, color='#1f77b4', alpha=0.7,
                    edgecolor='white', linewidth=0.4)
        ax_bot.axvline(css['mean'], color='black', lw=2, ls='--', alpha=0.7)
        ax_bot.axvline(-1.0, color='red', lw=0.8, ls=':', alpha=0.6)
        ax_bot.axvline(+1.0, color='red', lw=0.8, ls=':', alpha=0.6)
        _add_stats_box(ax_bot, _stats_lines([
            (r"$\mu$",    css['mean'], '.4f'),
            (r"$\sigma$", css['std'],  '.4f'),
            ("min",       css['min'],  '.4f'),
            ("max",       css['max'],  '.4f'),
        ]))
        ax_bot.set_title(
            r"Diagnostic cosine $\widetilde{S}(m)$ (display companion)",
            fontsize=10, fontweight='bold'
        )
        ax_bot.set_xlabel(r"$\widetilde{S}$")
        ax_bot.set_ylabel("count")
        _apply_axis_style(ax_bot)

        out_pdf = os.path.join(self.base_dir, 'figures', "Estudo_Operador_S.pdf")
        fig.savefig(out_pdf, dpi=300, format='pdf', bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Figure exported: {out_pdf}")
        return diag

    # =====================================================================
    # Operator chi -- Chirality (Eq. 8c)
    # =====================================================================
    def _render_operator_chi(self) -> dict:
        """
        Render the diagnostic figure for Operator chi (Eq. 8c). Three
        panels (1x3 GridSpec):
            [0] Heatmap of C_mu (Jacobian correlation matrix of the
                micro-space), angular axes in degrees.
            [1] Heatmap of D_chiral with color saturation at the 99th
                percentile and an annotation reporting the diagonal
                mean as a non-unit-norm diagnostic.
            [2] Polar chirality_scalar(theta) and stacked histogram.
        """
        op = self._operators['chi']
        result = op.apply()
        diag = op.diagnostics()
        self._log_diagnostics('chi', diag)

        chirality_scalar = result['scalar_per_ray']
        C_mu             = result['C_mu']
        D_chiral         = result['D_chiral']
        chs              = diag['chirality_stats']

        # Diagonal mean as a non-unit-norm diagnostic: in unit-norm
        # vectors, D_chiral[m,m] would be zero. Non-zero values
        # quantify how far mu_tensor deviates from unit norm at each
        # sector m.
        diag_mean = float(np.mean(np.diagonal(D_chiral)))

        # P99 saturation
        p99 = float(np.percentile(D_chiral, 99))
        if p99 <= 0.0:
            p99 = float(np.max(D_chiral)) if np.max(D_chiral) > 0 else 1.0

        # Angular axis in degrees
        theta_deg = np.degrees(self.theta_m_array)

        fig = plt.figure(figsize=(18, 5.5))
        gs = GridSpec(1, 3, figure=fig, wspace=0.35)
        fig.suptitle(
            r"Operator $\hat{\chi}$: Chirality (Eq. 8c)",
            fontsize=13, fontweight='bold', y=0.99
        )

        # Panel 1: C_mu heatmap (degrees, full diverging scale)
        ax = fig.add_subplot(gs[0])
        # vmin/vmax forced symmetric so that 0 is the divergence center.
        vmax = float(np.max(np.abs(C_mu)))
        if vmax <= 0.0:
            vmax = 1.0
        im = ax.imshow(C_mu, cmap='RdYlBu_r', aspect='auto', origin='lower',
                       extent=[0, 360, 0, 360], vmin=-vmax, vmax=vmax)
        ax.set_title(r"$C_\mu$: Jacobian Correlation of Micro-space",
                     **FONT_TITLE)
        ax.set_xlabel(r"$\theta_{m'}$ (degrees)")
        ax.set_ylabel(r"$\theta_m$ (degrees)")
        plt.colorbar(im, ax=ax, label=r"$\langle\mu_m|\mu_{m'}\rangle_J$",
                     shrink=0.85)

        # Panel 2: D_chiral with P99 saturation + diagonal-mean annotation
        ax = fig.add_subplot(gs[1])
        im = ax.imshow(D_chiral, cmap='YlOrRd', aspect='auto', origin='lower',
                       extent=[0, 360, 0, 360], vmin=0.0, vmax=p99)
        ax.set_title(r"$D_\chi$: $|C(m,m')-C(-m,-m')|$ (saturated at $P_{99}$)",
                     **FONT_TITLE)
        ax.set_xlabel(r"$\theta_{m'}$ (degrees)")
        ax.set_ylabel(r"$\theta_m$ (degrees)")
        plt.colorbar(im, ax=ax, label=r"$|\Delta C|$", shrink=0.85)
        _add_stats_box(ax, [
            rf"$P_{{99}} = {p99:.3e}$",
            rf"diag mean $= {diag_mean:.3e}$",
            r"(diag $> 0$: non-unit-norm $\mu$)",
        ])

        # Panel 3: polar + histogram
        gs_inner = gs[2].subgridspec(2, 1, hspace=0.55)
        ax_polar = fig.add_subplot(gs_inner[0], projection='polar')
        ax_polar.fill_between(self.theta_m_array, 0, chirality_scalar,
                              alpha=0.4, color='darkorange')
        ax_polar.plot(self.theta_m_array, chirality_scalar,
                      lw=1.2, color='darkorange')
        ax_polar.set_title("Chirality per angular sector",
                           pad=12, fontsize=10, fontweight='bold')

        ax_hist = fig.add_subplot(gs_inner[1])
        ax_hist.hist(chirality_scalar, bins=50, color='darkorange', alpha=0.7,
                     edgecolor='white', linewidth=0.4)
        ax_hist.axvline(chs['mean'], color='black', lw=2, ls='--', alpha=0.7)
        if self.population_data and 'chi' in self.population_data:
            pop = self.population_data['chi'].get('samples')
            if pop is not None:
                ax_hist.hist(pop, bins=50, color='#1f77b4', alpha=0.25,
                             edgecolor='none', label='Healthy population')
                ax_hist.legend(fontsize=7, loc='upper left')
        _add_stats_box(ax_hist, _stats_lines([
            (r"$\mu$",    chs['mean'],   '.3e'),
            (r"$\sigma$", chs['std'],    '.3e'),
            ("median",    chs['median'], '.3e'),
        ]))
        ax_hist.set_xlabel(r"$\sum_{m'}|C(m,m')-C(-m,-m')|$")
        ax_hist.set_ylabel("count")
        _apply_axis_style(ax_hist)

        out_pdf = os.path.join(self.base_dir, 'figures', "Estudo_Operador_chi.pdf")
        fig.savefig(out_pdf, dpi=300, format='pdf', bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Figure exported: {out_pdf}")
        return diag

    # =====================================================================
    # Operator Dr -- Radial Divergence (Eq. 8d)
    # =====================================================================
    def _render_operator_Dr(self) -> dict:
        """
        Render the diagnostic figure for Operator Dr (Eq. 8d). Three
        panels (1x3 GridSpec):
            [0] Profile of ||Delta mu_k||_2 along the radial axis for
                the reference ray.
            [1] Two stacked heatmaps:
                top: ||Delta mu_k||_2(m, k)
                bottom: ln(||Delta mu_k||_2 + epsilon)(m, k)
                (sugestao F5: log overlay reveals where the
                logarithmic compression / amplification regime acts).
                Angular axis in degrees.
            [2] Polar dr_scalar(theta) and stacked histogram.
        """
        op = self._operators['Dr']
        result = op.apply()
        diag = op.diagnostics()
        self._log_diagnostics('Dr', diag)

        dr_scalar     = result['scalar_per_ray']
        diff_norm_map = result['diff_norm_map']
        log_diff_map  = result['log_diff_map']
        ds            = diag['dr_stats']
        sel           = self.selected_m

        # Radial coordinate for the inter-band differences sits at
        # the midpoint between r_k and r_{k+1}; we use r_k[:N'-1] as a
        # canonical x-axis for the profile plot.
        r_x = self.r_k_array[:self.N_prime - 1]

        fig = plt.figure(figsize=(18, 5.5))
        gs = GridSpec(1, 3, figure=fig, wspace=0.32)
        fig.suptitle(
            r"Operator $\hat{D}_r$: Radial Divergence (Eq. 8d)  |  "
            rf"$m={sel}$",
            fontsize=13, fontweight='bold', y=0.99
        )

        # Panel 1: Profile of ||Delta mu_k||_2 for the reference ray
        ax = fig.add_subplot(gs[0])
        ax.plot(r_x, diff_norm_map[sel, :], 'k-', lw=1.5)
        ax.set_title(r"$\|\mu_{k+1,m} - \mu_{k,m}\|_2$ profile",
                     **FONT_TITLE)
        ax.set_xlabel(r"Radial position $r_k$ (px)")
        ax.set_ylabel(r"$\|\Delta\mu_k\|_2$")
        _apply_axis_style(ax)

        # Panel 2: stacked heatmaps (linear top, log bottom)
        gs_inner = gs[1].subgridspec(2, 1, hspace=0.40)
        ax_lin = fig.add_subplot(gs_inner[0])
        im_lin = ax_lin.imshow(
            diff_norm_map.T, cmap='hot', aspect='auto', origin='lower',
            extent=[0, 360, 0, self.N_prime - 1]
        )
        ax_lin.set_title(r"$\|\Delta\mu_k\|_2$ (linear)",
                         fontsize=10, fontweight='bold')
        ax_lin.set_xlabel(r"$\theta_m$ (degrees)")
        ax_lin.set_ylabel(r"radial band $k$")
        plt.colorbar(im_lin, ax=ax_lin, shrink=0.9)

        ax_log = fig.add_subplot(gs_inner[1])
        im_log = ax_log.imshow(
            log_diff_map.T, cmap='magma', aspect='auto', origin='lower',
            extent=[0, 360, 0, self.N_prime - 1]
        )
        ax_log.set_title(r"$\ln(\|\Delta\mu_k\|_2 + \epsilon)$ (log)",
                         fontsize=10, fontweight='bold')
        ax_log.set_xlabel(r"$\theta_m$ (degrees)")
        ax_log.set_ylabel(r"radial band $k$")
        plt.colorbar(im_log, ax=ax_log, shrink=0.9)

        # Panel 3: polar + histogram
        gs_panel3 = gs[2].subgridspec(2, 1, hspace=0.55)
        ax_polar = fig.add_subplot(gs_panel3[0], projection='polar')
        # Floor of the polar fill at min(dr_scalar) for visibility
        floor_value = float(np.min(dr_scalar))
        ax_polar.fill_between(self.theta_m_array, floor_value, dr_scalar,
                              alpha=0.4, color='purple')
        ax_polar.plot(self.theta_m_array, dr_scalar,
                      lw=1.2, color='darkviolet')
        ax_polar.set_title(
            r"$\sum_k \ln(\|\Delta\mu\| + \epsilon)$ per angular sector",
            pad=12, fontsize=10, fontweight='bold'
        )

        ax_hist = fig.add_subplot(gs_panel3[1])
        ax_hist.hist(dr_scalar, bins=50, color='darkviolet', alpha=0.7,
                     edgecolor='white', linewidth=0.4)
        ax_hist.axvline(ds['mean'], color='black', lw=2, ls='--', alpha=0.7)
        if self.population_data and 'Dr' in self.population_data:
            pop = self.population_data['Dr'].get('samples')
            if pop is not None:
                ax_hist.hist(pop, bins=50, color='#1f77b4', alpha=0.25,
                             edgecolor='none', label='Healthy population')
                ax_hist.legend(fontsize=7, loc='upper left')
        _add_stats_box(ax_hist, _stats_lines([
            (r"$\mu$",    ds['mean'], '.3f'),
            (r"$\sigma$", ds['std'],  '.3f'),
            ("min",       ds['min'],  '.3f'),
            ("max",       ds['max'],  '.3f'),
        ]))
        ax_hist.set_xlabel(r"$D_r$ scalar")
        ax_hist.set_ylabel("count")
        _apply_axis_style(ax_hist)

        out_pdf = os.path.join(self.base_dir, 'figures', "Estudo_Operador_Dr.pdf")
        fig.savefig(out_pdf, dpi=300, format='pdf', bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Figure exported: {out_pdf}")
        return diag

    # =====================================================================
    # Operator R -- Reciprocal (Eq. 8e)
    # =====================================================================
    def _render_operator_R(self) -> dict:
        """
        Render the diagnostic figure for Operator R (Eq. 8e). Three
        panels (1x3 GridSpec):
            [0] Frequency spectrum of the reference ray (all four
                Zernike modes, log scale, frequency in cycles/pixel
                per F4).
            [1] Heatmap of total spectral energy per ray per mode.
            [2] Tilt-Y diagnostic panel (m1 minor finding):
                top: spatial radial profile mu_2(k) vs mu_3(k) for
                     the reference ray.
                bottom: |FFT|/sqrt(N') spectra of mu_2 vs mu_3,
                        log-scale magnitude vs frequency in cycles/px.
                Allows visual triage between aliasing, anatomical
                anisotropy, and discrete-extraction quantization
                hypotheses for the anomalously low Tilt-Y energy
                concentration reported in earlier validation runs.
        """
        op = self._operators['R']
        result = op.apply()
        diag = op.diagnostics()
        self._log_diagnostics('R', diag)

        psi_R              = result['psi']
        spectral_energy    = result['spectral_energy']
        energy_concentration = result['energy_concentration']
        zero_per_mode      = result['zero_energy_rays_per_mode']
        K_freq             = result['K_freq']
        sel                = self.selected_m

        # Frequency axis in cycles per pixel:
        # f_k = k / (N' * delta_r), with delta_r in pixels.
        N = self.N_prime
        N_half = N // 2
        freq_index = np.arange(N)
        cycles_per_pixel = freq_index / (N * self.delta_r)

        fig = plt.figure(figsize=(18, 5.8))
        gs = GridSpec(1, 3, figure=fig, wspace=0.32)
        fig.suptitle(
            r"Operator $\hat{R}$: Reciprocal (Eq. 8e)  |  "
            rf"$m={sel}$",
            fontsize=13, fontweight='bold', y=0.99
        )

        # Panel 1: frequency spectrum of the reference ray
        ax = fig.add_subplot(gs[0])
        for q in range(self.Q):
            ax.semilogy(cycles_per_pixel[:N_half],
                        psi_R[sel, :N_half, q],
                        color=MODE_COLORS[q], lw=1.5,
                        label=MODE_LABELS[q])
        ax.set_title(r"Frequency spectrum (positive frequencies only)",
                     **FONT_TITLE)
        ax.set_xlabel(r"frequency (cycles / pixel)")
        ax.set_ylabel(r"$|\mathrm{FFT}(\mu_q)| / \sqrt{N'}$")
        ax.legend(fontsize=7, loc='best')
        _apply_axis_style(ax)

        # Panel 2: total spectral energy heatmap
        ax = fig.add_subplot(gs[1])
        im = ax.imshow(spectral_energy.T, cmap='viridis', aspect='auto',
                       origin='lower', extent=[0, 360, 0.5, self.Q + 0.5])
        ax.set_yticks([1, 2, 3, 4])
        ax.set_yticklabels(['Piston (q=1)', 'Tilt-X (q=2)',
                            'Tilt-Y (q=3)', 'Defocus (q=4)'])
        ax.set_title("Spectral energy per ray per Zernike mode",
                     **FONT_TITLE)
        ax.set_xlabel(r"$\theta_m$ (degrees)")
        ax.set_ylabel("Zernike mode")
        plt.colorbar(im, ax=ax,
                     label=r"$\sum_k |\widetilde{\mu}_{k,m,q}|^2$",
                     shrink=0.85)

        # Energy concentration overlay text
        ec_text = "\n".join([
            rf"low-freq fraction ($k<{K_freq}$):",
            rf"  Piston:  $\mu={diag['energy_stats'][0]['mean']:.3f}$",
            rf"  Tilt-X:  $\mu={diag['energy_stats'][1]['mean']:.3f}$",
            rf"  Tilt-Y:  $\mu={diag['energy_stats'][2]['mean']:.3f}$",
            rf"  Defocus: $\mu={diag['energy_stats'][3]['mean']:.3f}$",
        ])
        ax.text(0.02, 0.98, ec_text, transform=ax.transAxes,
                va='top', ha='left', fontsize=7, family='monospace',
                bbox=STATS_BBOX)

        # Panel 3: Tilt-Y diagnostic (m1 minor)
        gs_inner = gs[2].subgridspec(2, 1, hspace=0.55)

        # Top subpanel: spatial radial profile of mu_2 (Tilt-X) vs mu_3 (Tilt-Y)
        ax_top = fig.add_subplot(gs_inner[0])
        ax_top.plot(self.r_k_array, self.mu_tensor[sel, :, 1],
                    color=MODE_COLORS[1], lw=1.5, label=MODE_LABELS[1])
        ax_top.plot(self.r_k_array, self.mu_tensor[sel, :, 2],
                    color=MODE_COLORS[2], lw=1.5, label=MODE_LABELS[2])
        ax_top.axhline(0, color='gray', lw=0.6, alpha=0.6)
        ax_top.set_title(
            r"Tilt-X vs Tilt-Y radial profile (ray $m=" + str(sel) + r"$)",
            fontsize=10, fontweight='bold'
        )
        ax_top.set_xlabel(r"$r_k$ (px)")
        ax_top.set_ylabel(r"$\mu_q^{(k,m)}$")
        ax_top.legend(fontsize=7, loc='best')
        _apply_axis_style(ax_top)

        # Bottom subpanel: spectra of Tilt-X vs Tilt-Y
        ax_bot = fig.add_subplot(gs_inner[1])
        ax_bot.semilogy(cycles_per_pixel[:N_half],
                        psi_R[sel, :N_half, 1],
                        color=MODE_COLORS[1], lw=1.5, label=MODE_LABELS[1])
        ax_bot.semilogy(cycles_per_pixel[:N_half],
                        psi_R[sel, :N_half, 2],
                        color=MODE_COLORS[2], lw=1.5, label=MODE_LABELS[2])
        # Diagnostic stats on Tilt-Y
        tilty_concentration = diag['energy_stats'][2]['mean']
        tiltx_concentration = diag['energy_stats'][1]['mean']
        _add_stats_box(ax_bot, [
            rf"Tilt-X low-freq fraction $= {tiltx_concentration:.3f}$",
            rf"Tilt-Y low-freq fraction $= {tilty_concentration:.3f}$",
            rf"ratio Tilt-Y / Tilt-X $= {tilty_concentration / max(tiltx_concentration, 1e-30):.3f}$",
        ])
        ax_bot.set_title(
            r"Tilt-X vs Tilt-Y spectra (Tilt-Y diagnostic, m1 minor)",
            fontsize=10, fontweight='bold'
        )
        ax_bot.set_xlabel("frequency (cycles / pixel)")
        ax_bot.set_ylabel(r"$|\mathrm{FFT}(\mu_q)| / \sqrt{N'}$")
        ax_bot.legend(fontsize=7, loc='best')
        _apply_axis_style(ax_bot)

        out_pdf = os.path.join(self.base_dir, 'figures', "Estudo_Operador_R.pdf")
        fig.savefig(out_pdf, dpi=300, format='pdf', bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Figure exported: {out_pdf}")

        # Promote the zero_energy_rays_per_mode list into the diag for
        # the final summary log.
        diag['zero_energy_rays_per_mode'] = zero_per_mode
        return diag

    # =====================================================================
    # Pipeline
    # =====================================================================
    def run_pipeline(self, operators=None, population_data=None):
        """
        Execute validation for the requested operators.

        Parameters
        ----------
        operators : 'all', str, or list of str
            Which operators to validate. Valid names: O, S, chi, Dr, R.
        population_data : dict, optional
            Architectural hook (S.11 of the implementation plan):
            when present, activates population overlays on the
            histogram panels. Schema:
                {'O':   {'samples': ndarray (N_pop,)},
                 'S':   {'cohesion_samples': ndarray (N_pop,)},
                 'chi': {'samples': ndarray (N_pop,)},
                 'Dr':  {'samples': ndarray (N_pop,)},
                 'R':   {...}}
            Defaults to None (no overlay). Pass-through only at this
            stage; populated when N >= 2 healthy exams are available.
        """
        logging.info("=" * 60)
        logging.info("STUDY: PHASE I EXACERBATION OPERATOR VALIDATION")
        logging.info("=" * 60)

        self.population_data = population_data

        self._load_state()
        self._validate_input()
        self._build_operators()

        if operators is None or operators == 'all':
            operators = ['O', 'S', 'chi', 'Dr', 'R']
        elif isinstance(operators, str):
            operators = [operators]

        renderers = {
            'O':   self._render_operator_O,
            'S':   self._render_operator_S,
            'chi': self._render_operator_chi,
            'Dr':  self._render_operator_Dr,
            'R':   self._render_operator_R,
        }

        diagnostics = {}
        for op in operators:
            if op not in renderers:
                logging.warning(f"Unknown operator '{op}' -- skipped.")
                continue
            diagnostics[op] = renderers[op]()

        # ---- Final summary ----
        logging.info("=" * 60)
        logging.info("VALIDATION SUMMARY")
        logging.info("=" * 60)
        header = f"{'Op':<6} {'Correctness':<14} {'Status':<6}"
        logging.info(header)
        logging.info("-" * len(header))
        for name, diag in diagnostics.items():
            corr = diag.get('correctness_error', diag.get('parseval_error', float('nan')))
            st   = diag.get('status', 'N/A')
            logging.info(f"{name:<6} {corr:<14.2e} {st:<6}")
        n_pass = sum(1 for d in diagnostics.values() if d.get('status') == 'PASS')
        n_total = len(diagnostics)
        logging.info(f"\nResult: {n_pass}/{n_total} PASSED.")
        logging.info("=" * 60)


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase I exacerbation operator validation harness."
    )
    parser.add_argument(
        '--operator', type=str, default='all',
        choices=['all', 'O', 'S', 'chi', 'Dr', 'R'],
        help="Operator to validate (default: all)."
    )
    args = parser.parse_args()

    study = OperatorStudy()
    study.run_pipeline(operators=args.operator)
