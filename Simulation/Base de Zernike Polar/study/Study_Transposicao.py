# =============================================================================
# Study_Transposicao.py — Section 1 Validation: Spatial Transposition
# =============================================================================
# Orchestrates the execution of the polar transposition module with the
# Zernike basis, validates the properties of the resulting Hilbert space,
# and generates the vectorial visualisation of the output.
#
# Dependencies: Spatial_Transposition, lib_hilbert, lib_correlation
# Reference: README.md (v7.0) — Section 1 (Eqs. 1a, 1b, 2), D1, D3
# =============================================================================

import os
import sys
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ---------------------------------------------------------------------------
# Path resolution: this script lives in study/; libs live in ../lib/
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))   # .../study
_BASE = os.path.dirname(_HERE)                        # .../Base de Zernike Polar
_LIB  = os.path.join(_BASE, 'lib')
for _p in (_LIB, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Spatial_Transposition import SpatialTransposition
from lib_hilbert import HilbertSpace


class VisualAnalytics:
    """
    Dedicated class for the mathematical visual rendering of the
    Representation Space. Isolates figure manipulation from algorithmic
    logic.

    Usage:
        viewer = VisualAnalytics(st_module, mu_tensor, macro_tensor,
                                 selected_angle_deg=45.0)
        viewer.export_figure(out_pdf)
    """

    def __init__(self, st_module: SpatialTransposition,
                 mu_tensor: np.ndarray,
                 macro_tensor: np.ndarray,
                 selected_angle_deg: float = 45.0):
        self.st = st_module
        self.mu_tensor = mu_tensor
        self.macro_tensor = macro_tensor

        # Convert the requested angle (degrees, trigonometric circle) to
        # the nearest sector index in the discrete theta_m_array grid.
        # Reference: theta_m = m * Delta_theta, so m = round(theta_deg /
        # (360 / n_angular_sectors)).
        target_rad = np.radians(selected_angle_deg) % (2.0 * np.pi)
        self.sel_m = int(np.argmin(np.abs(self.st.theta_m_array - target_rad)))
        self.selected_angle_deg = selected_angle_deg
        logging.info(
            f"[VisualAnalytics] Requested angle: {selected_angle_deg:.1f} deg, "
            f"nearest sector: m={self.sel_m} "
            f"(theta={np.degrees(self.st.theta_m_array[self.sel_m]):.2f} deg)"
        )

        # Grid layout: 2 rows x 3 columns.
        # Column 0 (full height): anatomical overlay.
        # Column 1 (split): Zernike packet (top) + operator projection (bottom).
        # Column 2 (full height): RGB Jacobian spectral map.
        self.fig = plt.figure(figsize=(20, 7))
        self.gs = self.fig.add_gridspec(2, 3)
        self.ax1     = self.fig.add_subplot(self.gs[:, 0])
        self.ax2_top = self.fig.add_subplot(self.gs[0, 1])
        self.ax2_bot = self.fig.add_subplot(self.gs[1, 1], sharex=self.ax2_top)
        self.ax3     = self.fig.add_subplot(self.gs[:, 2])

    # -------------------------------------------------------------------------
    # Subplot 1: MRI brain with structured polar extraction overlay
    # -------------------------------------------------------------------------
    def _render_anatomical_mapping(self):
        """
        Render the anatomical reference panel: inverted-grayscale MRI with
        overlaid polar grid (radial bands, angular boundaries, selected
        sector wedge, and topological pivot).
        """
        inverted_image = 255 - self.st.image
        self.ax1.imshow(inverted_image, cmap='gray',
                        extent=[0, self.st.width, self.st.height, 0])
        self.ax1.set_title(
            r"Sensorial Extraction: Cartesian $\to$ Polar Transposition",
            fontsize=12, fontweight='bold'
        )
        self.ax1.axis('equal')

        # Topological pivot (symmetry centre, D1)
        self.ax1.plot(self.st.centerX, self.st.centerY, 'r+',
                      markersize=10, label="Topological Pivot (D1)")

        # Shannon exclusion boundary (r_min)
        circle_rmin = patches.Circle(
            (self.st.centerX, self.st.centerY), self.st.r_min,
            edgecolor='red', facecolor='none', lw=2, linestyle='--',
            label=r'$r_{min}$ Singularity'
        )
        self.ax1.add_patch(circle_rmin)

        # Radial depth bands (subsampled for visual clarity)
        step_r = max(1, self.st.n_radial_bands // 10)
        for r_k in self.st.r_k_array[::step_r]:
            circle = patches.Circle(
                (self.st.centerX, self.st.centerY), r_k,
                edgecolor='white', facecolor='none', lw=1.2, alpha=0.9
            )
            self.ax1.add_patch(circle)

        # Angular sector boundaries (lower edge of each sampled sector)
        step_m = max(1, self.st.n_angular_sectors // 36)
        first_boundary = True
        for t_m in self.st.theta_m_array[::step_m]:
            t_border = t_m - self.st.delta_theta / 2.0
            x_end   = self.st.centerX + self.st.R_max * np.cos(t_border)
            y_end   = self.st.centerY - self.st.R_max * np.sin(t_border)
            x_start = self.st.centerX + self.st.r_min * np.cos(t_border)
            y_start = self.st.centerY - self.st.r_min * np.sin(t_border)
            lbl = f'Angular Boundaries (every {step_m} deg)' if first_boundary else None
            self.ax1.plot([x_start, x_end], [y_start, y_end],
                          color='white', lw=1.2, alpha=0.9, label=lbl)
            first_boundary = False

        # Highlight of the sampled angular sector Omega_{sel_m}.
        # The extraction operates over the full wedge
        # [theta_m +/- Delta_theta/2] x [r_min, R_max], not a point ray.
        # The Wedge patch visualises the actual geometry.
        # Matplotlib Wedge uses degrees, counter-clockwise, y-up.
        # Our y-axis is inverted (imshow), so we negate the angles.
        sel_t = self.st.theta_m_array[self.sel_m]
        theta1_deg = -np.degrees(sel_t + self.st.delta_theta / 2.0)
        theta2_deg = -np.degrees(sel_t - self.st.delta_theta / 2.0)
        wedge = patches.Wedge(
            center=(self.st.centerX, self.st.centerY),
            r=self.st.R_max,
            theta1=theta1_deg,
            theta2=theta2_deg,
            width=self.st.R_max - self.st.r_min,
            facecolor='blue', alpha=0.25,
            edgecolor='blue', lw=2.0,
            label=rf'Sector $\Omega_{{k,{self.sel_m}}}$'
        )
        self.ax1.add_patch(wedge)
        self.ax1.legend(loc='upper right', fontsize=7)

        # Information box: sectorisation parameters (lower-left)
        info_text = (
            rf"$\Delta\theta = {np.degrees(self.st.delta_theta):.2f}^\circ$" + "\n"
            rf"$M = {self.st.n_angular_sectors}$ sectors" + "\n"
            rf"$N' = {self.st.n_radial_bands}$ bands" + "\n"
            rf"Tensor: $\mathbb{{R}}^{{{self.st.n_angular_sectors}"
            rf"\times{self.st.n_radial_bands}\times{self.st.n_zernike_modes}}}$"
        )
        self.ax1.text(
            0.02, 0.02, info_text, transform=self.ax1.transAxes,
            fontsize=7, verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.7),
            color='white', family='monospace'
        )

    # -------------------------------------------------------------------------
    # Subplot 2 top: Zernike packet (local polynomial basis)
    # -------------------------------------------------------------------------
    def _render_zernike_basis(self):
        """
        Render the Zernike coefficient profile of the reference ray
        (m = sel_m) as a function of radial depth r_k. Each of the four
        modes (Piston, Tilt-X, Tilt-Y, Defocus) is plotted separately.
        """
        self.ax2_top.set_title(
            rf"Zernike Packet Vector (Ray $m={self.sel_m}$)",
            fontsize=12, fontweight='bold'
        )
        V_m = self.st.V_tensor[self.sel_m]
        rx  = self.st.r_k_array

        self.ax2_top.plot(rx, V_m[:, 0], color='black',     lw=2,
                          label=r"$Z_0^0$: Piston (DC)")
        self.ax2_top.plot(rx, V_m[:, 1], color='green',     lw=1.5,
                          label=r"$Z_1^1$: Tilt-X ($\rho\cos\varphi$)")
        self.ax2_top.plot(rx, V_m[:, 2], color='darkorange', lw=1.5,
                          label=r"$Z_1^{-1}$: Tilt-Y ($\rho\sin\varphi$)")
        self.ax2_top.plot(rx, V_m[:, 3], color='purple',    lw=2,
                          label=r"$Z_2^0$: Defocus ($2\rho^2-1$)")
        self.ax2_top.set_ylabel('Jacobian-Normalised Amplitude')
        self.ax2_top.legend(loc='upper right', fontsize=8)
        self.ax2_top.grid(True, linestyle='--', alpha=0.6)
        plt.setp(self.ax2_top.get_xticklabels(), visible=False)

    # -------------------------------------------------------------------------
    # Subplot 2 bottom: Operator O projection (Macro vs Micro)
    # -------------------------------------------------------------------------
    def _render_operator_projection(self):
        """
        Render the action of the Intensity operator hat{O} (Eq. 8a) on
        both the macro-space and micro-space tensors of the reference ray,
        showing the relative amplitude profile along the radial axis.
        """
        self.ax2_bot.set_title(
            r"Directional Projection: Operator $\hat{O}$ Action",
            fontsize=11, fontweight='bold'
        )

        V_micro_m = self.mu_tensor[self.sel_m]
        V_macro_m = self.macro_tensor[self.sel_m]

        # Literal action of Operator O (Eq. 8a):
        # psi_O = mean(mu[:, 0]) * mu
        def apply_operator_O(tensor_ray):
            mean_intensity = np.mean(tensor_ray[:, 0])
            return mean_intensity * tensor_ray

        psi_O_micro = apply_operator_O(V_micro_m)
        psi_O_macro = apply_operator_O(V_macro_m)

        sig_micro = np.linalg.norm(psi_O_micro, axis=1)
        sig_macro = np.linalg.norm(psi_O_macro, axis=1)

        def normalise(sig):
            max_val = np.max(np.abs(sig))
            return sig / max_val if max_val > 0 else sig

        self.ax2_bot.plot(self.st.r_k_array, normalise(sig_macro),
                          color='blue', linestyle='-', lw=2,
                          label=r"$\hat{O}\,|C_{\mathrm{Macro}}\rangle$")
        self.ax2_bot.plot(self.st.r_k_array, normalise(sig_micro),
                          color='red', linestyle='-', lw=2,
                          label=r"$\hat{O}\,|C_{\mathrm{Micro}}\rangle$")
        self.ax2_bot.set_xlabel(r'Radial depth $r_k$ (px)')
        self.ax2_bot.set_ylabel('Relative amplitude')
        self.ax2_bot.legend(loc='upper right', fontsize=8)
        self.ax2_bot.grid(True, linestyle='--', alpha=0.6)

    # -------------------------------------------------------------------------
    # Subplot 3: RGB Jacobian spectral map
    # -------------------------------------------------------------------------
    def _render_rgb_jacobian_map(self):
        """
        Render the translational spectral RGB map weighted by the Jacobian
        (alpha channel). Channels:
            R = |Piston (DC)|  = |Z_0^0|
            G = |Tilt magnitude| = sqrt(Z_1^1^2 + Z_1^{-1}^2)
            B = |Defocus (curvature)| = |Z_2^0|
            alpha = r_k / R_max  (Jacobian weight, linear in radius)
        """
        self.ax3.set_title(
            r"RGB Spectral Vibration (Jacobian Metric via $\alpha$)",
            fontsize=12, fontweight='bold'
        )

        rgba_image = np.zeros(
            (self.st.n_angular_sectors, self.st.n_radial_bands, 4),
            dtype=np.float64
        )
        R_channel = np.abs(self.st.V_tensor[:, :, 0])   # DC (Piston)
        G_channel = np.sqrt(                              # Tilt magnitude
            self.st.V_tensor[:, :, 1]**2 +
            self.st.V_tensor[:, :, 2]**2
        )
        B_channel = np.abs(self.st.V_tensor[:, :, 3])   # Defocus / curvature

        def normalise_channel(ch):
            min_val, max_val = np.min(ch), np.max(ch)
            return ((ch - min_val) / (max_val - min_val)
                    if max_val > min_val else np.zeros_like(ch))

        rgba_image[:, :, 0] = normalise_channel(R_channel)
        rgba_image[:, :, 1] = normalise_channel(G_channel)
        rgba_image[:, :, 2] = normalise_channel(B_channel)

        # Alpha channel: Jacobian weight alpha proportional to r_k / R_max
        # (no subtraction of r_min). Remapped to [0.3, 1.0] for legibility.
        rk_matrix = np.tile(self.st.r_k_array, (self.st.n_angular_sectors, 1))
        alpha_channel = rk_matrix / np.max(rk_matrix)
        rgba_image[:, :, 3] = 0.3 + alpha_channel * 0.7   # [0,1] -> [0.3,1.0]

        self.ax3.imshow(
            rgba_image, aspect='auto', interpolation='nearest',
            extent=[self.st.r_k_array[0], self.st.r_k_array[-1],
                    self.st.n_angular_sectors - 1, 0]
        )
        self.ax3.set_xlabel(r'Radial depth $r_k$ (px)')
        self.ax3.set_ylabel(r'Angular ray $m$')

        legend_text = (
            r"R: DC ($Z_0^0$)" + "\n"
            r"G: $\|\nabla\|$ ($Z_1^{\pm 1}$)" + "\n"
            r"B: Curvature ($Z_2^0$)" + "\n"
            r"$\alpha$: Jacobian ($r_k / R_{\max}$)"
        )
        self.ax3.text(
            0.98, 0.98, legend_text, transform=self.ax3.transAxes,
            fontsize=8, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85)
        )

    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------
    def export_figure(self, out_pdf: str):
        """
        Orchestrate all four render methods and write the figure to disk.

        Parameters
        ----------
        out_pdf : str
            Absolute path of the output PDF file.
        """
        self._render_anatomical_mapping()
        self._render_zernike_basis()
        self._render_operator_projection()
        self._render_rgb_jacobian_map()
        plt.tight_layout()
        self.fig.savefig(out_pdf, dpi=300, format='pdf', bbox_inches='tight')
        logging.info(f"[VisualAnalytics] Vector figure compiled successfully: {out_pdf}")
        plt.close(self.fig)


class TranspositionStudy:
    """
    Orchestrator controller for the Spatial Transposition experiment.

    Runs the full pipeline:
        1. Polar transposition with Zernike basis (SpatialTransposition).
        2. Hilbert space validation (unitarity + basis orthogonality).
        3. Spectral bifurcation into Macro/Micro subspaces (GlobalCorrelation).
        4. State serialisation to Transposition_State.npz.
        5. Visual rendering via VisualAnalytics.

    Usage:
        study = TranspositionStudy(target_image="map.png")
        study.run_pipeline()
    """

    def __init__(self, target_image: str):
        self.base_dir = _BASE
        self.log_file = os.path.join(_BASE, 'log',     "Log_Transposicao_Espacial.txt")
        self.img_path = os.path.join(_BASE, 'input',   target_image)
        self.pdf_out  = os.path.join(_BASE, 'figures', "Estudo_Transposicao_Espacial.pdf")

        self._setup_logger()
        self.st_module = None

    # -------------------------------------------------------------------------
    # Logger
    # -------------------------------------------------------------------------
    def _setup_logger(self):
        """Configure dual logging to file (log/) and console."""
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
    # Hilbert space validation
    # -------------------------------------------------------------------------
    def _validate_hilbert_space(self) -> None:
        """
        Formal validations delegated to lib_hilbert.HilbertSpace.
        Checks Jacobian unitarity (Eq. 1b/2) and Zernike basis
        orthogonality via Monte Carlo integration.
        """
        logging.info("[Hilbert Audit] Evaluating metrics of the formed space...")

        hs = HilbertSpace(self.st_module.V_tensor, self.st_module.r_k_array)

        # 1. Jacobian unitarity
        if not hs.validate_unitarity():
            logging.error("CRITICAL FAILURE: Jacobian unitarity violated. Aborting.")
            sys.exit(1)

        # 2. Zernike basis orthogonality (standard Q=4 basis)
        def zernike_basis(rho, phi):
            """Standard Zernike basis (Q=4) for orthogonality verification."""
            z1 = 1.0                    # Z_0^0: Piston
            z2 = rho * np.cos(phi)      # Z_1^1: Tilt-X
            z3 = rho * np.sin(phi)      # Z_1^{-1}: Tilt-Y
            z4 = 2.0 * rho**2 - 1.0    # Z_2^0: Defocus
            return np.array([z1, z2, z3, z4])

        hs.validate_basis_orthogonality(zernike_basis)

    # -------------------------------------------------------------------------
    # State serialisation
    # -------------------------------------------------------------------------
    def _serialize_state(self) -> str:
        """
        Persist the main tensors and geometric metadata from the
        transposition + bifurcation pipeline to a compressed NPZ archive
        at the project root (Base de Zernike Polar/).

        Returns
        -------
        str
            Absolute path of the generated NPZ file.
        """
        out_path = os.path.join(self.base_dir, "Transposition_State.npz")
        st = self.st_module

        try:
            C_matrix     = self.gc.C_matrix
            eigenvalues  = self.gc.eigenvalues
            eigenvectors = self.gc.eigenvectors
            C_Macro      = self.gc.C_Macro
            C_Micro      = self.gc.C_Micro
        except AttributeError:
            logging.warning(
                "[Serialisation] GlobalCorrelation unavailable — "
                "spectral matrices will be empty arrays."
            )
            C_matrix = eigenvalues = eigenvectors = C_Macro = C_Micro = np.array([])

        np.savez_compressed(
            out_path,
            V_tensor          = st.V_tensor,
            mu_tensor         = self.mu_tensor,
            macro_tensor      = self.macro_tensor,
            r_k_array         = st.r_k_array,
            theta_m_array     = st.theta_m_array,
            C_matrix          = C_matrix,
            eigenvalues       = eigenvalues,
            eigenvectors      = eigenvectors,
            C_Macro           = C_Macro,
            C_Micro           = C_Micro,
            delta_r           = st.delta_r,
            delta_theta       = st.delta_theta,
            r_min             = st.r_min,
            R_max             = st.R_max,
            centerX           = st.centerX,
            centerY           = st.centerY,
            n_angular_sectors = st.n_angular_sectors,
            n_radial_bands    = st.n_radial_bands,
            n_zernike_modes   = st.n_zernike_modes,
            image_path        = self.img_path,
        )

        logging.info(f"[Serialisation] Pipeline state persisted at: {out_path}")
        return out_path

    # -------------------------------------------------------------------------
    # Main pipeline
    # -------------------------------------------------------------------------
    def run_pipeline(self):
        """
        Execute the full transposition study pipeline.

        Parameters (hard-coded below, documented inline):
            R_max                : maximum sector radius (px)
            r_min                : central core exclusion radius (px)
            n_angular_sectors    : number of angular sectors (M)
            delta_r              : radial band thickness (px)
            n_zernike_modes      : Zernike coefficients per packet (Q=4)
            skip_montecarlo      : True to use center_hint without MC search
            search_radius_factor : MC search radius scale (x r_min)
            center_hint          : initial (cx, cy) guess for MC
            mc_sectors_coarse    : sectors in MC Phase 1 (coarse)
            mc_sectors_fine      : sectors in MC Phase 2 (fine)
            mc_max_iter_phase1   : max iterations in MC Phase 1
            mc_max_iter_phase2   : max iterations in MC Phase 2
        """
        logging.info("=" * 58)
        logging.info("STUDY: SPATIAL TRANSPOSITION AND RIGOROUS POLYNOMIAL BASIS")
        logging.info("=" * 58)

        if not os.path.exists(self.img_path):
            logging.error(f"FATAL: Image not found at {self.img_path}. Aborting.")
            sys.exit(1)

        # Step 1: Polar transposition with Zernike basis
        self.st_module = SpatialTransposition(
            image_path           = self.img_path,
            R_max                = 360.0,
            r_min                = 29.0,
            n_angular_sectors    = 360,
            delta_r              = 2.0,
            n_zernike_modes      = 4,
            skip_montecarlo      = False,
            search_radius_factor = 1.25,
            center_hint          = (307, 333),
            mc_sectors_coarse    = 36,
            mc_sectors_fine      = 72,
            mc_max_iter_phase1   = 100,
            mc_max_iter_phase2   = 100,
        )
        self.st_module.execute_transposition()
        self._validate_hilbert_space()

        # Step 2: Spectral bifurcation (GlobalCorrelation)
        try:
            from lib_correlation import GlobalCorrelation
            logging.info(
                "[Study] lib_correlation imported successfully. "
                "Running real spectral bifurcation."
            )
            gc = GlobalCorrelation(
                self.st_module.V_tensor, self.st_module.r_k_array
            )
            self.gc = gc
            self.mu_tensor    = gc.compute_correlation_and_bifurcate(K_threshold=1)
            self.macro_tensor = gc.macro_tensor
        except ImportError:
            logging.warning(
                "[Study] 'lib_correlation' unavailable. "
                "Generating mock subspaces for visual rendering only."
            )
            # Fallback: mock subspaces (not orthogonal projectors; visual only)
            self.macro_tensor = self.st_module.V_tensor * 0.95
            self.mu_tensor    = self.st_module.V_tensor * 0.05

        # Step 3: Serialise state for downstream operator validation
        self._serialize_state()

        # Step 4: Visual rendering
        logging.info(
            "[Study] Transitioning Hilbert state to the visualisation engine..."
        )
        viewer = VisualAnalytics(
            self.st_module, self.mu_tensor, self.macro_tensor,
            selected_angle_deg=45.0
        )
        viewer.export_figure(self.pdf_out)

        logging.info("=" * 58)
        logging.info("STUDY COMPLETED SUCCESSFULLY.")
        logging.info("=" * 58)


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    study = TranspositionStudy(target_image="map.png")
    study.run_pipeline()
