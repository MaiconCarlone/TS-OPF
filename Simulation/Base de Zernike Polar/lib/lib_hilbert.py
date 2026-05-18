# =============================================================================
# lib_hilbert.py — Álgebra do Espaço de Hilbert com Métrica Jacobiana
# =============================================================================
# Concentra todas as operações algébricas fundamentais do espaço vetorial
# L² ponderado pelo Jacobiano polar (Eq. 2 do ResumoV5.md).
#
# Padrão arquitetural: classe centrada em estado (atributos de classe),
# métodos atuam sobre self sem retorno, seguindo o padrão de
# OPERADOR_Hilbert (HilbertOperators/hilbert_operators.py).
#
# Dependências: numpy, logging
# Referência: ResumoV5.md — Seção 1, Eqs. 1b, 2, D3 (S5)
# =============================================================================

import numpy as np
import logging


class HilbertSpace:
    """
    Infraestrutura algébrica do espaço de Hilbert real
    (ℝ^{n_radial_bands × n_zernike_modes}, ⟨·|·⟩_J).

    Operações incluídas:
    - Produto interno Jacobiano ⟨V_m | V_m'⟩_J (Eq. 2)
    - Norma Jacobiana ‖V_m‖_J
    - Normalização global sob métrica Jacobiana (Eq. 1b)
    - Validação de unitaridade
    - Verificação de ortogonalidade da base polinomial

    Correspondência: antigo OPERADOR_Hilbert (hilbert_operators.py),
    adaptado para tensores n_radial_bands × n_zernike_modes com métrica r_k.
    """

    def __init__(self, V_tensor: np.ndarray, r_k_array: np.ndarray):
        """
        Construtor: recebe o tensor de estados e a malha radial.

        Parameters:
            V_tensor (ndarray): Tensor de estados
                                (n_angular_sectors, n_radial_bands, n_zernike_modes).
                                Pode ser não-normalizado; a normalização é aplicada
                                explicitamente por normalize_jacobian().
            r_k_array (ndarray): Array de raios centrais das faixas radiais
                                 (n_radial_bands,).
        """
        # Tensor de estados de Hilbert: |V_m⟩_z ∈ ℝ^{n_radial_bands × n_zernike_modes}
        self.V_tensor = V_tensor

        # Malha radial: r_k para k=0..n_radial_bands-1
        self.r_k_array = r_k_array

        # Dimensões extraídas do tensor
        self.n_angular_sectors, self.n_radial_bands, self.n_zernike_modes = self.V_tensor.shape

    # -------------------------------------------------------------------------
    # Produto Interno Jacobiano — Eq. 2
    # -------------------------------------------------------------------------
    def inner_product_J(self, V_m1: np.ndarray, V_m2: np.ndarray) -> float:
        """
        Calcula o produto interno ponderado pelo Jacobiano entre dois
        estados angulares V_m1, V_m2 ∈ ℝ^{n_radial_bands × n_zernike_modes}.

        ⟨V_m | V_m'⟩_J = Σ_k r_k · (c_k,m)^T (c_k,m')    [Eq. 2]

        O fator r_k é |∂(x,y)/∂(r,θ)| = r, garantindo conservação de
        área física: setores periféricos (grande r_k) pesam mais que
        setores centrais (pequeno r_k).

        Parameters:
            V_m1 (ndarray): Primeiro estado (n_radial_bands, n_zernike_modes).
            V_m2 (ndarray): Segundo estado (n_radial_bands, n_zernike_modes).

        Returns:
            float: Escalar ⟨V_m1 | V_m2⟩_J.

        Referência: ResumoV5.md — Seção 1, Eq. 2.
        """
        # Produto escalar n_zernike_modes-dimensional em cada profundidade k
        dot_per_band = np.sum(V_m1 * V_m2, axis=1)  # shape: (n_radial_bands,)

        # Contração Jacobiana sobre o eixo radial
        return np.sum(self.r_k_array * dot_per_band)

    # -------------------------------------------------------------------------
    # Norma Jacobiana
    # -------------------------------------------------------------------------
    def norm_J(self, V_m: np.ndarray) -> float:
        """
        Calcula a norma Jacobiana de um estado: ‖V_m‖_J = √⟨V_m|V_m⟩_J.

        Parameters:
            V_m (ndarray): Estado (n_radial_bands, n_zernike_modes).

        Returns:
            float: Norma Jacobiana.
        """
        return np.sqrt(self.inner_product_J(V_m, V_m))

    # -------------------------------------------------------------------------
    # Normalização Jacobiana Global — Eq. 1b
    # -------------------------------------------------------------------------
    def normalize_jacobian(self) -> None:
        """
        Impõe ⟨V_m | V_m⟩_J = 1 para todo m, dividindo cada estado
        pela sua norma Jacobiana. Modifica self.V_tensor in-place.

        Raias com norma nula (fundo isolado) são preservadas e logadas
        como warning.

        Referência: ResumoV5.md — Seção 1, Eq. 1b (constante 𝒩).
        """
        logging.info("[HilbertSpace] Impondo condição de unitaridade sob Métrica Jacobiana...")

        for m in range(self.n_angular_sectors):
            norm_sq = self.inner_product_J(self.V_tensor[m], self.V_tensor[m])

            if norm_sq > 0:
                self.V_tensor[m] = self.V_tensor[m] / np.sqrt(norm_sq)
            else:
                logging.warning(
                    f"[HilbertSpace] Raia m={m}: norma Jacobiana nula (fundo isolado)."
                )

    # -------------------------------------------------------------------------
    # Validação de Unitaridade
    # -------------------------------------------------------------------------
    def validate_unitarity(self, tolerance: float = 1e-4) -> bool:
        """
        Verifica que ⟨V_m | V_m⟩_J ≈ 1 para todo m, dentro da tolerância.
        Loga resultado e aborta se NaN/Inf são detectados.

        Parameters:
            tolerance (float): Desvio máximo aceitável de 1.0.

        Returns:
            bool: True se unitaridade garantida, False caso contrário.

        Referência: ResumoV5.md — Seção 1, Eq. 1b + Eq. 2.
        """
        logging.info("[HilbertSpace] Auditoria: avaliando unitaridade do tensor...")

        # Verificação de integridade numérica
        if np.any(np.isnan(self.V_tensor)) or np.any(np.isinf(self.V_tensor)):
            logging.error(
                "[HilbertSpace] FALHA: NaN ou Inf detectados no tensor. Ruptura matemática."
            )
            return False

        # Computar normas Jacobianas para todas as raias
        norms = np.array([
            self.inner_product_J(self.V_tensor[m], self.V_tensor[m])
            for m in range(self.n_angular_sectors)
        ])

        max_n, min_n = np.max(norms), np.min(norms)

        if abs(max_n - 1.0) < tolerance and abs(min_n - 1.0) < tolerance:
            logging.info(
                f"[HilbertSpace] PASSOU: Unitaridade Jacobiana Garantida. "
                f"Erro máximo: {abs(max_n - 1.0):.2e}"
            )
            return True
        else:
            logging.warning(
                f"[HilbertSpace] ALERTA: Unitaridade Violada. "
                f"Min norma: {min_n:.4f}, Max norma: {max_n:.4f}"
            )
            return False

    # -------------------------------------------------------------------------
    # Verificação de Ortogonalidade da Base Zernike
    # -------------------------------------------------------------------------
    def validate_basis_orthogonality(self, basis_func, n_samples: int = 50000) -> bool:
        """
        Verifica numericamente a ortogonalidade da base polinomial utilizada
        na projeção Eq. 1a, via integração Monte Carlo sobre o disco unitário.

        A integral a estimar é:
            G_{pq} = ∫_0^1 ∫_0^{2π} φ_p(ρ,φ) · φ_q(ρ,φ) · ρ dρ dφ

        Amostragem: ρ = √u com u ~ U[0,1], φ ~ U[0,2π).
        A mudança de variável ρ = √u absorve o Jacobiano (ρ dρ → du/2),
        de modo que a integral se reduz a:
            G_{pq} = π · E_u[φ_p(√u, φ) · φ_q(√u, φ)]

        Para base ortogonal (Zernike), G deve ser diagonal.

        Parameters:
            basis_func (callable): Função que recebe (rho, phi) e retorna
                                   array de n_zernike_modes coeficientes de base.
            n_samples (int): Número de pontos Monte Carlo (default: 50000).

        Returns:
            bool: True se G é aproximadamente diagonal (off-diag < 0.05).
        """
        logging.info(
            f"[MC Ortogonalidade] Início: {n_samples} amostras, "
            f"n_zernike_modes={self.n_zernike_modes} modos, domínio: disco unitário ρ∈[0,1], φ∈[0,2π)."
        )

        # Amostragem uniforme no disco unitário: ρ = √u absorve Jacobiano ρ dρ
        u = np.random.uniform(0, 1, n_samples)
        rho_samples = np.sqrt(u)
        phi_samples = np.random.uniform(0, 2 * np.pi, n_samples)

        # Avaliar base em todos os pontos
        n_modes = self.n_zernike_modes
        basis_values = np.zeros((n_samples, n_modes))
        for i in range(n_samples):
            basis_values[i, :] = basis_func(rho_samples[i], phi_samples[i])

        # Estimar matriz de Gram via MC:
        # G_{pq} = π · E[φ_p · φ_q]  (Jacobiano já absorvido pela amostragem ρ=√u)
        gram = np.zeros((n_modes, n_modes))
        for p in range(n_modes):
            for q in range(n_modes):
                gram[p, q] = np.pi * np.mean(basis_values[:, p] * basis_values[:, q])

        # Valores teóricos para Zernike standard: ⟨Z_n^m, Z_n^m⟩ = π/(n+1)
        # Z_0^0: π/1 = π ≈ 3.1416
        # Z_1^1: π/2 ÷ 2 = π/4 ≈ 0.7854  (fator 1/2 pelo cos²φ)
        # Z_1^{-1}: π/4 ≈ 0.7854
        # Z_2^0: π/3 ≈ 1.0472
        teorico = np.array([np.pi, np.pi / 4, np.pi / 4, np.pi / 3])

        # Verificar diagonal-dominância
        diag = np.diag(gram)
        off_diag_max = np.max(np.abs(gram - np.diag(diag)))
        diag_erro = np.abs(diag - teorico)

        logging.info(
            f"[MC Ortogonalidade] Gram diagonal MC:     {np.round(diag, 4)}"
        )
        logging.info(
            f"[MC Ortogonalidade] Gram diagonal teórica: {np.round(teorico, 4)}"
        )
        logging.info(
            f"[MC Ortogonalidade] Erro diagonal absoluto: {np.round(diag_erro, 4)}"
        )
        logging.info(
            f"[MC Ortogonalidade] Off-diagonal máximo:   {off_diag_max:.4f} (limiar: 0.05)"
        )

        if off_diag_max < 0.05:
            logging.info(
                f"[MC Ortogonalidade] PASSOU: Base ortogonal confirmada ({n_samples} amostras)."
            )
            return True
        else:
            logging.warning(
                f"[MC Ortogonalidade] FALHOU: Off-diagonal {off_diag_max:.4f} > 0.05. "
                "Base pode não ser ortogonal."
            )
            return False
