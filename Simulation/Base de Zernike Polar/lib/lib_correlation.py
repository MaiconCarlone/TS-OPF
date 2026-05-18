import numpy as np
import logging

class GlobalCorrelation:
    """
    Lib 1: Correlação Global e Métrica Jacobiana.
    Integra os pacotes de transposição (V_tensor) estabelecendo matriz de interação
    com métrica Jacobiana, e realiza a bifurcação Macro/Micro (Teorema Espectral).
    """
    def __init__(self, V_tensor, r_k_array):
        self.V_tensor = V_tensor
        self.r_k_array = r_k_array
        self.n_angular_sectors, self.n_radial_bands, self.n_zernike_modes = self.V_tensor.shape
        self.n_macro_modes = 1
        self.n_noise_floor = 0

    def inner_product_J(self, V_m1, V_m2):
        # Produto de cada coeficiente polinomial (Piston, TiltX, TiltY, Defocus)
        dot_per_band = np.sum(V_m1 * V_m2, axis=1) # Dim: (n_radial_bands,)
        # Contração sobre o Jacobiano local iterado: Σ r_k * (c1·c1 + c2·c2...)
        return np.sum(self.r_k_array * dot_per_band)

    def compute_correlation_and_bifurcate(self, K_threshold=1):
        logging.info("[GlobalCorrelation] Computando Matriz Ortogonal C_z (Jacobiano Ponderado)...")
        self.n_macro_modes = K_threshold
        
        # Eq 3. Construção densa O(n_angular_sectors³) limitante da C_matrix
        self.C_matrix = np.zeros((self.n_angular_sectors, self.n_angular_sectors))
        for m1 in range(self.n_angular_sectors):
            for m2 in range(m1, self.n_angular_sectors): # Aproveitando simetria
                ip = self.inner_product_J(self.V_tensor[m1], self.V_tensor[m2])
                self.C_matrix[m1, m2] = ip
                self.C_matrix[m2, m1] = ip
                
        # Eq 4. Teorema espectral Ĉ_z |u_i⟩ = λ_i |u_i⟩
        eigenvalues, eigenvectors = np.linalg.eigh(self.C_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[idx]
        self.eigenvectors = eigenvectors[:, idx]
        
        logging.info(f"[GlobalCorrelation] Bifurcação. Autovalor Macro Dominante (λ_1): {self.eigenvalues[0]:.2e}")
        
        # Eqs 5 e 6: Cisão de Subespaços
        self.C_Macro = np.zeros((self.n_angular_sectors, self.n_angular_sectors))
        for i in range(self.n_macro_modes):
            u_i = self.eigenvectors[:, i]
            self.C_Macro += self.eigenvalues[i] * np.outer(u_i, u_i)
            
        self.C_Micro = np.zeros((self.n_angular_sectors, self.n_angular_sectors))
        for i in range(self.n_macro_modes, self.n_angular_sectors - self.n_noise_floor):
            u_i = self.eigenvectors[:, i]
            self.C_Micro += self.eigenvalues[i] * np.outer(u_i, u_i)
            
        # Eq 7: Isolar o campo resíduo (Micro-espaço) subjazendo anomalias para as 5 Lentes
        self.mu_tensor = np.zeros_like(self.V_tensor)
        for m in range(self.n_angular_sectors):
            coeffs = self.C_Micro[m, :]
            # Contração vetorial ponderando a influência da grade toda em m
            self.mu_tensor[m] = np.sum(self.V_tensor * coeffs[:, None, None], axis=0)

        # T6 — Macro-tensor: projeção análoga usando C_Macro (Eq. 5 aplicada como operador)
        # Ĉ_Macro |V_m⟩ = Σ_{m'} C_Macro(m,m') · |V_{m'}⟩
        # Disponibilizado como atributo de classe para evitar recomputação externa.
        self.macro_tensor = np.zeros_like(self.V_tensor)
        for m in range(self.n_angular_sectors):
            coeffs_macro = self.C_Macro[m, :]
            self.macro_tensor[m] = np.sum(self.V_tensor * coeffs_macro[:, None, None], axis=0)

        logging.info("[GlobalCorrelation] Destilação isolada concluída. μ_m e macro_tensor liberados.")
        return self.mu_tensor
