import os as _os
import sys as _sys
# Resolve lib/ directory relative to this file's location in study/
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'lib'))

import cv2
import numpy as np
import logging

from lib_hilbert import HilbertSpace

class SpatialTransposition:
    """
    Módulo Base 1: Transposição Espacial com Métrica Jacobiana.
    Arquitetura orientada a estado: todos os tensores e grades espaciais 
    são alocados no construtor e mutados por métodos de responsabilidade única.

    Parâmetros do Construtor:
        image_path (str):               Caminho para a imagem MRI (escala de cinza).
        R_max (float):                  Raio máximo do setor, em pixels. Obrigatório.
                                        Define até onde os setores se estendem desde o centro.
                                        Setores que ultrapassem a borda da imagem coletam
                                        apenas os pixels disponíveis (clip seguro).
        r_min (float):                  Raio mínimo de exclusão do núcleo central, em pixels.
                                        Obrigatório. Deve satisfazer r_min × Δr × Δθ ≥ 1.
                                        Usar o mesmo r_min garante consistência multiplano.
        n_angular_sectors (int):        Número de setores angulares (M). Default: 360.
        delta_r (float):                Espessura radial de cada faixa, em pixels. Default: 2.0.
        n_zernike_modes (int):          Número de coeficientes de Zernike por pacote. Default: 4.
        skip_montecarlo (bool):         True para usar center_hint sem busca MC. Default: False.
        search_radius_factor (float):   Fator de escala do raio de busca MC (×r_min). Default: 1.25.
        center_hint (tuple|None):       Chute inicial (cx, cy) para o MC. None = centro geométrico.
        mc_sectors_coarse (int):        Setores na Fase 1 MC (grossa). Default: 36.
        mc_sectors_fine (int):          Setores na Fase 2 MC (fina). Default: 72.
        mc_max_iter_phase1 (int):       Iterações máximas na Fase 1 MC. Default: 100.
        mc_max_iter_phase2 (int):       Iterações máximas na Fase 2 MC. Default: 100.
    """
    def __init__(self, image_path: str, R_max: float, r_min: float,
                 n_angular_sectors: int = 360, delta_r: float = 2.0,
                 n_zernike_modes: int = 4,
                 skip_montecarlo: bool = False, search_radius_factor: float = 1.25,
                 center_hint: tuple = None,
                 mc_sectors_coarse: int = 36, mc_sectors_fine: int = 72,
                 mc_max_iter_phase1: int = 100, mc_max_iter_phase2: int = 100):

        # =====================================================================
        # 1. Parâmetros Geométricos Fundamentais
        # =====================================================================

        # Caminho para a imagem de entrada (escala de cinza)
        self.image_path = image_path

        # Raio máximo do setor, em pixels (constante de entrada obrigatória).
        # Define o alcance radial da transposição desde o centro até a periferia.
        # Pode exceder os limites da imagem: setores parciais coletam
        # somente os pixels contidos na imagem (clip pelo bounding box).
        self.R_max = float(R_max)

        # Raio mínimo de exclusão do núcleo central, em pixels (constante de entrada obrigatória).
        # Define o início da zona de coleta radial. Deve satisfazer a condição
        # de Shannon/Nyquist: r_min × Δr × Δθ ≥ 1 (área do setor ≥ 1 pixel).
        # Se o valor fornecido violar esta condição, um ValueError é gerado.
        # Usar o mesmo r_min para todas as fatias garante consistência multiplano.
        self.r_min = float(r_min)

        # Número de setores angulares da transposição real.
        # Determina a resolução angular: Δθ = 2π / n_angular_sectors.
        self.n_angular_sectors = n_angular_sectors

        # Espessura radial de cada faixa, em pixels.
        # Determina a resolução radial: cada setor Ω_{k,m} tem profundidade Δr.
        self.delta_r = delta_r

        # Número de coeficientes de Zernike por Pacote de Informação Local.
        # Q=4 corresponde a: Piston (DC), Tilt-Radial, Tilt-Azimutal, Defocus.
        self.n_zernike_modes = n_zernike_modes

        # Abertura angular de cada setor (radianos)
        self.delta_theta = 2.0 * np.pi / self.n_angular_sectors

        # =====================================================================
        # 2. Parâmetros do Monte Carlo (D1)
        # =====================================================================

        # skip_montecarlo=True: usa center_hint (ou centro geométrico) sem busca MC.
        # Útil para execuções rápidas ou debugging.
        self.skip_montecarlo = skip_montecarlo

        # Fator de escala do raio de busca MC.
        # Multiplica r_min por este fator para definir a zona de busca.
        # Valor padrão: 1.25 (25% maior que a zona de exclusão).
        self.search_radius_factor = search_radius_factor

        # Chute inicial (cx, cy) para o centro MC.
        # Se None, usa o centro geométrico da imagem (width//2, height//2).
        self.center_hint = center_hint

        # Número de setores angulares usados APENAS na busca MC, para
        # avaliação rápida do custo Var{C_{m,m'}}. NÃO confundir com
        # n_angular_sectors, que é o número de setores da transposição real.
        # Valores menores aceleram a busca mas reduzem a resolução angular do custo.
        # mc_sectors_coarse: setores na Fase 1 (grossa). Padrão: 36 (~10°/setor).
        # mc_sectors_fine:   setores na Fase 2 (fina).  Padrão: 72 (~5°/setor).
        self.mc_sectors_coarse = mc_sectors_coarse
        self.mc_sectors_fine = mc_sectors_fine

        # Número máximo de iterações em cada fase do MC.
        self.mc_max_iter_phase1 = mc_max_iter_phase1
        self.mc_max_iter_phase2 = mc_max_iter_phase2

        # =====================================================================
        # 3. Buffers de Estado Espacial e Imagem (preenchidos na inicialização)
        # =====================================================================
        self.image = None
        self.height = 0
        self.width = 0
        self.centerX = 0.0
        self.centerY = 0.0
        self.n_radial_bands = 0

        # Malhas de coordenadas polares discretizadas
        self.r_k_array = None
        self.theta_m_array = None

        # Tensor de Estado de Hilbert: |V_m⟩ ∈ ℝ^{n_radial_bands × n_zernike_modes}
        self.V_tensor = None

        # =====================================================================
        # 4. Inicialização da Estrutura
        # =====================================================================
        self._load_and_validate_image()
        self._compute_topological_center()
        self._establish_spatial_grid()
        self._allocate_tensors()

    def _load_and_validate_image(self) -> None:
        """Carrega a imagem e define os limites brutos do plano 2D."""
        self.image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            raise FileNotFoundError(f"Erro Crítico: Imagem não localizada em {self.image_path}")
        self.height, self.width = self.image.shape
        logging.info(f"[Geometria] Imagem ingerida: {self.width}x{self.height} px.")

    def _compute_topological_center(self) -> None:
        """
        D1: Otimização do Centro de Simetria.

        Se skip_montecarlo=True: usa center_hint (ou centro geométrico).
        Se skip_montecarlo=False: Monte Carlo bifásico (D1, ResumoV5).

        Critério interno ao método:
            c* = argmin_{c ∈ Ω} Var{C_{m,m'}(c)}_{m ≠ m'}
        """
        # Centro de partida: center_hint se fornecido, senão centro geométrico
        if self.center_hint is not None:
            cx_start, cy_start = self.center_hint
        else:
            cx_start = self.width // 2
            cy_start = self.height // 2

        if self.skip_montecarlo:
            # Modo rápido: usa o center_hint (ou geométrico) sem busca MC
            self.centerX, self.centerY = cx_start, cy_start
            logging.info(
                f"[Geometria] Pivô Topológico fixado em: ({self.centerX}, {self.centerY}) "
                f"[Monte Carlo desativado — centro {'hint' if self.center_hint else 'geométrico'}]"
            )
            return

        from lib_montecarlo import monte_carlo_center_search

        # Zona de busca: r_min × search_radius_factor
        delta_theta_temp = 2.0 * np.pi / self.n_angular_sectors
        r_temp = 1.0
        while r_temp * self.delta_r * delta_theta_temp < 1.0:
            r_temp += 1.0
        # search_radius_factor escala o raio de busca (padrão: 1.25 = +25%)
        search_radius = r_temp * self.search_radius_factor

        result = monte_carlo_center_search(
            image=self.image,
            center_init=(cx_start, cy_start),
            search_radius=search_radius,
            M_search_coarse=self.mc_sectors_coarse,
            M_search_fine=self.mc_sectors_fine,
            delta_r=self.delta_r,
            n_zernike_modes=self.n_zernike_modes,
            max_iter_phase1=self.mc_max_iter_phase1,
            max_iter_phase2=self.mc_max_iter_phase2
        )

        self.centerX, self.centerY = result['center']
        logging.info(
            f"[Geometria] Pivô Topológico fixado em: ({self.centerX}, {self.centerY}) "
            f"[MC D1, custo={result['cost']:.6e}]"
        )

    def _establish_spatial_grid(self) -> None:
        """
        Valida r_min contra a condição de Nyquist e popula os arrays de
        coordenadas polares discretizadas.
        R_max e r_min são constantes de entrada, não são calculados aqui.
        """
        # R_max já definido pelo usuário como constante de entrada.
        # Aviso se R_max excede os limites da imagem a partir do centro.
        r_inscribed = min(self.centerX, self.width - self.centerX,
                          self.centerY, self.height - self.centerY)
        if self.R_max > r_inscribed:
            logging.warning(
                f"[Malha] R_max={self.R_max:.1f} excede o raio inscrito ({r_inscribed:.1f}). "
                f"Setores periféricos coletarão pixels parciais."
            )

        # Condição de Shannon/Nyquist: Área do setor >= Área do pixel (1.0)
        # Calcula o r_min mínimo admissível para os parâmetros atuais.
        r_nyquist = 1.0
        while (r_nyquist * self.delta_r * self.delta_theta) < 1.0:
            r_nyquist += 1.0

        # Validação: o r_min fornecido deve satisfazer a condição de Nyquist
        if self.r_min < r_nyquist:
            raise ValueError(
                f"[Malha] ERRO: r_min fornecido ({self.r_min:.1f}) viola a condição de "
                f"Shannon/Nyquist. Com delta_r={self.delta_r:.1f} e "
                f"delta_theta={np.degrees(self.delta_theta):.2f}°, o valor mínimo "
                f"admissível é r_min >= {r_nyquist:.1f}."
            )

        logging.info(
            f"[Malha] r_min fornecido: {self.r_min:.1f}, "
            f"r_min Nyquist: {r_nyquist:.1f} — Condição satisfeita."
        )

        self.n_radial_bands = int(np.floor((self.R_max - self.r_min) / self.delta_r))
        self.r_k_array = self.r_min + (np.arange(self.n_radial_bands) + 0.5) * self.delta_r
        self.theta_m_array = np.linspace(0, 2 * np.pi, self.n_angular_sectors, endpoint=False)

        logging.info(f"[Malha] Limites Radiais: r_min={self.r_min:.1f}, R_max={self.R_max:.1f}")
        logging.info(f"[Malha] Discretização: n_angular_sectors={self.n_angular_sectors}, n_radial_bands={self.n_radial_bands}")

    def _allocate_tensors(self) -> None:
        """Aloca o tensor V na memória estrita do construtor."""
        self.V_tensor = np.zeros((self.n_angular_sectors, self.n_radial_bands, self.n_zernike_modes), dtype=np.float64)

    def _evaluate_local_polar_basis(self, r_px: float, theta_px: float, r_c: float, theta_mid: float) -> np.ndarray:
        """
        Implementação dos Polinômios de Zernike Standard (Seção 1, Eq 1a).
        Projeta o ponto cartesiano para o disco unitário normalizado do setor
        e avalia os primeiros n_zernike_modes=4 polinômios de Zernike genuínos.

        Base utilizada (notação Noll/OSA):
            Z_0^0(ρ,φ) = 1                  — Piston (DC)
            Z_1^1(ρ,φ) = ρ·cos(φ)           — Tilt-X (gradiente radial)
            Z_1^{-1}(ρ,φ) = ρ·sin(φ)        — Tilt-Y (gradiente azimutal)
            Z_2^0(ρ,φ) = 2ρ² − 1            — Defocus (curvatura local)

        Ortogonalidade: ∫∫ Z_n^m Z_{n'}^{m'} ρ dρ dφ = [π/(n+1)]·δ_{nn'}δ_{mm'}
        sobre o disco unitário.

        Mapeamento afim do setor para coordenadas polares normalizadas:
            ρ = distância normalizada [0, 1] ao centro radial do setor
            φ = ângulo polar local no referencial do setor

        Referência: ResumoV5.md — Seção 1, S2–S3; Ponto Aberto nº 1.
        """
        # --- Mapeamento afim para coordenadas polares do disco unitário ---
        # Coordenadas normalizadas do ponto relativas ao centro do setor:
        #   u = (r_px - r_c) / (Δr/2)      → coordenada radial normalizada   ∈ [-1, 1]
        #   v = (θ_px - θ_mid) / (Δθ/2)    → coordenada angular normalizada  ∈ [-1, 1]
        # Ambas adimensionais, mesma escala — condição para arctan2 correto.
        d_theta = (theta_px - theta_mid + np.pi) % (2 * np.pi) - np.pi
        dr_norm = (r_px - r_c) / (self.delta_r / 2.0)
        dtheta_norm = d_theta / (self.delta_theta / 2.0)

        # ρ: distância euclidiana normalizada ao centro do setor [0, 1]
        rho = np.sqrt(dr_norm**2 + dtheta_norm**2)
        rho = np.clip(rho, 0.0, 1.0)

        # φ: ângulo polar no espaço normalizado (adimensional, mesma unidade em ambos os eixos)
        if abs(dr_norm) < 1e-12 and abs(dtheta_norm) < 1e-12:
            phi = 0.0
        else:
            phi = np.arctan2(dtheta_norm, dr_norm)

        # --- Polinômios de Zernike Standard (n_zernike_modes=4) ---
        z1 = 1.0                    # Z_0^0: Piston (DC)
        z2 = rho * np.cos(phi)      # Z_1^1: Tilt-X (gradiente radial)
        z3 = rho * np.sin(phi)      # Z_1^{-1}: Tilt-Y (gradiente azimutal)
        z4 = 2.0 * rho**2 - 1.0    # Z_2^0: Defocus (curvatura local)

        return np.array([z1, z2, z3, z4], dtype=np.float64)

    def _get_sector_bounding_box(self, r_c: float, theta_mid: float) -> tuple:
        """Calcula o Bounding Box cartesiano de um setor polar para otimizar iteração."""
        r_start = r_c - self.delta_r / 2.0
        r_end = r_c + self.delta_r / 2.0
        t_start = theta_mid - self.delta_theta / 2.0
        t_end = theta_mid + self.delta_theta / 2.0
        
        pts = [
            (self.centerX + r_start * np.cos(t_start), self.centerY - r_start * np.sin(t_start)),
            (self.centerX + r_end * np.cos(t_start), self.centerY - r_end * np.sin(t_start)),
            (self.centerX + r_start * np.cos(t_end), self.centerY - r_start * np.sin(t_end)),
            (self.centerX + r_end * np.cos(t_end), self.centerY - r_end * np.sin(t_end))
        ]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        
        # Clip seguro: setores que excedem a imagem coletam apenas pixels disponíveis
        x_min = max(0, int(np.floor(min(xs))))
        x_max = min(self.width - 1, int(np.ceil(max(xs))))
        y_min = max(0, int(np.floor(min(ys))))
        y_max = min(self.height - 1, int(np.ceil(max(ys))))
        
        return x_min, x_max, y_min, y_max, r_start, r_end, t_start, t_end

    def _extract_sector(self, m: int, k: int) -> None:
        """
        Extrai iterativamente os coeficientes c_{k,m} gerando a identidade estrutural.
        Modifica diretamente self.V_tensor[m, k].
        """
        r_c = self.r_k_array[k]
        # theta_m_array já posiciona ângulos como centros de setor (θ_m = m·Δθ)
        # Não adicionar offset Δθ/2; a convenção do ResumoV5 define θ_m como centro.
        theta_mid = self.theta_m_array[m]
        
        x_min, x_max, y_min, y_max, r_start, r_end, t_start, t_end = self._get_sector_bounding_box(r_c, theta_mid)
        
        coeffs = np.zeros(self.n_zernike_modes, dtype=np.float64)
        pixel_count = 0
        
        for y in range(y_min, y_max + 1):
            for x in range(x_min, x_max + 1):
                dx = x - self.centerX
                dy = -(y - self.centerY) # Y computacional é invertido
                
                r_px = np.sqrt(dx**2 + dy**2)
                theta_px = np.arctan2(dy, dx)
                if theta_px < 0:
                    theta_px += 2 * np.pi
                    
                # Condição de Pertencimento ao Setor (com tratamento para Wrap-around)
                in_theta = False
                if t_end > 2 * np.pi:
                    if theta_px >= t_start or theta_px <= (t_end - 2 * np.pi):
                        in_theta = True
                else:
                    if t_start <= theta_px <= t_end:
                        in_theta = True
                        
                if r_start <= r_px <= r_end and in_theta:
                    z_vals = self._evaluate_local_polar_basis(r_px, theta_px, r_c, theta_mid)
                    intens = float(self.image[y, x])
                    coeffs += intens * z_vals
                    pixel_count += 1

        # T3 — Convenção de integração (M2 do parecer):
        # A Eq. 1a define c_q = ∬_Ω I·φ_q dxdy (integral de área).
        # Aqui dividimos por pixel_count para obter a DENSIDADE ESPECTRAL
        # média por unidade de área do setor (média por pixel). Esta escolha
        # é consistente com a normalização Jacobiana global (Eq. 1b/Eq. 2)
        # que repondera cada faixa radial por r_k, restaurando a proporcionalidade
        # à área física. A alternativa (integral bruta sem divisão) produziria
        # coeficientes proporcionais ao número de pixels do setor, requerendo
        # normalização adicional. A presente convenção é documentada como
        # desvio intencional da formulação literal da Eq. 1a.
        #
        # --- Complemento de Rigor Físico (Análise por-fração): ---
        # No reticulado discreto, pixel_count ∝ r_k (área do setor ∝ r_k·Δr·Δθ).
        # Sem a divisão por pixel_count, os coeficientes seriam c_q ∝ r_k.
        # No produto interno Jacobiano (Eq. 2), isso levaria a uma ponderação
        # proporcional a Σ r_k · (r_k)² = Σ r_k³, introduzindo um viés cúbico
        # que super-representaria a periferia. A divisão por pixel_count restaura
        # c_q como uma densidade (unidade-independente), permitindo que a
        # ponderação linear prescrita pelo Jacobiano r_k (Eq. 2) seja mantida.
        if pixel_count > 0:
            self.V_tensor[m, k, :] = coeffs / pixel_count

    def _apply_jacobian_normalization(self) -> None:
        """
        Delegação da normalização Jacobiana (Eq. 1b, Eq. 2) para lib_hilbert.
        Instancia HilbertSpace, aplica normalização e sincroniza o tensor.
        """
        hs = HilbertSpace(self.V_tensor, self.r_k_array)
        hs.normalize_jacobian()
        self.V_tensor = hs.V_tensor

    def execute_transposition(self) -> np.ndarray:
        """
        Orquestra a pipeline de extração e retorna o Tensor no domínio formal polínomial.
        """
        logging.info("[Extração Sensorial] Iniciando integração vetorial...")
        
        for m in range(self.n_angular_sectors):
            for k in range(self.n_radial_bands):
                self._extract_sector(m, k)
                
        # Imprescindível para a Bifurcação Espectral não ser dominada pelo tamanho cartesiano
        self._apply_jacobian_normalization()
        
        logging.info(
            f"[Extração Sensorial] Concluída. Tensor Resultante: "
            f"R^({self.n_angular_sectors} x {self.n_radial_bands} x {self.n_zernike_modes})"
        )
        return self.V_tensor