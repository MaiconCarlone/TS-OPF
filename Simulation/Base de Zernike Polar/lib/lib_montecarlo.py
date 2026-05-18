# =============================================================================
# lib_montecarlo.py — Busca Monte Carlo do Centro de Simetria (D1)
# =============================================================================
# Implementação estática (funções puras) do critério D1 do ResumoV5:
#
#   c* = argmin_{c ∈ Ω} Var{C_{m,m'}(c)}_{m ≠ m'}
#
# Minimiza a variância dos elementos off-diagonal da matriz de correlação
# Jacobiana, maximizando a isonomia angular — critério auto-consistente.
#
# Otimizado com numba (@njit) para viabilizar a avaliação de ~100 candidatos
# (cada candidato requer transposição completa + correlação:
#  O(n_angular_sectors² × n_radial_bands + n_angular_sectors³)).
#
# Busca bifásica: Fase 1 (grossa) + Fase 2 (fina).
# Parâmetros de entrada: centro inicial, zona de busca, setores MC, max_iter.
#
# Dependências: numpy, numba, logging (logging apenas fora das funções JIT)
# Referência: ResumoV5.md — Seção D1
# =============================================================================

import numpy as np
import logging
from numba import njit, prange


# =============================================================================
# Kernel JIT: Avaliação de um candidato a centro
# =============================================================================

@njit(cache=True)
def _zernike_basis(r_px, theta_px, r_c, theta_mid, half_dr, half_dtheta):
    """
    Avalia os 4 polinômios de Zernike Standard no ponto (r_px, θ_px)
    relativo ao centro do setor (r_c, θ_mid).

    Coordenadas normalizadas (ambas adimensionais, mesma escala):
      u = (r_px - r_c) / half_dr       → radial   ∈ [-1, 1]
      v = (θ_px - θ_mid) / half_dtheta → angular  ∈ [-1, 1]
    """
    # Coordenadas normalizadas ao centro do setor
    d_theta = (theta_px - theta_mid + np.pi) % (2.0 * np.pi) - np.pi
    dr_norm = (r_px - r_c) / half_dr
    dtheta_norm = d_theta / half_dtheta

    # ρ: distância euclidiana normalizada ao centro do setor [0, 1]
    rho_sq = dr_norm * dr_norm + dtheta_norm * dtheta_norm
    if rho_sq > 1.0:
        rho_sq = 1.0
    rho = np.sqrt(rho_sq)

    # φ: ângulo polar no espaço normalizado (adimensional, mesma unidade em ambos os eixos)
    if abs(dr_norm) < 1e-12 and abs(dtheta_norm) < 1e-12:
        phi = 0.0
    else:
        phi = np.arctan2(dtheta_norm, dr_norm)

    # Zernike Standard (n_zernike_modes=4)
    z1 = 1.0                    # Z_0^0: Piston
    z2 = rho * np.cos(phi)      # Z_1^1: Tilt-X
    z3 = rho * np.sin(phi)      # Z_1^{-1}: Tilt-Y
    z4 = 2.0 * rho**2 - 1.0    # Z_2^0: Defocus

    return z1, z2, z3, z4


@njit(cache=True)
def _evaluate_candidate(image, cx, cy, n_sectors_search, delta_r, n_zernike_modes):
    """
    Para um centro candidato (cx, cy), executa:
    1. Estabelecimento da malha polar (r_min, R_max, n_radial_bands, r_k_array, θ_m_array)
    2. Extração de setores (transposição)
    3. Normalização Jacobiana
    4. Construção da matriz de correlação C_z
    5. Cálculo de Var{C_{m,m'}}_{m ≠ m'}

    Retorna o custo (variância off-diagonal). Valor menor = centro melhor.

    Parameters:
        image: imagem 2D (H, W) em float64
        cx, cy: centro candidato (coordenadas de pixel)
        n_sectors_search: número de setores angulares para esta avaliação
        delta_r: espessura radial das faixas
        n_zernike_modes: número de modos de Zernike (fixo = 4)

    Returns:
        float: custo = Var{C_{m,m'}}_{m ≠ m'}. -1.0 se candidato inválido.
    """
    height = image.shape[0]
    width = image.shape[1]

    # --- Malha Polar ---
    R_max = min(cx, width - cx, cy, height - cy)
    if R_max < 10.0:
        return -1.0  # candidato fora de limites viáveis

    delta_theta = 2.0 * np.pi / n_sectors_search
    half_dr = delta_r / 2.0
    half_dtheta = delta_theta / 2.0

    # Condição de Shannon/Nyquist: área mínima do setor >= 1 pixel
    r_min = 1.0
    while r_min * delta_r * delta_theta < 1.0:
        r_min += 1.0

    n_radial_bands = int(np.floor((R_max - r_min) / delta_r))
    if n_radial_bands < 2:
        return -1.0  # sem faixas radiais suficientes

    # Arrays de coordenadas
    r_k_array = np.empty(n_radial_bands)
    for k in range(n_radial_bands):
        r_k_array[k] = r_min + (k + 0.5) * delta_r

    theta_m_array = np.empty(n_sectors_search)
    for m in range(n_sectors_search):
        theta_m_array[m] = m * delta_theta

    # --- Transposição: Extração de Setores ---
    V_tensor = np.zeros((n_sectors_search, n_radial_bands, n_zernike_modes))

    for m in range(n_sectors_search):
        theta_mid = theta_m_array[m]
        for k in range(n_radial_bands):
            r_c = r_k_array[k]
            r_start = r_c - half_dr
            r_end = r_c + half_dr
            t_start = theta_mid - delta_theta / 2.0
            t_end = theta_mid + delta_theta / 2.0

            # Bounding box cartesiano simplificado
            # (conservador: usa r_end como raio máximo)
            x_min_bb = max(0, int(cx - r_end - 1))
            x_max_bb = min(width - 1, int(cx + r_end + 1))
            y_min_bb = max(0, int(cy - r_end - 1))
            y_max_bb = min(height - 1, int(cy + r_end + 1))

            c0 = 0.0
            c1 = 0.0
            c2 = 0.0
            c3 = 0.0
            px_count = 0

            for y in range(y_min_bb, y_max_bb + 1):
                for x in range(x_min_bb, x_max_bb + 1):
                    dx = float(x - cx)
                    dy = float(-(y - cy))

                    r_px = np.sqrt(dx * dx + dy * dy)
                    if r_px < r_start or r_px > r_end:
                        continue

                    theta_px = np.arctan2(dy, dx)
                    if theta_px < 0.0:
                        theta_px += 2.0 * np.pi

                    # Teste de pertencimento angular (com wrap-around)
                    in_theta = False
                    if t_end > 2.0 * np.pi:
                        if theta_px >= t_start or theta_px <= (t_end - 2.0 * np.pi):
                            in_theta = True
                    else:
                        if t_start <= theta_px <= t_end:
                            in_theta = True

                    if not in_theta:
                        continue

                    z1, z2, z3, z4 = _zernike_basis(r_px, theta_px, r_c, theta_mid, half_dr, half_dtheta)
                    intens = image[y, x]
                    c0 += intens * z1
                    c1 += intens * z2
                    c2 += intens * z3
                    c3 += intens * z4
                    px_count += 1

            if px_count > 0:
                inv_count = 1.0 / px_count
                V_tensor[m, k, 0] = c0 * inv_count
                V_tensor[m, k, 1] = c1 * inv_count
                V_tensor[m, k, 2] = c2 * inv_count
                V_tensor[m, k, 3] = c3 * inv_count

    # --- Normalização Jacobiana ---
    for m in range(n_sectors_search):
        norm_sq = 0.0
        for k in range(n_radial_bands):
            sq = 0.0
            for q in range(n_zernike_modes):
                sq += V_tensor[m, k, q] * V_tensor[m, k, q]
            norm_sq += r_k_array[k] * sq
        if norm_sq > 0.0:
            inv_norm = 1.0 / np.sqrt(norm_sq)
            for k in range(n_radial_bands):
                for q in range(n_zernike_modes):
                    V_tensor[m, k, q] *= inv_norm

    # --- Matriz de Correlação C_z (Eq. 3) ---
    C_matrix = np.zeros((n_sectors_search, n_sectors_search))
    for m1 in range(n_sectors_search):
        for m2 in range(m1, n_sectors_search):
            ip = 0.0
            for k in range(n_radial_bands):
                dot_per_mode = 0.0
                for q in range(n_zernike_modes):
                    dot_per_mode += V_tensor[m1, k, q] * V_tensor[m2, k, q]
                ip += r_k_array[k] * dot_per_mode
            C_matrix[m1, m2] = ip
            C_matrix[m2, m1] = ip

    # --- Custo: Var{C_{m,m'}}_{m ≠ m'} ---
    n_off = n_sectors_search * (n_sectors_search - 1)  # pares off-diagonal (inclui ambos)
    if n_off == 0:
        return -1.0

    sum_off = 0.0
    sum_sq_off = 0.0
    for m1 in range(n_sectors_search):
        for m2 in range(n_sectors_search):
            if m1 != m2:
                val = C_matrix[m1, m2]
                sum_off += val
                sum_sq_off += val * val

    mean_off = sum_off / n_off
    var_off = sum_sq_off / n_off - mean_off * mean_off

    # CV² (Coeficiente de Variação ao Quadrado): invariante à escala,
    # penaliza centros em regiões de fundo vazio onde μ → 0.
    # Preserva a ordenação entre candidatos não-degenerados (μ_A ≈ μ_B).
    if abs(mean_off) < 1e-12:
        return 1e9  # Penalidade: matriz degenerada ou nula

    cv_sq = var_off / (mean_off * mean_off)
    return cv_sq


# =============================================================================
# Orquestração bifásica (chamada do Python, loga progresso)
# =============================================================================

def monte_carlo_center_search(
    image: np.ndarray,
    center_init: tuple,
    search_radius: float,
    M_search_coarse: int = 36,
    M_search_fine: int = 72,
    delta_r: float = 2.0,
    n_zernike_modes: int = 4,
    max_iter_phase1: int = 100,
    max_iter_phase2: int = 100
) -> dict:
    """
    Busca Monte Carlo bifásica do centro ótimo de simetria (D1, ResumoV5).

    Fase 1 (Grossa): amostra candidatos uniformemente em
        [cx_init ± search_radius] × [cy_init ± search_radius].
        Usa M_search_coarse setores para avaliação rápida.
        Posições inteiras são deduplicadas para evitar reavaliações.

    Fase 2 (Fina): varredura exaustiva de TODAS as posições inteiras
        em ±2px do melhor da Fase 1 (máximo 25 candidatos), com
        M_search_fine setores para precisão máxima.

    Parameters:
        image (ndarray): Imagem 2D (H, W), float64, escala de cinza.
        center_init (tuple): Centro inicial (cx, cy) em pixels.
        search_radius (float): Raio da zona de busca (pixels) para Fase 1.
        M_search_coarse (int): Número de setores angulares na Fase 1.
        M_search_fine (int): Número de setores angulares na Fase 2.
        delta_r (float): Espessura radial das faixas.
        n_zernike_modes (int): Número de modos de Zernike (padrão: 4).
        max_iter_phase1 (int): Tentativas máximas na Fase 1.
        max_iter_phase2 (int): (Ignorado na Fase 2 — varredura exaustiva.)

    Returns:
        dict: {
            'center': (cx, cy) ótimo,
            'cost': custo final,
            'history_phase1': lista de (cx, cy, custo) da fase 1,
            'history_phase2': lista de (cx, cy, custo) da fase 2,
            'best_phase1': (cx, cy, custo) melhor candidato da fase 1
        }

    Referência: ResumoV5.md — Seção D1.
    """
    image_f64 = image.astype(np.float64)
    cx_init, cy_init = center_init

    # =================================================================
    # FASE 1: Varredura Grossa (Monte Carlo)
    # =================================================================
    logging.info(
        f"[MC D1] Fase 1 (Grossa): até {max_iter_phase1} candidatos, "
        f"n_sectors={M_search_coarse}, zona=±{search_radius:.0f}px em torno de ({cx_init}, {cy_init})."
    )

    # Pré-compilação JIT: executar uma vez fora do loop para forçar compilação
    _ = _evaluate_candidate(image_f64, cx_init, cy_init, M_search_coarse, delta_r, n_zernike_modes)
    logging.info("[MC D1] JIT compilado. Iniciando varredura...")

    best_cost_p1 = np.inf
    best_center_p1 = (cx_init, cy_init)
    history_p1 = []

    # Gerar candidatos e deduplicar posições inteiras
    np.random.seed(42)  # Reprodutibilidade
    candidates_x_raw = np.random.uniform(cx_init - search_radius, cx_init + search_radius, max_iter_phase1)
    candidates_y_raw = np.random.uniform(cy_init - search_radius, cy_init + search_radius, max_iter_phase1)

    # Deduplicar posições inteiras para não reavaliar o mesmo pixel
    seen_positions = set()
    candidates = []
    for i in range(max_iter_phase1):
        cx_cand = int(round(candidates_x_raw[i]))
        cy_cand = int(round(candidates_y_raw[i]))
        key = (cx_cand, cy_cand)
        if key not in seen_positions:
            seen_positions.add(key)
            candidates.append(key)

    n_unique = len(candidates)
    logging.info(f"[MC D1] Fase 1: {n_unique} posições únicas de {max_iter_phase1} amostras.")

    report_interval = max(1, n_unique // 10)

    for i, (cx_cand, cy_cand) in enumerate(candidates):
        cost = _evaluate_candidate(image_f64, cx_cand, cy_cand, M_search_coarse, delta_r, n_zernike_modes)

        if cost < 0.0:
            continue  # candidato inválido

        history_p1.append((cx_cand, cy_cand, cost))

        improved = ""
        if cost < best_cost_p1:
            best_cost_p1 = cost
            best_center_p1 = (cx_cand, cy_cand)
            improved = " ★"

        # Progresso a cada 10%
        if (i + 1) % report_interval == 0:
            pct = 100.0 * (i + 1) / n_unique
            logging.info(
                f"[MC D1] Fase 1: {pct:.0f}% ({i+1}/{n_unique}) — "
                f"Atual: ({cx_cand}, {cy_cand}) custo={cost:.6e} | "
                f"Melhor: ({best_center_p1[0]}, {best_center_p1[1]}) custo={best_cost_p1:.6e}{improved}"
            )

    logging.info(
        f"[MC D1] Fase 1 concluída. Melhor: ({best_center_p1[0]}, {best_center_p1[1]}), "
        f"Custo: {best_cost_p1:.6e}. Candidatos válidos: {len(history_p1)}/{n_unique}."
    )

    # =================================================================
    # FASE 2: Varredura Fina (Exaustiva)
    # =================================================================
    fine_radius = 2  # ±2px → varredura exaustiva de 5×5 = 25 posições
    cx_p1, cy_p1 = best_center_p1

    # Gerar TODAS as posições inteiras na zona fina
    candidates_p2 = []
    for dx in range(-fine_radius, fine_radius + 1):
        for dy in range(-fine_radius, fine_radius + 1):
            candidates_p2.append((cx_p1 + dx, cy_p1 + dy))

    n_p2 = len(candidates_p2)
    logging.info(
        f"[MC D1] Fase 2 (Fina, exaustiva): {n_p2} posições, "
        f"n_sectors={M_search_fine}, zona=±{fine_radius}px em torno de ({cx_p1}, {cy_p1})."
    )

    best_cost_p2 = best_cost_p1
    best_center_p2 = best_center_p1
    history_p2 = []

    report_interval_f = max(1, n_p2 // 10)

    for i, (cx_cand, cy_cand) in enumerate(candidates_p2):
        cost = _evaluate_candidate(image_f64, cx_cand, cy_cand, M_search_fine, delta_r, n_zernike_modes)

        if cost < 0.0:
            continue

        history_p2.append((cx_cand, cy_cand, cost))

        improved = ""
        if cost < best_cost_p2:
            best_cost_p2 = cost
            best_center_p2 = (cx_cand, cy_cand)
            improved = " ★"

        # Progresso a cada ~10%
        if (i + 1) % report_interval_f == 0 or i == n_p2 - 1:
            pct = 100.0 * (i + 1) / n_p2
            logging.info(
                f"[MC D1] Fase 2: {pct:.0f}% ({i+1}/{n_p2}) — "
                f"Atual: ({cx_cand}, {cy_cand}) custo={cost:.6e} | "
                f"Melhor: ({best_center_p2[0]}, {best_center_p2[1]}) custo={best_cost_p2:.6e}{improved}"
            )

    logging.info(
        f"[MC D1] Fase 2 concluída. Centro ótimo: ({best_center_p2[0]}, {best_center_p2[1]}), "
        f"Custo final: {best_cost_p2:.6e}. "
        f"Deslocamento total: Δx={best_center_p2[0] - cx_init}, Δy={best_center_p2[1] - cy_init}."
    )

    return {
        'center': best_center_p2,
        'cost': best_cost_p2,
        'history_phase1': history_p1,
        'history_phase2': history_p2,
        'best_phase1': (*best_center_p1, best_cost_p1)
    }
