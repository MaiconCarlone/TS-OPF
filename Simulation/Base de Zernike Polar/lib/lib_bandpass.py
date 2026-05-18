import numpy as np
import logging

class BandPassFilter:
    """
    Lib 3: Filtragem Passa-Banda.
    Atende à Seção 5 do ResumoV5 (Eqs 9 a 12).
    Purificação Cruzada das Auto-anomalias. Extração da vital Completude Espectral (\eta_y).
    """
    def __init__(self, operator_tensors, inner_product_J_func):
        # Dicionário com {'O': tensor, 'S': tensor...}
        self.operator_tensors = operator_tensors
        self.inner_product_J = inner_product_J_func
        self.M = list(self.operator_tensors.values())[0].shape[0]

    def purify_and_evaluate(self):
        logging.info("[BandPassFilter] Efetuando fechamento de sistema: Purificação Cruzada...")
        results = {}
        
        for name, psi_tensor in self.operator_tensors.items():
            C_y = np.zeros((self.M, self.M))
            
            # Eq 9: Nova matriz de correlação subespaço (\hat{C}_y) ponderada
            for m1 in range(self.M):
                for m2 in range(m1, self.M):
                    ip = self.inner_product_J(psi_tensor[m1], psi_tensor[m2])
                    C_y[m1, m2] = ip
                    C_y[m2, m1] = ip
                    
            # Eq 10: Diagonalização -> Auto-Anomalias
            eigenvalues, eigenvectors = np.linalg.eigh(C_y)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Definição dinâmica do P_y (Qtd de Modos retidos pela Purificação Eq 11).
            # Para não amputarmos precocemente os harmônicos na FASE 1, reteremos baseado no 
            # Spectral Gap, caindo para o Fallback de 2% (M=360 -> ~ 7 modos) se estático.
            diffs = np.abs(np.diff(eigenvalues))
            
            # Buscando descontinuidade abrupta em 95% do espectro limpo
            # Se for ruído puramente estocástico (chão liso), manterá poucos.
            threshold_gap = np.max(diffs) * 0.1
            gaps = np.where(diffs > threshold_gap)[0]
            
            if len(gaps) > 0:
                P_y = gaps[0] + 1
            else:
                P_y = max(1, int(self.M * 0.02))
                
            # Forçar um teto e um piso para P_y razoável
            P_y = np.clip(P_y, 1, self.M // 4)
            
            # Eq 12: Regra da Soma / Completude Espectral (\eta_y)
            tr_total = np.sum(eigenvalues)
            tr_kept = np.sum(eigenvalues[:P_y])
            eta_y = (tr_kept / tr_total) if tr_total > 1e-12 else 1.0
            
            logging.info(f"[BandPassFilter] Lente [ {name} ] -> Modos P_y = {P_y} | Completude (η_y) = {eta_y:.4f}")
            
            results[name] = {
                'C_y': C_y,             # Matriz Bruta
                'eigenval': eigenvalues,# Todo o espectro de anomalias
                'eigenvec': eigenvectors,
                'P_y': P_y,             # Ponto de corte
                'eta_y': eta_y          # Completude
            }
            
        logging.info("[BandPassFilter] Regra de fechamento do sistema processada com sucesso.")
        return results
