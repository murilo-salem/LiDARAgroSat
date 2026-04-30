Arquitetura geral
Nuvem de pontos normalizada
        ↓
Gerador de propostas (local maxima, CHM ou ambos)
        ↓
Para cada proposta: extrair vizinhança esférica de raio R
        ↓
PointNet local → [é árvore? / não é?] + [offset XY refinado]
        ↓
NMS 3D para remover duplicatas
        ↓
Matching + métricas

Fase 1 — Gerador de propostas
Use o baseline do paper como gerador de propostas com janela pequena (window_size = 0.5–1.0 m). Isso dá recall alto (~86%) com precisão baixa (~10%) — exatamente o que você quer: muitos candidatos, poucos perdidos. A rede vai filtrar os falsos positivos.
pythonproposals = local_maxima_filter(points_xyz, window_size=0.8, height_threshold=3.0)
# Espera-se ~5x mais candidatos que árvores reais

Fase 2 — Extração de vizinhanças locais
Para cada proposta, extraia uma esfera de pontos ao redor dela. O raio deve cobrir uma copa média — o paper reporta distância média de matching de ~0.89 m, então R = 2–3 m é razoável. Normalize as coordenadas em relação ao centro da proposta.
pythonfrom scipy.spatial import KDTree
import numpy as np

def extract_neighborhood(points_xyz, center, radius=2.5, n_points=512):
    tree = KDTree(points_xyz)
    idxs = tree.query_ball_point(center, r=radius)
    pts = points_xyz[idxs] - center  # centraliza em (0,0,0)
    
    if len(pts) == 0:
        return np.zeros((n_points, 3))
    
    # subsample ou pad para tamanho fixo
    if len(pts) >= n_points:
        choice = np.random.choice(len(pts), n_points, replace=False)
    else:
        choice = np.random.choice(len(pts), n_points, replace=True)
    
    return pts[choice]

Fase 3 — Labels de treinamento
Para cada proposta, rode o matching do paper contra o ground truth. Isso define:

Positivo (1): proposta matched a uma árvore GT (dist 2D ≤ 2 m)
Negativo (0): proposta sem match

Também extraia o offset XY da proposta até a árvore GT correspondente (para o cabeçalho de regressão).
pythondef label_proposals(proposals, ground_truth_xy, pos_threshold=2.0):
    from scipy.spatial.distance import cdist
    
    dists = cdist(proposals[:, :2], ground_truth_xy)
    labels = np.zeros(len(proposals), dtype=int)
    offsets = np.zeros((len(proposals), 2))
    
    for i, row in enumerate(dists):
        min_j = np.argmin(row)
        if row[min_j] <= pos_threshold:
            labels[i] = 1
            offsets[i] = ground_truth_xy[min_j] - proposals[i, :2]
    
    return labels, offsets

Fase 4 — Arquitetura PointNet local
A rede é simples: shared MLP nos pontos → max pooling global → MLP de classificação + regressão. Sem message passing entre vizinhanças, sem hierarquia — PointNet puro aplicado localmente.
pythonimport torch
import torch.nn as nn

class LocalPointNet(nn.Module):
    def __init__(self, n_points=512):
        super().__init__()
        
        # Shared MLP por ponto (equivale a Conv1d)
        self.point_mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
        )
        
        # Global feature após max pooling → 256-dim
        self.global_mlp = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
        )
        
        # Cabeça de classificação: é árvore ou não?
        self.cls_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)  # logit
        )
        
        # Cabeça de regressão: offset XY até o centro real
        self.reg_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2)  # delta_x, delta_y
        )
    
    def forward(self, x):
        # x: (B, N, 3) → transpor para (B, 3, N) para Conv1d
        x = x.transpose(1, 2)
        x = self.point_mlp(x)           # (B, 256, N)
        x = x.max(dim=2).values         # max pooling global → (B, 256)
        x = self.global_mlp(x)          # (B, 64)
        
        cls = self.cls_head(x).squeeze(-1)   # (B,) logit
        reg = self.reg_head(x)               # (B, 2) offset
        return cls, reg

Fase 5 — Treinamento
O dataset tem 3600 árvores em 10 parcelas. Use 8 parcelas para treino e 2 para validação (o paper lista plots 1–10 com stats individuais — escolha as 2 de maior diversidade como val, ex: plot 1 decíduo e plot 6 conífero).
python# Loss combinada: classificação + regressão apenas nos positivos
def combined_loss(cls_logits, reg_pred, labels, offsets, alpha=1.0):
    cls_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([5.0])  # compensa desbalanceamento ~10:1
    )(cls_logits, labels.float())
    
    mask = labels == 1
    if mask.sum() > 0:
        reg_loss = nn.SmoothL1Loss()(reg_pred[mask], offsets[mask])
    else:
        reg_loss = torch.tensor(0.0)
    
    return cls_loss + alpha * reg_loss
O desbalanceamento é real: com window_size = 0.8, você terá ~8–10 propostas por árvore real, logo ~90% dos exemplos são negativos. O pos_weight=5.0 compensa parcialmente — ajuste empiricamente.
Data augmentation importante para nuvens de pontos:

Rotação aleatória em Z (a copa é aproximadamente simétrica)
Jitter gaussiano leve nos pontos (simula erro do sensor)
Subsampling aleatório diferente a cada epoch (já coberto pela extração)


Fase 6 — Inferência e NMS
pythondef predict_and_refine(model, proposals, points_xyz, radius=2.5, n_points=512, 
                        cls_threshold=0.5, nms_radius=1.5):
    model.eval()
    neighborhoods = np.stack([
        extract_neighborhood(points_xyz, p, radius, n_points) 
        for p in proposals
    ])
    
    with torch.no_grad():
        x = torch.tensor(neighborhoods, dtype=torch.float32)
        cls_logits, reg_offsets = model(x)
        scores = torch.sigmoid(cls_logits).numpy()
        offsets = reg_offsets.numpy()
    
    # Aplicar offset de regressão às propostas
    refined = proposals[:, :2] + offsets
    
    # Filtrar por score
    keep = scores > cls_threshold
    refined = refined[keep]
    scores_kept = scores[keep]
    
    # NMS 2D: remove duplicatas próximas, mantém maior score
    return nms_2d(refined, scores_kept, radius=nms_radius)


def nms_2d(points, scores, radius=1.5):
    order = np.argsort(-scores)
    kept = []
    suppressed = set()
    
    tree = KDTree(points)
    for i in order:
        if i in suppressed:
            continue
        kept.append(i)
        neighbors = tree.query_ball_point(points[i], r=radius)
        suppressed.update(neighbors)
    
    return points[kept]

Expectativa de F1 e pontos críticos
Com esta arquitetura implementada fielmente:
ComponenteImpacto no F1Proposta com window 0.8 mrecall base ~0.86 — teto da pipelinePointNet classificadorremove ~80% dos FP → precision ~0.55–0.65Regressão de offsetreduz erro de distância, melhora matchingNMS bem calibradoevita double-counting
F1 esperado: 0.65–0.75, dependendo principalmente de quantas parcelas ficam no treino e da qualidade do balanceamento. O teto real é o recall do gerador de propostas — se o gerador perde uma árvore, a rede nunca a recupera.
O ponto mais frágil da implementação é o NMS: raio muito pequeno → duplicatas no resultado; raio muito grande → merge de árvores próximas. Calibre com as parcelas de validação antes de reportar métricas finais.