# Relatorio Completo de Runs e Variacoes

## 1. Escopo

Este arquivo consolida todos os experimentos efetivamente executados no repositório, cobrindo:

- pipeline principal multimodal `UAV LiDAR + RGB`;
- baselines e variantes fieis ao paper;
- variantes deep learning baseadas em nuvem de pontos;
- híbridos geométricos baseados em `watershed`, `local maxima` e propostas.

Importante: nem todos os números são diretamente comparáveis entre si, porque os protocolos mudam entre famílias:

- `LOPO` curto;
- `basic split` com treino longo;
- suíte de `3` splits;
- validação simples em `3` seeds;
- benchmark fiel “treina em tudo e mede em tudo” para métodos sem treino;
- validação fixa em `plot_01` e `plot_06` para PointNet estrito.

## 2. Resumo Executivo

### 2.1 Melhor resultado por família

| Família | Melhor método | Protocolo | Resultado principal |
|---|---|---|---:|
| Pipeline multimodal principal | `baseline_local_maxima` | `LOPO` prático | `F1 = 0.4063` |
| Pipeline multimodal principal | `baseline_local_maxima` | `basic split` longo | `F1 = 0.3725` |
| Híbridos multimodais comparáveis | `baseline_local_maxima` | suíte `3` splits | `F1 = 0.3785` |
| Melhor deep multimodal do projeto | `hybrid_proposal_refinement_multimodal_v2` | suíte `3` splits | `F1 = 0.3278` |
| Baseline fiel do paper | `local_maxima` otimizado por `Optuna` | `10` parcelas | `F1 = 0.4393` |
| Melhor baseline fiel clássico | `CHM + watershed` | `10` parcelas | `F1 = 0.5354` |
| Melhor fusão fiel sem deep pesado | `LiDAR + RGB object fusion` | `10` parcelas | `F1 = 0.5164` |
| Melhor PointNet fiel puro | `strict_focal_xyz` | validação `plot_01 + plot_06` | `F1 = 0.5505` |
| Híbrido `watershed + proposals` | `hybrid_watershed_proposals` | validação `plot_01 + plot_06` | `F1 = 0.5593` |

### 2.2 Tendências robustas

- No pipeline principal multimodal, a baseline clássica permaneceu como melhor método.
- O melhor deep do projeto principal foi o eixo `proposal + refinement`, em especial `Hybrid v2`.
- No benchmark fiel, `CHM + watershed` foi o melhor método global executado nas `10` parcelas.
- No PointNet estrito, o maior ganho veio de `threshold + NMS`; `Focal Loss` ajudou pouco além disso; `features multicanal` não ajudaram.
- O híbrido `watershed + proposals` ficou competitivo, mas não superou o `watershed` puro.

## 3. Pipeline Principal Multimodal

### 3.1 LOPO prático

**Objetivo**  
Validar o pipeline completo em `LOPO` com orçamento curto.

**Protocolo**  
`10` folds `LOPO`, perfil prático com `epochs=20`, `patience=5`.

**Métodos testados**  
`baseline_local_maxima`, `lidar_only`, `rgb_only`, `early_fusion`, `multimodal_mid_fusion`, `multimodal_late_fusion`.

**Resultados agregados**  
Fonte: `outputs/metrics/method_summary.csv`

| Método | Precision | Recall | F1 |
|---|---:|---:|---:|
| `baseline_local_maxima` | 0.3557 | 0.4889 | 0.4063 |
| `multimodal_mid_fusion` | 0.1243 | 0.3071 | 0.1539 |
| `multimodal_late_fusion` | 0.1044 | 0.2640 | 0.1455 |
| `early_fusion` | 0.0941 | 0.2974 | 0.1372 |
| `rgb_only` | 0.1010 | 0.2270 | 0.1343 |
| `lidar_only` | 0.1487 | 0.2662 | 0.0883 |

**Leitura**  
Útil como validação de pipeline, mas não como veredito científico final. A baseline foi claramente superior; `multimodal_mid_fusion` foi o melhor deep.

### 3.2 Basic split longo

**Objetivo**  
Rodar treino longo em um split único por parcela.

**Protocolo**  
Treino: `plot_04,01,08,09,07,02,06,03`; validação: `plot_10`; teste: `plot_05`; `120` épocas nominais com `early stopping`.

**Resultados em `plot_05`**  
Fonte: `outputs_basic_split/metrics/method_summary.csv`

| Método | Precision | Recall | F1 |
|---|---:|---:|---:|
| `baseline_local_maxima` | 0.2729 | 0.5865 | 0.3725 |
| `multimodal_mid_fusion` | 0.2590 | 0.3798 | 0.3080 |
| `early_fusion` | 0.0659 | 0.4904 | 0.1162 |
| `rgb_only` | 0.0401 | 0.2452 | 0.0690 |
| `multimodal_late_fusion` | 0.0346 | 0.1683 | 0.0574 |
| `lidar_only` | 0.0008 | 0.4856 | 0.0016 |

**Leitura**  
`multimodal_mid_fusion` melhorou muito com treino longo, mas ainda não passou a baseline. `lidar_only` colapsou por `FP`.

### 3.3 Ablação diagnóstica de decode e regime de treino

**Objetivo**  
Separar efeito de `decode tuning` do efeito do novo regime de treino.

**Protocolo**  
Mesmo `basic split` canônico. Variantes: `multimodal_mid_fusion_redecode`, `multimodal_mid_fusion_roi`, `lidar_only_redecode`, `lidar_only_roi`.

**Resultados em `plot_05`**

| Método | Precision | Recall | F1 |
|---|---:|---:|---:|
| `baseline_local_maxima` | 0.2729 | 0.5865 | 0.3725 |
| `multimodal_mid_fusion` | 0.2590 | 0.3798 | 0.3080 |
| `multimodal_mid_fusion_redecode` | 0.2211 | 0.5433 | 0.3143 |
| `multimodal_mid_fusion_roi` | 0.1254 | 0.5577 | 0.2048 |
| `lidar_only` | 0.0008 | 0.4856 | 0.0016 |
| `lidar_only_redecode` | 0.1022 | 0.2885 | 0.1509 |
| `lidar_only_roi` | 0.1789 | 0.4567 | 0.2571 |

**Leitura**  
O maior ganho veio de `decode tuning`, sobretudo em `lidar_only`. O novo regime de treino não melhorou o `mid_fusion` no teste.

### 3.4 Extensão completa do detector e híbrido inicial

**Objetivo**  
Testar `hard negatives`, `peakness`, `species head`, `deep supervision`, `adaptive NMS` e um refinador híbrido por propostas.

**Resultados em `plot_05`**

| Método | Precision | Recall | F1 |
|---|---:|---:|---:|
| `baseline_local_maxima` | 0.2729 | 0.5865 | 0.3725 |
| `multimodal_mid_fusion_redecode` | 0.2211 | 0.5433 | 0.3143 |
| `multimodal_mid_fusion_full` | 0.2007 | 0.5433 | 0.2931 |
| `hybrid_proposal_refinement_multimodal` | 0.1419 | 0.6394 | 0.2323 |

**Leitura**  
O detector denso completo elevou `recall`, mas não converteu isso em melhor `F1`. O híbrido mostrou potencial de cobertura, mas ainda com `FP` alto.

### 3.5 Hybrid v2 em três basic splits

**Objetivo**  
Medir se `proposal + refinement` melhora de forma consistente sobre `mid_fusion`.

**Protocolo**  
Suíte em `3` seeds/splits comparáveis.

**Resultados agregados**  
Fonte: `outputs_suite_aggregate/suite_method_summary.csv`

| Método | Precision mean | Recall mean | F1 mean |
|---|---:|---:|---:|
| `baseline_local_maxima` | 0.3410 | 0.4797 | 0.3785 |
| `hybrid_proposal_refinement_multimodal_v2` | 0.2930 | 0.3933 | 0.3278 |
| `multimodal_mid_fusion` | 0.2059 | 0.3634 | 0.2493 |

**Leitura**  
Primeira extensão que superou `mid_fusion` em `3/3` splits. Ainda abaixo da baseline, mas foi o melhor deep robusto do projeto principal.

### 3.6 Baseline como estágio 1 puro

**Objetivo**  
Usar apenas propostas de `CHM local maxima` no estágio 1, sem proposal net aprendida.

**Resultado em `plot_05`**

| Método | Precision | Recall | F1 |
|---|---:|---:|---:|
| `baseline_local_maxima` | 0.2729 | 0.5865 | 0.3725 |
| `hybrid_baseline_stage_refiner` | 0.1762 | 0.6394 | 0.2762 |

**Leitura**  
Subiu muito `recall`, mas perdeu muito em `precision`. Ficou abaixo de `Hybrid v2`.

### 3.7 Detector tri-modal `RGB + LiDAR + CHM`

**Objetivo**  
Testar ramo explícito de `CHM` e supervisão geométrica enriquecida.

**Resultados agregados**  
Fonte: `outputs_chm_context_aggregate/suite_method_summary.csv`

| Método | Precision mean | Recall mean | F1 mean |
|---|---:|---:|---:|
| `baseline_local_maxima` | 0.3410 | 0.4797 | 0.3785 |
| `multimodal_chm_context_interactive` | 0.1620 | 0.3839 | 0.1943 |

**Leitura**  
Não melhorou o benchmark. Reforçou empiricamente que o caminho mais promissor continua sendo o híbrido por propostas.

### 3.8 Validação simples de desenvolvimento

**Objetivo**  
Trocar `LOPO` por uma validação mais barata em `3` seeds fixos para desenvolvimento rápido.

**Resultados**  
Fonte: `outputs_dev_validation_simple/validation_summary.csv`

| Método | Precision mean | Recall mean | F1 mean |
|---|---:|---:|---:|
| `baseline_local_maxima` | 0.3698 | 0.4896 | 0.4153 |
| `hybrid_baseline_stage_refiner` | 0.2428 | 0.6017 | 0.3438 |
| `hybrid_proposal_refinement_multimodal_v2` | 0.3166 | 0.3618 | 0.3331 |
| `multimodal_mid_fusion` | 0.2493 | 0.4053 | 0.3066 |
| `multimodal_chm_context_interactive` | 0.1902 | 0.3861 | 0.2394 |

**Leitura**  
Boa para triagem. A baseline venceu `3/3`. Entre os deep, `hybrid_baseline_stage_refiner` teve maior `F1`.

### 3.9 Validação simples com pré-processamento explícito

**Objetivo**  
Medir efeito de um pré-processamento explicitamente reforçado: co-registro, referência saneada, `CHM features`.

**Resultados**  
Fonte: `outputs_dev_validation_preproc_explicit/validation_summary.csv`

| Método | Precision mean | Recall mean | F1 mean |
|---|---:|---:|---:|
| `baseline_local_maxima` | 0.3698 | 0.4896 | 0.4153 |
| `hybrid_baseline_stage_refiner` | 0.2441 | 0.5799 | 0.3422 |
| `hybrid_proposal_refinement_multimodal_v2` | 0.2612 | 0.4707 | 0.3343 |
| `multimodal_mid_fusion` | 0.2579 | 0.4445 | 0.3235 |
| `multimodal_chm_context_interactive` | 0.2325 | 0.1809 | 0.1407 |

**Leitura**  
O pré-processamento explícito ajudou mais o `multimodal_mid_fusion` do que as variantes híbridas.

### 3.10 Auditoria centro↔ápice e pseudo-rótulos

**Objetivo**  
Medir ruído geométrico entre centro de referência e ápice do dossel, e gerar pseudo-rótulos por concordância.

**Resultado principal**  
Deslocamento centro↔ápice:

- média: `0.8537 m`
- mediana: `0.9019 m`
- `P90 = 1.2298 m`

**Leitura**  
Há ruído geométrico relevante entre rótulo de campo e ápice do dossel. Isso explica parte da competitividade dos métodos geométricos.

## 4. Benchmark Fiel ao Paper

### 4.1 Baseline canônico fiel

**Objetivo**  
Reproduzir o baseline do paper/Kaggle diretamente sobre a nuvem `LAS`, sem RGB.

**Configuração canônica**

- `window_size = 2.0 m`
- `height_threshold = 3.0 m`
- `max_dist = 5.0 m`
- `max_height_diff = 3.0 m`

**Resultado**

- `Precision = 0.2967`
- `Recall = 0.5830`
- `F1 = 0.3933`

**Sweep manual**  
Melhor janela observada: `2.5 m`, `F1 = 0.4273`.

### 4.2 Baseline fiel otimizado com Optuna

**Objetivo**  
Otimizar `window_size` e `height_threshold`.

**Melhor trial**

- `window_size_m = 2.4`
- `height_threshold_m = 5.0`

**Resultado**

- `Precision = 0.4253`
- `Recall = 0.4542`
- `F1 = 0.4393`

**Leitura**  
Melhora clara sobre o caso canônico e sobre o sweep manual.

### 4.3 Local maxima com janela adaptativa

**Objetivo**  
Substituir a janela fixa por `window_size(z) = a*z + b`.

**Melhor configuração**

- `a = 0.035`
- `b = 2.5`
- `height_threshold = 5.0`

**Resultado**

- `Precision = 0.4553`
- `Recall = 0.5755`
- `F1 = 0.5084`

**Leitura**  
Melhora `9/10` parcelas e ganha muito em `recall`. Ainda abaixo do `watershed`.

### 4.4 CHM + watershed

**Objetivo**  
Trocar `local maxima` puro por segmentação de copas em `CHM`.

**Melhor configuração**

- `resolution_m = 0.1`
- `gaussian_sigma_m = 0.2`
- `min_distance_m = 0.8`
- `min_height_m = 5.9`
- `min_segment_area_m2 = 6.0`

**Resultado**

- `Precision = 0.5483`
- `Recall = 0.5230`
- `F1 = 0.5354`

**Leitura**  
Melhor baseline fiel global nas `10` parcelas. Superou o `local maxima` fixo em `10/10` parcelas.

### 4.5 Fusão orientada a objetos `LiDAR + RGB`

**Objetivo**  
Usar segmentação RGB clássica (`Felzenszwalb`) para refinar propostas LiDAR.

**Melhor configuração**

- `source_mode = adaptive`
- `felzenszwalb_scale = 50`
- `felzenszwalb_sigma = 0.9`
- `felzenszwalb_min_size = 192`
- `min_segment_chm_max_m = 6.5`

**Resultado**

- `Precision = 0.4978`
- `Recall = 0.5364`
- `F1 = 0.5164`

**Leitura**  
Superou o `local maxima` fixo e levemente o adaptativo, mas não superou o `watershed`.

### 4.6 Point cloud refiner

**Objetivo**  
Treinar um refinador local na nuvem ao redor de propostas LiDAR.

**Protocolo**  
`3` splits fixos com modos de proposta `adaptive` e `union`.

**Resultado**

| Modo | Precision mean | Recall mean | F1 mean |
|---|---:|---:|---:|
| `union` | 0.4454 | 0.5847 | 0.5025 |
| `adaptive` | 0.4487 | 0.5386 | 0.4865 |

**Leitura**  
Melhora sobre o baseline fixo antigo, mas não passa `watershed`.

## 5. PointNet Estrito e Melhorias

### 5.1 PointNet estrito fiel ao Pretendido

**Objetivo**  
Executar o desenho estrito do `Pretendido.md`: propostas `window=0.8 m`, vizinhança esférica, PointNet puro local, offset XY, NMS.

**Split**

- treino: `plot_02,03,04,05,07,08,09,10`
- validação: `plot_01,06`

**Resultado**

- propostas em validação: `P = 0.0369`, `R = 0.9280`, `F1 = 0.0707`
- PointNet final em validação: `P = 0.2522`, `R = 0.7869`, `F1 = 0.3783`

**Leitura**  
Cobertura de candidatos foi alta, mas a filtragem de `FP` foi insuficiente.

### 5.2 Melhorias sobre o PointNet estrito

**Variantes executadas**

- `strict_redecode_wide`
- `strict_focal_xyz`
- `strict_multifeature_bce`
- `strict_multifeature_focal`

**Resultados**

| Variante | Precision mean | Recall mean | F1 mean |
|---|---:|---:|---:|
| `strict_focal_xyz` | 0.5709 | 0.5347 | 0.5505 |
| `strict_redecode_wide` | 0.5805 | 0.5166 | 0.5444 |
| `strict_multifeature_focal` | 0.5134 | 0.5493 | 0.5302 |
| `strict_multifeature_bce` | 0.5278 | 0.5306 | 0.5262 |

**Leitura**

- o maior ganho veio de `threshold + NMS`;
- `Focal Loss` ajudou um pouco mais;
- `features multicanal` não ajudaram;
- o melhor PointNet fiel ficou muito próximo do melhor baseline fiel:
  - `strict_focal_xyz = 0.5505`
  - `CHM + watershed = 0.5614`

## 6. Híbridos Watershed + Propostas

### 6.1 Híbrido `watershed + proposals`

**Objetivo**  
Usar o `watershed` como filtro geométrico espacial sobre o gerador de propostas de alta revocação.

**Melhor configuração de validação**

- `resolution_m = 0.2`
- `smooth_sigma_px = 1.5`
- `min_distance_px = 3`
- `min_segment_area_m2 = 6.0`
- `proposal_window_m = 0.8`

**Resultados**

| Escopo | Precision mean | Recall mean | F1 mean |
|---|---:|---:|---:|
| validação (`plot_01`,`plot_06`) | 0.5684 | 0.5557 | 0.5593 |
| todas as `10` parcelas | 0.5353 | 0.5201 | 0.5230 |

**Comparação com `watershed` puro**

- validação: `delta F1 = -0.0021`
- `10` parcelas: `delta F1 = -0.0117`

**Leitura**  
Ficou praticamente empatado com o `watershed` puro em validação, mas ligeiramente pior. O diagnóstico mostrou `under` alto, indicando sub-segmentação como gargalo.

## 7. Ranking Prático dos Melhores Resultados

### 7.1 Pipeline principal

| Método | Protocolo | F1 |
|---|---|---:|
| `baseline_local_maxima` | `LOPO` prático | 0.4063 |
| `baseline_local_maxima` | suíte `3` splits | 0.3785 |
| `baseline_local_maxima` | `basic split` longo | 0.3725 |
| `hybrid_proposal_refinement_multimodal_v2` | suíte `3` splits | 0.3278 |
| `multimodal_mid_fusion_redecode` | `basic split` | 0.3143 |

### 7.2 Benchmark fiel

| Método | Protocolo | F1 |
|---|---|---:|
| `CHM + watershed` | `10` parcelas | 0.5354 |
| `strict_focal_xyz` | validação `plot_01+plot_06` | 0.5505 |
| `hybrid_watershed_proposals` | validação `plot_01+plot_06` | 0.5593 |
| `LiDAR + RGB object fusion` | `10` parcelas | 0.5164 |
| `local_maxima` adaptativo | `10` parcelas | 0.5084 |
| `point_cloud_refiner (union)` | `3` splits | 0.5025 |
| `local_maxima` fixo otimizado | `10` parcelas | 0.4393 |
| `baseline` canônico fiel | `10` parcelas | 0.3933 |

## 8. Conclusão Global

1. A baseline clássica permaneceu dominante no pipeline principal multimodal.
2. O melhor deep robusto do projeto principal foi `Hybrid v2`, mas ainda abaixo da baseline.
3. No benchmark fiel, `CHM + watershed` foi o melhor método global executado nas `10` parcelas.
4. O PointNet estrito só ficou competitivo depois de forte calibração de `threshold + NMS`; o melhor refinamento foi `strict_focal_xyz`.
5. O híbrido `watershed + proposals` não superou o `watershed` puro.
6. O padrão mais consistente do projeto inteiro foi: o ganho maior veio de bons priors geométricos e boa calibração de inferência, mais do que de aumento de complexidade arquitetural.

## 9. Documentos de Referência

- `Resultados.md`
- `ResultadosFaithfulPaper.md`
- `ResultadosFaithfulWatershed.md`
- `ResultadosFaithfulAdaptiveLocalMaxima.md`
- `ResultadosFaithfulObjectFusion.md`
- `ResultadosFaithfulPointCloudRefiner.md`
- `ResultadosFaithfulPointNetStrict.md`
- `ResultadosFaithfulPointNetImprovements.md`
- `ResultadosFaithfulHybridWatershedProposals.md`
