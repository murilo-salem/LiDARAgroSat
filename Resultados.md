# Resultados

## 1. Escopo deste documento

Este documento consolida os resultados efetivamente executados no repositório até o estado atual do projeto. Foram realizados dois regimes experimentais distintos:

1. `LOPO` (`Leave-One-Plot-Out`) com perfil prático/curto.
2. `Basic split` em nível de parcela com perfil longo (`120` épocas nominais no `base.yaml`, sujeito a `early stopping`).

O objetivo deste relatório é registrar:

- a descrição exata de cada experimento;
- as configurações usadas;
- as métricas quantitativas produzidas;
- o significado metodológico de cada conjunto de resultados;
- os caminhos dos artefatos gerados.

## 2. Dataset e contexto experimental

### 2.1 Dataset utilizado

Os experimentos foram executados no dataset Dubrovin real baixado do Kaggle:

- `sentinel3734/tree-detection-lidar-rgb`

Características relevantes:

- 10 parcelas;
- dados LiDAR (`LAS`);
- ortofotos RGB;
- inventário de campo com posições de árvores, espécie e alturas parciais.

### 2.2 Pré-processamento usado em todos os experimentos

Todos os resultados deste documento compartilham a mesma preparação de dados:

- resolução raster principal: `0.10 m`;
- `DTM` por quantil baixo em grade com refinamento por máscara de terreno;
- alturas normalizadas `z_norm = z - z_DTM`;
- tensor LiDAR estrutural com 13 canais;
- `RGB` reamostrado para a grade LiDAR;
- `CHM` derivado de `zmax`;
- tiles `512 x 512 px` com overlap;
- matching bipartido `one-to-one` com:
  - raio espacial `1.5 m`;
  - tolerância de altura `3 m` quando aplicável.

### 2.3 Métodos avaliados

Os métodos avaliados foram:

- `baseline_local_maxima`
- `lidar_only`
- `rgb_only`
- `early_fusion`
- `multimodal_mid_fusion`
- `multimodal_late_fusion`

## 3. Experimento A: LOPO com perfil prático

### 3.1 Objetivo

Este experimento foi executado para validar o pipeline completo em regime operacional, preservando o protocolo `LOPO` por parcela, mas reduzindo o custo de treino via um perfil mais curto.

### 3.2 Configuração usada

Arquivos de configuração:

- `configs/base.yaml`
- `configs/practical_run.yaml`
- `configs/<metodo>.yaml`

Parâmetros relevantes do perfil prático:

- `epochs = 20`
- `patience = 5`
- `validation_interval_epochs = 5`
- `batch_size = 2`
- `num_workers = 2`

Observação metodológica:

- este experimento é útil para triagem comparativa e validação end-to-end;
- ele não deve ser interpretado como estimativa final de desempenho máximo do modelo, porque o orçamento de treino foi explicitamente reduzido.

### 3.3 Protocolo

- Avaliação externa: `LOPO`, um fold por parcela.
- Validação interna por parcela única (`validation_plots_per_fold = 1`).
- A baseline clássica foi ajustada por validação em cada fold.
- Os modelos deep learning foram ajustados por `early stopping` usando `F1` no conjunto de validação.

### 3.4 Execução efetiva

Resumo do número de épocas efetivamente executadas por `early stopping`:

| Método | Nº de folds | Épocas médias | Mínimo | Máximo | Melhor `val_f1` médio |
|---|---:|---:|---:|---:|---:|
| `lidar_only` | 10 | 9.9 | 6 | 20 | 0.0921 |
| `rgb_only` | 10 | 6.0 | 6 | 6 | 0.1163 |
| `early_fusion` | 10 | 6.0 | 6 | 6 | 0.1347 |
| `multimodal_mid_fusion` | 10 | 7.0 | 6 | 16 | 0.1854 |
| `multimodal_late_fusion` | 10 | 8.4 | 6 | 20 | 0.1426 |

### 3.5 Resultado agregado principal

Fonte: `outputs/metrics/method_summary.csv`

| Método | Precision média | Recall médio | F1 médio | IC bootstrap 95% F1 | MAE localização (m) | MAE altura (m) |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_local_maxima` | 0.3557 | 0.4889 | 0.4063 | [0.3770, 0.4360] | 0.8319 | 1.3454 |
| `multimodal_mid_fusion` | 0.1243 | 0.3071 | 0.1539 | [0.1090, 0.1923] | 0.9002 | 1.1424 |
| `multimodal_late_fusion` | 0.1044 | 0.2640 | 0.1455 | [0.1004, 0.1966] | 0.9369 | 1.2886 |
| `early_fusion` | 0.0941 | 0.2974 | 0.1372 | [0.0926, 0.1836] | 0.9081 | 2.1003 |
| `rgb_only` | 0.1010 | 0.2270 | 0.1343 | [0.0831, 0.1921] | 0.9317 | NA |
| `lidar_only` | 0.1487 | 0.2662 | 0.0883 | [0.0248, 0.1680] | 0.8947 | 1.4335 |

### 3.6 Interpretação do resultado agregado

Neste regime prático:

- a `baseline_local_maxima` foi claramente superior aos modelos deep learning;
- entre os modelos, `multimodal_mid_fusion` foi o melhor;
- `multimodal_mid_fusion` superou as demais variantes deep, mas permaneceu substancialmente abaixo da baseline clássica;
- `lidar_only` foi instável e sofreu forte degradação por excesso de falsos positivos em alguns folds.

### 3.7 Comparação estatística contra a baseline

Fonte: `outputs/metrics/paired_statistics.csv`

| Comparação (`method_a` vs baseline) | Nº parcelas | Diferença média de F1 | Ganho relativo (%) | `p` |
|---|---:|---:|---:|---:|
| `early_fusion` vs `baseline_local_maxima` | 10 | -0.2691 | -65.09 | 1.0 |
| `lidar_only` vs `baseline_local_maxima` | 10 | -0.3180 | -77.43 | 1.0 |
| `multimodal_late_fusion` vs `baseline_local_maxima` | 10 | -0.2608 | -64.10 | 1.0 |
| `multimodal_mid_fusion` vs `baseline_local_maxima` | 10 | -0.2524 | -61.98 | 1.0 |
| `rgb_only` vs `baseline_local_maxima` | 10 | -0.2720 | -65.76 | 1.0 |

Não houve evidência, neste regime curto, de superioridade estatística de qualquer modelo deep sobre a baseline.

### 3.8 F1 por parcela no LOPO

Fonte derivada de `outputs/metrics/metrics_by_plot.csv`

| Parcela | Baseline | Mid fusion | Late fusion | Early fusion | RGB-only | LiDAR-only |
|---|---:|---:|---:|---:|---:|---:|
| `plot_01` | 0.4142 | 0.0310 | 0.3017 | 0.2715 | 0.3071 | 0.3448 |
| `plot_02` | 0.3810 | 0.2045 | 0.0740 | 0.2073 | 0.0820 | 0.0791 |
| `plot_03` | 0.3402 | 0.1824 | 0.1088 | 0.1616 | 0.0872 | 0.1997 |
| `plot_04` | 0.3623 | 0.0381 | 0.1217 | 0.1962 | 0.1561 | 0.0369 |
| `plot_05` | 0.4262 | 0.1146 | 0.0766 | 0.1231 | 0.0463 | 0.2012 |
| `plot_06` | 0.4429 | 0.2062 | 0.0948 | 0.1579 | 0.1166 | 0.0028 |
| `plot_07` | 0.3666 | 0.1901 | 0.1154 | 0.0340 | 0.1880 | 0.0036 |
| `plot_08` | 0.4729 | 0.1573 | 0.0874 | 0.0664 | 0.0000 | 0.0001 |
| `plot_09` | 0.4838 | 0.2443 | 0.2505 | 0.0267 | 0.1104 | 0.0017 |
| `plot_10` | 0.3729 | 0.1707 | 0.2238 | 0.1272 | 0.2489 | 0.0135 |

### 3.9 Baseline clássica no LOPO

Fonte: `outputs/predictions/baseline_local_maxima/fold_summary.csv`

O melhor conjunto de parâmetros da baseline variou por parcela, com predominância de:

- `window_radius_m`: `1.0` ou `1.5`
- `gaussian_sigma`: `0.7`, `1.0` ou `1.5`
- `min_height_m`: `2.0`, `3.0` ou `4.0`

Exemplo do `fold_00`:

- `window_radius_m = 1.0`
- `gaussian_sigma = 1.0`
- `min_height_m = 3.0`

### 3.10 Ganho da fusão vs LiDAR-only

Fonte: `outputs/metrics/fusion_gain_vs_baseline.csv`

O ganho de `multimodal_mid_fusion` sobre `lidar_only` foi heterogêneo:

- negativo em `plot_01`, `plot_03` e `plot_05`;
- positivo nas demais parcelas;
- especialmente forte em `plot_06`, `plot_07`, `plot_08`, `plot_09` e `plot_10`.

Diferenças `delta_f1 = F1(mid fusion) - F1(lidar_only)`:

| Parcela | `delta_f1` |
|---|---:|
| `plot_01` | -0.3137 |
| `plot_02` | 0.1255 |
| `plot_03` | -0.0173 |
| `plot_04` | 0.0012 |
| `plot_05` | -0.0866 |
| `plot_06` | 0.2034 |
| `plot_07` | 0.1865 |
| `plot_08` | 0.1573 |
| `plot_09` | 0.2426 |
| `plot_10` | 0.1572 |

### 3.11 Regressão exploratória do ganho da fusão

Fonte: `outputs/metrics/fusion_gain_regression.csv`

| Preditor | Inclinação | Intercepto | `R²` |
|---|---:|---:|---:|
| `canopy_height_variance` | -0.000782 | 0.1393 | -0.0016 |
| `vertical_roughness_mean` | -0.176972 | 0.3181 | 0.2510 |
| `tree_density_ha` | -0.000683 | 0.2724 | 0.2052 |
| `rgb_entropy` | -0.066814 | 0.3638 | 0.6574 |
| `conifer_fraction` | 0.000000 | 0.0960 | -0.0351 |

Interpretação:

- como análise exploratória, o sinal mais forte apareceu para `rgb_entropy`;
- porém, dado o baixo número de parcelas (`n=10`) e o perfil curto de treino, esses resultados devem ser tratados apenas como indícios e não como evidência confirmatória.

### 3.12 Artefatos do experimento A

Principais saídas:

- `outputs/metrics/method_summary.csv`
- `outputs/metrics/metrics_by_plot.csv`
- `outputs/metrics/paired_statistics.csv`
- `outputs/metrics/fusion_gain_vs_baseline.csv`
- `outputs/metrics/fusion_gain_regression.csv`
- `outputs/metrics/error_modes.csv`
- `outputs/predictions/<metodo>/fold_xx/*.csv`

## 4. Experimento B: Basic split com perfil longo

### 4.1 Objetivo

Este experimento foi executado para rodar os modelos com orçamento longo de treino, mas sem validação cruzada por `LOPO`, usando um único split básico em nível de parcela.

### 4.2 Configuração usada

Arquivos de configuração:

- `configs/base.yaml`
- `configs/basic_split.yaml`
- `configs/<metodo>.yaml`

Para os métodos multimodais, devido à concorrência de VRAM com outro processo de treinamento em execução na máquina, também foi necessário:

- `configs/batch1.yaml`

Parâmetros centrais:

- `epochs = 120`
- `patience = 20`
- `validation_interval_epochs = 1`
- `batch_size = 4` por default
- `batch_size = 1` nas variantes multimodais do split longo

### 4.3 Split usado

Split determinístico por parcela:

- treino: `plot_04, plot_01, plot_08, plot_09, plot_07, plot_02, plot_06, plot_03`
- validação: `plot_10`
- teste: `plot_05`

### 4.4 Observação metodológica

Este experimento fornece um teste importante do comportamento com treino longo, mas não oferece robustez estatística em nível de parcela, porque o conjunto de teste contém apenas uma única parcela.

### 4.5 Execução efetiva

Número de épocas realmente executadas:

| Método | Épocas executadas | Melhor `val_f1` |
|---|---:|---:|
| `lidar_only` | 22 | 0.0123 |
| `rgb_only` | 21 | 0.1731 |
| `early_fusion` | 21 | 0.2435 |
| `multimodal_mid_fusion` | 85 | 0.3922 |
| `multimodal_late_fusion` | 27 | 0.1076 |

### 4.6 Resultado principal no `plot_05`

Fonte: `outputs_basic_split/metrics/method_summary.csv`

| Método | Precision | Recall | F1 | MAE localização (m) | MAE altura (m) |
|---|---:|---:|---:|---:|---:|
| `baseline_local_maxima` | 0.2729 | 0.5865 | 0.3725 | 0.7300 | 1.0582 |
| `multimodal_mid_fusion` | 0.2590 | 0.3798 | 0.3080 | 0.8075 | 1.1980 |
| `early_fusion` | 0.0659 | 0.4904 | 0.1162 | 0.7869 | 1.4779 |
| `rgb_only` | 0.0401 | 0.2452 | 0.0690 | 1.0432 | NA |
| `multimodal_late_fusion` | 0.0346 | 0.1683 | 0.0574 | 1.0354 | 2.9738 |
| `lidar_only` | 0.0008 | 0.4856 | 0.0016 | 0.9019 | 1.3939 |

### 4.7 Interpretação do experimento B

No `basic split` longo:

- a baseline clássica continuou sendo o melhor método;
- `multimodal_mid_fusion` melhorou substancialmente em relação ao regime curto e foi o melhor deep model;
- a diferença entre `multimodal_mid_fusion` e a baseline caiu para `-0.0645` em `F1`;
- `lidar_only` colapsou por explosão de falsos positivos (`123044` FP no `plot_05`), tornando o `F1` praticamente nulo apesar de `recall` alto.

### 4.8 Métricas detalhadas no `plot_05`

Fonte: `outputs_basic_split/metrics/metrics_by_plot.csv`

| Método | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_local_maxima` | 122 | 325 | 86 | 0.2729 | 0.5865 | 0.3725 |
| `early_fusion` | 102 | 1445 | 106 | 0.0659 | 0.4904 | 0.1162 |
| `lidar_only` | 101 | 123044 | 107 | 0.0008 | 0.4856 | 0.0016 |
| `multimodal_late_fusion` | 35 | 977 | 173 | 0.0346 | 0.1683 | 0.0574 |
| `multimodal_mid_fusion` | 79 | 226 | 129 | 0.2590 | 0.3798 | 0.3080 |
| `rgb_only` | 51 | 1220 | 157 | 0.0401 | 0.2452 | 0.0690 |

### 4.9 Baseline clássica no `basic split`

Fonte: `outputs_basic_split/predictions/baseline_local_maxima/fold_00/best_params.json`

Parâmetros selecionados para o único fold:

- `window_radius_m = 1.0`
- `gaussian_sigma = 0.7`
- `min_height_m = 2.0`

### 4.10 Comparação contra a baseline no `basic split`

Fonte: `outputs_basic_split/metrics/paired_statistics.csv`

Como há apenas uma parcela de teste, não há teste pareado inferencial. Ainda assim, as diferenças observadas foram:

| Comparação | Diferença de F1 | Ganho relativo (%) |
|---|---:|---:|
| `multimodal_mid_fusion` vs `baseline_local_maxima` | -0.0645 | -17.32 |
| `early_fusion` vs `baseline_local_maxima` | -0.2563 | -68.80 |
| `rgb_only` vs `baseline_local_maxima` | -0.3036 | -81.49 |
| `multimodal_late_fusion` vs `baseline_local_maxima` | -0.3151 | -84.60 |
| `lidar_only` vs `baseline_local_maxima` | -0.3709 | -99.56 |

### 4.11 Ganho da fusão vs LiDAR-only

Fonte: `outputs_basic_split/metrics/fusion_gain_vs_baseline.csv`

No `plot_05`:

- `delta_f1 = F1(multimodal_mid_fusion) - F1(lidar_only) = 0.30635`

Esse ganho é numericamente forte, mas a interpretação deve ser cautelosa porque a baseline `lidar_only` teve comportamento degenerado nesse split.

### 4.12 Regressão de ganho vs heterogeneidade

Fonte: `outputs_basic_split/metrics/fusion_gain_regression.csv`

Não foi produzida regressão interpretável neste regime, porque havia apenas um plot de teste. O arquivo correspondente contém `NaN`.

### 4.13 Artefatos do experimento B

Principais saídas:

- `outputs_basic_split/metrics/method_summary.csv`
- `outputs_basic_split/metrics/metrics_by_plot.csv`
- `outputs_basic_split/metrics/paired_statistics.csv`
- `outputs_basic_split/metrics/fusion_gain_vs_baseline.csv`
- `outputs_basic_split/metrics/fusion_gain_regression.csv`
- `outputs_basic_split/metrics/error_modes.csv`
- `outputs_basic_split/predictions/<metodo>/fold_00/*.csv`

## 5. Síntese comparativa dos dois regimes

### 5.1 Tendências robustas observadas

Em ambos os regimes executados:

- a baseline clássica `CHM + local maxima` foi o melhor método global;
- `multimodal_mid_fusion` foi consistentemente o melhor deep model;
- `multimodal_mid_fusion` superou `multimodal_late_fusion`, `early_fusion`, `rgb_only` e, na maior parte dos cenários úteis, `lidar_only`;
- a modalidade LiDAR sozinha apresentou forte instabilidade, sobretudo por explosão de falsos positivos.

### 5.2 O que muda do regime curto para o longo

Quando o treinamento passa do perfil prático para o perfil longo:

- `multimodal_mid_fusion` melhora muito:
  - `F1 médio LOPO prático = 0.1539`
  - `F1 no basic split longo = 0.3080`
- `early_fusion` melhora pouco e permanece distante da baseline;
- `multimodal_late_fusion` não alcança o mesmo nível de `mid fusion`;
- a baseline permanece competitiva e não foi superada em nenhum dos regimes executados até agora.

### 5.3 Leitura científica atual

Com base apenas nos experimentos já executados:

- a hipótese `H1` do plano original ainda não foi confirmada;
- o melhor deep model implementado é `multimodal_mid_fusion`;
- o regime curto é útil como triagem, mas insuficiente para sustentar reivindicações fortes;
- o regime longo em `basic split` sugere que maior orçamento de treinamento aproxima o modelo multimodal da baseline;
- o próximo experimento cientificamente decisivo continua sendo `LOPO` com perfil longo, ainda não completado.

### 5.4 Ablação diagnóstica de treino e decode

Em `2026-03-26` foi executada uma ablação diagnóstica adicional no mesmo `basic split`, com três blocos:

- redecodificação dos checkpoints longos já existentes com `TTA + tuning` na validação;
- novo treino `multimodal_mid_fusion_roi` com `adaptive sigma`, `gradient accumulation = 4`, `min_epochs_before_stop = 25`, `patience = 15`, `TTA` final e `postprocess tuning`;
- novo treino `lidar_only_roi` com o mesmo regime.

Resumo dos resultados em `plot_05`:

| Método | Precision | Recall | F1 |
|---|---:|---:|---:|
| `baseline_local_maxima` | 0.2729 | 0.5865 | 0.3725 |
| `multimodal_mid_fusion` | 0.2590 | 0.3798 | 0.3080 |
| `multimodal_mid_fusion_redecode` | 0.2211 | 0.5433 | 0.3143 |
| `multimodal_mid_fusion_roi` | 0.1254 | 0.5577 | 0.2048 |
| `lidar_only` | 0.0008 | 0.4856 | 0.0016 |
| `lidar_only_redecode` | 0.1022 | 0.2885 | 0.1509 |
| `lidar_only_roi` | 0.1789 | 0.4567 | 0.2571 |

Leitura técnica imediata:

- `decode tuning` foi efetivo em ambos os casos, mas com naturezas diferentes;
- em `multimodal_mid_fusion`, o ganho veio principalmente por aumento de `recall`:
  - `F1: 0.3080 -> 0.3143`
  - `Recall: 0.3798 -> 0.5433`
  - `Precision: 0.2590 -> 0.2211`
- em `lidar_only`, o tuning de decode reduziu o colapso de falsos positivos:
  - `FP: 123044 -> 527`
  - `F1: 0.0016 -> 0.1509`
- o novo regime de treino `lidar_only_roi` trouxe ganho adicional real:
  - `F1: 0.1509 -> 0.2571`
  - `Precision: 0.1022 -> 0.1789`
- o novo regime de treino `multimodal_mid_fusion_roi` não generalizou bem no teste quando combinado com tuning por um único plot de validação:
  - com decode tunado, `F1 = 0.2048`;
  - com decode default sobre o mesmo checkpoint, `F1 = 0.2892`.

Interpretação:

- o problema de `lidar_only` era de fato parcialmente um problema de calibração de inferência, não apenas de arquitetura;
- em `multimodal_mid_fusion`, o principal limitante continua sendo o compromisso `precision-recall`;
- o novo regime aumentou a densidade de detecções e o `recall`, mas sem mecanismo suficiente para conter `FP` no teste;
- com apenas um `plot` de validação, o tuning de decode mostrou risco claro de sobreajuste para a variante `mid_fusion_roi`.

### 5.5 Extensão completa do detector e pipeline híbrido

Em `2026-03-26` foi implementado e testado o bloco restante do plano metodológico, com dois novos métodos:

- `multimodal_mid_fusion_full`:
  - `adaptive sigma`
  - `hard negative mining` em nível de tile e pixel
  - head auxiliar de `peakness`
  - head auxiliar de espécie em modo `coarse` (`conifer` vs `broadleaf`)
  - `deep supervision` multiescala
  - `adaptive NMS`
- `hybrid_proposal_refinement_multimodal`:
  - propostas pela união de `CHM local maxima` multiescala e picos de heatmap de `multimodal_mid_fusion`
  - refinador multimodal por patches com saídas de classificação, offset, altura e espécie
  - tuning final de `classification threshold + NMS + min_height`

Resumo em `plot_05`:

| Método | Precision | Recall | F1 |
|---|---:|---:|---:|
| `baseline_local_maxima` | 0.2729 | 0.5865 | 0.3725 |
| `multimodal_mid_fusion_redecode` | 0.2211 | 0.5433 | 0.3143 |
| `multimodal_mid_fusion` | 0.2590 | 0.3798 | 0.3080 |
| `multimodal_mid_fusion_full` | 0.2007 | 0.5433 | 0.2931 |
| `hybrid_proposal_refinement_multimodal` | 0.1419 | 0.6394 | 0.2323 |

Leitura técnica:

- o detector denso completo aumentou `recall`, mas não superou o `mid_fusion` simples redecodificado:
  - `multimodal_mid_fusion_redecode`: `F1 = 0.3143`
  - `multimodal_mid_fusion_full`: `F1 = 0.2931`
- a combinação `peakness + species + deep supervision + hard negatives` não resolveu o gargalo principal, que continua sendo o controle de `FP` quando o `recall` sobe;
- o híbrido confirmou o potencial de alta revocação:
  - `Recall = 0.6394`, o maior entre todos os métodos testados;
- porém o refinador ainda não está discriminando negativos difíceis com precisão suficiente:
  - `FP = 804`
  - `Precision = 0.1419`

Interpretação científica atual:

- o prior geométrico realmente ajuda a recuperar árvores perdidas, mas o sistema ainda carece de um mecanismo mais forte de supressão e reclassificação local;
- entre todas as variantes deep já executadas, o melhor compromisso continua sendo o `multimodal_mid_fusion` original com `decode tuning`;
- a direção mais promissora permanece o híbrido, mas o refinador ainda precisa de maior maturidade experimental para transformar o ganho de `recall` em ganho de `F1`.

Observação metodológica:

- o pipeline híbrido foi implementado por completo e executado fim a fim;
- devido ao custo computacional do banco de propostas, a rodada final do refinador foi conduzida em regime prático com `candidate_max_per_plot = 1536`, `batch_size = 32` e checkpoint final reavaliado por tuning explícito de validação.

### 5.6 Suíte `Hybrid v2` em três basic splits

Em `2026-03-26` foi executada uma suíte adicional mais controlada para testar a continuação mais promissora identificada na análise técnica:

- proposal model `multimodal_mid_fusion` em regime prático curto, com `adaptive sigma`, `TTA` e `decode tuning`;
- refinador `hybrid_proposal_refinement_multimodal_v2` com:
  - banco explícito de `hard negatives` em nível de candidato;
  - `focal loss` no head de classificação;
  - `features` tabulares de proposta:
    - confiança da proposta;
    - altura prevista;
    - `CHM` local;
    - `peakness` local;
    - densidade local;
    - indicador de origem `CHM` + `heatmap`;
  - calibração final específica por origem de proposta (`chm`, `heatmap`, `both`).

Para aumentar a robustez sem voltar ainda ao `LOPO` longo, a suíte foi repetida em três seeds de `basic split`:

| Seed | Treino | Validação | Teste |
|---|---|---|---|
| `13` | `plot_04, plot_01, plot_08, plot_09, plot_07, plot_02, plot_06, plot_03` | `plot_10` | `plot_05` |
| `21` | `plot_08, plot_01, plot_05, plot_02, plot_04, plot_10, plot_06, plot_09` | `plot_07` | `plot_03` |
| `34` | `plot_03, plot_05, plot_10, plot_04, plot_08, plot_02, plot_07, plot_01` | `plot_06` | `plot_09` |

Resumo por split:

| Split | Plot teste | Método | Precision | Recall | F1 |
|---|---|---|---:|---:|---:|
| `seed13` | `plot_05` | `baseline_local_maxima` | 0.2729 | 0.5865 | 0.3725 |
| `seed13` | `plot_05` | `multimodal_mid_fusion` | 0.1336 | 0.3365 | 0.1913 |
| `seed13` | `plot_05` | `hybrid_proposal_refinement_multimodal_v2` | 0.2148 | 0.4327 | 0.2871 |
| `seed21` | `plot_03` | `baseline_local_maxima` | 0.2744 | 0.4910 | 0.3521 |
| `seed21` | `plot_03` | `multimodal_mid_fusion` | 0.1812 | 0.4639 | 0.2606 |
| `seed21` | `plot_03` | `hybrid_proposal_refinement_multimodal_v2` | 0.2765 | 0.3855 | 0.3220 |
| `seed34` | `plot_09` | `baseline_local_maxima` | 0.4756 | 0.3617 | 0.4109 |
| `seed34` | `plot_09` | `multimodal_mid_fusion` | 0.3030 | 0.2898 | 0.2962 |
| `seed34` | `plot_09` | `hybrid_proposal_refinement_multimodal_v2` | 0.3879 | 0.3617 | 0.3743 |

Agregado sobre os três splits:

| Método | Nº splits | Precision média | Recall médio | F1 médio | Desvio padrão F1 |
|---|---:|---:|---:|---:|---:|
| `baseline_local_maxima` | 3 | 0.3410 | 0.4797 | 0.3785 | 0.0299 |
| `hybrid_proposal_refinement_multimodal_v2` | 3 | 0.2930 | 0.3933 | 0.3278 | 0.0439 |
| `multimodal_mid_fusion` | 3 | 0.2059 | 0.3634 | 0.2493 | 0.0534 |

Diferenças por split para o `Hybrid v2`:

| Split | `delta_f1` vs `mid_fusion` | `delta_f1` vs baseline |
|---|---:|---:|
| `seed13` | +0.0958 | -0.0854 |
| `seed21` | +0.0614 | -0.0300 |
| `seed34` | +0.0781 | -0.0366 |

Leitura técnica:

- o `Hybrid v2` superou o `multimodal_mid_fusion` em todos os três splits;
- o ganho em `F1` sobre `mid_fusion` foi consistente:
  - média `+0.0784`;
  - `3/3` vitórias pareadas;
- o `Hybrid v2` ainda ficou abaixo da baseline em todos os três splits:
  - média `-0.0507` em `F1`;
  - `0/3` vitórias pareadas;
- em `seed21`, o `Hybrid v2` praticamente igualou a baseline em precisão:
  - `0.2765` vs `0.2744`;
  - a perda residual ficou concentrada em `recall`.

Interpretação científica atual:

- a continuação proposta foi válida e produziu o melhor avanço real após o `mid_fusion` original;
- o híbrido com prior geométrico explícito mostrou ganho consistente de `F1` sobre o detector denso puro;
- a hipótese de que `proposal + refinement` é o caminho mais plausível para ultrapassar a baseline ficou reforçada;
- ainda assim, a baseline clássica permanece como melhor método global na média dos três splits executados.

### 5.7 Baseline como estágio 1 puro

Em `2026-03-26` foi implementada e testada uma variante mais estrita do princípio `baseline-as-proposal-stage`, sem proposal net:

- estágio 1:
  - propostas apenas por `CHM local maxima` multiescala;
  - múltiplos `window_radius_m` e `gaussian_sigma`;
  - deduplicação espacial entre propostas;
- estágio 2:
  - refinador multimodal por candidato (`RGB + LiDAR estrutural + CHM + peakness`);
  - heads de `objectness`, `dx/dy` e `height`;
  - rotulagem `one-to-one` entre candidatos e árvores de referência via matching bipartido local;
  - candidatos próximos do mesmo `GT` mas não selecionados como match passam a ser `hard negatives`.

Configuração principal:

- `configs/hybrid_baseline_stage_model.yaml`
- `configs/hybrid_baseline_stage_practical.yaml`

A execução foi feita no `basic split` canônico:

- treino: `plot_04, plot_01, plot_08, plot_09, plot_07, plot_02, plot_06, plot_03`
- validação: `plot_10`
- teste: `plot_05`

Resultado em `plot_05`:

| Método | Precision | Recall | F1 | MAE localização (m) | MAE altura (m) |
|---|---:|---:|---:|---:|---:|
| `baseline_local_maxima` | 0.2729 | 0.5865 | 0.3725 | 0.7300 | 1.0582 |
| `hybrid_baseline_stage_refiner` | 0.1762 | 0.6394 | 0.2762 | 0.7792 | 1.5676 |

Comparação com as melhores referências prévias no mesmo `plot_05`:

| Método | Precision | Recall | F1 |
|---|---:|---:|---:|
| `baseline_local_maxima` | 0.2729 | 0.5865 | 0.3725 |
| `multimodal_mid_fusion_redecode` | 0.2211 | 0.5433 | 0.3143 |
| `hybrid_proposal_refinement_multimodal_v2` | 0.2148 | 0.4327 | 0.2871 |
| `hybrid_baseline_stage_refiner` | 0.1762 | 0.6394 | 0.2762 |

Leitura técnica:

- o uso da baseline como estágio 1 puro aumentou o `recall` acima de todos os métodos anteriores nesse `plot`:
  - `Recall = 0.6394`;
- porém o refinador não controlou `FP` o suficiente:
  - `TP = 133`;
  - `FP = 622`;
  - `FN = 75`;
- o resultado final ficou abaixo do `Hybrid v2` com propostas `CHM + heatmap`:
  - `0.2762` vs `0.2871` em `F1`;
- também ficou abaixo do melhor detector denso já calibrado nesse split:
  - `multimodal_mid_fusion_redecode = 0.3143`.

Interpretação:

- a ideia de usar a baseline como gerador de propostas é válida e elevou a cobertura de árvores reais;
- no entanto, usar somente propostas derivadas do `CHM` piorou o compromisso `precision-recall` em relação ao híbrido com união `CHM + heatmap`;
- neste estado do projeto, a melhor leitura é que a baseline funciona melhor como componente dominante do estágio 1, mas ainda se beneficia de complementaridade com propostas aprendidas.

### 5.8 Detector tri-modal `RGB + LiDAR + CHM`

Em `2026-03-26` foi implementado um detector denso novo com foco nas mudanças sugeridas para o ramo de detecção raster:

- ramo explícito de `CHM` com `6` canais derivados:
  - `CHM`,
  - gradiente,
  - Laplaciano,
  - curvatura local,
  - `peakness`,
  - rugosidade local;
- fusão tri-modal interativa `RGB + LiDAR estrutural + CHM` por `gating` condicionado ao contexto do `CHM`;
- alvo geométrico auxiliar de `distance/crownness`;
- `sigma` adaptativo por altura e distância ao vizinho mais próximo;
- `snap` opcional ao ápice local com alvo bimodal fraco;
- pesos de dificuldade por densidade local, baixa altura relativa e `peakness`;
- contexto maior com supervisão apenas na região central do tile.

Configuração principal:

- `configs/multimodal_chm_context.yaml`
- `configs/chm_context_suite_practical.yaml`

O método foi avaliado na mesma suíte de `3` basic splits usada no `Hybrid v2`.

Resumo agregado:

| Método | Nº splits | Precision média | Recall médio | F1 médio |
|---|---:|---:|---:|---:|
| `baseline_local_maxima` | 3 | 0.3410 | 0.4797 | 0.3785 |
| `multimodal_chm_context_interactive` | 3 | 0.1620 | 0.3839 | 0.1943 |

Resultados por split:

| Split | Plot teste | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| `seed13` | `plot_05` | 0.0497 | 0.2500 | 0.0829 |
| `seed21` | `plot_03` | 0.1478 | 0.6687 | 0.2421 |
| `seed34` | `plot_09` | 0.2884 | 0.2331 | 0.2578 |

Leitura técnica:

- o novo detector tri-modal não superou nem o `multimodal_mid_fusion` antigo na suíte comparável;
- o ganho de `recall` em `seed21` não se converteu em `F1` competitivo devido à queda forte de `precision`;
- o acréscimo simultâneo de vários mecanismos de supervisão e fusão tornou o treinamento mais instável do que o pipeline híbrido.

Comparação direta com a suíte anterior:

| Método | Precision média | Recall médio | F1 médio |
|---|---:|---:|---:|
| `baseline_local_maxima` | 0.3410 | 0.4797 | 0.3785 |
| `hybrid_proposal_refinement_multimodal_v2` | 0.2930 | 0.3933 | 0.3278 |
| `multimodal_mid_fusion` | 0.2059 | 0.3634 | 0.2493 |
| `multimodal_chm_context_interactive` | 0.1620 | 0.3839 | 0.1943 |

Conclusão específica desta variante:

- tratar `CHM` como modalidade explícita e enriquecer a supervisão foi cientificamente defensável;
- porém, na implementação atual, isso não produziu ganho de benchmark;
- o resultado reforça que a estratégia híbrida por propostas continua sendo o melhor caminho empírico do projeto.

### 5.9 Auditoria centro↔ápice e pseudo-rótulos

Também foram implementadas duas peças auxiliares de metodologia:

- auditoria de alinhamento entre centro de referência e ápice local do `CHM`:
  - `scripts/audit_apex_alignment.py`
- gerador de pseudo-rótulos por concordância entre dois professores:
  - `scripts/generate_teacher_pseudolabels.py`

Na auditoria executada com raio de `1.0 m`, o deslocamento centro↔ápice foi substancial:

| Estatística global | Valor |
|---|---:|
| média | 0.8537 m |
| mediana | 0.9019 m |
| percentil 90 | 1.2298 m |
| percentil 95 | 1.3022 m |

Interpretação:

- há evidência forte de ruído geométrico relevante entre o centro anotado e o ápice local do dossel;
- isso sustenta cientificamente a inclusão de `snap` local e alvos bimodais fracos no treino;
- também ajuda a explicar por que a baseline geométrica continua competitiva.

## 6. Caminhos dos resultados

### 6.1 Resultados LOPO prático

- `outputs/metrics/method_summary.csv`
- `outputs/metrics/metrics_by_plot.csv`
- `outputs/metrics/paired_statistics.csv`
- `outputs/metrics/fusion_gain_vs_baseline.csv`
- `outputs/metrics/fusion_gain_regression.csv`
- `outputs/metrics/error_modes.csv`

### 6.2 Resultados basic split longo

- `outputs_basic_split/metrics/method_summary.csv`
- `outputs_basic_split/metrics/metrics_by_plot.csv`
- `outputs_basic_split/metrics/paired_statistics.csv`
- `outputs_basic_split/metrics/fusion_gain_vs_baseline.csv`
- `outputs_basic_split/metrics/fusion_gain_regression.csv`
- `outputs_basic_split/metrics/error_modes.csv`

### 6.3 Resultados da ablação diagnóstica

- `outputs_ablation_basic/metrics/method_summary.csv`
- `outputs_ablation_basic/metrics/metrics_by_plot.csv`
- `outputs_ablation_basic/metrics/paired_statistics.csv`
- `outputs_ablation_basic/predictions/multimodal_mid_fusion_roi/fold_00/best_decode_params.json`
- `outputs_ablation_basic/predictions/lidar_only_roi/fold_00/best_decode_params.json`
- `outputs_ablation_basic/diagnostics/multimodal_mid_fusion_redecode.json`
- `outputs_ablation_basic/diagnostics/lidar_only_redecode.json`

### 6.4 Resultados da extensão completa

- `outputs_complete_ablation/metrics/method_summary.csv`
- `outputs_complete_ablation/metrics/metrics_by_plot.csv`
- `outputs_complete_ablation/metrics/paired_statistics.csv`
- `outputs_complete_ablation/predictions/multimodal_mid_fusion_full/fold_00/best_decode_params.json`
- `outputs_complete_ablation/predictions/hybrid_proposal_refinement_multimodal/fold_00/best_params.json`
- `outputs_complete_ablation/predictions/hybrid_proposal_refinement_multimodal/fold_00/plot_05.csv`

### 6.5 Resultados da suíte `Hybrid v2`

- `outputs_suite_seed13/metrics/method_summary.csv`
- `outputs_suite_seed13/metrics/metrics_by_plot.csv`
- `outputs_suite_seed13/predictions/hybrid_proposal_refinement_multimodal_v2/fold_00/best_params.json`
- `outputs_suite_seed13/predictions/hybrid_proposal_refinement_multimodal_v2/fold_00/plot_05.csv`
- `outputs_suite_seed21/metrics/method_summary.csv`
- `outputs_suite_seed21/metrics/metrics_by_plot.csv`
- `outputs_suite_seed21/predictions/hybrid_proposal_refinement_multimodal_v2/fold_00/best_params.json`
- `outputs_suite_seed21/predictions/hybrid_proposal_refinement_multimodal_v2/fold_00/plot_03.csv`
- `outputs_suite_seed34/metrics/method_summary.csv`
- `outputs_suite_seed34/metrics/metrics_by_plot.csv`
- `outputs_suite_seed34/predictions/hybrid_proposal_refinement_multimodal_v2/fold_00/best_params.json`
- `outputs_suite_seed34/predictions/hybrid_proposal_refinement_multimodal_v2/fold_00/plot_09.csv`
- `outputs_suite_aggregate/suite_metrics_by_plot.csv`
- `outputs_suite_aggregate/suite_focus_metrics.csv`
- `outputs_suite_aggregate/suite_method_summary.csv`
- `outputs_suite_aggregate/suite_hybrid_deltas.csv`
- `outputs_suite_aggregate/suite_paired_comparisons.csv`

### 6.6 Resultados da variante baseline-first-stage

- `outputs_baseline_stage_basic/metrics/method_summary.csv`
- `outputs_baseline_stage_basic/metrics/metrics_by_plot.csv`
- `outputs_baseline_stage_basic/predictions/hybrid_baseline_stage_refiner/fold_00/best_params.json`
- `outputs_baseline_stage_basic/predictions/hybrid_baseline_stage_refiner/fold_00/validation_scores.json`
- `outputs_baseline_stage_basic/predictions/hybrid_baseline_stage_refiner/fold_00/plot_05.csv`

### 6.7 Resultados do detector tri-modal

- `outputs_suite_seed13/predictions/multimodal_chm_context_interactive/fold_00/best_decode_params.json`
- `outputs_suite_seed13/predictions/multimodal_chm_context_interactive/fold_00/plot_05.csv`
- `outputs_suite_seed21/predictions/multimodal_chm_context_interactive/fold_00/best_decode_params.json`
- `outputs_suite_seed21/predictions/multimodal_chm_context_interactive/fold_00/plot_03.csv`
- `outputs_suite_seed34/predictions/multimodal_chm_context_interactive/fold_00/best_decode_params.json`
- `outputs_suite_seed34/predictions/multimodal_chm_context_interactive/fold_00/plot_09.csv`
- `outputs_chm_context_aggregate/suite_metrics_by_plot.csv`
- `outputs_chm_context_aggregate/suite_method_summary.csv`
- `outputs_chm_context_aggregate/comparison_vs_previous_suite.csv`

### 6.8 Auditoria e pseudo-rótulos

- `outputs/analysis/apex_alignment/apex_offsets.csv`
- `outputs/analysis/apex_alignment/apex_offset_summary.csv`
- `outputs_pseudolabels_seed13/plot_05.csv`

## 7. Conclusão objetiva

Os resultados atualmente disponíveis mostram que:

1. o pipeline completo foi executado com sucesso em dados reais;
2. a baseline clássica continua sendo a referência mais forte;
3. `multimodal_mid_fusion` é a melhor arquitetura deep implementada até o momento;
4. o tuning de decode é parte essencial do desempenho e corrige grande parte do colapso de `lidar_only`;
5. o melhor ganho incremental observado até aqui em `mid_fusion` veio da calibração de inferência, não do novo regime composto de treino;
6. o `Hybrid v2` foi a primeira continuação a superar o `mid_fusion` de forma consistente em múltiplos splits básicos;
7. a extensão completa do detector denso elevou `recall`, mas não converteu esse ganho em `F1` superior ao melhor híbrido já obtido;
8. a variante `baseline-first-stage` pura elevou ainda mais o `recall`, mas perdeu em `precision` e não superou o `Hybrid v2` com propostas `CHM + heatmap`;
9. o detector tri-modal com ramo explícito de `CHM` e supervisão geométrica adicional não melhorou o benchmark na forma atual;
10. ainda não há evidência experimental de superação da baseline clássica pelo modelo multimodal no protocolo executado, embora a distância média tenha caído para cerca de `0.05` de `F1` na suíte `Hybrid v2`.
