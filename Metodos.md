# Metodos

## 1. Objetivo deste documento

Este documento externaliza, em um unico lugar, a arquitetura experimental implementada no repositorio, os protocolos de avaliacao, a suite de testes automatizados, a linha temporal das familias de experimentos e os resultados agregados de todas as runs executadas ate o momento.

O objetivo aqui nao e recontar apenas o melhor numero final. O objetivo e registrar:

- como o sistema foi organizado;
- quais familias metodologicas foram implementadas;
- quais protocolos e matching foram usados;
- quais ablations, extensoes e comparativos foram realmente executados;
- quais resultados foram obtidos em cada regime;
- quais conclusoes ficaram sustentadas por evidencia e quais linhas ficaram inconclusivas.

Este documento consolida principalmente:

- `ArquiteturaImplementadaAtual.md`
- `RelatorioCompleto.md`
- `Resultados.md`
- `ResultadosFaithfulPaper.md`
- `ResultadosFaithfulWatershed.md`
- `ResultadosFaithfulAdaptiveLocalMaxima.md`
- `ResultadosFaithfulObjectFusion.md`
- `ResultadosFaithfulPointCloudRefiner.md`
- `ResultadosFaithfulPointNetStrict.md`
- `ResultadosFaithfulPointNetImprovements.md`
- `ResultadosFaithfulHybridWatershedProposals.md`
- `docs/benchmark_protocol_v2.md`
- `TicketFinal.md`

Sempre que necessario, os numeros aqui referenciados foram consolidados diretamente dos CSVs e JSONs em `outputs*`.

Importante: nem todos os numeros descritos abaixo sao diretamente comparaveis entre si, porque o projeto acumulou regimes distintos ao longo do tempo:

- `LOPO` pratico;
- `basic split` longo;
- validacao simples em `3` seeds;
- suite comparavel de `3` splits;
- benchmark fiel em `10` parcelas;
- validacao fixa em `plot_01 + plot_06` para variantes PointNet fieis.

Por isso, ao longo do texto, toda familia de resultados e sempre descrita junto com o protocolo correspondente.

## 2. Problema cientifico e dataset

O projeto trata de `Individual Tree Detection` em florestas mistas densas com `UAV LiDAR + RGB`, tendo como problema central a deteccao de centros individuais de arvores em cenarios com:

- forte heterogeneidade estrutural;
- copas sobrepostas;
- mistura de folhosas e coniferas;
- frequente ausencia de um unico maximo dominante em individuos folhosos.

O dataset operacional usado no repositorio e o release publico do benchmark Dubrovin no Kaggle (`sentinel3734/tree-detection-lidar-rgb`), composto por:

- `10` parcelas;
- nuvem LiDAR por parcela em `LAS/LAZ`;
- ortofoto RGB de `7 cm`;
- inventario de campo com coordenadas, especie e alturas parciais quando disponiveis.

As parcelas sao tratadas como unidade estatistica e unidade de particionamento, para evitar vazamento espacial entre tiles vizinhos da mesma area.

## 3. Arquitetura do repositorio

O repositorio foi organizado em blocos funcionais.

### 3.1 Configuracao e benchmarking

Modulos principais:

- `src/lidaragrosat/config.py`
- `src/lidaragrosat/benchmarking.py`
- `src/lidaragrosat/experiment_manifest.py`

Responsabilidades:

- definir a arvore de configuracao experimental;
- compor YAMLs base e overrides;
- carregar e persistir `protocol_id`;
- construir nomes canonicos de experimento;
- gerar e carregar `experiment_manifest.json`.

No ciclo final do projeto, todo experimento agregavel passou a carregar:

- `protocol_id`;
- `stage1`;
- `stage2`;
- `labels`;
- `decode_params`;
- `commit_hash`;
- identificacao de dataset/split.

### 3.2 Ingestao, pre-processamento e targets

Modulos principais:

- `src/lidaragrosat/data/inventory.py`
- `src/lidaragrosat/data/reference.py`
- `src/lidaragrosat/data/lidar.py`
- `src/lidaragrosat/data/rgb.py`
- `src/lidaragrosat/data/preprocess.py`
- `src/lidaragrosat/data/qa.py`
- `src/lidaragrosat/data/targets.py`
- `src/lidaragrosat/data/auxiliary.py`
- `src/lidaragrosat/data/candidates.py`
- `src/lidaragrosat/data/dataset.py`
- `src/lidaragrosat/data/tiles.py`

Responsabilidades:

- descoberta do dataset por parcela;
- leitura e saneamento do inventario;
- leitura da nuvem LiDAR;
- construcao de `DTM`;
- normalizacao de altura;
- rasterizacao estrutural LiDAR;
- reamostragem e normalizacao do RGB;
- controle de qualidade multimodal;
- geracao de heatmaps, offsets, altura, species maps, distance maps e peakness maps;
- montagem de datasets globais e locais por proposta/segmento.

### 3.3 Modelos principais

Modulos principais:

- `src/lidaragrosat/models/blocks.py`
- `src/lidaragrosat/models/detector.py`
- `src/lidaragrosat/models/refiner.py`
- `src/lidaragrosat/stage2/models.py`
- `src/lidaragrosat/stage2/graph_models.py`

Familias implementadas:

- detector center-based raster `RGB-only`, `LiDAR-only`, `early`, `mid`, `late`;
- detector tri-modal com ramo explicito de `CHM`;
- refinadores locais `raster`, `point`, `hybrid`;
- variantes de ensemble, stacker logistico, `split predictor` e `segment_gnn`;
- PointNet local estrito na familia fiel.

### 3.4 Baselines geometricos e familia paper_faithful

Modulos principais:

- `src/lidaragrosat/baselines/local_maxima.py`
- `src/lidaragrosat/paper_faithful/method.py`
- `src/lidaragrosat/paper_faithful/adaptive_local_maxima.py`
- `src/lidaragrosat/paper_faithful/watershed.py`
- `src/lidaragrosat/paper_faithful/object_fusion.py`
- `src/lidaragrosat/paper_faithful/object_fusion_slic.py`
- `src/lidaragrosat/paper_faithful/hybrid_watershed_proposals.py`
- `src/lidaragrosat/paper_faithful/point_cloud_refiner.py`
- `src/lidaragrosat/paper_faithful/strict_pointnet.py`

Responsabilidades:

- baseline fiel do paper com `local maxima` em nuvem 3D;
- janela adaptativa por altura;
- `CHM + watershed`;
- fusao orientada a objetos `LiDAR + RGB`;
- fusao `watershed + proposals`;
- refinadores locais na nuvem;
- PointNet local fiel ao desenho de `Pretendido.md`.

### 3.5 Avaliacao, estatistica e relatorios

Modulos principais:

- `src/lidaragrosat/evaluation/matching.py`
- `src/lidaragrosat/evaluation/metrics.py`
- `src/lidaragrosat/evaluation/statistics.py`
- `src/lidaragrosat/analysis/scientific.py`
- `src/lidaragrosat/analysis/error_modes.py`
- `src/lidaragrosat/analysis/reporting.py`

Responsabilidades:

- matching um-para-um por criterios espaciais e de altura;
- calculo de `precision`, `recall`, `F1`, `location_mae_m`, `height_mae_m`, `TP`, `FP`, `FN`;
- comparacoes pareadas;
- ganho por parcela e por estagio;
- erros por tipo;
- agregacao automatica por manifesto/protocolo.

### 3.6 Linha alternativa de instancia/copa

Modulos principais:

- `src/lidaragrosat/instance_targets.py`
- `src/lidaragrosat/instance_baseline.py`

Responsabilidades:

- gerar `distance_transform`, `seedness`, `pseudo_crown_mask` e `star_convex_rays`;
- decodificar mapas de instancia;
- suportar o baseline inicial `distance-transform + watershed`.

## 4. Pipeline de dados e representacoes

### 4.1 Pipeline principal multimodal raster

O fluxo base do pipeline principal foi:

1. descoberta do dataset por parcela;
2. leitura da nuvem LiDAR;
3. construcao de `DTM` por quantil baixo e refinamento por residuo ao solo;
4. normalizacao `z_norm = z - z_DTM(x,y)`;
5. filtro de alturas fisicas;
6. rasterizacao estrutural LiDAR;
7. reamostragem do RGB para a grade LiDAR;
8. normalizacao radiometrica do RGB;
9. geracao do `CHM`;
10. QA de alinhamento e cobertura;
11. conversao do inventario em heatmap/offset/altura e mapas auxiliares;
12. tiling por parcela;
13. treino/inferencia por tiles;
14. agregacao por parcela;
15. matching e metricas.

### 4.2 Canais estruturais LiDAR

Os canais estruturais efetivamente usados no pipeline principal incluem:

- `zmax`
- `zmean`
- `zstd`
- `zp95`
- `density_total`
- `density_0p5_2`
- `density_2_5`
- `density_5_10`
- `density_10_20`
- `density_gt20`
- `cover_gt2`
- `cover_gt5`
- `vertical_roughness`

Em varias etapas posteriores tambem foram derivados:

- `CHM`;
- gradiente de `CHM`;
- laplaciano/curvatura local;
- mapas de `peakness`;
- descritores de segmento como area, elongacao, compactness e altura maxima.

### 4.3 Alvos supervisionados

Ao longo do projeto, os seguintes regimes de target foram implementados:

- heatmap gaussiano de centro;
- offset subpixel `(dx, dy)`;
- altura;
- `distance-to-center`;
- `peakness`;
- `species` em modo fino ou grosso;
- `difficulty` e `hard_negative_mask`;
- pseudo-copas de instancia para a linha `distance-transform`.

Tambem foram implementadas tres politicas de labels:

- `field_center`
- `apex_snapped`
- `mixed_center_apex`

### 4.4 Pre-processamento explicito

Uma fase posterior explicitou o pre-processamento que antes estava mais distribuido:

- correcao de shift RGB-LiDAR;
- saneamento da referencia;
- persistencia de `chm_features`;
- resumos de QA por parcela.

Nessa fase, o residual medio de co-registro caiu de aproximadamente `6.70 px` para `0.50 px`, com `9/10` parcelas ficando em residual `<= 1 px`.

### 4.5 Pipeline fiel do paper

Na familia `paper_faithful`, o pre-processamento foi separado do pipeline principal e seguiu o release/publicacao:

- deduplicacao a `1 mm`;
- uso dos pontos de solo `class 2`;
- interpolacao de terreno;
- normalizacao de altura;
- matching guloso deterministico ordenado por distancia;
- sem uso de RGB no baseline fiel canonico.

## 5. Protocolos experimentais e matching

## 5.1 Protocolos historicos

Antes do congelamento do benchmark, foram usados varios regimes:

| Regime | Uso principal | Observacao |
|---|---|---|
| `LOPO` pratico | validacao inicial do pipeline | curto, nao conclusivo para ciencia final |
| `basic split` longo | treino mais profundo em 1 split | comparavel apenas dentro do mesmo split |
| `3` basic splits | comparativo mais robusto entre alguns hibridos | base do `Hybrid v2` |
| validacao simples `3` seeds | desenvolvimento rapido | nao usa teste para triagem |
| validacao fixa `plot_01+plot_06` | PointNet estrito fiel | regime especializado |
| benchmark fiel `10` parcelas | metodos geometricos e treino local leve | referencia para familia paper_faithful |

## 5.2 Benchmark oficial V2

O projeto terminou com dois protocolos oficiais congelados.

### Protocolo A

- `protocol_id = protocol_a_v2_3split`
- suite comparavel de `3` splits em nivel de parcela
- foco em metodos aprendidos, refinadores locais e ablations
- metricas obrigatorias:
  - `precision`
  - `recall`
  - `f1`
  - `location_mae_m`
  - `height_mae_m`
  - `tp`
  - `fp`
  - `fn`

### Protocolo B

- `protocol_id = protocol_b_v2_faithful10`
- benchmark fiel nas `10` parcelas
- foco em Stage-1 geometrico, metodos sem treino e treino local leve
- mesmas metricas obrigatorias do Protocolo A

### Regras V2

- toda run agregavel deve carregar `protocol_id`;
- toda run agregavel deve ter `experiment_manifest.json`;
- o naming canonico e `family.stage1.stage2.labels.protocol.seed`;
- as tabelas oficiais devem preservar o `protocol_id`;
- runs historicas continuam uteis como contexto, mas nao entram em tabelas oficiais V2 sem migracao explicita.

## 5.3 Matching

Dois regimes de matching coexistem no repositorio.

### Matching do pipeline principal

Usado em `src/lidaragrosat/evaluation/matching.py`:

- matching um-para-um entre predicoes e GT;
- restricoes espaciais e de altura;
- usado com agregacao de metricas e estatistica por parcela.

### Matching fiel do paper

Usado na familia `paper_faithful`:

- matching guloso deterministico ordenado por distancia 2D crescente;
- `max_dist = 5.0 m`;
- `max_height_diff = 3.0 m` quando altura existe;
- sem Hungarian.

## 6. Suite de testes automatizados

Os testes automatizados cobrem contrato, regressao e sanidade dos principais blocos.

### 6.1 Benchmark e metadados

- `tests/test_benchmark_protocol_v2.py`
  - serializacao de `protocol_id`
  - separacao correta entre protocolos
  - construcao dos splits fixos do protocolo A
- `tests/test_experiment_manifest.py`
  - schema do manifesto
  - naming canonic
  - roundtrip de leitura/escrita
  - agregacao por metadados

### 6.2 Detector, losses e inferencia

- `tests/test_detector_advanced.py`
  - ramos auxiliares do detector
  - ramo de `CHM`
  - `deep supervision`
- `tests/test_inference_advanced.py`
  - sliding window com heads auxiliares
  - montagem de `chm_features`
  - integracao com `train_hybrid`
- `tests/test_losses_advanced.py`
  - composicao da `DetectionLoss`
  - termos de distancia, peakness, species e hard negatives

### 6.3 Targets e linha de instancia

- `tests/test_targets.py`
  - construcao basica do heatmap de centro
- `tests/test_targets_advanced.py`
  - apex snapping
  - species targets
  - mapas auxiliares
  - sigma adaptativo
  - hard negatives
- `tests/test_instance_targets.py`
  - alinhamento de `distance_transform`, `seedness`, `pseudo_crown_mask`, `star_convex_rays`
- `tests/test_instance_baseline.py`
  - decode do baseline de instancia

### 6.4 Matching, splits e familia paper_faithful

- `tests/test_matching.py`
  - respeito a restricoes espaciais e de altura
- `tests/test_splits.py`
  - integridade dos splits LOPO
- `tests/test_paper_faithful_adaptive.py`
  - comportamento do `adaptive_local_maxima`
- `tests/test_paper_faithful_object_fusion.py`
  - fusao de candidatos por segmento RGB

### 6.5 Stage-1 e Stage-2 locais

- `tests/test_stage1_api.py`
  - contrato unificado de Stage-1
  - coordenadas validas
  - segmentos e diagnosticos
- `tests/test_stage2_local_refiners.py`
  - `LocalProposalDataset`
  - forward de refinadores `raster`, `point`, `hybrid`
  - NMS/pos-processamento de Stage-2

Ao final do ciclo V2, as verificacoes finais recorrentes foram:

- `python3 -m compileall src scripts tests`
- `PYTHONPATH=src:. ./.venv/bin/pytest -q ...`

com subsets relevantes dos testes acima passando apos cada pacote de mudancas.

## 7. Catalogo completo de runs e ablations

Esta secao organiza todas as familias de runs efetivamente executadas.

## 7.1 Pipeline multimodal principal: detector center-based raster

### Run 1: LOPO pratico

Objetivo:

- validar o pipeline inteiro em `LOPO` com baixo custo;
- comparar `baseline_local_maxima` com modelos unimodais e multimodais.

Variantes executadas:

- `baseline_local_maxima`
- `lidar_only`
- `rgb_only`
- `early_fusion`
- `multimodal_mid_fusion`
- `multimodal_late_fusion`

Artefatos:

- `outputs/metrics/method_summary.csv`
- `outputs/metrics/metrics_by_plot.csv`
- `outputs/predictions/*`

Resultados agregados:

| Metodo | Precision | Recall | F1 |
|---|---:|---:|---:|
| `baseline_local_maxima` | 0.3557 | 0.4889 | 0.4063 |
| `multimodal_mid_fusion` | 0.1243 | 0.3071 | 0.1539 |
| `multimodal_late_fusion` | 0.1044 | 0.2640 | 0.1455 |
| `early_fusion` | 0.0941 | 0.2974 | 0.1372 |
| `rgb_only` | 0.1010 | 0.2270 | 0.1343 |
| `lidar_only` | 0.1487 | 0.2662 | 0.0883 |

Conclusao:

- serviu para validar a infraestrutura;
- nao serviu como evidencia forte contra a arquitetura deep, porque o treino ficou curto demais.

### Run 2: basic split longo

Objetivo:

- testar treino longo em um split unico de parcela.

Split:

- treino: `plot_04,01,08,09,07,02,06,03`
- validacao: `plot_10`
- teste: `plot_05`

Artefatos:

- `outputs_basic_split/metrics/method_summary.csv`
- `outputs_basic_split/metrics/metrics_by_plot.csv`

Resultados:

| Metodo | Precision | Recall | F1 |
|---|---:|---:|---:|
| `baseline_local_maxima` | 0.2729 | 0.5865 | 0.3725 |
| `multimodal_mid_fusion` | 0.2590 | 0.3798 | 0.3080 |
| `early_fusion` | 0.0659 | 0.4904 | 0.1162 |
| `rgb_only` | 0.0401 | 0.2452 | 0.0690 |
| `multimodal_late_fusion` | 0.0346 | 0.1683 | 0.0574 |
| `lidar_only` | 0.0008 | 0.4856 | 0.0016 |

Conclusao:

- `multimodal_mid_fusion` aproximou a baseline;
- `lidar_only` colapsou por explosao de `FP`.

### Run 3: ablation basic de decode e regime

Objetivo:

- separar efeito de calibracao do decode do efeito do re-treino.

Variantes:

- `multimodal_mid_fusion_redecode`
- `multimodal_mid_fusion_roi`
- `lidar_only_redecode`
- `lidar_only_roi`

Artefatos:

- `outputs_ablation_basic/metrics/method_summary.csv`
- `outputs_ablation_basic/diagnostics/*.json`

Resultados:

| Metodo | Precision | Recall | F1 |
|---|---:|---:|---:|
| `multimodal_mid_fusion_redecode` | 0.2211 | 0.5433 | 0.3143 |
| `multimodal_mid_fusion_roi` | 0.1254 | 0.5577 | 0.2048 |
| `lidar_only_redecode` | 0.1022 | 0.2885 | 0.1509 |
| `lidar_only_roi` | 0.1789 | 0.4567 | 0.2571 |

Conclusao:

- o decode era um gargalo enorme;
- boa parte do ganho do eixo LiDAR veio da redecodificacao/calibracao.

### Run 4: ablation completa do detector e hibrido inicial

Objetivo:

- testar `hard negatives`, `peakness`, `species head`, `deep supervision`, `adaptive NMS` e um refinador hibrido de propostas.

Variantes principais:

- `multimodal_mid_fusion_full`
- `hybrid_proposal_refinement_multimodal`

Artefatos:

- `outputs_complete_ablation/metrics/method_summary.csv`

Resultados:

| Metodo | Precision | Recall | F1 |
|---|---:|---:|---:|
| `multimodal_mid_fusion_full` | 0.2007 | 0.5433 | 0.2931 |
| `hybrid_proposal_refinement_multimodal` | 0.1419 | 0.6394 | 0.2323 |

Conclusao:

- o detector denso ganhou `recall`, mas nao converteu isso em melhor `F1`;
- o hibrido inicial mostrou potencial de cobertura, com `FP` ainda alto.

### Run 5: Hybrid v2 em 3 splits comparaveis

Objetivo:

- medir se `proposal + refinement` supera `mid_fusion` de forma consistente.

Artefatos:

- `outputs_suite_aggregate/suite_method_summary.csv`
- `outputs_suite_aggregate/suite_metrics_by_plot.csv`

Resultados:

| Metodo | Precision mean | Recall mean | F1 mean |
|---|---:|---:|---:|
| `baseline_local_maxima` | 0.3410 | 0.4797 | 0.3785 |
| `hybrid_proposal_refinement_multimodal_v2` | 0.2930 | 0.3933 | 0.3278 |
| `multimodal_mid_fusion` | 0.2059 | 0.3634 | 0.2493 |

Conclusao:

- `Hybrid v2` foi a primeira extensao a superar `mid_fusion` em `3/3` splits;
- ainda ficou abaixo da baseline classica.

### Run 6: baseline como Stage-1 puro

Objetivo:

- usar propostas do baseline como estagio 1 fixo para um refinador.

Artefatos:

- `outputs_baseline_stage_basic/metrics/method_summary.csv`
- `outputs_baseline_stage_seed13/*`
- `outputs_baseline_stage_seed21/*`
- `outputs_baseline_stage_seed34/*`

Resultado-chave no split canonico:

- `hybrid_baseline_stage_refiner`: `P=0.1762`, `R=0.6394`, `F1=0.2762`

Resultados em validacao simples:

- `hybrid_baseline_stage_refiner`: `F1=0.3438`

Conclusao:

- forte ganho de `recall`;
- perda de `precision` suficiente para manter o metodo abaixo da baseline.

### Run 7: detector tri-modal com ramo explicito de CHM

Objetivo:

- testar ramo dedicado de `CHM` com supervisao geometrica enriquecida.

Artefatos:

- `outputs_chm_context_aggregate/suite_method_summary.csv`

Resultado:

- `multimodal_chm_context_interactive`: `F1=0.1943` na suite de `3` splits

Conclusao:

- nao melhorou o benchmark;
- reforcou que o caminho mais promissor estava nos hibridos baseados em propostas.

### Run 8: validacao simples para desenvolvimento

Objetivo:

- substituir `LOPO` por um modo mais barato para triagem rapida.

Artefatos:

- `outputs_dev_validation_simple/validation_summary.csv`
- `outputs_dev_validation_preproc_explicit/validation_summary.csv`

Resultados sem pre-processamento explicito adicional:

| Metodo | Precision mean | Recall mean | F1 mean |
|---|---:|---:|---:|
| `baseline_local_maxima` | 0.3698 | 0.4896 | 0.4153 |
| `hybrid_baseline_stage_refiner` | 0.2428 | 0.6017 | 0.3438 |
| `hybrid_proposal_refinement_multimodal_v2` | 0.3166 | 0.3618 | 0.3331 |
| `multimodal_mid_fusion` | 0.2493 | 0.4053 | 0.3066 |
| `multimodal_chm_context_interactive` | 0.1902 | 0.3861 | 0.2394 |

Resultados com pre-processamento explicito:

| Metodo | Precision mean | Recall mean | F1 mean |
|---|---:|---:|---:|
| `baseline_local_maxima` | 0.3698 | 0.4896 | 0.4153 |
| `hybrid_baseline_stage_refiner` | 0.2441 | 0.5799 | 0.3422 |
| `hybrid_proposal_refinement_multimodal_v2` | 0.2612 | 0.4707 | 0.3343 |
| `multimodal_mid_fusion` | 0.2579 | 0.4445 | 0.3235 |
| `multimodal_chm_context_interactive` | 0.2325 | 0.1809 | 0.1407 |

Conclusao:

- validacao simples foi util para triagem;
- o pre-processamento explicito ajudou mais `multimodal_mid_fusion` do que os hibridos.

## 7.2 Auditorias e linha auxiliar de labels

### Auditoria centro-apice

Artefatos:

- `outputs/analysis/apex_alignment/apex_offset_summary.csv`

Resultado:

- media `0.8537 m`
- mediana `0.9019 m`
- `P90 = 1.2298 m`

Conclusao:

- o ruido geometrico centro-apice e real e relevante;
- isso justifica as politicas `apex_snapped` e `mixed_center_apex`.

## 7.3 Familia paper_faithful: benchmark geometrico fiel

### Run 9: baseline canonico fiel

Artefatos:

- `outputs_faithful_paper/canonical_summary.json`
- `outputs_faithful_paper/canonical_per_plot.csv`

Configuracao canonica:

- `window_size = 2.0 m`
- `height_threshold = 3.0 m`
- `max_dist = 5.0 m`
- `max_height_diff = 3.0 m`

Resultado:

- `Precision = 0.2967`
- `Recall = 0.5830`
- `F1 = 0.3933`

### Run 10: Optuna sobre o baseline fiel

Artefatos:

- `outputs_faithful_paper_optuna/best_summary.json`
- `outputs_faithful_paper_optuna/best_per_plot.csv`

Melhor trial:

- `window_size_m = 2.4`
- `height_threshold_m = 5.0`

Resultado:

- `Precision = 0.4253`
- `Recall = 0.4542`
- `F1 = 0.4393`

### Run 11: local maxima com janela adaptativa

Artefatos:

- `outputs_faithful_paper_adaptive_optuna/best_summary.json`

Melhor configuracao:

- `a = 0.035`
- `b = 2.5`
- `height_threshold = 5.0`

Resultado:

- `Precision = 0.4553`
- `Recall = 0.5755`
- `F1 = 0.5084`

### Run 12: CHM + watershed fiel

Artefatos:

- `outputs_faithful_paper_watershed_optuna/best_summary.json`
- `outputs_faithful_paper_watershed_optuna/best_per_plot.csv`

Melhor configuracao:

- `resolution_m = 0.1`
- `gaussian_sigma_m = 0.2`
- `min_distance_m = 0.8`
- `min_height_m = 5.9`
- `min_segment_area_m2 = 6.0`
- `hole_fill_radius_m = 0.3`
- `compactness = 0.0075`

Resultado:

- `Precision = 0.5483`
- `Recall = 0.5230`
- `F1 = 0.5354`

Conclusao:

- foi o melhor baseline fiel classico das `10` parcelas;
- superou o `local_maxima` otimizado em `10/10` parcelas.

### Run 13: object fusion com Felzenszwalb

Artefatos:

- `outputs_faithful_paper_object_fusion_optuna/best_summary.json`

Resultado:

- `Precision = 0.4978`
- `Recall = 0.5364`
- `F1 = 0.5164`

Conclusao:

- superou `local_maxima` fixo;
- nao superou `CHM + watershed`.

### Run 14: object fusion com SLIC multiescala

Artefatos:

- `outputs_faithful_paper_object_fusion_slic_optuna/best_summary.json`
- `outputs_faithful_paper_object_fusion_slic_optuna/comparison_vs_existing_global.csv`

Resultado:

- `Precision = 0.5731`
- `Recall = 0.4939`
- `F1 = 0.5306`

Comparacao:

- melhor que o `Felzenszwalb` antigo (`+0.0142` de `F1`);
- ainda abaixo do `watershed` fiel antigo (`-0.0048`) e bem abaixo do `watershed` do protocolo B V2.

### Run 15: hibrido watershed + proposals

Artefatos:

- `outputs_faithful_hybrid_watershed_proposals/*`

Melhor configuracao de validacao:

- `resolution_m = 0.2`
- `smooth_sigma_px = 1.5`
- `min_distance_px = 3`
- `min_segment_area_m2 = 6.0`
- `proposal_window_m = 0.8`

Resultados:

| Escopo | Precision | Recall | F1 |
|---|---:|---:|---:|
| validacao `plot_01 + plot_06` | 0.5684 | 0.5557 | 0.5593 |
| `10` parcelas | 0.5353 | 0.5201 | 0.5230 |

Conclusao:

- praticamente empatou com o `watershed` puro na validacao;
- ficou um pouco pior no agregado global;
- o diagnostico apontou `under-segmentation` como gargalo.

## 7.4 Familia point-based fiel

### Run 16: point cloud refiner local

Artefatos:

- `outputs_faithful_point_cloud_refiner/summary_by_source_mode.csv`
- `outputs_faithful_point_cloud_refiner/metrics_by_split.csv`

Resultados:

| Modo de proposta | Precision mean | Recall mean | F1 mean |
|---|---:|---:|---:|
| `union` | 0.4454 | 0.5847 | 0.5025 |
| `adaptive` | 0.4487 | 0.5386 | 0.4865 |

Conclusao:

- melhorou sobre o baseline fixo antigo;
- nao superou `watershed` nem `object fusion`.

### Run 17: PointNet estrito fiel ao Pretendido

Artefatos:

- `outputs_faithful_pointnet_strict/history.csv`
- `outputs_faithful_pointnet_strict/summary_by_partition.csv`
- `outputs_faithful_pointnet_strict/proposal_summary_by_partition.csv`
- `outputs_faithful_pointnet_strict/pretendido_checklist.csv`

Configuracao central:

- propostas por `local_maxima` com `window_size=0.8 m`
- `height_threshold = 3.0 m`
- esfera de raio `2.5 m`
- `512` pontos
- treino `8` parcelas, validacao `plot_01 + plot_06`
- `40` epocas

Resultado do gerador de propostas na validacao:

- `Precision = 0.0369`
- `Recall = 0.9280`
- `F1 = 0.0707`

Resultado final do PointNet:

- `Precision = 0.2522`
- `Recall = 0.7869`
- `F1 = 0.3783`

Conclusao:

- o regime de altissimo recall das propostas foi atingido;
- o PointNet puro nao filtrou `FP` o bastante.

### Run 18: melhorias sobre o PointNet estrito

Artefatos:

- `outputs_faithful_pointnet_improvements/variant_summary.csv`
- `outputs_faithful_pointnet_improvements/comparison_vs_best.csv`

Variantes:

- `strict_redecode_wide`
- `strict_focal_xyz`
- `strict_multifeature_bce`
- `strict_multifeature_focal`

Resultados:

| Variante | Precision mean | Recall mean | F1 mean |
|---|---:|---:|---:|
| `strict_focal_xyz` | 0.5709 | 0.5347 | 0.5505 |
| `strict_redecode_wide` | 0.5805 | 0.5166 | 0.5444 |
| `strict_multifeature_focal` | 0.5134 | 0.5493 | 0.5302 |
| `strict_multifeature_bce` | 0.5278 | 0.5306 | 0.5262 |

Conclusoes:

- o maior ganho veio de `threshold + NMS`;
- `Focal Loss` ajudou pouco alem disso;
- as `multifeatures` nao ajudaram;
- o melhor PointNet fiel ficou muito perto do melhor baseline fiel em validacao.

## 7.5 TicketFinal V2: benchmark oficial e matriz principal

Esta foi a fase em que o benchmark foi congelado e o repositorio recebeu a API unificada de `Stage-1`, o dataset local por proposta/segmento, os refinadores locais do protocolo oficial e a linha de relatorios V2.

### Run 19: Stage-1 oficial do protocolo B

Artefatos:

- `outputs_ticketfinal_stage1/metrics/watershed_summary.csv`
- `outputs_ticketfinal_stage1/metrics/adaptive_watershed_summary.csv`
- `outputs_ticketfinal_stage1/metrics/watershed_diagnostics.csv`

Resultados:

| Metodo | Precision | Recall | F1 | TP | FP | FN | Protocol |
|---|---:|---:|---:|---:|---:|---:|---|
| `watershed` | 0.6025 | 0.6854 | 0.6332 | 2464 | 1598 | 1138 | `protocol_b_v2_faithful10` |
| `adaptive_watershed` | 0.6549 | 0.4898 | 0.4943 | 1707 | 1073 | 1895 | `protocol_b_v2_faithful10` |

Conclusao:

- `watershed` fixo tornou-se o melhor resultado global do projeto no benchmark fiel;
- `adaptive_watershed` nao superou o fixo.

### Run 20: Stage-1-only no protocolo A

Artefatos:

- `outputs_ticketfinal_v2/metrics/stage1_only_summary.csv`

Resultados:

| Metodo | Precision | Recall | F1 |
|---|---:|---:|---:|
| `local_maxima` | 0.4552 | 0.2858 | 0.3492 |
| `watershed_fixo` | 0.3727 | 0.3544 | 0.3392 |
| `adaptive_watershed` | 0.3510 | 0.1580 | 0.1854 |

Conclusao:

- no protocolo A, `local_maxima` continuou forte como baseline oficial;
- `watershed_fixo` ficou muito competitivo;
- `adaptive_watershed` ficou claramente abaixo.

### Run 21: matriz principal Stage-1 x Stage-2 x label policy

Artefatos:

- `outputs_ticketfinal_v2/metrics/main_matrix_results.csv`
- `outputs_ticketfinal_v2/metrics/label_policy_ablation.csv`
- `outputs_ticketfinal_v2/reports/gain_by_stage.csv`
- `outputs_ticketfinal_v2/reports/gain_by_label.csv`
- `outputs_ticketfinal_v2/reports/gain_by_plot.csv`
- `outputs_ticketfinal_v2/reports/error_modes_stage2.csv`

Resultados principais da matriz:

| Stage-1 | Stage-2 | Labels | F1 |
|---|---|---|---:|
| `local_maxima` | `none` | `field_center` | 0.3492 |
| `watershed_fixo` | `none` | `field_center` | 0.3392 |
| `watershed_fixo` | `point` | `field_center` | 0.3460 |
| `watershed_fixo` | `point` | `apex_snapped` | 0.3438 |
| `watershed_fixo` | `point` | `mixed_center_apex` | 0.3439 |
| `watershed_fixo` | `raster` | `field_center` | 0.3420 |
| `watershed_fixo` | `raster` | `apex_snapped` | 0.3454 |
| `watershed_fixo` | `raster` | `mixed_center_apex` | 0.3442 |
| `watershed_fixo` | `hybrid` | `field_center` | 0.3441 |
| `watershed_fixo` | `hybrid` | `apex_snapped` | 0.3438 |
| `watershed_fixo` | `hybrid` | `mixed_center_apex` | 0.3449 |
| `adaptive_watershed` | `none` | `field_center` | 0.1854 |
| `adaptive_watershed` | `point` | `field_center` | 0.1787 |
| `adaptive_watershed` | `point` | `apex_snapped` | 0.1876 |
| `adaptive_watershed` | `point` | `mixed_center_apex` | 0.1873 |
| `adaptive_watershed` | `raster` | `field_center` | 0.1875 |
| `adaptive_watershed` | `raster` | `apex_snapped` | 0.1858 |
| `adaptive_watershed` | `raster` | `mixed_center_apex` | 0.1819 |
| `adaptive_watershed` | `hybrid` | `field_center` | 0.1785 |
| `adaptive_watershed` | `hybrid` | `apex_snapped` | 0.1888 |
| `adaptive_watershed` | `hybrid` | `mixed_center_apex` | 0.1862 |

Ganhos medios de Stage-2 sobre Stage-1:

| Stage-1 | Stage-2 | Delta F1 |
|---|---|---:|
| `watershed_fixo` | `point` | +0.0053 |
| `watershed_fixo` | `hybrid` | +0.0050 |
| `watershed_fixo` | `raster` | +0.0031 |
| `adaptive_watershed` | `point` | -0.0009 |
| `adaptive_watershed` | `hybrid` | -0.0009 |
| `adaptive_watershed` | `raster` | -0.0004 |

Conclusoes:

- `point` foi o melhor refinador da matriz oficial inicial;
- o ganho aprendido sobre `watershed_fixo` puro existiu, mas foi pequeno;
- `adaptive_watershed` nao sustentou refinamento competitivo.

### Run 22: ablation de labels

Artefatos:

- `outputs_ticketfinal_v2/metrics/label_policy_ablation.csv`

Resumo:

- `point`: `field_center` foi melhor que `apex_snapped` e `mixed`
- `hybrid`: `mixed_center_apex` foi levemente melhor em `F1`, mas pior em `location_mae`
- `raster`: `apex_snapped` e `mixed` ficaram um pouco acima de `field_center`

Conclusao:

- corrigir centro-apice nao trouxe ganho robusto e universal no protocolo A;
- o efeito dependeu da familia de refinador.

### Run 23: baseline `distance-transform + watershed`

Artefatos:

- `outputs_ticketfinal_v2/metrics/distance_transform_watershed_summary.csv`

Resultado:

- `F1 = 0.0`

Conclusao:

- a linha de instancia/copa ficou funcional em codigo, mas o baseline inicial nao produziu sinal competitivo.

### Run 24: ablation oficial de RGB

Artefatos:

- `outputs_ticketfinal_v2/metrics/rgb_contribution_ablation.csv`

Resultados:

| Modelo | Entrada | F1 |
|---|---|---:|
| `raster` | `lidar_chm` | 0.3449 |
| `raster` | `rgb_chm` | 0.3443 |
| `raster` | `rgb_lidar_chm` | 0.3468 |
| `hybrid` | `lidar_chm` | 0.3431 |
| `hybrid` | `rgb_chm` | 0.3516 |
| `hybrid` | `rgb_lidar_chm` | 0.3443 |

Conclusao:

- `RGB` agregou valor real no refinamento local;
- o melhor resultado novo do protocolo A passou a ser `watershed_fixo + hybrid + rgb_chm`, `F1 = 0.3516`.

### Run 25: ablation oficial de CHM

Artefatos:

- `outputs_ticketfinal_v2/metrics/chm_contribution_ablation.csv`

Resultados:

| Modelo | Modo CHM | F1 |
|---|---|---:|
| `raster` | `no_chm` | 0.3438 |
| `raster` | `with_chm` | 0.3456 |
| `raster` | `with_chm_geom` | 0.3435 |
| `hybrid` | `no_chm` | 0.3447 |
| `hybrid` | `with_chm` | 0.3403 |
| `hybrid` | `with_chm_geom` | 0.3452 |

Conclusao:

- `CHM` explicito no Stage-2 ajudou pouco e de forma inconsistente;
- o ganho mais robusto do protocolo A veio do `RGB`, nao do `CHM` adicional no refinador.

## 7.6 TicketFinal V2: extensoes apos a matriz oficial

### Run 26: proximas opcoes de alto retorno

Artefatos:

- `outputs_ticketfinal_next_options/metrics/next_options_summary.csv`

Variantes:

- `watershed_fixo.hybrid_rgb_strong`
- `watershed_fixo.hybrid_rgb_chm.recalibrated`
- `watershed_fixo.point_hybrid_ensemble`
- `watershed_fixo.segment_stacker_logreg`

Resultados:

| Metodo | Precision | Recall | F1 |
|---|---:|---:|---:|
| `watershed_fixo.hybrid_rgb_chm.recalibrated` | 0.3930 | 0.3471 | 0.3507 |
| `watershed_fixo.point_hybrid_ensemble` | 0.3828 | 0.3410 | 0.3435 |
| `watershed_fixo.hybrid_rgb_strong` | 0.3802 | 0.3384 | 0.3409 |
| `watershed_fixo.segment_stacker_logreg` | 0.3797 | 0.3326 | 0.3379 |

Conclusao:

- nenhuma dessas variantes superou o melhor atual `watershed_fixo + hybrid + rgb_chm = 0.3516`;
- a melhor foi a simples recalibracao do metodo ja vencedor.

### Run 27: faixa B com GNN e split predictor

Artefatos:

- `outputs_ticketfinal_faixa_b/metrics/gnn_split_summary.csv`

Resultados:

| Metodo | Precision | Recall | F1 |
|---|---:|---:|---:|
| `watershed_fixo.split_predictor` | 0.3882 | 0.3455 | 0.3480 |
| `watershed_fixo.segment_gnn` | 0.3809 | 0.3453 | 0.3422 |

Conclusao:

- `split_predictor` foi a melhor extensao nova da faixa B;
- ainda ficou abaixo do melhor metodo do protocolo A;
- `segment_gnn` nao compensou nesta formulacao simples.

## 8. Hierarquia atual dos resultados

Como o projeto tem mais de um benchmark, a hierarquia deve ser lida em dois niveis.

### 8.1 Melhor resultado global por regime

| Regime | Melhor metodo | F1 |
|---|---|---:|
| `protocol_b_v2_faithful10` | `watershed` | 0.6332 |
| validacao PointNet fiel | `strict_focal_xyz` | 0.5505 |
| benchmark fiel antigo `10` parcelas | `CHM + watershed` | 0.5354 |
| object fusion fiel | `SLIC multiescala` | 0.5306 |
| suite `3` splits historica | `baseline_local_maxima` | 0.3785 |
| melhor deep do pipeline principal | `hybrid_proposal_refinement_multimodal_v2` | 0.3278 |

### 8.2 Melhor resultado oficial do protocolo A

| Ordem | Metodo | F1 |
|---|---|---:|
| 1 | `watershed_fixo + hybrid + rgb_chm` | 0.3516 |
| 2 | `watershed_fixo + hybrid_rgb_chm.recalibrated` | 0.3507 |
| 3 | `local_maxima` | 0.3492 |
| 4 | `watershed_fixo + split_predictor` | 0.3480 |
| 5 | `watershed_fixo + raster + rgb_lidar_chm` | 0.3468 |
| 6 | `watershed_fixo + point + field_center` | 0.3460 |

## 9. Respostas atuais as perguntas centrais do plano

1. `adaptive_watershed` supera `watershed` fixo?  
Nao. Ficou claramente abaixo tanto no protocolo A quanto no protocolo B.

2. O refinamento local aprendido agrega ganho real sobre o Stage-1 puro?  
Sim, mas no protocolo A o ganho liquido foi pequeno. O melhor delta medio sobre `watershed_fixo` puro ficou em torno de `+0.005`.

3. O melhor refinador e `raster`, `point` ou `hybrid`?  
Na matriz oficial inicial, `point` foi o melhor. Depois das ablations de modalidades, o melhor numero do protocolo A passou a ser do `hybrid` com `rgb_chm`.

4. Corrigir centro-apice melhora o resultado?  
Nao de forma robusta e universal. O efeito foi pequeno e dependente da familia do refinador.

5. `RGB` agrega valor real no estagio local?  
Sim. A melhor evidencia foi `watershed_fixo + hybrid + rgb_chm = 0.3516`, acima das variantes sem RGB no mesmo protocolo A.

6. Vale migrar do problema de centro para o problema de instancia/copa?  
Ainda nao ha evidencia positiva com a implementacao atual. O baseline `distance-transform + watershed` ficou em `F1 = 0.0`, entao a linha continua experimental.

## 10. Conclusao global

O projeto convergiu para um padrao bastante claro.

- O maior ganho estrutural do repositorio veio da geometria forte de `CHM + watershed`.
- No benchmark fiel, `watershed` fixo tornou-se o melhor resultado global do projeto (`F1 = 0.6332`).
- No protocolo oficial de metodos aprendidos, o melhor resultado ficou com `watershed_fixo + hybrid + rgb_chm` (`F1 = 0.3516`).
- A linha principal de detectores raster densos ficou consistentemente abaixo dos hibridos baseados em propostas.
- A linha PointNet fiel so ficou competitiva depois de forte calibracao de `threshold + NMS`, e mesmo assim nao superou o melhor baseline geometrico.
- `RGB` agregou valor real no refinamento local; `CHM` adicional no Stage-2 agregou pouco; `adaptive_watershed` nao entregou; e a linha de instancia/copa ainda nao justificou migracao de formulacao.

Em sintese, a melhor narrativa sustentada pelos experimentos e:

- usar `watershed` fixo como Stage-1 geometrico forte;
- usar refinamento local leve, preferencialmente `hybrid` com `rgb_chm`, quando o objetivo for o protocolo A;
- usar o benchmark fiel de `10` parcelas para comparar familias geometricas e treino local leve;
- tratar todas as runs historicas fora do V2 como contexto util, mas nao como referencia oficial unica.

## 11. Mapa rapido das saidas

### Documentos consolidados

- `ArquiteturaImplementadaAtual.md`
- `Resultados.md`
- `RelatorioCompleto.md`
- `ResultadosFaithfulPaper.md`
- `ResultadosFaithfulWatershed.md`
- `ResultadosFaithfulAdaptiveLocalMaxima.md`
- `ResultadosFaithfulObjectFusion.md`
- `ResultadosFaithfulPointCloudRefiner.md`
- `ResultadosFaithfulPointNetStrict.md`
- `ResultadosFaithfulPointNetImprovements.md`
- `ResultadosFaithfulHybridWatershedProposals.md`

### Saidas centrais do benchmark oficial

- `outputs_ticketfinal_v2/metrics/main_matrix_results.csv`
- `outputs_ticketfinal_v2/metrics/rgb_contribution_ablation.csv`
- `outputs_ticketfinal_v2/metrics/chm_contribution_ablation.csv`
- `outputs_ticketfinal_stage1/metrics/watershed_summary.csv`
- `outputs_ticketfinal_stage1/metrics/adaptive_watershed_summary.csv`
- `outputs_ticketfinal_next_options/metrics/next_options_summary.csv`
- `outputs_ticketfinal_faixa_b/metrics/gnn_split_summary.csv`

### Saidas centrais da familia fiel

- `outputs_faithful_paper_optuna/best_summary.json`
- `outputs_faithful_paper_watershed_optuna/best_summary.json`
- `outputs_faithful_paper_object_fusion_optuna/best_summary.json`
- `outputs_faithful_paper_object_fusion_slic_optuna/best_summary.json`
- `outputs_faithful_paper_adaptive_optuna/best_summary.json`
- `outputs_faithful_point_cloud_refiner/summary_by_source_mode.csv`
- `outputs_faithful_pointnet_strict/summary_by_partition.csv`
- `outputs_faithful_pointnet_improvements/variant_summary.csv`

