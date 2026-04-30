# Arquitetura Implementada Atual

## 1. Escopo científico efetivamente implementado

O repositório implementa um pipeline experimental completo para detecção individual de árvores (`Individual Tree Detection`, ITD) em florestas mistas densas por fusão multimodal `UAV LiDAR + RGB`, com formulação center-based em grade raster, baseline clássica por máximos locais em `CHM`, modelos unimodais e multimodais em `PyTorch`, inferência por janelas deslizantes, matching bipartido um-para-um e agregação de métricas/estatísticas.

O sistema foi implementado para:

- preparação e harmonização multimodal por parcela;
- geração de `DTM`, `CHM` e tensor LiDAR estrutural multicanal;
- construção de alvos gaussianos de centro, offsets subpixel e altura opcional;
- treino de variantes `RGB-only`, `LiDAR-only`, `early fusion`, `mid fusion` e `late fusion`;
- avaliação em dois regimes:
  - `LOPO` (`Leave-One-Plot-Out`);
  - `basic split` em nível de parcela.

O código atual já foi executado no dataset Dubrovin real baixado do Kaggle (`sentinel3734/tree-detection-lidar-rgb`), tanto no regime `LOPO` curto/prático quanto no regime `basic split` com `120` épocas.

## 2. Organização lógica do sistema

O sistema está organizado em sete camadas:

1. Configuração experimental.
2. Ingestão e pré-processamento geoespacial.
3. Construção de targets e dataset para treino.
4. Modelagem deep learning e baseline clássica.
5. Treino e inferência.
6. Avaliação quantitativa e análise científica.
7. Entry points de execução.

Essas camadas residem principalmente em:

- `src/lidaragrosat/config.py`
- `src/lidaragrosat/data/*`
- `src/lidaragrosat/models/*`
- `src/lidaragrosat/training/*`
- `src/lidaragrosat/inference/*`
- `src/lidaragrosat/evaluation/*`
- `src/lidaragrosat/analysis/*`
- `scripts/*`

## 3. Configuração experimental

### 3.1 Arquivo central de configuração

O módulo `src/lidaragrosat/config.py` define a estrutura experimental via dataclasses:

- `PathConfig`
- `DataConfig`
- `PreprocessConfig`
- `AugmentationConfig`
- `ModelConfig`
- `TrainingConfig`
- `EvaluationConfig`
- `AnalysisConfig`
- `ExperimentConfig`

### 3.2 Funções de configuração

As principais funções/métodos são:

- `_merge_dicts(base, override)`: mescla YAMLs hierárquicos.
- `ExperimentConfig.from_files(*config_paths)`: compõe múltiplos arquivos de configuração.
- `ExperimentConfig.resolve_path(path_value)`: resolve caminhos relativos ao `root_dir`.
- `save_config(config, destination)`: persiste a configuração efetivamente usada.

### 3.3 Modos de split implementados

`DataConfig` suporta:

- `split_mode="lopo"`
- `split_mode="basic"`

Para `basic split`, existem os parâmetros:

- `basic_train_fraction`
- `basic_val_fraction`
- `basic_split_seed`

### 3.4 Configurações entregues

Os principais overrides disponíveis são:

- `configs/base.yaml`
- `configs/lidar_only.yaml`
- `configs/rgb_only.yaml`
- `configs/early_fusion.yaml`
- `configs/multimodal_mid.yaml`
- `configs/multimodal_late.yaml`
- `configs/practical_run.yaml`
- `configs/basic_split.yaml`
- `configs/batch1.yaml`

## 4. Ingestão de dados e padronização de entrada

### 4.1 Inventário e descoberta do dataset

O módulo `src/lidaragrosat/data/inventory.py` implementa:

- `PlotInventory`
- `discover_dataset(raw_dir, explicit_plot_ids=None)`
- `save_inventory(inventories, destination)`

O layout esperado por parcela é:

```text
data/raw/plot_xx/
  lidar.las|laz
  rgb.tif
  trees_reference.csv|gpkg|geojson|...
```

### 4.2 Referência florestal

O módulo `src/lidaragrosat/data/reference.py` implementa:

- `read_reference_table(reference_path)`
- `write_reference_table(table, destination)`

O loader suporta:

- `CSV`
- `Parquet`
- `GeoPackage`
- `GeoJSON`
- `Shapefile`

Internamente a referência é normalizada para colunas padronizadas:

- `tree_id`
- `x`
- `y`
- `species`
- `height`

## 5. Pré-processamento LiDAR

### 5.1 Estrutura da nuvem

O núcleo LiDAR reside em `src/lidaragrosat/data/lidar.py`.

Classe central:

- `LidarPointCloud(x, y, z, intensity=None)`

### 5.2 Funções implementadas

As funções principais são:

- `read_lidar(path)`
- `filter_noise(cloud, lower_q, upper_q)`
- `build_dtm(cloud, bounds, cell_size, ground_quantile)`
- `sample_grid(grid, bounds, resolution, x, y)`
- `normalize_heights(cloud, dtm, dtm_bounds, dtm_resolution)`
- `rasterize_structural_channels(cloud, normalized_z, bounds, resolution, channel_names)`
- `compute_plot_bounds(cloud)`

### 5.3 Método de terreno implementado

O terreno não é obtido por classificador externo tipo `PDAL/CSF`; o sistema implementa um método em duas passagens:

1. `DTM` inicial por quantil baixo (`ground_quantile`) em grade.
2. Seleção de pontos próximos à superfície inicial usando `ground_residual_threshold_m`.
3. `DTM` refinado pela nuvem filtrada de candidatos a terreno.

Trata-se de um método pragmático e reprodutível, adequado ao tamanho do benchmark, porém menos sofisticado que pipelines de classificação de terreno dedicados.

### 5.4 Normalização de altura

A normalização de altura implementa:

\[
z_{norm} = z - z_{DTM}(x,y)
\]

Após isso, os pontos são filtrados para o intervalo físico:

- `z_norm >= 0`
- `z_norm <= 80 m`

### 5.5 Canais estruturais implementados

O tensor estrutural LiDAR entregue pelo código inclui:

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

### 5.6 Observações metodológicas

- `zp95` é calculado por estatística de ordem por célula, de forma vetorizada.
- `vertical_roughness` é a raiz da variância local de `zmax` em vizinhança `3x3`, calculada com `uniform_filter`.
- Essas otimizações substituíram versões muito mais lentas baseadas em `generic_filter` e chamadas repetidas de `np.quantile`, mantendo o significado ecológico do canal e tornando a preparação operacional no dataset real.

## 6. Pré-processamento RGB e QA multimodal

### 6.1 Funções RGB

Em `src/lidaragrosat/data/rgb.py`:

- `read_rgb(rgb_path)`
- `normalize_rgb(rgb, clip_percentiles, standardize)`
- `resample_rgb_to_grid(rgb_path, transform, height, width)`
- `write_raster(array, destination, transform, crs=None, dtype="float32")`

### 6.2 Harmonização implementada

O RGB é:

1. lido do GeoTIFF original;
2. reamostrado para a grade LiDAR;
3. recortado/ajustado à mesma extensão;
4. normalizado com clipping robusto por percentis;
5. opcionalmente padronizado por canal.

### 6.3 QA

O módulo `src/lidaragrosat/data/qa.py` implementa:

- `QAReport`
- `fraction_points_inside_bounds(reference_table, bounds)`
- `height_outliers(reference_table, lower_q=0.01, upper_q=0.99)`
- `estimate_shift_pixels(chm, rgb, max_shift_px=8)`

O registro multimodal é avaliado por correlação entre bordas de `CHM` e intensidade média do RGB. A versão atual usa:

- `sobel` para bordas;
- `fftconvolve` para correlação;
- downsampling adaptativo para reduzir custo.

## 7. Preparação de parcela

O módulo `src/lidaragrosat/data/preprocess.py` implementa o fluxo por parcela.

Funções principais:

- `_filter_normalized_cloud(cloud, normalized_z, min_height)`
- `prepare_plot_bundle(plot_inventory, config)`
- `save_prepared_plot(bundle, artifacts, destination_dir)`

### 7.1 Saídas da preparação

Cada parcela processada contém:

- `rgb.npy`
- `lidar.npy`
- `chm.npy`
- `reference.csv`
- `manifest.json`
- `qa_report.json`
- `rgb_resampled.tif`
- `lidar_structural.tif`
- `chm.tif`
- `dtm.tif`
- `tiles.csv`

### 7.2 Estrutura interna

A abstração central é:

- `ProcessedPlotBundle(plot_id, rgb, lidar, chm, transform, reference, crs=None, metadata=None)`

Implementada em `src/lidaragrosat/data/dataset.py`.

## 8. Formulação supervisionada e targets

O problema é formulado como detecção center-based sobre raster.

Em `src/lidaragrosat/data/targets.py`:

- `TargetMaps`
- `_gaussian_radius_pixels(sigma_m, resolution)`
- `build_center_targets(reference_table, transform, raster_height, raster_width, sigma_m)`

### 8.1 Alvo de heatmap

Para cada árvore anotada, o target é:

\[
H(x,y)=\max_i \exp\left(-\frac{\|(x,y)-c_i\|^2}{2\sigma^2}\right)
\]

### 8.2 Alvos adicionais

Além do heatmap, o código gera:

- `offset[0]`: desvio subpixel em `x`
- `offset[1]`: desvio subpixel em `y`
- `height`: altura local da árvore
- `positive_mask`: máscara de pixels positivos

O target de altura é mascarado quando o rótulo não existe.

## 9. Tiles e dataset de treino

### 9.1 Tiles

O módulo `src/lidaragrosat/data/tiles.py` implementa:

- `TileRecord`
- `count_trees_per_tile(...)`
- `build_tiles(...)`

Os tiles são gerados por:

- tamanho fixo (`tile_size`)
- overlap configurável (`tile_overlap`)
- contagem de árvores por tile

### 9.2 Dataset PyTorch

Em `src/lidaragrosat/data/dataset.py`:

- `save_processed_plot`
- `load_processed_plot`
- `list_processed_plots`
- `TreeTileDataset`

### 9.3 Estratégia de dataset

`TreeTileDataset`:

- carrega `rgb`, `lidar` e `chm` por tile;
- usa cache por parcela de `TargetMaps`;
- aplica augmentations geométricas consistentes;
- retorna tensores PyTorch já com:
  - `heatmap`
  - `offset`
  - `height`
  - `height_mask`
  - `positive_mask`

### 9.4 Augmentations efetivamente implementadas

Implementadas:

- flip horizontal
- flip vertical
- rotações múltiplas de `90°`
- `color jitter` escalar no RGB
- `dropout` espacial leve no LiDAR

Parâmetros existem para pequenas rotações arbitrárias e translação, mas a implementação atual não aplica essas duas transformações de forma explícita.

## 10. Baseline clássica

O módulo `src/lidaragrosat/baselines/local_maxima.py` implementa a baseline por `CHM + local maxima`.

Componentes:

- `LocalMaximaParams`
- `detect_local_maxima(chm, transform, params)`
- `tune_local_maxima(...)`

### 10.1 Método

O `CHM` é:

1. suavizado com `gaussian_filter`;
2. submetido a `maximum_filter`;
3. limiarizado por altura mínima;
4. convertido em centros georreferenciados.

O tuning da baseline ocorre sobre o conjunto de validação, testando:

- raio da janela local;
- sigma gaussiano;
- altura mínima.

## 11. Arquitetura deep learning

### 11.1 Blocos básicos

Em `src/lidaragrosat/models/blocks.py`:

- `make_norm(kind, num_channels)`
- `ResidualConvBlock`
- `SimpleEncoder`
- `FusionBlock`
- `FPNDecoder`

### 11.2 Encoder

`SimpleEncoder` é uma pilha de blocos residuais convolucionais com strides progressivos, produzindo features multiescala.

### 11.3 Fusões implementadas

O sistema suporta:

- `RGB-only`
- `LiDAR-only`
- `early fusion`
- `multimodal mid fusion`
- `multimodal late fusion`

### 11.4 Estratégias

#### Early fusion

Concatenação direta de canais `RGB + LiDAR` na entrada.

#### Mid fusion

Implementada por `FusionBlock`, que aprende uma porta (`gate`) para combinar features RGB e LiDAR em cada escala:

- concatenação de features;
- convolução `1x1`;
- ativação sigmoide;
- mistura ponderada entre correntes;
- refinamento residual.

#### Late fusion

Cada modalidade possui encoder e decoder próprios; a fusão ocorre após decodificação por concatenação e refinamento residual.

### 11.5 Decoder

`FPNDecoder` implementa:

- projeção lateral `1x1`;
- upsampling bilinear;
- soma entre escalas;
- projeções intermediárias para compatibilizar números de canais;
- refinamento residual por escala.

Essa projeção intermediária foi adicionada após detectar incompatibilidade de canais em execução real.

### 11.6 Cabeças de saída

Em `src/lidaragrosat/models/detector.py`:

- `DetectionHeads`
- `CenterBasedTreeDetector`
- `build_detector_from_config(config)`

As cabeças produzem:

- `heatmap`
- `offset`
- `height`

## 12. Função de perda

Em `src/lidaragrosat/training/losses.py`:

- `sigmoid_focal_loss`
- `DetectionLoss`

### 12.1 Componentes da perda

A perda total é:

\[
\mathcal{L} = \lambda_{hm}\mathcal{L}_{hm} + \lambda_{off}\mathcal{L}_{off} + \lambda_{h}\mathcal{L}_{h}
\]

Onde:

- `L_hm`: `sigmoid focal loss` no heatmap;
- `L_off`: `L1` mascarada em pixels positivos;
- `L_h`: `Huber loss` mascarada nos pixels com altura válida.

## 13. Treino

O loop de treino está em `src/lidaragrosat/training/engine.py`.

Funções principais:

- `_to_device(batch, device)`
- `run_epoch(model, loader, optimizer, criterion, device, train)`
- `fit_model(model, train_loader, val_loader, config, device, checkpoint_dir, validation_callback=None)`

### 13.1 Características do treino

- `AdamW`
- `CosineAnnealingLR` ou `ReduceLROnPlateau`
- `mixed precision` com `torch.amp`
- `gradient clipping`
- `early stopping`
- persistência de `best.pt`
- histórico por época em `history.csv`

### 13.2 Critério de checkpoint

O sistema usa:

- perda de validação por default;
- ou métrica externa, tipicamente `F1`, via `validation_callback`.

Foi adicionado `validation_interval_epochs` para reduzir custo de validações pesadas em regime prático.

## 14. Inferência

Em `src/lidaragrosat/inference/sliding_window.py`:

- `PredictionMaps`
- `predict_plot_sliding_window(...)`
- `decode_tree_predictions(...)`

### 14.1 Estratégia

O raster completo é inferido por janelas com overlap. O sistema acumula:

- soma de heatmaps;
- soma de offsets;
- soma de alturas;
- contagem de contribuições por pixel.

Ao final, usa média por pixel.

### 14.2 Decodificação

O pós-processamento implementa:

- `sigmoid` no heatmap;
- `maximum_filter` para NMS local;
- limiar de confiança;
- correção subpixel via `offset`;
- associação opcional de altura.

## 15. Matching e métricas

### 15.1 Matching

Em `src/lidaragrosat/evaluation/matching.py`:

- `MatchResult`
- `match_tree_predictions(...)`

O matching é:

- bipartido;
- um-para-um;
- resolvido por `Hungarian algorithm` (`linear_sum_assignment`);
- com admissibilidade por:
  - distância espacial 2D;
  - tolerância de altura opcional.

### 15.2 Métricas

Em `src/lidaragrosat/evaluation/metrics.py`:

- `safe_divide`
- `summarize_matches`

As métricas implementadas são:

- `TP`
- `FP`
- `FN`
- `precision`
- `recall`
- `F1`
- `location_mae_m`
- `height_mae_m`

### 15.3 Estatística

Em `src/lidaragrosat/evaluation/statistics.py`:

- `paired_wilcoxon`
- `bootstrap_confidence_interval`

`paired_wilcoxon` foi adaptada para retornar `NaN` quando o número de parcelas testadas é insuficiente, como no caso de `basic split` com um único plot de teste.

## 16. Análise científica adicional

Em `src/lidaragrosat/analysis/scientific.py`:

- `compute_plot_descriptors`
- `build_gain_table`
- `fit_gain_vs_complexity`
- `classify_error_modes`
- `save_regression_table`

### 16.1 Descritores implementados por parcela

- `tree_density_ha`
- `canopy_height_variance`
- `vertical_roughness_mean`
- `rgb_entropy`
- `conifer_fraction`

### 16.2 Categorias de erro implementadas

Falsos positivos:

- `fp_border_artifact`
- `fp_fragmented_texture`
- `fp_duplicate`

Falsos negativos:

- `fn_overlapping_crown`
- `fn_no_dominant_apex`

Essas categorias são heurísticas, baseadas em geometria local e proximidade a borda.

## 17. Scripts de execução

### 17.1 `scripts/prepare_dataset.py`

Responsável por:

- descobrir parcelas;
- preparar rasters;
- salvar bundles processados;
- gerar `tiles.csv`;
- salvar inventário e resumo de QA;
- persistir splits.

### 17.2 `scripts/run_baseline.py`

Responsável por:

- carregar parcelas processadas;
- construir split `LOPO` ou `basic`;
- fazer tuning da baseline na validação;
- inferir no teste;
- salvar parâmetros e sumários.

### 17.3 `scripts/train_cv.py`

Responsável por:

- construir split experimental;
- amostrar tiles positivos e negativos;
- criar `DataLoader`;
- treinar modelo;
- validar via callback em plot completo;
- inferir no conjunto de teste;
- salvar checkpoints, histórico e predições.

Apesar do nome histórico `train_cv.py`, o script hoje suporta tanto `LOPO` quanto `basic split`.

### 17.4 `scripts/evaluate_experiments.py`

Responsável por:

- agregar predições salvas;
- aplicar matching unificado;
- produzir:
  - `metrics_by_plot.csv`
  - `method_summary.csv`
  - `paired_statistics.csv`
  - `plot_descriptors.csv`
  - `fusion_gain_vs_baseline.csv`
  - `fusion_gain_regression.csv`
  - `error_modes.csv`

### 17.5 `scripts/run_full_pipeline.py`

Orquestra o fluxo completo:

- preparação;
- baseline;
- treino das variantes;
- avaliação agregada.

## 18. Protocolos experimentais efetivamente implementados

### 18.1 LOPO

Implementado em `src/lidaragrosat/data/splits.py` por:

- `build_lopo_splits(plot_ids, validation_plots_per_fold=1)`

Cada fold contém:

- `train_plots`
- `val_plots`
- `test_plots`

### 18.2 Basic split

Implementado também em `src/lidaragrosat/data/splits.py` por:

- `build_basic_split(plot_ids, train_fraction=0.8, val_fraction=0.1, seed=13)`

Retorna um único `FoldSplit(fold_id=0, ...)`.

## 19. Execução real já realizada

No estado atual do repositório, já foram executados:

- preparação completa das 10 parcelas do Dubrovin;
- baseline clássica em `LOPO`;
- cinco modelos em `LOPO` com perfil prático;
- avaliação agregada desse regime;
- baseline e cinco modelos em `basic split`;
- avaliação agregada do `basic split` em regime longo.

Saídas principais:

- `outputs/*`
- `outputs_basic_split/*`

## 20. Ajustes feitos durante a execução real

Durante a execução no dataset real, a arquitetura foi refinada em pontos críticos:

1. Rugosidade vertical passou de `generic_filter(np.nanstd)` para cálculo vetorizado com `uniform_filter`.
2. O cálculo de quantis por célula foi vetorizado por ordenação em grupo.
3. A estimativa de shift RGB-LiDAR passou a usar `fftconvolve` com downsampling.
4. O `FPNDecoder` ganhou projeções intermediárias de canais entre escalas.
5. O treino ganhou `validation_interval_epochs`.
6. Foi adicionado suporte explícito a `basic split`.
7. Para caber em GPU já ocupada por outro treino externo, foi criado `configs/batch1.yaml`.

Esses ajustes não são apenas detalhes de engenharia; eles foram necessários para tornar a implementação cientificamente reproduzível e operacional em dados reais.

## 21. Limitações e diferenças em relação ao plano ideal

Com rigor metodológico, é importante registrar o que ainda não está no estado ideal:

- Não há pipeline dedicado de classificação de terreno por `PDAL` ou algoritmo especializado.
- O `zp95` é uma aproximação por estatística de ordem sem interpolação contínua.
- Não há automação completa de nested hyperparameter search para deep learning; o ajuste é conduzido por configurações e callback de validação.
- As augmentations de pequena rotação arbitrária e jitter translacional não estão implementadas apesar de existirem na configuração.
- A geração automatizada das figuras do paper não está totalmente integrada aos scripts principais, embora existam funções auxiliares e todas as tabelas/artefatos necessários para análise estejam sendo salvos.
- O `basic split` com um único plot de teste não permite inferência estatística forte.

## 22. Conclusão técnica

O repositório já contém uma implementação experimental coerente, funcional e executada em dados reais para o problema de ITD multimodal UAV LiDAR + RGB.

Em termos de arquitetura, o sistema atual entrega:

- preparação geoespacial reprodutível;
- formulação supervisionada center-based;
- baseline clássica controlada;
- famílias unimodais e multimodais comparáveis;
- treino com checkpointing e inferência full-plot;
- avaliação com matching unificado;
- análise quantitativa e interpretativa pós-experimento.

Ou seja, o projeto não está apenas em fase de desenho: ele já está implementado como pipeline científico-operacional end-to-end.
