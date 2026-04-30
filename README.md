# LiDARAgroSat

Pipeline técnico-científico para detecção individual de árvores em florestas mistas densas com fusão multimodal UAV LiDAR + RGB.

## Escopo implementado

- preparação e QA multimodal por parcela;
- geração de DTM, CHM e tensor estrutural LiDAR multicanal;
- harmonização RGB para a grade LiDAR;
- geração de heatmaps gaussianos, offsets e alvo opcional de altura;
- baseline clássica por máximos locais em CHM;
- modelos deep learning `RGB-only`, `LiDAR-only`, `early fusion`, `mid fusion` e `late fusion`;
- treino com validação cruzada `Leave-One-Plot-Out`;
- matching bipartido `one-to-one` com restrição espacial e de altura;
- agregação estatística, bootstrap, Wilcoxon pareado e análise exploratória de ganho vs heterogeneidade.

## Estrutura esperada dos dados

```text
data/
  raw/
    plot_01/
      lidar.laz
      rgb.tif
      trees_reference.gpkg
    plot_02/
      ...
```

Após a preparação:

```text
data/
  processed/
    plot_01/
      rgb.npy
      lidar.npy
      chm.npy
      reference.csv
      manifest.json
      qa_report.json
      rgb_resampled.tif
      lidar_structural.tif
      chm.tif
      dtm.tif
      tiles.csv
```

## Ambiente

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .[dev]
```

## Execução

Preparação:

```bash
python3 scripts/prepare_dataset.py --config configs/base.yaml
```

Baseline clássica:

```bash
python3 scripts/run_baseline.py --config configs/base.yaml
```

Treino dos modelos:

```bash
python3 scripts/train_cv.py --config configs/base.yaml configs/lidar_only.yaml
python3 scripts/train_cv.py --config configs/base.yaml configs/rgb_only.yaml
python3 scripts/train_cv.py --config configs/base.yaml configs/early_fusion.yaml
python3 scripts/train_cv.py --config configs/base.yaml configs/multimodal_mid.yaml
python3 scripts/train_cv.py --config configs/base.yaml configs/multimodal_late.yaml
```

Avaliação unificada:

```bash
python3 scripts/evaluate_experiments.py --config configs/base.yaml \
  --reference-method baseline_local_maxima \
  --fusion-method multimodal_mid_fusion \
  --baseline-method lidar_only
```

Pipeline completo:

```bash
python3 scripts/run_full_pipeline.py --base-config configs/base.yaml
```

## Saídas principais

- `outputs/predictions/<metodo>/fold_xx/*.csv`: centros preditos por parcela.
- `outputs/metrics/method_summary.csv`: Tabela 1 do paper.
- `outputs/metrics/paired_statistics.csv`: teste pareado entre métodos.
- `outputs/metrics/plot_descriptors.csv`: descritores estruturais/espectrais por parcela.
- `outputs/metrics/fusion_gain_regression.csv`: ganho da fusão vs heterogeneidade.
- `outputs/metrics/error_modes.csv`: categorização de erro.

## Observações

- O pipeline assume alinhamento geoespacial consistente entre LiDAR, RGB e referência.
- O código foi estruturado para o benchmark Dubrovin, mas aceita qualquer coleção de parcelas com o mesmo formato lógico.
- O treino usa `PyTorch`; a preparação geoespacial requer `rasterio`, `geopandas` e `laspy`.
