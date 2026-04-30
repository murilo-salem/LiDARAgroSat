# Resultados Faithful CHM + Watershed

## Implementação

Baseline separado em:

- [src/lidaragrosat/paper_faithful/watershed.py](/home/murilo/Documentos/LiDARAgroSat/src/lidaragrosat/paper_faithful/watershed.py)
- [scripts/run_optuna_faithful_watershed.py](/home/murilo/Documentos/LiDARAgroSat/scripts/run_optuna_faithful_watershed.py)

O pipeline reutiliza o mesmo cache normalizado da versão fiel do baseline em nuvem de pontos:

- deduplicação a `1 mm`
- normalização de altura via pontos de solo `class 2`
- matching guloso determinístico com `max_dist = 5 m` e `max_height_diff = 3 m`

Diferença metodológica:

- `local maxima` atual: opera diretamente na nuvem 3D
- `CHM + watershed`: rasteriza um `CHM`, preenche pequenos buracos locais, suaviza, extrai marcadores locais e segmenta copas por `watershed`

## Estudo Optuna

Saída completa em:

- [outputs_faithful_paper_watershed_optuna/best_summary.json](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_paper_watershed_optuna/best_summary.json)
- [outputs_faithful_paper_watershed_optuna/best_per_plot.csv](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_paper_watershed_optuna/best_per_plot.csv)
- [outputs_faithful_paper_watershed_optuna/comparison_vs_current_local_maxima.csv](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_paper_watershed_optuna/comparison_vs_current_local_maxima.csv)
- [outputs_faithful_paper_watershed_optuna/trials.csv](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_paper_watershed_optuna/trials.csv)

Configuração do melhor trial:

- `resolution_m = 0.1`
- `gaussian_sigma_m = 0.2`
- `min_distance_m = 0.8`
- `min_height_m = 5.9`
- `min_segment_area_m2 = 6.0`
- `hole_fill_radius_m = 0.3`
- `compactness = 0.0075`

Resumo global do melhor `CHM + watershed`:

- `TP = 1884`
- `FP = 1552`
- `FN = 1718`
- `Precision = 0.5483`
- `Recall = 0.5230`
- `F1 = 0.5354`

Resumo global do baseline atual `local maxima` otimizado:

- `TP = 1636`
- `FP = 2211`
- `FN = 1966`
- `Precision = 0.4253`
- `Recall = 0.4542`
- `F1 = 0.4393`

Ganho global do `CHM + watershed` sobre o baseline atual:

- `delta Precision = +0.1230`
- `delta Recall = +0.0689`
- `delta F1 = +0.0961`

## Resultados por Parcela

| Parcela | P watershed | R watershed | F1 watershed | P atual | R atual | F1 atual | delta F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `plot_01` | 0.5821 | 0.5405 | 0.5605 | 0.4516 | 0.6000 | 0.5153 | +0.0452 |
| `plot_02` | 0.5587 | 0.4822 | 0.5176 | 0.4945 | 0.3671 | 0.4214 | +0.0963 |
| `plot_03` | 0.4700 | 0.5422 | 0.5035 | 0.3923 | 0.4880 | 0.4349 | +0.0686 |
| `plot_04` | 0.4434 | 0.5556 | 0.4932 | 0.4007 | 0.4636 | 0.4298 | +0.0634 |
| `plot_05` | 0.5906 | 0.4856 | 0.5330 | 0.3863 | 0.5144 | 0.4412 | +0.0917 |
| `plot_06` | 0.5434 | 0.5828 | 0.5624 | 0.3250 | 0.5379 | 0.4052 | +0.1572 |
| `plot_07` | 0.5068 | 0.5490 | 0.5271 | 0.4262 | 0.5098 | 0.4643 | +0.0628 |
| `plot_08` | 0.5844 | 0.5484 | 0.5658 | 0.4030 | 0.5484 | 0.4646 | +0.1012 |
| `plot_09` | 0.5990 | 0.5076 | 0.5495 | 0.4520 | 0.3486 | 0.3936 | +0.1559 |
| `plot_10` | 0.6237 | 0.4672 | 0.5342 | 0.6208 | 0.2876 | 0.3931 | +0.1411 |

## Leitura Técnica

O `CHM + watershed` superou o baseline atual em `10/10` parcelas.

O padrão do ganho foi:

- aumento forte de precisão em quase todas as parcelas;
- aumento moderado de recall na maioria das parcelas;
- melhora grande em parcelas onde o `local maxima` atual tinha excesso de `FP`, como `plot_06`, `plot_09` e `plot_10`.

As maiores melhoras de `F1` ocorreram em:

- `plot_06`: `+0.1572`
- `plot_09`: `+0.1559`
- `plot_10`: `+0.1411`

As menores melhoras ocorreram em:

- `plot_01`: `+0.0452`
- `plot_07`: `+0.0628`
- `plot_04`: `+0.0634`

## Validação

Verificações executadas após a implementação:

- `python3 -m compileall src scripts`
- `PYTHONPATH=src:. ./.venv/bin/pytest -q tests`

Resultado: `11 passed`.
