# Resultados Faithful LiDAR + RGB Object-Oriented Fusion

## Escopo

Foi testada uma variante orientada a objetos de fusão `LiDAR + RGB`, separada do restante do pipeline, com:

- propostas LiDAR vindas de `local maxima` adaptativo, `watershed` ou união dos dois;
- segmentação de objetos no RGB por `Felzenszwalb`;
- filtragem e agrupamento das propostas LiDAR por objeto RGB.

Implementação:

- [src/lidaragrosat/paper_faithful/object_fusion.py](/home/murilo/Documentos/LiDARAgroSat/src/lidaragrosat/paper_faithful/object_fusion.py)
- [scripts/run_optuna_faithful_object_fusion.py](/home/murilo/Documentos/LiDARAgroSat/scripts/run_optuna_faithful_object_fusion.py)

## Observação sobre DeepForest

O pacote `DeepForest` foi instalado com sucesso, mas não pôde ser usado em inferência neste ambiente por incompatibilidade de runtime entre `torch` e `torchvision`:

- erro observado: `RuntimeError: operator torchvision::nms does not exist`

Por isso, a avaliação experimental foi feita com o fallback orientado a objetos clássico no RGB, que é executável e reprodutível no repositório atual.

## Melhor configuração encontrada

Saídas completas:

- [outputs_faithful_paper_object_fusion_optuna/best_summary.json](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_paper_object_fusion_optuna/best_summary.json)
- [outputs_faithful_paper_object_fusion_optuna/best_per_plot.csv](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_paper_object_fusion_optuna/best_per_plot.csv)
- [outputs_faithful_paper_object_fusion_optuna/comparison_vs_fixed.csv](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_paper_object_fusion_optuna/comparison_vs_fixed.csv)
- [outputs_faithful_paper_object_fusion_optuna/comparison_vs_adaptive.csv](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_paper_object_fusion_optuna/comparison_vs_adaptive.csv)
- [outputs_faithful_paper_object_fusion_optuna/comparison_vs_watershed.csv](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_paper_object_fusion_optuna/comparison_vs_watershed.csv)

Melhor trial:

- `source_mode = adaptive`
- `felzenszwalb_scale = 50.0`
- `felzenszwalb_sigma = 0.9`
- `felzenszwalb_min_size = 192`
- `min_segment_area_m2 = 0.5`
- `max_segment_area_m2 = 120.0`
- `area_per_tree_m2 = 24.0`
- `min_boundary_distance_m = 0.0`
- `min_segment_chm_max_m = 6.5`
- `candidate_merge_radius_m = 0.5`

## Resultado global

Fusão orientada a objetos:

- `TP = 1932`
- `FP = 1949`
- `FN = 1670`
- `Precision = 0.4978`
- `Recall = 0.5364`
- `F1 = 0.5164`

Comparação com os baselines atuais:

- `local maxima` fixo: `F1 = 0.4393`
- `local maxima` adaptativo: `F1 = 0.5084`
- `CHM + watershed`: `F1 = 0.5354`

Leitura objetiva:

- supera o baseline fixo com folga;
- supera o adaptativo por margem pequena;
- não supera o `watershed` LiDAR puro.

## Resultados por parcela

| Parcela | F1 fusão | F1 fixo | F1 adaptativo | F1 watershed |
|---|---:|---:|---:|---:|
| `plot_01` | 0.5496 | 0.5153 | 0.5507 | 0.5605 |
| `plot_02` | 0.4756 | 0.4214 | 0.4566 | 0.5176 |
| `plot_03` | 0.5207 | 0.4349 | 0.5106 | 0.5035 |
| `plot_04` | 0.4366 | 0.4298 | 0.4223 | 0.4932 |
| `plot_05` | 0.5760 | 0.4412 | 0.5423 | 0.5330 |
| `plot_06` | 0.5215 | 0.4052 | 0.5014 | 0.5624 |
| `plot_07` | 0.5270 | 0.4643 | 0.5258 | 0.5271 |
| `plot_08` | 0.5510 | 0.4646 | 0.5371 | 0.5658 |
| `plot_09` | 0.5215 | 0.3936 | 0.5355 | 0.5495 |
| `plot_10` | 0.4962 | 0.3931 | 0.5005 | 0.5342 |

## Comparação agregada

Contra `local maxima` fixo:

- `mean delta F1 by plot = +0.0812`
- vitórias: `10/10`

Contra `local maxima` adaptativo:

- `mean delta F1 by plot = +0.0093`
- vitórias: `7/10`

Contra `CHM + watershed`:

- `mean delta F1 by plot = -0.0171`
- vitórias: `2/10`

## Interpretação

O ganho do RGB apareceu, mas de forma mais modesta do que a hipótese otimista:

- o melhor regime usa o RGB como agrupador leve de objetos;
- a melhor fonte LiDAR continuou sendo a proposta `adaptive`, não a união com `watershed`;
- filtros fortes por borda e segmentos pequenos derrubaram o recall.

Empiricamente, neste benchmark:

- `RGB object fusion` melhora o baseline geométrico clássico;
- mas o `CHM + watershed` ainda captura melhor a estrutura de copa do que a segmentação RGB clássica sozinha consegue refinar.

## Validação

Verificações executadas:

- `python3 -m compileall src scripts tests`
- `PYTHONPATH=src:. ./.venv/bin/pytest -q tests`

Resultado: `14 passed`.
