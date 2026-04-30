# Resultados Faithful Hybrid Watershed Proposals

## Metodo

Esta variante combina:

- `watershed` no `CHM` para obter segmentos de copa
- propostas de alta revocacao vindas de `local maxima` com `window_size=0.8 m`
- um filtro por segmento: manter a proposta mais alta dentro de cada segmento; se o segmento nao tiver proposta, usar o maximo do `CHM` como fallback

Implementacao:

- [hybrid_watershed_proposals.py](/home/murilo/Documentos/LiDARAgroSat/src/lidaragrosat/paper_faithful/hybrid_watershed_proposals.py)
- [run_faithful_hybrid_watershed_proposals.py](/home/murilo/Documentos/LiDARAgroSat/scripts/run_faithful_hybrid_watershed_proposals.py)

Artefatos:

- [outputs_faithful_hybrid_watershed_proposals](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_hybrid_watershed_proposals)

## Melhor conjunto de parametros

Selecionado por grid search em `plot_01` e `plot_06`, salvo em [best_validation_params.json](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_hybrid_watershed_proposals/best_validation_params.json):

- `resolution_m = 0.2`
- `smooth_sigma_px = 1.5`
- `min_distance_px = 3`
- `min_segment_area_m2 = 6.0`
- `proposal_window_m = 0.8`

## Comparacao resumida com o watershed anterior

Tabela em [hybrid_vs_watershed_summary.csv](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_hybrid_watershed_proposals/hybrid_vs_watershed_summary.csv):

| Escopo | Hibrido Precision | Hibrido Recall | Hibrido F1 | Watershed F1 | Delta F1 |
|---|---:|---:|---:|---:|---:|
| `val` (`plot_01`, `plot_06`) | 0.5684 | 0.5557 | 0.5593 | 0.5614 | -0.0021 |
| `all` (`10` parcelas) | 0.5353 | 0.5201 | 0.5230 | 0.5347 | -0.0117 |

Conclusao direta: **nao foi melhor**. Na validacao o resultado ficou praticamente empatado com o watershed puro, mas ligeiramente abaixo. No agregado das 10 parcelas ficou um pouco pior.

## Comparacao com os outros metodos ja rodados

Val scope, em [comparison_val_scope.csv](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_hybrid_watershed_proposals/comparison_val_scope.csv):

| Metodo | F1 mean |
|---|---:|
| `chm_watershed` | 0.5614 |
| `hybrid_watershed_proposals` | 0.5593 |
| `rgb_lidar_object_fusion` | 0.5356 |
| `local_maxima_adaptive` | 0.5260 |
| `local_maxima_fixed` | 0.4603 |

All scope, em [comparison_all_scope.csv](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_hybrid_watershed_proposals/comparison_all_scope.csv):

| Metodo | F1 mean |
|---|---:|
| `chm_watershed` | 0.5347 |
| `hybrid_watershed_proposals` | 0.5230 |
| `rgb_lidar_object_fusion` | 0.5176 |
| `local_maxima_adaptive` | 0.5083 |
| `local_maxima_fixed` | 0.4364 |

## Resultados por parcela contra o watershed

Tabela em [comparison_vs_watershed_by_plot.csv](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_hybrid_watershed_proposals/comparison_vs_watershed_by_plot.csv):

| Parcela | F1 hibrido | F1 watershed | Delta |
|---|---:|---:|---:|
| `plot_01` | 0.5700 | 0.5605 | +0.0095 |
| `plot_02` | 0.4978 | 0.5176 | -0.0199 |
| `plot_03` | 0.5028 | 0.5035 | -0.0007 |
| `plot_04` | 0.4521 | 0.4932 | -0.0411 |
| `plot_05` | 0.5478 | 0.5330 | +0.0148 |
| `plot_06` | 0.5487 | 0.5624 | -0.0137 |
| `plot_07` | 0.5000 | 0.5271 | -0.0271 |
| `plot_08` | 0.5770 | 0.5658 | +0.0112 |
| `plot_09` | 0.5182 | 0.5495 | -0.0314 |
| `plot_10` | 0.5153 | 0.5342 | -0.0189 |

Saldo: o hibrido ganhou em `3/10` parcelas e perdeu em `7/10`.

## Diagnostico dos segmentos

Diagnosticos em:

- [validation_segment_diagnostics.csv](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_hybrid_watershed_proposals/validation_segment_diagnostics.csv)
- [segment_diagnostics_by_plot.csv](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_hybrid_watershed_proposals/segment_diagnostics_by_plot.csv)

Ponto principal:

- `under` ficou muito alto em todas as parcelas
- na validacao, medias:
  - `empty = 113.5`
  - `under = 288.5`

Isso indica que o gargalo dominante nao foi ruido fora de copa, e sim **sub-segmentacao**: muitos segmentos ainda cobriram varias arvores. Nessa situacao, “um candidato por segmento” inevitavelmente sacrifica `recall`.

## Leitura tecnica final

A ideia funcionou parcialmente:

- realmente cortou duplicatas e segurou o ruido do gerador de propostas;
- em algumas parcelas ganhou do watershed puro;
- mas o `watershed` ainda segmentou varias regioes com mais de uma arvore, e isso limitou o teto do metodo.

Em termos práticos, o filtro “um candidato por segmento” ficou forte demais para este `CHM` e esta escala de suavizacao. O resultado final foi competitivo, mas **nao superou** o watershed puro que ja tinhamos.
