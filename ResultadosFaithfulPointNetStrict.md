# Resultados Faithful PointNet Strict

## Escopo

Esta execucao implementa a versao estritamente fiel ao `Pretendido.md` para o teste de `deep learning direto na nuvem de pontos`, mantendo a implementacao anterior separada.

- modulo: [strict_pointnet.py](/home/murilo/Documentos/LiDARAgroSat/src/lidaragrosat/paper_faithful/strict_pointnet.py)
- runner: [run_faithful_pointnet_strict.py](/home/murilo/Documentos/LiDARAgroSat/scripts/run_faithful_pointnet_strict.py)
- artefatos: [outputs_faithful_pointnet_strict](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_pointnet_strict)

## Conformidade com o Pretendido

A checklist automatica ficou em [pretendido_checklist.csv](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_pointnet_strict/pretendido_checklist.csv).

Resumo:

- `yes`: nuvem normalizada, gerador de propostas `local maxima` com `window_size=0.8 m` e `height_threshold=3.0 m`, vizinhanca esferica `R=2.5 m`, `512` pontos, PointNet puro local, split `8 treino / 2 validacao`, loss `BCEWithLogits(pos_weight=5.0) + SmoothL1`, augmentations de rotacao/jitter, matching fiel do paper.
- `note`: o `Pretendido.md` conflita entre `NMS 3D` no diagrama geral e `nms_2d` na Fase 6. A run fiel seguiu a Fase 6 literalmente, portanto usou NMS 2D.

## Configuracao da run

- treino: `plot_02, plot_03, plot_04, plot_05, plot_07, plot_08, plot_09, plot_10`
- validacao: `plot_01, plot_06`
- propostas: `local_maxima_filter_3d(points_xyz, window_size=0.8, height_threshold=3.0)`
- labels de treino: positivo quando `dist2D <= 2.0 m`, offset XY ate o GT mais proximo
- melhor pos-processamento em validacao: `cls_threshold=0.70`, `nms_radius_m=1.80`
- treino: `40` epocas completas, sem early stop antes do limite

Historico de treino: [history.csv](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_pointnet_strict/history.csv)

## Estagio de propostas

O gerador de propostas entrou exatamente no regime desejado pelo desenho: recall muito alto e precisao muito baixa.

Resumo agregado em [proposal_summary_by_partition.csv](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_pointnet_strict/proposal_summary_by_partition.csv):

| Particao | Precision mean | Recall mean | F1 mean |
|---|---:|---:|---:|
| `proposal_train` | 0.0394 | 0.9133 | 0.0752 |
| `proposal_val` | 0.0369 | 0.9280 | 0.0707 |
| `proposal_all` | 0.0389 | 0.9163 | 0.0743 |

Por parcela, o recall das propostas variou de `0.7915` a `0.9648`, e a razao `proposals / trees` ficou entre `12.16x` e `35.86x`. Isso esta em [proposal_metrics_by_plot.csv](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_pointnet_strict/proposal_metrics_by_plot.csv).

## Resultado final do PointNet fiel

Resumo agregado em [summary_by_partition.csv](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_pointnet_strict/summary_by_partition.csv):

| Particao | TP | FP | FN | Precision mean | Recall mean | F1 mean | Distance mean (m) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `train` | 2032 | 6705 | 860 | 0.2310 | 0.7102 | 0.3456 | 0.8739 |
| `val` | 557 | 1673 | 153 | 0.2522 | 0.7869 | 0.3783 | 0.8607 |
| `all` | 2589 | 8378 | 1013 | 0.2352 | 0.7255 | 0.3521 | 0.8712 |

Metricas por parcela em [metrics_by_plot.csv](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_pointnet_strict/metrics_by_plot.csv):

| Parcela | Particao | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| `plot_01` | `val` | 0.3052 | 0.7738 | 0.4377 |
| `plot_02` | `train` | 0.2071 | 0.6740 | 0.3168 |
| `plot_03` | `train` | 0.2058 | 0.7289 | 0.3210 |
| `plot_04` | `train` | 0.1800 | 0.7050 | 0.2868 |
| `plot_05` | `train` | 0.1873 | 0.7500 | 0.2997 |
| `plot_06` | `val` | 0.1991 | 0.8000 | 0.3189 |
| `plot_07` | `train` | 0.2611 | 0.7206 | 0.3833 |
| `plot_08` | `train` | 0.2290 | 0.7771 | 0.3538 |
| `plot_09` | `train` | 0.2825 | 0.7102 | 0.4042 |
| `plot_10` | `train` | 0.2951 | 0.6158 | 0.3990 |

## Comparacao justa com os baselines fieis na validacao

Comparacao agregada nas parcelas de validacao (`plot_01`, `plot_06`) em [baseline_comparison_val.csv](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_pointnet_strict/baseline_comparison_val.csv):

| Metodo | Precision mean | Recall mean | F1 mean |
|---|---:|---:|---:|
| `chm_watershed` | 0.5627 | 0.5616 | 0.5614 |
| `rgb_lidar_object_fusion` | 0.5237 | 0.5606 | 0.5356 |
| `local_maxima_adaptive` | 0.4874 | 0.5869 | 0.5260 |
| `local_maxima_fixed` | 0.3883 | 0.5690 | 0.4603 |
| `pointnet_strict` | 0.2522 | 0.7869 | 0.3783 |

Leitura:

- o PointNet fiel entregou o maior `recall` medio entre os metodos comparados na validacao (`0.7869`);
- o problema continuou sendo `precision`, que caiu para `0.2522`;
- por isso, o `F1` final ficou abaixo de todos os baselines fieis de referencia.

## Comparacao por parcela

A tabela completa esta em [comparison_vs_baselines_by_plot.csv](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_pointnet_strict/comparison_vs_baselines_by_plot.csv).

Nos plots de validacao:

- `plot_01`: `PointNet strict F1 = 0.4377`, contra `0.5153` do `local_maxima_fixed`, `0.5507` do `adaptive`, `0.5605` do `watershed`, `0.5496` da `object_fusion`
- `plot_06`: `PointNet strict F1 = 0.3189`, contra `0.4052` do `local_maxima_fixed`, `0.5014` do `adaptive`, `0.5624` do `watershed`, `0.5215` da `object_fusion`

Nos `10` plots, o `PointNet strict` so superou o `local_maxima_fixed` em `plot_09` e `plot_10`, e ficou abaixo de `adaptive`, `watershed` e `object_fusion` em todas as parcelas.

## Conclusao objetiva

A implementacao ficou fiel ao `Pretendido.md`, inclusive no regime de propostas de altissimo recall. O gargalo nao foi cobertura de candidatos; foi a incapacidade do PointNet local, nessa formulacao pura, de eliminar falsos positivos suficientes. O comportamento observado foi:

- propostas: recall alto, precisao extremamente baixa, como esperado
- PointNet + offset + NMS: melhora forte sobre o gerador bruto, mas ainda insuficiente em precisao
- resultado final: `F1_val = 0.3783`, abaixo dos baselines fieis e bem abaixo do intervalo esperado de `0.65–0.75`

Em termos metodologicos, a run valida o desenho fiel, mas nao valida a hipotese de ganho desse PointNet local puro sobre os metodos geometricos fortes que ja estavam no benchmark.
