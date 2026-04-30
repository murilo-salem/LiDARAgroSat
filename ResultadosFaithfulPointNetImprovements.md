# Resultados Faithful PointNet Improvements

## Escopo

Esta ablação foi executada sobre a versao fiel do PointNet local, mantendo a mesma validacao fixa em `plot_01` e `plot_06`, e testando as mudancas pedidas:

- `strict_redecode_wide`: retuning amplo de `threshold + NMS`, sem retrain
- `strict_focal_xyz`: troca de `BCE` por `Focal Loss`, mantendo `XYZ`
- `strict_multifeature_bce`: entrada `7-D` com `XYZ + z_norm + dist_xy + intensity + return_ratio`, mantendo `BCE`
- `strict_multifeature_focal`: entrada `7-D` com `Focal Loss`

Runner: [run_faithful_pointnet_improvements.py](/home/murilo/Documentos/LiDARAgroSat/scripts/run_faithful_pointnet_improvements.py)  
Artefatos: [outputs_faithful_pointnet_improvements](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_pointnet_improvements)

## Referencias de comparacao

- `PointNet fiel original`: `F1_val = 0.3783`
- melhor baseline geral na mesma validacao: `CHM + watershed`, `F1_val = 0.5614`

Esses valores estao em [run_summary.json](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_pointnet_improvements/run_summary.json).

## Ranking das variantes

Tabela agregada em [variant_summary.csv](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_pointnet_improvements/variant_summary.csv):

| Variante | Features | Loss | Threshold | NMS (m) | Precision mean | Recall mean | F1 mean |
|---|---|---|---:|---:|---:|---:|---:|
| `strict_focal_xyz` | `XYZ` | `Focal` | 0.61 | 2.5 | 0.5709 | 0.5347 | 0.5505 |
| `strict_redecode_wide` | `XYZ` | `BCE` | 0.86 | 2.5 | 0.5805 | 0.5166 | 0.5444 |
| `strict_multifeature_focal` | `7-D` | `Focal` | 0.62 | 2.0 | 0.5134 | 0.5493 | 0.5302 |
| `strict_multifeature_bce` | `7-D` | `BCE` | 0.86 | 2.5 | 0.5278 | 0.5306 | 0.5262 |

## Comparacao com o melhor resultado anterior

Tabela em [comparison_vs_best.csv](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_pointnet_improvements/comparison_vs_best.csv):

| Variante | F1 mean | Delta vs PointNet fiel original | Delta vs melhor baseline (`CHM+watershed`) |
|---|---:|---:|---:|
| `strict_focal_xyz` | 0.5505 | +0.1722 | -0.0109 |
| `strict_redecode_wide` | 0.5444 | +0.1661 | -0.0171 |
| `strict_multifeature_focal` | 0.5302 | +0.1519 | -0.0312 |
| `strict_multifeature_bce` | 0.5262 | +0.1479 | -0.0353 |

## Leitura tecnica por mudanca

### 1. Threshold + NMS

Foi o maior ganho unitario.

- antes: `F1_val = 0.3783`
- depois: `F1_val = 0.5444`
- ganho: `+0.1661`

Interpretacao:

- o gargalo de calibracao era real e muito forte;
- o threshold otimo nao ficou baixo, e sim alto: `0.86`;
- o melhor `NMS` tambem ficou no limite superior do grid, `2.5 m`.

Ou seja, o modelo original estava deixando passar FP demais por decode permissivo.

### 2. Focal Loss

Sobre `XYZ`, `Focal Loss` ainda melhorou um pouco:

- `strict_redecode_wide`: `F1 = 0.5444`
- `strict_focal_xyz`: `F1 = 0.5505`
- ganho adicional: `+0.0061`

Leitura:

- houve melhora real, mas pequena;
- o principal efeito foi ajustar melhor o equilibrio `P/R`, sem mudar o quadro geral.

### 3. Features multicanal

Neste experimento, as features extras nao ajudaram.

- `strict_multifeature_bce`: `F1 = 0.5262`
- `strict_multifeature_focal`: `F1 = 0.5302`

Ambas ficaram abaixo de `strict_focal_xyz` e de `strict_redecode_wide`.

Leitura plausivel:

- `intensity` e `return_ratio` sao informativos em tese, mas aqui nao melhoraram a separacao final nesse setup;
- a simples extensao da entrada para `7-D` nao virou ganho automatico.

## Resultados por parcela de validacao

| Variante | `plot_01` F1 | `plot_06` F1 |
|---|---:|---:|
| `strict_focal_xyz` | 0.5910 | 0.5101 |
| `strict_redecode_wide` | 0.5554 | 0.5333 |
| `strict_multifeature_focal` | 0.5249 | 0.5355 |
| `strict_multifeature_bce` | 0.5452 | 0.5071 |

Arquivos:

- [strict_redecode_wide metrics](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_pointnet_improvements/strict_redecode_wide/metrics_by_plot.csv)
- [strict_focal_xyz metrics](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_pointnet_improvements/strict_focal_xyz/metrics_by_plot.csv)
- [strict_multifeature_bce metrics](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_pointnet_improvements/strict_multifeature_bce/metrics_by_plot.csv)
- [strict_multifeature_focal metrics](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_pointnet_improvements/strict_multifeature_focal/metrics_by_plot.csv)

## Conclusao objetiva

As mudancas pedidas produziram melhora grande e verificavel, mas nao do jeito inicialmente esperado.

- a maior melhora veio de `threshold + NMS` bem calibrados;
- `Focal Loss` ajudou um pouco alem disso;
- `features multicanal` nao ajudaram neste benchmark;
- o melhor resultado final foi `strict_focal_xyz` com `F1_val = 0.5505`.

Isso deixa o PointNet fiel praticamente empatado com o melhor baseline atual, mas ainda um pouco abaixo:

- `strict_focal_xyz`: `0.5505`
- `CHM + watershed`: `0.5614`

Em outras palavras, o metodo passou de claramente inferior para competitivamente proximo do melhor baseline, mas ainda nao o superou nesta validacao.
