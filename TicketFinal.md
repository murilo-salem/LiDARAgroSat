Épico 0 — Congelar o benchmark e a régua de comparação
TICKET 0.1 — Definir benchmark oficial da próxima iteração

Objetivo
Fixar o protocolo principal para evitar comparações frágeis entre regimes diferentes.

Escopo

Definir dois protocolos oficiais:
Protocolo A: suíte comparável de 3 splits.
Protocolo B: benchmark fiel em 10 parcelas para métodos sem treino ou com treino local leve.
Fixar métricas obrigatórias:
precision, recall, F1
location_mae_m
height_mae_m
contagem de TP/FP/FN
Fixar baselines obrigatórias em toda tabela:
baseline_local_maxima
CHM + watershed
hybrid_proposal_refinement_multimodal_v2

Critério de aceite

Documento docs/benchmark_protocol_v2.md criado.
Scripts de avaliação passam a rejeitar runs sem protocolo identificado.
Toda saída agregada carrega protocol_id.

Testes

Teste unitário de serialização do protocol_id.
Teste de regressão garantindo que evaluate_experiments.py separa corretamente os protocolos.
TICKET 0.2 — Padronizar naming e manifesto dos experimentos

Objetivo
Evitar novas ambiguidades entre famílias de runs.

Escopo

Criar esquema de nome:
family.stage1.stage2.labels.protocol.seed
Adicionar experiment_manifest.json em toda run com:
dataset
split
estágio 1
refinador
política de labels
parâmetros de decode
commit hash

Critério de aceite

Toda run nova gera manifesto válido.
Agregador lê manifesto e monta tabela consolidada sem heurística por nome de pasta.

Testes

Unitário para schema do manifesto.
Integração: agregação de 3 runs sintéticas com metadados diferentes.
Épico 1 — Tornar CHM + watershed o estágio 1 oficial
TICKET 1.1 — Extrair Stage-1 API unificada

Objetivo
Padronizar geradores de propostas/instâncias.

Escopo

Criar interface:
Stage1Generator.generate(plot_bundle) -> Stage1Output
Implementar adaptadores para:
local_maxima
watershed
adaptive_local_maxima
Stage1Output deve conter:
centros candidatos
segmentos/máscaras quando existirem
score geométrico
features geométricas básicas
metadados de origem

Critério de aceite

Os três métodos rodam via mesma interface.
O pipeline downstream deixa de depender de implementação específica.

Testes

Unitários para contrato da interface.
Teste de consistência: saída contém coordenadas válidas em CRS/pixel.
TICKET 1.2 — Promover CHM + watershed para baseline canônica Stage-1

Objetivo
Estabelecer o melhor estágio geométrico atual como default experimental.

Escopo

Portar a melhor configuração já validada.
Adicionar script:
scripts/run_stage1_watershed.py
Salvar:
segmentos
centros por segmento
área
altura máxima
estatísticas de forma

Critério de aceite

Reproduz o patamar esperado do watershed no benchmark fiel.
Outputs ficam prontos para alimentar refinadores.

Testes

Regressão de F1 no benchmark fiel dentro de tolerância.
Teste de estabilidade numérica em 2 resoluções diferentes.
TICKET 1.3 — Implementar adaptive watershed

Objetivo
Permitir parâmetros dependentes da estrutura local do dossel.

Escopo

Tornar adaptativos:
min_distance
min_segment_area
suavização
regra de seed
Variáveis condicionantes:
altura local
vertical_roughness
densidade local de árvores
rgb_entropy opcional
Suportar políticas:
tabela de regras
regressão simples
lookup por bins

Critério de aceite

Método roda nos mesmos scripts do watershed fixo.
É possível comparar fixo vs adaptativo apenas trocando config.

Testes

Unitário das regras adaptativas.
Teste sintético: CHM simples com duas regiões de complexidade distinta.
Benchmark A/B: watershed_fixo versus adaptive_watershed.
TICKET 1.4 — Diagnóstico de under/over-segmentation

Objetivo
Medir diretamente o gargalo do watershed.

Escopo

Implementar métricas por segmento:
under_segmented
over_segmented
mixed_segment
Relacionar isso com descritores da parcela.

Critério de aceite

Relatório por plot e agregado.
CSV watershed_diagnostics.csv.

Testes

Casos sintéticos com segmentos perfeitos, subsegmentados e supersegmentados.
Épico 2 — Refinadores locais sobre propostas geométricas
TICKET 2.1 — Dataset local por proposta/segmento

Objetivo
Criar base unificada para os refinadores locais.

Escopo

Implementar LocalProposalDataset.
Para cada proposta/segmento extrair:
crop RGB
crop CHM
crop LiDAR estrutural
recorte de pontos 3D
features geométricas do segmento
label de match com GT
Targets:
objectness
dx, dy
delta_h
split_flag
duplicate_flag

Critério de aceite

Dataset gera amostras determinísticas por seed.
Suporta modo segment-centered e proposal-centered.

Testes

Unitário de shape dos tensores.
Teste de alinhamento entre crop raster e pontos.
Teste de balanceamento positivo/negativo.
TICKET 2.2 — Refinador raster multimodal local

Objetivo
Treinar refinador local usando RGB + CHM + LiDAR estrutural.

Escopo

Backbone leve 2D local.
Heads:
classificação
offset
altura
split opcional
Entradas:
RGB
CHM
tensor estrutural LiDAR
features escalares do segmento

Critério de aceite

Roda fim a fim com watershed como estágio 1.
Produz tabela stage2_raster_summary.csv.

Testes

Unitário do forward.
Integração treino curto em subset.
Avaliação em 1 split fixo para sanity check.
TICKET 2.3 — Refinador point-based local

Objetivo
Explorar a melhor pista do PointNet: refinamento local 3D.

Escopo

Implementar variante local baseada em pontos:
xyz
normalized_z
intensidade opcional
Heads:
objectness
dx, dy
delta_h
Amostragem de pontos por segmento/proposta.

Critério de aceite

Reusa pipeline de propostas geométricas.
Reproduz comportamento esperado do eixo PointNet local.

Testes

Unitário de sampling de pontos.
Teste de invariância aproximada a permutação.
Integração com proposals reais em 1 plot.
TICKET 2.4 — Refinador híbrido local

Objetivo
Combinar sinal raster e point cloud no nível local.

Escopo

Encoder 2D local + encoder point-based local.
Fusão tardia ou por gating.
Comparar contra refinadores individuais.

Critério de aceite

Mesma API de inferência dos refinadores locais.
Tabela comparativa:
raster
point
hybrid

Testes

Unitário do forward multimodal.
Integração com Stage1Output.
Benchmark comparativo no protocolo principal.
TICKET 2.5 — Pós-processamento Stage-2

Objetivo
Padronizar decisão final após refinamento.

Escopo

Threshold de objectness
NMS final
merge/split rule
remoção de duplicatas
escolha de centro final:
geométrico
refinado
híbrido

Critério de aceite

Toda família Stage-2 usa o mesmo pós-processamento configurável.
Decode tuning fica isolado do treino.

Testes

Unitários de NMS/merge/split.
Teste com candidatos duplicados sintéticos.
Épico 3 — Estudo formal de ruído centro↔ápice
TICKET 3.1 — Pipeline de auditoria centro↔ápice reproduzível

Objetivo
Transformar a auditoria já feita em artefato oficial do projeto.

Escopo

Script:
scripts/audit_center_apex_noise.py
Medidas por árvore:
distância centro→ápice
diferença de altura
tipo estrutural local
Agregações por:
parcela
espécie
faixa de altura
densidade local

Critério de aceite

Gera:
center_apex_audit.csv
center_apex_summary.json
histogramas

Testes

Unitário de cálculo de distância.
Regressão dos agregados já observados.
TICKET 3.2 — Política de labels center, apex-snap, mixed

Objetivo
Permitir treino sob três regimes de supervisão.

Escopo

Implementar label modes:
field_center
apex_snapped
mixed_center_apex
Parâmetros:
raio máximo de snap
peso da componente snapped
fallback quando apex não é confiável

Critério de aceite

Mesmo dataset local e global suportam os três modos.
Config controlada por YAML.

Testes

Unitário do snapping.
Casos sintéticos com apex dentro/fora do raio.
Teste de consistência entre modos.
TICKET 3.3 — Ablação de ruído de label

Objetivo
Medir efeito de cada política de label nos refinadores.

Escopo

Rodar pelo menos:
watershed + raster refiner
watershed + point refiner
Em cada um:
center
apex-snap
mixed

Critério de aceite

Tabela label_policy_ablation.csv.
Conclusão clara sobre efeito em F1, location_mae_m.

Testes

Reprodutibilidade por seed.
Script de agregação específico.
Épico 4 — Linha alternativa de formulação por instância/copa
TICKET 4.1 — Framework de targets de instância

Objetivo
Preparar o repositório para sair de alvo puramente pontual.

Escopo

Criar módulo de targets alternativos:
distance_transform
seedness
pseudo_crown_mask
star_convex_rays se viável
Fonte inicial:
pseudo-copas derivadas de watershed
ou regiões fracas ao redor do GT

Critério de aceite

Targets salvos por plot.
Visualização disponível para QA.

Testes

Unitário dos mapas gerados.
Teste de alinhamento com GT.
TICKET 4.2 — Baseline de distance-transform + watershed aprendida

Objetivo
Testar formulação mais próxima de instância sem entrar direto em StarDist completo.

Escopo

Rede 2D local ou full-plot leve para prever:
mapa de distância
seedness
Separação final por watershed controlado.

Critério de aceite

Rodada comparável contra center-based e watershed puro.
Output integrado ao mesmo matching.

Testes

Unitário do decoder.
Sanity check visual em 2 plots.
Benchmark no protocolo A.
TICKET 4.3 — Spiking experiment de representação StarDist-like

Objetivo
Validar se vale investir pesado em representação star-convex.

Escopo

Implementação mínima:
raios por centro
loss simples
decode de polígonos
Restrito a subset de plots inicialmente.

Critério de aceite

Spike funcional, não necessariamente produção.
Decisão explícita:
continuar
congelar
descartar

Testes

Forward/backward.
Visualização de polígonos previstos.
Pequena ablação em subset.
Épico 5 — Comparativos oficiais e matriz experimental
TICKET 5.1 — Matriz oficial de experimentos da tese principal

Objetivo
Executar a comparação mínima que sustenta a narrativa científica.

Escopo
Rodar:

Stage-1
watershed_fixo
adaptive_watershed
Stage-2
sem refinador
raster_refiner
point_refiner
hybrid_refiner
Labels
field_center
apex_snapped
mixed_center_apex

Critério de aceite

Matriz central executada no protocolo principal.
Tabela única main_matrix_results.csv.

Testes

Checagem automática de completude da matriz.
Validação de que nenhuma célula ficou sem run.
TICKET 5.2 — Comparativo com e sem RGB

Objetivo
Responder se RGB agrega no refinamento local.

Escopo

Para refinador raster e híbrido:
LiDAR + CHM
RGB + CHM
RGB + LiDAR + CHM

Critério de aceite

Tabela rgb_contribution_ablation.csv.

Testes

Integridade dos canais de entrada.
Reprodutibilidade do split.
TICKET 5.3 — Comparativo com e sem CHM explícito no refinamento local

Objetivo
Quantificar o peso do CHM no estágio 2.

Escopo

Rodar:
sem CHM
com CHM
com CHM + geom features

Critério de aceite

Conclusão clara sobre utilidade do CHM no refinamento, sem confundir com o Stage-1.

Testes

Unitários de montagem das entradas.
TICKET 5.4 — Comparativo Stage-1-only vs Stage-1+Stage-2

Objetivo
Medir o ganho líquido real do refinamento aprendido.

Escopo
Comparar:

local_maxima
watershed
adaptive_watershed
watershed + raster
watershed + point
watershed + hybrid
adaptive_watershed + raster
adaptive_watershed + point
adaptive_watershed + hybrid

Critério de aceite

Tabela final pronta para paper.

Testes

Agregação automática.
Teste de ordenação por F1.
Épico 6 — Ferramentas de análise e escrita do paper
TICKET 6.1 — Relatório automático de ganho por estágio

Objetivo
Separar claramente ganho geométrico do ganho aprendido.

Escopo

Criar relatório:
ganho Stage-1
ganho Stage-2
ganho por política de label
ganho por tipo de parcela

Critério de aceite

CSVs e markdown gerados automaticamente.

Testes

Regressão sobre resultados sintéticos.
TICKET 6.2 — Relatório de falhas por tipo

Objetivo
Entender onde cada refinador ajuda.

Escopo

Expandir categorias:
under-segmentation
over-segmentation
duplicate
off-apex
border
occluded_neighbor
Relacionar falhas com método.

Critério de aceite

error_modes_stage2.csv gerado.

Testes

Unitários das heurísticas de classificação.
TICKET 6.3 — Gerador de tabelas e figuras finais

Objetivo
Fechar a narrativa do paper sem trabalho manual disperso.

Escopo

Tabelas:
matriz principal
ablação de labels
ablação RGB
Stage-1 vs Stage-2
Figuras:
exemplos bons
falhas típicas
centro↔ápice
under/over-segmentation

Critério de aceite

paper_tables/ e paper_figures/ gerados por script.

Testes

Smoke test de geração completa.
Ordem recomendada de execução
Sprint 1 — Infra + Stage-1
TICKET 0.1
TICKET 0.2
TICKET 1.1
TICKET 1.2
TICKET 1.3
TICKET 1.4
Sprint 2 — Dataset local + refinadores
TICKET 2.1
TICKET 2.2
TICKET 2.3
TICKET 2.5
Sprint 3 — Ruído de labels
TICKET 3.1
TICKET 3.2
TICKET 3.3
Sprint 4 — Refinador híbrido + comparativos
TICKET 2.4
TICKET 5.1
TICKET 5.2
TICKET 5.3
TICKET 5.4
Sprint 5 — Linha alternativa de instância/copa
TICKET 4.1
TICKET 4.2
TICKET 4.3
Sprint 6 — Paper readiness
TICKET 6.1
TICKET 6.2
TICKET 6.3
Definição de pronto do plano

O plano estará realmente completo quando você tiver respondido, com números comparáveis, estas 6 perguntas:

adaptive_watershed supera watershed fixo?
O refinamento local aprendido agrega ganho real sobre o Stage-1 puro?
O melhor refinador é raster, point ou hybrid?
Corrigir centro↔ápice melhora o resultado?
RGB agrega valor real no estágio local?
Vale a pena migrar do problema de centro para o problema de instância/copa?