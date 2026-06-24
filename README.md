# PIB-Nowcast

Um modelo de Nowcasting para o Produto Interno Bruto (PIB) brasileiro utilizando Fatores Dinâmicos de Frequência Mista (Dynamic Factor Model M/Q). O objetivo deste projeto é estimar o PIB trimestral em tempo real à medida que novos indicadores mensais são divulgados.

## 🏗️ Arquitetura do Projeto

O pipeline do projeto foi construído para ser modular, automatizado e orientado a metadados (Data-Driven):

- `data/series_spec.csv`: O arquivo mestre de especificação. Adicionar uma variável ao modelo requer apenas a inserção de uma linha neste arquivo informando código, fonte, blocos do fator dinâmico e ID da transformação de estacionaridade.
- `utils/get_data.py`: Módulo responsável pelas coletas via APIs públicas (BCB/SGS, Ipeadata, IBGE/SIDRA).
- `utils/transformations/`:
  - `seas_adj.py`: Integração com o X-13ARIMA-SEATS para expurgo de sazonalidade.
  - `stationarity.py`: Bateria de testes de raiz unitária por consenso (ADF, KPSS, Phillips-Perron, DFGLS) para identificar a ordem de integração.
  - `transform_pipeline.py`: Registro com as rotinas de transformação (Log, Box-Cox, Diff, etc.), mapeadas por um `transformation_id` numérico.
- `workflow/pipeline.py`: O orquestrador central. Ele compara as safras de dados (vintages), roda o ajuste sazonal, estacionariza e passa pelo Filtro de Kalman via `statsmodels.tsa.api.DynamicFactorMQ`.

## ⚙️ Fluxo Atual (Pipeline)

1. **Configuração e Metadados:** Leitura do `series_spec.csv`.
2. **Coleta de Vintages:** Extração da safra "Old" (salva localmente) e da safra "New" (coletada em tempo real).
3. **Tratamento:** Limpeza de sazonalidade via X-13 e aplicação das transformações estabilizadoras e de variância (via wrapper `make_stationary`).
4. **Filtro de Kalman / Fator Dinâmico:** Estimação do modelo `DynamicFactorMQ` para extrair os componentes comuns e lidar com valores faltantes (ragged edges).

## 🚀 Próximos Passos (To-Dos)

- [ ] **Expansão de Variáveis:** Adicionar novos indicadores na especificação `series_spec.csv` (ex: mais variáveis de Mercado de Trabalho, Crédito, Comércio Exterior, Arrecadação) para garantir maior densidade para o Fator Global.
- [ ] **Tuning Econométrico do Target:** Ajustar as restrições da série trimestral do PIB e hiperparâmetros (como `factor_orders`) para garantir que o algoritmo *Expectation-Maximization* (EM) do `statsmodels` atinja convergência em todas as execuções.
- [ ] **Computar Forecasts:** Extrair a projeção estruturada e consolidada gerada pelo modelo (Nowcast out-of-sample). Futuramente, criar módulo de avaliação pseudo-real-time (backtesting).
- [ ] **Computar News (Impacto das Revisões):** Implementar o algoritmo clássico de Nowcasting (e.g. Banbura et al. 2013) que lê o log de projeções e identifica "o quê" fez a expectativa do PIB subir ou descer, isolando o peso (*gain*) de cada nova divulgação de dados.