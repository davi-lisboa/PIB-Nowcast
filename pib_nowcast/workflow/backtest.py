# %% Bibliotecas
import sys
import gc
import ast
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import datetime as dt

from statsmodels.tsa.api import DynamicFactorMQ

# Certifique-se de que pib_nowcast está no PYTHONPATH
from pib_nowcast.config import SERIES_SPEC, START_DATE, OUTLIER_THRESHOLD, DATA_DIR, SKIP_SEAS_ADJ, MODEL_PARAMS_FILE
from pib_nowcast.utils.get_data import get_data_parallel
from pib_nowcast.utils.transformations import seas_adj_stl_parallel, make_stationary, deflate, remove_outliers
from pib_nowcast.utils.model_builder import build_dfm
from pib_nowcast.utils.transformations.transform_pipeline import QUARTERLY_YOY_LIKE_IDS, QUARTERLY_QOQ_LIKE_IDS

# %% 1. Configurações do Backtest

# Trimestres alvo e os meses nos quais a previsão será simulada.
# 2025Q4 -> Data de Referência PIB: '2025-12-01'
TARGETS = {
    '2025Q3': {
            'pib_date': pd.Timestamp('2025-09-01'),
            'simulation_months': [
                pd.Timestamp('2025-07-31'),
                pd.Timestamp('2025-08-31'),
                pd.Timestamp('2025-09-30'),
                pd.Timestamp('2025-10-31'),
                pd.Timestamp('2025-11-30'),
            ]},

    '2025Q4': {
        'pib_date': pd.Timestamp('2025-12-01'),
        'simulation_months': [
            pd.Timestamp('2025-10-31'),
            pd.Timestamp('2025-11-30'),
            pd.Timestamp('2025-12-31'),
            pd.Timestamp('2026-01-31'),
            pd.Timestamp('2026-02-28')
        ]
    },
    '2026Q1': {
        'pib_date': pd.Timestamp('2026-03-01'),
        'simulation_months': [
            pd.Timestamp('2026-01-31'),
            pd.Timestamp('2026-02-28'),
            pd.Timestamp('2026-03-31'),
            pd.Timestamp('2026-04-30'),
            pd.Timestamp('2026-05-31')
        ]
    }
}

RESULTS_FILE = DATA_DIR / 'backtest_results.xlsx'

# %% 2. Coleta de Dados Base

print(f"[{dt.datetime.now().time()}] Carregando metadados e baixando a base completa para a simulação...")
specs_df = pd.read_csv(SERIES_SPEC, sep=';')
start_date = START_DATE
fit_start_date = '2002-01-01'

full_data_raw = get_data_parallel(specs_df, start_date)

factors = specs_df.set_index('variable')['factors'].to_dict()
factors = {
    k: ast.literal_eval(v) if isinstance(v, str) else v
    for k, v in factors.items()
}

from pib_nowcast.config import MODEL_PARAMS_FILE
# %% 2.5 Preparação do Modelo Base (One-Time Fit)
print(f"[{dt.datetime.now().time()}] Preparando o modelo base a partir dos parâmetros fixos (dfm_params.csv)...")

if not MODEL_PARAMS_FILE.exists():
    raise FileNotFoundError(f"Arquivo {MODEL_PARAMS_FILE} não encontrado. Por favor, rode o pipeline completo uma vez para treinar e salvar os parâmetros do modelo e evitar estouro de memória no backtest.")

params = pd.read_csv(MODEL_PARAMS_FILE, index_col=0).squeeze("columns")

# Extraímos a lista exata de variáveis (colunas) que o dfm_params.csv espera.
# Isso garante que mesmo que alguma API tenha falhado hoje, o modelo vai
# ler os NaNs perfeitamente sem quebrar por descasamento de matriz.
expected_vars = []
for idx in params.index:
    if '->' in idx:
        expected_vars.append(idx.split('->')[1])
expected_columns = list(dict.fromkeys(expected_vars)) # unique mantendo ordem

print(f"[{dt.datetime.now().time()}] {len(expected_columns)} colunas esperadas pelo modelo foram encontradas nos parâmetros salvos.")

# Forçamos a base crua a ter exatamente as colunas que o modelo espera
full_data_raw = full_data_raw.reindex(columns=expected_columns)

# Tratamos a base toda uma vez apenas para construir a estrutura do modelo
full_data_defl = deflate(full_data_raw, specs_df)
if not SKIP_SEAS_ADJ:
    full_data_sa = seas_adj_stl_parallel(full_data_defl, specs_df)
else:
    full_data_sa = full_data_defl.copy()
full_data_stat = make_stationary(full_data_sa, specs_df)
full_data_stat = full_data_stat.loc[fit_start_date:, :]
full_data_stat = remove_outliers(full_data_stat, threshold=OUTLIER_THRESHOLD)

# IMPORTANTE: No backtest.py garantimos que as colunas e a ordem são as de dfm_params.csv,
# logo podemos inferir k_endog_monthly contando quantas dessas são mensais na spec.
# Filtrar specs apenas para as colunas esperadas
k_endog_monthly_base = min(full_data_stat.shape[1], specs_df[specs_df['variable'].isin(expected_columns)].query("frequency == 'Monthly'").shape[0])

model_base = build_dfm(
    endog_data=full_data_stat,
    k_endog_monthly=k_endog_monthly_base,
    factors=factors
)

trained_model = model_base.filter(params)

# Limpeza da base total para livrar RAM
del full_data_defl, full_data_sa, full_data_stat
gc.collect()

# %% 3. Loop de Simulação

results = []

for target_name, target_info in TARGETS.items():
    pib_target_date = target_info['pib_date']
    
    print(f"\n=======================================================")
    print(f" Iniciando Backtest para {target_name} (Alvo: {pib_target_date.date()})")
    print(f"=======================================================\n")
    
    for sim_date in target_info['simulation_months']:
        print(f"[{dt.datetime.now().time()}] Simulando visão de dados em: {sim_date.date()}")
        
        sim_data = full_data_raw.copy()
        
        # TRUNCAMENTO CRONOLÓGICO:
        sim_data.loc[sim_data.index > sim_date, :] = np.nan
        
        print("  -> Tratando dados...")
        sim_data_defl = deflate(sim_data, specs_df)
        
        if not SKIP_SEAS_ADJ:
            sim_data_sa = seas_adj_stl_parallel(sim_data_defl, specs_df)
        else:
            sim_data_sa = sim_data_defl.copy()
            
        sim_data_stat = make_stationary(sim_data_sa, specs_df)
        sim_data_stat = sim_data_stat.loc[fit_start_date:, :]
        sim_data_stat = remove_outliers(sim_data_stat, threshold=OUTLIER_THRESHOLD)
        
        k_endog_monthly = min(sim_data_stat.shape[1], specs_df.query("frequency == 'Monthly' ").shape[0])
        
        print("  -> Aplicando DynamicFactorMQ (filter)...")
        try:
            model = build_dfm(
                endog_data=sim_data_stat,
                k_endog_monthly=k_endog_monthly,
                factors=factors
            )
            # Rodamos apenas o filtro (filter) e não o suavizador (smooth),
            # pois o suavizador aloca as gigantescas matrizes de covariância (predicted_cov) 
            # e causa o ArrayMemoryError. Para a projeção final (predict), o filtro basta!
            try: model_res = model.smooth(params) 
            except: model_res = model.filter(params)
            
            # Predict
            pred = model_res.predict(start=pib_target_date, end=pib_target_date)
            predicted_value = pred['pib'].iloc[0] if 'pib' in pred.columns and not pred.empty else np.nan
            
            # actual_value = full_data_raw.loc[pib_target_date, 'pib'] if pib_target_date in full_data_raw.index else np.nan

            if pib_target_date in full_data_raw.index:
                pib_t_id = specs_df.query('variable == "pib"')['transformation_id'].iloc[0]
                if pib_t_id in QUARTERLY_QOQ_LIKE_IDS:
                    actual_value = (
                                    full_data_raw
                                    .loc[:, ['pib']]
                                    .dropna()
                                    .pct_change(1).multiply(100)
                                    .loc[pib_target_date, :]
                                )
                
                elif pib_t_id in QUARTERLY_YOY_LIKE_IDS:
                    actual_value = (
                                    full_data_raw
                                    .loc[:, ['pib']]
                                    .dropna()
                                    .pct_change(4).multiply(100)
                                    .loc[pib_target_date, :]
                                )
                # actual_value = full_data_raw.loc[pib_target_date, 'pib']

            else:
                actual_value = np.nan
            
            print(f"  => Predito: {predicted_value:.4f} | Realizado: {actual_value}")
            
            results.append({
                'Target_Quarter': target_name,
                'PIB_Date': pib_target_date.date(),
                'Simulation_Date': sim_date.date(),
                'Predicted_PIB_YoY': predicted_value,
                'Actual_PIB_YoY': actual_value,
                'Loglikelihood': model_res.llf
            })
            
        except Exception as e:
            print(f"  [ERRO] Falha ao treinar modelo para {sim_date.date()}: {e}")
            results.append({
                'Target_Quarter': target_name,
                'PIB_Date': pib_target_date.date(),
                'Simulation_Date': sim_date.date(),
                'Predicted_PIB_YoY': np.nan,
                'Actual_PIB_YoY': np.nan,
                'Loglikelihood': np.nan
            })
            
        # Clean up explicitly to prevent memory leaks
        try:
            del sim_data, sim_data_defl, sim_data_sa, sim_data_stat, model, model_res, pred
        except Exception:
            pass
        gc.collect()

# %% 4. Salvando Resultados

print(f"\n[{dt.datetime.now().time()}] Consolidando e salvando resultados...")
results_df = pd.DataFrame(results)

with pd.ExcelWriter(RESULTS_FILE) as writer:
    results_df.to_excel(writer, index=False, sheet_name='Backtest')

print(f"Backtest finalizado! Resultados salvos em {RESULTS_FILE}")
