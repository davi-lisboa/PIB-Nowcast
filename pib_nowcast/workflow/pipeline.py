# %% Bibliotecas
import os
from pathlib import Path
import sys

import pandas as pd
import numpy as np
import datetime as dt

from statsmodels.tsa.api import DynamicFactorMQ

from pib_nowcast.utils.get_data import get_bcb, get_pib, get_ipeadata, get_data

# %%

### Especifica caminho e primeira data
path_series_spec = r'../data/series_spec.csv'
specs_df = pd.read_csv(path_series_spec, sep=';')
start_date = '1996-01-01'

# %% Coletas

## Dataset completo, última run
path_old_full_data = r'../data/last_data_at_time.xlsx'
old_full_data = pd.read_excel(path_old_full_data, sheet_name='full_dataset', index_col='Date')

## Coleta dados mais recentes

### Junta tudo num df só
new_full_data = get_data(specs_df, start_date)

# %% Comparação

if old_full_data.equals(new_full_data):
    print("Sem dados novos ou revisões, encerrando processso.")
    sys.exit(0)

else:
    print('Houve atualização/revisão nos dados, prosseguindo com o processo.')

# %% Tratamentos 
## -> Ajuste sazonal
non_sa_vars = specs_df.query("seas_adj == 0")['variable'].to_list()

new_full_data_sa = new_full_data.copy(deep=True)

for col in non_sa_vars:
    new_full_data_sa[col] = x13_arima_analysis(new_full_data_sa[col]).seasadj
  
## -> Estacionarização


# %%
import ast

# Extrai os fatores especificados e corrige o tipo dos dados
factors = specs_df.set_index('variable')['factors'].to_dict()
factors = {
    k: ast.literal_eval(v) if isinstance(v, str) 
    else v
    for k, v in factors.items()
}

old_model = DynamicFactorMQ(
    endog = old_full_data,
    k_endog_monthly = specs_df.query("frequency == 'Monthly' ").shape[0],
    factors = factors,
    factor_orders = 4,
    # endog_qu

)

old_model_res = old_model.fit()

print(old_model_res.summary())

# %%

new_model = old_model_res.apply(
    endog = new_full_data,
    k_endog_monthly = specs_df.query("frequency == 'Monthly' ").shape[0],

)