# %% Bibliotecas
import os
from pathlib import Path
import sys

import pandas as pd
import numpy as np
import datetime as dt

from statsmodels.tsa.api import DynamicFactorMQ

from pib_nowcast.utils.get_data import get_bcb, get_pib, get_ipeadata

# %% Coletas

## Dataset completo, última run
path_old_full_data = r'../data/last_data_at_time.xlsx'
old_full_data = pd.read_excel(path_old_full_data, sheet_name='full_dataset', index_col='Date')

## Coleta dados mais recentes

### Especifica caminho e primeira data
path_series_spec = r'../data/series_spec.csv'
start_date = '1996-01-01'

### Efetua coletas no SGS, IPEA Data e PIB no SIDRA
bcb_df = get_bcb(series=path_series_spec, start=start_date)
ipea_df = get_ipeadata(series=path_series_spec, start=start_date)
pib_df = get_pib()

### Junta tudo num df só
new_full_data = pd.concat([bcb_df, ipea_df, pib_df], axis=1, join='outer')

# %% Comparação

if old_full_data.equals(new_full_data):
    print("Sem dados novos ou revisões, encerrando processso.")
    sys.exit(0)

else:
    print('Houve atualização/revisão nos dados, prosseguindo com o processo.')
# %%

