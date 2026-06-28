# %% Bibliotecas
import sys

import pandas as pd
import numpy as np
import datetime as dt

from statsmodels.tsa.api import DynamicFactorMQ

from pib_nowcast.config import SERIES_SPEC, LAST_DATA, DATA_DIR
from pib_nowcast.utils.get_data import get_data
from pib_nowcast.utils.transformations import seas_adj, make_stationary
from pib_nowcast.utils.news import get_news_impacts, get_new_forecasts

# %%

### Especifica caminho e primeira data
specs_df = pd.read_csv(SERIES_SPEC, sep=';')
start_date = '1996-01-01'

### Especificação de datas
today = dt.date.today()

# %% Coletas

## Dataset completo, última run
old_full_data = pd.read_excel(LAST_DATA, sheet_name='full_dataset', index_col='Date')

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

## -> Deflacionar valores
# WIP

## -> Ajuste sazonal
old_full_data_sa = seas_adj(old_full_data, specs_df)
new_full_data_sa = seas_adj(new_full_data, specs_df)

## -> Estacionarização
old_full_data_stat = make_stationary(old_full_data_sa, specs_df)
new_full_data_stat = make_stationary(new_full_data_sa, specs_df)

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
    endog = old_full_data_stat,
    k_endog_monthly = specs_df.query("frequency == 'Monthly' ").shape[0],
    factors = factors,
    factor_orders = 3,
    # endog_qu

)

old_model_res = old_model.fit()

print(old_model_res.summary())

# %%

new_model = old_model_res.apply(
    endog = new_full_data_stat,
    k_endog_monthly = specs_df.query("frequency == 'Monthly' ").shape[0],

)

# TO-DOs
# Ajustar transformações para M/M-like
# gerar primeiro excel/histórico de nowcasts

# separar datas relevantes
last_pib_date_timestamp = old_full_data['pib'].last_valid_index()
next_pib_quarter_timestamp = last_pib_date_timestamp + pd.DateOffset(months=3)

# Estimar news
news = new_model.news(
                        comparison=old_model_res, 
                        impacted_variable='pib', 
                        impact_date=next_pib_quarter_timestamp.strftime('%Y-%m-%d')
                    )


## -> Salvar impactos no histórico
get_news_impacts(news, save_to=DATA_DIR / 'news_impacts.xlsx')


## -> Salvar novos forecasts no histórico
get_new_forecasts(
    news=news, 
    new_model_res=new_model, 
    last_pib_date_timestamp=last_pib_date_timestamp, 
    next_pib_quarter_timestamp=next_pib_quarter_timestamp, 
    save_to=DATA_DIR / 'forecasts.xlsx'
)

