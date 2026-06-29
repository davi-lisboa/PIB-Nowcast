import sys
import psutil
import os
import pandas as pd
import datetime as dt

def get_mem():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 ** 2

def p(msg):
    print(msg, get_mem(), 'MB', flush=True)

p('Mem init:')
import ast
from statsmodels.tsa.api import DynamicFactorMQ
from pib_nowcast.config import SERIES_SPEC, LAST_DATA, DATA_DIR
from pib_nowcast.utils.get_data import get_data
from pib_nowcast.utils.transformations import seas_adj_stl_parallel, make_stationary
from pib_nowcast.utils.news import get_news_impacts, get_new_forecasts

p('Mem imports:')

specs_df = pd.read_csv(SERIES_SPEC, sep=';')
start_date = '1996-01-01'

old_full_data = pd.read_excel(LAST_DATA, sheet_name='full_dataset', index_col='Date')
p('Mem old data:')

new_full_data = old_full_data.copy()
new_full_data.iloc[-1, 0] = new_full_data.iloc[-1, 0] + 0.1 # fake change to avoid exit
p('Mem new data:')

old_full_data_sa = seas_adj_stl_parallel(old_full_data, specs_df)
new_full_data_sa = seas_adj_stl_parallel(new_full_data, specs_df)
old_full_data_stat = make_stationary(old_full_data_sa, specs_df)
new_full_data_stat = make_stationary(new_full_data_sa, specs_df)
p('Mem stats:')

factors = specs_df.set_index('variable')['factors'].to_dict()
factors = {k: ast.literal_eval(v) if isinstance(v, str) else v for k, v in factors.items()}

old_model = DynamicFactorMQ(
    endog = old_full_data_stat,
    k_endog_monthly = specs_df.query("frequency == 'Monthly' ").shape[0],
    factors = factors,
    factor_orders = 3,
)

p('Mem before fit:')
old_model_res = old_model.fit()
p('Mem after fit:')

new_model = old_model_res.apply(
    endog = new_full_data_stat,
    k_endog_monthly = specs_df.query("frequency == 'Monthly' ").shape[0],
)
p('Mem after apply:')

next_pib_quarter_timestamp = dt.date(2024, 12, 1) # dummy
try:
    print('Starting news...', flush=True)
    news = new_model.news(
        comparison=old_model_res, 
        impacted_variable='pib', 
        impact_date='2024-12-01',
        comparison_type='previous'
    )
    p('Mem after news:')
except Exception as e:
    print('Error in news:', e, flush=True)
