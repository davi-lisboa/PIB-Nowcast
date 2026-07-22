import pandas as pd
from pathlib import Path

# Raiz do projeto: resolve automaticamente independente de OS ou CWD
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Caminhos derivados
DATA_DIR = PROJECT_ROOT / "pib_nowcast" / "data"
X13_PATH = PROJECT_ROOT / "x13as" / "x13as"

SERIES_SPEC = DATA_DIR / "series_spec.csv"
LAST_DATA = DATA_DIR / "last_data_at_time.xlsx"

# Parâmetros Globais
START_DATE = '1996-01-01'
OUTLIER_THRESHOLD = 5
RECESSIONS = [
    pd.date_range(start='2008-10-01', end='2009-03-01', freq='MS').to_list(),
    pd.date_range(start='2014-04-01', end='2016-12-01', freq='MS').to_list(),
    pd.date_range(start='2020-01-01', end='2020-06-01', freq='MS').to_list()
]
