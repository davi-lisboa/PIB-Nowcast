import pandas as pd
from pathlib import Path

# Raiz do projeto: resolve automaticamente independente de OS ou CWD
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Caminhos derivados
DATA_DIR = PROJECT_ROOT / "pib_nowcast" / "data"
X13_PATH = PROJECT_ROOT / "x13as" / "x13as"
FIG_DIR = PROJECT_ROOT / "pib_nowcast" / "figures"

# SERIES_SPEC = DATA_DIR / "series_spec.csv"
SERIES_SPEC = DATA_DIR / "series_spec_MONTHLY.csv"
# SERIES_SPEC = DATA_DIR / "series_spec_ANNUAL.csv"
LAST_DATA = DATA_DIR / ( "last_data_at_time_" + ("MONTHLY" if "MONTHLY" in str(SERIES_SPEC) else "ANNUAL") + ".xlsx" )
MODEL_PARAMS_FILE = DATA_DIR / "dfm_params.csv"

# Controle de ajuste sazonal baseado no tipo de especificação
# Se a spec ativa usa transformações anuais (YoY, sdiff12, etc.),
# o STL não deve ser aplicado para evitar dupla dessazonalização.
SKIP_SEAS_ADJ = 'ANNUAL' in SERIES_SPEC.stem.upper()

# Parâmetros Globais
START_DATE = '1996-01-01'
OUTLIER_THRESHOLD = 3
RECESSIONS = [
    pd.date_range(start='2008-10-01', end='2009-03-01', freq='MS').to_list(),
    pd.date_range(start='2014-04-01', end='2016-12-01', freq='MS').to_list(),
    pd.date_range(start='2020-01-01', end='2020-06-01', freq='MS').to_list()
]
