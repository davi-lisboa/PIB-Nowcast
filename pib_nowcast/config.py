from pathlib import Path

# Raiz do projeto: resolve automaticamente independente de OS ou CWD
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Caminhos derivados
DATA_DIR = PROJECT_ROOT / "pib_nowcast" / "data"
X13_PATH = PROJECT_ROOT / "x13as" / "x13as"

SERIES_SPEC = DATA_DIR / "series_spec.csv"
LAST_DATA = DATA_DIR / "last_data_at_time.xlsx"
