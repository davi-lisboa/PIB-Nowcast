# %%
import logging

import pandas as pd
from statsmodels.tsa.x13 import x13_arima_analysis

from pib_nowcast.config import X13_PATH, SERIES_SPEC, LAST_DATA

logger = logging.getLogger(__name__)


# %%
def seas_adj(
        base_df: pd.DataFrame,
        specs_df: pd.DataFrame,
        x13_path: str | None = None,
) -> pd.DataFrame:
    """Aplica ajuste sazonal X-13 ARIMA às séries marcadas com seas_adj == 0.

    Parameters
    ----------
    base_df : pd.DataFrame
        DataFrame com as séries temporais (índice datetime).
    specs_df : pd.DataFrame
        Especificações das séries; deve conter colunas 'variable' e 'seas_adj'.
    x13_path : str | None
        Caminho para o executável do X-13. Se ``None``, usa ``X13_PATH`` do config.

    Returns
    -------
    pd.DataFrame
        Cópia de ``base_df`` com as séries dessazonalizadas.
    """
    if x13_path is None:
        x13_path = str(X13_PATH)

    non_sa_vars = specs_df.query("seas_adj == 0")["variable"].to_list()

    base_df_sa = base_df.copy(deep=True)

    for col in non_sa_vars:
        if col not in base_df_sa.columns:
            logger.warning(f"Coluna '{col}' não encontrada no DataFrame, pulando.")
            continue

        series = base_df_sa[[col]].dropna()

        if series.empty or len(series) < 36:
            logger.warning(f"Coluna '{col}' tem {len(series)} obs (mínimo ~36 para X-13), pulando.")
            continue

        try:
            sa_result = x13_arima_analysis(
                endog=series,
                trading=True,
                x12path=x13_path,
            ).seasadj

            # Realinha ao índice original preservando NaNs das pontas
            base_df_sa[col] = sa_result.reindex(base_df_sa.index)

        except Exception:
            logger.exception(f"Erro no ajuste sazonal da série '{col}', mantendo original.")

    return base_df_sa


# %%
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    ### Especifica caminho e primeira data
    specs_df = pd.read_csv(SERIES_SPEC, sep=";")

    ## Dataset completo, última run
    old_full_data = pd.read_excel(
        LAST_DATA, sheet_name="full_dataset", index_col="Date"
    )

    # Ajuste sazonal
    result = seas_adj(old_full_data, specs_df)
    display(result)
