# %%
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from statsmodels.tsa.x13 import x13_arima_analysis
from statsmodels.tsa.seasonal import STL

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


def _process_single_series(col: str, series: pd.DataFrame, x13_path: str) -> tuple[str, pd.Series | None]:
    """Função auxiliar para processar uma única série no X-13 ARIMA."""
    if series.empty or len(series) < 36:
        logger.warning(f"Coluna '{col}' tem {len(series)} obs (mínimo ~36 para X-13), pulando.")
        return col, None

    try:
        sa_result = x13_arima_analysis(
            endog=series,
            trading=True,
            x12path=x13_path,
        ).seasadj
        return col, sa_result
    except Exception:
        logger.exception(f"Erro no ajuste sazonal da série '{col}', mantendo original.")
        return col, None


def seas_adj_parallel(
        base_df: pd.DataFrame,
        specs_df: pd.DataFrame,
        x13_path: str | None = None,
        max_workers: int | None = None,
) -> pd.DataFrame:
    """Aplica ajuste sazonal X-13 ARIMA às séries marcadas com seas_adj == 0 de forma paralela.

    Parameters
    ----------
    base_df : pd.DataFrame
        DataFrame com as séries temporais (índice datetime).
    specs_df : pd.DataFrame
        Especificações das séries; deve conter colunas 'variable' e 'seas_adj'.
    x13_path : str | None
        Caminho para o executável do X-13. Se ``None``, usa ``X13_PATH`` do config.
    max_workers: int | None
        Número máximo de threads a serem utilizadas no pool de execução.
        Por padrão, utiliza min(32, os.cpu_count() + 4).

    Returns
    -------
    pd.DataFrame
        Cópia de ``base_df`` com as séries dessazonalizadas.
    """
    if x13_path is None:
        x13_path = str(X13_PATH)

    non_sa_vars = specs_df.query("seas_adj == 0")["variable"].to_list()
    base_df_sa = base_df.copy(deep=True)

    # Coleta todas as séries válidas para despachar
    jobs = {}
    for col in non_sa_vars:
        if col not in base_df_sa.columns:
            logger.warning(f"Coluna '{col}' não encontrada no DataFrame, pulando.")
            continue
            
        series = base_df_sa[[col]].dropna()
        jobs[col] = series

    # Se não houver max_workers, limita a um número seguro (ex: 4) para evitar esgotamento de recursos (WinError 1455)
    if max_workers is None:
        import os
        max_workers = min(4, (os.cpu_count() or 1))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_single_series, col, series, x13_path): col
            for col, series in jobs.items()
        }

        for future in as_completed(futures):
            col = futures[future]
            try:
                processed_col, sa_result = future.result()
                if sa_result is not None:
                    base_df_sa[processed_col] = sa_result.reindex(base_df_sa.index)
            except Exception as e:
                logger.error(f"Erro inesperado na thread da coluna '{col}': {e}")

    return base_df_sa


def seas_adj_stl(
        base_df: pd.DataFrame,
        specs_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aplica ajuste sazonal STL às séries marcadas com seas_adj == 0.
    
    Parameters
    ----------
    base_df : pd.DataFrame
        DataFrame com as séries temporais (índice datetime).
    specs_df : pd.DataFrame
        Especificações das séries; deve conter colunas 'variable', 'seas_adj' e 'frequency'.

    Returns
    -------
    pd.DataFrame
        Cópia de ``base_df`` com as séries dessazonalizadas.
    """
    non_sa_vars = specs_df.query("seas_adj == 0")["variable"].to_list()
    base_df_sa = base_df.copy(deep=True)

    for col in non_sa_vars:
        if col not in base_df_sa.columns:
            logger.warning(f"Coluna '{col}' não encontrada no DataFrame, pulando.")
            continue

        series = base_df_sa[[col]].dropna()

        freq_str = specs_df.loc[specs_df['variable'] == col, 'frequency'].iloc[0]
        period = {
                    'Quarterly': 4,
                    'Monthly': 12
                }

        if series.empty or len(series) < 2 * period:
            logger.warning(f"Coluna '{col}' tem {len(series)} obs (mínimo {2*period} para STL), pulando.")
            continue

        try:
            s = series[col]
            res = STL(s, period=period, robust=True).fit()
            sa_result = s - res.seasonal
            base_df_sa[col] = sa_result.reindex(base_df_sa.index)
        except Exception:
            logger.exception(f"Erro no ajuste sazonal STL da série '{col}', mantendo original.")

    return base_df_sa


def _process_single_series_stl(col: str, series: pd.DataFrame, period: int = 12) -> tuple[str, pd.Series | None]:
    """Função auxiliar para processar uma única série usando STL."""
    if series.empty or len(series) < 2 * period:
        logger.warning(f"Coluna '{col}' tem {len(series)} obs (mínimo {2*period} para STL), pulando.")
        return col, None

    try:
        s = series[col]
        # Aplica STL. robust=True ajuda com outliers
        res = STL(s, period=period, seasonal=period+1, robust=True).fit()
        # Série com ajuste sazonal = Observado - Sazonal
        sa_result = s - res.seasonal
        return col, sa_result
    except Exception:
        logger.exception(f"Erro no ajuste sazonal STL da série '{col}', mantendo original.")
        return col, None


def seas_adj_stl_parallel(
        base_df: pd.DataFrame,
        specs_df: pd.DataFrame,
        max_workers: int | None = None,
) -> pd.DataFrame:
    """Aplica ajuste sazonal STL às séries marcadas com seas_adj == 0 de forma paralela.

    Parameters
    ----------
    base_df : pd.DataFrame
        DataFrame com as séries temporais (índice datetime).
    specs_df : pd.DataFrame
        Especificações das séries; deve conter colunas 'variable', 'seas_adj' e 'frequency'.
    max_workers: int | None
        Número máximo de threads a serem utilizadas.

    Returns
    -------
    pd.DataFrame
        Cópia de ``base_df`` com as séries dessazonalizadas.
    """
    non_sa_vars = specs_df.query("seas_adj == 0")["variable"].to_list()
    base_df_sa = base_df.copy(deep=True)

    # Mapeamento fora do loop para não recriar o dicionário a cada iteração
    freq_map = {
    'Quarterly': 4,
    'Monthly': 12
            }

    jobs = {}
    for col in non_sa_vars:
        if col not in base_df_sa.columns:
            logger.warning(f"Coluna '{col}' não encontrada no DataFrame, pulando.")
            continue
            
        series = base_df_sa[[col]].dropna()
        freq_str = specs_df.loc[specs_df['variable'] == col, 'frequency'].iloc[0]
        period = freq_map.get(freq_str)

        jobs[col] = (series, period)

    # Para STL, não há problema em usar muitos workers (não usa arquivos temporários)
    if max_workers is None:
        import os
        max_workers = os.cpu_count() or 4

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_single_series_stl, col, series, period): col
            for col, (series, period) in jobs.items()
        }

        for future in as_completed(futures):
            col = futures[future]
            try:
                processed_col, sa_result = future.result()
                if sa_result is not None:
                    base_df_sa[processed_col] = sa_result.reindex(base_df_sa.index)
            except Exception as e:
                logger.error(f"Erro inesperado na thread da coluna '{col}': {e}")

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
