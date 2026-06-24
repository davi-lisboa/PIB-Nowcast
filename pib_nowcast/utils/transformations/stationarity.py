"""Testes agregados de estacionaridade para séries temporais.

Combina ADF, KPSS (statsmodels) e Phillips-Perron, DFGLS (arch) num
DataFrame resumo, com veredicto por maioria.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

logger = logging.getLogger(__name__)


def stationarity_tests(
    series: pd.Series,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Executa bateria de testes de estacionaridade sobre uma série.

    Parameters
    ----------
    series : pd.Series
        Série temporal univariada (sem NaN).
    alpha : float
        Nível de significância para os testes (default 5 %).

    Returns
    -------
    pd.DataFrame
        Colunas: ``test``, ``statistic``, ``p_value``, ``lags``,
        ``is_stationary``.  Uma linha por teste.
    """
    series = series.dropna()

    if len(series) < 20:
        logger.warning(
            f"Série com apenas {len(series)} obs — resultados podem ser pouco confiáveis."
        )

    results: list[dict] = []

    def _append_result(name: str, statistic, p_value, lags, alpha):
        """Adiciona resultado de um teste, tratando NaN em p_value."""
        if p_value is None or (isinstance(p_value, float) and np.isnan(p_value)):
            is_stat = None
        elif name == "KPSS":
            is_stat = p_value >= alpha  # H₀ invertida
        else:
            is_stat = p_value < alpha
        results.append({
            "test": name,
            "statistic": statistic,
            "p_value": p_value,
            "lags": lags,
            "is_stationary": is_stat,
        })

    def _append_nan(name: str):
        """Registra teste que falhou com NaN."""
        results.append({
            "test": name,
            "statistic": np.nan,
            "p_value": np.nan,
            "lags": np.nan,
            "is_stationary": None,
        })

    # --- ADF (H₀: raiz unitária → não estacionário) ---
    try:
        adf_stat, adf_p, adf_lags, *_ = adfuller(series, autolag="AIC")
        _append_result("ADF", adf_stat, adf_p, adf_lags, alpha)
    except Exception:
        logger.debug("Falha no teste ADF.", exc_info=True)
        _append_nan("ADF")

    # --- KPSS (H₀: estacionário → p < α significa NÃO estacionário) ---
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_stat, kpss_p, kpss_lags, _ = kpss(series, regression="c", nlags="auto")
        _append_result("KPSS", kpss_stat, kpss_p, kpss_lags, alpha)
    except Exception:
        logger.debug("Falha no teste KPSS.", exc_info=True)
        _append_nan("KPSS")

    # --- Phillips-Perron (H₀: raiz unitária → não estacionário) ---
    try:
        from arch.unitroot import PhillipsPerron
        pp = PhillipsPerron(series)
        _append_result("Phillips-Perron", pp.stat, pp.pvalue, pp.lags, alpha)
    except Exception:
        logger.debug("Falha no teste Phillips-Perron.", exc_info=True)
        _append_nan("Phillips-Perron")

    # --- DFGLS (H₀: raiz unitária → não estacionário) ---
    try:
        from arch.unitroot import DFGLS
        dfgls = DFGLS(series)
        _append_result("DFGLS", dfgls.stat, dfgls.pvalue, dfgls.lags, alpha)
    except Exception:
        logger.debug("Falha no teste DFGLS.", exc_info=True)
        _append_nan("DFGLS")

    return pd.DataFrame(results)


def is_stationary(
    series: pd.Series,
    alpha: float = 0.05,
) -> bool:
    """Retorna ``True`` se a maioria dos testes indicar estacionaridade.

    Parameters
    ----------
    series : pd.Series
        Série temporal univariada.
    alpha : float
        Nível de significância.

    Returns
    -------
    bool
        ``True`` se ≥ metade + 1 dos testes indicarem estacionaridade.
    """
    df = stationarity_tests(series, alpha=alpha)
    valid = df["is_stationary"].dropna()
    if valid.empty:
        return False
    n_stationary = valid.sum()
    return n_stationary > len(valid) / 2
