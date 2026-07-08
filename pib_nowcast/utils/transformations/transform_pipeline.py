"""Registro de pipelines de transformação para estacionaridade.

Cada pipeline é uma sequência nomeada de transformações aplicadas sobre
séries temporais sazonalmente ajustadas.  O ID numérico permite referência
compacta em tabelas, CSVs e na interface programática.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import boxcox, yeojohnson
import warnings

# ── Funções elementares de transformação ────────────────────────────────────

def _log(s: pd.Series) -> pd.Series:
    """Logaritmo natural. Requer s > 0."""
    return np.log(s)


def _boxcox(s: pd.Series) -> pd.Series:
    """Box-Cox (lambda estimado). Requer s > 0."""
    vals, _ = boxcox(s.values)
    return pd.Series(vals, index=s.index, name=s.name)


def _yeojohnson(s: pd.Series) -> pd.Series:
    """Yeo-Johnson (lambda estimado). Aceita qualquer sinal."""
    vals, _ = yeojohnson(s.values)
    return pd.Series(vals, index=s.index, name=s.name)


def _diff(s: pd.Series) -> pd.Series:
    """Primeira diferença."""
    return s.diff().dropna()


def _sdiff12(s: pd.Series) -> pd.Series:
    """Diferença sazonal lag=12 (dados mensais)."""
    return s.diff(12).dropna()


def _sdiff4(s: pd.Series) -> pd.Series:
    """Diferença sazonal lag=4 (dados trimestrais)."""
    return s.diff(4).dropna()


def _mom_pct(s: pd.Series) -> pd.Series:
    """Variação percentual MoM."""
    return s.pct_change(1).dropna() * 100

def _yoy_pct_monthly(s: pd.Series) -> pd.Series:
    """Variação percentual YoY (12 meses). Específica para séries mensais."""
    return s.pct_change(12).dropna() * 100


def _qoq_pct(s: pd.Series) -> pd.Series:
    """Variação percentual QoQ."""
    return s.pct_change(1).dropna() * 100


def _yoy_pct_quarterly(s: pd.Series) -> pd.Series:
    """Variação percentual YoY (4 trimestres). Específica para PIB."""
    return s.pct_change(4).dropna() * 100

def _multiply_100(s: pd.Series) -> pd.Series:
    """Multiplica a série por 100."""
    return s * 100


# ── Registro de pipelines ──────────────────────────────────────────────────
#
# Cada entrada: pipeline_id → (nome_legível, [funções], positivo_requerido)
#
# ``positivo_requerido`` indica se a série precisa ser estritamente positiva
# para que o pipeline seja aplicável (por causa de log / box-cox).
#
# IDs 0-12: séries mensais
# IDs 13-17: séries trimestrais (alguns IDs compartilhados com mensais)
#

PIPELINE_REGISTRY: dict[int, tuple[str, list, bool]] = {
    # ── Mensais ──
    0:  ("nível",              [],                           False),
    1:  ("diff",               [_diff],                      False),
    2:  ("log",                [_log],                       True),
    3:  ("log→diff",           [_log, _diff, _multiply_100], True),
    4:  ("boxcox",             [_boxcox],                    True),
    5:  ("boxcox→diff",        [_boxcox, _diff],             True),
    6:  ("yeojohnson",         [_yeojohnson],                False),
    7:  ("yeojohnson→diff",    [_yeojohnson, _diff],         False),
    8:  ("sdiff12",            [_sdiff12],                   False),
    9:  ("sdiff12→diff",       [_sdiff12, _diff],            False),
    10: ("log→sdiff12",        [_log, _sdiff12, _multiply_100], True),
    11: ("log→sdiff12→diff",   [_log, _sdiff12, _multiply_100, _diff], True),
    12: ("boxcox→sdiff12→diff", [_boxcox, _sdiff12, _diff],  True),
    13: ("mom_pct",            [_mom_pct],                   False),
    14: ("yoy_pct_monthly",    [_yoy_pct_monthly],           False),

    # ── Trimestrais ──
    15: ("qoq_pct",            [_qoq_pct],                   False),
    16: ("yoy_pct",            [_yoy_pct_quarterly],         False),
    17: ("sdiff4",             [_sdiff4],                    False),
    18: ("sdiff4→diff",        [_sdiff4, _diff],             False),
    19: ("log→sdiff4",         [_log, _sdiff4, _multiply_100], True),
    20: ("log→sdiff4→diff",    [_log, _sdiff4, _multiply_100, _diff], True),
}

# Mapeamento nome → id (útil para lookups a partir do nome legível)
PIPELINE_NAME_TO_ID: dict[str, int] = {
    name: pid for pid, (name, _, _) in PIPELINE_REGISTRY.items()
}

# IDs válidos para cada frequência
MONTHLY_PIPELINE_IDS: list[int] = list(range(0, 14+1))
QUARTERLY_PIPELINE_IDS: list[int] = [0, 1, 2, 3, 15, 16, 17, 18, 19, 20]


def apply_transform_pipeline(
    series: pd.Series,
    pipeline_id: int,
    *,
    min_obs: int = 20,
) -> pd.Series | None:
    """Aplica o pipeline de transformação identificado por ``pipeline_id``.

    Parameters
    ----------
    series : pd.Series
        Série temporal univariada (preferencialmente já com ajuste sazonal).
    pipeline_id : int
        Identificador numérico do pipeline conforme ``PIPELINE_REGISTRY``.
    min_obs : int
        Número mínimo de observações após cada transformação.
        Se a série ficar menor, retorna ``None``.

    Returns
    -------
    pd.Series | None
        Série transformada, ou ``None`` se o pipeline não for aplicável
        (série muito curta, valores não-positivos para log/boxcox, etc.).

    Raises
    ------
    KeyError
        Se ``pipeline_id`` não existir no registro.
    """
    if pipeline_id not in PIPELINE_REGISTRY:
        raise KeyError(
            f"pipeline_id={pipeline_id} inválido. "
            f"Válidos: {sorted(PIPELINE_REGISTRY.keys())}"
        )

    name, steps, needs_positive = PIPELINE_REGISTRY[pipeline_id]
    s = series.dropna().copy()

    try:
        for fn in steps:
            if needs_positive and fn in (_log, _boxcox) and (s <= 0).any():
                return None
            s = fn(s)
            if s.empty or len(s) < min_obs:
                return None

        if not np.isfinite(s).all():
            return None

        return s

    except Exception:
        return None


def make_stationary(df: pd.DataFrame, specs_df: pd.DataFrame) -> pd.DataFrame:
    """Aplica as transformações de estacionaridade em todas as colunas do DataFrame.
    
    A transformação adequada para cada série é extraída da coluna ``transformation_id``
    do DataFrame ``specs_df``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contendo as séries temporais originais ou sazonalmente ajustadas.
    specs_df : pd.DataFrame
        DataFrame contendo as especificações das séries, incluindo a coluna
        ``transformation_id`` correspondente ao pipeline de cada variável.

    Returns
    -------
    pd.DataFrame
        DataFrame com todas as séries transformadas e alinhadas temporalmente.
    """
    transformed_series = []
    
    for col in df.columns:
        if col not in specs_df['variable'].values:
            continue
            
        pipe_id = specs_df.loc[specs_df['variable'] == col, 'transformation_id'].iloc[0]
        
        # Ignora se não houver transformação definida (-1, nulo, etc.)
        if pd.isna(pipe_id) or pipe_id < 0:
            continue
            
        s_transformed = apply_transform_pipeline(df[col], int(pipe_id))
        if s_transformed is not None:
            transformed_series.append(s_transformed)
        else:
            warnings.warn(
                f"A série '{col}' foi descartada na estacionarização (pipeline_id={pipe_id}). "
                f"Possíveis causas: dados insuficientes ou valores <= 0 em transformações log/boxcox."
            )

    if not transformed_series:
        return pd.DataFrame(index=df.index)

    return pd.concat(transformed_series, axis=1)
