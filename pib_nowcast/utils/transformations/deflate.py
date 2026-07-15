"""Deflacionamento de series temporais nominais.

Utiliza o IPCA numero-indice (base: dez/1993 = 100) da tabela 1737 do SIDRA
(variavel 2266) como deflator padrao para converter series em valores nominais
(R$) para valores reais a precos constantes de uma data-base.

A coluna ``deflate`` em ``series_spec.csv`` segue a mesma convencao de
``seas_adj``:

- ``0``  -> serie nominal, **precisa** ser deflacionada;
- ``1``  -> serie ja em valores reais, pular;
- ``N/A`` -> nao se aplica (indices, quantidades etc.), pular.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constantes SIDRA ────────────────────────────────────────────────────────
# Tabela 1737 – IPCA: Serie historica com numero-indice, variacao mensal e
#               variacao acumulada (base: dez/1993 = 100)
# Variavel 2266 – IPCA - Numero-indice
_SIDRA_TABLE = "1737"
_SIDRA_VARIABLE = "2266"


# ── Funcoes internas ─────────────────────────────────────────────────────────

def _fetch_ipca_index() -> pd.Series:
    """Coleta o numero-indice do IPCA via SIDRA (tabela 1737, variavel 2266).

    Returns
    -------
    pd.Series
        Numero-indice do IPCA (base: dez/1993 = 100), indexado por datetime
        com datas no primeiro dia de cada mes.
    """
    import sidrapy

    raw = sidrapy.get_table(
        table_code=_SIDRA_TABLE,
        territorial_level="1",
        ibge_territorial_code="all",
        period="all",
        variable=_SIDRA_VARIABLE,
    )

    # Primeira linha (index 0) eh o cabecalho descritivo do SIDRA; descartar
    # D2C contem o codigo do periodo (YYYYMM), V contem o valor
    df = raw[["D2C", "V"]].iloc[1:].reset_index(drop=True)
    df.columns = ["date_code", "value"]

    # date_code vem no formato YYYYMM (ex: "199401")
    df["date"] = pd.to_datetime(df["date_code"], format="%Y%m")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    ipca_index = df.set_index("date")["value"].dropna()
    ipca_index.name = "ipca_index"

    logger.info(
        f"IPCA numero-indice coletado: {len(ipca_index)} periodos "
        f"({ipca_index.index.min().date()} a {ipca_index.index.max().date()})"
    )

    return ipca_index


def _rebase_index(
    ipca_index: pd.Series,
    base_date: str = "2022-12-01",
) -> pd.Series:
    """Renormaliza o numero-indice do IPCA para que ``base_date == 100``.

    Parameters
    ----------
    ipca_index : pd.Series
        Numero-indice original (base: dez/1993 = 100).
    base_date : str
        Data-base desejada para a normalizacao.

    Returns
    -------
    pd.Series
        Indice renormalizado (``base_date == 100``).
    """
    base_ts = pd.Timestamp(base_date)

    if base_ts not in ipca_index.index:
        idx = ipca_index.index.get_indexer([base_ts], method="nearest")[0]
        base_ts = ipca_index.index[idx]
        logger.info(
            f"Data-base {base_date} nao encontrada no indice. "
            f"Usando data mais proxima: {base_ts.date()}"
        )

    base_value = ipca_index.loc[base_ts]
    return (ipca_index / base_value) * 100


# ── Funcao principal ─────────────────────────────────────────────────────────

def deflate(
    base_df: pd.DataFrame,
    specs_df: pd.DataFrame,
    *,
    ipca_index: pd.Series | None = None,
    base_date: str = "2022-12-01",
) -> pd.DataFrame:
    """Deflaciona series marcadas com ``deflate == 0`` em ``specs_df``.

    Segue a mesma convencao de ``seas_adj``: filtra as variaveis que
    **precisam** ser deflacionadas (``deflate == 0``), aplica a divisao
    pelo indice de precos e retorna o DataFrame completo.

    Parameters
    ----------
    base_df : pd.DataFrame
        DataFrame com as series temporais (indice datetime).
    specs_df : pd.DataFrame
        Especificacoes das series; deve conter colunas ``'variable'`` e
        ``'deflate'``.
    ipca_index : pd.Series | None
        Numero-indice do IPCA. Se ``None``, sera coletado automaticamente
        via SIDRA (tabela 1737, variavel 2266).
    base_date : str
        Data-base para o indice de precos (default ``'2022-12-01'``).
        Todas as series serao expressas em precos constantes dessa data.

    Returns
    -------
    pd.DataFrame
        Copia de ``base_df`` com as series nominais convertidas para
        valores reais (precos constantes de ``base_date``).
    """
    # Identifica variaveis que precisam de deflacionamento
    nominal_vars = specs_df.query("deflate == 0")["variable"].to_list()

    if not nominal_vars:
        logger.info("Nenhuma variavel marcada para deflacionamento (deflate == 0).")
        return base_df.copy(deep=True)

    # Obtem ou constroi o indice de precos
    if ipca_index is None:
        ipca_index = _fetch_ipca_index()

    price_index = _rebase_index(ipca_index, base_date=base_date)

    base_df_deflated = base_df.copy(deep=True)

    for col in nominal_vars:
        if col not in base_df_deflated.columns:
            logger.warning(f"Coluna '{col}' nao encontrada no DataFrame, pulando deflacionamento.")
            continue

        series = base_df_deflated[col]

        # Alinha o indice de precos ao indice da serie
        aligned_index = price_index.reindex(series.index)

        if aligned_index.isna().all():
            logger.warning(
                f"Coluna '{col}': indice de precos nao possui datas compativeis. "
                f"Pulando deflacionamento."
            )
            continue

        n_missing = aligned_index.isna().sum()
        if n_missing > 0:
            # Identifica quais periodos ficaram sem deflator
            missing_dates = series.index[aligned_index.isna()]
            # Filtra apenas periodos que tem dado real (nao NaN) na serie original
            missing_with_data = missing_dates[series.loc[missing_dates].notna()]
            if len(missing_with_data) > 0:
                logger.warning(
                    f"Coluna '{col}': {len(missing_with_data)} periodo(s) com dados mas sem "
                    f"deflator disponivel ({missing_with_data[-1].date()}). "
                    f"Esses valores permanecerao em NaN."
                )
                # Valores sem deflator recebem NaN para nao misturar nominal com real
                base_df_deflated.loc[missing_with_data, col] = np.nan

        # Deflaciona: valor_real = valor_nominal / (indice / 100)
        deflated = series / (aligned_index / 100)
        # Apenas sobrescreve onde ha deflator valido
        valid_mask = aligned_index.notna()
        base_df_deflated.loc[valid_mask, col] = deflated.loc[valid_mask]

    return base_df_deflated


# ── Bloco para testes manuais ────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from pib_nowcast.config import SERIES_SPEC, LAST_DATA

    specs_df = pd.read_csv(SERIES_SPEC, sep=";")
    old_full_data = pd.read_excel(
        LAST_DATA, sheet_name="full_dataset", index_col="Date"
    )

    result = deflate(old_full_data, specs_df)

    nominal_vars = specs_df.query("deflate == 0")["variable"].to_list()
    for col in nominal_vars[:3]:
        if col in result.columns:
            print(f"\n--- {col} ---")
            print(f"  Original (tail): {old_full_data[col].dropna().tail(3).values}")
            print(f"  Deflated (tail): {result[col].dropna().tail(3).values}")
