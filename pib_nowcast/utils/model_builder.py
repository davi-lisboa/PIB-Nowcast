from statsmodels.tsa.api import DynamicFactorMQ

def build_dfm(endog_data, k_endog_monthly, factors):
    """
    Constrói e retorna a instância do DynamicFactorMQ centralizada.
    Garante que todos os scripts (pipeline, backtest, testes) usem exatamente
    a mesma arquitetura e ordens AR do modelo.
    """
    return DynamicFactorMQ(
        endog=endog_data,
        k_endog_monthly=k_endog_monthly,
        factors=factors,
        factor_multiplicities={'Global': 2},
        factor_orders={
            'Global': 4,
            'Output': 1,
            'Employment': 1,
            'Prices': 1,
            'Credit': 1
        }
    )
