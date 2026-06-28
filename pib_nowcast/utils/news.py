import os
import pandas as pd
import datetime as dt

def get_news_impacts(news, save_to: str | None = None) -> None:
    """Extrai os impactos (news) de uma dada atualização de modelo e os salva historicamente."""
    today = dt.date.today()

    # 1. get_impacts() retorna um DataFrame MultiIndexado com a decomposição do modelo.
    # 2. reset_index() remove os índices para transformá-los em colunas comuns.
    # 3. droplevel(1, axis=1) limpa a hierarquia de colunas indesejada do statsmodels.
    # 4. melt() pivota a tabela, transformando as variáveis (que estavam em colunas) em linhas.
    # 5. assign(update_date) fixa a data da rodada do nowcast atual.
    impacts_df = (
        news
        .get_impacts()
        .reset_index()
        .droplevel(level=1, axis=1)
        .melt(id_vars=['update date', 'updated variable'])
        .rename(columns={
                        'value': 'impact',
                        'update date': 'reference date'
                        })
        .assign(update_date=today)
    )

    if save_to:
        if os.path.exists(save_to):
            # Se já existir um histórico, carrega, anexa e sobrescreve
            history_df = pd.read_excel(save_to)
            final_df = pd.concat([history_df, impacts_df], ignore_index=True)
            final_df.to_excel(save_to, index=False)
        else:
            # Primeiro arquivo
            impacts_df.to_excel(save_to, index=False)
            
    return None


def get_new_forecasts(
    news, 
    new_model_res, 
    last_pib_date_timestamp, 
    next_pib_quarter_timestamp, 
    save_to: str | None = None
) -> None:
    """Extrai as projeções pontuais (mean) e intervalares (conf_int) e as salva historicamente."""
    today = dt.date.today()
    
    # Extrai do objeto news do statsmodels apenas a previsão média pontual ('estimate (new)')
    # Renomeia colunas para manter compatibilidade com o formato de arquivo de forecasts
    new_point_forecasts = (
        news
        .impacts
        [['estimate (new)']]
        .reset_index()
        .assign(
            update_date=today,
            estimate='predicted_mean',
            type = 'qoq'
            )
        .rename(columns={
            'impact date': 'reference date',
            'estimate (new)': 'forecast'
        })
    )

    # Gera intervalo de confiança iterando a função get_prediction no modelo recém atualizado
    # Filtra apenas a estimativa da variável alvo para a data do próximo PIB
    new_interval_forecasts = (
        new_model_res
        .get_prediction(
                        start=last_pib_date_timestamp,
                        end=next_pib_quarter_timestamp,
                        dynamic=False,
                        information_set='predicted'
                        ) 
        .conf_int()
        .loc[[next_pib_quarter_timestamp], ['lower pib', 'upper pib']]
        .reset_index()
        .rename(columns={
            'index': 'reference date',
            'lower pib': 'lower',
            'upper pib': 'upper',
        })
        .melt(id_vars='reference date', var_name='estimate', value_name='forecast')
        .assign(
            **{
                'impacted variable': 'pib',
                'update_date': today,
                'type': 'qoq'
            }
        )
    )

    # Unifica as estimativas pontuais e intervalares num mesmo formato tabular
    all_forecasts_df = pd.concat([new_point_forecasts, new_interval_forecasts], ignore_index=True)

    if save_to:
        if os.path.exists(save_to):
            # Histórico existe
            history_df = pd.read_excel(save_to)
            final_df = pd.concat([history_df, all_forecasts_df], ignore_index=True)
            final_df.to_excel(save_to, index=False)
        else:
            all_forecasts_df.to_excel(save_to, index=False)
            
    return None
