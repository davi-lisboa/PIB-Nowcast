import numpy as np
import pandas as pd

def remove_outliers(df: pd.DataFrame,  threshold: float = 10) -> pd.DataFrame:
    """
    Remove outliers de séries temporais baseado no intervalo interquartil (IQR).
    Substitui por NaN observações que estão a mais de 10 vezes o IQR de distância da média.
    
    Essa função é tipicamente aplicada após as séries terem sido estacionarizadas, 
    para evitar que choques extremos (como a pandemia de COVID-19) distorçam as 
    estimativas de fatores dinâmicos.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contendo as séries temporais.
        
    Returns
    -------
    pd.DataFrame
        Cópia de `df` com os outliers substituídos por np.nan.
    """
    # Computa a média e o intervalo interquartil (IQR)
    mean = df.mean()
    iqr = df.quantile([0.25, 0.75]).diff().T.iloc[:, 1]
    
    # Substitui entradas que estão a mais de 10 vezes o IQR de distância da média por NaN
    mask = np.abs(df - mean) > threshold * iqr
    
    treated = df.copy()
    treated[mask] = np.nan
    
    return treated
