from .seas_adj import seas_adj, seas_adj_parallel, seas_adj_stl, seas_adj_stl_parallel
from .stationarity import stationarity_tests, is_stationary
from .transform_pipeline import (
    PIPELINE_REGISTRY,
    PIPELINE_NAME_TO_ID,
    MONTHLY_PIPELINE_IDS,
    QUARTERLY_PIPELINE_IDS,
    apply_transform_pipeline,
    make_stationary,
)
from .deflate import deflate
from .outliers import remove_outliers

def preprocess_data(df, specs_df, skip_seas_adj, fit_start_date, outlier_threshold):
    """Executa todos os passos de pré-processamento de uma vez."""
    df_clean = deflate(df, specs_df)
    if not skip_seas_adj:
        df_clean = seas_adj_stl_parallel(df_clean, specs_df)
        
    df_clean = make_stationary(df_clean, specs_df).loc[fit_start_date:, :]
    return remove_outliers(df_clean, threshold=outlier_threshold)
