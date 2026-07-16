import pandas as pd
from tenacity import wait_fixed, retry, stop_after_attempt

@retry(stop=stop_after_attempt(5))
def get_ipeadata(
    series: str| dict| pd.DataFrame | None = None, 
    start: str|None = None, 
    **kwargs
    ):
    import pandas as pd
    import ipeadatapy as ipea
    
    if (isinstance(series, str)) and ('.csv' in series):

        series = (
                    pd.read_csv(series, sep=';')
                    .query("source == 'ipeadata'")
                )
        series = series.set_index('variable')['code'].to_dict()
    
    elif isinstance(series, pd.DataFrame):

        series = (
                    series
                    .query("source == 'ipeadata'")
                )

        series = series.set_index('variable')['code'].to_dict()

    if series == {}:
        print("Não foi localizada nenhuma série do IPEA Data no argumento `series`")
        return pd.DataFrame()

    ipea_dfs_list = []
    for name, code in series.items():
        temp = ipea.timeseries(series=code)
        temp = temp.iloc[:, [-1]]

        temp.columns = [name]

        ipea_dfs_list.append(temp)

    ipea_series = pd.concat(ipea_dfs_list, axis=1, join='outer')

    if (ipea_series is not None) and (start is not None) and (isinstance(start, str)):
        ipea_series = ipea_series.loc[start:, :]

    return ipea_series

@retry(stop=stop_after_attempt(5))
def get_ipeadata_parallel(
    series: str| dict| pd.DataFrame | None = None, 
    start: str|None = None, 
    **kwargs
    ):
    import pandas as pd
    import ipeadatapy as ipea
    import concurrent.futures
    
    if (isinstance(series, str)) and ('.csv' in series):
        series = (
                    pd.read_csv(series, sep=';')
                    .query("source == 'ipeadata'")
                )
        series = series.set_index('variable')['code'].to_dict()
    elif isinstance(series, pd.DataFrame):
        series = (
                    series
                    .query("source == 'ipeadata'")
                )
        series = series.set_index('variable')['code'].to_dict()

    if series == {}:
        print("Não foi localizada nenhuma série do IPEA Data no argumento `series`")
        return pd.DataFrame()

    def fetch_single_ipea(name, code):
        temp = ipea.timeseries(series=code)
        temp = temp.iloc[:, [-1]]
        temp.columns = [name]
        return temp

    ipea_dfs_list = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_name = {
            executor.submit(fetch_single_ipea, name, code): name 
            for name, code in series.items()
        }
        for future in concurrent.futures.as_completed(future_to_name):
            try:
                result = future.result()
                ipea_dfs_list.append(result)
            except Exception as e:
                name = future_to_name[future]
                print(f"Erro ao baixar a série {name} do IPEA: {e}")

    ipea_series = pd.concat(ipea_dfs_list, axis=1, join='outer') if ipea_dfs_list else pd.DataFrame()

    if (not ipea_series.empty) and (start is not None) and (isinstance(start, str)):
        ipea_series = ipea_series.loc[start:, :]

    return ipea_series

