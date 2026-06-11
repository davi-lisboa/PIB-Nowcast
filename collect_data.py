# %% get_bcb
from tenacity import wait_fixed, retry, stop_after_attempt

@retry(stop=stop_after_attempt(5))
def get_bcb(serie:dict, start:str|None = None):
    from bcb import sgs
    return sgs.get(serie, start)


series = {
                  # Output
                  'ind_extrativa':28504,
                  'ind_transformacao':28505,
                  'ind_bens_capital':28506,
                  'ind_bens_intermediarios':28507,
                  'ind_bens_consumo':28508,
                  'ind_construcao':28511,

                  'cons_energ_comercial':1402,
                  'cons_energ_indusrial':1404,

                  'pmc_ampliada': 28485,
                  'ibcbr_agro':29602,
                  'ibcbr_ind':29604,
                  'ibcbr_servicos':29606,

                  # Sentiment
                  'icc_fecomercio':4393,
                  'icea_fecomercio':4394,
                  'ief_fecomercio':4395,

                  'ics_fgv':20339,
                  'isas_fgv':20340,
                  'ies_fgv':20341,

                  # Employment
                  'caged_estoque':28784,
                  'renda_media':24399,
                  'tx_desemprego':24369,

                  # Prices
                  'ipca_12m':13522,
                  'ipca15': 7478,

                  'igpm':189,
                  'igpdi':190,
                  'incc':192,
                  'ipam':7450
              }


get_bcb(series)
# %% get_pib


def get_pib():
    
    import sidrapy
    # https://apisidra.ibge.gov.br/values/t/1620/n1/all/v/all/p/all/c11255/90707/d/v583%202


    return sidrapy.get_table(
                                table_code='1620', 
                                territorial_level='1', 
                                ibge_territorial_code='all', 
                                period='all'
                                variable='all', 
                                classification='11255/90707', 
                            )

get_pib()