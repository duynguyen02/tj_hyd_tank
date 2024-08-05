import pandas as pd

from tj_hyd_tank import TJHydTANK, TANKColNames, Subbasin, Reach

tank = TJHydTANK(
    'BAHADURABAD.basin',
    pd.read_csv('data_example.csv'),
    TANKColNames(
        date='Date',
        precipitation='P',
        evapotranspiration='E',
        discharge='Q'
    )
)

print(tank)

