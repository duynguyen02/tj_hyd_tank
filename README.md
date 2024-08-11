# TJ_HYD_NAM

Python implementation of Tank Hydrological model by Sugawara and Funiyuki (1956) , based on the original code
from [tank-model](https://github.com/nzahasan/tank-model) by [hckaraman](https://github.com/nzahasan)

### Installation

```
pip install tj_hyd_tank
```

### Getting Started

#### Prepare the Dataset
##### Dataset

The dataset should include columns: Date, Precipitation, Evapotranspiration, and Discharge, with column
names customizable.

| Date       | Q       | P   | E    |
|------------|---------|-----|------|
| 10/9/2016  | 0.25694 | 0   | 2.79 |
| 10/10/2016 | 0.25812 | 0   | 3.46 |
| 10/11/2016 | 0.30983 | 0   | 3.65 |
| 10/12/2016 | 0.31422 | 0   | 3.46 |
| 10/13/2016 | 0.30866 | 0   | 5.64 |
| 10/14/2016 | 0.30868 | 0   | 3.24 |
| 10/15/2016 | 0.31299 | 0   | 3.41 |
| ...        | ...     | ... | ...  |

The time intervals between dates must be equal (e.g., 24 hours) for the model to function accurately.

##### Basin file
HEC-HMS basin.

#### Quick start
```python
import pandas as pd

from tj_hyd_tank import TJHydTANK, TANKColNames, TANKConfig

df = pd.read_csv('data_example.csv')
tank_cols_name = TANKColNames(
    date='Date',
    precipitation='P',
    evapotranspiration='E',
    discharge='Q'
)
tank_config = TANKConfig(
    start_date=None,
    end_date=None,
    interval=24.0
)

tank = TJHydTANK(
    basin_file='CedarCreek.basin',
    df=df,
    tank_col_names=tank_cols_name,
    tank_config=tank_config
)
tank
```
#### Get basin_defs
```python
from tj_hyd_tank import Subbasin, Reach

for basin_def in tank.basin_defs:
    print(basin_def.name, basin_def.type)
    print(basin_def.stats)
    if isinstance(basin_def, (Subbasin, Reach)):
        print(basin_def.params)
```
#### Get root_node
```python
from tj_hyd_tank import Subbasin, Reach

for root_node in tank.root_node:
    print(root_node.name, root_node.type)
    print(root_node.stats)
    if isinstance(root_node, (Subbasin, Reach)):
        print(root_node.params)
```
#### Plot a comparison between Q_obs and Q_sim of a basin_def
```python
outlet1 = tank.get_basin_def_by_name('Outlet1')
if outlet1 is not None:
    tank.show_discharge(outlet1)
```
#### Reconfig and show Subbasin 's properties
```python
from tj_hyd_tank import SubbasinParams

w170 = tank.get_basin_def_by_name('W170')
if w170 is not None:
    if isinstance(w170, Subbasin):
        tank.reconfig_subbasin_params(
            w170,
            SubbasinParams(
                t0_is=0.02,
                t0_soc_uo=80.0
            )
        )
        print('Q_tank_0', w170.Q_tank_0.tolist())
        print('Q_tank_1', w170.Q_tank_1.tolist())
        print('Q_tank_2', w170.Q_tank_2.tolist())
        print('Q_tank_3', w170.Q_tank_3.tolist())
        print('bottom_outlet_flow_tank_0', w170.bottom_outlet_flow_tank_0.tolist())
        print('bottom_outlet_flow_tank_1', w170.bottom_outlet_flow_tank_1.tolist())
        print('bottom_outlet_flow_tank_2', w170.bottom_outlet_flow_tank_2.tolist())
```
#### Reconfig TANK model
```python
tank.reconfig_tank(
    TANKConfig(
        start_date=pd.to_datetime('09/10/2016', dayfirst=True, utc=True),
        end_date=pd.to_datetime('20/10/2016', dayfirst=True, utc=True)
    )
)
```
#### To DataFrame
```python
outlet1_df = tank.to_dataframe(outlet1)
outlet1_df
```
#### Get Logs
```python
print(tank.logs)
```