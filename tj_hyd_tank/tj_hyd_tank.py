import dataclasses
import datetime
import os
import pickle
from queue import Queue
from typing import Optional, List, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import minimize

from .basin_def import BasinDef, Subbasin, Reach, Junction, Sink, SubbasinParams, ReachParams, NSE
from .tank_exception import FileNotFoundException, MissingColumnsException, ColumnContainsEmptyDataException, \
    InvalidDatetimeException, InvalidDatetimeIntervalException, InvalidStartDateException, InvalidEndDateException, \
    InvalidDateRangeException
from .tj_hyd_tank_utils import build_basin_def_and_root_node


@dataclasses.dataclass
class TANKColNames:
    date: str = 'Date'
    precipitation: str = 'Precipitation'
    evapotranspiration: str = 'Evapotranspiration'
    discharge: str = 'Discharge'


@dataclasses.dataclass
class TANKConfig:
    start_date: Union[datetime.datetime, None] = None
    end_date: Union[datetime.datetime, None] = None
    interval: float = 24.0


class TJHydTANK:
    def __init__(
            self,
            basin_file: str,
            df: pd.DataFrame,
            tank_col_names: TANKColNames = TANKColNames(),
            tank_config: TANKConfig = TANKConfig()

    ):

        if not os.path.exists(basin_file):
            raise FileNotFoundException(basin_file)

        basin_defs, root_node = build_basin_def_and_root_node(basin_file)

        self._basin_defs = basin_defs
        self._root_node = root_node

        self._df: pd.DataFrame = df.copy()
        self._tank_col_names: TANKColNames = tank_col_names
        self._tank_config: TANKConfig = tank_config

        self._date: Optional[np.ndarray] = None
        self._P: Optional[np.ndarray] = None
        self._E: Optional[np.ndarray] = None
        self._Q_obs: Optional[np.ndarray] = None

        self._logs: List[str] = []

        # compute after setup basin
        self._run()

    @property
    def tank_config(self):
        return self._tank_config

    @property
    def date(self):
        return self._date

    @property
    def P(self):
        return self._P

    @property
    def E(self):
        return self._E

    @property
    def Q_obs(self):
        return self._Q_obs

    @property
    def logs(self):
        return self._logs

    def reconfig_tank(self, tank_config: TANKConfig):
        self._tank_config = tank_config
        self._run()

    @property
    def basin_defs(self):
        return self._basin_defs

    @property
    def root_node(self):
        return self._root_node

    def reconfig_subbasin_params(
            self,
            subbasin: Subbasin,
            subbasin_params: SubbasinParams,
    ):
        subbasin.params = subbasin_params
        self._run()

    def reconfig_reach_params(
            self,
            reach: Reach,
            reach_params: ReachParams,
    ):
        reach.params = reach_params
        self._run()

    def _validate_dataset(self):
        df = self._df.copy()
        required_columns = [
            self._tank_col_names.date,
            self._tank_col_names.precipitation,
            self._tank_col_names.evapotranspiration,
            self._tank_col_names.discharge
        ]

        for col in required_columns:
            if col not in df.columns:
                raise MissingColumnsException(col)

        for column in df.columns:
            if df[column].isnull().any():
                raise ColumnContainsEmptyDataException(column)

        try:
            df[self._tank_col_names.date] = pd.to_datetime(
                df[self._tank_col_names.date],
                utc=True
            )
        except Exception as _:
            str(_)
            raise InvalidDatetimeException()

        df['Interval'] = df[self._tank_col_names.date].diff()
        interval_hours = pd.Timedelta(hours=self._tank_config.interval)
        is_valid_interval_hours = df['Interval'].dropna().eq(interval_hours).all()
        if not is_valid_interval_hours:
            raise InvalidDatetimeIntervalException()
        df = df.drop(columns=['Interval'])

        start = 0
        end = df.size

        if self._tank_config.start_date is not None:
            valid = False
            for i in range(len(df.index)):
                if str(df[self._tank_col_names.date][i]) == str(self._tank_config.start_date):
                    valid = True
                    start = i
                    break

            if not valid:
                raise InvalidStartDateException()

        if self._tank_config.end_date is not None:
            valid = False
            for i in range(len(df.index)):
                if str(df[self._tank_col_names.date][i]) == str(self._tank_config.end_date):
                    valid = True
                    end = i + 1
                    break

            if not valid:
                raise InvalidEndDateException()

        if self._tank_config.start_date is not None and self._tank_config.end_date is not None:
            if self._tank_config.start_date > self._tank_config.end_date:
                raise InvalidDateRangeException()

        df = df[required_columns]
        self._df = df

        if end - start <= 3:
            raise InvalidDateRangeException("Minimum 4 rows of data required")

        return start, end

    def _build_computation_stack(self):
        computation_stack = []
        node_queue: Queue[BasinDef] = Queue()
        for root_node in self._root_node:
            node_queue.put(root_node)

        while not node_queue.empty():
            node = node_queue.get()
            computation_stack.append(node)

            if node.upstream:
                for child_node in node.upstream:
                    node_queue.put(child_node)

        return computation_stack

    def _init_data(self, start: int, end: int):
        proc_df: pd.DataFrame = self._df.copy()
        proc_df = proc_df.iloc[start: end]

        self._date = proc_df[self._tank_col_names.date].to_numpy()
        self._P = proc_df[self._tank_col_names.precipitation].to_numpy()
        self._E = proc_df[self._tank_col_names.evapotranspiration].to_numpy()
        self._Q_obs = proc_df[self._tank_col_names.discharge].to_numpy()

    def _tank_discharge(self, subbasin: Subbasin):
        params = subbasin.params
        if params.t0_soh_uo < params.t0_soh_lo:
            warnings_msg = f'WARNING-TANK-01 ({subbasin.name}): Invalid parameter upper outlet height is less than lower outlet height (Tank 0)'
            self._logs.append(warnings_msg)
            print(warnings_msg)

        time_step = self._P.shape[0]
        tank_storage = np.zeros((time_step, 4), dtype=np.float64)
        side_outlet_flow = np.zeros((time_step, 4), dtype=np.float64)
        bottom_outlet_flow = np.zeros((time_step, 3), dtype=np.float64)

        del_rf_et = self._P - self._E

        tank_storage[0, 0] = max(params.t0_is, 0)
        tank_storage[0, 1] = max(params.t1_is, 0)
        tank_storage[0, 2] = max(params.t2_is, 0)
        tank_storage[0, 3] = max(params.t3_is, 0)

        for t in np.arange(time_step):
            # TANK 0 : surface runoff
            side_outlet_flow[t, 0] = params.t0_soc_lo * max(tank_storage[t, 0] - params.t0_soh_lo, 0) \
                                     + params.t0_soc_uo * max(tank_storage[t, 0] - params.t0_soh_uo, 0)

            # TANK 1 : intermediate runoff
            side_outlet_flow[t, 1] = params.t1_soc * max(tank_storage[t, 1] - params.t1_soh, 0)
            # TANK 2 : sub-base runoff
            side_outlet_flow[t, 2] = params.t2_soc * max(tank_storage[t, 2] - params.t2_soh, 0)
            # TANK 3 : base-flow | Side outlet height = 0
            side_outlet_flow[t, 3] = params.t3_soc * tank_storage[t, 3]

            bottom_outlet_flow[t, 0] = params.t0_boc * tank_storage[t, 0]
            bottom_outlet_flow[t, 1] = params.t1_boc * tank_storage[t, 1]
            bottom_outlet_flow[t, 2] = params.t2_boc * tank_storage[t, 2]

            if t < (time_step - 1):
                tank_storage[t + 1, 0] = tank_storage[t, 0] + del_rf_et[t + 1] - (
                        side_outlet_flow[t, 0] + bottom_outlet_flow[t, 0])

                tank_storage[t + 1, 1] = tank_storage[t, 1] + bottom_outlet_flow[t, 0] - (
                        side_outlet_flow[t, 1] + bottom_outlet_flow[t, 1])

                tank_storage[t + 1, 2] = tank_storage[t, 2] + bottom_outlet_flow[t, 1] - (
                        side_outlet_flow[t, 2] + bottom_outlet_flow[t, 2])

                tank_storage[t + 1, 3] = tank_storage[t, 3] + bottom_outlet_flow[t, 2] - side_outlet_flow[t, 3]

                tank_storage[t + 1, 0] = max(tank_storage[t + 1, 0], 0)
                tank_storage[t + 1, 1] = max(tank_storage[t + 1, 1], 0)
                tank_storage[t + 1, 2] = max(tank_storage[t + 1, 2], 0)
                tank_storage[t + 1, 3] = max(tank_storage[t + 1, 3], 0)

            for i in range(4):
                total_tank_outflow = bottom_outlet_flow[t, i] + side_outlet_flow[t, i] if i <= 2 else side_outlet_flow[
                    t, i]

                if total_tank_outflow > tank_storage[t, i]:
                    warnings_msg = f'WARNING-TANK-02 ({subbasin.name}): Total outlet flow exceeded tank storage for tank {i} at timestep {t}'
                    self._logs.append(warnings_msg)
                    print(warnings_msg)

        unit_conv_coeff = (subbasin.area * 1000) / (self._tank_config.interval * 3600)
        discharge = unit_conv_coeff * side_outlet_flow.sum(axis=1)
        states = dict(
            tank_storage=tank_storage,
            side_outlet_flow=side_outlet_flow,
            bottom_outlet_flow=bottom_outlet_flow
        )

        # Set result
        subbasin.Q_tank_0 = tank_storage[0]
        subbasin.Q_tank_1 = tank_storage[1]
        subbasin.Q_tank_2 = tank_storage[2]
        subbasin.Q_tank_3 = tank_storage[3]

        subbasin.side_outlet_flow_tank_0 = side_outlet_flow[0]
        subbasin.side_outlet_flow_tank_1 = side_outlet_flow[1]
        subbasin.side_outlet_flow_tank_2 = side_outlet_flow[2]
        subbasin.side_outlet_flow_tank_3 = side_outlet_flow[3]

        subbasin.bottom_outlet_flow_tank_0 = bottom_outlet_flow[0]
        subbasin.bottom_outlet_flow_tank_1 = bottom_outlet_flow[1]
        subbasin.bottom_outlet_flow_tank_2 = bottom_outlet_flow[2]

        return discharge, states

    def _muskingum(
            self,
            inflow: np.ndarray,
            reach: Reach
    ):
        params = reach.params
        n_step: int = inflow.shape[0]
        outflow: np.ndarray = np.zeros(n_step, dtype=np.float64)

        c0: float = (-params.k * params.x + 0.5 * self._tank_config.interval) / (
                params.k * (1 - params.x) + 0.5 * self._tank_config.interval)
        c1: float = (params.k * params.x + 0.5 * self._tank_config.interval) / (
                params.k * (1 - params.x) + 0.5 * self._tank_config.interval)
        c2: float = (params.k * (1 - params.x) - 0.5 * self._tank_config.interval) / (
                params.k * (1 - params.x) + 0.5 * self._tank_config.interval)

        if (c0 + c1 + c2) > 1 or params.x > 0.5 or (self._tank_config.interval / params.k + params.x) > 1:
            warning_msg = f"WARNING-MUSKINGUM-01 ({reach.name}): violates k, x constraints"
            self._logs.append(warning_msg)
            print(warning_msg)

        outflow[0] = inflow[0]

        for t in np.arange(1, n_step):
            outflow[t] = c0 * inflow[t] + c1 * inflow[t - 1] + c2 * outflow[t - 1]

        return outflow

    def _compute(self):
        computation_stack = self._build_computation_stack()

        n_step = len(self._P)
        computation_result = pd.DataFrame()
        model_states = dict()

        while len(computation_stack) > 0:
            current_node = computation_stack.pop()

            if isinstance(current_node, Subbasin):
                computation_result[current_node.name], basin_states = self._tank_discharge(
                    current_node
                )
                model_states[current_node.name] = basin_states
            elif isinstance(current_node, Reach):
                sum_node = np.zeros(n_step, dtype=np.float64)
                for us_node in current_node.upstream:
                    sum_node += self._muskingum(
                        inflow=computation_result[us_node.name].to_numpy(),
                        reach=current_node
                    )
                computation_result[current_node.name] = sum_node
            elif isinstance(current_node, Sink) or isinstance(current_node, Junction):
                sum_node = np.zeros(n_step, dtype=np.float64)

                for us_node in current_node.upstream:
                    sum_node += computation_result[us_node.name].to_numpy()

                computation_result[current_node.name] = sum_node

        # set Q_sim for all BasinDef
        for basin_def in self._basin_defs:
            if basin_def.name in computation_result.columns:
                basin_def.Q_sim = computation_result[basin_def.name].to_numpy()

                # _compute_statistics
                basin_def.calculate_stats(self._Q_obs)

        return computation_result

    def _run(self):
        self._logs = []
        start, end = self._validate_dataset()
        self._init_data(start, end)
        _ = self._compute()

    def _params_stack(self):
        basin_def_order = []
        stacked_parameter = []

        for basin_def in self._basin_defs:
            if isinstance(basin_def, Subbasin) or isinstance(basin_def, Reach):
                if isinstance(basin_def, Subbasin):
                    stacked_parameter.extend(
                        basin_def.params.to_list()
                    )
                elif isinstance(basin_def, Reach):
                    stacked_parameter.extend(
                        basin_def.params.to_list()
                    )
                basin_def_order.append(basin_def)
        return basin_def_order, stacked_parameter

    @staticmethod
    def _update_basin_with_stack_params(basin_def_order: List[Union[Subbasin, Reach]], stacked_parameter: List[float]):
        subbasin_steps = len(SubbasinParams().to_list())
        reach_steps = len(ReachParams().to_list())
        _from = 0
        for basin_def in basin_def_order:
            if isinstance(basin_def, Subbasin):
                steps = _from + subbasin_steps
                basin_def.params = SubbasinParams(*stacked_parameter[_from: steps])
                _from = steps
            if isinstance(basin_def, Reach):
                steps = _from + reach_steps
                basin_def.params = ReachParams(*stacked_parameter[_from: steps])
                _from = steps

    def _stat_by_stacked_parameter(
            self, stacked_parameter: List[float], basin_def_order: List[Union[Subbasin, Reach]],
    ):
        self._update_basin_with_stack_params(
            basin_def_order,
            stacked_parameter
        )

        self._run()

        root_node = self._root_node[0]
        _nse = NSE(
            root_node.Q_sim,
            self._Q_obs
        )
        return 1 - _nse

    def optimize(self, eps: float = 0.01):
        basin_def_order, stacked_parameter = self._params_stack()

        upper_bound_stacked = list()
        lower_bound_stacked = list()

        for basin_def in basin_def_order:
            if isinstance(basin_def, Subbasin):
                upper_bound_stacked.extend(SubbasinParams.MaxBoundParams().to_list())
                lower_bound_stacked.extend(SubbasinParams().to_list())
            if isinstance(basin_def, Reach):
                upper_bound_stacked.extend(ReachParams.MaxBoundParams().to_list())
                lower_bound_stacked.extend(ReachParams().to_list())

        initial_guess = np.array(stacked_parameter)
        param_bounds = np.column_stack((lower_bound_stacked, upper_bound_stacked))

        optimizer = minimize(
            fun=self._stat_by_stacked_parameter,
            x0=initial_guess,
            args=basin_def_order,
            method='L-BFGS-B',
            bounds=param_bounds,
            options={
                'eps': eps
            }
        )
        self._update_basin_with_stack_params(
            basin_def_order,
            optimizer.x,
        )

    def get_basin_def_by_name(self, name: str) -> Union[
        Subbasin, Junction, Sink, Reach, None
    ]:
        for basin_def in self._basin_defs:
            if basin_def.name == name:
                return basin_def
        return None

    def show_discharge(self, basin_def: BasinDef):
        plt.figure()
        # for basin_def in self._basin_defs:
        #     plt.plot(self._date, basin_def.Q_sim, label=basin_def.name)
        plt.plot(self._date, self._Q_obs, label='Obs')
        plt.plot(self._date, basin_def.Q_sim, label=basin_def.name)
        plt.title('Observer vs Simulated Discharge')
        plt.xlabel('Date')
        plt.ylabel('Q')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def to_dataframe(self, basin_def: BasinDef):
        return pd.DataFrame({
            'Q_obs': self._Q_obs,
            'Q_sim': basin_def.Q_sim,
        })

    def save(self, filename: str = None):
        with open(f'{filename}.tjtank' if filename is not None else 'tank_model.tjtank', 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, model_path: str) -> 'TJHydTANK':
        with open(model_path, 'rb') as file:
            return pickle.load(file)

    def __str__(self):
        return f"""
TJ_HYD_TANK 🍃 🌧 ☔ 💦
FROM: {self._date[0]}
TO: {self._date[-1]}
basin_def: {" ".join([basin_def.name for basin_def in self._basin_defs])}
root_node: {" ".join([basin_def.name for basin_def in self._root_node])}
"""

    def __repr__(self):
        return self.__str__()
