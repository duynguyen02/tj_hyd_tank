import dataclasses
import datetime
import os
from queue import Queue
from typing import Optional

import numpy as np
import pandas as pd

from .tj_hyd_tank_utils import build_basin_def_and_root_node
from .basin_def import BasinDef, Subbasin, BasinDefType, Reach, Junction, Sink
from .tank_exception import FileNotFoundException, MissingColumnsException, ColumnContainsEmptyDataException, \
    InvalidDatetimeException, InvalidDatetimeIntervalException, InvalidStartDateException, InvalidEndDateException, \
    InvalidDateRangeException


@dataclasses.dataclass
class TANKColNames:
    date: str = 'Date'
    precipitation: str = 'Precipitation'
    evapotranspiration: str = 'Evapotranspiration'
    discharge: str = 'Discharge'


@dataclasses.dataclass
class TANKConfig:
    start_date: datetime.datetime | None = None
    end_date: datetime.datetime | None = None
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

        # compute after setup basin
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
            print(
                f'WARNING-TANK-01 ({subbasin.name}): Invalid parameter upper outlet height is less than lower outlet '
                f'height (Tank 0)')

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
                    print(
                        f'WARNING-TANK-02 ({subbasin.name}): Total outlet flow exceeded tank storage for tank {i} at timestep {t}')

        unit_conv_coeff = (subbasin.area * 1000) / (self._tank_config.interval * 3600)
        discharge = unit_conv_coeff * side_outlet_flow.sum(axis=1)
        states = dict(
            tank_storage=tank_storage,
            side_outlet_flow=side_outlet_flow,
            bottom_outlet_flow=bottom_outlet_flow
        )

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
            print(f"WARNING-MUSKINGUM-01 ({reach.name}): violates k, x constraints")

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

        for basin_def in self._basin_defs:
            if basin_def.name in computation_result.columns:
                basin_def.Q_sim = computation_result[basin_def.name].to_numpy()

    def _run(self):
        start, end = self._validate_dataset()
        self._init_data(start, end)
        self._compute()
