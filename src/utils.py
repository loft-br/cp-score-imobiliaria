import datetime as dt
from typing import Tuple

from .params import FEATURES


def get_data(df, cutoff_period, step) -> Tuple:

        if step == "train":
            df_step = df[
                df.dt_calendar < cutoff_period
            ]

        elif step == "test":
            df_step = df[
                (df.dt_calendar == cutoff_period) # >= cutoff_period)
                # & (df.dt_calendar != (dt.datetime.now()).strftime("%Y-%m"))
            ]

            print(df_step.dt_calendar.unique())
        
        return df_step[FEATURES], df_step["target"]