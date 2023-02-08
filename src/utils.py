import pandas as pd
from typing import Tuple

from .params import FEATURES


def get_step_data(df, cutoff_period, step) -> Tuple:

        if step == "train":
            df_step = df[df.dt_calendar < cutoff_period]

        elif step == "validation":
            df_step = df[df.dt_calendar == cutoff_period]
        
        elif step == "test":
            df_step = df[df.dt_calendar == cutoff_period + pd.DateOffset(months=1)]
        
        return df_step[FEATURES], df_step["target"]