import pandas as pd
from typing import Tuple

from .params import FEATURES


def get_data(df, cutoff_period, step) -> Tuple:

        if step == "train":
            df_step = df[
                df.dt_calendar < cutoff_period
            ]

        elif step == "validation":
            df_step = df[
                (df.dt_calendar >= cutoff_period)
                & (df.dt_calendar < "2022-12")
                # & (df.dt_calendar != (dt.datetime.now()).strftime("%Y-%m"))
            ]

        elif step == "test":
            df_step = df[
                (df.dt_calendar == cutoff_period)
                
            ]
        
        return df_step[FEATURES], df_step["target"]


def flatten(l, result):
    
    if isinstance(l, list):
        for i in l:
            flatten(i, result)
    else:
        if l in result:
            pass
        else:
            result.append(l)
    return result


def get_segments(y_pred):
    
    result=[]
    metrics = {
        'predictions': y_pred,
        'deciles': pd.qcut(y_pred, 10, labels=False),
        'target': y_pred 
    }

    segmentation_df = pd.DataFrame(metrics)

    decile_rates = segmentation_df.groupby('deciles')['target'].mean()
    bins = [0, [bin for bin in decile_rates.get([1, 3, 5, 7])], 1]
    
    label_segment = lambda x: pd.cut(
            x, 
            bins=flatten(bins, result), 
            labels=['A', 'B', 'C', 'D', 'E']
        )

    segmentation_df["segments"] = label_segment(y_pred)
    segment_rates = segmentation_df.groupby('segments')['target'].agg(['mean', 'size'])

    return segmentation_df, segment_rates
