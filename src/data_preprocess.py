import pandas as pd
import datetime as dt
from typing import Tuple


class DataProcessor:

    def __init__(self, df):
        self.df = df
        self.cutoff_period = pd.to_datetime("2022-06")

    def features_shift(
        self,
        group_by: list,
        columns_to_shift: dict
    ) -> pd.DataFrame:

        for shift, columns in columns_to_shift.items():

            self.df[columns] = self.df.groupby(group_by)[columns].transform(
                lambda x: x.shift(shift)
            )

        return self.df


def transform_data_type(df, columns_dtype, strftime=False):

    for dtype, columns in columns_dtype.items():
        if dtype == "datetime":
            for column in columns:
                df[column] = pd.to_datetime(df[column])

                if strftime:
                    df[column] = df[column].dt.strftime("%Y-%m")

        else:
            for column in columns:
                df[column] = df[column].astype(dtype)
    
    return df