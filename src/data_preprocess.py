import pandas as pd
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
