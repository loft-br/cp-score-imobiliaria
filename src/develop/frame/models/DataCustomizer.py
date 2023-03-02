import pandas as pd

from ..utils import get_data
from ....data_preprocess import DataProcessor, aggregate_data


class DataCustomizer:

    def __init__(self):
        self.df = get_data("src/develop/frame/data/df_imobs.csv")
        self.preprocess_data()

    def preprocess_data(self):

        self.df = (
            self.df
            .rename(columns={"dt_ativacao": "dt_calendar"})
            .sort_values(["id_imobiliaria", "dt_calendar"])
        )
        self.df["dt_calendar"] = pd.to_datetime(self.df["dt_calendar"])

    def create_target(self):

        self.df["target"] = (
            self.df["is_indemn_first_6months"]
            .transform(
                lambda x: 1 if x >= 1 else 0 
            )
        )

    def shift_features(self, **kwargs):

        processor = DataProcessor(self.df)

        self.df = processor.features_shift(
            group_by="id_imobiliaria",
            **kwargs
        )

    def transform_features(self, **kwargs):

        self.df["churn_rate"] = self.df["is_churn"] / self.df["is_activated"]
        
        for features, column_name in kwargs["aggregate_list"].items():
            self.df[column_name] = (
                aggregate_data(self.df, features, kwargs["window"])
            )

    def filter_new_imobs(self):
        self.df = (
            self.df.loc[self.df["is_active"] > 100, :]
            .reset_index(drop=True)
        )

    def export_data(self, column_to_filter):

        self.df = self.df[~self.df[column_to_filter].isnull()]
        print(self.df.shape)
        
        self.df.to_csv("src/develop/frame/data/df_model.csv", index=False)

        return print("\n<<< Model's data exported >>>")
