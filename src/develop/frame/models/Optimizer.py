import json
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..utils import get_data
from ...hyperopt import OptunaOptimize


class Optimizer():

    def __init__(self):
        self.df = get_data("src/develop/frame/data/df_model.csv")
        self.preprocess_data()

        self.best_result = None
        self.best_params = None

    def preprocess_data(self):

        self.df["dt_calendar"] = pd.to_datetime(self.df["dt_calendar"])
        self.df = self.df[(self.df.dt_calendar >= "2020-01")]

        print(self.df.shape)

    def find_best_params(self, **params):

        hyperopt = OptunaOptimize(
            self.df,
            eval_metric=roc_auc_score,
            **params
        )
        hyperopt.run()

        self.best_params = hyperopt.best_params
        self.best_result = hyperopt.best_result
        
        print(f"BEST AUC: {hyperopt.best_result}")

    def export_params(self):
        path = "src/develop/frame/data/params_et.json"

        with open(path, "w") as f:
            json.dump(self.best_params, f)

        return print("\n<<< Parameters optimized >>>")
