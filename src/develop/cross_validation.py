import numpy as np
import pandas as pd
import datetime as dt

from src.utils import get_data


class CrossValidation:

    def __init__(self, df, model, eval_metric):
        self.df = df
        self.model = model
        self.eval_metric = eval_metric

        self.cutoff_date = None

    def fit(self):

        results = []
        for cutoff_date in pd.date_range("2022-01", "2022-12", freq="MS"):

            self.cutoff_date = cutoff_date

            self.train_model()
            auc = self.test_model()
            results.append(auc)
            
            print(f"TRAIN ({(cutoff_date - pd.DateOffset(months=1)).strftime('%Y-%m')}) | VALIDATION ({cutoff_date.strftime('%Y-%m')}): (AUC={round(auc, 4)})")

        return np.mean(results)

    def train_model(self, **kwargs):
        
        X_train, y_train = get_data(self.df, self.cutoff_date, "train")
        self.model.fit(X_train, y_train)

        return self.model

    def test_model(self, **kwargs):

        X, y = get_data(self.df, self.cutoff_date, "test")

        y_pred = self.model.predict_proba(X)[:,1]
        auc = self.eval_metric(y, y_pred)

        return auc
