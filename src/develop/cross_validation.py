import numpy as np
import pandas as pd
# import datetime as dt

from src.utils import get_data


class CrossValidation:

    def __init__(self, df, model, eval_metric):
        self.df = df
        self.model = model
        self.eval_metric = eval_metric

        self.cohort = None

    def fit(self, from_, to_):

        results = []
        for cohort in pd.date_range(from_, to_, freq="MS"):

            self.cohort = cohort

            self.train_model()
            auc = self.validate_model()
            results.append(auc)
            
            # print(f"TRAINING UNTIL ({(cohort - pd.DateOffset(months=1)).strftime('%Y-%m')}) | VALIDATING ({cohort.strftime('%Y-%m')}): (AUC={round(auc, 4)})")
            print(f"TRAINING UNTIL ({(cohort - pd.DateOffset(months=1)).strftime('%Y-%m')}) | VALIDATING FROM ({cohort.strftime('%Y-%m')}) UNTIL ({to_}): (AUC={round(auc, 4)})")

        return round(np.mean(results), 4)

    def train_model(self, **kwargs):
        
        X_train, y_train = get_data(self.df, self.cohort, "train")
        self.model.fit(X_train, y_train)

        return self.model

    def validate_model(self, **kwargs):

        X, y = get_data(self.df, self.cohort, "validation")

        y_pred = self.model.predict_proba(X)[:,1]
        auc = self.eval_metric(y, y_pred)

        return round(auc, 4)
