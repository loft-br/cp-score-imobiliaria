import pandas as pd

from .utils import get_step_data


class ClassificationModel:

    def __init__(self, df, model, eval_metric, cutoff_period):
        self.df = df
        self.model = model
        self.eval_metric = eval_metric
        self.cutoff_period = cutoff_period

    def train_model(self, **kwargs):

        X_train, y_train = get_step_data(self.df, self.cutoff_period, "train")
        self.model.fit(X_train, y_train)

        return self.model

    def validate_model(self, **kwargs):
        X_valid, y_valid = get_step_data(self.df, self.cutoff_period, "validation")

        y_pred = self.model.predict_proba(X_valid)[:,1]
        auc = self.eval_metric(y_valid, y_pred)

        return auc            