import numpy as np
import pandas as pd

from .utils import get_data


class ClassificationModel:

    def __init__(self, df, model, eval_metric):
        self.df = df
        self.model = model
        self.eval_metric = eval_metric

        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None
        self.result = None

    def fit(self, cutoff_period):
        
        X_train, y_train = get_data(self.df, cutoff_period, step="train")
        self.model.fit(X_train, y_train)

        self.X_train, self.y_train = X_train, y_train

        return self

    def test_model(self, cutoff_period):

        X_test, y_test = get_data(self.df, cutoff_period, step="test")
        self.X_test, self.y_test = X_test, y_test

        y_pred = self.model.predict_proba(X_test)[:,1]
        auc = self.eval_metric(y_test, y_pred)

        self.result = round(auc, 4)

        print(f"TESTING ON ({cutoff_period})")

        return self.result


class TwoStepCalibration:
    
    """
    Class to use a two step model consisting of 
    a general purpose model and a logistic regression for calibration
    """

    def __init__(self, first_step_model, lr_model):
        
        self.first_step_model = first_step_model
        self.lr_model = lr_model 
        self.is_fitted = False
        self._estimator_type = 'classifier'
        
    def fit(self, X, y, sample_weight=None):
        
        self.classes_ = np.unique(y)
        
        self.first_step_model.fit(X, y)
        
        first_step_preds = self.first_step_model.predict_proba(X)[:,1]
        X_first_step = first_step_preds.reshape(-1, 1)
        
        self.lr_model.fit(X_first_step, y)

        # setting fitted variable
        self.is_fitted = True
        
    # method for predicting probabilities
    def predict_proba(self, X):
        
        first_step_preds = self.first_step_model.predict_proba(X)[:,1]
        X_first_step = first_step_preds.reshape(-1, 1)
        
        y_hat = self.lr_model.predict_proba(X_first_step)
        
        # retuning probabilities
        return y_hat
