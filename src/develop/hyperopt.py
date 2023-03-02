import optuna
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV

from src.utils import get_data
from src.develop.cross_validation import CrossValidation


class OptunaOptimize:

    def __init__(self, df, from_, to_, eval_metric, trials):
        self.df = df
        self.to_ = to_
        self.from_ = from_
        self.trials = trials
        self.eval_metric = eval_metric

        self.best_params = None
        self.best_result = None

    def run(self):
        
        print("Optimizing...")

        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=self.trials)

        print("Done...")

        self.best_params = study.best_params
        self.best_result = study.best_value

    
    def train(self, params):

        results = []
        model = ExtraTreesClassifier(**params)

        for cohort in pd.date_range(self.from_, self.to_, freq="MS"):

            X_train, y_train = get_data(self.df, cohort, "train")
            model.fit(X_train, y_train)

            X, y = get_data(self.df, cohort, "test")

            y_pred = model.predict_proba(X)[:,1]
            auc = self.eval_metric(y, y_pred)

            results.append(auc)
            
            print(f"TRAINING UNTIL ({(cohort - pd.DateOffset(months=1)).strftime('%Y-%m')}) | VALIDATING ({cohort.strftime('%Y-%m')}): (AUC={round(auc, 4)})")
        
        auc_cv = round(np.mean(results), 4)
        print(auc_cv)

        return auc_cv
    
    def objective(self, trial):
        params = {
            "random_state": 5,
            "oob_score": True,    # use out-of-bag samples to estimate the generalization score
            "bootstrap": True,
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 5),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 4, 8),
            "max_features":  trial.suggest_float("max_features", 0.3, 0.7, step=0.1),
            "max_samples": trial.suggest_float("max_samples", 0.5, 0.8, step=0.1),  # number of samples to draw when bootstrap is True
        }

        # Supress Ooptuna warnings
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        return self.train(params)


class GridSearch:

    def __init__(self, df, estimator):
        self.df = df 
        self.estimator = estimator

        self.searcher = None
        self.grid_options = {}

        self.load_grid_options()

    def fit(self, cutoff_date):

        search_domain = self.grid_options.get(self.estimator)
        self.searcher = GridSearchCV(
            search_domain["estimator"], search_domain["params"]
        )

        X, y = self.get_data(cutoff_date)
        self.searcher.fit(X, y)

        return self.searcher.best_params_

    def get_data(self, cutoff_date):

        X, y = get_data(self.df, cutoff_date, "train")
        print(X.shape, y.shape)

        return X, y
    
    def load_grid_options(self):
        self.grid_options = {
            "et": {
                "estimator": ExtraTreesClassifier(),
                "params": {
                    'n_estimators': [300, 400, 500],
                    'min_samples_leaf': [4, 6, 8, 10],
                    'bootstrap': [True],
                    'random_state': [5]
                }
            },
            "lgbm": {
                "estimator": LGBMClassifier(),
                "params": {
                    'metric': ['auc'],
                    'random_state': [5],
                    'objective': ['binary'],
                    'max_depth': [3, 4, 5],
                    'n_estimators': [300, 400, 500],
                    'subsample': [0.4, 0.5, 0.6, 0.7],
                    'colsample_bytree': [0.5, 0.6, 0.7, 0.8],
                    'learning_rate': [0.006, 0.008, 0.01]
                }
            }
        }
    