from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV

from src.utils import get_data


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
    