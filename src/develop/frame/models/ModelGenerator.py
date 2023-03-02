import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

from src.develop.frame.utils import get_data, get_params
from src.develop.cross_validation import CrossValidation
from src.train import ClassificationModel, TwoStepCalibration

class ModelGenerator:

    def __init__(self):
        self.df = get_data("src/develop/frame/data/df_model.csv")
        self.params = get_params("src/develop/frame/data/params_et.json")

        self.cv_result = None
        self.teste_result = None

        self.preprocess_data()

    def preprocess_data(self):

        self.df["dt_calendar"] = pd.to_datetime(self.df["dt_calendar"])
        self.df = self.df[(self.df.dt_calendar >= "2020-01")]

        print(self.df.shape)

    def run_pipeline(self, **pipe_params):

        print("\nInstantiating...")
        lr = LogisticRegression(**pipe_params["lr"])
        et = ExtraTreesClassifier(
            random_state=5, oob_score=True, bootstrap=True,
            **self.params
        )
        et_lr = TwoStepCalibration(et, lr)

        print("Validating...")
        et_cv = CrossValidation(self.df, et, roc_auc_score)
        et_cv.fit(**pipe_params["cv"])

        self.cv_result = et_cv.cv_result

        print("\nTesting...")
        et_model = ClassificationModel(self.df, et_lr, roc_auc_score)
        et_model.fit(**pipe_params["teste"])
        et_model.test_model(**pipe_params["teste"])

        self.teste_result = et_model.result
        print(et_model.result)

        print("\nGenerating past predictions...")
        self.segmentation_preds_hist(et_lr, pipe_params["historic"])

        return print("\n<<< Pipeline ended >>>")


    def segmentation_preds_hist(self, lr_model, cohorts):

        all_df_preds = []
        for cohort in cohorts:
            model = ClassificationModel(self.df, lr_model, roc_auc_score)

            model.fit(cutoff_period=cohort)
            model.test_model(cutoff_period=cohort)

            y_pred = lr_model.predict_proba(model.X_test)[:,1]

            df_ = self.df[self.df.dt_calendar == cohort][["dt_calendar", "id_imobiliaria"]]

            df_["predictions"] = y_pred
            df_["target"] = model.y_test

            ### Antiga segmentação apenas para fins comparativos #######
            # segmentation_df, segment_rates = get_segments(y_pred)
            # df_["old_segments"] =  np.array(segmentation_df["segments"])
            ############################################################

            all_df_preds.append(df_)

        df_preds = pd.concat(all_df_preds)
        df_preds.to_csv("src/develop/frame/data/df_preds.csv", index=False)

